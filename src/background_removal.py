#!/usr/bin/env python3
"""
background_removal.py

Production background remover with anatomical reasoning.

Public API stays compatible with your pipeline:

    from background_removal import BackgroundRemover
    bg = BackgroundRemover()
    out_path = bg.process(image_path, output_dir)

Core idea:
    Model fusion alone is not enough. This version adds an anatomy stage:
      1. Multi-model foreground probability:
         - YOLO ROI
         - SAM2
         - BiRefNet-style foreground model
         - human parser
         - MediaPipe pose segmentation/skeleton
         - depth consistency
         - adaptive skin detector
      2. Anatomical ownership:
         - pose/skeleton distance field
         - torso/head/limb tube prior
         - geodesic reconstruction from human-owned seeds
         - support ownership score per connected component
      3. Topology cleanup:
         - preserve valid negative spaces between arms/torso/legs
         - reject thin external protrusions
         - reject diagonal furniture/supports
         - reject rectangular/large background blocks
      4. Edge-only matting:
         - ViTMatte can soften the boundary but cannot invent foreground.

Install:
    pip install opencv-python pillow numpy torch torchvision transformers ultralytics scipy timm accelerate sentencepiece mediapipe

SAM2:
    git clone https://github.com/facebookresearch/sam2.git
    cd sam2 && pip install -e .

Notes:
    - The subject is assumed to be unclothed, so skin probability is a strong prior.
    - If optional models fail, the class continues with available signals unless fail_hard=True.
    - Debug output writes per-prior masks into output_dir/_debug_masks.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import cv2
import numpy as np
import torch
from PIL import Image


@dataclass
class ModelStatus:
    yolo: bool = False
    sam2: bool = False
    birefnet: bool = False
    parser: bool = False
    depth: bool = False
    pose: bool = False
    vitmatte: bool = False


@dataclass
class PosePrior:
    probability: np.ndarray
    skeleton: np.ndarray
    distance: np.ndarray
    landmarks: list[tuple[int, int, float]]
    tubes: np.ndarray


class BackgroundRemover:
    def __init__(
        self,
        yolo_model: str = "yolov8x-seg.pt",
        sam2_cfg: Optional[str] = "configs/sam2/sam2_hiera_l.yaml",
        sam2_ckpt: Optional[str] = "models/sam2/sam2_hiera_large.pt",
        birefnet_model: Optional[str] = "ZhengPeng7/BiRefNet",
        human_parser_model: Optional[str] = "fashn-ai/fashn-human-parser",
        depth_model: Optional[str] = "depth-anything/Depth-Anything-V2-Small-hf",
        vitmatte_model: Optional[str] = "hustvl/vitmatte-small-composition-1k",
        use_pose: bool = True,
        debug: bool = True,
        fail_hard: bool = False,
        device: Optional[str] = None,
        biref_side: int = 1024,
        min_subject_area: float = 0.006,
        max_subject_area: float = 0.68,
        threshold: float = 0.50,
    ) -> None:
        self.debug = debug
        self.fail_hard = fail_hard
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.biref_side = int(biref_side)
        self.min_subject_area = float(min_subject_area)
        self.max_subject_area = float(max_subject_area)
        self.threshold = float(threshold)

        self.status = ModelStatus()

        self.yolo = None
        self.sam_predictor = None
        self.biref_model = None
        self.parser_processor = None
        self.parser_model = None
        self.depth_processor = None
        self.depth_model = None
        self.vit_processor = None
        self.vitmatte = None
        self.mp_pose = None
        self.mp_pose_module = None

        self._load_yolo(yolo_model)
        self._load_sam2(sam2_cfg, sam2_ckpt)
        self._load_birefnet(birefnet_model)
        self._load_parser(human_parser_model)
        self._load_depth(depth_model)
        self._load_pose(use_pose)
        self._load_vitmatte(vitmatte_model)

        print(f"BackgroundRemover initialized on {self.device}.")
        print(f"Model status: {self.status}")

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_yolo(self, model: Optional[str]) -> None:
        if not model:
            return
        try:
            from ultralytics import YOLO

            print("Loading YOLO...")
            self.yolo = YOLO(model)
            self.status.yolo = True
        except Exception as exc:
            self._load_warning("YOLO", exc)

    def _load_sam2(self, cfg: Optional[str], ckpt: Optional[str]) -> None:
        if not cfg or not ckpt:
            return
        try:
            print("Loading SAM2...")
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            sam2 = build_sam2(cfg, ckpt, device=self.device)
            self.sam_predictor = SAM2ImagePredictor(sam2)
            self.status.sam2 = True
        except Exception as exc:
            self._load_warning("SAM2", exc)

    def _load_birefnet(self, model: Optional[str]) -> None:
        if not model:
            return
        try:
            print("Loading BiRefNet...")
            from transformers import AutoModelForImageSegmentation

            self.biref_model = AutoModelForImageSegmentation.from_pretrained(
                model,
                trust_remote_code=True,
            ).to(self.device).eval()
            self.status.birefnet = True
        except Exception as exc:
            self._load_warning("BiRefNet", exc)

    def _load_parser(self, model: Optional[str]) -> None:
        if not model:
            return
        try:
            print("Loading human parser...")
            from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

            self.parser_processor = AutoImageProcessor.from_pretrained(model)
            self.parser_model = SegformerForSemanticSegmentation.from_pretrained(model).to(self.device).eval()
            self.status.parser = True
        except Exception as exc:
            self._load_warning("human parser", exc)

    def _load_depth(self, model: Optional[str]) -> None:
        if not model:
            return
        try:
            print("Loading depth estimator...")
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation

            self.depth_processor = AutoImageProcessor.from_pretrained(model)
            self.depth_model = AutoModelForDepthEstimation.from_pretrained(model).to(self.device).eval()
            self.status.depth = True
        except Exception as exc:
            self._load_warning("depth estimator", exc)

    def _load_pose(self, use_pose: bool) -> None:
        if not use_pose:
            return
        try:
            print("Loading MediaPipe pose...")
            import mediapipe as mp

            self.mp_pose_module = mp.solutions.pose
            self.mp_pose = self.mp_pose_module.Pose(
                static_image_mode=True,
                model_complexity=2,
                enable_segmentation=True,
                min_detection_confidence=0.20,
            )
            self.status.pose = True
        except Exception as exc:
            self._load_warning("MediaPipe pose", exc)

    def _load_vitmatte(self, model: Optional[str]) -> None:
        if not model:
            return
        try:
            print("Loading ViTMatte...")
            from transformers import AutoImageProcessor, VitMatteForImageMatting

            self.vit_processor = AutoImageProcessor.from_pretrained(model)
            self.vitmatte = VitMatteForImageMatting.from_pretrained(model).to(self.device).eval()
            self.status.vitmatte = True
        except Exception as exc:
            self._load_warning("ViTMatte", exc)

    def _load_warning(self, name: str, exc: Exception) -> None:
        msg = f"{name} unavailable; continuing without it. Reason: {exc}"
        if self.fail_hard:
            raise RuntimeError(msg) from exc
        warnings.warn(msg)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, image_path, output_dir):
        image_path = Path(image_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Could not load image: {image_path}")

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]

        boxes = self._detect_person_boxes(bgr)
        parser_prob = self._run_parser(rgb)
        box = self._choose_subject_box(boxes, parser_prob, (h, w))
        if box is None:
            box = np.array([0, 0, w - 1, h - 1], dtype=np.float32)

        roi_prob = self._roi_probability((h, w), box)
        pose_prior = self._run_pose_prior(rgb, box)

        biref_prob = self._run_birefnet(rgb)
        sam_prob = self._run_sam2(rgb, box, parser_prob, pose_prior.probability, biref_prob)
        depth_prob = self._run_depth(rgb, box, parser_prob, sam_prob, biref_prob, pose_prior.probability)

        proposal_prob = self._weighted_average([
            (biref_prob, 1.20),
            (sam_prob, 1.00),
            (parser_prob, 1.00),
            (pose_prior.probability, 0.85),
            (roi_prob, 0.35),
        ])

        skin_prob, skin_seed = self._skin_probability(
            bgr=bgr,
            proposal_prob=proposal_prob,
            parser_prob=parser_prob,
            pose_prob=pose_prior.probability,
            box=box,
        )

        fused_prob = self._fuse_probabilities(
            sam=sam_prob,
            biref=biref_prob,
            parser=parser_prob,
            pose=pose_prior.probability,
            depth=depth_prob,
            skin=skin_prob,
            roi=roi_prob,
        )

        anatomy_ownership = self._anatomical_ownership(
            fused_prob=fused_prob,
            skin_prob=skin_prob,
            parser_prob=parser_prob,
            pose_prior=pose_prior,
            roi_prob=roi_prob,
            box=box,
        )

        binary = self._probability_to_subject(
            fused_prob=fused_prob,
            ownership=anatomy_ownership,
            skin_prob=skin_prob,
            parser_prob=parser_prob,
            pose_prior=pose_prior,
            sam_prob=sam_prob,
            biref_prob=biref_prob,
            depth_prob=depth_prob,
            roi_prob=roi_prob,
            box=box,
        )

        binary = self._anatomical_reconstruction(
            binary=binary,
            ownership=anatomy_ownership,
            fused_prob=fused_prob,
            skin_prob=skin_prob,
            parser_prob=parser_prob,
            pose_prior=pose_prior,
            box=box,
        )
        binary = self._reject_non_anatomical_components(
            binary=binary,
            ownership=anatomy_ownership,
            fused_prob=fused_prob,
            skin_prob=skin_prob,
            parser_prob=parser_prob,
            pose_prior=pose_prior,
            depth_prob=depth_prob,
            box=box,
        )
        binary = self._remove_limb_width_outliers(
            binary=binary,
            ownership=anatomy_ownership,
            pose_prior=pose_prior,
            skin_prob=skin_prob,
            parser_prob=parser_prob,
        )
        binary = self._preserve_anatomical_cutouts(
            binary=binary,
            fused_prob=fused_prob,
            skin_prob=skin_prob,
            parser_prob=parser_prob,
            pose_prior=pose_prior,
            box=box,
        )
        binary = self._repair_valid_body_dents(
            binary=binary,
            fused_prob=fused_prob,
            skin_prob=skin_prob,
            parser_prob=parser_prob,
            pose_prior=pose_prior,
            box=box,
        )
        binary = self._final_binary(
            binary=binary,
            ownership=anatomy_ownership,
            fused_prob=fused_prob,
            skin_prob=skin_prob,
            parser_prob=parser_prob,
            pose_prior=pose_prior,
            box=box,
        )

        alpha = self._matte_alpha(rgb, binary)
        alpha = self._final_alpha(
            alpha=alpha,
            binary=binary,
            ownership=anatomy_ownership,
            fused_prob=fused_prob,
            skin_prob=skin_prob,
            parser_prob=parser_prob,
            pose_prior=pose_prior,
            box=box,
        )

        rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = alpha

        out_path = output_dir / f"{image_path.stem}.png"
        cv2.imwrite(str(out_path), rgba)

        if self.debug:
            self._write_debug(
                output_dir / "_debug_masks",
                image_path.stem,
                {
                    "roi": roi_prob,
                    "pose_prob": pose_prior.probability,
                    "pose_skeleton": pose_prior.skeleton,
                    "pose_distance": 1.0 - np.clip(pose_prior.distance, 0, 1),
                    "pose_tubes": pose_prior.tubes,
                    "biref": biref_prob,
                    "sam": sam_prob,
                    "parser": parser_prob,
                    "depth": depth_prob,
                    "skin": skin_prob,
                    "skin_seed": skin_seed,
                    "proposal": proposal_prob,
                    "fused": fused_prob,
                    "ownership": anatomy_ownership,
                    "binary": binary,
                    "alpha": alpha,
                },
            )

        print(f"✓ Saved: {out_path}")
        return out_path

    # ------------------------------------------------------------------
    # Model maps
    # ------------------------------------------------------------------

    def _detect_person_boxes(self, bgr: np.ndarray) -> list[np.ndarray]:
        if self.yolo is None:
            return []
        h, w = bgr.shape[:2]
        try:
            result = self.yolo(bgr, verbose=False, conf=0.035, classes=[0])[0]
        except Exception as exc:
            self._runtime_warning("YOLO", exc)
            return []

        if result.boxes is None or len(result.boxes) == 0:
            return []

        boxes: list[tuple[float, np.ndarray]] = []
        for xyxy, conf, cls in zip(
            result.boxes.xyxy.cpu().numpy(),
            result.boxes.conf.cpu().numpy(),
            result.boxes.cls.cpu().numpy(),
        ):
            if int(cls) != 0:
                continue
            x1, y1, x2, y2 = xyxy.astype(np.float32)
            bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
            area = bw * bh / float(h * w)
            if area < self.min_subject_area:
                continue
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            score = float(conf) + 2.8 * area + 1.1 * bh / h + 0.15 * cy / h - 0.5 * abs(cx - 0.5 * w) / w
            pad_x, pad_y = 0.18 * bw, 0.16 * bh
            boxes.append((
                score,
                np.array([
                    max(0, x1 - pad_x),
                    max(0, y1 - pad_y),
                    min(w - 1, x2 + pad_x),
                    min(h - 1, y2 + pad_y),
                ], dtype=np.float32),
            ))

        boxes.sort(key=lambda x: x[0], reverse=True)
        return [b for _, b in boxes[:5]]

    def _choose_subject_box(
        self,
        boxes: Sequence[np.ndarray],
        parser_prob: np.ndarray,
        shape: tuple[int, int],
    ) -> Optional[np.ndarray]:
        h, w = shape
        if boxes:
            scored = []
            parser_mask = (parser_prob > 0.35).astype(np.uint8) * 255
            for b in boxes:
                area = ((b[2] - b[0]) * (b[3] - b[1])) / float(h * w)
                parser_cov = self._mask_box_coverage(parser_mask, b)
                scored.append((area * 0.35 + parser_cov, b))
            scored.sort(key=lambda x: x[0], reverse=True)
            return scored[0][1]

        box = self._bbox(parser_prob, 0.35)
        if box is None:
            return None

        x1, y1, x2, y2 = box
        bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
        return np.array([
            max(0, x1 - 0.18 * bw),
            max(0, y1 - 0.18 * bh),
            min(w - 1, x2 + 0.18 * bw),
            min(h - 1, y2 + 0.18 * bh),
        ], dtype=np.float32)

    def _run_parser(self, rgb: np.ndarray) -> np.ndarray:
        h, w = rgb.shape[:2]
        if self.parser_model is None:
            return np.zeros((h, w), np.float32)

        try:
            inputs = self.parser_processor(images=Image.fromarray(rgb), return_tensors="pt").to(self.device)
            self._match_input_dtype(inputs, self.parser_model)

            with torch.no_grad():
                outputs = self.parser_model(**inputs)

            logits = torch.nn.functional.interpolate(
                outputs.logits.float(),
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )
            probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

            id2label = self.parser_model.config.id2label
            keep_words = (
                "skin", "face", "hair", "head", "neck", "torso", "body",
                "arm", "hand", "leg", "foot", "left", "right", "upper", "lower",
            )
            drop_words = (
                "background", "bag", "hat", "cap", "glasses", "sunglass", "jewelry",
                "shoe", "sock", "dress", "skirt", "pants", "coat", "shirt", "top",
                "belt",
            )
            keep_ids = []
            for idx, label in id2label.items():
                s = str(label).lower()
                if any(d in s for d in drop_words):
                    continue
                if any(k in s for k in keep_words):
                    keep_ids.append(int(idx))

            if not keep_ids:
                keep_ids = [int(i) for i, label in id2label.items() if "background" not in str(label).lower()]

            prob = probs[keep_ids].sum(axis=0)
            prob = np.clip(prob, 0.0, 1.0).astype(np.float32)
            prob = cv2.GaussianBlur(prob, (5, 5), 0)
            return prob
        except Exception as exc:
            self._runtime_warning("parser", exc)
            return np.zeros((h, w), np.float32)

    def _run_birefnet(self, rgb: np.ndarray) -> np.ndarray:
        h, w = rgb.shape[:2]
        if self.biref_model is None:
            return np.zeros((h, w), np.float32)

        try:
            side = self.biref_side
            image = Image.fromarray(rgb).resize((side, side), Image.BILINEAR)
            arr = np.asarray(image).astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], np.float32)
            std = np.array([0.229, 0.224, 0.225], np.float32)
            arr = (arr - mean) / std
            tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)
            tensor = tensor.to(dtype=self._model_dtype(self.biref_model))

            with torch.no_grad():
                out = self.biref_model(tensor)

            pred = self._extract_first_tensor(out)
            if pred is None:
                return np.zeros((h, w), np.float32)

            if pred.ndim == 4:
                pred = pred[:, 0, :, :]
            elif pred.ndim == 3:
                pred = pred[0]
            else:
                pred = pred.reshape(1, side, side)[0]

            pred = torch.sigmoid(pred.float()).detach().cpu().numpy()
            pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR)
            pred = np.clip(pred, 0.0, 1.0).astype(np.float32)
            pred = self._correct_probability_polarity(pred, np.zeros_like(pred), None)
            return pred
        except Exception as exc:
            self._runtime_warning("BiRefNet", exc)
            return np.zeros((h, w), np.float32)

    def _run_sam2(
        self,
        rgb: np.ndarray,
        box: np.ndarray,
        parser_prob: np.ndarray,
        pose_prob: np.ndarray,
        biref_prob: np.ndarray,
    ) -> np.ndarray:
        h, w = rgb.shape[:2]
        if self.sam_predictor is None:
            return np.zeros((h, w), np.float32)

        try:
            self.sam_predictor.set_image(rgb)
            points, labels = self._sam_prompt_points((h, w), box, parser_prob, pose_prob, biref_prob)

            candidates: list[tuple[float, np.ndarray]] = []
            calls = [
                (dict(box=box.astype(np.float32), point_coords=points, point_labels=labels, multimask_output=True), 0.0),
                (dict(box=box.astype(np.float32), multimask_output=True), -0.12),
            ]

            for kwargs, bias in calls:
                masks, scores, _ = self.sam_predictor.predict(**kwargs)
                for m, s in zip(masks, scores):
                    prob = m.astype(np.float32)
                    prob = self._correct_probability_polarity(prob, parser_prob, box)
                    score = self._score_prob_mask(prob, parser_prob, box, float(s)) + bias
                    candidates.append((score, prob))

            if not candidates:
                return np.zeros((h, w), np.float32)

            candidates.sort(key=lambda x: x[0], reverse=True)
            best_score, best = candidates[0]
            maps = [(best, 1.0)]
            for score, prob in candidates[1:]:
                if score < best_score - 0.35:
                    continue
                if self._prob_iou(prob, best, 0.5) > 0.35 or self._prob_overlap(prob, parser_prob, 0.5) > 0.35:
                    maps.append((prob, max(0.15, math.exp(score - best_score))))

            out = self._weighted_average(maps)
            out = cv2.GaussianBlur(out, (5, 5), 0)
            return np.clip(out, 0.0, 1.0).astype(np.float32)
        except Exception as exc:
            self._runtime_warning("SAM2", exc)
            return np.zeros((h, w), np.float32)

    def _sam_prompt_points(
        self,
        shape: tuple[int, int],
        box: np.ndarray,
        parser_prob: np.ndarray,
        pose_prob: np.ndarray,
        biref_prob: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        h, w = shape
        x1, y1, x2, y2 = box.astype(float)
        bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)

        pos = [
            ((x1 + x2) / 2, y1 + 0.12 * bh),
            ((x1 + x2) / 2, y1 + 0.28 * bh),
            ((x1 + x2) / 2, y1 + 0.46 * bh),
            ((x1 + x2) / 2, y1 + 0.64 * bh),
            ((x1 + x2) / 2, y1 + 0.84 * bh),
            (x1 + 0.30 * bw, y1 + 0.48 * bh),
            (x1 + 0.70 * bw, y1 + 0.48 * bh),
            (x1 + 0.35 * bw, y1 + 0.88 * bh),
            (x1 + 0.65 * bw, y1 + 0.88 * bh),
        ]

        consensus = np.maximum.reduce([parser_prob, pose_prob * 0.9, biref_prob * 0.7])
        if np.max(consensus) > 0.1:
            pos.extend(self._sample_probability_centers(consensus, 0.35, 14))

        margin = max(10, int(min(h, w) * 0.025))
        neg = [
            (margin, margin), (w - margin - 1, margin),
            (margin, h - margin - 1), (w - margin - 1, h - margin - 1),
            (w / 2, margin), (w / 2, h - margin - 1),
            (margin, h / 2), (w - margin - 1, h / 2),
            (max(0, x1 - margin), y1 + 0.20 * bh),
            (max(0, x1 - margin), y1 + 0.50 * bh),
            (max(0, x1 - margin), y1 + 0.80 * bh),
            (min(w - 1, x2 + margin), y1 + 0.20 * bh),
            (min(w - 1, x2 + margin), y1 + 0.50 * bh),
            (min(w - 1, x2 + margin), y1 + 0.80 * bh),
            ((x1 + x2) / 2, max(0, y1 - margin)),
            ((x1 + x2) / 2, min(h - 1, y2 + margin)),
        ]

        pts = np.array(pos + neg, dtype=np.float32)
        pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
        lab = np.array([1] * len(pos) + [0] * len(neg), dtype=np.int32)
        return pts, lab

    def _run_pose_prior(self, rgb: np.ndarray, box: np.ndarray) -> PosePrior:
        h, w = rgb.shape[:2]
        zero = np.zeros((h, w), np.float32)
        if self.mp_pose is None:
            return PosePrior(zero, np.zeros((h, w), np.uint8), np.ones((h, w), np.float32), [], np.zeros((h, w), np.float32))

        try:
            result = self.mp_pose.process(rgb)
            prob = np.zeros((h, w), np.float32)
            skeleton = np.zeros((h, w), np.uint8)
            tubes = np.zeros((h, w), np.float32)
            landmarks: list[tuple[int, int, float]] = []

            if getattr(result, "segmentation_mask", None) is not None:
                seg = result.segmentation_mask.astype(np.float32)
                prob = np.maximum(prob, np.clip(seg, 0.0, 1.0))

            if getattr(result, "pose_landmarks", None):
                for lm in result.pose_landmarks.landmark:
                    vis = float(getattr(lm, "visibility", 1.0))
                    if vis < 0.15:
                        landmarks.append((-1, -1, vis))
                        continue
                    x = int(np.clip(lm.x * w, 0, w - 1))
                    y = int(np.clip(lm.y * h, 0, h - 1))
                    landmarks.append((x, y, vis))

                radius = max(8, int(min(h, w) * 0.020))
                for x, y, vis in landmarks:
                    if x >= 0 and y >= 0:
                        cv2.circle(skeleton, (x, y), max(3, radius // 2), 255, -1)

                connections = self._mediapipe_connections()
                for a, b in connections:
                    if a >= len(landmarks) or b >= len(landmarks):
                        continue
                    x1, y1, v1 = landmarks[a]
                    x2, y2, v2 = landmarks[b]
                    if x1 < 0 or x2 < 0 or min(v1, v2) < 0.15:
                        continue
                    thickness = self._limb_thickness(a, b, h, w)
                    cv2.line(skeleton, (x1, y1), (x2, y2), 255, max(2, thickness // 3))
                    cv2.line(tubes, (x1, y1), (x2, y2), 1.0, thickness)

                tubes = cv2.GaussianBlur(tubes, (15, 15), 0)
                if float(tubes.max()) > 1e-6:
                    tubes = np.clip(tubes / float(tubes.max()), 0.0, 1.0)

                prob = np.maximum(prob, tubes)

            prob *= self._roi_probability((h, w), box)
            prob = cv2.GaussianBlur(prob, (15, 15), 0)
            prob = np.clip(prob, 0.0, 1.0).astype(np.float32)

            # Distance field: 0 near skeleton/tubes, 1 far away.
            support = ((skeleton > 0) | (tubes > 0.15)).astype(np.uint8) * 255
            if np.count_nonzero(support) > 0:
                inv = (support == 0).astype(np.uint8)
                dist = cv2.distanceTransform(inv, cv2.DIST_L2, 5)
                scale = max(25.0, min(h, w) * 0.18)
                dist = np.clip(dist / scale, 0.0, 1.0).astype(np.float32)
            else:
                dist = np.ones((h, w), np.float32)

            return PosePrior(prob, skeleton, dist, landmarks, tubes.astype(np.float32))
        except Exception as exc:
            self._runtime_warning("pose", exc)
            return PosePrior(zero, np.zeros((h, w), np.uint8), np.ones((h, w), np.float32), [], np.zeros((h, w), np.float32))

    def _run_depth(
        self,
        rgb: np.ndarray,
        box: np.ndarray,
        parser_prob: np.ndarray,
        sam_prob: np.ndarray,
        biref_prob: np.ndarray,
        pose_prob: np.ndarray,
    ) -> np.ndarray:
        h, w = rgb.shape[:2]
        if self.depth_model is None:
            return np.zeros((h, w), np.float32)

        try:
            inputs = self.depth_processor(images=Image.fromarray(rgb), return_tensors="pt").to(self.device)
            self._match_input_dtype(inputs, self.depth_model)

            with torch.no_grad():
                outputs = self.depth_model(**inputs)

            depth = outputs.predicted_depth.float()
            depth = torch.nn.functional.interpolate(
                depth.unsqueeze(1),
                size=(h, w),
                mode="bicubic",
                align_corners=False,
            )[0, 0].detach().cpu().numpy().astype(np.float32)

            lo, hi = np.percentile(depth, [2, 98])
            d = np.clip((depth - lo) / max(1e-6, hi - lo), 0.0, 1.0)

            seed_prob = self._weighted_average([
                (parser_prob, 1.0),
                (sam_prob, 1.0),
                (biref_prob, 0.8),
                (pose_prob, 0.5),
            ])
            seed = (seed_prob > 0.62).astype(np.uint8)
            seed = cv2.bitwise_and(seed, self._expanded_box_mask((h, w), box, 0.05) // 255)

            if np.count_nonzero(seed) < 100:
                return np.zeros((h, w), np.float32)

            med = float(np.median(d[seed > 0]))
            mad = float(np.median(np.abs(d[seed > 0] - med)) + 1e-4)
            sigma = max(0.06, 2.5 * mad)

            prob = np.exp(-0.5 * ((d - med) / sigma) ** 2).astype(np.float32)
            prob *= self._roi_probability((h, w), box)
            prob = cv2.GaussianBlur(prob, (9, 9), 0)
            return np.clip(prob, 0.0, 1.0).astype(np.float32)
        except Exception as exc:
            self._runtime_warning("depth", exc)
            return np.zeros((h, w), np.float32)

    # ------------------------------------------------------------------
    # Probabilistic and anatomical priors
    # ------------------------------------------------------------------

    def _roi_probability(self, shape: tuple[int, int], box: Optional[np.ndarray]) -> np.ndarray:
        h, w = shape
        if box is None:
            return np.ones((h, w), dtype=np.float32) * 0.25

        x1, y1, x2, y2 = np.asarray(box, dtype=np.float32)
        bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
        px, py = 0.20 * bw, 0.20 * bh

        xx1 = int(max(0, x1 - px))
        yy1 = int(max(0, y1 - py))
        xx2 = int(min(w - 1, x2 + px))
        yy2 = int(min(h - 1, y2 + py))

        hard = np.zeros((h, w), np.float32)
        hard[yy1:yy2 + 1, xx1:xx2 + 1] = 1.0
        sigma = max(8.0, min(h, w) * 0.04)
        soft = cv2.GaussianBlur(hard, (0, 0), sigmaX=sigma, sigmaY=sigma)
        if float(soft.max()) > 1e-6:
            soft /= float(soft.max())
        return np.clip(soft, 0.0, 1.0).astype(np.float32)

    def _skin_probability(
        self,
        bgr: np.ndarray,
        proposal_prob: np.ndarray,
        parser_prob: np.ndarray,
        pose_prob: np.ndarray,
        box: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        h, w = proposal_prob.shape

        seed = (((parser_prob > 0.50) | (pose_prob > 0.60)) & (proposal_prob > 0.28)).astype(np.uint8) * 255
        if np.count_nonzero(seed) < 250:
            seed = (proposal_prob > 0.72).astype(np.uint8) * 255

        seed = self._erode(seed, 9, 1)

        x1, y1, x2, y2 = box.astype(int)
        bw, bh = max(1, x2 - x1), max(1, y2 - y1)
        central = np.zeros((h, w), np.uint8)
        central[
            max(0, int(y1 + 0.08 * bh)):min(h, int(y1 + 0.86 * bh)),
            max(0, int(x1 + 0.08 * bw)):min(w, int(x1 + 0.92 * bw)),
        ] = 255

        central_seed = cv2.bitwise_and(seed, central)
        if np.count_nonzero(central_seed) > 180:
            seed = central_seed

        if np.count_nonzero(seed) < 80:
            return np.zeros((h, w), np.float32), seed

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

        feat = np.dstack([
            hsv[:, :, 0],
            hsv[:, :, 1],
            hsv[:, :, 2],
            ycc[:, :, 1],
            ycc[:, :, 2],
            lab[:, :, 1],
            lab[:, :, 2],
        ]).astype(np.float32)

        samples = feat[seed > 0]
        med = np.median(samples, axis=0)
        mad = np.median(np.abs(samples - med), axis=0) + 1.0
        sigma = np.maximum(
            1.4826 * mad,
            np.array([7.0, 15.0, 20.0, 8.0, 8.0, 7.0, 7.0], dtype=np.float32),
        )

        hue_abs = np.abs(feat[:, :, 0] - med[0])
        d0 = np.minimum(hue_abs, 180.0 - hue_abs) / sigma[0]
        d1 = (feat[:, :, 1] - med[1]) / sigma[1]
        d2 = (feat[:, :, 2] - med[2]) / sigma[2]
        d3 = (feat[:, :, 3] - med[3]) / sigma[3]
        d4 = (feat[:, :, 4] - med[4]) / sigma[4]
        d5 = (feat[:, :, 5] - med[5]) / sigma[5]
        d6 = (feat[:, :, 6] - med[6]) / sigma[6]

        dist2 = d0*d0 + d1*d1 + 0.45*d2*d2 + d3*d3 + d4*d4 + 0.7*d5*d5 + 0.7*d6*d6
        prob = np.exp(-0.5 * dist2).astype(np.float32)

        roi = self._roi_probability((h, w), box)
        near = cv2.dilate(
            (proposal_prob > 0.22).astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (75, 75)),
            iterations=1,
        ).astype(np.float32)
        prob *= roi
        prob *= np.maximum(0.35, near)
        prob = cv2.GaussianBlur(prob, (5, 5), 0)
        return np.clip(prob, 0.0, 1.0).astype(np.float32), seed

    def _fuse_probabilities(
        self,
        sam: np.ndarray,
        biref: np.ndarray,
        parser: np.ndarray,
        pose: np.ndarray,
        depth: np.ndarray,
        skin: np.ndarray,
        roi: np.ndarray,
    ) -> np.ndarray:
        weighted = [
            (sam, 1.05),
            (biref, 1.35),
            (parser, 1.20),
            (pose, 1.05),
            (depth, 0.65),
            (skin, 1.95),
            (roi, 0.30),
        ]

        eps = 1e-4
        shape = roi.shape
        logit_sum = np.zeros(shape, np.float32)
        weight_sum = 0.0

        for p, weight in weighted:
            if p is None or p.shape != shape or float(np.max(p)) <= 1e-5:
                continue
            pp = np.clip(p.astype(np.float32), eps, 1.0 - eps)
            logit_sum += float(weight) * np.log(pp / (1.0 - pp))
            weight_sum += float(weight)

        if weight_sum <= 0:
            return np.zeros(shape, np.float32)

        logits = logit_sum / weight_sum
        logits -= 0.06
        fused = 1.0 / (1.0 + np.exp(-logits))
        fused = cv2.GaussianBlur(fused.astype(np.float32), (5, 5), 0)
        return np.clip(fused, 0.0, 1.0)

    def _anatomical_ownership(
        self,
        fused_prob: np.ndarray,
        skin_prob: np.ndarray,
        parser_prob: np.ndarray,
        pose_prior: PosePrior,
        roi_prob: np.ndarray,
        box: np.ndarray,
    ) -> np.ndarray:
        # Skeleton distance decay: close to skeleton/tubes means likely body ownership.
        dist_decay = np.exp(-3.5 * np.clip(pose_prior.distance, 0.0, 1.0)).astype(np.float32)

        # If pose failed, do not punish too hard.
        if float(np.max(pose_prior.probability)) < 0.05:
            dist_decay = np.ones_like(dist_decay) * 0.55

        support = (
            0.34 * fused_prob
            + 0.30 * skin_prob
            + 0.20 * parser_prob
            + 0.18 * pose_prior.probability
            + 0.14 * dist_decay
            + 0.06 * roi_prob
        )

        ownership = np.clip(support, 0.0, 1.0).astype(np.float32)

        # Penalize pixels far from skeleton with low skin/parser support.
        far = pose_prior.distance > 0.65
        weak = (skin_prob < 0.18) & (parser_prob < 0.18)
        ownership[far & weak] *= 0.35

        # Penalize likely background outside ROI.
        ownership[roi_prob < 0.05] *= 0.15

        ownership = cv2.GaussianBlur(ownership, (5, 5), 0)
        return np.clip(ownership, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Anatomy-aware binary mask
    # ------------------------------------------------------------------

    def _probability_to_subject(
        self,
        fused_prob,
        ownership,
        skin_prob,
        parser_prob,
        pose_prior,
        sam_prob,
        biref_prob,
        depth_prob,
        roi_prob,
        box,
    ):
        import cv2
        import numpy as np

        # IMPORTANT:
        # Trust the fused mask first. In your debug images this is the cleanest map.
        # Do not allow pose tubes or morphology to invent shape.
        strong = fused_prob > 0.50

        # Recover subject pixels that fused is slightly uncertain about, but only if
        # there is skin/parser evidence.
        soft_body = (
            (fused_prob > 0.34)
            & (
                (skin_prob > 0.24)
                | (parser_prob > 0.24)
                | ((sam_prob > 0.45) & (biref_prob > 0.35))
            )
            & (roi_prob > 0.05)
        )

        candidate = (strong | soft_body).astype(np.uint8) * 255

        # Remove obvious non-human regions. This is conservative; it should not reshape
        # arms/body just because pose is imperfect.
        non_body = (
            (fused_prob < 0.44)
            & (skin_prob < 0.16)
            & (parser_prob < 0.14)
            & (ownership < 0.26)
        )
        candidate[non_body] = 0

        seed = (
            (fused_prob > 0.62)
            | ((skin_prob > 0.45) & (parser_prob > 0.22))
            | ((parser_prob > 0.50) & (fused_prob > 0.30))
        ).astype(np.uint8) * 255

        if np.count_nonzero(seed) < 100:
            seed = (fused_prob > 0.58).astype(np.uint8) * 255

        out = self._reconstruct(seed, candidate)

        # If reconstruction becomes worse/smaller, fall back to direct fused threshold.
        direct = (fused_prob > 0.48).astype(np.uint8) * 255
        if np.count_nonzero(out) < np.count_nonzero(direct) * 0.78:
            out = direct

        return out


    def _anatomical_reconstruction(
        self,
        binary,
        ownership,
        fused_prob,
        skin_prob,
        parser_prob,
        pose_prior,
        box,
    ):
        import cv2
        import numpy as np

        # Previous version recovered too much from pose tubes and could introduce
        # elbow/background protrusions. This version only restores pixels that are
        # also supported by fused/skin/parser.
        near_body = cv2.dilate(
            binary,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)),
            iterations=1,
        )

        recover = (
            (fused_prob > 0.38)
            & (
                (skin_prob > 0.22)
                | (parser_prob > 0.22)
                | (ownership > 0.34)
            )
        ).astype(np.uint8) * 255

        recover = cv2.bitwise_and(recover, near_body)
        recover = cv2.bitwise_and(recover, self._expanded_box_mask(binary.shape, box, 0.10))

        allowed = cv2.bitwise_or(binary, recover)
        out = self._reconstruct(binary, allowed)

        return out


    def _reject_non_anatomical_components(
        self,
        binary: np.ndarray,
        ownership: np.ndarray,
        fused_prob: np.ndarray,
        skin_prob: np.ndarray,
        parser_prob: np.ndarray,
        pose_prior: PosePrior,
        depth_prob: np.ndarray,
        box: np.ndarray,
    ) -> np.ndarray:
        binary = self._bin(binary)
        num, labels, stats, cent = cv2.connectedComponentsWithStats(binary, 8)
        if num <= 1:
            return binary

        main = self._main_component(labels, stats, ownership, fused_prob, skin_prob, parser_prob, pose_prior.probability, box)
        main_area = stats[main, cv2.CC_STAT_AREA]
        main_cent = cent[main]
        out = np.zeros_like(binary)

        for i in range(1, num):
            comp = (labels == i).astype(np.uint8) * 255
            area = stats[i, cv2.CC_STAT_AREA]
            x, y, bw, bh, _ = stats[i]
            extent = area / max(1, bw * bh)
            aspect = max(bw, bh) / max(1, min(bw, bh))
            dist = float(np.linalg.norm(cent[i] - main_cent) / max(binary.shape))

            own = self._mean_prob(ownership, comp)
            f = self._mean_prob(fused_prob, comp)
            s = self._mean_prob(skin_prob, comp)
            p = self._mean_prob(parser_prob, comp)
            po = self._mean_prob(pose_prior.probability, comp)
            d = self._mean_prob(depth_prob, comp) if float(np.max(depth_prob)) > 0 else 0.0
            b = self._mask_box_coverage(comp, box)
            skel_near = 1.0 - self._mean_prob(pose_prior.distance, comp)

            background_block = (
                area > main_area * 0.030
                and extent > 0.36
                and s < 0.20
                and p < 0.16
                and po < 0.14
                and own < 0.34
            )

            diagonal_or_bar = (
                area > main_area * 0.006
                and aspect > 4.0
                and s < 0.22
                and p < 0.14
                and skel_near < 0.30
            )

            thin_weak = (
                area < main_area * 0.030
                and s < 0.17
                and p < 0.10
                and po < 0.10
                and f < 0.42
                and own < 0.35
            )

            keep = (
                i == main
                or (own > 0.34 and s > 0.20 and b > 0.02 and dist < 0.62)
                or (p > 0.22 and own > 0.24 and dist < 0.65)
                or (po > 0.25 and (s > 0.15 or p > 0.15) and dist < 0.65)
                or (d > 0.45 and s > 0.18 and own > 0.28 and dist < 0.55)
            )

            if keep and not background_block and not diagonal_or_bar and not thin_weak:
                out[labels == i] = 255

        return self._close(out, 7, 1)

    def _remove_limb_width_outliers(
        self,
        binary,
        ownership,
        pose_prior,
        skin_prob,
        parser_prob,
    ):
        # Disable aggressive pose-tube pruning.
        #
        # In your debug maps MediaPipe pose is only a crude stick figure. It is useful
        # as weak evidence, but not reliable enough to prune silhouette width.
        return binary

    def _preserve_anatomical_cutouts(
        self,
        binary: np.ndarray,
        fused_prob: np.ndarray,
        skin_prob: np.ndarray,
        parser_prob: np.ndarray,
        pose_prior: PosePrior,
        box: np.ndarray,
    ) -> np.ndarray:
        """
        Prevent valid negative spaces between arms/torso/legs from being filled.

        We mark internal holes as background if they have low skin/parser/ownership and
        are near pose skeleton/tube boundaries. This preserves arm-body triangles.
        """
        out = binary.copy()
        inv = cv2.bitwise_not(out)
        num, labels, stats, _ = cv2.connectedComponentsWithStats((inv > 0).astype(np.uint8), 8)
        h, w = out.shape

        for i in range(1, num):
            x, y, bw, bh, area = stats[i]
            touches = x == 0 or y == 0 or x + bw >= w or y + bh >= h
            if touches:
                continue

            hole = (labels == i).astype(np.uint8) * 255
            if area < 20:
                continue

            skin_m = self._mean_prob(skin_prob, hole)
            parser_m = self._mean_prob(parser_prob, hole)
            fused_m = self._mean_prob(fused_prob, hole)
            pose_near = 1.0 - self._mean_prob(pose_prior.distance, hole)

            # Valid cutout: little evidence of body, near body/limbs, not tiny noise.
            valid_negative_space = (
                area > out.size * 0.00035
                and skin_m < 0.22
                and parser_m < 0.22
                and fused_m < 0.40
                and pose_near > 0.12
            )

            # If the hole is very large, preserve it unless it has strong body evidence.
            large_negative_space = (
                area > out.size * 0.0015
                and skin_m < 0.28
                and parser_m < 0.28
            )

            if valid_negative_space or large_negative_space:
                out[hole > 0] = 0

        return out

    def _repair_valid_body_dents(
        self,
        binary,
        fused_prob,
        skin_prob,
        parser_prob,
        pose_prior,
        box,
    ):
        import cv2
        import numpy as np

        # Do NOT large-close the body. That was filling anatomical cutouts and changing
        # the clean fused shape.
        out = binary.copy()

        # Fill only tiny pinholes inside the body, not arm/torso negative spaces.
        out = self._fill_holes(out, max_hole_area=int(binary.size * 0.00035))

        # Very small local repair only where fused itself says body.
        closed = self._close(out, 9, 1)
        add = cv2.subtract(closed, out)

        valid_add = (
            (fused_prob > 0.50)
            | ((fused_prob > 0.40) & ((skin_prob > 0.24) | (parser_prob > 0.24)))
        ).astype(np.uint8) * 255

        add = cv2.bitwise_and(add, valid_add)

        # Only add tiny components; reject large bridges that close arm cutouts.
        num, labels, stats, _ = cv2.connectedComponentsWithStats((add > 0).astype(np.uint8), 8)
        small_add = np.zeros_like(add)
        max_add_area = max(20, int(binary.size * 0.00020))
        for i in range(1, num):
            if stats[i, cv2.CC_STAT_AREA] <= max_add_area:
                small_add[labels == i] = 255

        out = cv2.bitwise_or(out, small_add)
        out = self._preserve_anatomical_cutouts(out, fused_prob, skin_prob, parser_prob, pose_prior, box)
        return out

    def _final_binary(
        self,
        binary,
        ownership,
        fused_prob,
        skin_prob,
        parser_prob,
        pose_prior,
        box,
    ):
        import cv2
        import numpy as np

        binary = self._bin(binary)

        # Intersect with a slightly relaxed fused/body support map.
        # This removes elbow/background artefacts that survived reconstruction.
        support = (
            (fused_prob > 0.34)
            & (
                (skin_prob > 0.16)
                | (parser_prob > 0.14)
                | (ownership > 0.28)
                | (fused_prob > 0.55)
            )
        ).astype(np.uint8) * 255

        support = cv2.dilate(
            support,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=1,
        )

        binary = cv2.bitwise_and(binary, support)

        # Keep only supported connected components.
        binary = self._remove_small_components(binary, max(50, int(binary.size * 0.00010)))
        binary = self._keep_supported_components(
            binary,
            ownership,
            fused_prob,
            skin_prob,
            parser_prob,
            pose_prior.probability,
            box,
        )

        # Only tiny smoothing. Large closing destroys cutouts.
        binary = self._close(binary, 3, 1)

        # Preserve cutouts AFTER all smoothing.
        binary = self._preserve_anatomical_cutouts(
            binary,
            fused_prob,
            skin_prob,
            parser_prob,
            pose_prior,
            box,
        )

        # Fill only tiny pinholes. The small dot in your result is this category.
        binary = self._fill_holes(binary, max_hole_area=int(binary.size * 0.00030))

        return binary

    # ------------------------------------------------------------------
    # Matting
    # ------------------------------------------------------------------

    def _matte_alpha(self, rgb: np.ndarray, binary: np.ndarray) -> np.ndarray:
        if self.vitmatte is None:
            return binary.copy()

        try:
            trimap = self._make_trimap(binary)
            inputs = self.vit_processor(
                images=Image.fromarray(rgb),
                trimaps=Image.fromarray(trimap),
                return_tensors="pt",
            ).to(self.device)
            self._match_input_dtype(inputs, self.vitmatte)

            with torch.no_grad():
                outputs = self.vitmatte(**inputs)

            alpha = getattr(outputs, "alphas", None)
            if alpha is None:
                alpha = getattr(outputs, "alpha", None)
            if alpha is None and isinstance(outputs, dict):
                alpha = outputs.get("alphas", None)
                if alpha is None:
                    alpha = outputs.get("alpha", None)
            if alpha is None:
                return binary.copy()

            alpha = alpha[0, 0].detach().float().cpu().numpy()
            alpha = cv2.resize(alpha, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
            matte = np.clip(alpha * 255, 0, 255).astype(np.uint8)

            h, w = binary.shape
            inner_k = max(5, int(min(h, w) * 0.006) | 1)
            outer_k = max(9, int(min(h, w) * 0.010) | 1)

            inner = cv2.erode(binary, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (inner_k, inner_k)), iterations=1)
            outer = cv2.dilate(binary, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (outer_k, outer_k)), iterations=1)
            band = cv2.subtract(outer, inner)

            out = np.zeros_like(matte)
            out[inner > 0] = 255
            out[band > 0] = matte[band > 0]
            restore = (binary > 0) & (out < 225)
            out[restore] = 238
            out[outer == 0] = 0
            return out.astype(np.uint8)
        except Exception as exc:
            self._runtime_warning("ViTMatte", exc)
            return binary.copy()

    def _make_trimap(self, binary: np.ndarray) -> np.ndarray:
        h, w = binary.shape
        inner_k = max(5, int(min(h, w) * 0.006) | 1)
        outer_k = max(13, int(min(h, w) * 0.014) | 1)
        inner = cv2.erode(binary, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (inner_k, inner_k)), iterations=1)
        outer = cv2.dilate(binary, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (outer_k, outer_k)), iterations=1)
        trimap = np.zeros_like(binary)
        trimap[outer > 0] = 128
        trimap[inner > 0] = 255
        return trimap

    def _final_alpha(
        self,
        alpha: np.ndarray,
        binary: np.ndarray,
        ownership: np.ndarray,
        fused_prob: np.ndarray,
        skin_prob: np.ndarray,
        parser_prob: np.ndarray,
        pose_prior: PosePrior,
        box: np.ndarray,
    ) -> np.ndarray:
        allowed = cv2.dilate(binary, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
        alpha[allowed == 0] = 0

        fg = (alpha > 6).astype(np.uint8) * 255
        fg = self._keep_supported_components(fg, ownership, fused_prob, skin_prob, parser_prob, pose_prior.probability, box)
        alpha[fg == 0] = 0

        alpha[alpha < 5] = 0
        alpha[alpha > 247] = 255
        alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
        alpha[alpha < 4] = 0
        alpha[alpha > 250] = 255
        return alpha.astype(np.uint8)

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def _runtime_warning(self, name: str, exc: Exception) -> None:
        msg = f"{name} failed at runtime; using fallback. Reason: {exc}"
        if self.fail_hard:
            raise RuntimeError(msg) from exc
        warnings.warn(msg)

    @staticmethod
    def _model_dtype(model: torch.nn.Module) -> torch.dtype:
        try:
            return next(model.parameters()).dtype
        except StopIteration:
            return torch.float32

    def _match_input_dtype(self, inputs: Any, model: torch.nn.Module) -> None:
        dtype = self._model_dtype(model)
        if isinstance(inputs, dict):
            for k, v in list(inputs.items()):
                if torch.is_tensor(v) and v.is_floating_point():
                    inputs[k] = v.to(dtype=dtype)

    @staticmethod
    def _extract_first_tensor(obj: Any) -> Optional[torch.Tensor]:
        if torch.is_tensor(obj):
            return obj
        if isinstance(obj, dict):
            for key in ("logits", "preds", "out", "prediction", "result"):
                if key in obj:
                    found = BackgroundRemover._extract_first_tensor(obj[key])
                    if found is not None:
                        return found
            for value in obj.values():
                found = BackgroundRemover._extract_first_tensor(value)
                if found is not None:
                    return found
        if isinstance(obj, (list, tuple)):
            for item in reversed(obj):
                found = BackgroundRemover._extract_first_tensor(item)
                if found is not None:
                    return found
        return None

    @staticmethod
    def _weighted_average(items: Sequence[tuple[np.ndarray, float]]) -> np.ndarray:
        valid = [(p, float(w)) for p, w in items if p is not None and p.size > 1 and float(np.max(p)) > 1e-6 and w > 0]
        if not valid:
            if items:
                return np.zeros_like(items[0][0], dtype=np.float32)
            return np.zeros((1, 1), dtype=np.float32)
        shape = valid[0][0].shape
        acc = np.zeros(shape, np.float32)
        total = 0.0
        for p, w in valid:
            if p.shape != shape:
                continue
            acc += p.astype(np.float32) * w
            total += w
        if total <= 0:
            return np.zeros(shape, np.float32)
        return np.clip(acc / total, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def _bin(mask: np.ndarray) -> np.ndarray:
        return ((mask > 0).astype(np.uint8) * 255)

    @staticmethod
    def _open(mask: np.ndarray, k: int, it: int) -> np.ndarray:
        k |= 1
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)), iterations=it)

    @staticmethod
    def _close(mask: np.ndarray, k: int, it: int) -> np.ndarray:
        k |= 1
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)), iterations=it)

    @staticmethod
    def _erode(mask: np.ndarray, k: int, it: int) -> np.ndarray:
        k |= 1
        return cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)), iterations=it)

    @staticmethod
    def _bbox(prob_or_mask: np.ndarray, threshold: float = 0.5) -> Optional[np.ndarray]:
        ys, xs = np.where(prob_or_mask > threshold)
        if len(xs) == 0:
            return None
        return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)

    @staticmethod
    def _prob_iou(a: np.ndarray, b: np.ndarray, threshold: float) -> float:
        aa = a > threshold
        bb = b > threshold
        if not aa.any() or not bb.any():
            return 0.0
        return float(np.logical_and(aa, bb).sum() / max(1, np.logical_or(aa, bb).sum()))

    @staticmethod
    def _prob_overlap(a: np.ndarray, b: np.ndarray, threshold: float) -> float:
        aa = a > threshold
        bb = b > threshold
        if not aa.any() or not bb.any():
            return 0.0
        return float(np.logical_and(aa, bb).sum() / max(1, aa.sum()))

    def _score_prob_mask(self, prob: np.ndarray, parser_prob: np.ndarray, box: Optional[np.ndarray], base: float) -> float:
        mask = prob > 0.5
        area = float(mask.mean())
        if area < self.min_subject_area or area > 0.92:
            return -999.0
        p_iou = self._prob_iou(prob, parser_prob, 0.5) if float(np.max(parser_prob)) > 0.1 else 0.0
        p_ov = self._prob_overlap(prob, parser_prob, 0.5) if float(np.max(parser_prob)) > 0.1 else 0.0
        border = self._border_ratio(mask.astype(np.uint8) * 255)
        box_cov = self._mask_box_coverage(mask.astype(np.uint8) * 255, box) if box is not None else 0.0
        bbox = self._bbox(prob, 0.5)
        if bbox is None:
            return -999.0
        h, w = prob.shape
        x1, y1, x2, y2 = bbox
        height = (y2 - y1 + 1) / h
        width = (x2 - x1 + 1) / w
        center_penalty = abs(((x1 + x2) * 0.5) - w * 0.5) / w
        too_big = max(0.0, area - self.max_subject_area)
        return base + 2.2 * p_iou + 0.7 * p_ov + 1.0 * box_cov + 0.35 * height + 0.20 * width - 2.8 * border - 0.75 * center_penalty - 4.0 * too_big

    def _correct_probability_polarity(self, prob: np.ndarray, parser_prob: np.ndarray, box: Optional[np.ndarray]) -> np.ndarray:
        prob = np.clip(prob.astype(np.float32), 0.0, 1.0)
        inv = 1.0 - prob
        mask = prob > 0.5
        area = float(mask.mean())
        border = self._border_ratio(mask.astype(np.uint8) * 255)
        overlap = self._prob_overlap(prob, parser_prob, 0.5) if float(np.max(parser_prob)) > 0.1 else 0.0
        inv_overlap = self._prob_overlap(inv, parser_prob, 0.5) if float(np.max(parser_prob)) > 0.1 else 0.0
        box_cov = self._mask_box_coverage(mask.astype(np.uint8) * 255, box) if box is not None else 0.0
        inv_box_cov = self._mask_box_coverage((inv > 0.5).astype(np.uint8) * 255, box) if box is not None else 0.0
        if area > self.max_subject_area and border > 0.32:
            return inv
        if area > 0.50 and border > 0.42 and inv_box_cov >= box_cov * 0.65:
            return inv
        if float(np.max(parser_prob)) > 0.1 and inv_overlap > overlap + 0.12 and inv_box_cov >= box_cov * 0.65:
            return inv
        return prob

    @staticmethod
    def _mask_box_coverage(mask: np.ndarray, box: Optional[np.ndarray]) -> float:
        if box is None:
            return 0.0
        h, w = mask.shape
        x1, y1, x2, y2 = box.astype(int)
        x1, x2 = max(0, x1), min(w - 1, x2)
        y1, y2 = max(0, y1), min(h - 1, y2)
        if x2 <= x1 or y2 <= y1:
            return 0.0
        return float(np.mean(mask[y1:y2 + 1, x1:x2 + 1] > 0))

    @staticmethod
    def _border_ratio(mask: np.ndarray, px: int = 8) -> float:
        h, w = mask.shape
        b = max(1, min(px, h // 10, w // 10))
        border = np.zeros_like(mask, dtype=bool)
        border[:b, :] = True
        border[-b:, :] = True
        border[:, :b] = True
        border[:, -b:] = True
        return float(np.logical_and(mask > 0, border).sum() / max(1, border.sum()))

    @staticmethod
    def _expanded_box_mask(shape: tuple[int, int], box: np.ndarray, pad_frac: float) -> np.ndarray:
        h, w = shape
        x1, y1, x2, y2 = box.astype(float)
        bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
        xx1 = int(max(0, x1 - pad_frac * bw))
        xx2 = int(min(w - 1, x2 + pad_frac * bw))
        yy1 = int(max(0, y1 - pad_frac * bh))
        yy2 = int(min(h - 1, y2 + pad_frac * bh))
        out = np.zeros((h, w), np.uint8)
        out[yy1:yy2 + 1, xx1:xx2 + 1] = 255
        return out

    @staticmethod
    def _lower_box_mask(shape: tuple[int, int], box: np.ndarray, start_frac: float, pad_frac: float) -> np.ndarray:
        h, w = shape
        x1, y1, x2, y2 = box.astype(float)
        bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
        xx1 = int(max(0, x1 - pad_frac * bw))
        xx2 = int(min(w - 1, x2 + pad_frac * bw))
        yy1 = int(max(0, y1 + start_frac * bh))
        yy2 = int(min(h - 1, y2 + pad_frac * bh))
        out = np.zeros((h, w), np.uint8)
        out[yy1:yy2 + 1, xx1:xx2 + 1] = 255
        return out

    @staticmethod
    def _sample_probability_centers(prob: np.ndarray, threshold: float, max_points: int) -> list[tuple[float, float]]:
        mask = (prob > threshold).astype(np.uint8)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        comps = [(stats[i, cv2.CC_STAT_AREA], i) for i in range(1, num) if stats[i, cv2.CC_STAT_AREA] > 40]
        comps.sort(reverse=True)
        pts = []
        for _, i in comps[:max_points]:
            comp = (labels == i).astype(np.uint8)
            dist = cv2.distanceTransform(comp, cv2.DIST_L2, 5)
            y, x = np.unravel_index(np.argmax(dist), dist.shape)
            pts.append((float(x), float(y)))
        return pts

    @staticmethod
    def _remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
        num, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
        out = np.zeros_like(mask)
        for i in range(1, num):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                out[labels == i] = 255
        return out

    @staticmethod
    def _fill_holes(mask: np.ndarray, max_hole_area: int) -> np.ndarray:
        inv = cv2.bitwise_not(mask)
        num, labels, stats, _ = cv2.connectedComponentsWithStats((inv > 0).astype(np.uint8), 8)
        h, w = mask.shape
        out = mask.copy()
        for i in range(1, num):
            x, y, bw, bh, area = stats[i]
            touches = x == 0 or y == 0 or x + bw >= w or y + bh >= h
            if not touches and area <= max_hole_area:
                out[labels == i] = 255
        return out

    @staticmethod
    def _mean_prob(prob: np.ndarray, mask: np.ndarray) -> float:
        vals = prob[mask > 0]
        if vals.size == 0:
            return 0.0
        return float(np.mean(vals))

    def _main_component(
        self,
        labels: np.ndarray,
        stats: np.ndarray,
        ownership: np.ndarray,
        fused_prob: np.ndarray,
        skin_prob: np.ndarray,
        parser_prob: np.ndarray,
        pose_prob: np.ndarray,
        box: np.ndarray,
    ) -> int:
        best, best_score = 1, -1e18
        for i in range(1, stats.shape[0]):
            comp = (labels == i).astype(np.uint8) * 255
            area = stats[i, cv2.CC_STAT_AREA]
            own = self._mean_prob(ownership, comp)
            f = self._mean_prob(fused_prob, comp)
            s = self._mean_prob(skin_prob, comp)
            p = self._mean_prob(parser_prob, comp)
            po = self._mean_prob(pose_prob, comp)
            b = self._mask_box_coverage(comp, box)
            score = area * (1.0 + 1.9 * own + 1.2 * f + 2.0 * s + 1.5 * p + 0.9 * po + 0.5 * b)
            if score > best_score:
                best, best_score = i, score
        return best

    def _keep_supported_components(
        self,
        mask: np.ndarray,
        ownership: np.ndarray,
        fused_prob: np.ndarray,
        skin_prob: np.ndarray,
        parser_prob: np.ndarray,
        pose_prob: np.ndarray,
        box: np.ndarray,
    ) -> np.ndarray:
        num, labels, stats, cent = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
        if num <= 1:
            return mask

        main = self._main_component(labels, stats, ownership, fused_prob, skin_prob, parser_prob, pose_prob, box)
        main_area = stats[main, cv2.CC_STAT_AREA]
        main_cent = cent[main]
        out = np.zeros_like(mask)

        for i in range(1, num):
            comp = (labels == i).astype(np.uint8) * 255
            area = stats[i, cv2.CC_STAT_AREA]
            own = self._mean_prob(ownership, comp)
            f = self._mean_prob(fused_prob, comp)
            s = self._mean_prob(skin_prob, comp)
            p = self._mean_prob(parser_prob, comp)
            po = self._mean_prob(pose_prob, comp)
            b = self._mask_box_coverage(comp, box)
            dist = float(np.linalg.norm(cent[i] - main_cent) / max(mask.shape))
            keep = (
                i == main
                or (own > 0.34 and s > 0.20 and f > 0.18 and b > 0.02 and dist < 0.62 and area > max(35, main_area * 0.0012))
                or (p > 0.20 and own > 0.25 and dist < 0.65)
                or (po > 0.25 and (s > 0.16 or p > 0.16) and dist < 0.65)
                or (area > main_area * 0.025 and own > 0.40 and f > 0.38 and dist < 0.42)
            )
            if keep:
                out[labels == i] = 255
        return out

    @staticmethod
    def _reconstruct(seed: np.ndarray, allowed: np.ndarray, max_iter: int = 768) -> np.ndarray:
        seed = ((seed > 0) & (allowed > 0)).astype(np.uint8) * 255
        allowed = (allowed > 0).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cur = seed
        for _ in range(max_iter):
            nxt = cv2.dilate(cur, kernel, iterations=1)
            nxt = cv2.bitwise_and(nxt, allowed)
            if np.array_equal(nxt, cur):
                return nxt
            cur = nxt
        return cur

    def _mediapipe_connections(self) -> list[tuple[int, int]]:
        # MediaPipe PoseLandmark indices:
        # shoulders 11/12, elbows 13/14, wrists 15/16, hips 23/24,
        # knees 25/26, ankles 27/28, heels 29/30, feet 31/32,
        # nose 0, ears 7/8, mouth 9/10.
        return [
            (11, 12), (11, 23), (12, 24), (23, 24),
            (11, 13), (13, 15), (12, 14), (14, 16),
            (23, 25), (25, 27), (27, 29), (29, 31),
            (24, 26), (26, 28), (28, 30), (30, 32),
            (0, 11), (0, 12), (7, 11), (8, 12),
        ]

    def _limb_thickness(self, a: int, b: int, h: int, w: int) -> int:
        base = max(6, int(min(h, w) * 0.018))
        pair = {a, b}
        # torso/hips
        if pair in ({11, 12}, {11, 23}, {12, 24}, {23, 24}):
            return int(base * 3.3)
        # upper legs / upper arms
        if pair in ({23, 25}, {24, 26}, {11, 13}, {12, 14}):
            return int(base * 2.2)
        # lower legs / lower arms
        if pair in ({25, 27}, {26, 28}, {13, 15}, {14, 16}):
            return int(base * 1.6)
        # head/neck
        if 0 in pair or 7 in pair or 8 in pair:
            return int(base * 1.8)
        return int(base * 1.4)

    def _write_debug(self, dbg_dir: Path, stem: str, maps: dict[str, np.ndarray]) -> None:
        dbg_dir.mkdir(parents=True, exist_ok=True)
        for name, arr in maps.items():
            if arr is None:
                continue
            if arr.dtype == np.float32 or arr.dtype == np.float64:
                out = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
            else:
                out = arr
            cv2.imwrite(str(dbg_dir / f"{stem}_{name}.png"), out)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--out", default="out_bg_removed")
    parser.add_argument("--yolo", default="yolov8x-seg.pt")
    parser.add_argument("--sam2-cfg", default="configs/sam2/sam2_hiera_l.yaml")
    parser.add_argument("--sam2-ckpt", default="models/sam2/sam2_hiera_large.pt")
    parser.add_argument("--birefnet", default="ZhengPeng7/BiRefNet")
    parser.add_argument("--parser-model", default="fashn-ai/fashn-human-parser")
    parser.add_argument("--depth-model", default="depth-anything/Depth-Anything-V2-Small-hf")
    parser.add_argument("--no-sam2", action="store_true")
    parser.add_argument("--no-birefnet", action="store_true")
    parser.add_argument("--no-parser", action="store_true")
    parser.add_argument("--no-depth", action="store_true")
    parser.add_argument("--no-vitmatte", action="store_true")
    parser.add_argument("--no-pose", action="store_true")
    parser.add_argument("--fail-hard", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    remover = BackgroundRemover(
        yolo_model=args.yolo,
        sam2_cfg=None if args.no_sam2 else args.sam2_cfg,
        sam2_ckpt=None if args.no_sam2 else args.sam2_ckpt,
        birefnet_model=None if args.no_birefnet else args.birefnet,
        human_parser_model=None if args.no_parser else args.parser_model,
        depth_model=None if args.no_depth else args.depth_model,
        vitmatte_model=None if args.no_vitmatte else "hustvl/vitmatte-small-composition-1k",
        use_pose=not args.no_pose,
        fail_hard=args.fail_hard,
        threshold=args.threshold,
        debug=args.debug,
    )

    for p in args.inputs:
        remover.process(p, args.out)
