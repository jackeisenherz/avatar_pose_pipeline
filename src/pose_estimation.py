# src/pose_estimation.py
"""
ViTPose pose estimation without MMCV/MMPose.

Drop-in replacement for your existing src/pose_estimation.py.

This version uses:
- Hugging Face Transformers RT-DETR for person detection
- Hugging Face Transformers ViTPose / ViTPose++ for top-down pose
- Optional Ultralytics YOLO pose fallback

No mmcv, mmengine, mmdet, or mmpose required.

Output remains compatible with the existing pipeline:
{
  "image": "...png",
  "keypoints": [{"x": ..., "y": ..., "confidence": ...}, ...]
}

Additional metadata:
- pose_backend
- detector_model
- pose_model
- selected_person_index
- pose_quality_score
- candidate_scores
- bbox
- joint_sanity
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image


COCO17_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

COCO17_SKELETON = [
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
]

CORE_BODY_JOINTS = [5, 6, 11, 12, 13, 14, 15, 16]
TORSO_JOINTS = [5, 6, 11, 12]


class PoseEstimator:
    """
    Parameters
    ----------
    model_name:
        Backward-compatible argument. Your current main.py passes
        model_name="yolov8l-pose.pt". This class keeps that as the YOLO
        fallback model unless backend="yolo".

    backend:
        "vitpose", "yolo", or "auto".
        "vitpose" is default and uses Transformers only.
        "auto" tries ViTPose and falls back to YOLO on failure.

    vitpose_model:
        Good options:
        - "usyd-community/vitpose-base-simple"      stable, lighter
        - "usyd-community/vitpose-plus-base"        better if available
        - "usyd-community/vitpose-plus-huge"        best, heavy VRAM

    detector_model:
        RT-DETR detector used to find person boxes:
        - "PekingU/rtdetr_r50vd_coco_o365"
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        backend: str = "vitpose",
        device: Optional[str] = None,
        vitpose_model: str = "usyd-community/vitpose-base-simple",
        detector_model: str = "PekingU/rtdetr_r50vd_coco_o365",
        yolo_model: str = "yolo11x-pose.pt",
        detector_threshold: float = 0.25,
        min_joint_confidence: float = 0.10,
        prefer_alpha_main_person: bool = True,
        debug: bool = True,
        debug_dir_name: str = "_debug_pose",
        fallback_to_yolo: bool = True,
    ):
        if model_name and model_name.endswith(".pt"):
            yolo_model = model_name
        elif model_name and not model_name.endswith(".pt"):
            vitpose_model = model_name

        self.backend = backend
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.vitpose_model_name = vitpose_model
        self.detector_model_name = detector_model
        self.yolo_model_name = yolo_model
        self.detector_threshold = float(detector_threshold)
        self.min_joint_confidence = float(min_joint_confidence)
        self.prefer_alpha_main_person = bool(prefer_alpha_main_person)
        self.debug = bool(debug)
        self.debug_dir_name = debug_dir_name
        self.fallback_to_yolo = bool(fallback_to_yolo)

        self.detector_processor = None
        self.detector_model = None
        self.pose_processor = None
        self.pose_model = None
        self.yolo_model = None

        self.active_backend = None

        self._init_backend()

    # =========================================================
    # INIT
    # =========================================================

    def _init_backend(self):
        if self.backend not in {"vitpose", "yolo", "auto"}:
            raise ValueError("backend must be one of: vitpose, yolo, auto")

        if self.backend in {"vitpose", "auto"}:
            try:
                self._init_vitpose()
                self.active_backend = "vitpose"
                print(f"✓ Pose backend: ViTPose/Transformers on {self.device}")
                print(f"✓ Detector: {self.detector_model_name}")
                print(f"✓ Pose model: {self.vitpose_model_name}")
                return
            except Exception as exc:
                if self.backend == "vitpose":
                    raise
                warnings.warn(f"ViTPose init failed; falling back to YOLO. Reason: {exc}")

        self._init_yolo()
        self.active_backend = "yolo"
        print(f"✓ Pose backend: YOLO fallback on {self.device}")
        print(f"✓ YOLO model: {self.yolo_model_name}")

    def _init_vitpose(self):
        from transformers import AutoProcessor, RTDetrForObjectDetection, VitPoseForPoseEstimation

        self.detector_processor = AutoProcessor.from_pretrained(self.detector_model_name)
        self.detector_model = RTDetrForObjectDetection.from_pretrained(self.detector_model_name)
        self.detector_model.to(self.device).eval()

        self.pose_processor = AutoProcessor.from_pretrained(self.vitpose_model_name)
        self.pose_model = VitPoseForPoseEstimation.from_pretrained(self.vitpose_model_name)
        self.pose_model.to(self.device).eval()

    def _init_yolo(self):
        from ultralytics import YOLO

        self.yolo_model = YOLO(self.yolo_model_name)
        self.yolo_model.to("cuda" if self.device.startswith("cuda") else "cpu")

    # =========================================================
    # IMAGE HELPERS
    # =========================================================

    def _read_rgba_or_bgr(self, image_path: Path) -> np.ndarray:
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise RuntimeError(f"Could not read image: {image_path}")
        return image

    def _load_alpha_mask(self, image_path: Path) -> Optional[np.ndarray]:
        image = self._read_rgba_or_bgr(image_path)

        if image.ndim == 3 and image.shape[2] == 4:
            return (image[:, :, 3] > 10).astype(np.uint8)

        return None

    def _pil_rgb(self, image_path: Path) -> Image.Image:
        # PIL handles RGB conversion cleanly. If PNG has alpha, composite over black.
        image = Image.open(image_path).convert("RGBA")
        bg = Image.new("RGBA", image.size, (0, 0, 0, 255))
        image = Image.alpha_composite(bg, image)
        return image.convert("RGB")

    # =========================================================
    # DETECTION / SELECTION
    # =========================================================

    def _detect_person_boxes_vitpose(self, image: Image.Image) -> Tuple[np.ndarray, np.ndarray]:
        inputs = self.detector_processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.detector_model(**inputs)

        results = self.detector_processor.post_process_object_detection(
            outputs,
            target_sizes=torch.tensor([(image.height, image.width)], device=self.device),
            threshold=self.detector_threshold,
        )

        result = results[0]
        boxes_xyxy = result["boxes"].detach().cpu().numpy().astype(np.float32)
        labels = result["labels"].detach().cpu().numpy()
        scores = result["scores"].detach().cpu().numpy().astype(np.float32)

        # COCO person label is 0.
        keep = labels == 0
        boxes_xyxy = boxes_xyxy[keep]
        scores = scores[keep]

        return boxes_xyxy, scores

    def _fallback_alpha_box(self, alpha_mask: Optional[np.ndarray], image: Image.Image) -> Tuple[np.ndarray, np.ndarray]:
        if alpha_mask is None or alpha_mask.sum() == 0:
            w, h = image.size
            return np.array([[0, 0, w - 1, h - 1]], dtype=np.float32), np.array([0.25], dtype=np.float32)

        ys, xs = np.where(alpha_mask > 0)
        x1, x2 = float(xs.min()), float(xs.max())
        y1, y2 = float(ys.min()), float(ys.max())

        pad_x = 0.04 * max(1.0, x2 - x1)
        pad_y = 0.04 * max(1.0, y2 - y1)

        h, w = alpha_mask.shape[:2]
        box = np.array(
            [[
                max(0.0, x1 - pad_x),
                max(0.0, y1 - pad_y),
                min(float(w - 1), x2 + pad_x),
                min(float(h - 1), y2 + pad_y),
            ]],
            dtype=np.float32,
        )
        return box, np.array([0.20], dtype=np.float32)

    def _bbox_iou_with_mask(self, bbox_xyxy: np.ndarray, mask: Optional[np.ndarray]) -> float:
        if mask is None:
            return 0.0

        x1, y1, x2, y2 = bbox_xyxy.astype(int)
        h, w = mask.shape[:2]

        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h, y2))

        if x2 <= x1 or y2 <= y1:
            return 0.0

        box_mask = np.zeros_like(mask, dtype=np.uint8)
        box_mask[y1:y2, x1:x2] = 1

        inter = np.logical_and(box_mask, mask).sum()
        union = np.logical_or(box_mask, mask).sum()

        return float(inter / max(union, 1))

    def _select_main_box(
        self,
        boxes_xyxy: np.ndarray,
        detector_scores: np.ndarray,
        alpha_mask: Optional[np.ndarray],
    ) -> Tuple[int, List[Dict[str, float]]]:
        if len(boxes_xyxy) == 0:
            raise RuntimeError("No person boxes available")

        details = []
        best_idx = 0
        best_score = -1.0

        for i, box in enumerate(boxes_xyxy):
            det_score = float(detector_scores[i]) if i < len(detector_scores) else 0.0
            alpha_iou = self._bbox_iou_with_mask(box, alpha_mask)
            box_area = max(0.0, float((box[2] - box[0]) * (box[3] - box[1])))

            if alpha_mask is not None and self.prefer_alpha_main_person:
                score = 0.70 * alpha_iou + 0.25 * det_score + 0.05 * np.log1p(box_area)
            else:
                score = 0.75 * det_score + 0.25 * np.log1p(box_area)

            details.append(
                {
                    "candidate_index": int(i),
                    "detector_score": det_score,
                    "bbox_alpha_iou": float(alpha_iou),
                    "box_area": float(box_area),
                    "selection_score": float(score),
                }
            )

            if score > best_score:
                best_score = score
                best_idx = i

        return best_idx, details

    def _xyxy_to_xywh(self, boxes_xyxy: np.ndarray) -> np.ndarray:
        boxes = boxes_xyxy.copy().astype(np.float32)
        boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
        boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
        return boxes

    # =========================================================
    # VITPOSE INFERENCE
    # =========================================================

    def _infer_vitpose(self, image_path: Path, alpha_mask: Optional[np.ndarray]) -> Dict[str, Any]:
        image = self._pil_rgb(image_path)

        boxes_xyxy, detector_scores = self._detect_person_boxes_vitpose(image)

        if len(boxes_xyxy) == 0:
            boxes_xyxy, detector_scores = self._fallback_alpha_box(alpha_mask, image)

        selected_idx, candidate_scores = self._select_main_box(boxes_xyxy, detector_scores, alpha_mask)

        selected_xyxy = boxes_xyxy[selected_idx:selected_idx + 1]
        selected_xywh = self._xyxy_to_xywh(selected_xyxy)

        # Transformers ViTPose expects boxes as list per image, COCO xywh.
        inputs = self.pose_processor(
            image,
            boxes=[selected_xywh],
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            if "vitpose-plus" in self.vitpose_model_name.lower():
                dataset_index = torch.zeros(
                    (inputs["pixel_values"].shape[0],),
                    dtype=torch.long,
                    device=self.device,
                )
                try:
                    outputs = self.pose_model(**inputs, dataset_index=dataset_index)
                except TypeError:
                    outputs = self.pose_model(**inputs)
            else:
                outputs = self.pose_model(**inputs)

        pose_results = self.pose_processor.post_process_pose_estimation(
            outputs,
            boxes=[selected_xywh],
        )

        image_pose_result = pose_results[0]

        if len(image_pose_result) == 0:
            raise RuntimeError("ViTPose returned no pose results")

        pose = image_pose_result[0]

        keypoints = pose["keypoints"]
        scores = pose["scores"]

        if torch.is_tensor(keypoints):
            keypoints = keypoints.detach().cpu().numpy()

        if torch.is_tensor(scores):
            scores = scores.detach().cpu().numpy()

        keypoints = np.asarray(keypoints, dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32).reshape(-1, 1)

        kpts = np.concatenate([keypoints[:, :2], scores], axis=1)
        kpts = self._ensure_coco17(kpts)

        return {
            "backend": "vitpose_transformers",
            "detector_model": self.detector_model_name,
            "pose_model": self.vitpose_model_name,
            "keypoints": kpts,
            "bbox": selected_xyxy[0],
            "selected_person_index": int(selected_idx),
            "num_person_candidates": int(len(boxes_xyxy)),
            "candidate_scores": candidate_scores,
        }

    # =========================================================
    # YOLO FALLBACK
    # =========================================================

    def _infer_yolo(self, image_path: Path, alpha_mask: Optional[np.ndarray]) -> Dict[str, Any]:
        if self.yolo_model is None:
            self._init_yolo()

        results = self.yolo_model(
            str(image_path),
            verbose=False,
            device=self.device,
        )

        if not results or results[0].keypoints is None or len(results[0].keypoints) == 0:
            raise RuntimeError("YOLO returned no pose results")

        result = results[0]
        all_kpts = result.keypoints.data.cpu().numpy().astype(np.float32)

        boxes = None
        if result.boxes is not None and result.boxes.xyxy is not None:
            boxes = result.boxes.xyxy.cpu().numpy().astype(np.float32)

        candidate_scores = []
        best_idx = 0
        best_score = -1.0

        for i, kpts in enumerate(all_kpts):
            kpts = self._ensure_coco17(kpts)
            avg_conf = float(np.mean(kpts[:, 2]))

            if boxes is not None and i < len(boxes):
                bbox = boxes[i]
            else:
                bbox = self._bbox_from_keypoints(kpts)

            alpha_iou = self._bbox_iou_with_mask(bbox, alpha_mask) if bbox is not None else 0.0
            score = 0.65 * alpha_iou + 0.35 * avg_conf if alpha_mask is not None else avg_conf

            candidate_scores.append(
                {
                    "candidate_index": int(i),
                    "avg_confidence": avg_conf,
                    "bbox_alpha_iou": float(alpha_iou),
                    "selection_score": float(score),
                }
            )

            if score > best_score:
                best_score = score
                best_idx = i

        kpts = self._ensure_coco17(all_kpts[best_idx])
        bbox = boxes[best_idx] if boxes is not None and best_idx < len(boxes) else self._bbox_from_keypoints(kpts)

        return {
            "backend": "yolo",
            "detector_model": "yolo_pose",
            "pose_model": self.yolo_model_name,
            "keypoints": kpts,
            "bbox": bbox,
            "selected_person_index": int(best_idx),
            "num_person_candidates": int(len(all_kpts)),
            "candidate_scores": candidate_scores,
        }

    def _bbox_from_keypoints(self, keypoints: np.ndarray) -> Optional[np.ndarray]:
        valid = keypoints[:, 2] > self.min_joint_confidence
        if valid.sum() < 3:
            return None

        pts = keypoints[valid, :2]
        x1, y1 = pts.min(axis=0)
        x2, y2 = pts.max(axis=0)
        pad_x = 0.08 * max(1.0, x2 - x1)
        pad_y = 0.08 * max(1.0, y2 - y1)

        return np.array([x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y], dtype=np.float32)

    def _ensure_coco17(self, kpts: np.ndarray) -> np.ndarray:
        kpts = np.asarray(kpts, dtype=np.float32)

        if kpts.ndim != 2:
            raise ValueError("keypoints must be [N,C]")

        if kpts.shape[1] < 3:
            conf = np.ones((kpts.shape[0], 1), dtype=np.float32)
            kpts = np.concatenate([kpts[:, :2], conf], axis=1)

        kpts = kpts[:, :3]

        if kpts.shape[0] == 17:
            return kpts

        fixed = np.zeros((17, 3), dtype=np.float32)
        n = min(17, kpts.shape[0])
        fixed[:n] = kpts[:n]
        return fixed

    # =========================================================
    # QUALITY
    # =========================================================

    def _keypoint_mask_support(self, keypoints: np.ndarray, mask: Optional[np.ndarray]) -> float:
        if mask is None:
            return 0.0

        h, w = mask.shape[:2]
        valid = keypoints[:, 2] > self.min_joint_confidence

        if valid.sum() == 0:
            return 0.0

        support = 0
        total = 0

        for x, y, conf in keypoints[valid]:
            xi = int(round(float(x)))
            yi = int(round(float(y)))

            if 0 <= xi < w and 0 <= yi < h:
                support += int(mask[yi, xi] > 0)
                total += 1

        return float(support / max(total, 1))

    def _joint_sanity(self, kpts: np.ndarray) -> Dict[str, Any]:
        conf = kpts[:, 2]
        valid = conf > self.min_joint_confidence

        def limb_len(a, b):
            if not valid[a] or not valid[b]:
                return np.nan
            return float(np.linalg.norm(kpts[a, :2] - kpts[b, :2]))

        shoulder = limb_len(5, 6)
        hip = limb_len(11, 12)
        torso_l = limb_len(5, 11)
        torso_r = limb_len(6, 12)
        leg_l = limb_len(11, 13) + limb_len(13, 15)
        leg_r = limb_len(12, 14) + limb_len(14, 16)
        arm_l = limb_len(5, 7) + limb_len(7, 9)
        arm_r = limb_len(6, 8) + limb_len(8, 10)

        tests = {}

        tests["shoulder_width_valid"] = bool(np.isfinite(shoulder) and shoulder > 5.0)
        tests["hip_width_valid"] = bool(np.isfinite(hip) and hip > 5.0)

        if np.isfinite(torso_l) and np.isfinite(torso_r):
            tests["torso_balance"] = bool(min(torso_l, torso_r) / max(torso_l, torso_r, 1e-6) > 0.45)
        else:
            tests["torso_balance"] = False

        if np.isfinite(leg_l) and np.isfinite(leg_r):
            tests["leg_balance"] = bool(min(leg_l, leg_r) / max(leg_l, leg_r, 1e-6) > 0.35)
        else:
            tests["leg_balance"] = False

        if np.isfinite(arm_l) and np.isfinite(arm_r):
            tests["arm_balance"] = bool(min(arm_l, arm_r) / max(arm_l, arm_r, 1e-6) > 0.25)
        else:
            tests["arm_balance"] = False

        if valid[5] and valid[6] and valid[11] and valid[12]:
            shoulder_y = float((kpts[5, 1] + kpts[6, 1]) * 0.5)
            hip_y = float((kpts[11, 1] + kpts[12, 1]) * 0.5)
            tests["hips_below_shoulders"] = bool(hip_y > shoulder_y)
        else:
            tests["hips_below_shoulders"] = False

        passed = sum(1 for v in tests.values() if v)

        return {
            "tests": tests,
            "sanity_score": float(passed / max(len(tests), 1)),
            "lengths": {
                "shoulder_width_px": self._safe_float(shoulder),
                "hip_width_px": self._safe_float(hip),
                "torso_left_px": self._safe_float(torso_l),
                "torso_right_px": self._safe_float(torso_r),
                "leg_left_px": self._safe_float(leg_l),
                "leg_right_px": self._safe_float(leg_r),
                "arm_left_px": self._safe_float(arm_l),
                "arm_right_px": self._safe_float(arm_r),
            },
        }

    def _safe_float(self, value):
        if value is None or not np.isfinite(value):
            return None
        return float(value)

    def _quality_metadata(self, kpts: np.ndarray, alpha_mask: Optional[np.ndarray]) -> Dict[str, Any]:
        avg_conf = float(np.mean(kpts[:, 2]))
        core_conf = float(np.mean(kpts[CORE_BODY_JOINTS, 2]))
        torso_conf = float(np.mean(kpts[TORSO_JOINTS, 2]))
        support = self._keypoint_mask_support(kpts, alpha_mask)
        sanity = self._joint_sanity(kpts)

        support_term = support if alpha_mask is not None else 0.75

        quality = (
            0.30 * avg_conf +
            0.25 * core_conf +
            0.15 * torso_conf +
            0.15 * support_term +
            0.15 * sanity["sanity_score"]
        )

        quality = float(np.clip(quality, 0.0, 1.0))

        return {
            "avg_joint_confidence": avg_conf,
            "core_joint_confidence": core_conf,
            "torso_joint_confidence": torso_conf,
            "keypoint_alpha_support": support,
            "joint_sanity": sanity,
            "pose_quality_score": quality,
            "low_quality_pose": bool(quality < 0.50),
            "very_low_quality_pose": bool(quality < 0.35),
        }

    # =========================================================
    # OUTPUT / DEBUG
    # =========================================================

    def _format_output(self, image_path: Path, infer: Dict[str, Any], alpha_mask: Optional[np.ndarray]) -> Dict[str, Any]:
        kpts = infer["keypoints"]
        metadata = self._quality_metadata(kpts, alpha_mask)

        formatted = []
        for i, (x, y, conf) in enumerate(kpts):
            formatted.append(
                {
                    "name": COCO17_NAMES[i],
                    "x": float(x),
                    "y": float(y),
                    "confidence": float(conf),
                }
            )

        bbox = infer.get("bbox")
        if bbox is not None:
            bbox = np.asarray(bbox, dtype=np.float32).astype(float).tolist()

        output = {
            "image": str(image_path.name),
            "image_path": str(image_path),
            "pose_backend": infer.get("backend"),
            "detector_model": infer.get("detector_model"),
            "pose_model": infer.get("pose_model"),
            "selected_person_index": int(infer.get("selected_person_index", 0)),
            "num_person_candidates": int(infer.get("num_person_candidates", 1)),
            "bbox": bbox,
            "candidate_scores": infer.get("candidate_scores", []),
            "keypoint_format": "coco17",
            "keypoint_names": COCO17_NAMES,
            "keypoints": formatted,
        }

        output.update(metadata)
        return output

    def _save_debug_overlay(
        self,
        image_path: Path,
        output_dir: Path,
        kpts: np.ndarray,
        output: Dict[str, Any],
        alpha_mask: Optional[np.ndarray],
    ):
        if not self.debug:
            return

        debug_dir = output_dir / self.debug_dir_name
        debug_dir.mkdir(parents=True, exist_ok=True)

        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            return

        overlay = bgr.copy()

        if alpha_mask is not None:
            contours, _ = cv2.findContours(
                alpha_mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2)

        bbox = output.get("bbox")
        if bbox is not None and len(bbox) >= 4:
            x1, y1, x2, y2 = [int(round(v)) for v in bbox[:4]]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 180, 0), 2)

        for a, b in COCO17_SKELETON:
            if kpts[a, 2] <= self.min_joint_confidence or kpts[b, 2] <= self.min_joint_confidence:
                continue
            pa = tuple(np.round(kpts[a, :2]).astype(int))
            pb = tuple(np.round(kpts[b, :2]).astype(int))
            cv2.line(overlay, pa, pb, (0, 255, 0), 3)

        for i, (x, y, conf) in enumerate(kpts):
            color = (0, 0, 255) if conf < 0.35 else (255, 0, 0)
            center = (int(round(float(x))), int(round(float(y))))
            cv2.circle(overlay, center, 5, color, -1)
            cv2.putText(
                overlay,
                str(i),
                (center[0] + 5, center[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        text = (
            f"{output.get('pose_backend')} "
            f"q={output.get('pose_quality_score', 0.0):.3f} "
            f"support={output.get('keypoint_alpha_support', 0.0):.2f}"
        )

        cv2.putText(
            overlay,
            text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

        out_path = debug_dir / f"{image_path.stem}_pose_debug.png"
        cv2.imwrite(str(out_path), overlay)

    # =========================================================
    # PUBLIC API
    # =========================================================

    def process_image(self, image_path, output_dir):
        image_path = Path(image_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        alpha_mask = self._load_alpha_mask(image_path)

        try:
            if self.active_backend == "vitpose":
                infer = self._infer_vitpose(image_path, alpha_mask)
            elif self.active_backend == "yolo":
                infer = self._infer_yolo(image_path, alpha_mask)
            else:
                raise RuntimeError(f"Unknown backend: {self.active_backend}")
        except Exception as exc:
            if self.fallback_to_yolo and self.active_backend != "yolo":
                warnings.warn(f"ViTPose inference failed; falling back to YOLO. Reason: {exc}")
                self._init_yolo()
                self.active_backend = "yolo"
                infer = self._infer_yolo(image_path, alpha_mask)
                infer["fallback_reason"] = str(exc)
            else:
                raise

        output = self._format_output(image_path, infer, alpha_mask)

        out_file = output_dir / f"{image_path.stem}.json"
        with open(out_file, "w") as f:
            json.dump(output, f, indent=2)

        self._save_debug_overlay(
            image_path=image_path,
            output_dir=output_dir,
            kpts=infer["keypoints"],
            output=output,
            alpha_mask=alpha_mask,
        )

        return output


# =========================================================
# CLI
# =========================================================

def _list_images(input_path: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    if input_path.is_file():
        return [input_path]
    return sorted([p for p in input_path.iterdir() if p.suffix.lower() in exts])


def main():
    import argparse

    parser = argparse.ArgumentParser(description="ViTPose pose estimation without MMCV")
    parser.add_argument("--input", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--backend", choices=["vitpose", "yolo", "auto"], default="vitpose")
    parser.add_argument("--device", default=None)
    parser.add_argument("--vitpose-model", default="usyd-community/vitpose-base-simple")
    parser.add_argument("--detector-model", default="PekingU/rtdetr_r50vd_coco_o365")
    parser.add_argument("--yolo-model", default="yolo11x-pose.pt")
    parser.add_argument("--detector-threshold", type=float, default=0.25)
    parser.add_argument("--no-debug", action="store_true")

    args = parser.parse_args()

    estimator = PoseEstimator(
        backend=args.backend,
        device=args.device,
        vitpose_model=args.vitpose_model,
        detector_model=args.detector_model,
        yolo_model=args.yolo_model,
        detector_threshold=args.detector_threshold,
        debug=not args.no_debug,
    )

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    for image_path in _list_images(input_path):
        print(f"Pose: {image_path}")
        estimator.process_image(image_path, output_dir)


if __name__ == "__main__":
    main()
