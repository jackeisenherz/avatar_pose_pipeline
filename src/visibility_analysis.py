"""
Automatic custom-YOLO-only chest/breast visibility analyzer.

This version intentionally removes the earlier manual review/tagging UI and all
manual-label side effects. It is designed to produce more reliable JSON inputs
for the SMPL-X breast-only refinement stage by:

  * filtering detections with pose-aware chest ROI, alpha-mask support, and
    plausibility checks;
  * assigning left/right sides consistently from the pose midline;
  * deriving side-specific nipple/areola/bust/IMF evidence quality;
  * adding explicit breast-refinement gates for cleavage, landmark, projection,
    and IMF usage;
  * suppressing lateral/one-sided detections from driving cleavage/sternum shape;
  * writing richer debug overlays without requiring any GUI interaction.

Expected YOLO classes: bust, areola, nipple. The default model path is the
project-local custom detector:
    models/yolo/bust-areola-nipple-yollo11m-1280.pt
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


Point = Tuple[float, float]
BBox = Tuple[float, float, float, float]


# =============================================================================
# Utility helpers
# =============================================================================


def _as_float_pair(v: Sequence[float]) -> np.ndarray:
    return np.asarray([float(v[0]), float(v[1])], dtype=np.float32)


def _norm(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return np.zeros_like(v, dtype=np.float32)
    return (v / n).astype(np.float32)


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def _bbox_area(box: Sequence[float]) -> float:
    x1, y1, x2, y2 = map(float, box)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _bbox_center(box: Sequence[float]) -> List[float]:
    x1, y1, x2, y2 = map(float, box)
    return [(x1 + x2) * 0.5, (y1 + y2) * 0.5]


def _bbox_wh(box: Sequence[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = map(float, box)
    return max(0.0, x2 - x1), max(0.0, y2 - y1)


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    denom = _bbox_area(a) + _bbox_area(b) - inter
    return 0.0 if denom <= 1e-8 else inter / denom


def _point_poly_signed_distance(poly: np.ndarray, p: Sequence[float]) -> float:
    # OpenCV returns positive inside, zero on contour, negative outside.
    return float(cv2.pointPolygonTest(poly.astype(np.float32), (float(p[0]), float(p[1])), True))


def _point_poly_inside(poly: np.ndarray, p: Sequence[float]) -> bool:
    return _point_poly_signed_distance(poly, p) >= 0.0


def _safe_poly_bbox(poly: Sequence[Sequence[float]], w: Optional[int] = None, h: Optional[int] = None) -> Dict[str, int]:
    p = np.asarray(poly, dtype=np.float32)
    x1 = int(np.floor(float(p[:, 0].min())))
    y1 = int(np.floor(float(p[:, 1].min())))
    x2 = int(np.ceil(float(p[:, 0].max())))
    y2 = int(np.ceil(float(p[:, 1].max())))
    if w is not None:
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w, x2))
    if h is not None:
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h, y2))
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


def _rotated_rect_poly(center: np.ndarray, axis_x: np.ndarray, axis_y: np.ndarray, width: float, height: float) -> np.ndarray:
    c = np.asarray(center, dtype=np.float32)
    ax = _norm(np.asarray(axis_x, dtype=np.float32))
    ay = _norm(np.asarray(axis_y, dtype=np.float32))
    if np.linalg.norm(ax) < 1e-6:
        ax = np.array([1.0, 0.0], dtype=np.float32)
    if np.linalg.norm(ay) < 1e-6:
        ay = np.array([0.0, 1.0], dtype=np.float32)
    hx = ax * (0.5 * float(width))
    hy = ay * (0.5 * float(height))
    return np.asarray([c - hx - hy, c + hx - hy, c + hx + hy, c - hx + hy], dtype=np.float32)


def _clip_box(box: Sequence[float], w: int, h: int) -> List[float]:
    x1, y1, x2, y2 = map(float, box)
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    return [
        _clamp(x1, 0.0, max(float(w - 1), 0.0)),
        _clamp(y1, 0.0, max(float(h - 1), 0.0)),
        _clamp(x2, 0.0, float(w)),
        _clamp(y2, 0.0, float(h)),
    ]


def _box_alpha_support(alpha: np.ndarray, box: Sequence[float]) -> float:
    h, w = alpha.shape[:2]
    x1, y1, x2, y2 = _clip_box(box, w, h)
    ix1, iy1, ix2, iy2 = map(lambda z: int(round(z)), (x1, y1, x2, y2))
    ix1, iy1 = max(0, min(w - 1, ix1)), max(0, min(h - 1, iy1))
    ix2, iy2 = max(0, min(w, ix2)), max(0, min(h, iy2))
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    crop = alpha[iy1:iy2, ix1:ix2]
    return float((crop > 10).mean()) if crop.size else 0.0


def _load_rgba_bgr_alpha(path: Path) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None, None, None
    if img.ndim == 2:
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        alpha = np.ones(bgr.shape[:2], dtype=np.uint8) * 255
        rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = alpha
        return rgba, bgr, alpha
    if img.shape[2] == 4:
        return img, img[:, :, :3].copy(), img[:, :, 3].copy()
    bgr = img[:, :, :3].copy()
    alpha = np.ones(bgr.shape[:2], dtype=np.uint8) * 255
    rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = alpha
    return rgba, bgr, alpha


# =============================================================================
# Visibility analyzer
# =============================================================================


class VisibilityAnalyzer:
    """
    Automatic custom-YOLO-only visibility analyzer.

    Compatibility contract retained from the earlier analyzer:
      analyzer = VisibilityAnalyzer(...)
      result = analyzer.process(rgba_path, pose_json_path, output_dir)

    Removed features:
      * no tkinter UI;
      * no manual or review label writer;
      * no reviewed/manual flags;
      * no CLI review mode.
    """

    CLASS_IDS = {"bust": 0, "areola": 1, "nipple": 2}
    ID_TO_CLASS = {0: "bust", 1: "areola", 2: "nipple"}

    L_SHOULDER = 5
    R_SHOULDER = 6
    L_HIP = 11
    R_HIP = 12

    def __init__(
        self,
        yolo_model_path: str = "models/yolo/bust-areola-nipple-yollo11m-1280.pt",
        yolo_conf: float = 0.20,
        yolo_iou: float = 0.45,
        device: Optional[str] = None,
        debug: bool = True,
        # Kept only for backwards-compatible constructors. It is ignored.
        review_enabled: bool = False,
        review_label_subdir: str = "_review_labels",
    ) -> None:
        self.model_path = str(yolo_model_path)
        self.yolo_conf = float(yolo_conf)
        self.yolo_iou = float(yolo_iou)
        self.debug = bool(debug)
        self.review_enabled = False
        self.review_label_subdir = str(review_label_subdir)

        self.model = YOLO(self.model_path)
        if device is not None:
            self.model.to(device)

        self.class_aliases = {
            "bust": {"bust", "breast", "breasts", "boob", "boobs", "chest"},
            "areola": {"areola", "areolas"},
            "nipple": {"nipple", "nipples", "teat", "teats"},
        }

    # ------------------------------------------------------------------ public
    def process(self, rgba_path: Any, pose_json_path: Any, output_dir: Any) -> Optional[Dict[str, Any]]:
        rgba_path = Path(rgba_path)
        pose_json_path = Path(pose_json_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        rgba, bgr, alpha = _load_rgba_bgr_alpha(rgba_path)
        if rgba is None or bgr is None or alpha is None:
            return None

        try:
            pose_data = json.loads(Path(pose_json_path).read_text())
        except Exception:
            pose_data = {}

        keypoints = self._load_pose_keypoints(pose_data)
        h, w = bgr.shape[:2]

        chest_geom = self._compute_chest_geometry(keypoints, alpha, w, h)
        detections = self._run_custom_yolo(bgr)
        filtered = self._filter_detections_to_chest(detections, chest_geom, alpha, w, h)
        filtered = self._deduplicate_detections(filtered)
        sides = self._assign_side_detections(filtered, chest_geom)
        chest_analysis = self._derive_landmarks_from_boxes(sides, chest_geom, alpha)
        gates = self._compute_breast_fit_gates(chest_analysis, chest_geom)

        result = self._build_result(
            rgba_path=rgba_path,
            pose_json_path=pose_json_path,
            pose_data=pose_data,
            keypoints=keypoints,
            alpha=alpha,
            chest_geom=chest_geom,
            detections=detections,
            filtered=filtered,
            chest_analysis=chest_analysis,
            gates=gates,
        )

        json_path = output_dir / f"{rgba_path.stem}.json"
        json_path.write_text(json.dumps(result, indent=2))

        if self.debug:
            self._write_debug_images(output_dir, rgba_path.stem, bgr, alpha, chest_geom, filtered, chest_analysis, gates)

        return result

    def review_visibility_outputs(self, *args: Any, **kwargs: Any) -> bool:
        """Manual review has been removed by design."""
        print("Manual visibility review/tagging has been removed; analyzer is fully automatic.")
        return False

    # ------------------------------------------------------------------ pose / geometry
    def _load_pose_keypoints(self, pose_data: Dict[str, Any]) -> np.ndarray:
        arr: List[List[float]] = []
        kps = pose_data.get("keypoints", None)
        if kps is None and "people" in pose_data and pose_data["people"]:
            flat = pose_data["people"][0].get("pose_keypoints_2d", [])
            if flat:
                kps = [flat[i : i + 3] for i in range(0, len(flat), 3)]
        if kps is None:
            kps = []
        for kp in kps:
            if isinstance(kp, dict):
                arr.append([float(kp.get("x", 0.0)), float(kp.get("y", 0.0)), float(kp.get("confidence", kp.get("score", 0.0)))])
            elif isinstance(kp, (list, tuple, np.ndarray)) and len(kp) >= 2:
                arr.append([float(kp[0]), float(kp[1]), float(kp[2]) if len(kp) > 2 else 0.0])
        if not arr:
            return np.zeros((17, 3), dtype=np.float32)
        out = np.asarray(arr, dtype=np.float32)
        if out.shape[0] < 17:
            pad = np.zeros((17 - out.shape[0], 3), dtype=np.float32)
            out = np.concatenate([out, pad], axis=0)
        return out

    def _kp(self, keypoints: np.ndarray, idx: int) -> Tuple[np.ndarray, float]:
        if idx >= len(keypoints):
            return np.array([0.0, 0.0], dtype=np.float32), 0.0
        return keypoints[idx, :2].astype(np.float32).copy(), float(keypoints[idx, 2])

    def _compute_chest_geometry(self, keypoints: np.ndarray, alpha: np.ndarray, w: int, h: int) -> Dict[str, Any]:
        ls, lsc = self._kp(keypoints, self.L_SHOULDER)
        rs, rsc = self._kp(keypoints, self.R_SHOULDER)
        lh, lhc = self._kp(keypoints, self.L_HIP)
        rh, rhc = self._kp(keypoints, self.R_HIP)

        valid_shoulders = lsc > 0.20 and rsc > 0.20
        valid_hips = lhc > 0.20 and rhc > 0.20

        ys, xs = np.where(alpha > 10)
        if len(xs):
            sil_x1, sil_x2, sil_y1, sil_y2 = float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max())
        else:
            sil_x1, sil_x2, sil_y1, sil_y2 = 0.0, float(w - 1), 0.0, float(h - 1)
        sil_w = max(1.0, sil_x2 - sil_x1)
        sil_h = max(1.0, sil_y2 - sil_y1)

        if valid_shoulders:
            shoulder_mid = 0.5 * (ls + rs)
            shoulder_vec = ls - rs
            shoulder_width = float(np.linalg.norm(shoulder_vec))
        else:
            shoulder_mid = np.array([(sil_x1 + sil_x2) * 0.5, sil_y1 + 0.22 * sil_h], dtype=np.float32)
            shoulder_vec = np.array([max(60.0, 0.45 * sil_w), 0.0], dtype=np.float32)
            shoulder_width = float(np.linalg.norm(shoulder_vec))

        if valid_hips:
            hip_mid = 0.5 * (lh + rh)
        else:
            hip_mid = shoulder_mid + np.array([0.0, max(shoulder_width * 1.35, sil_h * 0.52)], dtype=np.float32)

        torso_vec = hip_mid - shoulder_mid
        torso_len = float(np.linalg.norm(torso_vec))
        if torso_len < 1e-4:
            torso_vec = np.array([0.0, 1.0], dtype=np.float32)
            torso_len = max(shoulder_width * 1.4, 100.0)
        torso_dir = _norm(torso_vec)

        chest_axis = _norm(shoulder_vec)
        if np.linalg.norm(chest_axis) < 1e-6:
            chest_axis = np.array([1.0, 0.0], dtype=np.float32)

        # Chest center is intentionally high enough for sternum/nipple/areola, but
        # ROI extends down to the IMF/lower-pole region.
        chest_center = shoulder_mid + torso_dir * (0.30 * torso_len)

        shoulder_ratio = shoulder_width / max(float(w), 1.0)
        if not valid_shoulders:
            view = "unknown"
            pose_quality = 0.45
        elif shoulder_ratio < 0.085:
            view = "lateral"
            pose_quality = 0.68
        elif shoulder_ratio < 0.18:
            view = "three_quarter"
            pose_quality = 0.84
        else:
            view = "frontal"
            pose_quality = 1.00

        # Pose-aware chest ROI. Frontal gets a wider two-breast ROI; lateral is
        # narrower and should not be used for cleavage fitting downstream.
        if view == "frontal":
            roi_w = max(shoulder_width * 1.28, 0.34 * sil_w, 120.0)
            roi_h = max(torso_len * 0.58, 0.34 * sil_h, 120.0)
        elif view == "three_quarter":
            roi_w = max(shoulder_width * 1.18, 0.25 * sil_w, 100.0)
            roi_h = max(torso_len * 0.62, 0.36 * sil_h, 120.0)
        else:
            roi_w = max(shoulder_width * 0.92, 0.18 * sil_w, 80.0)
            roi_h = max(torso_len * 0.68, 0.38 * sil_h, 120.0)

        roi_poly = _rotated_rect_poly(chest_center, chest_axis, torso_dir, roi_w, roi_h)
        midline_top = chest_center - torso_dir * (roi_h * 0.50)
        midline_bottom = chest_center + torso_dir * (roi_h * 0.62)

        # Subject-left side in image coordinates. If left shoulder has larger x
        # than right shoulder, subject left appears on image right.
        left_on_image = "right" if valid_shoulders and float(ls[0]) > float(rs[0]) else "left"

        return {
            "shoulder_mid": shoulder_mid.tolist(),
            "hip_mid": hip_mid.tolist(),
            "chest_center": chest_center.tolist(),
            "shoulder_width": float(shoulder_width),
            "shoulder_ratio": float(shoulder_ratio),
            "torso_length": float(torso_len),
            "chest_axis": chest_axis.tolist(),
            "torso_dir": torso_dir.tolist(),
            "roi_width": float(roi_w),
            "roi_height": float(roi_h),
            "roi_polygon": roi_poly.tolist(),
            "midline": [midline_top.tolist(), midline_bottom.tolist()],
            "view": view,
            "pose_quality": float(pose_quality),
            "left_on_image": left_on_image,
            "silhouette_bbox": {"x1": int(sil_x1), "y1": int(sil_y1), "x2": int(sil_x2), "y2": int(sil_y2)},
        }

    # ------------------------------------------------------------------ detector
    def _map_class(self, raw: str) -> Optional[str]:
        r = str(raw).strip().lower()
        for target, aliases in self.class_aliases.items():
            if r in aliases or any(a in r for a in aliases):
                return target
        # Some exported YOLO models keep numeric names.
        if r in {"0", "0.0"}:
            return "bust"
        if r in {"1", "1.0"}:
            return "areola"
        if r in {"2", "2.0"}:
            return "nipple"
        return None

    def _run_custom_yolo(self, bgr: np.ndarray) -> List[Dict[str, Any]]:
        result = self.model.predict(source=bgr, conf=self.yolo_conf, iou=self.yolo_iou, verbose=False)
        if not result:
            return []
        r = result[0]
        if getattr(r, "boxes", None) is None:
            return []
        names = r.names if hasattr(r, "names") else {}
        xyxy = r.boxes.xyxy.detach().cpu().numpy()
        conf = r.boxes.conf.detach().cpu().numpy()
        cls = r.boxes.cls.detach().cpu().numpy().astype(int)
        h, w = bgr.shape[:2]

        out: List[Dict[str, Any]] = []
        for i, row in enumerate(xyxy):
            raw_name = str(names.get(int(cls[i]), str(int(cls[i])))).lower()
            cls_name = self._map_class(raw_name)
            if cls_name is None:
                continue
            x1, y1, x2, y2 = _clip_box(row[:4], w, h)
            bw, bh = _bbox_wh((x1, y1, x2, y2))
            if bw < 2 or bh < 2:
                continue
            out.append({
                "class_id": self.CLASS_IDS[cls_name],
                "class_raw": raw_name,
                "class_name": cls_name,
                "confidence": float(conf[i]),
                "bbox_xyxy": [x1, y1, x2, y2],
                "center": _bbox_center((x1, y1, x2, y2)),
                "width": float(bw),
                "height": float(bh),
                "area": float(bw * bh),
            })
        return out

    def _filter_detections_to_chest(
        self,
        detections: List[Dict[str, Any]],
        chest_geom: Dict[str, Any],
        alpha: np.ndarray,
        w: int,
        h: int,
    ) -> List[Dict[str, Any]]:
        poly = np.asarray(chest_geom["roi_polygon"], dtype=np.float32)
        roi_w = float(chest_geom.get("roi_width", 1.0))
        roi_h = float(chest_geom.get("roi_height", 1.0))
        view = str(chest_geom.get("view", "unknown"))
        center = np.asarray(chest_geom["chest_center"], dtype=np.float32)
        torso_dir = _norm(np.asarray(chest_geom["torso_dir"], dtype=np.float32))
        chest_axis = _norm(np.asarray(chest_geom["chest_axis"], dtype=np.float32))

        filtered: List[Dict[str, Any]] = []
        for det in detections:
            cls_name = det["class_name"]
            cx, cy = det["center"]
            point = np.asarray([cx, cy], dtype=np.float32)
            signed_dist = _point_poly_signed_distance(poly, point)
            inside = signed_dist >= 0.0

            # Bust boxes can legitimately straddle the ROI edge; nipple/areola
            # must be inside or close to the ROI.
            margin = max(roi_w, roi_h) * (0.24 if cls_name == "bust" else 0.10)
            if not inside and abs(signed_dist) > margin:
                continue

            alpha_support = _box_alpha_support(alpha, det["bbox_xyxy"])
            if alpha_support < (0.015 if cls_name == "bust" else 0.035):
                continue

            bw, bh = float(det["width"]), float(det["height"])
            rel_w = bw / max(roi_w, 1.0)
            rel_h = bh / max(roi_h, 1.0)

            # Class-size plausibility. The goal is not strict detection purity;
            # it is to prevent huge or tiny spurious boxes from entering the fit.
            plausible = True
            if cls_name == "nipple":
                plausible = 0.008 <= rel_w <= 0.22 and 0.008 <= rel_h <= 0.22
            elif cls_name == "areola":
                plausible = 0.025 <= rel_w <= 0.36 and 0.025 <= rel_h <= 0.36
            elif cls_name == "bust":
                plausible = 0.12 <= rel_w <= 1.75 and 0.12 <= rel_h <= 1.85
            if not plausible:
                continue

            rel = point - center
            u = float(np.dot(rel, chest_axis) / max(roi_w * 0.5, 1.0))
            v = float(np.dot(rel, torso_dir) / max(roi_h * 0.5, 1.0))

            d = dict(det)
            d["alpha_support"] = float(alpha_support)
            d["roi_signed_distance"] = float(signed_dist)
            d["roi_u"] = float(u)
            d["roi_v"] = float(v)
            d["pose_view"] = view
            d["quality"] = float(det["confidence"]) * (0.50 + 0.50 * float(alpha_support))
            filtered.append(d)

        return filtered

    def _deduplicate_detections(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Per-class NMS with a conservative threshold. YOLO already performs NMS,
        # but this catches post-filter duplicates and overlapping aliases.
        out: List[Dict[str, Any]] = []
        for cls_name in ["nipple", "areola", "bust"]:
            ds = [d for d in detections if d["class_name"] == cls_name]
            ds = sorted(ds, key=lambda d: (float(d.get("quality", d.get("confidence", 0.0))), float(d.get("alpha_support", 0.0))), reverse=True)
            kept: List[Dict[str, Any]] = []
            thr = 0.35 if cls_name != "bust" else 0.45
            for d in ds:
                if all(_iou(d["bbox_xyxy"], k["bbox_xyxy"]) < thr for k in kept):
                    kept.append(d)
            out.extend(kept)
        return out

    # ------------------------------------------------------------------ side assignment
    def _side_of_point_image(self, point: Sequence[float], chest_geom: Dict[str, Any]) -> str:
        p = np.asarray(point, dtype=np.float32)
        a = np.asarray(chest_geom["midline"][0], dtype=np.float32)
        b = np.asarray(chest_geom["midline"][1], dtype=np.float32)
        v = b - a
        w = p - a
        cross = float(v[0] * w[1] - v[1] * w[0])
        return "left" if cross > 0 else "right"

    def _subject_side_from_image_side(self, image_side: str, chest_geom: Dict[str, Any]) -> str:
        left_on_image = str(chest_geom.get("left_on_image", "left"))
        if left_on_image == "left":
            return image_side
        return "left" if image_side == "right" else "right"

    def _assign_side_detections(self, filtered: List[Dict[str, Any]], chest_geom: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        sides = {"left": [], "right": [], "image_left": [], "image_right": []}
        for det in filtered:
            img_side = self._side_of_point_image(det["center"], chest_geom)
            subj_side = self._subject_side_from_image_side(img_side, chest_geom)
            d = dict(det)
            d["side_image"] = img_side
            d["side_subject"] = subj_side
            sides[subj_side].append(d)
            sides[f"image_{img_side}"].append(d)
        for k in list(sides.keys()):
            sides[k] = sorted(sides[k], key=lambda d: (float(d.get("quality", d.get("confidence", 0.0))), float(d.get("alpha_support", 0.0))), reverse=True)
        return sides

    def _best_det(self, dets: Sequence[Dict[str, Any]], name: str) -> Optional[Dict[str, Any]]:
        ds = [d for d in dets if d["class_name"] == name]
        if not ds:
            return None
        return max(ds, key=lambda d: (float(d.get("quality", d.get("confidence", 0.0))), float(d.get("alpha_support", 0.0))))

    # ------------------------------------------------------------------ landmark derivation
    def _empty_side(self) -> Dict[str, Any]:
        return {
            "visible": False,
            "nipple": None,
            "nipple_confidence": 0.0,
            "areola_center": None,
            "areola_confidence": 0.0,
            "areola_diameter_px": None,
            "bust_bbox": None,
            "bust_confidence": 0.0,
            "imf_curve": None,
            "imf_confidence": 0.0,
            "source": None,
            "quality": 0.0,
            "evidence": {
                "has_nipple": False,
                "has_areola": False,
                "has_bust": False,
                "has_imf": False,
            },
        }

    def _derive_landmarks_from_boxes(self, sides: Dict[str, List[Dict[str, Any]]], chest_geom: Dict[str, Any], alpha: np.ndarray) -> Dict[str, Any]:
        result = {
            "view": chest_geom["view"],
            "left_visible": False,
            "right_visible": False,
            "left": self._empty_side(),
            "right": self._empty_side(),
            "sternum": self._derive_sternum(chest_geom),
            "sources": {
                "primary": "custom_yolo_only",
                "deepnipple_used": False,
                "classical_used": False,
                "manual_review_used": False,
                "reviewable": False,
            },
        }
        for side in ["left", "right"]:
            side_res = self._derive_one_side(side, sides.get(side, []), chest_geom, alpha)
            result[side] = side_res
            result[f"{side}_visible"] = bool(side_res["visible"])

        result["both_sides_visible"] = bool(result["left_visible"] and result["right_visible"])
        result["quality"] = float((result["left"]["quality"] + result["right"]["quality"]) * 0.5)
        result["cleavage_evidence_confidence"] = self._cleavage_evidence_confidence(result, chest_geom)
        return result

    def _derive_one_side(self, side: str, dets: Sequence[Dict[str, Any]], chest_geom: Dict[str, Any], alpha: np.ndarray) -> Dict[str, Any]:
        bust = self._best_det(dets, "bust")
        areola = self._best_det(dets, "areola")
        nipple = self._best_det(dets, "nipple")
        out = self._empty_side()
        if bust is None and areola is None and nipple is None:
            return out

        out["visible"] = True
        if bust is not None:
            out["bust_bbox"] = [float(x) for x in bust["bbox_xyxy"]]
            out["bust_confidence"] = float(bust["confidence"])
            out["evidence"]["has_bust"] = True

        if areola is not None:
            out["areola_center"] = [float(areola["center"][0]), float(areola["center"][1])]
            out["areola_confidence"] = float(areola["confidence"])
            diameter = 0.5 * (float(areola["width"]) + float(areola["height"]))
            out["areola_diameter_px"] = float(diameter)
            out["evidence"]["has_areola"] = True

        if nipple is not None:
            out["nipple"] = [float(nipple["center"][0]), float(nipple["center"][1])]
            out["nipple_confidence"] = float(nipple["confidence"])
            out["source"] = "nipple_box"
            out["evidence"]["has_nipple"] = True
        elif areola is not None:
            out["nipple"] = [float(areola["center"][0]), float(areola["center"][1])]
            out["nipple_confidence"] = float(max(0.0, areola["confidence"] - 0.07))
            out["source"] = "areola_center"

        if out["areola_diameter_px"] is None and nipple is not None:
            out["areola_diameter_px"] = float(max(float(nipple["width"]), float(nipple["height"])) * 2.2)

        imf_curve, imf_conf = self._derive_imf_from_bust(bust, alpha, chest_geom)
        out["imf_curve"] = imf_curve
        out["imf_confidence"] = float(imf_conf)
        out["evidence"]["has_imf"] = bool(imf_curve is not None and imf_conf > 0.0)

        out["quality"] = self._side_quality(out, chest_geom)
        return out

    def _derive_sternum(self, chest_geom: Dict[str, Any]) -> Dict[str, Any]:
        a = np.asarray(chest_geom["midline"][0], dtype=np.float32)
        b = np.asarray(chest_geom["midline"][1], dtype=np.float32)
        c = 0.5 * (a + b)
        view = str(chest_geom.get("view", "unknown"))
        conf = 0.90 if view == "frontal" else 0.72 if view == "three_quarter" else 0.45
        return {"line": [a.tolist(), b.tolist()], "center": c.tolist(), "confidence": float(conf)}

    def _derive_imf_from_bust(
        self,
        bust: Optional[Dict[str, Any]],
        alpha: np.ndarray,
        chest_geom: Dict[str, Any],
    ) -> Tuple[Optional[List[List[float]]], float]:
        if bust is None:
            return None, 0.0
        x1, y1, x2, y2 = [int(round(v)) for v in bust["bbox_xyxy"]]
        h, w = alpha.shape[:2]
        x1, y1 = max(0, min(w - 1, x1)), max(0, min(h - 1, y1))
        x2, y2 = max(0, min(w, x2)), max(0, min(h, y2))
        if x2 <= x1 or y2 <= y1:
            return None, 0.0
        crop = alpha[y1:y2, x1:x2] > 10
        if crop.sum() < 20:
            return None, 0.0

        # Bottom envelope by vertical slices. Use a high percentile instead of
        # max to reduce isolated hair/mask spikes.
        pts: List[List[float]] = []
        n_bins = 32
        bin_edges = np.linspace(0, crop.shape[1], n_bins + 1)
        for i in range(n_bins):
            lo = int(max(0, math.floor(bin_edges[i])))
            hi = int(min(crop.shape[1], math.ceil(bin_edges[i + 1])))
            if hi <= lo:
                continue
            sl = crop[:, lo:hi]
            ys, _ = np.where(sl)
            if len(ys) >= 3:
                bx = x1 + 0.5 * (lo + hi)
                by = y1 + float(np.percentile(ys, 90.0))
                pts.append([float(bx), float(by)])
        if len(pts) < 5:
            return None, 0.0

        pts_np = np.asarray(pts, dtype=np.float32)
        x = pts_np[:, 0]
        y = pts_np[:, 1]
        x0 = float(x.mean())
        try:
            coeff = np.polyfit(x - x0, y, 2)
            xx = np.linspace(float(x.min()), float(x.max()), 32)
            yy = np.polyval(coeff, xx - x0)
            curve = np.stack([xx, yy], axis=1).astype(np.float32)
            roughness = float(np.mean(np.abs(y - np.polyval(coeff, x - x0)))) / max(float(y2 - y1), 1.0)
            conf = float(bust.get("confidence", 0.0)) * _clamp(1.0 - 2.0 * roughness, 0.45, 1.0)
            return curve.tolist(), _clamp(conf, 0.0, 0.90)
        except Exception:
            return pts_np.tolist(), float(min(0.70, bust.get("confidence", 0.0)))

    def _side_quality(self, side: Dict[str, Any], chest_geom: Dict[str, Any]) -> float:
        view = str(chest_geom.get("view", "unknown"))
        view_w = 1.0 if view == "frontal" else 0.75 if view == "three_quarter" else 0.45
        q = 0.0
        q += 0.30 * float(side.get("bust_confidence", 0.0))
        q += 0.30 * float(side.get("imf_confidence", 0.0))
        q += 0.22 * float(side.get("areola_confidence", 0.0))
        q += 0.18 * float(side.get("nipple_confidence", 0.0))
        return float(_clamp(q * view_w, 0.0, 1.0))

    def _cleavage_evidence_confidence(self, chest: Dict[str, Any], chest_geom: Dict[str, Any]) -> float:
        view = str(chest_geom.get("view", "unknown"))
        if view != "frontal":
            return 0.0
        if not (chest.get("left_visible") and chest.get("right_visible")):
            return 0.0
        l = chest["left"]
        r = chest["right"]
        ql = float(l.get("quality", 0.0))
        qr = float(r.get("quality", 0.0))
        # Cleavage needs both sides and at least one reliable nipple/areola pair.
        landmark_pair = min(
            max(float(l.get("nipple_confidence", 0.0)), float(l.get("areola_confidence", 0.0))),
            max(float(r.get("nipple_confidence", 0.0)), float(r.get("areola_confidence", 0.0))),
        )
        bust_pair = min(float(l.get("bust_confidence", 0.0)), float(r.get("bust_confidence", 0.0)))
        imf_pair = min(float(l.get("imf_confidence", 0.0)), float(r.get("imf_confidence", 0.0)))
        conf = 0.35 * min(ql, qr) + 0.30 * landmark_pair + 0.20 * bust_pair + 0.15 * imf_pair
        return float(_clamp(conf, 0.0, 1.0))

    # ------------------------------------------------------------------ gates for optimizer
    def _compute_breast_fit_gates(self, chest: Dict[str, Any], chest_geom: Dict[str, Any]) -> Dict[str, Any]:
        view = str(chest_geom.get("view", "unknown"))
        left = chest["left"]
        right = chest["right"]
        both = bool(chest.get("left_visible") and chest.get("right_visible"))

        left_landmark = max(float(left.get("nipple_confidence", 0.0)), float(left.get("areola_confidence", 0.0)))
        right_landmark = max(float(right.get("nipple_confidence", 0.0)), float(right.get("areola_confidence", 0.0)))
        landmark_conf = max(left_landmark, right_landmark)
        both_landmark_conf = min(left_landmark, right_landmark)
        bust_conf = max(float(left.get("bust_confidence", 0.0)), float(right.get("bust_confidence", 0.0)))
        imf_conf = max(float(left.get("imf_confidence", 0.0)), float(right.get("imf_confidence", 0.0)))
        both_bust_conf = min(float(left.get("bust_confidence", 0.0)), float(right.get("bust_confidence", 0.0)))
        cleavage_conf = float(chest.get("cleavage_evidence_confidence", 0.0))

        frontal = view == "frontal"
        three_q = view == "three_quarter"
        lateral = view == "lateral"

        # These are intentionally encoded in JSON so the optimizer can use them
        # without re-deriving visibility rules.
        use_cleavage = frontal and both and cleavage_conf >= 0.45 and both_landmark_conf >= 0.50 and both_bust_conf >= 0.45
        use_sternum = frontal and both and cleavage_conf >= 0.35
        use_landmarks = (frontal or three_q) and landmark_conf >= 0.50
        use_projection = bust_conf >= 0.35 and (frontal or three_q or lateral)
        use_imf = imf_conf >= 0.45 and (frontal or three_q or lateral)
        use_side_specific_only = not use_cleavage and (use_landmarks or use_projection or use_imf)

        return {
            "view": view,
            "use_for_breast_refine": bool(use_cleavage or use_landmarks or use_projection or use_imf),
            "use_for_cleavage": bool(use_cleavage),
            "use_for_sternum_valley": bool(use_sternum),
            "use_for_landmarks": bool(use_landmarks),
            "use_for_projection": bool(use_projection),
            "use_for_imf": bool(use_imf),
            "side_specific_only": bool(use_side_specific_only),
            "ignore_for_cleavage_reason": None if use_cleavage else self._cleavage_reject_reason(view, both, cleavage_conf, both_landmark_conf, both_bust_conf),
            "weights": {
                "cleavage": float(_clamp((cleavage_conf - 0.35) / 0.45, 0.0, 1.0)) if use_cleavage else 0.0,
                "sternum": float(_clamp((cleavage_conf - 0.25) / 0.50, 0.0, 1.0)) if use_sternum else 0.0,
                "landmarks": float(_clamp((landmark_conf - 0.35) / 0.45, 0.0, 1.0)) if use_landmarks else 0.0,
                "projection": float(_clamp((bust_conf - 0.25) / 0.50, 0.0, 1.0)) if use_projection else 0.0,
                "imf": float(_clamp((imf_conf - 0.30) / 0.50, 0.0, 1.0)) if use_imf else 0.0,
            },
            "confidence": {
                "left_landmark": float(left_landmark),
                "right_landmark": float(right_landmark),
                "both_landmark": float(both_landmark_conf),
                "bust": float(bust_conf),
                "both_bust": float(both_bust_conf),
                "imf": float(imf_conf),
                "cleavage": float(cleavage_conf),
            },
        }

    def _cleavage_reject_reason(self, view: str, both: bool, cleavage_conf: float, both_landmark: float, both_bust: float) -> str:
        if view != "frontal":
            return f"not_frontal:{view}"
        if not both:
            return "not_both_sides_visible"
        if both_landmark < 0.50:
            return "weak_bilateral_nipple_areola"
        if both_bust < 0.45:
            return "weak_bilateral_bust"
        if cleavage_conf < 0.45:
            return "weak_cleavage_evidence"
        return "unknown"

    # ------------------------------------------------------------------ output
    def _build_result(
        self,
        rgba_path: Path,
        pose_json_path: Path,
        pose_data: Dict[str, Any],
        keypoints: np.ndarray,
        alpha: np.ndarray,
        chest_geom: Dict[str, Any],
        detections: List[Dict[str, Any]],
        filtered: List[Dict[str, Any]],
        chest_analysis: Dict[str, Any],
        gates: Dict[str, Any],
    ) -> Dict[str, Any]:
        visible = keypoints[:, 2] > 0.5 if len(keypoints) else np.zeros(0, dtype=bool)
        avg_conf = float(keypoints[:, 2][visible].mean()) if visible.any() else 0.0
        silhouette_area = float((alpha > 10).mean())
        h, w = alpha.shape[:2]
        hip_visible = float((len(keypoints) > self.R_HIP) and (keypoints[self.L_HIP, 2] > 0.5) and (keypoints[self.R_HIP, 2] > 0.5))
        chest_visible = float(chest_analysis["left_visible"] or chest_analysis["right_visible"])
        quality_score = float(_clamp(avg_conf * float(chest_geom["pose_quality"]) * min(silhouette_area * 2.5, 1.0), 0.05, 1.0))
        image_weight = float(_clamp(avg_conf * (1.0 + float(chest_geom["pose_quality"])) * (0.5 + silhouette_area), 0.05, 2.5))

        return {
            "image": rgba_path.name,
            "image_path": str(rgba_path),
            "pose_json_path": str(pose_json_path),
            "model": {
                "type": "custom_yolo_only_auto_v2_no_manual_review",
                "path": self.model_path,
                "conf": self.yolo_conf,
                "iou": self.yolo_iou,
            },
            "crop_type": "torso",
            "image_weight": image_weight,
            "quality_score": quality_score,
            "visible_ratio": silhouette_area,
            "width_ratio": float(chest_geom["roi_width"] / max(w, 1)),
            "silhouette_area": silhouette_area,
            "visible_joints": int(visible.sum()),
            "avg_joint_confidence": avg_conf,
            "bbox": _safe_poly_bbox(chest_geom["roi_polygon"], w=w, h=h),
            "pose_view": chest_geom["view"],
            "pose_quality": chest_geom["pose_quality"],
            "chest_visible": chest_visible,
            "hip_visible": hip_visible,
            "chest": chest_analysis,
            "chest_geometry": chest_geom,
            "breast_fit_gates": gates,
            "detections": {
                "raw_count": len(detections),
                "filtered_count": len(filtered),
                "raw": detections,
                "filtered": filtered,
            },
            "review": {
                "enabled": False,
                "verified_by_user": False,
                "manual_review_removed": True,
            },
        }

    # ------------------------------------------------------------------ debug
    def _write_debug_images(
        self,
        output_dir: Path,
        stem: str,
        bgr: np.ndarray,
        alpha: np.ndarray,
        chest_geom: Dict[str, Any],
        boxes: List[Dict[str, Any]],
        chest: Dict[str, Any],
        gates: Dict[str, Any],
    ) -> None:
        dbg_dir = Path(output_dir) / "debug"
        dbg_dir.mkdir(parents=True, exist_ok=True)

        img = bgr.copy()
        poly = np.asarray(chest_geom["roi_polygon"], dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [poly], True, (255, 255, 0), 2)

        a = tuple(map(int, map(round, chest_geom["midline"][0])))
        b = tuple(map(int, map(round, chest_geom["midline"][1])))
        cv2.line(img, a, b, (0, 255, 255), 2)

        color_map = {"bust": (0, 255, 0), "areola": (0, 165, 255), "nipple": (0, 0, 255)}
        for d in boxes:
            x1, y1, x2, y2 = [int(round(v)) for v in d["bbox_xyxy"]]
            c = color_map.get(d["class_name"], (255, 255, 255))
            cv2.rectangle(img, (x1, y1), (x2, y2), c, 2)
            cv2.putText(
                img,
                f"{d['class_name']} {float(d.get('confidence', 0.0)):.2f}",
                (x1, max(12, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                c,
                1,
                cv2.LINE_AA,
            )

        for side, col in [("left", (0, 255, 255)), ("right", (255, 0, 255))]:
            s = chest[side]
            if s.get("nipple") is not None:
                x, y = s["nipple"]
                cv2.circle(img, (int(round(x)), int(round(y))), 6, col, -1)
                cv2.putText(img, f"{side} apex", (int(round(x)) + 8, int(round(y))), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
            if s.get("areola_center") is not None:
                x, y = s["areola_center"]
                r = max(3, int(round(float(s.get("areola_diameter_px") or 8) * 0.5)))
                cv2.circle(img, (int(round(x)), int(round(y))), r, col, 2)
            if s.get("imf_curve") is not None:
                pts = np.asarray(s["imf_curve"], dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(img, [pts], False, col, 2)

        txt = (
            f"view={gates['view']} cleav={int(gates['use_for_cleavage'])} "
            f"land={int(gates['use_for_landmarks'])} proj={int(gates['use_for_projection'])} "
            f"imf={int(gates['use_for_imf'])}"
        )
        cv2.putText(img, txt, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(img, txt, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)

        cv2.imwrite(str(dbg_dir / f"{stem}_overlay.png"), img)

        # Save a compact alpha/ROI candidate diagnostic as well.
        mask_vis = cv2.cvtColor(alpha, cv2.COLOR_GRAY2BGR)
        cv2.polylines(mask_vis, [poly], True, (255, 255, 0), 2)
        cv2.line(mask_vis, a, b, (0, 255, 255), 2)
        cv2.imwrite(str(dbg_dir / f"{stem}_mask_roi.png"), mask_vis)


# =============================================================================
# Optional CLI helper
# =============================================================================


def _discover_images(input_dir: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
    return sorted([p for p in input_dir.iterdir() if p.suffix.lower() in exts])


def _candidate_pose_path(pose_dir: Path, image_path: Path) -> Path:
    return pose_dir / f"{image_path.stem}.json"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Automatic custom-YOLO-only breast/chest visibility analyzer. No manual UI.")
    parser.add_argument("--images", type=str, required=False, help="Folder containing normalized RGBA/RGB images.")
    parser.add_argument("--pose", type=str, required=False, help="Folder containing matching pose JSON files.")
    parser.add_argument("--output", type=str, required=False, help="Output visibility folder.")
    parser.add_argument("--model", type=str, default="models/yolo/bust-areola-nipple-yollo11m-1280.pt")
    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--no-debug", action="store_true")
    args = parser.parse_args()

    if not args.images or not args.pose or not args.output:
        print("Provide --images, --pose, and --output to run batch analysis.")
        raise SystemExit(0)

    analyzer = VisibilityAnalyzer(
        yolo_model_path=args.model,
        yolo_conf=args.conf,
        yolo_iou=args.iou,
        debug=not args.no_debug,
    )
    images = _discover_images(Path(args.images))
    output_dir = Path(args.output)
    pose_dir = Path(args.pose)
    output_dir.mkdir(parents=True, exist_ok=True)

    ok = 0
    for img in images:
        pose_json = _candidate_pose_path(pose_dir, img)
        if not pose_json.exists():
            print(f"Skipping {img.name}: missing pose JSON {pose_json}")
            continue
        res = analyzer.process(img, pose_json, output_dir)
        ok += int(res is not None)
        if res is not None:
            gates = res.get("breast_fit_gates", {})
            print(
                f"{img.name}: view={res.get('pose_view')} "
                f"cleavage={int(gates.get('use_for_cleavage', False))} "
                f"landmark={int(gates.get('use_for_landmarks', False))} "
                f"projection={int(gates.get('use_for_projection', False))} "
                f"imf={int(gates.get('use_for_imf', False))}"
            )
    print(f"Processed {ok}/{len(images)} images.")
