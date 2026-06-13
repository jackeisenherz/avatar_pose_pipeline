#!/usr/bin/env python3
"""
Extract body, bust, chest-landmark and body-ratio measurements from canonical/refined SMPL-X NPZ.

Input:
    data/output/08_smplx/canonical_body.npz
    or
    data/output/09_refined/refined_body.npz

Optional:
    visibility JSON directory produced by the improved VisibilityAnalyzer.
    These JSONs may include:
        chest_analysis.left_nipple
        chest_analysis.right_nipple
        chest_analysis.left_areola_diameter_px
        chest_analysis.right_areola_diameter_px
        chest_analysis.left_imf_curve
        chest_analysis.right_imf_curve
        chest_analysis.sternum_midline
        chest_analysis.cleavage

Height:
    --height-cm is optional. If omitted, 165 cm is assumed.

Output:
    JSON with requested measurements.

Important:
    Bust-specific values are geometric estimates from the fitted mesh and/or
    fused image-derived chest landmarks. They are not clinical measurements
    unless the mesh and landmarks are scan-grade.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple, Optional, Any, Iterable, List

import numpy as np


# ============================================================
# BASIC GEOMETRY
# ============================================================

def _as_vertices_faces(npz_path: Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    data = np.load(npz_path, allow_pickle=True)
    payload = {k: data[k] for k in data.files}

    if "vertices" not in payload or "faces" not in payload:
        raise RuntimeError(f"NPZ must contain vertices and faces: {npz_path}")

    vertices = payload["vertices"]
    faces = payload["faces"]

    while vertices.ndim > 2:
        vertices = vertices[0]

    while faces.ndim > 2:
        faces = faces[0]

    return vertices.astype(np.float64), faces.astype(np.int64), payload


def _scale_vertices_to_cm(vertices: np.ndarray, known_height_cm: float) -> Tuple[np.ndarray, float]:
    y = vertices[:, 1]
    mesh_height = float(y.max() - y.min())

    if mesh_height <= 1e-8:
        raise RuntimeError("Invalid mesh height")

    scale = float(known_height_cm) / mesh_height
    return vertices * scale, scale


def _triangle_area(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return float(0.5 * np.linalg.norm(np.cross(b - a, c - a)))


def _surface_area(vertices: np.ndarray, faces: np.ndarray, vertex_mask: np.ndarray) -> float:
    if vertex_mask.sum() == 0:
        return 0.0

    fmask = (
        vertex_mask[faces[:, 0]] &
        vertex_mask[faces[:, 1]] &
        vertex_mask[faces[:, 2]]
    )

    selected = faces[fmask]

    total = 0.0
    for tri in selected:
        total += _triangle_area(vertices[tri[0]], vertices[tri[1]], vertices[tri[2]])

    return float(total)


def _convex_hull_2d(points: np.ndarray) -> np.ndarray:
    points = np.unique(points, axis=0)

    if len(points) <= 2:
        return points

    points = points[np.lexsort((points[:, 1], points[:, 0]))]

    def cross(o, a, b):
        return (
            (a[0] - o[0]) * (b[1] - o[1]) -
            (a[1] - o[1]) * (b[0] - o[0])
        )

    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in points[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return np.array(lower[:-1] + upper[:-1], dtype=np.float64)


def _perimeter(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0

    hull = _convex_hull_2d(points)

    if len(hull) < 3:
        return 0.0

    shifted = np.roll(hull, -1, axis=0)
    return float(np.linalg.norm(shifted - hull, axis=1).sum())


def _cross_section_points_y(vertices: np.ndarray, faces: np.ndarray, y_plane: float) -> np.ndarray:
    """
    Intersect mesh with horizontal plane y=y_plane.
    Return 2D x/z intersection points.
    """
    points = []

    for tri in faces:
        verts = vertices[tri]
        ys = verts[:, 1]

        if (ys.min() > y_plane) or (ys.max() < y_plane):
            continue

        for a, b in [(0, 1), (1, 2), (2, 0)]:
            y0 = ys[a]
            y1 = ys[b]

            if abs(y0 - y1) < 1e-10:
                continue

            if (y_plane - y0) * (y_plane - y1) <= 0.0:
                t = (y_plane - y0) / (y1 - y0)

                if 0.0 <= t <= 1.0:
                    p = verts[a] + t * (verts[b] - verts[a])
                    points.append([p[0], p[2]])

    if not points:
        return np.zeros((0, 2), dtype=np.float64)

    return np.asarray(points, dtype=np.float64)


def _circumference_at_y(vertices: np.ndarray, faces: np.ndarray, y_plane: float) -> float:
    pts = _cross_section_points_y(vertices, faces, y_plane)
    return _perimeter(pts)


def _best_circumference_in_band(
    vertices: np.ndarray,
    faces: np.ndarray,
    y_min: float,
    y_max: float,
    mode: str,
    samples: int = 48,
) -> Tuple[float, float]:
    best_val = None
    best_y = None
    for y in np.linspace(y_min, y_max, samples):
        c = _circumference_at_y(vertices, faces, float(y))
        if c <= 0:
            continue
        if best_val is None:
            best_val = c
            best_y = float(y)
        elif mode == "max" and c > best_val:
            best_val = c
            best_y = float(y)
        elif mode == "min" and c < best_val:
            best_val = c
            best_y = float(y)
    if best_val is None:
        return 0.0, float(0.5 * (y_min + y_max))
    return float(best_val), float(best_y)


def _fit_plane_xy_to_z(points: np.ndarray) -> Tuple[float, float, float]:
    """
    Fit z = ax + by + c.
    """
    if len(points) < 3:
        return 0.0, 0.0, float(points[:, 2].mean()) if len(points) else 0.0

    A = np.stack([points[:, 0], points[:, 1], np.ones(len(points))], axis=1)
    b = points[:, 2]

    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    return float(coef[0]), float(coef[1]), float(coef[2])


def _plane_eval(coef: Tuple[float, float, float], x: np.ndarray, y: np.ndarray) -> np.ndarray:
    a, b, c = coef
    return a * x + b * y + c


# ============================================================
# JSON / LANDMARK HELPERS
# ============================================================

def _safe_float(v: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if v is None:
            return default
        x = float(v)
        if not np.isfinite(x):
            return default
        return x
    except Exception:
        return default


def _weighted_mean(values: List[float], weights: List[float]) -> Optional[float]:
    pairs = [(float(v), max(float(w), 1e-6)) for v, w in zip(values, weights) if v is not None and np.isfinite(v)]
    if not pairs:
        return None
    vals = np.asarray([p[0] for p in pairs], dtype=np.float64)
    ww = np.asarray([p[1] for p in pairs], dtype=np.float64)
    return float((vals * ww).sum() / max(float(ww.sum()), 1e-8))


def _weighted_median(values: List[float], weights: List[float]) -> Optional[float]:
    pairs = [(float(v), max(float(w), 1e-6)) for v, w in zip(values, weights) if v is not None and np.isfinite(v)]
    if not pairs:
        return None
    vals = np.asarray([p[0] for p in pairs], dtype=np.float64)
    ww = np.asarray([p[1] for p in pairs], dtype=np.float64)
    idx = np.argsort(vals)
    vals = vals[idx]
    ww = ww[idx]
    cum = np.cumsum(ww)
    cutoff = 0.5 * ww.sum()
    return float(vals[np.searchsorted(cum, cutoff)])


def _point_visible(p: Dict[str, Any]) -> bool:
    return bool(p and p.get("visible", False) and p.get("x") is not None and p.get("y") is not None)


def _point_xy(p: Dict[str, Any]) -> Optional[np.ndarray]:
    if not _point_visible(p):
        return None
    x = _safe_float(p.get("x"))
    y = _safe_float(p.get("y"))
    if x is None or y is None:
        return None
    return np.asarray([x, y], dtype=np.float64)


def _dist2d(a: Dict[str, Any], b: Dict[str, Any]) -> Optional[float]:
    aa = _point_xy(a)
    bb = _point_xy(b)
    if aa is None or bb is None:
        return None
    return float(np.linalg.norm(aa - bb))


def _point_to_curve_distance_px(point: Dict[str, Any], curve: Dict[str, Any]) -> Optional[float]:
    p = _point_xy(point)
    if p is None or not curve or not curve.get("visible", False):
        return None
    pts = []
    for q in curve.get("points", []):
        if len(q) >= 2:
            x, y = _safe_float(q[0]), _safe_float(q[1])
            if x is not None and y is not None:
                pts.append([x, y])
    if not pts:
        return None
    arr = np.asarray(pts, dtype=np.float64)
    return float(np.min(np.linalg.norm(arr - p[None, :], axis=1)))


def _point_to_sternal_notch_px(point: Dict[str, Any], sternum_curve: Dict[str, Any]) -> Optional[float]:
    p = _point_xy(point)
    if p is None or not sternum_curve or not sternum_curve.get("visible", False):
        return None
    pts = []
    for q in sternum_curve.get("points", []):
        if len(q) >= 2:
            x, y = _safe_float(q[0]), _safe_float(q[1])
            if x is not None and y is not None:
                pts.append([x, y])
    if not pts:
        return None
    arr = np.asarray(pts, dtype=np.float64)
    # Topmost sternum point is used as sternal-notch proxy.
    notch = arr[np.argmin(arr[:, 1])]
    return float(np.linalg.norm(p - notch))


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _pose_pixel_height(pose_json: Optional[Dict[str, Any]]) -> Optional[float]:
    if not pose_json:
        return None
    ys = []
    for kp in pose_json.get("keypoints", []):
        if isinstance(kp, dict):
            c = _safe_float(kp.get("confidence"), 0.0)
            y = _safe_float(kp.get("y"))
        else:
            c = _safe_float(kp[2] if len(kp) > 2 else 0.0, 0.0)
            y = _safe_float(kp[1] if len(kp) > 1 else None)
        if c is not None and c > 0.35 and y is not None:
            ys.append(float(y))
    if len(ys) < 2:
        return None
    return float(max(ys) - min(ys))


def _visibility_pixel_height(vis_json: Dict[str, Any], pose_json: Optional[Dict[str, Any]]) -> Optional[float]:
    bbox = vis_json.get("bbox", {})
    y1, y2 = _safe_float(bbox.get("y1")), _safe_float(bbox.get("y2"))
    if y1 is not None and y2 is not None and y2 > y1:
        return float(y2 - y1)
    return _pose_pixel_height(pose_json)


def _matching_pose_json(visibility_path: Path, pose_dir: Optional[Path]) -> Optional[Dict[str, Any]]:
    if pose_dir is None or not pose_dir.exists():
        return None
    candidate = pose_dir / visibility_path.name
    if candidate.exists():
        return _load_json(candidate)
    # Some stages may use different stem but same .json.
    candidate = pose_dir / f"{visibility_path.stem}.json"
    return _load_json(candidate) if candidate.exists() else None


def _collect_visibility_paths(visibility_dir: Optional[Path]) -> List[Path]:
    if visibility_dir is None or not visibility_dir.exists():
        return []
    return sorted([p for p in visibility_dir.glob("*.json") if p.is_file()])


def _fuse_chest_landmarks_from_visibility(
    visibility_dir: Optional[Path],
    pose_dir: Optional[Path],
    height_cm: float,
) -> Dict[str, Any]:
    """
    Fuse per-image pixel landmarks from improved visibility_analysis.py and convert to cm.

    Pixel-to-cm is derived per image as:
        cm_per_px = height_cm / person_pixel_height

    If the image is cropped, this is an approximation. The output includes
    evidence and observation counts so low-confidence or inconsistent image
    evidence can be audited later.
    """
    paths = _collect_visibility_paths(visibility_dir)
    values: Dict[str, List[float]] = {
        "left_areola_diameter_cm": [],
        "right_areola_diameter_cm": [],
        "nipple_to_nipple_distance_cm": [],
        "left_nipple_to_imf_cm": [],
        "right_nipple_to_imf_cm": [],
        "left_nipple_to_sternal_notch_cm": [],
        "right_nipple_to_sternal_notch_cm": [],
    }
    weights: Dict[str, List[float]] = {k: [] for k in values.keys()}
    evidence = []

    def add(name: str, px_value: Optional[float], cm_per_px: float, weight: float):
        if px_value is None or not np.isfinite(px_value):
            return
        values[name].append(float(px_value) * float(cm_per_px))
        weights[name].append(max(float(weight), 1e-6))

    for path in paths:
        vis = _load_json(path)
        if not vis:
            continue
        pose = _matching_pose_json(path, pose_dir)
        person_px = _visibility_pixel_height(vis, pose)
        if person_px is None or person_px <= 1e-6:
            continue

        cm_per_px = float(height_cm) / float(person_px)
        chest = vis.get("chest_analysis", {})
        if not chest or not chest.get("available", False):
            continue

        left_nip = chest.get("left_nipple", {}) or {}
        right_nip = chest.get("right_nipple", {}) or {}
        left_imf = chest.get("left_imf_curve", {}) or {}
        right_imf = chest.get("right_imf_curve", {}) or {}
        sternum = chest.get("sternum_midline", {}) or {}

        chest_w = _safe_float(chest.get("chest_visibility_weight", vis.get("chest_visibility_weight", 0.0)), 0.0) or 0.0
        left_w = _safe_float(chest.get("left_breast_visibility", vis.get("left_breast_visibility", 0.0)), 0.0) or 0.0
        right_w = _safe_float(chest.get("right_breast_visibility", vis.get("right_breast_visibility", 0.0)), 0.0) or 0.0
        sternum_w = _safe_float(chest.get("sternum_visibility", vis.get("sternum_visibility", 0.0)), 0.0) or 0.0

        left_nip_conf = _safe_float(left_nip.get("confidence"), 0.0) or 0.0
        right_nip_conf = _safe_float(right_nip.get("confidence"), 0.0) or 0.0
        left_imf_conf = _safe_float(left_imf.get("confidence"), 0.0) or 0.0
        right_imf_conf = _safe_float(right_imf.get("confidence"), 0.0) or 0.0
        sternum_conf = _safe_float(sternum.get("confidence"), sternum_w) or sternum_w

        add(
            "left_areola_diameter_cm",
            _safe_float(left_nip.get("areola_diameter_px")),
            cm_per_px,
            left_w * left_nip_conf,
        )
        add(
            "right_areola_diameter_cm",
            _safe_float(right_nip.get("areola_diameter_px")),
            cm_per_px,
            right_w * right_nip_conf,
        )

        if _point_visible(left_nip) and _point_visible(right_nip):
            add(
                "nipple_to_nipple_distance_cm",
                _dist2d(left_nip, right_nip),
                cm_per_px,
                min(left_nip_conf, right_nip_conf) * chest_w,
            )

        add(
            "left_nipple_to_imf_cm",
            _point_to_curve_distance_px(left_nip, left_imf),
            cm_per_px,
            left_w * left_nip_conf * left_imf_conf,
        )
        add(
            "right_nipple_to_imf_cm",
            _point_to_curve_distance_px(right_nip, right_imf),
            cm_per_px,
            right_w * right_nip_conf * right_imf_conf,
        )

        add(
            "left_nipple_to_sternal_notch_cm",
            _point_to_sternal_notch_px(left_nip, sternum),
            cm_per_px,
            left_w * left_nip_conf * sternum_conf,
        )
        add(
            "right_nipple_to_sternal_notch_cm",
            _point_to_sternal_notch_px(right_nip, sternum),
            cm_per_px,
            right_w * right_nip_conf * sternum_conf,
        )

        evidence.append({
            "visibility_json": str(path),
            "image": vis.get("image"),
            "person_pixel_height": float(person_px),
            "cm_per_px": float(cm_per_px),
            "chest_visibility_weight": float(chest_w),
            "left_breast_visibility": float(left_w),
            "right_breast_visibility": float(right_w),
            "sternum_visibility": float(sternum_w),
            "left_nipple": left_nip,
            "right_nipple": right_nip,
            "left_imf_curve": left_imf,
            "right_imf_curve": right_imf,
            "sternum_midline": sternum,
        })

    fused: Dict[str, Any] = {}
    for key, vals in values.items():
        fused[key] = _weighted_median(vals, weights[key])
        fused[f"{key}_mean"] = _weighted_mean(vals, weights[key])
        fused[f"{key}_num_observations"] = int(len(vals))

    fused["evidence"] = evidence
    return fused


# ============================================================
# REGION AND LANDMARK ESTIMATION
# ============================================================

def _normalized_coords(vertices: np.ndarray) -> Dict[str, np.ndarray]:
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]

    return {
        "xn": (x - x.min()) / max(1e-8, float(x.max() - x.min())),
        "yn": (y - y.min()) / max(1e-8, float(y.max() - y.min())),
        "zn": (z - z.min()) / max(1e-8, float(z.max() - z.min())),
    }


def _front_direction(vertices: np.ndarray) -> int:
    """
    The fitting pipeline normally uses +z as front. This fallback checks
    whether +z has the larger extension.
    """
    z = vertices[:, 2]
    mid = np.median(z)
    top = np.percentile(z, 95) - mid
    bottom = mid - np.percentile(z, 5)
    return 1 if top >= bottom else -1


def _breast_region_masks(vertices: np.ndarray) -> Dict[str, np.ndarray]:
    nc = _normalized_coords(vertices)

    x = vertices[:, 0]
    x_mid = np.median(x)
    front_sign = _front_direction(vertices)
    z = front_sign * vertices[:, 2]
    zn = (z - z.min()) / max(1e-8, float(z.max() - z.min()))

    chest = (
        (nc["yn"] > 0.53) &
        (nc["yn"] < 0.74) &
        (nc["xn"] > 0.18) &
        (nc["xn"] < 0.82) &
        (zn > 0.45)
    )

    breast_band = (
        (nc["yn"] > 0.55) &
        (nc["yn"] < 0.70) &
        (nc["xn"] > 0.20) &
        (nc["xn"] < 0.80) &
        (zn > 0.48)
    )

    left = breast_band & (x < x_mid)
    right = breast_band & (x >= x_mid)

    sternum = (
        chest &
        (np.abs(x - x_mid) < 0.08 * (x.max() - x.min()))
    )

    underbust = (
        (nc["yn"] > 0.48) &
        (nc["yn"] < 0.58) &
        (nc["xn"] > 0.20) &
        (nc["xn"] < 0.80) &
        (zn > 0.38)
    )

    abdomen = (
        (nc["yn"] > 0.40) &
        (nc["yn"] < 0.58) &
        (nc["xn"] > 0.20) &
        (nc["xn"] < 0.80)
    )

    hips = (
        (nc["yn"] > 0.32) &
        (nc["yn"] < 0.52) &
        (nc["xn"] > 0.10) &
        (nc["xn"] < 0.90)
    )

    return {
        "chest": chest,
        "left_breast": left,
        "right_breast": right,
        "sternum": sternum,
        "underbust": underbust,
        "abdomen": abdomen,
        "hips": hips,
    }


def _weighted_apex(vertices: np.ndarray, mask: np.ndarray) -> Optional[np.ndarray]:
    if mask.sum() < 5:
        return None

    front_sign = _front_direction(vertices)
    pts = vertices[mask]
    z = front_sign * pts[:, 2]

    cutoff = np.percentile(z, 97.0)
    top = pts[z >= cutoff]

    if len(top) == 0:
        idx = np.argmax(z)
        return pts[idx]

    weights = front_sign * top[:, 2]
    weights = weights - weights.min() + 1e-6

    return (top * weights[:, None]).sum(axis=0) / weights.sum()


def _estimate_sternal_notch(vertices: np.ndarray) -> np.ndarray:
    nc = _normalized_coords(vertices)
    front_sign = _front_direction(vertices)

    x_mid = np.median(vertices[:, 0])
    span = vertices[:, 0].max() - vertices[:, 0].min()

    mask = (
        (nc["yn"] > 0.72) &
        (nc["yn"] < 0.86) &
        (np.abs(vertices[:, 0] - x_mid) < 0.12 * span)
    )

    if mask.sum() < 5:
        idx = np.argmin(np.abs(vertices[:, 0] - x_mid) + np.abs(nc["yn"] - 0.78))
        return vertices[idx]

    pts = vertices[mask]
    target_y = np.percentile(pts[:, 1], 60)
    target_z = np.percentile(front_sign * pts[:, 2], 45)

    score = (
        np.abs(pts[:, 0] - x_mid) +
        0.25 * np.abs(pts[:, 1] - target_y) +
        0.25 * np.abs(front_sign * pts[:, 2] - target_z)
    )

    return pts[np.argmin(score)]


def _estimate_imf_point(vertices: np.ndarray, breast_mask: np.ndarray, nipple: np.ndarray) -> np.ndarray:
    if breast_mask.sum() < 10:
        return nipple.copy()

    pts = vertices[breast_mask]
    x_span = max(1e-6, vertices[:, 0].max() - vertices[:, 0].min())
    local = pts[np.abs(pts[:, 0] - nipple[0]) < 0.09 * x_span]

    if len(local) < 5:
        local = pts

    front_sign = _front_direction(vertices)
    z = front_sign * local[:, 2]
    z_cut = np.percentile(z, 50)
    candidates = local[z >= z_cut]

    if len(candidates) < 3:
        candidates = local

    y_target = np.percentile(candidates[:, 1], 8)
    idx = np.argmin(np.abs(candidates[:, 1] - y_target))
    return candidates[idx]


def _base_width(vertices: np.ndarray, mask: np.ndarray) -> float:
    if mask.sum() < 5:
        return 0.0

    pts = vertices[mask]
    front_sign = _front_direction(vertices)
    z = front_sign * pts[:, 2]
    z_cut = np.percentile(z, 55)
    pts = pts[z >= z_cut]

    if len(pts) < 5:
        pts = vertices[mask]

    return float(pts[:, 0].max() - pts[:, 0].min())


def _projection_cm(vertices: np.ndarray, mask: np.ndarray, nipple: np.ndarray, sternum_mask: np.ndarray) -> float:
    if mask.sum() < 5:
        return 0.0

    front_sign = _front_direction(vertices)

    if sternum_mask.sum() >= 5:
        base_z = np.median(front_sign * vertices[sternum_mask, 2])
    else:
        pts = vertices[mask]
        base_z = np.percentile(front_sign * pts[:, 2], 20)

    return float(max(0.0, front_sign * nipple[2] - base_z))


def _breast_plane_and_depth(vertices: np.ndarray, masks: Dict[str, np.ndarray]) -> Tuple[Tuple[float, float, float], np.ndarray]:
    support = masks["sternum"] | masks["underbust"]

    if support.sum() < 20:
        support = masks["chest"] & ~(masks["left_breast"] | masks["right_breast"])

    pts = vertices[support]

    if len(pts) < 10:
        pts = vertices[masks["chest"]]

    front_sign = _front_direction(vertices)
    pts_fit = pts.copy()
    pts_fit[:, 2] = front_sign * pts_fit[:, 2]
    coef = _fit_plane_xy_to_z(pts_fit)

    v = vertices.copy()
    z_front = front_sign * v[:, 2]
    base = _plane_eval(coef, v[:, 0], v[:, 1])
    depth = np.maximum(0.0, z_front - base)

    return coef, depth


def _breast_volume(vertices: np.ndarray, faces: np.ndarray, mask: np.ndarray, depth: np.ndarray) -> float:
    if mask.sum() < 10:
        return 0.0

    fmask = mask[faces[:, 0]] & mask[faces[:, 1]] & mask[faces[:, 2]]
    selected = faces[fmask]

    volume = 0.0

    for tri in selected:
        pts = vertices[tri]
        xy = pts[:, :2]
        area_xy = 0.5 * abs(np.cross(xy[1] - xy[0], xy[2] - xy[0]))
        d = float(depth[tri].mean())
        volume += area_xy * d

    return float(max(0.0, volume))


def _ptosis_grade(nipple: np.ndarray, imf: np.ndarray) -> Tuple[int, float]:
    distance = float(nipple[1] - imf[1])

    if distance >= 1.0:
        grade = 0
    elif distance >= -1.0:
        grade = 1
    elif distance >= -3.0:
        grade = 2
    else:
        grade = 3

    return grade, abs(float(imf[1] - nipple[1]))


def _ptosis_grade_from_distance_cm(distance_cm: Optional[float]) -> Optional[int]:
    """
    Image-derived ptosis fallback based on nipple-to-IMF distance.
    This is heuristic and mainly useful for relative consistency.
    """
    if distance_cm is None:
        return None
    if distance_cm < 2.0:
        return 0
    if distance_cm < 4.0:
        return 1
    if distance_cm < 7.0:
        return 2
    return 3


def _prefer_image_measurement(image_value: Optional[float], mesh_value: float) -> float:
    if image_value is None or not np.isfinite(image_value):
        return float(mesh_value)
    return float(image_value)


# ============================================================
# MAIN MEASUREMENT EXTRACTION
# ============================================================

def extract_measurements(
    npz_path: Path,
    height_cm: Optional[float] = None,
    density_g_per_cm3: float = 0.95,
    flip_left_right: bool = False,
    visibility_dir: Optional[Path] = None,
    pose_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    if height_cm is None:
        height_cm = 165.0
        height_was_assumed = True
    else:
        height_was_assumed = False

    raw_vertices, faces, payload = _as_vertices_faces(npz_path)
    vertices, scale = _scale_vertices_to_cm(raw_vertices, float(height_cm))

    y_min, y_max = vertices[:, 1].min(), vertices[:, 1].max()
    height = y_max - y_min

    masks = _breast_region_masks(vertices)

    left_nipple = _weighted_apex(vertices, masks["left_breast"])
    right_nipple = _weighted_apex(vertices, masks["right_breast"])

    if left_nipple is None:
        left_nipple = vertices[masks["chest"]][0] if masks["chest"].sum() else vertices[0]

    if right_nipple is None:
        right_nipple = vertices[masks["chest"]][-1] if masks["chest"].sum() else vertices[0]

    overbust_y = float(np.mean([left_nipple[1], right_nipple[1]]))

    left_imf = _estimate_imf_point(vertices, masks["left_breast"], left_nipple)
    right_imf = _estimate_imf_point(vertices, masks["right_breast"], right_nipple)

    underbust_y = float(np.mean([left_imf[1], right_imf[1]]))

    overbust = _circumference_at_y(vertices, faces, overbust_y)
    underbust = _circumference_at_y(vertices, faces, underbust_y)

    # More robust waist / hip / BWH perimeters.
    waist_circ, waist_y = _best_circumference_in_band(
        vertices,
        faces,
        y_min + 0.40 * height,
        y_min + 0.58 * height,
        mode="min",
        samples=48,
    )
    hip_circ, hip_y = _best_circumference_in_band(
        vertices,
        faces,
        y_min + 0.32 * height,
        y_min + 0.52 * height,
        mode="max",
        samples=48,
    )

    left_base_width = _base_width(vertices, masks["left_breast"])
    right_base_width = _base_width(vertices, masks["right_breast"])

    left_projection = _projection_cm(vertices, masks["left_breast"], left_nipple, masks["sternum"])
    right_projection = _projection_cm(vertices, masks["right_breast"], right_nipple, masks["sternum"])

    sternum_notch = _estimate_sternal_notch(vertices)

    coef, depth = _breast_plane_and_depth(vertices, masks)

    left_volume = _breast_volume(vertices, faces, masks["left_breast"], depth)
    right_volume = _breast_volume(vertices, faces, masks["right_breast"], depth)

    left_weight = left_volume * density_g_per_cm3
    right_weight = right_volume * density_g_per_cm3

    left_area = _surface_area(vertices, faces, masks["left_breast"])
    right_area = _surface_area(vertices, faces, masks["right_breast"])

    left_grade_mesh, left_ptosis_dist_mesh = _ptosis_grade(left_nipple, left_imf)
    right_grade_mesh, right_ptosis_dist_mesh = _ptosis_grade(right_nipple, right_imf)

    nipple_distance_mesh = float(np.linalg.norm(left_nipple - right_nipple))
    left_nipple_to_imf_mesh = float(np.linalg.norm(left_nipple - left_imf))
    right_nipple_to_imf_mesh = float(np.linalg.norm(right_nipple - right_imf))
    left_nipple_to_notch_mesh = float(np.linalg.norm(left_nipple - sternum_notch))
    right_nipple_to_notch_mesh = float(np.linalg.norm(right_nipple - sternum_notch))

    left_conicity = float(left_projection / max(1e-6, left_base_width * 0.5))
    right_conicity = float(right_projection / max(1e-6, right_base_width * 0.5))

    waist_to_hip = float(waist_circ / max(1e-6, hip_circ))

    crotch_y = y_min + 0.47 * height
    leg_len = crotch_y - y_min
    leg_to_body_ratio = float(leg_len / max(1e-6, height))

    # Fuse image-derived chest landmarks and convert from px to cm.
    chest_landmarks = _fuse_chest_landmarks_from_visibility(
        visibility_dir=Path(visibility_dir) if visibility_dir else None,
        pose_dir=Path(pose_dir) if pose_dir else None,
        height_cm=float(height_cm),
    )

    left_nipple_to_imf = _prefer_image_measurement(chest_landmarks.get("left_nipple_to_imf_cm"), left_nipple_to_imf_mesh)
    right_nipple_to_imf = _prefer_image_measurement(chest_landmarks.get("right_nipple_to_imf_cm"), right_nipple_to_imf_mesh)
    nipple_distance = _prefer_image_measurement(chest_landmarks.get("nipple_to_nipple_distance_cm"), nipple_distance_mesh)
    left_nipple_to_notch = _prefer_image_measurement(chest_landmarks.get("left_nipple_to_sternal_notch_cm"), left_nipple_to_notch_mesh)
    right_nipple_to_notch = _prefer_image_measurement(chest_landmarks.get("right_nipple_to_sternal_notch_cm"), right_nipple_to_notch_mesh)

    left_grade_img = _ptosis_grade_from_distance_cm(chest_landmarks.get("left_nipple_to_imf_cm"))
    right_grade_img = _ptosis_grade_from_distance_cm(chest_landmarks.get("right_nipple_to_imf_cm"))
    left_grade = int(left_grade_img if left_grade_img is not None else left_grade_mesh)
    right_grade = int(right_grade_img if right_grade_img is not None else right_grade_mesh)

    result: Dict[str, Any] = {
        "bust_underbust_cm": underbust,
        "bust_overbust_cm": overbust,
        "bust_left_base_width_cm": left_base_width,
        "bust_right_base_width_cm": right_base_width,
        "bust_left_projection_cm": left_projection,
        "bust_right_projection_cm": right_projection,
        "bust_left_ptosis_grade": int(left_grade),
        "bust_right_ptosis_grade": int(right_grade),
        "bust_left_ptosis_distance_cm": left_nipple_to_imf,
        "bust_right_ptosis_distance_cm": right_nipple_to_imf,
        "bust_left_volume_cm3": left_volume,
        "bust_right_volume_cm3": right_volume,
        "bust_left_estimated_weight_g": left_weight,
        "bust_right_estimated_weight_g": right_weight,
        "bust_volume_asymmetry_cm3": abs(left_volume - right_volume),
        "bust_projection_asymmetry_cm": abs(left_projection - right_projection),
        "bust_base_width_asymmetry_cm": abs(left_base_width - right_base_width),
        "bust_nipple_to_nipple_distance_cm": nipple_distance,
        "bust_left_nipple_to_imf_cm": left_nipple_to_imf,
        "bust_right_nipple_to_imf_cm": right_nipple_to_imf,
        "bust_left_nipple_to_sternal_notch_cm": left_nipple_to_notch,
        "bust_right_nipple_to_sternal_notch_cm": right_nipple_to_notch,
        "bust_left_conicity_index": left_conicity,
        "bust_right_conicity_index": right_conicity,
        "bust_left_surface_area_cm2": left_area,
        "bust_right_surface_area_cm2": right_area,
        "leg_to_body_ratio": leg_to_body_ratio,
        "waist_to_hip_ratio": waist_to_hip,

        # Additional measurements requested
        "waist_cm": waist_circ,
        "hip_cm": hip_circ,
        "bust_waist_hip_perimeter_cm": {
            "bust_cm": overbust,
            "waist_cm": waist_circ,
            "hip_cm": hip_circ,
        },
        "bust_waist_hip_perimeter_label": f"{overbust:.1f}-{waist_circ:.1f}-{hip_circ:.1f} cm",

        # New image-derived chest landmark measurements, converted px -> cm
        "bust_left_areola_diameter_cm": chest_landmarks.get("left_areola_diameter_cm"),
        "bust_right_areola_diameter_cm": chest_landmarks.get("right_areola_diameter_cm"),
        "bust_left_areola_diameter_cm_mean": chest_landmarks.get("left_areola_diameter_cm_mean"),
        "bust_right_areola_diameter_cm_mean": chest_landmarks.get("right_areola_diameter_cm_mean"),
        "bust_left_areola_diameter_cm_num_observations": chest_landmarks.get("left_areola_diameter_cm_num_observations", 0),
        "bust_right_areola_diameter_cm_num_observations": chest_landmarks.get("right_areola_diameter_cm_num_observations", 0),

        # Preserve fused values for audit.
        "chest_landmark_measurements_cm": {
            k: v for k, v in chest_landmarks.items() if k != "evidence"
        },
        "chest_landmark_evidence": chest_landmarks.get("evidence", []),

        "_metadata": {
            "source_npz": str(npz_path),
            "known_height_cm": float(height_cm),
            "height_cm": float(height_cm),
            "height_was_assumed": bool(height_was_assumed),
            "default_height_cm": 165.0,
            "mesh_to_cm_scale": float(scale),
            "density_g_per_cm3": float(density_g_per_cm3),
            "visibility_dir": str(visibility_dir) if visibility_dir else None,
            "pose_dir": str(pose_dir) if pose_dir else None,
            "method": "mesh_geometry_plus_visibility_chest_landmark_fusion",
            "note": (
                "Nipple, IMF, sternum, ptosis, breast volume and weight are estimated. "
                "Image-derived chest landmarks are converted from pixels to cm using "
                "height_cm / per-image visible body pixel height. If height was omitted, "
                "165 cm is assumed."
            ),
        },
        "_landmarks_cm": {
            "left_nipple_xyz": left_nipple.tolist(),
            "right_nipple_xyz": right_nipple.tolist(),
            "left_imf_xyz": left_imf.tolist(),
            "right_imf_xyz": right_imf.tolist(),
            "sternal_notch_xyz": sternum_notch.tolist(),
        },
        "_mesh_slice_levels_cm": {
            "overbust_y_cm": overbust_y,
            "underbust_y_cm": underbust_y,
            "waist_y_cm": waist_y,
            "hip_y_cm": hip_y,
        },
    }

    if flip_left_right:
        result = _flip_left_right_metrics(result)

    return _json_clean(result)


def _flip_left_right_metrics(result: Dict) -> Dict:
    swapped = dict(result)

    for key in list(result.keys()):
        if key.startswith("bust_left_"):
            right_key = key.replace("bust_left_", "bust_right_")

            if right_key in result:
                swapped[key] = result[right_key]
                swapped[right_key] = result[key]

    # Extra left/right fields without bust_left prefix
    for a, b in [
        ("bust_left_areola_diameter_cm", "bust_right_areola_diameter_cm"),
        ("bust_left_areola_diameter_cm_mean", "bust_right_areola_diameter_cm_mean"),
        ("bust_left_areola_diameter_cm_num_observations", "bust_right_areola_diameter_cm_num_observations"),
    ]:
        if a in swapped and b in swapped:
            swapped[a], swapped[b] = swapped[b], swapped[a]

    lm = dict(result.get("_landmarks_cm", {}))

    for a, b in [
        ("left_nipple_xyz", "right_nipple_xyz"),
        ("left_imf_xyz", "right_imf_xyz"),
    ]:
        if a in lm and b in lm:
            lm[a], lm[b] = lm[b], lm[a]

    swapped["_landmarks_cm"] = lm
    return swapped


def _json_clean(value):
    if isinstance(value, dict):
        return {str(k): _json_clean(v) for k, v in value.items()}

    if isinstance(value, list):
        return [_json_clean(v) for v in value]

    if isinstance(value, tuple):
        return [_json_clean(v) for v in value]

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, (np.float32, np.float64)):
        return float(value)

    if isinstance(value, (np.int32, np.int64)):
        return int(value)

    if isinstance(value, float):
        if not np.isfinite(value):
            return None
        return value

    return value


def main():
    parser = argparse.ArgumentParser(description="Extract body/bust measurements from SMPL-X NPZ")

    parser.add_argument("--npz", required=True, type=str)
    parser.add_argument(
        "--height-cm",
        required=False,
        type=float,
        default=None,
        help="Known real model height in cm. If omitted, 165 cm is assumed.",
    )
    parser.add_argument("--output", required=True, type=str)

    parser.add_argument(
        "--visibility-dir",
        type=str,
        default=None,
        help="Directory containing visibility JSONs with chest_analysis landmarks.",
    )

    parser.add_argument(
        "--pose-dir",
        type=str,
        default=None,
        help="Optional directory containing pose JSONs. Used as fallback for pixel height scaling.",
    )

    parser.add_argument(
        "--density-g-per-cm3",
        type=float,
        default=0.95,
        help="Density used for estimated tissue weight. Default: 0.95 g/cm³",
    )

    parser.add_argument(
        "--flip-left-right",
        action="store_true",
        help="Swap left/right output labels if your mesh coordinate convention is reversed.",
    )

    args = parser.parse_args()

    measurements = extract_measurements(
        npz_path=Path(args.npz),
        height_cm=args.height_cm,
        density_g_per_cm3=args.density_g_per_cm3,
        flip_left_right=args.flip_left_right,
        visibility_dir=Path(args.visibility_dir) if args.visibility_dir else None,
        pose_dir=Path(args.pose_dir) if args.pose_dir else None,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(measurements, f, indent=2)

    print(f"✅ Measurements exported: {output_path}")

    for key in [
        "bust_underbust_cm",
        "bust_overbust_cm",
        "waist_cm",
        "hip_cm",
        "bust_waist_hip_perimeter_label",
        "bust_left_areola_diameter_cm",
        "bust_right_areola_diameter_cm",
        "bust_left_volume_cm3",
        "bust_right_volume_cm3",
        "waist_to_hip_ratio",
        "leg_to_body_ratio",
    ]:
        print(f"{key}: {measurements.get(key)}")


if __name__ == "__main__":
    main()
