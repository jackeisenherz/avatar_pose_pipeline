#!/usr/bin/env python3
"""
Extract body and bust measurements from canonical/refined SMPL-X NPZ.

Input:
    data/output/08_smplx/canonical_body.npz
    or
    data/output/09_refined/refined_body.npz

Required:
    --height-cm known real model height in centimeters

Output:
    JSON with requested measurements.

Important:
    Bust-specific values are geometric estimates from the fitted mesh.
    They are not clinical or scan-grade measurements unless the mesh has
    accurate nipple/IMF/sternal-notch landmarks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple, Optional

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


# ============================================================
# MAIN MEASUREMENT EXTRACTION
# ============================================================

def extract_measurements(
    npz_path: Path,
    height_cm: float,
    density_g_per_cm3: float = 0.95,
    flip_left_right: bool = False,
) -> Dict[str, float]:
    raw_vertices, faces, payload = _as_vertices_faces(npz_path)
    vertices, scale = _scale_vertices_to_cm(raw_vertices, height_cm)

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

    left_grade, left_ptosis_dist = _ptosis_grade(left_nipple, left_imf)
    right_grade, right_ptosis_dist = _ptosis_grade(right_nipple, right_imf)

    nipple_distance = float(np.linalg.norm(left_nipple - right_nipple))
    left_nipple_to_imf = float(np.linalg.norm(left_nipple - left_imf))
    right_nipple_to_imf = float(np.linalg.norm(right_nipple - right_imf))
    left_nipple_to_notch = float(np.linalg.norm(left_nipple - sternum_notch))
    right_nipple_to_notch = float(np.linalg.norm(right_nipple - sternum_notch))

    left_conicity = float(left_projection / max(1e-6, left_base_width * 0.5))
    right_conicity = float(right_projection / max(1e-6, right_base_width * 0.5))

    waist_y = y_min + 0.46 * height
    hip_y = y_min + 0.38 * height
    waist_circ = _circumference_at_y(vertices, faces, waist_y)
    hip_circ = _circumference_at_y(vertices, faces, hip_y)
    waist_to_hip = float(waist_circ / max(1e-6, hip_circ))

    crotch_y = y_min + 0.47 * height
    leg_len = crotch_y - y_min
    leg_to_body_ratio = float(leg_len / max(1e-6, height))

    result = {
        "bust_underbust_cm": underbust,
        "bust_overbust_cm": overbust,
        "bust_left_base_width_cm": left_base_width,
        "bust_right_base_width_cm": right_base_width,
        "bust_left_projection_cm": left_projection,
        "bust_right_projection_cm": right_projection,
        "bust_left_ptosis_grade": int(left_grade),
        "bust_right_ptosis_grade": int(right_grade),
        "bust_left_ptosis_distance_cm": left_ptosis_dist,
        "bust_right_ptosis_distance_cm": right_ptosis_dist,
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
        "_metadata": {
            "source_npz": str(npz_path),
            "known_height_cm": float(height_cm),
            "mesh_to_cm_scale": float(scale),
            "density_g_per_cm3": float(density_g_per_cm3),
            "method": "geometric_estimate_from_fitted_smplx_mesh",
            "note": (
                "Nipple, IMF, sternum, ptosis, breast volume and weight are estimated "
                "from mesh geometry. For production/clinical-grade values, provide "
                "explicit nipple/IMF/sternal-notch landmarks or a scan-grade mesh."
            ),
        },
        "_landmarks_cm": {
            "left_nipple_xyz": left_nipple.tolist(),
            "right_nipple_xyz": right_nipple.tolist(),
            "left_imf_xyz": left_imf.tolist(),
            "right_imf_xyz": right_imf.tolist(),
            "sternal_notch_xyz": sternum_notch.tolist(),
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

    return value


def main():
    parser = argparse.ArgumentParser(description="Extract body/bust measurements from SMPL-X NPZ")

    parser.add_argument("--npz", required=True, type=str)
    parser.add_argument("--height-cm", required=True, type=float)
    parser.add_argument("--output", required=True, type=str)

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
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(measurements, f, indent=2)

    print(f"✅ Measurements exported: {output_path}")

    for key in [
        "bust_underbust_cm",
        "bust_overbust_cm",
        "bust_left_volume_cm3",
        "bust_right_volume_cm3",
        "waist_to_hip_ratio",
        "leg_to_body_ratio",
    ]:
        print(f"{key}: {measurements[key]}")


if __name__ == "__main__":
    main()
