#!/usr/bin/env python3
"""
Generate an automatic SMPL-X female breast topology prior.

This script does NOT modify SMPL-X. It reads SMPLX_FEMALE.npz and writes:

  - smplx_female_breast_topology_prior.json
  - smplx_female_breast_soft_weights.npz
  - smplx_female_breast_prior_preview.obj
  - optional PNG preview plots

v4 fixes versus v3:
  - the visible preview no longer paints the abdomen guard in bright red by default;
    that red area was being mistaken for the IMF, even though the IMF bands were
    blue/purple in v3.
  - IMF selection is now derived from the lower envelope of the detected breast
    lobe vertices instead of a broad fixed vertical strip. This prevents abdomen
    vertices from leaking into the IMF band.
  - breast lobe lower bound is raised and made side-adaptive, so the generic
    prior sits on the anatomical breast region of the SMPL-X template rather
    than the upper abdomen.
  - guard regions remain exported in JSON/NPZ, but preview coloring of guards is
    opt-in with --color-guards to avoid misleading validation plots.

Coordinate convention expected for SMPL-X template:
  x = left/right, y = vertical up, z = depth. Positive z is assumed to be front.

Example:
  python generate_smplx_breast_topology_prior_v4.py \
    --smplx-npz /home/riley/dev/avatar_pose_pipeline/models/smplx/smplx/SMPLX_FEMALE.npz \
    --out-dir /home/riley/dev/avatar_pose_pipeline/assets \
    --write-preview-png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

VERSION = 4

# Preview colors. Guard colors are only used when --color-guards is passed.
COLORS = {
    "default": (0.68, 0.68, 0.68),
    "plus_x_breast": (1.00, 0.88, 0.05),       # yellow
    "minus_x_breast": (1.00, 0.30, 0.72),      # pink
    "plus_x_imf_band": (0.00, 0.80, 1.00),     # cyan
    "minus_x_imf_band": (0.80, 0.10, 1.00),    # magenta
    "sternum": (0.95, 0.95, 0.05),             # pale yellow
    "nipple_anchor": (1.00, 1.00, 1.00),       # white
    "armpit_guard": (0.10, 0.85, 0.10),        # green, opt-in preview
    "upper_chest_guard": (0.55, 0.38, 1.00),   # violet, opt-in preview
    "abdomen_guard": (0.35, 0.35, 0.35),       # muted, opt-in preview
}


def _as_array(npz: np.lib.npyio.NpzFile, key: str) -> np.ndarray:
    if key not in npz:
        raise KeyError(f"SMPL-X npz does not contain required key: {key}")
    return np.asarray(npz[key])


def load_template(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(str(path), allow_pickle=True)
    verts = _as_array(data, "v_template").astype(np.float64)
    if "f" in data:
        faces = np.asarray(data["f"], dtype=np.int64)
    elif "faces" in data:
        faces = np.asarray(data["faces"], dtype=np.int64)
    else:
        raise KeyError("SMPL-X npz does not contain 'f' or 'faces'")
    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError(f"Unexpected v_template shape: {verts.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"Unexpected faces shape: {faces.shape}")
    return verts, faces


def robust_normalize(values: np.ndarray, lo_q: float = 1.0, hi_q: float = 99.0) -> Tuple[np.ndarray, float, float]:
    lo = float(np.percentile(values, lo_q))
    hi = float(np.percentile(values, hi_q))
    scale = max(hi - lo, 1e-9)
    return (values - lo) / scale, lo, hi


def indices(mask: np.ndarray) -> List[int]:
    return np.flatnonzero(mask).astype(int).tolist()


def nearest_vertex(verts: np.ndarray, mask: np.ndarray, target: np.ndarray) -> int | None:
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return None
    d2 = np.sum((verts[idx] - target[None, :]) ** 2, axis=1)
    return int(idx[int(np.argmin(d2))])


def soften_from_groups(n: int, groups: Dict[str, List[int]]) -> Dict[str, np.ndarray]:
    weights: Dict[str, np.ndarray] = {}
    for name, idxs in groups.items():
        arr = np.zeros(n, dtype=np.float32)
        if idxs:
            arr[np.asarray(idxs, dtype=np.int64)] = 1.0
        weights[name] = arr
    return weights


def _side_masks(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return x > 0, x < 0


def _dynamic_imf_from_breast(
    verts: np.ndarray,
    breast_mask: np.ndarray,
    front_depth: np.ndarray,
    yn: np.ndarray,
    ax: np.ndarray,
    side_mask: np.ndarray,
) -> np.ndarray:
    """Return a narrow under-breast band from the lower envelope of a side breast.

    This is intentionally based on the selected breast lobe itself. The old v3
    fixed y strip could accidentally include upper abdomen vertices; this cannot,
    because the returned band is a subset of the breast mask plus a very small
    adjacent rim.
    """
    idx = np.flatnonzero(breast_mask & side_mask)
    if idx.size == 0:
        return np.zeros_like(breast_mask, dtype=bool)

    yvals = yn[idx]
    # Bottom 20-30% of the breast lobe, clipped to avoid isolated very low points.
    q08 = float(np.quantile(yvals, 0.08))
    q34 = float(np.quantile(yvals, 0.34))
    z_floor = float(np.quantile(front_depth[idx], 0.35))

    imf = breast_mask & side_mask & (yn >= q08) & (yn <= q34) & (front_depth >= z_floor)

    # Keep it laterally on the breast, not the sternum or arm side wall.
    xvals = ax[idx]
    x_lo = max(0.035, float(np.quantile(xvals, 0.10)))
    x_hi = min(0.300, float(np.quantile(xvals, 0.92)))
    imf &= (ax >= x_lo) & (ax <= x_hi)
    return imf


def build_prior(verts: np.ndarray, faces: np.ndarray, source_hint: str, front_sign: float) -> Tuple[dict, dict]:
    x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]
    x_abs = np.abs(x)
    yn, ylo, yhi = robust_normalize(y)
    zn, zlo, zhi = robust_normalize(z)
    x_scale = max(float(np.percentile(x_abs, 99.0)), 1e-9)
    ax = x_abs / x_scale

    front_depth = z * front_sign
    front_thr = float(np.percentile(front_depth, 38.0))
    high_front_thr = float(np.percentile(front_depth, 58.0))
    weak_front_thr = float(np.percentile(front_depth, 30.0))
    front = front_depth >= front_thr
    high_front = front_depth >= high_front_thr
    weak_front = front_depth >= weak_front_thr

    plus_x, minus_x = _side_masks(x)

    # Conservative anatomical chest window. The lower bound is deliberately above
    # the upper abdomen; IMF is later recovered from the breast lower envelope.
    breast_y = (yn >= 0.635) & (yn <= 0.815)
    breast_x = (ax >= 0.030) & (ax <= 0.285)
    breast_base = front & breast_y & breast_x

    plus_breast = breast_base & plus_x
    minus_breast = breast_base & minus_x

    # If a template variation is sparse under the stricter bounds, fall back to a
    # slightly wider but still chest-only selection.
    if int(plus_breast.sum()) < 80 or int(minus_breast.sum()) < 80:
        breast_y_fb = (yn >= 0.615) & (yn <= 0.825)
        breast_x_fb = (ax >= 0.025) & (ax <= 0.310)
        breast_base = front & breast_y_fb & breast_x_fb
        plus_breast = breast_base & plus_x
        minus_breast = breast_base & minus_x

    plus_imf = _dynamic_imf_from_breast(verts, plus_breast, front_depth, yn, ax, plus_x)
    minus_imf = _dynamic_imf_from_breast(verts, minus_breast, front_depth, yn, ax, minus_x)

    def split_lower_upper(breast: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        idx = np.flatnonzero(breast)
        if idx.size == 0:
            return breast.copy(), breast.copy()
        cut = float(np.quantile(yn[idx], 0.48))
        return breast & (yn <= cut), breast & (yn > cut)

    plus_lower, plus_upper = split_lower_upper(plus_breast)
    minus_lower, minus_upper = split_lower_upper(minus_breast)

    # Sternum remains narrow and only between the breasts.
    sternum = high_front & (ax <= 0.040) & (yn >= 0.645) & (yn <= 0.790)

    # Guards are for optimization regularization only; not part of breast/IMF.
    plus_armpit = weak_front & plus_x & (ax >= 0.225) & (ax <= 0.360) & (yn >= 0.735) & (yn <= 0.825)
    minus_armpit = weak_front & minus_x & (ax >= 0.225) & (ax <= 0.360) & (yn >= 0.735) & (yn <= 0.825)
    upper_chest = weak_front & (yn >= 0.805) & (yn <= 0.875) & (ax <= 0.345)
    abdomen = weak_front & (yn >= 0.485) & (yn <= 0.620) & (ax <= 0.285)

    def anchor_from(mask: np.ndarray) -> int | None:
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            return None
        # Prefer anterior vertices around the breast center, not high clavicle.
        y_center = float(np.quantile(yn[idx], 0.58))
        score = zn[idx] - 0.35 * np.abs(yn[idx] - y_center) - 0.08 * np.abs(ax[idx] - np.median(ax[idx]))
        return int(idx[int(np.argmax(score))])

    plus_nipple = anchor_from(plus_breast)
    minus_nipple = anchor_from(minus_breast)

    plus_imf_mid = nearest_vertex(
        verts, plus_imf, np.array([0.14 * x_scale, float(np.median(y[np.flatnonzero(plus_imf)])) if plus_imf.any() else np.percentile(y, 67), np.percentile(z, 78)])
    )
    minus_imf_mid = nearest_vertex(
        verts, minus_imf, np.array([-0.14 * x_scale, float(np.median(y[np.flatnonzero(minus_imf)])) if minus_imf.any() else np.percentile(y, 67), np.percentile(z, 78)])
    )
    sternum_mid = nearest_vertex(verts, sternum, np.array([0.0, np.percentile(y, 70), np.percentile(z, 82)]))

    vertex_groups = {
        # Backward-compatible names. For new code prefer plus_x/minus_x aliases.
        "left_breast": indices(plus_breast),
        "right_breast": indices(minus_breast),
        "left_imf_band": indices(plus_imf),
        "right_imf_band": indices(minus_imf),
        "left_lower_pole": indices(plus_lower),
        "right_lower_pole": indices(minus_lower),
        "left_upper_pole": indices(plus_upper),
        "right_upper_pole": indices(minus_upper),
        "sternum": indices(sternum),
        "left_armpit_guard": indices(plus_armpit),
        "right_armpit_guard": indices(minus_armpit),
        "upper_chest_guard": indices(upper_chest),
        "abdomen_guard": indices(abdomen),
        # Unambiguous aliases.
        "plus_x_breast": indices(plus_breast),
        "minus_x_breast": indices(minus_breast),
        "plus_x_imf_band": indices(plus_imf),
        "minus_x_imf_band": indices(minus_imf),
        "plus_x_armpit_guard": indices(plus_armpit),
        "minus_x_armpit_guard": indices(minus_armpit),
    }

    anchors = {
        "left_nipple_prior_vertex": plus_nipple,
        "right_nipple_prior_vertex": minus_nipple,
        "left_imf_mid_vertex": plus_imf_mid,
        "right_imf_mid_vertex": minus_imf_mid,
        "sternum_mid_vertex": sternum_mid,
        "plus_x_nipple_prior_vertex": plus_nipple,
        "minus_x_nipple_prior_vertex": minus_nipple,
        "plus_x_imf_mid_vertex": plus_imf_mid,
        "minus_x_imf_mid_vertex": minus_imf_mid,
    }

    warnings: List[str] = []
    for name in ("plus_x_breast", "minus_x_breast", "plus_x_imf_band", "minus_x_imf_band"):
        if len(vertex_groups[name]) == 0:
            warnings.append(f"empty group: {name}")
    if len(vertex_groups["abdomen_guard"]) > 0:
        # This warning is intentionally explanatory because the preview confusion
        # came from interpreting abdomen guard color as IMF.
        warnings.append("abdomen_guard is exported for regularization only; it is not the IMF band")

    prior = {
        "version": VERSION,
        "model": "SMPL-X",
        "gender": "female",
        "source_model_path_hint": source_hint,
        "vertex_count": int(verts.shape[0]),
        "face_count": int(faces.shape[0]),
        "coordinate_convention": {
            "template_x_axis": "left_right",
            "template_y_axis": "vertical_up",
            "template_depth_axis": 2,
            "template_front_sign": float(front_sign),
            "note": "left/right keys are backward-compatible template-side aliases; plus_x/minus_x keys are unambiguous.",
        },
        "generation": {
            "method": "automatic_template_coordinate_prior_v4_dynamic_imf",
            "front_threshold_percentile": 38,
            "high_front_threshold_percentile": 58,
            "weak_front_threshold_percentile": 30,
            "x_scale": float(x_scale),
            "breast_y_norm_range": [0.635, 0.815],
            "breast_abs_x_norm_range": [0.030, 0.285],
            "imf_method": "lower_envelope_quantile_of_each_side_breast_lobe",
            "sternum_y_norm_range": [0.645, 0.790],
            "sternum_abs_x_norm_max": 0.040,
            "armpit_guard_x_norm_range": [0.225, 0.360],
            "armpit_guard_y_norm_range": [0.735, 0.825],
            "notes": "v4 derives IMF from the breast lower envelope and keeps abdomen guard visually muted by default. No per-subject manual labels are required.",
            "warnings": warnings,
        },
        "anchors": anchors,
        "vertex_groups": vertex_groups,
    }

    weights = soften_from_groups(verts.shape[0], vertex_groups)
    return prior, weights


def write_obj(path: Path, verts: np.ndarray, faces: np.ndarray, prior: dict, color_guards: bool = False) -> None:
    colors = np.tile(np.asarray(COLORS["default"], dtype=np.float64), (verts.shape[0], 1))
    groups = prior["vertex_groups"]

    # Draw guards first only when requested. Breast/IMF/anchors overwrite them.
    if color_guards:
        for key in ("upper_chest_guard", "left_armpit_guard", "right_armpit_guard", "abdomen_guard"):
            if key in groups:
                cname = "armpit_guard" if "armpit" in key else key
                colors[np.asarray(groups[key], dtype=np.int64)] = COLORS[cname]

    for key in ("plus_x_breast", "minus_x_breast", "sternum", "plus_x_imf_band", "minus_x_imf_band"):
        if key in groups and len(groups[key]):
            colors[np.asarray(groups[key], dtype=np.int64)] = COLORS[key if key in COLORS else key]

    for k, v in prior.get("anchors", {}).items():
        if v is not None and 0 <= int(v) < verts.shape[0]:
            colors[int(v)] = COLORS["nipple_anchor"]

    with open(path, "w", encoding="utf-8") as f:
        f.write("# SMPL-X breast topology prior preview v4\n")
        f.write("# yellow/pink=breast, cyan/magenta=IMF, white=nipple anchors\n")
        if color_guards:
            f.write("# green/violet/muted gray=guards; guards are not IMF\n")
        for p, c in zip(verts, colors):
            f.write(f"v {p[0]:.8f} {p[1]:.8f} {p[2]:.8f} {c[0]:.6f} {c[1]:.6f} {c[2]:.6f}\n")
        for tri in faces:
            f.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")


def write_preview_pngs(out_dir: Path, stem: str, verts: np.ndarray, prior: dict, color_guards: bool = False) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"Could not write preview PNGs because matplotlib is unavailable: {exc}")
        return

    groups = prior["vertex_groups"]
    colors = np.tile(np.asarray(COLORS["default"], dtype=np.float64), (verts.shape[0], 1))
    if color_guards:
        for key in ("upper_chest_guard", "left_armpit_guard", "right_armpit_guard", "abdomen_guard"):
            if key in groups:
                cname = "armpit_guard" if "armpit" in key else key
                colors[np.asarray(groups[key], dtype=np.int64)] = COLORS[cname]
    for key in ("plus_x_breast", "minus_x_breast", "sternum", "plus_x_imf_band", "minus_x_imf_band"):
        if key in groups and len(groups[key]):
            colors[np.asarray(groups[key], dtype=np.int64)] = COLORS[key]
    for v in prior.get("anchors", {}).values():
        if v is not None and 0 <= int(v) < verts.shape[0]:
            colors[int(v)] = COLORS["nipple_anchor"]

    plots = [
        ("front_xy", 0, 1, "x", "y"),
        ("side_zy", 2, 1, "z", "y"),
        ("top_xz", 0, 2, "x", "z"),
    ]
    for name, a, b, xlabel, ylabel in plots:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        ax.scatter(verts[:, a], verts[:, b], s=5, c=colors)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(name + (" guards" if color_guards else " breast_imf_only"))
        ax.grid(True, alpha=0.35)
        fig.tight_layout()
        fig.savefig(out_dir / f"{stem}_{name}.png", dpi=180)
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate SMPL-X female breast topology prior v4")
    ap.add_argument("--smplx-npz", required=True, type=Path, help="Path to SMPLX_FEMALE.npz")
    ap.add_argument("--out-dir", required=True, type=Path, help="Output directory")
    ap.add_argument("--stem", default="smplx_female_breast", help="Output filename stem")
    ap.add_argument("--front-sign", type=float, default=1.0, choices=[-1.0, 1.0], help="Use -1 if the template front is negative z")
    ap.add_argument("--write-preview-png", action="store_true", help="Also write PNG preview plots")
    ap.add_argument("--color-guards", action="store_true", help="Color guard regions in previews. Off by default to avoid confusing guards with IMF.")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    verts, faces = load_template(args.smplx_npz)
    prior, weights = build_prior(verts, faces, str(args.smplx_npz), args.front_sign)

    json_path = args.out_dir / f"{args.stem}_topology_prior.json"
    npz_path = args.out_dir / f"{args.stem}_soft_weights.npz"
    obj_path = args.out_dir / f"{args.stem}_prior_preview.obj"

    json_path.write_text(json.dumps(prior, indent=2), encoding="utf-8")
    np.savez_compressed(npz_path, **weights)
    write_obj(obj_path, verts, faces, prior, color_guards=args.color_guards)
    if args.write_preview_png:
        write_preview_pngs(args.out_dir, f"{args.stem}_prior_preview", verts, prior, color_guards=args.color_guards)

    print(f"Wrote {json_path}")
    print(f"Wrote {npz_path}")
    print(f"Wrote {obj_path}")
    if prior["generation"].get("warnings"):
        print("Warnings:")
        for w in prior["generation"]["warnings"]:
            print(f"  - {w}")


if __name__ == "__main__":
    main()
