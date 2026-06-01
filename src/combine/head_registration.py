from __future__ import annotations

from dataclasses import dataclass, asdict
import math
import numpy as np


@dataclass
class RegistrationConfig:
    enabled: bool = True

    # v10: tighter SMPL-X reference. v9 selected too much lower neck/collar,
    # producing target width 0.197 and scale 1.19 in the user's summary.
    smplx_head_top_margin_ratio: float = 0.020
    smplx_head_lower_ratio: float = 0.825
    smplx_head_lateral_radius_ratio: float = 0.120
    smplx_head_forward_radius_ratio: float = 0.105

    # Source reference trims lower FLAME skirt/neck; this reduces scale inflation.
    source_lower_trim_ratio: float = 0.120
    source_upper_trim_ratio: float = 0.015

    iterations: int = 16
    sample_points: int = 900
    trim_fraction: float = 0.60
    damping: float = 0.55

    # Scale correction relative to coarse init.
    scale_delta_min: float = 0.86
    scale_delta_max: float = 1.00

    # Absolute scale clamps. This is the key v10 safety rail: the fused FLAME
    # head is already roughly in SMPL-X units, so scale 1.19 is not acceptable.
    absolute_scale_min: float = 0.82
    absolute_scale_max: float = 1.08

    # Conservative global multiplier applied to coarse target dimensions to avoid
    # matching SMPL-X ears/collar width as face width.
    target_scale_shrink: float = 0.94

    allow_rotation: bool = False
    max_rotation_degrees: float = 8.0
    forward_icp_weight: float = 0.50

    debug: bool = True


class SimilarityHeadRegistrar:
    """
    Landmark-free closed-loop registration of fused FLAME/DECA head to fitted
    SMPL-X head.

    v10 changes:
      - tighter target extraction: avoid neck/collar and shoulder/cut artifacts
      - lower source trim: avoid FLAME lower neck skirt driving scale
      - conservative absolute scale clamp: prevents the observed 1.19 oversized head
      - scale_delta_max default 1.00: ICP may shrink but not grow beyond coarse fit
    """

    def __init__(self, config: RegistrationConfig | None = None):
        self.config = config or RegistrationConfig()

    def register(self, source_vertices, body_vertices, body_info, axes, source_faces=None):
        source_vertices = np.asarray(source_vertices, dtype=np.float32)
        body_vertices = np.asarray(body_vertices, dtype=np.float32)

        target = self.extract_smplx_head_reference(body_vertices, body_info, axes)
        source = self.extract_source_reference(source_vertices, axes)

        if len(target["points"]) < 32 or len(source["points"]) < 32:
            transform = self._identity_transform()
            transform["success"] = False
            transform["reason"] = "not_enough_reference_points"
            return source_vertices.copy(), transform, {"target": target, "source": source}

        coarse = self.coarse_similarity_init(source["points"], target["points"], axes)
        refined = self.robust_icp_refine(source["points"], target["points"], coarse, axes)
        transform = refined if refined.get("success") else coarse

        f_axis = axes["forward"]
        if self.config.forward_icp_weight < 1.0:
            transformed_center = self.apply_transform(source["center"][None, :], transform)[0]
            target_center = target["center"]
            correction = target_center[f_axis] - transformed_center[f_axis]
            transform["translation"][f_axis] += (1.0 - self.config.forward_icp_weight) * correction

        registered = self.apply_transform(source_vertices, transform)

        transform["matrix"] = self.make_matrix(
            scale=transform["scale"],
            rotation=transform["rotation"],
            translation=transform["translation"],
        )
        transform["config"] = asdict(self.config)
        transform["target_stats"] = self._stats_for_json(target)
        transform["source_stats"] = self._stats_for_json(source)

        return registered.astype(np.float32), self._json_safe(transform), {
            "target": target,
            "source": source,
        }

    # =========================================================
    # REFERENCE EXTRACTION
    # =========================================================

    def extract_smplx_head_reference(self, vertices, body_info, axes):
        v_axis = axes["vertical"]
        l_axis = axes["lateral"]
        f_axis = axes["forward"]

        y = vertices[:, v_axis]
        lateral = vertices[:, l_axis]
        forward = vertices[:, f_axis]

        center = np.asarray(body_info["neck_center"], dtype=np.float32)
        body_height = float(body_info["height"])

        lower = body_info["min"] + self.config.smplx_head_lower_ratio * body_height
        upper = body_info["max"] - self.config.smplx_head_top_margin_ratio * body_height

        lateral_radius = self.config.smplx_head_lateral_radius_ratio * body_height
        forward_radius = self.config.smplx_head_forward_radius_ratio * body_height

        ell = (
            ((lateral - center[l_axis]) / max(lateral_radius, 1e-6)) ** 2 +
            ((forward - center[f_axis]) / max(forward_radius, 1e-6)) ** 2
        )

        mask = (y >= lower) & (y <= upper) & (ell <= 1.0)

        # Fallback is still tighter than v9.
        if mask.sum() < 64:
            mask = (
                (y >= lower) &
                (y <= upper) &
                (np.abs(lateral - center[l_axis]) < lateral_radius * 1.10) &
                (np.abs(forward - center[f_axis]) < forward_radius * 1.10)
            )

        pts = vertices[mask].astype(np.float32)
        return self._reference_record(pts, mask, "smplx_head_reference", axes)

    def extract_source_reference(self, vertices, axes):
        v_axis = axes["vertical"]

        y = vertices[:, v_axis]
        h_min = float(y.min())
        h_max = float(y.max())
        h = max(h_max - h_min, 1e-8)

        lower = h_min + self.config.source_lower_trim_ratio * h
        upper = h_max - self.config.source_upper_trim_ratio * h

        mask = (y >= lower) & (y <= upper)
        pts = vertices[mask].astype(np.float32)

        return self._reference_record(pts, mask, "source_head_reference", axes)

    def _reference_record(self, pts, mask, name, axes):
        if len(pts) == 0:
            center = np.zeros(3, dtype=np.float32)
            bbox_min = np.zeros(3, dtype=np.float32)
            bbox_max = np.zeros(3, dtype=np.float32)
            dims = np.zeros(3, dtype=np.float32)
        else:
            center = pts.mean(axis=0).astype(np.float32)
            bbox_min = np.percentile(pts, 5, axis=0).astype(np.float32)
            bbox_max = np.percentile(pts, 95, axis=0).astype(np.float32)
            dims = bbox_max - bbox_min

        return {
            "name": name,
            "points": pts,
            "mask": mask,
            "count": int(len(pts)),
            "center": center,
            "bbox_min": bbox_min,
            "bbox_max": bbox_max,
            "dims": dims.astype(np.float32),
            "width_lateral": float(dims[axes["lateral"]]) if len(pts) else 0.0,
            "height_vertical": float(dims[axes["vertical"]]) if len(pts) else 0.0,
            "depth_forward": float(dims[axes["forward"]]) if len(pts) else 0.0,
        }

    # =========================================================
    # INITIALIZATION / ICP
    # =========================================================

    def coarse_similarity_init(self, source_points, target_points, axes):
        src = np.asarray(source_points, dtype=np.float32)
        tgt = np.asarray(target_points, dtype=np.float32)

        src_dims = np.percentile(src, 95, axis=0) - np.percentile(src, 5, axis=0)
        tgt_dims = np.percentile(tgt, 95, axis=0) - np.percentile(tgt, 5, axis=0)
        tgt_dims = tgt_dims * float(self.config.target_scale_shrink)

        scale_candidates = []
        weights = []

        # v10: lateral dominates. Vertical may include neck/collar differences.
        for axis, weight in [
            (axes["lateral"], 0.70),
            (axes["vertical"], 0.20),
            (axes["forward"], 0.10),
        ]:
            if src_dims[axis] > 1e-8 and tgt_dims[axis] > 1e-8:
                scale_candidates.append(tgt_dims[axis] / src_dims[axis])
                weights.append(weight)

        if not scale_candidates:
            scale = 1.0
        else:
            scale_candidates = np.asarray(scale_candidates, dtype=np.float32)
            weights = np.asarray(weights, dtype=np.float32)
            weights = weights / weights.sum()
            scale = float(np.sum(scale_candidates * weights))

        scale = float(np.clip(scale, self.config.absolute_scale_min, self.config.absolute_scale_max))

        src_center = src.mean(axis=0)
        tgt_center = tgt.mean(axis=0)

        rotation = np.eye(3, dtype=np.float32)
        translation = (tgt_center - scale * src_center).astype(np.float32)

        return {
            "success": True,
            "method": "coarse_tight_head_similarity_v10",
            "scale": float(scale),
            "coarse_scale": float(scale),
            "rotation": rotation,
            "translation": translation,
            "error": None,
            "scale_candidates": scale_candidates.tolist() if len(scale_candidates) else [],
            "src_dims": src_dims.astype(np.float32),
            "tgt_dims_shrunk": tgt_dims.astype(np.float32),
        }

    def robust_icp_refine(self, source_points, target_points, init_transform, axes):
        src_full = np.asarray(source_points, dtype=np.float32)
        tgt_full = np.asarray(target_points, dtype=np.float32)

        src = self._sample_points(src_full, self.config.sample_points)
        tgt = self._sample_points(tgt_full, self.config.sample_points)

        scale0 = float(init_transform["scale"])
        scale = scale0
        rotation = np.asarray(init_transform["rotation"], dtype=np.float32).copy()
        translation = np.asarray(init_transform["translation"], dtype=np.float32).copy()

        last_error = None

        for _ in range(int(self.config.iterations)):
            transformed = scale * (src @ rotation.T) + translation[None, :]
            nearest, distances = self._nearest_neighbors(transformed, tgt)

            if len(distances) < 8:
                break

            trim_n = max(8, int(len(distances) * self.config.trim_fraction))
            keep = np.argsort(distances)[:trim_n]

            p = src[keep]
            q = nearest[keep]

            update = self._fit_similarity(p, q, allow_rotation=self.config.allow_rotation)

            new_scale = update["scale"]
            min_scale = max(scale0 * self.config.scale_delta_min, self.config.absolute_scale_min)
            max_scale = min(scale0 * self.config.scale_delta_max, self.config.absolute_scale_max)
            new_scale = float(np.clip(new_scale, min_scale, max_scale))

            if self.config.allow_rotation:
                new_rotation = self._clamp_rotation(update["rotation"], self.config.max_rotation_degrees)
            else:
                new_rotation = np.eye(3, dtype=np.float32)

            new_translation = update["translation"]
            damp = float(self.config.damping)
            scale = (1.0 - damp) * scale + damp * new_scale
            translation = (1.0 - damp) * translation + damp * new_translation
            rotation = new_rotation.astype(np.float32)

            last_error = float(np.mean(distances[keep]))

        scale = float(np.clip(scale, self.config.absolute_scale_min, self.config.absolute_scale_max))

        return {
            "success": True,
            "method": "robust_trimmed_icp_similarity_v10",
            "scale": float(scale),
            "coarse_scale": float(scale0),
            "rotation": rotation.astype(np.float32),
            "translation": translation.astype(np.float32),
            "error": last_error,
            "iterations": int(self.config.iterations),
        }

    def _fit_similarity(self, source, target, allow_rotation=False):
        source = np.asarray(source, dtype=np.float32)
        target = np.asarray(target, dtype=np.float32)

        src_center = source.mean(axis=0)
        tgt_center = target.mean(axis=0)

        src0 = source - src_center
        tgt0 = target - tgt_center

        if allow_rotation:
            H = src0.T @ tgt0
            U, S, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = Vt.T @ U.T
        else:
            R = np.eye(3, dtype=np.float32)

        src_rot = src0 @ R.T
        denom = float(np.sum(src_rot ** 2)) + 1e-8
        scale = float(np.sum(src_rot * tgt0) / denom)
        scale = max(scale, 1e-6)

        translation = tgt_center - scale * (src_center @ R.T)

        return {
            "scale": float(scale),
            "rotation": R.astype(np.float32),
            "translation": translation.astype(np.float32),
        }

    def _nearest_neighbors(self, query, target):
        query = np.asarray(query, dtype=np.float32)
        target = np.asarray(target, dtype=np.float32)

        try:
            from scipy.spatial import cKDTree
            tree = cKDTree(target)
            distances, indices = tree.query(query, k=1, workers=-1)
            return target[indices], distances.astype(np.float32)
        except Exception:
            nearest = []
            distances = []
            chunk = 256
            for i in range(0, len(query), chunk):
                q = query[i:i + chunk]
                d2 = ((q[:, None, :] - target[None, :, :]) ** 2).sum(axis=2)
                idx = np.argmin(d2, axis=1)
                nearest.append(target[idx])
                distances.append(np.sqrt(d2[np.arange(len(q)), idx]))
            return np.vstack(nearest), np.concatenate(distances).astype(np.float32)

    def _sample_points(self, points, n):
        points = np.asarray(points, dtype=np.float32)
        if len(points) <= n:
            return points
        rng = np.random.default_rng(12345)
        idx = rng.choice(len(points), size=n, replace=False)
        return points[idx]

    # =========================================================
    # TRANSFORMS / UTILS
    # =========================================================

    def apply_transform(self, vertices, transform):
        vertices = np.asarray(vertices, dtype=np.float32)
        scale = float(transform["scale"])
        rotation = np.asarray(transform["rotation"], dtype=np.float32)
        translation = np.asarray(transform["translation"], dtype=np.float32)
        return (scale * (vertices @ rotation.T) + translation[None, :]).astype(np.float32)

    def make_matrix(self, scale, rotation, translation):
        M = np.eye(4, dtype=np.float32)
        M[:3, :3] = float(scale) * np.asarray(rotation, dtype=np.float32)
        M[:3, 3] = np.asarray(translation, dtype=np.float32)
        return M

    def _identity_transform(self):
        return {
            "success": True,
            "method": "identity",
            "scale": 1.0,
            "coarse_scale": 1.0,
            "rotation": np.eye(3, dtype=np.float32),
            "translation": np.zeros(3, dtype=np.float32),
            "matrix": np.eye(4, dtype=np.float32),
            "error": None,
        }

    def _clamp_rotation(self, R, max_degrees):
        R = np.asarray(R, dtype=np.float32)
        trace = float(np.trace(R))
        cos_angle = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
        angle = math.degrees(math.acos(cos_angle))
        if angle > max_degrees:
            return np.eye(3, dtype=np.float32)
        return R

    def _stats_for_json(self, ref):
        return {
            "name": ref.get("name"),
            "count": int(ref.get("count", 0)),
            "center": ref.get("center"),
            "bbox_min": ref.get("bbox_min"),
            "bbox_max": ref.get("bbox_max"),
            "dims": ref.get("dims"),
            "width_lateral": ref.get("width_lateral"),
            "height_vertical": ref.get("height_vertical"),
            "depth_forward": ref.get("depth_forward"),
        }

    def _json_safe(self, value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {str(k): self._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._json_safe(v) for v in value]
        return value
