from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
import numpy as np

from .mesh_io import load_npz_mesh, read_obj, save_npz_mesh, write_obj
from .head_registration import SimilarityHeadRegistrar, RegistrationConfig


@dataclass
class HeadBodyFusionConfig:
    """
    SMPL-X + fused head combine with closed-loop auto-registration.

    v10:
      - tighter SMPL-X head reference excludes neck/collar
      - registration has absolute scale clamp to prevent the observed 1.19 scale
      - source lower trim is stronger to prevent FLAME neck skirt scale inflation
    """

    vertical_axis: str = "auto"

    neck_cutoff_ratio: float = 0.895
    neck_overlap_ratio: float = 0.010
    remove_body_head: bool = True
    head_remove_lateral_ratio: float = 0.135
    head_remove_forward_ratio: float = 0.115

    clip_head_neck: bool = True
    head_clip_bottom_ratio: float = 0.16

    register_head: bool = True
    registration_iterations: int = 16
    registration_sample_points: int = 900
    registration_trim_fraction: float = 0.60
    registration_scale_delta_min: float = 0.86
    registration_scale_delta_max: float = 1.00
    registration_absolute_scale_min: float = 0.82
    registration_absolute_scale_max: float = 1.08
    registration_target_scale_shrink: float = 0.94
    registration_allow_rotation: bool = False
    registration_forward_icp_weight: float = 0.50

    # Manual post-registration offsets. Normally keep zero.
    head_scale: float = 1.0
    head_vertical_offset: float = 0.0
    head_lateral_offset: float = 0.0
    head_forward_offset: float = 0.0

    connect_neck: bool = False
    debug: bool = True


class HeadBodyFusion:
    AXIS_NAME_TO_INDEX = {"x": 0, "y": 1, "z": 2}

    def __init__(self, config: HeadBodyFusionConfig | None = None):
        self.config = config or HeadBodyFusionConfig()

    def fuse(self, body_npz_path, head_obj_path, output_npz_path, output_obj_path, summary_path=None):
        body_npz_path = Path(body_npz_path)
        head_obj_path = Path(head_obj_path)
        output_npz_path = Path(output_npz_path)
        output_obj_path = Path(output_obj_path)
        summary_path = Path(summary_path) if summary_path else output_npz_path.with_suffix(".json")

        if not body_npz_path.exists():
            raise FileNotFoundError(f"Missing body NPZ: {body_npz_path}")
        if not head_obj_path.exists():
            raise FileNotFoundError(f"Missing head OBJ: {head_obj_path}")

        body_vertices, body_faces = load_npz_mesh(body_npz_path)
        raw_head_vertices, raw_head_faces, _, _ = read_obj(head_obj_path)

        axes = self._infer_axes(body_vertices)
        body_info = self._measure_body(body_vertices, axes)

        clipped_head_vertices, clipped_head_faces, clip_info = self._clip_head_neck(
            raw_head_vertices,
            raw_head_faces,
            axes,
        )

        if self.config.register_head:
            registrar = SimilarityHeadRegistrar(
                RegistrationConfig(
                    enabled=True,
                    iterations=self.config.registration_iterations,
                    sample_points=self.config.registration_sample_points,
                    trim_fraction=self.config.registration_trim_fraction,
                    scale_delta_min=self.config.registration_scale_delta_min,
                    scale_delta_max=self.config.registration_scale_delta_max,
                    absolute_scale_min=self.config.registration_absolute_scale_min,
                    absolute_scale_max=self.config.registration_absolute_scale_max,
                    target_scale_shrink=self.config.registration_target_scale_shrink,
                    allow_rotation=self.config.registration_allow_rotation,
                    forward_icp_weight=self.config.registration_forward_icp_weight,
                    debug=self.config.debug,
                )
            )

            registered_head_vertices, registration, registration_refs = registrar.register(
                source_vertices=clipped_head_vertices,
                body_vertices=body_vertices,
                body_info=body_info,
                axes=axes,
                source_faces=clipped_head_faces,
            )
        else:
            registered_head_vertices = clipped_head_vertices.copy()
            registration = {
                "success": True,
                "method": "registration_disabled",
                "scale": 1.0,
                "rotation": np.eye(3, dtype=np.float32),
                "translation": np.zeros(3, dtype=np.float32),
                "matrix": np.eye(4, dtype=np.float32),
            }
            registration_refs = {}

        registered_head_vertices = self._apply_manual_post_adjust(
            registered_head_vertices,
            body_info,
            axes,
        )

        body_cut = self._cut_body_head(body_vertices, body_faces, body_info, axes)

        combined_vertices, combined_faces, bridge_info = self._combine_meshes(
            body_vertices=body_cut["vertices"],
            body_faces=body_cut["faces"],
            head_vertices=registered_head_vertices,
            head_faces=clipped_head_faces,
        )

        save_npz_mesh(
            output_npz_path,
            combined_vertices,
            combined_faces,
            source_body=str(body_npz_path),
            source_head=str(head_obj_path),
            registration_matrix=np.asarray(registration.get("matrix", np.eye(4)), dtype=np.float32),
            head_clip_info=np.asarray(json.dumps(self._json_safe(clip_info))),
        )

        write_obj(
            output_obj_path,
            combined_vertices,
            combined_faces,
            comments=[
                "Generated by avatar_pose_pipeline HeadBodyFusion v10 registration",
                "v10: tighter target + absolute scale clamp",
                f"body={body_npz_path}",
                f"head={head_obj_path}",
            ],
        )

        summary = {
            "source_body": str(body_npz_path),
            "source_head": str(head_obj_path),
            "output_npz": str(output_npz_path),
            "output_obj": str(output_obj_path),
            "config": asdict(self.config),
            "axes": axes,
            "body": body_info,
            "head_clip": clip_info,
            "registration": registration,
            "body_cut": {
                "kept_vertices": int(len(body_cut["vertices"])),
                "kept_faces": int(len(body_cut["faces"])),
                "removed_vertices": int(body_cut["removed_vertices"]),
                "removed_faces": int(body_cut["removed_faces"]),
                "arm_safe_removal": True,
                "removal_rule": "remove only above neck cutoff AND inside central head/neck capsule",
            },
            "bridge": bridge_info,
            "combined": {
                "vertices": int(len(combined_vertices)),
                "faces": int(len(combined_faces)),
            },
            "notes": [
                "v10 fixes oversized heads by using a tighter SMPL-X head target and absolute registration scale clamp.",
                "The uploaded v9 summary showed target width 0.197, source width 0.162, and final scale 1.191, which is too large.",
                "If still too large globally, lower --combine-registration-absolute-scale-max to 1.03.",
                "If too small globally, raise --combine-registration-absolute-scale-max to 1.12.",
            ],
        }

        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(self._json_safe(summary), indent=2))

        if self.config.debug:
            print("✅ SMPL-X + head registration fusion complete")
            print(f"   body: {body_npz_path}")
            print(f"   head: {head_obj_path}")
            print(f"   obj:  {output_obj_path}")
            print(f"   npz:  {output_npz_path}")
            print(f"   registration method: {registration.get('method')}")
            print(f"   registration scale:  {registration.get('scale')}")
            print(f"   registration error:  {registration.get('error')}")
            print(f"   removed body vertices: {body_cut['removed_vertices']}")

        return summary

    def _infer_axes(self, body_vertices):
        if self.config.vertical_axis != "auto":
            vertical = self.AXIS_NAME_TO_INDEX[self.config.vertical_axis]
        else:
            ranges = body_vertices.max(axis=0) - body_vertices.min(axis=0)
            vertical = int(np.argmax(ranges))

        remaining = [i for i in range(3) if i != vertical]
        ranges = body_vertices.max(axis=0) - body_vertices.min(axis=0)

        if ranges[remaining[0]] >= ranges[remaining[1]]:
            lateral = remaining[0]
            forward = remaining[1]
        else:
            lateral = remaining[1]
            forward = remaining[0]

        return {"vertical": int(vertical), "lateral": int(lateral), "forward": int(forward)}

    def _axis_values(self, vertices, axis):
        return np.asarray(vertices[:, int(axis)], dtype=np.float32)

    def _measure_body(self, vertices, axes):
        v_axis = axes["vertical"]
        l_axis = axes["lateral"]
        f_axis = axes["forward"]

        y = self._axis_values(vertices, v_axis)
        body_min = float(y.min())
        body_max = float(y.max())
        body_height = body_max - body_min

        if body_height <= 1e-8:
            raise RuntimeError("Body height is near zero; cannot align head.")

        neck_y = body_min + self.config.neck_cutoff_ratio * body_height
        overlap = self.config.neck_overlap_ratio * body_height

        lateral = self._axis_values(vertices, l_axis)
        forward = self._axis_values(vertices, f_axis)

        l_med = float(np.median(lateral))
        f_med = float(np.median(forward))
        l_span = float(np.percentile(lateral, 90) - np.percentile(lateral, 10))
        f_span = float(np.percentile(forward, 90) - np.percentile(forward, 10))

        neck_band = np.abs(y - neck_y) < max(overlap * 3.0, body_height * 0.012)
        central = (
            np.abs(lateral - l_med) < max(0.15 * body_height, 0.24 * l_span, 1e-6)
        ) & (
            np.abs(forward - f_med) < max(0.09 * body_height, 0.32 * f_span, 1e-6)
        )

        neck_candidates = neck_band & central
        if neck_candidates.sum() < 12:
            neck_candidates = neck_band & (np.abs(lateral - l_med) < 0.20 * body_height)
        if neck_candidates.sum() < 12:
            neck_candidates = neck_band

        if neck_candidates.sum() < 12:
            neck_center = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
            neck_center[v_axis] = neck_y
            neck_center[l_axis] = l_med
            neck_center[f_axis] = f_med
        else:
            neck_center = vertices[neck_candidates].mean(axis=0).astype(np.float32)

        return {
            "min": body_min,
            "max": body_max,
            "height": float(body_height),
            "neck_y": float(neck_y),
            "neck_overlap": float(overlap),
            "body_attach_y": float(neck_y - overlap * 0.15),
            "neck_center": neck_center,
            "bbox_min": vertices.min(axis=0).astype(np.float32),
            "bbox_max": vertices.max(axis=0).astype(np.float32),
            "lateral_median": l_med,
            "forward_median": f_med,
            "original_top_height": float(body_max - neck_y),
        }

    def _clip_head_neck(self, vertices, faces, axes):
        vertices = np.asarray(vertices, dtype=np.float32)
        faces = np.asarray(faces, dtype=np.int64)

        if not self.config.clip_head_neck:
            return vertices.copy(), faces.copy(), {
                "enabled": False,
                "removed_vertices": 0,
                "removed_faces": 0,
                "clip_y": None,
            }

        v_axis = axes["vertical"]
        y = self._axis_values(vertices, v_axis)
        h_min = float(y.min())
        h_max = float(y.max())
        h = max(h_max - h_min, 1e-8)

        clip_y = h_min + self.config.head_clip_bottom_ratio * h
        keep = y >= clip_y

        if keep.sum() < 0.55 * len(vertices):
            clip_y = h_min + 0.10 * h
            keep = y >= clip_y

        old_to_new = -np.ones(len(vertices), dtype=np.int64)
        old_to_new[keep] = np.arange(int(keep.sum()), dtype=np.int64)

        keep_face = keep[faces].all(axis=1)

        new_vertices = vertices[keep].copy()
        new_faces = old_to_new[faces[keep_face]].astype(np.int64)

        return new_vertices, new_faces, {
            "enabled": True,
            "clip_bottom_ratio": float(self.config.head_clip_bottom_ratio),
            "clip_y": float(clip_y),
            "removed_vertices": int((~keep).sum()),
            "removed_faces": int((~keep_face).sum()),
            "remaining_vertices": int(len(new_vertices)),
            "remaining_faces": int(len(new_faces)),
        }

    def _apply_manual_post_adjust(self, vertices, body_info, axes):
        vertices = np.asarray(vertices, dtype=np.float32).copy()

        if abs(self.config.head_scale - 1.0) > 1e-8:
            center = vertices.mean(axis=0)
            vertices = (vertices - center[None, :]) * float(self.config.head_scale) + center[None, :]

        offset = np.zeros(3, dtype=np.float32)
        offset[axes["vertical"]] = self.config.head_vertical_offset * body_info["height"]
        offset[axes["lateral"]] = self.config.head_lateral_offset * body_info["height"]
        offset[axes["forward"]] = self.config.head_forward_offset * body_info["height"]

        vertices += offset[None, :]
        return vertices.astype(np.float32)

    def _cut_body_head(self, vertices, faces, body_info, axes):
        if not self.config.remove_body_head:
            return {
                "vertices": vertices.copy(),
                "faces": faces.copy(),
                "removed_vertices": 0,
                "removed_faces": 0,
                "old_to_new": np.arange(len(vertices), dtype=np.int64),
            }

        v_axis = axes["vertical"]
        l_axis = axes["lateral"]
        f_axis = axes["forward"]

        y = self._axis_values(vertices, v_axis)
        lateral = self._axis_values(vertices, l_axis)
        forward = self._axis_values(vertices, f_axis)

        cutoff = body_info["neck_y"] - body_info["neck_overlap"]
        center = np.asarray(body_info["neck_center"], dtype=np.float32)

        lateral_radius = self.config.head_remove_lateral_ratio * body_info["height"]
        forward_radius = self.config.head_remove_forward_ratio * body_info["height"]

        ell = (
            ((lateral - center[l_axis]) / max(lateral_radius, 1e-6)) ** 2 +
            ((forward - center[f_axis]) / max(forward_radius, 1e-6)) ** 2
        )

        remove_vertex = (y >= cutoff) & (ell < 1.0)
        hard_lateral_limit = np.abs(lateral - center[l_axis]) < 0.16 * body_info["height"]
        remove_vertex = remove_vertex & hard_lateral_limit

        keep_vertex = ~remove_vertex
        old_to_new = -np.ones(len(vertices), dtype=np.int64)
        old_to_new[keep_vertex] = np.arange(keep_vertex.sum())
        keep_face = keep_vertex[faces].all(axis=1)

        return {
            "vertices": vertices[keep_vertex].copy(),
            "faces": old_to_new[faces[keep_face]].astype(np.int64),
            "removed_vertices": int(remove_vertex.sum()),
            "removed_faces": int((~keep_face).sum()),
            "old_to_new": old_to_new,
        }

    def _combine_meshes(self, body_vertices, body_faces, head_vertices, head_faces):
        body_vertices = np.asarray(body_vertices, dtype=np.float32)
        body_faces = np.asarray(body_faces, dtype=np.int64)
        head_vertices = np.asarray(head_vertices, dtype=np.float32)
        head_faces = np.asarray(head_faces, dtype=np.int64)

        head_offset = len(body_vertices)
        combined_vertices = np.vstack([body_vertices, head_vertices])
        combined_faces = np.vstack([body_faces, head_faces + head_offset])

        return combined_vertices.astype(np.float32), combined_faces.astype(np.int64), {
            "enabled": bool(self.config.connect_neck),
            "success": False,
            "faces_added": 0,
            "note": "Bridge disabled in v10. Registration + overlap is preferred; seam blending should be handled later.",
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
        if isinstance(value, Path):
            return str(value)
        return value
