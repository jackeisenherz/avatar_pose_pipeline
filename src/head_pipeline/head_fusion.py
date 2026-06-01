from pathlib import Path
import json
import shutil
import sys

import cv2
import numpy as np


def _patch_numpy_for_deca():
    aliases = {
        "bool": bool,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
        "str": str,
        "unicode": str,
    }

    for name, value in aliases.items():
        if not hasattr(np, name):
            setattr(np, name, value)


def _json_safe(value):
    try:
        import torch
        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
    except Exception:
        pass

    if isinstance(value, np.ndarray):
        return {
            "__ndarray__": True,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    return str(value)


def _public_mesh_result(mesh_result):
    return {
        k: v
        for k, v in mesh_result.items()
        if k not in {"vertices", "faces"}
    }


class HeadFusion:
    """
    Multi-crop canonical head fusion with stricter bad-crop rejection.

    Key fix:
        The previous texture selector over-trusted UV texture coverage. A bad crop
        that contains only one eye can still produce a UV texture with high
        non-black coverage and therefore poison the canonical texture.

    This version separates:
        - geometry fusion: multi-crop FLAME shape fusion
        - texture selection: strict crop + UV anatomical validation

    Texture strategy:
        1. Reject textures from invalid / partial / eye-only crops.
        2. Prefer best single frontal/high-quality UV texture.
        3. Average only when several textures pass strict checks.
        4. Repair small black holes with valid-mask fill + inpaint.
        5. Skip texture entirely rather than exporting a poisoned texture.

    Expected outputs:
        canonical_head_params.npz
        canonical_fused_head.obj
        canonical_head_texture.png              if texture passes checks
        canonical_head_texture_quality_report.json
        rejected_texture_candidates.json
        canonical_fused_head_textured.obj       if UV layout + texture exist
    """

    SHAPE_KEYS = ["shape", "shapecode", "shape_code", "shape_params", "betas", "beta", "identity", "id"]
    EXP_KEYS = ["exp", "expression", "expr", "expression_params", "expcode", "exp_code"]
    POSE_KEYS = ["pose", "pose_params", "flame_pose", "full_pose", "global_pose", "head_pose"]
    TEX_KEYS = ["tex", "texture", "texcode", "tex_code", "texture_code", "albedo", "albedocode"]
    CAM_KEYS = ["cam", "camera", "cam_params"]
    LIGHT_KEYS = ["light", "lights", "lighting", "lit"]

    def __init__(
        self,
        deca_root="external/DECA",
        device="cuda",
        min_score=0.20,
        top_k=8,
        neutralize_expression=True,
        neutralize_pose=True,
        allow_obj_fallback=True,
        texture_mode="best",
        texture_top_k=5,
        texture_size=1024,
        texture_min_quality=0.58,
        texture_average_min_quality=0.68,
        texture_black_threshold=14,
        strict_texture_validation=True,
    ):
        self.deca_root = Path(deca_root)
        self.device = device
        self.min_score = float(min_score)
        self.top_k = int(top_k)
        self.neutralize_expression = bool(neutralize_expression)
        self.neutralize_pose = bool(neutralize_pose)
        self.allow_obj_fallback = bool(allow_obj_fallback)

        # IMPORTANT: default is best, not average.
        self.texture_mode = str(texture_mode)
        self.texture_top_k = int(texture_top_k)
        self.texture_size = int(texture_size)
        self.texture_min_quality = float(texture_min_quality)
        self.texture_average_min_quality = float(texture_average_min_quality)
        self.texture_black_threshold = int(texture_black_threshold)
        self.strict_texture_validation = bool(strict_texture_validation)

    # =========================================================
    # MAIN
    # =========================================================

    def fuse_from_recon_dir(self, recon_dir, crop_index_path=None, output_dir=None):
        recon_dir = Path(recon_dir)
        output_dir = Path(output_dir) if output_dir else recon_dir / "fused"
        output_dir.mkdir(parents=True, exist_ok=True)

        crop_metadata = self._load_crop_metadata(crop_index_path)
        param_files = sorted(recon_dir.rglob("*.mat")) + sorted(recon_dir.rglob("*.npz"))
        key_report = self._write_key_report(param_files, output_dir)
        records = self._collect_records(recon_dir=recon_dir, crop_metadata=crop_metadata)

        summary = {
            "recon_dir": str(recon_dir),
            "output_dir": str(output_dir),
            "num_param_files": len(param_files),
            "key_report": str(key_report),
            "num_parameter_records": len(records),
            "used_parameter_fusion": False,
            "used_obj_fallback": False,
            "selected": [],
            "mesh": {},
            "texture": {},
            "preview": {},
            "warnings": [],
            "note": (
                "This is a standalone canonical head. Final texture seam blending "
                "should happen later when the head is combined with the body."
            ),
        }

        if records:
            selected = self._select_records(records)
            fused = self._weighted_average(selected)

            params_path = output_dir / "canonical_head_params.npz"
            np.savez(
                params_path,
                shape=fused["shape"],
                expression=fused["expression"],
                pose=fused["pose"],
                texture=fused["texture"] if fused["texture"] is not None else np.asarray([], dtype=np.float32),
                weights=np.asarray([r["weight"] for r in selected], dtype=np.float32),
                source_files=np.asarray([str(r["param_path"]) for r in selected]),
            )

            mesh_result = self._decode_or_fallback(
                fused=fused,
                selected=selected,
                recon_dir=recon_dir,
                output_dir=output_dir,
            )

            texture_result = self._build_canonical_uv_texture(
                selected=selected,
                recon_dir=recon_dir,
                output_dir=output_dir,
            )

            preview_result = self._create_textured_preview(
                mesh_result=mesh_result,
                texture_result=texture_result,
                selected=selected,
                recon_dir=recon_dir,
                output_dir=output_dir,
            )

            if not texture_result.get("success"):
                summary["warnings"].append(
                    "No acceptable UV texture was found. Geometry was exported; texture was skipped."
                )

            if not preview_result.get("success"):
                summary["warnings"].append(
                    "Textured preview OBJ was not created because UV layout or acceptable texture was missing."
                )

            summary.update(
                {
                    "used_parameter_fusion": True,
                    "num_selected": len(selected),
                    "selected": [
                        {
                            "param_path": str(r["param_path"]),
                            "score": float(r["score"]),
                            "weight": float(r["weight"]),
                            "crop_quality": float(r.get("crop_quality", 0.0)),
                            "validation_score": float(r.get("validation_score", 0.0)),
                            "pose_penalty": float(r.get("pose_penalty", 0.0)),
                            "expression_penalty": float(r.get("expression_penalty", 0.0)),
                            "matched_keys": r.get("matched_keys", {}),
                        }
                        for r in selected
                    ],
                    "params_path": str(params_path),
                    "mesh": _public_mesh_result(mesh_result),
                    "texture": texture_result,
                    "preview": preview_result,
                }
            )

            summary_path = output_dir / "head_fusion_summary.json"
            summary_path.write_text(json.dumps(_json_safe(summary), indent=2))

            print("✅ Canonical head fusion complete")
            print(f"📄 Params: {params_path}")

            if mesh_result.get("obj_path"):
                print(f"🙂 Mesh OBJ: {mesh_result['obj_path']}")

            if texture_result.get("success"):
                print(f"🎨 Texture:  {texture_result['texture_path']}")
            else:
                print("⚠ No acceptable UV texture found.")

            if preview_result.get("success"):
                print(f"👀 Preview:  {preview_result['preview_obj']}")
            else:
                print("⚠ No textured preview created.")

            return _json_safe(summary)

        if self.allow_obj_fallback:
            fallback = self._obj_fallback_fusion(recon_dir=recon_dir, crop_metadata=crop_metadata, output_dir=output_dir)
            texture_result = self._build_canonical_uv_texture(selected=[], recon_dir=recon_dir, output_dir=output_dir)
            preview_result = self._create_textured_preview(
                mesh_result=fallback,
                texture_result=texture_result,
                selected=[],
                recon_dir=recon_dir,
                output_dir=output_dir,
            )

            summary.update(
                {
                    "used_obj_fallback": True,
                    "mesh": _public_mesh_result(fallback),
                    "texture": texture_result,
                    "preview": preview_result,
                }
            )

            if not texture_result.get("success"):
                summary["warnings"].append("No acceptable UV texture found.")

            summary_path = output_dir / "head_fusion_summary.json"
            summary_path.write_text(json.dumps(_json_safe(summary), indent=2))
            return _json_safe(summary)

        summary_path = output_dir / "head_fusion_summary.json"
        summary_path.write_text(json.dumps(_json_safe(summary), indent=2))
        raise RuntimeError(f"No usable parameter records found. See key report: {key_report}")

    # =========================================================
    # METADATA / PARAM LOADING
    # =========================================================

    def _write_key_report(self, param_files, output_dir):
        report = []
        for path in param_files:
            path = Path(path)
            try:
                data = self._load_mat(path) if path.suffix.lower() == ".mat" else self._load_npz(path)
                flat = {}
                self._flatten_named_values(data, flat)
                report.append(
                    {
                        "file": str(path),
                        "top_level_keys": sorted(list(data.keys())),
                        "flattened_keys": sorted(list(flat.keys()))[:300],
                        "num_flattened_keys": len(flat),
                        "shape_key_found": self._find_key(flat, self.SHAPE_KEYS)[0],
                        "exp_key_found": self._find_key(flat, self.EXP_KEYS)[0],
                        "pose_key_found": self._find_key(flat, self.POSE_KEYS)[0],
                        "tex_key_found": self._find_key(flat, self.TEX_KEYS)[0],
                    }
                )
            except Exception as exc:
                report.append({"file": str(path), "error": str(exc)})

        report_path = output_dir / "head_fusion_key_report.json"
        report_path.write_text(json.dumps(_json_safe(report), indent=2))
        return report_path

    def _load_crop_metadata(self, crop_index_path):
        if crop_index_path is None:
            return {}
        crop_index_path = Path(crop_index_path)
        if not crop_index_path.exists():
            return {}

        with open(crop_index_path, "r") as f:
            index = json.load(f)

        metadata = {}
        for item in index:
            crop_path = Path(item.get("crop_path", ""))
            meta_path = item.get("metadata_path")
            data = dict(item)

            if meta_path and Path(meta_path).exists():
                try:
                    with open(meta_path, "r") as mf:
                        data.update(json.load(mf))
                except Exception:
                    pass

            if crop_path.name:
                metadata[crop_path.stem] = data
                metadata[crop_path.name] = data

        return metadata

    def _collect_records(self, recon_dir, crop_metadata):
        records = []

        for p in sorted(recon_dir.rglob("*.mat")):
            r = self._record_from_data(p, self._load_mat(p), crop_metadata)
            if r:
                records.append(r)

        for p in sorted(recon_dir.rglob("*.npz")):
            r = self._record_from_data(p, self._load_npz(p), crop_metadata)
            if r:
                records.append(r)

        unique = {}
        for r in records:
            unique[str(r["param_path"])] = r
        return list(unique.values())

    def _load_mat(self, path):
        from scipy.io import loadmat
        raw = loadmat(path, squeeze_me=True, struct_as_record=False)
        return {k: v for k, v in raw.items() if not k.startswith("__")}

    def _load_npz(self, path):
        raw = np.load(path, allow_pickle=True)
        return {k: raw[k] for k in raw.files}

    def _record_from_data(self, param_path, data, crop_metadata):
        flat = {}
        self._flatten_named_values(data, flat)

        shape_key, shape = self._find_key(flat, self.SHAPE_KEYS)
        if shape is None:
            return None

        exp_key, exp = self._find_key(flat, self.EXP_KEYS)
        pose_key, pose = self._find_key(flat, self.POSE_KEYS)
        tex_key, tex = self._find_key(flat, self.TEX_KEYS)
        cam_key, cam = self._find_key(flat, self.CAM_KEYS)
        light_key, light = self._find_key(flat, self.LIGHT_KEYS)

        shape = self._flatten_code(shape)
        if not self._looks_like_code(shape, 5, 500):
            return None

        exp = self._flatten_code(exp) if exp is not None else np.zeros(50, dtype=np.float32)
        pose = self._flatten_code(pose) if pose is not None else np.zeros(6, dtype=np.float32)
        tex = self._flatten_code(tex) if tex is not None else None
        cam = self._flatten_code(cam) if cam is not None else None
        light = self._flatten_code(light) if light is not None else None

        meta = self._match_metadata(param_path, crop_metadata)

        crop_quality = float(meta.get("final_quality_score", meta.get("quality_score", 0.5)))
        validation_score = float(meta.get("crop_validation_score", 0.5))
        centeredness = float(meta.get("centeredness_score", 0.5))
        pose_penalty = self._pose_penalty(pose)
        expression_penalty = self._expression_penalty(exp)

        score = (
            0.30 * crop_quality +
            0.25 * validation_score +
            0.15 * centeredness +
            0.20 * (1.0 - pose_penalty) +
            0.10 * (1.0 - expression_penalty)
        )
        score = float(np.clip(score, 0.0, 1.0))

        return {
            "param_path": Path(param_path),
            "shape": shape,
            "expression": exp,
            "pose": pose,
            "texture": tex,
            "camera": cam,
            "light": light,
            "metadata": meta,
            "crop_quality": crop_quality,
            "validation_score": validation_score,
            "centeredness": centeredness,
            "pose_penalty": pose_penalty,
            "expression_penalty": expression_penalty,
            "score": score,
            "weight": 0.0,
            "matched_keys": {
                "shape": shape_key,
                "expression": exp_key,
                "pose": pose_key,
                "texture": tex_key,
                "camera": cam_key,
                "light": light_key,
            },
        }

    def _flatten_named_values(self, obj, out, prefix=""):
        if obj is None:
            return
        if isinstance(obj, dict):
            for k, v in obj.items():
                name = f"{prefix}.{k}" if prefix else str(k)
                self._flatten_named_values(v, out, name)
            return
        if hasattr(obj, "_fieldnames"):
            for k in obj._fieldnames:
                name = f"{prefix}.{k}" if prefix else str(k)
                self._flatten_named_values(getattr(obj, k), out, name)
            return
        if isinstance(obj, np.ndarray):
            if obj.dtype == object:
                if obj.size == 1:
                    self._flatten_named_values(obj.item(), out, prefix)
                    return
                for i, item in enumerate(obj.flat):
                    self._flatten_named_values(item, out, f"{prefix}[{i}]")
                return
            out[prefix] = obj
            return
        if np.isscalar(obj):
            out[prefix] = np.asarray(obj)
            return
        out[prefix] = obj

    def _find_key(self, flat, aliases):
        for alias in aliases:
            alias_l = alias.lower()
            for key, value in flat.items():
                leaf = key.split(".")[-1].lower()
                if leaf == alias_l:
                    arr = self._as_numeric_array(value)
                    if arr is not None:
                        return key, arr

        for alias in aliases:
            alias_l = alias.lower()
            for key, value in flat.items():
                leaf = key.split(".")[-1].lower()
                if alias_l in leaf:
                    arr = self._as_numeric_array(value)
                    if arr is not None:
                        return key, arr

        return None, None

    def _as_numeric_array(self, value):
        try:
            arr = np.asarray(value)
        except Exception:
            return None
        if arr.dtype == object or not np.issubdtype(arr.dtype, np.number):
            return None
        arr = np.asarray(arr, dtype=np.float32)
        if arr.size == 0 or np.isnan(arr).any() or np.isinf(arr).any():
            return None
        return arr

    def _flatten_code(self, arr):
        arr = np.asarray(arr, dtype=np.float32)
        arr = np.squeeze(arr)
        if arr.ndim > 1:
            arr = arr.reshape(-1)
        return arr.astype(np.float32)

    def _looks_like_code(self, arr, min_len=5, max_len=500):
        arr = np.asarray(arr).reshape(-1)
        return min_len <= len(arr) <= max_len

    def _match_metadata(self, param_path, crop_metadata):
        param_path = Path(param_path)
        for c in [param_path.stem, param_path.name, param_path.parent.name, param_path.parent.stem]:
            if c in crop_metadata:
                return crop_metadata[c]
        for key, value in crop_metadata.items():
            if key in param_path.stem or param_path.stem in key:
                return value
        return {}

    # =========================================================
    # GEOMETRY FUSION
    # =========================================================

    def _select_records(self, records):
        records = sorted(records, key=lambda r: r["score"], reverse=True)
        selected = [r for r in records if r["score"] >= self.min_score][: self.top_k]
        if not selected:
            selected = records[: min(1, len(records))]

        scores = np.asarray([max(r["score"], 1e-6) for r in selected], dtype=np.float32)
        weights = scores / scores.sum()
        for r, w in zip(selected, weights):
            r["weight"] = float(w)
        return selected

    def _weighted_average(self, selected):
        shape_len = len(selected[0]["shape"])
        exp_len = len(selected[0]["expression"])
        pose_len = len(selected[0]["pose"])
        tex_items = [r["texture"] for r in selected if r["texture"] is not None]
        tex_len = len(tex_items[0]) if tex_items else None

        shape = np.zeros(shape_len, dtype=np.float32)
        exp = np.zeros(exp_len, dtype=np.float32)
        pose = np.zeros(pose_len, dtype=np.float32)
        tex = np.zeros(tex_len, dtype=np.float32) if tex_len is not None else None
        tex_weight_sum = 0.0

        for r in selected:
            w = float(r["weight"])
            shape += w * self._fit_length(r["shape"], shape_len)
            exp += w * self._fit_length(r["expression"], exp_len)
            pose += w * self._fit_length(r["pose"], pose_len)
            if tex is not None and r["texture"] is not None:
                tex += w * self._fit_length(r["texture"], tex_len)
                tex_weight_sum += w

        if tex is not None and tex_weight_sum > 1e-8:
            tex /= tex_weight_sum
        if self.neutralize_expression:
            exp = np.zeros_like(exp)
        if self.neutralize_pose:
            pose = np.zeros_like(pose)

        return {
            "shape": shape.astype(np.float32),
            "expression": exp.astype(np.float32),
            "pose": pose.astype(np.float32),
            "texture": tex.astype(np.float32) if tex is not None else None,
        }

    def _fit_length(self, arr, length):
        arr = np.asarray(arr, dtype=np.float32).reshape(-1)
        out = np.zeros(length, dtype=np.float32)
        n = min(length, len(arr))
        out[:n] = arr[:n]
        return out

    def _pose_penalty(self, pose):
        if pose is None or len(pose) == 0:
            return 0.0
        pose = np.asarray(pose, dtype=np.float32).reshape(-1)
        mag = float(np.linalg.norm(pose[: min(6, len(pose))]))
        return float(np.clip(mag / 1.2, 0.0, 1.0))

    def _expression_penalty(self, exp):
        if exp is None or len(exp) == 0:
            return 0.0
        mag = float(np.linalg.norm(np.asarray(exp, dtype=np.float32).reshape(-1)))
        return float(np.clip(mag / 3.0, 0.0, 1.0))

    # =========================================================
    # MESH
    # =========================================================

    def _decode_or_fallback(self, fused, selected, recon_dir, output_dir):
        decode_result = self._decode_with_deca(fused=fused, output_dir=output_dir)
        if decode_result.get("success"):
            return decode_result
        return self._fallback_best_obj(
            selected=selected,
            recon_dir=recon_dir,
            output_dir=output_dir,
            reason=decode_result.get("error", "decode failed"),
        )

    def _decode_with_deca(self, fused, output_dir):
        try:
            _patch_numpy_for_deca()
            import torch

            deca_root = self.deca_root.resolve()
            if str(deca_root) not in sys.path:
                sys.path.insert(0, str(deca_root))

            from decalib.deca import DECA
            from decalib.utils.config import cfg as deca_cfg
            from .mesh_io import write_obj

            device = self.device if torch.cuda.is_available() and self.device == "cuda" else "cpu"
            deca = DECA(config=deca_cfg, device=device)

            shape = torch.tensor(fused["shape"][None, :], dtype=torch.float32, device=device)
            exp = torch.tensor(fused["expression"][None, :], dtype=torch.float32, device=device)
            pose = torch.tensor(fused["pose"][None, :], dtype=torch.float32, device=device)

            try:
                out = deca.flame(shape_params=shape, expression_params=exp, pose_params=pose)
                vertices = out[0] if isinstance(out, (tuple, list)) else out.vertices
            except Exception:
                decoded = deca.decode(
                    {"shape": shape, "exp": exp, "pose": pose},
                    rendering=False,
                    vis_lmk=False,
                    return_vis=False,
                )
                vertices = decoded["verts"]

            vertices_np = vertices.detach().cpu().numpy()[0]

            if hasattr(deca.flame, "faces_tensor"):
                faces = deca.flame.faces_tensor.detach().cpu().numpy()
            elif hasattr(deca.flame, "faces"):
                faces = np.asarray(deca.flame.faces)
            else:
                raise RuntimeError("Could not find FLAME faces")

            obj_path = output_dir / "canonical_fused_head.obj"
            write_obj(obj_path, vertices_np, faces)

            return {
                "success": True,
                "method": "deca_flame_decode",
                "obj_path": str(obj_path),
                "vertices": vertices_np,
                "faces": faces,
            }
        except Exception as exc:
            return {"success": False, "method": "deca_flame_decode", "error": str(exc)}

    def _fallback_best_obj(self, selected, recon_dir, output_dir, reason):
        obj_candidates = self._find_obj_candidates(recon_dir, preferred_stem=Path(selected[0]["param_path"]).stem)
        if not obj_candidates:
            return {"success": False, "method": "fallback_best_obj", "error": f"No OBJ fallback found. Decode error: {reason}"}
        src = obj_candidates[0]
        dst = output_dir / "canonical_fused_head_fallback_best.obj"
        shutil.copy2(src, dst)
        return {"success": True, "method": "fallback_best_obj", "obj_path": str(dst), "source_obj": str(src), "warning": f"Could not decode averaged FLAME params; copied best OBJ. Reason: {reason}"}

    def _obj_fallback_fusion(self, recon_dir, crop_metadata, output_dir):
        from .mesh_io import read_obj_vertices_faces, write_obj

        obj_candidates = self._find_obj_candidates(recon_dir)
        if not obj_candidates:
            return {"success": False, "method": "obj_fallback_fusion", "error": "No OBJ files found"}

        scored = []
        for p in obj_candidates:
            meta = self._match_metadata(p, crop_metadata)
            score = float(meta.get("final_quality_score", meta.get("quality_score", 0.5)))
            scored.append((score, p))
        scored.sort(reverse=True, key=lambda x: x[0])
        selected = scored[: self.top_k]

        meshes = []
        face_ref = None
        v_count = None
        for score, p in selected:
            try:
                v, f = read_obj_vertices_faces(p)
                if v_count is None:
                    v_count = len(v)
                    face_ref = f
                if len(v) == v_count:
                    meshes.append((score, v, p))
            except Exception:
                continue

        if not meshes:
            src = scored[0][1]
            dst = output_dir / "canonical_fused_head_fallback_best.obj"
            shutil.copy2(src, dst)
            return {"success": True, "method": "copy_best_obj", "obj_path": str(dst), "source_obj": str(src)}

        weights = np.asarray([max(m[0], 1e-6) for m in meshes], dtype=np.float32)
        weights = weights / weights.sum()
        vertices = np.zeros_like(meshes[0][1], dtype=np.float32)
        for w, (_, v, _) in zip(weights, meshes):
            vertices += w * v

        obj_path = output_dir / "canonical_fused_head_obj_average.obj"
        write_obj(obj_path, vertices, face_ref)
        return {"success": True, "method": "obj_vertex_average", "obj_path": str(obj_path), "num_meshes": len(meshes), "sources": [str(m[2]) for m in meshes], "warning": "Used OBJ vertex average because no FLAME parameter records were found."}

    def _find_obj_candidates(self, recon_dir, preferred_stem=None):
        all_objs = sorted(Path(recon_dir).rglob("*.obj"))
        objs = [p for p in all_objs if "template" not in p.name.lower()]
        if not objs:
            objs = all_objs
        if preferred_stem:
            objs = sorted(objs, key=lambda p: 0 if preferred_stem in p.stem or p.stem in preferred_stem else 1)
        return objs

    # =========================================================
    # TEXTURE QUALITY + BUILDING
    # =========================================================

    def _build_canonical_uv_texture(self, selected, recon_dir, output_dir):
        texture_items = self._find_uv_texture_images(selected=selected, recon_dir=recon_dir)
        scored = []
        rejected = []

        for recon_score, path in texture_items:
            img = self._load_texture_image(path)
            record = self._record_for_texture_path(path, selected)
            source_crop_path = self._source_crop_for_texture_path(path, record)
            q = self._score_texture_quality(img, recon_score, record, source_crop_path)

            item = {
                "path": path,
                "image": img,
                "quality": float(q["texture_quality"]),
                "metrics": q,
                "reconstruction_score": float(recon_score),
                "source_crop_path": str(source_crop_path) if source_crop_path else None,
            }

            if self._texture_candidate_allowed(q):
                scored.append(item)
            else:
                rejected.append(
                    {
                        "path": str(path),
                        "quality": float(q["texture_quality"]),
                        "reject_reasons": q.get("reject_reasons", []),
                        "metrics": _json_safe(q),
                        "source_crop_path": str(source_crop_path) if source_crop_path else None,
                    }
                )

        scored.sort(key=lambda x: x["quality"], reverse=True)
        rejected.sort(key=lambda x: x["quality"], reverse=True)

        quality_report_path = output_dir / "canonical_head_texture_quality_report.json"
        rejected_report_path = output_dir / "rejected_texture_candidates.json"

        quality_report = [
            {
                "path": str(item["path"]),
                "quality": float(item["quality"]),
                "metrics": _json_safe(item["metrics"]),
                "reconstruction_score": float(item["reconstruction_score"]),
                "source_crop_path": item["source_crop_path"],
            }
            for item in scored
        ]

        quality_report_path.write_text(json.dumps(quality_report, indent=2))
        rejected_report_path.write_text(json.dumps(rejected, indent=2))

        if not scored:
            return {
                "success": False,
                "method": "none",
                "quality_report": str(quality_report_path),
                "rejected_report": str(rejected_report_path),
                "message": (
                    "No UV texture passed strict crop/anatomy validation. "
                    "Texture export skipped to avoid poisoning the canonical head."
                ),
            }

        best = scored[0]

        if best["quality"] < self.texture_min_quality:
            return {
                "success": False,
                "method": "skipped_low_quality",
                "quality_report": str(quality_report_path),
                "rejected_report": str(rejected_report_path),
                "best_quality": float(best["quality"]),
                "best_source": str(best["path"]),
                "message": "Best UV texture quality is below threshold; texture export skipped.",
            }

        good_for_average = [
            item for item in scored
            if item["quality"] >= self.texture_average_min_quality
        ][: self.texture_top_k]

        use_average = (
            self.texture_mode == "weighted_average" and
            len(good_for_average) >= 2
        )

        if use_average:
            texture, valid_mask, used = self._weighted_average_valid_pixels(good_for_average)
            method = "weighted_average_valid_pixels"
        else:
            texture = best["image"].copy()
            valid_mask = self._valid_texture_mask(texture)
            used = [best]
            method = "best_uv_texture_strict"

        repaired = self._repair_texture(texture, valid_mask)
        texture_path = output_dir / "canonical_head_texture.png"
        cv2.imwrite(str(texture_path), repaired)

        debug_mask_path = output_dir / "canonical_head_texture_valid_mask.png"
        cv2.imwrite(str(debug_mask_path), (valid_mask.astype(np.uint8) * 255))

        return {
            "success": True,
            "method": method,
            "texture_path": str(texture_path),
            "quality_report": str(quality_report_path),
            "rejected_report": str(rejected_report_path),
            "valid_mask_path": str(debug_mask_path),
            "best_quality": float(best["quality"]),
            "best_source": str(best["path"]),
            "source_images": [str(item["path"]) for item in used],
            "used_count": len(used),
            "repair": "valid-mask dilation + OpenCV inpaint + invalid-area feathering",
        }

    def _texture_candidate_allowed(self, metrics):
        if not self.strict_texture_validation:
            return metrics["texture_quality"] >= self.texture_min_quality

        reject_reasons = metrics.get("reject_reasons", [])
        if reject_reasons:
            return False

        return metrics["texture_quality"] >= self.texture_min_quality

    def _find_uv_texture_images(self, selected, recon_dir):
        recon_dir = Path(recon_dir)
        selected_by_stem = {Path(r["param_path"]).stem: float(r["score"]) for r in selected}

        positive = ["uv_texture_gt", "uv_texture", "albedo"]
        negative = [
            "vis",
            "depth",
            "kpt",
            "landmark",
            "normal",
            "shape",
            "render",
            "input",
            "crop",
            "detail",
            "geometry",
            "codedict_tex",  # coefficient visualization, not a reliable UV texture
        ]

        candidates = []
        for ext in ["*.png", "*.jpg", "*.jpeg"]:
            for p in sorted(recon_dir.rglob(ext)):
                name = p.stem.lower()

                if not any(k in name for k in positive):
                    continue

                if any(k in name for k in negative):
                    continue

                score = 0.25
                for stem, s in selected_by_stem.items():
                    if stem in p.stem or p.stem in stem:
                        score = max(score, s)

                candidates.append((score, p))

        unique = {}
        for score, path in candidates:
            unique[str(path)] = (score, path)

        return sorted(unique.values(), key=lambda x: x[0], reverse=True)

    def _record_for_texture_path(self, texture_path, selected):
        texture_path = Path(texture_path)
        for r in selected:
            stem = Path(r["param_path"]).stem
            parent = Path(r["param_path"]).parent.name
            if stem in texture_path.stem or parent in texture_path.stem or parent == texture_path.parent.name:
                return r
        return None

    def _source_crop_for_texture_path(self, texture_path, record):
        if record:
            meta = record.get("metadata", {})
            for k in ["crop_path", "image_path", "source_image", "source_path"]:
                p = meta.get(k)
                if p and Path(p).exists():
                    return Path(p)

        # Try the paired input crop copied into recon/input_crops.
        texture_path = Path(texture_path)
        recon_dir = None
        for parent in texture_path.parents:
            if parent.name == "recon":
                recon_dir = parent
                break

        if recon_dir is not None:
            candidates = list((recon_dir / "input_crops").glob(f"{texture_path.parent.name}.*"))
            if candidates:
                return candidates[0]

        return None

    def _load_texture_image(self, path):
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Could not read image: {path}")
        img = cv2.resize(img, (self.texture_size, self.texture_size), interpolation=cv2.INTER_AREA)
        return img

    def _valid_texture_mask(self, img):
        img = np.asarray(img, dtype=np.uint8)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        not_black = gray > self.texture_black_threshold
        b, g, r = cv2.split(img)
        color_sum = b.astype(np.int32) + g.astype(np.int32) + r.astype(np.int32)
        valid = not_black & (color_sum > self.texture_black_threshold * 3)

        valid_u8 = valid.astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        valid_u8 = cv2.morphologyEx(valid_u8, cv2.MORPH_OPEN, kernel)
        valid_u8 = cv2.morphologyEx(valid_u8, cv2.MORPH_CLOSE, kernel)
        return valid_u8.astype(bool)

    def _score_texture_quality(self, img, reconstruction_score, record=None, source_crop_path=None):
        img = np.asarray(img, dtype=np.uint8)
        h, w = img.shape[:2]
        valid = self._valid_texture_mask(img)

        non_black_ratio = float(valid.mean())

        y1, y2 = int(h * 0.14), int(h * 0.86)
        x1, x2 = int(w * 0.15), int(w * 0.85)
        face_region = valid[y1:y2, x1:x2]
        face_region_validity = float(face_region.mean()) if face_region.size > 0 else 0.0

        low_banding_score = self._low_banding_score(img, valid)
        uv_symmetry_score = self._uv_symmetry_score(img, valid)
        uv_anatomy_score = self._uv_anatomy_score(img, valid)
        crop_completeness_score = self._crop_completeness_score(record, source_crop_path)

        texture_quality = (
            0.18 * non_black_ratio +
            0.18 * face_region_validity +
            0.16 * low_banding_score +
            0.18 * uv_symmetry_score +
            0.18 * uv_anatomy_score +
            0.12 * crop_completeness_score
        )

        reject_reasons = []

        # Hard gates for the actual failure case: partial eye crop / bad crop.
        if crop_completeness_score < 0.42:
            reject_reasons.append("source_crop_incomplete_or_eye_only")

        if uv_symmetry_score < 0.30:
            reject_reasons.append("uv_texture_too_asymmetric")

        if uv_anatomy_score < 0.28:
            reject_reasons.append("uv_anatomy_check_failed")

        if face_region_validity < 0.55:
            reject_reasons.append("low_central_face_validity")

        if non_black_ratio < 0.35:
            reject_reasons.append("too_much_empty_black_area")

        # If the texture has lots of coverage but terrible anatomy, this is usually a wrong crop.
        if non_black_ratio > 0.45 and uv_anatomy_score < 0.36:
            reject_reasons.append("coverage_high_but_face_anatomy_bad")

        return {
            "texture_quality": float(np.clip(texture_quality, 0.0, 1.0)),
            "non_black_ratio": non_black_ratio,
            "face_region_validity": face_region_validity,
            "low_banding_score": float(low_banding_score),
            "uv_symmetry_score": float(uv_symmetry_score),
            "uv_anatomy_score": float(uv_anatomy_score),
            "crop_completeness_score": float(crop_completeness_score),
            "crop_reconstruction_quality": float(np.clip(reconstruction_score, 0.0, 1.0)),
            "reject_reasons": reject_reasons,
        }

    def _low_banding_score(self, img, valid):
        img_f = img.astype(np.float32) / 255.0
        gray = cv2.cvtColor((img_f * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

        if valid.mean() < 0.05:
            return 0.0

        masked = gray.copy()
        masked[~valid] = np.nan

        col_mean = np.nanmean(masked, axis=0)
        row_mean = np.nanmean(masked, axis=1)

        col_mean = np.nan_to_num(col_mean, nan=float(np.nanmean(col_mean)))
        row_mean = np.nan_to_num(row_mean, nan=float(np.nanmean(row_mean)))

        col_var = float(np.std(col_mean))
        row_var = float(np.std(row_mean))
        valid_gray = gray[valid]
        overall = float(np.std(valid_gray)) + 1e-6

        vertical_banding = col_var / overall
        directional_excess = max(0.0, col_var - 0.75 * row_var)
        banding_score = vertical_banding + directional_excess * 4.0

        low_banding_score = 1.0 - np.clip((banding_score - 0.20) / 0.55, 0.0, 1.0)
        return float(low_banding_score)

    def _uv_symmetry_score(self, img, valid):
        """
        UV face textures should be roughly bilaterally balanced.
        A one-eye crop usually creates a very asymmetric UV texture.
        """
        img_f = img.astype(np.float32) / 255.0
        gray = cv2.cvtColor((img_f * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

        h, w = gray.shape
        x1, x2 = int(w * 0.18), int(w * 0.82)
        y1, y2 = int(h * 0.12), int(h * 0.88)

        g = gray[y1:y2, x1:x2]
        m = valid[y1:y2, x1:x2]

        if m.mean() < 0.2:
            return 0.0

        left = g[:, : g.shape[1] // 2]
        right = np.fliplr(g[:, g.shape[1] - left.shape[1] :])
        lm = m[:, : left.shape[1]]
        rm = np.fliplr(m[:, m.shape[1] - left.shape[1] :])
        both = lm & rm

        if both.mean() < 0.12:
            return 0.0

        diff = np.abs(left - right)[both]
        mad = float(np.mean(diff)) if diff.size else 1.0

        # Dark feature mass should also be reasonably balanced.
        dark = (g < 0.28) & m
        dark_left = float(dark[:, : dark.shape[1] // 2].mean())
        dark_right = float(dark[:, dark.shape[1] // 2 :].mean())
        dark_balance = 1.0 - min(1.0, abs(dark_left - dark_right) / max(dark_left + dark_right, 1e-4))

        symmetry = (1.0 - np.clip(mad / 0.30, 0.0, 1.0)) * 0.65 + dark_balance * 0.35
        return float(np.clip(symmetry, 0.0, 1.0))

    def _uv_anatomy_score(self, img, valid):
        """
        Very lightweight sanity check for UV face layout.

        It does not identify a person; it only checks whether the UV texture
        has plausible face content:
          - central skin-like face area exists
          - eye-band contains two roughly balanced dark feature regions
          - mouth/nose area is not dominated by a single giant dark/warped patch
        """
        img = np.asarray(img, dtype=np.uint8)
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        H, S, V = cv2.split(hsv)

        # broad skin-like mask; intentionally loose.
        skin = (
            (H >= 0) & (H <= 30) &
            (S >= 20) & (S <= 190) &
            (V >= 45)
        ) | (
            (H >= 160) & (H <= 179) &
            (S >= 20) & (S <= 190) &
            (V >= 45)
        )

        skin = skin & valid

        # Expected facial oval region in FLAME/DECA UV.
        face_y1, face_y2 = int(h * 0.16), int(h * 0.83)
        face_x1, face_x2 = int(w * 0.20), int(w * 0.80)
        face_skin = skin[face_y1:face_y2, face_x1:face_x2]
        face_valid = valid[face_y1:face_y2, face_x1:face_x2]

        face_skin_ratio = float(face_skin.mean()) if face_skin.size else 0.0
        face_valid_ratio = float(face_valid.mean()) if face_valid.size else 0.0

        # Eye band: two sides should both contain some dark detail, not one giant single-eye patch.
        eye_y1, eye_y2 = int(h * 0.30), int(h * 0.53)
        eye_x1, eye_x2 = int(w * 0.20), int(w * 0.80)
        eye_gray = gray[eye_y1:eye_y2, eye_x1:eye_x2]
        eye_valid = valid[eye_y1:eye_y2, eye_x1:eye_x2]

        dark = (eye_gray < 75) & eye_valid

        if dark.size == 0:
            eye_balance_score = 0.0
            giant_dark_penalty = 1.0
        else:
            left_dark = float(dark[:, : dark.shape[1] // 2].mean())
            right_dark = float(dark[:, dark.shape[1] // 2 :].mean())
            eye_balance_score = 1.0 - min(1.0, abs(left_dark - right_dark) / max(left_dark + right_dark, 1e-4))

            # connected dark components; one huge component is suspicious.
            dark_u8 = dark.astype(np.uint8)
            n, labels, stats, _ = cv2.connectedComponentsWithStats(dark_u8, connectivity=8)
            if n <= 1:
                giant_dark_penalty = 0.4
            else:
                areas = stats[1:, cv2.CC_STAT_AREA]
                largest_ratio = float(areas.max() / max(dark_u8.size, 1))
                giant_dark_penalty = 1.0 - np.clip((largest_ratio - 0.035) / 0.11, 0.0, 1.0)

        # Mouth/nose band should not be almost entirely black.
        lower_y1, lower_y2 = int(h * 0.48), int(h * 0.72)
        lower_x1, lower_x2 = int(w * 0.30), int(w * 0.70)
        lower_gray = gray[lower_y1:lower_y2, lower_x1:lower_x2]
        lower_valid = valid[lower_y1:lower_y2, lower_x1:lower_x2]
        lower_black_ratio = float(((lower_gray < 45) & lower_valid).mean()) if lower_gray.size else 1.0
        lower_score = 1.0 - np.clip((lower_black_ratio - 0.05) / 0.25, 0.0, 1.0)

        anatomy = (
            0.30 * np.clip(face_skin_ratio / 0.45, 0.0, 1.0) +
            0.20 * np.clip(face_valid_ratio / 0.70, 0.0, 1.0) +
            0.25 * eye_balance_score +
            0.15 * giant_dark_penalty +
            0.10 * lower_score
        )

        return float(np.clip(anatomy, 0.0, 1.0))

    def _crop_completeness_score(self, record, source_crop_path):
        """
        Reject eye-only / partial crops before they can affect texture.

        Uses metadata when available; otherwise uses image content heuristics.
        """
        meta_score = None

        if record is not None:
            meta = record.get("metadata", {})
            candidates = [
                "full_head_score",
                "head_completeness_score",
                "crop_validation_score",
                "final_quality_score",
                "quality_score",
                "face_validation_score",
            ]
            vals = []
            for k in candidates:
                if k in meta:
                    try:
                        vals.append(float(meta[k]))
                    except Exception:
                        pass

            if vals:
                meta_score = float(np.clip(np.mean(vals), 0.0, 1.0))

            # Hard metadata flags if your cropper writes them.
            for k in ["is_partial_face", "is_eye_only", "bad_crop", "invalid_crop"]:
                if bool(meta.get(k, False)):
                    return 0.0

        image_score = None
        if source_crop_path and Path(source_crop_path).exists():
            image_score = self._crop_image_completeness_score(Path(source_crop_path))

        if meta_score is not None and image_score is not None:
            return float(0.45 * meta_score + 0.55 * image_score)
        if image_score is not None:
            return float(image_score)
        if meta_score is not None:
            return float(meta_score)

        # Unknown crop quality: do not fully reject, but keep conservative.
        return 0.50

    def _crop_image_completeness_score(self, path):
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            return 0.25

        h, w = img.shape[:2]
        if h < 64 or w < 64:
            return 0.0

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        H, S, V = cv2.split(hsv)

        # Non-background / non-black coverage.
        foreground = gray > 18

        # Broad skin-like pixels.
        skin = (
            (H >= 0) & (H <= 30) & (S >= 18) & (S <= 205) & (V >= 40)
        ) | (
            (H >= 160) & (H <= 179) & (S >= 18) & (S <= 205) & (V >= 40)
        )
        skin = skin & foreground

        fg_ratio = float(foreground.mean())
        skin_ratio = float(skin.mean())

        # Bounding box of foreground. Eye-only crops usually have a tiny/odd box.
        ys, xs = np.where(foreground)
        if len(xs) == 0:
            return 0.0

        bbox_w = (xs.max() - xs.min() + 1) / float(w)
        bbox_h = (ys.max() - ys.min() + 1) / float(h)
        bbox_area = bbox_w * bbox_h

        # Face/head crops should occupy a meaningful area but not just a single corner.
        area_score = np.clip((bbox_area - 0.18) / 0.45, 0.0, 1.0)
        size_score = np.clip(min(bbox_w, bbox_h) / 0.48, 0.0, 1.0)

        # Centeredness.
        cx = (xs.min() + xs.max()) / 2.0 / max(w, 1)
        cy = (ys.min() + ys.max()) / 2.0 / max(h, 1)
        center_dist = np.sqrt((cx - 0.5) ** 2 + (cy - 0.50) ** 2)
        centeredness = 1.0 - np.clip(center_dist / 0.45, 0.0, 1.0)

        # Dark feature balance. A one-eye crop has extreme dark asymmetry.
        dark = (gray < 70) & foreground
        left_dark = float(dark[:, : w // 2].mean())
        right_dark = float(dark[:, w // 2 :].mean())
        dark_balance = 1.0 - min(1.0, abs(left_dark - right_dark) / max(left_dark + right_dark, 1e-4))

        # Reject crops with too much single dark component.
        n, labels, stats, _ = cv2.connectedComponentsWithStats(dark.astype(np.uint8), connectivity=8)
        if n > 1:
            largest_dark = float(stats[1:, cv2.CC_STAT_AREA].max() / max(h * w, 1))
        else:
            largest_dark = 0.0
        giant_dark_ok = 1.0 - np.clip((largest_dark - 0.08) / 0.18, 0.0, 1.0)

        # Very low skin coverage often means crop is only eye/hair/background.
        skin_score = np.clip((skin_ratio - 0.06) / 0.22, 0.0, 1.0)

        score = (
            0.22 * area_score +
            0.18 * size_score +
            0.18 * centeredness +
            0.18 * skin_score +
            0.14 * dark_balance +
            0.10 * giant_dark_ok
        )

        # Hard penalties.
        if bbox_area < 0.12:
            score *= 0.20
        if min(bbox_w, bbox_h) < 0.35:
            score *= 0.35
        if skin_ratio < 0.04:
            score *= 0.35

        return float(np.clip(score, 0.0, 1.0))

    def _weighted_average_valid_pixels(self, items):
        acc = None
        wacc = None
        used = []

        for item in items:
            img = item["image"].astype(np.float32)
            valid = self._valid_texture_mask(item["image"])
            w = float(max(item["quality"], 1e-6))

            if acc is None:
                acc = np.zeros_like(img, dtype=np.float32)
                wacc = np.zeros(img.shape[:2], dtype=np.float32)

            acc += img * valid[:, :, None].astype(np.float32) * w
            wacc += valid.astype(np.float32) * w
            used.append(item)

        avg = np.zeros_like(acc, dtype=np.float32)
        valid_any = wacc > 1e-8
        avg[valid_any] = acc[valid_any] / wacc[valid_any, None]

        return np.clip(avg, 0, 255).astype(np.uint8), valid_any, used

    def _repair_texture(self, img, valid_mask):
        img = np.asarray(img, dtype=np.uint8).copy()
        valid_mask = valid_mask.astype(bool)
        invalid = ~valid_mask

        if invalid.mean() < 0.001:
            return img

        invalid_u8 = invalid.astype(np.uint8) * 255
        kernel = np.ones((3, 3), np.uint8)
        invalid_for_inpaint = cv2.dilate(invalid_u8, kernel, iterations=1)

        filled = img.copy()
        current_valid = valid_mask.astype(np.uint8)

        for _ in range(24):
            if current_valid.all():
                break
            dilated = cv2.dilate(filled, kernel, iterations=1)
            dilated_valid = cv2.dilate(current_valid, kernel, iterations=1)
            update = (current_valid == 0) & (dilated_valid > 0)
            filled[update] = dilated[update]
            current_valid[update] = 1

        try:
            inpainted = cv2.inpaint(filled, invalid_for_inpaint, 3, cv2.INPAINT_TELEA)
        except Exception:
            inpainted = filled

        feather = cv2.GaussianBlur(invalid_for_inpaint.astype(np.float32) / 255.0, (0, 0), 2.0)
        feather = np.clip(feather[:, :, None], 0.0, 1.0)

        repaired = img.astype(np.float32) * (1.0 - feather) + inpainted.astype(np.float32) * feather
        repaired[invalid] = inpainted[invalid]

        return np.clip(repaired, 0, 255).astype(np.uint8)

    # =========================================================
    # TEXTURED PREVIEW
    # =========================================================

    def _create_textured_preview(self, mesh_result, texture_result, selected, recon_dir, output_dir):
        if not mesh_result.get("success"):
            return {"success": False, "message": "No mesh for preview"}
        if not texture_result.get("success"):
            return {"success": False, "message": "No UV texture for preview"}

        try:
            from .mesh_io import extract_uv_layout_from_obj, read_obj_vertices_faces, write_mtl, write_obj_with_uv

            texture_path = Path(texture_result["texture_path"])
            uv_source = self._find_best_uv_obj(selected=selected, recon_dir=recon_dir)
            if uv_source is None:
                return {"success": False, "message": "No source OBJ with UV coordinates found. Make sure DECA saved textured OBJ files."}

            uv_layout = extract_uv_layout_from_obj(uv_source)
            if uv_layout is None:
                return {"success": False, "message": f"Source OBJ has no usable UV layout: {uv_source}"}

            if "vertices" in mesh_result and "faces" in mesh_result:
                vertices = np.asarray(mesh_result["vertices"], dtype=np.float32)
                faces = np.asarray(mesh_result["faces"], dtype=np.int64)
            else:
                vertices, faces = read_obj_vertices_faces(mesh_result["obj_path"])

            if len(faces) != len(uv_layout["face_uvs"]):
                return {
                    "success": False,
                    "message": (
                        f"Face count mismatch between canonical mesh and UV source. "
                        f"canonical={len(faces)}, uv_source={len(uv_layout['face_uvs'])}"
                    ),
                    "uv_source_obj": str(uv_source),
                }

            preview_obj = output_dir / "canonical_fused_head_textured.obj"
            preview_mtl = output_dir / "canonical_fused_head_textured.mtl"
            local_texture = output_dir / "canonical_head_texture.png"

            if Path(texture_path).resolve() != local_texture.resolve():
                shutil.copy2(texture_path, local_texture)

            write_mtl(preview_mtl, texture_filename=local_texture.name)
            write_obj_with_uv(
                preview_obj,
                vertices=vertices,
                faces=faces,
                uv_coords=uv_layout["uv_coords"],
                face_uvs=uv_layout["face_uvs"],
                mtl_name=preview_mtl.name,
            )

            return {
                "success": True,
                "preview_obj": str(preview_obj),
                "preview_mtl": str(preview_mtl),
                "preview_texture": str(local_texture),
                "uv_source_obj": str(uv_source),
            }
        except Exception as exc:
            return {"success": False, "message": str(exc)}

    def _find_best_uv_obj(self, selected, recon_dir):
        from .mesh_io import extract_uv_layout_from_obj

        all_objs = self._find_obj_candidates(recon_dir)
        selected_stems = [Path(r["param_path"]).stem for r in selected]

        def rank_obj(p):
            p = Path(p)
            has_match = any(stem in p.stem or p.stem in stem for stem in selected_stems)
            return 0 if has_match else 1

        all_objs = sorted(all_objs, key=rank_obj)

        for p in all_objs:
            uv = extract_uv_layout_from_obj(p)
            if uv is not None:
                return p

        return None
