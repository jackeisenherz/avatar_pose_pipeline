
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import json, cv2, numpy as np
from .uv_utils import read_obj_with_uv, write_textured_obj, render_vertex_visibility_map, barycentric_sample_image

@dataclass
class BodyTextureFusionConfig:
    texture_size: int = 2048
    min_view_weight: float = 0.05
    erode_mask_px: int = 3
    feather_px: int = 9
    inpaint_radius: int = 3
    blend_mode: str = "weighted"
    save_debug: bool = True

class BodyTextureFusion:
    def __init__(self, config: BodyTextureFusionConfig | None = None):
        self.config = config or BodyTextureFusionConfig()

    def fuse(self, mesh_obj_path, image_paths, alpha_paths, camera_json_paths, output_dir):
        output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
        verts, faces, uvs, face_uvs, _, _ = read_obj_with_uv(mesh_obj_path)
        if uvs is None or face_uvs is None:
            raise RuntimeError("Mesh OBJ has no UVs.")
        ts = int(self.config.texture_size)
        accum = np.zeros((ts, ts, 3), np.float32)
        weights = np.zeros((ts, ts), np.float32)
        best_rgb = np.zeros((ts, ts, 3), np.uint8)
        best_score = np.zeros((ts, ts), np.float32)
        per_view = []
        for idx, (img_p, alpha_p, cam_p) in enumerate(zip(image_paths, alpha_paths, camera_json_paths)):
            img = cv2.imread(str(img_p), cv2.IMREAD_COLOR)
            alpha = cv2.imread(str(alpha_p), cv2.IMREAD_UNCHANGED)
            if img is None or alpha is None or not Path(cam_p).exists():
                continue
            amask = alpha[:,:,3] if alpha.ndim == 3 and alpha.shape[2] == 4 else (alpha if alpha.ndim == 2 else cv2.cvtColor(alpha, cv2.COLOR_BGR2GRAY))
            cam = json.loads(Path(cam_p).read_text())
            uv_vis = render_vertex_visibility_map(verts, faces, uvs, face_uvs, cam, image_size=(img.shape[1], img.shape[0]), texture_size=ts)
            rgb_tex, valid_mask, view_score = barycentric_sample_image(uv_vis, img, amask, erode_px=self.config.erode_mask_px)
            if view_score <= 0:
                per_view.append({"image": str(img_p), "used": False, "score": 0.0}); continue
            if self.config.blend_mode == "best":
                take = (valid_mask > 0) & (view_score > best_score)
                best_rgb[take] = rgb_tex[take]; best_score[take] = view_score
            else:
                accum += rgb_tex.astype(np.float32) * valid_mask[...,None] * view_score
                weights += valid_mask.astype(np.float32) * view_score
                take = (valid_mask > 0) & (view_score > best_score)
                best_rgb[take] = rgb_tex[take]; best_score[take] = view_score
            per_view.append({"image": str(img_p), "used": True, "score": float(view_score)})
            if self.config.save_debug:
                cv2.imwrite(str(output_dir / f"view_{idx:03d}_valid_mask.png"), (valid_mask * 255).astype(np.uint8))
        if self.config.blend_mode == "best":
            tex = best_rgb.copy(); valid = best_score > 0
        else:
            tex = np.zeros_like(best_rgb); valid = weights > 1e-8
            tex[valid] = (accum[valid] / weights[valid,None]).clip(0,255).astype(np.uint8)
            miss = ~valid; tex[miss] = best_rgb[miss]; valid = valid | (best_score > 0)
        tex = self._repair_texture(tex, valid.astype(np.uint8))
        texture_path = output_dir / "canonical_body_texture.png"; cv2.imwrite(str(texture_path), tex)
        out_obj = output_dir / "textured_avatar.obj"; out_mtl = output_dir / "textured_avatar.mtl"
        write_textured_obj(mesh_obj_path, out_obj, out_mtl, texture_path.name)
        summary = {"mesh_obj": str(mesh_obj_path), "texture_path": str(texture_path), "textured_obj": str(out_obj), "num_input_images": len(image_paths), "num_views_used": int(sum(1 for x in per_view if x["used"])), "coverage_ratio": float(valid.mean()), "views": per_view, "config": asdict(self.config)}
        (output_dir / "texture_summary.json").write_text(json.dumps(summary, indent=2))
        return summary

    def _repair_texture(self, tex, valid_mask):
        inv = (valid_mask == 0).astype(np.uint8) * 255
        if inv.any():
            tex = cv2.inpaint(tex, inv, self.config.inpaint_radius, cv2.INPAINT_TELEA)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.config.feather_px, self.config.feather_px))
        halo = cv2.dilate(valid_mask, k, iterations=1) - cv2.erode(valid_mask, k, iterations=1)
        blur = cv2.GaussianBlur(tex, (0,0), 1.0); tex[halo > 0] = blur[halo > 0]
        return tex
