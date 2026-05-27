from pathlib import Path
import json

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from smplx_fit.renderer import SilhouetteRenderer
from smplx_fit.body_regions import BodyRegionWeights
from smplx_fit.losses import silhouette_loss
from .laplacian import LaplacianRegularizer


class MeshRefiner:
    """
    Smooth normal-only mesh refinement with anti-bloating.

    Positive scalar offset = outward expansion.
    Negative scalar offset = inward contraction.

    This version:
    - uses normal-only scalar offsets,
    - smooths offsets every iteration,
    - scales camera focal length if refinement image_size differs from SMPL-X,
    - uses conservative silhouette pressure,
    - penalizes positive/outward offsets to avoid making the body fatter,
    - logs full multiline diagnostics.
    """

    def __init__(
        self,
        image_size=512,
        offset_limit=0.006,
        lr=1e-4,
        smooth_steps=6,
        smooth_alpha=0.55,
        final_smooth_steps=20,
        debug=True,
        debug_every=25,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.image_size = int(image_size)
        self.offset_limit = float(offset_limit)
        self.lr = float(lr)
        self.smooth_steps = int(smooth_steps)
        self.smooth_alpha = float(smooth_alpha)
        self.final_smooth_steps = int(final_smooth_steps)
        self.debug = bool(debug)
        self.debug_every = int(debug_every)

        print(f"✓ MeshRefiner device: {self.device}")
        print(f"✓ Image size: {self.image_size}")
        print(f"✓ Normal-only offset limit: ±{self.offset_limit}")
        print(f"✓ LR: {self.lr}")
        print(f"✓ Smooth steps / iter: {self.smooth_steps}")
        print(f"✓ Final smooth steps: {self.final_smooth_steps}")
        print(f"✓ Render debug: {self.debug}")

        self.renderer = SilhouetteRenderer(image_size=self.image_size, device=self.device)

    def _load_target_mask(self, image_path):
        rgba = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if rgba is None:
            raise RuntimeError(f"Could not load image: {image_path}")

        if rgba.ndim == 3 and rgba.shape[2] == 4:
            alpha = rgba[:, :, 3]
        else:
            gray = cv2.cvtColor(rgba, cv2.COLOR_BGR2GRAY)
            alpha = (gray > 5).astype(np.uint8) * 255

        mask = (alpha > 10).astype(np.float32)
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        return torch.tensor(mask, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)

    def _distance_maps_from_mask(self, mask_tensor):
        mask_np = mask_tensor[0, 0].detach().cpu().numpy().astype(np.uint8)
        fg = (mask_np > 0).astype(np.uint8)

        dist_out = cv2.distanceTransform(1 - fg, cv2.DIST_L2, 5)
        dist_in = cv2.distanceTransform(fg, cv2.DIST_L2, 5)

        dist_out = dist_out / max(float(dist_out.max()), 1e-6)
        dist_in = dist_in / max(float(dist_in.max()), 1e-6)

        dist_out = torch.tensor(dist_out, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
        dist_in = torch.tensor(dist_in, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
        return dist_out, dist_in

    def _build_edges(self, faces):
        f = faces.long()
        edges = torch.cat([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], dim=0)
        edges = torch.sort(edges, dim=1).values
        edges = torch.unique(edges, dim=0)
        return edges.to(self.device)

    def _smooth_values(self, values, edges, steps=5, alpha=0.7):
        if steps <= 0:
            return values

        out = values
        src = edges[:, 0]
        dst = edges[:, 1]

        for _ in range(steps):
            base = out[0]
            sums = torch.zeros_like(base)
            counts = torch.zeros(base.shape[0], 1, dtype=base.dtype, device=base.device)

            sums.index_add_(0, src, base[dst])
            sums.index_add_(0, dst, base[src])

            ones = torch.ones(src.shape[0], 1, dtype=base.dtype, device=base.device)
            counts.index_add_(0, src, ones)
            counts.index_add_(0, dst, ones)

            neighbor_mean = sums / counts.clamp(min=1.0)
            out = ((1.0 - alpha) * base + alpha * neighbor_mean).unsqueeze(0)

        return out

    def _vertex_normals(self, vertices, faces):
        v = vertices[0]
        f = faces.long()

        v0 = v[f[:, 0]]
        v1 = v[f[:, 1]]
        v2 = v[f[:, 2]]

        face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)

        normals = torch.zeros_like(v)
        normals.index_add_(0, f[:, 0], face_normals)
        normals.index_add_(0, f[:, 1], face_normals)
        normals.index_add_(0, f[:, 2], face_normals)

        normals = F.normalize(normals, dim=-1, eps=1e-8)
        return normals.unsqueeze(0)

    def _distance_silhouette_loss(self, rendered_mask, target_mask, dist_out, dist_in):
        false_positive = rendered_mask * (1.0 - target_mask)
        false_negative = target_mask * (1.0 - rendered_mask)
        fp_loss = (false_positive * dist_out).mean()
        fn_loss = (false_negative * dist_in).mean()
        return fp_loss + 0.20 * fn_loss

    def _iou_loss(self, rendered_mask, target_mask, eps=1e-6):
        intersection = (rendered_mask * target_mask).sum(dim=(1, 2, 3))
        union = (rendered_mask + target_mask - rendered_mask * target_mask).sum(dim=(1, 2, 3))
        return 1.0 - ((intersection + eps) / (union + eps)).mean()

    def _edge_loss(self, rendered_mask, target_mask):
        kernel_x = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], dtype=torch.float32, device=self.device).unsqueeze(0)
        kernel_y = torch.tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], dtype=torch.float32, device=self.device).unsqueeze(0)

        pred_x = F.conv2d(rendered_mask, kernel_x, padding=1)
        pred_y = F.conv2d(rendered_mask, kernel_y, padding=1)
        tgt_x = F.conv2d(target_mask, kernel_x, padding=1)
        tgt_y = F.conv2d(target_mask, kernel_y, padding=1)

        pred_edge = torch.sqrt(pred_x ** 2 + pred_y ** 2 + 1e-6)
        tgt_edge = torch.sqrt(tgt_x ** 2 + tgt_y ** 2 + 1e-6)
        return torch.abs(pred_edge - tgt_edge).mean()

    def _load_scaled_camera(self, data):
        canonical_image_size = self.image_size
        if "image_size" in data.files:
            saved_size = np.asarray(data["image_size"]).reshape(-1)
            if len(saved_size) > 0:
                canonical_image_size = int(saved_size[0])

        scale = float(self.image_size) / float(canonical_image_size)

        focal_np = data["focal_length"] if "focal_length" in data.files else np.array([[1500.0, 1500.0]], dtype=np.float32)
        focal_np = np.asarray(focal_np, dtype=np.float32).reshape(1, 2)
        focal_np *= scale

        focal_length = torch.tensor(focal_np, dtype=torch.float32, device=self.device)
        camera_center = torch.tensor([[self.image_size / 2.0, self.image_size / 2.0]], dtype=torch.float32, device=self.device)

        print(
            f"✓ Refiner camera scale: canonical={canonical_image_size}, "
            f"refine={self.image_size}, scale={scale:.4f}, "
            f"focal=({focal_length[0,0].item():.2f}, {focal_length[0,1].item():.2f})"
        )
        return focal_length, camera_center

    def _save_mask_debug(self, rendered_mask, target_mask, out_path):
        pred = rendered_mask[0, 0].detach().cpu().numpy()
        tgt = target_mask[0, 0].detach().cpu().numpy()

        pred = (pred > 0.5).astype(np.uint8) * 255
        tgt = (tgt > 0.5).astype(np.uint8) * 255

        overlay = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
        overlay[:, :, 1] = tgt
        overlay[:, :, 2] = pred

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), overlay)

    def _export_obj(self, vertices, faces, obj_path):
        vertices = np.asarray(vertices)
        faces = np.asarray(faces)

        while vertices.ndim > 2:
            vertices = vertices[0]

        while faces.ndim > 2:
            faces = faces[0]

        vertices = vertices.astype(np.float32)
        faces = faces.astype(np.int32)

        if np.isnan(vertices).any() or np.isinf(vertices).any():
            raise RuntimeError("Cannot export OBJ: vertices contain NaN or Inf")

        obj_path = Path(obj_path)
        obj_path.parent.mkdir(parents=True, exist_ok=True)

        with open(obj_path, "w") as f:
            for v in vertices:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")
            for tri in faces:
                a, b, c = tri + 1
                f.write(f"f {int(a)} {int(b)} {int(c)}\n")

        print(f"✅ OBJ exported: {obj_path}")

    def refine(self, canonical_body_path, image_paths=None, visibility_paths=None, output_path=None, iterations=300):
        print("\n🚀 Starting anti-bloat smooth normal-only refinement")

        canonical_body_path = Path(canonical_body_path)
        data = np.load(canonical_body_path, allow_pickle=True)

        required = ["vertices", "faces", "per_image_vertices", "translations"]
        missing = [k for k in required if k not in data.files]
        if missing:
            raise RuntimeError(f"canonical_body.npz is missing refinement fields: {missing}. Re-run multi-image optimization first.")

        base_vertices = torch.tensor(data["vertices"], dtype=torch.float32, device=self.device)
        if base_vertices.dim() == 2:
            base_vertices = base_vertices.unsqueeze(0)

        faces = torch.tensor(data["faces"], dtype=torch.long, device=self.device)
        if faces.dim() == 3:
            faces = faces[0]

        faces_render = faces.unsqueeze(0)
        edges = self._build_edges(faces)

        per_image_vertices = torch.tensor(data["per_image_vertices"], dtype=torch.float32, device=self.device)
        translations = torch.tensor(data["translations"], dtype=torch.float32, device=self.device)

        if per_image_vertices.dim() != 3:
            raise RuntimeError("Expected per_image_vertices with shape [N, V, 3]")

        num_images, num_vertices, _ = per_image_vertices.shape

        if image_paths is None and "image_paths" in data.files:
            image_paths = [Path(str(p)) for p in data["image_paths"]]
        if visibility_paths is None and "visibility_json_paths" in data.files:
            visibility_paths = [Path(str(p)) for p in data["visibility_json_paths"]]
        if image_paths is None or visibility_paths is None:
            raise RuntimeError("image_paths and visibility_paths must be passed, or saved in canonical_body.npz")

        image_paths = [Path(p) for p in image_paths]
        visibility_paths = [Path(p) for p in visibility_paths]

        if len(image_paths) != num_images:
            raise RuntimeError(f"image_paths length {len(image_paths)} does not match per_image_vertices {num_images}")
        if len(visibility_paths) != num_images:
            raise RuntimeError(f"visibility_paths length {len(visibility_paths)} does not match per_image_vertices {num_images}")

        if output_path is None:
            output_path = canonical_body_path.parent / "refined_body.npz"
        output_path = Path(output_path)
        debug_dir = output_path.parent / "_render_debug"

        focal_length, camera_center = self._load_scaled_camera(data)

        masks, dist_out_all, dist_in_all, metadata_all = [], [], [], []
        for img_path, vis_path in zip(image_paths, visibility_paths):
            mask = self._load_target_mask(img_path)
            masks.append(mask)

            dist_out, dist_in = self._distance_maps_from_mask(mask)
            dist_out_all.append(dist_out)
            dist_in_all.append(dist_in)

            with open(vis_path, "r") as f:
                metadata_all.append(json.load(f))

        normals = self._vertex_normals(base_vertices, faces).detach()

        offset_scalars = nn.Parameter(torch.zeros(1, num_vertices, 1, device=self.device))
        optimizer = torch.optim.Adam([offset_scalars], lr=self.lr)
        laplacian = LaplacianRegularizer(faces, device=self.device)

        progress = tqdm(range(iterations), desc="Refining", dynamic_ncols=True, leave=True)

        for iteration in progress:
            optimizer.zero_grad()

            offset_scalars_clamped = torch.clamp(offset_scalars, -self.offset_limit, self.offset_limit)
            offset_scalars_smooth = self._smooth_values(
                offset_scalars_clamped,
                edges,
                steps=self.smooth_steps,
                alpha=self.smooth_alpha,
            )

            vertex_offsets = normals * offset_scalars_smooth
            total_loss = 0.0
            last = {}

            for i in range(num_images):
                vertices_i = per_image_vertices[i:i + 1] + vertex_offsets

                rendered = self.renderer.render(
                    vertices=vertices_i,
                    faces=faces_render,
                    focal_length=focal_length,
                    principal_point=camera_center,
                    translation=translations[i:i + 1],
                )

                rendered_mask = rendered[..., 3].unsqueeze(1)
                metadata = metadata_all[i]

                visibility_mask = torch.ones_like(masks[i])
                border = int(self.image_size * 0.12)

                if metadata.get("truncated_top", False):
                    visibility_mask[:, :, :border, :] *= 0.4
                if metadata.get("truncated_bottom", False):
                    visibility_mask[:, :, -border:, :] *= 0.4
                if metadata.get("truncated_left", False):
                    visibility_mask[:, :, :, :border] *= 0.4
                if metadata.get("truncated_right", False):
                    visibility_mask[:, :, :, -border:] *= 0.4

                region_weights = BodyRegionWeights.create_weight_map(
                    height=self.image_size,
                    width=self.image_size,
                    device=self.device,
                )

                if region_weights.dim() == 3:
                    region_weights = region_weights.unsqueeze(0)

                chest_boost = 1.0 + float(metadata.get("chest_visible", 0.0)) * 0.08
                hip_boost = 1.0 + float(metadata.get("hip_visible", 0.0)) * 0.08
                region_weights = region_weights * chest_boost * hip_boost

                loss_sil = silhouette_loss(rendered_mask, masks[i], visibility_mask, region_weights)
                loss_dist = self._distance_silhouette_loss(rendered_mask, masks[i], dist_out_all[i], dist_in_all[i])
                loss_iou = self._iou_loss(rendered_mask, masks[i])
                loss_edge = self._edge_loss(rendered_mask, masks[i])

                image_weight = float(metadata.get("image_weight", 1.0))
                total_loss = total_loss + image_weight * (
                    0.08 * loss_sil +
                    3.00 * loss_dist +
                    0.45 * loss_iou +
                    0.03 * loss_edge
                )

                if self.debug and i == 0 and iteration % self.debug_every == 0:
                    self._save_mask_debug(rendered_mask, masks[i], debug_dir / f"iter_{iteration:04d}_img_{i:03d}.png")

                last = {"sil": loss_sil, "dist": loss_dist, "iou": loss_iou, "edge": loss_edge}

            loss_lap_xyz = laplacian.offset_loss(vertex_offsets)
            loss_lap_scalar = laplacian.offset_loss(offset_scalars_smooth)
            loss_offset = (offset_scalars_smooth ** 2).mean()
            loss_raw_offset = (offset_scalars_clamped ** 2).mean()

            positive_offsets = torch.relu(offset_scalars_smooth)
            loss_outward = positive_offsets.mean()
            loss_outward_sq = (positive_offsets ** 2).mean()

            total = (
                total_loss +
                60.0 * loss_lap_xyz +
                120.0 * loss_lap_scalar +
                150.0 * loss_offset +
                25.0 * loss_raw_offset +
                80.0 * loss_outward +
                300.0 * loss_outward_sq
            )

            total.backward()
            optimizer.step()

            with torch.no_grad():
                offset_scalars.clamp_(-self.offset_limit, self.offset_limit)
                smoothed_param = self._smooth_values(offset_scalars, edges, steps=self.smooth_steps, alpha=self.smooth_alpha)
                offset_scalars.copy_(torch.clamp(smoothed_param, -self.offset_limit, self.offset_limit))

            if iteration % 10 == 0:
                progress.set_postfix({
                    "total": f"{total.item():.2f}",
                    "iou": f"{last.get('iou', torch.tensor(0.0)).item():.4f}",
                    "sil": f"{last.get('sil', torch.tensor(0.0)).item():.4f}",
                })

            if iteration % 25 == 0:
                tqdm.write(
                    "\n"
                    f"[anti-bloat refine iter {iteration:04d}]\n"
                    f"  total:        {total.item():.6f}\n"
                    f"  sil:          {last.get('sil', torch.tensor(0.0)).item():.6f}\n"
                    f"  dist:         {last.get('dist', torch.tensor(0.0)).item():.6f}\n"
                    f"  iou:          {last.get('iou', torch.tensor(0.0)).item():.6f}\n"
                    f"  edge:         {last.get('edge', torch.tensor(0.0)).item():.6f}\n"
                    f"  lap_xyz:      {loss_lap_xyz.item():.8f}\n"
                    f"  lap_scalar:   {loss_lap_scalar.item():.8f}\n"
                    f"  off_smooth:   {loss_offset.item():.8f}\n"
                    f"  off_raw:      {loss_raw_offset.item():.8f}\n"
                    f"  outward:      {loss_outward.item():.8f}\n"
                    f"  outward_sq:   {loss_outward_sq.item():.8f}\n"
                    f"  clamp:        ±{self.offset_limit:.4f}\n"
                )

        with torch.no_grad():
            final_scalars = torch.clamp(offset_scalars, -self.offset_limit, self.offset_limit)
            final_scalars = self._smooth_values(final_scalars, edges, steps=self.final_smooth_steps, alpha=self.smooth_alpha)
            final_scalars = torch.clamp(final_scalars, -self.offset_limit, self.offset_limit)

            final_offsets = normals * final_scalars
            final_vertices = base_vertices + final_offsets

        result = {
            "vertices": final_vertices.detach().cpu().numpy(),
            "faces": faces.detach().cpu().numpy(),
            "offsets": final_offsets.detach().cpu().numpy(),
            "offset_scalars": final_scalars.detach().cpu().numpy(),
            "source_canonical": str(canonical_body_path),
            "num_images": num_images,
            "image_size": np.array([self.image_size], dtype=np.int32),
            "focal_length": focal_length.detach().cpu().numpy(),
            "camera_center": camera_center.detach().cpu().numpy(),
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(output_path, **result)

        obj_path = output_path.with_suffix(".obj")
        self._export_obj(result["vertices"], result["faces"], obj_path)

        print(f"\n✅ Refined mesh saved:\n{output_path}")
        return result
