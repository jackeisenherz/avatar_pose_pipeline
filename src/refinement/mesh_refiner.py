# Drop-in replacement for: src/refinement/mesh_refiner.py
"""
Safer multi-view mesh refiner.

This class is intended for the phase after the initial SMPL-X/body fitting.
It refines a canonical mesh using low-frequency normal-direction offsets while
preserving the base body topology and avoiding aggressive freeform deformation.

Expected inputs are compatible with the earlier MeshRefiner:

    MeshRefiner(...).refine(
        canonical_body_path=...,
        image_paths=...,
        visibility_paths=...,
        output_path=...,
        iterations=...
    )

The code is conservative by design:
- scalar displacement along canonical vertex normals
- edge/Laplacian smoothing on scalar offsets
- offset magnitude penalty
- edge-length preservation
- anti-bloat penalty
- optional view weighting
- debug overlays
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from smplx_fit.renderer import SilhouetteRenderer

try:
    from .laplacian import LaplacianRegularizer
except Exception:  # pragma: no cover - allows direct script-style imports
    from laplacian import LaplacianRegularizer


class MeshRefiner:
    """
    Conservative low-frequency contour refiner.

    Important guidance
    ------------------
    This refiner should improve silhouette/body-contour alignment after the
    initial fit, but it should not be expected to fix bad camera pose, bad
    global scale, wrong body orientation, or severe keypoint/segmentation errors.

    If the rendered silhouette is far from the mask before refinement, rerun or
    improve the initial fitting stage first.
    """

    def __init__(
        self,
        image_size: int = 512,
        device: Optional[str] = None,

        # Geometry parameterization
        max_offset: float = 0.0060,
        smooth_steps: int = 10,
        smooth_alpha: float = 0.65,
        allow_inward: bool = True,
        allow_outward: bool = True,

        # Optimization
        lr: float = 0.0025,
        min_lr: float = 0.00025,
        grad_clip: float = 1.0,
        view_batch_size: int = 2,
        early_stop_patience: int = 80,
        early_stop_min_delta: float = 1e-5,

        # Silhouette/data losses
        silhouette_weight: float = 0.25,
        signed_distance_weight: float = 7.0,
        iou_weight: float = 1.15,
        boundary_weight: float = 0.08,
        false_positive_weight: float = 1.0,
        false_negative_weight: float = 0.45,

        # Regularization
        scalar_edge_weight: float = 500.0,
        scalar_lap_weight: float = 120.0,
        offset_l2_weight: float = 90.0,
        outward_weight: float = 90.0,
        edge_length_weight: float = 0.03,
        normal_consistency_weight: float = 0.01,

        # Debug
        debug: bool = True,
        debug_every: int = 25,
        debug_max_images: int = 6,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.image_size = int(image_size)

        self.max_offset = float(max_offset)
        self.smooth_steps = int(smooth_steps)
        self.smooth_alpha = float(smooth_alpha)
        self.allow_inward = bool(allow_inward)
        self.allow_outward = bool(allow_outward)

        self.lr = float(lr)
        self.min_lr = float(min_lr)
        self.grad_clip = float(grad_clip)
        self.view_batch_size = max(1, int(view_batch_size))
        self.early_stop_patience = int(early_stop_patience)
        self.early_stop_min_delta = float(early_stop_min_delta)

        self.silhouette_weight = float(silhouette_weight)
        self.signed_distance_weight = float(signed_distance_weight)
        self.iou_weight = float(iou_weight)
        self.boundary_weight = float(boundary_weight)
        self.false_positive_weight = float(false_positive_weight)
        self.false_negative_weight = float(false_negative_weight)

        self.scalar_edge_weight = float(scalar_edge_weight)
        self.scalar_lap_weight = float(scalar_lap_weight)
        self.offset_l2_weight = float(offset_l2_weight)
        self.outward_weight = float(outward_weight)
        self.edge_length_weight = float(edge_length_weight)
        self.normal_consistency_weight = float(normal_consistency_weight)

        self.debug = bool(debug)
        self.debug_every = int(debug_every)
        self.debug_max_images = int(debug_max_images)

        self.renderer = SilhouetteRenderer(
            image_size=self.image_size,
            device=self.device,
        )

        print(f"✓ Refiner device: {self.device}")
        print(f"✓ Refiner max offset: ±{self.max_offset:.4f}")
        print(f"✓ Refiner view batch size: {self.view_batch_size}")

    # ------------------------------------------------------------------
    # IO
    # ------------------------------------------------------------------

    def _load_npz(self, path):
        data = np.load(path, allow_pickle=True)
        return {k: data[k] for k in data.files}

    def _as_vertices(self, arr):
        arr = np.asarray(arr)
        while arr.ndim > 2:
            arr = arr[0]
        if arr.ndim != 2 or arr.shape[-1] != 3:
            raise ValueError(f"vertices must resolve to [V,3], got {arr.shape}")
        return arr.astype(np.float32)

    def _as_faces(self, arr):
        arr = np.asarray(arr)
        while arr.ndim > 2:
            arr = arr[0]
        if arr.ndim != 2 or arr.shape[-1] != 3:
            raise ValueError(f"faces must resolve to [F,3], got {arr.shape}")
        return arr.astype(np.int64)

    def _load_mask(self, image_path):
        rgba = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if rgba is None:
            raise RuntimeError(f"Could not load image/mask: {image_path}")

        if rgba.ndim == 3 and rgba.shape[2] == 4:
            alpha = rgba[:, :, 3]
        elif rgba.ndim == 2:
            alpha = rgba
        else:
            gray = cv2.cvtColor(rgba, cv2.COLOR_BGR2GRAY)
            alpha = (gray > 5).astype(np.uint8) * 255

        mask = (alpha > 10).astype(np.float32)
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        # Small cleanup. Conservative: do not aggressively close/open.
        mask_u8 = (mask > 0.5).astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=1)

        return torch.as_tensor(mask_u8, dtype=torch.float32, device=self.device).view(1, 1, self.image_size, self.image_size)

    def _load_visibility_weight(self, visibility_path) -> float:
        """
        Optional per-view weighting.

        Supports:
        - missing/None path -> 1.0
        - .npz with keys like visibility, visible, mask
        - image masks
        - plain text numeric value

        This is intentionally permissive because visibility files vary by project.
        """
        if visibility_path is None:
            return 1.0

        p = Path(visibility_path)
        if not p.exists():
            return 1.0

        try:
            if p.suffix.lower() == ".npz":
                d = np.load(p, allow_pickle=True)
                for key in ("visibility", "visible", "mask", "weights"):
                    if key in d.files:
                        arr = np.asarray(d[key])
                        return float(np.clip(arr.mean(), 0.05, 1.0))
            elif p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".webp"):
                m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                if m is not None:
                    return float(np.clip((m > 10).mean(), 0.05, 1.0))
            else:
                txt = p.read_text(encoding="utf-8", errors="ignore").strip()
                return float(np.clip(float(txt), 0.05, 1.0))
        except Exception:
            return 1.0

        return 1.0

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    def _smooth_scalars(self, scalars, regularizer: LaplacianRegularizer, steps=None, alpha=None):
        steps = self.smooth_steps if steps is None else int(steps)
        alpha = self.smooth_alpha if alpha is None else float(alpha)
        if steps <= 0:
            return scalars

        out = scalars
        for _ in range(steps):
            neigh = regularizer.neighbor_mean(out)
            out = (1.0 - alpha) * out + alpha * neigh
        return out

    def _vertex_normals(self, vertices, regularizer: LaplacianRegularizer):
        return regularizer.vertex_normals(vertices)

    def _clamp_scalars(self, scalars):
        lo = -self.max_offset if self.allow_inward else 0.0
        hi = self.max_offset if self.allow_outward else 0.0
        return torch.clamp(scalars, lo, hi)

    # ------------------------------------------------------------------
    # Image losses
    # ------------------------------------------------------------------

    def _distance_maps_from_mask(self, mask_tensor):
        mask_np = mask_tensor[0, 0].detach().cpu().numpy().astype(np.uint8)
        fg = (mask_np > 0).astype(np.uint8)

        dist_to_fg = cv2.distanceTransform(1 - fg, cv2.DIST_L2, 5)
        dist_to_bg = cv2.distanceTransform(fg, cv2.DIST_L2, 5)

        # Normalize to [0,1] but keep relative distances useful.
        dist_to_fg = dist_to_fg / max(float(dist_to_fg.max()), 1e-6)
        dist_to_bg = dist_to_bg / max(float(dist_to_bg.max()), 1e-6)

        signed = dist_to_fg - dist_to_bg  # outside positive, inside negative

        return (
            torch.as_tensor(dist_to_fg, dtype=torch.float32, device=self.device).view(1, 1, self.image_size, self.image_size),
            torch.as_tensor(dist_to_bg, dtype=torch.float32, device=self.device).view(1, 1, self.image_size, self.image_size),
            torch.as_tensor(signed, dtype=torch.float32, device=self.device).view(1, 1, self.image_size, self.image_size),
        )

    def _silhouette_loss(self, rendered, target):
        # Weighted BCE-like L1: penalize protrusions a bit more than missing silhouette.
        fp = rendered * (1.0 - target)
        fn = target * (1.0 - rendered)
        return self.false_positive_weight * fp.mean() + self.false_negative_weight * fn.mean()

    def _signed_distance_loss(self, rendered, target, dist_out, dist_in):
        false_positive = rendered * (1.0 - target)
        false_negative = target * (1.0 - rendered)
        return (
            self.false_positive_weight * (false_positive * dist_out).mean()
            + self.false_negative_weight * (false_negative * dist_in).mean()
        )

    def _iou_loss(self, rendered, target, eps=1e-6):
        intersection = (rendered * target).sum(dim=(1, 2, 3))
        union = (rendered + target - rendered * target).sum(dim=(1, 2, 3))
        return 1.0 - ((intersection + eps) / (union + eps)).mean()

    def _boundary_loss(self, rendered, target):
        # Sobel edges. This is weakly weighted because hard silhouettes are noisy.
        kernel_x = torch.tensor(
            [[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]],
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        kernel_y = torch.tensor(
            [[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]],
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        px = F.conv2d(rendered, kernel_x, padding=1)
        py = F.conv2d(rendered, kernel_y, padding=1)
        tx = F.conv2d(target, kernel_x, padding=1)
        ty = F.conv2d(target, kernel_y, padding=1)

        pe = torch.sqrt(px.square() + py.square() + 1e-6)
        te = torch.sqrt(tx.square() + ty.square() + 1e-6)
        return torch.abs(pe - te).mean()

    # ------------------------------------------------------------------
    # Debug
    # ------------------------------------------------------------------

    def _save_debug(self, debug_dir, iteration, image_index, target_mask, rendered_mask):
        if not self.debug:
            return

        debug_dir.mkdir(parents=True, exist_ok=True)

        target = target_mask[0, 0].detach().cpu().numpy()
        render = rendered_mask[0, 0].detach().cpu().numpy()

        target = (target > 0.5).astype(np.uint8)
        render = (render > 0.5).astype(np.uint8)

        img = np.zeros((target.shape[0], target.shape[1], 3), dtype=np.uint8)

        overlap = (target == 1) & (render == 1)
        target_only = (target == 1) & (render == 0)
        render_only = (target == 0) & (render == 1)

        # RGB before cvtColor:
        img[target_only] = [0, 255, 0]      # green = target not covered
        img[render_only] = [255, 0, 0]      # red = mesh protrusion
        img[overlap] = [255, 255, 0]        # yellow = overlap

        out_path = debug_dir / f"refine_iter_{iteration:04d}_img_{image_index:03d}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    # ------------------------------------------------------------------
    # Camera helpers
    # ------------------------------------------------------------------

    def _select_camera_param(self, tensor, i, batch_size):
        """
        Accepts camera params shaped [1,*] or [N,*].
        Returns [batch_size,*] for views i:i+batch_size.
        """
        if tensor.shape[0] == batch_size:
            return tensor
        if tensor.shape[0] == 1:
            return tensor.expand(batch_size, *tensor.shape[1:])
        return tensor[i : i + batch_size]

    # ------------------------------------------------------------------
    # Main
    # ------------------------------------------------------------------

    def refine(
        self,
        canonical_body_path,
        image_paths,
        visibility_paths=None,
        output_path=None,
        iterations: int = 300,
    ):
        print("\n🚀 Starting conservative multi-view mesh refinement")

        canonical_body_path = Path(canonical_body_path)
        if output_path is None:
            output_path = canonical_body_path.with_name(canonical_body_path.stem + "_refined.npz")
        output_path = Path(output_path)
        debug_dir = output_path.parent / "_render_debug"

        image_paths = [Path(p) for p in image_paths]
        if visibility_paths is None:
            visibility_paths = [None] * len(image_paths)
        else:
            visibility_paths = list(visibility_paths)
            if len(visibility_paths) != len(image_paths):
                raise ValueError("visibility_paths must be None or the same length as image_paths")

        payload = self._load_npz(canonical_body_path)

        neutral_vertices_np = self._as_vertices(payload["vertices"])
        faces_np = self._as_faces(payload["faces"])

        faces = torch.as_tensor(faces_np, dtype=torch.long, device=self.device)

        neutral_vertices = torch.as_tensor(
            neutral_vertices_np,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        num_views = len(image_paths)

        if "per_image_vertices" in payload:
            per_image_vertices_np = np.asarray(payload["per_image_vertices"], dtype=np.float32)
            if per_image_vertices_np.ndim == 2:
                per_image_vertices_np = per_image_vertices_np[None, :, :]
            if per_image_vertices_np.shape[0] == 1 and num_views > 1:
                per_image_vertices_np = np.repeat(per_image_vertices_np, num_views, axis=0)
        else:
            per_image_vertices_np = np.repeat(neutral_vertices_np[None, :, :], num_views, axis=0)

        if per_image_vertices_np.shape[0] != num_views:
            raise ValueError(
                f"per_image_vertices has {per_image_vertices_np.shape[0]} views, "
                f"but image_paths has {num_views}"
            )

        per_image_vertices = torch.as_tensor(per_image_vertices_np, dtype=torch.float32, device=self.device)

        translations_np = payload.get(
            "translations",
            np.tile(np.array([[0.0, 0.0, 5.0]], dtype=np.float32), (num_views, 1)),
        )
        translations = torch.as_tensor(translations_np, dtype=torch.float32, device=self.device)
        if translations.ndim == 1:
            translations = translations.view(1, -1)
        if translations.shape[0] == 1 and num_views > 1:
            translations = translations.expand(num_views, -1)

        focal_np = payload.get("focal_length", np.array([[1500.0, 1500.0]], dtype=np.float32))
        focal = torch.as_tensor(focal_np, dtype=torch.float32, device=self.device)
        if focal.ndim == 1:
            focal = focal.view(1, 2)

        camera_center_np = payload.get(
            "camera_center",
            np.array([[self.image_size / 2.0, self.image_size / 2.0]], dtype=np.float32),
        )
        camera_center = torch.as_tensor(camera_center_np, dtype=torch.float32, device=self.device)
        if camera_center.ndim == 1:
            camera_center = camera_center.view(1, 2)

        masks = [self._load_mask(p) for p in image_paths]
        dist_maps = [self._distance_maps_from_mask(m) for m in masks]

        view_weights = torch.as_tensor(
            [self._load_visibility_weight(p) for p in visibility_paths],
            dtype=torch.float32,
            device=self.device,
        )
        view_weights = view_weights / view_weights.mean().clamp_min(1e-6)

        regularizer = LaplacianRegularizer(faces, num_vertices=neutral_vertices.shape[1], device=self.device)
        normals = self._vertex_normals(neutral_vertices, regularizer).detach()

        scalar_offsets = nn.Parameter(
            torch.zeros(
                1,
                neutral_vertices.shape[1],
                1,
                dtype=torch.float32,
                device=self.device,
            )
        )

        optimizer = torch.optim.AdamW([scalar_offsets], lr=self.lr, weight_decay=0.0)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, int(iterations)),
            eta_min=self.min_lr,
        )

        best_loss = float("inf")
        best_scalars = None
        stale = 0
        last_losses = {}

        progress = tqdm(range(int(iterations)), desc="Refining", dynamic_ncols=True)

        for iteration in progress:
            optimizer.zero_grad(set_to_none=True)

            raw_scalars = self._clamp_scalars(scalar_offsets)
            smooth_scalars = self._smooth_scalars(raw_scalars, regularizer)
            vertex_offsets = normals * smooth_scalars

            total_data_loss = torch.zeros((), dtype=torch.float32, device=self.device)

            # Multi-view batching to reduce optimizer overhead while controlling VRAM.
            for start in range(0, num_views, self.view_batch_size):
                end = min(start + self.view_batch_size, num_views)
                bsz = end - start

                verts = per_image_vertices[start:end] + vertex_offsets.expand(bsz, -1, -1)

                rendered = self.renderer.render(
                    vertices=verts,
                    faces=faces.unsqueeze(0).expand(bsz, -1, -1),
                    focal_length=self._select_camera_param(focal, start, bsz),
                    principal_point=self._select_camera_param(camera_center, start, bsz),
                    translation=translations[start:end],
                )

                rendered_masks = rendered[..., 3].unsqueeze(1)

                batch_loss = torch.zeros((), dtype=torch.float32, device=self.device)

                for local_idx, global_idx in enumerate(range(start, end)):
                    rendered_mask = rendered_masks[local_idx : local_idx + 1]
                    target_mask = masks[global_idx]
                    dist_out, dist_in, _signed = dist_maps[global_idx]

                    loss_sil = self._silhouette_loss(rendered_mask, target_mask)
                    loss_dist = self._signed_distance_loss(rendered_mask, target_mask, dist_out, dist_in)
                    loss_iou = self._iou_loss(rendered_mask, target_mask)
                    loss_boundary = self._boundary_loss(rendered_mask, target_mask)

                    image_loss = (
                        self.silhouette_weight * loss_sil
                        + self.signed_distance_weight * loss_dist
                        + self.iou_weight * loss_iou
                        + self.boundary_weight * loss_boundary
                    )

                    image_loss = image_loss * view_weights[global_idx]
                    batch_loss = batch_loss + image_loss

                    last_losses = {
                        "sil": loss_sil.detach(),
                        "dist": loss_dist.detach(),
                        "iou": loss_iou.detach(),
                        "boundary": loss_boundary.detach(),
                    }

                    if (
                        self.debug
                        and (iteration % self.debug_every == 0 or iteration == int(iterations) - 1)
                        and global_idx < self.debug_max_images
                    ):
                        self._save_debug(
                            debug_dir=debug_dir,
                            iteration=iteration,
                            image_index=global_idx,
                            target_mask=target_mask,
                            rendered_mask=rendered_mask,
                        )

                total_data_loss = total_data_loss + batch_loss

            total_data_loss = total_data_loss / max(float(num_views), 1.0)

            loss_edge = regularizer.edge_smoothness(smooth_scalars, robust=False)
            loss_lap = regularizer.uniform_laplacian_loss(smooth_scalars, robust=True)
            loss_l2 = smooth_scalars.square().mean()
            loss_outward = torch.relu(smooth_scalars).square().mean()

            refined_neutral = neutral_vertices + vertex_offsets
            loss_edge_len = regularizer.edge_length_loss(refined_neutral, neutral_vertices, relative=True)
            loss_normals = regularizer.normal_consistency_loss(refined_neutral)

            total_loss = (
                total_data_loss
                + self.scalar_edge_weight * loss_edge
                + self.scalar_lap_weight * loss_lap
                + self.offset_l2_weight * loss_l2
                + self.outward_weight * loss_outward
                + self.edge_length_weight * loss_edge_len
                + self.normal_consistency_weight * loss_normals
            )

            total_loss.backward()

            if self.grad_clip and self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_([scalar_offsets], max_norm=self.grad_clip)

            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                scalar_offsets.copy_(self._clamp_scalars(scalar_offsets))

            current = float(total_loss.detach().item())
            if current + self.early_stop_min_delta < best_loss:
                best_loss = current
                best_scalars = scalar_offsets.detach().clone()
                stale = 0
            else:
                stale += 1

            if iteration % 25 == 0:
                tqdm.write(
                    f"\n[refine iter {iteration:04d}]\n"
                    f"  total:      {current:.6f}\n"
                    f"  data:       {total_data_loss.item():.6f}\n"
                    f"  sil:        {last_losses.get('sil', torch.tensor(0.0)).item():.6f}\n"
                    f"  dist:       {last_losses.get('dist', torch.tensor(0.0)).item():.6f}\n"
                    f"  iou:        {last_losses.get('iou', torch.tensor(0.0)).item():.6f}\n"
                    f"  boundary:   {last_losses.get('boundary', torch.tensor(0.0)).item():.6f}\n"
                    f"  edge:       {loss_edge.item():.8f}\n"
                    f"  lap:        {loss_lap.item():.8f}\n"
                    f"  off_l2:     {loss_l2.item():.8f}\n"
                    f"  outward:    {loss_outward.item():.8f}\n"
                    f"  edge_len:   {loss_edge_len.item():.8f}\n"
                    f"  normal:     {loss_normals.item():.8f}\n"
                    f"  lr:         {optimizer.param_groups[0]['lr']:.7f}\n"
                    f"  clamp:      ±{self.max_offset:.4f}"
                )

            progress.set_postfix(
                {
                    "total": f"{current:.3f}",
                    "data": f"{total_data_loss.item():.3f}",
                    "iou": f"{last_losses.get('iou', torch.tensor(0.0)).item():.3f}",
                    "stale": stale,
                }
            )

            if self.early_stop_patience > 0 and stale >= self.early_stop_patience:
                tqdm.write(
                    f"Early stopping refinement at iter {iteration}; "
                    f"no improvement for {self.early_stop_patience} iterations."
                )
                break

        with torch.no_grad():
            if best_scalars is None:
                best_scalars = scalar_offsets.detach().clone()

            final_scalars = self._smooth_scalars(
                self._clamp_scalars(best_scalars),
                regularizer,
            )
            final_offsets = normals * final_scalars
            refined_vertices = neutral_vertices + final_offsets
            refined_per_image_vertices = per_image_vertices + final_offsets.expand(num_views, -1, -1)

        result = dict(payload)
        result["vertices"] = refined_vertices.detach().cpu().numpy()
        result["faces"] = faces_np
        result["refinement_offsets"] = final_offsets.detach().cpu().numpy()
        result["refinement_scalar_offsets"] = final_scalars.detach().cpu().numpy()
        result["per_image_vertices"] = refined_per_image_vertices.detach().cpu().numpy()
        result["refinement_max_offset"] = np.array([self.max_offset], dtype=np.float32)
        result["refinement_best_loss"] = np.array([best_loss], dtype=np.float32)
        result["refinement_method"] = np.array(["conservative_multiview_low_frequency_normal_offsets"])

        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(output_path, **result)

        print(f"✅ Refined body saved: {output_path}")
        return result
