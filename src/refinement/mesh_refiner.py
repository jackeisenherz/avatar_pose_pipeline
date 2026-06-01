# Drop-in replacement for: src/refinement/mesh_refiner.py

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from smplx_fit.renderer import SilhouetteRenderer


class MeshRefiner:
    """
    Safe, low-frequency contour refiner.

    This refiner intentionally avoids aggressive freeform deformation. It learns
    a very small scalar displacement field along the canonical vertex normals,
    heavily smooths it, penalizes outward bloat, and writes debug overlays.

    Interface-compatible with:
        mesh_refiner.refine(
            canonical_body_path=...,
            image_paths=...,
            visibility_paths=...,
            output_path=...,
            iterations=...
        )
    """

    def __init__(
        self,
        image_size=512,
        device=None,
        max_offset=0.0040,
        smooth_steps=12,
        smooth_alpha=0.75,
        silhouette_weight=0.30,
        distance_weight=8.0,
        iou_weight=1.30,
        edge_weight=0.05,
        laplacian_weight=350.0,
        offset_l2_weight=80.0,
        outward_weight=120.0,
        debug=True,
        debug_every=25,
        debug_max_images=6,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.image_size = int(image_size)
        self.max_offset = float(max_offset)
        self.smooth_steps = int(smooth_steps)
        self.smooth_alpha = float(smooth_alpha)

        self.silhouette_weight = float(silhouette_weight)
        self.distance_weight = float(distance_weight)
        self.iou_weight = float(iou_weight)
        self.edge_weight = float(edge_weight)
        self.laplacian_weight = float(laplacian_weight)
        self.offset_l2_weight = float(offset_l2_weight)
        self.outward_weight = float(outward_weight)

        self.debug = bool(debug)
        self.debug_every = int(debug_every)
        self.debug_max_images = int(debug_max_images)

        self.renderer = SilhouetteRenderer(
            image_size=self.image_size,
            device=self.device,
        )

        print(f"✓ Refiner device: {self.device}")
        print(f"✓ Refiner max offset: ±{self.max_offset:.4f}")

    # =========================================================
    # IO
    # =========================================================

    def _load_npz(self, path):
        data = np.load(path, allow_pickle=True)
        return {k: data[k] for k in data.files}

    def _as_vertices(self, arr):
        arr = np.asarray(arr)
        while arr.ndim > 2:
            arr = arr[0]
        return arr.astype(np.float32)

    def _as_faces(self, arr):
        arr = np.asarray(arr)
        while arr.ndim > 2:
            arr = arr[0]
        return arr.astype(np.int64)

    def _load_mask(self, image_path):
        rgba = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

        if rgba is None:
            raise RuntimeError(f"Could not load image: {image_path}")

        if rgba.ndim == 3 and rgba.shape[2] == 4:
            alpha = rgba[:, :, 3]
        else:
            gray = cv2.cvtColor(rgba, cv2.COLOR_BGR2GRAY)
            alpha = (gray > 5).astype(np.uint8) * 255

        mask = (alpha > 10).astype(np.float32)
        mask = cv2.resize(
            mask,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_NEAREST,
        )

        return torch.tensor(
            mask,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0).unsqueeze(0)

    # =========================================================
    # GEOMETRY
    # =========================================================

    def _build_edges(self, faces, num_vertices):
        edges = torch.cat(
            [
                faces[:, [0, 1]],
                faces[:, [1, 2]],
                faces[:, [2, 0]],
            ],
            dim=0,
        )
        edges = torch.sort(edges, dim=1).values
        edges = torch.unique(edges, dim=0)
        return edges.long()

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

    def _smooth_scalars(self, scalars, edges, steps=None, alpha=None):
        steps = self.smooth_steps if steps is None else int(steps)
        alpha = self.smooth_alpha if alpha is None else float(alpha)

        if steps <= 0:
            return scalars

        out = scalars
        src = edges[:, 0]
        dst = edges[:, 1]

        for _ in range(steps):
            base = out[0]
            sums = torch.zeros_like(base)
            counts = torch.zeros(
                base.shape[0],
                1,
                dtype=base.dtype,
                device=base.device,
            )

            sums.index_add_(0, src, base[dst])
            sums.index_add_(0, dst, base[src])

            ones = torch.ones(
                src.shape[0],
                1,
                dtype=base.dtype,
                device=base.device,
            )

            counts.index_add_(0, src, ones)
            counts.index_add_(0, dst, ones)

            neighbor_mean = sums / counts.clamp(min=1.0)
            out = ((1.0 - alpha) * base + alpha * neighbor_mean).unsqueeze(0)

        return out

    # =========================================================
    # LOSSES
    # =========================================================

    def _distance_maps_from_mask(self, mask_tensor):
        mask_np = mask_tensor[0, 0].detach().cpu().numpy().astype(np.uint8)
        fg = (mask_np > 0).astype(np.uint8)

        dist_out = cv2.distanceTransform(1 - fg, cv2.DIST_L2, 5)
        dist_in = cv2.distanceTransform(fg, cv2.DIST_L2, 5)

        dist_out = dist_out / max(float(dist_out.max()), 1e-6)
        dist_in = dist_in / max(float(dist_in.max()), 1e-6)

        return (
            torch.tensor(dist_out, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0),
            torch.tensor(dist_in, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0),
        )

    def _silhouette_loss(self, rendered, target):
        return torch.abs(rendered - target).mean()

    def _distance_loss(self, rendered, target, dist_out, dist_in):
        false_positive = rendered * (1.0 - target)
        false_negative = target * (1.0 - rendered)
        return (false_positive * dist_out).mean() + 0.25 * (false_negative * dist_in).mean()

    def _iou_loss(self, rendered, target, eps=1e-6):
        intersection = (rendered * target).sum(dim=(1, 2, 3))
        union = (rendered + target - rendered * target).sum(dim=(1, 2, 3))
        return 1.0 - ((intersection + eps) / (union + eps)).mean()

    def _edge_loss(self, rendered, target):
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

        pe = torch.sqrt(px ** 2 + py ** 2 + 1e-6)
        te = torch.sqrt(tx ** 2 + ty ** 2 + 1e-6)

        return torch.abs(pe - te).mean()

    def _laplacian_scalar_loss(self, scalars, edges):
        src = edges[:, 0]
        dst = edges[:, 1]
        return ((scalars[:, src] - scalars[:, dst]) ** 2).mean()

    def _outward_bloat_loss(self, scalars):
        return (torch.relu(scalars) ** 2).mean()

    # =========================================================
    # DEBUG
    # =========================================================

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

        img[target_only] = [0, 255, 0]
        img[render_only] = [255, 0, 0]
        img[overlap] = [255, 255, 0]

        out_path = debug_dir / f"refine_iter_{iteration:04d}_img_{image_index:03d}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    # =========================================================
    # MAIN
    # =========================================================

    def refine(
        self,
        canonical_body_path,
        image_paths,
        visibility_paths,
        output_path,
        iterations=300,
    ):
        print("\n🚀 Starting safe low-frequency refinement")

        canonical_body_path = Path(canonical_body_path)
        output_path = Path(output_path)
        debug_dir = output_path.parent / "_render_debug"

        payload = self._load_npz(canonical_body_path)

        neutral_vertices_np = self._as_vertices(payload["vertices"])
        faces_np = self._as_faces(payload["faces"])

        faces = torch.tensor(
            faces_np,
            dtype=torch.long,
            device=self.device,
        )

        neutral_vertices = torch.tensor(
            neutral_vertices_np,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        if "per_image_vertices" in payload:
            per_image_vertices_np = payload["per_image_vertices"].astype(np.float32)
        else:
            per_image_vertices_np = np.repeat(
                neutral_vertices_np[None, :, :],
                len(image_paths),
                axis=0,
            )

        per_image_vertices = torch.tensor(
            per_image_vertices_np,
            dtype=torch.float32,
            device=self.device,
        )

        translations = torch.tensor(
            payload.get(
                "translations",
                np.tile(np.array([[0.0, 0.0, 5.0]], dtype=np.float32), (len(image_paths), 1)),
            ),
            dtype=torch.float32,
            device=self.device,
        )

        focal = payload.get("focal_length", np.array([[1500.0, 1500.0]], dtype=np.float32))
        focal = torch.tensor(focal, dtype=torch.float32, device=self.device)

        if focal.ndim == 1:
            focal = focal.view(1, 2)

        camera_center = payload.get(
            "camera_center",
            np.array([[self.image_size / 2.0, self.image_size / 2.0]], dtype=np.float32),
        )

        camera_center = torch.tensor(
            camera_center,
            dtype=torch.float32,
            device=self.device,
        )

        if camera_center.ndim == 1:
            camera_center = camera_center.view(1, 2)

        masks = [self._load_mask(p) for p in image_paths]
        dist_maps = [self._distance_maps_from_mask(m) for m in masks]

        edges = self._build_edges(faces, neutral_vertices.shape[1])
        normals = self._vertex_normals(neutral_vertices, faces)

        scalar_offsets = nn.Parameter(
            torch.zeros(
                1,
                neutral_vertices.shape[1],
                1,
                dtype=torch.float32,
                device=self.device,
            )
        )

        optimizer = torch.optim.Adam([scalar_offsets], lr=0.003)

        progress = tqdm(range(iterations), desc="Refining", dynamic_ncols=True)

        last = {}

        for iteration in progress:
            optimizer.zero_grad()

            raw_scalars = torch.clamp(
                scalar_offsets,
                -self.max_offset,
                self.max_offset,
            )

            smooth_scalars = self._smooth_scalars(raw_scalars, edges)
            vertex_offsets = normals * smooth_scalars

            total_loss = torch.tensor(0.0, device=self.device)

            for i in range(len(image_paths)):
                verts = per_image_vertices[i:i + 1] + vertex_offsets

                rendered = self.renderer.render(
                    vertices=verts,
                    faces=faces.unsqueeze(0),
                    focal_length=focal,
                    principal_point=camera_center,
                    translation=translations[i:i + 1],
                )

                rendered_mask = rendered[..., 3].unsqueeze(1)

                target_mask = masks[i]
                dist_out, dist_in = dist_maps[i]

                loss_sil = self._silhouette_loss(rendered_mask, target_mask)
                loss_dist = self._distance_loss(rendered_mask, target_mask, dist_out, dist_in)
                loss_iou = self._iou_loss(rendered_mask, target_mask)
                loss_edge = self._edge_loss(rendered_mask, target_mask)

                image_loss = (
                    self.silhouette_weight * loss_sil +
                    self.distance_weight * loss_dist +
                    self.iou_weight * loss_iou +
                    self.edge_weight * loss_edge
                )

                total_loss = total_loss + image_loss

                last = {
                    "sil": loss_sil,
                    "dist": loss_dist,
                    "iou": loss_iou,
                    "edge": loss_edge,
                }

                if self.debug and (iteration % self.debug_every == 0 or iteration == iterations - 1) and i < self.debug_max_images:
                    self._save_debug(
                        debug_dir=debug_dir,
                        iteration=iteration,
                        image_index=i,
                        target_mask=target_mask,
                        rendered_mask=rendered_mask,
                    )

            loss_lap = self._laplacian_scalar_loss(smooth_scalars, edges)
            loss_l2 = (smooth_scalars ** 2).mean()
            loss_outward = self._outward_bloat_loss(smooth_scalars)

            total_loss = total_loss + (
                self.laplacian_weight * loss_lap +
                self.offset_l2_weight * loss_l2 +
                self.outward_weight * loss_outward
            )

            total_loss.backward()
            optimizer.step()

            with torch.no_grad():
                scalar_offsets.clamp_(-self.max_offset, self.max_offset)

            if iteration % 25 == 0:
                tqdm.write(
                    f"\n[safe refine iter {iteration:04d}]\n"
                    f"  total:      {total_loss.item():.6f}\n"
                    f"  sil:        {last.get('sil', torch.tensor(0.0)).item():.6f}\n"
                    f"  dist:       {last.get('dist', torch.tensor(0.0)).item():.6f}\n"
                    f"  iou:        {last.get('iou', torch.tensor(0.0)).item():.6f}\n"
                    f"  edge:       {last.get('edge', torch.tensor(0.0)).item():.6f}\n"
                    f"  lap:        {loss_lap.item():.8f}\n"
                    f"  off_l2:     {loss_l2.item():.8f}\n"
                    f"  outward:    {loss_outward.item():.8f}\n"
                    f"  clamp:      ±{self.max_offset:.4f}"
                )

            progress.set_postfix({
                "total": f"{total_loss.item():.2f}",
                "iou": f"{last.get('iou', torch.tensor(0.0)).item():.4f}",
                "sil": f"{last.get('sil', torch.tensor(0.0)).item():.4f}",
            })

        with torch.no_grad():
            final_scalars = self._smooth_scalars(
                torch.clamp(scalar_offsets, -self.max_offset, self.max_offset),
                edges,
            )
            final_offsets = normals * final_scalars
            refined_vertices = neutral_vertices + final_offsets
            refined_per_image_vertices = per_image_vertices + final_offsets

        result = dict(payload)
        result["vertices"] = refined_vertices.detach().cpu().numpy()
        result["faces"] = faces_np
        result["refinement_offsets"] = final_offsets.detach().cpu().numpy()
        result["refinement_scalar_offsets"] = final_scalars.detach().cpu().numpy()
        result["per_image_vertices"] = refined_per_image_vertices.detach().cpu().numpy()
        result["refinement_max_offset"] = np.array([self.max_offset], dtype=np.float32)
        result["refinement_method"] = np.array(["safe_low_frequency_normal_offsets"])

        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(output_path, **result)

        print(f"✅ Refined body saved: {output_path}")
        return result
