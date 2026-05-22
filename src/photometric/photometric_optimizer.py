# ============================================================
# FILE:
# src/photometric/photometric_optimizer.py
# ============================================================

import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

from tqdm import tqdm

from .rgb_renderer import RGBRenderer

from .photometric_losses import (
    photometric_loss,
    texture_smoothness_loss
)

from refinement.laplacian import (
    LaplacianRegularizer
)


class PhotometricOptimizer:

    def __init__(
        self,
        image_size=1024,
        texture_size=2048
    ):

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(
            f"✓ Photometric optimizer:"
            f" {self.device}"
        )

        self.image_size = image_size

        self.texture_size = texture_size

        self.renderer = RGBRenderer(
            image_size=image_size,
            device=self.device
        )

    def optimize(
        self,
        refined_mesh_path,
        image_paths,
        visibility_paths,
        texture_path,
        output_path,
        iterations=2000
    ):

        print(
            "\n🚀 Starting photometric optimization"
        )

        # =====================================================
        # LOAD MESH
        # =====================================================

        mesh_data = np.load(
            refined_mesh_path,
            allow_pickle=True
        )

        vertices = torch.tensor(
            mesh_data["vertices"],
            dtype=torch.float32,
            device=self.device
        )

        faces = torch.tensor(
            mesh_data["faces"],
            dtype=torch.long,
            device=self.device
        )

        num_vertices = vertices.shape[1]

        # =====================================================
        # LOAD TEXTURE
        # =====================================================

        texture_img = cv2.imread(
            str(texture_path)
        )

        texture_img = cv2.cvtColor(
            texture_img,
            cv2.COLOR_BGR2RGB
        )

        texture_img = cv2.resize(
            texture_img,
            (
                self.texture_size,
                self.texture_size
            )
        )

        texture_map = torch.tensor(
            texture_img,
            dtype=torch.float32,
            device=self.device
        ) / 255.0

        texture_map = nn.Parameter(
            texture_map.unsqueeze(0)
        )

        # =====================================================
        # FREEFORM REFINEMENT
        # =====================================================

        vertex_offsets = nn.Parameter(
            torch.zeros(
                1,
                num_vertices,
                3,
                device=self.device
            )
        )

        optimizer = torch.optim.Adam(
            [
                vertex_offsets,
                texture_map
            ],
            lr=0.0003
        )

        # =====================================================
        # UV PLACEHOLDER
        # =====================================================

        verts_uvs = torch.rand(
            1,
            num_vertices,
            2,
            device=self.device
        )

        faces_uvs = faces.unsqueeze(0)

        # =====================================================
        # LAPLACIAN
        # =====================================================

        laplacian = LaplacianRegularizer(
            faces,
            num_vertices,
            device=self.device
        )

        # =====================================================
        # CAMERA
        # =====================================================

        cameras = self.renderer.create_camera()

        # =====================================================
        # LOAD TARGET IMAGES
        # =====================================================

        target_rgbs = []

        visibility_masks = []

        metadata_all = []

        for img_path, vis_path in zip(
            image_paths,
            visibility_paths
        ):

            rgba = cv2.imread(
                str(img_path),
                cv2.IMREAD_UNCHANGED
            )

            if rgba.shape[2] == 4:

                rgb = rgba[:, :, :3]

                alpha = rgba[:, :, 3]

            else:

                rgb = rgba

                alpha = np.ones(
                    rgb.shape[:2],
                    dtype=np.uint8
                ) * 255

            rgb = cv2.cvtColor(
                rgb,
                cv2.COLOR_BGR2RGB
            )

            rgb = cv2.resize(
                rgb,
                (
                    self.image_size,
                    self.image_size
                )
            )

            alpha = cv2.resize(
                alpha,
                (
                    self.image_size,
                    self.image_size
                )
            )

            rgb = torch.tensor(
                rgb,
                dtype=torch.float32,
                device=self.device
            ) / 255.0

            mask = torch.tensor(
                (
                    alpha > 10
                ).astype(np.float32),
                dtype=torch.float32,
                device=self.device
            )

            target_rgbs.append(
                rgb.unsqueeze(0)
            )

            visibility_masks.append(
                mask.unsqueeze(0)
            )

            with open(vis_path, "r") as f:

                metadata = json.load(f)

            metadata_all.append(metadata)

        # =====================================================
        # OPTIMIZATION LOOP
        # =====================================================

        progress = tqdm(
            range(iterations),
            desc="Photometric"
        )

        for iteration in progress:

            optimizer.zero_grad()

            refined_vertices = (
                vertices +
                vertex_offsets
            )

            total_loss = 0.0

            for i in range(len(image_paths)):

                rendered_rgb = (
                    self.renderer.render(
                        vertices=refined_vertices,
                        faces=faces,
                        verts_uvs=verts_uvs,
                        faces_uvs=faces_uvs,
                        texture_map=texture_map,
                        cameras=cameras
                    )
                )

                rgb_loss = photometric_loss(
                    rendered_rgb,
                    target_rgbs[i],
                    visibility_masks[i]
                )

                image_weight = metadata_all[i][
                    "image_weight"
                ]

                total_loss += (
                    image_weight *
                    rgb_loss
                )

            # =============================================
            # REGULARIZATION
            # =============================================

            lap_loss = laplacian.loss(
                refined_vertices
            )

            offset_loss = (
                vertex_offsets ** 2
            ).mean()

            tex_smooth = texture_smoothness_loss(
                texture_map
            )

            total = (

                100.0 * total_loss +

                3.0 * lap_loss +

                0.1 * offset_loss +

                0.05 * tex_smooth
            )

            total.backward()

            optimizer.step()

            # =============================================
            # CLAMP
            # =============================================

            with torch.no_grad():

                vertex_offsets.clamp_(
                    -0.05,
                    0.05
                )

                texture_map.clamp_(
                    0.0,
                    1.0
                )

            if iteration % 10 == 0:

                progress.set_postfix({
                    "loss":
                        f"{total.item():.4f}"
                })

        # =====================================================
        # SAVE RESULTS
        # =====================================================

        final_vertices = (
            vertices +
            vertex_offsets
        )

        final_texture = (
            texture_map[0]
            .detach()
            .cpu()
            .numpy()
        )

        final_texture = (
            final_texture * 255
        ).astype(np.uint8)

        final_texture = cv2.cvtColor(
            final_texture,
            cv2.COLOR_RGB2BGR
        )

        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        texture_out = (
            output_path.parent /
            "optimized_texture.png"
        )

        cv2.imwrite(
            str(texture_out),
            final_texture
        )

        np.savez(
            output_path,
            vertices=final_vertices
                .detach()
                .cpu()
                .numpy(),
            faces=faces
                .detach()
                .cpu()
                .numpy()
        )

        print(
            f"\n✅ Photometric optimization complete"
        )

        return {
            "mesh": output_path,
            "texture": texture_out
        }