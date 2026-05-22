from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from smplx_fit.renderer import SilhouetteRenderer
from smplx_fit.losses import silhouette_loss
from smplx_fit.body_regions import BodyRegionWeights

from .laplacian import LaplacianRegularizer


class MeshRefiner:

    def __init__(
        self,
        image_size=1024
    ):

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(f"✓ MeshRefiner device: {self.device}")

        self.image_size = image_size

        self.renderer = SilhouetteRenderer(
            image_size=image_size,
            device=self.device
        )

    def refine(
        self,
        canonical_body_path,
        image_paths,
        visibility_paths,
        output_path,
        iterations=1500
    ):

        print("\n🚀 Starting freeform refinement")

        data = np.load(
            canonical_body_path,
            allow_pickle=True
        )

        base_vertices = torch.tensor(
            data["vertices"],
            dtype=torch.float32,
            device=self.device
        )

        faces = torch.tensor(
            data["faces"],
            dtype=torch.long,
            device=self.device
        )

        num_vertices = base_vertices.shape[1]

        # =====================================================
        # LEARNABLE OFFSETS
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
            [vertex_offsets],
            lr=0.0005
        )

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
        # LOAD MASKS
        # =====================================================

        masks = []
        metadata_all = []

        for img_path, vis_path in zip(
            image_paths,
            visibility_paths
        ):

            rgba = cv2.imread(
                str(img_path),
                cv2.IMREAD_UNCHANGED
            )

            alpha = rgba[:, :, 3]

            mask = (
                alpha > 10
            ).astype(np.float32)

            mask = cv2.resize(
                mask,
                (
                    self.image_size,
                    self.image_size
                )
            )

            mask = torch.tensor(
                mask,
                dtype=torch.float32,
                device=self.device
            ).unsqueeze(0)

            masks.append(mask)

            import json

            with open(vis_path, "r") as f:

                metadata = json.load(f)

            metadata_all.append(metadata)

        # =====================================================
        # OPTIMIZATION LOOP
        # =====================================================

        progress = tqdm(
            range(iterations),
            desc="Refining"
        )

        for iteration in progress:

            optimizer.zero_grad()

            refined_vertices = (
                base_vertices +
                vertex_offsets
            )

            total_loss = 0.0

            # =============================================
            # MULTI-IMAGE SILHOUETTE FIT
            # =============================================

            for i in range(len(image_paths)):

                rendered_mask = (
                    self.renderer.render(
                        vertices=refined_vertices,
                        faces=faces,
                        cameras=cameras
                    )
                )

                metadata = metadata_all[i]

                visibility_mask = torch.ones_like(
                    masks[i]
                )

                # =========================================
                # TRUNCATION HANDLING
                # =========================================

                if metadata["truncated_top"]:

                    visibility_mask[
                        :, :120, :
                    ] *= 0.5

                if metadata["truncated_bottom"]:

                    visibility_mask[
                        :, -120:, :
                    ] *= 0.5

                # =========================================
                # REGION WEIGHTS
                # =========================================

                region_weights = (
                    BodyRegionWeights
                    .create_weight_map(
                        height=self.image_size,
                        width=self.image_size,
                        device=self.device
                    )
                )

                chest_boost = (
                    1.0 +
                    metadata["chest_visible"]
                )

                hip_boost = (
                    1.0 +
                    metadata["hip_visible"]
                )

                region_weights *= (
                    chest_boost *
                    hip_boost
                )

                sil_loss = silhouette_loss(
                    rendered_mask,
                    masks[i],
                    visibility_mask,
                    region_weights
                )

                image_weight = metadata[
                    "image_weight"
                ]

                total_loss += (
                    image_weight *
                    sil_loss
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

            # =============================================
            # FINAL LOSS
            # =============================================

            total = (

                50.0 * total_loss +

                5.0 * lap_loss +

                0.1 * offset_loss
            )

            total.backward()

            optimizer.step()

            # =============================================
            # LIMIT OFFSET MAGNITUDE
            # =============================================

            with torch.no_grad():

                vertex_offsets.clamp_(
                    -0.08,
                    0.08
                )

            if iteration % 10 == 0:

                progress.set_postfix({
                    "loss":
                        f"{total.item():.4f}"
                })

        # =====================================================
        # FINAL MESH
        # =====================================================

        final_vertices = (
            base_vertices +
            vertex_offsets
        )

        result = {

            "vertices":
                final_vertices
                .detach()
                .cpu()
                .numpy(),

            "faces":
                faces
                .detach()
                .cpu()
                .numpy(),

            "offsets":
                vertex_offsets
                .detach()
                .cpu()
                .numpy()
        }

        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        np.savez(
            output_path,
            **result
        )

        print(
            f"\n✅ Refined mesh saved:"
            f"\n{output_path}"
        )

        return result