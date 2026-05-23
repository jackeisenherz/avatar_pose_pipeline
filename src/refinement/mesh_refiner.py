from pathlib import Path
import json

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
    # debug:      256px, 30–50 iterations
    # validation: 512px, 100–300 iterations
    # final:      1024px, 500–1500 iterations
    # via CLI --refine-iterations 30
    def __init__(self, image_size=256):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.image_size = image_size

        print(f"✓ MeshRefiner device: {self.device}")

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

        data = np.load(canonical_body_path, allow_pickle=True)

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

        if faces.dim() == 2:
            faces_render = faces.unsqueeze(0)
        else:
            faces_render = faces

        num_vertices = base_vertices.shape[1]

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

        laplacian = LaplacianRegularizer(
            faces if faces.dim() == 2 else faces[0],
            num_vertices,
            device=self.device
        )

        focal_length = torch.tensor(
            [[1500.0, 1500.0]],
            device=self.device
        )

        camera_center = torch.tensor(
            [[self.image_size / 2.0, self.image_size / 2.0]],
            device=self.device
        )

        translation = torch.tensor(
            [[0.0, 0.0, 5.0]],
            device=self.device
        )

        masks = []
        metadata_all = []

        for img_path, vis_path in zip(image_paths, visibility_paths):
            rgba = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)

            if rgba is None:
                raise RuntimeError(f"Could not load image: {img_path}")

            if rgba.shape[2] == 4:
                alpha = rgba[:, :, 3]
            else:
                gray = cv2.cvtColor(rgba, cv2.COLOR_BGR2GRAY)
                alpha = (gray > 5).astype(np.uint8) * 255

            mask = (alpha > 10).astype(np.float32)

            mask = cv2.resize(
                mask,
                (self.image_size, self.image_size),
                interpolation=cv2.INTER_NEAREST
            )

            mask = torch.tensor(
                mask,
                dtype=torch.float32,
                device=self.device
            )

            mask = mask.unsqueeze(0).unsqueeze(0)

            masks.append(mask)

            with open(vis_path, "r") as f:
                metadata_all.append(json.load(f))

        progress = tqdm(
            range(iterations),
            desc="Refining"
        )

        for iteration in progress:
            optimizer.zero_grad()

            refined_vertices = base_vertices + vertex_offsets
            total_sil_loss = 0.0

            for i in range(len(image_paths)):
                rendered = self.renderer.render(
                    vertices=refined_vertices,
                    faces=faces_render,
                    focal_length=focal_length,
                    principal_point=camera_center,
                    translation=translation
                )

                rendered_mask = rendered[..., 3].unsqueeze(1)

                metadata = metadata_all[i]

                visibility_mask = torch.ones_like(masks[i])

                border = int(self.image_size * 0.12)

                if metadata.get("truncated_top", False):
                    visibility_mask[:, :, :border, :] *= 0.5

                if metadata.get("truncated_bottom", False):
                    visibility_mask[:, :, -border:, :] *= 0.5

                if metadata.get("truncated_left", False):
                    visibility_mask[:, :, :, :border] *= 0.5

                if metadata.get("truncated_right", False):
                    visibility_mask[:, :, :, -border:] *= 0.5

                region_weights = BodyRegionWeights.create_weight_map(
                    height=self.image_size,
                    width=self.image_size,
                    device=self.device
                )

                chest_boost = 1.0 + float(metadata.get("chest_visible", 0.0))
                hip_boost = 1.0 + float(metadata.get("hip_visible", 0.0))

                region_weights = region_weights.unsqueeze(0)
                region_weights *= chest_boost * hip_boost

                sil_loss = silhouette_loss(
                    rendered_mask,
                    masks[i],
                    visibility_mask,
                    region_weights
                )

                image_weight = float(metadata.get("image_weight", 1.0))
                total_sil_loss += image_weight * sil_loss

            lap_loss = laplacian.loss(refined_vertices)
            offset_loss = (vertex_offsets ** 2).mean()

            total = (
                50.0 * total_sil_loss +
                5.0 * lap_loss +
                0.1 * offset_loss
            )

            total.backward()
            optimizer.step()

            with torch.no_grad():
                vertex_offsets.clamp_(-0.08, 0.08)

            if iteration % 10 == 0:
                progress.set_postfix({
                    "loss": f"{total.item():.4f}",
                    "sil": f"{float(total_sil_loss):.4f}",
                    "lap": f"{float(lap_loss):.4f}",
                    "off": f"{float(offset_loss):.6f}",
                })

        final_vertices = base_vertices + vertex_offsets

        result = {
            "vertices": final_vertices.detach().cpu().numpy(),
            "faces": faces.detach().cpu().numpy(),
            "offsets": vertex_offsets.detach().cpu().numpy(),
        }

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        np.savez(output_path, **result)

        print(f"\n✅ Refined mesh saved:\n{output_path}")

        return result