# ============================================================
# FILE:
# src/photometric/texture_projection.py
# ============================================================

from pathlib import Path

import cv2
import numpy as np
import torch


class TextureProjector:

    def __init__(
        self,
        texture_size=2048
    ):

        self.texture_size = texture_size

    def create_initial_texture(
        self,
        image_paths,
        output_path
    ):

        texture = np.zeros(
            (
                self.texture_size,
                self.texture_size,
                3
            ),
            dtype=np.float32
        )

        weights = np.zeros(
            (
                self.texture_size,
                self.texture_size,
                1
            ),
            dtype=np.float32
        )

        # =====================================================
        # SIMPLE INITIAL VERSION
        # =====================================================

        for img_path in image_paths:

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

            rgb = cv2.resize(
                rgb,
                (
                    self.texture_size,
                    self.texture_size
                )
            )

            alpha = cv2.resize(
                alpha,
                (
                    self.texture_size,
                    self.texture_size
                )
            )

            mask = (
                alpha > 10
            ).astype(np.float32)[..., None]

            texture += rgb * mask

            weights += mask

        texture /= np.maximum(weights, 1e-6)

        texture = texture.astype(np.uint8)

        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        cv2.imwrite(
            str(output_path),
            texture
        )

        print(
            f"✓ Initial texture saved:"
            f"\n{output_path}"
        )

        return texture