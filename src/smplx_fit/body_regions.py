import torch


class BodyRegionWeights:

    @staticmethod
    def create_weight_map(
        height,
        width,
        device="cpu"
    ):

        weights = torch.ones(
            1,
            height,
            width,
            device=device
        )

        # =================================================
        # NORMALIZED GRID
        # =================================================

        y = torch.linspace(
            0.0,
            1.0,
            height,
            device=device
        ).view(height, 1)

        x = torch.linspace(
            0.0,
            1.0,
            width,
            device=device
        ).view(1, width)

        # =================================================
        # CHEST REGION
        # =================================================

        chest_mask = (
            (y > 0.22) &
            (y < 0.42) &
            (x > 0.22) &
            (x < 0.78)
        )

        # =================================================
        # WAIST REGION
        # =================================================

        waist_mask = (
            (y > 0.40) &
            (y < 0.58) &
            (x > 0.25) &
            (x < 0.75)
        )

        # =================================================
        # HIP / GLUTE REGION
        # =================================================

        hip_mask = (
            (y > 0.55) &
            (y < 0.82) &
            (x > 0.18) &
            (x < 0.82)
        )

        # =================================================
        # SHOULDERS
        # =================================================

        shoulder_mask = (
            (y > 0.15) &
            (y < 0.30) &
            (x > 0.10) &
            (x < 0.90)
        )
        # =================================================
        # APPLY REGION WEIGHTS
        # =================================================

        weights[0][chest_mask] *= 3.0
        weights[0][waist_mask] *= 2.0
        weights[0][hip_mask] *= 3.5
        weights[0][shoulder_mask] *= 1.8

        # =================================================
        # EDGE FALLBACK
        # =================================================

        edge_margin = int(width * 0.05)

        weights[:, :, :edge_margin] *= 0.5
        weights[:, :, -edge_margin:] *= 0.5

        return weights