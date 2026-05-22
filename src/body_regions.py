import torch


class BodyRegionWeights:

    @staticmethod
    def create_weight_map(height, width, device="cuda"):

        weights = torch.ones(
            (1, height, width),
            device=device
        )

        y = torch.linspace(
            0,
            1,
            height,
            device=device
        ).view(height, 1)

        # =====================================================
        # CHEST REGION
        # =====================================================

        chest_mask = (
            (y > 0.22) &
            (y < 0.48)
        )

        weights[:, chest_mask.squeeze(), :] *= 3.0

        # =====================================================
        # WAIST
        # =====================================================

        waist_mask = (
            (y > 0.48) &
            (y < 0.60)
        )

        weights[:, waist_mask.squeeze(), :] *= 1.5

        # =====================================================
        # GLUTE / HIP REGION
        # =====================================================

        glute_mask = (
            (y > 0.60) &
            (y < 0.78)
        )

        weights[:, glute_mask.squeeze(), :] *= 2.5

        # =====================================================
        # LEGS
        # =====================================================

        leg_mask = (
            (y > 0.78)
        )

        weights[:, leg_mask.squeeze(), :] *= 1.2

        return weights