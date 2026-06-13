import torch


class BodyRegionWeights:
    """
    Coarse screen-space body-region weights for the multi-image optimizer.

    This class is intentionally conservative. It should not define precise
    breast/chest anatomy; the visibility analyzer + SMPL-X topology prior are
    responsible for breast-only fitting. These maps are only coarse silhouette
    / residual weights used by the main body fit.

    Output compatibility:
        create_weight_map(...) returns a tensor with shape [1, H, W], matching
        the previous implementation.
    """

    @staticmethod
    def _normalized_grid(height, width, device="cpu", dtype=torch.float32):
        height = int(height)
        width = int(width)
        if height <= 0 or width <= 0:
            raise ValueError(f"height and width must be positive, got {height}x{width}")

        y = torch.linspace(0.0, 1.0, height, device=device, dtype=dtype).view(height, 1)
        x = torch.linspace(0.0, 1.0, width, device=device, dtype=dtype).view(1, width)
        return x, y

    @staticmethod
    def _rect_mask(x, y, *, x1, x2, y1, y2):
        return (x >= float(x1)) & (x <= float(x2)) & (y >= float(y1)) & (y <= float(y2))

    @staticmethod
    def _soft_vertical_band(x, *, center=0.5, half_width=0.055):
        """Soft central strip used to protect sternum/cleavage from broad chest boosting."""
        hw = max(float(half_width), 1e-6)
        return torch.exp(-0.5 * ((x - float(center)) / hw) ** 2).clamp(0.0, 1.0)

    @staticmethod
    def region_maps(height, width, device="cpu", dtype=torch.float32):
        """
        Return coarse [H, W] boolean/soft maps.

        These maps are image-space heuristics and are deliberately broad. They
        are safe as weighting priors, but they should not be used as anatomical
        labels for breast-only deformation.
        """
        x, y = BodyRegionWeights._normalized_grid(height, width, device=device, dtype=dtype)

        # Broad torso bands. Values are tuned to be less aggressive than the old
        # implementation, especially around the sternum/cleavage strip.
        shoulder = BodyRegionWeights._rect_mask(x, y, x1=0.10, x2=0.90, y1=0.15, y2=0.31)
        chest_full = BodyRegionWeights._rect_mask(x, y, x1=0.20, x2=0.80, y1=0.22, y2=0.43)
        waist = BodyRegionWeights._rect_mask(x, y, x1=0.24, x2=0.76, y1=0.40, y2=0.59)
        abdomen = BodyRegionWeights._rect_mask(x, y, x1=0.27, x2=0.73, y1=0.36, y2=0.63)
        hip = BodyRegionWeights._rect_mask(x, y, x1=0.17, x2=0.83, y1=0.54, y2=0.80)
        glute = BodyRegionWeights._rect_mask(x, y, x1=0.19, x2=0.81, y1=0.58, y2=0.83)

        # Breast lobes exclude a central strip. This prevents the main body fit
        # from over-rewarding a single connected chest mass.
        left_breast = BodyRegionWeights._rect_mask(x, y, x1=0.22, x2=0.47, y1=0.25, y2=0.43)
        right_breast = BodyRegionWeights._rect_mask(x, y, x1=0.53, x2=0.78, y1=0.25, y2=0.43)
        breast_lobes = left_breast | right_breast

        sternum_strip_soft = BodyRegionWeights._soft_vertical_band(x, center=0.5, half_width=0.050)
        sternum = (
            BodyRegionWeights._rect_mask(x, y, x1=0.45, x2=0.55, y1=0.22, y2=0.49)
        ).float() * sternum_strip_soft
        cleavage = (
            BodyRegionWeights._rect_mask(x, y, x1=0.43, x2=0.57, y1=0.27, y2=0.45)
        ).float() * sternum_strip_soft

        # Underbust / IMF-ish support. Still broad; exact IMF comes from visibility JSON.
        underbust = BodyRegionWeights._rect_mask(x, y, x1=0.22, x2=0.78, y1=0.39, y2=0.50)
        left_underbust = BodyRegionWeights._rect_mask(x, y, x1=0.22, x2=0.48, y1=0.39, y2=0.50)
        right_underbust = BodyRegionWeights._rect_mask(x, y, x1=0.52, x2=0.78, y1=0.39, y2=0.50)

        return {
            "shoulder": shoulder.float(),
            "chest_full": chest_full.float(),
            "chest": (chest_full.float() * (1.0 - 0.70 * cleavage)).clamp(0.0, 1.0),
            "left_breast": left_breast.float(),
            "right_breast": right_breast.float(),
            "breast": breast_lobes.float(),
            "breast_lobes": breast_lobes.float(),
            "sternum": sternum.float(),
            "cleavage": cleavage.float(),
            "underbust": underbust.float(),
            "left_underbust": left_underbust.float(),
            "right_underbust": right_underbust.float(),
            "waist": waist.float(),
            "abdomen": abdomen.float(),
            "hip": hip.float(),
            "hips": hip.float(),
            "glute": glute.float(),
            "glutes": glute.float(),
        }

    @staticmethod
    def create_weight_map(height, width, device="cpu"):
        """
        Create a conservative coarse body-region residual weight map.

        Returns:
            torch.Tensor [1, H, W]

        Design notes:
        - Previous chest/hip multipliers were very aggressive (3.0/3.5).
        - The new chest boost is modest and excludes the central sternum strip.
        - The central cleavage/sternum strip is deliberately de-emphasized for
          broad silhouette fitting so the main optimizer does not create a
          connected 'tent' before the breast-only phase.
        - Hips/glutes remain weighted enough for lower-body shape recovery, but
          not so strongly that they dominate camera/pose fitting.
        """
        maps = BodyRegionWeights.region_maps(height, width, device=device)
        weights = torch.ones(1, int(height), int(width), device=device, dtype=torch.float32)

        # Conservative positive weighting.
        weights = weights + 0.35 * maps["shoulder"].unsqueeze(0)
        weights = weights + 0.32 * maps["chest"].unsqueeze(0)
        weights = weights + 0.22 * maps["breast"].unsqueeze(0)
        weights = weights + 0.38 * maps["waist"].unsqueeze(0)
        weights = weights + 0.30 * maps["abdomen"].unsqueeze(0)
        weights = weights + 0.65 * maps["hip"].unsqueeze(0)
        weights = weights + 0.45 * maps["glute"].unsqueeze(0)
        weights = weights + 0.20 * maps["underbust"].unsqueeze(0)

        # Protect the central sternum/cleavage band from broad chest boosting.
        # This does not forbid breast-only valley fitting; it only prevents the
        # main generic silhouette fit from rewarding inter-breast mass.
        sternum_suppression = (0.35 * maps["sternum"] + 0.45 * maps["cleavage"]).clamp(0.0, 0.60)
        weights = weights * (1.0 - sternum_suppression.unsqueeze(0))

        # Edge fallback: reduce reliance on near-border pixels/crops.
        edge_margin = max(1, int(width * 0.05))
        weights[:, :, :edge_margin] *= 0.55
        weights[:, :, -edge_margin:] *= 0.55

        return weights.clamp(min=0.35, max=2.25)

    @staticmethod
    def create_anti_bloat_map(height, width, device="cpu"):
        """
        Optional false-positive / bloat penalty map with sternum protection.

        Existing code may not call this method, but it is useful for newer
        optimizers. Shape: [1, H, W].
        """
        maps = BodyRegionWeights.region_maps(height, width, device=device)
        weights = torch.ones(1, int(height), int(width), device=device, dtype=torch.float32)

        # Allow true breast/glute projection a little, but keep central sternum
        # and abdomen/waist strongly controlled.
        weights = weights * (1.0 - 0.18 * maps["breast"].unsqueeze(0))
        weights = weights * (1.0 - 0.12 * maps["glute"].unsqueeze(0))
        weights = weights + 0.42 * maps["abdomen"].unsqueeze(0)
        weights = weights + 0.30 * maps["waist"].unsqueeze(0)
        weights = weights + 0.70 * maps["sternum"].unsqueeze(0)
        weights = weights + 0.90 * maps["cleavage"].unsqueeze(0)

        edge_margin = max(1, int(width * 0.05))
        weights[:, :, :edge_margin] *= 0.75
        weights[:, :, -edge_margin:] *= 0.75
        return weights.clamp(min=0.45, max=2.50)
