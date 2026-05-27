import torch


class RegionAwareMasks:
    """
    Region helpers for canonical SMPL-X optimization.

    There are two kinds of masks here:

    1. Screen-space masks:
       Used directly in silhouette losses. These are reliable because the
       target and rendered masks are both 2D images.

    2. Template vertex masks:
       Saved for later refinement/analysis. They are heuristic and based on
       normalized SMPL-X template coordinates. They are useful for region-aware
       refinement, but canonical fitting still mainly uses screen-space losses.
    """

    @staticmethod
    def screen_region_maps(height, width, device="cpu"):
        """
        Returns region maps with shape [1, 1, H, W].

        Coordinate convention:
            y = 0 top
            y = 1 bottom
        """

        y = torch.linspace(
            0.0,
            1.0,
            height,
            device=device,
        ).view(1, 1, height, 1)

        x = torch.linspace(
            0.0,
            1.0,
            width,
            device=device,
        ).view(1, 1, 1, width)

        # Broad image-space anatomical bands.
        # These are intentionally conservative because pose/camera changes the
        # exact screen location of body parts.
        chest = (
            (y >= 0.22) &
            (y <= 0.43) &
            (x >= 0.18) &
            (x <= 0.82)
        ).float()

        breast = (
            (y >= 0.25) &
            (y <= 0.42) &
            (x >= 0.22) &
            (x <= 0.78)
        ).float()

        waist = (
            (y >= 0.40) &
            (y <= 0.58) &
            (x >= 0.22) &
            (x <= 0.78)
        ).float()

        hips = (
            (y >= 0.54) &
            (y <= 0.78) &
            (x >= 0.16) &
            (x <= 0.84)
        ).float()

        glutes = (
            (y >= 0.58) &
            (y <= 0.82) &
            (x >= 0.18) &
            (x <= 0.82)
        ).float()

        abdomen = (
            (y >= 0.38) &
            (y <= 0.62) &
            (x >= 0.25) &
            (x <= 0.75)
        ).float()

        return {
            "chest": chest,
            "breast": breast,
            "waist": waist,
            "hips": hips,
            "glutes": glutes,
            "abdomen": abdomen,
        }

    @staticmethod
    def silhouette_region_weights(height, width, device="cpu"):
        """
        Pixel weights for silhouette mismatch.

        Breast/chest and hip/glute regions receive moderately higher weight.
        This encourages canonical SMPL-X betas to use these regions during
        fitting, without letting them dominate the whole loss.
        """

        maps = RegionAwareMasks.screen_region_maps(
            height,
            width,
            device,
        )

        weights = torch.ones(
            1,
            1,
            height,
            width,
            device=device,
        )

        weights = weights + 0.50 * maps["chest"]
        weights = weights + 0.35 * maps["breast"]
        weights = weights + 0.25 * maps["waist"]
        weights = weights + 0.45 * maps["hips"]
        weights = weights + 0.35 * maps["glutes"]

        return weights

    @staticmethod
    def anti_bloat_weights(height, width, device="cpu"):
        """
        Pixel weights for false-positive / bloat penalties.

        Important:
        Breast and glute regions receive lower anti-bloat weight because their
        true identity-specific shape may legitimately project outward. Abdomen
        and generic torso remain more strongly controlled.
        """

        maps = RegionAwareMasks.screen_region_maps(
            height,
            width,
            device,
        )

        weights = torch.ones(
            1,
            1,
            height,
            width,
            device=device,
        )

        # Allow breast/glute projection more than generic torso expansion.
        weights = weights * (1.0 - 0.55 * maps["breast"])
        weights = weights * (1.0 - 0.30 * maps["chest"])
        weights = weights * (1.0 - 0.35 * maps["glutes"])
        weights = weights * (1.0 - 0.20 * maps["hips"])

        # Keep abdomen/waist from inflating.
        weights = weights + 0.35 * maps["abdomen"]
        weights = weights + 0.20 * maps["waist"]

        return weights.clamp(
            min=0.15,
            max=2.0,
        )

    @staticmethod
    def template_vertex_masks(vertices):
        """
        Heuristic vertex masks from a neutral SMPL-X template.

        vertices:
            torch.Tensor [V, 3] or [1, V, 3]

        Returns bool masks with shape [V].

        These masks are deliberately approximate and are mostly intended to be
        saved for downstream region-aware refinement. They do not need exact
        anatomical labels to be useful for weighting.
        """

        if vertices.dim() == 3:
            vertices = vertices[0]

        x = vertices[:, 0]
        y = vertices[:, 1]
        z = vertices[:, 2]

        x_n = (x - x.min()) / (x.max() - x.min()).clamp(min=1e-8)
        y_n = (y - y.min()) / (y.max() - y.min()).clamp(min=1e-8)
        z_n = (z - z.min()) / (z.max() - z.min()).clamp(min=1e-8)

        # y_n: 0 feet, 1 head
        # z_n: direction depends on model convention; use broad front-ish band.
        upper_torso = (
            (y_n > 0.48) &
            (y_n < 0.72) &
            (x_n > 0.22) &
            (x_n < 0.78)
        )

        breast = (
            (y_n > 0.54) &
            (y_n < 0.70) &
            (x_n > 0.24) &
            (x_n < 0.76) &
            (z_n > 0.45)
        )

        chest = (
            (y_n > 0.52) &
            (y_n < 0.73) &
            (x_n > 0.18) &
            (x_n < 0.82)
        )

        abdomen = (
            (y_n > 0.38) &
            (y_n < 0.56) &
            (x_n > 0.24) &
            (x_n < 0.76)
        )

        waist = (
            (y_n > 0.34) &
            (y_n < 0.50) &
            (x_n > 0.22) &
            (x_n < 0.78)
        )

        hips = (
            (y_n > 0.23) &
            (y_n < 0.42) &
            (x_n > 0.16) &
            (x_n < 0.84)
        )

        glutes = (
            (y_n > 0.22) &
            (y_n < 0.43) &
            (x_n > 0.18) &
            (x_n < 0.82) &
            (z_n < 0.55)
        )

        return {
            "upper_torso": upper_torso,
            "breast": breast,
            "chest": chest,
            "abdomen": abdomen,
            "waist": waist,
            "hips": hips,
            "glutes": glutes,
        }

    @staticmethod
    def masks_to_numpy(vertex_masks):
        return {
            name: mask.detach().cpu().numpy().astype(bool)
            for name, mask in vertex_masks.items()
        }
