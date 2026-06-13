import torch


class RegionAwareMasks:
    """
    Conservative region helpers for canonical SMPL-X optimization.

    The masks in this file intentionally do *not* define the anatomical breast
    model.  They are support masks for screen-space silhouette weighting,
    anti-bloat protection, and debug/export vertex masks.  The breast-only
    soft-tissue phase should still use the dedicated topology prior and the
    visibility analyser's `breast_fit_gates` as the authoritative source.

    Key design changes vs the old helper:
      - the generic breast screen mask excludes the central sternum strip;
      - sternum/cleavage pixels keep strong anti-bloat penalties;
      - left/right breast, underbust, sternum, cleavage, and guard maps are
        exposed explicitly for optimizer diagnostics and optional weighting;
      - template vertex masks include diagnostic left/right/sternum/IMF/guard
        groups, while remaining heuristic.
    """

    @staticmethod
    def _rect(x, y, x0, x1, y0, y1):
        return ((x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)).float()

    @staticmethod
    def _ellipse(x, y, cx, cy, rx, ry):
        return ((((x - cx) / max(float(rx), 1e-6)) ** 2 + ((y - cy) / max(float(ry), 1e-6)) ** 2) <= 1.0).float()

    @staticmethod
    def _soften(binary_map, strength=0.15):
        """Return a slightly softened 0/1 map without changing shape/device."""
        # Keep this dependency-free: no convolutions or OpenCV here.  The light
        # blend avoids extreme discontinuities while preserving hard intent.
        return binary_map.clamp(0.0, 1.0) * (1.0 - float(strength)) + binary_map.clamp(0.0, 1.0).detach() * float(strength)

    @staticmethod
    def screen_region_maps(height, width, device="cpu"):
        """
        Returns float region maps with shape [1, 1, H, W].

        Coordinate convention:
            x = 0 left,  x = 1 right
            y = 0 top,   y = 1 bottom

        These are deliberately conservative screen-space priors.  They should
        not override pose-aware visibility analysis or topology priors.
        """
        y = torch.linspace(0.0, 1.0, int(height), device=device).view(1, 1, int(height), 1)
        x = torch.linspace(0.0, 1.0, int(width), device=device).view(1, 1, 1, int(width))
        z = torch.zeros(1, 1, int(height), int(width), device=device)

        chest = RegionAwareMasks._rect(x, y, 0.17, 0.83, 0.21, 0.46)
        upper_chest_guard = RegionAwareMasks._rect(x, y, 0.23, 0.77, 0.17, 0.285)
        sternum = RegionAwareMasks._rect(x, y, 0.455, 0.545, 0.225, 0.465)
        cleavage = RegionAwareMasks._rect(x, y, 0.415, 0.585, 0.245, 0.455)

        left_lobe = RegionAwareMasks._ellipse(x, y, 0.365, 0.345, 0.155, 0.115)
        right_lobe = RegionAwareMasks._ellipse(x, y, 0.635, 0.345, 0.155, 0.115)
        breast_rect = RegionAwareMasks._rect(x, y, 0.215, 0.785, 0.245, 0.445)
        left_breast = (left_lobe * breast_rect * (1.0 - sternum)).clamp(0.0, 1.0)
        right_breast = (right_lobe * breast_rect * (1.0 - sternum)).clamp(0.0, 1.0)
        breast_full = breast_rect
        breast = (left_breast + right_breast).clamp(0.0, 1.0)

        underbust = RegionAwareMasks._rect(x, y, 0.225, 0.775, 0.385, 0.505) * (1.0 - 0.60 * sternum)
        abdomen = RegionAwareMasks._rect(x, y, 0.245, 0.755, 0.435, 0.635)
        waist = RegionAwareMasks._rect(x, y, 0.225, 0.775, 0.395, 0.585)
        hips = RegionAwareMasks._rect(x, y, 0.155, 0.845, 0.535, 0.785)
        glutes = RegionAwareMasks._rect(x, y, 0.18, 0.82, 0.58, 0.825)

        left_imf = left_breast * RegionAwareMasks._rect(x, y, 0.22, 0.50, 0.375, 0.485)
        right_imf = right_breast * RegionAwareMasks._rect(x, y, 0.50, 0.78, 0.375, 0.485)
        imf = (left_imf + right_imf).clamp(0.0, 1.0)

        # breast_fit is the safe local screen prior for breast-specific residuals:
        # lobes + IMF, but not central sternum/cleavage bridge.
        breast_fit = (breast + 0.35 * imf).clamp(0.0, 1.0) * (1.0 - 0.90 * sternum) * (1.0 - 0.60 * cleavage)
        breast_fit = breast_fit.clamp(0.0, 1.0)

        armpit_left = RegionAwareMasks._rect(x, y, 0.14, 0.28, 0.22, 0.43)
        armpit_right = RegionAwareMasks._rect(x, y, 0.72, 0.86, 0.22, 0.43)
        armpit_guard = (armpit_left + armpit_right).clamp(0.0, 1.0)
        abdomen_guard = RegionAwareMasks._rect(x, y, 0.25, 0.75, 0.49, 0.67)

        return {
            # Backwards-compatible names used by the optimizer.
            "chest": chest,
            "breast": breast,
            "waist": waist,
            "hips": hips,
            "glutes": glutes,
            "abdomen": abdomen,
            # New stricter diagnostic/control regions.
            "breast_full": breast_full,
            "breast_fit": breast_fit,
            "left_breast": left_breast,
            "right_breast": right_breast,
            "underbust": underbust.clamp(0.0, 1.0),
            "left_imf": left_imf,
            "right_imf": right_imf,
            "imf": imf,
            "sternum": sternum,
            "cleavage": cleavage,
            "upper_chest_guard": upper_chest_guard,
            "armpit_left": armpit_left,
            "armpit_right": armpit_right,
            "armpit_guard": armpit_guard,
            "abdomen_guard": abdomen_guard,
            "zero": z,
        }

    @staticmethod
    def silhouette_region_weights(height, width, device="cpu"):
        """
        Pixel weights for silhouette mismatch.

        The old helper gave a broad central chest/breast boost.  This version
        keeps modest lobe/underbust attention but avoids encouraging a central
        sternum bridge.  Cleavage/sternum receive neutral-to-low silhouette
        weight because false positives there are handled by anti_bloat_weights.
        """
        maps = RegionAwareMasks.screen_region_maps(height, width, device)
        weights = torch.ones(1, 1, int(height), int(width), device=device)

        weights = weights + 0.22 * maps["chest"]
        weights = weights + 0.30 * maps["breast_fit"]
        weights = weights + 0.16 * maps["underbust"]
        weights = weights + 0.18 * maps["waist"]
        weights = weights + 0.38 * maps["hips"]
        weights = weights + 0.28 * maps["glutes"]

        # Do not reward filling the sternum/cleavage strip as a silhouette fix.
        weights = weights * (1.0 - 0.18 * maps["sternum"])
        weights = weights * (1.0 - 0.10 * maps["cleavage"])
        return weights.clamp(min=0.65, max=2.0)

    @staticmethod
    def anti_bloat_weights(height, width, device="cpu"):
        """
        Pixel weights for false-positive / bloat penalties.

        Breast lobes and glute regions can project outward, but the central
        sternum/cleavage strip, abdomen, and upper chest should not inflate.
        This directly targets the 'tent between the breasts' failure mode.
        """
        maps = RegionAwareMasks.screen_region_maps(height, width, device)
        weights = torch.ones(1, 1, int(height), int(width), device=device)

        # Allow true lobe/glute projection, not central chest bridges.
        weights = weights * (1.0 - 0.38 * maps["breast_fit"])
        weights = weights * (1.0 - 0.24 * maps["glutes"])
        weights = weights * (1.0 - 0.12 * maps["hips"])

        # Strongly penalize central false positive mass and torso inflation.
        weights = weights + 0.95 * maps["sternum"]
        weights = weights + 0.65 * maps["cleavage"]
        weights = weights + 0.45 * maps["upper_chest_guard"]
        weights = weights + 0.45 * maps["abdomen"]
        weights = weights + 0.25 * maps["waist"]
        weights = weights + 0.35 * maps["abdomen_guard"]
        weights = weights + 0.20 * maps["armpit_guard"]

        return weights.clamp(min=0.20, max=2.60)

    @staticmethod
    def template_vertex_masks(vertices):
        """
        Heuristic vertex masks from a neutral SMPL-X template.

        Args:
            vertices: torch.Tensor [V, 3] or [1, V, 3]

        Returns bool masks with shape [V].  These are for diagnostics/export and
        non-authoritative safeguards.  Dedicated breast topology prior remains
        the source of truth for breast soft-tissue fitting.
        """
        if vertices.dim() == 3:
            vertices = vertices[0]

        x = vertices[:, 0]
        y = vertices[:, 1]
        z = vertices[:, 2]

        x_n = (x - x.min()) / (x.max() - x.min()).clamp(min=1e-8)
        y_n = (y - y.min()) / (y.max() - y.min()).clamp(min=1e-8)
        z_n = (z - z.min()) / (z.max() - z.min()).clamp(min=1e-8)

        # y_n: 0 feet, 1 head.  z_n is model-convention dependent, so use it
        # only as a soft/front-ish heuristic and keep masks broad.
        frontish = z_n > 0.45
        backish = z_n < 0.58
        mid_x = (x_n > 0.455) & (x_n < 0.545)

        upper_torso = (y_n > 0.48) & (y_n < 0.73) & (x_n > 0.18) & (x_n < 0.82)
        chest = (y_n > 0.515) & (y_n < 0.735) & (x_n > 0.17) & (x_n < 0.83)
        upper_chest_guard = (y_n > 0.675) & (y_n < 0.785) & (x_n > 0.24) & (x_n < 0.76)
        sternum = (y_n > 0.535) & (y_n < 0.705) & mid_x & frontish
        cleavage = (y_n > 0.535) & (y_n < 0.695) & (x_n > 0.415) & (x_n < 0.585) & frontish

        left_breast = (y_n > 0.545) & (y_n < 0.705) & (x_n > 0.24) & (x_n < 0.485) & frontish
        right_breast = (y_n > 0.545) & (y_n < 0.705) & (x_n > 0.515) & (x_n < 0.76) & frontish
        breast_full = (y_n > 0.54) & (y_n < 0.705) & (x_n > 0.235) & (x_n < 0.765) & frontish
        breast = (left_breast | right_breast) & (~sternum)

        left_imf = (y_n > 0.515) & (y_n < 0.585) & (x_n > 0.25) & (x_n < 0.49) & frontish
        right_imf = (y_n > 0.515) & (y_n < 0.585) & (x_n > 0.51) & (x_n < 0.75) & frontish
        imf = left_imf | right_imf
        underbust = (y_n > 0.49) & (y_n < 0.60) & (x_n > 0.23) & (x_n < 0.77) & frontish

        abdomen = (y_n > 0.36) & (y_n < 0.56) & (x_n > 0.24) & (x_n < 0.76)
        waist = (y_n > 0.33) & (y_n < 0.51) & (x_n > 0.22) & (x_n < 0.78)
        hips = (y_n > 0.22) & (y_n < 0.43) & (x_n > 0.16) & (x_n < 0.84)
        glutes = (y_n > 0.22) & (y_n < 0.43) & (x_n > 0.18) & (x_n < 0.82) & backish

        left_armpit_guard = (y_n > 0.54) & (y_n < 0.72) & (x_n > 0.14) & (x_n < 0.28)
        right_armpit_guard = (y_n > 0.54) & (y_n < 0.72) & (x_n > 0.72) & (x_n < 0.86)
        armpit_guard = left_armpit_guard | right_armpit_guard
        abdomen_guard = (y_n > 0.37) & (y_n < 0.55) & (x_n > 0.27) & (x_n < 0.73)
        breast_fit = breast & (~sternum) & (~armpit_guard) & (~upper_chest_guard) & (~abdomen_guard)

        return {
            # Backwards-compatible masks.
            "upper_torso": upper_torso,
            "breast": breast,
            "chest": chest,
            "abdomen": abdomen,
            "waist": waist,
            "hips": hips,
            "glutes": glutes,
            # New diagnostic masks.
            "breast_full": breast_full,
            "breast_fit": breast_fit,
            "left_breast": left_breast,
            "right_breast": right_breast,
            "underbust": underbust,
            "left_imf": left_imf,
            "right_imf": right_imf,
            "imf": imf,
            "sternum": sternum,
            "cleavage": cleavage,
            "upper_chest_guard": upper_chest_guard,
            "left_armpit_guard": left_armpit_guard,
            "right_armpit_guard": right_armpit_guard,
            "armpit_guard": armpit_guard,
            "abdomen_guard": abdomen_guard,
        }

    @staticmethod
    def masks_to_numpy(vertex_masks):
        return {name: mask.detach().cpu().numpy().astype(bool) for name, mask in vertex_masks.items()}
