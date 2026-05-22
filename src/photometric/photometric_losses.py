# ============================================================
# FILE:
# src/photometric/photometric_losses.py
# ============================================================

import torch
import torch.nn.functional as F


def photometric_loss(
    rendered_rgb,
    target_rgb,
    visibility_mask
):

    diff = torch.abs(
        rendered_rgb - target_rgb
    )

    diff = diff.mean(dim=-1)

    weighted = diff * visibility_mask

    return weighted.mean()


def texture_smoothness_loss(
    texture_map
):

    loss_x = (
        texture_map[:, :, 1:] -
        texture_map[:, :, :-1]
    ).abs().mean()

    loss_y = (
        texture_map[:, 1:, :] -
        texture_map[:, :-1, :]
    ).abs().mean()

    return loss_x + loss_y