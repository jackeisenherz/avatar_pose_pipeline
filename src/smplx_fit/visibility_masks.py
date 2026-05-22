import cv2
import numpy as np


def create_visibility_mask(
    rgba,
    metadata
):
    """
    Creates visibility-aware optimization mask.

    Returns:
        float32 mask in [0,1]
    """

    h, w = rgba.shape[:2]

    mask = np.ones(
        (h, w),
        dtype=np.float32
    )

    crop_type = metadata["crop_type"]

    bbox = metadata["bbox"]

    y1 = bbox["y1"]
    y2 = bbox["y2"]

    # =====================================================
    # CROPPED BOTTOM
    # =====================================================

    if metadata["truncated_bottom"]:

        fade_start = int(y2 * 0.95)

        for y in range(fade_start, h):

            alpha = 1.0 - (
                (y - fade_start)
                / max(1, h - fade_start)
            )

            mask[y, :] *= alpha

    # =====================================================
    # CROPPED TOP
    # =====================================================

    if metadata["truncated_top"]:

        fade_end = int(y1 * 1.05)

        for y in range(0, fade_end):

            alpha = y / max(1, fade_end)

            mask[y, :] *= alpha

    # =====================================================
    # IMAGE TYPE PRIORS
    # =====================================================

    if crop_type == "torso":

        # Lower body uncertain
        lower_start = int(h * 0.65)

        mask[lower_start:, :] *= 0.2

    elif crop_type == "american":

        lower_start = int(h * 0.80)

        mask[lower_start:, :] *= 0.5

    elif crop_type == "closeup":

        lower_start = int(h * 0.40)

        mask[lower_start:, :] *= 0.05

    # =====================================================
    # BLUR FOR SMOOTH GRADIENTS
    # =====================================================

    mask = cv2.GaussianBlur(
        mask,
        (31, 31),
        0
    )

    mask = np.clip(mask, 0.0, 1.0)

    return mask