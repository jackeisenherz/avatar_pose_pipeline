import torch
import torch.nn.functional as F


# =========================================================
# KEYPOINT LOSS
# =========================================================

def keypoint_loss(
    pred_keypoints,
    gt_keypoints,
    confidence,
    joint_weights=None
):

    diff = (
        pred_keypoints - gt_keypoints
    ) ** 2

    diff = diff.sum(dim=-1)

    weighted = diff * confidence

    if joint_weights is not None:

        weighted = weighted * joint_weights

    return weighted.mean()


# =========================================================
# SILHOUETTE LOSS
# =========================================================

def silhouette_loss(
    pred_mask,
    target_mask,
    visibility_mask,
    region_weights
):

    diff = (
        pred_mask - target_mask
    ) ** 2

    weighted = (
        diff *
        visibility_mask *
        region_weights
    )

    return weighted.mean()


# =========================================================
# PSEUDO LANDMARK LOSS
# =========================================================

def pseudo_landmark_loss(
    pred_value,
    target_value,
    weight=1.0
):

    if pred_value is None:
        return 0.0

    if target_value is None:
        return 0.0

    loss = (
        pred_value - target_value
    ) ** 2

    return loss.mean() * weight


# =========================================================
# SHAPE PRIOR
# =========================================================

def shape_prior_loss(
    betas
):

    return (betas ** 2).mean()


# =========================================================
# POSE PRIOR
# =========================================================

def pose_prior_loss(
    body_pose
):

    return (body_pose ** 2).mean()


# =========================================================
# TRANSLATION PRIOR
# =========================================================

def translation_loss(
    transl
):

    return (transl ** 2).mean()


# =========================================================
# SMOOTHNESS
# =========================================================

def smoothness_loss(
    vertices
):

    return (
        (
            vertices[:, 1:] -
            vertices[:, :-1]
        ) ** 2
    ).mean()


# =========================================================
# LAPLACIAN REGULARIZATION
# =========================================================

def laplacian_loss(
    offsets,
    neighbors
):

    total = 0.0

    count = 0

    for vid, nbs in neighbors.items():

        if len(nbs) == 0:
            continue

        v = offsets[:, vid]

        nb = offsets[:, nbs].mean(dim=1)

        total += (
            (v - nb) ** 2
        ).mean()

        count += 1

    if count == 0:
        return 0.0

    return total / count