import json

import numpy as np
import torch

from pytorch3d.renderer import (
    PerspectiveCameras
)


# =========================================================
# CAMERA
# =========================================================

def create_camera(
    focal_length,
    camera_center,
    image_size,
    device="cpu"
):

    cameras = PerspectiveCameras(

        focal_length=focal_length,

        principal_point=camera_center,

        image_size=torch.tensor(
            [[image_size, image_size]],
            device=device
        ),

        in_ndc=False,

        device=device
    )

    return cameras


# =========================================================
# PERSPECTIVE PROJECTION
# =========================================================

def perspective_projection(
    points,
    translation,
    focal_length,
    camera_center,
    rotation=None
):
    """
    points:
        [B, N, 3]
    """

    batch_size = points.shape[0]

    if rotation is None:

        rotation = torch.eye(
            3,
            device=points.device
        ).unsqueeze(0).expand(
            batch_size,
            -1,
            -1
        )

    # =====================================================
    # ROTATE
    # =====================================================

    rotated = torch.einsum(
        "bij,bkj->bki",
        rotation,
        points
    )

    # =====================================================
    # TRANSLATE
    # =====================================================

    translated = (
        rotated +
        translation.unsqueeze(1)
    )

    # =====================================================
    # PERSPECTIVE DIVISION
    # =====================================================

    projected = (
        translated[:, :, :2]
        /
        translated[:, :, 2:].clamp(min=1e-6)
    )

    # =====================================================
    # APPLY FOCAL
    # =====================================================

    projected_x = (
        projected[:, :, 0]
        *
        focal_length[:, 0].unsqueeze(1)
    )

    projected_y = (
        projected[:, :, 1]
        *
        focal_length[:, 1].unsqueeze(1)
    )

    projected = torch.stack(
        [projected_x, projected_y],
        dim=-1
    )

    # =====================================================
    # APPLY CAMERA CENTER
    # =====================================================

    projected[:, :, 0] += (
        camera_center[:, 0]
        .unsqueeze(1)
    )

    projected[:, :, 1] += (
        camera_center[:, 1]
        .unsqueeze(1)
    )

    return projected


# =========================================================
# RODRIGUES
# =========================================================

def batch_rodrigues(theta):

    l1norm = torch.norm(
        theta + 1e-8,
        p=2,
        dim=1
    )

    angle = l1norm.unsqueeze(-1)

    normalized = theta / angle

    angle = angle * 0.5

    v_cos = torch.cos(angle)

    v_sin = torch.sin(angle)

    quat = torch.cat(
        [v_cos, v_sin * normalized],
        dim=1
    )

    return quat_to_rotmat(quat)


# =========================================================
# QUATERNION -> ROTMAT
# =========================================================

def quat_to_rotmat(quat):

    norm_quat = quat / quat.norm(
        p=2,
        dim=1,
        keepdim=True
    )

    w, x, y, z = norm_quat.unbind(dim=1)

    B = quat.size(0)

    rotMat = torch.stack([

        1 - 2*y*y - 2*z*z,
        2*x*y - 2*w*z,
        2*w*y + 2*x*z,

        2*w*z + 2*x*y,
        1 - 2*x*x - 2*z*z,
        2*y*z - 2*w*x,

        2*x*z - 2*w*y,
        2*w*x + 2*y*z,
        1 - 2*x*x - 2*y*y

    ], dim=1).view(B, 3, 3)

    return rotMat


# =========================================================
# POSE JSON LOADER
# =========================================================

def load_pose_json(json_path):

    with open(json_path, "r") as f:

        data = json.load(f)

    keypoints = np.array(
        [
            [
                kp["x"],
                kp["y"],
                kp["confidence"]
            ]
            for kp in data["keypoints"]
        ],
        dtype=np.float32
    )

    return {
        "keypoints": keypoints
    }


# =========================================================
# VISIBILITY JSON LOADER
# =========================================================

def load_visibility_json(json_path):

    with open(json_path, "r") as f:

        data = json.load(f)

    return data