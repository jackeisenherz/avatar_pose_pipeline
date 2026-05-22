import json
from pathlib import Path

import cv2
import numpy as np

import torch
import torch.nn as nn

import smplx

from tqdm import tqdm

from .renderer import SilhouetteRenderer

from .utils import (
    perspective_projection,
    load_pose_json,
    load_visibility_json,
    create_camera
)

from .joint_mapper import (
    SMPLXJointMapper
)

from .body_regions import (
    BodyRegionWeights
)

from .pseudo_landmarks import (
    PseudoLandmarkExtractor
)

from .body_measurements import (
    BodyMeasurements
)

from .losses import (
    keypoint_loss,
    silhouette_loss,
    shape_prior_loss,
    pose_prior_loss,
    translation_loss,
    pseudo_landmark_loss
)


class MultiImageOptimizer:

    # TODO: once confirmed working, increase image_size back to 1024
    def __init__(
        self,
        model_path,
        gender="female",
        image_size=512
    ):

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(f"✓ Optimizer device: {self.device}")

        self.image_size = image_size

        # =================================================
        # SMPL-X
        # =================================================

        self.model = smplx.create(
            model_path=model_path,
            model_type="smplx",
            gender=gender,
            use_pca=False,
            num_betas=10,
            ext="npz"
        ).to(self.device)

        # =================================================
        # RENDERER
        # =================================================

        self.renderer = SilhouetteRenderer(
            image_size=image_size,
            device=self.device
        )

    # =====================================================
    # MAIN OPTIMIZATION
    # =====================================================
    # TODO: once confirmed working, increase again to 1000 iterations
    def optimize(
        self,
        image_paths,
        pose_json_paths,
        visibility_json_paths,
        output_path,
        iterations=1000
    ):

        print("\n🚀 Starting multi-image optimization")

        num_images = len(image_paths)

        print(f"✓ Images: {num_images}")

        # =================================================
        # SHARED SHAPE
        # =================================================

        betas = nn.Parameter(
            torch.zeros(
                1,
                10,
                device=self.device
            )
        )

        # =================================================
        # PER IMAGE PARAMETERS
        # =================================================

        body_poses = nn.ParameterList()

        global_orients = nn.ParameterList()

        translations = nn.ParameterList()

        gt_keypoints_all = []

        confidence_all = []

        masks_all = []

        metadata_all = []

        pseudo_landmarks_all = []

        # =================================================
        # LOAD DATA
        # =================================================

        for i in range(num_images):

            # ---------------------------------------------
            # POSE
            # ---------------------------------------------

            pose_data = load_pose_json(
                pose_json_paths[i]
            )

            visibility_data = load_visibility_json(
                visibility_json_paths[i]
            )

            metadata_all.append(
                visibility_data
            )

            # ---------------------------------------------
            # KEYPOINTS
            # ---------------------------------------------

            keypoints = torch.tensor(
                pose_data["keypoints"][:, :2],
                dtype=torch.float32,
                device=self.device
            ).unsqueeze(0)

            confidence = torch.tensor(
                pose_data["keypoints"][:, 2],
                dtype=torch.float32,
                device=self.device
            ).unsqueeze(0)

            gt_keypoints_all.append(
                keypoints
            )

            confidence_all.append(
                confidence
            )

            # ---------------------------------------------
            # LOAD SILHOUETTE
            # ---------------------------------------------

            rgba = cv2.imread(
                str(image_paths[i]),
                cv2.IMREAD_UNCHANGED
            )

            if rgba.shape[2] == 4:

                alpha = rgba[:, :, 3]

                mask = (
                    alpha > 10
                ).astype(np.float32)

            else:

                gray = cv2.cvtColor(
                    rgba,
                    cv2.COLOR_BGR2GRAY
                )

                mask = (
                    gray > 5
                ).astype(np.float32)

            mask = cv2.resize(
                mask,
                (
                    self.image_size,
                    self.image_size
                )
            )

            # ---------------------------------------------
            # PSEUDO LANDMARKS
            # ---------------------------------------------

            pseudo_landmarks = (
                PseudoLandmarkExtractor.extract(
                    mask
                )
            )

            pseudo_landmarks_all.append(
                pseudo_landmarks
            )

            # ---------------------------------------------
            # TORCH MASK
            # ---------------------------------------------

            mask_tensor = torch.tensor(
                mask,
                dtype=torch.float32,
                device=self.device
            )

            # [H,W]
            # -> [1,H,W]
            # -> [1,1,H,W]

            mask_tensor = (
                mask_tensor
                .unsqueeze(0)
                .unsqueeze(0)
            )

            masks_all.append(
                mask_tensor
            )

            # ---------------------------------------------
            # INIT POSE
            # ---------------------------------------------

            body_poses.append(
                nn.Parameter(
                    torch.zeros(
                        1,
                        63,
                        device=self.device
                    )
                )
            )

            global_orients.append(
                nn.Parameter(
                    torch.zeros(
                        1,
                        3,
                        device=self.device
                    )
                )
            )

            translations.append(
                nn.Parameter(
                    torch.tensor(
                        [[0.0, 0.0, 5.0]],
                        device=self.device
                    )
                )
            )

        # =================================================
        # OPTIMIZER
        # =================================================

        params = (
            [betas] +
            list(body_poses.parameters()) +
            list(global_orients.parameters()) +
            list(translations.parameters())
        )

        optimizer = torch.optim.Adam(
            params,
            lr=0.01
        )

        # =================================================
        # CAMERA
        # =================================================

        focal_length = torch.tensor(
            [[1500.0, 1500.0]],
            device=self.device
        )

        camera_center = torch.tensor(
            [[
                self.image_size / 2,
                self.image_size / 2
            ]],
            device=self.device
        )

        # =================================================
        # OPTIMIZATION LOOP
        # =================================================

        progress = tqdm(
            range(iterations),
            desc="Optimizing"
        )

        for iteration in progress:

            optimizer.zero_grad()

            total_loss = 0.0

            # =============================================
            # LOOP OVER IMAGES
            # =============================================

            for i in range(num_images):

                # -----------------------------------------
                # MODEL FORWARD
                # -----------------------------------------

                output = self.model(
                    betas=betas,
                    body_pose=body_poses[i],
                    global_orient=global_orients[i],
                    transl=translations[i],
                    return_verts=True
                )

                vertices = output.vertices

                # -----------------------------------------
                # JOINT MAPPING
                # -----------------------------------------

                joints = (
                    SMPLXJointMapper
                    .smplx_to_coco17(
                        output.joints
                    )
                )

                # -----------------------------------------
                # PROJECT JOINTS
                # -----------------------------------------

                projected_joints = (
                    perspective_projection(
                        joints,
                        translation=translations[i],
                        focal_length=focal_length,
                        camera_center=camera_center
                    )
                )

                # -----------------------------------------
                # RENDER SILHOUETTE
                # -----------------------------------------

                rendered = self.renderer.render(
                    vertices=vertices,
                    faces=self.model.faces_tensor.unsqueeze(0),
                    focal_length=focal_length,
                    principal_point=camera_center,
                    translation=translations[i]
                )

                rendered_mask = rendered[..., 3].unsqueeze(1)

                # -----------------------------------------
                # VISIBILITY MASK
                # -----------------------------------------

                metadata = metadata_all[i]

                visibility_mask = torch.ones_like(
                    masks_all[i]
                )

                if metadata["truncated_top"]:

                    visibility_mask[
                        :, :120, :
                    ] *= 0.5

                if metadata["truncated_bottom"]:

                    visibility_mask[
                        :, -120:, :
                    ] *= 0.5

                if metadata["truncated_left"]:

                    visibility_mask[
                        :, :, :120
                    ] *= 0.5

                if metadata["truncated_right"]:

                    visibility_mask[
                        :, :, -120:
                    ] *= 0.5

                # -----------------------------------------
                # REGION WEIGHTS
                # -----------------------------------------

                region_weights = (
                    BodyRegionWeights
                    .create_weight_map(
                        height=self.image_size,
                        width=self.image_size,
                        device=self.device
                    )
                )

                chest_boost = (
                    1.0 +
                    metadata["chest_visible"]
                )

                hip_boost = (
                    1.0 +
                    metadata["hip_visible"] * 0.5
                )

                region_weights *= (
                    chest_boost *
                    hip_boost
                )

                # -----------------------------------------
                # BODY MEASUREMENTS
                # -----------------------------------------

                chest_width = (
                    BodyMeasurements.width(
                        vertices,
                        BodyMeasurements.CHEST_VERTICES
                    )
                )

                waist_width = (
                    BodyMeasurements.width(
                        vertices,
                        BodyMeasurements.WAIST_VERTICES
                    )
                )

                hip_width = (
                    BodyMeasurements.width(
                        vertices,
                        BodyMeasurements.HIP_VERTICES
                    )
                )

                pseudo_data = (
                    pseudo_landmarks_all[i]
                )

                loss_pseudo = 0.0

                # =====================================================
                # CHEST
                # =====================================================

                if pseudo_data["chest"] is not None:

                    target_width = (
                        pseudo_data["chest"]["width"]
                        / self.image_size
                    )

                    loss_pseudo += pseudo_landmark_loss(
                        chest_width,
                        target_width,
                        weight=30.0
                    )

                # =====================================================
                # WAIST
                # =====================================================

                if pseudo_data["waist"] is not None:

                    target_width = (
                        pseudo_data["waist"]["width"]
                        / self.image_size
                    )

                    loss_pseudo += pseudo_landmark_loss(
                        waist_width,
                        target_width,
                        weight=20.0
                    )

                # =====================================================
                # HIPS / GLUTES
                # =====================================================

                if pseudo_data["hips"] is not None:

                    target_width = (
                        pseudo_data["hips"]["width"]
                        / self.image_size
                    )

                    loss_pseudo += pseudo_landmark_loss(
                        hip_width,
                        target_width,
                        weight=35.0
                    )

                # -----------------------------------------
                # LOSSES
                # -----------------------------------------

                loss_kp = keypoint_loss(
                    projected_joints,
                    gt_keypoints_all[i],
                    confidence_all[i]
                )

                loss_sil = silhouette_loss(
                    rendered_mask,
                    masks_all[i],
                    visibility_mask,
                    region_weights
                )

                loss_shape = shape_prior_loss(
                    betas
                )

                loss_pose = pose_prior_loss(
                    body_poses[i]
                )

                loss_trans = translation_loss(
                    translations[i]
                )

                # -----------------------------------------
                # IMAGE WEIGHT
                # -----------------------------------------

                image_weight = metadata[
                    "image_weight"
                ]

                # -----------------------------------------
                # FINAL IMAGE LOSS
                # -----------------------------------------
                image_loss = image_weight * (
                    0.00005 * loss_kp +
                    40.0 * loss_sil +
                    3.0 * loss_pseudo +
                    0.05 * loss_shape +
                    0.01 * loss_pose +
                    0.01 * loss_trans
                )
                total_loss += image_loss

            # =============================================
            # BACKPROP
            # =============================================

            total_loss.backward()

            optimizer.step()

            # =============================================
            # LOGGING
            # =============================================

            if iteration % 10 == 0:
                progress.set_postfix({
                    "total": f"{total_loss.item():.2f}",
                    "kp": f"{loss_kp.item():.2f}",
                    "sil": f"{loss_sil.item():.4f}",
                    "pseudo": f"{float(loss_pseudo):.4f}",
                })

        # =================================================
        # FINAL FORWARD
        # =================================================

        output = self.model(

            betas=betas,

            body_pose=torch.zeros(
                1,
                63,
                device=self.device
            ),

            global_orient=torch.zeros(
                1,
                3,
                device=self.device
            ),

            transl=torch.zeros(
                1,
                3,
                device=self.device
            ),

            return_verts=True
        )

        # =================================================
        # SAVE
        # =================================================

        result = {

            "betas":
                betas.detach()
                .cpu()
                .numpy(),

            "vertices":
                output.vertices.detach()
                .cpu()
                .numpy(),

            "joints":
                output.joints.detach()
                .cpu()
                .numpy(),

            "faces":
                self.model.faces,

            "num_images":
                num_images
        }

        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        np.savez(
            output_path,
            **result
        )

        print(
            f"\n✅ Saved canonical body:"
            f"\n{output_path}"
        )

        return result