from pathlib import Path

import cv2
import json
import torch
import numpy as np
import torch.nn.functional as F
import smplx

from torch.optim import Adam

from .utils import (
    perspective_projection,
    batch_rodrigues,
)

from .losses import (
    keypoint_loss,
    shape_prior_loss,
    pose_prior_loss,
    translation_loss,
    silhouette_loss,
)

from .renderer import SilhouetteRenderer


class SMPLXFitter:

    def __init__(
        self,
        model_path,
        gender="female",
        device=None
    ):
        self.device = (
            device
            if device is not None
            else (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        )

        print(f"✓ SMPL-X device: {self.device}")

        self.model = smplx.create(
            model_path=model_path,
            model_type="smplx",
            gender=gender,
            use_face_contour=True,
            use_pca=False,
            num_betas=10,
            ext="npz"
        ).to(self.device)

        self.faces = torch.tensor(
            self.model.faces.astype(np.int64),
            dtype=torch.long,
            device=self.device
        ).unsqueeze(0)

        self.renderer = SilhouetteRenderer(
            image_size=1024,
            device=self.device
        )

    def fit_image(
        self,
        image_path,
        pose_json,
        output_dir,
        iterations=300
    ):
        image_path = Path(image_path)
        pose_json = Path(pose_json)
        output_dir = Path(output_dir)

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        # =====================================================
        # LOAD IMAGE
        # =====================================================

        rgba = cv2.imread(
            str(image_path),
            cv2.IMREAD_UNCHANGED
        )

        if rgba is None:
            raise RuntimeError(
                f"Could not load image: {image_path}"
            )

        if rgba.shape[2] < 4:
            raise RuntimeError(
                "Expected RGBA image with alpha channel"
            )

        h, w = rgba.shape[:2]

        # =====================================================
        # TARGET SILHOUETTE
        # =====================================================

        alpha = rgba[:, :, 3]

        target_mask = (
            torch.tensor(
                alpha,
                dtype=torch.float32,
                device=self.device
            ) / 255.0
        )

        target_mask = target_mask.unsqueeze(0).unsqueeze(0)

        # =====================================================
        # LOAD POSE KEYPOINTS
        # =====================================================

        with open(pose_json, "r") as f:
            pose_data = json.load(f)

        keypoints = pose_data["keypoints"]

        keypoints_2d = []

        for kp in keypoints:
            keypoints_2d.append([
                kp["x"],
                kp["y"],
                kp["confidence"]
            ])

        keypoints_2d = torch.tensor(
            keypoints_2d,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)

        gt_points = keypoints_2d[:, :, :2]

        confidence = keypoints_2d[:, :, 2]

        # =====================================================
        # OPTIMIZATION VARIABLES
        # =====================================================

        body_pose = torch.zeros(
            (1, 63),
            dtype=torch.float32,
            device=self.device,
            requires_grad=True
        )

        betas = torch.zeros(
            (1, 10),
            dtype=torch.float32,
            device=self.device,
            requires_grad=True
        )

        global_orient = torch.zeros(
            (1, 3),
            dtype=torch.float32,
            device=self.device,
            requires_grad=True
        )

        transl = torch.tensor(
            [[0.0, 0.0, 5.0]],
            dtype=torch.float32,
            device=self.device,
            requires_grad=True
        )

        # =====================================================
        # OPTIMIZER
        # =====================================================

        optimizer = Adam(
            [
                body_pose,
                betas,
                global_orient,
                transl
            ],
            lr=0.01
        )

        # =====================================================
        # CAMERA
        # =====================================================

        focal_length = torch.tensor(
            [[5000.0, 5000.0]],
            dtype=torch.float32,
            device=self.device
        )

        camera_center = torch.tensor(
            [[w / 2.0, h / 2.0]],
            dtype=torch.float32,
            device=self.device
        )

        # =====================================================
        # OPTIMIZATION LOOP
        # =====================================================

        print(
            f"🚀 Fitting SMPL-X: {image_path.name}"
        )

        for i in range(iterations):

            optimizer.zero_grad()

            model_output = self.model(
                betas=betas,
                body_pose=body_pose,
                global_orient=global_orient,
                transl=None,
                return_verts=True
            )

            vertices = model_output.vertices

            joints_3d = model_output.joints
            # COCO-17 mapping from SMPL-X joints
            coco_indices = torch.tensor([
                55,  # nose
                12,  # left eye
                17,  # right eye
                19,  # left ear
                21,  # right ear
                16,  # left shoulder
                17,  # right shoulder
                18,  # left elbow
                19,  # right elbow
                20,  # left wrist
                21,  # right wrist
                1,   # left hip
                2,   # right hip
                4,   # left knee
                5,   # right knee
                7,   # left ankle
                8,   # right ankle
            ], device=self.device)

            joints_3d = joints_3d[:, coco_indices]
            
            rotation = batch_rodrigues(
                torch.zeros(
                    (1, 3),
                    device=self.device
                )
            )

            projected_joints = perspective_projection(
                joints_3d,
                rotation=rotation,
                translation=torch.zeros(
                    (1, 3),
                    device=self.device
                ),
                focal_length=focal_length,
                camera_center=camera_center
            )

            # =====================================================
            # SILHOUETTE RENDERING
            # =====================================================

            rendered = self.renderer.render(
                vertices=vertices,
                faces=self.faces,
                focal_length=focal_length,
                principal_point=camera_center,
                translation=transl
            )

            predicted_mask = rendered[..., 3]

            predicted_mask = predicted_mask.unsqueeze(1)

            target_resized = F.interpolate(
                target_mask,
                size=predicted_mask.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

            # =====================================================
            # LOSSES
            # =====================================================

            loss_kp = keypoint_loss(
                projected_joints,
                gt_points,
                confidence
            )

            loss_sil = silhouette_loss(
                predicted_mask,
                target_resized
            )

            loss_shape = shape_prior_loss(
                betas
            )

            loss_pose = pose_prior_loss(
                body_pose
            )

            loss_trans = translation_loss(
                transl
            )

            total_loss = (
                10.0 * loss_kp
                + 25.0 * loss_sil
                + 0.001 * loss_shape
                + 0.001 * loss_pose
                + 0.001 * loss_trans
            )

            total_loss.backward()

            optimizer.step()

            # =====================================================
            # LOGGING
            # =====================================================

            if i % 20 == 0 or i == iterations - 1:

                print(
                    f"[{i:03d}/{iterations}] "
                    f"total={total_loss.item():.4f} "
                    f"kp={loss_kp.item():.4f} "
                    f"sil={loss_sil.item():.4f}"
                )

        # =====================================================
        # FINAL FORWARD PASS
        # =====================================================

        with torch.no_grad():

            final_output = self.model(
                betas=betas,
                body_pose=body_pose,
                global_orient=global_orient,
                transl=None,
                return_verts=True
            )

        result = {
            "vertices": (
                final_output.vertices
                .detach()
                .cpu()
                .numpy()
            ),

            "joints": (
                final_output.joints
                .detach()
                .cpu()
                .numpy()
            ),

            "betas": (
                betas.detach()
                .cpu()
                .numpy()
            ),

            "body_pose": (
                body_pose.detach()
                .cpu()
                .numpy()
            ),

            "global_orient": (
                global_orient.detach()
                .cpu()
                .numpy()
            ),

            "translation": (
                transl.detach()
                .cpu()
                .numpy()
            ),
        }

        output_file = (
            output_dir /
            f"{image_path.stem}_smplx.npz"
        )

        np.savez_compressed(
            output_file,
            **result
        )

        print(
            f"✓ Saved SMPL-X fit: {output_file}"
        )

        return output_file