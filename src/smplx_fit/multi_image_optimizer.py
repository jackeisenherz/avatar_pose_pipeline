import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import smplx
from tqdm import tqdm

from pytorch3d.renderer import PerspectiveCameras

from .renderer import SilhouetteRenderer
from .utils import load_pose_json, load_visibility_json
from .joint_mapper import SMPLXJointMapper
from .body_regions import BodyRegionWeights
from .region_masks import RegionAwareMasks
from .losses import (
    silhouette_loss,
    shape_prior_loss,
    pose_prior_loss,
    translation_loss,
)


class MultiImageOptimizer:
    """
    Region-aware multi-image canonical SMPL-X optimizer.

    Goals:
    - Improve canonical SMPL-X shape before freeform refinement.
    - Preserve identity-specific breast/chest and hip/glute variation.
    - Avoid generic smoothing/anti-bloat suppressing breast volume.
    - Avoid solving silhouettes by globally inflating the body.

    Major features:
    - keypoint coordinate scaling
    - PyTorch3D projection/render camera consistency
    - staged camera/pose/shape optimization
    - screen-space chest/waist/hip/breast/glute width and area constraints
    - region-aware anti-bloat weighting
    - normalized image weights
    - saved vertex region masks for later refinement
    """

    COCO_LIMBS = [
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        (5, 11), (6, 12), (11, 12),
        (11, 13), (13, 15), (12, 14), (14, 16),
    ]

    BODY_JOINTS = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

    BAND_SPECS = {
        "chest": (0.22, 0.43),
        "breast": (0.25, 0.42),
        "waist": (0.40, 0.58),
        "hips": (0.54, 0.78),
        "glutes": (0.58, 0.82),
    }

    def __init__(
        self,
        model_path,
        gender="female",
        image_size=512,
        pseudo_weight=0.0,
        optimize_focal=True,
        base_focal=1500.0,
        debug=True,
        debug_every=25,
        min_keypoint_conf=0.20,
        width_weight=1.25,
        area_weight=0.65,
        anti_bloat_weight=0.35,
        breast_preserve_weight=1.0,
        glute_preserve_weight=0.7,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.image_size = int(image_size)
        self.pseudo_weight = float(pseudo_weight)
        self.optimize_focal = bool(optimize_focal)
        self.base_focal = float(base_focal)
        self.debug = bool(debug)
        self.debug_every = int(debug_every)
        self.min_keypoint_conf = float(min_keypoint_conf)

        self.width_weight = float(width_weight)
        self.area_weight = float(area_weight)
        self.anti_bloat_weight = float(anti_bloat_weight)
        self.breast_preserve_weight = float(breast_preserve_weight)
        self.glute_preserve_weight = float(glute_preserve_weight)

        print(f"✓ Optimizer device: {self.device}")
        print(f"✓ Image size: {self.image_size}")
        print(f"✓ Pseudo weight arg accepted for main.py compatibility: {self.pseudo_weight}")
        print(f"✓ Width weight: {self.width_weight}")
        print(f"✓ Area weight: {self.area_weight}")
        print(f"✓ Anti-bloat weight: {self.anti_bloat_weight}")
        print(f"✓ Breast preserve weight: {self.breast_preserve_weight}")
        print(f"✓ Glute preserve weight: {self.glute_preserve_weight}")
        print(f"✓ Optimize focal: {self.optimize_focal}")
        print(f"✓ Render debug: {self.debug}")

        self.model = smplx.create(
            model_path=model_path,
            model_type="smplx",
            gender=gender,
            use_pca=False,
            num_betas=10,
            ext="npz",
        ).to(self.device)

        self.renderer = SilhouetteRenderer(
            image_size=self.image_size,
            device=self.device,
        )

        self.region_maps = RegionAwareMasks.screen_region_maps(
            self.image_size,
            self.image_size,
            self.device,
        )

        self.sil_region_weights = RegionAwareMasks.silhouette_region_weights(
            self.image_size,
            self.image_size,
            self.device,
        )

        self.anti_bloat_map = RegionAwareMasks.anti_bloat_weights(
            self.image_size,
            self.image_size,
            self.device,
        )

        self.joint_weights = torch.tensor(
            [
                0.20, 0.10, 0.10, 0.10, 0.10,
                2.50, 2.50,
                3.00, 3.00,
                3.50, 3.50,
                2.50, 2.50,
                2.50, 2.50,
                2.00, 2.00,
            ],
            dtype=torch.float32,
            device=self.device,
        ).view(1, 17)

    # =========================================================
    # CAMERA / PROJECTION
    # =========================================================

    def _make_camera(self, focal_length, camera_center, translation):
        batch_size = translation.shape[0]
        R = torch.eye(3, device=self.device).unsqueeze(0).repeat(batch_size, 1, 1)

        image_size = torch.tensor(
            [[self.image_size, self.image_size]],
            dtype=torch.float32,
            device=self.device,
        ).repeat(batch_size, 1)

        if focal_length.shape[0] == 1 and batch_size > 1:
            focal_length = focal_length.repeat(batch_size, 1)

        if camera_center.shape[0] == 1 and batch_size > 1:
            camera_center = camera_center.repeat(batch_size, 1)

        return PerspectiveCameras(
            focal_length=focal_length,
            principal_point=camera_center,
            R=R,
            T=translation,
            image_size=image_size,
            in_ndc=False,
            device=self.device,
        )

    def _project_points_screen(self, points, focal_length, camera_center, translation):
        cameras = self._make_camera(
            focal_length=focal_length,
            camera_center=camera_center,
            translation=translation,
        )

        screen = cameras.transform_points_screen(
            points,
            image_size=((self.image_size, self.image_size),),
            with_xyflip=True,
        )

        return screen[:, :, :2]

    def _current_focal(self, log_focal_scale):
        if log_focal_scale is None:
            focal_value = torch.tensor(self.base_focal, device=self.device)
        else:
            focal_value = self.base_focal * torch.exp(log_focal_scale[0])

        focal_value = torch.clamp(focal_value, 800.0, 4000.0)
        return torch.stack([focal_value, focal_value]).view(1, 2)

    # =========================================================
    # LOADERS
    # =========================================================

    def _load_image_shape(self, image_path):
        img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

        if img is None:
            raise RuntimeError(f"Could not load image: {image_path}")

        h, w = img.shape[:2]

        return h, w

    def _load_scaled_pose(self, pose_json_path, image_path):
        pose_data = load_pose_json(pose_json_path)
        keypoints = pose_data["keypoints"].astype(np.float32)

        img_h, img_w = self._load_image_shape(image_path)

        keypoints[:, 0] *= self.image_size / float(img_w)
        keypoints[:, 1] *= self.image_size / float(img_h)

        return keypoints

    def _load_target_mask_np(self, image_path):
        rgba = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

        if rgba is None:
            raise RuntimeError(f"Could not load image: {image_path}")

        if rgba.ndim == 3 and rgba.shape[2] == 4:
            alpha = rgba[:, :, 3]
        else:
            gray = cv2.cvtColor(rgba, cv2.COLOR_BGR2GRAY)
            alpha = (gray > 5).astype(np.uint8) * 255

        mask = (alpha > 10).astype(np.float32)

        mask = cv2.resize(
            mask,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_NEAREST,
        )

        return mask

    def _load_target_mask(self, image_path):
        mask = self._load_target_mask_np(image_path)

        return torch.tensor(
            mask,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0).unsqueeze(0)

    def _bbox_from_mask(self, mask_np):
        ys, xs = np.where(mask_np > 0.5)

        if len(xs) == 0:
            return None

        return {
            "cx": float(xs.mean()),
            "cy": float(ys.mean()),
            "w": int(xs.max() - xs.min() + 1),
            "h": int(ys.max() - ys.min() + 1),
            "area": float(mask_np.mean()),
        }

    def _initial_translation_from_mask_and_keypoints(self, mask_np, keypoints):
        bbox = self._bbox_from_mask(mask_np)

        if bbox is None:
            return torch.tensor([[0.0, 0.0, 5.0]], dtype=torch.float32, device=self.device)

        reliable = keypoints[:, 2] > self.min_keypoint_conf
        reliable_body = reliable[self.BODY_JOINTS]

        if reliable_body.sum() >= 4:
            body_kps = keypoints[self.BODY_JOINTS][reliable_body]
            cx = float(body_kps[:, 0].mean())
            cy = float(body_kps[:, 1].mean())
        else:
            cx = float(bbox["cx"])
            cy = float(bbox["cy"])

        h_px = max(float(bbox["h"]), 32.0)

        body_height_m = 1.65
        z = self.base_focal * body_height_m / h_px
        z = float(np.clip(z, 2.5, 8.0))

        tx = (cx - self.image_size / 2.0) * z / self.base_focal
        ty = (cy - self.image_size / 2.0) * z / self.base_focal

        return torch.tensor([[tx, ty, z]], dtype=torch.float32, device=self.device)

    def _distance_maps_from_mask(self, mask_tensor):
        mask_np = mask_tensor[0, 0].detach().cpu().numpy().astype(np.uint8)
        fg = (mask_np > 0).astype(np.uint8)

        dist_out = cv2.distanceTransform(1 - fg, cv2.DIST_L2, 5)
        dist_in = cv2.distanceTransform(fg, cv2.DIST_L2, 5)

        dist_out = dist_out / max(float(dist_out.max()), 1e-6)
        dist_in = dist_in / max(float(dist_in.max()), 1e-6)

        dist_out_t = torch.tensor(
            dist_out,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0).unsqueeze(0)

        dist_in_t = torch.tensor(
            dist_in,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0).unsqueeze(0)

        return dist_out_t, dist_in_t

    # =========================================================
    # SCREEN-SPACE REGION METRICS
    # =========================================================

    def _mask_band_widths(self, mask_tensor):
        widths = {}

        for name, (y0, y1) in self.BAND_SPECS.items():
            a = int(round(y0 * self.image_size))
            b = int(round(y1 * self.image_size))
            a = max(0, min(self.image_size - 1, a))
            b = max(a + 1, min(self.image_size, b))

            band = mask_tensor[:, :, a:b, :]
            row_mass = band.mean(dim=2).sum(dim=-1) / float(self.image_size)

            widths[name] = row_mass.mean()

        return widths

    def _region_areas(self, mask_tensor):
        areas = {}

        for name, region in self.region_maps.items():
            denom = region.sum().clamp(min=1.0)
            areas[name] = (mask_tensor * region).sum() / denom

        return areas

    def _width_loss(self, rendered_mask, target_widths, metadata):
        rendered_widths = self._mask_band_widths(rendered_mask)

        loss = torch.tensor(0.0, device=self.device)
        active = 0.0

        chest_visible = float(metadata.get("chest_visible", 0.0))
        hip_visible = float(metadata.get("hip_visible", 0.0))
        crop_type = metadata.get("crop_type", "")

        if chest_visible > 0.40:
            loss = loss + 1.00 * (rendered_widths["chest"] - target_widths["chest"]).pow(2)
            loss = loss + 1.25 * (rendered_widths["breast"] - target_widths["breast"]).pow(2)
            active += 2.25

        if crop_type in ["full_body", "american"]:
            loss = loss + 0.65 * (rendered_widths["waist"] - target_widths["waist"]).pow(2)
            active += 0.65

        if hip_visible > 0.40:
            loss = loss + 0.90 * (rendered_widths["hips"] - target_widths["hips"]).pow(2)
            loss = loss + 0.65 * (rendered_widths["glutes"] - target_widths["glutes"]).pow(2)
            active += 1.55

        if active <= 0.0:
            return torch.tensor(0.0, device=self.device)

        return loss / active

    def _regional_area_loss(self, rendered_mask, target_areas, metadata):
        rendered_areas = self._region_areas(rendered_mask)

        loss = torch.tensor(0.0, device=self.device)
        active = 0.0

        chest_visible = float(metadata.get("chest_visible", 0.0))
        hip_visible = float(metadata.get("hip_visible", 0.0))

        if chest_visible > 0.40:
            # Breast/chest area is allowed to be identity-specific.
            loss = loss + self.breast_preserve_weight * (
                rendered_areas["breast"] - target_areas["breast"]
            ).pow(2)

            loss = loss + 0.60 * (
                rendered_areas["chest"] - target_areas["chest"]
            ).pow(2)

            active += self.breast_preserve_weight + 0.60

        if hip_visible > 0.40:
            loss = loss + self.glute_preserve_weight * (
                rendered_areas["glutes"] - target_areas["glutes"]
            ).pow(2)

            loss = loss + 0.50 * (
                rendered_areas["hips"] - target_areas["hips"]
            ).pow(2)

            active += self.glute_preserve_weight + 0.50

        if active <= 0.0:
            return torch.tensor(0.0, device=self.device)

        return loss / active

    def _anti_bloat_loss(self, rendered_mask, target_mask):
        false_positive = rendered_mask * (1.0 - target_mask)

        # Region-aware: breast and glute expansion is penalized less than
        # abdomen/waist/global expansion.
        return (false_positive * self.anti_bloat_map).mean()

    # =========================================================
    # DEBUG
    # =========================================================

    def _save_mask_debug(
        self,
        rendered_mask,
        target_mask,
        projected_joints,
        target_joints,
        confidence,
        out_path,
    ):
        pred = rendered_mask[0, 0].detach().cpu().numpy()
        tgt = target_mask[0, 0].detach().cpu().numpy()

        pred = (pred > 0.5).astype(np.uint8) * 255
        tgt = (tgt > 0.5).astype(np.uint8) * 255

        overlay = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
        overlay[:, :, 1] = tgt
        overlay[:, :, 2] = pred

        pj = projected_joints[0].detach().cpu().numpy()
        gj = target_joints[0].detach().cpu().numpy()
        cf = confidence[0].detach().cpu().numpy()

        for idx in range(min(len(pj), len(gj))):
            gx, gy = int(round(gj[idx, 0])), int(round(gj[idx, 1]))
            px, py = int(round(pj[idx, 0])), int(round(pj[idx, 1]))

            if 0 <= gx < self.image_size and 0 <= gy < self.image_size:
                if cf[idx] >= self.min_keypoint_conf:
                    cv2.circle(overlay, (gx, gy), 4, (255, 255, 0), -1)
                else:
                    cv2.circle(overlay, (gx, gy), 2, (128, 128, 0), -1)

            if 0 <= px < self.image_size and 0 <= py < self.image_size:
                cv2.circle(overlay, (px, py), 4, (0, 0, 255), -1)

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(out_path), overlay)

    # =========================================================
    # KEYPOINT LOSSES
    # =========================================================

    def _has_enough_keypoints(self, confidence):
        body_conf = confidence[:, self.BODY_JOINTS]
        count = (body_conf > self.min_keypoint_conf).sum()

        return int(count.detach().cpu()) >= 4

    def _weighted_keypoint_loss(self, pred, target, confidence):
        if not self._has_enough_keypoints(confidence):
            return torch.tensor(0.0, device=self.device)

        confidence = torch.where(
            confidence > self.min_keypoint_conf,
            confidence,
            torch.zeros_like(confidence),
        )

        weights = confidence * self.joint_weights

        diff = ((pred - target) / float(self.image_size)) ** 2
        diff = diff.sum(dim=-1)

        denom = weights.sum().clamp(min=1.0)

        return (diff * weights).sum() / denom

    def _bone_direction_loss(self, pred, target, confidence):
        total = torch.tensor(0.0, device=self.device)
        count = torch.tensor(0.0, device=self.device)

        for a, b in self.COCO_LIMBS:
            conf = torch.minimum(confidence[:, a], confidence[:, b])

            if float(conf.max().detach().cpu()) < self.min_keypoint_conf:
                continue

            pred_vec = pred[:, b] - pred[:, a]
            tgt_vec = target[:, b] - target[:, a]

            pred_vec = pred_vec / pred_vec.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            tgt_vec = tgt_vec / tgt_vec.norm(dim=-1, keepdim=True).clamp(min=1e-6)

            limb_loss = ((pred_vec - tgt_vec) ** 2).sum(dim=-1)

            total = total + (limb_loss * conf).mean()
            count = count + 1.0

        return total / count.clamp(min=1.0)

    def _keypoint_center_scale_loss(self, pred, target, confidence):
        if not self._has_enough_keypoints(confidence):
            return torch.tensor(0.0, device=self.device)

        confidence = torch.where(
            confidence > self.min_keypoint_conf,
            confidence,
            torch.zeros_like(confidence),
        )

        weights = confidence * self.joint_weights
        denom = weights.sum(dim=1, keepdim=True).clamp(min=1.0)

        pred_center = (pred * weights.unsqueeze(-1)).sum(dim=1) / denom
        tgt_center = (target * weights.unsqueeze(-1)).sum(dim=1) / denom

        center_loss = (((pred_center - tgt_center) / self.image_size) ** 2).sum(dim=-1).mean()

        pred_spread = torch.sqrt(
            (((pred - pred_center.unsqueeze(1)) ** 2).sum(dim=-1) * weights).sum(dim=1)
            / denom.squeeze(1)
        )

        tgt_spread = torch.sqrt(
            (((target - tgt_center.unsqueeze(1)) ** 2).sum(dim=-1) * weights).sum(dim=1)
            / denom.squeeze(1)
        )

        scale_loss = (((pred_spread - tgt_spread) / self.image_size) ** 2).mean()

        return center_loss + scale_loss

    # =========================================================
    # SILHOUETTE LOSSES
    # =========================================================

    def _distance_silhouette_loss(self, rendered_mask, target_mask, dist_out, dist_in):
        false_positive = rendered_mask * (1.0 - target_mask)
        false_negative = target_mask * (1.0 - rendered_mask)

        fp_loss = (false_positive * dist_out).mean()
        fn_loss = (false_negative * dist_in).mean()

        # Lower false-negative pressure than earlier versions to reduce
        # global expansion, while regional breast/glute area losses preserve
        # legitimate local volume.
        return fp_loss + 0.30 * fn_loss

    def _iou_loss(self, rendered_mask, target_mask, eps=1e-6):
        intersection = (rendered_mask * target_mask).sum(dim=(1, 2, 3))

        union = (
            rendered_mask +
            target_mask -
            rendered_mask * target_mask
        ).sum(dim=(1, 2, 3))

        return 1.0 - ((intersection + eps) / (union + eps)).mean()

    def _edge_loss(self, rendered_mask, target_mask):
        kernel_x = torch.tensor(
            [[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]],
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        kernel_y = torch.tensor(
            [[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]],
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        pred_x = F.conv2d(rendered_mask, kernel_x, padding=1)
        pred_y = F.conv2d(rendered_mask, kernel_y, padding=1)

        tgt_x = F.conv2d(target_mask, kernel_x, padding=1)
        tgt_y = F.conv2d(target_mask, kernel_y, padding=1)

        pred_edge = torch.sqrt(pred_x ** 2 + pred_y ** 2 + 1e-6)
        tgt_edge = torch.sqrt(tgt_x ** 2 + tgt_y ** 2 + 1e-6)

        return torch.abs(pred_edge - tgt_edge).mean()

    # =========================================================
    # STAGES
    # =========================================================

    def _phase_config(self, iteration, iterations):
        frac = iteration / max(1, iterations - 1)

        if frac < 0.15:
            return {
                "name": "camera_pose",
                "kp": 80.0,
                "bone": 20.0,
                "center": 50.0,
                "sil": 0.02,
                "dist": 0.50,
                "iou": 0.25,
                "edge": 0.00,
                "width": 0.00,
                "area": 0.00,
                "bloat": 0.00,
                "shape": 1.00,
                "beta": 0.75,
                "pose": 0.02,
                "trans": 0.05,
                "focal_reg": 0.10,
                "train_betas": False,
                "train_pose": True,
                "train_orient": True,
                "train_trans": True,
                "train_focal": True,
                "lr": 0.02,
            }

        if frac < 0.55:
            return {
                "name": "pose",
                "kp": 60.0,
                "bone": 25.0,
                "center": 30.0,
                "sil": 0.10,
                "dist": 2.00,
                "iou": 0.75,
                "edge": 0.03,
                "width": 0.00,
                "area": 0.00,
                "bloat": 0.05,
                "shape": 0.80,
                "beta": 0.50,
                "pose": 0.015,
                "trans": 0.05,
                "focal_reg": 0.05,
                "train_betas": False,
                "train_pose": True,
                "train_orient": True,
                "train_trans": True,
                "train_focal": True,
                "lr": 0.015,
            }

        if frac < 0.88:
            return {
                "name": "region_shape",
                "kp": 30.0,
                "bone": 15.0,
                "center": 15.0,
                "sil": 0.35,
                "dist": 7.00,
                "iou": 1.40,
                "edge": 0.08,
                "width": self.width_weight,
                "area": self.area_weight,
                "bloat": self.anti_bloat_weight,
                "shape": 0.28,
                "beta": 0.10,
                "pose": 0.02,
                "trans": 0.06,
                "focal_reg": 0.03,
                "train_betas": True,
                "train_pose": True,
                "train_orient": True,
                "train_trans": True,
                "train_focal": True,
                "lr": 0.005,
            }

        return {
            "name": "final",
            "kp": 25.0,
            "bone": 10.0,
            "center": 10.0,
            "sil": 0.30,
            "dist": 8.00,
            "iou": 1.80,
            "edge": 0.10,
            "width": self.width_weight * 0.75,
            "area": self.area_weight * 0.75,
            "bloat": self.anti_bloat_weight * 1.20,
            "shape": 0.42,
            "beta": 0.20,
            "pose": 0.025,
            "trans": 0.08,
            "focal_reg": 0.00,
            "train_betas": True,
            "train_pose": True,
            "train_orient": True,
            "train_trans": True,
            "train_focal": False,
            "lr": 0.003,
        }

    def _set_trainable(self, params, flag):
        for p in params:
            p.requires_grad_(flag)

    def _make_optimizer(
        self,
        betas,
        body_poses,
        global_orients,
        translations,
        log_focal_scale,
        cfg,
    ):
        betas.requires_grad_(cfg["train_betas"])

        self._set_trainable(body_poses.parameters(), cfg["train_pose"])
        self._set_trainable(global_orients.parameters(), cfg["train_orient"])
        self._set_trainable(translations.parameters(), cfg["train_trans"])

        if log_focal_scale is not None:
            log_focal_scale.requires_grad_(cfg["train_focal"])

        params = []

        if betas.requires_grad:
            params.append(
                {
                    "params": [betas],
                    "lr": cfg["lr"] * 0.20,
                }
            )

        pose_params = [p for p in body_poses.parameters() if p.requires_grad]
        orient_params = [p for p in global_orients.parameters() if p.requires_grad]
        trans_params = [p for p in translations.parameters() if p.requires_grad]

        if pose_params:
            params.append(
                {
                    "params": pose_params,
                    "lr": cfg["lr"],
                }
            )

        if orient_params:
            params.append(
                {
                    "params": orient_params,
                    "lr": cfg["lr"] * 0.5,
                }
            )

        if trans_params:
            params.append(
                {
                    "params": trans_params,
                    "lr": cfg["lr"] * 0.5,
                }
            )

        if log_focal_scale is not None and log_focal_scale.requires_grad:
            params.append(
                {
                    "params": [log_focal_scale],
                    "lr": cfg["lr"] * 0.1,
                }
            )

        if not params:
            params = [{"params": [translations[0]], "lr": cfg["lr"]}]

        return torch.optim.Adam(params)

    # =========================================================
    # OPTIMIZATION
    # =========================================================

    def optimize(
        self,
        image_paths,
        pose_json_paths,
        visibility_json_paths,
        output_path,
        iterations=1000,
    ):
        print("\n🚀 Starting region-aware multi-image optimization")

        output_path = Path(output_path)
        debug_dir = output_path.parent / "_render_debug"

        num_images = len(image_paths)
        print(f"✓ Images: {num_images}")

        betas = nn.Parameter(
            torch.zeros(
                1,
                10,
                device=self.device,
            )
        )

        initial_betas = torch.zeros_like(betas).detach()

        log_focal_scale = (
            nn.Parameter(torch.zeros(1, device=self.device))
            if self.optimize_focal
            else None
        )

        body_poses = nn.ParameterList()
        global_orients = nn.ParameterList()
        translations = nn.ParameterList()

        gt_keypoints_all = []
        confidence_all = []
        masks_all = []
        dist_out_all = []
        dist_in_all = []
        metadata_all = []
        target_widths_all = []
        target_areas_all = []
        image_weights = []

        for i in range(num_images):
            keypoints_np = self._load_scaled_pose(
                pose_json_paths[i],
                image_paths[i],
            )

            metadata = load_visibility_json(
                visibility_json_paths[i]
            )

            metadata_all.append(metadata)

            keypoints = torch.tensor(
                keypoints_np[:, :2],
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)

            confidence = torch.tensor(
                keypoints_np[:, 2],
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)

            gt_keypoints_all.append(keypoints)
            confidence_all.append(confidence)

            mask_tensor = self._load_target_mask(
                image_paths[i]
            )

            masks_all.append(mask_tensor)

            dist_out, dist_in = self._distance_maps_from_mask(
                mask_tensor
            )

            dist_out_all.append(dist_out)
            dist_in_all.append(dist_in)

            target_widths_all.append(
                self._mask_band_widths(mask_tensor)
            )

            target_areas_all.append(
                self._region_areas(mask_tensor)
            )

            image_weights.append(
                float(metadata.get("image_weight", 1.0))
            )

            mask_np = (
                mask_tensor[0, 0]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

            body_poses.append(
                nn.Parameter(torch.zeros(1, 63, device=self.device))
            )

            global_orients.append(
                nn.Parameter(torch.zeros(1, 3, device=self.device))
            )

            translations.append(
                nn.Parameter(
                    self._initial_translation_from_mask_and_keypoints(
                        mask_np,
                        keypoints_np,
                    )
                )
            )

        mean_weight = max(
            float(np.mean(image_weights)),
            1e-6,
        )

        image_weights = [
            float(np.clip(w / mean_weight, 0.25, 2.0))
            for w in image_weights
        ]

        camera_center = torch.tensor(
            [[self.image_size / 2.0, self.image_size / 2.0]],
            device=self.device,
        )

        optimizer = None
        active_phase = None

        progress = tqdm(
            range(iterations),
            desc="Optimizing",
            dynamic_ncols=True,
            leave=True,
        )

        for iteration in progress:
            cfg = self._phase_config(
                iteration,
                iterations,
            )

            if active_phase != cfg["name"]:
                active_phase = cfg["name"]

                optimizer = self._make_optimizer(
                    betas,
                    body_poses,
                    global_orients,
                    translations,
                    log_focal_scale,
                    cfg,
                )

            optimizer.zero_grad()

            total_loss = 0.0
            last_losses = {}

            focal_length = self._current_focal(
                log_focal_scale
            )

            for i in range(num_images):
                output = self.model(
                    betas=betas,
                    body_pose=body_poses[i],
                    global_orient=global_orients[i],
                    transl=None,
                    return_verts=True,
                )

                vertices = output.vertices

                joints = SMPLXJointMapper.smplx_to_coco17(
                    output.joints
                )

                projected_joints = self._project_points_screen(
                    joints,
                    focal_length=focal_length,
                    camera_center=camera_center,
                    translation=translations[i],
                )

                rendered = self.renderer.render(
                    vertices=vertices,
                    faces=self.model.faces_tensor.unsqueeze(0),
                    focal_length=focal_length,
                    principal_point=camera_center,
                    translation=translations[i],
                )

                rendered_mask = rendered[..., 3].unsqueeze(1)

                metadata = metadata_all[i]

                visibility_mask = torch.ones_like(
                    masks_all[i]
                )

                border = int(self.image_size * 0.12)

                if metadata.get("truncated_top", False):
                    visibility_mask[:, :, :border, :] *= 0.4

                if metadata.get("truncated_bottom", False):
                    visibility_mask[:, :, -border:, :] *= 0.4

                if metadata.get("truncated_left", False):
                    visibility_mask[:, :, :, :border] *= 0.4

                if metadata.get("truncated_right", False):
                    visibility_mask[:, :, :, -border:] *= 0.4

                region_weights = BodyRegionWeights.create_weight_map(
                    height=self.image_size,
                    width=self.image_size,
                    device=self.device,
                )

                if region_weights.dim() == 3:
                    region_weights = region_weights.unsqueeze(0)

                # Combine older broad body weights with new region-aware maps.
                chest_visible = float(metadata.get("chest_visible", 0.0))
                hip_visible = float(metadata.get("hip_visible", 0.0))

                region_weights = (
                    region_weights *
                    self.sil_region_weights *
                    (1.0 + 0.15 * chest_visible * self.region_maps["breast"]) *
                    (1.0 + 0.12 * hip_visible * self.region_maps["glutes"])
                )

                loss_kp = self._weighted_keypoint_loss(
                    projected_joints,
                    gt_keypoints_all[i],
                    confidence_all[i],
                )

                loss_bone = self._bone_direction_loss(
                    projected_joints,
                    gt_keypoints_all[i],
                    confidence_all[i],
                )

                loss_center = self._keypoint_center_scale_loss(
                    projected_joints,
                    gt_keypoints_all[i],
                    confidence_all[i],
                )

                loss_sil = silhouette_loss(
                    rendered_mask,
                    masks_all[i],
                    visibility_mask,
                    region_weights,
                )

                loss_dist = self._distance_silhouette_loss(
                    rendered_mask,
                    masks_all[i],
                    dist_out_all[i],
                    dist_in_all[i],
                )

                loss_iou = self._iou_loss(
                    rendered_mask,
                    masks_all[i],
                )

                loss_edge = self._edge_loss(
                    rendered_mask,
                    masks_all[i],
                )

                loss_width = self._width_loss(
                    rendered_mask,
                    target_widths_all[i],
                    metadata,
                )

                loss_area = self._regional_area_loss(
                    rendered_mask,
                    target_areas_all[i],
                    metadata,
                )

                loss_bloat = self._anti_bloat_loss(
                    rendered_mask,
                    masks_all[i],
                )

                loss_shape = shape_prior_loss(
                    betas
                )

                loss_beta_stability = (
                    (betas - initial_betas) ** 2
                ).mean()

                loss_pose = pose_prior_loss(
                    body_poses[i]
                )

                loss_trans = translation_loss(
                    translations[i]
                )

                if log_focal_scale is None:
                    loss_focal = torch.tensor(
                        0.0,
                        device=self.device,
                    )
                else:
                    loss_focal = (
                        log_focal_scale ** 2
                    ).mean()

                image_weight = image_weights[i]

                image_loss = image_weight * (
                    cfg["kp"]        * loss_kp +
                    cfg["bone"]      * loss_bone +
                    cfg["center"]    * loss_center +
                    cfg["sil"]       * loss_sil +
                    cfg["dist"]      * loss_dist +
                    cfg["iou"]       * loss_iou +
                    cfg["edge"]      * loss_edge +
                    cfg["width"]     * loss_width +
                    cfg["area"]      * loss_area +
                    cfg["bloat"]     * loss_bloat +
                    cfg["shape"]     * loss_shape +
                    cfg["beta"]      * loss_beta_stability +
                    cfg["pose"]      * loss_pose +
                    cfg["trans"]     * loss_trans +
                    cfg["focal_reg"] * loss_focal
                )

                total_loss += image_loss

                if self.debug and i == 0 and iteration % self.debug_every == 0:
                    self._save_mask_debug(
                        rendered_mask,
                        masks_all[i],
                        projected_joints,
                        gt_keypoints_all[i],
                        confidence_all[i],
                        debug_dir / f"iter_{iteration:04d}_{cfg['name']}.png",
                    )

                last_losses = {
                    "kp": loss_kp,
                    "bone": loss_bone,
                    "center": loss_center,
                    "sil": loss_sil,
                    "dist": loss_dist,
                    "iou": loss_iou,
                    "edge": loss_edge,
                    "width": loss_width,
                    "area": loss_area,
                    "bloat": loss_bloat,
                    "shape": loss_shape,
                    "beta": loss_beta_stability,
                    "pose": loss_pose,
                    "trans": loss_trans,
                    "focal": focal_length[0, 0],
                }

            total_loss.backward()
            optimizer.step()

            with torch.no_grad():
                betas.clamp_(-2.5, 2.5)

                for t in translations:
                    t[:, 2].clamp_(2.0, 10.0)

                if log_focal_scale is not None:
                    log_focal_scale.clamp_(
                        np.log(800.0 / self.base_focal),
                        np.log(4000.0 / self.base_focal),
                    )

            if iteration % 10 == 0:
                progress.set_postfix(
                    {
                        "phase": cfg["name"],
                        "total": f"{total_loss.item():.2f}",
                        "iou": f"{last_losses['iou'].item():.4f}",
                        "sil": f"{last_losses['sil'].item():.4f}",
                    }
                )

            if iteration % 25 == 0:
                tqdm.write(
                    "\n"
                    f"[iter {iteration:04d}] phase={cfg['name']}\n"
                    f"  total: {total_loss.item():.6f}\n"
                    f"  kp:    {last_losses['kp'].item():.6f}\n"
                    f"  bone:  {last_losses['bone'].item():.6f}\n"
                    f"  ctr:   {last_losses['center'].item():.6f}\n"
                    f"  sil:   {last_losses['sil'].item():.6f}\n"
                    f"  dist:  {last_losses['dist'].item():.6f}\n"
                    f"  iou:   {last_losses['iou'].item():.6f}\n"
                    f"  edge:  {last_losses['edge'].item():.6f}\n"
                    f"  width: {last_losses['width'].item():.6f}\n"
                    f"  area:  {last_losses['area'].item():.6f}\n"
                    f"  bloat: {last_losses['bloat'].item():.6f}\n"
                    f"  shape: {last_losses['shape'].item():.6f}\n"
                    f"  beta:  {last_losses['beta'].item():.6f}\n"
                    f"  pose:  {last_losses['pose'].item():.6f}\n"
                    f"  trans: {last_losses['trans'].item():.6f}\n"
                    f"  focal: {last_losses['focal'].item():.2f}\n"
                )

        with torch.no_grad():
            final_focal = self._current_focal(
                log_focal_scale
            )

            neutral_output = self.model(
                betas=betas,
                body_pose=torch.zeros(1, 63, device=self.device),
                global_orient=torch.zeros(1, 3, device=self.device),
                transl=None,
                return_verts=True,
            )

            per_image_vertices = []
            per_image_joints = []

            for i in range(num_images):
                out_i = self.model(
                    betas=betas,
                    body_pose=body_poses[i],
                    global_orient=global_orients[i],
                    transl=None,
                    return_verts=True,
                )

                per_image_vertices.append(
                    out_i.vertices.detach().cpu().numpy()[0]
                )

                per_image_joints.append(
                    out_i.joints.detach().cpu().numpy()[0]
                )

            vertex_region_masks = RegionAwareMasks.template_vertex_masks(
                neutral_output.vertices.detach()[0]
            )

        result = {
            "betas": betas.detach().cpu().numpy(),
            "vertices": neutral_output.vertices.detach().cpu().numpy(),
            "joints": neutral_output.joints.detach().cpu().numpy(),
            "faces": self.model.faces,
            "num_images": num_images,

            "per_image_vertices": np.stack(
                per_image_vertices,
                axis=0,
            ),
            "per_image_joints": np.stack(
                per_image_joints,
                axis=0,
            ),
            "body_poses": np.stack(
                [p.detach().cpu().numpy()[0] for p in body_poses],
                axis=0,
            ),
            "global_orients": np.stack(
                [g.detach().cpu().numpy()[0] for g in global_orients],
                axis=0,
            ),
            "translations": np.stack(
                [t.detach().cpu().numpy()[0] for t in translations],
                axis=0,
            ),
            "focal_length": final_focal.detach().cpu().numpy(),
            "camera_center": camera_center.detach().cpu().numpy(),
            "image_size": np.array([self.image_size], dtype=np.int32),
            "image_paths": np.array([str(p) for p in image_paths]),
            "visibility_json_paths": np.array(
                [str(p) for p in visibility_json_paths]
            ),
        }

        for name, mask in RegionAwareMasks.masks_to_numpy(
            vertex_region_masks
        ).items():
            result[f"vertex_mask_{name}"] = mask

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        np.savez(
            output_path,
            **result,
        )

        print(f"\n✅ Saved canonical body:\n{output_path}")

        return result
