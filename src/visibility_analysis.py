import json
from pathlib import Path

import cv2
import numpy as np


class VisibilityAnalyzer:

    def __init__(self):

        # COCO indices
        self.chest_joints = [5, 6]
        self.hip_joints = [11, 12]
        self.knee_joints = [13, 14]
        self.ankle_joints = [15, 16]

    def process(
        self,
        rgba_path,
        pose_json_path,
        output_dir
    ):

        output_dir = Path(output_dir)

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        # =====================================================
        # LOAD IMAGE
        # =====================================================

        rgba = cv2.imread(
            str(rgba_path),
            cv2.IMREAD_UNCHANGED
        )

        if rgba is None:
            return None

        h, w = rgba.shape[:2]

        # =====================================================
        # LOAD POSE
        # =====================================================

        with open(pose_json_path, "r") as f:
            pose_data = json.load(f)

        keypoints = np.array(
            pose_data["keypoints"],
            dtype=np.float32
        )

        if len(keypoints) == 0:
            return None

        # =====================================================
        # CONFIDENCE
        # =====================================================

        visibility = keypoints[:, 2]

        visible_mask = visibility > 0.5

        visible_joints = int(
            visible_mask.sum()
        )

        avg_confidence = float(
            visibility[visible_mask].mean()
        ) if visible_joints > 0 else 0.0

        # =====================================================
        # BOUNDING BOX
        # =====================================================

        visible_points = keypoints[
            visible_mask
        ]

        if len(visible_points) == 0:
            return None

        x1 = np.min(visible_points[:, 0])
        y1 = np.min(visible_points[:, 1])

        x2 = np.max(visible_points[:, 0])
        y2 = np.max(visible_points[:, 1])

        bbox_width = x2 - x1
        bbox_height = y2 - y1

        # =====================================================
        # BODY COVERAGE
        # =====================================================

        height_ratio = bbox_height / h
        width_ratio = bbox_width / w

        # =====================================================
        # TRUNCATION DETECTION
        # =====================================================

        margin = 0.03

        truncated_top = y1 < (h * margin)

        truncated_bottom = y2 > (
            h * (1.0 - margin)
        )

        truncated_left = x1 < (
            w * margin
        )

        truncated_right = x2 > (
            w * (1.0 - margin)
        )

        # =====================================================
        # SILHOUETTE AREA
        # =====================================================

        if rgba.shape[2] == 4:

            alpha = rgba[:, :, 3]

            silhouette_area = (
                alpha > 10
            ).sum() / (h * w)

        else:

            silhouette_area = 0.0

        # =====================================================
        # CROP TYPE
        # =====================================================

        if (
            height_ratio > 0.75 and
            not truncated_top and
            not truncated_bottom
        ):

            crop_type = "full_body"

        elif height_ratio > 0.45:

            crop_type = "american"

        else:

            crop_type = "torso"

        # =====================================================
        # QUALITY SCORE
        # =====================================================

        quality_score = 1.0

        # Penalize truncation
        truncations = sum([
            truncated_top,
            truncated_bottom,
            truncated_left,
            truncated_right
        ])

        quality_score *= (
            1.0 - truncations * 0.15
        )

        # Joint confidence
        quality_score *= avg_confidence

        # Silhouette quality
        quality_score *= min(
            silhouette_area * 3.0,
            1.0
        )

        # Coverage quality
        quality_score *= min(
            height_ratio * 1.5,
            1.0
        )

        quality_score = max(
            quality_score,
            0.05
        )

        # =====================================================
        # REGION VISIBILITY
        # =====================================================

        chest_visible = float(
            (
                visibility[
                    self.chest_joints
                ] > 0.5
            ).mean()
        )

        hip_visible = float(
            (
                visibility[
                    self.hip_joints
                ] > 0.5
            ).mean()
        )

        leg_visible = float(
            (
                visibility[
                    self.knee_joints +
                    self.ankle_joints
                ] > 0.5
            ).mean()
        )

        # =====================================================
        # FINAL IMAGE WEIGHT
        # =====================================================

        image_weight = quality_score

        # Full body bonus
        if crop_type == "full_body":

            image_weight *= 2.0

        elif crop_type == "american":

            image_weight *= 1.5

        # Region importance
        image_weight *= (
            1.0 +
            0.5 * chest_visible +
            0.3 * hip_visible +
            0.2 * leg_visible
        )

        # =====================================================
        # RESULT
        # =====================================================

        result = {

            "image":
                rgba_path.name,

            "crop_type":
                crop_type,

            "image_weight":
                float(image_weight),

            "quality_score":
                float(quality_score),

            "visible_ratio":
                float(height_ratio),

            "width_ratio":
                float(width_ratio),

            "silhouette_area":
                float(silhouette_area),

            "visible_joints":
                int(visible_joints),

            "avg_joint_confidence":
                float(avg_confidence),

            "truncated_top":
                bool(truncated_top),

            "truncated_bottom":
                bool(truncated_bottom),

            "truncated_left":
                bool(truncated_left),

            "truncated_right":
                bool(truncated_right),

            # =============================================
            # REGION VISIBILITY
            # =============================================

            "chest_visible":
                chest_visible,

            "hip_visible":
                hip_visible,

            "leg_visible":
                leg_visible,

            # =============================================
            # BBOX
            # =============================================

            "bbox": {

                "x1": int(x1),
                "y1": int(y1),

                "x2": int(x2),
                "y2": int(y2)
            }
        }

        # =====================================================
        # SAVE
        # =====================================================

        out_path = (
            output_dir /
            f"{rgba_path.stem}.json"
        )

        with open(out_path, "w") as f:

            json.dump(
                result,
                f,
                indent=2
            )

        return result