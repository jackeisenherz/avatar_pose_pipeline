import cv2
import numpy as np
import torch


class PseudoLandmarkExtractor:

    @staticmethod
    def extract(mask):

        """
        mask:
            [H, W]
        """

        H, W = mask.shape

        ys, xs = np.where(mask > 0.5)

        if len(xs) == 0:
            return None

        top = ys.min()
        bottom = ys.max()

        height = bottom - top

        # =================================================
        # BODY SLICES
        # =================================================

        chest_y = int(top + height * 0.32)

        waist_y = int(top + height * 0.50)

        hip_y = int(top + height * 0.68)

        # =================================================
        # EXTRACT WIDTHS
        # =================================================

        chest = PseudoLandmarkExtractor._extract_width(
            mask,
            chest_y
        )

        waist = PseudoLandmarkExtractor._extract_width(
            mask,
            waist_y
        )

        hips = PseudoLandmarkExtractor._extract_width(
            mask,
            hip_y
        )

        return {
            "chest": chest,
            "waist": waist,
            "hips": hips
        }

    @staticmethod
    def _extract_width(mask, y):

        row = mask[y]

        xs = np.where(row > 0.5)[0]

        if len(xs) < 2:
            return None

        left = xs.min()
        right = xs.max()

        center = (left + right) / 2

        return {
            "left": np.array([left, y]),
            "right": np.array([right, y]),
            "center": np.array([center, y]),
            "width": float(right - left)
        }