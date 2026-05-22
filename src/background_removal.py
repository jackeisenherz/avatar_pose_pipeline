from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO


class BackgroundRemover:
    """
    Production-grade avatar background remover.

    Pipeline:
        1. YOLO human segmentation
        2. Conservative mask selection
        3. Trimap generation
        4. MODNet alpha matting
        5. Foreground protection
        6. Alpha-only refinement
        7. RGBA export

    IMPORTANT:
        - RGB pixels are NEVER modified
        - Only alpha is refined
        - Conservative strategy avoids cutting body parts
    """

    def __init__(
        self,
        yolo_model="yolov8x-seg.pt",
        modnet_onnx="modnet.onnx",
    ):
        print("Loading YOLO segmentation model...")
        self.yolo = YOLO(yolo_model)

        print("Loading MODNet...")
        self.session = ort.InferenceSession(
            modnet_onnx,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )

        self.input_name = self.session.get_inputs()[0].name

        print("Background remover initialized.")

    def process(self, image_path, output_dir):
        image_path = Path(image_path)
        output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        orig_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

        if orig_bgr is None:
            raise RuntimeError(f"Failed loading image: {image_path}")

        h, w = orig_bgr.shape[:2]

        # =========================================================
        # 1. YOLO PERSON SEGMENTATION
        # =========================================================

        coarse_mask = self._get_conservative_person_mask(orig_bgr)

        # =========================================================
        # 2. TRIMAP GENERATION
        # =========================================================

        trimap = self._generate_trimap(coarse_mask)

        # =========================================================
        # 3. MODNET ALPHA MATTING
        # =========================================================

        alpha = self._run_modnet(orig_bgr)

        # =========================================================
        # 4. FOREGROUND PROTECTION
        # =========================================================

        alpha = self._protect_foreground(alpha, coarse_mask)

        # =========================================================
        # 5. EDGE REFINEMENT (ALPHA ONLY)
        # =========================================================

        alpha = self._refine_alpha(alpha)

        # =========================================================
        # 6. APPLY TRIMAP CONSTRAINTS
        # =========================================================

        alpha[trimap == 255] = 255
        alpha[trimap == 0] = 0

        # =========================================================
        # 7. EXPORT RGBA
        # =========================================================

        rgba = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = alpha

        output_path = output_dir / f"{image_path.stem}.png"

        cv2.imwrite(str(output_path), rgba)

        print(f"✓ Saved: {output_path}")

        return output_path

    # =============================================================
    # PERSON DETECTION
    # =============================================================

    def _get_conservative_person_mask(self, image):
        h, w = image.shape[:2]

        results = self.yolo(image, verbose=False)[0]

        if results.masks is None:
            raise RuntimeError("No segmentation masks detected.")

        masks = results.masks.data.cpu().numpy()

        center = (w // 2, h // 2)

        best_mask = None
        best_score = -1

        for mask in masks:

            if mask.shape[:2] != (h, w):
                mask = cv2.resize(
                    mask,
                    (w, h),
                    interpolation=cv2.INTER_LINEAR
                )

            area = mask.sum()

            center_bonus = (
                1_000_000
                if mask[center[1], center[0]] > 0.5
                else 0
            )

            score = area + center_bonus

            if score > best_score:
                best_score = score
                best_mask = mask

        mask = (best_mask > 0.35).astype(np.uint8) * 255

        # Conservative expansion
        kernel = np.ones((9, 9), np.uint8)

        mask = cv2.dilate(mask, kernel, iterations=2)

        return mask

    # =============================================================
    # TRIMAP
    # =============================================================

    def _generate_trimap(self, mask):
        kernel = np.ones((15, 15), np.uint8)

        sure_fg = cv2.erode(mask, kernel, iterations=2)
        sure_bg = cv2.dilate(mask, kernel, iterations=3)

        trimap = np.full(mask.shape, 128, dtype=np.uint8)

        trimap[sure_fg == 255] = 255
        trimap[sure_bg == 0] = 0

        return trimap

    # =============================================================
    # MODNET
    # =============================================================

    def _run_modnet(self, bgr):
        h, w = bgr.shape[:2]

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        input_size = 512

        resized = cv2.resize(rgb, (input_size, input_size))

        tensor = resized.astype(np.float32) / 255.0

        tensor = (tensor - 0.5) / 0.5

        tensor = np.transpose(tensor, (2, 0, 1))

        tensor = np.expand_dims(tensor, axis=0)

        matte = self.session.run(
            None,
            {self.input_name: tensor}
        )[0]

        matte = matte[0][0]

        matte = cv2.resize(
            matte,
            (w, h),
            interpolation=cv2.INTER_LINEAR
        )

        matte = np.clip(matte * 255, 0, 255).astype(np.uint8)

        return matte

    # =============================================================
    # FOREGROUND PROTECTION
    # =============================================================

    def _protect_foreground(self, alpha, coarse_mask):
        """
        Prevent body parts from disappearing.
        False positives are preferred over false negatives.
        """

        protected = np.maximum(alpha, coarse_mask)

        return protected

    # =============================================================
    # ALPHA REFINEMENT ONLY
    # =============================================================

    def _refine_alpha(self, alpha):
        """
        IMPORTANT:
            ONLY refine alpha.
            NEVER touch RGB.
        """

        alpha = cv2.GaussianBlur(alpha, (3, 3), 0)

        alpha = cv2.normalize(
            alpha,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        )

        return alpha