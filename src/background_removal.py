from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO


class BackgroundRemover:
    """
    Avatar-safe background remover.

    Main principles:
    - Filter YOLO detections to PERSON class only.
    - Never let non-person objects become foreground.
    - Preserve original RGB pixels.
    - Refine alpha only.
    - Prefer keeping extra subject pixels over cutting the body.
    - Save debug masks for inspection.
    """

    def __init__(
        self,
        yolo_model="yolov8x-seg.pt",
        modnet_onnx="models/modnet/modnet.onnx",
        confidence=0.15,
        debug=True
    ):
        self.debug = debug
        self.confidence = confidence

        print("Loading YOLO person segmentation model...")
        self.yolo = YOLO(yolo_model)

        self.use_modnet = Path(modnet_onnx).exists()

        if self.use_modnet:
            print("Loading MODNet...")
            self.session = ort.InferenceSession(
                modnet_onnx,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            )
            self.input_name = self.session.get_inputs()[0].name
        else:
            print(f"MODNet not found at {modnet_onnx}; using YOLO-only alpha.")
            self.session = None
            self.input_name = None

        print("Background remover initialized.")

    def process(self, image_path, output_dir):
        image_path = Path(image_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        orig_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

        if orig_bgr is None:
            raise RuntimeError(f"Failed loading image: {image_path}")

        # 1. Person-only YOLO mask
        person_mask = self._get_person_mask(orig_bgr)

        # 2. If YOLO fails, fail loudly instead of producing wrong masks
        if person_mask is None or person_mask.sum() == 0:
            raise RuntimeError(
                f"No reliable person mask found for {image_path.name}. "
                "Do not continue; wrong masks poison the reconstruction."
            )

        # 3. Conservative cleanup
        person_mask = self._clean_person_mask(person_mask)

        # 4. Optional MODNet alpha, constrained by YOLO person area
        if self.use_modnet:
            alpha = self._run_modnet(orig_bgr)
            alpha = self._combine_yolo_and_modnet(alpha, person_mask)
        else:
            alpha = person_mask.copy()

        # 5. Alpha-only refinement
        alpha = self._refine_alpha(alpha, person_mask)

        # 6. Export RGBA without touching RGB
        rgba = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = alpha

        output_path = output_dir / f"{image_path.stem}.png"
        cv2.imwrite(str(output_path), rgba)

        if self.debug:
            debug_dir = output_dir / "_debug_masks"
            debug_dir.mkdir(parents=True, exist_ok=True)

            cv2.imwrite(
                str(debug_dir / f"{image_path.stem}_person_mask.png"),
                person_mask
            )

            cv2.imwrite(
                str(debug_dir / f"{image_path.stem}_alpha.png"),
                alpha
            )

        print(f"✓ Saved: {output_path}")

        return output_path

    # =============================================================
    # YOLO PERSON MASK
    # =============================================================

    def _get_person_mask(self, image):
        h, w = image.shape[:2]

        results = self.yolo(
            image,
            verbose=False,
            conf=self.confidence,
            classes=[0]
        )[0]

        if results.masks is None or results.boxes is None:
            return None

        masks = results.masks.data.cpu().numpy()
        boxes = results.boxes

        if len(masks) == 0:
            return None

        person_masks = []

        for idx, mask in enumerate(masks):
            cls = int(boxes.cls[idx].item())

            # COCO class 0 = person
            if cls != 0:
                continue

            conf = float(boxes.conf[idx].item())

            if conf < self.confidence:
                continue

            if mask.shape[:2] != (h, w):
                mask = cv2.resize(
                    mask,
                    (w, h),
                    interpolation=cv2.INTER_LINEAR
                )

            mask_u8 = (mask > 0.20).astype(np.uint8) * 255

            area = mask_u8.sum() / 255

            if area < h * w * 0.01:
                continue

            person_masks.append((mask_u8, conf, area))

        if len(person_masks) == 0:
            return None

        # Select main subject only.
        # The real model should be the largest person-like component,
        # usually lower / central in the image.
        image_center_x = w / 2
        image_center_y = h / 2

        best_mask = None
        best_score = -1

        for mask_u8, conf, area in person_masks:
            ys, xs = np.where(mask_u8 > 0)

            if len(xs) == 0:
                continue

            x1, x2 = xs.min(), xs.max()
            y1, y2 = ys.min(), ys.max()

            bbox_w = x2 - x1
            bbox_h = y2 - y1

            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            center_dist = abs(cx - image_center_x) / w
            vertical_bonus = cy / h

            # Reject tiny wall/photo people
            if area < h * w * 0.03:
                continue

            # Reject detections that are too high in the frame,
            # common for wall pictures/posters.
            if cy < h * 0.25:
                continue

            score = (
                area * 1.0
                + conf * h * w * 0.2
                + vertical_bonus * h * w * 0.15
                - center_dist * h * w * 0.25
            )

            if score > best_score:
                best_score = score
                best_mask = mask_u8

        if best_mask is None:
            # fallback: largest detected person
            best_mask = max(person_masks, key=lambda x: x[2])[0]

        return best_mask

    # =============================================================
    # MASK CLEANUP
    # =============================================================

    def _clean_person_mask(self, mask):
        mask = mask.astype(np.uint8)

        # Do NOT aggressively close; it fills arm/body gaps.
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)

        # Only tiny dilation for safety.
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.dilate(mask, kernel_dilate, iterations=1)

        return mask

    # =============================================================
    # MODNET
    # =============================================================

    def _run_modnet(self, bgr):
        h, w = bgr.shape[:2]

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        input_size = 512

        resized = cv2.resize(
            rgb,
            (input_size, input_size),
            interpolation=cv2.INTER_AREA
        )

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
    # COMBINE YOLO + MODNET
    # =============================================================

    def _combine_yolo_and_modnet(self, modnet_alpha, person_mask):
        h, w = person_mask.shape

        allowed = cv2.dilate(
            person_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)),
            iterations=1
        )

        alpha = np.zeros((h, w), dtype=np.uint8)
        alpha[allowed > 0] = modnet_alpha[allowed > 0]

        # Protect only strong MODNet foreground, not the full YOLO mask.
        sure_fg = (
            (person_mask > 0) &
            (modnet_alpha > 170)
        ).astype(np.uint8) * 255

        sure_fg = cv2.erode(
            sure_fg,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=1
        )

        alpha[sure_fg > 0] = 255

        # Carve internal gaps between arms/body.
        gap_candidates = (
            (person_mask > 0) &
            (modnet_alpha < 80)
        ).astype(np.uint8) * 255

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            gap_candidates,
            connectivity=8
        )

        min_gap_area = h * w * 0.001

        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]

            if area >= min_gap_area:
                alpha[labels == label] = 0

        return alpha
    # =============================================================
    # ALPHA REFINEMENT ONLY
    # =============================================================

    def _refine_alpha(self, alpha, person_mask):
        alpha = alpha.astype(np.uint8)

        alpha[alpha < 15] = 0
        alpha[alpha > 240] = 255

        # Smooth only alpha edge.
        alpha = cv2.GaussianBlur(alpha, (3, 3), 0)

        # Protect only very confident inner body.
        core = (
            (person_mask > 0) &
            (alpha > 180)
        ).astype(np.uint8) * 255

        core = cv2.erode(
            core,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
            iterations=1
        )

        alpha[core > 0] = 255

        return alpha