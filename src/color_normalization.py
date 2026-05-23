from pathlib import Path
import json
import cv2
import numpy as np


class ColorNormalizer:
    """
    Avatar-oriented image normalization.

    Goals:
    - preserve original texture detail
    - preserve alpha
    - normalize foreground exposure
    - reduce white balance drift
    - avoid destructive beautification
    - emit metadata for later photometric optimization
    """

    def __init__(
        self,
        target_luma=145.0,
        target_contrast=52.0,
        clahe_clip=1.25,
        clahe_grid=(8, 8),
        skin_warmth=0.0,
        preserve_sharpness=True,
        write_metadata=True
    ):
        self.target_luma = target_luma
        self.target_contrast = target_contrast
        self.clahe_clip = clahe_clip
        self.clahe_grid = clahe_grid
        self.skin_warmth = skin_warmth
        self.preserve_sharpness = preserve_sharpness
        self.write_metadata = write_metadata

    def process(self, image_path, output_dir):
        image_path = Path(image_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            return None

        bgr, alpha = self._split_alpha(image)

        mask = self._foreground_mask(alpha, bgr)
        before_stats = self._stats(bgr, mask)

        work = bgr.astype(np.float32)

        work = self._gray_world_white_balance(work, mask)
        work = self._robust_exposure_normalize(work, mask)
        work = self._mild_luminance_clahe(work, mask)

        if self.skin_warmth != 0.0:
            work = self._controlled_warmth(work, mask, amount=self.skin_warmth)

        work = np.clip(work, 0, 255).astype(np.uint8)

        if self.preserve_sharpness:
            work = self._texture_safe_blend(original=bgr, normalized=work, mask=mask)

        after_stats = self._stats(work, mask)

        output = self._merge_alpha(work, alpha)
        output_path = output_dir / image_path.with_suffix(".png").name
        cv2.imwrite(str(output_path), output)

        if self.write_metadata:
            meta_path = output_dir / f"{image_path.stem}_normalization.json"
            with open(meta_path, "w") as f:
                json.dump(
                    {
                        "image": image_path.name,
                        "output": output_path.name,
                        "before": before_stats,
                        "after": after_stats,
                        "settings": {
                            "target_luma": self.target_luma,
                            "target_contrast": self.target_contrast,
                            "clahe_clip": self.clahe_clip,
                            "clahe_grid": self.clahe_grid,
                            "skin_warmth": self.skin_warmth,
                            "preserve_sharpness": self.preserve_sharpness,
                        },
                    },
                    f,
                    indent=2
                )

        return output_path

    def _split_alpha(self, image):
        if len(image.shape) == 3 and image.shape[2] == 4:
            return image[:, :, :3], image[:, :, 3]
        return image[:, :, :3], None

    def _merge_alpha(self, bgr, alpha):
        if alpha is None:
            return bgr
        bgra = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
        bgra[:, :, 3] = alpha
        return bgra

    def _foreground_mask(self, alpha, bgr):
        if alpha is not None:
            mask = (alpha > 20).astype(np.uint8)
        else:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            mask = (gray > 5).astype(np.uint8)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        return mask.astype(bool)

    def _stats(self, bgr, mask):
        if mask.sum() < 100:
            mask = np.ones(bgr.shape[:2], dtype=bool)

        pixels = bgr[mask].astype(np.float32)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l = lab[:, :, 0][mask].astype(np.float32)

        return {
            "b_mean": float(pixels[:, 0].mean()),
            "g_mean": float(pixels[:, 1].mean()),
            "r_mean": float(pixels[:, 2].mean()),
            "luma_mean": float(l.mean()),
            "luma_std": float(l.std()),
            "foreground_pixels": int(mask.sum()),
        }

    def _gray_world_white_balance(self, bgr_float, mask):
        if mask.sum() < 100:
            return bgr_float

        pixels = bgr_float[mask]
        means = pixels.mean(axis=0)
        gray = means.mean()

        scale = gray / np.maximum(means, 1e-6)
        scale = np.clip(scale, 0.85, 1.15)

        corrected = bgr_float * scale.reshape(1, 1, 3)
        return np.clip(corrected, 0, 255)

    def _robust_exposure_normalize(self, bgr_float, mask):
        lab = cv2.cvtColor(np.clip(bgr_float, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        l_float = l.astype(np.float32)
        fg = l_float[mask]

        if fg.size < 100:
            return bgr_float

        current_mean = np.percentile(fg, 50)
        current_p10 = np.percentile(fg, 10)
        current_p90 = np.percentile(fg, 90)
        current_contrast = max(current_p90 - current_p10, 1.0)

        gain = self.target_contrast / current_contrast
        gain = np.clip(gain, 0.75, 1.25)

        shift = self.target_luma - current_mean
        shift = np.clip(shift, -18, 18)

        l_new = (l_float - current_mean) * gain + current_mean + shift
        l_new = np.clip(l_new, 0, 255).astype(np.uint8)

        merged = cv2.merge((l_new, a, b))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR).astype(np.float32)

    def _mild_luminance_clahe(self, bgr_float, mask):
        bgr_u8 = np.clip(bgr_float, 0, 255).astype(np.uint8)
        lab = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip,
            tileGridSize=self.clahe_grid
        )

        l_clahe = clahe.apply(l)

        # Blend CLAHE gently to avoid fake skin texture.
        l_out = cv2.addWeighted(l, 0.75, l_clahe, 0.25, 0)

        merged = cv2.merge((l_out, a, b))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR).astype(np.float32)

    def _controlled_warmth(self, bgr_float, mask, amount=0.03):
        lab = cv2.cvtColor(np.clip(bgr_float, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        a = a.astype(np.float32)
        b = b.astype(np.float32)

        # Very small warmth shift only. Avoid global color stylization.
        b[mask] += 255.0 * amount
        a[mask] += 255.0 * amount * 0.25

        merged = cv2.merge((
            l,
            np.clip(a, 0, 255).astype(np.uint8),
            np.clip(b, 0, 255).astype(np.uint8)
        ))

        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR).astype(np.float32)

    def _texture_safe_blend(self, original, normalized, mask):
        """
        Preserve high-frequency texture from original while using normalized low-frequency color.
        This avoids destroying pores, skin gradients, and reconstruction features.
        """
        original_f = original.astype(np.float32)
        normalized_f = normalized.astype(np.float32)

        original_blur = cv2.GaussianBlur(original_f, (0, 0), 3.0)
        normalized_blur = cv2.GaussianBlur(normalized_f, (0, 0), 3.0)

        high_freq = original_f - original_blur
        reconstructed = normalized_blur + high_freq

        # Only apply strongly on foreground.
        mask_f = mask.astype(np.float32)[:, :, None]
        out = reconstructed * mask_f + normalized_f * (1.0 - mask_f)

        return np.clip(out, 0, 255).astype(np.uint8)