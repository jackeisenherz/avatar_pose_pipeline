from pathlib import Path
import cv2

class ColorNormalizer:
    def process(self, image_path, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            return None

        if image.shape[2] == 4:
            bgr = image[:, :, :3]
            alpha = image[:, :, 3]
        else:
            bgr = image
            alpha = None

        # Gentle warm-up to fix blue skin tone
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Mild contrast + warm skin tone
        clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8,8))
        l = clahe.apply(l)
        b = cv2.add(b, 5)   # slight yellow/warm boost

        merged = cv2.merge((l, a, b))
        normalized = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

        if alpha is not None:
            normalized = cv2.cvtColor(normalized, cv2.COLOR_BGR2BGRA)
            normalized[:, :, 3] = alpha

        output_path = output_dir / image_path.name
        cv2.imwrite(str(output_path), normalized)

        return output_path