# src/pose_estimation.py
import json
from pathlib import Path
from ultralytics import YOLO

class PoseEstimator:
    def __init__(self, model_name="yolov8l-pose.pt"):
        self.model = YOLO(model_name)
        self.model.to('cuda' if self.model.device.type == 'cuda' else 'cpu')

    def process_image(self, image_path, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = self.model(str(image_path), verbose=False)

        if not results or len(results[0].keypoints) == 0:
            return None

        # Take first person
        kpts = results[0].keypoints.data[0].cpu().numpy()  # (17, 3) -> x, y, conf

        formatted = []
        for x, y, conf in kpts:
            formatted.append({
                "x": float(x),
                "y": float(y),
                "confidence": float(conf)
            })

        output = {
            "image": str(image_path.name),
            "keypoints": formatted
        }

        out_file = output_dir / f"{image_path.stem}.json"
        with open(out_file, "w") as f:
            json.dump(output, f, indent=2)

        return output