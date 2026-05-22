import json
from pathlib import Path

import cv2
import mediapipe as mp

class FaceLandmarkExtractor:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )

    def process_image(self, image_path, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        image = cv2.imread(str(image_path))
        if image is None:
            return None

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        landmarks_output = []
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for lm in face_landmarks.landmark:
                    landmarks_output.append({
                        "x": float(lm.x),
                        "y": float(lm.y),
                        "z": float(lm.z)
                    })

        output = {
            "image": str(image_path.name),
            "landmarks": landmarks_output
        }

        out_file = output_dir / f"{image_path.stem}.json"
        with open(out_file, "w") as f:
            json.dump(output, f, indent=2)

        return output