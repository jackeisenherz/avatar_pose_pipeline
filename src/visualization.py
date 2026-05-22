from pathlib import Path
import cv2

def draw_pose(image_path, pose_json, output_path):
    """
    Draw YOLOv8 Pose keypoints (17 points) on the image.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"⚠ Could not read image: {image_path}")
        return

    # YOLOv8 Pose keypoint connections (skeleton)
    skeleton = [
        (0, 1), (0, 2), (1, 3), (2, 4),           # head + ears
        (5, 6), (5, 11), (6, 12),                  # shoulders
        (11, 12),                                  # hips
        (5, 7), (7, 9), (6, 8), (8, 10),          # arms
        (11, 13), (13, 15), (12, 14), (14, 16)    # legs
    ]

    keypoints = pose_json.get("keypoints", [])

    # Draw keypoints
    for i, kp in enumerate(keypoints):
        x = int(kp["x"])
        y = int(kp["y"])
        conf = kp.get("confidence", 0)

        if conf > 0.3:  # only draw confident points
            color = (0, 255, 0) if conf > 0.7 else (0, 165, 255)
            cv2.circle(image, (x, y), 5, color, -1)
            cv2.putText(image, str(i), (x + 5, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Draw skeleton lines
    for start_idx, end_idx in skeleton:
        if start_idx < len(keypoints) and end_idx < len(keypoints):
            start = keypoints[start_idx]
            end = keypoints[end_idx]

            if start["confidence"] > 0.3 and end["confidence"] > 0.3:
                pt1 = (int(start["x"]), int(start["y"]))
                pt2 = (int(end["x"]), int(end["y"]))
                cv2.line(image, pt1, pt2, (0, 255, 255), 2)

    output_path = Path(output_path)
    cv2.imwrite(str(output_path), image)
    print(f"   📸 Visualization saved: {output_path.name}")