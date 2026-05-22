
from pathlib import Path

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]

def list_images(folder):
    folder = Path(folder)

    images = []

    for ext in IMAGE_EXTENSIONS:
        images.extend(folder.glob(f"*{ext}"))
        images.extend(folder.glob(f"*{ext.upper()}"))

    return sorted(images)
