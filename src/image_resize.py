
from PIL import Image
from pathlib import Path

class ImageResizer:
    def __init__(self, max_resolution=2048):
        self.max_resolution = max_resolution

    def process(self, image_path, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        image = Image.open(image_path)

        width, height = image.size

        scale = min(
            self.max_resolution / width,
            self.max_resolution / height,
            1.0
        )

        new_size = (
            int(width * scale),
            int(height * scale)
        )

        resized = image.resize(new_size)

        output_path = output_dir / image_path.name

        resized.save(output_path)

        return output_path
