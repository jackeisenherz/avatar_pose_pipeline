import numpy as np
from pathlib import Path

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]

def list_images(folder):
    folder = Path(folder)

    images = []

    for ext in IMAGE_EXTENSIONS:
        images.extend(folder.glob(f"*{ext}"))
        images.extend(folder.glob(f"*{ext.upper()}"))

    return sorted(images)

def export_npz_to_obj(npz_path, obj_path):

    data = np.load(
        npz_path,
        allow_pickle=True
    )

    vertices = data["vertices"]
    faces = data["faces"]

    # ============================================
    # REMOVE BATCH DIMENSIONS
    # ============================================

    while vertices.ndim > 2:
        vertices = vertices[0]

    while faces.ndim > 2:
        faces = faces[0]

    vertices = vertices.astype(np.float32)
    faces = faces.astype(np.int32)

    # ============================================
    # VALIDATION
    # ============================================

    if np.isnan(vertices).any():
        raise RuntimeError(
            "Vertices contain NaN"
        )

    if np.isinf(vertices).any():
        raise RuntimeError(
            "Vertices contain Inf"
        )

    # ============================================
    # WRITE OBJ
    # ============================================

    with open(obj_path, "w") as f:

        for v in vertices:

            f.write(
                f"v {v[0]} {v[1]} {v[2]}\n"
            )

        for face in faces:

            a = int(face[0]) + 1
            b = int(face[1]) + 1
            c = int(face[2]) + 1

            f.write(
                f"f {a} {b} {c}\n"
            )

    print(f"✅ OBJ exported: {obj_path}")