from pathlib import Path
import numpy as np


def load_npz_mesh(path):
    path = Path(path)
    data = np.load(path, allow_pickle=True)

    if "vertices" not in data or "faces" not in data:
        raise KeyError(f"NPZ must contain vertices and faces: {path}")

    vertices = np.asarray(data["vertices"], dtype=np.float32)
    faces = np.asarray(data["faces"], dtype=np.int64)

    while vertices.ndim > 2:
        vertices = vertices[0]
    while faces.ndim > 2:
        faces = faces[0]

    return vertices, faces


def save_npz_mesh(path, vertices, faces, **extra):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "vertices": np.asarray(vertices, dtype=np.float32),
        "faces": np.asarray(faces, dtype=np.int64),
    }
    payload.update(extra)
    np.savez(path, **payload)


def read_obj(path):
    path = Path(path)

    vertices = []
    texcoords = []
    faces = []
    face_texcoords = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            if line.startswith("v "):
                p = line.split()
                vertices.append([float(p[1]), float(p[2]), float(p[3])])

            elif line.startswith("vt "):
                p = line.split()
                texcoords.append([float(p[1]), float(p[2])])

            elif line.startswith("f "):
                parsed_v = []
                parsed_t = []

                for item in line.split()[1:]:
                    parts = item.split("/")
                    vi = int(parts[0]) - 1
                    ti = int(parts[1]) - 1 if len(parts) > 1 and parts[1] else None
                    parsed_v.append(vi)
                    parsed_t.append(ti)

                for i in range(1, len(parsed_v) - 1):
                    faces.append([parsed_v[0], parsed_v[i], parsed_v[i + 1]])
                    face_texcoords.append([parsed_t[0], parsed_t[i], parsed_t[i + 1]])

    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)

    texcoords = np.asarray(texcoords, dtype=np.float32) if texcoords else None
    has_uv = texcoords is not None and face_texcoords and all(
        all(v is not None for v in tri) for tri in face_texcoords
    )
    face_texcoords = np.asarray(face_texcoords, dtype=np.int64) if has_uv else None

    return vertices, faces, texcoords, face_texcoords


def write_obj(path, vertices, faces, comments=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)

    with open(path, "w", encoding="utf-8") as f:
        if comments:
            for c in comments:
                f.write(f"# {c}\n")

        for v in vertices:
            f.write(f"v {float(v[0]):.8f} {float(v[1]):.8f} {float(v[2]):.8f}\n")

        for tri in faces:
            a, b, c = tri[:3] + 1
            f.write(f"f {int(a)} {int(b)} {int(c)}\n")
