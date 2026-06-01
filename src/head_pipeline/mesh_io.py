from pathlib import Path
import numpy as np


def read_obj_full(path):
    """
    Read OBJ vertices, texcoords, normals, and faces.

    faces are returned as list of triangles:
        [(v_idx, vt_idx, vn_idx), ...]
    where vt_idx/vn_idx may be None.
    """
    path = Path(path)
    vertices = []
    texcoords = []
    normals = []
    faces = []

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

            elif line.startswith("vn "):
                p = line.split()
                normals.append([float(p[1]), float(p[2]), float(p[3])])

            elif line.startswith("f "):
                items = line.split()[1:]
                parsed = []

                for item in items:
                    parts = item.split("/")
                    vi = int(parts[0]) - 1 if parts[0] else None
                    ti = int(parts[1]) - 1 if len(parts) > 1 and parts[1] else None
                    ni = int(parts[2]) - 1 if len(parts) > 2 and parts[2] else None
                    parsed.append((vi, ti, ni))

                # triangulate fan if needed
                if len(parsed) == 3:
                    faces.append(parsed)
                elif len(parsed) > 3:
                    for i in range(1, len(parsed) - 1):
                        faces.append([parsed[0], parsed[i], parsed[i + 1]])

    return {
        "vertices": np.asarray(vertices, dtype=np.float32),
        "texcoords": np.asarray(texcoords, dtype=np.float32) if texcoords else None,
        "normals": np.asarray(normals, dtype=np.float32) if normals else None,
        "faces": faces,
    }


def read_obj_vertices_faces(path):
    obj = read_obj_full(path)

    faces = []
    for face in obj["faces"]:
        faces.append([int(face[0][0]), int(face[1][0]), int(face[2][0])])

    return obj["vertices"], np.asarray(faces, dtype=np.int64)


def extract_uv_layout_from_obj(path):
    """
    Return UV layout from a DECA/FLAME OBJ if available.

    Returns:
        uv_coords: [T, 2]
        face_uvs: [F, 3]
        face_vertices: [F, 3]
    """
    obj = read_obj_full(path)

    if obj["texcoords"] is None or len(obj["texcoords"]) == 0:
        return None

    face_vertices = []
    face_uvs = []

    for face in obj["faces"]:
        fv = []
        fu = []

        for vi, ti, _ in face:
            if vi is None or ti is None:
                return None
            fv.append(int(vi))
            fu.append(int(ti))

        face_vertices.append(fv)
        face_uvs.append(fu)

    return {
        "uv_coords": obj["texcoords"],
        "face_uvs": np.asarray(face_uvs, dtype=np.int64),
        "face_vertices": np.asarray(face_vertices, dtype=np.int64),
    }


def write_obj(path, vertices, faces, mtl_name=None, material_name="material0"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)

    while vertices.ndim > 2:
        vertices = vertices[0]

    while faces.ndim > 2:
        faces = faces[0]

    with open(path, "w", encoding="utf-8") as f:
        if mtl_name:
            f.write(f"mtllib {mtl_name}\n")
            f.write(f"usemtl {material_name}\n")

        for v in vertices:
            f.write(f"v {float(v[0])} {float(v[1])} {float(v[2])}\n")

        for tri in faces:
            a, b, c = tri[:3] + 1
            f.write(f"f {int(a)} {int(b)} {int(c)}\n")


def write_obj_with_uv(
    path,
    vertices,
    faces,
    uv_coords,
    face_uvs,
    mtl_name,
    material_name="material0",
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)
    uv_coords = np.asarray(uv_coords, dtype=np.float32)
    face_uvs = np.asarray(face_uvs, dtype=np.int64)

    while vertices.ndim > 2:
        vertices = vertices[0]

    while faces.ndim > 2:
        faces = faces[0]

    while uv_coords.ndim > 2:
        uv_coords = uv_coords[0]

    while face_uvs.ndim > 2:
        face_uvs = face_uvs[0]

    with open(path, "w", encoding="utf-8") as f:
        if mtl_name:
            f.write(f"mtllib {mtl_name}\n")

        f.write(f"usemtl {material_name}\n")

        for v in vertices:
            f.write(f"v {float(v[0])} {float(v[1])} {float(v[2])}\n")

        for uv in uv_coords:
            f.write(f"vt {float(uv[0])} {float(uv[1])}\n")

        for tri, tri_uv in zip(faces, face_uvs):
            a, b, c = tri[:3] + 1
            ua, ub, uc = tri_uv[:3] + 1
            f.write(f"f {int(a)}/{int(ua)} {int(b)}/{int(ub)} {int(c)}/{int(uc)}\n")


def write_mtl(path, texture_filename, material_name="material0"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"newmtl {material_name}\n")
        f.write("Ka 1.000 1.000 1.000\n")
        f.write("Kd 1.000 1.000 1.000\n")
        f.write("Ks 0.000 0.000 0.000\n")
        f.write("Ke 0.000 0.000 0.000\n")
        f.write("Ns 10.000\n")
        f.write("d 1.0\n")
        f.write("illum 2\n")
        f.write(f"map_Kd {texture_filename}\n")
