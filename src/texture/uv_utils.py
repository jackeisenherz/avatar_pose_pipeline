
from pathlib import Path
import numpy as np

def read_obj_with_uv(path):
    path = Path(path); verts=[]; uvs=[]; faces=[]; face_uvs=[]; mtl_name=None; tex_name=None
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("mtllib "): mtl_name = line.split(maxsplit=1)[1].strip()
            elif line.startswith("v "):
                _,x,y,z = line.strip().split()[:4]; verts.append([float(x),float(y),float(z)])
            elif line.startswith("vt "):
                _,u,v = line.strip().split()[:3]; uvs.append([float(u),float(v)])
            elif line.startswith("f "):
                vs=[]; ts=[]
                for tok in line.strip().split()[1:]:
                    p=tok.split("/"); vs.append(int(p[0])-1); ts.append(int(p[1])-1 if len(p)>1 and p[1] else 0)
                for i in range(1, len(vs)-1):
                    faces.append([vs[0],vs[i],vs[i+1]]); face_uvs.append([ts[0],ts[i],ts[i+1]])
    return np.asarray(verts,np.float32), np.asarray(faces,np.int64), (np.asarray(uvs,np.float32) if uvs else None), (np.asarray(face_uvs,np.int64) if face_uvs else None), mtl_name, tex_name

def write_textured_obj(source_obj, output_obj, output_mtl, texture_filename):
    source_obj=Path(source_obj); output_obj=Path(output_obj); output_mtl=Path(output_mtl); output_obj.parent.mkdir(parents=True, exist_ok=True)
    lines = source_obj.read_text(encoding="utf-8", errors="ignore").splitlines(True)
    with open(output_obj, "w", encoding="utf-8") as f:
        f.write(f"mtllib {output_mtl.name}\n"); wrote=False
        for line in lines:
            if line.startswith("mtllib "): continue
            if line.startswith("usemtl "):
                if not wrote: f.write("usemtl avatarMat\n"); wrote=True
                continue
            f.write(line)
        if not wrote: f.write("usemtl avatarMat\n")
    with open(output_mtl, "w", encoding="utf-8") as f:
        f.write("newmtl avatarMat\nKa 1.000 1.000 1.000\nKd 1.000 1.000 1.000\nKs 0.000 0.000 0.000\n")
        f.write(f"map_Kd {texture_filename}\n")

def project_points(points, cam):
    pts=np.asarray(points,np.float32); t=np.asarray(cam["translation"], np.float32).reshape(1,3); f=np.asarray(cam["focal_length"], np.float32).reshape(-1); c=np.asarray(cam["camera_center"], np.float32).reshape(-1)
    p = pts + t; z = np.clip(p[:,2], 1e-6, None); x = p[:,0]/z*f[0] + c[0]; y = p[:,1]/z*f[1] + c[1]
    return np.stack([x,y,z], axis=1)

def render_vertex_visibility_map(verts, faces, uvs, face_uvs, camera, image_size, texture_size=2048):
    proj = project_points(verts, camera); uv_img = np.zeros((texture_size, texture_size, 3), np.float32); uv_valid = np.zeros((texture_size, texture_size), np.uint8); h,w=texture_size,texture_size
    for fi, tri in enumerate(faces):
        uvtri = uvs[face_uvs[fi]].copy(); uvtri[:,1] = 1.0 - uvtri[:,1]
        px = np.stack([uvtri[:,0]*(w-1), uvtri[:,1]*(h-1)], axis=1).astype(np.float32)
        minx,maxx=max(int(np.floor(px[:,0].min())),0),min(int(np.ceil(px[:,0].max())),w-1)
        miny,maxy=max(int(np.floor(px[:,1].min())),0),min(int(np.ceil(px[:,1].max())),h-1)
        if maxx <= minx or maxy <= miny: continue
        P = proj[tri,:2].astype(np.float32); Z = proj[tri,2].astype(np.float32); A = px
        denom = ((A[1,1]-A[2,1])*(A[0,0]-A[2,0]) + (A[2,0]-A[1,0])*(A[0,1]-A[2,1]))
        if abs(denom) < 1e-8: continue
        for yy in range(miny, maxy+1):
            for xx in range(minx, maxx+1):
                p = np.array([xx+0.5, yy+0.5], np.float32)
                w1 = ((A[1,1]-A[2,1])*(p[0]-A[2,0]) + (A[2,0]-A[1,0])*(p[1]-A[2,1]))/denom
                w2 = ((A[2,1]-A[0,1])*(p[0]-A[2,0]) + (A[0,0]-A[2,0])*(p[1]-A[2,1]))/denom
                w3 = 1.0-w1-w2
                if w1 < 0 or w2 < 0 or w3 < 0: continue
                src_xy = w1*P[0] + w2*P[1] + w3*P[2]; z = w1*Z[0] + w2*Z[1] + w3*Z[2]
                uv_img[yy,xx,0:2] = src_xy; uv_img[yy,xx,2] = z; uv_valid[yy,xx] = 1
    return {"proj_map": uv_img, "valid": uv_valid}

def barycentric_sample_image(uv_vis, img, alpha_mask, erode_px=3):
    import cv2
    proj = uv_vis["proj_map"]; valid = uv_vis["valid"].copy()
    if erode_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_px*2+1, erode_px*2+1)); alpha_mask = cv2.erode(alpha_mask, k, iterations=1)
    h, w = img.shape[:2]; out = np.zeros((proj.shape[0], proj.shape[1], 3), np.uint8); ok = np.zeros((proj.shape[0], proj.shape[1]), np.uint8); fg_ratio = float((alpha_mask > 8).mean())
    for y in range(proj.shape[0]):
        for x in range(proj.shape[1]):
            if not valid[y,x]: continue
            sx, sy = proj[y,x,0], proj[y,x,1]; ix, iy = int(round(sx)), int(round(sy))
            if ix < 0 or iy < 0 or ix >= w or iy >= h: continue
            if alpha_mask[iy, ix] < 8: continue
            out[y,x] = img[iy,ix]; ok[y,x] = 1
    return out, ok, fg_ratio * float(ok.mean())
