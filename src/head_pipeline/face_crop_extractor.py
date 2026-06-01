from pathlib import Path
import json
import math
import cv2
import numpy as np


class FaceCropExtractor:
    """
    Background-guided robust head crop extractor.

    Uses:
    - original full-resolution image for crop generation
    - optional resized alpha mask from body background removal as a PRIOR only
    - multiple detection candidates
    - crop validation
    - rejection of obvious false crops (feet, pillows, paintings)

    Main idea:
    1. detect candidate faces on full-res image
    2. map resized body alpha to full-res and use it as a person prior
    3. reject candidates that are outside the person silhouette
    4. generate multiple crop variants
    5. validate each crop again
    6. keep the best crop

    This avoids using resized alpha for final cropping quality while still benefiting
    from the body mask to suppress background false positives.
    """

    LEFT_EYE_OUTER = 33
    RIGHT_EYE_OUTER = 263
    CHIN = 152
    FOREHEAD = 10
    NOSE_TIP = 1
    LEFT_FACE = 234
    RIGHT_FACE = 454

    def __init__(
        self,
        crop_size=768,
        padding=2.1,
        min_face_px=70,
        debug=True,
        max_candidates=18,
        validation_threshold=0.22,
        use_alpha_prior=True,
        alpha_weight=0.35,
    ):
        self.crop_size = int(crop_size)
        self.padding = float(padding)
        self.min_face_px = int(min_face_px)
        self.debug = bool(debug)
        self.max_candidates = int(max_candidates)
        self.validation_threshold = float(validation_threshold)
        self.use_alpha_prior = bool(use_alpha_prior)
        self.alpha_weight = float(alpha_weight)

        self.mp_face_mesh = None
        self.mp_face_detection = None
        self.haar = None

        try:
            import mediapipe as mp
            self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=6,
                refine_landmarks=True,
                min_detection_confidence=0.25,
            )
            self.mp_face_detection = mp.solutions.face_detection.FaceDetection(
                model_selection=1,
                min_detection_confidence=0.25,
            )
            print("✓ FaceCropExtractor: MediaPipe loaded")
        except Exception as exc:
            print(f"⚠ MediaPipe unavailable: {exc}")

        haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.haar = cv2.CascadeClassifier(haar_path)

    # ---------------------------------------------------------
    # public
    # ---------------------------------------------------------

    def extract_directory(self, input_dir, output_dir, extensions=(".jpg", ".jpeg", ".png", ".webp")):
        input_dir = Path(input_dir)
        image_paths = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in extensions])
        return self.extract_images(image_paths, output_dir)

    def extract_images(self, image_paths, output_dir):
        output_dir = Path(output_dir)
        crop_dir = output_dir / "crops"
        meta_dir = output_dir / "metadata"
        debug_dir = output_dir / "debug"
        crop_dir.mkdir(parents=True, exist_ok=True)
        meta_dir.mkdir(parents=True, exist_ok=True)
        debug_dir.mkdir(parents=True, exist_ok=True)

        results = []
        for image_path in image_paths:
            try:
                r = self.extract_one(image_path, crop_dir, meta_dir, debug_dir)
                if r is not None:
                    results.append(r)
            except Exception as exc:
                print(f"❌ Face crop failed for {Path(image_path).name}: {exc}")
                import traceback
                traceback.print_exc()

        index_path = output_dir / "head_crops_index.json"
        with open(index_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"✓ Extracted {len(results)} head crops")
        print(f"✓ Crop index: {index_path}")
        return results

    def extract_one(self, image_path, crop_dir, meta_dir, debug_dir):
        image_path = Path(image_path)
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Could not load image: {image_path}")

        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        alpha_prior = self._load_alpha_prior(image_path, (h, w))

        base_candidates = []
        base_candidates.extend(self._detect_face_detection_candidates(rgb))
        base_candidates.extend(self._detect_facemesh_candidates(rgb))
        base_candidates.extend(self._detect_haar_candidates(bgr))

        if not base_candidates:
            print(f"⚠ No face detected: {image_path.name}")
            return None

        filtered = []
        for c in base_candidates:
            alpha_score = self._alpha_face_score(c["center"], c["base_size"], alpha_prior)
            c["alpha_score"] = float(alpha_score)
            c["base_quality"] = float(0.75 * c["base_quality"] + self.alpha_weight * alpha_score)
            # Hard reject obvious background detections if alpha prior exists
            if alpha_prior is not None and alpha_score < 0.10:
                continue
            filtered.append(c)

        if not filtered:
            filtered = base_candidates

        crop_candidates = self._generate_crop_candidates(bgr, filtered)
        scored = []

        for candidate in crop_candidates:
            validation = self._validate_crop(candidate["crop"])
            final_score = self._score_candidate(candidate, validation, (h, w), alpha_prior)
            candidate["validation"] = validation
            candidate["final_score"] = float(final_score)
            scored.append(candidate)

        scored.sort(key=lambda x: x["final_score"], reverse=True)
        best = scored[0]

        crop_path = crop_dir / f"{image_path.stem}_head.png"
        meta_path = meta_dir / f"{image_path.stem}_head.json"
        debug_path = debug_dir / f"{image_path.stem}_head_debug.jpg"
        cand_path = debug_dir / f"{image_path.stem}_head_candidates.jpg"

        cv2.imwrite(str(crop_path), best["crop"])

        metadata = {
            "source_image": str(image_path),
            "crop_path": str(crop_path),
            "image_width": int(w),
            "image_height": int(h),
            "center": [float(best["center"][0]), float(best["center"][1])],
            "estimated_face_size_px": float(best["base_size"]),
            "crop_size_px": float(best["crop_size_px"]),
            "padding": float(best["padding"]),
            "bbox_xyxy": [float(v) for v in best["bbox_xyxy"]],
            "rotation_degrees": float(best["angle_deg"]),
            "base_detector": best["base_detector"],
            "base_confidence": float(best["base_confidence"]),
            "alpha_score": float(best.get("alpha_score", 0.0)),
            "face_size_score": float(best["face_size_score"]),
            "centeredness_score": float(best["centeredness_score"]),
            "rotation_score": float(best["rotation_score"]),
            "crop_validation_score": float(best["validation"]["score"]),
            "final_quality_score": float(best["final_score"]),
            "validation": best["validation"],
            "num_candidates": len(scored),
            "top_candidates": [
                {
                    "rank": i,
                    "score": float(c["final_score"]),
                    "validation_score": float(c["validation"]["score"]),
                    "alpha_score": float(c.get("alpha_score", 0.0)),
                    "detector": c["base_detector"],
                    "padding": float(c["padding"]),
                    "angle_deg": float(c["angle_deg"]),
                    "bbox_xyxy": [float(v) for v in c["bbox_xyxy"]],
                }
                for i, c in enumerate(scored[:8])
            ],
        }

        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        if self.debug:
            self._write_debug(bgr, alpha_prior, scored, best, debug_path, cand_path)

        if best["validation"]["score"] < self.validation_threshold:
            print(f"⚠ Weak crop kept for {image_path.name}: {best['validation']['score']:.3f}")

        return {
            "source_image": str(image_path),
            "crop_path": str(crop_path),
            "metadata_path": str(meta_path),
            "debug_path": str(debug_path),
            "candidates_debug_path": str(cand_path),
            "quality_score": float(best["final_score"]),
            "crop_validation_score": float(best["validation"]["score"]),
        }

    # ---------------------------------------------------------
    # alpha prior
    # ---------------------------------------------------------

    def _load_alpha_prior(self, image_path, target_shape):
        if not self.use_alpha_prior:
            return None

        image_path = Path(image_path)
        # expected project layout:
        # data/input/foo.jpg
        # data/output/02_no_background/foo.png
        # try common path pattern by filename
        project_root = None
        parts = image_path.parts
        if "data" in parts:
            idx = parts.index("data")
            project_root = Path(*parts[:idx]) if idx > 0 else Path(".")
        else:
            project_root = image_path.parent.parent.parent

        candidates = []
        if project_root is not None:
            candidates.extend([
                project_root / "data" / "output" / "02_no_background" / f"{image_path.stem}.png",
                project_root / "output" / "02_no_background" / f"{image_path.stem}.png",
            ])

        alpha = None
        for p in candidates:
            if p.exists():
                rgba = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
                if rgba is not None and rgba.ndim == 3 and rgba.shape[2] == 4:
                    alpha = rgba[:, :, 3]
                    break

        if alpha is None:
            return None

        th, tw = target_shape
        alpha = cv2.resize(alpha, (tw, th), interpolation=cv2.INTER_LINEAR)
        alpha = alpha.astype(np.float32) / 255.0
        alpha = cv2.GaussianBlur(alpha, (0, 0), 3.0)
        return alpha

    def _alpha_face_score(self, center, face_size, alpha_prior):
        if alpha_prior is None:
            return 0.5

        h, w = alpha_prior.shape[:2]
        cx, cy = int(round(center[0])), int(round(center[1]))
        r = max(6, int(round(face_size * 0.35)))

        x1 = max(0, cx - r)
        y1 = max(0, cy - r)
        x2 = min(w, cx + r)
        y2 = min(h, cy + r)

        patch = alpha_prior[y1:y2, x1:x2]
        if patch.size == 0:
            return 0.0

        mean_alpha = float(patch.mean())

        # Also test an area slightly below center (head should connect to body)
        y1b = max(0, cy)
        y2b = min(h, cy + 2 * r)
        patch_below = alpha_prior[y1b:y2b, x1:x2]
        below_alpha = float(patch_below.mean()) if patch_below.size > 0 else 0.0

        return float(np.clip(0.6 * mean_alpha + 0.4 * below_alpha, 0.0, 1.0))

    # ---------------------------------------------------------
    # detection
    # ---------------------------------------------------------

    def _detect_face_detection_candidates(self, rgb):
        if self.mp_face_detection is None:
            return []
        h, w = rgb.shape[:2]
        result = self.mp_face_detection.process(rgb)
        candidates = []
        if not result.detections:
            return candidates

        for idx, det in enumerate(result.detections):
            bbox = det.location_data.relative_bounding_box
            x = bbox.xmin * w
            y = bbox.ymin * h
            bw = bbox.width * w
            bh = bbox.height * h
            size = max(bw, bh)
            if size < self.min_face_px:
                continue
            center = np.array([x + 0.5 * bw, y + 0.5 * bh], dtype=np.float32)
            score = float(det.score[0]) if det.score else 0.75
            candidates.append({
                "base_detector": "mediapipe_detection",
                "face_index": int(idx),
                "center": center,
                "base_size": float(size),
                "angle_deg": 0.0,
                "base_confidence": score,
                "base_quality": 0.5 * score + 0.3 * self._face_size_score(size, (h, w)) + 0.2 * self._centeredness_score(center, (h, w)),
                "landmarks": [],
            })
        return candidates

    def _detect_facemesh_candidates(self, rgb):
        if self.mp_face_mesh is None:
            return []

        h, w = rgb.shape[:2]
        result = self.mp_face_mesh.process(rgb)
        if not result.multi_face_landmarks:
            return []

        out = []
        for face_index, face_landmarks in enumerate(result.multi_face_landmarks):
            pts = np.array([[lm.x * w, lm.y * h, lm.z] for lm in face_landmarks.landmark], dtype=np.float32)
            xs = pts[:, 0]
            ys = pts[:, 1]
            bbox_w = float(xs.max() - xs.min())
            bbox_h = float(ys.max() - ys.min())
            size = max(bbox_w, bbox_h)
            if size < self.min_face_px:
                continue
            left_eye = pts[self.LEFT_EYE_OUTER, :2]
            right_eye = pts[self.RIGHT_EYE_OUTER, :2]
            angle = math.degrees(math.atan2(float((right_eye-left_eye)[1]), float((right_eye-left_eye)[0])))
            center_ids = [self.NOSE_TIP, self.CHIN, self.FOREHEAD, self.LEFT_FACE, self.RIGHT_FACE]
            center = pts[center_ids, :2].mean(axis=0)
            conf = 0.85
            out.append({
                "base_detector": "mediapipe_facemesh",
                "face_index": int(face_index),
                "center": center.astype(np.float32),
                "base_size": float(size),
                "angle_deg": float(angle),
                "base_confidence": float(conf),
                "base_quality": 0.4 * conf + 0.3 * self._face_size_score(size, (h, w)) + 0.3 * self._centeredness_score(center, (h, w)),
                "landmarks": [],
            })
        return out

    def _detect_haar_candidates(self, bgr):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        detections = self.haar.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=3, minSize=(self.min_face_px, self.min_face_px))
        out = []
        for idx, (x, y, fw, fh) in enumerate(detections):
            size = float(max(fw, fh))
            center = np.array([x + 0.5 * fw, y + 0.5 * fh], dtype=np.float32)
            conf = 0.50
            out.append({
                "base_detector": "opencv_haar",
                "face_index": int(idx),
                "center": center,
                "base_size": size,
                "angle_deg": 0.0,
                "base_confidence": conf,
                "base_quality": 0.4 * conf + 0.3 * self._face_size_score(size, (h, w)) + 0.3 * self._centeredness_score(center, (h, w)),
                "landmarks": [],
            })
        return out

    # ---------------------------------------------------------
    # candidates and validation
    # ---------------------------------------------------------

    def _generate_crop_candidates(self, image, base_candidates):
        padding_values = [self.padding, self.padding * 0.9, self.padding * 1.1]
        angle_offsets = [0.0, -5.0, 5.0]
        variants = []

        base_candidates = sorted(base_candidates, key=lambda c: c["base_quality"], reverse=True)[: self.max_candidates]

        for base in base_candidates:
            for pad in padding_values:
                for da in angle_offsets:
                    angle = float(base["angle_deg"]) + da
                    crop_size_px = float(base["base_size"]) * float(pad)
                    crop, M, bbox = self._aligned_crop(image, base["center"], crop_size_px, angle, self.crop_size)
                    variants.append({**base, "padding": float(pad), "crop_size_px": float(crop_size_px), "angle_deg": float(angle), "crop": crop, "transform": M, "bbox_xyxy": bbox})
        return variants

    def _aligned_crop(self, image, center, size, angle_deg, output_size):
        center = np.asarray(center, dtype=np.float32)
        scale = output_size / float(size)
        M = cv2.getRotationMatrix2D((float(center[0]), float(center[1])), float(angle_deg), scale)
        M[0, 2] += output_size * 0.5 - center[0]
        M[1, 2] += output_size * 0.5 - center[1]
        crop = cv2.warpAffine(image, M, (output_size, output_size), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)
        half = 0.5 * float(size)
        bbox = [float(center[0] - half), float(center[1] - half), float(center[0] + half), float(center[1] + half)]
        return crop, M, bbox

    def _validate_crop(self, crop_bgr):
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        h, w = crop_bgr.shape[:2]
        vals = []

        if self.mp_face_detection is not None:
            r = self.mp_face_detection.process(crop_rgb)
            if r.detections:
                best_score = 0.0
                best_item = None
                for d in r.detections:
                    bbox = d.location_data.relative_bounding_box
                    x = bbox.xmin * w
                    y = bbox.ymin * h
                    bw = bbox.width * w
                    bh = bbox.height * h
                    size = max(bw, bh)
                    center = np.array([x + 0.5 * bw, y + 0.5 * bh], dtype=np.float32)
                    s = float(d.score[0]) if d.score else 0.75
                    item_score = 0.45 * s + 0.30 * self._crop_face_size_score(size, (h, w)) + 0.25 * self._centeredness_score(center, (h, w))
                    if item_score > best_score:
                        best_score = item_score
                        best_item = {
                            "score": float(item_score),
                            "method": "mediapipe_detection",
                            "face_present": True,
                            "face_centeredness": float(self._centeredness_score(center, (h, w))),
                            "face_size_score": float(self._crop_face_size_score(size, (h, w))),
                            "rotation_score": 0.8,
                        }
                if best_item is not None:
                    vals.append(best_item)

        detections = self.haar.detectMultiScale(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY), scaleFactor=1.08, minNeighbors=3, minSize=(30, 30))
        if len(detections) > 0:
            best_score = 0.0
            best_item = None
            for x, y, fw, fh in detections:
                size = float(max(fw, fh))
                center = np.array([x + 0.5 * fw, y + 0.5 * fh], dtype=np.float32)
                item_score = 0.25 * 0.55 + 0.40 * self._crop_face_size_score(size, crop_bgr.shape[:2]) + 0.35 * self._centeredness_score(center, crop_bgr.shape[:2])
                if item_score > best_score:
                    best_score = item_score
                    best_item = {
                        "score": float(item_score),
                        "method": "opencv_haar",
                        "face_present": True,
                        "face_centeredness": float(self._centeredness_score(center, crop_bgr.shape[:2])),
                        "face_size_score": float(self._crop_face_size_score(size, crop_bgr.shape[:2])),
                        "rotation_score": 0.65,
                    }
            if best_item is not None:
                vals.append(best_item)

        if not vals:
            return {"score": 0.0, "method": "none", "face_present": False, "face_centeredness": 0.0, "face_size_score": 0.0, "rotation_score": 0.0}
        vals.sort(key=lambda v: v["score"], reverse=True)
        return vals[0]

    # ---------------------------------------------------------
    # scoring helpers
    # ---------------------------------------------------------

    def _score_candidate(self, candidate, validation, image_shape, alpha_prior):
        face_size_score = self._face_size_score(candidate["base_size"], image_shape)
        centeredness_score = self._centeredness_score(candidate["center"], image_shape)
        rotation_score = self._rotation_score(candidate["angle_deg"])
        validation_score = float(validation.get("score", 0.0))
        detector_conf = float(candidate.get("base_confidence", 0.0))
        alpha_score = self._alpha_face_score(candidate["center"], candidate["base_size"], alpha_prior)

        candidate["face_size_score"] = float(face_size_score)
        candidate["centeredness_score"] = float(centeredness_score)
        candidate["rotation_score"] = float(rotation_score)

        score = (
            0.22 * detector_conf +
            0.18 * face_size_score +
            0.12 * centeredness_score +
            0.30 * validation_score +
            0.08 * rotation_score +
            0.10 * alpha_score
        )
        return score

    def _face_size_score(self, size_px, image_shape):
        h, w = image_shape
        rel = float(size_px) / float(min(h, w))
        score = (rel - 0.04) / (0.25 - 0.04)
        return float(np.clip(score, 0.0, 1.0))

    def _crop_face_size_score(self, size_px, crop_shape):
        h, w = crop_shape
        rel = float(size_px) / float(min(h, w))
        center = 0.42
        width = 0.32
        score = 1.0 - abs(rel - center) / width
        return float(np.clip(score, 0.0, 1.0))

    def _centeredness_score(self, center, image_shape):
        h, w = image_shape
        cx, cy = float(center[0]), float(center[1])
        dx = abs(cx - 0.5 * w) / max(1.0, 0.5 * w)
        dy = abs(cy - 0.5 * h) / max(1.0, 0.5 * h)
        return float(np.clip(1.0 - 0.65 * dx - 0.35 * dy, 0.0, 1.0))

    def _rotation_score(self, angle_deg):
        return float(np.clip(1.0 - abs(float(angle_deg)) / 35.0, 0.0, 1.0))

    # ---------------------------------------------------------
    # debug
    # ---------------------------------------------------------

    def _write_debug(self, bgr, alpha_prior, candidates, best, debug_path, candidates_debug_path):
        dbg = bgr.copy()
        if alpha_prior is not None:
            alpha_vis = (np.clip(alpha_prior, 0, 1) * 255).astype(np.uint8)
            alpha_vis = cv2.applyColorMap(alpha_vis, cv2.COLORMAP_VIRIDIS)
            dbg = cv2.addWeighted(dbg, 0.75, alpha_vis, 0.25, 0)

        for rank, c in enumerate(candidates[:8]):
            x1, y1, x2, y2 = [int(round(v)) for v in c["bbox_xyxy"]]
            color = (0, 255, 0) if c is best else (0, 180, 255)
            cv2.rectangle(dbg, (x1, y1), (x2, y2), color, 2)
            label = f"#{rank} s={c['final_score']:.2f} v={c['validation']['score']:.2f} a={c.get('alpha_score',0):.2f}"
            cv2.putText(dbg, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

        cv2.imwrite(str(debug_path), dbg)

        tiles = []
        for rank, c in enumerate(candidates[:6]):
            crop = cv2.resize(c["crop"], (256, 256))
            txt = f"#{rank} s={c['final_score']:.3f} v={c['validation']['score']:.3f}"
            cv2.putText(crop, txt, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 2, cv2.LINE_AA)
            tiles.append(crop)

        if tiles:
            rows = []
            for i in range(0, len(tiles), 3):
                row = tiles[i:i+3]
                while len(row) < 3:
                    row.append(np.zeros_like(tiles[0]))
                rows.append(np.concatenate(row, axis=1))
            panel = np.concatenate(rows, axis=0)
            cv2.imwrite(str(candidates_debug_path), panel)
