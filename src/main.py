#!/usr/bin/env python3
"""Avatar Pose Pipeline with step control."""

import argparse
from pathlib import Path
from tqdm import tqdm

from bootstrap import ensure_dependencies
from utils import list_images, export_npz_to_obj

from image_resize import ImageResizer
from background_removal import BackgroundRemover
from color_normalization import ColorNormalizer
from pose_estimation import PoseEstimator
from face_landmarks import FaceLandmarkExtractor
from visibility_analysis import VisibilityAnalyzer
from visualization import draw_pose

from smplx_fit.multi_image_optimizer import MultiImageOptimizer
from refinement.mesh_refiner import MeshRefiner
from photometric.texture_projection import TextureProjector
from photometric.photometric_optimizer import PhotometricOptimizer

STAGES = ["preprocess", "smplx", "refine", "texture", "photometric"]


def should_run(stage, only, skip):
    if only:
        return stage in only
    return stage not in skip


def collect_existing_preprocessed(dirs):
    processed_images = sorted(dirs["normalized"].glob("*.png"))
    valid_images, pose_jsons, visibility_jsons = [], [], []
    for img in processed_images:
        pose_json = dirs["pose"] / f"{img.stem}.json"
        visibility_json = dirs["visibility"] / f"{img.stem}.json"
        if pose_json.exists() and visibility_json.exists():
            valid_images.append(img)
            pose_jsons.append(pose_json)
            visibility_jsons.append(visibility_json)
    return valid_images, pose_jsons, visibility_jsons


def preprocess_images(input_dir, dirs, args):
    resizer = ImageResizer(max_resolution=args.max_resolution)
    bg_remover = BackgroundRemover()
    color_normalizer = ColorNormalizer(target_luma=145, target_contrast=52, clahe_clip=1.25, skin_warmth=0.0)
    pose_estimator = PoseEstimator(model_name="yolov8l-pose.pt")
    face_extractor = FaceLandmarkExtractor()
    visibility_analyzer = VisibilityAnalyzer()
    image_paths = list_images(input_dir)
    print(f"✅ Found {len(image_paths)} images\n")
    processed_images, pose_jsons, visibility_jsons = [], [], []
    for img_path in tqdm(image_paths, desc="Processing"):
        try:
            print(f"\n→ {img_path.name}")
            current = resizer.process(img_path, dirs["resized"])
            print("   ✓ resized")
            current = bg_remover.process(current, dirs["no_bg"])
            print("   ✓ background removed")
            current = color_normalizer.process(current, dirs["normalized"])
            print("   ✓ normalized")
            pose_data = pose_estimator.process_image(current, dirs["pose"])
            if not pose_data:
                print("   ⚠ no pose")
                continue
            print("   ✓ pose")
            face_data = face_extractor.process_image(current, dirs["face"])
            if face_data:
                print("   ✓ face")
            pose_json = dirs["pose"] / f"{current.stem}.json"
            visibility_data = visibility_analyzer.process(current, pose_json, dirs["visibility"])
            if not visibility_data:
                print("   ⚠ visibility failed")
                continue
            print("   ✓ visibility")
            draw_pose(current, pose_data, dirs["viz"] / f"{current.stem}_viz.jpg")
            print("   ✓ visualization")
            processed_images.append(current)
            pose_jsons.append(pose_json)
            visibility_jsons.append(dirs["visibility"] / f"{current.stem}.json")
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
    return processed_images, pose_jsons, visibility_jsons


def main():
    parser = argparse.ArgumentParser(description="Avatar Pose Pipeline")
    parser.add_argument("--input", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--max-resolution", type=int, default=2048)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--refine-iterations", type=int, default=1500)
    parser.add_argument("--photo-iterations", type=int, default=2000)
    parser.add_argument("--smplx-image-size", type=int, default=512)
    parser.add_argument("--refine-image-size", type=int, default=512)
    parser.add_argument("--pseudo-weight", type=float, default=0.03)
    parser.add_argument("--only", nargs="+", choices=STAGES, default=None)
    parser.add_argument("--skip", nargs="+", choices=STAGES, default=[])
    parser.add_argument("--no-render-debug", action="store_true")
    parser.add_argument("--debug-every", type=int, default=25)
    args = parser.parse_args()

    input_dir = Path(args.input).resolve()
    output_root = Path(args.output).resolve()
    if not input_dir.exists():
        raise FileNotFoundError(input_dir)
    dirs = {"resized": output_root / "01_resized", "no_bg": output_root / "02_no_background", "normalized": output_root / "03_normalized", "pose": output_root / "04_pose", "face": output_root / "05_face_landmarks", "viz": output_root / "06_visualization", "visibility": output_root / "07_visibility", "smplx": output_root / "08_smplx", "refined": output_root / "09_refined", "texture": output_root / "10_texture", "photometric": output_root / "11_photometric"}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    print("\n🚀 Starting pipeline\n")
    print(f"Input:  {input_dir}")
    print(f"Output: {output_root}")
    print(f"Only:   {args.only}")
    print(f"Skip:   {args.skip}\n")
    ensure_dependencies()
    canonical_output = dirs["smplx"] / "canonical_body.npz"
    refined_output = dirs["refined"] / "refined_body.npz"
    texture_output = dirs["texture"] / "initial_texture.png"
    photometric_output = dirs["photometric"] / "photometric_body.npz"
    if should_run("preprocess", args.only, args.skip):
        processed_images, pose_jsons, visibility_jsons = preprocess_images(input_dir, dirs, args)
    else:
        print("⏭ Skipping preprocessing; collecting existing preprocessed files")
        processed_images, pose_jsons, visibility_jsons = collect_existing_preprocessed(dirs)
    if len(processed_images) == 0:
        print("❌ No valid preprocessed images found")
        return
    print(f"✅ Usable images: {len(processed_images)}")
    if should_run("smplx", args.only, args.skip):
        print("\n🚀 Multi-image SMPL-X optimization\n")
        multi_optimizer = MultiImageOptimizer(model_path="models/smplx", gender="female", image_size=args.smplx_image_size, pseudo_weight=args.pseudo_weight, debug=not args.no_render_debug, debug_every=args.debug_every)
        multi_optimizer.optimize(image_paths=processed_images, pose_json_paths=pose_jsons, visibility_json_paths=visibility_jsons, output_path=canonical_output, iterations=args.iterations)
        export_npz_to_obj(canonical_output, dirs["smplx"] / "canonical_body.obj")
        print("✅ Canonical body created")
    else:
        print("⏭ Skipping SMPL-X optimization")
    if should_run("refine", args.only, args.skip):
        if not canonical_output.exists():
            raise FileNotFoundError(f"Missing canonical body: {canonical_output}")
        print("\n🚀 Freeform refinement\n")
        mesh_refiner = MeshRefiner(image_size=args.refine_image_size)
        mesh_refiner.refine(canonical_body_path=canonical_output, image_paths=processed_images, visibility_paths=visibility_jsons, output_path=refined_output, iterations=args.refine_iterations)
        export_npz_to_obj(refined_output, dirs["refined"] / "refined_body.obj")
        print("✅ Freeform refinement complete")
    else:
        print("⏭ Skipping refinement")
    if should_run("texture", args.only, args.skip):
        print("\n🚀 Creating initial texture\n")
        texture_projector = TextureProjector()
        texture_projector.create_initial_texture(image_paths=processed_images, output_path=texture_output)
        print("✅ Initial texture created")
    else:
        print("⏭ Skipping texture projection")
    if should_run("photometric", args.only, args.skip):
        if not refined_output.exists():
            raise FileNotFoundError(f"Missing refined body: {refined_output}")
        if not texture_output.exists():
            raise FileNotFoundError(f"Missing texture: {texture_output}")
        print("\n🚀 Photometric optimization\n")
        photometric_optimizer = PhotometricOptimizer()
        photometric_optimizer.optimize(refined_mesh_path=refined_output, image_paths=processed_images, visibility_paths=visibility_jsons, texture_path=texture_output, output_path=photometric_output, iterations=args.photo_iterations)
        print("✅ Photometric optimization complete")
    else:
        print("⏭ Skipping photometric optimization")
    print("\n✅ Pipeline finished")
    print(f"📁 Output: {output_root}")
    print(f"🧍 Canonical:   {canonical_output}")
    print(f"🧍 Refined:     {refined_output}")
    print(f"🎨 Texture:     {texture_output}")
    print(f"✨ Photometric: {photometric_output}")


if __name__ == "__main__":
    main()
