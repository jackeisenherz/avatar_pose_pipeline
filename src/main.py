#!/usr/bin/env python3
"""
Avatar Pose Pipeline
"""

import argparse
from pathlib import Path
from tqdm import tqdm

from bootstrap import ensure_dependencies
from utils import list_images

from image_resize import ImageResizer
from background_removal import BackgroundRemover
from color_normalization import ColorNormalizer

from pose_estimation import PoseEstimator
from face_landmarks import FaceLandmarkExtractor
from visibility_analysis import VisibilityAnalyzer

from visualization import draw_pose

from smplx_fit.multi_image_optimizer import MultiImageOptimizer

from photometric.texture_projection import TextureProjector
from photometric.photometric_optimizer import PhotometricOptimizer

from refinement.mesh_refiner import MeshRefiner


def main():

    parser = argparse.ArgumentParser(
        description="Avatar Pose Pipeline"
    )

    parser.add_argument("--input", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)

    parser.add_argument(
        "--max-resolution",
        type=int,
        default=2048
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=1000
    )

    parser.add_argument(
        "--photo-iterations",
        type=int,
        default=2000
    )

    args = parser.parse_args()

    input_dir = Path(args.input).resolve()
    output_root = Path(args.output).resolve()

    print(f"🔍 Input: {input_dir}")

    if not input_dir.exists():
        raise FileNotFoundError(input_dir)

    # =====================================================
    # OUTPUT DIRS
    # =====================================================

    dirs = {
        "resized": output_root / "01_resized",
        "no_bg": output_root / "02_no_background",
        "normalized": output_root / "03_normalized",
        "pose": output_root / "04_pose",
        "face": output_root / "05_face_landmarks",
        "viz": output_root / "06_visualization",
        "visibility": output_root / "07_visibility",
        "smplx": output_root / "08_smplx",
        "refined": output_root / "09_refined",
        "texture": output_root / "10_texture",
        "photometric": output_root / "11_photometric",
    }

    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    print("\n🚀 Starting pipeline...\n")

    ensure_dependencies()

    # =====================================================
    # COMPONENTS
    # =====================================================

    resizer = ImageResizer(
        max_resolution=args.max_resolution
    )

    bg_remover = BackgroundRemover()

    color_normalizer = ColorNormalizer()

    pose_estimator = PoseEstimator(
        model_name="yolov8l-pose.pt"
    )

    face_extractor = FaceLandmarkExtractor()

    visibility_analyzer = VisibilityAnalyzer()

    multi_optimizer = MultiImageOptimizer(
        model_path="models/smplx",
        gender="female",
        image_size=1024
    )

    mesh_refiner = MeshRefiner()

    texture_projector = TextureProjector()

    photometric_optimizer = PhotometricOptimizer()

    # =====================================================
    # INPUT IMAGES
    # =====================================================

    image_paths = list_images(input_dir)

    print(f"✅ Found {len(image_paths)} images\n")

    processed_images = []
    pose_jsons = []
    visibility_jsons = []

    # =====================================================
    # PREPROCESSING
    # =====================================================

    for img_path in tqdm(
        image_paths,
        desc="Processing"
    ):

        try:

            print(f"\n→ {img_path.name}")

            current = img_path

            # ---------------------------------------------
            # Resize
            # ---------------------------------------------

            current = resizer.process(
                current,
                dirs["resized"]
            )

            print("   ✓ resized")

            # ---------------------------------------------
            # Background removal
            # ---------------------------------------------

            current = bg_remover.process(
                current,
                dirs["no_bg"]
            )

            print("   ✓ background removed")

            # ---------------------------------------------
            # Color normalization
            # ---------------------------------------------

            current = color_normalizer.process(
                current,
                dirs["normalized"]
            )

            print("   ✓ normalized")

            # ---------------------------------------------
            # Pose estimation
            # ---------------------------------------------

            pose_data = pose_estimator.process_image(
                current,
                dirs["pose"]
            )

            if not pose_data:
                print("   ⚠ no pose")
                continue

            print("   ✓ pose")

            # ---------------------------------------------
            # Face landmarks
            # ---------------------------------------------

            face_data = face_extractor.process_image(
                current,
                dirs["face"]
            )

            if face_data:
                print("   ✓ face")

            # ---------------------------------------------
            # Visibility metadata
            # ---------------------------------------------

            visibility_data = visibility_analyzer.process(
                current,
                dirs["pose"] / f"{current.stem}.json",
                dirs["visibility"]
            )

            if not visibility_data:
                print("   ⚠ visibility failed")
                continue

            print("   ✓ visibility")

            # ---------------------------------------------
            # Visualization
            # ---------------------------------------------

            viz_path = (
                dirs["viz"] /
                f"{current.stem}_viz.jpg"
            )

            draw_pose(
                current,
                pose_data,
                viz_path
            )

            print("   ✓ visualization")

            # ---------------------------------------------
            # Collect
            # ---------------------------------------------

            processed_images.append(current)

            pose_jsons.append(
                dirs["pose"] /
                f"{current.stem}.json"
            )

            visibility_jsons.append(
                dirs["visibility"] /
                f"{current.stem}.json"
            )

        except Exception as e:

            print(f"❌ Error: {e}")

            import traceback
            traceback.print_exc()

    # =====================================================
    # VALIDATION
    # =====================================================

    if len(processed_images) == 0:

        print("❌ No valid images")
        return

    # =====================================================
    # MULTI-IMAGE SMPL-X OPTIMIZATION
    # =====================================================

    print("\n🚀 Multi-image optimization...\n")

    canonical_output = (
        dirs["smplx"] /
        "canonical_body.npz"
    )

    multi_optimizer.optimize(
        image_paths=processed_images,
        pose_json_paths=pose_jsons,
        visibility_json_paths=visibility_jsons,
        output_path=canonical_output,
        iterations=args.iterations
    )

    print("✅ Canonical body created")

    # =====================================================
    # FREEFORM REFINEMENT
    # =====================================================

    print("\n🚀 Freeform refinement...\n")

    mesh_refiner.refine(
        canonical_body_path=canonical_output,
        image_paths=normalized_images,
        visibility_paths=visibility_jsons,
        output_path=dirs["refined"] / "refined_body.npz"
    )

    print("✅ Freeform refinement complete")

    # =====================================================
    # INITIAL TEXTURE PROJECTION
    # =====================================================

    print("\n🚀 Creating initial texture...\n")

    texture_output = (
        dirs["texture"] /
        "initial_texture.png"
    )

    texture_projector.create_initial_texture(
        image_paths=processed_images,
        output_path=texture_output
    )

    print("✅ Initial texture created")

    # =====================================================
    # PHOTOMETRIC OPTIMIZATION
    # =====================================================

    print("\n🚀 Photometric optimization...\n")

    photometric_output = (
        dirs["photometric"] /
        "photometric_body.npz"
    )

    photometric_optimizer.optimize(
        refined_mesh_path=refined_output,
        image_paths=processed_images,
        visibility_paths=visibility_jsons,
        texture_path=texture_output,
        output_path=photometric_output,
        iterations=args.photo_iterations
    )

    print("\n✅ Pipeline finished\n")

    print(f"📁 Output: {output_root}")

    print(
        f"🧍 Canonical mesh:"
        f"\n{canonical_output}"
    )

    print(
        f"🧍 Refined mesh:"
        f"\n{refined_output}"
    )

    print(
        f"🎨 Texture:"
        f"\n{texture_output}"
    )

    print(
        f"✨ Final avatar:"
        f"\n{photometric_output}"
    )


if __name__ == "__main__":
    main()