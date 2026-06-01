#!/usr/bin/env python3
"""Avatar Pose Pipeline with step control + standalone head reconstruction + fusion."""

import argparse
import os
from pathlib import Path

from bootstrap import ensure_dependencies


# Runtime imports are intentionally delayed until after ensure_dependencies().
# This prevents missing packages such as mediapipe from crashing before
# bootstrap can install/check dependencies.
tqdm = None
list_images = None
export_npz_to_obj = None

ImageResizer = None
BackgroundRemover = None
ColorNormalizer = None
PoseEstimator = None
FaceLandmarkExtractor = None
VisibilityAnalyzer = None
draw_pose = None

MultiImageOptimizer = None
MeshRefiner = None
TextureProjector = None
PhotometricOptimizer = None

BodyTextureFusion = None
BodyTextureFusionConfig = None
FaceCropExtractor = None
FaceReconstructionRunner = None
HeadFusion = None
HeadBodyFusion = None
HeadBodyFusionConfig = None
extract_measurements = None


def load_runtime_imports():
    global tqdm, list_images, export_npz_to_obj
    global ImageResizer, BackgroundRemover, ColorNormalizer
    global PoseEstimator, FaceLandmarkExtractor, VisibilityAnalyzer, draw_pose
    global MultiImageOptimizer, MeshRefiner, TextureProjector, PhotometricOptimizer
    global BodyTextureFusion, BodyTextureFusionConfig
    global FaceCropExtractor, FaceReconstructionRunner, HeadFusion
    global HeadBodyFusion, HeadBodyFusionConfig
    global extract_measurements

    from tqdm import tqdm as _tqdm
    from utils import list_images as _list_images, export_npz_to_obj as _export_npz_to_obj

    from image_resize import ImageResizer as _ImageResizer
    from background_removal import BackgroundRemover as _BackgroundRemover
    from color_normalization import ColorNormalizer as _ColorNormalizer
    from pose_estimation import PoseEstimator as _PoseEstimator
    from face_landmarks import FaceLandmarkExtractor as _FaceLandmarkExtractor
    from visibility_analysis import VisibilityAnalyzer as _VisibilityAnalyzer
    from visualization import draw_pose as _draw_pose

    from smplx_fit.multi_image_optimizer import MultiImageOptimizer as _MultiImageOptimizer
    from refinement.mesh_refiner import MeshRefiner as _MeshRefiner
    from photometric.texture_projection import TextureProjector as _TextureProjector
    from photometric.photometric_optimizer import PhotometricOptimizer as _PhotometricOptimizer

    tqdm = _tqdm
    list_images = _list_images
    export_npz_to_obj = _export_npz_to_obj

    ImageResizer = _ImageResizer
    BackgroundRemover = _BackgroundRemover
    ColorNormalizer = _ColorNormalizer
    PoseEstimator = _PoseEstimator
    FaceLandmarkExtractor = _FaceLandmarkExtractor
    VisibilityAnalyzer = _VisibilityAnalyzer
    draw_pose = _draw_pose

    MultiImageOptimizer = _MultiImageOptimizer
    MeshRefiner = _MeshRefiner
    TextureProjector = _TextureProjector
    PhotometricOptimizer = _PhotometricOptimizer

    try:
        from texturing.body_texture_fusion import BodyTextureFusion as _BodyTextureFusion
        from texturing.body_texture_fusion import BodyTextureFusionConfig as _BodyTextureFusionConfig
        BodyTextureFusion = _BodyTextureFusion
        BodyTextureFusionConfig = _BodyTextureFusionConfig
    except Exception:
        BodyTextureFusion = None
        BodyTextureFusionConfig = None

    try:
        from head_pipeline.face_crop_extractor import FaceCropExtractor as _FaceCropExtractor
        from head_pipeline.face_reconstruction_runner import FaceReconstructionRunner as _FaceReconstructionRunner
        from head_pipeline.head_fusion import HeadFusion as _HeadFusion
        FaceCropExtractor = _FaceCropExtractor
        FaceReconstructionRunner = _FaceReconstructionRunner
        HeadFusion = _HeadFusion
    except Exception:
        FaceCropExtractor = None
        FaceReconstructionRunner = None
        HeadFusion = None

    try:
        from combine.head_body_fusion import HeadBodyFusion as _HeadBodyFusion
        from combine.head_body_fusion import HeadBodyFusionConfig as _HeadBodyFusionConfig
        HeadBodyFusion = _HeadBodyFusion
        HeadBodyFusionConfig = _HeadBodyFusionConfig
    except Exception:
        HeadBodyFusion = None
        HeadBodyFusionConfig = None

    try:
        from measurements.extract_body_measurements import extract_measurements as _extract_measurements
        extract_measurements = _extract_measurements
    except Exception:
        extract_measurements = None


STAGES = [
    "preprocess",
    "smplx",
    "refine",
    "measure",
    "head",
    "combine",
    "texture",
    "photometric",
]


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
    color_normalizer = ColorNormalizer(
        target_luma=145,
        target_contrast=52,
        clahe_clip=1.25,
        skin_warmth=0.0,
    )
    pose_estimator = PoseEstimator(
        backend=args.pose_backend,
        device=args.pose_device,
        vitpose_model=args.vitpose_model,
        detector_model=args.pose_detector_model,
        yolo_model=args.pose_yolo_model,
        detector_threshold=args.pose_detector_threshold,
        debug=not args.no_pose_debug,
        fallback_to_yolo=not args.no_pose_yolo_fallback,
    )
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
            print("   ✓ face" if face_data else "   ⚠ no face")

            pose_json = dirs["pose"] / f"{current.stem}.json"

            visibility_data = visibility_analyzer.process(
                current,
                pose_json,
                dirs["visibility"],
            )

            if not visibility_data:
                print("   ⚠ visibility failed")
                continue

            print("   ✓ visibility")

            draw_pose(
                current,
                pose_data,
                dirs["viz"] / f"{current.stem}_viz.jpg",
            )

            print("   ✓ visualization")

            processed_images.append(current)
            pose_jsons.append(pose_json)
            visibility_jsons.append(dirs["visibility"] / f"{current.stem}.json")

        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()

    return processed_images, pose_jsons, visibility_jsons


def default_head_recon_command():
    env_cmd = os.environ.get("HEAD_RECON_COMMAND", "").strip()

    if env_cmd:
        return env_cmd

    deca_demo = Path("external/DECA/demos/demo_reconstruct.py")

    if deca_demo.exists():
        return (
            f"python {deca_demo} "
            "-i {input_dir} "
            "-s {output_dir} "
            "--iscrop True "
            "--saveObj True "
            "--saveMat True "
            "--useTex True "
            "--extractTex True "
            "--saveImages True "
            "--saveDepth True "
            "--saveKpt True "
            "--rasterizer_type=pytorch3d"
        )

    emoca_demo = Path("external/emoca/demos/test_emoca_on_images.py")

    if emoca_demo.exists():
        return (
            f"python {emoca_demo} "
            "--input_folder {input_dir} "
            "--output_folder {output_dir}"
        )

    return None


def run_head_stage(input_dir, dirs, args):
    """
    Head stage:
      - crop extraction
      - standalone per-crop reconstruction
      - optional fusion into one canonical head OBJ/NPZ

    It does NOT attach/register the head to SMPL-X.
    That belongs to the future combine stage.
    """

    if FaceCropExtractor is None or FaceReconstructionRunner is None:
        raise ImportError(
            "Missing src/head_pipeline package. Copy head_pipeline/ into src/head_pipeline/ first."
        )

    head_dir = dirs["head"]
    crop_index = head_dir / "head_crops_index.json"
    recon_dir = head_dir / "recon"
    fused_dir = head_dir / "fused"

    if not args.head_skip_crop:
        print("\n🚀 Head extraction\n")

        cropper = FaceCropExtractor(
            crop_size=args.head_crop_size,
            padding=args.head_padding,
            debug=True,
        )

        crop_results = cropper.extract_directory(
            input_dir=input_dir,
            output_dir=head_dir,
        )

        if len(crop_results) == 0:
            print("⚠ No head crops extracted; skipping head reconstruction/fusion")
            return
    else:
        print("⏭ Skipping head crop extraction")
        if not crop_index.exists():
            raise FileNotFoundError(f"Missing crop index: {crop_index}")

    recon_result = None

    if args.head_reconstruct:
        recon_command = args.head_recon_command or default_head_recon_command()

        if not recon_command:
            raise RuntimeError(
                "Head reconstruction requested, but no reconstruction command was found.\n"
                "Either install DECA at external/DECA, pass --head-recon-command, "
                "or set HEAD_RECON_COMMAND."
            )

        if "--saveMat" not in recon_command and "DECA" in recon_command:
            print("⚠ DECA command does not include --saveMat True.")
            print("⚠ Fusion needs MAT files. The command will reconstruct OBJs but fusion may fail.")

        print("\n🚀 Head reconstruction\n")

        runner = FaceReconstructionRunner(
            command_template=recon_command,
        )

        recon_result = runner.run_from_crop_index(
            crop_index_path=crop_index,
            output_dir=recon_dir,
        )

        obj_files = recon_result.get("obj_files", []) if recon_result else []
        mat_files = recon_result.get("mat_files", []) if recon_result else []
        npz_files = recon_result.get("npz_files", []) if recon_result else []

        print("✅ Head per-crop reconstruction complete")
        print(f"🙂 Per-crop OBJ files: {len(obj_files)}")
        print(f"📄 MAT parameter files: {len(mat_files)}")
        print(f"📄 NPZ parameter files: {len(npz_files)}")

    else:
        print("⏭ Head reconstruction disabled. Use --head-reconstruct or --head-full.")

    if args.head_fuse:
        if HeadFusion is None:
            raise ImportError(
                "Missing src/head_pipeline/head_fusion.py. Install the head_fusion_update files first."
            )

        print("\n🚀 Fusing canonical head\n")

        fusion = HeadFusion(
            deca_root=args.deca_root,
            device=args.head_fusion_device,
            top_k=args.head_fusion_top_k,
            min_score=args.head_fusion_min_score,
            texture_mode=args.head_texture_mode,
            texture_top_k=args.head_texture_top_k,
            texture_min_quality=args.head_texture_min_quality,
            texture_average_min_quality=args.head_texture_average_min_quality,
        )

        summary = fusion.fuse_from_recon_dir(
            recon_dir=recon_dir,
            crop_index_path=crop_index,
            output_dir=fused_dir,
        )

        print("✅ Canonical head fusion complete")
        print(f"📁 Fused head folder: {fused_dir}")

        obj_path = summary.get("mesh", {}).get("obj_path")
        params_path = summary.get("params_path")

        if obj_path:
            print(f"🙂 Canonical head OBJ: {obj_path}")

        if params_path:
            print(f"📄 Canonical head NPZ: {params_path}")

    else:
        print("⏭ Head fusion disabled. Use --head-fuse or --head-full.")

    print("✅ Head stage complete")
    print(f"📁 Head output: {head_dir}")


def find_default_head_obj(dirs):
    fused = dirs["head"] / "fused"
    candidates = [
        fused / "canonical_fused_head.obj",
        fused / "canonical_fused_head_fallback_best.obj",
        fused / "canonical_fused_head_obj_average.obj",
        fused / "canonical_fused_head_textured.obj",
    ]
    for p in candidates:
        if p.exists():
            return p
    found = sorted(fused.glob("*.obj"))
    return found[0] if found else None


def choose_body_for_combine(dirs, args):
    canonical_output = dirs["smplx"] / "canonical_body.npz"
    refined_output = dirs["refined"] / "refined_body.npz"
    if args.combine_body_source == "canonical":
        return canonical_output
    if args.combine_body_source == "refined":
        return refined_output
    return refined_output if refined_output.exists() else canonical_output


def run_combine_stage(dirs, args):
    if HeadBodyFusion is None or HeadBodyFusionConfig is None:
        raise ImportError("Missing src/combine package. Copy combine/ into src/combine/ first.")

    body_path = Path(args.combine_body_path) if args.combine_body_path else choose_body_for_combine(dirs, args)
    head_path = Path(args.combine_head_obj) if args.combine_head_obj else find_default_head_obj(dirs)

    if not body_path.exists():
        raise FileNotFoundError(f"Missing body mesh for combine: {body_path}")
    if head_path is None or not head_path.exists():
        raise FileNotFoundError("Missing fused head OBJ. Run the head stage first or pass --combine-head-obj.")

    combined_dir = dirs["combined"]
    combined_dir.mkdir(parents=True, exist_ok=True)

    output_npz = combined_dir / f"{args.combine_output_name}.npz"
    output_obj = combined_dir / f"{args.combine_output_name}.obj"
    summary_path = combined_dir / f"{args.combine_output_name}_summary.json"

    print("\n🚀 Combining SMPL-X body + fused head with auto-registration\n")
    print(f"Body: {body_path}")
    print(f"Head: {head_path}")
    print(f"Output OBJ: {output_obj}")

    config = HeadBodyFusionConfig(
        vertical_axis=args.combine_vertical_axis,
        neck_cutoff_ratio=args.combine_neck_cutoff_ratio,
        neck_overlap_ratio=args.combine_neck_overlap_ratio,
        remove_body_head=not args.combine_keep_body_head,
        head_remove_lateral_ratio=args.combine_head_remove_lateral_ratio,
        head_remove_forward_ratio=args.combine_head_remove_forward_ratio,
        clip_head_neck=args.combine_clip_head_neck,
        head_clip_bottom_ratio=args.combine_head_clip_bottom_ratio,
        register_head=not args.combine_no_register_head,
        registration_iterations=args.combine_registration_iterations,
        registration_sample_points=args.combine_registration_sample_points,
        registration_trim_fraction=args.combine_registration_trim_fraction,
        registration_scale_delta_min=args.combine_registration_scale_delta_min,
        registration_scale_delta_max=args.combine_registration_scale_delta_max,
        registration_absolute_scale_min=args.combine_registration_absolute_scale_min,
        registration_absolute_scale_max=args.combine_registration_absolute_scale_max,
        registration_target_scale_shrink=args.combine_registration_target_scale_shrink,
        registration_allow_rotation=args.combine_registration_allow_rotation,
        registration_forward_icp_weight=args.combine_registration_forward_icp_weight,
        head_scale=args.combine_head_scale,
        head_vertical_offset=args.combine_head_vertical_offset,
        head_lateral_offset=args.combine_head_lateral_offset,
        head_forward_offset=args.combine_head_forward_offset,
        connect_neck=args.combine_connect_neck,
        debug=not args.no_render_debug,
    )

    fusion = HeadBodyFusion(config=config)
    summary = fusion.fuse(
        body_npz_path=body_path,
        head_obj_path=head_path,
        output_npz_path=output_npz,
        output_obj_path=output_obj,
        summary_path=summary_path,
    )

    print("✅ Combine stage complete")
    print(f"🔗 Combined OBJ: {output_obj}")
    print(f"📄 Combined NPZ: {output_npz}")
    print(f"🧾 Summary:      {summary_path}")
    return summary


def collect_texture_inputs(dirs, processed_images, args):
    """
    Collect source images, alpha masks and per-image camera JSONs for UV texture fusion.

    The camera JSON files are exported by the updated MultiImageOptimizer:
      data/output/08_smplx/camera/<image_stem>_camera.json
    """

    image_paths = list(processed_images or [])

    if not image_paths:
        image_paths = sorted(dirs["normalized"].glob("*.png"))

    if not image_paths:
        raise FileNotFoundError(
            "No normalized input images found for texture fusion. "
            "Run preprocessing first or do not run texture in isolation."
        )

    alpha_paths = []
    camera_paths = []

    camera_dir = dirs["smplx"] / "camera"

    for img_path in image_paths:
        img_path = Path(img_path)
        stem = img_path.stem

        alpha_path = dirs["no_bg"] / f"{stem}.png"
        camera_path = camera_dir / f"{stem}_camera.json"

        alpha_paths.append(alpha_path)
        camera_paths.append(camera_path)

    missing_alpha = [p for p in alpha_paths if not p.exists()]
    missing_camera = [p for p in camera_paths if not p.exists()]

    if missing_alpha:
        raise FileNotFoundError(
            "Missing background-removed alpha image needed for UV texture fusion: "
            f"{missing_alpha[0]}"
        )

    if missing_camera:
        raise FileNotFoundError(
            "Missing per-image camera JSON needed for UV texture fusion: "
            f"{missing_camera[0]}\n"
            "Rerun the SMPL-X stage with the updated multi_image_optimizer.py that exports camera JSON files."
        )

    return image_paths, alpha_paths, camera_paths


def find_texture_mesh_obj(dirs, args):
    if args.texture_mesh_obj:
        mesh_obj = Path(args.texture_mesh_obj)
    else:
        mesh_obj = dirs["combined"] / f"{args.combine_output_name}.obj"

    if not mesh_obj.exists():
        raise FileNotFoundError(
            f"Missing combined OBJ for UV texture fusion: {mesh_obj}\n"
            "Run --only combine first, or pass --texture-mesh-obj."
        )

    return mesh_obj


def run_uv_texture_stage(dirs, processed_images, args):
    """
    Multi-view UV texture stitching.

    This replaces the old single initial texture projection when --texture-mode uv.
    It uses the combined body/head OBJ, per-image camera JSONs, normalized photos,
    and background-removed alpha masks.
    """

    if BodyTextureFusion is None or BodyTextureFusionConfig is None:
        raise ImportError(
            "Missing src/texturing package. Copy the body_texture_fusion_update files into src/texturing/ first."
        )

    mesh_obj = find_texture_mesh_obj(dirs, args)
    image_paths, alpha_paths, camera_paths = collect_texture_inputs(
        dirs=dirs,
        processed_images=processed_images,
        args=args,
    )

    texture_dir = dirs["texture"]
    texture_dir.mkdir(parents=True, exist_ok=True)

    print("\n🚀 Multi-view UV texture fusion\n")
    print(f"Mesh:    {mesh_obj}")
    print(f"Images:  {len(image_paths)}")
    print(f"Output:  {texture_dir}")

    fuser = BodyTextureFusion(
        BodyTextureFusionConfig(
            texture_size=args.texture_size,
            blend_mode=args.texture_blend_mode,
            erode_mask_px=args.texture_erode_mask_px,
            feather_px=args.texture_feather_px,
            inpaint_radius=args.texture_inpaint_radius,
            save_debug=not args.no_render_debug,
        )
    )

    summary = fuser.fuse(
        mesh_obj_path=mesh_obj,
        image_paths=image_paths,
        alpha_paths=alpha_paths,
        camera_json_paths=camera_paths,
        output_dir=texture_dir,
    )

    print("✅ UV texture fusion complete")
    print(f"🎨 Texture:      {texture_dir / 'canonical_body_texture.png'}")
    print(f"🧵 Textured OBJ: {texture_dir / 'textured_avatar.obj'}")
    print(f"🧾 Summary:      {texture_dir / 'texture_summary.json'}")

    return summary


def choose_body_for_measurement(dirs, args):
    canonical_output = dirs["smplx"] / "canonical_body.npz"
    refined_output = dirs["refined"] / "refined_body.npz"
    combined_output = dirs["combined"] / f"{args.combine_output_name}.npz"

    if args.measurement_body_path:
        return Path(args.measurement_body_path)

    if args.measurement_body_source == "canonical":
        return canonical_output

    if args.measurement_body_source == "refined":
        return refined_output

    if args.measurement_body_source == "combined":
        return combined_output

    if refined_output.exists():
        return refined_output

    if canonical_output.exists():
        return canonical_output

    return combined_output


def run_measurement_stage(dirs, args):
    if extract_measurements is None:
        raise ImportError(
            "Missing src/measurements/extract_body_measurements.py. "
            "Copy the measurement extractor into src/measurements/ first."
        )

    if args.height_cm is None:
        raise ValueError(
            "The measurement stage requires --height-cm, for example: --height-cm 172"
        )

    body_path = choose_body_for_measurement(dirs, args)

    if not body_path.exists():
        raise FileNotFoundError(f"Missing body NPZ for measurement: {body_path}")

    output_path = (
        Path(args.measurement_output)
        if args.measurement_output
        else dirs["measurements"] / "body_measurements.json"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("\n🚀 Extracting body measurements\n")
    print(f"Body NPZ:  {body_path}")
    print(f"Height cm: {args.height_cm}")
    print(f"Output:    {output_path}")

    measurements = extract_measurements(
        npz_path=body_path,
        height_cm=args.height_cm,
        density_g_per_cm3=args.measurement_density_g_per_cm3,
        flip_left_right=args.measurement_flip_left_right,
    )

    import json

    with open(output_path, "w") as f:
        json.dump(measurements, f, indent=2)

    print("✅ Measurement extraction complete")
    print(f"📏 Measurements: {output_path}")

    return measurements


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

    # Pose controls.
    # Default uses Hugging Face Transformers ViTPose without MMCV/MMPose.
    parser.add_argument(
        "--pose-backend",
        choices=["vitpose", "yolo", "auto"],
        default="vitpose",
        help="Pose backend. Default: vitpose. Uses YOLO only as fallback or if selected.",
    )
    parser.add_argument(
        "--vitpose-model",
        type=str,
        default="usyd-community/vitpose-plus-base",
        help="Hugging Face ViTPose / ViTPose++ model. Default: usyd-community/vitpose-plus-base.",
    )
    parser.add_argument(
        "--pose-detector-model",
        type=str,
        default="PekingU/rtdetr_r50vd_coco_o365",
        help="Hugging Face RT-DETR person detector used before ViTPose.",
    )
    parser.add_argument(
        "--pose-yolo-model",
        type=str,
        default="yolo11x-pose.pt",
        help="Ultralytics YOLO pose model used only when --pose-backend yolo or as fallback.",
    )
    parser.add_argument(
        "--pose-device",
        type=str,
        default=None,
        help="Pose device, e.g. cuda, cuda:0, or cpu. Default: auto.",
    )
    parser.add_argument(
        "--pose-detector-threshold",
        type=float,
        default=0.25,
        help="RT-DETR person detector confidence threshold.",
    )
    parser.add_argument(
        "--no-pose-debug",
        action="store_true",
        help="Disable pose debug overlays in data/output/04_pose/_debug_pose.",
    )
    parser.add_argument(
        "--no-pose-yolo-fallback",
        action="store_true",
        help="Disable YOLO fallback if ViTPose inference fails.",
    )

    # Head controls.
    parser.add_argument(
        "--head",
        action="store_true",
        help="Run isolated head crop extraction only.",
    )

    parser.add_argument(
        "--head-full",
        action="store_true",
        help="Run isolated head crop extraction + reconstruction + fusion. Does not attach to SMPL-X.",
    )

    parser.add_argument("--head-skip-crop", action="store_true")
    parser.add_argument("--head-reconstruct", action="store_true")
    parser.add_argument("--head-fuse", action="store_true")
    parser.add_argument("--head-crop-size", type=int, default=768)
    parser.add_argument("--head-padding", type=float, default=2.1)
    parser.add_argument("--head-recon-command", type=str, default=None)
    parser.add_argument("--deca-root", type=str, default="external/DECA")
    parser.add_argument("--head-fusion-device", type=str, default="cuda")
    parser.add_argument("--head-fusion-top-k", type=int, default=8)
    parser.add_argument("--head-fusion-min-score", type=float, default=0.20)

    parser.add_argument("--head-texture-mode", choices=["best", "weighted_average"], default="best")
    parser.add_argument("--head-texture-top-k", type=int, default=5)
    parser.add_argument("--head-texture-min-quality", type=float, default=0.45)
    parser.add_argument("--head-texture-average-min-quality", type=float, default=0.62)

    # SMPL-X + head combine controls.
    parser.add_argument("--combine-body-source", choices=["auto", "canonical", "refined"], default="auto")
    parser.add_argument("--combine-body-path", type=str, default=None)
    parser.add_argument("--combine-head-obj", type=str, default=None)
    parser.add_argument("--combine-output-name", type=str, default="final_avatar")

    parser.add_argument("--combine-vertical-axis", choices=["auto", "x", "y", "z"], default="auto")
    parser.add_argument("--combine-neck-cutoff-ratio", type=float, default=0.895)
    parser.add_argument("--combine-neck-overlap-ratio", type=float, default=0.010)

    parser.add_argument("--combine-clip-head-neck", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--combine-head-clip-bottom-ratio", type=float, default=0.16)

    parser.add_argument("--combine-no-register-head", action="store_true")
    parser.add_argument("--combine-registration-iterations", type=int, default=16)
    parser.add_argument("--combine-registration-sample-points", type=int, default=900)
    parser.add_argument("--combine-registration-trim-fraction", type=float, default=0.60)
    parser.add_argument("--combine-registration-scale-delta-min", type=float, default=0.86)
    parser.add_argument("--combine-registration-scale-delta-max", type=float, default=1.00)
    parser.add_argument("--combine-registration-absolute-scale-min", type=float, default=0.82)
    parser.add_argument("--combine-registration-absolute-scale-max", type=float, default=1.08)
    parser.add_argument("--combine-registration-target-scale-shrink", type=float, default=0.94)
    parser.add_argument("--combine-registration-allow-rotation", action="store_true")
    parser.add_argument("--combine-registration-forward-icp-weight", type=float, default=0.50)

    # Manual post-registration offsets. Normally keep these at zero.
    parser.add_argument("--combine-head-scale", type=float, default=1.0)
    parser.add_argument("--combine-head-vertical-offset", type=float, default=0.0)
    parser.add_argument("--combine-head-lateral-offset", type=float, default=0.0)
    parser.add_argument("--combine-head-forward-offset", type=float, default=0.0)

    parser.add_argument("--combine-head-remove-lateral-ratio", type=float, default=0.135)
    parser.add_argument("--combine-head-remove-forward-ratio", type=float, default=0.115)

    parser.add_argument("--combine-connect-neck", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--combine-keep-body-head", action="store_true")

    # Measurement controls.
    parser.add_argument(
        "--height-cm",
        type=float,
        default=None,
        help="Known real model height in centimeters. Required for the measure stage.",
    )
    parser.add_argument(
        "--measurement-body-source",
        choices=["auto", "canonical", "refined", "combined"],
        default="auto",
        help="Which fitted body NPZ to measure. Auto prefers refined, then canonical.",
    )
    parser.add_argument(
        "--measurement-body-path",
        type=str,
        default=None,
        help="Optional explicit NPZ path to measure.",
    )
    parser.add_argument(
        "--measurement-output",
        type=str,
        default=None,
        help="Optional explicit output JSON path.",
    )
    parser.add_argument(
        "--measurement-density-g-per-cm3",
        type=float,
        default=0.95,
        help="Density used for estimated bust tissue weight.",
    )
    parser.add_argument(
        "--measurement-flip-left-right",
        action="store_true",
        help="Swap left/right measurement labels if your coordinate convention is reversed.",
    )

    # Texture controls.
    # "uv" performs multi-view UV texture stitching from all photos.
    # "initial" keeps the old simple TextureProjector behavior.
    parser.add_argument("--texture-mode", choices=["uv", "initial"], default="uv")
    parser.add_argument("--texture-size", type=int, default=2048)
    parser.add_argument("--texture-blend-mode", choices=["weighted", "best"], default="weighted")
    parser.add_argument("--texture-mesh-obj", type=str, default=None)
    parser.add_argument("--texture-erode-mask-px", type=int, default=3)
    parser.add_argument("--texture-feather-px", type=int, default=9)
    parser.add_argument("--texture-inpaint-radius", type=int, default=3)


    args = parser.parse_args()

    if args.head_full:
        args.only = ["head"]
        args.head_reconstruct = True
        args.head_fuse = True

    elif args.head and args.only is None:
        args.only = ["head"]

    # Full pipeline default: when no specific stage shortcut is provided and
    # the head stage is not skipped, run head extraction + reconstruction + fusion.
    if args.only is None and not args.head and "head" not in args.skip:
        args.head_reconstruct = True
        args.head_fuse = True


    input_dir = Path(args.input).resolve()
    output_root = Path(args.output).resolve()

    if not input_dir.exists():
        raise FileNotFoundError(input_dir)

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
        "head": output_root / "12_head",
        "measurements": output_root / "13_measurements",
        "combined": output_root / "14_combined",
    }

    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    print("\n🚀 Starting pipeline\n")
    print(f"Input:     {input_dir}")
    print(f"Output:    {output_root}")
    print(f"Only:      {args.only}")
    print(f"Skip:      {args.skip}")
    print(f"Head:      {args.head}")
    print(f"Head full: {args.head_full}")
    print(f"Head recon:{args.head_reconstruct}")
    print(f"Head fuse: {args.head_fuse}")
    print(f"Pose:      {args.pose_backend}")
    print(f"ViTPose:   {args.vitpose_model}")
    print()

    ensure_dependencies()
    load_runtime_imports()

    canonical_output = dirs["smplx"] / "canonical_body.npz"
    refined_output = dirs["refined"] / "refined_body.npz"
    texture_output = dirs["texture"] / "initial_texture.png"
    photometric_output = dirs["photometric"] / "photometric_body.npz"

    needs_preprocessed = not (
        args.only == ["head"] or
        args.only == ["combine"] or
        args.only == ["measure"]
    )

    if should_run("preprocess", args.only, args.skip):
        processed_images, pose_jsons, visibility_jsons = preprocess_images(
            input_dir,
            dirs,
            args,
        )
    elif needs_preprocessed:
        print("⏭ Skipping preprocessing; collecting existing preprocessed files")
        processed_images, pose_jsons, visibility_jsons = collect_existing_preprocessed(dirs)
    else:
        processed_images, pose_jsons, visibility_jsons = [], [], []

    if needs_preprocessed and len(processed_images) == 0:
        print("❌ No valid preprocessed images found")
        return

    if processed_images:
        print(f"✅ Usable images: {len(processed_images)}")

    if should_run("smplx", args.only, args.skip):
        print("\n🚀 Multi-image SMPL-X optimization\n")

        multi_optimizer = MultiImageOptimizer(
            model_path="models/smplx",
            gender="female",
            image_size=args.smplx_image_size,
            pseudo_weight=args.pseudo_weight,
            debug=not args.no_render_debug,
            debug_every=args.debug_every,
        )

        multi_optimizer.optimize(
            image_paths=processed_images,
            pose_json_paths=pose_jsons,
            visibility_json_paths=visibility_jsons,
            output_path=canonical_output,
            iterations=args.iterations,
        )

        export_npz_to_obj(
            canonical_output,
            dirs["smplx"] / "canonical_body.obj",
        )

        print("✅ Canonical body created")
    else:
        print("⏭ Skipping SMPL-X optimization")

    if should_run("refine", args.only, args.skip):
        if not canonical_output.exists():
            raise FileNotFoundError(f"Missing canonical body: {canonical_output}")

        print("\n🚀 Freeform refinement\n")

        mesh_refiner = MeshRefiner(
            image_size=args.refine_image_size,
        )

        mesh_refiner.refine(
            canonical_body_path=canonical_output,
            image_paths=processed_images,
            visibility_paths=visibility_jsons,
            output_path=refined_output,
            iterations=args.refine_iterations,
        )

        export_npz_to_obj(
            refined_output,
            dirs["refined"] / "refined_body.obj",
        )

        print("✅ Freeform refinement complete")
    else:
        print("⏭ Skipping refinement")

    if should_run("measure", args.only, args.skip):
        run_measurement_stage(
            dirs,
            args,
        )
    else:
        print("⏭ Skipping measurement stage")

    if should_run("head", args.only, args.skip):
        run_head_stage(
            input_dir,
            dirs,
            args,
        )
    else:
        print("⏭ Skipping head stage")


    if should_run("combine", args.only, args.skip):
        run_combine_stage(
            dirs,
            args,
        )
    else:
        print("⏭ Skipping combine stage")

    if should_run("texture", args.only, args.skip):
        if args.texture_mode == "uv":
            run_uv_texture_stage(
                dirs=dirs,
                processed_images=processed_images,
                args=args,
            )
            texture_output = dirs["texture"] / "canonical_body_texture.png"
        else:
            print("\n🚀 Creating initial texture\n")

            texture_projector = TextureProjector()

            texture_projector.create_initial_texture(
                image_paths=processed_images,
                output_path=texture_output,
            )

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

        photometric_optimizer.optimize(
            refined_mesh_path=refined_output,
            image_paths=processed_images,
            visibility_paths=visibility_jsons,
            texture_path=texture_output,
            output_path=photometric_output,
            iterations=args.photo_iterations,
        )

        print("✅ Photometric optimization complete")
    else:
        print("⏭ Skipping photometric optimization")

    print("\n✅ Pipeline finished")
    print(f"📁 Output:      {output_root}")
    print(f"🧍 Canonical:   {canonical_output}")
    print(f"🧍 Refined:     {refined_output}")
    print(f"🙂 Head:        {dirs['head']}")
    print(f"📏 Measurements:{dirs['measurements']}")
    print(f"🔗 Combined:    {dirs['combined']}")
    print(f"🎨 Texture:     {texture_output}")
    print(f"✨ Photometric: {photometric_output}")


if __name__ == "__main__":
    main()
