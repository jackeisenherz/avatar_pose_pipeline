#!/usr/bin/env python3
from pathlib import Path
import argparse
import json

from head_pipeline.face_crop_extractor import FaceCropExtractor
from head_pipeline.face_reconstruction_runner import FaceReconstructionRunner
from head_pipeline.head_fusion import HeadFusion


def main():
    parser = argparse.ArgumentParser(description="Head extraction + standalone reconstruction + fusion")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--crop-size", type=int, default=768)
    parser.add_argument("--padding", type=float, default=2.1)

    parser.add_argument("--reconstruct", action="store_true")
    parser.add_argument("--recon-command", type=str, default=None)

    parser.add_argument("--fuse", action="store_true")
    parser.add_argument("--deca-root", type=str, default="external/DECA")
    parser.add_argument("--fusion-top-k", type=int, default=8)
    parser.add_argument("--fusion-min-score", type=float, default=0.20)

    args = parser.parse_args()

    input_dir = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cropper = FaceCropExtractor(
        crop_size=args.crop_size,
        padding=args.padding,
        debug=True,
    )

    crop_results = cropper.extract_directory(input_dir=input_dir, output_dir=output_dir)
    crop_index = output_dir / "head_crops_index.json"

    recon_result = None

    if args.reconstruct:
        if not args.recon_command:
            raise RuntimeError("--reconstruct requires --recon-command")

        runner = FaceReconstructionRunner(command_template=args.recon_command)
        recon_result = runner.run_from_crop_index(
            crop_index_path=crop_index,
            output_dir=output_dir / "recon",
        )

    fusion_result = None

    if args.fuse:
        fusion = HeadFusion(
            deca_root=args.deca_root,
            top_k=args.fusion_top_k,
            min_score=args.fusion_min_score,
        )

        fusion_result = fusion.fuse_from_recon_dir(
            recon_dir=output_dir / "recon",
            crop_index_path=crop_index,
            output_dir=output_dir / "fused",
        )

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "num_crops": len(crop_results),
        "crop_index": str(crop_index),
        "reconstruction": recon_result,
        "fusion": fusion_result,
        "note": "Head is not registered to SMPL-X here. Use combine stage later.",
    }

    with open(output_dir / "head_pipeline_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("✅ Head pipeline finished")
    print(f"📁 Output: {output_dir}")


if __name__ == "__main__":
    main()
