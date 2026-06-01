from pathlib import Path
import json
import shlex
import shutil
import subprocess


class FaceReconstructionRunner:
    """
    Runs external DECA/EMOCA reconstruction on extracted crops.

    Important for fusion:
        Your command should save parameters, not only OBJ.

    For DECA, add:
        --saveMat True

    Example:
        python external/DECA/demos/demo_reconstruct.py -i {input_dir} -s {output_dir}
        --iscrop True --saveObj True --saveMat True --saveDepth True --saveKpt True
        --rasterizer_type=pytorch3d
    """

    def __init__(self, command_template=None, dry_run=False):
        self.command_template = command_template
        self.dry_run = bool(dry_run)

    def run_from_crop_index(self, crop_index_path, output_dir):
        crop_index_path = Path(crop_index_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(crop_index_path, "r") as f:
            crop_index = json.load(f)

        crop_dir = output_dir / "input_crops"
        crop_dir.mkdir(parents=True, exist_ok=True)

        for item in crop_index:
            src = Path(item["crop_path"])
            dst = crop_dir / src.name
            shutil.copy2(src, dst)

        return self.run(crop_dir, output_dir)

    def run(self, input_dir, output_dir):
        input_dir = Path(input_dir).resolve()
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        if not self.command_template:
            raise RuntimeError("No reconstruction command_template was provided.")

        cmd = self.command_template.format(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
        )

        print("🚀 Running face reconstruction command:")
        print(cmd)

        if self.dry_run:
            return {
                "command": cmd,
                "output_dir": str(output_dir),
                "dry_run": True,
            }

        process = subprocess.run(
            shlex.split(cmd),
            cwd=str(Path.cwd()),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        log_path = output_dir / "reconstruction_log.txt"
        log_path.write_text(process.stdout)

        if process.returncode != 0:
            raise RuntimeError(
                f"Face reconstruction failed with return code {process.returncode}. "
                f"See log: {log_path}"
            )

        obj_files = sorted(output_dir.rglob("*.obj"))
        mat_files = sorted(output_dir.rglob("*.mat"))
        npz_files = sorted(output_dir.rglob("*.npz"))

        result = {
            "command": cmd,
            "output_dir": str(output_dir),
            "log_path": str(log_path),
            "obj_files": [str(p) for p in obj_files],
            "mat_files": [str(p) for p in mat_files],
            "npz_files": [str(p) for p in npz_files],
        }

        with open(output_dir / "reconstruction_result.json", "w") as f:
            json.dump(result, f, indent=2)

        print(f"✓ Reconstruction finished.")
        print(f"✓ OBJ files: {len(obj_files)}")
        print(f"✓ MAT files: {len(mat_files)}")
        print(f"✓ NPZ files: {len(npz_files)}")

        if len(mat_files) == 0 and len(npz_files) == 0:
            print("⚠ No parameter files found. Fusion needs --saveMat True for DECA.")

        return result
