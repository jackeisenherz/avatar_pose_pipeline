import importlib
import os
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.request import urlretrieve


_BOOTSTRAP_FILE = Path(__file__).resolve()
# bootstrap.py may live either in the project root or in src/.
# Use the repository root so external/DECA is not
# accidentally created under src/external, which would cause repeated clones
# and repeated model downloads.
if _BOOTSTRAP_FILE.parent.name == "src":
    PROJECT_ROOT = _BOOTSTRAP_FILE.parents[1]
else:
    PROJECT_ROOT = _BOOTSTRAP_FILE.parent

EXTERNAL_DIR = PROJECT_ROOT / "external"
DECA_DIR = EXTERNAL_DIR / "DECA"
DECA_REPO = "https://github.com/YadiraF/DECA.git"

GENERIC_MODEL_URL = (
    "https://huggingface.co/camenduru/BakedAvatar/resolve/"
    "bd4c39da1f55e1a17bd19bff5e8044ff5a76bd92/generic_model.pkl"
)

FLAME_ALBEDO_URL = (
    "https://huggingface.co/camenduru/TalkingHead/resolve/main/"
    "FLAME_albedo_from_BFM.npz"
)

DECA_MODEL_URL = (
    "https://huggingface.co/camenduru/show/resolve/"
    "d95d9e7901b981d744390e052e05432499d3106c/"
    "models/models_deca/data/deca_model.tar"
)

MIN_MODEL_BYTES = {
    "generic_model.pkl": 1_000_000,
    "FLAME_albedo_from_BFM.npz": 10_000,
    "deca_model.tar": 1_000_000,
}


def is_package_installed(module_name):
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False


def run(cmd, cwd=None, env=None):
    print(" ".join(str(x) for x in cmd))
    merged_env = os.environ.copy()
    merged_env.setdefault("PYTHONNOUSERSITE", "1")
    merged_env.setdefault("SETUPTOOLS_USE_DISTUTILS", "stdlib")
    if env:
        merged_env.update(env)
    subprocess.check_call(
        [str(x) for x in cmd],
        cwd=str(cwd) if cwd else None,
        env=merged_env,
    )


def pip_install(args):
    run([sys.executable, "-m", "pip", "install", "--no-user", *args])


def pip_uninstall(args):
    run([sys.executable, "-m", "pip", "uninstall", "-y", *args])


def pip_install_force(args):
    run([sys.executable, "-m", "pip", "install", "--no-user", "--force-reinstall", *args])


def pip_install_no_build_isolation(args):
    run([
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-user",
        "--no-build-isolation",
        *args,
    ])


def distribution_installed(distribution_name):
    try:
        from importlib.metadata import version
        version(distribution_name)
        return True
    except Exception:
        return False


def clear_imported_package(package_name):
    """
    Remove a package and all submodules from sys.modules.

    This matters after repairing packages with pip during the same Python
    process. Without clearing sys.modules, Python can keep returning the old
    broken already-imported module.
    """
    prefix = package_name + "."
    for name in list(sys.modules.keys()):
        if name == package_name or name.startswith(prefix):
            del sys.modules[name]


def pip_install_if_missing(module_name, pip_args):
    if is_package_installed(module_name):
        return False

    pip_install(pip_args)
    return True


def mediapipe_debug_info():
    try:
        import mediapipe as mp
        version = getattr(mp, "__version__", "unknown")
        path = getattr(mp, "__file__", None)
        has_solutions = hasattr(mp, "solutions")
        return {
            "version": version,
            "path": str(path),
            "has_solutions": bool(has_solutions),
        }
    except Exception as exc:
        return {
            "version": None,
            "path": None,
            "has_solutions": False,
            "error": repr(exc),
        }


def mediapipe_has_solutions():
    clear_imported_package("mediapipe")
    try:
        import mediapipe as mp
        if not hasattr(mp, "solutions"):
            return False
        return hasattr(mp.solutions, "face_mesh") and hasattr(mp.solutions, "pose")
    except Exception:
        return False


def ensure_mediapipe_installed():
    """
    Validate the classic MediaPipe Solutions API used by this project.

    Newer MediaPipe releases removed `mediapipe.solutions`, while this pipeline
    still uses:
        mediapipe.solutions.face_mesh
        mediapipe.solutions.pose

    Also, when we repair MediaPipe with pip inside bootstrap, we must clear
    sys.modules before re-checking, otherwise the old broken module remains
    cached in the running Python process.
    """
    if mediapipe_has_solutions():
        info = mediapipe_debug_info()
        print(f"✅ MediaPipe solutions API available: {info.get('version')} at {info.get('path')}")
        return False

    before = mediapipe_debug_info()
    print("⚠ MediaPipe is missing or does not expose mediapipe.solutions.")
    print(f"   Current MediaPipe info: {before}")
    print("   Installing classic MediaPipe with Solutions API: mediapipe==0.10.21")

    try:
        pip_uninstall(["mediapipe", "mediapipe-nightly"])
    except Exception:
        pass

    clear_imported_package("mediapipe")

    # 0.10.21 is the last broadly useful classic line before the newer
    # no-solutions releases. It is newer than 0.10.14 but still exposes
    # mp.solutions on supported Python versions.
    pip_install_force(["mediapipe==0.10.21"])

    clear_imported_package("mediapipe")

    if not mediapipe_has_solutions():
        after = mediapipe_debug_info()
        project_shadow = ""
        path = after.get("path")
        if path and str(PROJECT_ROOT) in path:
            project_shadow = (
                "\nA local project file/folder appears to be shadowing MediaPipe. "
                "Rename/remove it: " + path
            )

        message = (
            "MediaPipe installed, but mediapipe.solutions is still unavailable.\n"
            f"Before repair: {before}\n"
            f"After repair:  {after}\n"
            "Run this diagnostic:\n"
            "  python - <<'PY'\n"
            "  import mediapipe, sys\n"
            "  print('file:', mediapipe.__file__)\n"
            "  print('version:', getattr(mediapipe, '__version__', None))\n"
            "  print('has solutions:', hasattr(mediapipe, 'solutions'))\n"
            "  print('\\n'.join(sys.path))\n"
            "  PY"
            f"{project_shadow}"
        )

        # The current pipeline can run with the newer ViTPose-based pose path
        # and does not need to hard-fail here. Set AVATAR_REQUIRE_MEDIAPIPE=1
        # if you want the older strict behavior.
        if os.environ.get("AVATAR_REQUIRE_MEDIAPIPE", "0") == "1":
            raise RuntimeError(message)

        print("⚠ " + message.replace("\n", "\n  "))
        print("⚠ Continuing without MediaPipe Solutions API. Set AVATAR_REQUIRE_MEDIAPIPE=1 to make this fatal.")
        return False

    info = mediapipe_debug_info()
    print(f"✅ MediaPipe repaired: {info.get('version')} at {info.get('path')}")
    return True


def ensure_chumpy_installed():
    """
    chumpy==0.70 is a legacy setup.py package. Its setup.py imports `pip`,
    which breaks under modern PEP517 build isolation with:

        ModuleNotFoundError: No module named 'pip'
    """
    if distribution_installed("chumpy"):
        try:
            import chumpy  # noqa: F401
            print("✅ chumpy already present")
            return False
        except Exception as exc:
            print(f"⚠ chumpy package is present but import raised: {exc}")
            print("   Continuing; runtime compatibility patches may still handle this.")

    print("Installing legacy chumpy with --no-build-isolation...")
    pip_install_no_build_isolation(["chumpy==0.70"])
    return True


def file_is_present(path, min_bytes=1):
    path = Path(path)
    return path.exists() and path.is_file() and path.stat().st_size >= min_bytes


def find_existing_file(candidate_paths, min_bytes=1):
    for path in candidate_paths:
        path = Path(path)
        if file_is_present(path, min_bytes=min_bytes):
            return path
    return None


def copy_or_symlink_existing(src, dest):
    src = Path(src)
    dest = Path(dest)

    if file_is_present(dest, min_bytes=1):
        print(f"✅ Already present: {dest}")
        return False

    dest.parent.mkdir(parents=True, exist_ok=True)

    try:
        rel_src = os.path.relpath(src, start=dest.parent)
        dest.symlink_to(rel_src)
        print(f"✅ Linked {dest} -> {rel_src}")
    except Exception:
        shutil.copy2(src, dest)
        print(f"✅ Copied {src} -> {dest}")

    return True


def download_if_missing(url, dest, min_bytes=1, alternate_paths=None):
    dest = Path(dest)
    alternate_paths = [Path(p) for p in (alternate_paths or [])]
    candidates = [dest] + alternate_paths

    existing = find_existing_file(candidates, min_bytes=min_bytes)
    if existing is not None:
        print(f"✅ Model already present: {existing}")
        if existing != dest:
            copy_or_symlink_existing(existing, dest)
        return False

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".download")

    if tmp.exists():
        tmp.unlink()

    print(f"⬇ Downloading: {url}\n  -> {dest}")
    urlretrieve(url, tmp)

    if not file_is_present(tmp, min_bytes=min_bytes):
        size = tmp.stat().st_size if tmp.exists() else 0
        raise RuntimeError(
            f"Download failed or produced too-small file: {tmp} "
            f"({size} bytes, expected >= {min_bytes})"
        )

    tmp.replace(dest)
    return True


def ensure_deca_repo():
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)

    if (DECA_DIR / "decalib").exists() and (DECA_DIR / "demos").exists():
        print(f"✅ DECA already present: {DECA_DIR}")
        return False

    if DECA_DIR.exists() and any(DECA_DIR.iterdir()):
        print(f"⚠ DECA directory exists but looks incomplete: {DECA_DIR}")
        print("   Leaving it untouched. Remove it manually if you want a clean clone.")
        return False

    print("⬇ Cloning DECA into external/DECA...")
    run(["git", "clone", DECA_REPO, str(DECA_DIR)])
    return True


def patch_deca_demo_reconstruct():
    script = DECA_DIR / "demos" / "demo_reconstruct.py"

    if not script.exists():
        print(f"⚠ Cannot patch missing DECA demo: {script}")
        return False

    text = script.read_text(errors="ignore")
    changed = False

    numpy_patch = """
# ---- avatar_pose_pipeline NumPy compatibility patch ----
import numpy as _avatar_np
for _avatar_name, _avatar_value in {
    "bool": bool,
    "int": int,
    "float": float,
    "complex": complex,
    "object": object,
    "str": str,
    "unicode": str,
}.items():
    if not hasattr(_avatar_np, _avatar_name):
        setattr(_avatar_np, _avatar_name, _avatar_value)
# ---- end avatar_pose_pipeline patch ----
"""

    helper_patch = """
# ---- avatar_pose_pipeline DECA code-saving helpers ----
def _avatar_tensor_to_numpy(value):
    try:
        import torch
        if torch.is_tensor(value):
            return value.detach().cpu().numpy()
    except Exception:
        pass

    if isinstance(value, dict):
        return {k: _avatar_tensor_to_numpy(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [_avatar_tensor_to_numpy(v) for v in value]

    return value


def _avatar_first_batch(value):
    import numpy as np
    arr = _avatar_tensor_to_numpy(value)
    if isinstance(arr, np.ndarray) and arr.ndim > 0 and arr.shape[0] == 1:
        return arr[0]
    return arr


def _avatar_build_code_mat(codedict, opdict):
    mat = {}

    for src_key, dst_key in [
        ("shape", "shape"),
        ("exp", "exp"),
        ("pose", "pose"),
        ("tex", "tex"),
        ("cam", "cam"),
        ("light", "light"),
        ("detail", "detail"),
    ]:
        if src_key in codedict:
            mat[dst_key] = _avatar_first_batch(codedict[src_key])

    if "shape" in mat:
        mat["shapecode"] = mat["shape"]
        mat["shape_params"] = mat["shape"]

    if "exp" in mat:
        mat["expression"] = mat["exp"]
        mat["expression_params"] = mat["exp"]

    if "pose" in mat:
        mat["pose_params"] = mat["pose"]
        mat["flame_pose"] = mat["pose"]

    if "tex" in mat:
        mat["texcode"] = mat["tex"]
        mat["texture_code"] = mat["tex"]

    for key in [
        "verts",
        "trans_verts",
        "landmarks2d",
        "landmarks3d",
        "landmarks3d_world",
    ]:
        if key in opdict:
            try:
                mat[key] = _avatar_first_batch(opdict[key])
            except Exception:
                pass

    return mat


def _avatar_save_codes(save_dir, name, codedict, opdict):
    from pathlib import Path
    import numpy as np
    from scipy.io import savemat

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    mat = _avatar_build_code_mat(codedict, opdict)

    if not mat:
        return

    savemat(str(save_dir / f"{name}_codes.mat"), mat)
    np.savez(str(save_dir / f"{name}_codes.npz"), **mat)
# ---- end avatar_pose_pipeline DECA code-saving helpers ----
"""

    if "avatar_pose_pipeline NumPy compatibility patch" not in text:
        text = numpy_patch + "\n" + text
        changed = True

    if "avatar_pose_pipeline DECA code-saving helpers" not in text:
        text = helper_patch + "\n" + text
        changed = True

    if "avatar_pose_pipeline extract_tex config patch" not in text:
        config_patch = """
    # ---- avatar_pose_pipeline extract_tex config patch ----
    try:
        deca_cfg.defrost()
    except Exception:
        pass
    try:
        deca_cfg.model.extract_tex = bool(args.extractTex)
    except Exception:
        pass
    try:
        deca_cfg.model.use_tex = bool(args.useTex)
    except Exception:
        pass
    try:
        deca_cfg.freeze()
    except Exception:
        pass
    # ---- end avatar_pose_pipeline extract_tex config patch ----
"""

        markers = [
            "deca = DECA(config = deca_cfg, device=device)",
            "deca = DECA(config=deca_cfg, device=device)",
            "deca = DECA(config = cfg, device=device)",
            "deca = DECA(config=cfg, device=device)",
        ]

        inserted = False
        for marker in markers:
            if marker in text:
                text = text.replace(marker, config_patch + "\n    " + marker, 1)
                inserted = True
                changed = True
                break

        if not inserted:
            print("⚠ Could not find DECA construction line for extract_tex patch.")

    if "avatar_pose_pipeline save encoded FLAME codes" not in text:
        save_patch = """
            # ---- avatar_pose_pipeline save encoded FLAME codes ----
            try:
                _avatar_save_codes(os.path.join(savefolder, name), name, codedict, opdict)
            except Exception as _avatar_exc:
                print(f"⚠ Could not save DECA codes for {name}: {_avatar_exc}")
            # ---- end avatar_pose_pipeline save encoded FLAME codes ----
"""

        markers = [
            "opdict, visdict = deca.decode(codedict)",
            "opdict, visdict = deca.decode(codedict, rendering=True)",
            "opdict, visdict = deca.decode(codedict, rendering = True)",
        ]

        inserted = False
        for marker in markers:
            if marker in text:
                text = text.replace(marker, marker + "\n" + save_patch, 1)
                inserted = True
                changed = True
                break

        if not inserted and "return_vis=True)" in text:
            text = text.replace("return_vis=True)", "return_vis=True)\n" + save_patch, 1)
            inserted = True
            changed = True

        if not inserted:
            print("⚠ Could not find DECA decode line for code-saving patch.")
            print("   If MAT/NPZ code files are not produced, patch demo_reconstruct.py manually.")

    if changed:
        backup = script.with_suffix(script.suffix + ".avatar_backup")
        if not backup.exists():
            shutil.copy2(script, backup)
            print(f"✅ Created DECA demo backup: {backup}")

        script.write_text(text)
        print(f"✅ Patched DECA demo in place: {script}")
    else:
        print(f"✅ DECA demo already patched: {script}")

    return changed


def install_deca_python_dependencies():
    print("🔧 Checking DECA Python dependencies...")

    deps = [
        ("face_alignment", ["face-alignment"]),
        ("skimage", ["scikit-image"]),
        ("yaml", ["PyYAML"]),
        ("yacs", ["yacs"]),
        ("kornia", ["kornia"]),
        ("matplotlib", ["matplotlib"]),
        ("imageio", ["imageio"]),
        ("scipy", ["scipy"]),
        ("trimesh", ["trimesh"]),
    ]

    for module, pip_args in deps:
        pip_install_if_missing(module, pip_args)

    ensure_chumpy_installed()



def ensure_legacy_numpy_inspect_compat():
    sitecustomize = PROJECT_ROOT / "sitecustomize.py"
    patch = """# Auto-generated by avatar_pose_pipeline bootstrap.
import inspect
try:
    from collections import namedtuple
    if not hasattr(inspect, "getargspec"):
        ArgSpec = namedtuple("ArgSpec", "args varargs keywords defaults")
        def getargspec(func):
            spec = inspect.getfullargspec(func)
            return ArgSpec(spec.args, spec.varargs, spec.varkw, spec.defaults)
        inspect.getargspec = getargspec
except Exception:
    pass

try:
    import numpy as _np
    for _name, _value in {
        "bool": bool,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
        "str": str,
        "unicode": str,
    }.items():
        if not hasattr(_np, _name):
            setattr(_np, _name, _value)
except Exception:
    pass
"""
    if sitecustomize.exists() and sitecustomize.read_text(errors="ignore") == patch:
        return False
    sitecustomize.write_text(patch)
    print(f"✅ Wrote legacy compatibility patch: {sitecustomize}")
    return True


def ensure_deca_models():
    data_dir = DECA_DIR / "data"
    flame2020_dir = data_dir / "FLAME2020"
    data_dir.mkdir(parents=True, exist_ok=True)
    flame2020_dir.mkdir(parents=True, exist_ok=True)

    generic_data = data_dir / "generic_model.pkl"
    generic_flame2020 = flame2020_dir / "generic_model.pkl"
    albedo = data_dir / "FLAME_albedo_from_BFM.npz"
    deca_model = data_dir / "deca_model.tar"

    min_generic = MIN_MODEL_BYTES["generic_model.pkl"]
    min_albedo = MIN_MODEL_BYTES["FLAME_albedo_from_BFM.npz"]
    min_deca = MIN_MODEL_BYTES["deca_model.tar"]

    existing_generic = find_existing_file(
        [generic_data, generic_flame2020],
        min_bytes=min_generic,
    )

    if existing_generic is not None:
        print(f"✅ FLAME generic model already present: {existing_generic}")
        if existing_generic != generic_data:
            copy_or_symlink_existing(existing_generic, generic_data)
        if existing_generic != generic_flame2020:
            copy_or_symlink_existing(existing_generic, generic_flame2020)
    else:
        download_if_missing(GENERIC_MODEL_URL, generic_data, min_bytes=min_generic)
        copy_or_symlink_existing(generic_data, generic_flame2020)

    download_if_missing(FLAME_ALBEDO_URL, albedo, min_bytes=min_albedo)
    download_if_missing(DECA_MODEL_URL, deca_model, min_bytes=min_deca)



def ensure_dependencies():
    print("🔧 Checking dependencies...")
    ensure_legacy_numpy_inspect_compat()

    required_modules = [
        "torch",
        "torchvision",
        "cv2",
        "numpy",
        "PIL",
        "tqdm",
        "ultralytics",
        "onnxruntime",
        "transformers",
        "accelerate",
        "timm",
        "hydra",
        "iopath",
        "sam2",
        "smplx",
        "pytorch3d",
        "face_alignment",
        "skimage",
        "yaml",
        "yacs",
        "kornia",
        "scipy",
        "trimesh",
    ]

    missing = [m for m in required_modules if not is_package_installed(m)]

    if missing:
        print(f"⚠ Missing modules: {missing}")
        print("Installing missing dependencies only...")

        common_packages = [
            ("cv2", ["opencv-python"]),
            ("numpy", ["numpy"]),
            ("PIL", ["Pillow"]),
            ("tqdm", ["tqdm"]),
            ("ultralytics", ["ultralytics"]),
            ("onnxruntime", ["onnxruntime-gpu"]),
            ("transformers", ["transformers"]),
            ("accelerate", ["accelerate"]),
            ("timm", ["timm"]),
            ("hydra", ["hydra-core"]),
            ("iopath", ["iopath"]),
            ("smplx", ["smplx"]),
            ("skimage", ["scikit-image"]),
            ("yaml", ["PyYAML"]),
            ("yacs", ["yacs"]),
            ("kornia", ["kornia"]),
            ("scipy", ["scipy"]),
            ("trimesh", ["trimesh"]),
            ("face_alignment", ["face-alignment"]),
            ("matplotlib", ["matplotlib"]),
            ("imageio", ["imageio"]),
        ]

        for module, package_args in common_packages:
            pip_install_if_missing(module, package_args)

        ensure_mediapipe_installed()
        ensure_chumpy_installed()

        if not is_package_installed("sam2"):
            print("Installing SAM2...")
            pip_install(["git+https://github.com/facebookresearch/sam2.git"])

        if not is_package_installed("pytorch3d"):
            print("Installing PyTorch3D from source...")
            pip_install([
                "--no-build-isolation",
                "git+https://github.com/facebookresearch/pytorch3d.git@stable",
            ])
    else:
        print("✅ Python dependencies already installed.")

    ensure_mediapipe_installed()
    ensure_chumpy_installed()

    ensure_deca_repo()
    install_deca_python_dependencies()
    ensure_deca_models()
    patch_deca_demo_reconstruct()


    import torch
    print("✅ Dependencies checked.")
    print(f"Torch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")


if __name__ == "__main__":
    ensure_dependencies()
