import importlib
import subprocess
import sys


def is_package_installed(module_name):
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False


def pip_install(args):
    subprocess.check_call([sys.executable, "-m", "pip", "install", *args])


def ensure_dependencies():
    print("🔧 Checking dependencies...")

    required_modules = [
        "torch",
        "torchvision",
        "cv2",
        "numpy",
        "PIL",
        "tqdm",
        "ultralytics",
        "mediapipe",
        "onnxruntime",
        "transformers",
        "accelerate",
        "timm",
        "hydra",
        "iopath",
        "sam2",
        "smplx",
        "pytorch3d",
    ]

    missing = [
        m for m in required_modules
        if not is_package_installed(m)
    ]

    if not missing:
        print("✅ Dependencies already installed.")
        import torch
        print(f"GPU available: {torch.cuda.is_available()}")
        return

    print(f"⚠ Missing modules: {missing}")

    print("Installing lightweight/common dependencies...")

    pip_install([
        "setuptools<81",
        "wheel",
        "packaging==24.2",
        "--force-reinstall"
    ])

    pip_install([
        "opencv-python",
        "numpy",
        "Pillow",
        "tqdm",
        "ultralytics",
        "mediapipe==0.10.14",
        "onnxruntime-gpu",
        "transformers",
        "accelerate",
        "timm",
        "hydra-core",
        "iopath",
        "smplx",
        "scikit-image",
    ])

    if not is_package_installed("sam2"):
        print("Installing SAM2...")
        pip_install([
            "git+https://github.com/facebookresearch/sam2.git"
        ])

    if not is_package_installed("pytorch3d"):
        print("Installing PyTorch3D from source...")
        pip_install([
            "--no-build-isolation",
            "git+https://github.com/facebookresearch/pytorch3d.git@stable"
        ])

    import torch
    print("✅ Dependencies installed.")
    print(f"Torch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")


if __name__ == "__main__":
    ensure_dependencies()