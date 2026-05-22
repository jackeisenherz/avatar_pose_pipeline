# src/bootstrap.py
import importlib
import subprocess
import sys

def is_package_installed(package_name):
    try:
        importlib.import_module(package_name.replace("-", "_"))
        return True
    except ImportError:
        return False

def ensure_dependencies():
    print("🔧 Checking dependencies...")

    # Only run heavy installs if needed
    if is_package_installed("ultralytics") and is_package_installed("mediapipe"):
        print("✅ Core dependencies already installed.")
        import torch
        print(f"GPU available: {torch.cuda.is_available()}")
        return

    print("Installing missing dependencies...")

    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "setuptools<81", "wheel", "packaging==24.2", "--force-reinstall"
    ])

    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "ultralytics", "rembg", "mediapipe==0.10.14", "opencv-python", "Pillow", "tqdm"
    ])

    print("✅ Dependencies installed successfully!")
    import torch
    print(f"GPU available: {torch.cuda.is_available()}")


if __name__ == "__main__":
    ensure_dependencies()