
import subprocess
import sys
from pathlib import Path

REQUIREMENTS_FILE = Path(__file__).parent / "requirements.txt"

def install_requirements():
    print("Checking and installing dependencies...")

    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "pip"
    ])

    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "install",
        "-r",
        str(REQUIREMENTS_FILE)
    ])

    print("Dependencies installed successfully.")

if __name__ == "__main__":
    install_requirements()
