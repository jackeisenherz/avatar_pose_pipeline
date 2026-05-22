
# Avatar Pipeline Installer

This package now includes automatic dependency installation.

## Features

- detects missing packages
- installs dependencies automatically
- bootstraps environment at runtime
- supports clean machine deployment

## Usage

```bash
python src/main.py --input data/input --output outputs
```

OR manually install everything:

```bash
python install_dependencies.py
```

## Notes

The first launch may take several minutes because:
- PyTorch is large
- MMPose installs additional components
- ONNXRuntime downloads binaries
