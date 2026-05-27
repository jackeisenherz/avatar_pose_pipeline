
# Avatar Pipeline Installer

Install PyTorch separately first:
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 \
  --index-url https://download.pytorch.org/whl/cu124

Then:
pip install -r requirements.txt --no-build-isolation


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

Only preprocessing
```bash
python3 src/main.py --input data/input --output data/output --only preprocess
```

Reuse preprocessing, run SMPL-X only
```bash
python3 src/main.py --input data/input --output data/output --only smplx
```

Skip preprocessing and photometric
```bash
python3 src/main.py --input data/input --output data/output --skip preprocess photometric
```

Disable render debug overlays
```bash
python3 src/main.py --input data/input --output data/output --no-render-debug
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
