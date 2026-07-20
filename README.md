# LiftHolo

**LiftHolo: Lifted Amplitude-Guided Deep Learning for Accelerated High-Quality Computer-Generated Holography**

LiftHolo is a deep learning framework for holographic image super-resolution and phase-only hologram generation. It combines an efficient amplitude super-resolution branch with complex-valued convolutional networks for phase estimation and refinement, enabling high-quality hologram reconstruction with minimal parameters.

> **Note**: This repository contains the reference implementation accompanying a manuscript currently under **major revision** at *Optics Letters*. Code is provided for reproducibility and community feedback; please cite the preprint or repository if you use this code prior to official publication.

## Features

- **Ultra-lightweight**: ~44K parameters by default, with an optional ~23K corrected single-PixelShuffle variant
- **Hybrid architecture**: Amplitude SR + complex-valued phase estimation + phase refinement
- **End-to-end hologram generation**: From low-resolution amplitude to high-resolution phase-only hologram
- **Angular Spectrum Method (ASM)**: Built-in wave propagation simulation

## Installation

```bash
git clone https://github.com/Arcohyp/LiftHolo.git
cd LiftHolo
pip install -r requirements.txt
```

## Usage

### Training

Train LiftHolo on your own dataset:

```bash
python scripts/train.py \
    --config configs/default.yaml \
    --data-dir /path/to/your/dataset \
    --val-dir /path/to/validation/set \
    --output-dir ./experiments \
    --epochs 30 \
    --batch-size 1 \
    --lr 0.001
```

**Dataset format:**
```
your_dataset/
├── hr/              # High-resolution amplitude images (3840x2160)
│   ├── 0001.png
│   └── ...
└── lr/              # Low-resolution amplitude images (1920x1080 for x2)
    ├── 0001.png
    └── ...
```

You can use public datasets like DIV2K and downsample them to create LR inputs.

### Inference

After training, run inference with your checkpoint:

```bash
python scripts/inference.py \
    --config configs/default.yaml \
    --input data/sample.png \
    --checkpoint checkpoints/liftholo_best.pth \
    --output results/hologram.png
```

### Python API

```python
import torch
from liftHolo import LiftHolo

# Initialize model (default ~44K param variant; use single_pixelshuffle=True for ~23K)
model = LiftHolo(scale_factor=2, amp_channels=24)
model.eval()

# Load input
lr_amp = torch.randn(1, 1, 1080, 1920)  # Low-resolution amplitude
phase = torch.zeros_like(lr_amp)

# Optical parameters (from paper)
z = 0.15          # Propagation distance (m)
pitch = 3.6e-6    # SLM pixel pitch (m)
wavelength = 532e-9  # Wavelength (m)

# Forward pass
with torch.no_grad():
    holophase, sr_amp = model(lr_amp, phase, z, True, pitch, wavelength, None)
```

## Project Structure

```
LiftHolo/
├── liftHolo/           # Core package
│   ├── model.py        # LiftHolo model definition
│   ├── complex_layers.py  # Embedded complex-valued layers
│   └── utils.py        # Utility functions (ASM propagation)
├── scripts/
│   ├── train.py        # Training script
│   └── inference.py    # Inference script
├── configs/
│   └── default.yaml    # Default configuration (paper parameters)
├── results/samples/    # Sample results
└── README.md
```

## Paper Parameters

The default configuration uses the same optical parameters as the paper:
- Wavelength: **532 nm** (green)
- SLM pixel pitch: **3.6 μm**
- Propagation distance: **150 mm**
- Target resolution: **3840 × 2160** (4K UHD)
- Input resolution: **1920 × 1080** (for x2 super-resolution)

### Model variants

The default model (`amp_channels=24`) corresponds to the main LiftHolo model reported in the paper (~44K parameters, DIV2K PSNR 33.72 dB). We also provide a corrected single-PixelShuffle EASNet variant that halves the parameter count to ~23K and reduces latency to ~50 ms while retaining comparable fidelity (DIV2K PSNR 32.56 dB). To use it, set `single_pixelshuffle: true` in the config or pass `single_pixelshuffle=True` when constructing the model:

```python
from liftHolo import LiftHolo

model = LiftHolo(scale_factor=2, amp_channels=24, single_pixelshuffle=True)
```

## Citation

If you use LiftHolo in your research, please cite this repository directly until the associated paper is officially published:

```bibtex
@software{liftholo2026,
  title={LiftHolo: Lifted Amplitude-Guided Deep Learning for Accelerated High-Quality Computer-Generated Holography},
  author={Cheng, Xirun and Liu, Yong and Wang, Quan and Liu, Xiang and Zhang, Chaofan and Gao, Zhenyu},
  year={2026},
  url={https://github.com/Arcohyp/LiftHolo}
}
```

> **Note**: A manuscript describing this work is currently under peer review. The BibTeX entry above will be updated with the official publication details once the paper is accepted.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
