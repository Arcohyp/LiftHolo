import os
import argparse

import torch
import yaml
import numpy as np
from PIL import Image

from liftHolo import LiftHolo, propagation_ASM


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def preprocess_image(image_path, size=None):
    """Load and preprocess an image to amplitude tensor."""
    img = Image.open(image_path).convert('L')
    if size is not None:
        img = img.resize(size)
    img_np = np.array(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0)
    return tensor


def postprocess_phase(phase_tensor):
    """Convert phase tensor to numpy image."""
    phase_np = phase_tensor.squeeze().detach().cpu().numpy()
    phase_np = (phase_np - phase_np.min()) / (phase_np.max() - phase_np.min() + 1e-8)
    phase_np = (phase_np * 255).astype(np.uint8)
    return phase_np


def main():
    parser = argparse.ArgumentParser(description='LiftHolo Inference')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to config file')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input amplitude image')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='results/hologram.png',
                        help='Path to save output hologram')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Initialize model
    model_cfg = cfg['model']
    optical_cfg = cfg['optical']
    model = LiftHolo(
        scale_factor=model_cfg['scale_factor'],
        amp_channels=model_cfg['amp_channels'],
        name=model_cfg['name']
    ).to(device)

    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"Loaded checkpoint from {args.checkpoint}")
    model.eval()

    # Load input
    lr_amp = preprocess_image(args.input).to(device)
    phase = torch.zeros_like(lr_amp).to(device)

    # Optical parameters
    z = optical_cfg['z']
    pitch = optical_cfg['pitch']
    wavelength = optical_cfg['wavelength']
    pad = True

    # Precompute transfer function
    H = None  # Will be computed inside propagation_ASM if not provided

    with torch.no_grad():
        holophase, sr_amp = model(lr_amp, phase, z, pad, pitch, wavelength, H)

    # Save output
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    holo_img = Image.fromarray(postprocess_phase(holophase))
    holo_img.save(args.output)
    print(f"Saved hologram to {args.output}")

    # Optionally save amplitude SR
    amp_path = args.output.replace('.png', '_amp.png')
    amp_img = Image.fromarray(postprocess_phase(sr_amp))
    amp_img.save(amp_path)
    print(f"Saved amplitude SR to {amp_path}")


if __name__ == '__main__':
    main()
