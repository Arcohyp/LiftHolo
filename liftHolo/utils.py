"""
Optical propagation utilities adapted from the Neural Holography codebase.

This code is released under the Creative Commons Attribution-NonCommercial 4.0
International license (CC BY-NC). In a nutshell:
    - The license is only for non-commercial use (commercial licenses can be
      obtained from Stanford).
    - The material is provided as-is, with no warranties whatsoever.
    - If you publish any code, data, or scientific work based on this, please
      cite the original work.

Technical Paper:
Y. Peng, S. Choi, N. Padmanaban, G. Wetzstein. Neural Holography with
Camera-in-the-loop Training. ACM TOG (SIGGRAPH Asia), 2020.
"""

import math

import torch
import torch.nn.functional as F
import numpy as np


def _fftshift(tensor):
    """Apply fftshift to a tensor."""
    dim = (-2, -1)
    shifts = [tensor.shape[d] // 2 for d in dim]
    return torch.roll(tensor, shifts=shifts, dims=dim)


def _ifftshift(tensor):
    """Apply ifftshift to a tensor."""
    dim = (-2, -1)
    shifts = [-(tensor.shape[d] // 2) for d in dim]
    return torch.roll(tensor, shifts=shifts, dims=dim)


def _crop_image(img, target_shape, pytorch=True, stacked_complex=False):
    """Crop image to target shape."""
    if pytorch:
        h, w = target_shape
        current_h, current_w = img.shape[-2:]
        start_h = (current_h - h) // 2
        start_w = (current_w - w) // 2
        return img[..., start_h:start_h + h, start_w:start_w + w]
    return img


def _pad_image(img, target_shape, padval=0, stacked_complex=False):
    """Pad image to target shape."""
    h, w = target_shape
    current_h, current_w = img.shape[-2:]
    pad_h = h - current_h
    pad_w = w - current_w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    return F.pad(img, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=padval)


def _polar_to_rect(mag, angle):
    """Convert polar to rectangular coordinates."""
    return mag * torch.cos(angle), mag * torch.sin(angle)


def propagation_ASM(u_in, feature_size, wavelength, z, linear_conv=True,
                    padtype='zero', return_H=False, precomped_H=None,
                    return_H_exp=False, precomped_H_exp=None,
                    dtype=torch.float32):
    """
    Angular Spectrum Method (ASM) for hologram propagation.

    Based on Neural Holography (Peng et al., SIGGRAPH Asia 2020).
    """
    if linear_conv:
        input_resolution = u_in.size()[-2:]
        conv_size = [i * 2 for i in input_resolution]
        if padtype == 'zero':
            padval = 0
        elif padtype == 'median':
            padval = torch.median(torch.pow((u_in ** 2).sum(-1), 0.5))
        u_in = _pad_image(u_in, conv_size, padval=padval, stacked_complex=False)

    if precomped_H is None and precomped_H_exp is None:
        field_resolution = u_in.size()
        num_y, num_x = field_resolution[2], field_resolution[3]
        dy, dx = feature_size
        y, x = (dy * float(num_y), dx * float(num_x))

        fy = np.linspace(-1 / (2 * dy) + 0.5 / (2 * y), 1 / (2 * dy) - 0.5 / (2 * y), num_y)
        fx = np.linspace(-1 / (2 * dx) + 0.5 / (2 * x), 1 / (2 * dx) - 0.5 / (2 * x), num_x)

        FX, FY = np.meshgrid(fx, fy)

        HH = 2 * math.pi * np.sqrt(1 / wavelength ** 2 - (FX ** 2 + FY ** 2))

        H_exp = torch.tensor(HH, dtype=dtype).to(u_in.device)
        H_exp = torch.reshape(H_exp, (1, 1, *H_exp.size()))

    elif precomped_H_exp is not None:
        H_exp = precomped_H_exp

    if precomped_H is None:
        H_exp = torch.mul(H_exp, z)

        fy_max = 1 / np.sqrt((2 * z * (1 / y)) ** 2 + 1) / wavelength
        fx_max = 1 / np.sqrt((2 * z * (1 / x)) ** 2 + 1) / wavelength
        H_filter = torch.tensor(((np.abs(FX) < fx_max) & (np.abs(FY) < fy_max)).astype(np.uint8), dtype=dtype)

        H_real, H_imag = _polar_to_rect(H_filter.to(u_in.device), H_exp)

        H = torch.stack((H_real, H_imag), 4)
        H = _ifftshift(H)
        H = torch.view_as_complex(H)
    else:
        H = precomped_H

    if return_H_exp:
        return H_exp
    if return_H:
        return H

    U1 = torch.fft.fftn(_ifftshift(u_in), dim=(-2, -1), norm='ortho')
    U2 = H * U1
    u_out = _fftshift(torch.fft.ifftn(U2, dim=(-2, -1), norm='ortho'))

    if linear_conv:
        return _crop_image(u_out, input_resolution, pytorch=True, stacked_complex=False)
    else:
        return u_out
