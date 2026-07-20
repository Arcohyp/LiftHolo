import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .complex_layers import ComplexConvTranspose2d, ComplexConv2d, complex_relu
from .utils import propagation_ASM


def _match_complex_size(target, source):
    t_h, t_w = target.shape[-2:]
    s_h, s_w = source.shape[-2:]
    if s_h != t_h or s_w != t_w:
        source = torch.complex(
            F.interpolate(source.real, size=(t_h, t_w), mode='bilinear', align_corners=False),
            F.interpolate(source.imag, size=(t_h, t_w), mode='bilinear', align_corners=False)
        )
    return source


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.cov1 = nn.Sequential(ComplexConv2d(in_channels, out_channels, 3, stride=2, padding=1))

    def forward(self, x):
        return complex_relu(self.cov1(x))


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.cov1 = nn.Sequential(ComplexConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1))

    def forward(self, x):
        return complex_relu(self.cov1(x))


class UpBlockNoAct(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.cov1 = nn.Sequential(ComplexConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1))

    def forward(self, x):
        return self.cov1(x)


class ComplexSR(nn.Module):
    """Lightweight complex-domain amplitude+phase super-resolution"""
    def __init__(self, scale_factor=2, channels=32):
        super().__init__()
        self.scale_factor = scale_factor

        self.conv_first = nn.Conv2d(2, channels, 5, padding=2)

        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv_up = nn.Conv2d(channels, channels * scale_factor * scale_factor, 3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)

        self.conv_out = nn.Conv2d(channels, 2, 3, padding=1)

        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.lrelu(self.conv_first(x))
        x = self.body(x)
        x = self.lrelu(self.conv_up(x))
        x = self.pixel_shuffle(x)
        x = self.conv_out(x)
        return x


class AmplitudeSR(nn.Module):
    """Ultra-lightweight amplitude-only super-resolution with pixel shuffle"""
    def __init__(self, scale_factor=2, channels=24, single_pixelshuffle=False):
        super().__init__()
        self.scale_factor = scale_factor
        self.single_pixelshuffle = single_pixelshuffle and scale_factor == 2

        self.conv1 = nn.Conv2d(1, channels, 5, padding=2)

        if self.single_pixelshuffle:
            # Corrected variant: one PixelShuffle for x2, halving parameters.
            self.conv2 = nn.Conv2d(channels, channels * scale_factor * scale_factor, 3, padding=1)
            self.pixel_shuffle1 = nn.PixelShuffle(scale_factor)
            self.conv3 = None
            self.pixel_shuffle2 = None
        else:
            self.conv2 = nn.Conv2d(channels, channels * 4, 3, padding=1)
            self.pixel_shuffle1 = nn.PixelShuffle(2)
            self.conv3 = nn.Conv2d(channels, channels * 4, 3, padding=1)
            self.pixel_shuffle2 = nn.PixelShuffle(2)

        self.conv_out = nn.Conv2d(channels, 1, 3, padding=1)

        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.lrelu(self.conv1(x))
        x = self.lrelu(self.conv2(x))

        x = self.pixel_shuffle1(x)
        if self.conv3 is not None:
            x = self.lrelu(self.conv3(x))
            x = self.pixel_shuffle2(x)

        x = self.conv_out(x)

        return torch.sigmoid(x)


class PhaseRefiner(nn.Module):
    """Lighter CCNN2 for hologram phase refinement"""
    def __init__(self, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor

        self.netdown1 = DownBlock(2, 2)
        self.netdown2 = DownBlock(2, 4)

        self.netup2 = UpBlock(4, 2)
        self.netup1 = UpBlockNoAct(2, 1)

    def forward(self, x):
        x = torch.complex(x, torch.zeros_like(x))

        out1 = self.netdown1(x)
        out2 = self.netdown2(out1)

        out12 = self.netup2(out2)
        out_matched = _match_complex_size(out12, out1)
        out13 = self.netup1(out12 + out_matched)

        holophase = torch.atan2(out13.imag, out13.real)
        return holophase


class PhaseEstimator(nn.Module):
    """Lighter phase estimation network"""
    def __init__(self, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor

        mid_channels = 4 * scale_factor * scale_factor
        self.upscale = nn.Sequential(
            nn.Conv2d(1, mid_channels, 3, padding=1),
            nn.PixelShuffle(scale_factor),
            nn.ReLU(),
        )

        self.netdown1 = DownBlock(4, 2)
        self.netdown2 = DownBlock(2, 4)

        self.netup2 = UpBlock(4, 2)
        self.netup1 = UpBlockNoAct(2, 1)

    def forward(self, x):
        x = self.upscale(x)
        x = torch.complex(x, torch.zeros_like(x))

        out1 = self.netdown1(x)
        out2 = self.netdown2(out1)

        out15 = self.netup2(out2)
        out_matched = _match_complex_size(out15, out1)
        out16 = self.netup1(out15 + out_matched)

        predictphase = torch.atan2(out16.imag, out16.real)
        return predictphase


class LiftHolo(nn.Module):
    """LiftHolo: amplitude super-resolution guided holographic phase generation"""
    def __init__(self, scale_factor=2, amp_channels=24, single_pixelshuffle=False, name='liftholo'):
        super().__init__()
        self.name = name
        self.scale_factor = scale_factor

        self.amp_sr = AmplitudeSR(scale_factor, amp_channels, single_pixelshuffle)
        self.ccnn1_phase = PhaseEstimator(scale_factor)
        self.ccnn2 = PhaseRefiner(scale_factor)

        total_params = sum(p.numel() for p in self.parameters())
        print(f"LiftHolo parameters: {total_params:,}")

    def forward(self, lr_amp, phase, z, pad, pitch, wavelength, H):
        target_h = lr_amp.shape[2] * self.scale_factor
        target_w = lr_amp.shape[3] * self.scale_factor

        sr_amp = self.amp_sr(lr_amp)
        if sr_amp.shape[2] != target_h or sr_amp.shape[3] != target_w:
            sr_amp = F.interpolate(sr_amp, size=(target_h, target_w), mode='bilinear', align_corners=False)

        predict_phase = self.ccnn1_phase(lr_amp)
        predict_phase = F.interpolate(predict_phase, size=(target_h, target_w),
                                      mode='bilinear', align_corners=False)

        predict_complex = torch.complex(
            sr_amp * torch.cos(predict_phase),
            sr_amp * torch.sin(predict_phase)
        )

        slmfield = propagation_ASM(
            u_in=predict_complex, z=z, linear_conv=pad,
            feature_size=[pitch, pitch], wavelength=wavelength,
            precomped_H=H
        )

        slmamp = torch.pow(slmfield.real**2 + slmfield.imag**2, 0.5)
        slmphase = torch.atan2(slmfield.imag, slmfield.real)

        slm_amp_phase = torch.cat((slmamp, slmphase), dim=-3)

        holophase = self.ccnn2(slm_amp_phase)

        return holophase, sr_amp
