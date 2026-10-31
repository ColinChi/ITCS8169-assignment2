"""Batched, per-sample image augmentation on the GPU.

Inputs are uint8 (B, 3, H, W) tensors from the cache; outputs are normalized
float tensors. Random-resized-crop and horizontal flip are a single affine warp.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

MEAN, STD = 0.45, 0.25  # the dataset is 15/16 grayscale; one constant per channel is enough


def _u(b: int, lo: float, hi: float, dev) -> torch.Tensor:
    return torch.empty(b, device=dev).uniform_(lo, hi)


def _mask(b: int, p: float, dev) -> torch.Tensor:
    return (torch.rand(b, 1, 1, 1, device=dev) < p)


def to_float(x: torch.Tensor) -> torch.Tensor:
    return x.float().div_(255)


def normalize(x: torch.Tensor) -> torch.Tensor:
    return (x - MEAN) / STD


def resized_crop(x: torch.Tensor, size: int, scale=(0.08, 1.0), ratio=(3 / 4, 4 / 3), flip=True) -> torch.Tensor:
    """Per-sample RandomResizedCrop (+ horizontal flip) of a float batch via one affine grid."""
    b, dev = x.shape[0], x.device
    area = _u(b, scale[0], scale[1], dev)
    ar = torch.exp(_u(b, math.log(ratio[0]), math.log(ratio[1]), dev))
    w = torch.sqrt(area * ar).clamp(max=1.0)  # crop width / height as a fraction of the image
    h = torch.sqrt(area / ar).clamp(max=1.0)
    cx = (torch.rand(b, device=dev) * 2 - 1) * (1 - w)
    cy = (torch.rand(b, device=dev) * 2 - 1) * (1 - h)
    sx = w * torch.where(torch.rand(b, device=dev) < 0.5, -1.0, 1.0) if flip else w
    theta = torch.zeros(b, 2, 3, device=dev)
    theta[:, 0, 0], theta[:, 0, 2], theta[:, 1, 1], theta[:, 1, 2] = sx, cx, h, cy
    grid = F.affine_grid(theta, (b, x.shape[1], size, size), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)


def _gray(x: torch.Tensor) -> torch.Tensor:
    return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]


def color_jitter(x: torch.Tensor, p: float, s: float) -> torch.Tensor:
    """Brightness, contrast and saturation, each scaled by a factor in [1-s, 1+s]."""
    if p <= 0 or s <= 0:
        return x
    b, dev = x.shape[0], x.device
    f = lambda: _u(b, 1 - s, 1 + s, dev).view(b, 1, 1, 1)  # noqa: E731
    y = x * f()
    m = _gray(y).mean((1, 2, 3), keepdim=True)
    y = (y - m) * f() + m
    g = _gray(y)
    y = ((y - g) * f() + g).clamp(0, 1)
    return torch.where(_mask(b, p, dev), y, x)


def random_gray(x: torch.Tensor, p: float) -> torch.Tensor:
    if p <= 0:
        return x
    return torch.where(_mask(x.shape[0], p, x.device), _gray(x).expand_as(x), x)


def gaussian_blur(x: torch.Tensor, p: float, sigma=(0.1, 2.0)) -> torch.Tensor:
    """Per-sample separable Gaussian blur; sigma is given relative to a 224 px image."""
    if p <= 0:
        return x
    b, c, h, w = x.shape
    dev, scale = x.device, h / 224
    sig = _u(b, sigma[0] * scale, sigma[1] * scale, dev)
    k = 2 * math.ceil(3 * sigma[1] * scale) + 1
    t = torch.arange(k, device=dev, dtype=x.dtype) - k // 2
    ker = torch.exp(-t.view(1, -1) ** 2 / (2 * sig.view(-1, 1) ** 2))
    ker = (ker / ker.sum(1, keepdim=True)).repeat_interleave(c, 0)  # (b*c, k)
    y = x.reshape(1, b * c, h, w)
    y = F.conv2d(F.pad(y, (k // 2, k // 2, 0, 0), mode="reflect"), ker.view(b * c, 1, 1, k), groups=b * c)
    y = F.conv2d(F.pad(y, (0, 0, k // 2, k // 2), mode="reflect"), ker.view(b * c, 1, k, 1), groups=b * c)
    return torch.where(_mask(b, p, dev), y.view(b, c, h, w), x)


def solarize(x: torch.Tensor, p: float) -> torch.Tensor:
    if p <= 0:
        return x
    return torch.where(_mask(x.shape[0], p, x.device), torch.where(x < 0.5, x, 1 - x), x)


def ssl_view(x8: torch.Tensor, size: int, scale, jitter=0.4, jitter_p=0.8, gray_p=0.2,
             blur_p=0.5, solarize_p=0.0) -> torch.Tensor:
    x = resized_crop(to_float(x8), size, scale)
    x = color_jitter(x, jitter_p, jitter)
    x = random_gray(x, gray_p)
    x = gaussian_blur(x, blur_p)
    x = solarize(x, solarize_p)
    return normalize(x)


def train_view(x8: torch.Tensor, size: int, crop_min: float = 0.35) -> torch.Tensor:
    """Supervised training augmentation: crop + flip + mild photometric jitter."""
    x = resized_crop(to_float(x8), size, (crop_min, 1.0))
    x = color_jitter(x, 0.5, 0.3)
    return normalize(x)


def eval_view(x8: torch.Tensor, size: int) -> torch.Tensor:
    """Deterministic: the whole (square) image resized to ``size``."""
    x = to_float(x8)
    if x.shape[-1] != size:
        x = F.interpolate(x, size=(size, size), mode="bilinear", antialias=True, align_corners=False)
    return normalize(x)
