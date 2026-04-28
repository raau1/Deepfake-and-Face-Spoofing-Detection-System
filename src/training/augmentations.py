"""
Compression-aware augmentation transforms for in-the-wild deepfake detection.

These transforms address the "in-the-wild generalisation gap" observed during
FastAPI demonstration: the baseline mixed/ensemble/temporal models classify
YouTube-sourced deepfakes as real with high confidence because YouTube's
VP9/AV1 re-encoding destroys the sub-pixel artefacts the networks learned to
rely on. Training with aggressive JPEG compression, downscale/upscale passes,
blur, colour jitter and Gaussian noise forces the model to learn manipulation
cues that survive the social-media distribution pipeline.
"""

from __future__ import annotations

import io
import random
from typing import List, Optional, Tuple

import torch
from PIL import Image, ImageFilter
from torchvision import transforms


class RandomJPEGCompression:
    """Randomly re-encode a PIL image as JPEG at a random quality level."""

    def __init__(self, quality_range: Tuple[int, int] = (30, 95), p: float = 0.7):
        self.quality_range = quality_range
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img
        quality = random.randint(self.quality_range[0], self.quality_range[1])
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return Image.open(buf).convert("RGB")


class RandomDownscaleUpscale:
    """Bilinear downscale and re-upscale to simulate low-bitrate streaming.

    YouTube's adaptive bitrate ladder serves heavily down-sampled copies of
    many videos; the browser then stretches them back up. Applying the same
    round-trip at training time teaches the network that resolution loss is
    not a cue for authenticity.
    """

    def __init__(self, scale_range: Tuple[float, float] = (0.4, 0.9), p: float = 0.5):
        self.scale_range = scale_range
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img
        w, h = img.size
        s = random.uniform(self.scale_range[0], self.scale_range[1])
        new_w = max(1, int(w * s))
        new_h = max(1, int(h * s))
        return img.resize((new_w, new_h), Image.BILINEAR).resize((w, h), Image.BILINEAR)


class RandomGaussianBlurPIL:
    """Gaussian blur with a random radius applied in PIL space."""

    def __init__(self, radius_range: Tuple[float, float] = (0.1, 1.5), p: float = 0.3):
        self.radius_range = radius_range
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img
        r = random.uniform(self.radius_range[0], self.radius_range[1])
        return img.filter(ImageFilter.GaussianBlur(radius=r))


class RandomGaussianNoise:
    """Additive Gaussian noise applied after ToTensor, before Normalize.

    The noise is added in [0, 1] tensor space so the standard-deviation is
    directly comparable to pixel-level perturbation magnitudes.
    """

    def __init__(self, std_range: Tuple[float, float] = (0.0, 0.04), p: float = 0.3):
        self.std_range = std_range
        self.p = p

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if random.random() >= self.p:
            return tensor
        std = random.uniform(self.std_range[0], self.std_range[1])
        return (tensor + torch.randn_like(tensor) * std).clamp(0.0, 1.0)


def get_robust_train_transforms(
    image_size: int = 299,
    normalize_mean: Optional[List[float]] = None,
    normalize_std: Optional[List[float]] = None,
) -> transforms.Compose:
    """Training pipeline with compression-aware augmentations.

    Combines the baseline geometric augmentations (horizontal flip, small
    rotation) with the social-media-distribution augmentations (downscale/
    upscale, JPEG re-encoding, blur, stronger colour jitter, Gaussian noise).
    """
    if normalize_mean is None:
        normalize_mean = [0.5, 0.5, 0.5]
    if normalize_std is None:
        normalize_std = [0.5, 0.5, 0.5]

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        RandomDownscaleUpscale(scale_range=(0.4, 0.9), p=0.5),
        RandomJPEGCompression(quality_range=(30, 95), p=0.7),
        RandomGaussianBlurPIL(radius_range=(0.1, 1.5), p=0.3),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        transforms.ToTensor(),
        RandomGaussianNoise(std_range=(0.0, 0.04), p=0.3),
        transforms.Normalize(mean=normalize_mean, std=normalize_std),
    ])


def get_eval_transforms(
    image_size: int = 299,
    normalize_mean: Optional[List[float]] = None,
    normalize_std: Optional[List[float]] = None,
) -> transforms.Compose:
    """Deterministic eval/val pipeline (resize + normalize only)."""
    if normalize_mean is None:
        normalize_mean = [0.5, 0.5, 0.5]
    if normalize_std is None:
        normalize_std = [0.5, 0.5, 0.5]

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize_mean, std=normalize_std),
    ])
