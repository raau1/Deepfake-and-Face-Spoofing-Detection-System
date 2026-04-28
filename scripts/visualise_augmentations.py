"""Generate Figure 5.6: 1x5 grid of one face under the four compression-aware
augmentations applied at the strongest end of their training ranges
(`augmentations.py` defaults). Original on the left, then JPEG re-encode,
downscale+upscale, PIL Gaussian blur, additive Gaussian noise."""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageFilter


def jpeg_recode(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def downscale_upscale(img: Image.Image, scale: float) -> Image.Image:
    w, h = img.size
    return (img.resize((max(1, int(w * scale)), max(1, int(h * scale))),
                       Image.BILINEAR)
              .resize((w, h), Image.BILINEAR))


def pil_blur(img: Image.Image, radius: float) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def gaussian_noise(img: Image.Image, std: float, seed: int = 0) -> Image.Image:
    arr = np.asarray(img, dtype=np.float32) / 255.0
    rng = np.random.default_rng(seed)
    noisy = np.clip(arr + rng.normal(0.0, std, arr.shape), 0.0, 1.0)
    return Image.fromarray((noisy * 255.0).astype(np.uint8))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--dpi", type=int, default=220)
    args = ap.parse_args()

    base = Image.open(args.input).convert("RGB")

    panels = [
        ("(a) Original", base),
        ("(b) JPEG re-encode (q=30)", jpeg_recode(base, 30)),
        ("(c) Downscale + upscale (0.4x)", downscale_upscale(base, 0.4)),
        ("(d) Gaussian blur (radius=1.5)", pil_blur(base, 1.5)),
        ("(e) Gaussian noise (std=0.04)", gaussian_noise(base, 0.04, seed=42)),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(15, 3.6), dpi=args.dpi)
    for ax, (title, img) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight", facecolor="white")
    print(f"[visualise_augmentations] wrote {args.output}")


if __name__ == "__main__":
    main()
