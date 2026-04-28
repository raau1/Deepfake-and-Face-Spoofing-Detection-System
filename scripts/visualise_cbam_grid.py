"""Generate Figure 6.12: two 4x4 Grad-CAM grids placed side-by-side, one
at `backbone.act4` and one after the CBAM block, on the same row labels
(FF++ Real, FF++ Fake, Celeb-DF Real, Celeb-DF Fake)."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

ROW_ORDER = [
    ("ffpp_real",   "FF++ Real"),
    ("ffpp_fake",   "FF++ Fake"),
    ("celebdf_real", "Celeb-DF Real"),
    ("celebdf_fake", "Celeb-DF Fake"),
]


def fill_quadrant(axes_block, src_dir: Path):
    for r, (prefix, label) in enumerate(ROW_ORDER):
        files = sorted(src_dir.glob(f"{prefix}_*.png"))
        for c in range(4):
            ax = axes_block[r][c]
            if c < len(files):
                img = Image.open(files[c]).convert("RGB")
                ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(label, fontsize=10, weight="bold",
                              rotation=90, labelpad=8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--act4-dir", required=True)
    ap.add_argument("--after-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--dpi", type=int, default=160)
    args = ap.parse_args()

    fig = plt.figure(figsize=(20, 9.5), dpi=args.dpi)
    gs = fig.add_gridspec(4, 8, hspace=0.18, wspace=0.05,
                          left=0.04, right=0.99, top=0.92, bottom=0.04)
    left_axes = [[fig.add_subplot(gs[r, c]) for c in range(4)] for r in range(4)]
    right_axes = [[fig.add_subplot(gs[r, c + 4]) for c in range(4)] for r in range(4)]

    fill_quadrant(left_axes, Path(args.act4_dir))
    fill_quadrant(right_axes, Path(args.after_dir))

    fig.text(0.265, 0.955, "(a) Grad-CAM at backbone.act4",
             ha="center", va="bottom", fontsize=12, weight="bold")
    fig.text(0.74, 0.955, "(b) Grad-CAM after the CBAM block",
             ha="center", va="bottom", fontsize=12, weight="bold")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight", facecolor="white")
    print(f"[visualise_cbam_grid] wrote {args.output}")


if __name__ == "__main__":
    main()
