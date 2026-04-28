"""Tile a directory of Grad-CAM 'original | heatmap' PNGs into a 4x4 grid
with row labels. Used for Figure 6.8 (v2 robust sanity sweep), Figure 6.9
(v3 robust sanity sweep), and Figure 6.12 (CBAM act4 + after, two
sub-grids). Filenames must follow the convention <split>_<class>_NN_*.png
where split in {ffpp, celebdf} and class in {real, fake}."""
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True,
                    help="Directory containing the 16 sanity PNGs.")
    ap.add_argument("--output", required=True)
    ap.add_argument("--title", default="")
    ap.add_argument("--dpi", type=int, default=160)
    args = ap.parse_args()

    src = Path(args.dir)

    fig, axes = plt.subplots(4, 4, figsize=(15, 9.5), dpi=args.dpi)

    for r, (prefix, label) in enumerate(ROW_ORDER):
        files = sorted(src.glob(f"{prefix}_*.png"))
        if len(files) < 4:
            for c in range(4):
                axes[r][c].set_title(f"{prefix}: missing", fontsize=9, color="red")
                axes[r][c].axis("off")
            continue
        for c in range(4):
            img = Image.open(files[c]).convert("RGB")
            axes[r][c].imshow(img)
            axes[r][c].set_xticks([])
            axes[r][c].set_yticks([])
            if c == 0:
                axes[r][c].set_ylabel(label, fontsize=11, weight="bold",
                                      rotation=90, labelpad=10)
            axes[r][c].set_title(files[c].stem.split("_", 2)[2][:28],
                                 fontsize=7.5, color="#444444")

    if args.title:
        fig.suptitle(args.title, fontsize=12, weight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.97))
    else:
        fig.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight", facecolor="white")
    print(f"[visualise_gradcam_grid] wrote {args.output}")


if __name__ == "__main__":
    main()
