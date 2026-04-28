"""Generate Figure 6.10: 2x4 grid of the 8 YouTube deepfake test-clip
frames with v3 robust Grad-CAM overlays, annotated with the per-frame
p(fake) values reported in section 6.10.2."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

P_FAKE = [0.027, 0.273, 0.423, 0.184, 0.262, 0.206, 0.271, 0.289]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--dpi", type=int, default=180)
    args = ap.parse_args()

    src = Path(args.dir)
    files = sorted(src.glob("deepfake_degrass_f*.png"))
    if len(files) < 8:
        raise RuntimeError(f"Expected 8 frames in {src}, found {len(files)}")

    fig, axes = plt.subplots(2, 4, figsize=(15, 6.5), dpi=args.dpi)
    for i, ax in enumerate(axes.flatten()):
        img = Image.open(files[i]).convert("RGB")
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        verdict = "FAKE" if P_FAKE[i] > 0.5 else "REAL"
        colour = "#b73a3a" if verdict == "FAKE" else "#1f6536"
        ax.set_title(f"frame {i}   p(fake) = {P_FAKE[i]:.3f}   verdict = {verdict}",
                     fontsize=10, color=colour)

    fig.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight", facecolor="white")
    print(f"[visualise_youtube_gradcam] wrote {args.output}")


if __name__ == "__main__":
    main()
