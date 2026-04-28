"""Generate Figure 4.1: end-to-end system architecture diagram.

Renders the six-stage pipeline from §4.1 with the shared preprocessing block
feeding both the offline evaluation path and the FastAPI online path, and
the inference branch fanning out to the four model families before video-
level aggregation.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


COLOURS = {
    "input": "#e6f2ff",
    "input_edge": "#3a78bd",
    "prep": "#fff3d6",
    "prep_edge": "#c98a1f",
    "model": "#e6f7ec",
    "model_edge": "#2a8c4a",
    "agg": "#f3e8ff",
    "agg_edge": "#7b3ec2",
    "verdict": "#ffe1e1",
    "verdict_edge": "#b73a3a",
    "group_face": "#fff8e8",
    "group_inf": "#eef8f1",
}


def box(ax, x, y, w, h, text, fc, ec, fontsize=9, weight="normal"):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.4, edgecolor=ec, facecolor=fc,
    )
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text,
            ha="center", va="center", fontsize=fontsize, weight=weight,
            wrap=True)


def arrow(ax, x1, y1, x2, y2, color="#444444"):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.3,
                        shrinkA=2, shrinkB=2),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--dpi", type=int, default=220)
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(13, 8), dpi=args.dpi)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 9)
    ax.axis("off")

    bw, bh = 3.4, 0.7

    box(ax, 1.0, 8.0, bw, bh,
        "Offline batch\n(data/, scripts/evaluate_model.py)",
        COLOURS["input"], COLOURS["input_edge"], fontsize=8.5)
    box(ax, 8.6, 8.0, bw, bh,
        "Online upload\n(POST /api/predict, FastAPI)",
        COLOURS["input"], COLOURS["input_edge"], fontsize=8.5)

    grp_x, grp_y, grp_w, grp_h = 0.6, 3.6, 11.8, 4.0
    grp = patches.FancyBboxPatch(
        (grp_x, grp_y), grp_w, grp_h,
        boxstyle="round,pad=0.05,rounding_size=0.08",
        linewidth=1.0, edgecolor="#a8862e",
        facecolor=COLOURS["group_face"], alpha=0.55, zorder=0,
    )
    ax.add_patch(grp)
    ax.text(grp_x + 0.15, grp_y + grp_h - 0.25,
            "Shared preprocessing  (src/preprocessing/PreprocessingPipeline)",
            fontsize=9, weight="bold", color="#7b5e16")

    stages = [
        ("(1) VideoProcessor.extract_frames\n32 frames, uniform via numpy.linspace", 6.6),
        ("(2) FaceExtractor / MTCNN\nmin face size 60 px, post_process=False", 5.8),
        ("(3) FaceAligner / MediaPipe FaceMesh\n6 + 6 eye landmarks, arctan, affine warp", 5.0),
        ("(4) Crop and resize\n299x299, margin 0.3, normalise [-1, 1]", 4.2),
    ]
    sx = 4.8
    for text, sy in stages:
        box(ax, sx, sy, bw, bh, text,
            COLOURS["prep"], COLOURS["prep_edge"], fontsize=8.2)

    arrow(ax, 1.0 + bw / 2, 8.0, sx + bw / 2, 6.6 + bh)
    arrow(ax, 8.6 + bw / 2, 8.0, sx + bw / 2, 6.6 + bh)
    for i in range(len(stages) - 1):
        _, sy_top = stages[i]
        _, sy_bot = stages[i + 1]
        arrow(ax, sx + bw / 2, sy_top, sx + bw / 2, sy_bot + bh)

    inf_grp_x, inf_grp_y, inf_grp_w, inf_grp_h = 0.6, 1.6, 11.8, 1.6
    inf_grp = patches.FancyBboxPatch(
        (inf_grp_x, inf_grp_y), inf_grp_w, inf_grp_h,
        boxstyle="round,pad=0.05,rounding_size=0.08",
        linewidth=1.0, edgecolor="#2a8c4a",
        facecolor=COLOURS["group_inf"], alpha=0.55, zorder=0,
    )
    ax.add_patch(inf_grp)
    ax.text(inf_grp_x + 0.15, inf_grp_y + inf_grp_h - 0.22,
            "(5) Inference  (src/models/, four checkpoints loaded once at startup)",
            fontsize=9, weight="bold", color="#1f6536")

    mw, mh = 2.5, 0.6
    model_y = 1.85
    models = [
        ("Mixed\nXception 20.8M", 0.85),
        ("Robust\nXception 20.8M (FT)", 3.55),
        ("Ensemble\nXception + EfficientNet", 6.45),
        ("Temporal\nXception + LSTM", 9.35),
    ]
    for label, mx in models:
        box(ax, mx, model_y, mw, mh, label,
            COLOURS["model"], COLOURS["model_edge"], fontsize=8)

    fan_top_y = 4.2
    fan_bot_y = model_y + mh
    for label, mx in models:
        arrow(ax, sx + bw / 2, fan_top_y, mx + mw / 2, fan_top_y - 0.15)
        arrow(ax, mx + mw / 2, fan_top_y - 0.15, mx + mw / 2, fan_bot_y)

    agg_w, agg_h = 4.2, 0.65
    agg_x = (13 - agg_w) / 2
    agg_y = 0.85
    box(ax, agg_x, agg_y, agg_w, agg_h,
        "(6) Video-level aggregation\nmean over softmax(fake), threshold 0.5, inconclusive band [0.4, 0.6]",
        COLOURS["agg"], COLOURS["agg_edge"], fontsize=8.5)
    for label, mx in models:
        arrow(ax, mx + mw / 2, model_y, agg_x + agg_w / 2, agg_y + agg_h)

    vw, vh = 4.2, 0.55
    vx = (13 - vw) / 2
    vy = 0.05
    box(ax, vx, vy, vw, vh,
        "Verdict   {REAL, INCONCLUSIVE, FAKE}   +  p(fake)  +  optional Grad-CAM",
        COLOURS["verdict"], COLOURS["verdict_edge"], fontsize=8.5, weight="bold")
    arrow(ax, agg_x + agg_w / 2, agg_y, vx + vw / 2, vy + vh)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight", facecolor="white")
    print(f"[visualise_architecture] wrote {args.output}")


if __name__ == "__main__":
    main()
