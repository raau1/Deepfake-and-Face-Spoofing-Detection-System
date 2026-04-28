"""Generate Figure 6.13: DFDC out-of-family AUC bar chart for the four
model families, overlaid with the Dolhansky et al. published XceptionNet
baseline band (0.65 to 0.72)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

ORDER = [
    ("mixed",    "Mixed v2",    "#4c8eda"),
    ("ensemble", "Ensemble v3", "#3aa55a"),
    ("robust",   "Robust v3",   "#f6a14a"),
    ("temporal", "Temporal v3", "#9b6bd1"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    aucs = []
    accs = []
    for slug, _, _ in ORDER:
        with open(Path(args.results_dir) / f"dfdc_eval_{slug}.json", "r",
                  encoding="utf-8") as f:
            d = json.load(f)["summary"]
        aucs.append(d["auc"])
        accs.append(d["video_accuracy"])

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=args.dpi)

    band = patches.Rectangle((-0.6, 0.65), len(ORDER) + 1.2, 0.07,
                             linewidth=0, facecolor="#b73a3a", alpha=0.13,
                             zorder=0)
    ax.add_patch(band)
    ax.text(2.5, 0.76,
            "Dolhansky et al. XceptionNet baseline band (0.65 to 0.72)",
            ha="center", va="center", fontsize=9, color="#7a2828",
            style="italic")

    x = np.arange(len(ORDER))
    colours = [c for _, _, c in ORDER]
    labels = [lbl for _, lbl, _ in ORDER]
    bars = ax.bar(x, aucs, width=0.55, color=colours, edgecolor="#222")

    for i, b in enumerate(bars):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.008,
                f"{aucs[i]:.4f}",
                ha="center", va="bottom", fontsize=10, weight="bold")

    ax.axhline(y=0.5, linestyle=":", color="#444", linewidth=1.0, alpha=0.7)
    ax.text(len(ORDER) - 0.05, 0.5, "random (AUC 0.5)",
            ha="left", va="center", fontsize=8, color="#444",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=2))

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("AUC-ROC on DFDC out-of-family clips", fontsize=10)
    ax.set_title("DFDC out-of-family AUC by model family (676 balanced clips)",
                 fontsize=11)
    ax.set_xlim(-0.6, len(ORDER) + 0.6)
    ax.set_ylim(0.4, 0.85)
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight", facecolor="white")
    print(f"[visualise_dfdc_bars] wrote {args.output}")


if __name__ == "__main__":
    main()
