"""Generate Figure 6.6: 2x2 grid comparing v0.1.0 (flipped-alignment) vs
v0.2.0 (corrected) baseline ROC curves on FF++ and Celeb-DF v2. Shows the
alignment bug deflated v0.1.0's apparent in-domain perfection slightly
(FF++) while substantially under-stating cross-dataset capability
(Celeb-DF jumps from 51.85% to 71.63%)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

V01_DIR = Path("outputs/archive/v0.1.0-flipped-alignment/results")
V02_DIR = Path("outputs/results")


def load(json_path: Path):
    with open(json_path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return d["roc_curve"]["fpr"], d["roc_curve"]["tpr"], d["frame_level"]["auc"]


def plot_one(ax, fpr, tpr, auc, title, colour):
    ax.plot(fpr, tpr, color=colour, linewidth=1.7, label=f"ROC (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#444", linewidth=1.0,
            label="Random classifier")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.005)
    ax.set_xlabel("False Positive Rate", fontsize=9)
    ax.set_ylabel("True Positive Rate", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.95)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Project root")
    ap.add_argument("--output", required=True)
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    root = Path(args.root)

    v01_ff = load(root / V01_DIR / "evaluation_results_ff++.json")
    v01_cd = load(root / V01_DIR / "evaluation_results_celebdf.json")
    v02_ff = load(root / V02_DIR / "evaluation_results_ff++.json")
    v02_cd = load(root / V02_DIR / "evaluation_results_celebdf.json")

    fig, axes = plt.subplots(2, 2, figsize=(10, 9), dpi=args.dpi)

    plot_one(axes[0][0], *v01_ff,
             f"v0.1.0 baseline on FF++ (AUC = {v01_ff[2] * 100:.2f}%)",
             "#b73a3a")
    plot_one(axes[0][1], *v02_ff,
             f"v0.2.0 baseline on FF++ (AUC = {v02_ff[2] * 100:.2f}%)",
             "#1f6536")
    plot_one(axes[1][0], *v01_cd,
             f"v0.1.0 baseline on Celeb-DF (AUC = {v01_cd[2] * 100:.2f}%)",
             "#b73a3a")
    plot_one(axes[1][1], *v02_cd,
             f"v0.2.0 baseline on Celeb-DF (AUC = {v02_cd[2] * 100:.2f}%)",
             "#1f6536")

    fig.suptitle("Baseline ROC: v0.1.0 (flipped alignment) vs v0.2.0 (corrected)",
                 fontsize=12, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight", facecolor="white")
    print(f"[visualise_v01_vs_v02] wrote {args.output}")


if __name__ == "__main__":
    main()
