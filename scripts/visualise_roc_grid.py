"""Render a 2x2 grid of ROC curves for one model across the four evaluation
datasets. Reads `roc_curve` arrays + AUC from the per-dataset evaluation
JSONs and overlays the random-classifier diagonal for reference. Used for
Figures 6.4 (ensemble v3), 6.5 (temporal v3), and 6.7 (robust v2)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

DATASETS = [
    ("ff++",          "FaceForensics++"),
    ("celebdf",       "Celeb-DF v2"),
    ("dfd",           "DeepFakeDetection"),
    ("wilddeepfake",  "WildDeepfake"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True,
                    help="Directory containing <model>_eval_<dataset>/evaluation_results_<dataset>.json")
    ap.add_argument("--prefix", required=True,
                    help="Subdirectory prefix, e.g. 'ensemble' for ensemble_eval_<dataset>/")
    ap.add_argument("--title", required=True,
                    help="Figure suptitle, e.g. 'Ensemble v3 ROC curves'")
    ap.add_argument("--output", required=True)
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    fig, axes = plt.subplots(2, 2, figsize=(10, 8.5), dpi=args.dpi)
    axes = axes.flatten()

    for ax, (slug, label) in zip(axes, DATASETS):
        json_path = Path(args.results_dir) / f"{args.prefix}_eval_{slug}" / f"evaluation_results_{slug}.json"
        if not json_path.exists():
            ax.set_title(f"{label}: missing", fontsize=10, color="red")
            ax.axis("off")
            continue
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        fpr = data["roc_curve"]["fpr"]
        tpr = data["roc_curve"]["tpr"]
        auc = data["frame_level"]["auc"]

        ax.plot(fpr, tpr, color="#1f77b4", linewidth=1.6,
                label=f"ROC (AUC = {auc:.4f})")
        ax.plot([0, 1], [0, 1], linestyle="--", color="#444444",
                linewidth=1.0, label="Random classifier")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.005)
        ax.set_xlabel("False Positive Rate", fontsize=9)
        ax.set_ylabel("True Positive Rate", fontsize=9)
        ax.set_title(f"{label} (AUC = {auc * 100:.2f}%)", fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.legend(loc="lower right", fontsize=8.5, framealpha=0.95)

    fig.suptitle(args.title, fontsize=12, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight", facecolor="white")
    print(f"[visualise_roc_grid] wrote {args.output}")


if __name__ == "__main__":
    main()
