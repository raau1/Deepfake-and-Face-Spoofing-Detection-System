"""
DFDC (Deepfake Detection Challenge) Out-of-Family Evaluation.

Runs one of the v3 checkpoints (robust / ensemble / temporal) across a balanced
sample of DFDC clips and reports AUC, EER, and video-level accuracy in the same
table shape as the per-dataset evaluations in §6.10.3. Unlike FF++ / Celeb-DF /
DFD / WildDeepfake - all of which share at least one generator family with this
project's training mix - DFDC was produced with DFAE, MM/NN, NTH, FSGAN, and
StyleGAN-based face-swaps, none of which appear in training. This is therefore
the first genuinely out-of-family evaluation in the project.

Label source: each DFDC partition ships with a metadata.json keyed by filename,
containing {"label": "FAKE"|"REAL", "split": "train", "original": <real_fname>}
for fakes. The script reads labels directly from this file rather than from
folder structure.

Sampling: DFDC is ~81% FAKE by count, so random sampling inflates AUC via
class-prior imbalance. This script defaults to balanced sampling (equal REAL /
FAKE) with a fixed seed for reproducibility.

Per-clip efficiency: faces are extracted *once* per clip via the shared
PreprocessingPipeline, then the same [T, 3, 299, 299] tensor is passed through
every requested model. This makes a 3-model sweep roughly 3× faster than
calling InferenceService.predict() three times per clip.

Usage:
    # Balanced 400-clip sweep across all three v3 checkpoints:
    python scripts/evaluate_dfdc.py \
        --dfdc-dir data/dfdc/dfdc_train_part_01 \
        --num-clips 400

    # Smaller first-pass sanity check:
    python scripts/evaluate_dfdc.py \
        --dfdc-dir data/dfdc/dfdc_train_part_01 \
        --num-clips 100 \
        --models temporal

    # Save worst misclassifications for qualitative follow-up (Grad-CAM figure):
    python scripts/evaluate_dfdc.py \
        --dfdc-dir data/dfdc/dfdc_train_part_01 \
        --num-clips 400 \
        --save-failures
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import yaml
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm

from src.api.inference import InferenceService


def load_dfdc_metadata(dfdc_dirs: List[Path]) -> Dict[str, Dict]:
    """Pool metadata across one or more DFDC partition directories.

    Returns {mp4_filename: {'label': 'FAKE'|'REAL', 'dir': Path}} where 'dir'
    records which partition the file came from so the caller can resolve the
    full path later. Duplicate filenames across partitions keep the first
    entry and warn (DFDC filenames are globally unique, so this should not
    trigger in practice).
    """
    out: Dict[str, Dict] = {}
    for d in dfdc_dirs:
        meta_path = d / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"metadata.json missing at {meta_path}. "
                "Make sure you extracted the DFDC zip fully."
            )
        with open(meta_path) as f:
            meta = json.load(f)
        for name, entry in meta.items():
            if name in out:
                print(f"[!] Duplicate filename across partitions: {name} (keeping first)")
                continue
            out[name] = {"label": entry["label"], "dir": d}
    return out


def sample_balanced(
    metadata: Dict[str, Dict],
    n_per_class: int,
    seed: int,
) -> List[Tuple[str, str, Path]]:
    """Return a shuffled list of (filename, label, source_dir) tuples.

    Sampling is balanced across REAL/FAKE and capped by the smaller class.
    The source_dir is needed because filenames alone are not enough to
    resolve the video path when metadata is pooled from several partitions.
    """
    reals = [k for k, v in metadata.items() if v["label"] == "REAL"]
    fakes = [k for k, v in metadata.items() if v["label"] == "FAKE"]
    rng = random.Random(seed)
    rng.shuffle(reals)
    rng.shuffle(fakes)
    n = min(n_per_class, len(reals), len(fakes))
    if n < n_per_class:
        print(
            f"[!] Requested {n_per_class} per class but only {n} available. "
            f"Using {n}. (Real: {len(reals)}, Fake: {len(fakes)})"
        )
    clips = (
        [(f, "REAL", metadata[f]["dir"]) for f in reals[:n]]
        + [(f, "FAKE", metadata[f]["dir"]) for f in fakes[:n]]
    )
    rng.shuffle(clips)
    return clips


def compute_eer(labels: np.ndarray, probs: np.ndarray) -> float:
    """Equal error rate: the operating point where FPR == FNR."""
    fpr, tpr, _ = roc_curve(labels, probs)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[idx] + fnr[idx]) / 2.0)


def evaluate_clip(
    svc: InferenceService,
    video_path: Path,
    models_to_run: List[str],
) -> Dict[str, Dict[str, float]]:
    """Extract faces once and evaluate every requested model on the same tensor.

    Returns {model_name: {"prob": float, "verdict": str, "frames": int}}.
    Raises if face extraction fails so the caller can log and skip.
    """
    faces = svc._extract_faces(str(video_path))
    if not faces:
        raise RuntimeError("no faces detected")
    tensor = svc._faces_to_tensor(faces)  # [T, 3, 299, 299] on device

    out = {}
    for name in models_to_run:
        model = svc.models[name]
        if name == "temporal":
            fake_prob, _ = svc._infer_temporal(model, tensor)
        else:
            fake_prob, _ = svc._infer_framewise(model, tensor)
        verdict = "FAKE" if fake_prob >= svc.threshold else "REAL"
        out[name] = {
            "prob": float(fake_prob),
            "verdict": verdict,
            "frames": len(faces),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="DFDC out-of-family evaluation")
    ap.add_argument(
        "--dfdc-dir",
        required=True,
        nargs="+",
        help="One or more extracted dfdc_train_part_XX/ directories "
             "(each must contain .mp4 files + metadata.json). "
             "Metadata is pooled before balanced sampling - pass multiple "
             "partitions to increase the REAL-class ceiling.",
    )
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument(
        "--models",
        nargs="+",
        default=["robust", "ensemble", "temporal"],
        help="Which FastAPI checkpoint slots to evaluate (default: all three v3)",
    )
    ap.add_argument(
        "--num-clips",
        type=int,
        default=400,
        help="Total clips, split 50/50 REAL/FAKE (default: 400)",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", default="outputs/results")
    ap.add_argument(
        "--save-failures",
        action="store_true",
        help="Also write the top-10 worst-misclassified clips per model",
    )
    args = ap.parse_args()

    dfdc_dirs = [Path(d).resolve() for d in args.dfdc_dir]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- metadata + sampling ----------
    print(f"[*] Loading DFDC metadata from {len(dfdc_dirs)} partition(s):")
    for d in dfdc_dirs:
        print(f"      {d}")
    meta = load_dfdc_metadata(dfdc_dirs)
    n_real = sum(1 for v in meta.values() if v["label"] == "REAL")
    n_fake = sum(1 for v in meta.values() if v["label"] == "FAKE")
    print(f"    Pooled: {len(meta)} total ({n_real} REAL, {n_fake} FAKE)")

    n_per_class = args.num_clips // 2
    clips = sample_balanced(meta, n_per_class, args.seed)
    print(f"[*] Sampled {len(clips)} clips balanced across REAL/FAKE (seed={args.seed})")

    # ---------- models ----------
    with open(args.config) as f:
        config = yaml.safe_load(f)

    print("[*] Loading models via InferenceService (reads config.api.checkpoints)...")
    svc = InferenceService(config=config)
    available = set(svc.loaded_model_names())
    requested = set(args.models)
    missing = requested - available
    if missing:
        print(f"[!] Requested but not loaded: {sorted(missing)} - skipping")
    models_to_run = [m for m in args.models if m in available]
    if not models_to_run:
        print("[!] No requested models are loaded. Aborting.")
        sys.exit(1)
    print(f"[*] Evaluating: {models_to_run}")

    # ---------- inference loop ----------
    results: Dict[str, Dict[str, Dict]] = {m: {} for m in models_to_run}
    failed = []  # (filename, reason)

    for fname, label, src_dir in tqdm(clips, desc="Clips"):
        video_path = src_dir / fname
        if not video_path.exists():
            failed.append((fname, "file not found"))
            continue
        try:
            per_model = evaluate_clip(svc, video_path, models_to_run)
        except Exception as exc:
            failed.append((fname, str(exc)))
            continue
        for m, r in per_model.items():
            results[m][fname] = {"label": label, "partition": src_dir.name, **r}

    # ---------- metrics ----------
    summary = {}
    for m in models_to_run:
        y_true, y_prob, y_pred = [], [], []
        for fname, r in results[m].items():
            y_true.append(1 if r["label"] == "FAKE" else 0)
            y_prob.append(r["prob"])
            y_pred.append(1 if r["verdict"] == "FAKE" else 0)
        y_true = np.array(y_true, dtype=int)
        y_prob = np.array(y_prob, dtype=float)
        y_pred = np.array(y_pred, dtype=int)
        both_classes_present = len(np.unique(y_true)) > 1
        auc = roc_auc_score(y_true, y_prob) if both_classes_present else float("nan")
        eer = compute_eer(y_true, y_prob) if both_classes_present else float("nan")
        acc = float((y_pred == y_true).mean()) if len(y_true) else float("nan")
        summary[m] = {
            "n_clips": int(len(y_true)),
            "auc": float(auc),
            "eer": float(eer),
            "video_accuracy": float(acc),
            "n_failed": sum(1 for f in failed),
        }

    # ---------- persist per-model JSON ----------
    for m in models_to_run:
        path = out_dir / f"dfdc_eval_{m}.json"
        with open(path, "w") as f:
            json.dump(
                {
                    "model": m,
                    "dfdc_partitions": [str(d) for d in dfdc_dirs],
                    "num_clips_sampled": len(clips),
                    "n_per_class": n_per_class,
                    "seed": args.seed,
                    "summary": summary[m],
                    "per_clip": results[m],
                    "failed_clips": failed,
                },
                f,
                indent=2,
            )
        print(f"[+] Wrote {path}")

    # ---------- combined markdown table ----------
    md_path = out_dir / "dfdc_eval_summary.md"
    with open(md_path, "w") as f:
        f.write("# DFDC Out-of-Family Evaluation\n\n")
        f.write(f"- Partitions: {', '.join(f'`{d.name}`' for d in dfdc_dirs)}\n")
        f.write(
            f"- Clips sampled: {len(clips)} "
            f"(balanced {n_per_class} REAL / {n_per_class} FAKE)\n"
        )
        f.write(f"- Seed: {args.seed}\n")
        f.write(f"- Failed clips: {len(failed)}\n\n")
        f.write("| Model | Clips | AUC-ROC | EER | Video Acc |\n")
        f.write("|---|---|---|---|---|\n")
        for m in models_to_run:
            s = summary[m]
            f.write(
                f"| {m} | {s['n_clips']} | {s['auc']:.4f} | "
                f"{s['eer']:.4f} | {s['video_accuracy']:.4f} |\n"
            )
    print(f"[+] Wrote {md_path}")

    # ---------- optional: worst misclassifications ----------
    if args.save_failures:
        fail_md = out_dir / "dfdc_eval_worst_cases.md"
        with open(fail_md, "w") as f:
            f.write("# DFDC - Worst Misclassifications (top 10 per model)\n\n")
            f.write(
                "Candidates for qualitative follow-up. "
                "Fakes-classified-as-real are false negatives; "
                "reals-classified-as-fake are false positives.\n\n"
            )
            for m in models_to_run:
                f.write(f"## {m}\n\n")
                rec = results[m]
                fakes = [(k, r) for k, r in rec.items() if r["label"] == "FAKE"]
                reals = [(k, r) for k, r in rec.items() if r["label"] == "REAL"]
                fakes.sort(key=lambda x: x[1]["prob"])  # lowest p(fake) first
                reals.sort(key=lambda x: -x[1]["prob"])  # highest p(fake) first
                f.write("**Fakes confidently labelled REAL (false negatives):**\n\n")
                for k, r in fakes[:10]:
                    f.write(f"- `{k}` - p(fake) = {r['prob']:.4f}\n")
                f.write("\n**Reals confidently labelled FAKE (false positives):**\n\n")
                for k, r in reals[:10]:
                    f.write(f"- `{k}` - p(fake) = {r['prob']:.4f}\n")
                f.write("\n")
        print(f"[+] Wrote {fail_md}")

    # ---------- console summary ----------
    if failed:
        print(f"\n[!] {len(failed)} clip(s) failed to process "
              f"(see failed_clips in per-model JSON)")

    print("\n" + "=" * 62)
    print("DFDC Out-of-Family Evaluation Summary")
    print("=" * 62)
    print(f"{'Model':<12} {'Clips':<8} {'AUC':<10} {'EER':<10} {'VideoAcc':<10}")
    for m in models_to_run:
        s = summary[m]
        print(
            f"{m:<12} {s['n_clips']:<8} "
            f"{s['auc']:<10.4f} {s['eer']:<10.4f} {s['video_accuracy']:<10.4f}"
        )
    print("=" * 62)


if __name__ == "__main__":
    main()
