"""Generate Figure 5.5: stacked stage-breakdown bar chart for all four model
families. Loads the InferenceService once, runs the same per-stage profiler
the standalone script uses (decode, detect+align, tensor prep, model
forward), aggregates over `runs` repetitions, saves JSON, and renders a
single matplotlib stacked-bar figure with end-to-end FPS annotated above
each bar."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from src.api.inference import InferenceService

MODELS = ["mixed", "robust", "ensemble", "temporal"]


def time_it(fn, *args, sync_cuda=False, **kwargs):
    if sync_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    if sync_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()
    return out, time.perf_counter() - t0


def profile_one(svc, video_path, model_name):
    pipeline = svc.pipeline
    _, t_info = time_it(pipeline.video_processor.get_video_info, video_path)
    frames, t_frames = time_it(
        pipeline.video_processor.extract_frames, video_path, pipeline.frames_per_video
    )

    process_times = []
    faces = []
    for frame in frames:
        face, dt = time_it(pipeline.process_frame, frame)
        process_times.append(dt)
        if face is not None:
            faces.append(face)

    if faces:
        tensor, t_tensor = time_it(svc._faces_to_tensor, faces)
    else:
        tensor, t_tensor = None, 0.0

    t_model = 0.0
    if tensor is not None:
        model = svc.models[model_name]
        if model_name == "temporal":
            _, t_model = time_it(svc._infer_temporal, model, tensor,
                                 sync_cuda=True, use_tta=False)
        else:
            _, t_model = time_it(svc._infer_framewise, model, tensor,
                                 sync_cuda=True, use_tta=False)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    svc.predict(video_path, model_name=model_name, use_tta=False, include_gradcam=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_e2e = time.perf_counter() - t0

    return {
        "frames_decoded": len(frames),
        "faces_detected": len(faces),
        "t_decode": t_info + t_frames,
        "t_detect_align": sum(process_times),
        "t_tensor": t_tensor,
        "t_model": t_model,
        "t_e2e": t_e2e,
    }


def aggregate(rows):
    keys = ["frames_decoded", "faces_detected",
            "t_decode", "t_detect_align", "t_tensor", "t_model", "t_e2e"]
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=False, default="")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--json-out", required=True)
    ap.add_argument("--fig-out", required=True)
    ap.add_argument("--dpi", type=int, default=220)
    ap.add_argument("--from-json", action="store_true",
                    help="Skip profiling, read JSON, just re-render figure.")
    args = ap.parse_args()

    if args.from_json:
        with open(args.json_out, "r", encoding="utf-8") as f:
            payload = json.load(f)
        results = payload["models"]
        args.video = payload.get("video", args.video)
        print(f"[profile_all] Loaded existing JSON, skipping profile run.")
    else:
        with open(PROJECT_ROOT / "config" / "config.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        print(f"[profile_all] CUDA={torch.cuda.is_available()}, loading InferenceService...")
        svc = InferenceService(config)
        print(f"[profile_all] Loaded models: {svc.loaded_model_names()}")

        results = {}
        for m in MODELS:
            if m not in svc.loaded_model_names():
                print(f"[profile_all] Skipping '{m}' (not loaded).")
                continue
            print(f"[profile_all] Warming up '{m}'...")
            for _ in range(args.warmup):
                profile_one(svc, args.video, m)
            rows = []
            for i in range(args.runs):
                row = profile_one(svc, args.video, m)
                rows.append(row)
                print(f"[profile_all]   {m} run {i + 1}/{args.runs}: e2e={row['t_e2e']:.2f}s")
            results[m] = aggregate(rows)

        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump({"video": args.video, "warmup": args.warmup,
                       "runs": args.runs, "models": results}, f, indent=2)
        print(f"[profile_all] JSON saved to {args.json_out}")

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(10, 7.5), dpi=args.dpi,
        gridspec_kw={"height_ratios": [2.4, 1]}, sharex=True,
    )
    model_order = [m for m in MODELS if m in results]
    x = np.arange(len(model_order))

    decode_raw = np.array([results[m]["t_decode"] for m in model_order])
    detect_raw = np.array([results[m]["t_detect_align"] for m in model_order])
    tensor_raw = np.array([results[m]["t_tensor"] for m in model_order])
    model_raw = np.array([results[m]["t_model"] for m in model_order])
    e2e = np.array([results[m]["t_e2e"] for m in model_order])
    frames_decoded = np.array([results[m]["frames_decoded"] for m in model_order])
    fps = np.array([frames_decoded[i] / e2e[i] if e2e[i] > 0 else 0.0
                    for i in range(len(model_order))])

    sum_raw = decode_raw + detect_raw + tensor_raw + model_raw
    scale = np.where(sum_raw > 0, e2e / sum_raw, 1.0)
    decode = decode_raw * scale
    detect = detect_raw * scale
    tensor = tensor_raw * scale
    model_t = model_raw * scale

    width = 0.55
    colours = {
        "decode": "#4c8eda",
        "detect": "#f6a14a",
        "tensor": "#9b6bd1",
        "model":  "#3aa55a",
        "fps":    "#3a78bd",
    }

    ax_top.bar(x, decode, width, label="Video decode + sample", color=colours["decode"])
    ax_top.bar(x, detect, width, bottom=decode,
               label="MTCNN detect + MediaPipe align", color=colours["detect"])
    ax_top.bar(x, tensor, width, bottom=decode + detect,
               label="Tensor prep (CPU to GPU)", color=colours["tensor"])
    ax_top.bar(x, model_t, width, bottom=decode + detect + tensor,
               label="Model forward", color=colours["model"])

    for i in range(len(model_order)):
        ax_top.text(x[i], e2e[i] + 0.05,
                    f"{e2e[i]:.2f} s",
                    ha="center", va="bottom", fontsize=9, weight="bold")

    ax_top.set_ylabel("Wall-time per video (s)\n(shorter = faster)", fontsize=10)
    ax_top.set_title(f"Inference stage breakdown over {args.runs} runs on "
                     f"{Path(args.video).name}", fontsize=11)
    ax_top.set_ylim(0, max(e2e) * 1.20)
    ax_top.grid(axis="y", linestyle=":", alpha=0.4)
    ax_top.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
                  fontsize=8.5, frameon=False, borderaxespad=0)

    bars_fps = ax_bot.bar(x, fps, width, color=colours["fps"], edgecolor="#1f4f80")
    for i, b in enumerate(bars_fps):
        ax_bot.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.15,
                    f"{fps[i]:.1f} FPS",
                    ha="center", va="bottom", fontsize=9, weight="bold")
    ax_bot.axhline(y=10.0, linestyle="--", color="#b73a3a",
                   linewidth=1.2, alpha=0.8)
    ax_bot.text(len(model_order) - 0.5, 10.15,
                "10 FPS interim-report target",
                ha="right", va="bottom", fontsize=8, color="#b73a3a")

    ax_bot.set_ylabel("Throughput (FPS)\n(taller = faster)", fontsize=10)
    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels([m.capitalize() for m in model_order], fontsize=10)
    ax_bot.set_ylim(0, max(max(fps) * 1.25, 12))
    ax_bot.grid(axis="y", linestyle=":", alpha=0.4)

    fig.tight_layout()
    Path(args.fig_out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.fig_out, bbox_inches="tight", facecolor="white")
    print(f"[profile_all] Figure saved to {args.fig_out}")


if __name__ == "__main__":
    main()
