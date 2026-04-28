"""Profile the InferenceService pipeline to find the speed bottleneck.

Breaks down per-video wall-time into: video decode + frame sampling,
per-frame detect+align (MTCNN inside extract_face + MediaPipe inside
align_face), tensor prep, and model inference. Also times a real
end-to-end svc.predict() call, which is the number that matters for
the interim-report 10+ FPS claim.

Usage:
    venv/Scripts/python.exe scripts/profile_inference.py \\
        --video data/dfdc/dfdc_train_part_1/aassnaulhq.mp4 \\
        --model robust --warmup 1 --runs 3
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import yaml

from src.api.inference import InferenceService


def time_it(fn, *args, sync_cuda=False, **kwargs):
    if sync_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    if sync_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()
    return out, time.perf_counter() - t0


def profile_stages(svc: InferenceService, video_path: str, model_name: str):
    """Stage-by-stage breakdown that mirrors the production predict() path."""
    pipeline = svc.pipeline

    # --- Stage 1: video decode + frame sampling ---
    _, t_info = time_it(pipeline.video_processor.get_video_info, video_path)
    frames, t_frames = time_it(
        pipeline.video_processor.extract_frames, video_path, pipeline.frames_per_video
    )

    # --- Stage 2: detect + align, per frame (== process_frame, production path) ---
    process_times: list[float] = []
    faces = []
    for frame in frames:
        face, dt = time_it(pipeline.process_frame, frame)
        process_times.append(dt)
        if face is not None:
            faces.append(face)

    # --- Stage 3: tensor prep ---
    if faces:
        tensor, t_tensor = time_it(svc._faces_to_tensor, faces)
    else:
        tensor, t_tensor = None, 0.0

    # --- Stage 4: model inference (single pass, no TTA) ---
    t_model = 0.0
    if tensor is not None:
        model = svc.models[model_name]
        if model_name == "temporal":
            _, t_model = time_it(svc._infer_temporal, model, tensor, sync_cuda=True, use_tta=False)
        else:
            _, t_model = time_it(svc._infer_framewise, model, tensor, sync_cuda=True, use_tta=False)

    return {
        "frames_decoded": len(frames),
        "faces_detected": len(faces),
        "t_info": t_info,
        "t_frames": t_frames,
        "t_process_total": sum(process_times),
        "t_process_mean": float(np.mean(process_times)) if process_times else 0.0,
        "t_tensor": t_tensor,
        "t_model": t_model,
    }


def profile_end_to_end(svc: InferenceService, video_path: str, model_name: str):
    """Time a real svc.predict() call — this is the production path."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = svc.predict(video_path, model_name=model_name, use_tta=False, include_gradcam=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return result, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", default="robust",
                    choices=["mixed", "robust", "ensemble", "temporal"])
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()

    with open(PROJECT_ROOT / "config" / "config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print(f"[profile] Loading InferenceService (CUDA={torch.cuda.is_available()})...")
    svc = InferenceService(config)
    print(f"[profile] Loaded models: {svc.loaded_model_names()}")

    # Warmup
    for i in range(args.warmup):
        print(f"[profile] Warmup {i + 1}/{args.warmup}...")
        profile_stages(svc, args.video, args.model)
        profile_end_to_end(svc, args.video, args.model)

    # Stage breakdown runs
    rows = []
    for i in range(args.runs):
        t0 = time.perf_counter()
        row = profile_stages(svc, args.video, args.model)
        row["t_wall"] = time.perf_counter() - t0
        rows.append(row)
        print(f"[profile] Stage-breakdown run {i + 1}/{args.runs}: wall={row['t_wall']:.2f}s")

    # End-to-end predict() runs
    e2e_times = []
    for i in range(args.runs):
        _, t_e2e = profile_end_to_end(svc, args.video, args.model)
        e2e_times.append(t_e2e)
        print(f"[profile] End-to-end predict() run {i + 1}/{args.runs}: {t_e2e:.2f}s")

    def avg(key):
        return float(np.mean([r[key] for r in rows]))

    wall = avg("t_wall")
    n_det = avg("frames_decoded")
    n_face = avg("faces_detected")
    e2e_mean = float(np.mean(e2e_times))

    print()
    print(f"=== Average over {args.runs} runs on {args.video} (model={args.model}) ===")
    print(f"Frames decoded:    {n_det:.0f}")
    print(f"Faces detected:    {n_face:.0f}")
    print(f"Stage-sum wall:    {wall:.3f}s")
    print(f"predict() wall:    {e2e_mean:.3f}s   <-- production number")
    print()

    def pct(x):
        return 100.0 * x / wall if wall > 0 else 0.0

    print(f"{'Stage':<32} {'Time (s)':>10} {'% of wall':>10} {'Per-frame (ms)':>18}")
    print("-" * 72)
    for label, k, per_frame_key in [
        ("Video info + decode", "t_frames", None),
        ("Detect + align (process_frame)", "t_process_total", "t_process_mean"),
        ("Tensor prep (CPU->GPU)", "t_tensor", None),
        ("Model inference", "t_model", None),
    ]:
        t = avg(k)
        per_frame = (avg(per_frame_key) * 1000) if per_frame_key else (t / n_det * 1000 if n_det else 0)
        print(f"{label:<32} {t:>10.3f} {pct(t):>9.1f}% {per_frame:>17.1f}")

    covered = sum(avg(k) for _, k, _ in [
        (None, "t_frames", None),
        (None, "t_process_total", None),
        (None, "t_tensor", None),
        (None, "t_model", None),
    ])
    print(f"{'Sum of stages':<32} {covered:>10.3f} {pct(covered):>9.1f}%")
    print(f"{'(Unaccounted: Python/GC)':<32} {wall - covered:>10.3f} {pct(wall - covered):>9.1f}%")

    fps_stage = n_det / wall if wall > 0 else 0.0
    fps_e2e = n_det / e2e_mean if e2e_mean > 0 else 0.0
    print()
    print(f"Throughput (stage-sum):   {fps_stage:.2f} FPS")
    print(f"Throughput (predict()):   {fps_e2e:.2f} FPS   <-- report this")
    print(f"Interim-report target: 10+ FPS  -->  "
          f"{'HIT' if fps_e2e >= 10 else f'MISS (factor {10 / fps_e2e:.2f}x short)'}")


if __name__ == "__main__":
    main()
