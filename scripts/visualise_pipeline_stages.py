"""Generate Figure 3.4: four-panel preprocessing pipeline visualisation.

Loads one frame from a FaceForensics++ video, runs MTCNN detection,
MediaPipe FaceMesh landmark extraction, and the project's FaceAligner,
then renders the four pipeline stages side-by-side.

Usage:
    venv/Scripts/python.exe scripts/visualise_pipeline_stages.py \\
        --video data/FaceForensics++/original/000.mp4 \\
        --frame-index 60 \\
        --output outputs/figures/fig_3_4_pipeline.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import torch
from facenet_pytorch import MTCNN

from src.preprocessing.face_aligner import FaceAligner


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--frame-index", type=int, default=60)
    ap.add_argument("--output", required=True)
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame_index)
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {args.frame_index} from {args.video}")
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    H, W = frame_rgb.shape[:2]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mtcnn = MTCNN(keep_all=False, device=device, post_process=False)
    boxes, probs = mtcnn.detect(frame_rgb)
    if boxes is None or len(boxes) == 0:
        raise RuntimeError("MTCNN found no face in the selected frame.")
    bbox = boxes[0].astype(float)

    fm = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
    )
    res = fm.process(frame_rgb)
    fm.close()
    if not res.multi_face_landmarks:
        raise RuntimeError("FaceMesh found no face in the selected frame.")
    lms = res.multi_face_landmarks[0].landmark
    pts = np.array([[lm.x * W, lm.y * H] for lm in lms], dtype=np.float32)

    fw, fh = bbox[2] - bbox[0], bbox[3] - bbox[1]
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    crop_size = max(fw, fh) * 1.6
    x1 = max(0, int(cx - crop_size / 2))
    y1 = max(0, int(cy - crop_size / 2))
    x2 = min(W, int(cx + crop_size / 2))
    y2 = min(H, int(cy + crop_size / 2))
    face_crop = frame_rgb[y1:y2, x1:x2]
    pts_in_crop = pts.copy()
    pts_in_crop[:, 0] -= x1
    pts_in_crop[:, 1] -= y1

    aligner = FaceAligner(output_size=299, margin=0.3)
    aligned_bgr = aligner.align_face(frame_bgr, landmarks=pts, output_size=299)
    if aligned_bgr is None:
        raise RuntimeError("FaceAligner.align_face returned None.")
    aligned_rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), dpi=args.dpi)

    axes[0].imshow(frame_rgb)
    axes[0].set_title(f"(a) Raw frame {args.frame_index}", fontsize=10)

    axes[1].imshow(frame_rgb)
    rect = patches.Rectangle(
        (bbox[0], bbox[1]), fw, fh,
        linewidth=2.0, edgecolor="#39ff14", facecolor="none",
    )
    axes[1].add_patch(rect)
    axes[1].text(
        bbox[0], bbox[1] - 8,
        f"MTCNN p={float(probs[0]):.3f}",
        fontsize=8, color="black",
        bbox=dict(facecolor="#39ff14", alpha=0.85, pad=2, edgecolor="none"),
    )
    axes[1].set_title("(b) MTCNN detection", fontsize=10)

    axes[2].imshow(face_crop)
    axes[2].scatter(pts_in_crop[:, 0], pts_in_crop[:, 1],
                    s=3, c="#ff3366", alpha=0.85, edgecolors="none")
    axes[2].set_title("(c) MediaPipe FaceMesh (468 landmarks)", fontsize=10)

    axes[3].imshow(aligned_rgb)
    axes[3].set_title("(d) Aligned 299x299 crop", fontsize=10)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight", facecolor="white")
    print(f"[visualise_pipeline_stages] wrote {args.output}")


if __name__ == "__main__":
    main()
