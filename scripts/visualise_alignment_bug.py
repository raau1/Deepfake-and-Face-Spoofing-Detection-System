"""Generate Figure 5.1: side-by-side v0.1.0 (flipped) vs v0.2.0 (correct)
alignment, by reproducing the v0.1.0 bug (swapped LEFT/RIGHT eye index
arrays) on the same frame the correct aligner is run against. The original
upside-down crops were overwritten when the dataset was re-preprocessed,
so the v0.1.0 panel is reconstructed from the documented bug rather than
loaded from the archive directory."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np

from src.preprocessing.face_aligner import FaceAligner


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--frame-index", type=int, default=60)
    ap.add_argument("--output", required=True)
    ap.add_argument("--dpi", type=int, default=220)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame_index)
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {args.frame_index} from {args.video}")
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    H, W = frame_rgb.shape[:2]

    fm = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1,
        refine_landmarks=False, min_detection_confidence=0.5,
    )
    res = fm.process(frame_rgb)
    fm.close()
    if not res.multi_face_landmarks:
        raise RuntimeError("FaceMesh found no face in the selected frame.")
    lms = res.multi_face_landmarks[0].landmark
    pts = np.array([[lm.x * W, lm.y * H] for lm in lms], dtype=np.float32)

    aligner_correct = FaceAligner(output_size=299, margin=0.3)
    aligned_correct_bgr = aligner_correct.align_face(frame_bgr, landmarks=pts, output_size=299)

    aligner_bug = FaceAligner(output_size=299, margin=0.3)
    aligner_bug.LEFT_EYE_INDICES = FaceAligner.RIGHT_EYE_INDICES
    aligner_bug.RIGHT_EYE_INDICES = FaceAligner.LEFT_EYE_INDICES
    aligned_bug_bgr = aligner_bug.align_face(frame_bgr, landmarks=pts, output_size=299)

    if aligned_correct_bgr is None or aligned_bug_bgr is None:
        raise RuntimeError("Aligner returned None for one of the two crops.")

    aligned_correct_rgb = cv2.cvtColor(aligned_correct_bgr, cv2.COLOR_BGR2RGB)
    aligned_bug_rgb = cv2.cvtColor(aligned_bug_bgr, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4.5), dpi=args.dpi)

    axes[0].imshow(aligned_bug_rgb)
    axes[0].set_title("(a) v0.1.0 - swapped eye indices",
                      fontsize=10, color="#b73a3a")
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    axes[1].imshow(aligned_correct_rgb)
    axes[1].set_title("(b) v0.2.0 - corrected eye indices",
                      fontsize=10, color="#1f6536")
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    fig.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight", facecolor="white")
    print(f"[visualise_alignment_bug] wrote {args.output}")


if __name__ == "__main__":
    main()
