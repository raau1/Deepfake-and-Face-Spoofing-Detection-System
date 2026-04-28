"""Generate Figure 3.3: MediaPipe FaceMesh 468-landmark topology overlaid on a
reference face, with the two 6-landmark eye clusters used by FaceAligner
highlighted.

Usage (inside the API container):
    docker exec deepfake-api python scripts/visualise_facemesh.py \\
        --input  /app/scripts/_tmp/face_in.jpg \\
        --output /app/outputs/uploads/fig_3_3_facemesh.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np

LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to a frontal face image (RGB or BGR).")
    ap.add_argument("--output", required=True, help="Where to save the annotated PNG.")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    img_bgr = cv2.imread(args.input)
    if img_bgr is None:
        raise FileNotFoundError(args.input)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
    )
    result = face_mesh.process(img_rgb)
    face_mesh.close()

    if not result.multi_face_landmarks:
        raise RuntimeError("FaceMesh found no face in the input image.")

    lms = result.multi_face_landmarks[0].landmark
    pts = np.array([[lm.x * w, lm.y * h] for lm in lms], dtype=np.float32)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=args.dpi)
    ax.imshow(img_rgb)

    ax.scatter(pts[:, 0], pts[:, 1], s=4, c="#cccccc", alpha=0.85,
               edgecolors="none", label="468 FaceMesh landmarks")

    le = pts[LEFT_EYE_INDICES]
    re = pts[RIGHT_EYE_INDICES]
    ax.scatter(le[:, 0], le[:, 1], s=42, c="#1f77b4",
               edgecolors="white", linewidths=0.8, zorder=3,
               label=f"Left-eye cluster {LEFT_EYE_INDICES}")
    ax.scatter(re[:, 0], re[:, 1], s=42, c="#d62728",
               edgecolors="white", linewidths=0.8, zorder=3,
               label=f"Right-eye cluster {RIGHT_EYE_INDICES}")

    le_c = le.mean(axis=0)
    re_c = re.mean(axis=0)
    ax.plot([le_c[0], re_c[0]], [le_c[1], re_c[1]],
            color="#ffffff", linewidth=1.5, linestyle="--", alpha=0.9, zorder=2)
    ax.scatter([le_c[0], re_c[0]], [le_c[1], re_c[1]],
               s=70, c="#ffd700", edgecolors="black", linewidths=1.0, zorder=4,
               marker="X", label="Eye centres (cluster means)")

    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.18),
              ncol=1, frameon=False, fontsize=8)
    fig.tight_layout()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight", facecolor="white")
    print(f"[visualise_facemesh] wrote {args.output}")


if __name__ == "__main__":
    main()
