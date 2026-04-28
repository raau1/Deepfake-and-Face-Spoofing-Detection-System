"""
Grad-CAM diagnostic script for deepfake XceptionNet.

Produces heatmap overlays showing which facial regions the classifier attends
to when producing its verdict. Used to diagnose whether the model is looking at
manipulation cues (face interior, blend boundary) or taking shortcuts (hairline,
background, compression artefacts at image edges).

Usage (single image):
    python scripts/gradcam_diagnose.py \
        --checkpoint outputs/models/robust_v2/best_model_robust.pth \
        --image path/to/face.jpg \
        --output outputs/gradcam/one.png

Usage (video - extracts faces with existing MTCNN pipeline first):
    python scripts/gradcam_diagnose.py \
        --checkpoint outputs/models/robust_v2/best_model_robust.pth \
        --video path/to/clip.mp4 \
        --output-dir outputs/gradcam/clip \
        --max-frames 8

Usage (sanity check on FF++ / Celeb-DF):
    python scripts/gradcam_diagnose.py \
        --checkpoint outputs/models/robust_v2/best_model_robust.pth \
        --sanity \
        --output-dir outputs/gradcam/sanity
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from torchvision import transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from src.models.xception import create_xception


def load_model(ckpt_path: Path, device: torch.device, config: dict) -> torch.nn.Module:
    """Load a robust checkpoint, auto-detecting whether it was trained with CBAM.

    The CBAM flag is stored in the checkpoint by train_xception_robust.py so
    eval scripts don't need to be told which architecture to instantiate.
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    use_cbam = bool(ckpt.get("use_cbam", False)) if isinstance(ckpt, dict) else False
    cbam_reduction = int(ckpt.get("cbam_reduction", 16)) if isinstance(ckpt, dict) else 16
    model = create_xception(
        num_classes=config["model"]["xception"]["num_classes"],
        dropout=config["model"]["xception"]["dropout"],
        pretrained=False,
        use_cbam=use_cbam,
        cbam_reduction=cbam_reduction,
    )
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device).eval()
    if use_cbam:
        print(f"[*] Loaded CBAM-equipped Xception (reduction={cbam_reduction})")
    return model


def preprocess(image_pil: Image.Image, size: int, mean, std) -> torch.Tensor:
    tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return tf(image_pil).unsqueeze(0)


def run_gradcam(
    model: torch.nn.Module,
    image_pil: Image.Image,
    device: torch.device,
    target_class: int,
    size: int,
    mean, std,
    after_cbam: bool = False,
) -> tuple[np.ndarray, float, float]:
    """Return heatmap-overlay image, p(real), p(fake).

    If the model has CBAM and after_cbam=True, the Grad-CAM target is the CBAM
    output (shows attention *after* the gating). Otherwise the target is
    backbone.act4 (the pre-attention feature map). For non-CBAM models the
    target is always backbone.act4.
    """
    input_tensor = preprocess(image_pil, size, mean, std).to(device)

    if after_cbam and getattr(model, "use_cbam", False) and model.cbam is not None:
        target_layer = model.cbam
    else:
        target_layer = model.backbone.act4

    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=[ClassifierOutputTarget(target_class)],
    )[0]

    # Build RGB overlay using the resized input.
    rgb_resized = np.asarray(image_pil.resize((size, size)).convert("RGB")) / 255.0
    overlay = show_cam_on_image(rgb_resized.astype(np.float32), grayscale_cam, use_rgb=True)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return overlay, float(probs[0]), float(probs[1])


def side_by_side(original: np.ndarray, overlay: np.ndarray, caption: str, out_path: Path) -> None:
    """Write a 2-panel PNG with the original face next to the Grad-CAM overlay."""
    h, w = overlay.shape[:2]
    orig_resized = cv2.resize(original, (w, h), interpolation=cv2.INTER_AREA)
    gap = np.full((h, 8, 3), 255, dtype=np.uint8)
    canvas = np.concatenate([orig_resized, gap, overlay.astype(np.uint8)], axis=1)

    # Add caption band
    band_h = 36
    band = np.full((band_h, canvas.shape[1], 3), 32, dtype=np.uint8)
    cv2.putText(band, caption, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    canvas = np.concatenate([canvas, band], axis=0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def extract_faces_from_video(video_path: Path, max_frames: int) -> list[np.ndarray]:
    """Use the project's existing preprocessing pipeline to extract face crops."""
    from src.preprocessing.video_processor import VideoProcessor
    from src.preprocessing.face_extractor import FaceExtractor

    processor = VideoProcessor(frames_per_video=max_frames)
    frames = processor.extract_frames(str(video_path))
    extractor = FaceExtractor()
    faces = []
    for frame in frames:
        result = extractor.extract_face(frame)
        if result is not None:
            faces.append(result)
    return faces


def sample_random_image(root: Path, label_class: str, n: int) -> list[Path]:
    """Pick n random face images from a processed dataset root/class/*/face_*.jpg."""
    class_dir = root / label_class
    if not class_dir.exists():
        return []
    # Nested layout: root/class/video_id/face_*.jpg
    candidates: list[Path] = []
    for video_dir in list(class_dir.iterdir())[:200]:  # cap search
        if video_dir.is_dir():
            candidates.extend(list(video_dir.glob("*.jpg"))[:3])
            if len(candidates) > 500:
                break
    if not candidates:
        return []
    random.shuffle(candidates)
    return candidates[:n]


def main() -> None:
    ap = argparse.ArgumentParser(description="Grad-CAM diagnostic for XceptionNet deepfake detector")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--image", type=str, default=None, help="Single input image")
    ap.add_argument("--video", type=str, default=None, help="Input video (faces extracted via MTCNN pipeline)")
    ap.add_argument("--sanity", action="store_true",
                    help="Run on 4 random FF++ real, 4 FF++ fake, 4 Celeb-DF real, 4 Celeb-DF fake")
    ap.add_argument("--max-frames", type=int, default=8)
    ap.add_argument("--output", type=str, default=None)
    ap.add_argument("--output-dir", type=str, default="outputs/gradcam")
    ap.add_argument("--target", choices=["pred", "real", "fake"], default="pred",
                    help="Which class to explain (pred=model's own prediction)")
    ap.add_argument("--after-cbam", action="store_true",
                    help="For CBAM-equipped models: target the CBAM output rather than the "
                         "pre-attention act4 feature map. Use this to visualise the effect of "
                         "the attention gating directly. Ignored for non-CBAM models.")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(Path(args.checkpoint), device, cfg)
    size = cfg["preprocessing"]["image"].get("size", 299)
    mean = cfg["preprocessing"]["image"].get("normalize_mean", [0.5, 0.5, 0.5])
    std = cfg["preprocessing"]["image"].get("normalize_std", [0.5, 0.5, 0.5])

    out_root = Path(args.output_dir)
    jobs: list[tuple[str, np.ndarray]] = []  # (tag, face_bgr_as_rgb_np)

    if args.image:
        img = Image.open(args.image).convert("RGB")
        jobs.append((Path(args.image).stem, np.asarray(img)))
    elif args.video:
        faces = extract_faces_from_video(Path(args.video), args.max_frames)
        if not faces:
            print(f"[!] No faces extracted from {args.video}")
            sys.exit(1)
        for i, face_bgr in enumerate(faces):
            jobs.append((f"{Path(args.video).stem}_f{i:02d}", cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)))
    elif args.sanity:
        processed_root = Path(cfg["data"]["processed"]["root"])
        datasets = [
            ("ffpp_real", processed_root / "FaceForensics++_AllTypes", "real"),
            ("ffpp_fake", processed_root / "FaceForensics++_AllTypes", "fake"),
            ("celebdf_real", processed_root / "Celeb-DF-v2", "real_celeb"),
            ("celebdf_fake", processed_root / "Celeb-DF-v2", "fake"),
        ]
        for tag, root, cls in datasets:
            for i, img_path in enumerate(sample_random_image(root, cls, 4)):
                img = Image.open(img_path).convert("RGB")
                jobs.append((f"{tag}_{i:02d}_{img_path.parent.name}", np.asarray(img)))
    else:
        print("Must provide --image, --video, or --sanity")
        sys.exit(2)

    # Pick the explanation target.
    target_map = {"real": 0, "fake": 1}

    print(f"\n[*] Running Grad-CAM on {len(jobs)} face(s) (device={device})")
    summary: list[str] = []
    for tag, face_rgb in jobs:
        pil = Image.fromarray(face_rgb)
        # Decide target
        if args.target == "pred":
            # Quick forward pass to pick argmax class
            with torch.no_grad():
                probs_raw = torch.softmax(model(preprocess(pil, size, mean, std).to(device)), dim=1)[0]
                tgt = int(probs_raw.argmax().item())
        else:
            tgt = target_map[args.target]
        overlay, p_real, p_fake = run_gradcam(
            model, pil, device, tgt, size, mean, std, after_cbam=args.after_cbam,
        )
        verdict = "FAKE" if p_fake > 0.5 else "REAL"
        label = "fake" if tgt == 1 else "real"
        caption = f"{tag}  explain:{label}  verdict:{verdict}  p(real)={p_real:.3f}  p(fake)={p_fake:.3f}"
        if args.output and len(jobs) == 1:
            out_path = Path(args.output)
        else:
            out_path = out_root / f"{tag}.png"
        side_by_side(face_rgb, overlay, caption, out_path)
        summary.append(caption)
        print(f"  {caption}  ->  {out_path}")

    print("\n[*] Summary:")
    for line in summary:
        print(f"  {line}")
    print(f"\n[+] Outputs in: {out_root if not args.output else Path(args.output).parent}")


if __name__ == "__main__":
    main()
