"""
XceptionNet Robust Fine-tune Script

Fine-tunes the mixed-trained XceptionNet checkpoint with compression-aware
augmentations (JPEG re-encoding, downscale/upscale, Gaussian noise, stronger
colour jitter) to close the in-the-wild generalisation gap observed during
FastAPI demonstration - the baseline models classify YouTube-sourced
deepfakes as real with high confidence because YouTube re-encoding destroys
the sub-pixel artefacts the networks learned to rely on.

Start state:  outputs/models/mixed/best_model_mixed.pth
End state:    outputs/models/robust/best_model_robust.pth

Default configuration uses a low learning rate (1e-5) and 10 epochs so the
model retains its in-distribution performance while learning the compression
invariances.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Make the project root importable so `from src.*` and `from scripts.*` work
# whether this script is invoked as `python scripts/train_xception_robust.py`
# or `python -m scripts.train_xception_robust`.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch
import yaml

from src.models.xception import create_xception
from scripts.train_xception_mixed import MixedDatasetLoader, Trainer


def _load_weights_from_checkpoint(
    model: torch.nn.Module,
    ckpt_path: Path,
    device: torch.device,
    strict: bool = True,
) -> None:
    """Load only the model weights from a mixed-training checkpoint.

    strict=False tolerates missing keys (used when loading a non-CBAM checkpoint
    into a CBAM-equipped model - the cbam.* tensors stay at their random init).
    """
    print(f"Loading weights from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    missing, unexpected = model.load_state_dict(state, strict=strict)
    if missing:
        print(f"  [load] Missing keys (left at init): {len(missing)} (e.g. {missing[:3]})")
    if unexpected:
        print(f"  [load] Unexpected keys (ignored): {len(unexpected)} (e.g. {unexpected[:3]})")
    print("Weights loaded successfully.")


def _patch_trainer_for_robust_output(
    trainer: Trainer,
    use_cbam: bool = False,
    cbam_reduction: int = 16,
) -> None:
    """Replace Trainer.save_checkpoint so best checkpoints save as best_model_robust.pth.

    The original Trainer hard-codes 'best_model_mixed.pth' which would clobber
    the base checkpoint. We also tag the saved checkpoint with model_type so
    evaluate_model.py can auto-detect it, and record use_cbam / cbam_reduction
    so downstream loaders can rebuild the architecture without a manual flag.
    """
    def save_checkpoint(filename: str, is_best: bool = False) -> None:
        ckpt = {
            "epoch": trainer.current_epoch,
            "model_state_dict": trainer.model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "scheduler_state_dict": trainer.scheduler.state_dict(),
            "best_val_auc": trainer.best_val_auc,
            "history": trainer.history,
            "config": trainer.config,
            "model_type": "robust",
            "use_cbam": use_cbam,
            "cbam_reduction": cbam_reduction,
        }
        if trainer.scaler is not None:
            ckpt["scaler_state_dict"] = trainer.scaler.state_dict()
        torch.save(ckpt, trainer.checkpoint_dir / filename)
        if is_best:
            torch.save(ckpt, trainer.checkpoint_dir / "best_model_robust.pth")
            print(f"Saved best robust model with AUC: {trainer.best_val_auc:.4f}")

    trainer.save_checkpoint = save_checkpoint  # type: ignore[assignment]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune mixed XceptionNet with compression-aware augmentation"
    )
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override epoch count (default: training_robust.epochs in config)")
    parser.add_argument("--learning-rate", type=float, default=None,
                        help="Override LR (default: training_robust.learning_rate in config)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size (default: training.batch_size in config)")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Checkpoint to fine-tune from (default: best_model_mixed.pth). "
                             "Ignored if --from-imagenet is set.")
    parser.add_argument("--from-imagenet", action="store_true",
                        help="Start from ImageNet pretrained Xception instead of a prior checkpoint. "
                             "Use this for clean shortcut-mitigation runs where we want no deepfake "
                             "bias inherited from earlier training (e.g. the 'YouTube codec -> real' "
                             "correlation baked into best_model_mixed.pth by Celeb-DF's real_youtube "
                             "class). Needs more epochs (~25) than fine-tuning (~10) to converge.")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--include-dfd", action="store_true",
                        help="Include DFD (DeepFakeDetection) standalone release in training mix.")
    parser.add_argument("--include-wilddeepfake", action="store_true",
                        help="Include WildDeepfake (train split) in training mix. Expects flat layout at data/wilddeepfake/train/{real,fake}.")
    parser.add_argument("--dfd-dir", type=str, default=None,
                        help="Override preprocessed DFD directory (default: <processed_root>/DFD).")
    parser.add_argument("--wilddeepfake-dir", type=str, default=None,
                        help="Override WildDeepfake train directory (default: data/wilddeepfake/train).")
    parser.add_argument("--wilddeepfake-frames", type=int, default=20,
                        help="Frames per video for WildDeepfake (default 20; overrides training_frames_per_video). "
                             "WildDeepfake has ~150 frames/video so 5 leaves it under-sampled at ~3%%.")
    parser.add_argument("--keep-dfd-in-ffpp", action="store_true",
                        help="Keep the 1000 DeepFakeDetection_* videos inside FF++ even when --include-dfd "
                             "is set. Default: drop them to avoid double-counting (v2 shortcut).")
    parser.add_argument("--keep-real-youtube", action="store_true",
                        help="Keep Celeb-DF's real_youtube/ class in training. Default: drop it to remove "
                             "the 'YouTube codec -> real' shortcut the v2 checkpoint learned.")
    parser.add_argument("--no-weighted-sampler", action="store_true",
                        help="Disable WeightedRandomSampler and fall back to the old balance_classes "
                             "oversampling (which duplicates minority-class samples in memory). "
                             "Only use this for ablation comparisons against v2.")
    parser.add_argument("--use-cbam", action="store_true",
                        help="Insert CBAM (channel + spatial attention) between Xception's act4 "
                             "feature map and the global pool. Aimed at the §6.10.2 Grad-CAM "
                             "diagnosis: v3 heatmaps collapsed onto the central nose-mouth-cheek "
                             "region regardless of class. CBAM forces the network to reason over "
                             "a learned weighting of channels and spatial positions rather than "
                             "only the highest-activation region.")
    parser.add_argument("--cbam-reduction", type=int, default=16,
                        help="CBAM channel-attention reduction ratio (default 16, per the paper).")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Override DataLoader num_workers (default: training.num_workers in config). "
                             "Set to 0 for single-process debugging / smoke-test runs on Windows where "
                             "worker spawning can fight the GPU driver.")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run a single forward+backward step and exit. Used to verify plumbing before launching a long run.")
    args = parser.parse_args()

    # ---- Load config ----
    config_path = Path(args.config)
    with open(config_path) as f:
        config = yaml.safe_load(f)

    robust_cfg = config.get("training_robust", {})

    # Resolve paths + hyperparameters with clear precedence: CLI > config > hard default.
    models_dir = Path(config.get("output", {}).get("models_dir", "outputs/models"))
    if args.resume_from is None:
        args.resume_from = robust_cfg.get(
            "resume_from", str(models_dir / "mixed" / "best_model_mixed.pth")
        )
    if args.output_dir is None:
        args.output_dir = robust_cfg.get("output_dir", str(models_dir / "robust"))
    if args.epochs is None:
        args.epochs = int(robust_cfg.get("epochs", 10))
    if args.learning_rate is None:
        args.learning_rate = float(robust_cfg.get("learning_rate", 1e-5))
    if args.batch_size is None:
        args.batch_size = int(config.get("training", {}).get("batch_size", 32))
    if args.num_workers is not None:
        config["training"]["num_workers"] = args.num_workers

    # Inject into the config dict so Trainer picks them up.
    config["training"]["batch_size"] = args.batch_size
    config["training"]["epochs"] = args.epochs
    config["training"]["optimizer"]["learning_rate"] = args.learning_rate
    config["training"]["checkpoint_dir"] = args.output_dir
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # ---- Device ----
    cfg_device = config.get("hardware", {}).get("device", "cuda")
    if cfg_device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        cfg_device = "cpu"
    device = torch.device(cfg_device)
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ---- Model: ImageNet init (clean slate) or fine-tune from prior checkpoint ----
    if args.use_cbam:
        print(f"[cbam] Using CBAM attention (reduction={args.cbam_reduction}).")
    if args.from_imagenet:
        print("\n[from-imagenet] Creating XceptionNet with ImageNet pretrained weights "
              "(no prior deepfake bias).")
        model = create_xception(
            num_classes=config["model"]["xception"]["num_classes"],
            dropout=config["model"]["xception"]["dropout"],
            pretrained=True,
            use_cbam=args.use_cbam,
            cbam_reduction=args.cbam_reduction,
        )
        resume_path = None
    else:
        print("\nCreating XceptionNet and loading prior-training weights...")
        model = create_xception(
            num_classes=config["model"]["xception"]["num_classes"],
            dropout=config["model"]["xception"]["dropout"],
            pretrained=False,
            use_cbam=args.use_cbam,
            cbam_reduction=args.cbam_reduction,
        )
        resume_path = Path(args.resume_from)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        # strict=False tolerates the new cbam.* parameters when --use-cbam is set
        # against a non-CBAM checkpoint - cbam weights stay at their random init
        # and the rest of the backbone loads from the prior run.
        _load_weights_from_checkpoint(model, resume_path, device, strict=not args.use_cbam)

    # ---- Data (robust augmentation on train split only) ----
    ff_dir = Path(config["data"]["processed"]["faceforensics"])
    celebdf_dir = Path(config["data"]["processed"]["celebdf"])
    pp = config.get("preprocessing", {})
    img_cfg = pp.get("image", {})
    train_cfg = config.get("training", {})

    # Build the dataset list. FF++ and Celeb-DF are always in; DFD and
    # WildDeepfake are opt-in so we can compare three- vs four-way mixes
    # without changing the script.
    processed_root = Path(
        config.get("data", {}).get("processed", {}).get("root", "data/processed")
    )
    data_root = Path(config.get("data", {}).get("root", "data"))

    # v3 shortcut-mitigation flags.
    # Grad-CAM on the v2 checkpoint showed three dataset-level shortcuts:
    #   1) FF++_AllTypes/fake already contains 1000 DeepFakeDetection_* videos;
    #      running with --include-dfd then gave the model those 1000 twice.
    #   2) Celeb-DF's real_youtube/ is the only class that's pure YouTube
    #      pristine content - the model learned "YouTube codec -> real" and
    #      classified YouTube deepfakes as real by the same shortcut.
    #   3) WildDeepfake was undersampled at frames_per_video=5 (~3% of content)
    #      so its generalisation signal didn't reach the model.
    # v3 fixes all three below.
    ffpp_exclude_prefixes = ["DeepFakeDetection_"] if args.include_dfd and not args.keep_dfd_in_ffpp else None
    celebdf_exclude_classes = None if args.keep_real_youtube else ["real_youtube"]

    dataset_entries: list = [
        {
            "name": "FaceForensics++",
            "path": ff_dir,
            "flat_layout": False,
            "exclude_prefixes": ffpp_exclude_prefixes,
        },
        {
            "name": "Celeb-DF-v2",
            "path": celebdf_dir,
            "flat_layout": False,
            "exclude_classes": celebdf_exclude_classes,
        },
    ]
    if ffpp_exclude_prefixes:
        print(f"[+] FF++ filter: excluding videos starting with {ffpp_exclude_prefixes} "
              f"(DFD standalone release supplies these instead)")
    if celebdf_exclude_classes:
        print(f"[+] Celeb-DF filter: excluding classes {celebdf_exclude_classes} "
              f"('YouTube -> real' shortcut mitigation)")
    if args.include_dfd:
        dfd_dir = Path(args.dfd_dir) if args.dfd_dir else processed_root / "DFD"
        dataset_entries.append({"name": "DFD", "path": dfd_dir, "flat_layout": False})
        print(f"[+] Including DFD from: {dfd_dir}")
    if args.include_wilddeepfake:
        wd_dir = (
            Path(args.wilddeepfake_dir)
            if args.wilddeepfake_dir
            else data_root / "wilddeepfake" / "train"
        )
        # Higher frames_per_video for WildDeepfake - it has ~150 frames/video
        # and was previously under-sampled at the default 5. 20 lifts the
        # effective sampling rate from ~3% to ~13%, closer to FF++'s ~15%.
        dataset_entries.append({
            "name": "WildDeepfake",
            "path": wd_dir,
            "flat_layout": True,
            "frames_per_video": args.wilddeepfake_frames,
        })
        print(f"[+] Including WildDeepfake (flat layout, "
              f"{args.wilddeepfake_frames} frames/video) from: {wd_dir}")

    loader = MixedDatasetLoader(
        dataset_dirs=dataset_entries,
        batch_size=train_cfg["batch_size"],
        num_workers=train_cfg.get("num_workers", 4),
        train_split=train_cfg.get("train_split", 0.8),
        val_split=train_cfg.get("val_split", 0.1),
        image_size=img_cfg.get("size", 299),
        frames_per_video=pp.get("training_frames_per_video", 5),
        normalize_mean=img_cfg.get("normalize_mean"),
        normalize_std=img_cfg.get("normalize_std"),
        augmentation="robust",
        use_weighted_sampler=not args.no_weighted_sampler,
    )
    train_loader, val_loader, test_loader = loader.create_dataloaders()

    # ---- Trainer ----
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
    )
    _patch_trainer_for_robust_output(trainer, use_cbam=args.use_cbam, cbam_reduction=args.cbam_reduction)

    # ---- Smoke test path: one batch, then exit ----
    if args.smoke_test:
        print("\n[smoke-test] Running one forward+backward step to verify plumbing...")
        trainer.model.train()
        images, labels = next(iter(train_loader))
        images = images.to(device)
        labels = labels.to(device)
        trainer.optimizer.zero_grad()
        outputs = trainer.model(images)
        loss = trainer.criterion(outputs, labels)
        loss.backward()
        trainer.optimizer.step()
        print(f"[smoke-test] OK - batch shape {tuple(images.shape)}, loss {loss.item():.4f}")
        return

    # ---- Train ----
    trainer.train(config["training"]["epochs"])

    # ---- Final test-set evaluation ----
    print("\nFinal evaluation on combined test set...")
    trainer.val_loader = test_loader
    test_loss, test_acc, test_auc = trainer.validate()
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"Test AUC: {test_auc:.4f}")

    results = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "test_auc": float(test_auc),
        "resumed_from": str(resume_path) if resume_path is not None else "imagenet",
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "augmentation": "robust",
        "timestamp": datetime.now().isoformat(),
    }
    results_path = Path(args.output_dir) / "test_results_robust.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
