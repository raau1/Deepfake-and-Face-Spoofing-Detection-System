"""
Preprocess the DFD (DeepFakeDetection) standalone release.

This is the full 3068-fake / 363-real DFD release from Google. The FaceForensics++
`DeepFakeDetection` sub-folder already contains 1000 videos from this same source,
so running this script brings the remaining ~2000 fakes (and the full set of real
sequences) into the training mix. The goal here is extra generator-inside-DFD
identity coverage rather than a new generator family - WildDeepfake / DFDC would
be required for a true out-of-family test.

Source layout (as delivered):
    data/DFD_manipulated_sequences/DFD_manipulated_sequences/*.mp4   (3068 fakes,
        nested one level because the archive ships as a folder-in-folder)
    data/DFD_original sequences/*.mp4                                 (363 reals,
        note the space in the folder name - kept as-is to avoid renaming user data)

Output layout (matches the existing FF++/Celeb-DF processed directories so the
same MixedDatasetLoader can consume it):
    data/processed/DFD/real/<video_id>/face_*.jpg
    data/processed/DFD/fake/<video_id>/face_*.jpg
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict

import yaml

# Make src.* importable when running as `python scripts/preprocess_dfd.py`.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.preprocessing.pipeline import PreprocessingPipeline


def preprocess_dfd(
    fake_dir: Path,
    real_dir: Path,
    output_dir: Path,
    preprocessing_config: dict,
) -> Dict:
    """Run the standard MTCNN + MediaPipe pipeline over the DFD dataset."""
    fake_dir = Path(fake_dir)
    real_dir = Path(real_dir)
    output_dir = Path(output_dir)

    print("PREPROCESSING DFD (DeepFakeDetection) STANDALONE RELEASE")
    print("=" * 60)
    print(f"[*] Fake source:  {fake_dir}")
    print(f"[*] Real source:  {real_dir}")
    print(f"[*] Output:       {output_dir}")

    if not fake_dir.exists():
        raise FileNotFoundError(f"Fake directory missing: {fake_dir}")
    if not real_dir.exists():
        raise FileNotFoundError(f"Real directory missing: {real_dir}")

    fake_count = len(list(fake_dir.glob("*.mp4")))
    real_count = len(list(real_dir.glob("*.mp4")))
    print(f"[*] Fake videos found: {fake_count}")
    print(f"[*] Real videos found: {real_count}")
    if fake_count == 0 or real_count == 0:
        raise RuntimeError("No .mp4 files found in one of the source directories.")

    pp = preprocessing_config or {}
    alignment = pp.get("alignment", {})
    pipeline = PreprocessingPipeline(
        output_size=alignment.get("output_size", 299),
        frames_per_video=pp.get("frames_per_video", 32),
        sampling_strategy=pp.get("frame_sampling", "uniform"),
        use_alignment=alignment.get("enabled", True),
        margin=alignment.get("margin", 0.3),
        min_face_size=pp.get("min_face_size", 60),
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    start = datetime.now()
    stats = pipeline.process_dataset(
        input_dirs={"real": real_dir, "fake": fake_dir},
        output_dir=output_dir,
        save_format="jpg",
        quality=95,
    )
    duration = datetime.now() - start
    pipeline.close()

    if stats is not None:
        stats["source_fake_dir"] = str(fake_dir)
        stats["source_real_dir"] = str(real_dir)
        stats["duration"] = str(duration)
        stats["dataset"] = "DFD"
        meta_path = output_dir / "dfd_preprocessing_stats.json"
        with open(meta_path, "w") as f:
            json.dump(stats, f, indent=2, default=str)
        print(f"\n[+] Stats written to: {meta_path}")

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print(f"[+] Duration: {duration}")
    print(f"[+] Output:   {output_dir}")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess the DFD standalone release")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument(
        "--fake-dir",
        type=str,
        default=None,
        help="Override fake source directory (default: data/DFD_manipulated_sequences/DFD_manipulated_sequences)",
    )
    parser.add_argument(
        "--real-dir",
        type=str,
        default=None,
        help="Override real source directory (default: data/DFD_original sequences)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory (default: <processed_root>/DFD)",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    config = {}
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)

    data_root = Path(config.get("data", {}).get("root", "data"))
    processed_root = Path(
        config.get("data", {}).get("processed", {}).get("root", "data/processed")
    )

    if args.fake_dir is None:
        args.fake_dir = str(
            data_root / "DFD_manipulated_sequences" / "DFD_manipulated_sequences"
        )
    if args.real_dir is None:
        args.real_dir = str(data_root / "DFD_original sequences")
    if args.output_dir is None:
        args.output_dir = str(processed_root / "DFD")

    stats = preprocess_dfd(
        fake_dir=Path(args.fake_dir),
        real_dir=Path(args.real_dir),
        output_dir=Path(args.output_dir),
        preprocessing_config=config.get("preprocessing", {}),
    )
    if stats is None:
        sys.exit(1)

    print("\n[*] Next step - retrain robust model with DFD included:")
    print("    python scripts/train_xception_robust.py --include-dfd")


if __name__ == "__main__":
    main()
