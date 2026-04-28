"""
Preprocess Celeb-DF-v2 dataset for deepfake detection training.

Extracts faces from Celeb-real, YouTube-real, and Celeb-synthesis videos.
Real sources (Celeb-real + YouTube-real) are kept as separate labels for
per-source statistics, then both feed into the 'real' class during training.
"""

import sys
import yaml
from pathlib import Path
from typing import Dict

# Adds project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.preprocessing.pipeline import PreprocessingPipeline


def preprocess_celebdf(
    celebdf_root: Path,
    output_dir: Path,
    frames_per_video: int = 32,
    preprocessing_config: dict = None
) -> Dict:
    """
    Preprocess Celeb-DF-v2 dataset.

    Args:
        celebdf_root: Root directory of Celeb-DF-v2 dataset
        output_dir: Output directory for processed faces
        frames_per_video: Frames to extract per video
        preprocessing_config: Preprocessing settings from config.yaml

    Returns:
        stats: Processing statistics
    """
    celebdf_root = Path(celebdf_root)
    output_dir = Path(output_dir)

    print("PREPROCESSING CELEB-DF-V2 DATASET")
    print("=" * 60)
    print(f"\n[*] Source: {celebdf_root}")
    print(f"[*] Output: {output_dir}")
    print(f"[*] Frames per video: {frames_per_video}")

    # Initialize pipeline using settings from config if provided
    pp = preprocessing_config or {}
    alignment = pp.get('alignment', {})
    pipeline = PreprocessingPipeline(
        output_size=alignment.get('output_size', 299),
        frames_per_video=pp.get('frames_per_video', frames_per_video),
        sampling_strategy=pp.get('frame_sampling', 'uniform'),
        use_alignment=alignment.get('enabled', True),
        margin=alignment.get('margin', 0.3),
        min_face_size=pp.get('min_face_size', 60)
    )

    # Define input directories
    input_dirs = {
        'real_celeb': celebdf_root / 'Celeb-real',
        'real_youtube': celebdf_root / 'YouTube-real',
        'fake': celebdf_root / 'Celeb-synthesis'
    }

    # Process dataset
    stats = pipeline.process_dataset(
        input_dirs=input_dirs,
        output_dir=output_dir / 'Celeb-DF-v2',
        save_format='jpg',
        quality=95
    )

    pipeline.close()

    print("\n" + "=" * 60)
    print("CELEB-DF-V2 PREPROCESSING COMPLETE")
    print("=" * 60)

    return stats


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Preprocess Celeb-DF-v2 dataset'
    )
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to config file')
    parser.add_argument('--celebdf-root', type=str,
                        default=None,
                        help='Root directory of Celeb-DF-v2 dataset (default: from config.yaml)')
    parser.add_argument('--output-dir', type=str,
                        default=None,
                        help='Output directory for processed data (default: from config.yaml)')
    parser.add_argument('--frames', type=int, default=32,
                        help='Frames to extract per video')

    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    config = {}
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)

    # Resolve paths from config if not provided on command line
    if args.celebdf_root is None:
        args.celebdf_root = config.get('data', {}).get('celebdf', {}).get('root', 'data/Celeb-DF-v2')
    if args.output_dir is None:
        args.output_dir = config.get('data', {}).get('processed', {}).get('root', 'data/processed')

    # Run preprocessing
    stats = preprocess_celebdf(
        celebdf_root=Path(args.celebdf_root),
        output_dir=Path(args.output_dir),
        frames_per_video=args.frames,
        preprocessing_config=config.get('preprocessing', {})
    )

    if stats is None:
        print("\n[!] Preprocessing failed!")
        sys.exit(1)

    print("\n[+] All done")


if __name__ == '__main__':
    main()
