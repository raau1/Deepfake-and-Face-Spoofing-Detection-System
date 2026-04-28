"""
Preprocess all FaceForensics++ manipulation types for comprehensive training.

This script processes all 6 manipulation methods available in FF++:
1. Deepfakes
2. Face2Face
3. FaceSwap
4. FaceShifter
5. NeuralTextures
6. DeepFakeDetection

All fake videos are combined into a single 'fake' category for training,
but statistics are tracked per manipulation type for analysis.
"""

import sys
import yaml
from pathlib import Path
from typing import Dict
import json
from datetime import datetime

# Adds project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.preprocessing.pipeline import PreprocessingPipeline


def preprocess_ff_all_types(
    ff_root: Path,
    output_dir: Path,
    frames_per_video: int = 32,
    manipulation_types: list = None,
    preprocessing_config: dict = None
) -> Dict:
    """
    Preprocess all FF++ manipulation types.

    Args:
        ff_root: Root directory of FaceForensics++ dataset
        output_dir: Output directory for processed faces
        frames_per_video: Frames to extract per video
        manipulation_types: List of manipulation types to process (None = all)
        preprocessing_config: Preprocessing settings from config.yaml

    Returns:
        stats: Combined processing statistics
    """
    ff_root = Path(ff_root)
    output_dir = Path(output_dir)

    # All available FF++ manipulation types
    if manipulation_types is None:
        manipulation_types = [
            'Deepfakes',
            'Face2Face',
            'FaceSwap',
            'NeuralTextures',
            'DeepFakeDetection',
            'FaceShifter'
        ]

    print("PREPROCESSING ALL FACEFORENSICS++ MANIPULATION TYPES")
    print("="*60)
    print(f"\n[*] Source: {ff_root}")
    print(f"[*] Output: {output_dir}")
    print(f"[*] Frames per video: {frames_per_video}")
    print(f"[*] Manipulation types to process: {len(manipulation_types)}")

    # Checks which manipulation directories exist
    available_types = []
    for manip_type in manipulation_types:
        manip_dir = ff_root / manip_type
        if manip_dir.exists():
            video_count = len(list(manip_dir.glob("*.mp4")))
            available_types.append(manip_type)
            print(f"  [+] {manip_type}: {video_count} videos")
        else:
            print(f"  [!] {manip_type}: NOT FOUND (skipping)")

    print(f"\n[*] Processing {len(available_types)} manipulation types")

    # Initializes pipeline using settings from config if provided
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

    # Gets original/real videos directory
    original_dir = ff_root / 'original_reencoded'
    if not original_dir.exists():
        original_dir = ff_root / 'original'

    if not original_dir.exists():
        print(f"\n[!] ERROR: Original videos directory not found")
        print(f"   Looked for: {ff_root / 'original_reencoded'}")
        print(f"   Looked for: {ff_root / 'original'}")
        return None

    real_video_count = len(list(original_dir.glob("*.mp4")))
    print(f"\n[*] Real videos: {real_video_count} from {original_dir.name}")

    # Create input directories dictionary
    # All fake types will be processed into a single 'fake' output directory
    # But tracks statistics per manipulation type
    input_dirs = {'real': original_dir}

    # Adds all fake manipulation types
    for manip_type in available_types:
        # Uses manipulation type name as key for tracking
        input_dirs[f'fake_{manip_type}'] = ff_root / manip_type

    print(f"\n[*] Starting preprocessing...")
    print(f"   This will take approximately {len(input_dirs) * 30} minutes")
    print(f"   Progress will be saved incrementally\n")

    # Process dataset
    start_time = datetime.now()

    # Use temporary output directory
    temp_output = output_dir / 'FaceForensics++_AllTypes_temp'

    stats = pipeline.process_dataset(
        input_dirs=input_dirs,
        output_dir=temp_output,
        save_format='jpg',
        quality=95
    )

    end_time = datetime.now()
    duration = end_time - start_time

    pipeline.close()

    # Merges all fake_* directories into a single 'fake' directory
    print(f"\n[*] Merging all manipulation types into single 'fake' directory...")

    final_output = output_dir / 'FaceForensics++_AllTypes'
    final_output.mkdir(parents=True, exist_ok=True)

    # Copy real directory
    real_src = temp_output / 'real'
    real_dst = final_output / 'real'
    if real_src.exists():
        print(f"  [+] Copying real videos...")
        import shutil
        if real_dst.exists():
            shutil.rmtree(real_dst)
        shutil.copytree(real_src, real_dst)

    # Merges all fake_* directories into single fake directory
    fake_dst = final_output / 'fake'
    fake_dst.mkdir(parents=True, exist_ok=True)

    total_fake_videos = 0
    for manip_type in available_types:
        fake_src = temp_output / f'fake_{manip_type}'
        if fake_src.exists():
            print(f"  [+] Merging {manip_type}...")
            # Copy all video directories from this manipulation type
            for video_dir in fake_src.iterdir():
                if video_dir.is_dir():
                    video_dst = fake_dst / f"{manip_type}_{video_dir.name}"
                    shutil.copytree(video_dir, video_dst)
                    total_fake_videos += 1

    print(f"  [+] Merged {total_fake_videos} fake videos from {len(available_types)} manipulation types")

    # Copies metadata to final location
    metadata_src = temp_output / 'metadata.json'
    if metadata_src.exists():
        import shutil
        shutil.copy(metadata_src, final_output / 'metadata.json')

    # Cleans up temporary directory
    print(f"  [+] Cleaning up temporary files...")
    import shutil
    shutil.rmtree(temp_output)

    print(f"  [+] Final dataset ready at: {final_output}")

    # Adds metadata about which manipulation types were included
    if stats:
        stats['manipulation_types'] = available_types
        stats['processing_duration'] = str(duration)
        stats['frames_per_video'] = frames_per_video

    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE")
    print("="*60)
    print(f"\n[+] Total duration: {duration}")
    print(f"[+] Output directory: {output_dir / 'FaceForensics++_AllTypes'}")
    print(f"\n[*] To view detailed statistics, run:")
    print(f"    python scripts/show_preprocessing_stats.py")

    return stats


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Preprocess all FaceForensics++ manipulation types'
    )
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to config file')
    parser.add_argument('--ff-root', type=str,
                        default=None,
                        help='Root directory of FaceForensics++ dataset (default: from config.yaml)')
    parser.add_argument('--output-dir', type=str,
                        default=None,
                        help='Output directory for processed data (default: from config.yaml)')
    parser.add_argument('--frames', type=int, default=32,
                        help='Frames to extract per video')
    parser.add_argument('--types', type=str, nargs='+',
                        default=None,
                        help='Specific manipulation types to process (default: all)')

    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    config = {}
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)

    # Resolve paths from config if not provided on command line
    if args.ff_root is None:
        args.ff_root = config.get('data', {}).get('faceforensics', {}).get('root', 'data/FaceForensics++')
    if args.output_dir is None:
        args.output_dir = config.get('data', {}).get('processed', {}).get('root', 'data/processed')

    # Run preprocessing, passing preprocessing settings from config
    stats = preprocess_ff_all_types(
        ff_root=Path(args.ff_root),
        output_dir=Path(args.output_dir),
        frames_per_video=args.frames,
        manipulation_types=args.types,
        preprocessing_config=config.get('preprocessing', {})
    )

    if stats is None:
        print("\n[!] Preprocessing failed!")
        sys.exit(1)

    print("\n[+] All done")
    print("\n[*] Next step: Train on the new dataset")
    print("   python scripts/train_xception_mixed.py \\")
    print("     --ff-dir data/processed/FaceForensics++_AllTypes \\")
    print("     --celebdf-dir data/processed/Celeb-DF-v2 \\")
    print("     --epochs 30 --batch-size 32")


if __name__ == '__main__':
    main()