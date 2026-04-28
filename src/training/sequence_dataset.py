"""
Sequence Dataset for Temporal Deepfake Detection
Returns fixed-length sequences of frames per video, maintaining temporal order.

Used by the temporal LSTM model which needs to see multiple frames from the
same video in order to detect frame-to-frame inconsistencies.
"""

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from PIL import Image
import numpy as np
import random


class SequenceDataset(Dataset):
    """
    Dataset that returns sequences of frames per video.

    Each sample is a video represented as a fixed-length sequence of face images.
    Frames are sorted by filename to maintain temporal order (face_0000.jpg, face_0001.jpg, ...).

    Directory structure (nested layout, same as DeepfakeDataset default):
        root/
            real/
                video1/
                    face_0000.jpg
                    face_0001.jpg
                    ...
                video2/
                    ...
            fake/
                video1/
                    ...

    Flat layout (used for WildDeepfake), enabled with flat_layout=True:
        root/
            real/
                0_0.png    <- video_id=0, frame_idx=0
                0_1.png    <- video_id=0, frame_idx=1
                42_137.png <- video_id=42, frame_idx=137
                ...
            fake/
                ...
        Files are grouped by the prefix before the last underscore (the video_id)
        and sorted numerically by the trailing frame_idx so temporal order is preserved.
    """

    def __init__(
        self,
        root_dir: Union[str, Path],
        split: str = 'train',
        sequence_length: int = 32,
        image_size: int = 299,
        normalize_mean: Optional[List[float]] = None,
        normalize_std: Optional[List[float]] = None,
        augment: bool = False,
        exclude_classes: Optional[List[str]] = None,
        exclude_video_prefixes: Optional[List[str]] = None,
        flat_layout: bool = False,
    ):
        """
        Initialize sequence dataset.

        Args:
            root_dir: Root directory containing 'real' and 'fake' subdirectories
            split: Data split ('train', 'val', 'test')
            sequence_length: Fixed number of frames per video sequence
            image_size: Image size for transforms
            normalize_mean: Normalization mean
            normalize_std: Normalization std
            augment: Whether to apply data augmentation (train only)
            exclude_classes: Class directory names to skip entirely (e.g. ['real_youtube']
                to remove the 'YouTube codec -> real' shortcut from Celeb-DF).
            exclude_video_prefixes: Video directory name prefixes to skip (e.g.
                ['DeepFakeDetection_'] to drop FF++_AllTypes' DFD subset when training
                with DFD standalone).
            flat_layout: If True, parse 'class/<video_id>_<frame_idx>.png' instead
                of 'class/<video_id>/face_*.jpg'. Used for WildDeepfake.
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.sequence_length = sequence_length
        self.image_size = image_size
        self.normalize_mean = normalize_mean or [0.5, 0.5, 0.5]
        self.normalize_std = normalize_std or [0.5, 0.5, 0.5]
        self.exclude_classes = set(exclude_classes or ())
        self.exclude_video_prefixes = tuple(exclude_video_prefixes or ())
        self.flat_layout = flat_layout

        self.transform = self._get_transforms(augment and split == 'train')

        # Each item is (video_dir, label, [sorted frame paths])
        self.videos: List[Tuple[str, int, List[Path]]] = []
        self._load_videos()

        real_count = sum(1 for _, l, _ in self.videos if l == 0)
        fake_count = sum(1 for _, l, _ in self.videos if l == 1)
        print(f"SequenceDataset ({split}): {len(self.videos)} videos, "
              f"seq_len={sequence_length}")
        print(f"  Real: {real_count}, Fake: {fake_count}")

    def _get_transforms(self, augment: bool) -> transforms.Compose:
        """Get image transforms."""
        normalize = transforms.Normalize(
            mean=self.normalize_mean,
            std=self.normalize_std
        )

        if augment:
            return transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                normalize
            ])
        else:
            return transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                normalize
            ])

    def _load_videos(self):
        """Load all videos from directory structure (dispatches by layout)."""
        if self.flat_layout:
            self._load_videos_flat()
        else:
            self._load_videos_nested()

    def _load_videos_nested(self):
        """Nested layout: root/<class>/<video_id>/face_*.jpg"""
        class_dirs = {
            'real': 0, 'fake': 1,
            'real_celeb': 0, 'real_youtube': 0
        }

        for class_name, label in class_dirs.items():
            if class_name in self.exclude_classes:
                continue
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                continue

            for video_dir in sorted(class_dir.iterdir()):
                if not video_dir.is_dir():
                    continue
                if self.exclude_video_prefixes and video_dir.name.startswith(self.exclude_video_prefixes):
                    continue

                # Get frames sorted by name (temporal order)
                frames = sorted(
                    list(video_dir.glob('*.jpg')) + list(video_dir.glob('*.png'))
                )

                if len(frames) < 2:
                    continue  # Need at least 2 frames for temporal analysis

                self.videos.append((str(video_dir), label, frames))

    def _load_videos_flat(self):
        """Flat layout: root/<class>/<video_id>_<frame_idx>.png

        Groups by the prefix before the last underscore (the video_id), and
        sorts each group numerically by the trailing frame_idx so the LSTM
        sees frames in true temporal order. Plain lexicographic sort is wrong
        here ('0_10' < '0_2'), which is the bug a straight glob-and-sort would
        introduce.
        """
        class_dirs = {'real': 0, 'fake': 1}

        for class_name, label in class_dirs.items():
            if class_name in self.exclude_classes:
                continue
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                continue

            video_groups: Dict[str, List[Tuple[int, Path]]] = {}
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() not in ('.png', '.jpg', '.jpeg'):
                    continue
                stem = img_path.stem
                if self.exclude_video_prefixes and stem.startswith(self.exclude_video_prefixes):
                    continue

                if '_' in stem:
                    video_id, frame_str = stem.rsplit('_', 1)
                    try:
                        frame_idx = int(frame_str)
                    except ValueError:
                        # Suffix isn't numeric - fall back to whole-stem grouping
                        # with index 0 so the file is still loadable.
                        video_id, frame_idx = stem, 0
                else:
                    video_id, frame_idx = stem, 0

                key = str(class_dir / video_id)
                video_groups.setdefault(key, []).append((frame_idx, img_path))

            for video_key, indexed_frames in video_groups.items():
                indexed_frames.sort(key=lambda pair: pair[0])
                frames = [p for _, p in indexed_frames]

                if len(frames) < 2:
                    continue  # Need at least 2 frames for temporal analysis

                self.videos.append((video_key, label, frames))

    def _sample_frames(self, frames: List[Path]) -> List[Path]:
        """
        Sample a fixed-length sequence from available frames.

        Uses uniform sampling to cover the full temporal extent.
        If fewer frames than sequence_length, repeats the last frame.
        """
        n_available = len(frames)

        if n_available >= self.sequence_length:
            # Uniform sampling across the video
            indices = np.linspace(0, n_available - 1, self.sequence_length, dtype=int)
            return [frames[i] for i in indices]
        else:
            # Use all frames, pad with last frame
            sampled = list(frames)
            while len(sampled) < self.sequence_length:
                sampled.append(frames[-1])
            return sampled

    def __len__(self) -> int:
        return len(self.videos)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a video sequence.

        Args:
            idx: Video index

        Returns:
            sequence: Frame sequence tensor [T, 3, H, W]
            label: Class label (0=real, 1=fake)
        """
        video_dir, label, all_frames = self.videos[idx]

        # Sample fixed-length sequence
        sampled_frames = self._sample_frames(all_frames)

        # Load and transform frames
        frame_tensors = []
        for frame_path in sampled_frames:
            image = Image.open(frame_path).convert('RGB')
            tensor = self.transform(image)
            frame_tensors.append(tensor)

        # Stack into [T, C, H, W]
        sequence = torch.stack(frame_tensors, dim=0)
        return sequence, label

    def get_video_paths(self) -> List[str]:
        """Get all video directory paths (for splitting)."""
        return [v[0] for v in self.videos]
