"""
PyTorch Dataset for Deepfake Detection
Handles loading of preprocessed face images for training and evaluation.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Union
import cv2
import numpy as np
from PIL import Image
import json
import random


class DeepfakeDataset(Dataset):
    """
    Dataset for deepfake detection.
    
    Loads preprocessed face images from directory structure:
    root/
        real/
            video1/
                face_0000.jpg
                face_0001.jpg
            video2/
                ...
        fake/
            video1/
                ...
    """
    
    def __init__(
        self,
        root_dir: Union[str, Path],
        split: str = 'train',
        transform: Optional[transforms.Compose] = None,
        frames_per_video: int = 1,
        balance_classes: bool = True,
        image_size: int = 299,
        normalize_mean: Optional[List[float]] = None,
        normalize_std: Optional[List[float]] = None,
        augmentation: str = 'standard',
        flat_layout: bool = False,
        exclude_video_prefixes: Optional[List[str]] = None,
        exclude_classes: Optional[List[str]] = None,
    ):
        """
        Initialize the dataset.

        Args:
            root_dir: Root directory. In nested layout (default) contains
                'real/<video_id>/face_*.jpg' subdirectories. In flat layout
                contains 'real/<video_id>_<frame>.png' files directly (used
                for WildDeepfake which ships as pre-extracted face PNGs).
            split: Data split ('train', 'val', 'test')
            transform: Image transforms to apply
            frames_per_video: Number of frames to sample per video
            balance_classes: Whether to balance class distribution
            image_size: Expected image size
            normalize_mean: Per-channel normalization mean (default [0.5, 0.5, 0.5])
            normalize_std: Per-channel normalization std (default [0.5, 0.5, 0.5])
            augmentation: 'standard' (baseline flip/rotation/colour-jitter) or
                'robust' (adds JPEG re-encoding, downscale/upscale, blur, noise
                - see src/training/augmentations.py). Only affects split='train'.
            flat_layout: If True, parse 'real/<video_id>_<frame>.png' to group
                frames by video_id prefix. If False (default), expect the
                nested 'real/<video_id>/face_*.jpg' layout our preprocessing
                pipeline produces.
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.frames_per_video = frames_per_video
        self.balance_classes = balance_classes
        self.image_size = image_size
        self.normalize_mean = normalize_mean if normalize_mean is not None else [0.5, 0.5, 0.5]
        self.normalize_std = normalize_std if normalize_std is not None else [0.5, 0.5, 0.5]
        self.augmentation = augmentation
        self.flat_layout = flat_layout
        # Filters used to combat dataset-level shortcuts discovered in v2:
        #   - exclude_video_prefixes: e.g. ["DeepFakeDetection_"] when the
        #     DFD standalone release is also loaded, to prevent seeing the
        #     same 1000 videos twice per epoch.
        #   - exclude_classes: e.g. ["real_youtube"] on Celeb-DF, to remove
        #     the "YouTube codec -> real" shortcut.
        self.exclude_video_prefixes = tuple(exclude_video_prefixes or ())
        self.exclude_classes = set(exclude_classes or ())

        # Default transforms if none provided
        if transform is None:
            self.transform = self._get_default_transforms(split)
        else:
            self.transform = transform
        
        # Load samples
        self.samples = []
        self.labels = []
        self.video_to_frames = {}
        
        self._load_samples()
        
        print(f"DeepfakeDataset ({split}): {len(self.samples)} samples")
        print(f"  Real: {sum(1 for l in self.labels if l == 0)}")
        print(f"  Fake: {sum(1 for l in self.labels if l == 1)}")
    
    def _get_default_transforms(self, split: str) -> transforms.Compose:
        """Get default transforms based on split and augmentation mode."""
        normalize = transforms.Normalize(
            mean=self.normalize_mean,
            std=self.normalize_std
        )

        if split == 'train':
            if self.augmentation == 'robust':
                # Imported lazily so non-training code paths don't pay the cost.
                from src.training.augmentations import get_robust_train_transforms
                return get_robust_train_transforms(
                    image_size=self.image_size,
                    normalize_mean=self.normalize_mean,
                    normalize_std=self.normalize_std,
                )
            return transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
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
    
    def _load_samples(self):
        """Load all samples from directory structure."""
        if self.flat_layout:
            self._load_samples_flat()
        else:
            self._load_samples_nested()

        # Balance classes if requested
        if self.balance_classes and self.split == 'train':
            self._balance_classes()

    def _load_samples_nested(self):
        """Standard layout: root/{real,fake}/<video_id>/face_*.jpg"""
        class_dirs = {
            'real': 0,
            'fake': 1,
            # Also handle Celeb-DF structure
            'real_celeb': 0,
            'real_youtube': 0
        }

        for class_name, label in class_dirs.items():
            if class_name in self.exclude_classes:
                continue
            class_dir = self.root_dir / class_name

            if not class_dir.exists():
                continue

            video_dirs = [d for d in class_dir.iterdir() if d.is_dir()]

            for video_dir in video_dirs:
                if self.exclude_video_prefixes and video_dir.name.startswith(self.exclude_video_prefixes):
                    continue
                face_images = list(video_dir.glob('*.jpg')) + list(video_dir.glob('*.png'))

                if not face_images:
                    continue

                self.video_to_frames[str(video_dir)] = {
                    'frames': face_images,
                    'label': label
                }

                if self.frames_per_video == -1:
                    for img_path in face_images:
                        self.samples.append(img_path)
                        self.labels.append(label)
                else:
                    sampled = random.sample(
                        face_images,
                        min(self.frames_per_video, len(face_images))
                    )
                    for img_path in sampled:
                        self.samples.append(img_path)
                        self.labels.append(label)

    def _load_samples_flat(self):
        """Flat layout: root/{real,fake}/<video_id>_<frame>.png

        Used for WildDeepfake, which ships as pre-extracted 224x224 face PNGs
        named like '0_0.png', '0_1.png', '42_137.png'. We parse the video_id
        prefix (everything before the last underscore) so the MixedDatasetLoader
        can still split by video to prevent frame-level leakage.
        """
        class_dirs = {'real': 0, 'fake': 1}

        for class_name, label in class_dirs.items():
            if class_name in self.exclude_classes:
                continue
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                continue

            # Group files by video_id prefix.
            video_groups: Dict[str, List[Path]] = {}
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() not in ('.png', '.jpg', '.jpeg'):
                    continue
                stem = img_path.stem
                if self.exclude_video_prefixes and stem.startswith(self.exclude_video_prefixes):
                    continue
                # 'a_b_c_42' -> video_id='a_b_c', frame='42'. Fall back to the
                # whole stem if there's no underscore (treat the file as its own video).
                if '_' in stem:
                    video_id = stem.rsplit('_', 1)[0]
                else:
                    video_id = stem
                # Namespace the synthetic key with the class folder so real/
                # video_id=0 doesn't collide with fake/video_id=0.
                key = str(class_dir / video_id)
                video_groups.setdefault(key, []).append(img_path)

            for video_key, frames in video_groups.items():
                self.video_to_frames[video_key] = {
                    'frames': frames,
                    'label': label,
                }

                if self.frames_per_video == -1:
                    for img_path in frames:
                        self.samples.append(img_path)
                        self.labels.append(label)
                else:
                    sampled = random.sample(
                        frames,
                        min(self.frames_per_video, len(frames))
                    )
                    for img_path in sampled:
                        self.samples.append(img_path)
                        self.labels.append(label)
    
    def _balance_classes(self):
        """Balance class distribution by oversampling minority class."""
        real_indices = [i for i, l in enumerate(self.labels) if l == 0]
        fake_indices = [i for i, l in enumerate(self.labels) if l == 1]
        
        if len(real_indices) == 0 or len(fake_indices) == 0:
            return
        
        # Oversample minority class
        if len(real_indices) < len(fake_indices):
            diff = len(fake_indices) - len(real_indices)
            extra_indices = random.choices(real_indices, k=diff)
            for idx in extra_indices:
                self.samples.append(self.samples[idx])
                self.labels.append(self.labels[idx])
        else:
            diff = len(real_indices) - len(fake_indices)
            extra_indices = random.choices(fake_indices, k=diff)
            for idx in extra_indices:
                self.samples.append(self.samples[idx])
                self.labels.append(self.labels[idx])
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample.
        
        Args:
            idx: Sample index
            
        Returns:
            image: Transformed image tensor
            label: Class label (0=real, 1=fake)
        """
        img_path = self.samples[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)
        
        return image, label
    
    def get_video_samples(self, video_path: str) -> List[Tuple[torch.Tensor, int]]:
        """Get all samples from a specific video."""
        if video_path not in self.video_to_frames:
            return []
        
        video_info = self.video_to_frames[video_path]
        samples = []
        
        for img_path in video_info['frames']:
            image = Image.open(img_path).convert('RGB')
            if self.transform is not None:
                image = self.transform(image)
            samples.append((image, video_info['label']))
        
        return samples


class VideoDataset(Dataset):
    """
    Dataset that loads videos directly (for testing/inference).
    Processes videos on-the-fly instead of using preprocessed faces.
    """
    
    def __init__(
        self,
        video_paths: List[Path],
        labels: List[int],
        transform: Optional[transforms.Compose] = None,
        frames_per_video: int = 32,
        face_extractor=None,
        face_aligner=None
    ):
        """
        Initialize video dataset.
        
        Args:
            video_paths: List of video file paths
            labels: List of labels (0=real, 1=fake)
            transform: Image transforms
            frames_per_video: Frames to extract per video
            face_extractor: FaceExtractor instance
            face_aligner: FaceAligner instance
        """
        self.video_paths = video_paths
        self.labels = labels
        self.transform = transform
        self.frames_per_video = frames_per_video
        self.face_extractor = face_extractor
        self.face_aligner = face_aligner
        
        if transform is None:
            # VideoDataset is not yet implemented (raises NotImplementedError in __getitem__)
            # Normalization values here are placeholders for when it is implemented
            self.transform = transforms.Compose([
                transforms.Resize((299, 299)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
    
    def __len__(self) -> int:
        return len(self.video_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get video frames as a tensor stack."""
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        # This would use VideoProcessor and FaceExtractor
        # For now, return placeholder
        # In practice, you'd process the video and extract faces
        
        raise NotImplementedError(
            "VideoDataset.__getitem__ requires face extraction implementation"
        )


def create_dataloaders(
    root_dir: Union[str, Path],
    batch_size: int = 32,
    num_workers: int = 4,
    train_split: float = 0.8,
    val_split: float = 0.1,
    image_size: int = 299,
    frames_per_video: int = 1,
    normalize_mean: Optional[List[float]] = None,
    normalize_std: Optional[List[float]] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.

    Args:
        root_dir: Root directory with processed faces
        batch_size: Batch size
        num_workers: Number of data loading workers
        train_split: Fraction for training
        val_split: Fraction for validation
        image_size: Image size
        frames_per_video: Frames per video to use
        normalize_mean: Per-channel normalization mean (default [0.5, 0.5, 0.5])
        normalize_std: Per-channel normalization std (default [0.5, 0.5, 0.5])

    Returns:
        train_loader, val_loader, test_loader
    """
    root_dir = Path(root_dir)

    # Create full dataset to get all samples
    full_dataset = DeepfakeDataset(
        root_dir=root_dir,
        split='train',
        frames_per_video=-1,  # Use all frames
        balance_classes=False,
        image_size=image_size,
        normalize_mean=normalize_mean,
        normalize_std=normalize_std
    )
    
    # Get unique videos
    video_paths = list(full_dataset.video_to_frames.keys())
    random.shuffle(video_paths)
    
    # Split by videos (not frames) to prevent data leakage
    n_videos = len(video_paths)
    n_train = int(n_videos * train_split)
    n_val = int(n_videos * val_split)
    
    train_videos = set(video_paths[:n_train])
    val_videos = set(video_paths[n_train:n_train + n_val])
    test_videos = set(video_paths[n_train + n_val:])
    
    # Create split datasets
    def create_split_dataset(videos: set, split: str) -> DeepfakeDataset:
        dataset = DeepfakeDataset(
            root_dir=root_dir,
            split=split,
            frames_per_video=frames_per_video,
            balance_classes=(split == 'train'),
            image_size=image_size,
            normalize_mean=normalize_mean,
            normalize_std=normalize_std
        )
        
        # Filter to only include videos in this split
        filtered_samples = []
        filtered_labels = []
        
        for i, sample_path in enumerate(dataset.samples):
            video_dir = str(sample_path.parent)
            if video_dir in videos:
                filtered_samples.append(sample_path)
                filtered_labels.append(dataset.labels[i])
        
        dataset.samples = filtered_samples
        dataset.labels = filtered_labels
        
        return dataset
    
    train_dataset = create_split_dataset(train_videos, 'train')
    val_dataset = create_split_dataset(val_videos, 'val')
    test_dataset = create_split_dataset(test_videos, 'test')
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


# Test function
def test_dataset():
    """Test the dataset class."""
    print("Testing DeepfakeDataset...")
    
    # This will only work if you have processed data
    test_root = Path("C:/FINAL YEAR PROJECT/data/processed/FaceForensics++")
    
    if test_root.exists():
        dataset = DeepfakeDataset(
            root_dir=test_root,
            split='train',
            frames_per_video=5
        )
        
        if len(dataset) > 0:
            image, label = dataset[0]
            print(f"Sample shape: {image.shape}")
            print(f"Sample label: {label}")
    else:
        print(f"Test directory not found: {test_root}")
        print("Run preprocessing first to create the dataset")
    
    print("Dataset test complete!")


if __name__ == '__main__':
    test_dataset()
