"""
XceptionNet Mixed Training Script
Train XceptionNet on combined FaceForensics++ and Celeb-DF-v2 datasets.

This addresses overfitting to FF++ by training on diverse deepfake types.
Target: 70-75% AUC on Celeb-DF-v2 cross-dataset evaluation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler
from torch.amp import autocast, GradScaler
from pathlib import Path
from typing import Tuple, List
import yaml
import json
import argparse
from datetime import datetime
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score

from src.models.xception import create_xception
from src.training.dataset import DeepfakeDataset


class MixedDatasetLoader:
    """
    Creates dataloaders that combine multiple deepfake datasets.
    Ensures balanced sampling from each dataset during training.
    """

    def __init__(
        self,
        dataset_dirs: List[Tuple[str, Path]],
        batch_size: int = 32,
        num_workers: int = 4,
        train_split: float = 0.8,
        val_split: float = 0.1,
        image_size: int = 299,
        frames_per_video: int = 5,
        normalize_mean: List[float] = None,
        normalize_std: List[float] = None,
        augmentation: str = 'standard',
        use_weighted_sampler: bool = False,
    ):
        """
        Initialize mixed dataset loader.

        Args:
            dataset_dirs: List of (name, path) tuples for each dataset
            batch_size: Batch size
            num_workers: Number of data loading workers
            train_split: Fraction for training
            val_split: Fraction for validation
            image_size: Image size
            frames_per_video: Frames per video to use
            normalize_mean: Per-channel normalization mean (default [0.5, 0.5, 0.5])
            normalize_std: Per-channel normalization std (default [0.5, 0.5, 0.5])
            augmentation: 'standard' or 'robust' - propagated to the train split
                only. See src/training/augmentations.py.
        """
        self.dataset_dirs = dataset_dirs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_split = train_split
        self.val_split = val_split
        self.image_size = image_size
        self.frames_per_video = frames_per_video
        self.normalize_mean = normalize_mean if normalize_mean is not None else [0.5, 0.5, 0.5]
        self.normalize_std = normalize_std if normalize_std is not None else [0.5, 0.5, 0.5]
        self.augmentation = augmentation
        # When True, the train DataLoader uses WeightedRandomSampler so batches
        # are class-balanced without duplicating minority samples in memory.
        # This replaces the old `balance_classes=True` oversampling path that
        # was causing the v2 checkpoint to memorise Celeb-DF's 890 real videos.
        self.use_weighted_sampler = use_weighted_sampler

    def create_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create train, validation, and test dataloaders from multiple datasets.

        Returns:
            train_loader, val_loader, test_loader
        """
        train_datasets = []
        val_datasets = []
        test_datasets = []


        for entry in self.dataset_dirs:
            # Entry formats (oldest -> newest):
            #   (name, path)
            #   (name, path, flat_layout)
            #   dict with keys: name, path, flat_layout, frames_per_video,
            #                   exclude_prefixes, exclude_classes
            if isinstance(entry, dict):
                dataset_name = entry["name"]
                dataset_path = entry["path"]
                flat_layout = entry.get("flat_layout", False)
                per_dataset_frames = entry.get("frames_per_video", self.frames_per_video)
                exclude_prefixes = entry.get("exclude_prefixes", None)
                exclude_classes = entry.get("exclude_classes", None)
            elif len(entry) == 3:
                dataset_name, dataset_path, flat_layout = entry
                per_dataset_frames = self.frames_per_video
                exclude_prefixes = None
                exclude_classes = None
            else:
                dataset_name, dataset_path = entry
                flat_layout = False
                per_dataset_frames = self.frames_per_video
                exclude_prefixes = None
                exclude_classes = None

            print(f"\nLoading {dataset_name} from: {dataset_path}"
                  + (" (flat layout)" if flat_layout else "")
                  + (f" [frames_per_video={per_dataset_frames}]" if per_dataset_frames != self.frames_per_video else "")
                  + (f" [exclude_prefixes={exclude_prefixes}]" if exclude_prefixes else "")
                  + (f" [exclude_classes={exclude_classes}]" if exclude_classes else ""))

            if not dataset_path.exists():
                print(f"WARNING: Dataset path not found: {dataset_path}")
                continue

            # Create full dataset to get all videos
            full_dataset = DeepfakeDataset(
                root_dir=dataset_path,
                split='train',
                frames_per_video=-1,  # Use all frames to get video list
                balance_classes=False,
                image_size=self.image_size,
                flat_layout=flat_layout,
                exclude_video_prefixes=exclude_prefixes,
                exclude_classes=exclude_classes,
            )

            # Get unique videos
            video_paths = list(full_dataset.video_to_frames.keys())

            # Shuffle videos for random split
            np.random.shuffle(video_paths)

            # Split by videos (not frames) to prevent data leakage
            n_videos = len(video_paths)
            n_train = int(n_videos * self.train_split)
            n_val = int(n_videos * self.val_split)

            train_videos = set(video_paths[:n_train])
            val_videos = set(video_paths[n_train:n_train + n_val])
            test_videos = set(video_paths[n_train + n_val:])

            print(f"  Total videos: {n_videos}")
            print(f"  Train videos: {len(train_videos)}")
            print(f"  Val videos: {len(val_videos)}")
            print(f"  Test videos: {len(test_videos)}")

            # Create split datasets (pass per-dataset overrides through).
            train_ds = self._create_split_dataset(
                dataset_path, train_videos, 'train', flat_layout=flat_layout,
                frames_per_video=per_dataset_frames,
                exclude_prefixes=exclude_prefixes,
                exclude_classes=exclude_classes,
            )
            val_ds = self._create_split_dataset(
                dataset_path, val_videos, 'val', flat_layout=flat_layout,
                frames_per_video=per_dataset_frames,
                exclude_prefixes=exclude_prefixes,
                exclude_classes=exclude_classes,
            )
            test_ds = self._create_split_dataset(
                dataset_path, test_videos, 'test', flat_layout=flat_layout,
                frames_per_video=per_dataset_frames,
                exclude_prefixes=exclude_prefixes,
                exclude_classes=exclude_classes,
            )

            train_datasets.append(train_ds)
            val_datasets.append(val_ds)
            test_datasets.append(test_ds)

            print(f"  Train samples: {len(train_ds)}")
            print(f"  Val samples: {len(val_ds)}")
            print(f"  Test samples: {len(test_ds)}")

        # Combine datasets
        print("\nCombining datasets...")
        train_combined = ConcatDataset(train_datasets)
        val_combined = ConcatDataset(val_datasets)
        test_combined = ConcatDataset(test_datasets)

        print(f"Combined train samples: {len(train_combined)}")
        print(f"Combined val samples: {len(val_combined)}")
        print(f"Combined test samples: {len(test_combined)}")

        # Create dataloaders.
        # When `use_weighted_sampler` is on, build per-sample weights so that
        # each class contributes equal expected frequency per batch - but no
        # sample is ever duplicated in memory (unlike the old balance_classes
        # path which copied Celeb-DF's 890 reals 6x and caused memorisation).
        train_sampler = None
        train_shuffle = True
        if self.use_weighted_sampler:
            all_labels: List[int] = []
            for ds in train_datasets:
                all_labels.extend(ds.labels)
            labels_tensor = torch.tensor(all_labels, dtype=torch.long)
            class_counts = torch.bincount(labels_tensor, minlength=2).float()
            # weight_per_class is inverse-frequency; normalise so samples_per_epoch
            # matches the original dataset size.
            class_weights = 1.0 / class_counts.clamp(min=1.0)
            sample_weights = class_weights[labels_tensor]
            train_sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True,
            )
            train_shuffle = False
            real_count = int(class_counts[0].item())
            fake_count = int(class_counts[1].item())
            print(f"\n[sampler] WeightedRandomSampler active - "
                  f"real={real_count}, fake={fake_count}, "
                  f"expected batch balance 50/50 without duplicating samples.")

        train_loader = DataLoader(
            train_combined,
            batch_size=self.batch_size,
            shuffle=train_shuffle,
            sampler=train_sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )

        val_loader = DataLoader(
            val_combined,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

        test_loader = DataLoader(
            test_combined,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

        return train_loader, val_loader, test_loader

    def _create_split_dataset(
        self,
        root_dir: Path,
        videos: set,
        split: str,
        flat_layout: bool = False,
        frames_per_video: int = None,
        exclude_prefixes: List[str] = None,
        exclude_classes: List[str] = None,
    ) -> DeepfakeDataset:
        """Create dataset for a specific split."""
        # Augmentation only applies to the train split; val/test stay deterministic.
        split_augmentation = self.augmentation if split == 'train' else 'standard'
        # If the caller asked for a weighted sampler, disable the in-dataset
        # balance_classes duplication - the sampler handles class balance for
        # us without memorising the minority class.
        use_balance_classes = (split == 'train') and not self.use_weighted_sampler
        dataset = DeepfakeDataset(
            root_dir=root_dir,
            split=split,
            frames_per_video=frames_per_video if frames_per_video is not None else self.frames_per_video,
            balance_classes=use_balance_classes,
            image_size=self.image_size,
            normalize_mean=self.normalize_mean,
            normalize_std=self.normalize_std,
            augmentation=split_augmentation,
            flat_layout=flat_layout,
            exclude_video_prefixes=exclude_prefixes,
            exclude_classes=exclude_classes,
        )

        # Filter to only include videos in this split.
        # Nested layout: the video key is the per-video folder (sample.parent).
        # Flat layout: the video key is class_dir / video_id, where video_id is
        #   everything before the last underscore in the filename stem (matches
        #   DeepfakeDataset._load_samples_flat).
        filtered_samples = []
        filtered_labels = []

        for i, sample_path in enumerate(dataset.samples):
            if flat_layout:
                stem = sample_path.stem
                video_id = stem.rsplit('_', 1)[0] if '_' in stem else stem
                video_key = str(sample_path.parent / video_id)
            else:
                video_key = str(sample_path.parent)
            if video_key in videos:
                filtered_samples.append(sample_path)
                filtered_labels.append(dataset.labels[i])

        dataset.samples = filtered_samples
        dataset.labels = filtered_labels

        return dataset


class Trainer:
    """Trainer class for mixed dataset training."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        device: torch.device
    ):
        """Initialize trainer."""
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Optimizer
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config['training']['optimizer']['learning_rate'],
            weight_decay=config['training']['optimizer'].get('weight_decay', 0)
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['training']['epochs'],
            eta_min=config['training']['scheduler'].get('min_lr', 1e-7)
        )

        # Mixed precision scaler
        self.scaler = GradScaler('cuda') if config['hardware'].get('mixed_precision', True) else None

        # Training state
        self.current_epoch = 0
        self.best_val_auc = 0.0
        self.early_stop_counter = 0

        # History
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_auc': [],
            'lr': []
        }

        # Output directories
        self.checkpoint_dir = Path(config['training']['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nTrainer initialized on {device}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")

        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass with mixed precision
            if self.scaler is not None:
                with autocast('cuda'):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                # Backward pass
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })

        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total

        return avg_loss, accuracy

    @torch.no_grad()
    def validate(self) -> Tuple[float, float, float]:
        """Validate the model."""
        self.model.eval()

        total_loss = 0.0
        all_labels = []
        all_probs = []
        all_preds = []

        for images, labels in tqdm(self.val_loader, desc="Validating"):
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            total_loss += loss.item()

            # Get predictions and probabilities
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of fake
            all_preds.extend(predicted.cpu().numpy())

        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * accuracy_score(all_labels, all_preds)

        # Calculate AUC
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            auc = 0.5  # If only one class present

        return avg_loss, accuracy, auc

    def save_checkpoint(self, filename: str, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_auc': self.best_val_auc,
            'history': self.history,
            'config': self.config
        }

        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        save_path = self.checkpoint_dir / filename
        torch.save(checkpoint, save_path)

        if is_best:
            best_path = self.checkpoint_dir / 'best_model_mixed.pth'
            torch.save(checkpoint, best_path)
            print(f"Saved best model with AUC: {self.best_val_auc:.4f}")

    def train(self, epochs: int):
        """Main training loop."""
        print(f"\nStarting training for {epochs} epochs")
        print("=" * 60)

        early_stop_patience = self.config['training']['early_stopping']['patience']

        for epoch in range(epochs):
            self.current_epoch = epoch

            # Train
            train_loss, train_acc = self.train_epoch()

            # Validate
            val_loss, val_acc, val_auc = self.validate()

            # Update learning rate
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']

            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_auc'].append(val_auc)
            self.history['lr'].append(current_lr)

            # Print epoch summary
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val AUC: {val_auc:.4f}")
            print(f"  Learning Rate: {current_lr:.6f}")

            # Check for improvement
            # min_delta: improvement must exceed this threshold to reset patience counter
            min_delta = self.config['training']['early_stopping'].get('min_delta', 0.001)
            is_best = val_auc > self.best_val_auc
            improved_enough = val_auc > self.best_val_auc + min_delta
            if is_best:
                self.best_val_auc = val_auc
            if improved_enough:
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1

            # Save checkpoint
            self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pth', is_best)

            # Early stopping
            if self.early_stop_counter >= early_stop_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break

        print("\nTraining complete!")
        print(f"Best validation AUC: {self.best_val_auc:.4f}")

        # Save final history
        history_path = self.checkpoint_dir / 'training_history_mixed.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description='Train XceptionNet on mixed FF++ and Celeb-DF-v2'
    )
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to config file')
    parser.add_argument('--ff-dir', type=str,
                        default=None,
                        help='Path to FF++ processed dataset (default: from config.yaml)')
    parser.add_argument('--celebdf-dir', type=str,
                        default=None,
                        help='Path to Celeb-DF-v2 processed dataset (default: from config.yaml)')
    parser.add_argument('--output-dir', type=str,
                        default=None,
                        help='Directory to save model checkpoints (default: from config.yaml)')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')

    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        # Use default config
        config = {
            'training': {
                'batch_size': args.batch_size,
                'epochs': args.epochs,
                'optimizer': {
                    'learning_rate': 0.0001,
                    'weight_decay': 0.00001
                },
                'scheduler': {
                    'min_lr': 0.000001
                },
                'early_stopping': {
                    'patience': 7
                },
                'checkpoint_dir': args.output_dir
            },
            'hardware': {
                'mixed_precision': True
            },
            'model': {
                'xception': {
                    'num_classes': 2,
                    'dropout': 0.5,
                    'pretrained': True
                }
            }
        }

    # Resolve paths from config if not provided on command line
    if args.ff_dir is None:
        args.ff_dir = config.get('data', {}).get('processed', {}).get('faceforensics', 'data/processed/FaceForensics++_AllTypes')
    if args.celebdf_dir is None:
        args.celebdf_dir = config.get('data', {}).get('processed', {}).get('celebdf', 'data/processed/Celeb-DF-v2')
    if args.output_dir is None:
        models_dir = config.get('output', {}).get('models_dir', 'outputs/models')
        args.output_dir = str(Path(models_dir) / 'mixed')

    # Override config with command line arguments
    config['training']['batch_size'] = args.batch_size
    config['training']['epochs'] = args.epochs
    config['training']['checkpoint_dir'] = args.output_dir

    # Setup device, read preference from config, fall back to CPU if CUDA unavailable
    cfg_device = config.get('hardware', {}).get('device', 'cuda')
    if cfg_device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        cfg_device = 'cpu'
    device = torch.device(cfg_device)
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Create model
    print("\nCreating XceptionNet model...")
    model = create_xception(
        num_classes=config['model']['xception']['num_classes'],
        dropout=config['model']['xception']['dropout'],
        pretrained=config['model']['xception']['pretrained']
    )

    # Setup datasets
    dataset_dirs = [
        ('FaceForensics++', Path(args.ff_dir)),
        ('Celeb-DF-v2', Path(args.celebdf_dir))
    ]

    # Create mixed dataloaders
    pp = config.get('preprocessing', {})
    img_cfg = pp.get('image', {})
    train_cfg = config.get('training', {})
    mixed_loader = MixedDatasetLoader(
        dataset_dirs=dataset_dirs,
        batch_size=train_cfg['batch_size'],
        num_workers=train_cfg.get('num_workers', 4),
        train_split=train_cfg.get('train_split', 0.8),
        val_split=train_cfg.get('val_split', 0.1),
        image_size=img_cfg.get('size', 299),
        frames_per_video=pp.get('training_frames_per_video', 5),
        normalize_mean=img_cfg.get('normalize_mean', [0.5, 0.5, 0.5]),
        normalize_std=img_cfg.get('normalize_std', [0.5, 0.5, 0.5])
    )

    train_loader, val_loader, test_loader = mixed_loader.create_dataloaders()

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )

    # Train
    trainer.train(config['training']['epochs'])

    # Final evaluation on test set
    print("\nFinal evaluation on combined test set...")
    trainer.val_loader = test_loader
    test_loss, test_acc, test_auc = trainer.validate()
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"Test AUC: {test_auc:.4f}")

    # Save final test results
    test_results = {
        'test_loss': float(test_loss),
        'test_accuracy': float(test_acc),
        'test_auc': float(test_auc),
        'timestamp': datetime.now().isoformat()
    }

    results_path = Path(args.output_dir) / 'test_results_mixed.json'
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(test_results, f, indent=2)

    print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    main()