"""
Temporal LSTM Training Script
Train an LSTM temporal model on top of a frozen pre-trained XceptionNet backbone.

The backbone extracts 2048-dim spatial embeddings per frame; the LSTM learns
to detect temporal inconsistencies across the frame sequence.

v3 revision (2026-04-21): adds the same shortcut-mitigation filters used by
train_xception_robust.py and train_ensemble.py (drop Celeb-DF real_youtube,
drop FF++ DeepFakeDetection_* overlap, optional DFD inclusion). Also defaults
the backbone to robust_v3's best_model_robust.pth, so the LSTM trains on top
of shortcut-free per-frame embeddings rather than v2 ones. WildDeepfake is
excluded from temporal training because SequenceDataset does not yet support
the flat layout that WildDeepfake uses.

Usage:
    python scripts/train_temporal.py
    python scripts/train_temporal.py --include-dfd --epochs 30
    python scripts/train_temporal.py --backbone-checkpoint outputs/models/mixed/best_model_mixed.pth
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torch.amp import autocast, GradScaler
from typing import Tuple, List, Optional
import yaml
import json
import argparse
from datetime import datetime
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score

from src.models.temporal import create_temporal
from src.training.sequence_dataset import SequenceDataset


class MixedSequenceLoader:
    """
    Creates dataloaders of video sequences from multiple datasets.
    Each sample is a full video (sequence of frames), not individual frames.

    Dataset entries accept either legacy (name, path) tuples or v3-style dicts:
        {
            'name': 'FaceForensics++',
            'path': Path(...),
            'exclude_classes': ['real_youtube'],        # optional
            'exclude_prefixes': ['DeepFakeDetection_'], # optional
        }
    """

    def __init__(
        self,
        dataset_dirs: List,
        batch_size: int = 4,
        num_workers: int = 2,
        train_split: float = 0.8,
        val_split: float = 0.1,
        sequence_length: int = 32,
        image_size: int = 299,
        normalize_mean: Optional[List[float]] = None,
        normalize_std: Optional[List[float]] = None,
    ):
        self.dataset_dirs = dataset_dirs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_split = train_split
        self.val_split = val_split
        self.sequence_length = sequence_length
        self.image_size = image_size
        self.normalize_mean = normalize_mean or [0.5, 0.5, 0.5]
        self.normalize_std = normalize_std or [0.5, 0.5, 0.5]

    @staticmethod
    def _normalise_entry(entry) -> dict:
        """Accept both (name, path) tuples and dict entries."""
        if isinstance(entry, dict):
            return entry
        name, path = entry
        return {'name': name, 'path': path}

    def create_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        train_datasets = []
        val_datasets = []
        test_datasets = []

        for raw_entry in self.dataset_dirs:
            entry = self._normalise_entry(raw_entry)
            dataset_name = entry['name']
            dataset_path = Path(entry['path'])
            exclude_classes = entry.get('exclude_classes')
            exclude_prefixes = entry.get('exclude_prefixes')
            flat_layout = bool(entry.get('flat_layout', False))

            print(f"\nLoading {dataset_name} from: {dataset_path}")
            if flat_layout:
                print(f"  flat_layout=True")
            if exclude_classes:
                print(f"  exclude_classes={exclude_classes}")
            if exclude_prefixes:
                print(f"  exclude_video_prefixes={exclude_prefixes}")

            if not dataset_path.exists():
                print(f"WARNING: Dataset path not found: {dataset_path}")
                continue

            # Load all videos (filters applied inside SequenceDataset._load_videos)
            full_dataset = SequenceDataset(
                root_dir=dataset_path,
                split='all',
                sequence_length=self.sequence_length,
                image_size=self.image_size,
                normalize_mean=self.normalize_mean,
                normalize_std=self.normalize_std,
                augment=False,
                exclude_classes=exclude_classes,
                exclude_video_prefixes=exclude_prefixes,
                flat_layout=flat_layout,
            )

            # Split by video
            video_indices = list(range(len(full_dataset.videos)))
            np.random.shuffle(video_indices)

            n_videos = len(video_indices)
            n_train = int(n_videos * self.train_split)
            n_val = int(n_videos * self.val_split)

            train_indices = set(video_indices[:n_train])
            val_indices = set(video_indices[n_train:n_train + n_val])
            test_indices = set(video_indices[n_train + n_val:])

            print(f"  Total videos: {n_videos}")
            print(f"  Train: {len(train_indices)}, Val: {len(val_indices)}, "
                  f"Test: {len(test_indices)}")

            # Create split datasets by filtering videos
            train_ds = self._create_split(dataset_path, full_dataset,
                                          train_indices, 'train', augment=True,
                                          exclude_classes=exclude_classes,
                                          exclude_prefixes=exclude_prefixes,
                                          flat_layout=flat_layout)
            val_ds = self._create_split(dataset_path, full_dataset,
                                        val_indices, 'val', augment=False,
                                        exclude_classes=exclude_classes,
                                        exclude_prefixes=exclude_prefixes,
                                        flat_layout=flat_layout)
            test_ds = self._create_split(dataset_path, full_dataset,
                                         test_indices, 'test', augment=False,
                                         exclude_classes=exclude_classes,
                                         exclude_prefixes=exclude_prefixes,
                                         flat_layout=flat_layout)

            train_datasets.append(train_ds)
            val_datasets.append(val_ds)
            test_datasets.append(test_ds)

        print("\nCombining datasets...")
        train_combined = ConcatDataset(train_datasets)
        val_combined = ConcatDataset(val_datasets)
        test_combined = ConcatDataset(test_datasets)

        print(f"Combined - Train: {len(train_combined)} videos, "
              f"Val: {len(val_combined)}, Test: {len(test_combined)}")

        train_loader = DataLoader(
            train_combined, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True, drop_last=True
        )
        val_loader = DataLoader(
            val_combined, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True
        )
        test_loader = DataLoader(
            test_combined, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True
        )

        return train_loader, val_loader, test_loader

    def _create_split(self, root_dir, full_dataset, indices, split, augment,
                      exclude_classes=None, exclude_prefixes=None,
                      flat_layout=False):
        """Create a dataset containing only the videos at the given indices."""
        dataset = SequenceDataset(
            root_dir=root_dir,
            split=split,
            sequence_length=self.sequence_length,
            image_size=self.image_size,
            normalize_mean=self.normalize_mean,
            normalize_std=self.normalize_std,
            augment=augment,
            exclude_classes=exclude_classes,
            exclude_video_prefixes=exclude_prefixes,
            flat_layout=flat_layout,
        )
        # Filter to only videos at specified indices
        dataset.videos = [full_dataset.videos[i] for i in sorted(indices)]
        return dataset


class TemporalTrainer:
    """Trainer for the temporal LSTM model."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        device: torch.device,
        output_dir: str
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        self.criterion = nn.CrossEntropyLoss()

        # Only optimise trainable parameters (LSTM + classifier)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(
            trainable_params,
            lr=config['training']['optimizer']['learning_rate'],
            weight_decay=config['training']['optimizer'].get('weight_decay', 0)
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['training']['epochs'],
            eta_min=config['training']['scheduler'].get('min_lr', 1e-7)
        )

        self.scaler = GradScaler('cuda') if (
            config['hardware'].get('mixed_precision', True) and device.type == 'cuda'
        ) else None

        self.current_epoch = 0
        self.best_val_auc = 0.0
        self.early_stop_counter = 0

        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [], 'val_auc': [], 'lr': []
        }

        self.checkpoint_dir = Path(output_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        trainable_count = sum(p.numel() for p in trainable_params)
        print(f"\nTemporalTrainer initialized on {device}")
        print(f"Trainable parameters: {trainable_count:,}")

    def train_epoch(self) -> Tuple[float, float]:
        self.model.train()
        # Keep backbone in eval mode
        self.model.backbone.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        for batch_idx, (sequences, labels) in enumerate(pbar):
            # sequences: [B, T, C, H, W]
            sequences = sequences.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            if self.scaler is not None:
                with autocast('cuda'):
                    outputs = self.model(sequences)
                    loss = self.criterion(outputs, labels)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(sequences)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })

        return total_loss / len(self.train_loader), 100. * correct / total

    @torch.no_grad()
    def validate(self) -> Tuple[float, float, float]:
        self.model.eval()
        total_loss = 0.0
        all_labels = []
        all_probs = []
        all_preds = []

        for sequences, labels in tqdm(self.val_loader, desc="Validating"):
            sequences = sequences.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(sequences)
            loss = self.criterion(outputs, labels)
            total_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * accuracy_score(all_labels, all_preds)

        try:
            auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            auc = 0.5

        return avg_loss, accuracy, auc

    def save_checkpoint(self, filename: str, is_best: bool = False):
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_auc': self.best_val_auc,
            'history': self.history,
            'config': self.config,
            'model_type': 'temporal'
        }
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        torch.save(checkpoint, self.checkpoint_dir / filename)

        if is_best:
            best_path = self.checkpoint_dir / 'best_model_temporal.pth'
            torch.save(checkpoint, best_path)
            print(f"Saved best temporal model with AUC: {self.best_val_auc:.4f}")

    def train(self, epochs: int):
        print(f"\nStarting temporal training for {epochs} epochs")
        print("=" * 60)

        early_stop_patience = self.config['training']['early_stopping']['patience']

        for epoch in range(epochs):
            self.current_epoch = epoch

            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc, val_auc = self.validate()

            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']

            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_auc'].append(val_auc)
            self.history['lr'].append(current_lr)

            print(f"\nEpoch {epoch + 1}/{epochs}")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, "
                  f"Val AUC: {val_auc:.4f}")
            print(f"  Learning Rate: {current_lr:.6f}")

            min_delta = self.config['training']['early_stopping'].get('min_delta', 0.001)
            is_best = val_auc > self.best_val_auc
            improved_enough = val_auc > self.best_val_auc + min_delta
            if is_best:
                self.best_val_auc = val_auc
            if improved_enough:
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1

            self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pth', is_best)

            if self.early_stop_counter >= early_stop_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break

        print("\nTraining complete!")
        print(f"Best validation AUC: {self.best_val_auc:.4f}")

        history_path = self.checkpoint_dir / 'training_history_temporal.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description='Train temporal LSTM model on mixed datasets'
    )
    parser.add_argument('--config', type=str, default='config/config.yaml')
    parser.add_argument('--backbone-checkpoint', type=str, default=None,
                        help='Path to pre-trained XceptionNet checkpoint '
                             '(default: outputs/models/robust_v3/best_model_robust.pth - the '
                             'shortcut-mitigated backbone. Pass outputs/models/mixed/best_model_mixed.pth '
                             'to match the pre-v3 temporal training.)')
    parser.add_argument('--ff-dir', type=str, default=None)
    parser.add_argument('--celebdf-dir', type=str, default=None)
    parser.add_argument('--dfd-dir', type=str, default=None,
                        help='Override preprocessed DFD directory (default: <processed_root>/DFD).')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Default: outputs/models/temporal_v3')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size in videos (default 4 - each video is '
                             '32 frames through backbone)')
    parser.add_argument('--sequence-length', type=int, default=32,
                        help='Number of frames per video sequence')
    parser.add_argument('--lstm-hidden', type=int, default=None)
    parser.add_argument('--lstm-layers', type=int, default=None)
    parser.add_argument('--bidirectional', action='store_true', default=False)
    parser.add_argument('--num-workers', type=int, default=None,
                        help='Override DataLoader num_workers. Windows: 0-2 if you hit error 1455.')

    # v3 shortcut-mitigation flags (mirror scripts/train_xception_robust.py).
    parser.add_argument('--include-dfd', action='store_true',
                        help='Include DFD (DeepFakeDetection) standalone release in training mix.')
    parser.add_argument('--include-wilddeepfake', action='store_true',
                        help='Include WildDeepfake (train split) in training mix. '
                             'SequenceDataset uses flat_layout grouping by <video_id>_<frame_idx>.png.')
    parser.add_argument('--wilddeepfake-dir', type=str, default=None,
                        help='Override WildDeepfake train directory (default: data/wilddeepfake/train).')
    parser.add_argument('--keep-dfd-in-ffpp', action='store_true',
                        help='Keep DeepFakeDetection_* videos inside FF++ even when --include-dfd is set. '
                             'Default: drop them to avoid double-counting (v2 shortcut).')
    parser.add_argument('--keep-real-youtube', action='store_true',
                        help="Keep Celeb-DF's real_youtube/ class in training. "
                             "Default: drop it to remove the 'YouTube codec -> real' shortcut.")

    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        print(f"[!] Config not found at {config_path}, using defaults")
        config = {
            'training': {
                'batch_size': args.batch_size, 'epochs': args.epochs,
                'optimizer': {'learning_rate': 0.0001, 'weight_decay': 0.00001},
                'scheduler': {'min_lr': 0.000001},
                'early_stopping': {'patience': 7, 'min_delta': 0.001},
            },
            'hardware': {'mixed_precision': True, 'device': 'cuda'},
            'model': {
                'xception': {'num_classes': 2, 'dropout': 0.5, 'pretrained': True},
                'temporal': {
                    'lstm_hidden': 512, 'lstm_layers': 2,
                    'lstm_dropout': 0.3, 'classifier_dropout': 0.5,
                    'bidirectional': False, 'sequence_length': 32
                }
            }
        }

    # Resolve paths
    if args.ff_dir is None:
        args.ff_dir = config.get('data', {}).get('processed', {}).get(
            'faceforensics', 'data/processed/FaceForensics++_AllTypes')
    if args.celebdf_dir is None:
        args.celebdf_dir = config.get('data', {}).get('processed', {}).get(
            'celebdf', 'data/processed/Celeb-DF-v2')
    models_dir_path = Path(config.get('output', {}).get('models_dir', 'outputs/models'))
    if args.output_dir is None:
        args.output_dir = str(models_dir_path / 'temporal_v3')

    if args.backbone_checkpoint is None:
        args.backbone_checkpoint = str(
            models_dir_path / 'robust_v3' / 'best_model_robust.pth'
        )

    if args.num_workers is not None:
        config.setdefault('training', {})['num_workers'] = args.num_workers

    # Resolve LSTM config
    temporal_cfg = config.get('model', {}).get('temporal', {})
    lstm_hidden = args.lstm_hidden or temporal_cfg.get('lstm_hidden', 512)
    lstm_layers = args.lstm_layers or temporal_cfg.get('lstm_layers', 2)
    lstm_dropout = temporal_cfg.get('lstm_dropout', 0.3)
    classifier_dropout = temporal_cfg.get('classifier_dropout', 0.5)
    bidirectional = args.bidirectional or temporal_cfg.get('bidirectional', False)
    sequence_length = args.sequence_length or temporal_cfg.get('sequence_length', 32)

    # Override config
    config['training']['batch_size'] = args.batch_size
    config['training']['epochs'] = args.epochs

    # Setup device
    cfg_device = config.get('hardware', {}).get('device', 'cuda')
    if cfg_device == 'cuda' and not torch.cuda.is_available():
        print("[!] CUDA not available, falling back to CPU")
        cfg_device = 'cpu'
    device = torch.device(cfg_device)
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"Memory: {props.total_memory / 1e9:.1f} GB")

    # Create temporal model
    print("\nCreating temporal model (XceptionNet backbone + LSTM)...")
    xception_cfg = config.get('model', {}).get('xception', {})

    model = create_temporal(
        num_classes=xception_cfg.get('num_classes', 2),
        backbone_dropout=xception_cfg.get('dropout', 0.5),
        pretrained_backbone=False,  # We'll load weights from checkpoint
        lstm_hidden=lstm_hidden,
        lstm_layers=lstm_layers,
        lstm_dropout=lstm_dropout,
        classifier_dropout=classifier_dropout,
        bidirectional=bidirectional,
        freeze_backbone=True
    )

    # Load pre-trained backbone weights
    backbone_path = Path(args.backbone_checkpoint)
    if backbone_path.exists():
        model.load_backbone_weights(str(backbone_path), device)
    else:
        print(f"[!] WARNING: Backbone checkpoint not found: {backbone_path}")
        print("    Training will use random ImageNet-pretrained backbone")

    # Setup datasets - v3 shortcut-mitigation mirrors scripts/train_xception_robust.py
    processed_root = Path(
        config.get('data', {}).get('processed', {}).get('root', 'data/processed')
    )
    data_root = Path(config.get('data', {}).get('root', 'data'))

    ffpp_exclude_prefixes = (
        ['DeepFakeDetection_'] if args.include_dfd and not args.keep_dfd_in_ffpp else None
    )
    celebdf_exclude_classes = None if args.keep_real_youtube else ['real_youtube']

    dataset_entries: list = [
        {
            'name': 'FaceForensics++',
            'path': Path(args.ff_dir),
            'exclude_prefixes': ffpp_exclude_prefixes,
        },
        {
            'name': 'Celeb-DF-v2',
            'path': Path(args.celebdf_dir),
            'exclude_classes': celebdf_exclude_classes,
        },
    ]
    if ffpp_exclude_prefixes:
        print(f"[+] FF++ filter: excluding videos starting with {ffpp_exclude_prefixes} "
              f"(DFD standalone release supplies these instead)")
    if celebdf_exclude_classes:
        print(f"[+] Celeb-DF filter: excluding classes {celebdf_exclude_classes} "
              f"('YouTube -> real' shortcut mitigation)")
    if args.include_dfd:
        dfd_dir = Path(args.dfd_dir) if args.dfd_dir else processed_root / 'DFD'
        dataset_entries.append({'name': 'DFD', 'path': dfd_dir})
        print(f"[+] Including DFD from: {dfd_dir}")
    if args.include_wilddeepfake:
        wd_dir = (
            Path(args.wilddeepfake_dir)
            if args.wilddeepfake_dir
            else data_root / 'wilddeepfake' / 'train'
        )
        dataset_entries.append({
            'name': 'WildDeepfake',
            'path': wd_dir,
            'flat_layout': True,
        })
        print(f"[+] Including WildDeepfake (flat layout) from: {wd_dir}")

    pp = config.get('preprocessing', {})
    img_cfg = pp.get('image', {})

    mixed_loader = MixedSequenceLoader(
        dataset_dirs=dataset_entries,
        batch_size=args.batch_size,
        num_workers=config.get('training', {}).get('num_workers', 2),
        train_split=config.get('training', {}).get('train_split', 0.8),
        val_split=config.get('training', {}).get('val_split', 0.1),
        sequence_length=sequence_length,
        image_size=img_cfg.get('size', 299),
        normalize_mean=img_cfg.get('normalize_mean', [0.5, 0.5, 0.5]),
        normalize_std=img_cfg.get('normalize_std', [0.5, 0.5, 0.5])
    )

    train_loader, val_loader, test_loader = mixed_loader.create_dataloaders()

    # Create trainer
    trainer = TemporalTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        output_dir=args.output_dir
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

    # Save test results
    test_results = {
        'test_loss': float(test_loss),
        'test_accuracy': float(test_acc),
        'test_auc': float(test_auc),
        'model_type': 'temporal',
        'lstm_hidden': lstm_hidden,
        'lstm_layers': lstm_layers,
        'bidirectional': bidirectional,
        'sequence_length': sequence_length,
        'backbone_checkpoint': str(backbone_path),
        'timestamp': datetime.now().isoformat()
    }

    results_path = Path(args.output_dir) / 'test_results_temporal.json'
    with open(results_path, 'w') as f:
        json.dump(test_results, f, indent=2)

    print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    main()
