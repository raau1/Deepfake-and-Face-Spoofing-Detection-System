"""
Ensemble Training Script - XceptionNet + EfficientNet-B4
Train an ensemble model on combined deepfake datasets.

This builds on the mixed-dataset training approach (train_xception_mixed.py)
by replacing the single XceptionNet model with an ensemble of XceptionNet
and EfficientNet-B4, fusing their predictions for improved robustness.

v3 revision (2026-04-21): reuses the shared MixedDatasetLoader so the same
shortcut-mitigation toggles available on the robust Xception script (drop
Celeb-DF's real_youtube class, drop FF++'s DeepFakeDetection_* overlap with
the DFD standalone release, optionally include DFD and WildDeepfake) apply
here too. Without these the ensemble inherits the v2 shortcut learned by
its Xception backbone.

Usage:
    python scripts/train_ensemble.py
    python scripts/train_ensemble.py --include-dfd --include-wilddeepfake \\
        --fusion weighted --epochs 30 --batch-size 16
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from typing import Tuple
import yaml
import json
import argparse
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score

from src.models.ensemble import create_ensemble
from scripts.train_xception_mixed import MixedDatasetLoader


class Trainer:
    """Trainer class for ensemble model training."""

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

        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config['training']['optimizer']['learning_rate'],
            weight_decay=config['training']['optimizer'].get('weight_decay', 0)
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['training']['epochs'],
            eta_min=config['training']['scheduler'].get('min_lr', 1e-7)
        )

        self.scaler = GradScaler('cuda') if config['hardware'].get('mixed_precision', True) else None

        self.current_epoch = 0
        self.best_val_auc = 0.0
        self.early_stop_counter = 0

        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [], 'val_auc': [], 'lr': []
        }

        self.checkpoint_dir = Path(output_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nTrainer initialized on {device}")
        print(f"Ensemble parameters: {sum(p.numel() for p in model.parameters()):,}")

    def train_epoch(self) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            if self.scaler is not None:
                with autocast('cuda'):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
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

        for images, labels in tqdm(self.val_loader, desc="Validating"):
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)
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
            'model_type': 'ensemble'
        }
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        torch.save(checkpoint, self.checkpoint_dir / filename)

        if is_best:
            best_path = self.checkpoint_dir / 'best_model_ensemble.pth'
            torch.save(checkpoint, best_path)
            print(f"Saved best ensemble model with AUC: {self.best_val_auc:.4f}")

    def train(self, epochs: int):
        print(f"\nStarting ensemble training for {epochs} epochs")
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
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val AUC: {val_auc:.4f}")
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

        history_path = self.checkpoint_dir / 'training_history_ensemble.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description='Train XceptionNet + EfficientNet-B4 ensemble on mixed datasets'
    )
    parser.add_argument('--config', type=str, default='config/config.yaml')
    parser.add_argument('--ff-dir', type=str, default=None,
                        help='Path to FF++ processed dataset (default: from config.yaml)')
    parser.add_argument('--celebdf-dir', type=str, default=None,
                        help='Path to Celeb-DF-v2 processed dataset (default: from config.yaml)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save model checkpoints (default: outputs/models/ensemble_v3)')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size (default 16 - ensemble uses ~2x memory of single model)')
    parser.add_argument('--fusion', type=str, default=None,
                        choices=['mean', 'max', 'weighted'],
                        help='Fusion strategy (default: from config.yaml)')
    parser.add_argument('--num-workers', type=int, default=None,
                        help='Override DataLoader num_workers (default: training.num_workers in config). '
                             'On Windows set this low (0-2) if you hit paging file / shared memory errors.')

    # v3 shortcut-mitigation flags (mirror scripts/train_xception_robust.py)
    parser.add_argument('--include-dfd', action='store_true',
                        help='Include DFD (DeepFakeDetection) standalone release in training mix.')
    parser.add_argument('--include-wilddeepfake', action='store_true',
                        help='Include WildDeepfake (train split). Expects flat layout at '
                             'data/wilddeepfake/train/{real,fake}.')
    parser.add_argument('--dfd-dir', type=str, default=None,
                        help='Override preprocessed DFD directory (default: <processed_root>/DFD).')
    parser.add_argument('--wilddeepfake-dir', type=str, default=None,
                        help='Override WildDeepfake train directory (default: data/wilddeepfake/train).')
    parser.add_argument('--wilddeepfake-frames', type=int, default=20,
                        help='Frames per video for WildDeepfake (default 20).')
    parser.add_argument('--keep-dfd-in-ffpp', action='store_true',
                        help='Keep DeepFakeDetection_* videos inside FF++ even when --include-dfd is set. '
                             'Default: drop them to avoid the double-counting shortcut.')
    parser.add_argument('--keep-real-youtube', action='store_true',
                        help="Keep Celeb-DF's real_youtube/ class in training. "
                             "Default: drop it to remove the 'YouTube codec -> real' shortcut.")
    parser.add_argument('--no-weighted-sampler', action='store_true',
                        help='Disable WeightedRandomSampler and fall back to balance_classes duplication '
                             '(matches pre-v3 ensemble behaviour). Default: weighted sampler on, matching robust v3.')

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
                'checkpoint_dir': args.output_dir or 'outputs/models/ensemble'
            },
            'hardware': {'mixed_precision': True, 'device': 'cuda'},
            'model': {
                'xception': {'num_classes': 2, 'dropout': 0.5, 'pretrained': True},
                'efficientnet': {'num_classes': 2, 'dropout': 0.5, 'pretrained': True},
                'ensemble': {'fusion': 'mean', 'xception_weight': 0.5}
            }
        }

    # Resolve paths from config if not provided
    if args.ff_dir is None:
        args.ff_dir = config.get('data', {}).get('processed', {}).get('faceforensics', 'data/processed/FaceForensics++_AllTypes')
    if args.celebdf_dir is None:
        args.celebdf_dir = config.get('data', {}).get('processed', {}).get('celebdf', 'data/processed/Celeb-DF-v2')
    if args.output_dir is None:
        models_dir = config.get('output', {}).get('models_dir', 'outputs/models')
        args.output_dir = str(Path(models_dir) / 'ensemble_v3')
    if args.num_workers is not None:
        config.setdefault('training', {})['num_workers'] = args.num_workers

    # Resolve fusion from config if not overridden
    ensemble_cfg = config.get('model', {}).get('ensemble', {})
    if args.fusion is None:
        args.fusion = ensemble_cfg.get('fusion', 'mean')

    # Override config with CLI args
    config['training']['batch_size'] = args.batch_size
    config['training']['epochs'] = args.epochs
    config['training']['checkpoint_dir'] = args.output_dir

    # Setup device
    cfg_device = config.get('hardware', {}).get('device', 'cuda')
    if cfg_device == 'cuda' and not torch.cuda.is_available():
        print("[!] Warning: CUDA not available, falling back to CPU")
        cfg_device = 'cpu'
    device = torch.device(cfg_device)
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Create ensemble model
    print("\nCreating ensemble model (XceptionNet + EfficientNet-B4)...")
    model_cfg = config.get('model', {})
    xception_cfg = model_cfg.get('xception', {})
    efficientnet_cfg = model_cfg.get('efficientnet', {})

    model = create_ensemble(
        num_classes=xception_cfg.get('num_classes', 2),
        dropout=xception_cfg.get('dropout', 0.5),
        pretrained=xception_cfg.get('pretrained', True),
        fusion=args.fusion,
        xception_weight=ensemble_cfg.get('xception_weight', 0.5)
    )

    print(f"Fusion strategy: {args.fusion}")

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
            'flat_layout': False,
            'exclude_prefixes': ffpp_exclude_prefixes,
        },
        {
            'name': 'Celeb-DF-v2',
            'path': Path(args.celebdf_dir),
            'flat_layout': False,
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
        dataset_entries.append({'name': 'DFD', 'path': dfd_dir, 'flat_layout': False})
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
            'frames_per_video': args.wilddeepfake_frames,
        })
        print(f"[+] Including WildDeepfake (flat layout, "
              f"{args.wilddeepfake_frames} frames/video) from: {wd_dir}")

    pp = config.get('preprocessing', {})
    img_cfg = pp.get('image', {})
    train_cfg = config.get('training', {})
    mixed_loader = MixedDatasetLoader(
        dataset_dirs=dataset_entries,
        batch_size=train_cfg['batch_size'],
        num_workers=train_cfg.get('num_workers', 4),
        train_split=train_cfg.get('train_split', 0.8),
        val_split=train_cfg.get('val_split', 0.1),
        image_size=img_cfg.get('size', 299),
        frames_per_video=pp.get('training_frames_per_video', 5),
        normalize_mean=img_cfg.get('normalize_mean', [0.5, 0.5, 0.5]),
        normalize_std=img_cfg.get('normalize_std', [0.5, 0.5, 0.5]),
        augmentation='standard',
        use_weighted_sampler=not args.no_weighted_sampler,
    )

    train_loader, val_loader, test_loader = mixed_loader.create_dataloaders()

    # Create trainer
    trainer = Trainer(
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
        'model_type': 'ensemble',
        'fusion': args.fusion,
        'timestamp': datetime.now().isoformat()
    }

    results_path = Path(args.output_dir) / 'test_results_ensemble.json'
    with open(results_path, 'w') as f:
        json.dump(test_results, f, indent=2)

    print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    main()
