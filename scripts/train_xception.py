"""
XceptionNet Training Script
Train baseline XceptionNet model for deepfake detection.
As specified in Section 1.2 and 5.2 of the Interim Report.

Target performance (Section 1.4):
- FaceForensics++ AUC: 92-95%
- Celeb-DF v2 AUC: 70-75% (cross-dataset)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from pathlib import Path
from typing import Tuple
import yaml
import json
import argparse
from datetime import datetime
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

from src.models.xception import create_xception
from src.training.dataset import create_dataloaders


class Trainer:
    """
    Trainer class for deepfake detection models.
    
    Handles training loop, validation, checkpointing, and logging.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        device: torch.device
    ):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            device: Device to train on
        """
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
        self.best_val_loss = float('inf')
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
        
        print(f"Trainer initialized on {device}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    def train_epoch(self) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Returns:
            avg_loss: Average training loss
            accuracy: Training accuracy
        """
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
        """
        Validate the model.
        
        Returns:
            avg_loss: Average validation loss
            accuracy: Validation accuracy
            auc: Area under ROC curve
        """
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
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"Saved best model with AUC: {self.best_val_auc:.4f}")
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_auc = checkpoint['best_val_auc']
        self.history = checkpoint['history']
        
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def train(self, epochs: int):
        """
        Main training loop.
        
        Args:
            epochs: Number of epochs to train
        """
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
        history_path = self.checkpoint_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Train XceptionNet for deepfake detection')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to config file')
    parser.add_argument('--data-dir', type=str,
                        default=None,
                        help='Path to processed dataset (default: from config.yaml)')
    parser.add_argument('--output-dir', type=str,
                        default=None,
                        help='Directory to save model checkpoints (default: from config.yaml)')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
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
    if args.data_dir is None:
        args.data_dir = config.get('data', {}).get('processed', {}).get('faceforensics', 'data/processed/FaceForensics++')
    if args.output_dir is None:
        args.output_dir = config.get('output', {}).get('models_dir', 'outputs/models')

    # Override config with command line arguments
    config['training']['batch_size'] = args.batch_size
    config['training']['epochs'] = args.epochs
    config['training']['checkpoint_dir'] = args.output_dir
    
    # Setup device, read preference from config, fall back to CPU if CUDA unavailable
    cfg_device = config.get('hardware', {}).get('device', 'cuda')
    if cfg_device == 'cuda' and not torch.cuda.is_available():
        print("[!] Warning: CUDA not available, falling back to CPU")
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
    
    # Create dataloaders
    print(f"\nLoading dataset from: {args.data_dir}")
    
    data_dir = Path(args.data_dir)
    
    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        print("Please run preprocessing first:")
        print("  python src/preprocessing/pipeline.py --dataset ff++")
        return
    
    pp = config.get('preprocessing', {})
    img_cfg = pp.get('image', {})
    train_cfg = config.get('training', {})
    train_loader, val_loader, test_loader = create_dataloaders(
        root_dir=data_dir,
        batch_size=train_cfg['batch_size'],
        num_workers=train_cfg.get('num_workers', 4),
        train_split=train_cfg.get('train_split', 0.8),
        val_split=train_cfg.get('val_split', 0.1),
        image_size=img_cfg.get('size', 299),
        frames_per_video=pp.get('training_frames_per_video', 5),
        normalize_mean=img_cfg.get('normalize_mean', [0.5, 0.5, 0.5]),
        normalize_std=img_cfg.get('normalize_std', [0.5, 0.5, 0.5])
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train
    trainer.train(config['training']['epochs'])
    
    # Final evaluation on test set
    print("\nFinal evaluation on test set...")
    trainer.val_loader = test_loader
    test_loss, test_acc, test_auc = trainer.validate()
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"Test AUC: {test_auc:.4f}")


if __name__ == '__main__':
    main()