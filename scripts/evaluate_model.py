"""
Comprehensive Model Evaluation Script for Deepfake Detection

Evaluates trained models on test datasets with detailed metrics:
- Frame-level metrics: Accuracy, AUC, Precision, Recall, F1
- Video-level metrics: Mean aggregation accuracy
- Advanced metrics: EER, Precision@95%Recall
- Visualizations: Confusion matrix, ROC curve

Usage:
    python scripts/evaluate_model.py --checkpoint outputs/models/best_model.pth --dataset ff++
    python scripts/evaluate_model.py --checkpoint outputs/models/best_model.pth --dataset celebdf
"""

import argparse
import json
import sys
import yaml
from pathlib import Path
from typing import Dict, Tuple, List
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

# Import from src
from src.models.xception import XceptionNetTimm
from src.models.ensemble import EnsembleModel
from src.models.temporal import TemporalModel
from src.training.dataset import DeepfakeDataset, create_dataloaders
from src.training.sequence_dataset import SequenceDataset


class ModelEvaluator:
    """Comprehensive model evaluation with frame-level and video-level metrics"""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        output_dir: Path,
        aggregation: str = 'mean',
        targets: Dict = None
    ):
        self.model = model
        self.device = device
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # aggregation: how frame-level results are combined into video-level prediction
        # 'mean' , average fake probability across frames, threshold at 0.5
        # 'majority' ,  majority vote on binary per-frame predictions
        # 'max'  ,  fake if any frame's fake probability exceeds 0.5
        # Note: with frames_per_video=1 (current default), aggregation has no effect
        # since there is only one frame per video, frame-level and video-level metrics are identical.
        self.aggregation = aggregation
        self.targets = targets or {}  # performance targets from config['evaluation']['targets']

        # Results storage
        self.frame_predictions = []
        self.frame_labels = []
        self.frame_probs = []
        self.video_predictions = defaultdict(list)  # video_name -> [binary predictions]
        self.video_probs = defaultdict(list)         # video_name -> [fake probabilities]
        self.video_labels = {}                       # video_name -> label

    def evaluate(self, dataloader: DataLoader, split_name: str = "test") -> Dict:
        """
        Evaluate model on a dataset

        Args:
            dataloader: DataLoader for the test dataset
            split_name: Name of the split (e.g., 'test', 'celebdf')

        Returns:
            Dictionary containing all evaluation metrics
        """
        print(f"\n{'='*60}")
        print(f"Evaluating on {split_name.upper()} set")
        print(f"{'='*60}\n")

        self.model.eval()

        # Reset storage
        self.frame_predictions = []
        self.frame_labels = []
        self.frame_probs = []
        self.video_predictions = defaultdict(list)
        self.video_probs = defaultdict(list)
        self.video_labels = {}

        # Detect if this is a temporal (sequence) dataloader
        is_temporal = hasattr(dataloader.dataset, 'videos') or (
            hasattr(dataloader.dataset, 'datasets') and
            any(hasattr(d, 'videos') for d in dataloader.dataset.datasets)
        )

        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(tqdm(dataloader, desc=f"Evaluating {split_name}")):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = self.model(inputs)
                probs = torch.softmax(outputs, dim=1)[:, 1]  # Probability of fake
                predictions = (probs > 0.5).long()

                # Store results
                self.frame_predictions.extend(predictions.cpu().numpy())
                self.frame_labels.extend(labels.cpu().numpy())
                self.frame_probs.extend(probs.cpu().numpy())

                # Get video names for video-level metrics
                batch_start_idx = batch_idx * dataloader.batch_size
                for i in range(len(labels)):
                    sample_idx = batch_start_idx + i
                    if sample_idx < len(dataloader.dataset):
                        if is_temporal:
                            # SequenceDataset: each sample IS a video
                            # Get video name from the dataset's videos list
                            ds = dataloader.dataset
                            # Handle ConcatDataset
                            if hasattr(ds, 'datasets'):
                                cumulative = 0
                                for sub_ds in ds.datasets:
                                    if sample_idx - cumulative < len(sub_ds):
                                        video_dir = sub_ds.videos[sample_idx - cumulative][0]
                                        break
                                    cumulative += len(sub_ds)
                                else:
                                    video_dir = f"video_{sample_idx}"
                            else:
                                video_dir = ds.videos[sample_idx][0]
                            video_name = Path(video_dir).name
                        else:
                            # DeepfakeDataset: each sample is a frame
                            img_path = str(dataloader.dataset.samples[sample_idx])
                            video_name = Path(img_path).parent.name

                        pred = predictions[i].item()
                        label = labels[i].item()

                        self.video_predictions[video_name].append(pred)
                        self.video_probs[video_name].append(probs[i].item())
                        self.video_labels[video_name] = label

        # Compute all metrics
        results = self._compute_metrics(split_name)

        # Save results
        self._save_results(results, split_name)

        # Create visualizations
        self._create_visualizations(split_name)

        # Print summary
        self._print_summary(results)

        return results

    def _compute_metrics(self, split_name: str) -> Dict:
        """Compute comprehensive evaluation metrics"""

        # Convert to numpy arrays
        y_true = np.array(self.frame_labels)
        y_pred = np.array(self.frame_predictions)
        y_prob = np.array(self.frame_probs)

        # Frame-level metrics
        frame_acc = accuracy_score(y_true, y_pred)
        frame_precision = precision_score(y_true, y_pred, zero_division=0)
        frame_recall = recall_score(y_true, y_pred, zero_division=0)
        frame_f1 = f1_score(y_true, y_pred, zero_division=0)
        frame_auc = roc_auc_score(y_true, y_prob)

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        # Equal Error Rate (EER)
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        fnr = 1 - tpr
        eer_threshold_idx = np.nanargmin(np.absolute((fnr - fpr)))
        eer = (fpr[eer_threshold_idx] + fnr[eer_threshold_idx]) / 2

        # Precision at 95% Recall
        precision_at_95_recall = self._compute_precision_at_recall(y_true, y_prob, target_recall=0.95)

        # Video-level metrics, aggregation method from config['evaluation']['aggregation']
        video_true = []
        video_pred = []
        for video_name, predictions in self.video_predictions.items():
            video_true.append(self.video_labels[video_name])
            probs_for_video = self.video_probs[video_name]
            if self.aggregation == 'mean':
                # Average fake probability across frames, threshold at 0.5
                video_pred.append(1 if np.mean(probs_for_video) > 0.5 else 0)
            elif self.aggregation == 'majority':
                # Binary majority vote: fake if >50% of frames predicted fake
                video_pred.append(1 if np.mean(predictions) > 0.5 else 0)
            elif self.aggregation == 'max':
                # Fake if any single frame exceeds 0.5 fake probability
                video_pred.append(1 if np.max(probs_for_video) > 0.5 else 0)
            else:
                video_pred.append(1 if np.mean(probs_for_video) > 0.5 else 0)

        video_true = np.array(video_true)
        video_pred = np.array(video_pred)

        video_acc = accuracy_score(video_true, video_pred)
        video_precision = precision_score(video_true, video_pred, zero_division=0)
        video_recall = recall_score(video_true, video_pred, zero_division=0)
        video_f1 = f1_score(video_true, video_pred, zero_division=0)

        # Aggregate results
        results = {
            'split': split_name,
            'frame_level': {
                'accuracy': float(frame_acc),
                'precision': float(frame_precision),
                'recall': float(frame_recall),
                'f1_score': float(frame_f1),
                'auc': float(frame_auc),
                'eer': float(eer),
                'precision_at_95_recall': float(precision_at_95_recall),
                'confusion_matrix': {
                    'tn': int(tn),
                    'fp': int(fp),
                    'fn': int(fn),
                    'tp': int(tp)
                },
                'total_samples': len(y_true)
            },
            'video_level': {
                'accuracy': float(video_acc),
                'precision': float(video_precision),
                'recall': float(video_recall),
                'f1_score': float(video_f1),
                'total_videos': len(video_true)
            },
            'roc_curve': {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'thresholds': thresholds.tolist()
            }
        }

        return results

    def _compute_precision_at_recall(self, y_true: np.ndarray, y_prob: np.ndarray, target_recall: float = 0.95) -> float:
        """Compute precision at a specific recall threshold"""
        precision_vals, recall_vals, _ = self._precision_recall_curve(y_true, y_prob)

        # Find the precision at the target recall
        idx = np.where(recall_vals >= target_recall)[0]
        if len(idx) > 0:
            return precision_vals[idx[0]]
        return 0.0

    def _precision_recall_curve(self, y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute precision-recall curve"""
        
        # Sort by probability
        desc_score_indices = np.argsort(y_prob)[::-1]
        y_prob = y_prob[desc_score_indices]
        y_true = y_true[desc_score_indices]

        # Compute precision and recall at each threshold
        distinct_value_indices = np.where(np.diff(y_prob))[0]
        threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

        tps = np.cumsum(y_true)[threshold_idxs]
        fps = 1 + threshold_idxs - tps

        precision = tps / (tps + fps)
        recall = tps / tps[-1]
        thresholds = y_prob[threshold_idxs]

        return precision, recall, thresholds

    def _save_results(self, results: Dict, split_name: str):
        """Save results to JSON file"""
        output_file = self.output_dir / f"evaluation_results_{split_name}.json"

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)

        print(f"\n[+] Results saved to: {output_file}")

    def _create_visualizations(self, split_name: str):
        """Create confusion matrix and ROC curve visualizations"""

        y_true = np.array(self.frame_labels)
        y_pred = np.array(self.frame_predictions)
        y_prob = np.array(self.frame_probs)

        # Create figure with subplots
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 1. Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                    xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
        axes[0].set_title(f'Confusion Matrix - {split_name.upper()}')
        axes[0].set_ylabel('True Label')
        axes[0].set_xlabel('Predicted Label')

        # 2. ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)

        axes[1].plot(fpr, tpr, label=f'ROC (AUC = {auc:.4f})', linewidth=2)
        axes[1].plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].set_title(f'ROC Curve - {split_name.upper()}')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        # Save figure
        output_file = self.output_dir / f"evaluation_plots_{split_name}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"[+] Visualizations saved to: {output_file}")

    def _print_summary(self, results: Dict):
        """Print comprehensive results summary"""


        print(f"\n{'='*60}")
        print(f"EVALUATION SUMMARY - {results['split'].upper()}")
        print(f"{'='*60}\n")

        # Frame-level metrics
        frame = results['frame_level']
        print("[*] FRAME-LEVEL METRICS:")
        print(f"  Total Samples: {frame['total_samples']:,}")
        print(f"  Accuracy:      {frame['accuracy']*100:.2f}%")
        print(f"  AUC-ROC:       {frame['auc']*100:.2f}%")
        print(f"  Precision:     {frame['precision']*100:.2f}%")
        print(f"  Recall:        {frame['recall']*100:.2f}%")
        print(f"  F1-Score:      {frame['f1_score']*100:.2f}%")
        print(f"  EER:           {frame['eer']*100:.2f}%")
        print(f"  P@95%R:        {frame['precision_at_95_recall']*100:.2f}%")

        # Confusion matrix
        cm = frame['confusion_matrix']
        print(f"\n  Confusion Matrix:")
        print(f"    True Negatives:  {cm['tn']:,}")
        print(f"    False Positives: {cm['fp']:,}")
        print(f"    False Negatives: {cm['fn']:,}")
        print(f"    True Positives:  {cm['tp']:,}")

        # Video-level metrics
        video = results['video_level']
        print(f"\n[*] VIDEO-LEVEL METRICS ({self.aggregation.upper()} aggregation):")
        print(f"  Total Videos:  {video['total_videos']:,}")
        print(f"  Accuracy:      {video['accuracy']*100:.2f}%")
        print(f"  Precision:     {video['precision']*100:.2f}%")
        print(f"  Recall:        {video['recall']*100:.2f}%")
        print(f"  F1-Score:      {video['f1_score']*100:.2f}%")

        # Performance assessment
        self._print_performance_assessment(results)

        print(f"\n{'='*60}\n")

    def _print_performance_assessment(self, results: Dict):
        """Print performance assessment against project targets"""

        split = results['split']
        auc = results['frame_level']['auc']
        eer = results['frame_level']['eer']
        p_at_95r = results['frame_level']['precision_at_95_recall']
        video_acc = results['video_level']['accuracy']

        print(f"\n[*] PERFORMANCE ASSESSMENT:")

        if 'ff' in split.lower() or 'faceforensics' in split.lower():
            # FaceForensics++ targets, thresholds from config['evaluation']['targets']
            t_auc = self.targets.get('faceforensics_auc', 0.92)
            t_eer = self.targets.get('eer_ff', 0.08)
            print(f"  Target AUC:           92-95%")
            print(f"  Achieved AUC:         {auc*100:.2f}% {'[PASS]' if auc >= t_auc else '[FAIL]'}")
            print(f"  Target EER:           5-8%")
            print(f"  Achieved EER:         {eer*100:.2f}% {'[PASS]' if eer <= t_eer else '[FAIL]'}")
            print(f"  Target P@95%R:        >90%")
            print(f"  Achieved P@95%R:      {p_at_95r*100:.2f}% {'[PASS]' if p_at_95r >= 0.90 else '[FAIL]'}")
            print(f"  Target Video Acc:     95-98%")
            print(f"  Achieved Video Acc:   {video_acc*100:.2f}% {'[PASS]' if video_acc >= 0.95 else '[FAIL]'}")

        elif 'celeb' in split.lower():
            # Celeb-DF v2 targets (cross-dataset), thresholds from config['evaluation']['targets']
            t_auc = self.targets.get('celebdf_auc', 0.70)
            t_eer = self.targets.get('eer_celebdf', 0.18)
            print(f"  Target AUC:           70-75%")
            print(f"  Achieved AUC:         {auc*100:.2f}% {'[PASS]' if auc >= t_auc else '[FAIL]'}")
            print(f"  Target EER:           12-18%")
            print(f"  Achieved EER:         {eer*100:.2f}% {'[PASS]' if eer <= t_eer else '[FAIL]'}")
            print(f"  Target P@95%R:        >75%")
            print(f"  Achieved P@95%R:      {p_at_95r*100:.2f}% {'[PASS]' if p_at_95r >= 0.75 else '[FAIL]'}")
            print(f"  Target Video Acc:     75-80%")
            print(f"  Achieved Video Acc:   {video_acc*100:.2f}% {'[PASS]' if video_acc >= 0.75 else '[FAIL]'}")


def load_model(checkpoint_path: Path, device: torch.device, config: dict = None):
    """
    Load trained model from checkpoint.
    Supports XceptionNet, Ensemble, and Temporal models.

    Returns:
        (model, model_type) tuple - model_type is 'xception', 'ensemble', or 'temporal'
    """

    print(f"\n[*] Loading model from: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Detect model type from checkpoint metadata or filename
    model_type = None
    if isinstance(checkpoint, dict):
        model_type = checkpoint.get('model_type', None)
    if model_type is None and 'ensemble' in checkpoint_path.stem:
        model_type = 'ensemble'
    if model_type is None and 'temporal' in checkpoint_path.stem:
        model_type = 'temporal'

    # Initialize the correct model
    if model_type == 'temporal':
        temporal_cfg = (config or {}).get('model', {}).get('temporal', {})
        xception_cfg = (config or {}).get('model', {}).get('xception', {})
        model = TemporalModel(
            num_classes=xception_cfg.get('num_classes', 2),
            backbone_dropout=xception_cfg.get('dropout', 0.5),
            pretrained_backbone=False,
            lstm_hidden=temporal_cfg.get('lstm_hidden', 512),
            lstm_layers=temporal_cfg.get('lstm_layers', 2),
            lstm_dropout=temporal_cfg.get('lstm_dropout', 0.3),
            classifier_dropout=temporal_cfg.get('classifier_dropout', 0.5),
            bidirectional=temporal_cfg.get('bidirectional', False),
            freeze_backbone=True
        )
        print(f"[*] Model type: Temporal (XceptionNet + LSTM)")
    elif model_type == 'ensemble':
        ensemble_cfg = (config or {}).get('model', {}).get('ensemble', {})
        model = EnsembleModel(
            num_classes=2, pretrained=False,
            fusion=ensemble_cfg.get('fusion', 'mean'),
            xception_weight=ensemble_cfg.get('xception_weight', 0.5)
        )
        print(f"[*] Model type: Ensemble (XceptionNet + EfficientNet-B4)")
    else:
        model_type = 'xception'
        # Auto-detect CBAM from checkpoint metadata so a CBAM-trained robust
        # checkpoint can be loaded without a manual flag.
        use_cbam = bool(checkpoint.get('use_cbam', False)) if isinstance(checkpoint, dict) else False
        cbam_reduction = int(checkpoint.get('cbam_reduction', 16)) if isinstance(checkpoint, dict) else 16
        model = XceptionNetTimm(
            num_classes=2,
            pretrained=False,
            use_cbam=use_cbam,
            cbam_reduction=cbam_reduction,
        )
        print(f"[*] Model type: XceptionNet{' + CBAM' if use_cbam else ''}")

    # Load weights
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'unknown')
        print(f"[+] Loaded checkpoint from epoch {epoch}")
        best_auc = checkpoint.get('best_val_auc')
        history = checkpoint.get('history', {})
        train_acc = history.get('train_acc', [])
        if train_acc and isinstance(epoch, int) and epoch < len(train_acc):
            print(f"   Training accuracy: {train_acc[epoch]:.2f}%")
        if best_auc is not None:
            print(f"   Best validation AUC: {best_auc:.4f}")
    else:
        model.load_state_dict(checkpoint)
        print(f"[+] Loaded model state dict")

    model = model.to(device)
    model.eval()

    return model, model_type


def create_test_dataloader(
    dataset_name: str,
    data_dir: Path,
    batch_size: int = 32,
    num_workers: int = 4,
    normalize_mean: List[float] = None,
    normalize_std: List[float] = None,
    flat_layout: bool = False
) -> DataLoader:
    """Create dataloader for testing (frame-level models)"""

    print(f"\n[*] Creating test dataloader for: {dataset_name}")

    # Create dataset
    dataset = DeepfakeDataset(
        root_dir=data_dir,
        split='test',
        transform=None,  # Uses default transforms (no augmentation for test)
        frames_per_video=1,
        balance_classes=False,  # Don't balance test set
        normalize_mean=normalize_mean,
        normalize_std=normalize_std,
        flat_layout=flat_layout,
    )

    print(f"[+] Test dataset created: {len(dataset)} samples")

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader


def create_test_sequence_dataloader(
    dataset_name: str,
    data_dir: Path,
    sequence_length: int = 32,
    batch_size: int = 4,
    num_workers: int = 2,
    normalize_mean: List[float] = None,
    normalize_std: List[float] = None,
    flat_layout: bool = False,
) -> DataLoader:
    """Create dataloader for testing temporal models (video-level sequences)"""

    print(f"\n[*] Creating test sequence dataloader for: {dataset_name} "
          f"(flat_layout={flat_layout})")

    dataset = SequenceDataset(
        root_dir=data_dir,
        split='test',
        sequence_length=sequence_length,
        normalize_mean=normalize_mean,
        normalize_std=normalize_std,
        augment=False,
        flat_layout=flat_layout,
    )

    print(f"[+] Test dataset created: {len(dataset)} videos")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader


def main():
    parser = argparse.ArgumentParser(description='Evaluate deepfake detection model')

    # Config argument
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to config file')

    # Model arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pth file)')

    # Dataset arguments
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['ff++', 'faceforensics', 'celebdf', 'celeb-df-v2',
                                 'dfd', 'wilddeepfake'],
                        help='Dataset to evaluate on')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Path to processed dataset (default: from config.yaml)')

    # Evaluation arguments
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size for evaluation (default: from config.yaml)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save evaluation results (default: from config.yaml)')

    # Hardware arguments
    parser.add_argument('--device', type=str, default=None,
                        choices=['cuda', 'cpu'],
                        help='Device to use for evaluation (default: from config.yaml)')

    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    config = {}
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)

    # Setup paths
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"[!] Error: Checkpoint not found at {checkpoint_path}")
        sys.exit(1)

    # Determine data directory from config if not provided on command line
    if args.data_dir is None:
        processed = config.get('data', {}).get('processed', {})
        if args.dataset in ['ff++', 'faceforensics']:
            args.data_dir = processed.get('faceforensics', 'data/processed/FaceForensics++')
        elif args.dataset in ['celebdf', 'celeb-df-v2']:
            args.data_dir = processed.get('celebdf', 'data/processed/Celeb-DF-v2')
        elif args.dataset == 'dfd':
            processed_root = processed.get('root', 'data/processed')
            args.data_dir = str(Path(processed_root) / 'DFD')
        elif args.dataset == 'wilddeepfake':
            # Use the pristine test split WildDeepfake ships with -
            # guaranteed not to overlap with the train/ subset used for training.
            data_root = config.get('data', {}).get('root', 'data')
            args.data_dir = str(Path(data_root) / 'wilddeepfake' / 'test')

    # Resolve evaluation settings from config if not provided on CLI
    eval_cfg = config.get('evaluation', {})
    if args.batch_size is None:
        args.batch_size = eval_cfg.get('batch_size', 32)
    if args.device is None:
        args.device = config.get('hardware', {}).get('device', 'cuda')

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"[!] Error: Dataset not found at {data_dir}")
        print(f"   Have you run preprocessing on this dataset?")
        sys.exit(1)

    # Setup device, fall back to CPU if CUDA unavailable
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("[!] Warning: CUDA not available, falling back to CPU")
        args.device = 'cpu'
    device = torch.device(args.device)
    print(f"\n[*] Using device: {device}")

    # Load model
    model, model_type = load_model(checkpoint_path, device, config)

    # Auto-route output directory based on model type
    if args.output_dir is None:
        base_results_dir = config.get('output', {}).get('results_dir', 'outputs/results')
        dataset_tag = args.dataset.replace('celeb-df-v2', 'celebdf').replace('faceforensics', 'ff++')
        if model_type == 'temporal':
            args.output_dir = str(Path(base_results_dir) / f"temporal_eval_{dataset_tag}")
        elif model_type == 'ensemble':
            args.output_dir = str(Path(base_results_dir) / f"ensemble_eval_{dataset_tag}")
        elif 'robust' in checkpoint_path.stem:
            args.output_dir = str(Path(base_results_dir) / f"robust_eval_{dataset_tag}")
        elif 'mixed' in checkpoint_path.stem:
            args.output_dir = str(Path(base_results_dir) / f"mixed_eval_{dataset_tag}")
        else:
            args.output_dir = base_results_dir

    output_dir = Path(args.output_dir)

    # Resolve preprocessing settings for consistent normalization
    pp_cfg = config.get('preprocessing', {})
    img_cfg = pp_cfg.get('image', {})
    normalize_mean = img_cfg.get('normalize_mean', [0.5, 0.5, 0.5])
    normalize_std = img_cfg.get('normalize_std', [0.5, 0.5, 0.5])
    num_workers = config.get('training', {}).get('num_workers', 4)

    # Create test dataloader - temporal models need sequence dataloader
    if model_type == 'temporal':
        temporal_cfg = config.get('model', {}).get('temporal', {})
        seq_len = temporal_cfg.get('sequence_length', 32)
        # Smaller batch size for temporal (each sample = 32 frames)
        temporal_batch = min(args.batch_size, 4)
        test_loader = create_test_sequence_dataloader(
            args.dataset,
            data_dir,
            sequence_length=seq_len,
            batch_size=temporal_batch,
            num_workers=min(num_workers, 2),
            normalize_mean=normalize_mean,
            normalize_std=normalize_std,
            flat_layout=(args.dataset == 'wilddeepfake'),
        )
    else:
        test_loader = create_test_dataloader(
            args.dataset,
            data_dir,
            batch_size=args.batch_size,
            num_workers=num_workers,
            normalize_mean=normalize_mean,
            normalize_std=normalize_std,
            flat_layout=(args.dataset == 'wilddeepfake'),
        )

    # Create evaluator
    evaluator = ModelEvaluator(
        model,
        device,
        output_dir,
        aggregation=eval_cfg.get('aggregation', 'mean'),
        targets=eval_cfg.get('targets', {})
    )

    # Run evaluation
    results = evaluator.evaluate(test_loader, split_name=args.dataset)

    print(f"\n[+] Evaluation complete!")
    print(f"[*] Results saved to: {output_dir}")


if __name__ == '__main__':
    main()