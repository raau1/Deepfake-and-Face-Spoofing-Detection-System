"""
Temporal Analysis Model for Deepfake Detection
Combines a frozen XceptionNet backbone with an LSTM to detect
temporal inconsistencies across video frame sequences.

As specified in Interim Report Section 1.2 and 2.3:
"Temporal analysis methods examine frame-to-frame consistency,
using LSTM networks to detect unnatural transitions and
inconsistent facial movements across video sequences."

Architecture:
    Input: [B, T, 3, 299, 299], batch of T-frame video sequences
    XceptionNet backbone (frozen): extracts 2048-dim embedding per frame
    LSTM: processes [B, T, 2048] sequence, captures temporal patterns
    Classifier: final hidden state → [B, 2] (real/fake)
"""

import torch
import torch.nn as nn
from .xception import XceptionNetTimm


class TemporalModel(nn.Module):
    """
    Temporal deepfake detection using XceptionNet + LSTM.

    The XceptionNet backbone is frozen, only the LSTM and classifier
    are trained. This is both memory-efficient and leverages the
    spatial features already learned by the pre-trained XceptionNet.
    """

    def __init__(
        self,
        num_classes: int = 2,
        backbone_dropout: float = 0.5,
        pretrained_backbone: bool = True,
        lstm_hidden: int = 512,
        lstm_layers: int = 2,
        lstm_dropout: float = 0.3,
        classifier_dropout: float = 0.5,
        bidirectional: bool = False,
        freeze_backbone: bool = True
    ):
        """
        Initialize temporal model.

        Args:
            num_classes: Number of output classes
            backbone_dropout: Dropout for XceptionNet backbone
            pretrained_backbone: Whether backbone uses ImageNet weights
            lstm_hidden: LSTM hidden state dimension
            lstm_layers: Number of LSTM layers
            lstm_dropout: Dropout between LSTM layers (only if lstm_layers > 1)
            classifier_dropout: Dropout before final classifier
            bidirectional: Whether LSTM processes sequence in both directions
            freeze_backbone: Whether to freeze backbone weights (should be True)
        """
        super().__init__()

        self.num_classes = num_classes
        self.lstm_hidden = lstm_hidden
        self.bidirectional = bidirectional
        self.freeze_backbone = freeze_backbone

        # Spatial backbone - XceptionNet
        self.backbone = XceptionNetTimm(
            num_classes=num_classes,
            dropout=backbone_dropout,
            pretrained=pretrained_backbone
        )
        self.feature_dim = self.backbone.feature_dim  # 2048

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

        # Temporal module - LSTM
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0,
            bidirectional=bidirectional
        )

        # Classifier
        lstm_output_dim = lstm_hidden * 2 if bidirectional else lstm_hidden
        self.classifier_dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(lstm_output_dim, num_classes)

        # Initialize classifier
        nn.init.normal_(self.classifier.weight, 0, 0.01)
        nn.init.constant_(self.classifier.bias, 0)

        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"TemporalModel initialized:")
        print(f"  Backbone features: {self.feature_dim}")
        print(f"  LSTM: {lstm_layers} layers, {lstm_hidden} hidden, "
              f"bidirectional={bidirectional}")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Frozen backbone parameters: {total_params - trainable_params:,}")

    def load_backbone_weights(self, checkpoint_path: str, device: torch.device):
        """
        Load pre-trained backbone weights from a checkpoint.

        Args:
            checkpoint_path: Path to a trained XceptionNet or ensemble checkpoint
            device: Device to load weights on
        """
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = checkpoint.get('model_state_dict', checkpoint)

        # Handle different checkpoint formats
        backbone_state = {}
        for key, value in state_dict.items():
            # From standalone XceptionNet checkpoint
            if key.startswith('backbone.') or key.startswith('dropout.') or key.startswith('fc.'):
                backbone_state[key] = value
            # From ensemble checkpoint (xception sub-model)
            elif key.startswith('xception.'):
                new_key = key[len('xception.'):]
                backbone_state[new_key] = value

        if backbone_state:
            self.backbone.load_state_dict(backbone_state, strict=False)
            print(f"Loaded backbone weights from: {checkpoint_path}")
            print(f"  Loaded {len(backbone_state)} parameter tensors")
        else:
            print(f"WARNING: No matching backbone weights found in {checkpoint_path}")

    def extract_embeddings(self, x: torch.Tensor, chunk_size: int = 16) -> torch.Tensor:
        """
        Extract frame embeddings using frozen backbone.

        Processes frames in chunks to manage GPU memory.

        Args:
            x: Input frames [B, T, C, H, W]
            chunk_size: Number of frames to process at once

        Returns:
            embeddings: [B, T, feature_dim]
        """
        B, T, C, H, W = x.shape
        embeddings = []

        for i in range(0, T, chunk_size):
            chunk = x[:, i:i + chunk_size]  # [B, chunk, C, H, W]
            chunk_T = chunk.size(1)
            chunk_flat = chunk.reshape(B * chunk_T, C, H, W)

            with torch.no_grad():
                emb = self.backbone.get_embedding(chunk_flat)  # [B*chunk, 2048]

            emb = emb.view(B, chunk_T, -1)  # [B, chunk, 2048]
            embeddings.append(emb)

        return torch.cat(embeddings, dim=1)  # [B, T, 2048]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: frames → backbone embeddings → LSTM → classifier.

        Args:
            x: Input tensor [B, T, C, H, W] - batch of frame sequences

        Returns:
            logits: Output logits [B, num_classes]
        """
        # Ensure backbone is in eval mode even during training
        if self.freeze_backbone:
            self.backbone.eval()

        # Extract spatial embeddings
        embeddings = self.extract_embeddings(x)  # [B, T, 2048]

        # Temporal analysis
        lstm_out, (h_n, c_n) = self.lstm(embeddings)
        # h_n: [num_layers * num_directions, B, hidden]

        # Use final hidden state
        if self.bidirectional:
            # Concatenate forward and backward final hidden states
            features = torch.cat([h_n[-2], h_n[-1]], dim=1)  # [B, hidden*2]
        else:
            features = h_n[-1]  # [B, hidden]

        # Classify
        features = self.classifier_dropout(features)
        logits = self.classifier(features)  # [B, 2]
        return logits

    def forward_from_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass from pre-extracted embeddings (skips backbone).

        Args:
            embeddings: Pre-extracted embeddings [B, T, feature_dim]

        Returns:
            logits: Output logits [B, num_classes]
        """
        lstm_out, (h_n, c_n) = self.lstm(embeddings)

        if self.bidirectional:
            features = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            features = h_n[-1]

        features = self.classifier_dropout(features)
        logits = self.classifier(features)
        return logits

    def get_temporal_attention(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get per-frame LSTM output magnitudes (for analysis/visualisation).
        Higher values indicate frames the LSTM found more informative.

        Args:
            x: Input tensor [B, T, C, H, W]

        Returns:
            attention: Per-frame importance scores [B, T]
        """
        if self.freeze_backbone:
            self.backbone.eval()

        embeddings = self.extract_embeddings(x)
        lstm_out, _ = self.lstm(embeddings)  # [B, T, hidden]

        # L2 norm of each frame's LSTM output as importance score
        attention = torch.norm(lstm_out, dim=2)  # [B, T]
        # Normalise to [0, 1]
        attention = attention / (attention.max(dim=1, keepdim=True)[0] + 1e-8)
        return attention


def create_temporal(
    num_classes: int = 2,
    backbone_dropout: float = 0.5,
    pretrained_backbone: bool = True,
    lstm_hidden: int = 512,
    lstm_layers: int = 2,
    lstm_dropout: float = 0.3,
    classifier_dropout: float = 0.5,
    bidirectional: bool = False,
    freeze_backbone: bool = True
) -> nn.Module:
    """
    Factory function to create temporal model.

    Args:
        num_classes: Number of output classes
        backbone_dropout: Dropout for XceptionNet backbone
        pretrained_backbone: Whether backbone uses ImageNet weights
        lstm_hidden: LSTM hidden dimension
        lstm_layers: Number of LSTM layers
        lstm_dropout: Dropout between LSTM layers
        classifier_dropout: Dropout before classifier
        bidirectional: Whether to use bidirectional LSTM
        freeze_backbone: Whether to freeze backbone weights

    Returns:
        model: TemporalModel instance
    """
    return TemporalModel(
        num_classes=num_classes,
        backbone_dropout=backbone_dropout,
        pretrained_backbone=pretrained_backbone,
        lstm_hidden=lstm_hidden,
        lstm_layers=lstm_layers,
        lstm_dropout=lstm_dropout,
        classifier_dropout=classifier_dropout,
        bidirectional=bidirectional,
        freeze_backbone=freeze_backbone
    )
