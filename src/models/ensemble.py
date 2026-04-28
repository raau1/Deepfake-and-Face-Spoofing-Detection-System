"""
Ensemble Model for Deepfake Detection
Combines XceptionNet and EfficientNet-B4 predictions for improved robustness.

As specified in Section 1.2 and 4.2.5 of the Interim Report:
"Ensemble CNN Approach (Selected): Combines XceptionNet's depth-wise separable
convolutions with EfficientNet's compound scaling, providing a balance between
accuracy and inference speed."

Fusion strategies:
- mean: Average logits from both models (default)
- max: Take the maximum fake probability from either model (most sensitive)
- weighted: Learned weighted combination of logits
"""

import torch
import torch.nn as nn
    
from .xception import XceptionNetTimm
from .efficientnet import EfficientNetB4


class EnsembleModel(nn.Module):
    """
    Ensemble combining XceptionNet and EfficientNet-B4.

    Both models process the same input independently. Their outputs
    are fused into a single prediction using a configurable strategy.
    """

    def __init__(
        self,
        num_classes: int = 2,
        dropout: float = 0.5,
        pretrained: bool = True,
        fusion: str = 'mean',
        xception_weight: float = 0.5
    ):
        """
        Initialize ensemble model.

        Args:
            num_classes: Number of output classes
            dropout: Dropout rate for both sub-models
            pretrained: Whether to use ImageNet pretrained weights
            fusion: Fusion strategy - 'mean', 'max', or 'weighted'
            xception_weight: Weight for XceptionNet in weighted fusion (EfficientNet gets 1 - this)
        """
        super().__init__()

        self.num_classes = num_classes
        self.fusion = fusion

        # Sub-models
        self.xception = XceptionNetTimm(num_classes, dropout, pretrained)
        self.efficientnet = EfficientNetB4(num_classes, dropout, pretrained)

        # For weighted fusion
        if fusion == 'weighted':
            self.xception_weight = nn.Parameter(torch.tensor(xception_weight))
        else:
            self.register_buffer('xception_weight', torch.tensor(xception_weight))

        total_params = sum(p.numel() for p in self.parameters())
        print(f"Ensemble initialized: {total_params:,} total parameters, fusion={fusion}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through both models with fusion.

        Args:
            x: Input tensor [B, 3, 299, 299]

        Returns:
            logits: Fused output logits [B, num_classes]
        """
        logits_x = self.xception(x)
        logits_e = self.efficientnet(x)

        if self.fusion == 'mean':
            return (logits_x + logits_e) / 2.0

        elif self.fusion == 'max':
            # Take whichever model gives higher fake probability
            probs_x = torch.softmax(logits_x, dim=1)
            probs_e = torch.softmax(logits_e, dim=1)
            use_x = probs_x[:, 1] >= probs_e[:, 1]
            result = torch.where(use_x.unsqueeze(1), logits_x, logits_e)
            return result

        elif self.fusion == 'weighted':
            w = torch.sigmoid(self.xception_weight)  # Constrain to [0, 1]
            return w * logits_x + (1 - w) * logits_e

        else:
            return (logits_x + logits_e) / 2.0

    def forward_features(self, x: torch.Tensor):
        """Extract features from both models."""
        feat_x = self.xception.forward_features(x)
        feat_e = self.efficientnet.forward_features(x)
        return feat_x, feat_e

    def get_individual_predictions(self, x: torch.Tensor):
        """
        Get predictions from each model separately (for analysis).

        Returns:
            xception_logits, efficientnet_logits
        """
        with torch.no_grad():
            logits_x = self.xception(x)
            logits_e = self.efficientnet(x)
        return logits_x, logits_e


def create_ensemble(
    num_classes: int = 2,
    dropout: float = 0.5,
    pretrained: bool = True,
    fusion: str = 'mean',
    xception_weight: float = 0.5
) -> nn.Module:
    """
    Factory function to create ensemble model.

    Args:
        num_classes: Number of output classes
        dropout: Dropout rate
        pretrained: Whether to use pretrained weights
        fusion: Fusion strategy - 'mean', 'max', or 'weighted'
        xception_weight: Initial weight for XceptionNet in weighted fusion

    Returns:
        model: Ensemble model
    """
    return EnsembleModel(num_classes, dropout, pretrained, fusion, xception_weight)
