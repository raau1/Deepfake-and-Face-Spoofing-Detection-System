"""
EfficientNet-B4 Model for Deepfake Detection
As specified in Section 1.2 and 4.2.5 of the Interim Report.

EfficientNet uses compound scaling (depth, width, resolution) to achieve
higher accuracy than XceptionNet at the cost of slower inference.

Reference:
M. Tan and Q. Le, "EfficientNet: Rethinking Model Scaling for CNNs," ICML 2019
"""

import torch
import torch.nn as nn
import timm


class EfficientNetB4(nn.Module):
    """
    EfficientNet-B4 for binary deepfake detection using timm library.

    Loads the pretrained EfficientNet-B4 architecture from timm and replaces
    the classifier with a 2-class (real/fake) output layer.
    """

    def __init__(
        self,
        num_classes: int = 2,
        dropout: float = 0.5,
        pretrained: bool = True
    ):
        """
        Initialize EfficientNet-B4 using timm.

        Args:
            num_classes: Number of output classes
            dropout: Dropout rate
            pretrained: Whether to use ImageNet pretrained weights
        """
        super().__init__()

        self.num_classes = num_classes

        # Load pretrained EfficientNet-B4 from timm
        self.backbone = timm.create_model(
            'efficientnet_b4',
            pretrained=pretrained,
            num_classes=0,  # Remove classifier
            global_pool='avg'
        )

        # Get feature dimension
        self.feature_dim = self.backbone.num_features

        # Custom classifier
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.feature_dim, num_classes)

        # Initialize classifier
        nn.init.normal_(self.fc.weight, 0, 0.01)
        nn.init.constant_(self.fc.bias, 0)

        print(f"EfficientNetB4 initialized with {self.feature_dim} features")

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features."""
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [B, 3, 299, 299]

        Returns:
            logits: Output logits [B, num_classes]
        """
        x = self.forward_features(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Get feature embedding."""
        return self.forward_features(x)


def create_efficientnet(
    num_classes: int = 2,
    dropout: float = 0.5,
    pretrained: bool = True
) -> nn.Module:
    """
    Factory function to create EfficientNet-B4 model.

    Args:
        num_classes: Number of output classes
        dropout: Dropout rate
        pretrained: Whether to use pretrained weights

    Returns:
        model: EfficientNet-B4 model
    """
    return EfficientNetB4(num_classes, dropout, pretrained)
