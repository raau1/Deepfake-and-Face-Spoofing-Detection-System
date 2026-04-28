"""
XceptionNet Model for Deepfake Detection
As specified in Section 1.2, 2.1, and 4.2.5 of the Interim Report.

XceptionNet uses depthwise separable convolutions to capture manipulation artifacts,
established as a baseline in FaceForensics++ [2].

Optional CBAM attention (channel + spatial) can be inserted between the backbone's
final feature map (act4) and the global average pool, to force multi-region
reasoning rather than the central-region collapse diagnosed on the v3 line in
Dissertation §6.10.2.

References:
[3] F. Chollet, "Xception: Deep Learning with Depthwise Separable Convolutions," CVPR 2017
    Woo et al., "CBAM: Convolutional Block Attention Module," ECCV 2018
"""

import torch
import torch.nn as nn
import timm

from src.models.cbam import CBAM


class XceptionNetTimm(nn.Module):
    """
    XceptionNet for binary deepfake detection using timm library.

    Loads the pretrained Xception architecture from timm and replaces
    the classifier with a 2-class (real/fake) output layer.

    When use_cbam=True, CBAM is inserted between the 2048-channel act4 feature
    map and the global average pool. Grad-CAM's preferred target layer in this
    mode is `model.cbam` (CBAM's output shows the effect of the attention
    gating); targeting `model.backbone.act4` still works and shows the
    pre-attention feature importance.
    """

    def __init__(
        self,
        num_classes: int = 2,
        dropout: float = 0.5,
        pretrained: bool = True,
        use_cbam: bool = False,
        cbam_reduction: int = 16,
    ):
        """
        Initialize XceptionNet using timm.

        Args:
            num_classes: Number of output classes
            dropout: Dropout rate
            pretrained: Whether to use ImageNet pretrained weights
            use_cbam: Insert a CBAM block between act4 and the global pool
            cbam_reduction: Channel-attention reduction ratio (CBAM default = 16)
        """
        super().__init__()

        self.num_classes = num_classes
        self.use_cbam = use_cbam

        # When CBAM is off we keep the original shape: backbone pools internally
        # and returns [B, 2048]. When CBAM is on we need the unpooled [B,2048,H,W]
        # feature map, so we strip timm's built-in pool and do it ourselves.
        global_pool = '' if use_cbam else 'avg'
        self.backbone = timm.create_model(
            'xception',
            pretrained=pretrained,
            num_classes=0,
            global_pool=global_pool,
        )

        self.feature_dim = self.backbone.num_features

        if use_cbam:
            self.cbam = CBAM(self.feature_dim, reduction=cbam_reduction)
            self.pool = nn.AdaptiveAvgPool2d(1)
        else:
            self.cbam = None
            self.pool = None

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.feature_dim, num_classes)

        nn.init.normal_(self.fc.weight, 0, 0.01)
        nn.init.constant_(self.fc.bias, 0)

        print(
            f"XceptionNetTimm initialized with {self.feature_dim} features"
            f"{' + CBAM' if use_cbam else ''}"
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features (post-pool, shape [B, feature_dim])."""
        x = self.backbone(x)
        if self.use_cbam:
            # Shape: [B, 2048, H, W]
            x = self.cbam(x)
            x = self.pool(x).flatten(1)
        return x

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


def create_xception(
    num_classes: int = 2,
    dropout: float = 0.5,
    pretrained: bool = True,
    use_cbam: bool = False,
    cbam_reduction: int = 16,
) -> nn.Module:
    """
    Factory function to create XceptionNet model.

    Args:
        num_classes: Number of output classes
        dropout: Dropout rate
        pretrained: Whether to use pretrained weights
        use_cbam: Whether to insert CBAM attention after act4
        cbam_reduction: Channel-attention reduction ratio for CBAM

    Returns:
        model: XceptionNet model
    """
    return XceptionNetTimm(
        num_classes=num_classes,
        dropout=dropout,
        pretrained=pretrained,
        use_cbam=use_cbam,
        cbam_reduction=cbam_reduction,
    )


# Test function
def test_xception():
    """Test XceptionNet model."""
    print("Testing XceptionNet...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create model
    model = create_xception(num_classes=2, pretrained=True)
    model = model.to(device)
    model.eval()

    # Test forward pass
    dummy_input = torch.randn(2, 3, 299, 299).to(device)

    with torch.no_grad():
        output = model(dummy_input)
        embedding = model.get_embedding(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Embedding shape: {embedding.shape}")

    # Test probabilities
    probs = torch.softmax(output, dim=1)
    print(f"Probabilities: {probs}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    print("XceptionNet test complete!")


if __name__ == '__main__':
    test_xception()
