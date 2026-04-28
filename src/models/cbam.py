"""
Convolutional Block Attention Module (CBAM).

CBAM sequentially infers attention maps along two separate dimensions, channel and spatial, then multiplies the input feature map by each mask.
Used here to force the Xception backbone to reason over multiple spatial
regions rather than collapsing onto the central nose-mouth-cheek cluster
that Grad-CAM diagnosed on the v3 line (see Dissertation §6.10.2).

Reference:
    Woo et al., "CBAM: Convolutional Block Attention Module", ECCV 2018.
"""

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """Squeeze-excitation-style channel attention.

    Global-average and global-max pool the [B, C, H, W] feature map to two
    [B, C, 1, 1] descriptors, run both through a shared 2-layer MLP with
    reduction ratio r, sum, then sigmoid to produce the [B, C, 1, 1] mask.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        return torch.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """Spatial attention over the H×W grid.

    Reduces the channel axis with a channel-wise avg+max concat, then a
    single 7×7 conv with sigmoid produces the [B, 1, H, W] mask.
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        return torch.sigmoid(self.conv(concat))


class CBAM(nn.Module):
    """Channel attention followed by spatial attention, residual-style."""

    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction=reduction)
        self.spatial_attention = SpatialAttention(kernel_size=spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x
