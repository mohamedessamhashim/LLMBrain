"""Cross-attention modules for LLM-conditioned segmentation."""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    """Cross-attention layer for conditioning vision features on text embeddings.

    Implements multi-head cross-attention where:
    - Query: Vision features (from Swin UNETR encoder/decoder)
    - Key/Value: Text embeddings (from LLaMA)

    All projections map into a fixed-size common attention space so that
    text information is not excessively compressed at high-resolution
    decoder stages.

    Args:
        vision_dim: Dimension of vision features.
        text_dim: Dimension of text embeddings.
        common_dim: Dimension of the shared Q/K/V attention space.
        num_heads: Number of attention heads.
        dropout: Dropout rate.
        bias: Whether to use bias in projections.
    """

    def __init__(
        self,
        vision_dim: int,
        text_dim: int,
        common_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        bias: bool = True,
    ):
        super().__init__()

        self.vision_dim = vision_dim
        self.text_dim = text_dim
        self.common_dim = common_dim
        self.num_heads = num_heads
        self.head_dim = common_dim // num_heads

        assert common_dim % num_heads == 0, "common_dim must be divisible by num_heads"

        # Query projection (vision -> common attention space)
        self.q_proj = nn.Linear(vision_dim, common_dim, bias=bias)

        # Key and Value projections (text -> common attention space)
        self.k_proj = nn.Linear(text_dim, common_dim, bias=bias)
        self.v_proj = nn.Linear(text_dim, common_dim, bias=bias)

        # Output projection (common attention space -> vision)
        self.out_proj = nn.Linear(common_dim, vision_dim, bias=bias)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(
        self,
        vision_features: torch.Tensor,
        text_embeddings: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply cross-attention.

        Args:
            vision_features: (batch, num_patches, vision_dim) or (batch, D, H, W, vision_dim)
            text_embeddings: (batch, seq_len, text_dim)
            text_mask: (batch, seq_len) attention mask for text

        Returns:
            Attended vision features with same shape as input.
        """
        # Handle 3D spatial input
        spatial_shape = None
        if vision_features.dim() == 5:
            B, D, H, W, C = vision_features.shape
            spatial_shape = (D, H, W)
            vision_features = vision_features.view(B, D * H * W, C)

        batch_size, num_patches, _ = vision_features.shape
        seq_len = text_embeddings.size(1)

        # Project queries, keys, values into common attention space
        Q = self.q_proj(vision_features)  # (B, num_patches, common_dim)
        K = self.k_proj(text_embeddings)  # (B, seq_len, common_dim)
        V = self.v_proj(text_embeddings)  # (B, seq_len, common_dim)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, num_patches, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # Now: (B, num_heads, num_patches/seq_len, head_dim)

        # Compute attention scores
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        # (B, num_heads, num_patches, seq_len)

        # Apply text mask if provided
        if text_mask is not None:
            # Expand mask for multi-head attention
            text_mask = text_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, seq_len)
            attn_weights = attn_weights.masked_fill(text_mask == 0, float("-inf"))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)
        # (B, num_heads, num_patches, head_dim)

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, num_patches, self.common_dim)

        # Output projection
        output = self.out_proj(attn_output)

        # Restore spatial shape if needed
        if spatial_shape is not None:
            output = output.view(batch_size, *spatial_shape, self.vision_dim)

        return output


class CrossAttentionBlock(nn.Module):
    """Cross-attention block with residual connection and normalization.

    Combines cross-attention with layer normalization and feed-forward network.

    Args:
        vision_dim: Dimension of vision features.
        text_dim: Dimension of text embeddings.
        common_dim: Dimension of the shared Q/K/V attention space.
        num_heads: Number of attention heads.
        mlp_ratio: Ratio for MLP hidden dimension.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        vision_dim: int,
        text_dim: int,
        common_dim: int = 512,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(vision_dim)
        self.cross_attn = CrossAttention(
            vision_dim=vision_dim,
            text_dim=text_dim,
            common_dim=common_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.norm2 = nn.LayerNorm(vision_dim)
        mlp_hidden = int(vision_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(vision_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, vision_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        vision_features: torch.Tensor,
        text_embeddings: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with residual connections.

        Args:
            vision_features: Vision features from encoder/decoder.
            text_embeddings: Text embeddings from LLM.
            text_mask: Attention mask for text.

        Returns:
            Conditioned vision features.
        """
        # Cross-attention with residual
        normed = self.norm1(vision_features)
        attn_out = self.cross_attn(normed, text_embeddings, text_mask)
        vision_features = vision_features + attn_out

        # MLP with residual
        normed = self.norm2(vision_features)
        mlp_out = self.mlp(normed)
        vision_features = vision_features + mlp_out

        return vision_features


class SpatialCrossAttention(nn.Module):
    """Cross-attention for 3D spatial features.

    Designed for integration with Swin UNETR decoder stages.
    Handles the spatial dimensions of 3D medical images.

    When the number of spatial tokens (D*H*W) exceeds ``max_tokens``,
    the features are spatially downsampled with average pooling before
    cross-attention and then upsampled back.  This keeps memory usage
    bounded for high-resolution decoder stages.

    Args:
        in_channels: Number of input channels.
        text_dim: Dimension of text embeddings.
        common_dim: Dimension of the shared Q/K/V attention space.
        num_heads: Number of attention heads.
        dropout: Dropout rate.
        max_tokens: Token count threshold above which spatial downsampling
            is applied before cross-attention.
    """

    def __init__(
        self,
        in_channels: int,
        text_dim: int,
        common_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_tokens: int = 8192,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.max_tokens = max_tokens
        self.cross_attn_block = CrossAttentionBlock(
            vision_dim=in_channels,
            text_dim=text_dim,
            common_dim=common_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

    @staticmethod
    def _pool_factor(num_tokens: int, max_tokens: int) -> int:
        """Compute the smallest power-of-2 pooling factor that brings
        ``num_tokens`` under ``max_tokens``."""
        factor = 1
        while num_tokens // (factor ** 3) > max_tokens:
            factor *= 2
        return factor

    def forward(
        self,
        x: torch.Tensor,
        text_embeddings: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply spatial cross-attention.

        Args:
            x: (batch, channels, D, H, W) vision features
            text_embeddings: (batch, seq_len, text_dim) text embeddings
            text_mask: (batch, seq_len) attention mask

        Returns:
            Conditioned features with same shape as input.
        """
        B, C, D, H, W = x.shape
        num_tokens = D * H * W

        # Downsample if spatial volume is too large for attention
        pool_k = self._pool_factor(num_tokens, self.max_tokens) if num_tokens > self.max_tokens else 1

        if pool_k > 1:
            x_down = F.avg_pool3d(x, kernel_size=pool_k, stride=pool_k)
            Bd, Cd, Dd, Hd, Wd = x_down.shape
        else:
            x_down = x
            Dd, Hd, Wd = D, H, W

        # Reshape: (B, C, D', H', W') -> (B, D'*H'*W', C)
        x_flat = x_down.permute(0, 2, 3, 4, 1).reshape(B, Dd * Hd * Wd, C)

        # Apply cross-attention
        x_attended = self.cross_attn_block(x_flat, text_embeddings, text_mask)

        # Reshape back: (B, D'*H'*W', C) -> (B, C, D', H', W')
        x_out = x_attended.reshape(B, Dd, Hd, Wd, C).permute(0, 4, 1, 2, 3)

        # Upsample back to original resolution if downsampled
        if pool_k > 1:
            x_out = F.interpolate(x_out, size=(D, H, W), mode="trilinear", align_corners=False)

        return x_out


class MultiScaleCrossAttention(nn.Module):
    """Multi-scale cross-attention for Swin UNETR decoder.

    Applies cross-attention at multiple scales/resolutions in the decoder.

    Args:
        feature_sizes: List of feature sizes at each scale.
        text_dim: Dimension of text embeddings.
        common_dim: Dimension of the shared Q/K/V attention space.
        num_heads: Number of attention heads.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        feature_sizes: Tuple[int, ...] = (768, 384, 192, 96, 48),
        text_dim: int = 3072,
        common_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.cross_attn_layers = nn.ModuleList([
            SpatialCrossAttention(
                in_channels=feat_size,
                text_dim=text_dim,
                common_dim=common_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
            for feat_size in feature_sizes
        ])

    def forward(
        self,
        features: Tuple[torch.Tensor, ...],
        text_embeddings: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, ...]:
        """Apply cross-attention at each scale.

        Args:
            features: Tuple of features at different scales.
            text_embeddings: Text embeddings from LLM.
            text_mask: Attention mask for text.

        Returns:
            Tuple of conditioned features.
        """
        conditioned = []
        for feat, cross_attn in zip(features, self.cross_attn_layers):
            conditioned.append(cross_attn(feat, text_embeddings, text_mask))
        return tuple(conditioned)
