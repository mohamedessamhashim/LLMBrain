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

    Args:
        vision_dim: Dimension of vision features.
        text_dim: Dimension of text embeddings.
        num_heads: Number of attention heads.
        dropout: Dropout rate.
        bias: Whether to use bias in projections.
    """

    def __init__(
        self,
        vision_dim: int,
        text_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        bias: bool = True,
    ):
        super().__init__()

        self.vision_dim = vision_dim
        self.text_dim = text_dim
        self.num_heads = num_heads
        self.head_dim = vision_dim // num_heads

        assert vision_dim % num_heads == 0, "vision_dim must be divisible by num_heads"

        # Query projection (vision -> attention space)
        self.q_proj = nn.Linear(vision_dim, vision_dim, bias=bias)

        # Key and Value projections (text -> attention space)
        self.k_proj = nn.Linear(text_dim, vision_dim, bias=bias)
        self.v_proj = nn.Linear(text_dim, vision_dim, bias=bias)

        # Output projection
        self.out_proj = nn.Linear(vision_dim, vision_dim, bias=bias)

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

        # Project queries, keys, values
        Q = self.q_proj(vision_features)  # (B, num_patches, vision_dim)
        K = self.k_proj(text_embeddings)  # (B, seq_len, vision_dim)
        V = self.v_proj(text_embeddings)  # (B, seq_len, vision_dim)

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
        attn_output = attn_output.view(batch_size, num_patches, self.vision_dim)

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
        num_heads: Number of attention heads.
        mlp_ratio: Ratio for MLP hidden dimension.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        vision_dim: int,
        text_dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(vision_dim)
        self.cross_attn = CrossAttention(
            vision_dim=vision_dim,
            text_dim=text_dim,
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

    Args:
        in_channels: Number of input channels.
        text_dim: Dimension of text embeddings.
        num_heads: Number of attention heads.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        in_channels: int,
        text_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.cross_attn_block = CrossAttentionBlock(
            vision_dim=in_channels,
            text_dim=text_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

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

        # Reshape: (B, C, D, H, W) -> (B, D*H*W, C)
        x_flat = x.permute(0, 2, 3, 4, 1).reshape(B, D * H * W, C)

        # Apply cross-attention
        x_attended = self.cross_attn_block(x_flat, text_embeddings, text_mask)

        # Reshape back: (B, D*H*W, C) -> (B, C, D, H, W)
        x_out = x_attended.reshape(B, D, H, W, C).permute(0, 4, 1, 2, 3)

        return x_out


class MultiScaleCrossAttention(nn.Module):
    """Multi-scale cross-attention for Swin UNETR decoder.

    Applies cross-attention at multiple scales/resolutions in the decoder.

    Args:
        feature_sizes: List of feature sizes at each scale.
        text_dim: Dimension of text embeddings.
        num_heads: Number of attention heads.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        feature_sizes: Tuple[int, ...] = (768, 384, 192, 96, 48),
        text_dim: int = 3072,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.cross_attn_layers = nn.ModuleList([
            SpatialCrossAttention(
                in_channels=feat_size,
                text_dim=text_dim,
                num_heads=min(num_heads, feat_size // 64),  # Ensure divisibility
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
