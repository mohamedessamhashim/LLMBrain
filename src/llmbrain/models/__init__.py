"""Model architectures for LLM-conditioned brain tumor segmentation."""

from .cross_attention import (
    CrossAttention,
    CrossAttentionBlock,
    MultiScaleCrossAttention,
    SpatialCrossAttention,
)
from .llm_encoder import CachedLLMEncoder, LLaMAEncoder
from .swin_unetr import (
    LLMConditionedSwinUNETR,
    SwinUNETRBaseline,
    get_llm_conditioned_model,
    get_swin_unetr,
)

__all__ = [
    # Vision backbone
    "SwinUNETRBaseline",
    "get_swin_unetr",
    # LLM-conditioned model
    "LLMConditionedSwinUNETR",
    "get_llm_conditioned_model",
    # LLM encoder
    "LLaMAEncoder",
    "CachedLLMEncoder",
    # Cross-attention
    "CrossAttention",
    "CrossAttentionBlock",
    "SpatialCrossAttention",
    "MultiScaleCrossAttention",
]
