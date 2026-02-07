"""Swin UNETR model with LLM cross-attention conditioning for brain tumor segmentation.

Supports three pretrained weight modes:
  - "full":    Loads the complete BraTS-pretrained Swin UNETR (encoder + decoder).
               Best for fine-tuning on UCSF-PDGM since both encoder and decoder start
               from strong segmentation-specific weights.
  - "encoder": Loads only the self-supervised pretrained Swin ViT encoder backbone.
               Useful if your task differs significantly from BraTS segmentation.
  - False:     Random initialization (train from scratch).
  - str path:  Loads weights from a local checkpoint file.
"""

from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR

from .cross_attention import SpatialCrossAttention

# MONAI model zoo URLs
_ENCODER_ONLY_URL = (
    "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/"
    "download/0.8.1/model_swinvit.pt"
)
_FULL_MODEL_URL = (
    "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/"
    "download/0.8.1/swin_unetr.base_5000ep_f48_lr2e-4_pretrained.pt"
)


def _load_pretrained(
    model: SwinUNETR,
    pretrained: Union[bool, str],
) -> None:
    """Load pretrained weights into a SwinUNETR model.

    Args:
        model: The SwinUNETR model instance.
        pretrained: One of:
            - True / "full": Load complete BraTS-pretrained model (encoder + decoder).
            - "encoder": Load self-supervised encoder weights only.
            - False / None: No pretrained weights.
            - A file path string: Load weights from a local .pt file.
    """
    if pretrained is False or pretrained is None:
        return

    if pretrained is True or pretrained == "full":
        try:
            state_dict = torch.hub.load_state_dict_from_url(
                _FULL_MODEL_URL, progress=True, map_location="cpu"
            )
            # The full checkpoint stores state_dict under "state_dict" key
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            model.load_state_dict(state_dict, strict=False)
            print("Loaded full pretrained Swin UNETR weights (encoder + decoder)")
        except Exception as e:
            print(f"Could not load full pretrained weights: {e}")
            print("Falling back to encoder-only weights...")
            _load_pretrained(model, "encoder")
        return

    if pretrained == "encoder":
        try:
            state_dict = torch.hub.load_state_dict_from_url(
                _ENCODER_ONLY_URL, progress=True, map_location="cpu"
            )
            model.swinViT.load_state_dict(state_dict["state_dict"], strict=False)
            print("Loaded pretrained Swin UNETR encoder weights (self-supervised)")
        except Exception as e:
            print(f"Could not load encoder weights: {e}")
            print("Using random initialization")
        return

    # Treat as a local file path
    path = Path(pretrained)
    if path.exists():
        state_dict = torch.load(str(path), map_location="cpu")
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        elif "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded pretrained weights from {path}")
    else:
        print(f"Pretrained weight file not found: {path}")
        print("Using random initialization")


class SwinUNETRBaseline(nn.Module):
    """Vision-only Swin UNETR baseline (no LLM conditioning).

    Args:
        img_size: Input image size (H, W, D).
        in_channels: Number of input channels (4 for T1, T1ce, T2, FLAIR).
        out_channels: Number of output classes (4 for BG, NCR, ED, ET).
        feature_size: Base feature size for Swin Transformer.
        pretrained: Weight loading mode — True/"full" (recommended), "encoder", False,
                    or a path to a local .pt checkpoint.
        use_checkpoint: Whether to use gradient checkpointing.
    """

    def __init__(
        self,
        img_size: Tuple[int, int, int] = (96, 96, 96),
        in_channels: int = 4,
        out_channels: int = 4,
        feature_size: int = 48,
        pretrained: Union[bool, str] = True,
        use_checkpoint: bool = False,
    ):
        super().__init__()

        self.model = SwinUNETR(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            use_checkpoint=use_checkpoint,
        )

        _load_pretrained(self.model, pretrained)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class LLMConditionedSwinUNETR(nn.Module):
    """Swin UNETR with LLM-conditioned cross-attention in the decoder.

    Architecture:
        1. Swin UNETR encoder extracts multi-scale 3D vision features
        2. LLaMA 3B encodes clinical prompts into sequence embeddings
        3. Cross-attention layers inject text semantics into decoder skip
           connections at each resolution level
        4. Conditioned features are decoded to produce the segmentation map

    The cross-attention is applied at the bottleneck and each decoder stage,
    allowing the model to leverage clinical context (diagnosis, IDH status,
    MGMT methylation, etc.) to guide segmentation.

    Args:
        img_size: Input image size (H, W, D).
        in_channels: Number of input channels.
        out_channels: Number of output classes.
        feature_size: Base feature size for Swin Transformer.
        text_dim: Dimension of LLM text embeddings.
        cross_attn_heads: Number of heads in cross-attention.
        cross_attn_dropout: Dropout rate for cross-attention.
        pretrained_vision: Weight loading mode — True/"full" (recommended), "encoder",
                          False, or a path to a local .pt checkpoint.
        use_checkpoint: Whether to use gradient checkpointing.
    """

    def __init__(
        self,
        img_size: Tuple[int, int, int] = (96, 96, 96),
        in_channels: int = 4,
        out_channels: int = 4,
        feature_size: int = 48,
        text_dim: int = 3072,
        cross_attn_heads: int = 8,
        cross_attn_dropout: float = 0.1,
        pretrained_vision: Union[bool, str] = True,
        use_checkpoint: bool = False,
    ):
        super().__init__()

        self.feature_size = feature_size
        self.text_dim = text_dim

        # Vision backbone: Swin UNETR
        self.swin_unetr = SwinUNETR(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            use_checkpoint=use_checkpoint,
        )

        _load_pretrained(self.swin_unetr, pretrained_vision)

        # Cross-attention layers at each decoder stage.
        # Swin UNETR decoder stages have these channel counts (feature_size=48):
        #   enc0: 48,  enc1: 48,  enc2: 96,  enc3: 192,  bottleneck: 768
        # These correspond to the encoder outputs fed as skip connections.
        decoder_channels = [
            feature_size * 16,  # bottleneck: 768
            feature_size * 8,   # dec3 input: 384
            feature_size * 4,   # dec2 input: 192
            feature_size * 2,   # dec1 input: 96
        ]

        self.cross_attn_layers = nn.ModuleList()
        for ch in decoder_channels:
            n_heads = max(1, min(cross_attn_heads, ch // 64))
            self.cross_attn_layers.append(
                SpatialCrossAttention(
                    in_channels=ch,
                    text_dim=text_dim,
                    num_heads=n_heads,
                    dropout=cross_attn_dropout,
                )
            )

        # Gating parameters: learnable scalars initialised near zero so the
        # model starts close to the pretrained vision-only behaviour.
        self.gate_params = nn.ParameterList([
            nn.Parameter(torch.zeros(1)) for _ in decoder_channels
        ])

    def forward(
        self,
        x: torch.Tensor,
        text_embeddings: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with optional LLM conditioning.

        Args:
            x: Input image tensor (B, C, D, H, W).
            text_embeddings: Sequence embeddings from LLaMA (B, seq_len, text_dim).
                            If None, behaves as a vision-only model.
            text_mask: Attention mask for text (B, seq_len).

        Returns:
            Segmentation logits (B, out_channels, D, H, W).
        """
        # --- Swin UNETR encoder ---
        hidden_states = self.swin_unetr.swinViT(x, self.swin_unetr.normalize)
        # hidden_states is a list of encoder outputs at increasing depth

        enc0 = self.swin_unetr.encoder1(x)            # stem features
        enc1 = self.swin_unetr.encoder2(hidden_states[0])
        enc2 = self.swin_unetr.encoder3(hidden_states[1])
        enc3 = self.swin_unetr.encoder4(hidden_states[2])
        bottleneck = self.swin_unetr.encoder10(hidden_states[4])

        # --- Cross-attention conditioning ---
        if text_embeddings is not None:
            # Condition bottleneck and skip connections
            gate0 = torch.sigmoid(self.gate_params[0])
            bottleneck = bottleneck + gate0 * self.cross_attn_layers[0](
                bottleneck, text_embeddings, text_mask
            )

            gate1 = torch.sigmoid(self.gate_params[1])
            enc3 = enc3 + gate1 * self.cross_attn_layers[1](
                enc3, text_embeddings, text_mask
            )

            gate2 = torch.sigmoid(self.gate_params[2])
            enc2 = enc2 + gate2 * self.cross_attn_layers[2](
                enc2, text_embeddings, text_mask
            )

            gate3 = torch.sigmoid(self.gate_params[3])
            enc1 = enc1 + gate3 * self.cross_attn_layers[3](
                enc1, text_embeddings, text_mask
            )

        # --- Swin UNETR decoder ---
        dec3 = self.swin_unetr.decoder5(bottleneck, hidden_states[3])
        dec2 = self.swin_unetr.decoder4(dec3, enc3)
        dec1 = self.swin_unetr.decoder3(dec2, enc2)
        dec0 = self.swin_unetr.decoder2(dec1, enc1)
        out = self.swin_unetr.decoder1(dec0, enc0)
        logits = self.swin_unetr.out(out)

        return logits


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def get_swin_unetr(
    img_size: Tuple[int, int, int] = (96, 96, 96),
    in_channels: int = 4,
    out_channels: int = 4,
    feature_size: int = 48,
    pretrained: Union[bool, str] = True,
    use_checkpoint: bool = False,
    device: Optional[str] = None,
) -> SwinUNETRBaseline:
    """Create a vision-only Swin UNETR baseline.

    Args:
        pretrained: Weight loading mode:
            - True / "full": Load complete BraTS-pretrained model (recommended).
            - "encoder": Load self-supervised encoder weights only.
            - False: Random initialization.
            - str path: Load from a local .pt checkpoint.
    """
    model = SwinUNETRBaseline(
        img_size=img_size,
        in_channels=in_channels,
        out_channels=out_channels,
        feature_size=feature_size,
        pretrained=pretrained,
        use_checkpoint=use_checkpoint,
    )
    if device is not None:
        model = model.to(device)
    return model


def get_llm_conditioned_model(
    img_size: Tuple[int, int, int] = (96, 96, 96),
    in_channels: int = 4,
    out_channels: int = 4,
    feature_size: int = 48,
    text_dim: int = 3072,
    cross_attn_heads: int = 8,
    cross_attn_dropout: float = 0.1,
    pretrained_vision: Union[bool, str] = True,
    use_checkpoint: bool = False,
    device: Optional[str] = None,
) -> LLMConditionedSwinUNETR:
    """Create an LLM-conditioned Swin UNETR model.

    Args:
        pretrained_vision: Weight loading mode for the vision backbone:
            - True / "full": Load complete BraTS-pretrained model (recommended).
            - "encoder": Load self-supervised encoder weights only.
            - False: Random initialization.
            - str path: Load from a local .pt checkpoint.
    """
    model = LLMConditionedSwinUNETR(
        img_size=img_size,
        in_channels=in_channels,
        out_channels=out_channels,
        feature_size=feature_size,
        text_dim=text_dim,
        cross_attn_heads=cross_attn_heads,
        cross_attn_dropout=cross_attn_dropout,
        pretrained_vision=pretrained_vision,
        use_checkpoint=use_checkpoint,
    )
    if device is not None:
        model = model.to(device)
    return model
