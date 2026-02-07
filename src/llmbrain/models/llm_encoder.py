"""LLM Encoder using LLaMA 3B for clinical text embeddings."""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


class LLaMAEncoder(nn.Module):
    """LLaMA 3B encoder for extracting clinical text embeddings.

    Extracts contextualized embeddings from clinical prompts using
    a frozen or fine-tunable LLaMA model.

    Args:
        model_name: HuggingFace model name for LLaMA.
        freeze: Whether to freeze LLM weights.
        output_dim: Output embedding dimension (projects from LLM hidden size).
        pooling: Pooling strategy ('last', 'mean', 'cls').
        device: Device to load model on.
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-3B",
        freeze: bool = True,
        output_dim: Optional[int] = None,
        pooling: str = "last",
        device: Optional[str] = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = True,
    ):
        super().__init__()

        self.model_name = model_name
        self.freeze = freeze
        self.pooling = pooling
        self.device = device

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model with quantization for memory efficiency
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            self.llm = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
            )
        elif load_in_8bit:
            self.llm = AutoModelForCausalLM.from_pretrained(
                model_name,
                load_in_8bit=True,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            self.llm = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )

        # Freeze LLM weights if specified
        if freeze:
            for param in self.llm.parameters():
                param.requires_grad = False
            self.llm.eval()

        # Get hidden size from model config
        self.hidden_size = self.llm.config.hidden_size

        # Optional projection layer
        self.projection = None
        if output_dim is not None and output_dim != self.hidden_size:
            self.projection = nn.Linear(self.hidden_size, output_dim)
            self.output_dim = output_dim
        else:
            self.output_dim = self.hidden_size

    def encode_text(
        self,
        texts: List[str],
        max_length: int = 128,
    ) -> torch.Tensor:
        """Encode text prompts to embeddings.

        Args:
            texts: List of clinical prompt strings.
            max_length: Maximum token length.

        Returns:
            Tensor of shape (batch_size, output_dim).
        """
        # Tokenize
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        # Move to device
        inputs = {k: v.to(self.llm.device) for k, v in inputs.items()}

        # Get hidden states
        with torch.no_grad() if self.freeze else torch.enable_grad():
            outputs = self.llm(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )

        # Get last layer hidden states
        hidden_states = outputs.hidden_states[-1]  # (batch, seq_len, hidden_size)

        # Pool embeddings
        if self.pooling == "last":
            # Use last non-padding token
            attention_mask = inputs["attention_mask"]
            seq_lengths = attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
            embeddings = hidden_states[batch_indices, seq_lengths]
        elif self.pooling == "mean":
            # Mean pooling over non-padding tokens
            attention_mask = inputs["attention_mask"].unsqueeze(-1)
            embeddings = (hidden_states * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
        elif self.pooling == "cls":
            # Use first token (CLS-style)
            embeddings = hidden_states[:, 0]
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling}")

        # Project if needed
        if self.projection is not None:
            embeddings = self.projection(embeddings.float())

        return embeddings

    def forward(
        self,
        texts: List[str],
        max_length: int = 128,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            texts: List of clinical prompt strings.
            max_length: Maximum token length.

        Returns:
            Text embeddings tensor.
        """
        return self.encode_text(texts, max_length)

    def get_sequence_embeddings(
        self,
        texts: List[str],
        max_length: int = 128,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get full sequence embeddings for cross-attention.

        Args:
            texts: List of clinical prompt strings.
            max_length: Maximum token length.

        Returns:
            Tuple of (sequence_embeddings, attention_mask).
            sequence_embeddings: (batch, seq_len, hidden_size)
            attention_mask: (batch, seq_len)
        """
        # Tokenize
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        inputs = {k: v.to(self.llm.device) for k, v in inputs.items()}

        with torch.no_grad() if self.freeze else torch.enable_grad():
            outputs = self.llm(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )

        hidden_states = outputs.hidden_states[-1]

        # Project if needed
        if self.projection is not None:
            hidden_states = self.projection(hidden_states.float())

        return hidden_states, inputs["attention_mask"]


class CachedLLMEncoder(LLaMAEncoder):
    """LLM Encoder with embedding caching for efficiency.

    Caches computed embeddings to avoid redundant computation
    during training when the same prompts are used repeatedly.
    """

    def __init__(self, *args, cache_size: int = 1000, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache = {}
        self.cache_size = cache_size

    def encode_text(
        self,
        texts: List[str],
        max_length: int = 128,
    ) -> torch.Tensor:
        """Encode with caching."""
        # Check cache
        cache_keys = [f"{t}_{max_length}" for t in texts]
        cached_indices = []
        uncached_indices = []
        uncached_texts = []

        for i, (text, key) in enumerate(zip(texts, cache_keys)):
            if key in self.cache:
                cached_indices.append(i)
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        # Compute uncached embeddings
        if uncached_texts:
            new_embeddings = super().encode_text(uncached_texts, max_length)

            # Update cache
            for i, (text, key) in enumerate(zip(uncached_texts, [cache_keys[j] for j in uncached_indices])):
                if len(self.cache) < self.cache_size:
                    self.cache[key] = new_embeddings[i].detach().cpu()

        # Assemble results
        embeddings = torch.zeros(len(texts), self.output_dim, device=self.llm.device)

        for i, idx in enumerate(uncached_indices):
            embeddings[idx] = new_embeddings[i] if uncached_texts else None

        for idx in cached_indices:
            embeddings[idx] = self.cache[cache_keys[idx]].to(self.llm.device)

        return embeddings

    def clear_cache(self):
        """Clear the embedding cache."""
        self.cache.clear()
