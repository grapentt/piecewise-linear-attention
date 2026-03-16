"""Generic encoder for LRA tasks.

This encoder architecture is shared across all LRA tasks. Only the
embedding layer and task head differ per task.

Reuses TransformerBlock from translation_transformer.py for consistency.
"""
import torch
import torch.nn as nn
from typing import Optional

# Import from existing piecewise-linear-attention models
from piecewise_linear_attention.models.translation_transformer import TransformerBlock


class LRAEncoder(nn.Module):
    """Generic encoder for all LRA tasks.

    This architecture is shared across ListOps, Text, Image, Retrieval,
    and Pathfinder tasks. Only the embedding layer differs per task.

    Follows encoder-only architecture (like BERT) with:
    - Token + positional embeddings
    - Stack of TransformerBlocks (reused from translation)
    - Classification head
    """

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        num_classes: int,
        emb_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        mlp_dim: int = 1024,
        attention_type: str = "standard",
        dropout: float = 0.1,
        pooling_mode: str = "CLS",
        device: Optional[torch.device] = None,
    ):
        """Initialize LRA encoder.

        Args:
            vocab_size: Size of vocabulary (includes PAD token at index 0)
            max_seq_len: Maximum sequence length
            num_classes: Number of classification classes
            emb_dim: Embedding dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            mlp_dim: MLP hidden dimension
            attention_type: Type of attention ("standard", "linear", "piecewise")
            dropout: Dropout rate
            pooling_mode: Pooling strategy ("CLS" or "MEAN")
            device: Device to use
        """
        super().__init__()

        self.emb_dim = emb_dim
        self.pooling_mode = pooling_mode
        self.max_seq_len = max_seq_len

        # Token embeddings (vocab_size includes padding token at 0)
        self.token_embedding = nn.Embedding(
            vocab_size, emb_dim, padding_idx=0
        )

        # Positional embeddings (learnable, as in LRA)
        # Add 1 for CLS token if using CLS pooling
        pos_len = max_seq_len + 1 if pooling_mode == "CLS" else max_seq_len
        self.position_embedding = nn.Embedding(pos_len, emb_dim)

        # CLS token (if using CLS pooling)
        if pooling_mode == "CLS":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
            nn.init.normal_(self.cls_token, std=0.02)

        self.dropout = nn.Dropout(dropout)

        # Encoder layers - REUSE TransformerBlock from translation!
        # is_decoder=False means encoder-only (bidirectional attention)
        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_dim=emb_dim,
                num_heads=num_heads,
                mlp_hidden_dim=mlp_dim,
                is_decoder=False,  # Encoder: non-causal attention
                dropout=dropout,
                attention_type=attention_type,
                device=device,
            )
            for _ in range(num_layers)
        ])

        # Final layer norm
        self.norm = nn.LayerNorm(emb_dim)

        # Classification head
        self.classifier = nn.Linear(emb_dim, num_classes)

        if device is not None:
            self.to(device)

    def forward(
        self,
        input_ids: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            input_ids: (batch, seq_len) - token IDs
            padding_mask: (batch, seq_len) - True for valid, False for padding

        Returns:
            logits: (batch, num_classes)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Token embeddings
        x = self.token_embedding(input_ids)  # (batch, seq_len, emb_dim)

        # Add CLS token if using CLS pooling
        if self.pooling_mode == "CLS":
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
            # Update padding mask for CLS token
            if padding_mask is not None:
                cls_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=device)
                padding_mask = torch.cat([cls_mask, padding_mask], dim=1)
            seq_len += 1

        # Positional embeddings
        positions = torch.arange(seq_len, device=device)
        x = x + self.position_embedding(positions)
        x = self.dropout(x)

        # Encoder layers
        # TransformerBlock expects (x, encoder_states)
        # For encoder-only, encoder_states=None
        for layer in self.layers:
            x = layer(x, encoder_states=None)

        # Final norm
        x = self.norm(x)

        # Pooling
        if self.pooling_mode == "CLS":
            pooled = x[:, 0]  # Use CLS token
        elif self.pooling_mode == "MEAN":
            if padding_mask is not None:
                # Mask out padding tokens before mean
                mask = padding_mask.unsqueeze(-1).float()
                pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            else:
                pooled = x.mean(dim=1)
        else:
            raise ValueError(f"Unknown pooling mode: {self.pooling_mode}")

        # Classification
        logits = self.classifier(pooled)

        return logits
