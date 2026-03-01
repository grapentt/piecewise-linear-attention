"""Configurable Transformer for comparing attention mechanisms.

This transformer allows plugging in different attention types (standard, linear, piecewise)
for comparative analysis on machine translation tasks.
"""

import sys
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import our configurable multi-head attention
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from piecewise_linear_attention.models.multihead import MultiHeadAttention


class MLP(nn.Module):
    """Feed-forward network used in transformer blocks."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        self.up_projection = nn.Linear(input_dim, hidden_dim)
        self.down_projection = nn.Linear(hidden_dim, input_dim)

        if device is not None:
            self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        up_projected = self.up_projection(x)
        activated = F.gelu(up_projected)
        down_projected = self.down_projection(activated)
        return down_projected


class TransformerBlock(nn.Module):
    """Transformer block with configurable attention mechanism."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_hidden_dim: int,
        is_decoder: bool = False,
        dropout: float = 0.1,
        attention_type: str = "standard",
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        self.is_decoder = is_decoder

        # Self-attention with configurable mechanism
        # Decoder self-attention is ALWAYS causal (prevents looking at future tokens)
        # Encoder self-attention is non-causal (bidirectional)
        self.self_attn = MultiHeadAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            attention_type=attention_type,
            dropout=dropout,
            causal=is_decoder,
            device=device,
        )
        self.self_attn_layer_norm = nn.LayerNorm(hidden_dim)
        self.self_attn_dropout = nn.Dropout(dropout)

        # Cross-attention for decoder (no causal masking)
        if is_decoder:
            self.cross_attn = MultiHeadAttention(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                attention_type=attention_type,
                dropout=dropout,
                causal=False,  # Cross-attention is never causal
                device=device,
            )
            self.cross_attn_layer_norm = nn.LayerNorm(hidden_dim)
            self.cross_attn_dropout = nn.Dropout(dropout)

        # Feed-forward network
        self.mlp = MLP(hidden_dim, mlp_hidden_dim, device)
        self.mlp_layer_norm = nn.LayerNorm(hidden_dim)
        self.mlp_dropout = nn.Dropout(dropout)

        if device is not None:
            self.to(device)

    def forward(
        self,
        x: torch.Tensor,
        encoder_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through transformer block.

        Args:
            x: Input tensor [batch, seq_len, hidden_dim]
            encoder_states: Encoder states for cross-attention (decoder only) [batch, encoder_len, hidden_dim]

        Returns:
            Output tensor [batch, seq_len, hidden_dim]
        """
        # Self-attention (causal if decoder, bidirectional if encoder)
        sa_out = self.self_attn(x, x)
        sa_out = self.self_attn_dropout(sa_out)
        x = self.self_attn_layer_norm(x + sa_out)

        # Cross-attention (decoder only)
        if self.is_decoder and encoder_states is not None:
            ca_out = self.cross_attn(x, encoder_states)
            ca_out = self.cross_attn_dropout(ca_out)
            x = self.cross_attn_layer_norm(x + ca_out)

        # Feed-forward network
        mlp_out = self.mlp(x)
        mlp_out = self.mlp_dropout(mlp_out)
        x = self.mlp_layer_norm(x + mlp_out)

        return x


class ConfigurableTransformer(nn.Module):
    """Transformer with configurable attention mechanism for comparative analysis."""

    def __init__(
        self,
        source_dictionary: Dict[str, int],
        target_dictionary: Dict[str, int],
        hidden_dim: int = 256,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        num_heads: int = 4,
        mlp_hidden_dim: int = 768,
        padding_index: int = 0,
        max_seq_len: int = 256,
        attention_type: str = "standard",
        device: Optional[str] = "cpu",
    ):
        """Initialize transformer with chosen attention type.

        Args:
            source_dictionary: Source language vocabulary
            target_dictionary: Target language vocabulary
            hidden_dim: Model dimension
            num_encoder_layers: Number of encoder layers
            num_decoder_layers: Number of decoder layers
            num_heads: Number of attention heads
            mlp_hidden_dim: Hidden dimension of feed-forward network
            padding_index: Padding token index
            max_seq_len: Maximum sequence length
            attention_type: "standard", "linear", or "piecewise"
            device: Device to place model on
        """
        super().__init__()

        self.source_dictionary = source_dictionary
        self.target_dictionary = target_dictionary
        self.id_to_token = {v: k for k, v in target_dictionary.items()}

        self.padding_index = padding_index
        self.num_heads = num_heads
        self.attention_type = attention_type
        self.device = device

        # Embeddings
        self.encoder_embedding = nn.Embedding(len(source_dictionary), hidden_dim)
        self.decoder_embedding = nn.Embedding(len(target_dictionary), hidden_dim)

        # Positional encodings
        self.encoder_positional_encoding = nn.Parameter(
            torch.zeros(1, max_seq_len, hidden_dim)
        )
        self.decoder_positional_encoding = nn.Parameter(
            torch.zeros(1, max_seq_len, hidden_dim)
        )

        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                mlp_hidden_dim=mlp_hidden_dim,
                is_decoder=False,
                attention_type=attention_type,
                device=device,
            )
            for _ in range(num_encoder_layers)
        ])

        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            TransformerBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                mlp_hidden_dim=mlp_hidden_dim,
                is_decoder=True,
                attention_type=attention_type,
                device=device,
            )
            for _ in range(num_decoder_layers)
        ])

        # Output head
        self.head = nn.Linear(hidden_dim, len(target_dictionary))

        if device is not None:
            self.to(device)

    def encode(self, source_indices: torch.Tensor) -> torch.Tensor:
        """Encode source sequence."""
        batch_size, seq_len = source_indices.shape

        # Embed and add positional encoding
        embedded = self.encoder_embedding(source_indices)
        encoded = embedded + self.encoder_positional_encoding[:, :seq_len, :]

        # Pass through encoder layers
        for layer in self.encoder_layers:
            encoded = layer(encoded)

        return encoded

    def decode(
        self,
        target_indices: torch.Tensor,
        encoder_states: torch.Tensor,
    ) -> torch.Tensor:
        """Decode target sequence given encoder states.

        Causal masking is handled internally by the attention mechanism (causal=True).
        """
        _, seq_len = target_indices.shape

        # Embed and add positional encoding
        embedded = self.decoder_embedding(target_indices)
        decoded = embedded + self.decoder_positional_encoding[:, :seq_len, :]

        # Pass through decoder layers
        # Causal masking is handled internally by decoder self-attention
        for layer in self.decoder_layers:
            decoded = layer(decoded, encoder_states=encoder_states)

        return decoded

    def forward(
        self,
        source_indices: torch.Tensor,
        target_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for training."""
        # Encode source
        encoder_states = self.encode(source_indices)

        # Decode target
        decoded = self.decode(target_indices, encoder_states)

        # Project to vocabulary
        logits = self.head(decoded)

        return logits

    def translate(
        self,
        source_indices: torch.Tensor,
        max_len: int = 100,
        sos_token_id: int = 2,
        eos_token_id: int = 3,
    ) -> torch.Tensor:
        """Translate source sequence (greedy decoding)."""
        self.eval()
        with torch.no_grad():
            # Encode source
            encoder_states = self.encode(source_indices)

            batch_size = source_indices.shape[0]

            # Start with SOS token
            target_indices = torch.full(
                (batch_size, 1),
                sos_token_id,
                dtype=torch.long,
                device=source_indices.device
            )

            # Generate tokens one by one
            for _ in range(max_len):
                # Decode current sequence
                decoded = self.decode(target_indices, encoder_states)

                # Get logits for last token
                logits = self.head(decoded[:, -1, :])

                # Greedy decoding
                next_token = logits.argmax(dim=-1, keepdim=True)

                # Append to sequence
                target_indices = torch.cat([target_indices, next_token], dim=1)

                # Stop if all sequences have EOS token
                if (next_token == eos_token_id).all():
                    break

            return target_indices
