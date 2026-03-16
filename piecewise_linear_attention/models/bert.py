"""BERT model with configurable attention mechanisms.

This module implements a BERT (Bidirectional Encoder Representations from Transformers)
model that can use StandardAttention, LinearAttention, or PiecewiseAttention for
comparative analysis on language understanding tasks.

Reference: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
           (Devlin et al., 2019)
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .multihead import MultiHeadAttention


class BERTEmbeddings(nn.Module):
    """BERT embeddings: token + position + segment embeddings."""

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        max_position_embeddings: int = 512,
        type_vocab_size: int = 2,
        dropout: float = 0.1,
    ):
        """Initialize BERT embeddings.

        Args:
            vocab_size: Size of vocabulary
            hidden_dim: Hidden dimension
            max_position_embeddings: Maximum sequence length
            type_vocab_size: Number of segment types (usually 2 for sentence A/B)
            dropout: Dropout probability
        """
        super().__init__()

        self.token_embeddings = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_dim)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_dim)

        self.layer_norm = nn.LayerNorm(hidden_dim, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute BERT embeddings.

        Args:
            input_ids: Token indices of shape (batch, seq_len)
            token_type_ids: Segment indices of shape (batch, seq_len)
            position_ids: Position indices of shape (batch, seq_len)

        Returns:
            Embeddings of shape (batch, seq_len, hidden_dim)
        """
        batch_size, seq_len = input_ids.shape

        # Create position IDs if not provided
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # Create token type IDs if not provided
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # Compute embeddings
        token_embeds = self.token_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        token_type_embeds = self.token_type_embeddings(token_type_ids)

        # Sum embeddings
        embeddings = token_embeds + position_embeds + token_type_embeds

        # Layer norm and dropout
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class BERTBlock(nn.Module):
    """BERT encoder block with configurable attention."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        intermediate_dim: int,
        dropout: float = 0.1,
        attention_type: str = "standard",
        device: Optional[torch.device] = None,
    ):
        """Initialize BERT block.

        Args:
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads
            intermediate_dim: Intermediate dimension for feed-forward network
            dropout: Dropout probability
            attention_type: "standard", "linear", or "piecewise"
            device: Device to place module on
        """
        super().__init__()

        # Self-attention
        self.attention = MultiHeadAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            attention_type=attention_type,
            dropout=dropout,
            causal=False,  # BERT uses bidirectional attention
            device=device,
        )

        # Layer norm after attention
        self.attention_layer_norm = nn.LayerNorm(hidden_dim, eps=1e-12)
        self.attention_dropout = nn.Dropout(dropout)

        # Feed-forward network
        self.intermediate = nn.Linear(hidden_dim, intermediate_dim)
        self.intermediate_act = nn.GELU()
        self.output = nn.Linear(intermediate_dim, hidden_dim)

        # Layer norm after feed-forward
        self.output_layer_norm = nn.LayerNorm(hidden_dim, eps=1e-12)
        self.output_dropout = nn.Dropout(dropout)

        if device is not None:
            self.to(device)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass through BERT block.

        Args:
            hidden_states: Input tensor of shape (batch, seq_len, hidden_dim)

        Returns:
            Output tensor of shape (batch, seq_len, hidden_dim)
        """
        # Self-attention with residual connection
        attention_output = self.attention(hidden_states, hidden_states)
        attention_output = self.attention_dropout(attention_output)
        hidden_states = self.attention_layer_norm(hidden_states + attention_output)

        # Feed-forward network with residual connection
        intermediate_output = self.intermediate(hidden_states)
        intermediate_output = self.intermediate_act(intermediate_output)
        layer_output = self.output(intermediate_output)
        layer_output = self.output_dropout(layer_output)
        hidden_states = self.output_layer_norm(hidden_states + layer_output)

        return hidden_states


class BERTModel(nn.Module):
    """BERT model with configurable attention mechanism.

    This is the base BERT encoder without task-specific heads.
    Use BERTForMaskedLM or BERTForSequenceClassification for specific tasks.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        intermediate_dim: int = 3072,
        max_position_embeddings: int = 512,
        type_vocab_size: int = 2,
        dropout: float = 0.1,
        attention_type: str = "standard",
        device: Optional[str] = "cpu",
    ):
        """Initialize BERT model.

        Args:
            vocab_size: Size of vocabulary
            hidden_dim: Hidden dimension (768 for BERT-base, 1024 for BERT-large)
            num_layers: Number of encoder layers (12 for base, 24 for large)
            num_heads: Number of attention heads (12 for base, 16 for large)
            intermediate_dim: FFN intermediate dimension (usually 4 * hidden_dim)
            max_position_embeddings: Maximum sequence length
            type_vocab_size: Number of segment types
            dropout: Dropout probability
            attention_type: "standard", "linear", or "piecewise"
            device: Device to place model on
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.attention_type = attention_type

        # Embeddings
        self.embeddings = BERTEmbeddings(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            dropout=dropout,
        )

        # Encoder blocks
        self.encoder_blocks = nn.ModuleList([
            BERTBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                intermediate_dim=intermediate_dim,
                dropout=dropout,
                attention_type=attention_type,
                device=device,
            )
            for _ in range(num_layers)
        ])

        # Pooler for [CLS] token
        self.pooler = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        if device is not None:
            self.to(device)

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through BERT.

        Args:
            input_ids: Token indices of shape (batch, seq_len)
            token_type_ids: Segment indices of shape (batch, seq_len)
            attention_mask: Attention mask of shape (batch, seq_len)
                          (1 for tokens to attend to, 0 for padding)

        Returns:
            sequence_output: Full sequence representations (batch, seq_len, hidden_dim)
            pooled_output: Pooled [CLS] token representation (batch, hidden_dim)
        """
        # Compute embeddings
        hidden_states = self.embeddings(input_ids, token_type_ids)

        # Apply encoder blocks
        for block in self.encoder_blocks:
            hidden_states = block(hidden_states)

        # Pooled output from [CLS] token (first token)
        pooled_output = self.pooler(hidden_states[:, 0])

        return hidden_states, pooled_output


class BERTForSequenceClassification(nn.Module):
    """BERT for sequence classification tasks (e.g., sentiment analysis, GLUE)."""

    def __init__(
        self,
        bert_model: BERTModel,
        num_labels: int,
        dropout: float = 0.1,
    ):
        """Initialize classification model.

        Args:
            bert_model: Pre-initialized BERT model
            num_labels: Number of classification labels
            dropout: Dropout probability for classification head
        """
        super().__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(bert_model.hidden_dim, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for classification.

        Args:
            input_ids: Token indices of shape (batch, seq_len)
            token_type_ids: Segment indices of shape (batch, seq_len)
            attention_mask: Attention mask of shape (batch, seq_len)

        Returns:
            Logits of shape (batch, num_labels)
        """
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


class BERTForMaskedLM(nn.Module):
    """BERT for masked language modeling (pre-training)."""

    def __init__(self, bert_model: BERTModel):
        """Initialize masked LM model.

        Args:
            bert_model: Pre-initialized BERT model
        """
        super().__init__()
        self.bert = bert_model

        # MLM head
        self.mlm_head = nn.Sequential(
            nn.Linear(bert_model.hidden_dim, bert_model.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(bert_model.hidden_dim, eps=1e-12),
            nn.Linear(bert_model.hidden_dim, bert_model.vocab_size),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for masked language modeling.

        Args:
            input_ids: Token indices of shape (batch, seq_len)
            token_type_ids: Segment indices of shape (batch, seq_len)
            attention_mask: Attention mask of shape (batch, seq_len)

        Returns:
            Logits of shape (batch, seq_len, vocab_size)
        """
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask)
        prediction_scores = self.mlm_head(sequence_output)
        return prediction_scores


# Convenience configurations

def bert_tiny(vocab_size: int, num_labels: int = 2, attention_type: str = "standard", **kwargs):
    """BERT-Tiny for quick experiments."""
    bert = BERTModel(
        vocab_size=vocab_size,
        hidden_dim=128,
        num_layers=2,
        num_heads=2,
        intermediate_dim=512,
        attention_type=attention_type,
        **kwargs
    )
    return BERTForSequenceClassification(bert, num_labels)


def bert_base(vocab_size: int, num_labels: int = 2, attention_type: str = "standard", **kwargs):
    """BERT-Base: 110M parameters."""
    bert = BERTModel(
        vocab_size=vocab_size,
        hidden_dim=768,
        num_layers=12,
        num_heads=12,
        intermediate_dim=3072,
        attention_type=attention_type,
        **kwargs
    )
    return BERTForSequenceClassification(bert, num_labels)


__all__ = [
    "BERTModel",
    "BERTForSequenceClassification",
    "BERTForMaskedLM",
    "bert_tiny",
    "bert_base",
]
