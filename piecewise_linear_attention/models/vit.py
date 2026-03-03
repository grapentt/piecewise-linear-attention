"""Vision Transformer (ViT) with configurable attention mechanisms.

This module implements a Vision Transformer that can use StandardAttention,
LinearAttention, or PiecewiseAttention for comparative analysis on image
classification tasks.

Reference: "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
           (Dosovitskiy et al., 2021)
"""

from typing import Optional

import torch
import torch.nn as nn

from .multihead import MultiHeadAttention


class PatchEmbedding(nn.Module):
    """Split image into patches and embed them.

    Converts an image of shape (batch, channels, height, width) into a sequence
    of patch embeddings of shape (batch, num_patches, embed_dim).
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
    ):
        """Initialize patch embedding layer.

        Args:
            img_size: Input image size (assumed square)
            patch_size: Size of each patch (assumed square)
            in_channels: Number of input channels (3 for RGB)
            embed_dim: Embedding dimension
        """
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # Use Conv2d to extract patches and embed them
        self.projection = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert image to patch embeddings.

        Args:
            x: Image tensor of shape (batch, channels, height, width)

        Returns:
            Patch embeddings of shape (batch, num_patches, embed_dim)
        """
        # (batch, embed_dim, H/P, W/P)
        x = self.projection(x)

        # Flatten patches: (batch, embed_dim, num_patches)
        x = x.flatten(2)

        # Transpose: (batch, num_patches, embed_dim)
        x = x.transpose(1, 2)

        return x


class ViTBlock(nn.Module):
    """Vision Transformer block with configurable attention."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attention_type: str = "standard",
        device: Optional[torch.device] = None,
    ):
        """Initialize ViT block.

        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            mlp_ratio: Ratio of MLP hidden dim to embed dim
            dropout: Dropout rate
            attention_type: "standard", "linear", or "piecewise"
            device: Device to place module on
        """
        super().__init__()

        # Layer norm before attention (ViT uses pre-norm)
        self.norm1 = nn.LayerNorm(embed_dim)

        # Multi-head self-attention (non-causal for ViT)
        self.attn = MultiHeadAttention(
            hidden_dim=embed_dim,
            num_heads=num_heads,
            attention_type=attention_type,
            dropout=dropout,
            causal=False,  # ViT uses bidirectional attention
            device=device,
        )

        # Layer norm before MLP
        self.norm2 = nn.LayerNorm(embed_dim)

        # MLP with GELU activation
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

        if device is not None:
            self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ViT block.

        Args:
            x: Input tensor of shape (batch, seq_len, embed_dim)

        Returns:
            Output tensor of shape (batch, seq_len, embed_dim)
        """
        # Self-attention with residual connection
        x = x + self.attn(self.norm1(x), self.norm1(x))

        # MLP with residual connection
        x = x + self.mlp(self.norm2(x))

        return x


class VisionTransformer(nn.Module):
    """Vision Transformer with configurable attention mechanism.

    This implementation follows the original ViT paper and can be configured
    to use different attention mechanisms for comparative analysis.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attention_type: str = "standard",
        device: Optional[str] = "cpu",
    ):
        """Initialize Vision Transformer.

        Args:
            img_size: Input image size (assumed square)
            patch_size: Size of each patch
            in_channels: Number of input channels (3 for RGB)
            num_classes: Number of output classes
            embed_dim: Embedding dimension
            depth: Number of transformer blocks
            num_heads: Number of attention heads
            mlp_ratio: Ratio of MLP hidden dim to embed dim
            dropout: Dropout rate
            attention_type: "standard", "linear", or "piecewise"
            device: Device to place model on
        """
        super().__init__()

        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.attention_type = attention_type

        # Patch embedding
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        # Class token (learnable)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Positional embeddings (learnable)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        self.dropout = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            ViTBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attention_type=attention_type,
                device=device,
            )
            for _ in range(depth)
        ])

        # Final layer norm
        self.norm = nn.LayerNorm(embed_dim)

        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

        if device is not None:
            self.to(device)

    def _init_weights(self, m):
        """Initialize weights following ViT paper."""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ViT.

        Args:
            x: Image tensor of shape (batch, channels, height, width)

        Returns:
            Logits of shape (batch, num_classes)
        """
        batch_size = x.shape[0]

        # Patch embedding: (batch, num_patches, embed_dim)
        x = self.patch_embed(x)

        # Prepend class token: (batch, 1 + num_patches, embed_dim)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add positional embeddings
        x = x + self.pos_embed
        x = self.dropout(x)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final layer norm
        x = self.norm(x)

        # Extract class token and apply classification head
        cls_token_final = x[:, 0]
        logits = self.head(cls_token_final)

        return logits


# Convenience configurations (matching ViT paper)

def vit_tiny(num_classes: int = 1000, attention_type: str = "standard", **kwargs):
    """ViT-Tiny: 5M parameters."""
    return VisionTransformer(
        embed_dim=192, depth=12, num_heads=3, mlp_ratio=4.0,
        num_classes=num_classes, attention_type=attention_type, **kwargs
    )


def vit_small(num_classes: int = 1000, attention_type: str = "standard", **kwargs):
    """ViT-Small: 22M parameters."""
    return VisionTransformer(
        embed_dim=384, depth=12, num_heads=6, mlp_ratio=4.0,
        num_classes=num_classes, attention_type=attention_type, **kwargs
    )


def vit_base(num_classes: int = 1000, attention_type: str = "standard", **kwargs):
    """ViT-Base: 86M parameters."""
    return VisionTransformer(
        embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0,
        num_classes=num_classes, attention_type=attention_type, **kwargs
    )


__all__ = [
    "VisionTransformer",
    "vit_tiny",
    "vit_small",
    "vit_base",
]
