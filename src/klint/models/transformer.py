"""Causal Transformer backbone with RoPE and RMSNorm."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from klint.models.rmsnorm import RMSNorm
from klint.models.rope import apply_rotary_pos_emb


class CausalSelfAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention with Rotary Position Embeddings.
    """
    def __init__(
        self,
        d_model: int = 480,
        n_heads: int = 10,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model {d_model} must be divisible by n_heads {n_heads}"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, seq_len, d_model)
            cos, sin: (seq_len, head_dim) rotary position embedding caches
            mask: Optional causal mask
        """
        B, T, C = x.shape

        # Linear projections
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Scaled dot-product attention (uses FlashAttention-2 / Memory-Efficient attention on GPU)
        dropout_p = self.dropout.p if self.training else 0.0
        if mask is None:
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=dropout_p,
                is_causal=True,
            )
        else:
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask,
                dropout_p=dropout_p,
                is_causal=False,
            )

        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.out_proj(out)


class FeedForward(nn.Module):
    """
    Two-layer MLP with 4x expansion and GELU activation.
    """
    def __init__(self, d_model: int = 480, d_ff: int = 1920, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(self.act(self.w1(x))))


class CausalTransformerBlock(nn.Module):
    """
    Pre-LayerNorm Causal Transformer block using RMSNorm.
    """
    def __init__(
        self,
        d_model: int = 480,
        n_heads: int = 10,
        d_ff: int = 1920,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = FeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Pre-norm residual attention
        x = x + self.attn(self.attn_norm(x), cos, sin, mask)
        # Pre-norm residual feedforward
        x = x + self.ffn(self.ffn_norm(x))
        return x


class CausalTransformerBackbone(nn.Module):
    """
    10-layer decoder-only causal Transformer backbone.
    """
    def __init__(
        self,
        d_model: int = 480,
        n_layers: int = 10,
        n_heads: int = 10,
        d_ff: int = 1920,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            CausalTransformerBlock(
                d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout
            )
            for _ in range(n_layers)
        ])
        self.final_norm = RMSNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, cos, sin, mask)
        return self.final_norm(x)
