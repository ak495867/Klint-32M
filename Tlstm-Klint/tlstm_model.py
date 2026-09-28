"""
Temporal LSTM (TLSTM) Student Architecture for Klint Foundation Distillation.

A compact, causal temporal recurrent network designed to distill knowledge from
the flagship Klint-32M v2 foundation model (~28.6M parameters) into an ultra-fast,
low-parameter student model (~0.59M parameters, ~48x compression).
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TLSTMConfig:
    """Configuration for Temporal LSTM (TLSTM) student model."""
    d_model: int = 128
    hidden_dim: int = 128
    num_layers: int = 2
    price_vocab_size: int = 512
    range_vocab_size: int = 256
    activity_vocab_size: int = 256
    dropout: float = 0.1
    use_temporal_attention: bool = True
    context_bars: int = 64
    max_seq_len: int = 2048


class CausalTemporalAttention(nn.Module):
    """
    Lightweight Causal Multi-Head / Single-Head Temporal Attention.
    Allows the recurrent student to perform causal self-attention across
    the unrolled LSTM hidden states without breaking causality.
    """
    def __init__(self, hidden_dim: int = 128, num_heads: int = 2, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, hidden_dim)
        Returns: (B, T, hidden_dim) with residual connection
        """
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T, D)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T, D)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T, D)

        # Causal mask: position t can only attend to <= t
        causal_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, H, T, T)
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, v)  # (B, H, T, D)
        context = context.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(context)
        return self.norm(x + out)


class TLSTMKlint(nn.Module):
    """
    Temporal LSTM Student Model for Klint Causal Factor Streams.
    
    Ingests interleaved factor sequences [P_0, R_0, A_0, P_1, R_1, A_1, ...],
    processes temporal dependencies with a 2-layer causal LSTM + causal attention,
    and predicts factor logits and continuous return estimates.
    """
    def __init__(self, config: Optional[TLSTMConfig] = None):
        super().__init__()
        self.config = config or TLSTMConfig()
        d = self.config.d_model
        h = self.config.hidden_dim

        # Factor token embeddings
        self.embed_price = nn.Embedding(self.config.price_vocab_size, d)
        self.embed_range = nn.Embedding(self.config.range_vocab_size, d)
        self.embed_activity = nn.Embedding(self.config.activity_vocab_size, d)
        self.embed_factor_type = nn.Embedding(3, d)

        # 2-Layer Temporal LSTM Backbone
        self.lstm = nn.LSTM(
            input_size=d,
            hidden_size=h,
            num_layers=self.config.num_layers,
            batch_first=True,
            dropout=self.config.dropout if self.config.num_layers > 1 else 0.0,
            bidirectional=False,  # Strictly causal
        )

        # Causal Temporal Attention (Optional / Enabled by default)
        if self.config.use_temporal_attention:
            self.temporal_attn = CausalTemporalAttention(
                hidden_dim=h,
                num_heads=2,
                dropout=self.config.dropout,
            )
        else:
            self.temporal_attn = None

        # Layer Normalization
        self.ln_out = nn.LayerNorm(h)

        # Factor Prediction Heads
        self.head_price = nn.Linear(h, self.config.price_vocab_size, bias=False)
        self.head_range = nn.Linear(h, self.config.range_vocab_size, bias=False)
        self.head_activity = nn.Linear(h, self.config.activity_vocab_size, bias=False)

        # Direct Return Estimation Head (auxiliary regression)
        self.head_return = nn.Linear(h, 1)

    def count_parameters(self) -> int:
        """Returns total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def embed_interleaved(self, sequence: torch.Tensor) -> torch.Tensor:
        """
        Embeds a flattened sequence of tokens where positions mod 3 are:
        0 -> Price code in [0, 511]
        1 -> Range code in [0, 255]
        2 -> Activity code in [0, 255]
        """
        B, seq_len = sequence.shape
        device = sequence.device

        # Factor type indices: [0, 1, 2, 0, 1, 2, ...]
        factor_types = torch.arange(seq_len, device=device) % 3
        type_emb = self.embed_factor_type(factor_types).unsqueeze(0)  # (1, seq_len, d)

        # Masks for each factor stream
        mask_p = (factor_types == 0)
        mask_r = (factor_types == 1)
        mask_a = (factor_types == 2)

        x = torch.zeros(B, seq_len, self.config.d_model, device=device, dtype=torch.float32)

        if mask_p.any():
            tok_p = torch.clamp(sequence[:, mask_p], 0, self.config.price_vocab_size - 1)
            x[:, mask_p] = self.embed_price(tok_p)
        if mask_r.any():
            tok_r = torch.clamp(sequence[:, mask_r], 0, self.config.range_vocab_size - 1)
            x[:, mask_r] = self.embed_range(tok_r)
        if mask_a.any():
            tok_a = torch.clamp(sequence[:, mask_a], 0, self.config.activity_vocab_size - 1)
            x[:, mask_a] = self.embed_activity(tok_a)

        return x + type_emb

    def forward(
        self,
        sequence: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Dict[str, Any]:
        """
        Forward pass over interleaved sequence of factor tokens.

        Args:
            sequence: (B, seq_len) token IDs.
            targets: Optional (B, seq_len) target token IDs.
            hidden_state: Optional (h_0, c_0) tuple for recurrent streaming inference.

        Returns:
            Dict containing:
                - 'hidden': (B, seq_len, hidden_dim)
                - 'logits_price': (B, seq_len, 512)
                - 'logits_range': (B, seq_len, 256)
                - 'logits_activity': (B, seq_len, 256)
                - 'pred_return': (B, seq_len, 1)
                - 'next_hidden_state': Tuple (h_n, c_n)
                - 'loss': Optional hard cross-entropy loss if targets provided
        """
        B, seq_len = sequence.shape
        device = sequence.device

        # 1. Embed interleaved factor sequence
        x = self.embed_interleaved(sequence)  # (B, seq_len, d_model)

        # 2. Causal LSTM Backbone
        lstm_out, next_hidden = self.lstm(x, hidden_state)  # (B, seq_len, hidden_dim)

        # 3. Causal Temporal Attention (if enabled)
        if self.temporal_attn is not None:
            hidden = self.temporal_attn(lstm_out)
        else:
            hidden = lstm_out

        hidden = self.ln_out(hidden)

        # 4. Compute Factor Logits & Continuous Return Prediction
        logits_p = self.head_price(hidden)       # (B, seq_len, 512)
        logits_r = self.head_range(hidden)       # (B, seq_len, 256)
        logits_a = self.head_activity(hidden)    # (B, seq_len, 256)
        pred_return = self.head_return(hidden)   # (B, seq_len, 1)

        out = {
            "hidden": hidden,
            "logits_price": logits_p,
            "logits_range": logits_r,
            "logits_activity": logits_a,
            "pred_return": pred_return,
            "next_hidden_state": next_hidden,
        }

        # 5. Compute hard target loss if targets provided
        if targets is not None:
            next_factor_type = (torch.arange(seq_len, device=device) + 1) % 3
            loss = torch.tensor(0.0, device=device)
            count = 0

            mask_to_p = (next_factor_type == 0)
            mask_to_r = (next_factor_type == 1)
            mask_to_a = (next_factor_type == 2)

            if mask_to_p.any():
                lp = logits_p[:, mask_to_p].reshape(-1, self.config.price_vocab_size)
                tp = targets[:, mask_to_p].reshape(-1)
                loss = loss + F.cross_entropy(lp, tp)
                count += 1

            if mask_to_r.any():
                lr = logits_r[:, mask_to_r].reshape(-1, self.config.range_vocab_size)
                tr = targets[:, mask_to_r].reshape(-1)
                loss = loss + F.cross_entropy(lr, tr)
                count += 1

            if mask_to_a.any():
                la = logits_a[:, mask_to_a].reshape(-1, self.config.activity_vocab_size)
                ta = targets[:, mask_to_a].reshape(-1)
                loss = loss + F.cross_entropy(la, ta)
                count += 1

            out["loss"] = loss / max(count, 1)

        return out
