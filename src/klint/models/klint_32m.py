"""Klint-32M Foundation Model: Full model integration, causal factor routing, and generation."""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from klint.models.rope import RotaryEmbedding
from klint.models.transformer import CausalTransformerBackbone


@dataclass
class KlintConfig:
    """Configuration for Klint-32M model."""
    d_model: int = 480
    n_layers: int = 10
    n_heads: int = 10
    d_ff: int = 1920
    price_vocab_size: int = 512
    range_vocab_size: int = 256
    activity_vocab_size: int = 256
    max_seq_len: int = 4096
    dropout: float = 0.1
    rope_base: float = 10000.0
    gradient_checkpointing: bool = False


class Klint32M(nn.Module):
    """
    Klint-32M: 10-layer, 480-hidden, 10-head causal autoregressive foundation model.
    
    Generates financial trajectories following the causal factor order:
    Price -> Range (conditioned on Price) -> Activity (conditioned on Price & Range).
    """
    def __init__(self, config: Optional[KlintConfig] = None):
        super().__init__()
        self.config = config or KlintConfig()

        d = self.config.d_model

        # Token embeddings for each factor stream
        self.embed_price = nn.Embedding(self.config.price_vocab_size, d)
        self.embed_range = nn.Embedding(self.config.range_vocab_size, d)
        self.embed_activity = nn.Embedding(self.config.activity_vocab_size, d)

        # Factor type embeddings: 0 = Price, 1 = Range, 2 = Activity
        self.embed_factor_type = nn.Embedding(3, d)

        # Rotary position embeddings
        head_dim = d // self.config.n_heads
        self.rope = RotaryEmbedding(dim=head_dim, max_seq_len=self.config.max_seq_len, base=self.config.rope_base)

        # 10-layer causal transformer backbone
        self.backbone = CausalTransformerBackbone(
            d_model=d,
            n_layers=self.config.n_layers,
            n_heads=self.config.n_heads,
            d_ff=self.config.d_ff,
            dropout=self.config.dropout,
            gradient_checkpointing=self.config.gradient_checkpointing,
        )

        # Factor-specific prediction heads
        self.head_price = nn.Linear(d, self.config.price_vocab_size, bias=False)
        self.head_range = nn.Linear(d, self.config.range_vocab_size, bias=False)
        self.head_activity = nn.Linear(d, self.config.activity_vocab_size, bias=False)

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

        # Separate masks for each factor stream
        mask_p = (factor_types == 0)
        mask_r = (factor_types == 1)
        mask_a = (factor_types == 2)

        x = torch.zeros(B, seq_len, self.config.d_model, device=device, dtype=torch.float32)

        if mask_p.any():
            x[:, mask_p] = self.embed_price(sequence[:, mask_p])
        if mask_r.any():
            x[:, mask_r] = self.embed_range(sequence[:, mask_r])
        if mask_a.any():
            x[:, mask_a] = self.embed_activity(sequence[:, mask_a])

        return x + type_emb

    def forward(
        self,
        sequence: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass over interleaved sequence of factor tokens.

        Args:
            sequence: (B, seq_len) token IDs.
            targets: Optional (B, seq_len) target token IDs for next-token prediction.

        Returns:
            Dict containing:
                - 'hidden': (B, seq_len, d_model)
                - 'logits_price': (B, seq_len, 512)
                - 'logits_range': (B, seq_len, 256)
                - 'logits_activity': (B, seq_len, 256)
                - 'loss': Cross-entropy loss (if targets provided)
        """
        B, seq_len = sequence.shape
        device = sequence.device

        # Embed input tokens
        x = self.embed_interleaved(sequence)

        # Get RoPE frequencies
        cos, sin = self.rope(seq_len, device=device)

        # Transformer forward pass
        hidden = self.backbone(x, cos, sin)

        # Compute factor-specific logits
        logits_p = self.head_price(hidden)
        logits_r = self.head_range(hidden)
        logits_a = self.head_activity(hidden)

        out = {
            "hidden": hidden,
            "logits_price": logits_p,
            "logits_range": logits_r,
            "logits_activity": logits_a,
        }

        if targets is not None:
            # Shift for autoregressive loss: position i predicts targets[:, i]
            # Next factor type depends on position i:
            # position i % 3 == 0 (Price): next target is Range (type 1)
            # position i % 3 == 1 (Range): next target is Activity (type 2)
            # position i % 3 == 2 (Activity): next target is next bar Price (type 0)
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

    @torch.no_grad()
    def generate_tokens(
        self,
        prefix_sequence: torch.Tensor,
        num_bars: int = 10,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
    ) -> torch.Tensor:
        """
        Autoregressively generates future factor tokens adhering strictly to
        the causal factor sequence: Price -> Range -> Activity.

        Args:
            prefix_sequence: (1, current_len) starting prompt tokens.
            num_bars: Number of future financial bars to generate (3 * num_bars tokens).
        """
        self.eval()
        device = prefix_sequence.device
        current_seq = prefix_sequence.clone()
        total_tokens_to_generate = num_bars * 3

        for _ in range(total_tokens_to_generate):
            seq_len = current_seq.shape[1]
            if seq_len >= self.config.max_seq_len:
                break

            # Current position determines which factor comes next
            # Sequence has length seq_len. Last token was at index seq_len - 1.
            # Next token to predict is at index seq_len:
            next_type = seq_len % 3

            out = self.forward(current_seq)
            last_hidden = out["hidden"][:, -1, :]  # (1, d)

            if next_type == 0:  # Next is Price
                logits = self.head_price(last_hidden) / max(temperature, 1e-5)
            elif next_type == 1:  # Next is Range
                logits = self.head_range(last_hidden) / max(temperature, 1e-5)
            else:  # Next is Activity
                logits = self.head_activity(last_hidden) / max(temperature, 1e-5)

            # Top-K filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float("-inf")

            # Top-P (nucleus) filtering
            if 0.0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift right to keep first token above threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[0, indices_to_remove] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)

            current_seq = torch.cat([current_seq, next_token], dim=1)

        return current_seq
