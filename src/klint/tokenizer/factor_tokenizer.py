"""Multi-stream Factor Tokenizer managing price, range, and activity codebooks."""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
from klint.tokenizer.rvq import ResidualVectorQuantizer


class FactorTokenizer(nn.Module):
    """
    Tokenizes the three causal financial factor streams using dedicated RVQ codebooks:
    - Price-path: 512 codes
    - Range-shape: 256 codes
    - Activity: 256 codes
    """
    def __init__(
        self,
        price_vocab_size: int = 512,
        range_vocab_size: int = 256,
        activity_vocab_size: int = 256,
        latent_dim: int = 64,
        decay: float = 0.99,
    ):
        super().__init__()
        self.price_vocab_size = price_vocab_size
        self.range_vocab_size = range_vocab_size
        self.activity_vocab_size = activity_vocab_size
        self.latent_dim = latent_dim

        # Input projections: continuous factors -> latent_dim
        self.proj_price = nn.Sequential(
            nn.Linear(2, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.proj_range = nn.Sequential(
            nn.Linear(3, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.proj_activity = nn.Sequential(
            nn.Linear(1, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )

        # RVQ codebooks
        self.rvq_price = ResidualVectorQuantizer(price_vocab_size, latent_dim, num_stages=1, decay=decay)
        self.rvq_range = ResidualVectorQuantizer(range_vocab_size, latent_dim, num_stages=1, decay=decay)
        self.rvq_activity = ResidualVectorQuantizer(activity_vocab_size, latent_dim, num_stages=1, decay=decay)

        # Output projections: latent_dim -> continuous factors
        self.dec_price = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, 2),
        )
        self.dec_range = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, 3),
        )
        self.dec_activity = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, 1),
        )

    def encode(
        self,
        price_path: torch.Tensor,
        range_shape: torch.Tensor,
        activity: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encodes continuous factor streams into discrete codebook tokens.
        
        Args:
            price_path: (B, T, 2)
            range_shape: (B, T, 3)
            activity: (B, T, 1)

        Returns:
            p_tokens: (B, T)
            r_tokens: (B, T)
            a_tokens: (B, T)
            total_vq_loss: scalar commitment loss
        """
        z_p = self.proj_price(price_path)
        z_r = self.proj_range(range_shape)
        z_a = self.proj_activity(activity)

        q_p, idx_p, loss_p = self.rvq_price(z_p)
        q_r, idx_r, loss_r = self.rvq_range(z_r)
        q_a, idx_a, loss_a = self.rvq_activity(z_a)

        # idx shape is (num_stages, B, T) -> squeeze stage dim for 1 stage
        p_tokens = idx_p.squeeze(0)
        r_tokens = idx_r.squeeze(0)
        a_tokens = idx_a.squeeze(0)

        total_loss = loss_p + loss_r + loss_a
        return p_tokens, r_tokens, a_tokens, total_loss

    def forward(
        self,
        price_path: torch.Tensor,
        range_shape: torch.Tensor,
        activity: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass for training the autoencoder with straight-through gradients.
        """
        z_p = self.proj_price(price_path)
        z_r = self.proj_range(range_shape)
        z_a = self.proj_activity(activity)

        q_p, idx_p, loss_p = self.rvq_price(z_p)
        q_r, idx_r, loss_r = self.rvq_range(z_r)
        q_a, idx_a, loss_a = self.rvq_activity(z_a)

        rec_p = self.dec_price(q_p)
        rec_r = self.dec_range(q_r)
        rec_a = self.dec_activity(q_a)

        return {
            "rec_price": rec_p,
            "rec_range": rec_r,
            "rec_activity": rec_a,
            "tokens_price": idx_p.squeeze(0),
            "tokens_range": idx_r.squeeze(0),
            "tokens_activity": idx_a.squeeze(0),
            "vq_loss": loss_p + loss_r + loss_a,
        }

    def decode_tokens(
        self,
        p_tokens: torch.Tensor,
        r_tokens: torch.Tensor,
        a_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Decodes discrete token IDs into continuous factor estimates.
        """
        emb_p = self.rvq_price.layers[0].embedding
        emb_r = self.rvq_range.layers[0].embedding
        emb_a = self.rvq_activity.layers[0].embedding

        z_p = torch.nn.functional.embedding(p_tokens, emb_p)
        z_r = torch.nn.functional.embedding(r_tokens, emb_r)
        z_a = torch.nn.functional.embedding(a_tokens, emb_a)

        rec_price = self.dec_price(z_p)
        rec_range = self.dec_range(z_r)
        rec_activity = self.dec_activity(z_a)

        return rec_price, rec_range, rec_activity

    @staticmethod
    def interleave(
        p_tokens: torch.Tensor,
        r_tokens: torch.Tensor,
        a_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Interleaves (B, T) factor tokens into a flattened (B, 3T) sequence:
        [P_1, R_1, A_1, P_2, R_2, A_2, ..., P_T, R_T, A_T]
        """
        B, T = p_tokens.shape
        stacked = torch.stack([p_tokens, r_tokens, a_tokens], dim=2)  # (B, T, 3)
        return stacked.reshape(B, 3 * T)

    @staticmethod
    def deinterleave(
        sequence: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Deinterleaves (B, 3T) sequence back into (B, T) factor tokens.
        """
        B, seq_len = sequence.shape
        assert seq_len % 3 == 0, f"Sequence length {seq_len} must be a multiple of 3."
        T = seq_len // 3
        reshaped = sequence.view(B, T, 3)
        return reshaped[:, :, 0], reshaped[:, :, 1], reshaped[:, :, 2]
