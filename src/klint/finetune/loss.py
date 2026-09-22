"""PnL-Weighted Cross-Entropy Loss with Asymmetric Directional Penalty for Klint-32M Fine-tuning."""

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from klint.tokenizer.factor_tokenizer import FactorTokenizer


class PnLWeightedCrossEntropyLoss(nn.Module):
    """
    Financial utility loss aligning foundation autoregressive cross-entropy with trading PnL.

    Formulation:
        L_total = (L_price + L_range + L_activity) / 3.0

        Where:
        L_price = mean( w_t * CE(logits_p, targets_p) ) + gamma_dir * L_dir
        w_t     = 1.0 + lambda_pnl * clamp(|r_realized| * 100, 0, max_weight_clip)
        L_dir   = mean( ReLU( - r_pred_expected * r_realized ) * 100 )

        L_range = mean( CE(logits_r, targets_r) )
        L_act   = mean( CE(logits_a, targets_a) )
    """
    def __init__(
        self,
        tokenizer: Optional[FactorTokenizer] = None,
        lambda_pnl: float = 2.0,
        gamma_dir: float = 1.0,
        max_weight_clip: float = 10.0,
        price_vocab_size: int = 512,
        range_vocab_size: int = 256,
        activity_vocab_size: int = 256,
    ):
        super().__init__()
        self.lambda_pnl = float(lambda_pnl)
        self.gamma_dir = float(gamma_dir)
        self.max_weight_clip = float(max_weight_clip)
        self.price_vocab_size = price_vocab_size
        self.range_vocab_size = range_vocab_size
        self.activity_vocab_size = activity_vocab_size

        if tokenizer is not None:
            self.init_codebook_returns(tokenizer)
        else:
            # Placeholder codebook returns if initialized without tokenizer
            self.register_buffer("code_returns", torch.zeros(price_vocab_size, dtype=torch.float32))

    def init_codebook_returns(self, tokenizer: FactorTokenizer):
        """Precomputes the stationary body return r_body for all 512 price codebook tokens."""
        with torch.no_grad():
            emb_p = tokenizer.rvq_price.layers[0].embedding  # (512, latent_dim)
            dec = tokenizer.dec_price(emb_p)                 # (512, 2): [r_gap, r_body]
            code_body_returns = dec[:, 1].detach().clone()   # (512,)
        self.register_buffer("code_returns", code_body_returns.float())

    def forward(
        self,
        logits_price: torch.Tensor,
        logits_range: torch.Tensor,
        logits_activity: torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes the PnL-weighted cross-entropy loss and directional penalty.

        Args:
            logits_price: (B, seq_len, price_vocab_size)
            logits_range: (B, seq_len, range_vocab_size)
            logits_activity: (B, seq_len, activity_vocab_size)
            targets: (B, seq_len) next-token targets

        Returns:
            total_loss: scalar differentiable tensor
            metrics: dict of logging metrics (hit_rate, loss_dir, loss_price, etc.)
        """
        B, seq_len = targets.shape
        device = targets.device

        # In an interleaved sequence:
        # pos % 3 == 0 (P): next target is Range (type 1)
        # pos % 3 == 1 (R): next target is Activity (type 2)
        # pos % 3 == 2 (A): next target is next bar Price (type 0)
        next_factor_type = (torch.arange(seq_len, device=device) + 1) % 3

        mask_to_p = (next_factor_type == 0)
        mask_to_r = (next_factor_type == 1)
        mask_to_a = (next_factor_type == 2)

        loss_price = torch.tensor(0.0, device=device)
        loss_range = torch.tensor(0.0, device=device)
        loss_act = torch.tensor(0.0, device=device)
        loss_dir = torch.tensor(0.0, device=device)
        hit_rate = 0.5
        mean_pnl_weight = 1.0

        # Ensure code_returns is on the target device
        code_returns = self.code_returns.to(device)

        # 1. Price Factor Loss (Financial utility & PnL weighting)
        if mask_to_p.any():
            lp = logits_price[:, mask_to_p].reshape(-1, self.price_vocab_size)
            tp = torch.clamp(targets[:, mask_to_p].reshape(-1), 0, self.price_vocab_size - 1)

            # Continuous realized return corresponding to target token
            r_realized = code_returns[tp]  # (N_p,)

            # Expected predicted return via softmax distribution over codebook
            probs_p = F.softmax(lp, dim=-1)  # (N_p, 512)
            r_pred = torch.sum(probs_p * code_returns, dim=-1)  # (N_p,)

            # Directional penalty (ReLU hinge loss on wrong sign)
            # If r_pred and r_realized have opposite signs, -r_pred * r_realized > 0
            dir_penalty = F.relu(-r_pred * r_realized) * 100.0
            loss_dir = torch.mean(dir_penalty)

            # Directional hit rate
            correct_sign = ((r_pred > 0) == (r_realized > 0)).float()
            hit_rate = float(torch.mean(correct_sign).item())

            # Sample weight proportional to realized price movement
            sample_weight = 1.0 + self.lambda_pnl * torch.clamp(
                torch.abs(r_realized) * 100.0, 0.0, self.max_weight_clip
            )
            mean_pnl_weight = float(torch.mean(sample_weight).item())

            # Cross-entropy with sample weighting
            ce_unreduced = F.cross_entropy(lp, tp, reduction="none")  # (N_p,)
            weighted_ce = torch.mean(sample_weight * ce_unreduced)

            loss_price = weighted_ce + self.gamma_dir * loss_dir

        # 2. Range Factor Loss
        if mask_to_r.any():
            lr = logits_range[:, mask_to_r].reshape(-1, self.range_vocab_size)
            tr = torch.clamp(targets[:, mask_to_r].reshape(-1), 0, self.range_vocab_size - 1)
            loss_range = F.cross_entropy(lr, tr)

        # 3. Activity Factor Loss
        if mask_to_a.any():
            la = logits_activity[:, mask_to_a].reshape(-1, self.activity_vocab_size)
            ta = torch.clamp(targets[:, mask_to_a].reshape(-1), 0, self.activity_vocab_size - 1)
            loss_act = F.cross_entropy(la, ta)

        total_loss = (loss_price + loss_range + loss_act) / 3.0

        metrics = {
            "loss_total": float(total_loss.item()),
            "loss_price": float(loss_price.item()),
            "loss_range": float(loss_range.item()),
            "loss_activity": float(loss_act.item()),
            "loss_dir": float(loss_dir.item()),
            "hit_rate": hit_rate,
            "mean_pnl_weight": mean_pnl_weight,
        }

        return total_loss, metrics
