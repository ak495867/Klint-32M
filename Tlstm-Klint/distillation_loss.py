"""
Knowledge Distillation Loss for Klint-32M Teacher -> TLSTM Student.

Combines:
  1. Soft Logit Distillation (KL Divergence with Temperature Scaling)
  2. Hard Target Cross-Entropy on Realized Factor Tokens
  3. PnL-Weighted Loss Amplification on High-Volatility Bars
  4. Directional Hinge Loss Penalizing Wrong-Way Trades
"""

from typing import Optional, Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class KlintDistillationLoss(nn.Module):
    """
    Knowledge Distillation Loss transferring foundation teacher capabilities
    into the lightweight Temporal LSTM student.
    """
    def __init__(
        self,
        temperature: float = 2.0,
        alpha_kd: float = 0.6,          # 60% teacher dark knowledge, 40% hard targets
        lambda_pnl: float = 2.0,        # PnL volatility weighting
        gamma_dir: float = 1.0,         # Directional hinge penalty
        codebook_returns: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.temperature = temperature
        self.alpha_kd = alpha_kd
        self.lambda_pnl = lambda_pnl
        self.gamma_dir = gamma_dir

        if codebook_returns is not None:
            self.register_buffer("codebook_returns", codebook_returns)
        else:
            self.codebook_returns = None

    def forward(
        self,
        student_logits_p: torch.Tensor,     # (B, T, 512)
        student_logits_r: torch.Tensor,     # (B, T, 256)
        student_logits_a: torch.Tensor,     # (B, T, 256)
        teacher_logits_p: torch.Tensor,     # (B, T, 512)
        teacher_logits_r: torch.Tensor,     # (B, T, 256)
        teacher_logits_a: torch.Tensor,     # (B, T, 256)
        targets: torch.Tensor,              # (B, T)
        realized_returns: Optional[torch.Tensor] = None,  # (B, T)
        student_pred_return: Optional[torch.Tensor] = None, # (B, T, 1)
    ) -> Dict[str, torch.Tensor]:
        """
        Computes composite distillation loss across all 3 factor streams.
        """
        B, seq_len = targets.shape
        device = targets.device

        next_factor_type = (torch.arange(seq_len, device=device) + 1) % 3
        mask_p = (next_factor_type == 0)
        mask_r = (next_factor_type == 1)
        mask_a = (next_factor_type == 2)

        # 1. PnL Weights: w_t = 1.0 + lambda_pnl * min(|r_body| * 100, 10.0)
        if realized_returns is not None and self.lambda_pnl > 0:
            pnl_weights = 1.0 + self.lambda_pnl * torch.clamp(
                torch.abs(realized_returns) * 100.0, max=10.0
            )  # (B, T)
        else:
            pnl_weights = torch.ones(B, seq_len, device=device)

        # -----------------------------------------------------------------
        # Soft Knowledge Distillation (KL Divergence)
        # KL(P_student || P_teacher) = sum P_teacher * log(P_teacher / P_student)
        # In PyTorch F.kl_div(input=log_probs_student, target=probs_teacher)
        # -----------------------------------------------------------------
        tau = self.temperature
        tau_sq = tau * tau

        loss_kd = torch.tensor(0.0, device=device)
        loss_ce = torch.tensor(0.0, device=device)
        count_factors = 0

        # Price Factor Distillation
        if mask_p.any():
            s_p = student_logits_p[:, mask_p]  # (B, N_p, 512)
            t_p = teacher_logits_p[:, mask_p]  # (B, N_p, 512)
            y_p = targets[:, mask_p]           # (B, N_p)
            w_p = pnl_weights[:, mask_p]       # (B, N_p)

            # Soft targets from teacher
            p_teacher = F.softmax(t_p / tau, dim=-1)
            log_p_student = F.log_softmax(s_p / tau, dim=-1)

            # KL divergence per element: (B, N_p)
            kl_elem = F.kl_div(log_p_student, p_teacher, reduction="none").sum(dim=-1) * tau_sq
            kd_p = (kl_elem * w_p).mean()

            # Hard cross entropy
            ce_elem = F.cross_entropy(s_p.reshape(-1, 512), y_p.reshape(-1), reduction="none").view(B, -1)
            ce_p = (ce_elem * w_p).mean()

            loss_kd = loss_kd + kd_p
            loss_ce = loss_ce + ce_p
            count_factors += 1

        # Range Factor Distillation
        if mask_r.any():
            s_r = student_logits_r[:, mask_r]
            t_r = teacher_logits_r[:, mask_r]
            y_r = targets[:, mask_r]
            w_r = pnl_weights[:, mask_r]

            p_teacher_r = F.softmax(t_r / tau, dim=-1)
            log_p_student_r = F.log_softmax(s_r / tau, dim=-1)

            kl_elem_r = F.kl_div(log_p_student_r, p_teacher_r, reduction="none").sum(dim=-1) * tau_sq
            kd_r = (kl_elem_r * w_r).mean()

            ce_elem_r = F.cross_entropy(s_r.reshape(-1, 256), y_r.reshape(-1), reduction="none").view(B, -1)
            ce_r = (ce_elem_r * w_r).mean()

            loss_kd = loss_kd + kd_r
            loss_ce = loss_ce + ce_r
            count_factors += 1

        # Activity Factor Distillation
        if mask_a.any():
            s_a = student_logits_a[:, mask_a]
            t_a = teacher_logits_a[:, mask_a]
            y_a = targets[:, mask_a]
            w_a = pnl_weights[:, mask_a]

            p_teacher_a = F.softmax(t_a / tau, dim=-1)
            log_p_student_a = F.log_softmax(s_a / tau, dim=-1)

            kl_elem_a = F.kl_div(log_p_student_a, p_teacher_a, reduction="none").sum(dim=-1) * tau_sq
            kd_a = (kl_elem_a * w_a).mean()

            ce_elem_a = F.cross_entropy(s_a.reshape(-1, 256), y_a.reshape(-1), reduction="none").view(B, -1)
            ce_a = (ce_elem_a * w_a).mean()

            loss_kd = loss_kd + kd_a
            loss_ce = loss_ce + ce_a
            count_factors += 1

        if count_factors > 0:
            loss_kd = loss_kd / count_factors
            loss_ce = loss_ce / count_factors

        # -----------------------------------------------------------------
        # Directional Hinge Loss
        # ReLU(- expected_return * realized_return) * 100
        # -----------------------------------------------------------------
        loss_dir = torch.tensor(0.0, device=device)
        if mask_p.any() and realized_returns is not None and self.gamma_dir > 0:
            s_p = student_logits_p[:, mask_p]  # (B, N_p, 512)
            r_real = realized_returns[:, mask_p] # (B, N_p)

            if self.codebook_returns is not None:
                probs = F.softmax(s_p, dim=-1)
                exp_ret = torch.matmul(probs, self.codebook_returns)  # (B, N_p)
            elif student_pred_return is not None:
                exp_ret = student_pred_return[:, mask_p].squeeze(-1)
            else:
                exp_ret = None

            if exp_ret is not None:
                hinge = F.relu(-exp_ret * r_real) * 100.0
                loss_dir = hinge.mean() * self.gamma_dir

        # Total Distillation Loss
        total_loss = (1.0 - self.alpha_kd) * loss_ce + self.alpha_kd * loss_kd + loss_dir

        return {
            "loss": total_loss,
            "loss_kd": loss_kd,
            "loss_ce": loss_ce,
            "loss_dir": loss_dir,
        }
