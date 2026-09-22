"""Residual Vector Quantization (RVQ) with EMA updates and dead-code revival."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class VectorQuantizer(nn.Module):
    """
    Vector Quantizer with exponential moving average (EMA) codebook updates
    and dead-code revival to guarantee high codebook utilization.
    """
    def __init__(
        self,
        vocab_size: int,
        dim: int,
        beta: float = 0.25,
        decay: float = 0.99,
        eps: float = 1e-5,
        dead_code_threshold: int = 1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.beta = beta
        self.decay = decay
        self.eps = eps
        self.dead_code_threshold = dead_code_threshold

        # Initialize codebook with normalized random gaussian vectors
        embedding = torch.randn(vocab_size, dim)
        embedding = F.normalize(embedding, p=2, dim=-1)
        self.register_buffer("embedding", embedding)
        self.register_buffer("ema_count", torch.zeros(vocab_size))
        self.register_buffer("ema_weight", embedding.clone())

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantizes input tensor x.
        
        Args:
            x: Input tensor of shape (..., dim)
            
        Returns:
            quantized: Quantized tensor with straight-through gradient (..., dim)
            indices: Codebook indices of shape (...)
            loss: Commitment loss (scalar)
        """
        input_shape = x.shape
        flat_x = x.reshape(-1, self.dim)  # (N, D)

        # Distances: ||x - e||^2 = ||x||^2 - 2 x e^T + ||e||^2
        x_sq = torch.sum(flat_x ** 2, dim=-1, keepdim=True)
        e_sq = torch.sum(self.embedding ** 2, dim=-1)
        distances = x_sq - 2.0 * torch.matmul(flat_x, self.embedding.t()) + e_sq

        # Find closest codebook vectors
        indices = torch.argmin(distances, dim=-1)  # (N,)
        quantized = F.embedding(indices, self.embedding.detach())  # (N, D)

        # EMA codebook update during training (no gradient tracking)
        if self.training:
            with torch.no_grad():
                encodings = F.one_hot(indices, self.vocab_size).float()  # (N, K)
                counts = encodings.sum(dim=0)  # (K,)
                dw = torch.matmul(encodings.t(), flat_x)  # (K, D)

                self.ema_count.mul_(self.decay).add_(counts, alpha=1.0 - self.decay)
                self.ema_weight.mul_(self.decay).add_(dw, alpha=1.0 - self.decay)

                # Laplace smoothing for counts
                n = self.ema_count.sum()
                smoothed_count = (
                    (self.ema_count + self.eps) / (n + self.vocab_size * self.eps) * n
                )

                # Update embedding
                new_embedding = self.ema_weight / smoothed_count.unsqueeze(-1)
                self.embedding.copy_(new_embedding)

                # Dead-code revival
                dead_codes = self.ema_count < self.dead_code_threshold
                if dead_codes.any() and flat_x.size(0) > 0:
                    num_dead = int(dead_codes.sum().item())
                    # Sample random vectors from current batch
                    rand_indices = torch.randint(0, flat_x.size(0), (num_dead,), device=flat_x.device)
                    revived_vectors = flat_x[rand_indices].detach()
                    self.embedding[dead_codes] = revived_vectors
                    self.ema_weight[dead_codes] = revived_vectors
                    self.ema_count[dead_codes] = self.dead_code_threshold + 1.0

        # Commitment loss
        loss = self.beta * F.mse_loss(quantized.detach(), flat_x)

        # Straight-through estimator
        quantized = flat_x + (quantized - flat_x).detach()

        return quantized.view(input_shape), indices.view(input_shape[:-1]), loss


class ResidualVectorQuantizer(nn.Module):
    """
    Residual Vector Quantizer (RVQ) chaining multiple VQ stages for coarse-to-fine quantization.
    """
    def __init__(
        self,
        vocab_size: int,
        dim: int,
        num_stages: int = 1,
        beta: float = 0.25,
        decay: float = 0.99,
    ):
        super().__init__()
        self.num_stages = num_stages
        self.layers = nn.ModuleList([
            VectorQuantizer(vocab_size=vocab_size, dim=dim, beta=beta, decay=decay)
            for _ in range(num_stages)
        ])

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Tensor of shape (..., dim)
            
        Returns:
            quantized: Aggregated quantized tensor (..., dim)
            all_indices: Stage indices of shape (num_stages, ...)
            total_loss: Sum of commitment losses across stages
        """
        residual = x
        quantized_out = torch.zeros_like(x)
        indices_list = []
        total_loss = torch.tensor(0.0, device=x.device)

        for layer in self.layers:
            q_stage, idx_stage, loss_stage = layer(residual)
            quantized_out = quantized_out + q_stage
            residual = residual - q_stage
            indices_list.append(idx_stage)
            total_loss = total_loss + loss_stage

        all_indices = torch.stack(indices_list, dim=0)
        return quantized_out, all_indices, total_loss
