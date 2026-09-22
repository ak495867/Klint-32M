"""Factor-to-OHLCV decoder structurally guaranteeing candle geometric invariants."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Optional
import numpy as np


class GeometricDecoder(nn.Module):
    """
    Decodes continuous factor streams into structurally valid OHLCV candlestick trajectories.
    
    Guarantees:
    - High >= max(Open, Close) is strictly satisfied for all inputs.
    - Low <= min(Open, Close) is strictly satisfied for all inputs.
    - Open, High, Low, Close > 0.
    - Volume >= 0.
    """
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        price_path: torch.Tensor,
        range_shape: torch.Tensor,
        activity: torch.Tensor,
        anchor_price: Union[torch.Tensor, float] = 100.0,
        baseline_volume: float = 1000.0,
    ) -> torch.Tensor:
        """
        Decodes factors into OHLCV tensor.

        Args:
            price_path: (B, T, 2) [r_gap, r_body]
            range_shape: (B, T, 3) [log_range_latent, upper_wick_latent, lower_wick_latent]
            activity: (B, T, 1) [log_vol_latent]
            anchor_price: Scalar or (B,) tensor representing price at t=0
            baseline_volume: Reference average volume

        Returns:
            ohlcv: (B, T, 5) with columns [Open, High, Low, Close, Volume]
        """
        B, T, _ = price_path.shape
        device = price_path.device

        if isinstance(anchor_price, (int, float)):
            curr_close = torch.full((B,), float(anchor_price), device=device, dtype=price_path.dtype)
        elif anchor_price.ndim == 0:
            curr_close = anchor_price.repeat(B).to(device)
        else:
            curr_close = anchor_price.to(device)

        ohlcv_bars = []

        # Clamp raw outputs to physically reasonable bounds to avoid numerical overflow
        r_gap = torch.clamp(price_path[:, :, 0], -0.3, 0.3)
        r_body = torch.clamp(price_path[:, :, 1], -0.3, 0.3)
        log_range_lat = range_shape[:, :, 0]
        u_lat = range_shape[:, :, 1]
        l_lat = range_shape[:, :, 2]
        vol_lat = torch.clamp(activity[:, :, 0], -5.0, 5.0)

        for t in range(T):
            open_t = curr_close * torch.exp(r_gap[:, t])
            close_t = open_t * torch.exp(r_body[:, t])

            max_oc = torch.maximum(open_t, close_t)
            min_oc = torch.minimum(open_t, close_t)
            body_span = torch.abs(close_t - open_t)

            # Intrabar expansion beyond the body
            extra_range = F.softplus(log_range_lat[:, t]) * 0.05 * open_t

            # Wicks are strictly non-negative offsets
            upper_wick = F.softplus(u_lat[:, t]) * extra_range
            lower_wick = F.softplus(l_lat[:, t]) * extra_range

            high_t = max_oc + upper_wick
            low_t = min_oc - lower_wick

            # Strictly guarantee low > 0
            low_t = torch.maximum(low_t, min_oc * 0.001)

            vol_t = baseline_volume * torch.exp(vol_lat[:, t])

            bar_t = torch.stack([open_t, high_t, low_t, close_t, vol_t], dim=-1)  # (B, 5)
            ohlcv_bars.append(bar_t)
            curr_close = close_t

        return torch.stack(ohlcv_bars, dim=1)  # (B, T, 5)
