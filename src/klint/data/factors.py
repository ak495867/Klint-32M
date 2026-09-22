"""Causal factor decomposition separating price-path, range-shape, and activity streams."""

from dataclasses import dataclass
from typing import Union, Tuple, Optional
import numpy as np
import torch


@dataclass
class FactorStreams:
    """Container for the three causal factor streams."""
    price_path: Union[np.ndarray, torch.Tensor]    # (N, 2): [r_gap, r_body]
    range_shape: Union[np.ndarray, torch.Tensor]   # (N, 3): [log_range, upper_wick_ratio, lower_wick_ratio]
    activity: Union[np.ndarray, torch.Tensor]      # (N, 1): [log_rel_volume]
    anchor_price: float                            # Initial base price for exact un-normalization


class FactorDecomposer:
    """
    Decomposes raw OHLCV market bars into stationary, causal factor representations.
    
    Guarantees:
    - Zero lookahead bias: all features at bar t depend strictly on current and past bars.
    - Scale invariance: features are expressed as stationary returns and ratios.
    """
    def __init__(self, volume_ema_span: int = 60, eps: float = 1e-8):
        self.volume_ema_span = volume_ema_span
        self.eps = eps

    def decompose(
        self,
        ohlcv: Union[np.ndarray, torch.Tensor],
    ) -> FactorStreams:
        """
        Decomposes (N, 5) OHLCV data into 3 stationary factor streams.

        Args:
            ohlcv: Array or Tensor of shape (N, 5) with [Open, High, Low, Close, Volume].

        Returns:
            FactorStreams containing price_path, range_shape, and activity tensors.
        """
        is_torch = isinstance(ohlcv, torch.Tensor)
        if is_torch:
            device = ohlcv.device
            arr = ohlcv.detach().cpu().numpy().astype(np.float64)
        else:
            arr = np.asarray(ohlcv, dtype=np.float64)

        N = len(arr)
        if N == 0:
            raise ValueError("Input OHLCV array is empty.")

        open_p = arr[:, 0]
        high_p = arr[:, 1]
        low_p = arr[:, 2]
        close_p = arr[:, 3]
        volume = arr[:, 4]

        # 1. Price-Path Stream: [r_gap, r_body]
        # r_gap: ln(Open_t / Close_{t-1}), for t=0: r_gap = 0.0
        prev_close = np.roll(close_p, 1)
        prev_close[0] = open_p[0]
        r_gap = np.log(np.maximum(open_p, self.eps) / np.maximum(prev_close, self.eps))
        r_body = np.log(np.maximum(close_p, self.eps) / np.maximum(open_p, self.eps))
        price_path = np.stack([r_gap, r_body], axis=-1)

        # 2. Range-Shape Stream: [log_range, upper_wick_ratio, lower_wick_ratio]
        log_range = np.log(np.maximum(high_p, self.eps) / np.maximum(low_p, self.eps))
        full_range = high_p - low_p + self.eps
        max_oc = np.maximum(open_p, close_p)
        min_oc = np.minimum(open_p, close_p)

        upper_wick_ratio = np.clip((high_p - max_oc) / full_range, 0.0, 1.0)
        lower_wick_ratio = np.clip((min_oc - low_p) / full_range, 0.0, 1.0)
        range_shape = np.stack([log_range, upper_wick_ratio, lower_wick_ratio], axis=-1)

        # 3. Activity Stream: [log_relative_volume]
        # Causal EMA of volume
        alpha = 2.0 / (self.volume_ema_span + 1.0)
        vol_ema = np.zeros_like(volume)
        current_ema = volume[0]
        for t in range(N):
            current_ema = alpha * volume[t] + (1.0 - alpha) * current_ema
            vol_ema[t] = current_ema

        log_rel_vol = np.log(1.0 + volume) - np.log(1.0 + vol_ema)
        activity = log_rel_vol[:, None]

        anchor_price = float(open_p[0])

        if is_torch:
            return FactorStreams(
                price_path=torch.from_numpy(price_path).float().to(device),
                range_shape=torch.from_numpy(range_shape).float().to(device),
                activity=torch.from_numpy(activity).float().to(device),
                anchor_price=anchor_price,
            )

        return FactorStreams(
            price_path=price_path.astype(np.float32),
            range_shape=range_shape.astype(np.float32),
            activity=activity.astype(np.float32),
            anchor_price=anchor_price,
        )

    def reconstruct(
        self,
        factors: FactorStreams,
        baseline_volume: float = 100.0,
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Reconstructs OHLCV array from FactorStreams, preserving candle invariants.
        """
        is_torch = isinstance(factors.price_path, torch.Tensor)
        if is_torch:
            p = factors.price_path.detach().cpu().numpy()
            r = factors.range_shape.detach().cpu().numpy()
            a = factors.activity.detach().cpu().numpy()
        else:
            p = factors.price_path
            r = factors.range_shape
            a = factors.activity

        N = len(p)
        ohlcv = np.zeros((N, 5), dtype=np.float64)
        current_close = factors.anchor_price

        for t in range(N):
            r_gap, r_body = p[t, 0], p[t, 1]
            log_range, u_ratio, l_ratio = r[t, 0], r[t, 1], r[t, 2]
            v_rel = a[t, 0]

            open_p = current_close * np.exp(r_gap)
            close_p = open_p * np.exp(r_body)

            # High and low from range geometry
            body_span = np.abs(close_p - open_p)
            implied_range = max(open_p, close_p) * (np.exp(log_range) - 1.0)
            range_val = max(body_span, implied_range)

            high_p = max(open_p, close_p) + u_ratio * range_val
            low_p = min(open_p, close_p) - l_ratio * range_val
            volume = np.expm1(np.log1p(baseline_volume) + v_rel)
            volume = max(0.0, volume)

            ohlcv[t] = [open_p, high_p, low_p, close_p, volume]
            current_close = close_p

        if is_torch:
            return torch.from_numpy(ohlcv).float().to(factors.price_path.device)
        return ohlcv.astype(np.float32)
