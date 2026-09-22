"""Evaluation metrics for Klint architecture validation."""

import torch
import numpy as np
from typing import Dict, Any, Union
from klint.data.validator import validate_ohlcv


def compute_candle_validity_rate(
    ohlcv: Union[np.ndarray, torch.Tensor]
) -> float:
    """
    Computes percentage of generated candles that satisfy physical market geometry:
    High >= max(Open, Close) and Low <= min(Open, Close).
    """
    if isinstance(ohlcv, torch.Tensor):
        arr = ohlcv.detach().cpu().numpy()
    else:
        arr = np.asarray(ohlcv)

    if arr.ndim == 3:
        arr = arr.reshape(-1, arr.shape[-1])

    is_valid, report = validate_ohlcv(arr, raise_on_error=False)
    if report["total_bars"] == 0:
        return 0.0

    invalids = report["invalid_high_count"] + report["invalid_low_count"]
    valid_count = max(0, report["total_bars"] - invalids)
    return float(valid_count / report["total_bars"]) * 100.0


def compute_codebook_perplexity(
    tokens: Union[np.ndarray, torch.Tensor],
    vocab_size: int,
) -> Dict[str, float]:
    """
    Computes codebook entropy, perplexity, and active code utilization rate.
    """
    if isinstance(tokens, torch.Tensor):
        arr = tokens.detach().cpu().numpy().flatten()
    else:
        arr = np.asarray(tokens).flatten()

    total_tokens = len(arr)
    if total_tokens == 0:
        return {"perplexity": 0.0, "utilization_pct": 0.0}

    counts = np.bincount(arr, minlength=vocab_size)[:vocab_size]
    active_codes = int((counts > 0).sum())
    utilization_pct = (active_codes / vocab_size) * 100.0

    probs = counts / total_tokens
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log(probs))
    perplexity = float(np.exp(entropy))

    return {
        "active_codes": active_codes,
        "vocab_size": vocab_size,
        "utilization_pct": float(utilization_pct),
        "entropy": float(entropy),
        "perplexity": perplexity,
    }


def compute_reconstruction_mae(
    real_ohlcv: Union[np.ndarray, torch.Tensor],
    pred_ohlcv: Union[np.ndarray, torch.Tensor],
) -> Dict[str, float]:
    """
    Computes Mean Absolute Error (MAE) per OHLCV channel.
    """
    if isinstance(real_ohlcv, torch.Tensor):
        y = real_ohlcv.detach().cpu().numpy()
    else:
        y = np.asarray(real_ohlcv)

    if isinstance(pred_ohlcv, torch.Tensor):
        y_hat = pred_ohlcv.detach().cpu().numpy()
    else:
        y_hat = np.asarray(pred_ohlcv)

    diff = np.abs(y - y_hat)
    return {
        "mae_open": float(np.mean(diff[..., 0])),
        "mae_high": float(np.mean(diff[..., 1])),
        "mae_low": float(np.mean(diff[..., 2])),
        "mae_close": float(np.mean(diff[..., 3])),
        "mae_volume": float(np.mean(diff[..., 4])),
        "mae_total": float(np.mean(diff)),
    }
