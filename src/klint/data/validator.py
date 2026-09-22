"""Data validation module ensuring structural consistency and market invariants."""

from typing import Dict, Any, Union, Tuple
import numpy as np
import torch


class DataValidationError(Exception):
    """Raised when market data violates fundamental financial invariants."""
    pass


def validate_ohlcv(
    data: Union[np.ndarray, torch.Tensor],
    tolerance: float = 1e-6,
    raise_on_error: bool = False,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Validates that market bar data strictly conforms to financial invariants:
    1. High >= max(Open, Close) - tolerance
    2. Low <= min(Open, Close) + tolerance
    3. Open, High, Low, Close > 0
    4. Volume >= 0
    5. No NaNs or Infs

    Args:
        data: Array or Tensor with shape (N, 5) or (N, C) where cols 0:5 are O, H, L, C, V.
        tolerance: Numerical tolerance for floating point comparisons.
        raise_on_error: If True, raises DataValidationError on invariant breach.

    Returns:
        is_valid: Boolean indicating whether all invariants hold.
        report: Dictionary with violation counts and diagnostics.
    """
    if isinstance(data, torch.Tensor):
        arr = data.detach().cpu().numpy()
    else:
        arr = np.asarray(data)

    if arr.ndim < 2 or arr.shape[-1] < 5:
        err = f"Expected data with at least 5 columns (O, H, L, C, V), got shape {arr.shape}"
        if raise_on_error:
            raise DataValidationError(err)
        return False, {"error": err}

    open_p = arr[:, 0]
    high_p = arr[:, 1]
    low_p = arr[:, 2]
    close_p = arr[:, 3]
    volume = arr[:, 4]

    nan_count = int(np.isnan(arr[:, :5]).sum())
    inf_count = int(np.isinf(arr[:, :5]).sum())

    non_positive_price = int(
        (open_p <= 0).sum() + (high_p <= 0).sum() + (low_p <= 0).sum() + (close_p <= 0).sum()
    )
    negative_volume = int((volume < 0).sum())

    max_oc = np.maximum(open_p, close_p)
    min_oc = np.minimum(open_p, close_p)

    invalid_high_mask = high_p < (max_oc - tolerance)
    invalid_low_mask = low_p > (min_oc + tolerance)
    high_low_inversion = int((high_p < low_p - tolerance).sum())

    invalid_high_count = int(invalid_high_mask.sum())
    invalid_low_count = int(invalid_low_mask.sum())

    total_violations = (
        nan_count + inf_count + non_positive_price + negative_volume +
        invalid_high_count + invalid_low_count + high_low_inversion
    )

    is_valid = (total_violations == 0)

    report = {
        "is_valid": is_valid,
        "total_bars": len(arr),
        "nan_count": nan_count,
        "inf_count": inf_count,
        "non_positive_price_count": non_positive_price,
        "negative_volume_count": negative_volume,
        "invalid_high_count": invalid_high_count,
        "invalid_low_count": invalid_low_count,
        "high_low_inversion_count": high_low_inversion,
        "total_violations": total_violations,
    }

    if not is_valid and raise_on_error:
        raise DataValidationError(
            f"Market invariants breached: {invalid_high_count} high < max(O, C), "
            f"{invalid_low_count} low > min(O, C), {non_positive_price} non-positive prices, "
            f"{negative_volume} negative volumes."
        )

    return is_valid, report
