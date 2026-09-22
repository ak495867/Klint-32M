"""Chronologically purged & embargoed walk-forward cross-validation engine."""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import numpy as np


@dataclass
class WalkForwardFoldResult:
    """Performance evaluation for a single walk-forward window."""
    fold_idx: int
    is_bars: int
    oos_bars: int
    is_sharpe: float
    oos_sharpe: float
    is_hit_rate: float
    oos_hit_rate: float
    is_cum_return_pct: float
    oos_cum_return_pct: float
    is_max_drawdown_pct: float
    oos_max_drawdown_pct: float
    efficiency_ratio: float           # OOS Sharpe / IS Sharpe
    oos_equity_curve: np.ndarray
    oos_returns: np.ndarray


@dataclass
class WalkForwardResult:
    """Aggregated walk-forward cross-validation performance scorecard."""
    num_folds: int
    folds: List[WalkForwardFoldResult]
    mean_is_sharpe: float
    mean_oos_sharpe: float
    mean_is_hit_rate: float
    mean_oos_hit_rate: float
    walk_forward_efficiency_ratio: float   # Mean OOS Sharpe / Mean IS Sharpe
    consistency_rate_pct: float            # % of folds where OOS Sharpe > 0
    stitched_oos_cum_return_pct: float     # Total return stitching all OOS periods
    stitched_oos_sharpe: float             # Sharpe ratio of stitched OOS return stream
    stitched_oos_max_drawdown_pct: float
    stitched_oos_equity_curve: np.ndarray


class WalkForwardEngine:
    """
    Executes purged and embargoed rolling walk-forward cross-validation.
    """

    def __init__(self, embargo_bars: int = 20):
        self.embargo_bars = embargo_bars

    def compute_metrics(
        self,
        pred_returns: np.ndarray,
        actual_returns: np.ndarray,
        fee_bps: float = 5.0,
    ) -> Tuple[float, float, float, float, np.ndarray, np.ndarray]:
        """Calculates (Sharpe, Hit Rate %, Cumulative Return %, Max Drawdown %, equity, strat_returns)."""
        fee = fee_bps / 10000.0
        N = len(pred_returns)
        if N == 0:
            return 0.0, 0.0, 0.0, 0.0, np.array([1.0]), np.array([0.0])

        signals = np.zeros(N)
        signals[pred_returns > fee] = 1.0
        signals[pred_returns < -fee] = -1.0

        strat_returns = signals * actual_returns - (np.abs(signals) * fee)

        # Hit rate
        valid = (pred_returns != 0) & (actual_returns != 0)
        if np.any(valid):
            hit_rate = float(np.mean(np.sign(pred_returns[valid]) == np.sign(actual_returns[valid])) * 100.0)
        else:
            hit_rate = 50.0

        # Equity & DD
        equity = np.cumprod(1.0 + strat_returns)
        cum_ret = float(equity[-1] - 1.0) * 100.0
        peaks = np.maximum.accumulate(equity)
        drawdowns = (equity - peaks) / peaks * 100.0
        mdd = float(np.min(drawdowns))

        # Sharpe
        mean_r = np.mean(strat_returns)
        std_r = np.std(strat_returns) + 1e-8
        sharpe = float((mean_r / std_r) * np.sqrt(252))

        return sharpe, hit_rate, cum_ret, mdd, equity, strat_returns

    def evaluate_walk_forward(
        self,
        pred_returns: np.ndarray,
        actual_returns: np.ndarray,
        num_folds: int = 5,
        fee_bps: float = 5.0,
    ) -> WalkForwardResult:
        """
        Partitions the sequential series into K chronologically ordered rolling windows
        with embargo gaps, calculating In-Sample and Out-Of-Sample metrics for each.
        """
        N = len(pred_returns)
        if N < 50:
            raise ValueError(f"Series length ({N}) is too short for walk-forward validation.")

        fold_results: List[WalkForwardFoldResult] = []
        all_oos_returns: List[np.ndarray] = []

        # Calculate fold boundaries
        # For K folds, we divide total bars into K+1 segments:
        # Fold 1: IS = Segment 0, OOS = Segment 1
        # Fold 2: IS = Segment 0+1, OOS = Segment 2 (expanding window)
        segment_size = N // (num_folds + 1)

        for k in range(1, num_folds + 1):
            is_end = k * segment_size
            oos_start = is_end + self.embargo_bars
            oos_end = min((k + 1) * segment_size, N)

            if oos_start >= oos_end:
                # If embargo is larger than segment, adjust
                oos_start = is_end + 1

            # In-Sample segment
            is_pred = pred_returns[:is_end]
            is_act = actual_returns[:is_end]

            # Out-of-Sample segment
            oos_pred = pred_returns[oos_start:oos_end]
            oos_act = actual_returns[oos_start:oos_end]

            if len(oos_pred) == 0:
                continue

            is_sharpe, is_hit, is_ret, is_mdd, _, _ = self.compute_metrics(is_pred, is_act, fee_bps=fee_bps)
            oos_sharpe, oos_hit, oos_ret, oos_mdd, oos_eq, oos_strat_r = self.compute_metrics(oos_pred, oos_act, fee_bps=fee_bps)

            efficiency = oos_sharpe / (is_sharpe + 1e-8) if is_sharpe > 0 else (1.0 if oos_sharpe > 0 else 0.0)

            fold_results.append(WalkForwardFoldResult(
                fold_idx=k,
                is_bars=len(is_pred),
                oos_bars=len(oos_pred),
                is_sharpe=is_sharpe,
                oos_sharpe=oos_sharpe,
                is_hit_rate=is_hit,
                oos_hit_rate=oos_hit,
                is_cum_return_pct=is_ret,
                oos_cum_return_pct=oos_ret,
                is_max_drawdown_pct=is_mdd,
                oos_max_drawdown_pct=oos_mdd,
                efficiency_ratio=float(efficiency),
                oos_equity_curve=oos_eq,
                oos_returns=oos_strat_r,
            ))
            all_oos_returns.append(oos_strat_r)

        # Aggregate metrics
        mean_is_sh = float(np.mean([f.is_sharpe for f in fold_results]))
        mean_oos_sh = float(np.mean([f.oos_sharpe for f in fold_results]))
        mean_is_hit = float(np.mean([f.is_hit_rate for f in fold_results]))
        mean_oos_hit = float(np.mean([f.oos_hit_rate for f in fold_results]))

        wfer = mean_oos_sh / (mean_is_sh + 1e-8) if mean_is_sh > 0 else 0.0
        profitable_folds = sum(1 for f in fold_results if f.oos_sharpe > 0)
        consistency = (profitable_folds / len(fold_results)) * 100.0 if fold_results else 0.0

        # Stitched continuous OOS equity curve
        stitched_returns = np.concatenate(all_oos_returns) if all_oos_returns else np.array([0.0])
        stitched_eq = np.cumprod(1.0 + stitched_returns)
        stitched_ret = float(stitched_eq[-1] - 1.0) * 100.0
        stitched_peaks = np.maximum.accumulate(stitched_eq)
        stitched_mdd = float(np.min((stitched_eq - stitched_peaks) / stitched_peaks * 100.0))

        st_mean = np.mean(stitched_returns)
        st_std = np.std(stitched_returns) + 1e-8
        stitched_sh = float((st_mean / st_std) * np.sqrt(252))

        return WalkForwardResult(
            num_folds=len(fold_results),
            folds=fold_results,
            mean_is_sharpe=mean_is_sh,
            mean_oos_sharpe=mean_oos_sh,
            mean_is_hit_rate=mean_is_hit,
            mean_oos_hit_rate=mean_oos_hit,
            walk_forward_efficiency_ratio=float(wfer),
            consistency_rate_pct=consistency,
            stitched_oos_cum_return_pct=stitched_ret,
            stitched_oos_sharpe=stitched_sh,
            stitched_oos_max_drawdown_pct=stitched_mdd,
            stitched_oos_equity_curve=stitched_eq,
        )
