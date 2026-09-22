"""Multi-year chronological expanding window walk-forward validation."""

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import numpy as np

from klint.eval.core_forecasting import calc_ic_rankic


@dataclass
class ExpandingWindowFold:
    """Evaluation result for a single expanding walk-forward window."""
    fold_idx: int
    train_bars: int
    test_bars: int
    test_ic: float
    test_rank_ic: float
    test_hit_rate_pct: float
    test_sharpe: float
    test_cum_return_pct: float
    test_max_drawdown_pct: float


@dataclass
class ExpandingWalkForwardResult:
    """Aggregated expanding window walk-forward scorecard."""
    num_folds: int
    folds: List[ExpandingWindowFold]
    mean_test_ic: float
    mean_test_rank_ic: float
    mean_test_hit_rate: float
    mean_test_sharpe: float
    consistency_pct: float             # % folds with Sharpe > 0
    stitched_equity: np.ndarray


class ExpandingWalkForwardValidator:
    """Executes multi-year expanding window validation across historical regimes."""

    def __init__(self, embargo_bars: int = 20, fee_bps: float = 5.0):
        self.embargo = embargo_bars
        self.fee = fee_bps / 10000.0

    def evaluate_expanding_series(
        self,
        pred_returns: np.ndarray,
        actual_returns: np.ndarray,
        min_train_ratio: float = 0.4,
        num_folds: int = 5,
    ) -> ExpandingWalkForwardResult:
        """
        Partitions time-series into expanding historical training windows
        and non-overlapping forward testing windows with embargo separation.
        """
        N = len(pred_returns)
        if N < 100:
            raise ValueError(f"Series length {N} is too short for multi-year expanding validation.")

        min_train_len = int(N * min_train_ratio)
        remaining = N - min_train_len
        step = remaining // num_folds

        folds: List[ExpandingWindowFold] = []
        stitched_returns: List[float] = []

        for k in range(num_folds):
            train_end = min_train_len + k * step
            test_start = train_end + self.embargo
            test_end = min(test_start + step, N)

            if test_start >= test_end:
                continue

            sub_pred = pred_returns[test_start:test_end]
            sub_act = actual_returns[test_start:test_end]

            ic, rank_ic = calc_ic_rankic(sub_pred, sub_act)

            # Strategy returns
            sig = np.zeros(len(sub_pred))
            sig[sub_pred > self.fee] = 1.0
            sig[sub_pred < -self.fee] = -1.0
            strat_r = sig * sub_act - (np.abs(sig) * self.fee)

            valid = (sub_pred != 0) & (sub_act != 0)
            hit = float(np.mean(np.sign(sub_pred[valid]) == np.sign(sub_act[valid])) * 100.0) if np.any(valid) else 50.0

            eq = np.cumprod(1.0 + strat_r)
            cum_ret = float(eq[-1] - 1.0) * 100.0
            peaks = np.maximum.accumulate(eq)
            mdd = float(np.min((eq - peaks) / peaks * 100.0))

            mean_r = np.mean(strat_r)
            std_r = np.std(strat_r) + 1e-8
            sharpe = float((mean_r / std_r) * np.sqrt(252))

            folds.append(ExpandingWindowFold(
                fold_idx=k + 1,
                train_bars=train_end,
                test_bars=len(sub_pred),
                test_ic=ic,
                test_rank_ic=rank_ic,
                test_hit_rate_pct=hit,
                test_sharpe=sharpe,
                test_cum_return_pct=cum_ret,
                test_max_drawdown_pct=mdd,
            ))
            stitched_returns.extend(strat_r.tolist())

        stitched_eq = np.cumprod(1.0 + np.array(stitched_returns)) if stitched_returns else np.array([1.0])
        mean_ic = float(np.mean([f.test_ic for f in folds])) if folds else 0.0
        mean_rank_ic = float(np.mean([f.test_rank_ic for f in folds])) if folds else 0.0
        mean_hit = float(np.mean([f.test_hit_rate_pct for f in folds])) if folds else 50.0
        mean_sh = float(np.mean([f.test_sharpe for f in folds])) if folds else 0.0

        consistency = float(np.mean([f.test_sharpe > 0 for f in folds]) * 100.0) if folds else 0.0

        return ExpandingWalkForwardResult(
            num_folds=len(folds),
            folds=folds,
            mean_test_ic=mean_ic,
            mean_test_rank_ic=mean_rank_ic,
            mean_test_hit_rate=mean_hit,
            mean_test_sharpe=mean_sh,
            consistency_pct=consistency,
            stitched_equity=stitched_eq,
        )
