"""Transaction cost sensitivity sweep and friction decay analysis."""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np


@dataclass
class CostLevelResult:
    """Performance metrics at a specific fee level."""
    fee_bps: float
    fee_ratio: float
    trades_count: int
    annualized_sharpe: float
    annualized_sortino: float
    cumulative_return_pct: float
    max_drawdown_pct: float
    win_rate_pct: float
    profit_factor: float
    equity_curve: np.ndarray


@dataclass
class CostSensitivityResult:
    """Aggregated cost sensitivity analysis across fee levels."""
    fee_levels: List[float]
    results_by_fee: List[CostLevelResult]
    breakeven_fee_bps: Optional[float]      # Maximum fee before Sharpe drops below 0
    profitable_breakeven_bps: Optional[float] # Maximum fee before Cumulative Return drops below 0
    decay_rate_sharpe_per_bps: float        # d(Sharpe)/d(Fee)
    robustness_score: float                 # Score 0-100 indicating resistance to fees


class CostSensitivityEngine:
    """
    Simulates trading performance across a spectrum of fee & slippage regimes.
    """

    DEFAULT_FEES_BPS = [0.0, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0]

    def evaluate_returns_at_fee(
        self,
        pred_returns: np.ndarray,
        actual_returns: np.ndarray,
        fee_bps: float,
    ) -> CostLevelResult:
        """Evaluates strategy metrics under a specific fee level."""
        fee = fee_bps / 10000.0
        N = len(pred_returns)

        strategy_returns = np.zeros(N)
        signals = np.zeros(N)

        for i in range(N):
            exp_r = pred_returns[i]
            real_r = actual_returns[i]

            # Causal threshold signal: trade only when expected alpha exceeds friction
            sig = 0.0
            if exp_r > fee:
                sig = 1.0
            elif exp_r < -fee:
                sig = -1.0

            signals[i] = sig
            # Return after transaction friction
            strategy_returns[i] = sig * real_r - (abs(sig) * fee)

        # Performance calculations
        trades_idx = np.where(signals != 0.0)[0]
        trades_count = len(trades_idx)

        equity = np.cumprod(1.0 + strategy_returns)
        cum_ret = float(equity[-1] - 1.0) * 100.0

        peaks = np.maximum.accumulate(equity)
        drawdowns = (equity - peaks) / peaks * 100.0
        mdd = float(np.min(drawdowns))

        mean_r = np.mean(strategy_returns)
        std_r = np.std(strategy_returns) + 1e-8
        downside_std = np.std(strategy_returns[strategy_returns < 0]) + 1e-8

        sharpe = float((mean_r / std_r) * np.sqrt(252))
        sortino = float((mean_r / downside_std) * np.sqrt(252))

        if trades_count > 0:
            traded_returns = strategy_returns[trades_idx]
            win_rate = float(np.sum(traded_returns > 0) / trades_count * 100.0)
            gains = np.sum(traded_returns[traded_returns > 0])
            losses = np.abs(np.sum(traded_returns[traded_returns < 0]))
            profit_factor = float(gains / (losses + 1e-8))
        else:
            win_rate = 0.0
            profit_factor = 0.0

        return CostLevelResult(
            fee_bps=fee_bps,
            fee_ratio=fee,
            trades_count=trades_count,
            annualized_sharpe=sharpe,
            annualized_sortino=sortino,
            cumulative_return_pct=cum_ret,
            max_drawdown_pct=mdd,
            win_rate_pct=win_rate,
            profit_factor=profit_factor,
            equity_curve=equity,
        )

    def run_sweep(
        self,
        pred_returns: np.ndarray,
        actual_returns: np.ndarray,
        fee_grid: Optional[List[float]] = None,
    ) -> CostSensitivityResult:
        """Executes full fee sweep across provided or default fee grid."""
        pred_returns = np.asarray(pred_returns, dtype=np.float64)
        actual_returns = np.asarray(actual_returns, dtype=np.float64)
        fees = fee_grid or self.DEFAULT_FEES_BPS

        results: List[CostLevelResult] = []
        for fee_bps in fees:
            res = self.evaluate_returns_at_fee(pred_returns, actual_returns, fee_bps)
            results.append(res)

        sharpes = np.array([r.annualized_sharpe for r in results])
        returns = np.array([r.cumulative_return_pct for r in results])
        fees_arr = np.array(fees)

        # Calculate break-even fee for Sharpe (linear interpolation)
        breakeven_sharpe: Optional[float] = None
        if sharpes[0] > 0:
            below_zero = np.where(sharpes <= 0)[0]
            if len(below_zero) > 0:
                idx = below_zero[0]
                # Linear interpolation between idx-1 and idx
                f0, f1 = fees_arr[idx - 1], fees_arr[idx]
                s0, s1 = sharpes[idx - 1], sharpes[idx]
                if s0 != s1:
                    breakeven_sharpe = float(f0 + (0.0 - s0) * (f1 - f0) / (s1 - s0))
                else:
                    breakeven_sharpe = float(f0)
            else:
                breakeven_sharpe = float(fees_arr[-1])  # Remains profitable across entire grid
        else:
            breakeven_sharpe = 0.0

        # Calculate break-even fee for Cumulative Return
        breakeven_ret: Optional[float] = None
        if returns[0] > 0:
            below_zero_ret = np.where(returns <= 0)[0]
            if len(below_zero_ret) > 0:
                idx = below_zero_ret[0]
                f0, f1 = fees_arr[idx - 1], fees_arr[idx]
                r0, r1 = returns[idx - 1], returns[idx]
                if r0 != r1:
                    breakeven_ret = float(f0 + (0.0 - r0) * (f1 - f0) / (r1 - r0))
                else:
                    breakeven_ret = float(f0)
            else:
                breakeven_ret = float(fees_arr[-1])
        else:
            breakeven_ret = 0.0

        # Linear regression slope d(Sharpe)/d(Fee)
        if len(fees_arr) > 1:
            slope, _ = np.polyfit(fees_arr, sharpes, 1)
        else:
            slope = 0.0

        # Robustness score: ratio of breakeven fee to standard 10 bps fee, scaled 0 to 100
        # If breakeven is 20 bps, robustness = 100. If 10 bps, robustness = 50.
        robustness = float(np.clip((breakeven_sharpe / 20.0) * 100.0, 0.0, 100.0))

        return CostSensitivityResult(
            fee_levels=fees,
            results_by_fee=results,
            breakeven_fee_bps=breakeven_sharpe,
            profitable_breakeven_bps=breakeven_ret,
            decay_rate_sharpe_per_bps=float(slope),
            robustness_score=robustness,
        )
