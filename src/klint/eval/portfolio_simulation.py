"""Standardized Top-K portfolio simulation and transaction cost friction stress test."""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import numpy as np


@dataclass
class PortfolioFeeLevelResult:
    """Portfolio performance metrics at a specific transaction cost level."""
    fee_bps: float
    cagr_pct: float
    annualized_return_pct: float
    annualized_sharpe: float
    annualized_sortino: float
    max_drawdown_pct: float
    calmar_ratio: float
    annualized_turnover_pct: float
    win_rate_pct: float
    profit_factor: float
    information_ratio: float
    equity_curve: np.ndarray


@dataclass
class PortfolioSimulationResult:
    """Aggregated portfolio simulation results across all transaction fee regimes."""
    top_k: int
    num_assets: int
    time_periods: int
    results_by_fee: List[PortfolioFeeLevelResult]
    breakeven_fee_bps: Optional[float]
    best_sharpe_zero_cost: float
    benchmark_annualized_return_pct: float
    benchmark_sharpe: float


class PortfolioSimulator:
    """
    Simulates Top-K cross-sectional portfolios across transaction cost friction levels.
    """

    DEFAULT_FEE_GRID = [0.0, 5.0, 10.0, 25.0, 50.0, 100.0, 200.0]

    def simulate_portfolio(
        self,
        pred_returns_matrix: np.ndarray,   # (T, num_assets)
        actual_returns_matrix: np.ndarray, # (T, num_assets)
        top_k: int = 3,
        fee_grid: Optional[List[float]] = None,
    ) -> PortfolioSimulationResult:
        """
        Ranks assets cross-sectionally at each time step, forms equal-weight Top-K portfolio,
        tracks portfolio turnover, and sweeps transaction fees.
        """
        T, num_assets = pred_returns_matrix.shape
        top_k = min(top_k, num_assets)
        fees = fee_grid or self.DEFAULT_FEE_GRID

        # Benchmark: 1/N equal-weight buy-and-hold
        bm_returns = np.mean(actual_returns_matrix, axis=1)
        bm_mean = np.mean(bm_returns)
        bm_std = np.std(bm_returns) + 1e-8
        bm_sharpe = float((bm_mean / bm_std) * np.sqrt(252))
        bm_ann_ret = float(((1.0 + bm_mean) ** 252 - 1.0) * 100.0)

        # Precompute selected weights at each step
        weights = np.zeros((T, num_assets))
        for t in range(T):
            scores = pred_returns_matrix[t]
            top_indices = np.argsort(scores)[-top_k:]
            weights[t, top_indices] = 1.0 / top_k

        # Gross portfolio returns before costs
        gross_returns = np.sum(weights * actual_returns_matrix, axis=1)

        # Turnover calculation: sum(|w_t - w_{t-1}|)
        turnovers = np.zeros(T)
        turnovers[0] = 1.0  # Initial portfolio construction
        for t in range(1, T):
            turnovers[t] = np.sum(np.abs(weights[t] - weights[t-1]))
        ann_turnover = float(np.mean(turnovers) * 252 * 100.0)

        results_by_fee: List[PortfolioFeeLevelResult] = []

        for fee_bps in fees:
            fee_ratio = fee_bps / 10000.0
            # Transaction friction = turnover * fee_ratio
            net_returns = gross_returns - (turnovers * fee_ratio)

            # Cumulative equity
            equity = np.cumprod(1.0 + net_returns)
            cum_ret = float(equity[-1] - 1.0)

            # Annualized Return & CAGR
            years = max(T / 252.0, 0.05)
            cagr = float(((equity[-1]) ** (1.0 / years) - 1.0) * 100.0) if equity[-1] > 0 else -100.0
            ann_ret = float(np.mean(net_returns) * 252.0 * 100.0)

            # Risk metrics
            peaks = np.maximum.accumulate(equity)
            drawdowns = (equity - peaks) / peaks * 100.0
            mdd = float(np.min(drawdowns))
            calmar = float(cagr / (abs(mdd) + 1e-8)) if mdd < 0 else 0.0

            mean_r = np.mean(net_returns)
            std_r = np.std(net_returns) + 1e-8
            downside_std = np.std(net_returns[net_returns < 0]) + 1e-8

            sharpe = float((mean_r / std_r) * np.sqrt(252))
            sortino = float((mean_r / downside_std) * np.sqrt(252))

            # Win Rate & Profit Factor
            win_rate = float(np.mean(net_returns > 0) * 100.0)
            gains = np.sum(net_returns[net_returns > 0])
            losses = np.abs(np.sum(net_returns[net_returns < 0]))
            pf = float(gains / (losses + 1e-8))

            # Information Ratio vs 1/N Benchmark
            active_returns = net_returns - bm_returns
            track_err = np.std(active_returns) + 1e-8
            ir = float((np.mean(active_returns) / track_err) * np.sqrt(252))

            results_by_fee.append(PortfolioFeeLevelResult(
                fee_bps=fee_bps,
                cagr_pct=cagr,
                annualized_return_pct=ann_ret,
                annualized_sharpe=sharpe,
                annualized_sortino=sortino,
                max_drawdown_pct=mdd,
                calmar_ratio=calmar,
                annualized_turnover_pct=ann_turnover,
                win_rate_pct=win_rate,
                profit_factor=pf,
                information_ratio=ir,
                equity_curve=equity,
            ))

        # Solve break-even fee for Sharpe
        sharpes = [r.annualized_sharpe for r in results_by_fee]
        fees_arr = np.array(fees)
        breakeven: Optional[float] = None
        if sharpes[0] > 0:
            below_zero = np.where(np.array(sharpes) <= 0)[0]
            if len(below_zero) > 0:
                idx = below_zero[0]
                f0, f1 = fees_arr[idx - 1], fees_arr[idx]
                s0, s1 = sharpes[idx - 1], sharpes[idx]
                breakeven = float(f0 + (0.0 - s0) * (f1 - f0) / (s1 - s0 + 1e-8))
            else:
                breakeven = float(fees_arr[-1])
        else:
            breakeven = 0.0

        return PortfolioSimulationResult(
            top_k=top_k,
            num_assets=num_assets,
            time_periods=T,
            results_by_fee=results_by_fee,
            breakeven_fee_bps=breakeven,
            best_sharpe_zero_cost=sharpes[0],
            benchmark_annualized_return_pct=bm_ann_ret,
            benchmark_sharpe=bm_sharpe,
        )
