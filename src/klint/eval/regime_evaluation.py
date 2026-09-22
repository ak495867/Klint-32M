"""7-State market regime-conditioned evaluation engine."""

from dataclasses import dataclass
from typing import Dict, List, Any
import numpy as np

from klint.eval.core_forecasting import calc_ic_rankic


@dataclass
class RegimePerformanceMetric:
    """Evaluation metrics for a specific market regime."""
    regime: str
    sample_count: int
    ic: float
    rank_ic: float
    mae_pct: float
    directional_accuracy_pct: float
    annualized_sharpe: float


@dataclass
class RegimeConditionedResult:
    """Aggregated performance scorecard conditioned on the 7 market regimes."""
    regimes: List[RegimePerformanceMetric]
    regime_breakdown: Dict[str, RegimePerformanceMetric]


class RegimeConditionedEvaluator:
    """
    Partitions market observations into 7 canonical regimes:
    Bull, Bear, Sideways, High Volatility, Low Volatility, Crash, and Recovery.
    """

    REGIMES = [
        "Bull",
        "Bear",
        "Sideways",
        "High Volatility",
        "Low Volatility",
        "Crash",
        "Recovery",
    ]

    def segment_regimes(self, returns: np.ndarray) -> Dict[str, np.ndarray]:
        """Classifies each time step into one or more market regimes."""
        N = len(returns)
        std_r = np.std(returns) + 1e-8
        mean_r = np.mean(returns)

        # Volatility: 10-bar rolling standard deviation
        vol = np.array([np.std(returns[max(0, i - 10):i + 1]) for i in range(N)])
        vol_q20 = np.percentile(vol, 20)
        vol_q80 = np.percentile(vol, 80)

        # Drawdown from running peak
        equity = np.cumprod(1.0 + returns)
        peaks = np.maximum.accumulate(equity)
        drawdowns = (equity - peaks) / peaks

        masks = {
            "Bull": returns > (mean_r + 0.5 * std_r),
            "Bear": returns < (mean_r - 0.5 * std_r),
            "Sideways": np.abs(returns - mean_r) <= (0.5 * std_r),
            "High Volatility": vol >= vol_q80,
            "Low Volatility": vol <= vol_q20,
            "Crash": returns < (mean_r - 2.5 * std_r),
            "Recovery": np.zeros(N, dtype=bool),
        }

        # Recovery: 3 bars following a Crash
        crash_indices = np.where(masks["Crash"])[0]
        for c_idx in crash_indices:
            masks["Recovery"][c_idx + 1:min(N, c_idx + 4)] = True

        return masks

    def evaluate_regimes(
        self,
        pred_returns: np.ndarray,
        actual_returns: np.ndarray,
        fee_bps: float = 5.0,
    ) -> RegimeConditionedResult:
        """Evaluates model metrics conditioned on each of the 7 market regimes."""
        N = len(actual_returns)
        masks = self.segment_regimes(actual_returns)
        fee = fee_bps / 10000.0

        results: List[RegimePerformanceMetric] = []
        res_dict: Dict[str, RegimePerformanceMetric] = {}

        for regime_name in self.REGIMES:
            mask = masks[regime_name]
            count = int(np.sum(mask))

            if count < 5:
                metric = RegimePerformanceMetric(
                    regime=regime_name,
                    sample_count=count,
                    ic=0.0,
                    rank_ic=0.0,
                    mae_pct=0.0,
                    directional_accuracy_pct=50.0,
                    annualized_sharpe=0.0,
                )
            else:
                p_sub = pred_returns[mask]
                a_sub = actual_returns[mask]

                ic, rank_ic = calc_ic_rankic(p_sub, a_sub)
                mae = float(np.mean(np.abs(p_sub - a_sub)) * 100.0)

                valid = (p_sub != 0) & (a_sub != 0)
                hit = float(np.mean(np.sign(p_sub[valid]) == np.sign(a_sub[valid])) * 100.0) if np.any(valid) else 50.0

                sig = np.zeros(count)
                sig[p_sub > fee] = 1.0
                sig[p_sub < -fee] = -1.0
                strat_r = sig * a_sub - (np.abs(sig) * fee)

                m_r = np.mean(strat_r)
                s_r = np.std(strat_r) + 1e-8
                sharpe = float((m_r / s_r) * np.sqrt(252))

                metric = RegimePerformanceMetric(
                    regime=regime_name,
                    sample_count=count,
                    ic=ic,
                    rank_ic=rank_ic,
                    mae_pct=mae,
                    directional_accuracy_pct=hit,
                    annualized_sharpe=sharpe,
                )

            results.append(metric)
            res_dict[regime_name] = metric

        return RegimeConditionedResult(
            regimes=results,
            regime_breakdown=res_dict,
        )
