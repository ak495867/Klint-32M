"""Monte Carlo simulation, block bootstrap resampling, and permutation hypothesis testing."""

from dataclasses import dataclass
from typing import Dict, Any, Tuple
import numpy as np


@dataclass
class MonteCarloResult:
    """Quantitative results from Monte Carlo simulation suite."""
    num_paths: int
    path_length: int
    observed_sharpe: float
    observed_cum_return_pct: float
    observed_max_drawdown_pct: float
    p_value_sharpe: float              # Probability observed Sharpe occurred by luck
    p_value_return: float              # Probability observed return occurred by luck
    var_95_pct: float                  # 95% Value at Risk (% loss)
    var_99_pct: float                  # 99% Value at Risk (% loss)
    cvar_95_pct: float                 # 95% Conditional VaR / Expected Shortfall
    cvar_99_pct: float                 # 99% Conditional VaR / Expected Shortfall
    terminal_equity_p05: float         # 5th percentile terminal equity
    terminal_equity_p25: float         # 25th percentile
    terminal_equity_p50: float         # Median terminal equity
    terminal_equity_p75: float         # 75th percentile
    terminal_equity_p95: float         # 95th percentile
    mdd_median_pct: float              # Median Max Drawdown across simulations
    mdd_p95_pct: float                 # 95th percentile (stress) Max Drawdown
    equity_paths_sample: np.ndarray    # Sample of simulated equity curves (e.g. 100 paths)
    percentile_curves: Dict[str, np.ndarray]  # P5, P25, P50, P75, P95 curves over time
    mdd_distribution: np.ndarray       # All simulated maximum drawdowns


class MonteCarloEngine:
    """Engine executing block bootstrap and permutation stress tests on return streams."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def run_permutation_test(
        self,
        returns: np.ndarray,
        num_permutations: int = 2000,
    ) -> Tuple[float, float, np.ndarray]:
        """
        Shuffles the sequence of returns to test the null hypothesis:
        The strategy's performance was due to random ordering rather than timing skill.
        """
        if len(returns) == 0:
            return 1.0, 1.0, np.array([0.0])

        obs_mean = np.mean(returns)
        obs_std = np.std(returns) + 1e-8
        obs_sharpe = float((obs_mean / obs_std) * np.sqrt(252))
        obs_cum_return = float(np.prod(1.0 + returns) - 1.0)

        perm_sharpes = np.empty(num_permutations)
        perm_returns = np.empty(num_permutations)

        for i in range(num_permutations):
            perm = self.rng.permutation(returns)
            p_mean = np.mean(perm)
            p_std = np.std(perm) + 1e-8
            perm_sharpes[i] = (p_mean / p_std) * np.sqrt(252)
            perm_returns[i] = np.prod(1.0 + perm) - 1.0

        p_val_sharpe = float(np.mean(perm_sharpes >= obs_sharpe))
        p_val_return = float(np.mean(perm_returns >= obs_cum_return))

        return p_val_sharpe, p_val_return, perm_sharpes

    def run_block_bootstrap(
        self,
        returns: np.ndarray,
        num_paths: int = 2000,
        block_size: int = 5,
    ) -> np.ndarray:
        """
        Generates synthetic return paths using circular block bootstrap
        to preserve short-term volatility clustering and autocorrelation.
        """
        N = len(returns)
        if N < block_size:
            return np.tile(returns, (num_paths, 1))

        num_blocks = int(np.ceil(N / block_size))
        # Valid start indices for circular blocks
        indices = np.arange(N)
        circular_indices = np.concatenate([indices, indices[:block_size - 1]])

        simulated_paths = np.empty((num_paths, N), dtype=np.float64)

        for p in range(num_paths):
            starts = self.rng.integers(0, N, size=num_blocks)
            path_idx = []
            for s in starts:
                path_idx.extend(circular_indices[s:s + block_size])
            simulated_paths[p] = returns[np.array(path_idx[:N])]

        return simulated_paths

    def run_full_stress_test(
        self,
        returns: np.ndarray,
        num_paths: int = 2000,
        block_size: int = 5,
    ) -> MonteCarloResult:
        """Runs comprehensive Monte Carlo stress testing suite on strategy returns."""
        returns = np.asarray(returns, dtype=np.float64)
        N = len(returns)

        if N == 0:
            raise ValueError("Returns array is empty.")

        # 1. Observed metrics
        obs_mean = np.mean(returns)
        obs_std = np.std(returns) + 1e-8
        obs_sharpe = float((obs_mean / obs_std) * np.sqrt(252))
        obs_equity = np.cumprod(1.0 + returns)
        obs_cum_return = float(obs_equity[-1] - 1.0) * 100.0
        obs_peaks = np.maximum.accumulate(obs_equity)
        obs_mdd = float(np.min((obs_equity - obs_peaks) / obs_peaks * 100.0))

        # 2. Permutation Hypothesis Test
        p_val_sharpe, p_val_return, _ = self.run_permutation_test(
            returns, num_permutations=min(num_paths, 2000)
        )

        # 3. Block Bootstrap Simulation
        sim_returns = self.run_block_bootstrap(returns, num_paths=num_paths, block_size=block_size)
        # Compute cumulative equity paths: shape (num_paths, N)
        equity_paths = np.cumprod(1.0 + sim_returns, axis=1)

        # 4. Drawdowns per simulated path
        sim_peaks = np.maximum.accumulate(equity_paths, axis=1)
        sim_drawdowns = (equity_paths - sim_peaks) / sim_peaks * 100.0
        mdd_dist = np.min(sim_drawdowns, axis=1)  # (num_paths,)

        # 5. Terminal equity percentiles
        terminal_equities = equity_paths[:, -1]
        p05 = float(np.percentile(terminal_equities, 5))
        p25 = float(np.percentile(terminal_equities, 25))
        p50 = float(np.percentile(terminal_equities, 50))
        p75 = float(np.percentile(terminal_equities, 75))
        p95 = float(np.percentile(terminal_equities, 95))

        # 6. VaR and CVaR (1-bar horizon)
        var_95 = float(np.abs(np.percentile(returns, 5))) * 100.0
        var_99 = float(np.abs(np.percentile(returns, 1))) * 100.0
        cvar_95 = float(np.abs(np.mean(returns[returns <= np.percentile(returns, 5)]))) * 100.0
        cvar_99 = float(np.abs(np.mean(returns[returns <= np.percentile(returns, 1)]))) * 100.0

        # 7. Time-step percentiles across trajectory
        percentile_curves = {
            "p05": np.percentile(equity_paths, 5, axis=0),
            "p25": np.percentile(equity_paths, 25, axis=0),
            "p50": np.percentile(equity_paths, 50, axis=0),
            "p75": np.percentile(equity_paths, 75, axis=0),
            "p95": np.percentile(equity_paths, 95, axis=0),
        }

        # Subsample 100 paths for visual rendering
        sample_indices = self.rng.choice(num_paths, size=min(100, num_paths), replace=False)
        sample_paths = equity_paths[sample_indices]

        return MonteCarloResult(
            num_paths=num_paths,
            path_length=N,
            observed_sharpe=obs_sharpe,
            observed_cum_return_pct=obs_cum_return,
            observed_max_drawdown_pct=obs_mdd,
            p_value_sharpe=p_val_sharpe,
            p_value_return=p_val_return,
            var_95_pct=var_95,
            var_99_pct=var_99,
            cvar_95_pct=cvar_95,
            cvar_99_pct=cvar_99,
            terminal_equity_p05=p05,
            terminal_equity_p25=p25,
            terminal_equity_p50=p50,
            terminal_equity_p75=p75,
            terminal_equity_p95=p95,
            mdd_median_pct=float(np.median(mdd_dist)),
            mdd_p95_pct=float(np.percentile(mdd_dist, 5)),  # 5th percentile of negative numbers is deep DD
            equity_paths_sample=sample_paths,
            percentile_curves=percentile_curves,
            mdd_distribution=mdd_dist,
        )
