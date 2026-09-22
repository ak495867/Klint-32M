"""Unit tests for the quantitative stress testing and validation suite."""

import pytest
import numpy as np

from klint.stress_test.monte_carlo import MonteCarloEngine, MonteCarloResult
from klint.stress_test.cost_sensitivity import CostSensitivityEngine, CostSensitivityResult
from klint.stress_test.walk_forward import WalkForwardEngine, WalkForwardResult
from klint.stress_test.oneshot_generalization import OODAssetEvaluation, OneShotGeneralizationResult


def test_monte_carlo_engine():
    """Verify Monte Carlo simulation, block bootstrap, and risk percentiles."""
    np.random.seed(42)
    # Generate 150 daily returns with slight positive drift
    returns = np.random.normal(loc=0.001, scale=0.015, size=150)

    engine = MonteCarloEngine(seed=42)
    res = engine.run_full_stress_test(returns, num_paths=200, block_size=5)

    assert isinstance(res, MonteCarloResult)
    assert res.num_paths == 200
    assert res.path_length == 150

    # Test percentile ordering: P5 <= P25 <= P50 <= P75 <= P95
    assert res.terminal_equity_p05 <= res.terminal_equity_p25
    assert res.terminal_equity_p25 <= res.terminal_equity_p50
    assert res.terminal_equity_p50 <= res.terminal_equity_p75
    assert res.terminal_equity_p75 <= res.terminal_equity_p95

    # Test VaR and CVaR properties: Expected Shortfall (CVaR) >= VaR
    assert res.cvar_95_pct >= res.var_95_pct - 1e-6
    assert res.cvar_99_pct >= res.var_99_pct - 1e-6

    # Test permutation p-value bounds
    assert 0.0 <= res.p_value_sharpe <= 1.0
    assert 0.0 <= res.p_value_return <= 1.0


def test_cost_sensitivity_sweep():
    """Verify transaction fee decay monotonicity and break-even calculation."""
    np.random.seed(42)
    # Strategy with positive alpha
    actual_returns = np.random.normal(loc=0.0005, scale=0.02, size=200)
    pred_returns = actual_returns + np.random.normal(0, 0.005, size=200)

    engine = CostSensitivityEngine()
    fees = [0.0, 5.0, 10.0, 25.0, 50.0]
    res = engine.run_sweep(pred_returns, actual_returns, fee_grid=fees)

    assert isinstance(res, CostSensitivityResult)
    assert len(res.results_by_fee) == len(fees)

    # Higher fee friction must decay net returns
    cum_returns = [r.cumulative_return_pct for r in res.results_by_fee]
    assert cum_returns[0] >= cum_returns[-1]

    # Break-even fee must be positive if initial strategy is profitable
    if res.results_by_fee[0].annualized_sharpe > 0:
        assert res.breakeven_fee_bps is not None
        assert res.breakeven_fee_bps > 0.0


def test_walk_forward_purged_embargo():
    """Verify chronological partitioning, embargo separation, and efficiency ratios."""
    np.random.seed(42)
    actual_returns = np.random.normal(loc=0.0008, scale=0.018, size=300)
    pred_returns = actual_returns + np.random.normal(0, 0.004, size=300)

    engine = WalkForwardEngine(embargo_bars=15)
    res = engine.evaluate_walk_forward(pred_returns, actual_returns, num_folds=4, fee_bps=5.0)

    assert isinstance(res, WalkForwardResult)
    assert res.num_folds == 4
    assert len(res.folds) == 4

    # Verify each fold has non-zero bars and valid metrics
    for fold in res.folds:
        assert fold.is_bars > 0
        assert fold.oos_bars > 0
        assert fold.is_hit_rate >= 0.0
        assert fold.oos_hit_rate >= 0.0

    # Stitched equity curve length must equal total OOS bars across folds
    total_oos = sum(f.oos_bars for f in res.folds)
    assert len(res.stitched_oos_equity_curve) == total_oos + 1 or len(res.stitched_oos_equity_curve) == total_oos
    assert 0.0 <= res.consistency_rate_pct <= 100.0


def test_oneshot_generalization_aggregation():
    """Verify OOD generalization metrics structure and aggregation."""
    e1 = OODAssetEvaluation(
        ticker="SPY",
        name="SPDR S&P 500",
        asset_class="Equities",
        bars_count=200,
        ce_loss=2.1,
        perplexity=8.16,
        directional_accuracy=54.5,
        wasserstein_dist=0.008,
        annualized_sharpe=0.85,
        max_drawdown_pct=-4.2,
        win_rate_pct=52.0,
        invariant_validity_pct=100.0,
        pred_returns=np.array([0.01, -0.01]),
        actual_returns=np.array([0.012, -0.008]),
    )
    e2 = OODAssetEvaluation(
        ticker="GLD",
        name="SPDR Gold",
        asset_class="Commodities",
        bars_count=200,
        ce_loss=2.3,
        perplexity=9.97,
        directional_accuracy=53.0,
        wasserstein_dist=0.009,
        annualized_sharpe=0.60,
        max_drawdown_pct=-3.8,
        win_rate_pct=51.0,
        invariant_validity_pct=100.0,
        pred_returns=np.array([0.005, -0.005]),
        actual_returns=np.array([0.006, -0.004]),
    )

    evals = [e1, e2]
    mean_ppl = float(np.mean([e.perplexity for e in evals]))
    mean_acc = float(np.mean([e.directional_accuracy for e in evals]))

    res = OneShotGeneralizationResult(
        total_assets=2,
        asset_evaluations=evals,
        mean_perplexity=mean_ppl,
        mean_directional_accuracy=mean_acc,
        mean_sharpe=0.725,
        mean_wasserstein_dist=0.0085,
        asset_class_breakdown={
            "Equities": {"count": 1, "mean_accuracy": 54.5},
            "Commodities": {"count": 1, "mean_accuracy": 53.0},
        },
    )

    assert res.total_assets == 2
    assert res.mean_directional_accuracy == 53.75
    assert "Equities" in res.asset_class_breakdown
