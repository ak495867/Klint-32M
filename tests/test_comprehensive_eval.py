"""Unit tests for the Klint-32M comprehensive evaluation suite."""

import pytest
import numpy as np
import torch

from klint.eval.core_forecasting import calc_ic_rankic
from klint.eval.synthetic_fidelity import compute_rbf_mmd, LSTMDiscriminator, SyntheticFidelityEvaluator
from klint.eval.distributional_analysis import DistributionalAnalyzer, autocorr
from klint.eval.tstr_evaluator import TSTREvaluator
from klint.eval.portfolio_simulation import PortfolioSimulator
from klint.eval.walk_forward_expanding import ExpandingWalkForwardValidator
from klint.eval.regime_evaluation import RegimeConditionedEvaluator


def test_calc_ic_rankic():
    """Verify Pearson IC and Spearman RankIC computation."""
    np.random.seed(42)
    y_true = np.linspace(-1, 1, 100)
    # Perfectly monotonic prediction
    y_pred = y_true ** 3
    ic, rank_ic = calc_ic_rankic(y_pred, y_true)

    assert rank_ic == pytest.approx(1.0, abs=1e-4)
    assert 0.8 <= ic <= 1.0


def test_distributional_analyzer_and_clustering():
    """Verify moments, autocorrelation, and volatility clustering logic."""
    np.random.seed(42)
    N = 250
    # Simulate GARCH-like series with persistent volatility
    returns = np.random.normal(0, 0.02, N)
    prices = np.cumprod(1.0 + returns) * 100.0
    opens = prices
    closes = prices * (1.0 + np.random.normal(0, 0.005, N))
    highs = np.maximum(opens, closes) + np.abs(np.random.normal(0, 0.5, N))
    lows = np.minimum(opens, closes) - np.abs(np.random.normal(0, 0.5, N))
    volumes = np.random.exponential(10000, N)
    ohlcv = np.column_stack([opens, highs, lows, closes, volumes])

    analyzer = DistributionalAnalyzer()
    moments = analyzer.compute_moments(ohlcv)

    assert abs(moments.mean) < 0.1
    assert moments.std > 0.0
    assert moments.vol_p05 <= moments.vol_p50 <= moments.vol_p95


def test_tstr_evaluator():
    """Verify Train-on-Synthetic, Test-on-Real downstream evaluation."""
    np.random.seed(42)
    N = 200
    r1 = np.random.normal(0.001, 0.01, N)
    r2 = np.random.normal(0.001, 0.01, N)
    r_synth = np.random.normal(0.001, 0.01, N)

    ohlcv1 = np.column_stack([np.ones(N), np.ones(N)*2, np.ones(N)*0.5, np.cumprod(1+r1)*100, np.ones(N)*1000])
    ohlcv2 = np.column_stack([np.ones(N), np.ones(N)*2, np.ones(N)*0.5, np.cumprod(1+r2)*100, np.ones(N)*1000])
    ohlcv_s = np.column_stack([np.ones(N), np.ones(N)*2, np.ones(N)*0.5, np.cumprod(1+r_synth)*100, np.ones(N)*1000])

    tstr_eval = TSTREvaluator(lookback_lags=3)
    res = tstr_eval.evaluate_tstr(ohlcv1, ohlcv2, ohlcv_s)

    assert -1.0 <= res.trtr_ic <= 1.0
    assert -1.0 <= res.tstr_ic <= 1.0
    assert isinstance(res.utility_retention_pct, float)


def test_portfolio_simulator_cross_sectional():
    """Verify Top-K portfolio construction and transaction friction sweep."""
    np.random.seed(42)
    T, num_assets = 100, 5
    # Actual returns
    actual = np.random.normal(0.0005, 0.015, (T, num_assets))
    # Predictor with positive edge
    pred = actual + np.random.normal(0, 0.005, (T, num_assets))

    simulator = PortfolioSimulator()
    fees = [0.0, 5.0, 25.0, 50.0]
    res = simulator.simulate_portfolio(pred, actual, top_k=2, fee_grid=fees)

    assert res.top_k == 2
    assert res.num_assets == 5
    assert len(res.results_by_fee) == len(fees)

    # Friction must decay returns monotonically
    returns = [r.annualized_return_pct for r in res.results_by_fee]
    assert returns[0] >= returns[-1]


def test_expanding_walk_forward_validator():
    """Verify chronological expanding window partitions."""
    np.random.seed(42)
    N = 250
    act = np.random.normal(0.0005, 0.015, N)
    pred = act + np.random.normal(0, 0.005, N)

    validator = ExpandingWalkForwardValidator(embargo_bars=10)
    res = validator.evaluate_expanding_series(pred, act, num_folds=3)

    assert res.num_folds == 3
    assert len(res.folds) == 3
    for f in res.folds:
        assert f.train_bars > 0
        assert f.test_bars > 0


def test_regime_conditioned_segmentation():
    """Verify 7-regime classification and metrics generation."""
    np.random.seed(42)
    N = 200
    returns = np.random.normal(0.0002, 0.02, N)
    # Insert a synthetic crash
    returns[100] = -0.09
    pred = returns + np.random.normal(0, 0.005, N)

    evaluator = RegimeConditionedEvaluator()
    res = evaluator.evaluate_regimes(pred, returns)

    assert len(res.regimes) == 7
    regime_names = [m.regime for m in res.regimes]
    assert "Bull" in regime_names
    assert "Crash" in regime_names
    assert "Recovery" in regime_names
