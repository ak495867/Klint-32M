"""Unit tests for the V2 out-of-sample multi-test suite."""

import os
import shutil
import tempfile
import pytest
import numpy as np

import sys
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("V2-tests"))

from fresh_universe import get_fresh_300_universe, TRAINING_EXCLUSION_TICKERS
from gpu_evaluator import AssetEvaluationResult
from test_battery import V2TestBattery


def test_fresh_universe_zero_leakage():
    """Verify that the fresh universe contains >= 300 assets with ZERO training leakage."""
    universe = get_fresh_300_universe(max_assets=350)
    assert len(universe) >= 300

    tickers = set(a["ticker"] for a in universe)
    overlap = tickers.intersection(TRAINING_EXCLUSION_TICKERS)
    assert len(overlap) == 0, f"Critical leak! Overlap with training: {overlap}"

    # Verify diversity across asset classes
    classes = set(a["asset_class"] for a in universe)
    assert len(classes) >= 5


def test_v2_test_battery_all_10_tests():
    """Verify that all 10 tests run and generate all 20 required plot files."""
    tmp_dir = tempfile.mkdtemp(prefix="v2_multitest_test_")
    try:
        np.random.seed(42)
        mock_results = {}
        for i in range(10):
            ticker = f"TEST_ASSET_{i}"
            n_bars = 120
            # Slight positive drift
            realized = np.random.normal(loc=0.0008, scale=0.015, size=n_bars)
            pred = realized * 0.5 + np.random.normal(0, 0.005, size=n_bars)
            prob_up = np.clip(0.5 + pred * 10, 0.1, 0.9)
            prob_down = 1.0 - prob_up
            signals = np.where(pred > 0, 1, -1)
            prices = 100.0 * np.cumprod(1.0 + realized)
            volumes = np.random.uniform(1000, 5000, size=n_bars)

            res = AssetEvaluationResult(
                ticker=ticker,
                asset_class="Equities" if i < 5 else "Crypto",
                name=f"Test Asset {i}",
                dates_idx=np.arange(n_bars),
                predicted_returns=pred,
                realized_returns=realized,
                prob_up=prob_up,
                prob_down=prob_down,
                signals=signals,
                prices=prices,
                volumes=volumes,
            )
            mock_results[ticker] = res

        battery = V2TestBattery(mock_results, output_dir=tmp_dir)
        test_metrics = battery.run_all_10_tests()

        # Check all 10 tests executed
        expected_tests = [
            "monte_carlo", "ic", "sharpe", "deflated_sharpe",
            "information_ratio", "shock", "placebo",
            "walk_forward", "noise", "friction"
        ]
        for t in expected_tests:
            assert t in test_metrics, f"Missing test: {t}"

        # Verify all 20 plot files exist
        all_plots = []
        for t in expected_tests:
            plots = test_metrics[t]["plots"]
            assert len(plots) == 2, f"Expected 2 plots for test {t}, got {len(plots)}"
            for p in plots:
                assert os.path.exists(p), f"Plot file not generated: {p}"
                all_plots.append(p)

        assert len(all_plots) == 20
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
