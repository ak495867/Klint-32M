"""Unit tests for causal factor decomposition and data validation."""

import os
import pytest
import numpy as np
import torch
from klint.data.validator import validate_ohlcv, DataValidationError
from klint.data.factors import FactorDecomposer


def test_validator_detects_bad_candles():
    """Verify validator flags invalid highs and lows."""
    # Bar with high < max(open, close)
    bad_high = np.array([[100.0, 95.0, 90.0, 105.0, 10.0]])
    is_valid, report = validate_ohlcv(bad_high)
    assert is_valid is False
    assert report["invalid_high_count"] == 1

    # Bar with low > min(open, close)
    bad_low = np.array([[100.0, 110.0, 102.0, 95.0, 10.0]])
    is_valid, report = validate_ohlcv(bad_low)
    assert is_valid is False
    assert report["invalid_low_count"] == 1

    with pytest.raises(DataValidationError):
        validate_ohlcv(bad_high, raise_on_error=True)


def test_factor_decomposition_shapes_and_values():
    """Verify factor extraction outputs and round-trip reconstruction."""
    N = 100
    np.random.seed(42)
    # Generate random walk candles
    prices = np.cumprod(1.0 + np.random.randn(N) * 0.01) * 100.0
    opens = prices
    closes = prices * (1.0 + np.random.randn(N) * 0.005)
    highs = np.maximum(opens, closes) + np.abs(np.random.randn(N)) * 0.5
    lows = np.minimum(opens, closes) - np.abs(np.random.randn(N)) * 0.5
    volumes = np.random.exponential(100.0, size=N)

    ohlcv = np.stack([opens, highs, lows, closes, volumes], axis=1)

    is_valid, _ = validate_ohlcv(ohlcv)
    assert is_valid is True

    decomposer = FactorDecomposer()
    factors = decomposer.decompose(ohlcv)

    assert factors.price_path.shape == (N, 2)
    assert factors.range_shape.shape == (N, 3)
    assert factors.activity.shape == (N, 1)

    # Check wick ratios are bounded within [0, 1]
    assert np.all(factors.range_shape[:, 1] >= 0.0)
    assert np.all(factors.range_shape[:, 1] <= 1.0)
    assert np.all(factors.range_shape[:, 2] >= 0.0)
    assert np.all(factors.range_shape[:, 2] <= 1.0)

    # Test reconstruction preserves candle geometry
    rec_ohlcv = decomposer.reconstruct(factors)
    assert rec_ohlcv.shape == (N, 5)
    rec_valid, _ = validate_ohlcv(rec_ohlcv, raise_on_error=True)
    assert rec_valid is True


def test_factor_decomposition_on_real_sol_data():
    """Verify factor decomposition on real Solana 1m data if present."""
    sol_path = os.path.join("data", "SOL.npy")
    if not os.path.exists(sol_path):
        pytest.skip("data/SOL.npy not found")

    data = np.load(sol_path, allow_pickle=True)
    # Sample first 2000 bars
    ohlcv = data[:2000, 1:6].astype(np.float64)

    is_valid, report = validate_ohlcv(ohlcv)
    assert is_valid is True
    assert report["total_violations"] == 0

    decomposer = FactorDecomposer()
    factors = decomposer.decompose(ohlcv)

    assert not np.isnan(factors.price_path).any()
    assert not np.isnan(factors.range_shape).any()
    assert not np.isnan(factors.activity).any()
