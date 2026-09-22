"""Unit tests verifying structural candlestick geometric invariant guarantees."""

import pytest
import torch
import numpy as np
from klint.tokenizer.geometric_decoder import GeometricDecoder
from klint.data.validator import validate_ohlcv


def test_geometric_decoder_invariants_random():
    """Verify that random continuous factors always decode to valid candlesticks."""
    torch.manual_seed(42)
    decoder = GeometricDecoder()

    # Generate 100 batches of 50 bars with wide random factor distributions
    B, T = 16, 64
    price_path = torch.randn(B, T, 2) * 2.0
    range_shape = torch.randn(B, T, 3) * 3.0
    activity = torch.randn(B, T, 1) * 2.0
    anchor_price = torch.rand(B) * 500.0 + 10.0

    ohlcv = decoder(price_path, range_shape, activity, anchor_price=anchor_price)

    assert ohlcv.shape == (B, T, 5)
    flat_ohlcv = ohlcv.view(-1, 5)

    is_valid, report = validate_ohlcv(flat_ohlcv, raise_on_error=True)
    assert is_valid is True
    assert report["invalid_high_count"] == 0
    assert report["invalid_low_count"] == 0
    assert report["non_positive_price_count"] == 0
    assert report["negative_volume_count"] == 0


def test_geometric_decoder_extreme_values():
    """Verify invariant preservation under extreme edge-case inputs."""
    decoder = GeometricDecoder()

    # Test extreme positive, negative, and zero inputs
    for scale in [0.0, -10.0, 10.0, 100.0, -100.0]:
        price_path = torch.full((2, 10, 2), scale)
        range_shape = torch.full((2, 10, 3), scale)
        activity = torch.full((2, 10, 1), scale)

        ohlcv = decoder(price_path, range_shape, activity, anchor_price=50.0)
        is_valid, report = validate_ohlcv(ohlcv.view(-1, 5), raise_on_error=True)
        assert is_valid is True
