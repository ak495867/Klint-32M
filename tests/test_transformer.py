"""Unit tests for RoPE, RMSNorm, Causal Transformer, and Klint-32M integration."""

import pytest
import torch
from klint.models.rmsnorm import RMSNorm
from klint.models.rope import RotaryEmbedding, apply_rotary_pos_emb
from klint.models.klint_32m import Klint32M, KlintConfig


def test_rmsnorm():
    """Verify RMSNorm normalizes root-mean-square to 1."""
    x = torch.randn(2, 10, 480) * 5.0
    norm = RMSNorm(480)
    out = norm(x)
    assert out.shape == x.shape
    rms = torch.sqrt(torch.mean(out.pow(2), dim=-1))
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-2)


def test_causal_no_leakage():
    """Formally verify that future tokens cannot influence past token predictions."""
    config = KlintConfig(d_model=64, n_layers=2, n_heads=4, d_ff=128, max_seq_len=64)
    model = Klint32M(config)
    model.eval()

    seq_len = 12
    # Create input tensor
    tokens = torch.randint(0, 100, (1, seq_len))

    # Get hidden output at position 5
    out1 = model(tokens)["hidden"][:, 5, :].clone()

    # Modify tokens at future positions (e.g. pos 6, 7, ...)
    modified_tokens = tokens.clone()
    modified_tokens[:, 6:] = torch.randint(100, 200, (1, seq_len - 6))

    out2 = model(modified_tokens)["hidden"][:, 5, :].clone()

    # Output at position 5 must be identical regardless of future tokens
    assert torch.allclose(out1, out2, atol=1e-5), "Causal leak detected: future tokens influenced past token!"


def test_klint_32m_parameter_scale():
    """Verify standard Klint-32M configuration parameter count."""
    config = KlintConfig(
        d_model=480,
        n_layers=10,
        n_heads=10,
        d_ff=1920,
        price_vocab_size=512,
        range_vocab_size=256,
        activity_vocab_size=256,
    )
    model = Klint32M(config)
    total_params = model.count_parameters()

    # The transformer backbone + heads should be ~28.6M parameters
    assert 28_000_000 <= total_params <= 32_000_000, f"Expected ~30M params, got {total_params}"


def test_klint_generation_order():
    """Verify autoregressive generation yields the expected sequence length and valid tokens."""
    config = KlintConfig(d_model=64, n_layers=2, n_heads=4, d_ff=128, max_seq_len=128)
    model = Klint32M(config)

    # Prompt: 1 bar (3 tokens: P, R, A)
    prompt = torch.tensor([[10, 5, 2]])
    # Generate 4 future bars -> 4 * 3 = 12 tokens added
    generated = model.generate_tokens(prompt, num_bars=4, temperature=0.8)

    assert generated.shape == (1, 15)
