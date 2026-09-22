"""Unit tests for RVQ and multi-stream FactorTokenizer."""

import pytest
import torch
from klint.tokenizer.rvq import VectorQuantizer, ResidualVectorQuantizer
from klint.tokenizer.factor_tokenizer import FactorTokenizer


def test_vector_quantizer_forward_and_loss():
    """Verify codebook assignment and commitment loss."""
    vq = VectorQuantizer(vocab_size=32, dim=16)
    x = torch.randn(4, 10, 16)
    quantized, indices, loss = vq(x)

    assert quantized.shape == x.shape
    assert indices.shape == (4, 10)
    assert indices.min() >= 0
    assert indices.max() < 32
    assert loss.item() >= 0.0


def test_factor_tokenizer_roundtrip():
    """Verify factor tokenizer encoding, interleaving, and decoding."""
    tokenizer = FactorTokenizer(
        price_vocab_size=512,
        range_vocab_size=256,
        activity_vocab_size=256,
        latent_dim=64,
    )

    B, T = 2, 20
    price_path = torch.randn(B, T, 2)
    range_shape = torch.randn(B, T, 3)
    activity = torch.randn(B, T, 1)

    p_tok, r_tok, a_tok, loss = tokenizer.encode(price_path, range_shape, activity)

    assert p_tok.shape == (B, T)
    assert r_tok.shape == (B, T)
    assert a_tok.shape == (B, T)

    assert p_tok.max() < 512
    assert r_tok.max() < 256
    assert a_tok.max() < 256

    # Test interleaving and de-interleaving
    interleaved = tokenizer.interleave(p_tok, r_tok, a_tok)
    assert interleaved.shape == (B, 3 * T)

    p_unp, r_unp, a_unp = tokenizer.deinterleave(interleaved)
    assert torch.equal(p_tok, p_unp)
    assert torch.equal(r_tok, r_unp)
    assert torch.equal(a_tok, a_unp)

    # Test continuous decoding
    rec_p, rec_r, rec_a = tokenizer.decode_tokens(p_tok, r_tok, a_tok)
    assert rec_p.shape == (B, T, 2)
    assert rec_r.shape == (B, T, 3)
    assert rec_a.shape == (B, T, 1)
