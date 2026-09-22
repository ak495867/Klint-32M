"""Example demonstrating end-to-end factor decomposition, tokenization, generation, and geometric decoding."""

import os
import sys

# Ensure src directory is in sys.path regardless of working directory
sys.path.insert(0, os.path.abspath("src"))
sys.path.insert(0, os.path.abspath("../src"))
if os.path.exists("/content/Klint-32M/src"):
    sys.path.insert(0, "/content/Klint-32M/src")

import torch
import numpy as np

from klint.data.validator import validate_ohlcv
from klint.data.factors import FactorDecomposer
from klint.tokenizer.factor_tokenizer import FactorTokenizer
from klint.tokenizer.geometric_decoder import GeometricDecoder
from klint.models.klint_32m import Klint32M, KlintConfig


def main():
    print("=" * 60)
    print(" Klint: Generative Financial Trajectory Pipeline Demo")
    print("=" * 60)

    # 1. Load data
    data_path = os.path.join("data", "SOL.npy")
    if os.path.exists(data_path):
        raw = np.load(data_path, allow_pickle=True)
        ohlcv = raw[:1024, 1:6].astype(np.float64)
        print(f"Loaded {len(ohlcv)} bars from {data_path}")
    else:
        print("Using synthetic sample bars...")
        # Synthetic random walk
        N = 1024
        prices = np.cumprod(1.0 + np.random.randn(N) * 0.002) * 100.0
        opens = prices
        closes = prices * (1.0 + np.random.randn(N) * 0.001)
        highs = np.maximum(opens, closes) + np.random.exponential(0.1, size=N)
        lows = np.minimum(opens, closes) - np.random.exponential(0.1, size=N)
        volumes = np.random.exponential(500.0, size=N)
        ohlcv = np.stack([opens, highs, lows, closes, volumes], axis=1)

    # 2. Validate input invariants
    is_valid, report = validate_ohlcv(ohlcv)
    print(f"Input Data Invariant Check: valid={is_valid}, violations={report['total_violations']}")
    assert is_valid, "Input data contains invalid candles!"

    # 3. Factor Decomposition
    decomposer = FactorDecomposer()
    factors = decomposer.decompose(ohlcv)
    print("Factor Streams Extracted:")
    print(f"  Price-Path Stream:  shape={factors.price_path.shape} [r_gap, r_body]")
    print(f"  Range-Shape Stream: shape={factors.range_shape.shape} [log_range, upper_ratio, lower_ratio]")
    print(f"  Activity Stream:    shape={factors.activity.shape} [log_rel_volume]")

    # 4. Tokenizer
    tokenizer = FactorTokenizer()
    p_path = torch.from_numpy(factors.price_path).unsqueeze(0).float()
    r_shape = torch.from_numpy(factors.range_shape).unsqueeze(0).float()
    act = torch.from_numpy(factors.activity).unsqueeze(0).float()

    p_tok, r_tok, a_tok, vq_loss = tokenizer.encode(p_path, r_shape, act)
    tokens_seq = tokenizer.interleave(p_tok, r_tok, a_tok)
    print(f"Tokenized Sequence Shape: {tokens_seq.shape} (total factor tokens: {tokens_seq.size(1)})")

    # 5. Model: Klint-32M
    config = KlintConfig()
    model = Klint32M(config)
    print(f"Klint-32M Total Trainable Parameters: {model.count_parameters():,}")

    # 6. Autoregressive Generation: Sample next 10 bars
    print("\nAutoregressively generating 10 future bars...")
    # Use the last 10 bars (30 tokens) as prompt
    prompt_tokens = tokens_seq[:, -30:]
    generated_tokens = model.generate_tokens(prompt_tokens, num_bars=10, temperature=0.8, top_k=20)
    new_tokens = generated_tokens[:, 30:]  # (1, 30)
    print(f"Generated {new_tokens.size(1)} new tokens (3 tokens/bar * 10 bars)")

    # 7. Decode generated tokens back to factors
    new_p_tok, new_r_tok, new_a_tok = tokenizer.deinterleave(new_tokens)
    rec_p, rec_r, rec_a = tokenizer.decode_tokens(new_p_tok, new_r_tok, new_a_tok)

    # 8. Factor-to-OHLCV Geometric Decoder
    decoder = GeometricDecoder()
    last_close = float(ohlcv[-1, 3])
    synthetic_ohlcv = decoder(rec_p, rec_r, rec_a, anchor_price=last_close)

    # 9. Verify generated invariants
    synth_arr = synthetic_ohlcv.squeeze(0).detach().numpy()
    synth_valid, synth_report = validate_ohlcv(synth_arr)
    print(f"\nGenerated Synthetic Candles Invariant Check: valid={synth_valid}")
    print(f"Violations: {synth_report['total_violations']}")
    assert synth_valid, "Generated candles breached physical geometry!"

    print("\nSample Generated 10 Synthetic Bars [Open, High, Low, Close, Volume]:")
    print(f"{'Open':>10} {'High':>10} {'Low':>10} {'Close':>10} {'Volume':>12}")
    print("-" * 56)
    for bar in synth_arr:
        print(f"{bar[0]:10.4f} {bar[1]:10.4f} {bar[2]:10.4f} {bar[3]:10.4f} {bar[4]:12.2f}")

    print("\nAll physical candle invariants strictly satisfied (High >= max(O,C), Low <= min(O,C), Vol >= 0).")
    print("=" * 60)


if __name__ == "__main__":
    main()
