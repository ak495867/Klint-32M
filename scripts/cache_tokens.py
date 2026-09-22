"""Stage 2: Pre-encoding market bars into discrete tokens for fast Transformer training."""

import os
import sys

# Ensure src directory is in sys.path regardless of working directory
sys.path.insert(0, os.path.abspath("src"))
sys.path.insert(0, os.path.abspath("../src"))
if os.path.exists("/content/Klint-32M/src"):
    sys.path.insert(0, "/content/Klint-32M/src")

import argparse
import numpy as np
import torch

from klint.data.factors import FactorDecomposer
from klint.tokenizer.factor_tokenizer import FactorTokenizer


def cache_tokens(
    data_path: str = "data/SOL.npy",
    tokenizer_path: str = "checkpoints/tokenizer_best.pt",
    output_path: str = "data/sol_tokens.pt",
    batch_size: int = 4096,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    print("=" * 60)
    print(f" Klint: Stage 2 - Pre-encoding Market Tokens on {device.upper()}")
    print("=" * 60)

    # 1. Load data
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file {data_path} not found.")

    raw = np.load(data_path, allow_pickle=True)
    if raw.shape[-1] == 11:
        ohlcv = raw[:, 1:6].astype(np.float64)
    else:
        ohlcv = raw[:, :5].astype(np.float64)

    total_bars = len(ohlcv)
    print(f"Loaded {total_bars:,} bars from {data_path}")

    # 2. Decompose factors
    print("Extracting factor streams...")
    decomposer = FactorDecomposer()
    factors = decomposer.decompose(ohlcv)

    p_tensor = torch.from_numpy(factors.price_path).float()   # (N, 2)
    r_tensor = torch.from_numpy(factors.range_shape).float()  # (N, 3)
    a_tensor = torch.from_numpy(factors.activity).float()     # (N, 1)

    # 3. Load Tokenizer
    tokenizer = FactorTokenizer().to(device)
    if os.path.exists(tokenizer_path):
        print(f"Loading trained tokenizer weights from {tokenizer_path}...")
        tokenizer.load_state_dict(torch.load(tokenizer_path, map_location=device))
    else:
        print(f"Warning: {tokenizer_path} not found. Using initialized tokenizer codebooks.")

    tokenizer.eval()

    # 4. Batch Tokenization
    print("Encoding all bars into discrete factor codes...")
    all_p_tok, all_r_tok, all_a_tok = [], [], []

    with torch.no_grad():
        for i in range(0, total_bars, batch_size):
            end_idx = min(i + batch_size, total_bars)
            bp = p_tensor[i:end_idx].unsqueeze(0).to(device)
            br = r_tensor[i:end_idx].unsqueeze(0).to(device)
            ba = a_tensor[i:end_idx].unsqueeze(0).to(device)

            p_tok, r_tok, a_tok, _ = tokenizer.encode(bp, br, ba)

            all_p_tok.append(p_tok.squeeze(0).cpu())
            all_r_tok.append(r_tok.squeeze(0).cpu())
            all_a_tok.append(a_tok.squeeze(0).cpu())

    p_tokens = torch.cat(all_p_tok, dim=0)  # (N,)
    r_tokens = torch.cat(all_r_tok, dim=0)  # (N,)
    a_tokens = torch.cat(all_a_tok, dim=0)  # (N,)

    # Interleave into (N, 3) where cols are [P, R, A]
    token_matrix = torch.stack([p_tokens, r_tokens, a_tokens], dim=-1)  # (N, 3)
    flattened_tokens = token_matrix.view(-1)  # (3N,)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    torch.save(
        {
            "token_matrix": token_matrix,         # (N, 3)
            "flattened_tokens": flattened_tokens, # (3N,)
            "num_bars": total_bars,
            "anchor_price": factors.anchor_price,
        },
        output_path,
    )

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Successfully cached {total_bars:,} bars ({len(flattened_tokens):,} tokens) to {output_path} ({size_mb:.2f} MB)")
    print("Stage 2 Complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/SOL.npy")
    parser.add_argument("--tokenizer_path", type=str, default="checkpoints/tokenizer_best.pt")
    parser.add_argument("--output_path", type=str, default="data/sol_tokens.pt")
    parser.add_argument("--batch_size", type=int, default=8192)
    args = parser.parse_args()

    cache_tokens(
        data_path=args.data_path,
        tokenizer_path=args.tokenizer_path,
        output_path=args.output_path,
        batch_size=args.batch_size,
    )
