"""Stage 1: Pre-training the Factor Tokenizer (RVQ) codebooks on market bars."""

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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from klint.data.validator import validate_ohlcv
from klint.data.factors import FactorDecomposer
from klint.tokenizer.factor_tokenizer import FactorTokenizer
from klint.training.metrics import compute_codebook_perplexity


def train_tokenizer(
    data_path: str = "data/SOL.npy",
    epochs: int = 10,
    batch_size: int = 64,
    window_size: int = 256,
    learning_rate: float = 1e-3,
    save_dir: str = "checkpoints",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    print("=" * 60)
    print(f" Klint: Stage 1 - Tokenizer (RVQ) Pre-training on {device.upper()}")
    print("=" * 60)

    os.makedirs(save_dir, exist_ok=True)

    # 1. Load data
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file {data_path} not found.")

    raw = np.load(data_path, allow_pickle=True)
    if raw.shape[-1] == 11:
        ohlcv = raw[:, 1:6].astype(np.float64)
    else:
        ohlcv = raw[:, :5].astype(np.float64)

    print(f"Loaded {len(ohlcv):,} bars from {data_path}")
    is_valid, report = validate_ohlcv(ohlcv[:10000])
    print(f"Sample Invariant Check: valid={is_valid}, violations={report['total_violations']}")

    # 2. Decompose factors across entire dataset
    print("Extracting stationary factor streams across all bars...")
    decomposer = FactorDecomposer()
    factors = decomposer.decompose(ohlcv)

    p_tensor = torch.from_numpy(factors.price_path).float()   # (N, 2)
    r_tensor = torch.from_numpy(factors.range_shape).float()  # (N, 3)
    a_tensor = torch.from_numpy(factors.activity).float()     # (N, 1)

    # Cut into windows of length window_size
    N = len(p_tensor)
    num_windows = N // window_size
    p_windows = p_tensor[:num_windows * window_size].view(num_windows, window_size, 2)
    r_windows = r_tensor[:num_windows * window_size].view(num_windows, window_size, 3)
    a_windows = a_tensor[:num_windows * window_size].view(num_windows, window_size, 1)

    # Chronological 80/20 train/val split
    split_idx = int(num_windows * 0.8)
    train_dataset = TensorDataset(p_windows[:split_idx], r_windows[:split_idx], a_windows[:split_idx])
    val_dataset = TensorDataset(p_windows[split_idx:], r_windows[split_idx:], a_windows[split_idx:])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"Windows: {len(train_dataset)} train, {len(val_dataset)} val (window size = {window_size} bars)")

    # 3. Model & Optimizer
    tokenizer = FactorTokenizer().to(device)
    optimizer = torch.optim.AdamW(tokenizer.parameters(), lr=learning_rate, weight_decay=1e-4)

    best_val_loss = float("inf")
    save_path = os.path.join(save_dir, "tokenizer_best.pt")

    for epoch in range(1, epochs + 1):
        tokenizer.train()
        total_loss = 0.0
        recon_loss_total = 0.0

        for batch_p, batch_r, batch_a in train_loader:
            batch_p = batch_p.to(device)
            batch_r = batch_r.to(device)
            batch_a = batch_a.to(device)

            optimizer.zero_grad()

            # Forward pass with straight-through gradient
            out = tokenizer(batch_p, batch_r, batch_a)
            rec_p, rec_r, rec_a = out["rec_price"], out["rec_range"], out["rec_activity"]
            vq_loss = out["vq_loss"]

            # Reconstruction losses
            loss_p = F.mse_loss(rec_p, batch_p)
            loss_r = F.mse_loss(rec_r, batch_r)
            loss_a = F.mse_loss(rec_a, batch_a)
            recon_loss = loss_p + loss_r + loss_a

            loss = recon_loss + vq_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            recon_loss_total += recon_loss.item()

        avg_train_loss = total_loss / len(train_loader)
        avg_recon_loss = recon_loss_total / len(train_loader)

        # Validation
        tokenizer.eval()
        val_loss = 0.0
        all_p_tok, all_r_tok, all_a_tok = [], [], []

        with torch.no_grad():
            for batch_p, batch_r, batch_a in val_loader:
                batch_p = batch_p.to(device)
                batch_r = batch_r.to(device)
                batch_a = batch_a.to(device)

                out = tokenizer(batch_p, batch_r, batch_a)
                rec_p, rec_r, rec_a = out["rec_price"], out["rec_range"], out["rec_activity"]
                vq_loss = out["vq_loss"]

                recon_loss = (
                    F.mse_loss(rec_p, batch_p) +
                    F.mse_loss(rec_r, batch_r) +
                    F.mse_loss(rec_a, batch_a)
                )
                val_loss += (recon_loss + vq_loss).item()

                all_p_tok.append(out["tokens_price"].cpu())
                all_r_tok.append(out["tokens_range"].cpu())
                all_a_tok.append(out["tokens_activity"].cpu())

        avg_val_loss = val_loss / len(val_loader)

        # Check codebook utilization on val set
        p_perp = compute_codebook_perplexity(torch.cat(all_p_tok), 512)
        r_perp = compute_codebook_perplexity(torch.cat(all_r_tok), 256)
        a_perp = compute_codebook_perplexity(torch.cat(all_a_tok), 256)

        print(
            f"Epoch {epoch:2d}/{epochs:2d} | Train Loss: {avg_train_loss:.4f} (Recon: {avg_recon_loss:.4f}) | "
            f"Val Loss: {avg_val_loss:.4f} | Codebook Util: P={p_perp['utilization_pct']:.1f}% "
            f"R={r_perp['utilization_pct']:.1f}% A={a_perp['utilization_pct']:.1f}%"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(tokenizer.state_dict(), save_path)
            print(f"  --> Saved best tokenizer checkpoint to {save_path}")

    print("\nStage 1 Tokenizer Pre-training Complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/SOL.npy")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    args = parser.parse_args()

    train_tokenizer(
        data_path=args.data_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        save_dir=args.save_dir,
    )
