#!/usr/bin/env python
"""
Institutional PnL-Weighted Fine-Tuning CLI for Klint-32M v2.

Upgrades the foundation model into Klint-32M v2 across multi-asset cross-market regimes.
"""

import argparse
import os
import sys
import torch

from klint.eval.bundle_loader import get_or_create_release_bundle
from klint.finetune.multi_asset_dataset import build_finetune_dataloaders, DEFAULT_FINETUNE_TICKERS
from klint.finetune.pnl_trainer import KlintPnLTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Klint-32M v2 Institutional Fine-Tuning Engine")
    parser.add_argument("--base_bundle", type=str, default="checkpoints/klint_32m_release.pt",
                        help="Path to base release bundle (auto-downloaded from HF if missing)")
    parser.add_argument("--output_bundle", type=str, default="checkpoints/klint_32m_v2_release.pt",
                        help="Path to save the upgraded Klint-32M v2 bundle")
    parser.add_argument("--tickers", type=str, default=None,
                        help="Comma-separated list of ticker symbols (defaults to 12 cross-market assets)")
    parser.add_argument("--local_files", type=str, default="data/SOL.npy",
                        help="Comma-separated paths to local .npy market data files")
    parser.add_argument("--steps", type=int, default=500,
                        help="Total fine-tuning optimization steps")
    parser.add_argument("--eval_interval", type=int, default=50,
                        help="Evaluation frequency in steps")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Initial learning rate for AdamW")
    parser.add_argument("--min_lr", type=float, default=1e-6,
                        help="Minimum learning rate for cosine annealing")
    parser.add_argument("--lambda_pnl", type=float, default=2.0,
                        help="PnL loss weighting factor lambda")
    parser.add_argument("--gamma_dir", type=float, default=1.0,
                        help="Directional hinge penalty factor gamma")
    parser.add_argument("--context_bars", type=int, default=256,
                        help="Sliding window context size in bars")
    parser.add_argument("--stride_bars", type=int, default=64,
                        help="Sliding window stride in bars")
    parser.add_argument("--period", type=str, default="2y",
                        help="Historical data period from yfinance")
    parser.add_argument("--interval", type=str, default="1h",
                        help="Bar resolution from yfinance (1h, 1d, 15m)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Hardware execution device")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 75)
    print("  KLINT-32M v2 INSTITUTIONAL CROSS-MARKET FINE-TUNING")
    print("=" * 75)
    print(f"Device: {args.device}")
    print(f"Base bundle: {args.base_bundle}")
    print(f"Output bundle: {args.output_bundle}")

    # 1. Warm-start: Load base bundle (or auto-package / download from HF)
    model, tokenizer, _, _, meta = get_or_create_release_bundle(
        bundle_path=args.base_bundle,
        device=args.device,
    )
    print(f"Loaded base model: {model.count_parameters():,} parameters. Base Meta: {meta}")

    # 2. Parse data sources
    if args.tickers:
        tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    else:
        tickers = DEFAULT_FINETUNE_TICKERS

    local_files = [f.strip() for f in args.local_files.split(",") if f.strip() and os.path.exists(f.strip())]

    print(f"Ingesting {len(tickers)} tickers and {len(local_files)} local files...")

    # 3. Build multi-asset dataloaders
    train_loader, val_loader = build_finetune_dataloaders(
        tickers=tickers,
        local_files=local_files,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        context_bars=args.context_bars,
        stride_bars=args.stride_bars,
        period=args.period,
        interval=args.interval,
    )

    # 4. Initialize PnL Trainer
    trainer = KlintPnLTrainer(
        model=model,
        tokenizer=tokenizer,
        learning_rate=args.lr,
        min_lr=args.min_lr,
        total_steps=args.steps,
        lambda_pnl=args.lambda_pnl,
        gamma_dir=args.gamma_dir,
        device=args.device,
    )

    # Evaluate baseline performance prior to fine-tuning
    print("\nEvaluating baseline zero-shot / pre-trained performance...")
    baseline_metrics = trainer.evaluate(val_loader, max_batches=20)
    print(f"Baseline Val Loss: {baseline_metrics['val_loss_total']:.4f} | Baseline Hit Rate: {baseline_metrics['val_hit_rate']*100:.2f}%\n")

    # 5. Run fine-tuning
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        steps=args.steps,
        eval_interval=args.eval_interval,
        save_path=args.output_bundle,
    )


if __name__ == "__main__":
    main()
