"""Stage 3: Training the Klint-32M Foundation Model with mixed precision and gradient accumulation."""

import os
import math
import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from klint.models.klint_32m import Klint32M, KlintConfig


class CachedTokenDataset(Dataset):
    """Zero-overhead dataset reading directly from pre-encoded token array."""
    def __init__(
        self,
        token_matrix: torch.Tensor,
        context_bars: int = 1024,
        stride: int = 8,
    ):
        self.context_bars = context_bars
        self.context_tokens = context_bars * 3
        self.flattened = token_matrix.view(-1).long()
        self.total_tokens = len(self.flattened)

        # Slice start indices (in steps of 3 tokens = 1 bar)
        max_start = self.total_tokens - self.context_tokens
        self.starts = list(range(0, max_start + 1, stride * 3))

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int) -> torch.Tensor:
        s = self.starts[idx]
        e = s + self.context_tokens
        return self.flattened[s:e]


def get_lr(it: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    """Cosine learning rate schedule with linear warmup."""
    if it < warmup_steps:
        return max_lr * (it + 1) / (warmup_steps + 1)
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def train_klint_32m(
    tokens_path: str = "data/sol_tokens.pt",
    save_dir: str = "checkpoints",
    max_steps: int = 10000,
    batch_size: int = 4,
    grad_accum_steps: int = 4,
    learning_rate: float = 3e-4,
    warmup_steps: int = 500,
    eval_interval: int = 250,
    save_interval: int = 500,
    context_bars: int = 1024,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    print("=" * 60)
    print(f" Klint: Stage 3 - Training Klint-32M Foundation Model on {device.upper()}")
    print("=" * 60)

    os.makedirs(save_dir, exist_ok=True)

    # 1. Load cached tokens
    if not os.path.exists(tokens_path):
        raise FileNotFoundError(f"Cached tokens {tokens_path} not found. Run scripts/cache_tokens.py first!")

    cached = torch.load(tokens_path, map_location="cpu")
    token_matrix = cached["token_matrix"]  # (N, 3)
    total_bars = len(token_matrix)
    print(f"Loaded {total_bars:,} bars ({total_bars * 3:,} tokens) from {tokens_path}")

    # Chronological split with 1,440-bar embargo
    embargo = 1440
    train_end = int(total_bars * 0.85)
    val_start = train_end + embargo

    train_tokens = token_matrix[:train_end]
    val_tokens = token_matrix[val_start:]

    train_dataset = CachedTokenDataset(train_tokens, context_bars=context_bars, stride=8)
    val_dataset = CachedTokenDataset(val_tokens, context_bars=context_bars, stride=32)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    print(f"Train windows: {len(train_dataset):,} | Val windows: {len(val_dataset):,}")
    print(f"Sequence length: {context_bars} bars = {context_bars * 3} tokens")
    print(f"Effective batch size: {batch_size * grad_accum_steps} (Micro-batch={batch_size}, Accum={grad_accum_steps})")

    # 2. Model setup
    config = KlintConfig(max_seq_len=context_bars * 3)
    model = Klint32M(config).to(device)
    print(f"Trainable Parameters: {model.count_parameters():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    model.train()
    step = 0
    best_val_loss = float("inf")
    start_time = time.time()
    train_iter = iter(train_loader)

    while step < max_steps:
        optimizer.zero_grad()
        accum_loss = 0.0

        lr = get_lr(step, warmup_steps, max_steps, learning_rate, learning_rate * 0.1)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        for _ in range(grad_accum_steps):
            try:
                seq = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                seq = next(train_iter)

            seq = seq.to(device)
            inputs = seq[:, :-1]
            targets = seq[:, 1:]

            with torch.amp.autocast("cuda", enabled=(device == "cuda"), dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
                out = model(inputs, targets=targets)
                loss = out["loss"] / grad_accum_steps

            scaler.scale(loss).backward()
            accum_loss += loss.item() * grad_accum_steps

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        step += 1

        if step % 20 == 0 or step == 1:
            tokens_per_sec = (step * batch_size * grad_accum_steps * context_bars * 3) / max(time.time() - start_time, 1e-5)
            print(f"Step {step:5d}/{max_steps:5d} | Loss: {accum_loss:.4f} | LR: {lr:.2e} | Speed: {tokens_per_sec:,.0f} tok/s")

        # Evaluation
        if step % eval_interval == 0:
            model.eval()
            val_loss = 0.0
            val_batches = 0
            with torch.no_grad():
                for v_seq in val_loader:
                    v_seq = v_seq.to(device)
                    v_in = v_seq[:, :-1]
                    v_tgt = v_seq[:, 1:]
                    with torch.amp.autocast("cuda", enabled=(device == "cuda"), dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
                        v_out = model(v_in, targets=v_tgt)
                    val_loss += v_out["loss"].item()
                    val_batches += 1
                    if val_batches >= 20:  # Eval on 20 batches for speed
                        break

            avg_val_loss = val_loss / max(val_batches, 1)
            print(f"\n[EVAL] Step {step} | Validation Cross-Entropy Loss: {avg_val_loss:.4f}\n")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_path = os.path.join(save_dir, "klint_32m_best.pt")
                torch.save(
                    {
                        "step": step,
                        "model_state_dict": model.state_dict(),
                        "config": config,
                        "val_loss": best_val_loss,
                    },
                    best_path,
                )
                print(f"  --> Saved new best checkpoint to {best_path}")

            model.train()

        # Regular checkpoint
        if step % save_interval == 0:
            ckpt_path = os.path.join(save_dir, f"klint_32m_step_{step}.pt")
            torch.save(
                {
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "config": config,
                },
                ckpt_path,
            )
            print(f"Saved periodic checkpoint: {ckpt_path}")

    print("\nTraining Finished Successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens_path", type=str, default="data/sol_tokens.pt")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--eval_interval", type=int, default=250)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--context_bars", type=int, default=1024)
    args = parser.parse_args()

    train_klint_32m(
        tokens_path=args.tokens_path,
        save_dir=args.save_dir,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        context_bars=args.context_bars,
    )
