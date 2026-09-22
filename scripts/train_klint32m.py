"""Stage 3: Training the Klint-32M Foundation Model with mixed precision and gradient accumulation."""

import os
import sys

# Ensure src directory is in sys.path regardless of working directory
sys.path.insert(0, os.path.abspath("src"))
sys.path.insert(0, os.path.abspath("../src"))
if os.path.exists("/content/Klint-32M/src"):
    sys.path.insert(0, "/content/Klint-32M/src")

# Prevent CUDA memory fragmentation on GPUs
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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
    resume_from: Optional[str] = None,
    max_steps: int = 10000,
    batch_size: int = 8,
    grad_accum_steps: int = 4,
    learning_rate: float = 3e-4,
    warmup_steps: int = 500,
    eval_interval: int = 250,
    save_interval: int = 500,
    context_bars: int = 256,
    gradient_checkpointing: bool = False,
    compile_model: bool = False,
    use_in_vram_sampling: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    print("=" * 60)
    print(f" Klint: Stage 3 - High-Speed Training Klint-32M on {device.upper()}")
    print("=" * 60)

    os.makedirs(save_dir, exist_ok=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 1. Load cached tokens
    if not os.path.exists(tokens_path):
        raise FileNotFoundError(f"Cached tokens {tokens_path} not found. Run scripts/cache_tokens.py first!")

    cached = torch.load(tokens_path, map_location="cpu", weights_only=False)
    token_matrix = cached["token_matrix"]  # (N, 3)
    total_bars = len(token_matrix)
    print(f"Loaded {total_bars:,} bars ({total_bars * 3:,} tokens) from {tokens_path}")

    # Chronological split with 1,440-bar embargo
    embargo = 1440
    train_end = int(total_bars * 0.85)
    val_start = train_end + embargo

    train_tokens = token_matrix[:train_end].view(-1).long()
    val_tokens = token_matrix[val_start:].view(-1).long()

    seq_len = context_bars * 3
    print(f"Sequence length: {context_bars} bars = {seq_len} tokens")
    print(f"Effective batch size: {batch_size * grad_accum_steps} (Micro-batch={batch_size}, Accum={grad_accum_steps})")
    print(f"Gradient Checkpointing: {gradient_checkpointing}")

    # Move tokens directly to GPU for zero CPU-overhead tensor slicing
    if use_in_vram_sampling and device == "cuda":
        print("Using In-VRAM Direct Tensor Slicing (Zero CPU-GPU memory bus overhead)")
        train_gpu = train_tokens.to(device)
        val_gpu = val_tokens.to(device)
        max_train_start = len(train_gpu) - seq_len
        max_val_start = len(val_gpu) - seq_len
        indices_template = torch.arange(seq_len, device=device).unsqueeze(0)
    else:
        train_dataset = CachedTokenDataset(token_matrix[:train_end], context_bars=context_bars, stride=8)
        val_dataset = CachedTokenDataset(token_matrix[val_start:], context_bars=context_bars, stride=32)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
        train_iter = iter(train_loader)

    # 2. Model setup
    config = KlintConfig(max_seq_len=max(seq_len, 4096), gradient_checkpointing=gradient_checkpointing)
    model = Klint32M(config).to(device)
    print(f"Trainable Parameters: {model.count_parameters():,}")

    step = 0
    best_val_loss = float("inf")

    # Resumption support
    if resume_from:
        if os.path.exists(resume_from):
            print(f"--> Resuming weights from: {resume_from}")
            ckpt = torch.load(resume_from, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            if "step" in ckpt:
                step = ckpt["step"]
                print(f"--> Resumed starting step: {step:,}")
            if "val_loss" in ckpt and ckpt["val_loss"] is not None:
                best_val_loss = ckpt["val_loss"]
                print(f"--> Previous best validation loss: {best_val_loss:.4f}")
        else:
            print(f"Warning: Checkpoint {resume_from} not found. Starting from scratch.")

    if compile_model and hasattr(torch, "compile"):
        print("Enabling torch.compile() for fused kernels...")
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"torch.compile warning: {e}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    model.train()
    start_time = time.time()
    last_log_time = time.time()
    last_log_step = step

    while step < max_steps:
        optimizer.zero_grad()
        accum_loss = 0.0

        lr = get_lr(step, warmup_steps, max_steps, learning_rate, learning_rate * 0.1)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        for _ in range(grad_accum_steps):
            if use_in_vram_sampling and device == "cuda":
                # Pure GPU tensor slicing
                start_bars = torch.randint(0, max_train_start // 3, (batch_size,), device=device) * 3
                batch_indices = start_bars.unsqueeze(1) + indices_template
                seq = train_gpu[batch_indices]
            else:
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
            accum_loss += loss.item()

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
            val_batches = 20
            with torch.no_grad():
                for _ in range(val_batches):
                    if use_in_vram_sampling and device == "cuda":
                        v_start = torch.randint(0, max_val_start // 3, (batch_size,), device=device) * 3
                        v_indices = v_start.unsqueeze(1) + indices_template
                        v_seq = val_gpu[v_indices]
                    else:
                        try:
                            v_seq = next(val_iter)
                        except (StopIteration, NameError):
                            val_iter = iter(val_loader)
                            v_seq = next(val_iter)
                        v_seq = v_seq.to(device)

                    v_in = v_seq[:, :-1]
                    v_tgt = v_seq[:, 1:]
                    with torch.amp.autocast("cuda", enabled=(device == "cuda"), dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
                        v_out = model(v_in, targets=v_tgt)
                    val_loss += v_out["loss"].item()

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
    parser.add_argument("--resume_from", type=str, default="checkpoints\\klint_32m_best.pt", help="Path to checkpoint to resume training from")
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--eval_interval", type=int, default=200)
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--context_bars", type=int, default=256)
    parser.add_argument("--no_gradient_checkpointing", action="store_true")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile()")
    args = parser.parse_args()

    train_klint_32m(
        tokens_path=args.tokens_path,
        save_dir=args.save_dir,
        resume_from=args.resume_from,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        context_bars=args.context_bars,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        compile_model=args.compile,
    )
