"""Institutional PnL-Weighted Fine-Tuning Engine for Klint-32M v2."""

import os
import time
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from klint.models.klint_32m import Klint32M, KlintConfig
from klint.tokenizer.factor_tokenizer import FactorTokenizer
from klint.finetune.loss import PnLWeightedCrossEntropyLoss


class KlintPnLTrainer:
    """
    Fine-tuning trainer optimizing Klint-32M with financial utility and directional hinge losses.
    
    Upgrades pre-trained models into specialized trading engines with:
    - PnL-weighted cross entropy (higher penalty on volatile bars)
    - Directional hinge penalty (penalizing opposite-sign return forecasts)
    - Warm-start learning rates and cosine annealing
    - Direct Hit Rate and utility tracking
    """
    def __init__(
        self,
        model: Klint32M,
        tokenizer: FactorTokenizer,
        criterion: Optional[PnLWeightedCrossEntropyLoss] = None,
        learning_rate: float = 1e-4,
        min_lr: float = 1e-6,
        total_steps: int = 1000,
        weight_decay: float = 0.01,
        grad_clip: float = 1.0,
        lambda_pnl: float = 2.0,
        gamma_dir: float = 1.0,
        device: Optional[str] = None,
    ):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device)
        self.tokenizer = tokenizer.to(self.device)
        self.tokenizer.eval()  # Factor tokenizer RVQ codebook is frozen during fine-tuning

        self.criterion = criterion or PnLWeightedCrossEntropyLoss(
            tokenizer=self.tokenizer,
            lambda_pnl=lambda_pnl,
            gamma_dir=gamma_dir,
        ).to(self.device)

        self.grad_clip = grad_clip
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.95),
            eps=1e-8,
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max(total_steps, 100),
            eta_min=min_lr,
        )

        self.step_count = 0
        self.best_val_loss = float("inf")
        self.best_val_hit_rate = 0.0

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Executes a single optimization step."""
        self.model.train()
        self.optimizer.zero_grad()

        inputs = batch["inputs"].to(self.device)
        targets = batch["targets"].to(self.device)

        out = self.model(inputs)
        loss, metrics = self.criterion(
            out["logits_price"],
            out["logits_range"],
            out["logits_activity"],
            targets,
        )

        loss.backward()
        if self.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

        self.optimizer.step()
        self.scheduler.step()

        self.step_count += 1
        metrics["lr"] = float(self.optimizer.param_groups[0]["lr"])
        metrics["step"] = self.step_count
        return metrics

    @torch.no_grad()
    def eval_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Runs a validation step."""
        self.model.eval()

        inputs = batch["inputs"].to(self.device)
        targets = batch["targets"].to(self.device)

        out = self.model(inputs)
        _, metrics = self.criterion(
            out["logits_price"],
            out["logits_range"],
            out["logits_activity"],
            targets,
        )
        return metrics

    def evaluate(self, val_loader: DataLoader, max_batches: Optional[int] = None) -> Dict[str, float]:
        """Evaluates model over validation dataloader."""
        self.model.eval()
        accum: Dict[str, float] = {}
        total_batches = 0

        for i, batch in enumerate(val_loader):
            if max_batches and i >= max_batches:
                break
            metrics = self.eval_step(batch)
            for k, v in metrics.items():
                accum[k] = accum.get(k, 0.0) + v
            total_batches += 1

        if total_batches == 0:
            return {"loss_total": float("nan"), "hit_rate": 0.5}

        return {f"val_{k}": v / total_batches for k, v in accum.items()}

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        steps: int = 500,
        eval_interval: int = 50,
        save_path: str = "checkpoints/klint_32m_v2_release.pt",
        log_callback: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """
        Runs the full fine-tuning loop for a specified number of gradient steps.
        """
        print("=" * 70)
        print(f"KLINT-32M v2 FINANCIAL FINE-TUNING ENGINE")
        print(f"Device: {self.device} | Total Steps: {steps} | Eval Every: {eval_interval} steps")
        print(f"PnL Lambda: {self.criterion.lambda_pnl:.1f} | Directional Gamma: {self.criterion.gamma_dir:.1f}")
        print("=" * 70)

        train_iter = iter(train_loader)
        start_time = time.time()
        history = {"step": [], "loss": [], "hit_rate": [], "val_loss": [], "val_hit_rate": []}

        for step in range(1, steps + 1):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            metrics = self.train_step(batch)

            if step % eval_interval == 0 or step == steps:
                val_metrics = self.evaluate(val_loader, max_batches=20)
                elapsed = time.time() - start_time
                steps_per_sec = step / max(elapsed, 1e-4)

                val_loss = val_metrics["val_loss_total"]
                val_hr = val_metrics["val_hit_rate"]

                print(
                    f"Step {step:4d}/{steps:4d} | "
                    f"Train Loss: {metrics['loss_total']:.3f} (Dir: {metrics['loss_dir']:.2f}) | "
                    f"Train HitRate: {metrics['hit_rate']*100:.1f}% | "
                    f"Val Loss: {val_loss:.3f} | "
                    f"Val HitRate: {val_hr*100:.1f}% | "
                    f"LR: {metrics['lr']:.2e} | "
                    f"{steps_per_sec:.1f} step/s"
                )

                history["step"].append(step)
                history["loss"].append(metrics["loss_total"])
                history["hit_rate"].append(metrics["hit_rate"])
                history["val_loss"].append(val_loss)
                history["val_hit_rate"].append(val_hr)

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_val_hit_rate = val_hr
                    self.save_release_bundle(save_path, step, val_loss, val_hr)

                if log_callback:
                    log_callback(step, metrics, val_metrics)

        print("=" * 70)
        print(f"Fine-tuning completed in {time.time() - start_time:.1f}s.")
        print(f"Best Validation Loss: {self.best_val_loss:.4f} | Best Hit Rate: {self.best_val_hit_rate*100:.2f}%")
        print(f"Upgraded release checkpoint saved at: {save_path}")
        print("=" * 70)

        return history

    def save_release_bundle(
        self,
        save_path: str,
        step: int,
        val_loss: float,
        val_hit_rate: float,
    ):
        """Packages weights and configuration into an all-in-one v2 release bundle."""
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        bundle = {
            "config": self.model.config,
            "model_state_dict": self.model.state_dict(),
            "tokenizer_state_dict": self.tokenizer.state_dict(),
            "training_meta": {
                "base_model": "Klint-32M",
                "version": "v2",
                "fine_tune_step": step,
                "val_loss": float(val_loss),
                "val_hit_rate": float(val_hit_rate),
                "lambda_pnl": float(self.criterion.lambda_pnl),
                "gamma_dir": float(self.criterion.gamma_dir),
                "model_parameters": self.model.count_parameters(),
                "author": "Akhilesh Varma (akhverm)",
                "created_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
        }
        torch.save(bundle, save_path)
