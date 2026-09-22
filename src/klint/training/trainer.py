"""Trainer for Klint-32M foundation model and tokenizer."""

import os
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from klint.models.klint_32m import Klint32M
from klint.tokenizer.factor_tokenizer import FactorTokenizer


class KlintTrainer:
    """
    Orchestrates training of the Klint-32M causal autoregressive foundation model.
    """
    def __init__(
        self,
        model: Klint32M,
        tokenizer: FactorTokenizer,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.01,
        grad_clip: float = 1.0,
        device: Optional[torch.device] = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.tokenizer = tokenizer.to(self.device)
        self.grad_clip = grad_clip

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """
        Executes a single optimization step on a batch of market factor windows.
        """
        self.model.train()
        self.optimizer.zero_grad()

        p_path = batch["price_path"].to(self.device)
        r_shape = batch["range_shape"].to(self.device)
        act = batch["activity"].to(self.device)

        # 1. Tokenize continuous factors into discrete tokens
        with torch.no_grad():
            p_tok, r_tok, a_tok, _ = self.tokenizer.encode(p_path, r_shape, act)
            interleaved = self.tokenizer.interleave(p_tok, r_tok, a_tok)  # (B, 3T)

        # Autoregressive sequence modeling: input tokens predict next tokens
        inputs = interleaved[:, :-1]
        targets = interleaved[:, 1:]

        out = self.model(inputs, targets=targets)
        loss = out["loss"]

        loss.backward()
        if self.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()

        return float(loss.item())

    @torch.no_grad()
    def eval_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """
        Evaluates cross-entropy validation loss on a held-out batch.
        """
        self.model.eval()

        p_path = batch["price_path"].to(self.device)
        r_shape = batch["range_shape"].to(self.device)
        act = batch["activity"].to(self.device)

        p_tok, r_tok, a_tok, _ = self.tokenizer.encode(p_path, r_shape, act)
        interleaved = self.tokenizer.interleave(p_tok, r_tok, a_tok)

        inputs = interleaved[:, :-1]
        targets = interleaved[:, 1:]

        out = self.model(inputs, targets=targets)
        return float(out["loss"].item())

    def save_checkpoint(self, path: str):
        """Saves model and tokenizer state dictionaries."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "tokenizer_state_dict": self.tokenizer.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )

    def load_checkpoint(self, path: str):
        """Loads model and tokenizer checkpoints."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.tokenizer.load_state_dict(checkpoint["tokenizer_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
