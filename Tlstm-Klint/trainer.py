"""
Knowledge Distillation Trainer: Klint-32M Teacher -> TLSTM Student.

Distills the 28.6M causal Transformer foundation teacher into the lightweight
0.59M Temporal LSTM student via soft probability matching (temperature scaling),
hard factor cross-entropy, PnL weighting, and directional hinge penalties.
"""

from __future__ import annotations

import os
import time
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Internal imports
try:
    from .tlstm_model import TLSTMKlint, TLSTMConfig
    from .distillation_loss import KlintDistillationLoss
    from .data_loader import FreshUniverseDataLoader
except (ImportError, ValueError):
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from tlstm_model import TLSTMKlint, TLSTMConfig
    from distillation_loss import KlintDistillationLoss
    from data_loader import FreshUniverseDataLoader

from klint.models.klint_32m import Klint32M, KlintConfig
from klint.tokenizer.factor_tokenizer import FactorTokenizer
from klint.tokenizer.geometric_decoder import GeometricDecoder
from klint.finetune.multi_asset_dataset import INSTITUTIONAL_100_TICKERS, CORE_12_TICKERS


class TokenSequenceDataset(Dataset):
    """Dataset of sliding factor token sequences and realized returns for distillation."""
    def __init__(
        self,
        token_sequences: List[torch.Tensor],
        realized_returns_list: Optional[List[torch.Tensor]] = None,
    ):
        self.sequences = token_sequences
        self.realized_returns = realized_returns_list

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seq = self.sequences[idx]
        if self.realized_returns is not None:
            ret = self.realized_returns[idx]
        else:
            ret = torch.zeros_like(seq, dtype=torch.float32)
        return seq, ret


class DistillationTrainer:
    """
    Distillation Trainer that transfers Klint-32M v2 foundation representations
    into the compact Temporal LSTM student model.
    """
    def __init__(
        self,
        teacher_checkpoint: str = "checkpoints/klint_32m_v2_release.pt",
        student_config: Optional[TLSTMConfig] = None,
        device: Optional[str] = None,
        temperature: float = 2.0,
        alpha_kd: float = 0.6,
        lambda_pnl: float = 2.0,
        gamma_dir: float = 1.0,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.student_config = student_config or TLSTMConfig()
        self.temperature = temperature
        self.alpha_kd = alpha_kd
        self.lambda_pnl = lambda_pnl
        self.gamma_dir = gamma_dir
        self.lr = lr
        self.weight_decay = weight_decay

        # 1. Load Teacher Model
        self.teacher, self.tokenizer, self.decoder, self.teacher_config = self._load_teacher(teacher_checkpoint)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        # 2. Precompute RVQ Codebook Expected Returns for Directional Alignment
        self.codebook_returns = self._precompute_price_codebook_returns()

        # 3. Instantiate Student Model
        self.student = TLSTMKlint(self.student_config).to(self.device)

        # 4. Distillation Loss Function
        self.loss_fn = KlintDistillationLoss(
            temperature=self.temperature,
            alpha_kd=self.alpha_kd,
            lambda_pnl=self.lambda_pnl,
            gamma_dir=self.gamma_dir,
            codebook_returns=self.codebook_returns,
        )

        # 5. Optimizer
        self.optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
        )

    def _load_teacher(self, checkpoint_path: str):
        """Loads teacher bundle with fallbacks and Hugging Face download."""
        if not os.path.exists(checkpoint_path):
            alt_paths = [
                os.path.join("..", checkpoint_path),
                os.path.join("/content/Klint-32M", checkpoint_path),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", checkpoint_path),
                "checkpoints/klint_32m_release.pt",
                "../checkpoints/klint_32m_release.pt",
            ]
            for alt in alt_paths:
                if os.path.exists(alt):
                    checkpoint_path = alt
                    break

        if not os.path.exists(checkpoint_path):
            try:
                from huggingface_hub import hf_hub_download
                print(f"[Trainer] Teacher {checkpoint_path} not found. Fetching from HF akhverm/Klint-32M...")
                checkpoint_path = hf_hub_download(repo_id="akhverm/Klint-32M", filename="klint_32m_v2_release.pt")
            except Exception:
                raise FileNotFoundError(f"Could not load teacher from {checkpoint_path}")

        print(f"[Trainer] Loading Klint-32M Teacher from: {checkpoint_path} to {self.device}")
        bundle = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        cfg = bundle.get("config", KlintConfig())
        teacher = Klint32M(cfg).to(self.device)
        teacher.load_state_dict(bundle["model_state_dict"])

        tokenizer = FactorTokenizer().to(self.device)
        tok_state = bundle.get("tokenizer_state_dict", bundle.get("tokenizer_state", None))
        if tok_state is not None:
            tokenizer.load_state_dict(tok_state)

        decoder = GeometricDecoder().to(self.device)
        return teacher, tokenizer, decoder, cfg

    @torch.no_grad()
    def _precompute_price_codebook_returns(self) -> torch.Tensor:
        """Extracts continuous body returns for all 512 price tokens."""
        if hasattr(self.tokenizer.rvq_price, "layers"):
            codebook_weights = self.tokenizer.rvq_price.layers[0].embedding.to(self.device)
            continuous_factors = self.tokenizer.dec_price(codebook_weights)
            return continuous_factors[:, 1]
        elif hasattr(self.tokenizer.rvq_price, "codebooks"):
            codebook_weights = self.tokenizer.rvq_price.codebooks[0].to(self.device)
            continuous_factors = self.tokenizer.dec_price(codebook_weights)
            return continuous_factors[:, 1]
        else:
            p_indices = torch.arange(512, device=self.device)
            dummy = torch.zeros_like(p_indices)
            rec_p, _, _ = self.tokenizer.decode_tokens(p_indices, dummy, dummy)
            return rec_p[:, 1]

    def prepare_training_data(
        self,
        tickers: Optional[List[str]] = None,
        context_bars: int = 64,
        stride_bars: int = 16,
        period: str = "1y",
        interval: str = "1d",
        verbose: bool = True,
    ) -> DataLoader:
        """
        Ingests multi-asset data, extracts stationary factor streams,
        tokenizes with the pre-trained RVQ tokenizer, and slices into sliding windows.
        """
        if tickers is None:
            tickers = INSTITUTIONAL_100_TICKERS

        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_cache")
        loader = FreshUniverseDataLoader(
            cache_dir=cache_dir,
            period=period,
            interval=interval,
            min_bars=context_bars + 10,
        )

        assets_meta = [{"ticker": t, "name": t, "asset_class": "Multi-Asset"} for t in tickers]
        dataset = loader.load_universe(assets_meta, verbose=verbose)

        token_windows = []
        return_windows = []
        window_tokens = context_bars * 3
        stride_tokens = stride_bars * 3

        if verbose:
            print(f"[Trainer] Tokenizing and slicing sequence windows across {len(dataset)} assets...")

        with torch.no_grad():
            for asset_data in dataset.values():
                ohlcv = asset_data["ohlcv"]
                factors = asset_data["factors"]

                f_price = torch.from_numpy(factors.price_path).float().unsqueeze(0).to(self.device)
                f_range = torch.from_numpy(factors.range_shape).float().unsqueeze(0).to(self.device)
                f_act = torch.from_numpy(factors.activity).float().unsqueeze(0).to(self.device)

                p_tok, r_tok, a_tok, _ = self.tokenizer.encode(f_price, f_range, f_act)
                all_tokens = FactorTokenizer.interleave(p_tok, r_tok, a_tok).squeeze(0).cpu()  # (3 * T,)

                # Price returns for PnL weighting
                returns = torch.from_numpy(factors.price_path[:, 1]).float().cpu()  # (T,)
                # Repeat across the 3 token positions per bar
                token_returns = returns.repeat_interleave(3)  # (3 * T,)

                total_tokens = len(all_tokens)
                for start in range(0, total_tokens - window_tokens, stride_tokens):
                    end = start + window_tokens
                    token_windows.append(all_tokens[start:end])
                    return_windows.append(token_returns[start:end])

        if verbose:
            print(f"[Trainer] Extracted {len(token_windows)} training sequence windows of length {window_tokens} tokens.")

        if not token_windows:
            raise ValueError(f"No valid sequence windows could be extracted across {len(dataset)} loaded assets.")

        ds = TokenSequenceDataset(token_windows, return_windows)
        effective_bs = min(32, len(token_windows))
        drop_last = len(token_windows) >= 32
        dataloader = DataLoader(ds, batch_size=effective_bs, shuffle=True, drop_last=drop_last)
        return dataloader

    def train_distillation(
        self,
        dataloader: DataLoader,
        epochs: int = 5,
        save_checkpoint: str = "checkpoints/tlstm_klint_distilled.pt",
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Executes knowledge distillation training loop.
        """
        start_time = time.time()
        os.makedirs(os.path.dirname(save_checkpoint) or ".", exist_ok=True)

        teacher_params = self.teacher.count_parameters()
        student_params = self.student.count_parameters()
        compression_ratio = teacher_params / student_params

        if verbose:
            print("=" * 70)
            print(f"🚀 Starting Knowledge Distillation Training: Klint-32M v2 -> TLSTM Student")
            print(f"   * Teacher Parameters: {teacher_params:,} (~28.6M)")
            print(f"   * Student Parameters: {student_params:,} (~0.59M)")
            print(f"   * Compression Ratio:  {compression_ratio:.1f}x Parameter Reduction (~98%)")
            print(f"   * Epochs:             {epochs}")
            print(f"   * Device:             {self.device}")
            print("=" * 70)

        history: Dict[str, List[float]] = {
            "total_loss": [],
            "kd_loss": [],
            "ce_loss": [],
            "dir_loss": [],
        }

        self.student.train()
        global_step = 0

        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            epoch_kd = 0.0
            epoch_ce = 0.0
            epoch_dir = 0.0
            num_batches = 0

            for batch_seq, batch_ret in dataloader:
                batch_seq = batch_seq.to(self.device)  # (B, T)
                batch_ret = batch_ret.to(self.device)  # (B, T)

                # Targets are next tokens: input is seq[:, :-1], target is seq[:, 1:]
                inputs = batch_seq[:, :-1]
                targets = batch_seq[:, 1:]
                returns_slice = batch_ret[:, 1:]

                # 1. Teacher Forward Pass (Frozen)
                with torch.no_grad():
                    t_out = self.teacher(inputs)
                    t_p = t_out["logits_price"]
                    t_r = t_out["logits_range"]
                    t_a = t_out["logits_activity"]

                # 2. Student Forward Pass
                self.optimizer.zero_grad()
                s_out = self.student(inputs)
                s_p = s_out["logits_price"]
                s_r = s_out["logits_range"]
                s_a = s_out["logits_activity"]
                s_pred_ret = s_out.get("pred_return", None)

                # 3. Compute Composite Distillation Loss
                loss_dict = self.loss_fn(
                    student_logits_p=s_p,
                    student_logits_r=s_r,
                    student_logits_a=s_a,
                    teacher_logits_p=t_p,
                    teacher_logits_r=t_r,
                    teacher_logits_a=t_a,
                    targets=targets,
                    realized_returns=returns_slice,
                    student_pred_return=s_pred_ret,
                )

                loss = loss_dict["loss"]
                loss.backward()
                nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)
                self.optimizer.step()

                epoch_loss += loss.item()
                epoch_kd += loss_dict["loss_kd"].item()
                epoch_ce += loss_dict["loss_ce"].item()
                epoch_dir += loss_dict["loss_dir"].item()
                num_batches += 1
                global_step += 1

            avg_loss = epoch_loss / max(num_batches, 1)
            avg_kd = epoch_kd / max(num_batches, 1)
            avg_ce = epoch_ce / max(num_batches, 1)
            avg_dir = epoch_dir / max(num_batches, 1)

            history["total_loss"].append(avg_loss)
            history["kd_loss"].append(avg_kd)
            history["ce_loss"].append(avg_ce)
            history["dir_loss"].append(avg_dir)

            if verbose:
                print(
                    f"[Epoch {epoch:02d}/{epochs:02d}] "
                    f"Total Loss: {avg_loss:.4f} | "
                    f"KD Loss: {avg_kd:.4f} | "
                    f"Hard CE: {avg_ce:.4f} | "
                    f"Dir Hinge: {avg_dir:.4f}"
                )

        elapsed = time.time() - start_time

        # Save Distilled Student Bundle
        bundle = {
            "model_state_dict": self.student.state_dict(),
            "config": self.student_config,
            "tokenizer_state_dict": self.tokenizer.state_dict(),
            "training_history": history,
            "teacher_info": {
                "teacher_params": teacher_params,
                "student_params": student_params,
                "compression_ratio": compression_ratio,
                "elapsed_time": elapsed,
            },
        }
        torch.save(bundle, save_checkpoint)

        if verbose:
            print("-" * 70)
            print(f"✅ Distillation completed in {elapsed:.1f}s!")
            print(f"📦 Saved distilled student model bundle to: {save_checkpoint}")
            print("-" * 70)

        return {
            "history": history,
            "elapsed_time": elapsed,
            "compression_ratio": compression_ratio,
            "checkpoint_path": save_checkpoint,
        }
