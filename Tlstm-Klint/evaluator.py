"""
GPU-Accelerated Evaluation Engine for Distilled TLSTM Student Across 300+ Fresh Assets.

Computes expected returns, directional probabilities, and trading signals with
high-throughput GPU batching, zero data leakage, and strict chronological integrity.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn.functional as F

# Flexible imports
try:
    from .tlstm_model import TLSTMKlint, TLSTMConfig
except (ImportError, ValueError):
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from tlstm_model import TLSTMKlint, TLSTMConfig

from klint.tokenizer.factor_tokenizer import FactorTokenizer
from klint.tokenizer.geometric_decoder import GeometricDecoder


@dataclass
class AssetEvaluationResult:
    """Holds per-asset evaluation time-series and trading performance."""
    ticker: str
    name: str
    asset_class: str
    realized_returns: np.ndarray      # Ground truth bar body return
    predicted_returns: np.ndarray     # Student expected return sum_k P(k)*r_body(k)
    signals: np.ndarray               # Position signal {-1, 0, 1}
    prob_up: np.ndarray               # P(return > 0)
    prob_down: np.ndarray             # P(return < 0)
    prices: np.ndarray                # Close prices for return tracking


class TLSTMGPUEvaluator:
    """
    High-throughput GPU evaluation engine for the distilled Temporal LSTM student.
    """
    def __init__(
        self,
        checkpoint_path: str = "checkpoints/tlstm_klint_distilled.pt",
        device: Optional[str] = None,
        context_bars: int = 64,
        batch_size: int = 64,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.context_bars = context_bars
        self.context_tokens = context_bars * 3
        self.batch_size = batch_size

        self.model, self.tokenizer, self.decoder, self.config = self._load_bundle(checkpoint_path)
        self.model.eval()
        self.tokenizer.eval()

        # Precompute RVQ Price Codebook Expected Returns on GPU
        self.codebook_returns = self._precompute_price_codebook_returns()

    def _load_bundle(self, checkpoint_path: str):
        """Loads student bundle with fallbacks."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Could not locate student checkpoint at: {checkpoint_path}")

        print(f"[TLSTMEvaluator] Loading Distilled TLSTM bundle from: {checkpoint_path} to {self.device}")
        bundle = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        cfg = bundle.get("config", TLSTMConfig())
        model = TLSTMKlint(cfg).to(self.device)
        model.load_state_dict(bundle["model_state_dict"])

        tokenizer = FactorTokenizer().to(self.device)
        tok_state = bundle.get("tokenizer_state_dict", None)
        if tok_state is not None:
            tokenizer.load_state_dict(tok_state)

        decoder = GeometricDecoder().to(self.device)
        return model, tokenizer, decoder, cfg

    @torch.no_grad()
    def _precompute_price_codebook_returns(self) -> torch.Tensor:
        """
        Decodes all 512 discrete price tokens into continuous return representations.
        Shape: (512,) on self.device
        """
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

    @torch.no_grad()
    def evaluate_universe(
        self,
        dataset: Dict[str, Dict[str, Any]],
        signal_threshold: float = 0.0005,  # 5 bps return threshold
        prob_threshold: float = 0.51,      # 51% directional probability threshold
        verbose: bool = True,
    ) -> Dict[str, AssetEvaluationResult]:
        """
        Runs GPU-accelerated evaluation across all fresh assets.
        """
        results: Dict[str, AssetEvaluationResult] = {}
        total_assets = len(dataset)
        start_eval_time = time.time()

        if verbose:
            print(f"[TLSTMEvaluator] Starting GPU evaluation across {total_assets} fresh assets...")

        asset_list = list(dataset.values())
        total_windows_evaluated = 0

        for asset_data in asset_list:
            ticker = asset_data["ticker"]
            asset_class = asset_data["asset_class"]
            name = asset_data["name"]
            ohlcv = asset_data["ohlcv"]
            factors = asset_data["factors"]

            num_bars = len(ohlcv)
            if num_bars <= self.context_bars + 2:
                continue

            # Tokenize all bars for this asset
            f_price = torch.from_numpy(factors.price_path).float().unsqueeze(0).to(self.device)
            f_range = torch.from_numpy(factors.range_shape).float().unsqueeze(0).to(self.device)
            f_act = torch.from_numpy(factors.activity).float().unsqueeze(0).to(self.device)

            p_tok, r_tok, a_tok, _ = self.tokenizer.encode(f_price, f_range, f_act)
            all_tokens = FactorTokenizer.interleave(p_tok, r_tok, a_tok).squeeze(0)  # (3 * num_bars,)

            # Build sliding context windows
            num_eval_bars = num_bars - self.context_bars
            windows = []
            for b in range(self.context_bars, num_bars):
                end_tok = b * 3
                start_tok = end_tok - self.context_tokens
                windows.append(all_tokens[start_tok:end_tok])

            if not windows:
                continue

            all_windows = torch.stack(windows, dim=0)  # (num_eval_bars, context_tokens)
            total_windows_evaluated += num_eval_bars

            pred_returns = []
            p_up_list = []
            p_down_list = []

            for i in range(0, num_eval_bars, self.batch_size):
                batch_toks = all_windows[i : i + self.batch_size]  # (B, context_tokens)
                out = self.model(batch_toks)

                # The last token in the window is Activity (A_{b-1})
                # The model's prediction at this position is for Price (P_b)
                if isinstance(out, dict):
                    last_logits = out["logits_price"][:, -1, :]  # (B, 512)
                else:
                    last_logits = out[:, -1, :512]

                probs = F.softmax(last_logits, dim=-1)  # (B, 512)

                # Vectorized expected return: sum_k P(k) * r_body(k)
                exp_ret = torch.matmul(probs, self.codebook_returns)  # (B,)

                # Directional Probabilities
                up_mask = self.codebook_returns > 0
                down_mask = self.codebook_returns < 0
                prob_up = probs[:, up_mask].sum(dim=-1)
                prob_down = probs[:, down_mask].sum(dim=-1)

                pred_returns.append(exp_ret.cpu().numpy())
                p_up_list.append(prob_up.cpu().numpy())
                p_down_list.append(prob_down.cpu().numpy())

            pred_ret_arr = np.concatenate(pred_returns)
            prob_up_arr = np.concatenate(p_up_list)
            prob_down_arr = np.concatenate(p_down_list)

            # Ground truth realized body return for evaluation bars
            realized_body_returns = factors.price_path[self.context_bars : num_bars, 1]
            close_prices = ohlcv[self.context_bars : num_bars, 3]

            # Generate Trading Signals: +1 for Long, -1 for Short, 0 for Neutral
            signals = np.zeros(num_eval_bars, dtype=np.float32)
            long_cond = (pred_ret_arr > signal_threshold) & (prob_up_arr > prob_threshold)
            short_cond = (pred_ret_arr < -signal_threshold) & (prob_down_arr > prob_threshold)
            signals[long_cond] = 1.0
            signals[short_cond] = -1.0

            results[ticker] = AssetEvaluationResult(
                ticker=ticker,
                name=name,
                asset_class=asset_class,
                realized_returns=realized_body_returns,
                predicted_returns=pred_ret_arr,
                signals=signals,
                prob_up=prob_up_arr,
                prob_down=prob_down_arr,
                prices=close_prices,
            )

        elapsed = time.time() - start_eval_time
        fps = total_windows_evaluated / max(elapsed, 1e-4)

        if verbose:
            print(f"[TLSTMEvaluator] Successfully evaluated {len(results)} assets ({total_windows_evaluated:,} windows) in {elapsed:.2f}s ({fps:.1f} windows/sec)!")

        return results
