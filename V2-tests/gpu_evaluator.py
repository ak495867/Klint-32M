"""
Vectorized GPU-Accelerated Evaluator for Klint-32M v2 Out-of-Sample Market Trajectories.

Runs parallel batched forward passes on CUDA/GPU to compute:
- Expected next-bar returns (soft-decoded from discrete RVQ codebook)
- Directional probabilities P(Up) and P(Down)
- Directional signals (-1, 0, +1)
- Realized returns and transaction-cost-adjusted strategy returns
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn.functional as F

from klint.models.klint_32m import Klint32M, KlintConfig
from klint.tokenizer.factor_tokenizer import FactorTokenizer
from klint.tokenizer.geometric_decoder import GeometricDecoder


@dataclass
class AssetEvaluationResult:
    """Contains time-series predictions and realized returns for an asset."""
    ticker: str
    asset_class: str
    name: str
    dates_idx: np.ndarray
    predicted_returns: np.ndarray      # Shape (N,)
    realized_returns: np.ndarray       # Shape (N,)
    prob_up: np.ndarray                # Shape (N,)
    prob_down: np.ndarray              # Shape (N,)
    signals: np.ndarray                # Shape (N,) in {-1, 0, +1}
    prices: np.ndarray                 # Shape (N,) Close prices
    volumes: np.ndarray                # Shape (N,) Volumes


class GPUEvaluator:
    """
    Vectorized GPU evaluation engine for Klint-32M v2.
    Processes 300+ assets in parallel batches with zero Python per-bar loops.
    """

    def __init__(
        self,
        checkpoint_path: str = "checkpoints/klint_32m_v2_release.pt",
        device: Optional[str] = None,
        context_bars: int = 64,
        batch_size: int = 64,
    ):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.context_bars = context_bars
        self.context_tokens = context_bars * 3
        self.batch_size = batch_size

        self.model, self.tokenizer, self.decoder, self.config = self._load_bundle(checkpoint_path)
        self.model.eval()
        self.tokenizer.eval()

        # Precompute RVQ Price Codebook Expected Returns on GPU
        self.codebook_returns = self._precompute_price_codebook_returns()

    def _load_bundle(self, checkpoint_path: str):
        """Loads all-in-one release bundle with fallbacks."""
        if not os.path.exists(checkpoint_path):
            fallback_v1 = "checkpoints/klint_32m_release.pt"
            if os.path.exists(fallback_v1):
                checkpoint_path = fallback_v1
            else:
                # Try downloading from HF if available
                try:
                    from huggingface_hub import hf_hub_download
                    print(f"[GPUEvaluator] Checkpoint {checkpoint_path} not found. Attempting download from HF akhverm/Klint-32M...")
                    checkpoint_path = hf_hub_download(repo_id="akhverm/Klint-32M", filename="klint_32m_v2_release.pt")
                except Exception:
                    raise FileNotFoundError(f"Could not locate checkpoint at {checkpoint_path} or on Hugging Face.")

        print(f"[GPUEvaluator] Loading Klint-32M bundle from: {checkpoint_path} to {self.device}")
        bundle = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        cfg = bundle.get("config", KlintConfig())
        model = Klint32M(cfg).to(self.device)
        model.load_state_dict(bundle["model_state_dict"])

        tokenizer = FactorTokenizer().to(self.device)
        tok_state = bundle.get("tokenizer_state_dict", bundle.get("tokenizer_state", None))
        if tok_state is not None:
            tokenizer.load_state_dict(tok_state)

        decoder = GeometricDecoder().to(self.device)
        return model, tokenizer, decoder, cfg

    @torch.no_grad()
    def _precompute_price_codebook_returns(self) -> torch.Tensor:
        """
        Decodes all 512 discrete price tokens into their continuous [r_gap, r_body]
        representations and extracts the body return component.
        Shape: (512,) on self.device
        """
        if hasattr(self.tokenizer.rvq_price, "layers"):
            codebook_weights = self.tokenizer.rvq_price.layers[0].embedding.to(self.device)  # (512, 64)
            continuous_factors = self.tokenizer.dec_price(codebook_weights)                  # (512, 2)
            r_body = continuous_factors[:, 1]                                                # (512,)
            return r_body
        elif hasattr(self.tokenizer.rvq_price, "codebooks"):
            codebook_weights = self.tokenizer.rvq_price.codebooks[0].to(self.device)
            continuous_factors = self.tokenizer.dec_price(codebook_weights)
            return continuous_factors[:, 1]
        else:
            p_indices = torch.arange(self.config.price_vocab_size, device=self.device)
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
        Vectorized cross-asset evaluation.
        Batches sequence windows across all assets to maximize GPU throughput.
        """
        results: Dict[str, AssetEvaluationResult] = {}
        total_assets = len(dataset)
        if verbose:
            print(f"[GPUEvaluator] Starting GPU-accelerated evaluation across {total_assets} fresh assets...")

        # Process assets
        asset_list = list(dataset.values())
        processed_count = 0

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
            # Convert factors to tensors with batch dimension (1, num_bars, dim)
            f_price = torch.from_numpy(factors.price_path).float().unsqueeze(0).to(self.device)
            f_range = torch.from_numpy(factors.range_shape).float().unsqueeze(0).to(self.device)
            f_act = torch.from_numpy(factors.activity).float().unsqueeze(0).to(self.device)

            p_tok, r_tok, a_tok, _ = self.tokenizer.encode(f_price, f_range, f_act)
            all_tokens = FactorTokenizer.interleave(p_tok, r_tok, a_tok).squeeze(0)  # (3 * num_bars,)

            # Build sliding context windows
            # Predict bar t for t in [context_bars, num_bars - 1]
            num_eval_bars = num_bars - self.context_bars
            windows = []
            for b in range(self.context_bars, num_bars):
                # Prompt window tokens: from (b - context_bars)*3 to b*3
                end_tok = b * 3
                start_tok = end_tok - self.context_tokens
                windows.append(all_tokens[start_tok:end_tok])

            if not windows:
                continue

            # Stack into batches and run on GPU
            all_windows = torch.stack(windows, dim=0)  # (num_eval_bars, context_tokens)

            pred_returns = []
            p_up_list = []
            p_down_list = []

            for i in range(0, num_eval_bars, self.batch_size):
                batch_toks = all_windows[i : i + self.batch_size]  # (B, context_tokens)
                logits = self.model(batch_toks)                    # (B, context_tokens, vocab_size)

                # The last token in the window is Activity (A_{b-1})
                # The model's prediction at this position is for Price (P_b)
                last_logits = logits[:, -1, : self.config.price_vocab_size]  # (B, 512)
                probs = F.softmax(last_logits, dim=-1)                       # (B, 512)

                # Vectorized expected return: sum_k P(k) * r_body(k)
                exp_ret = torch.matmul(probs, self.codebook_returns)         # (B,)

                # Vectorized Directional Probabilities
                up_mask = self.codebook_returns > 0                          # (512,)
                down_mask = self.codebook_returns < 0                        # (512,)
                prob_up = probs[:, up_mask].sum(dim=-1)                      # (B,)
                prob_down = probs[:, down_mask].sum(dim=-1)                  # (B,)

                pred_returns.append(exp_ret.cpu().numpy())
                p_up_list.append(prob_up.cpu().numpy())
                p_down_list.append(prob_down.cpu().numpy())

            pred_returns_arr = np.concatenate(pred_returns, axis=0)
            prob_up_arr = np.concatenate(p_up_list, axis=0)
            prob_down_arr = np.concatenate(p_down_list, axis=0)

            # Compute realized close-to-close returns
            eval_closes = ohlcv[self.context_bars :, 3]
            prior_closes = ohlcv[self.context_bars - 1 : -1, 3]
            realized_returns_arr = (eval_closes - prior_closes) / np.maximum(prior_closes, 1e-8)

            # Generate Alpha Trading Signals: {-1, 0, +1}
            signals = np.zeros(num_eval_bars, dtype=np.int32)
            long_cond = (pred_returns_arr > signal_threshold) & (prob_up_arr >= prob_threshold)
            short_cond = (pred_returns_arr < -signal_threshold) & (prob_down_arr >= prob_threshold)
            signals[long_cond] = 1
            signals[short_cond] = -1

            eval_res = AssetEvaluationResult(
                ticker=ticker,
                asset_class=asset_class,
                name=name,
                dates_idx=np.arange(self.context_bars, num_bars),
                predicted_returns=pred_returns_arr,
                realized_returns=realized_returns_arr,
                prob_up=prob_up_arr,
                prob_down=prob_down_arr,
                signals=signals,
                prices=eval_closes,
                volumes=ohlcv[self.context_bars :, 4],
            )
            results[ticker] = eval_res
            processed_count += 1

            if verbose and (processed_count % 50 == 0 or processed_count == total_assets):
                print(f"  Processed {processed_count}/{total_assets} fresh assets...")

        if verbose:
            print(f"[GPUEvaluator] Evaluation finished for {len(results)} assets.")

        return results
