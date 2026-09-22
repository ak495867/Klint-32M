"""Multi-asset cross-market dataset for fine-tuning Klint-32M across 100 liquid institutional assets."""

from __future__ import annotations

import os
import concurrent.futures
from typing import List, Dict, Optional, Tuple, Literal, Any
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from klint.benchmark.data_fetcher import MultiAssetDataFetcher
from klint.data.factors import FactorDecomposer
from klint.tokenizer.factor_tokenizer import FactorTokenizer


# Curated universe of 100 premier liquid institutional assets across 6 major asset classes
INSTITUTIONAL_100_TICKERS: List[str] = [
    # US Equities: Mega-cap Tech & High-Growth (16)
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AVGO",
    "AMD", "QCOM", "INTC", "CRM", "ORCL", "ADBE", "PLTR", "CSCO",
    # US Equities: Financials (9)
    "JPM", "V", "MA", "BAC", "WFC", "MS", "GS", "BLK", "COIN",
    # US Equities: Healthcare (6)
    "LLY", "UNH", "JNJ", "ABBV", "MRK", "PFE",
    # US Equities: Consumer & Retail (5)
    "HD", "COST", "WMT", "PG", "KO",
    # US Equities: Industrials & Energy (4)
    "CAT", "GE", "XOM", "CVX",
    # Global & Sector ETFs (20)
    "SPY", "QQQ", "IWM", "DIA", "VOO", "VTI",
    "XLK", "XLF", "XLV", "XLE", "XLI", "XLY", "XLP", "XLU", "XLB",
    "SMH", "SOXX", "ARKK", "EEM", "INDA",
    # Crypto Macro (15)
    "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD",
    "ADA-USD", "DOGE-USD", "AVAX-USD", "LINK-USD", "DOT-USD",
    "NEAR-USD", "ATOM-USD", "LTC-USD", "BCH-USD", "XLM-USD",
    # Commodities (10)
    "GLD", "SLV", "USO", "UNG", "DBC", "CPER", "PPLT", "WEAT", "CORN", "SOYB",
    # Rates & Fixed Income (10)
    "TLT", "IEF", "SHY", "BND", "AGG", "LQD", "HYG", "JNK", "TIP", "BIL",
    # Foreign Exchange / Global Macro (5)
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X",
]

CORE_12_TICKERS: List[str] = [
    "SPY", "QQQ", "AAPL", "NVDA", "MSFT", "TSLA",
    "BTC-USD", "ETH-USD", "SOL-USD",
    "GLD", "USO", "TLT",
]

DEFAULT_FINETUNE_TICKERS: List[str] = INSTITUTIONAL_100_TICKERS


class MultiAssetFineTuneDataset(Dataset):
    """
    Unified multi-asset token dataset for causal foundation model fine-tuning.
    
    Ingests diverse cross-market assets across Equities, ETFs, Crypto, Commodities,
    Rates, and Forex. Sanitizes candle invariants, computes stationary factors,
    encodes via Residual Vector Quantization (RVQ), and slices into overlapping
    autoregressive token sequences.
    """
    def __init__(
        self,
        tickers: Optional[List[str]] = None,
        local_files: Optional[List[str]] = None,
        tokenizer: Optional[FactorTokenizer] = None,
        split: Literal["train", "val"] = "train",
        val_ratio: float = 0.15,
        context_bars: int = 256,
        stride_bars: int = 64,
        period: str = "2y",
        interval: str = "1h",
        cache_dir: str = "data/yfinance_cache",
        max_workers: int = 8,
        device: str = "cpu",
    ):
        super().__init__()
        self.tickers = DEFAULT_FINETUNE_TICKERS if tickers is None else tickers
        self.local_files = [] if local_files is None else local_files
        self.split = split
        self.val_ratio = val_ratio
        self.context_bars = context_bars
        self.stride_bars = stride_bars
        self.seq_len = context_bars * 3  # 3 tokens per bar (P, R, A)
        self.stride_tokens = stride_bars * 3
        self.max_workers = max_workers
        self.device = device

        self.decomposer = FactorDecomposer()
        self.tokenizer = tokenizer or FactorTokenizer()
        self.tokenizer.to(device).eval()

        self.fetcher = MultiAssetDataFetcher(cache_dir=cache_dir)
        self.samples: List[Tuple[torch.Tensor, str]] = []  # List of (token_tensor, ticker)

        self._build_dataset(period=period, interval=interval)

    def _build_dataset(self, period: str, interval: str):
        """Fetches data, validates, tokenizes, and creates sliding sequence windows."""
        all_asset_tokens: Dict[str, torch.Tensor] = {}

        # 1. Load local .npy files first if provided
        for file_path in self.local_files:
            if os.path.exists(file_path):
                ticker = os.path.splitext(os.path.basename(file_path))[0]
                try:
                    raw_data = np.load(file_path)
                    ohlcv = self._sanitize_raw_ohlcv(raw_data)
                    tokens = self._tokenize_ohlcv(ohlcv)
                    if tokens is not None and len(tokens) >= self.seq_len:
                        all_asset_tokens[ticker] = tokens
                except Exception as e:
                    print(f"Warning: Failed loading local file {file_path}: {e}")

        # 2. Ingest tickers in parallel using ThreadPoolExecutor for fast network I/O
        if len(self.tickers) > 0:
            total_target = len(self.tickers)
            print(f"Ingesting {total_target} assets in parallel (period={period}, interval={interval})...")

            def _fetch_and_tokenize(t: str) -> Optional[Tuple[str, torch.Tensor]]:
                try:
                    asset_info = self.fetcher.fetch_asset(
                        t, period=period, interval=interval, min_bars=self.context_bars + 10
                    )
                    if asset_info is not None:
                        tokens = self._tokenize_ohlcv(asset_info["ohlcv"])
                        if tokens is not None and len(tokens) >= self.seq_len:
                            return t, tokens
                except Exception:
                    pass
                return None

            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {executor.submit(_fetch_and_tokenize, t): t for t in self.tickers}
                completed = 0
                for future in concurrent.futures.as_completed(futures):
                    completed += 1
                    res = future.result()
                    if res is not None:
                        t, tok = res
                        all_asset_tokens[t] = tok
                    if completed % 25 == 0 or completed == total_target:
                        print(f"  --> Progress: {completed}/{total_target} processed ({len(all_asset_tokens)} active datasets)")

        # 3. Fallback: If no online data is available, generate synthetic multi-asset trajectories
        if len(all_asset_tokens) == 0:
            print("Notice: No live or cached assets found. Generating synthetic multi-asset calibration data.")
            for dummy_ticker in ["SYN_EQ", "SYN_CRYPTO", "SYN_MACRO"]:
                synth_ohlcv = self._generate_synthetic_ohlcv(bars=2000)
                tokens = self._tokenize_ohlcv(synth_ohlcv)
                if tokens is not None:
                    all_asset_tokens[dummy_ticker] = tokens

        # 4. Partition each asset chronologically into train/val sliding windows
        for ticker, tokens in all_asset_tokens.items():
            total_tokens = len(tokens)
            split_idx = int(total_tokens * (1.0 - self.val_ratio))
            embargo_tokens = self.seq_len  # Embargo equal to context window

            if self.split == "train":
                asset_tokens = tokens[:split_idx]
            else:
                start_val = split_idx + embargo_tokens
                if start_val >= total_tokens - self.seq_len:
                    start_val = split_idx  # Fallback if series is short
                asset_tokens = tokens[start_val:]

            num_tokens = len(asset_tokens)
            if num_tokens >= self.seq_len:
                for start in range(0, num_tokens - self.seq_len + 1, self.stride_tokens):
                    window = asset_tokens[start : start + self.seq_len]
                    self.samples.append((window, ticker))

        print(f"MultiAssetFineTuneDataset ({self.split}): Created {len(self.samples)} sequence windows across {len(all_asset_tokens)} assets.")

    def _sanitize_raw_ohlcv(self, raw_data: np.ndarray) -> np.ndarray:
        """Extracts and sanitizes OHLCV invariant bounds."""
        if raw_data.ndim == 2 and raw_data.shape[1] >= 5:
            if raw_data.shape[1] == 11:
                ohlcv = raw_data[:, 1:6].astype(np.float64)
            else:
                ohlcv = raw_data[:, :5].astype(np.float64)
        else:
            raise ValueError(f"Invalid raw data shape: {raw_data.shape}")

        open_p = ohlcv[:, 0]
        high_p = np.maximum(ohlcv[:, 1], np.maximum(open_p, ohlcv[:, 3]))
        low_p = np.minimum(ohlcv[:, 2], np.minimum(open_p, ohlcv[:, 3]))
        low_p = np.maximum(low_p, 1e-6)
        close_p = ohlcv[:, 3]
        vol = np.maximum(ohlcv[:, 4], 0.0)

        return np.stack([open_p, high_p, low_p, close_p, vol], axis=-1)

    def _tokenize_ohlcv(self, ohlcv: np.ndarray) -> Optional[torch.Tensor]:
        """Decomposes OHLCV into factors and quantizes into interleaved token tensor."""
        try:
            factors = self.decomposer.decompose(ohlcv)
            p_path = torch.from_numpy(factors.price_path).float().to(self.device).unsqueeze(0)
            r_shape = torch.from_numpy(factors.range_shape).float().to(self.device).unsqueeze(0)
            activity = torch.from_numpy(factors.activity).float().to(self.device).unsqueeze(0)

            with torch.no_grad():
                p_tok, r_tok, a_tok, _ = self.tokenizer.encode(p_path, r_shape, activity)
                interleaved = self.tokenizer.interleave(p_tok, r_tok, a_tok).squeeze(0)  # (3T,)

            return interleaved.cpu()
        except Exception as e:
            print(f"Error tokenizing OHLCV: {e}")
            return None

    def _generate_synthetic_ohlcv(self, bars: int = 2000) -> np.ndarray:
        """Generates realistic synthetic OHLCV adhering strictly to financial bounds."""
        np.random.seed(42)
        returns = np.random.normal(0.0002, 0.015, size=bars)
        price = 100.0 * np.exp(np.cumsum(returns))

        open_p = price
        close_p = price * np.exp(np.random.normal(0, 0.005, size=bars))
        ranges = np.abs(np.random.normal(0.01, 0.005, size=bars)) * price
        high_p = np.maximum(open_p, close_p) + ranges * 0.6
        low_p = np.maximum(np.minimum(open_p, close_p) - ranges * 0.4, 1e-4)
        volume = np.random.lognormal(mean=10.0, sigma=0.8, size=bars)

        return np.stack([open_p, high_p, low_p, close_p, volume], axis=-1)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        tokens, ticker = self.samples[idx]
        # Autoregressive sequence modeling: inputs predict targets shifted by 1
        inputs = tokens[:-1].long()
        targets = tokens[1:].long()

        return {
            "inputs": inputs,
            "targets": targets,
            "ticker": ticker,
        }


def build_finetune_dataloaders(
    tickers: Optional[List[str]] = None,
    local_files: Optional[List[str]] = None,
    tokenizer: Optional[FactorTokenizer] = None,
    batch_size: int = 16,
    context_bars: int = 256,
    stride_bars: int = 64,
    period: str = "2y",
    interval: str = "1h",
    val_ratio: float = 0.15,
    max_workers: int = 8,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """Factory creating train and validation DataLoaders for fine-tuning."""
    target_tickers = DEFAULT_FINETUNE_TICKERS if tickers is None else tickers
    train_ds = MultiAssetFineTuneDataset(
        tickers=target_tickers,
        local_files=local_files,
        tokenizer=tokenizer,
        split="train",
        val_ratio=val_ratio,
        context_bars=context_bars,
        stride_bars=stride_bars,
        period=period,
        interval=interval,
        max_workers=max_workers,
    )
    val_ds = MultiAssetFineTuneDataset(
        tickers=target_tickers,
        local_files=local_files,
        tokenizer=tokenizer,
        split="val",
        val_ratio=val_ratio,
        context_bars=context_bars,
        stride_bars=stride_bars,
        period=period,
        interval=interval,
        max_workers=max_workers,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader
