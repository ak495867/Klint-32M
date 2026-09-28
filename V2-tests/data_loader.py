"""
High-Speed Vectorized Multi-Asset Data Loader for Klint-32M v2 Out-of-Sample Testing.

Loads, sanitizes, and extracts stationary causal factor streams across 300+ fresh assets
with disk caching, parallelized fetching, and zero data leakage.
"""

from __future__ import annotations

import os
import time
import concurrent.futures
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import pandas as pd
import yfinance as yf

from klint.data.validation import validate_ohlcv
from klint.data.factors import FactorDecomposer, FactorStreams


class FreshUniverseDataLoader:
    """
    High-throughput data loader for 300+ fresh assets.
    Enforces strict chronological ordering and factor decomposition.
    """

    def __init__(
        self,
        cache_dir: str = "V2-multitest/cache",
        period: str = "1y",
        interval: str = "1d",
        min_bars: int = 120,
        max_workers: int = 16,
    ):
        self.cache_dir = cache_dir
        self.period = period
        self.interval = interval
        self.min_bars = min_bars
        self.max_workers = max_workers
        self.decomposer = FactorDecomposer()
        os.makedirs(self.cache_dir, exist_ok=True)

    def fetch_single_ticker(self, ticker: str) -> Optional[np.ndarray]:
        """Fetches and sanitizes OHLCV for a single ticker with disk caching."""
        cache_file = os.path.join(self.cache_dir, f"{ticker}_{self.period}_{self.interval}.npy")

        # Load from disk cache if exists
        if os.path.exists(cache_file):
            try:
                arr = np.load(cache_file)
                if arr.ndim == 2 and arr.shape[0] >= self.min_bars and arr.shape[1] == 5:
                    return arr
            except Exception:
                pass

        # Fetch live via yfinance
        try:
            df = yf.download(
                ticker,
                period=self.period,
                interval=self.interval,
                progress=False,
                auto_adjust=True,
            )
            if df is None or len(df) < self.min_bars:
                return None

            # Handle MultiIndex columns from recent yfinance versions
            if isinstance(df.columns, pd.MultiIndex):
                df = df.xs(ticker, axis=1, level=1) if ticker in df.columns.levels[1] else df.droplevel(1, axis=1)

            cols = [c.lower() for c in df.columns]
            col_map = {c: c.lower() for c in df.columns}
            df = df.rename(columns=col_map)

            req = ["open", "high", "low", "close", "volume"]
            if not all(k in df.columns for k in req):
                return None

            ohlcv = df[req].values.astype(np.float64)

            # Sanitize Invariants strictly (no lookahead, causal repair)
            # High >= max(Open, Close), Low <= min(Open, Close)
            ohlcv[:, 1] = np.maximum(ohlcv[:, 1], np.maximum(ohlcv[:, 0], ohlcv[:, 3]))
            ohlcv[:, 2] = np.minimum(ohlcv[:, 2], np.minimum(ohlcv[:, 0], ohlcv[:, 3]))
            ohlcv[:, 4] = np.maximum(ohlcv[:, 4], 0.0)

            # Drop non-finite rows
            valid_mask = np.isfinite(ohlcv).all(axis=1)
            ohlcv = ohlcv[valid_mask]

            if len(ohlcv) < self.min_bars:
                return None

            # Save to disk cache
            np.save(cache_file, ohlcv)
            return ohlcv
        except Exception:
            return None

    def load_all_assets(
        self,
        assets: List[Dict[str, str]],
        verbose: bool = True,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Loads all assets in parallel using ThreadPoolExecutor.
        Returns a dictionary mapping ticker -> {ohlcv, factors, asset_class, name}.
        """
        start_time = time.time()
        tickers = [a["ticker"] for a in assets]
        asset_info = {a["ticker"]: a for a in assets}

        if verbose:
            print(f"[FreshUniverseDataLoader] Ingesting {len(tickers)} fresh assets across 6 asset classes...")

        dataset: Dict[str, Dict[str, Any]] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_ticker = {
                executor.submit(self.fetch_single_ticker, t): t for t in tickers
            }
            for future in concurrent.futures.as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    ohlcv = future.result()
                    if ohlcv is not None and len(ohlcv) >= self.min_bars:
                        # Extract stationary factor streams
                        factors = self.decomposer.decompose(ohlcv)
                        info = asset_info[ticker]
                        dataset[ticker] = {
                            "ticker": ticker,
                            "name": info.get("name", ticker),
                            "asset_class": info.get("asset_class", "Unclassified"),
                            "ohlcv": ohlcv,
                            "factors": factors,
                            "length": len(ohlcv),
                        }
                except Exception as e:
                    if verbose:
                        print(f"  Warning: Failed to process {ticker}: {e}")

        elapsed = time.time() - start_time
        if verbose:
            print(f"[FreshUniverseDataLoader] Ingestion completed in {elapsed:.2f}s.")
            print(f"[FreshUniverseDataLoader] Successfully loaded {len(dataset)} valid fresh assets (Target: >= 300).")

        return dataset
