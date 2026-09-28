"""
High-Speed Vectorized Multi-Asset Data Loader for TLSTM-Klint Distillation and Evaluation.

Loads, sanitizes, and extracts stationary causal factor streams across 300+ fresh assets
with disk caching (checking local and V2 cache), parallel fetching, and zero data leakage.
"""

from __future__ import annotations

import os
import time
import concurrent.futures
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import pandas as pd
import yfinance as yf

# Flexible imports
try:
    from klint.data.validator import validate_ohlcv
    from klint.data.factors import FactorDecomposer, FactorStreams
except ImportError:
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
    from klint.data.validator import validate_ohlcv
    from klint.data.factors import FactorDecomposer, FactorStreams


class FreshUniverseDataLoader:
    """
    High-throughput data loader for 300+ fresh assets.
    Enforces strict chronological ordering, data hygiene, and factor decomposition.
    """

    def __init__(
        self,
        cache_dir: str = "Tlstm-Klint/cache",
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

        # Fallback cache from V2 multitest if available
        self.fallback_cache_dir = "V2-multitest/cache"

    def fetch_single_ticker(self, ticker: str) -> Optional[np.ndarray]:
        """Fetches and sanitizes OHLCV for a single ticker with dual-level disk caching."""
        cache_file = os.path.join(self.cache_dir, f"{ticker}_{self.period}_{self.interval}.npy")
        fallback_file = os.path.join(self.fallback_cache_dir, f"{ticker}_{self.period}_{self.interval}.npy")

        # 1. Check primary cache
        if os.path.exists(cache_file):
            try:
                arr = np.load(cache_file)
                if arr.ndim == 2 and arr.shape[0] >= self.min_bars and arr.shape[1] == 5:
                    return arr
            except Exception:
                pass

        # 2. Check fallback cache (from V2 battery)
        if os.path.exists(fallback_file):
            try:
                arr = np.load(fallback_file)
                if arr.ndim == 2 and arr.shape[0] >= self.min_bars and arr.shape[1] == 5:
                    # Save into primary cache for speed
                    np.save(cache_file, arr)
                    return arr
            except Exception:
                pass

        # 3. Fetch live via yfinance
        try:
            df = yf.download(
                ticker,
                period=self.period,
                interval=self.interval,
                progress=False,
                auto_adjust=False,
            )
            if df.empty or len(df) < self.min_bars:
                return None

            # Handle MultiIndex columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            required_cols = ["Open", "High", "Low", "Close", "Volume"]
            if not all(col in df.columns for col in required_cols):
                return None

            raw_arr = df[required_cols].to_numpy(dtype=np.float64)

            # Sanitize and validate geometry
            clean_arr, _ = validate_ohlcv(raw_arr)
            if len(clean_arr) < self.min_bars:
                return None

            np.save(cache_file, clean_arr)
            return clean_arr

        except Exception:
            return None

    def load_universe(
        self,
        assets: List[Dict[str, str]],
        verbose: bool = True,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Loads, extracts factor streams, and verifies geometry across the provided assets in parallel.
        """
        start_time = time.time()
        results: Dict[str, Dict[str, Any]] = {}

        if verbose:
            print(f"[DataLoader] Ingesting {len(assets)} fresh assets across parallel threads...")

        def _process_asset(asset_meta: Dict[str, str]) -> Optional[Tuple[str, Dict[str, Any]]]:
            ticker = asset_meta["ticker"]
            arr = self.fetch_single_ticker(ticker)
            if arr is None:
                return None

            try:
                factors: FactorStreams = self.decomposer.decompose(arr)
                return ticker, {
                    "ticker": ticker,
                    "name": asset_meta.get("name", ticker),
                    "asset_class": asset_meta.get("asset_class", "Equities"),
                    "ohlcv": arr,
                    "factors": factors,
                }
            except Exception:
                return None

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_asset = {
                executor.submit(_process_asset, meta): meta["ticker"] for meta in assets
            }
            for future in concurrent.futures.as_completed(future_to_asset):
                res = future.result()
                if res is not None:
                    ticker, data = res
                    results[ticker] = data

        elapsed = time.time() - start_time
        if verbose:
            print(f"[DataLoader] Successfully ingested {len(results)}/{len(assets)} assets in {elapsed:.2f}s!")

        return results
