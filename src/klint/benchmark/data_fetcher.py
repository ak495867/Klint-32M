"""Robust multi-asset data fetcher utilizing yfinance with disk caching and invariant validation."""

import os
import time
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import yfinance as yf

from klint.data.validator import validate_ohlcv


class MultiAssetDataFetcher:
    """
    Downloads historical OHLCV data for arbitrary assets using yfinance,
    caches locally on disk to prevent rate-limiting, and enforces financial invariants.
    """
    def __init__(self, cache_dir: str = "data/yfinance_cache"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_cache_path(self, ticker: str, period: str, interval: str) -> str:
        safe_ticker = ticker.replace("=", "_").replace("-", "_").replace("^", "_")
        return os.path.join(self.cache_dir, f"{safe_ticker}_{period}_{interval}.csv")

    def fetch_asset(
        self,
        ticker: str,
        period: str = "1y",
        interval: str = "1d",
        force_download: bool = False,
        min_bars: int = 50,
    ) -> Optional[Dict[str, Any]]:
        """
        Fetches historical OHLCV data for a single asset.
        
        Returns:
            dict containing:
                - 'ticker': str
                - 'ohlcv': np.ndarray of shape (N, 5) [O, H, L, C, V]
                - 'dates': pd.DatetimeIndex
                - 'bars_count': int
        """
        cache_file = self._get_cache_path(ticker, period, interval)

        # 1. Try to load from disk cache
        if not force_download and os.path.exists(cache_file):
            try:
                df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                if len(df) >= min_bars:
                    return self._process_dataframe(ticker, df)
            except Exception:
                pass  # Re-download if cache corrupted

        # 2. Download from yfinance
        try:
            time.sleep(0.05)  # Respectful rate-limiting
            ticker_obj = yf.Ticker(ticker)
            df = ticker_obj.history(period=period, interval=interval, auto_adjust=True)

            if df is None or len(df) < min_bars:
                return None

            # Clean and filter columns
            required_cols = ["Open", "High", "Low", "Close", "Volume"]
            if not all(col in df.columns for col in required_cols):
                return None

            df = df[required_cols].dropna()
            # Cache to disk
            df.to_csv(cache_file)

            return self._process_dataframe(ticker, df)

        except Exception as e:
            return None

    def _process_dataframe(self, ticker: str, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Cleans and validates OHLCV dataframe."""
        raw_arr = df[["Open", "High", "Low", "Close", "Volume"]].values.astype(np.float64)

        # Sanitize candle invariants: guarantee high >= max(open, close), low <= min(open, close)
        open_p = raw_arr[:, 0]
        high_p = raw_arr[:, 1]
        low_p = raw_arr[:, 2]
        close_p = raw_arr[:, 3]
        vol = np.maximum(raw_arr[:, 4], 0.0)

        # Fix any micro-tick reporting discrepancies from vendor
        high_clean = np.maximum(high_p, np.maximum(open_p, close_p))
        low_clean = np.minimum(low_p, np.minimum(open_p, close_p))
        low_clean = np.maximum(low_clean, 1e-6)

        clean_ohlcv = np.stack([open_p, high_clean, low_clean, close_p, vol], axis=-1)

        is_valid, report = validate_ohlcv(clean_ohlcv)
        if not is_valid or len(clean_ohlcv) == 0:
            return None

        return {
            "ticker": ticker,
            "ohlcv": clean_ohlcv,
            "dates": df.index,
            "bars_count": len(clean_ohlcv),
        }

    def fetch_universe(
        self,
        assets: List[Dict[str, str]],
        period: str = "1y",
        interval: str = "1d",
        verbose: bool = True,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Fetches and caches data across an entire universe of assets.
        """
        results = {}
        total = len(assets)
        if verbose:
            print(f"Fetching market data for {total} assets (period={period}, interval={interval})...")

        for idx, item in enumerate(assets, 1):
            t = item["ticker"]
            asset_data = self.fetch_asset(t, period=period, interval=interval)
            if asset_data is not None:
                asset_data["name"] = item.get("name", t)
                asset_data["asset_class"] = item.get("asset_class", "Unknown")
                results[t] = asset_data

            if verbose and (idx % 25 == 0 or idx == total):
                print(f"  --> Processed {idx}/{total} assets ({len(results)} valid datasets fetched)")

        return results
