"""Cross-asset and multi-frequency evaluation suite across global asset classes."""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import numpy as np
import torch

from klint.models.klint_32m import Klint32M
from klint.tokenizer.factor_tokenizer import FactorTokenizer
from klint.tokenizer.geometric_decoder import GeometricDecoder
from klint.benchmark.data_fetcher import MultiAssetDataFetcher
from klint.eval.core_forecasting import calc_ic_rankic
from klint.data.factors import FactorDecomposer


@dataclass
class AssetFrequencyResult:
    """Evaluation result for an asset at a specific frequency and volume regime."""
    ticker: str
    asset_class: str
    frequency: str
    bars_count: int
    uses_volume: bool
    directional_accuracy_pct: float
    ic: float
    rank_ic: float
    mae_pct: float
    annualized_sharpe: float


class CrossAssetEvaluator:
    """Evaluates Klint across diverse asset domains, sampling frequencies, and OHLC vs OHLCV."""

    DEFAULT_UNIVERSE = {
        "Equities": ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA"],
        "Indices": ["^GSPC", "^IXIC", "^NSEI"],
        "Crypto": ["BTC-USD", "ETH-USD", "SOL-USD"],
        "FX": ["EURUSD=X", "GBPUSD=X", "USDJPY=X"],
        "Commodities": ["GLD", "USO", "SLV"],
    }

    def __init__(
        self,
        model: Klint32M,
        tokenizer: FactorTokenizer,
        decoder: GeometricDecoder,
        data_fetcher: MultiAssetDataFetcher,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer.to(device)
        self.tokenizer.eval()
        self.decoder = decoder.to(device)
        self.fetcher = data_fetcher
        self.device = device
        self.decomposer = FactorDecomposer()

    @torch.no_grad()
    def evaluate_single_series(
        self,
        ohlcv: np.ndarray,
        ticker: str,
        asset_class: str,
        frequency: str = "1d",
        uses_volume: bool = True,
        context_bars: int = 64,
        eval_bars: int = 150,
        horizon: int = 5,
    ) -> Optional[AssetFrequencyResult]:
        """Evaluates model performance on a single OHLC(V) time series."""
        N = len(ohlcv)
        if N < context_bars + eval_bars + horizon:
            return None

        # If volume is disabled, mask volume to 0 (tests OHLC-only robustness)
        eval_ohlcv = ohlcv.copy()
        if not uses_volume:
            eval_ohlcv[:, 4] = 0.0

        factors = self.decomposer.decompose(eval_ohlcv)
        p_tensor = torch.from_numpy(factors.price_path).unsqueeze(0).to(self.device)
        r_tensor = torch.from_numpy(factors.range_shape).unsqueeze(0).to(self.device)
        a_tensor = torch.from_numpy(factors.activity).unsqueeze(0).to(self.device)

        p_tok, r_tok, a_tok, _ = self.tokenizer.encode(p_tensor, r_tensor, a_tensor)
        interleaved = self.tokenizer.interleave(p_tok, r_tok, a_tok).squeeze(0)

        emb_p = self.tokenizer.rvq_price.layers[0].embedding
        dec_p_weights = self.tokenizer.dec_price(emb_p)
        code_returns = dec_p_weights[:, 1]  # (512,)

        start_bar = N - eval_bars
        actual_returns = []
        pred_returns = []
        strat_returns = []

        fee = 0.0005  # 5 bps

        for b in range(start_bar, N):
            h_start = (b - context_bars) * 3
            h_end = b * 3
            prompt = interleaved[h_start:h_end].unsqueeze(0)

            out = self.model(prompt)
            last_h = out["hidden"][:, -1, :]
            logits_p = self.model.head_price(last_h).squeeze(0)
            probs = torch.softmax(logits_p, dim=-1)

            exp_r = float(torch.sum(probs * code_returns).item())
            real_r = float(factors.price_path[b, 1])

            pred_returns.append(exp_r)
            actual_returns.append(real_r)

            sig = 1.0 if exp_r > fee else (-1.0 if exp_r < -fee else 0.0)
            strat_returns.append(sig * real_r - (abs(sig) * fee))

        pred_returns = np.array(pred_returns)
        actual_returns = np.array(actual_returns)
        strat_returns = np.array(strat_returns)

        ic, rank_ic = calc_ic_rankic(pred_returns, actual_returns)

        valid = (pred_returns != 0) & (actual_returns != 0)
        hit_rate = float(np.mean(np.sign(pred_returns[valid]) == np.sign(actual_returns[valid])) * 100.0) if np.any(valid) else 50.0

        mae = float(np.mean(np.abs(pred_returns - actual_returns)) * 100.0)

        mean_r = np.mean(strat_returns)
        std_r = np.std(strat_returns) + 1e-8
        # Annualization factor based on frequency
        ann_factor = np.sqrt(252) if frequency == "1d" else (np.sqrt(252 * 7) if frequency == "1h" else np.sqrt(252 * 7 * 12))
        sharpe = float((mean_r / std_r) * ann_factor)

        return AssetFrequencyResult(
            ticker=ticker,
            asset_class=asset_class,
            frequency=frequency,
            bars_count=N,
            uses_volume=uses_volume,
            directional_accuracy_pct=hit_rate,
            ic=ic,
            rank_ic=rank_ic,
            mae_pct=mae,
            annualized_sharpe=sharpe,
        )

    def run_cross_asset_matrix(
        self,
        universe: Optional[Dict[str, List[str]]] = None,
        frequencies: Optional[List[str]] = None,
        evaluate_ohlc_only: bool = True,
        max_assets_per_class: int = 3,
    ) -> List[AssetFrequencyResult]:
        """Runs evaluation across domain classes and frequencies."""
        target_universe = universe or self.DEFAULT_UNIVERSE
        target_freqs = frequencies or ["1d"]
        results: List[AssetFrequencyResult] = []

        print(f"\nRunning Cross-Asset Evaluation Matrix across {len(target_universe)} asset classes...")
        for asset_class, tickers in target_universe.items():
            for ticker in tickers[:max_assets_per_class]:
                for freq in target_freqs:
                    period = "1y" if freq == "1d" else ("3mo" if freq == "1h" else "1mo")
                    asset_data = self.fetcher.fetch_asset(ticker, period=period, interval=freq)
                    if asset_data is None:
                        continue
                    ohlcv = asset_data["ohlcv"]

                    # 1. Standard OHLCV
                    res_ohlcv = self.evaluate_single_series(ohlcv, ticker, asset_class, frequency=freq, uses_volume=True)
                    if res_ohlcv is not None:
                        results.append(res_ohlcv)
                        print(f"  --> {ticker:10s} ({asset_class:11s} | {freq:2s} | OHLCV) | Hit: {res_ohlcv.directional_accuracy_pct:5.1f}% | IC: {res_ohlcv.ic:+.3f} | Sharpe: {res_ohlcv.annualized_sharpe:+.2f}")

                    # 2. OHLC Only (No Volume)
                    if evaluate_ohlc_only:
                        res_ohlc = self.evaluate_single_series(ohlcv, ticker, asset_class, frequency=freq, uses_volume=False)
                        if res_ohlc is not None:
                            results.append(res_ohlc)
                            print(f"  --> {ticker:10s} ({asset_class:11s} | {freq:2s} | OHLC ) | Hit: {res_ohlc.directional_accuracy_pct:5.1f}% | IC: {res_ohlc.ic:+.3f} | Sharpe: {res_ohlc.annualized_sharpe:+.2f}")

        return results
