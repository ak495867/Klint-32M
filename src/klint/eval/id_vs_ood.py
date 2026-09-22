"""In-Distribution vs Out-Of-Distribution (ID vs OOD) generalization benchmark."""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import numpy as np

from klint.eval.cross_asset_eval import CrossAssetEvaluator, AssetFrequencyResult
from klint.benchmark.data_fetcher import MultiAssetDataFetcher


@dataclass
class IDvsOODResult:
    """Quantitative comparison of In-Distribution vs Out-Of-Distribution performance."""
    id_assets: List[str]
    ood_assets: List[str]
    id_mean_ic: float
    ood_mean_ic: float
    id_mean_rank_ic: float
    ood_mean_rank_ic: float
    id_mean_hit_rate: float
    ood_mean_hit_rate: float
    id_mean_sharpe: float
    ood_mean_sharpe: float
    retention_ratio_ic: float         # ood_ic / id_ic
    retention_ratio_hit_rate: float   # ood_hit / id_hit
    generalization_conclusion: str    # Structural Invariant vs Asset Memorization


class IDvsOODEvaluator:
    """Evaluates whether Klint-32M learns universal market invariants or memorizes training assets."""

    DEFAULT_ID_ASSETS = ["SOL-USD", "BTC-USD", "ETH-USD"]
    DEFAULT_OOD_ASSETS = ["^NSEI", "^GDAXI", "EURUSD=X", "GLD", "TSLA"]

    def __init__(self, cross_evaluator: CrossAssetEvaluator, data_fetcher: MultiAssetDataFetcher):
        self.evaluator = cross_evaluator
        self.fetcher = data_fetcher

    def evaluate_id_vs_ood(
        self,
        id_tickers: Optional[List[str]] = None,
        ood_tickers: Optional[List[str]] = None,
        period: str = "1y",
        interval: str = "1d",
    ) -> IDvsOODResult:
        """Compares ID and OOD cohorts."""
        id_list = id_tickers or self.DEFAULT_ID_ASSETS
        ood_list = ood_tickers or self.DEFAULT_OOD_ASSETS

        print("\nEvaluating In-Distribution (ID) Cohort...")
        id_results: List[AssetFrequencyResult] = []
        for t in id_list:
            d = self.fetcher.fetch_asset(t, period=period, interval=interval)
            if d is not None:
                res = self.evaluator.evaluate_single_series(d["ohlcv"], t, "ID_Crypto", frequency=interval)
                if res is not None:
                    id_results.append(res)
                    print(f"  --> ID:  {t:10s} | Hit Rate: {res.directional_accuracy_pct:5.1f}% | IC: {res.ic:+.3f} | Sharpe: {res.annualized_sharpe:+.2f}")

        print("\nEvaluating Out-of-Distribution (OOD) Cohort...")
        ood_results: List[AssetFrequencyResult] = []
        for t in ood_list:
            d = self.fetcher.fetch_asset(t, period=period, interval=interval)
            if d is not None:
                res = self.evaluator.evaluate_single_series(d["ohlcv"], t, "OOD_Global", frequency=interval)
                if res is not None:
                    ood_results.append(res)
                    print(f"  --> OOD: {t:10s} | Hit Rate: {res.directional_accuracy_pct:5.1f}% | IC: {res.ic:+.3f} | Sharpe: {res.annualized_sharpe:+.2f}")

        id_ic = float(np.mean([r.ic for r in id_results])) if id_results else 0.0
        ood_ic = float(np.mean([r.ic for r in ood_results])) if ood_results else 0.0

        id_rank_ic = float(np.mean([r.rank_ic for r in id_results])) if id_results else 0.0
        ood_rank_ic = float(np.mean([r.rank_ic for r in ood_results])) if ood_results else 0.0

        id_hit = float(np.mean([r.directional_accuracy_pct for r in id_results])) if id_results else 50.0
        ood_hit = float(np.mean([r.directional_accuracy_pct for r in ood_results])) if ood_results else 50.0

        id_sh = float(np.mean([r.annualized_sharpe for r in id_results])) if id_results else 0.0
        ood_sh = float(np.mean([r.annualized_sharpe for r in ood_results])) if ood_results else 0.0

        ret_ic = ood_ic / (id_ic + 1e-8) if id_ic != 0 else 1.0
        ret_hit = ood_hit / (id_hit + 1e-8) if id_hit != 0 else 1.0

        if ood_hit >= 52.0 and ret_hit >= 0.90:
            conclusion = "Structural Invariant Learning: Model transfers alpha robustly to unseen markets."
        elif ood_hit >= 50.0:
            conclusion = "Moderate Generalization: Retains baseline predictability with slight OOD degradation."
        else:
            conclusion = "Domain Overfitting: Performance collapses when transferred out of distribution."

        return IDvsOODResult(
            id_assets=id_list,
            ood_assets=ood_list,
            id_mean_ic=id_ic,
            ood_mean_ic=ood_ic,
            id_mean_rank_ic=id_rank_ic,
            ood_mean_rank_ic=ood_rank_ic,
            id_mean_hit_rate=id_hit,
            ood_mean_hit_rate=ood_hit,
            id_mean_sharpe=id_sh,
            ood_mean_sharpe=ood_sh,
            retention_ratio_ic=float(ret_ic),
            retention_ratio_hit_rate=float(ret_hit),
            generalization_conclusion=conclusion,
        )
