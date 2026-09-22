"""Klint-32M Comprehensive Foundation Model Evaluation Suite."""

from klint.eval.bundle_loader import get_or_create_release_bundle
from klint.eval.core_forecasting import (
    CoreForecastingEvaluator,
    CoreForecastingResult,
    HorizonForecastMetric,
    ReturnForecastMetric,
    VolatilityForecastMetric,
    calc_ic_rankic,
)
from klint.eval.cross_asset_eval import CrossAssetEvaluator, AssetFrequencyResult
from klint.eval.id_vs_ood import IDvsOODEvaluator, IDvsOODResult
from klint.eval.synthetic_fidelity import SyntheticFidelityEvaluator, SyntheticFidelityResult
from klint.eval.distributional_analysis import DistributionalAnalyzer, DistributionalComparisonResult, DistributionalMoments
from klint.eval.tstr_evaluator import TSTREvaluator, TSTRResult
from klint.eval.portfolio_simulation import PortfolioSimulator, PortfolioSimulationResult, PortfolioFeeLevelResult
from klint.eval.walk_forward_expanding import ExpandingWalkForwardValidator, ExpandingWalkForwardResult, ExpandingWindowFold
from klint.eval.regime_evaluation import RegimeConditionedEvaluator, RegimeConditionedResult, RegimePerformanceMetric
from klint.eval.eval_plotter import ComprehensiveEvalPlotter

__all__ = [
    "get_or_create_release_bundle",
    "CoreForecastingEvaluator",
    "CoreForecastingResult",
    "HorizonForecastMetric",
    "ReturnForecastMetric",
    "VolatilityForecastMetric",
    "calc_ic_rankic",
    "CrossAssetEvaluator",
    "AssetFrequencyResult",
    "IDvsOODEvaluator",
    "IDvsOODResult",
    "SyntheticFidelityEvaluator",
    "SyntheticFidelityResult",
    "DistributionalAnalyzer",
    "DistributionalComparisonResult",
    "DistributionalMoments",
    "TSTREvaluator",
    "TSTRResult",
    "PortfolioSimulator",
    "PortfolioSimulationResult",
    "PortfolioFeeLevelResult",
    "ExpandingWalkForwardValidator",
    "ExpandingWalkForwardResult",
    "ExpandingWindowFold",
    "RegimeConditionedEvaluator",
    "RegimeConditionedResult",
    "RegimePerformanceMetric",
    "ComprehensiveEvalPlotter",
]
