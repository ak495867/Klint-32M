"""Quantitative multi-asset benchmarking suite for Klint foundation models."""

from klint.benchmark.universe import BENCHMARK_UNIVERSE, get_universe_by_class, get_all_tickers
from klint.benchmark.data_fetcher import MultiAssetDataFetcher
from klint.benchmark.evaluator import MultiAssetEvaluator, AssetBenchmarkResult
from klint.benchmark.reporter import BenchmarkReporter
from klint.benchmark.plotter import BenchmarkPlotter

__all__ = [
    "BENCHMARK_UNIVERSE",
    "get_universe_by_class",
    "get_all_tickers",
    "MultiAssetDataFetcher",
    "MultiAssetEvaluator",
    "AssetBenchmarkResult",
    "BenchmarkReporter",
    "BenchmarkPlotter",
]
