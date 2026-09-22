"""Institutional Stress Testing & Validation Engine for Klint-32M Foundation Model."""

import os
import sys

# Ensure src is in python path
sys.path.insert(0, os.path.abspath("src"))
sys.path.insert(0, os.path.abspath("../src"))
if os.path.exists("/content/Klint-32M/src"):
    sys.path.insert(0, "/content/Klint-32M/src")

import argparse
import numpy as np
import pandas as pd
import torch
from typing import Dict, Any, List

from klint.models.klint_32m import Klint32M, KlintConfig
from klint.tokenizer.factor_tokenizer import FactorTokenizer
from klint.benchmark.evaluator import MultiAssetEvaluator
from klint.benchmark.data_fetcher import DataFetcher
from klint.stress_test.monte_carlo import MonteCarloEngine
from klint.stress_test.cost_sensitivity import CostSensitivityEngine
from klint.stress_test.walk_forward import WalkForwardEngine
from klint.stress_test.oneshot_generalization import OneShotGeneralizationEngine
from klint.stress_test.stress_plotter import StressPlotter


def load_model_and_tokenizer(
    checkpoint_path: str,
    tokenizer_path: str,
    device: str = "cpu",
):
    """Loads Klint-32M and Factor Tokenizer checkpoints safely."""
    print(f"Loading Klint-32M from: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "config" in state:
        cfg = state["config"]
    else:
        cfg = KlintConfig()

    model = Klint32M(cfg).to(device)
    weights = state.get("model_state", state.get("model", state))
    model.load_state_dict(weights, strict=False)
    model.eval()

    print(f"Loading Factor Tokenizer from: {tokenizer_path}")
    tok_state = torch.load(tokenizer_path, map_location=device, weights_only=False)
    tokenizer = FactorTokenizer().to(device)
    tok_weights = tok_state.get("tokenizer_state", tok_state.get("model_state", tok_state))
    tokenizer.load_state_dict(tok_weights, strict=False)
    tokenizer.eval()

    return model, tokenizer


def run_stress_test_suite(
    checkpoint: str = "checkpoints/klint_32m_best.pt",
    tokenizer_checkpoint: str = "checkpoints/tokenizer_best.pt",
    data_path: str = "data/SOL.npy",
    ticker: str = "SOL",
    tests: str = "all",
    monte_carlo_runs: int = 2000,
    walkforward_folds: int = 5,
    output_dir: str = "benchmarks/stress_tests",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    fee_bps: float = 5.0,
    context_bars: int = 64,
    eval_bars: int = 250,
):
    """Executes the full institutional quantitative stress testing suite."""
    reports_dir = os.path.join(output_dir, "reports")
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    print("=" * 70)
    print(f" KLINT-32M INSTITUTIONAL STRESS TESTING SUITE ({device.upper()})")
    print("=" * 70)

    # 1. Load Architecture
    model, tokenizer = load_model_and_tokenizer(checkpoint, tokenizer_checkpoint, device=device)
    evaluator = MultiAssetEvaluator(model, tokenizer, device=device, fee_bps=fee_bps)
    plotter = StressPlotter(output_dir=plots_dir)

    # 2. Ingest Primary Data & Generate Baseline Out-Of-Sample Predictions
    print(f"\nIngesting primary baseline data: {data_path} ({ticker})...")
    if os.path.exists(data_path):
        ohlcv = np.load(data_path)
    else:
        # Fallback to fetcher
        fetcher = DataFetcher(cache_dir="data/yfinance_cache")
        asset_d = fetcher.fetch_asset_data(ticker, period="1y", interval="1d")
        if asset_d is None:
            raise FileNotFoundError(f"Could not load data for {ticker} from {data_path} or yfinance.")
        ohlcv = asset_d["ohlcv"]

    base_data = {"ticker": ticker, "name": f"{ticker} Baseline", "asset_class": "Crypto", "ohlcv": ohlcv}
    print(f"Running causal out-of-sample simulation on {len(ohlcv):,} bars...")
    base_res = evaluator.evaluate_asset(base_data, context_bars=context_bars, eval_horizon=eval_bars)
    if base_res is None:
        raise RuntimeError("Failed to evaluate baseline asset.")

    # Reconstruct strategy returns
    equity = base_res.equity_curve
    strat_returns = np.diff(equity) / equity[:-1]
    # Invert back to get predicted and actual return series
    factors = evaluator.decomposer.decompose(ohlcv)
    N = len(ohlcv)
    eval_len = len(strat_returns)
    actual_returns = np.array([float(factors.price_path[b, 1]) for b in range(N - eval_len, N)])
    pred_returns = (strat_returns + (np.abs(np.sign(strat_returns)) * (fee_bps / 10000.0))) / (np.sign(strat_returns) + 1e-8)
    pred_returns = np.nan_to_num(pred_returns, nan=0.0, posinf=0.0, neginf=0.0)

    active_tests = [t.strip().lower() for t in tests.split(",")]
    run_all = "all" in active_tests

    # =========================================================================
    # Test 1: Monte Carlo Permutation & Resampling
    # =========================================================================
    if run_all or "mc" in active_tests or "monte_carlo" in active_tests:
        print("\n" + "-" * 70)
        print(f" TEST 1: MONTE CARLO PERMUTATION & BLOCK BOOTSTRAP ({monte_carlo_runs:,} PATHS)")
        print("-" * 70)
        mc_engine = MonteCarloEngine(seed=42)
        mc_res = mc_engine.run_full_stress_test(strat_returns, num_paths=monte_carlo_runs)

        # Export CSVs
        mc_summary = pd.DataFrame([{
            "ticker": ticker,
            "paths_simulated": mc_res.num_paths,
            "observed_sharpe": mc_res.observed_sharpe,
            "permutation_p_value_sharpe": mc_res.p_value_sharpe,
            "permutation_p_value_return": mc_res.p_value_return,
            "var_95_pct": mc_res.var_95_pct,
            "var_99_pct": mc_res.var_99_pct,
            "cvar_95_pct": mc_res.cvar_95_pct,
            "cvar_99_pct": mc_res.cvar_99_pct,
            "terminal_equity_p05": mc_res.terminal_equity_p05,
            "terminal_equity_p50_median": mc_res.terminal_equity_p50,
            "terminal_equity_p95": mc_res.terminal_equity_p95,
            "mdd_median_pct": mc_res.mdd_median_pct,
            "mdd_p95_pct_stress": mc_res.mdd_p95_pct,
        }])
        mc_summary_path = os.path.join(reports_dir, "monte_carlo_summary.csv")
        mc_summary.to_csv(mc_summary_path, index=False)

        # Plot
        cone_plot = plotter.plot_monte_carlo_cone(mc_res, ticker=ticker)
        print(f"  --> Permutation p-value:    {mc_res.p_value_sharpe:.4f} ({'Statistically Significant' if mc_res.p_value_sharpe < 0.05 else 'Not Significant'})")
        print(f"  --> Median Terminal Wealth: {mc_res.terminal_equity_p50:.2f}x initial capital")
        print(f"  --> 95% Stress Drawdown:    {mc_res.mdd_p95_pct:.2f}%")
        print(f"  --> 99% 1-Bar VaR:          {mc_res.var_99_pct:.2f}% (Expected Shortfall: {mc_res.cvar_99_pct:.2f}%)")
        print(f"  --> Saved Plot: {cone_plot}")

    # =========================================================================
    # Test 2: Transaction Cost Sensitivity Sweep
    # =========================================================================
    if run_all or "cost" in active_tests or "cost_sensitivity" in active_tests:
        print("\n" + "-" * 70)
        print(" TEST 2: TRANSACTION FRICTION & FEE DECAY SWEEP (0 to 50 bps)")
        print("-" * 70)
        cost_engine = CostSensitivityEngine()
        cost_res = cost_engine.run_sweep(pred_returns, actual_returns)

        # Export CSV
        cost_df = pd.DataFrame([{
            "fee_bps": r.fee_bps,
            "annualized_sharpe": r.annualized_sharpe,
            "annualized_sortino": r.annualized_sortino,
            "cumulative_return_pct": r.cumulative_return_pct,
            "max_drawdown_pct": r.max_drawdown_pct,
            "trades_count": r.trades_count,
            "win_rate_pct": r.win_rate_pct,
            "profit_factor": r.profit_factor,
        } for r in cost_res.results_by_fee])
        cost_csv_path = os.path.join(reports_dir, "cost_sensitivity_sweep.csv")
        cost_df.to_csv(cost_csv_path, index=False)

        # Plot
        cost_plot = plotter.plot_cost_sensitivity(cost_res, ticker=ticker)
        be_str = f"{cost_res.breakeven_fee_bps:.1f} bps" if cost_res.breakeven_fee_bps is not None else "N/A"
        print(f"  --> Critical Break-Even Fee: {be_str}")
        print(f"  --> Sharpe Decay Elasticity: {cost_res.decay_rate_sharpe_per_bps:.4f} Sharpe / bps")
        print(f"  --> Friction Robustness:     {cost_res.robustness_score:.1f} / 100")
        print(f"  --> Saved Plot: {cost_plot}")

    # =========================================================================
    # Test 3: Chronologically Purged & Embargoed Walk-Forward Cross-Validation
    # =========================================================================
    if run_all or "wf" in active_tests or "walkforward" in active_tests:
        print("\n" + "-" * 70)
        print(f" TEST 3: PURGED & EMBARGOED WALK-FORWARD VALIDATION ({walkforward_folds} FOLDS)")
        print("-" * 70)
        wf_engine = WalkForwardEngine(embargo_bars=10)
        wf_res = wf_engine.evaluate_walk_forward(pred_returns, actual_returns, num_folds=walkforward_folds, fee_bps=fee_bps)

        # Export CSV
        wf_df = pd.DataFrame([{
            "fold_idx": f.fold_idx,
            "is_bars": f.is_bars,
            "oos_bars": f.oos_bars,
            "is_sharpe": f.is_sharpe,
            "oos_sharpe": f.oos_sharpe,
            "is_hit_rate": f.is_hit_rate,
            "oos_hit_rate": f.oos_hit_rate,
            "is_cum_return_pct": f.is_cum_return_pct,
            "oos_cum_return_pct": f.oos_cum_return_pct,
            "efficiency_ratio": f.efficiency_ratio,
        } for f in wf_res.folds])
        wf_csv_path = os.path.join(reports_dir, "walk_forward_folds.csv")
        wf_df.to_csv(wf_csv_path, index=False)

        # Plot
        wf_plot = plotter.plot_walk_forward(wf_res, ticker=ticker)
        print(f"  --> Mean In-Sample Sharpe:       {wf_res.mean_is_sharpe:+.2f}")
        print(f"  --> Mean Out-Of-Sample Sharpe:   {wf_res.mean_oos_sharpe:+.2f}")
        print(f"  --> Walk-Forward Efficiency (WFER): {wf_res.walk_forward_efficiency_ratio:.2f} ({'Robust' if wf_res.walk_forward_efficiency_ratio >= 0.5 else 'Degraded'})")
        print(f"  --> Fold Consistency Rate:      {wf_res.consistency_rate_pct:.1f}%")
        print(f"  --> Stitched Continuous Sharpe:  {wf_res.stitched_oos_sharpe:+.2f}")
        print(f"  --> Saved Plot: {wf_plot}")

    # =========================================================================
    # Test 4: One-Shot & Zero-Shot Out-of-Distribution Generalization
    # =========================================================================
    if run_all or "oneshot" in active_tests or "ood" in active_tests:
        print("\n" + "-" * 70)
        print(" TEST 4: ONE-SHOT / ZERO-SHOT OOD GENERALIZATION (10 GLOBAL ASSETS)")
        print("-" * 70)
        fetcher = DataFetcher(cache_dir=os.path.join(output_dir, "yfinance_cache"))
        os_engine = OneShotGeneralizationEngine(model, tokenizer, device=device, fee_bps=fee_bps)
        os_res = os_engine.run_generalization_suite(fetcher, context_bars=context_bars, eval_bars=eval_bars)

        # Export CSVs
        os_df = pd.DataFrame([{
            "ticker": e.ticker,
            "name": e.name,
            "asset_class": e.asset_class,
            "bars_count": e.bars_count,
            "directional_accuracy_pct": e.directional_accuracy,
            "perplexity": e.perplexity,
            "wasserstein_distance": e.wasserstein_dist,
            "annualized_sharpe": e.annualized_sharpe,
            "max_drawdown_pct": e.max_drawdown_pct,
            "win_rate_pct": e.win_rate_pct,
            "invariant_validity_pct": e.invariant_validity_pct,
        } for e in os_res.asset_evaluations])
        os_csv_path = os.path.join(reports_dir, "oneshot_generalization_summary.csv")
        os_df.to_csv(os_csv_path, index=False)

        class_df = pd.DataFrame([
            {"asset_class": k, **v} for k, v in os_res.asset_class_breakdown.items()
        ])
        class_csv_path = os.path.join(reports_dir, "oneshot_asset_class_breakdown.csv")
        class_df.to_csv(class_csv_path, index=False)

        # Plot
        os_plot = plotter.plot_oneshot_generalization(os_res)
        print(f"\n  --> Mean OOD Hit Rate:      {os_res.mean_directional_accuracy:.2f}% (vs 50% random)")
        print(f"  --> Mean Token Perplexity:  {os_res.mean_perplexity:.2f}")
        print(f"  --> Mean OOD Sharpe:        {os_res.mean_sharpe:+.2f}")
        print(f"  --> Mean Wasserstein Dist:  {os_res.mean_wasserstein_dist:.4f}")
        print(f"  --> Saved Plot: {os_plot}")

    print("\n" + "=" * 70)
    print(" ALL INSTITUTIONAL STRESS TESTS COMPLETED SUCCESSFULLY!")
    print(f" CSV Reports saved to: {reports_dir}")
    print(f" High-Resolution Plots saved to: {plots_dir}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Institutional Quantitative Stress Testing Suite for Klint-32M")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/klint_32m_best.pt")
    parser.add_argument("--tokenizer_checkpoint", type=str, default="checkpoints/tokenizer_best.pt")
    parser.add_argument("--data_path", type=str, default="data/SOL.npy")
    parser.add_argument("--ticker", type=str, default="SOL")
    parser.add_argument("--tests", type=str, default="all", help="Comma-separated: all, mc, cost, wf, oneshot")
    parser.add_argument("--monte_carlo_runs", type=int, default=2000)
    parser.add_argument("--walkforward_folds", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="benchmarks/stress_tests")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--fee_bps", type=float, default=5.0)
    parser.add_argument("--context_bars", type=int, default=64)
    parser.add_argument("--eval_bars", type=int, default=200)

    args = parser.parse_args()

    run_stress_test_suite(
        checkpoint=args.checkpoint,
        tokenizer_checkpoint=args.tokenizer_checkpoint,
        data_path=args.data_path,
        ticker=args.ticker,
        tests=args.tests,
        monte_carlo_runs=args.monte_carlo_runs,
        walkforward_folds=args.walkforward_folds,
        output_dir=args.output_dir,
        device=args.device,
        fee_bps=args.fee_bps,
        context_bars=args.context_bars,
        eval_bars=args.eval_bars,
    )


if __name__ == "__main__":
    main()
