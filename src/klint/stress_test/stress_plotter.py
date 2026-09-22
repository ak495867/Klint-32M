"""Publication-quality visualization plotter for stress testing and validation metrics."""

import os
from typing import Dict, Any, List
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from klint.stress_test.monte_carlo import MonteCarloResult
from klint.stress_test.cost_sensitivity import CostSensitivityResult
from klint.stress_test.walk_forward import WalkForwardResult
from klint.stress_test.oneshot_generalization import OneShotGeneralizationResult


class StressPlotter:
    """Renders high-resolution publication-quality stress testing figures."""

    def __init__(self, output_dir: str = "benchmarks/stress_tests/plots"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        # Apply clean dark-mode financial styling
        plt.style.use("seaborn-v0_8-darkgrid" if "seaborn-v0_8-darkgrid" in plt.style.available else "default")

    def plot_monte_carlo_cone(self, mc_res: MonteCarloResult, ticker: str = "SOL") -> str:
        """Plots 5th-95th percentile confidence cone and Max Drawdown distribution."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={"width_ratios": [2.2, 1]})

        T = mc_res.path_length
        bars = np.arange(T)

        # 1. Confidence Cone
        p05 = mc_res.percentile_curves["p05"]
        p25 = mc_res.percentile_curves["p25"]
        p50 = mc_res.percentile_curves["p50"]
        p75 = mc_res.percentile_curves["p75"]
        p95 = mc_res.percentile_curves["p95"]

        # Light background sample paths
        for path in mc_res.equity_paths_sample[:35]:
            ax1.plot(bars, path, color="#34495e", alpha=0.15, linewidth=0.8)

        # Percentile ribbons
        ax1.fill_between(bars, p05, p95, color="#3498db", alpha=0.20, label="5th - 95th Percentile Cone")
        ax1.fill_between(bars, p25, p75, color="#2980b9", alpha=0.35, label="25th - 75th Percentile Cone")
        ax1.plot(bars, p50, color="#f1c40f", linewidth=2.2, linestyle="--", label="Median Simulation ($P_{50}$)")

        ax1.axhline(1.0, color="#7f8c8d", linestyle=":", alpha=0.7, label="Initial Capital (1.0)")
        ax1.set_title(f"Klint-32M: Monte Carlo Stationary Block Bootstrap ({mc_res.num_paths:,} Paths) - {ticker}", fontsize=13, fontweight="bold", pad=12)
        ax1.set_xlabel("Time Horizon (Bars)", fontsize=11)
        ax1.set_ylabel("Cumulative Wealth ($W_t / W_0$)", fontsize=11)
        ax1.legend(loc="upper left", framealpha=0.9)

        # Text scorecard inside ax1
        info_text = (
            f"Observed Sharpe: {mc_res.observed_sharpe:+.2f}\n"
            f"Permutation p-val: {mc_res.p_value_sharpe:.4f}\n"
            f"Terminal $P_{{50}}$: {mc_res.terminal_equity_p50:.2f}x\n"
            f"Terminal $P_{{05}}$ (Stress): {mc_res.terminal_equity_p05:.2f}x\n"
            f"99% 1-Bar VaR: {mc_res.var_99_pct:.2f}%\n"
            f"99% Expected Shortfall: {mc_res.cvar_99_pct:.2f}%"
        )
        ax1.text(0.02, 0.04, info_text, transform=ax1.transAxes, fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="#2c3e50", alpha=0.85, edgecolor="#ecf0f1", textcolor="white"))

        # 2. Max Drawdown Distribution
        ax2.hist(mc_res.mdd_distribution, bins=35, color="#e74c3c", alpha=0.75, edgecolor="#c0392b")
        ax2.axvline(mc_res.mdd_median_pct, color="#f1c40f", linestyle="--", linewidth=2.0, label=f"Median MDD ({mc_res.mdd_median_pct:.1f}%)")
        ax2.axvline(mc_res.mdd_p95_pct, color="#c0392b", linestyle="-", linewidth=2.2, label=f"95% Stress MDD ({mc_res.mdd_p95_pct:.1f}%)")
        ax2.axvline(mc_res.observed_max_drawdown_pct, color="#2ecc71", linestyle=":", linewidth=2.2, label=f"Observed ({mc_res.observed_max_drawdown_pct:.1f}%)")

        ax2.set_title("Simulated Max Drawdown Distribution", fontsize=12, fontweight="bold", pad=12)
        ax2.set_xlabel("Maximum Peak-to-Trough Drawdown (%)", fontsize=11)
        ax2.set_ylabel("Frequency", fontsize=11)
        ax2.legend(loc="upper left", framealpha=0.9, fontsize=9)

        plt.tight_layout()
        filepath = os.path.join(self.output_dir, "monte_carlo_confidence_cone.png")
        fig.savefig(filepath, dpi=200)
        plt.close(fig)
        return filepath

    def plot_cost_sensitivity(self, cost_res: CostSensitivityResult, ticker: str = "SOL") -> str:
        """Plots Sharpe decay and cumulative return vs fee friction."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        fees = np.array(cost_res.fee_levels)
        sharpes = np.array([r.annualized_sharpe for r in cost_res.results_by_fee])
        returns = np.array([r.cumulative_return_pct for r in cost_res.results_by_fee])

        # 1. Sharpe vs Fee
        ax1.plot(fees, sharpes, marker="o", color="#3498db", linewidth=2.5, markersize=6, label="Annualized Sharpe")
        ax1.axhline(0.0, color="#e74c3c", linestyle="--", linewidth=1.5, alpha=0.8, label="Zero-Sharpe Threshold")

        if cost_res.breakeven_fee_bps is not None:
            ax1.axvline(cost_res.breakeven_fee_bps, color="#e67e22", linestyle=":", linewidth=2.0,
                        label=f"Break-Even Fee: {cost_res.breakeven_fee_bps:.1f} bps")
            ax1.scatter([cost_res.breakeven_fee_bps], [0.0], color="#e74c3c", s=100, zorder=5)

        ax1.set_title(f"Transaction Cost Friction: Sharpe Decay Curve - {ticker}", fontsize=12, fontweight="bold", pad=10)
        ax1.set_xlabel("Transaction Friction per Trade (Basis Points)", fontsize=11)
        ax1.set_ylabel("Annualized Sharpe Ratio", fontsize=11)
        ax1.legend(loc="upper right", framealpha=0.9)

        # 2. Cumulative Return vs Fee
        ax2.plot(fees, returns, marker="s", color="#2ecc71", linewidth=2.5, markersize=6, label="Net Cumulative Return (%)")
        ax2.axhline(0.0, color="#e74c3c", linestyle="--", linewidth=1.5, alpha=0.8, label="Breakeven (0% Return)")

        if cost_res.profitable_breakeven_bps is not None:
            ax2.axvline(cost_res.profitable_breakeven_bps, color="#e67e22", linestyle=":", linewidth=2.0,
                        label=f"Net PnL Breakeven: {cost_res.profitable_breakeven_bps:.1f} bps")

        ax2.set_title("Net Strategy Return vs Execution Cost", fontsize=12, fontweight="bold", pad=10)
        ax2.set_xlabel("Transaction Friction per Trade (Basis Points)", fontsize=11)
        ax2.set_ylabel("Net Cumulative Return (%)", fontsize=11)
        ax2.legend(loc="upper right", framealpha=0.9)

        plt.tight_layout()
        filepath = os.path.join(self.output_dir, "cost_sensitivity_decay.png")
        fig.savefig(filepath, dpi=200)
        plt.close(fig)
        return filepath

    def plot_walk_forward(self, wf_res: WalkForwardResult, ticker: str = "SOL") -> str:
        """Plots Walk-Forward fold comparisons and continuous OOS equity."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.5))

        folds = [f.fold_idx for f in wf_res.folds]
        is_sh = [f.is_sharpe for f in wf_res.folds]
        oos_sh = [f.oos_sharpe for f in wf_res.folds]

        x = np.arange(len(folds))
        width = 0.35

        # 1. Fold Sharpe Comparison
        ax1.bar(x - width/2, is_sh, width, label="In-Sample (IS) Sharpe", color="#3498db", alpha=0.85)
        ax1.bar(x + width/2, oos_sh, width, label="Out-Of-Sample (OOS) Sharpe", color="#2ecc71", alpha=0.85)

        ax1.axhline(0.0, color="#7f8c8d", linestyle="-", linewidth=1.0)
        ax1.set_xticks(x)
        ax1.set_xticklabels([f"Fold {f}" for f in folds])
        ax1.set_title(f"Walk-Forward Cross-Validation (WFER: {wf_res.walk_forward_efficiency_ratio:.2f}) - {ticker}",
                      fontsize=12, fontweight="bold", pad=10)
        ax1.set_xlabel("Chronological Rolling Fold (with Embargo)", fontsize=11)
        ax1.set_ylabel("Annualized Sharpe Ratio", fontsize=11)
        ax1.legend(loc="upper right", framealpha=0.9)

        # 2. Stitched Out-of-Sample Equity Curve
        oos_equity = wf_res.stitched_oos_equity_curve
        ax2.plot(oos_equity, color="#9b59b6", linewidth=2.2, label="Stitched Continuous OOS Equity")
        ax2.axhline(1.0, color="#7f8c8d", linestyle=":", alpha=0.7)

        ax2.set_title(f"Stitched Out-of-Sample Equity (Sharpe: {wf_res.stitched_oos_sharpe:+.2f} | Return: {wf_res.stitched_oos_cum_return_pct:+.1f}%)",
                      fontsize=12, fontweight="bold", pad=10)
        ax2.set_xlabel("Out-Of-Sample Chronological Bars", fontsize=11)
        ax2.set_ylabel("Cumulative Wealth ($W_t / W_0$)", fontsize=11)
        ax2.legend(loc="upper left", framealpha=0.9)

        plt.tight_layout()
        filepath = os.path.join(self.output_dir, "walkforward_stability.png")
        fig.savefig(filepath, dpi=200)
        plt.close(fig)
        return filepath

    def plot_oneshot_generalization(self, os_res: OneShotGeneralizationResult) -> str:
        """Plots cross-asset class generalization accuracy, perplexity, and Sharpe."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        tickers = [e.ticker for e in os_res.asset_evaluations]
        classes = [e.asset_class for e in os_res.asset_evaluations]
        accs = [e.directional_accuracy for e in os_res.asset_evaluations]
        sharpes = [e.annualized_sharpe for e in os_res.asset_evaluations]

        # Colors by asset class
        class_colors = {
            "Equities": "#3498db",
            "Crypto": "#e67e22",
            "Commodities": "#f1c40f",
            "Forex": "#2ecc71",
            "Fixed Income": "#9b59b6",
        }
        bar_colors = [class_colors.get(c, "#95a5a6") for c in classes]

        y_pos = np.arange(len(tickers))

        # 1. Directional Accuracy
        bars1 = ax1.barh(y_pos, accs, color=bar_colors, alpha=0.85, edgecolor="#2c3e50")
        ax1.axvline(50.0, color="#e74c3c", linestyle="--", linewidth=2.0, label="Random Walk Baseline (50.0%)")
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(tickers, fontsize=10, fontweight="bold")
        ax1.invert_yaxis()
        ax1.set_title("Zero-Shot Directional Hit Rate Across Unseen Assets", fontsize=12, fontweight="bold", pad=10)
        ax1.set_xlabel("Directional Hit Rate (%)", fontsize=11)
        ax1.set_xlim(40, max(max(accs) + 5, 60))
        ax1.legend(loc="lower right", framealpha=0.9)

        # 2. Out-of-sample Sharpe Ratio
        bars2 = ax2.barh(y_pos, sharpes, color=bar_colors, alpha=0.85, edgecolor="#2c3e50")
        ax2.axvline(0.0, color="#e74c3c", linestyle="-", linewidth=1.2, alpha=0.8)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels([])
        ax2.invert_yaxis()
        ax2.set_title("Zero-Shot Strategy Sharpe Ratio (5 bps Fee)", fontsize=12, fontweight="bold", pad=10)
        ax2.set_xlabel("Annualized Sharpe Ratio", fontsize=11)

        # Legend for asset classes
        from matplotlib.patches import Patch
        legend_patches = [Patch(color=col, label=cls) for cls, col in class_colors.items() if cls in classes]
        ax2.legend(handles=legend_patches, loc="lower right", framealpha=0.9, title="Asset Class")

        plt.tight_layout()
        filepath = os.path.join(self.output_dir, "oneshot_generalization_radar.png")
        fig.savefig(filepath, dpi=200)
        plt.close(fig)
        return filepath
