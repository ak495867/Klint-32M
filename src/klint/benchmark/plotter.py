"""Generates publication-quality charts for multi-asset drawdown, accuracy, Sharpe, and rankings."""

import os
from typing import List, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from klint.benchmark.evaluator import AssetBenchmarkResult


class BenchmarkPlotter:
    """
    Generates and saves publication-quality visualization charts.
    """
    def __init__(self, output_dir: str = "benchmarks/plots"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        # Clean styling
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    def plot_drawdown_curves(self, results: List[AssetBenchmarkResult]) -> str:
        """Plots cumulative equity curve and underwater drawdown curve."""
        min_len = min(len(r.equity_curve) for r in results)
        stacked_eq = np.stack([r.equity_curve[:min_len] for r in results], axis=0)
        port_eq = np.mean(stacked_eq, axis=0)

        peaks = np.maximum.accumulate(port_eq)
        drawdown = (port_eq - peaks) / peaks * 100.0

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [2.5, 1.5]})

        # Equity Curve
        ax1.plot(port_eq, color="#2563eb", linewidth=2.2, label=f"Klint-32M Portfolio ({len(results)} Assets)")
        ax1.axhline(1.0, color="#94a3b8", linestyle="--", linewidth=1.0)
        ax1.set_title(f"Multi-Asset Portfolio Equity Curve ({len(results)} Global Assets)", fontsize=14, fontweight="bold")
        ax1.set_ylabel("Cumulative Wealth ($1.00 base)", fontsize=11)
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)

        # Underwater Drawdown
        ax2.plot(drawdown, color="#dc2626", linewidth=1.8, label="Underwater Drawdown (%)")
        ax2.fill_between(range(len(drawdown)), drawdown, 0, color="#fca5a5", alpha=0.5)
        ax2.set_title("Portfolio Drawdown Dynamics", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Drawdown (%)", fontsize=11)
        ax2.set_xlabel("Out-of-Sample Evaluation Steps", fontsize=11)
        ax2.set_ylim(min(np.min(drawdown) * 1.15, -2.0), 1.0)
        ax2.legend(loc="lower left")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, "drawdown_equity_curves.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        return save_path

    def plot_accuracy_distribution(self, results: List[AssetBenchmarkResult]) -> str:
        """Plots distribution of directional accuracy across assets vs 50% random walk."""
        accuracies = [r.directional_accuracy for r in results]
        mean_acc = float(np.mean(accuracies))
        median_acc = float(np.median(accuracies))
        pct_above_50 = float(np.mean(np.array(accuracies) > 50.0)) * 100.0

        fig, ax = plt.subplots(figsize=(10, 6))
        n, bins, patches = ax.hist(accuracies, bins=25, color="#3b82f6", edgecolor="#1e3a8a", alpha=0.8)

        ax.axvline(50.0, color="#dc2626", linestyle="--", linewidth=2.0, label="Random Walk (50.0%)")
        ax.axvline(mean_acc, color="#10b981", linestyle="-", linewidth=2.2, label=f"Mean Accuracy ({mean_acc:.2f}%)")

        ax.set_title(f"Directional Prediction Accuracy Distribution ({len(results)} Assets)", fontsize=14, fontweight="bold")
        ax.set_xlabel("Directional Hit Rate / Accuracy (%)", fontsize=11)
        ax.set_ylabel("Asset Count", fontsize=11)

        # Stats text box
        stats_text = (
            f"Total Assets: {len(results)}\n"
            f"Mean: {mean_acc:.2f}%\n"
            f"Median: {median_acc:.2f}%\n"
            f"Assets > 50%: {pct_above_50:.1f}%"
        )
        ax.text(
            0.03, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="#cbd5e1")
        )

        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, "directional_accuracy_dist.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        return save_path

    def plot_sharpe_rankings(self, results: List[AssetBenchmarkResult], top_n: int = 25) -> str:
        """Plots ranked horizontal bar chart of top performing assets by Sharpe ratio."""
        sorted_res = sorted(results, key=lambda x: x.annualized_sharpe, reverse=True)[:top_n]
        tickers = [r.ticker for r in sorted_res][::-1]
        sharpes = [r.annualized_sharpe for r in sorted_res][::-1]
        classes = [r.asset_class for r in sorted_res][::-1]

        class_colors = {
            "Equities": "#3b82f6",
            "ETFs": "#8b5cf6",
            "Crypto": "#f59e0b",
            "Commodities": "#10b981",
            "FX": "#ec4899",
            "Fixed_Income": "#64748b",
        }
        bar_colors = [class_colors.get(c, "#3b82f6") for c in classes]

        fig, ax = plt.subplots(figsize=(10, 9))
        bars = ax.barh(tickers, sharpes, color=bar_colors, edgecolor="#334155", alpha=0.85)

        ax.axvline(0.0, color="#475569", linestyle="-", linewidth=1.0)
        ax.axvline(1.0, color="#10b981", linestyle="--", linewidth=1.2, label="Sharpe = 1.0 (Good)")
        ax.axvline(2.0, color="#059669", linestyle="--", linewidth=1.2, label="Sharpe = 2.0 (Excellent)")

        ax.set_title(f"Top {top_n} Assets Ranked by Annualized Sharpe Ratio", fontsize=14, fontweight="bold")
        ax.set_xlabel("Annualized Sharpe Ratio", fontsize=11)
        ax.set_ylabel("Ticker", fontsize=11)

        # Legend for asset classes
        legend_handles = [
            mpatches.Patch(color=col, label=cls_name)
            for cls_name, col in class_colors.items()
            if cls_name in classes
        ]
        ax.legend(handles=legend_handles, title="Asset Class", loc="lower right")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, "asset_sharpe_ranking.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        return save_path

    def plot_asset_class_comparison(self, results: List[AssetBenchmarkResult]) -> str:
        """Plots 2x2 comparison grid across asset classes."""
        import pandas as pd
        df = pd.DataFrame([
            {
                "Class": r.asset_class,
                "Accuracy": r.directional_accuracy,
                "Sharpe": r.annualized_sharpe,
                "MaxDrawdown": r.max_drawdown_pct,
                "Return": r.cumulative_return_pct,
            }
            for r in results
        ])

        classes = sorted(df["Class"].unique())
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        metrics = [
            ("Accuracy", "Directional Accuracy (%)", axes[0, 0]),
            ("Sharpe", "Annualized Sharpe Ratio", axes[0, 1]),
            ("MaxDrawdown", "Maximum Drawdown (%)", axes[1, 0]),
            ("Return", "Cumulative Strategy Return (%)", axes[1, 1]),
        ]

        for col, title, ax in metrics:
            data_by_class = [df[df["Class"] == c][col].dropna().values for c in classes]
            ax.boxplot(data_by_class, tick_labels=classes, patch_artist=True,
                       boxprops=dict(facecolor="#93c5fd", color="#1e40af"),
                       medianprops=dict(color="#b91c1c", linewidth=2.0))
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.tick_params(axis="x", rotation=25)
            ax.grid(True, alpha=0.3)

        fig.suptitle("Cross-Asset Class Performance Comparison", fontsize=15, fontweight="bold")
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, "asset_class_comparison.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        return save_path

    def plot_all(self, results: List[AssetBenchmarkResult]) -> Dict[str, str]:
        """Generates and exports all visualization plots."""
        print("\nGenerating Publication-Grade Benchmark Plots...")
        p_dd = self.plot_drawdown_curves(results)
        p_acc = self.plot_accuracy_distribution(results)
        p_shp = self.plot_sharpe_rankings(results)
        p_cls = self.plot_asset_class_comparison(results)

        print(f"  1. Drawdown Curves: {p_dd}")
        print(f"  2. Accuracy Distribution: {p_acc}")
        print(f"  3. Sharpe Rankings: {p_shp}")
        print(f"  4. Asset Class Comparison: {p_cls}")

        return {
            "drawdown": p_dd,
            "accuracy": p_acc,
            "sharpe_ranking": p_shp,
            "class_comparison": p_cls,
        }
