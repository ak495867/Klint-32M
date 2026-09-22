"""Publication-grade visualization plotter for the comprehensive evaluation suite."""

import os
from typing import Dict, List, Any
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.stats as stats

from klint.eval.core_forecasting import CoreForecastingResult
from klint.eval.portfolio_simulation import PortfolioSimulationResult
from klint.eval.distributional_analysis import DistributionalComparisonResult
from klint.eval.tstr_evaluator import TSTRResult
from klint.eval.regime_evaluation import RegimeConditionedResult


class ComprehensiveEvalPlotter:
    """Renders high-resolution publication-quality benchmark figures."""

    def __init__(self, output_dir: str = "results/plots"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        plt.style.use("seaborn-v0_8-darkgrid" if "seaborn-v0_8-darkgrid" in plt.style.available else "default")

    def plot_core_forecasting(self, core_res: CoreForecastingResult) -> str:
        """Plots IC, RankIC, and Directional Accuracy across forecasting horizons."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # 1. Return Forecasting IC & RankIC
        ret_h = [m.horizon for m in core_res.return_metrics]
        ret_ic = [m.ic for m in core_res.return_metrics]
        ret_rank_ic = [m.rank_ic for m in core_res.return_metrics]

        ax1.plot(ret_h, ret_ic, marker="o", color="#3498db", linewidth=2.2, label="Pearson IC")
        ax1.plot(ret_h, ret_rank_ic, marker="s", color="#2ecc71", linewidth=2.2, label="Spearman RankIC")
        ax1.axhline(0.0, color="#e74c3c", linestyle="--", alpha=0.7)
        ax1.set_title(f"Return Forecasting: IC & RankIC vs Horizon ({core_res.ticker})", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Forecast Horizon $H$ (Bars)", fontsize=11)
        ax1.set_ylabel("Correlation Coefficient", fontsize=11)
        ax1.legend(loc="upper right", framealpha=0.9)

        # 2. Price Forecasting Directional Accuracy
        price_h = [m.horizon for m in core_res.price_metrics]
        price_acc = [m.directional_accuracy_pct for m in core_res.price_metrics]

        ax2.plot(price_h, price_acc, marker="^", color="#f1c40f", linewidth=2.5, label="Directional Accuracy (%)")
        ax2.axhline(50.0, color="#e74c3c", linestyle="--", linewidth=1.5, label="Random Walk Baseline (50%)")
        ax2.set_title("Price Forecasting: Directional Hit Rate vs Horizon", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Forecast Horizon $H$ (Bars)", fontsize=11)
        ax2.set_ylabel("Directional Accuracy (%)", fontsize=11)
        ax2.set_ylim(40, max(max(price_acc) + 5, 65))
        ax2.legend(loc="lower right", framealpha=0.9)

        plt.tight_layout()
        path = os.path.join(self.output_dir, "forecasting_horizons_ic_rankic.png")
        fig.savefig(path, dpi=200)
        plt.close(fig)
        return path

    def plot_distributional_stylized_facts(
        self,
        real_ohlcv: np.ndarray,
        synthetic_ohlcv: np.ndarray,
        dist_res: DistributionalComparisonResult,
    ) -> str:
        """Plots 4-panel financial stylized facts: Return distribution, QQ-plot, ACF, and Volatility clustering."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 11))
        r_ret = np.diff(np.log(real_ohlcv[:, 3] + 1e-8))
        s_ret = np.diff(np.log(synthetic_ohlcv[:, 3] + 1e-8))

        # Panel A: Return Distribution KDE
        ax_a = axes[0, 0]
        ax_a.hist(r_ret, bins=50, density=True, alpha=0.45, color="#3498db", label=f"Real Returns (Kurtosis={dist_res.real_moments.kurtosis:.1f})")
        ax_a.hist(s_ret, bins=50, density=True, alpha=0.45, color="#2ecc71", label=f"Synthetic (Kurtosis={dist_res.synthetic_moments.kurtosis:.1f})")
        ax_a.set_title("Panel A: Return Distribution (Fat-Tail Leptokurtosis)", fontsize=11, fontweight="bold")
        ax_a.set_xlabel("Log Return", fontsize=10)
        ax_a.set_ylabel("Density", fontsize=10)
        ax_a.legend(loc="upper right", framealpha=0.9)

        # Panel B: QQ-Plot (Real vs Synthetic)
        ax_b = axes[0, 1]
        percs = np.linspace(1, 99, 100)
        r_quant = np.percentile(r_ret, percs)
        s_quant = np.percentile(s_ret, percs)
        ax_b.scatter(r_quant, s_quant, color="#e67e22", s=25, alpha=0.85, label="Quantile Pairs")
        lims = [min(r_quant.min(), s_quant.min()), max(r_quant.max(), s_quant.max())]
        ax_b.plot(lims, lims, color="#e74c3c", linestyle="--", label="1:1 Identity Reference")
        ax_b.set_title("Panel B: Empirical Quantile-Quantile (QQ) Plot", fontsize=11, fontweight="bold")
        ax_b.set_xlabel("Real Return Quantiles", fontsize=10)
        ax_b.set_ylabel("Synthetic Return Quantiles", fontsize=10)
        ax_b.legend(loc="upper left", framealpha=0.9)

        # Panel C: Raw Return Autocorrelation ACF(r_t)
        ax_c = axes[1, 0]
        lags = np.arange(1, 11)
        r_acf_raw = [stats.pearsonr(r_ret[:-k], r_ret[k:])[0] for k in lags]
        s_acf_raw = [stats.pearsonr(s_ret[:-k], s_ret[k:])[0] for k in lags]
        ax_c.plot(lags, r_acf_raw, marker="o", color="#3498db", label="Real $ACF(r_t)$")
        ax_c.plot(lags, s_acf_raw, marker="s", color="#2ecc71", label="Synthetic $ACF(r_t)$")
        ax_c.axhline(0.0, color="#7f8c8d", linestyle="--")
        ax_c.set_title("Panel C: Raw Return Autocorrelation (Weak Predictability)", fontsize=11, fontweight="bold")
        ax_c.set_xlabel("Lag $k$ (Bars)", fontsize=10)
        ax_c.set_ylabel("Correlation", fontsize=10)
        ax_c.set_ylim(-0.15, 0.25)
        ax_c.legend(loc="upper right", framealpha=0.9)

        # Panel D: Volatility Clustering ACF(|r_t|)
        ax_d = axes[1, 1]
        r_abs = np.abs(r_ret)
        s_abs = np.abs(s_ret)
        r_acf_abs = [stats.pearsonr(r_abs[:-k], r_abs[k:])[0] for k in lags]
        s_acf_abs = [stats.pearsonr(s_abs[:-k], s_abs[k:])[0] for k in lags]
        ax_d.plot(lags, r_acf_abs, marker="o", color="#3498db", linewidth=2.0, label="Real $ACF(|r_t|)$")
        ax_d.plot(lags, s_acf_abs, marker="s", color="#2ecc71", linewidth=2.0, label="Synthetic $ACF(|r_t|)$")
        ax_d.set_title("Panel D: Volatility Clustering $ACF(|r_t|)$ (ARCH/GARCH Effect)", fontsize=11, fontweight="bold")
        ax_d.set_xlabel("Lag $k$ (Bars)", fontsize=10)
        ax_d.set_ylabel("Absolute Return Correlation", fontsize=10)
        ax_d.legend(loc="upper right", framealpha=0.9)

        plt.tight_layout()
        path = os.path.join(self.output_dir, "distributional_qq_acf_pacf.png")
        fig.savefig(path, dpi=200)
        plt.close(fig)
        return path

    def plot_portfolio_stress(self, port_res: PortfolioSimulationResult) -> str:
        """Plots Top-K portfolio equity curves across transaction fee levels."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # 1. Equity Curves
        colors = ["#2ecc71", "#3498db", "#9b59b6", "#e67e22", "#e74c3c", "#34495e", "#7f8c8d"]
        for idx, r in enumerate(port_res.results_by_fee):
            c = colors[idx % len(colors)]
            ax1.plot(r.equity_curve, color=c, linewidth=2.0, label=f"{r.fee_bps:g} bps (Sharpe: {r.annualized_sharpe:+.2f})")

        ax1.axhline(1.0, color="#7f8c8d", linestyle=":", alpha=0.8)
        ax1.set_title(f"Top-{port_res.top_k} Portfolio Equity Curves Across Trading Friction", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Trading Periods", fontsize=11)
        ax1.set_ylabel("Cumulative Wealth ($W_t / W_0$)", fontsize=11)
        ax1.legend(loc="upper left", framealpha=0.9, fontsize=9)

        # 2. Sharpe vs Fee Friction
        fees = [r.fee_bps for r in port_res.results_by_fee]
        sharpes = [r.annualized_sharpe for r in port_res.results_by_fee]
        ax2.plot(fees, sharpes, marker="o", color="#3498db", linewidth=2.5, markersize=6, label="Strategy Sharpe")
        ax2.axhline(0.0, color="#e74c3c", linestyle="--", label="Zero-Sharpe Threshold")
        ax2.axhline(port_res.benchmark_sharpe, color="#f1c40f", linestyle=":", label=f"1/N Benchmark Sharpe ({port_res.benchmark_sharpe:+.2f})")

        if port_res.breakeven_fee_bps is not None:
            ax2.axvline(port_res.breakeven_fee_bps, color="#e67e22", linestyle=":", linewidth=2.0,
                        label=f"Break-Even: {port_res.breakeven_fee_bps:.1f} bps")

        ax2.set_title("Portfolio Sharpe Decay vs Transaction Fee Grid", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Friction per Trade (Basis Points)", fontsize=11)
        ax2.set_ylabel("Annualized Sharpe Ratio", fontsize=11)
        ax2.legend(loc="upper right", framealpha=0.9)

        plt.tight_layout()
        path = os.path.join(self.output_dir, "portfolio_equity_curves.png")
        fig.savefig(path, dpi=200)
        plt.close(fig)
        return path

    def plot_regimes(self, regime_res: RegimeConditionedResult) -> str:
        """Plots performance across the 7 market regimes."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        regimes = [m.regime for m in regime_res.regimes]
        sharpes = [m.annualized_sharpe for m in regime_res.regimes]
        hit_rates = [m.directional_accuracy_pct for m in regime_res.regimes]

        colors = ["#2ecc71" if s > 0 else "#e74c3c" for s in sharpes]

        # 1. Sharpe by regime
        ax1.barh(regimes, sharpes, color=colors, alpha=0.85, edgecolor="#2c3e50")
        ax1.axvline(0.0, color="#ffffff", linestyle="-", linewidth=1.0)
        ax1.set_title("Strategy Sharpe Ratio Across 7 Market Regimes", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Annualized Sharpe Ratio", fontsize=11)
        ax1.invert_yaxis()

        # 2. Hit Rate by regime
        ax2.barh(regimes, hit_rates, color="#3498db", alpha=0.85, edgecolor="#2c3e50")
        ax2.axvline(50.0, color="#e74c3c", linestyle="--", linewidth=1.5, label="Random Walk Baseline (50%)")
        ax2.set_title("Directional Hit Rate (%) Across 7 Market Regimes", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Hit Rate (%)", fontsize=11)
        ax2.set_xlim(35, max(max(hit_rates) + 5, 65))
        ax2.legend(loc="lower right", framealpha=0.9)
        ax2.invert_yaxis()

        plt.tight_layout()
        path = os.path.join(self.output_dir, "regime_radar_breakdown.png")
        fig.savefig(path, dpi=200)
        plt.close(fig)
        return path

    def plot_tstr(self, tstr_res: TSTRResult) -> str:
        """Plots downstream utility comparison between TRTR and TSTR."""
        fig, ax = plt.subplots(figsize=(8, 5))
        categories = ["Pearson IC", "Spearman RankIC"]
        trtr_vals = [tstr_res.trtr_ic, tstr_res.trtr_rank_ic]
        tstr_vals = [tstr_res.tstr_ic, tstr_res.tstr_rank_ic]

        x = np.arange(len(categories))
        width = 0.35

        ax.bar(x - width/2, trtr_vals, width, label="Train Real, Test Real (TRTR)", color="#3498db", alpha=0.85)
        ax.bar(x + width/2, tstr_vals, width, label="Train Synthetic, Test Real (TSTR)", color="#2ecc71", alpha=0.85)

        ax.axhline(0.0, color="#7f8c8d", linestyle="-")
        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=11, fontweight="bold")
        ax.set_ylabel("Correlation Coefficient on Unseen Real Data", fontsize=11)
        ax.set_title(f"TSTR Utility Benchmark: Downstream Model Transfer (Retention: {tstr_res.utility_retention_pct:.1f}%)",
                     fontsize=12, fontweight="bold")
        ax.legend(loc="upper right", framealpha=0.9)

        plt.tight_layout()
        path = os.path.join(self.output_dir, "tstr_comparison_chart.png")
        fig.savefig(path, dpi=200)
        plt.close(fig)
        return path
