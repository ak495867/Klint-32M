# Klint-32M Comprehensive Foundation Model Evaluation Protocol

*Inspired by state-of-the-art financial time-series foundation benchmarks (Kronos, Shi et al., 2025).*

---

## 1. Core Forecasting Benchmark

The primary evaluation assesses whether Klint-32M captures multi-horizon price paths, returns, and realized volatility:

### 1.1 Price Forecasting
Conditioned on historical OHLCV context ($T = 64$ bars), Klint-32M autoregressively forecasts future trajectories across multiple horizons:
$$H \in \{5, 10, 20, 50, 100\} \text{ bars}$$

**Reported Metrics**:
* **Mean Absolute Error (MAE)**: $\frac{1}{H}\sum_{h=1}^H |\hat{P}_{t+h} - P_{t+h}|$
* **Root Mean Squared Error (RMSE)**: $\sqrt{\frac{1}{H}\sum_{h=1}^H (\hat{P}_{t+h} - P_{t+h})^2}$
* **Mean Absolute Percentage Error (MAPE %)**: $\frac{1}{H}\sum_{h=1}^H \left| \frac{\hat{P}_{t+h} - P_{t+h}}{P_{t+h}} \right| \times 100\%$
* **Pearson Information Coefficient (IC)**: $\text{corr}(\hat{P}, P)$
* **Spearman Rank Information Coefficient (RankIC)**: $\text{corr}(\text{rank}(\hat{P}), \text{rank}(P))$
* **Directional Accuracy (%)**: $\frac{1}{H}\sum_{h=1}^H \mathbb{I}\left( \text{sign}(\hat{P}_{t+h} - P_t) = \text{sign}(P_{t+h} - P_t) \right) \times 100\%$

### 1.2 Return Forecasting
Evaluates terminal price returns derived from predicted terminal prices and historical close:
$$r_{t, H} = \frac{\hat{P}_{t+H}}{P_t} - 1 \quad \text{vs} \quad r_{t, H}^{\text{real}} = \frac{P_{t+H}}{P_t} - 1$$

Evaluated across horizons: $H \in \{1, 5, 10, 20, 50\}$ bars.

$$\text{IC} = \text{corr}(\hat{r}, r), \quad \text{RankIC} = \text{corr}(\text{rank}(\hat{r}), \text{rank}(r))$$

### 1.3 Realized Volatility (RV) Forecasting
Evaluates high-frequency realized volatility computed from squared log returns:
$$RV_{t, H} = \sum_{i=1}^{H-1} \left[ \ln(P_{t+i+1}) - \ln(P_{t+i}) \right]^2$$

**Reported Metrics**: MAE, RMSE, $R^2$, Pearson Correlation, Rank Correlation across Low Volatility, Normal Volatility, and High Volatility regimes.

---

## 2. Cross-Asset & Multi-Frequency Matrix

Evaluates cross-market generalizability across 5 major global asset classes:

| Domain | Curated Assets | Evaluated Frequencies |
| :--- | :--- | :--- |
| **Equities** | `SPY`, `QQQ`, `AAPL`, `MSFT`, `NVDA`, `TSLA` | `5m`, `1h`, `1d` |
| **Global Indices** | `^GSPC` (S&P 500), `^IXIC` (NASDAQ), `^NSEI` (NIFTY 50) | `1h`, `1d` |
| **Cryptocurrencies**| `BTC-USD`, `ETH-USD`, `SOL-USD` | `5m`, `1h`, `1d` |
| **Foreign Exchange**| `EURUSD=X`, `GBPUSD=X`, `USDJPY=X` | `1h`, `1d` |
| **Commodities** | `GLD` (Gold), `USO` (Crude Oil), `SLV` (Silver) | `1h`, `1d` |

* **Ablation on Volume**: Evaluated both with full OHLCV and with volume masked to zero (OHLC only) to test robustness on volume-fragmented instruments like FX and decentralized crypto.

---

## 3. In-Distribution vs Out-of-Distribution (ID vs OOD)

Quantifies whether Klint-32M learns universal invariant market structures or memorizes asset-specific trajectories:

* **In-Distribution (ID)**: High-volatility crypto assets matching the training distribution (`SOL-USD`, `BTC-USD`, `ETH-USD`).
* **Out-of-Distribution (OOD)**: Completely unseen markets, countries, and instruments (`^NSEI` / NIFTY 50, `^GDAXI` / DAX, `EURUSD=X`, `GLD`, `TSLA`).

$$\Delta_{\text{OOD}} = \text{Performance}_{\text{ID}} - \text{Performance}_{\text{OOD}}$$
* Target: Retention ratio $\frac{\text{Performance}_{\text{OOD}}}{\text{Performance}_{\text{ID}}} \ge 0.85$.

---

## 4. Synthetic Market Generation Fidelity

Evaluates whether synthetic trajectories synthesized by Klint-32M are statistically indistinguishable from empirical reality:

### 4.1 LSTM Discriminator & Discriminative Score
Trains a 2-layer post-hoc PyTorch LSTM classifier to distinguish Real market sequences (label 1) from Synthetic Klint trajectories (label 0):
$$\text{Discriminative Score} = \left| \text{Accuracy}_{\text{test}} - 0.50 \right|$$
* A score approaching **0.00** (discriminator test accuracy $\sim 50\%$) proves that generated paths are indistinguishable from real market data.

### 4.2 Distribution Divergence
* **Maximum Mean Discrepancy (MMD)** with multi-scale Gaussian RBF kernels:
  $$\text{MMD}^2(P, Q) = \mathbb{E}[k(x, x')] + \mathbb{E}[k(y, y')] - 2\mathbb{E}[k(x, y)]$$
* **2-Wasserstein Distance** on log return sequences.
* **Precision & Recall for Generative Distributions**:
  * Precision: % of synthetic return states supported by empirical market distributions.
  * Recall: % of the empirical market manifold covered by generated scenarios.

---

## 5. Distributional Stylized Facts Verification

Ensures generated paths exhibit the fundamental empirical stylized facts of financial econometrics:
1. **Fat-Tail Leptokurtosis**: Excess kurtosis of returns $> 0$ (heavy tails).
2. **Weak Linear Autocorrelation**: Autocorrelation of raw returns $ACF(r_t) \approx 0$ for lags $k \ge 1$.
3. **Persistent Volatility Clustering**: Autocorrelation of absolute returns $ACF(|r_t|) > 0$ with slow monotonic decay (ARCH/GARCH effect).
4. **Leverage Effect**: Negative cross-correlation between price return and subsequent volatility: $\text{corr}(r_t, \sigma_{t+1}) < 0$.
5. **Empirical QQ Plots**: Linear alignment against empirical quantiles.

---

## 6. Train-on-Synthetic, Test-on-Real (TSTR)

Validates downstream algorithmic utility:
1. Model A (TRTR): Downstream forecaster trained on **Real** market data $\to$ Evaluated on unseen Real test set.
2. Model B (TSTR): Identical downstream forecaster trained strictly on **Synthetic** Klint-32M trajectories $\to$ Evaluated on the same unseen Real test set.

$$\Delta \text{IC} = \text{IC}_{\text{TSTR}} - \text{IC}_{\text{TRTR}}, \quad \Delta \text{RankIC} = \text{RankIC}_{\text{TSTR}} - \text{RankIC}_{\text{TRTR}}$$
* High utility retention ($\ge 80\%$) confirms that Klint synthetic data encodes genuine predictive market dynamics.

---

## 7. Standardized Top-K Investment Portfolio Simulation

* Evaluates a standardized cross-sectional Top-$K$ long portfolio:
  1. Ranks universe of assets by predicted return $\hat{r}_{t, H}$.
  2. Allocates equal weights across Top-$K$ assets with turnover tracking and minimum holding periods.
  3. Computes: CAGR (%), Annualized Return (%), Sharpe Ratio, Sortino Ratio, Maximum Drawdown (MDD %), Calmar Ratio, Annualized Turnover (%), Win Rate (%), Profit Factor, Information Ratio.
* **Transaction Cost Friction Grid**:
  * Sweeps execution friction across $[0, 5, 10, 25, 50, 100, 200]$ basis points to identify the critical break-even fee ($F_{\text{crit}}$).

---

## 8. Multi-Year Chronological Expanding Walk-Forward Validation

Evaluates stability across expanding historical partitions with embargo separation:
* Window 1: Train 2018-2020 $\to$ Test 2021
* Window 2: Train 2018-2021 $\to$ Test 2022
* Window 3: Train 2018-2022 $\to$ Test 2023
* Window 4: Train 2018-2023 $\to$ Test 2024
* Window 5: Train 2018-2024 $\to$ Test 2025/2026

Computes Walk-Forward Efficiency Ratio ($\text{WFER} = \frac{\text{Mean OOS Sharpe}}{\text{Mean IS Sharpe}}$) to detect temporal overfitting.

---

## 9. 7-State Market Regime-Conditioned Evaluation

Segments test observations into 7 canonical financial regimes:
1. **Bull**: Sustained positive drift ($r > +0.5\sigma$)
2. **Bear**: Sustained negative drift ($r < -0.5\sigma$)
3. **Sideways**: Low directional drift ($|r| \le 0.5\sigma$)
4. **High Volatility**: Volatility in top 20th percentile
5. **Low Volatility**: Volatility in bottom 20th percentile
6. **Crash**: Sudden drawdown $> 2.5\sigma$
7. **Recovery**: Rebound within 3 bars following a crash

Reports IC, RankIC, MAE, Sharpe Ratio, and Directional Hit Rate for every regime.

---

## 10. Output Organization (`results/`)

All evaluation outputs are saved to `results/`:
* `results/metrics/`: CSV spreadsheets for every experiment.
* `results/plots/`: High-resolution figures (IC vs Horizon, QQ plots, ACF, Portfolio curves, Regime breakdown).
* `results/reports/`: Master evaluation markdown summary report.
