# Klint Evaluation Protocol & Institutional Leakage Controls

## 1. Chronological Partitioning & Purged Embargo Windows

Financial time-series data exhibits long-range temporal autocorrelation in volatility, volume, and order flow. Standard random $K$-fold cross-validation results in severe lookahead leakage and catastrophic backtest overfitting.

Klint enforces strict **chronological partitioning** with an **embargo window** between splits:

```text
[ ---- TRAIN SPLIT ---- ] [ EMBARGO ] [ ---- VAL SPLIT ---- ] [ EMBARGO ] [ ---- TEST SPLIT ---- ]
```

* **Embargo Duration ($E$)**: Minimum 1,440 bars (24 hours on 1-minute intraday data) or 20 bars on daily data between any split boundary.
* **Causal Feature Computation**: Feature statistics (e.g., volume EMA) are computed strictly online and causally with zero retrospective normalization.
* **Zero Contamination**: Training samples cannot condition on test or validation factor tokens.

---

## 2. Core Quantitative Benchmark Metrics

Model checkpoints are evaluated against four quantitative dimensions:

### 2.1 Directional Predictive Alpha & Calibration
* **Directional Hit Rate (Accuracy %)**:
  $$\text{Hit Rate} = \frac{1}{T} \sum_{t=1}^T \mathbb{I}\left(\text{sign}(\hat{r}_t) = \text{sign}(r_t)\right) \times 100\%$$
* **Expected Next-Bar Alpha**:
  $$\hat{r}_t = \mathbb{E}_{P(p_t)}[r^{\text{body}}] = \sum_{k=1}^{K_p} P(p_t = k \mid S_{<t}) \cdot \text{Decoder}_p(e_k)$$
* **Null Hypothesis Significance ($p$-value)**:
  Empirical permutation test against randomized trade sequences: $H_0: \text{Sharpe}_{\text{observed}} \le \text{Sharpe}_{\text{luck}}$.

### 2.2 Risk-Adjusted Return Metrics
* **Annualized Sharpe Ratio** (252-day basis):
  $$\text{Sharpe} = \frac{\mathbb{E}[R_{\text{strat}}] - R_f}{\sigma(R_{\text{strat}})} \times \sqrt{252}$$
* **Annualized Sortino Ratio** (downside semi-variance):
  $$\text{Sortino} = \frac{\mathbb{E}[R_{\text{strat}}] - R_f}{\sqrt{\frac{1}{T} \sum_{t=1}^T \min(0, R_{\text{strat}, t})^2}} \times \sqrt{252}$$
* **Maximum Peak-to-Trough Drawdown (MDD %)**:
  $$\text{MDD} = \min_{t \in [1, T]} \left( \frac{W_t - \max_{s \le t} W_s}{\max_{s \le t} W_s} \right) \times 100\%$$
* **Profit Factor**:
  $$\text{Profit Factor} = \frac{\sum \text{Gains}}{\sum |\text{Losses}| + \epsilon}$$

### 2.3 Structural Candle Geometric Integrity
* **Candle Invariant Validity Rate**:
  $$\text{Pass Rate} = \frac{1}{T} \sum_{t=1}^T \mathbb{I}\left(H_t \ge \max(O_t, C_t) \land L_t \le \min(O_t, C_t)\right) \times 100\%$$
  **Benchmark Target**: Guaranteed $100.00\%$ by structural Softplus geometric decoding.

### 2.4 Distributional Fidelity
* **2-Wasserstein Distance** between model return expectation distribution $\mathcal{P}$ and empirical market returns $\mathcal{Q}$:
  $$W_2(\mathcal{P}, \mathcal{Q}) = \left( \inf_{\gamma \in \Pi(\mathcal{P}, \mathcal{Q})} \int |x - y|^2 \, d\gamma(x, y) \right)^{1/2}$$
* **Continuous Ranked Probability Score (CRPS)** evaluated over horizons $h \in \{1, 5, 15, 30, 60\}$ bars.

---

## 3. Institutional Stress Testing Battery

### 3.1 Monte Carlo Stationary Block Bootstrap
To stress-test strategy returns without destroying volatility clustering:
1. Resamples $N = 2,000$ synthetic return paths using circular block bootstrapping with block size $b=5$ bars.
2. Evaluates the terminal wealth distribution across confidence ribbons: $P_5, P_{25}, P_{50}, P_{75}, P_{95}$.
3. Computes tail-risk metrics:
   * **95% & 99% Value at Risk (VaR)**: 1-bar percentage loss threshold.
   * **95% & 99% Conditional VaR (Expected Shortfall / CVaR)**: Mean loss conditioned on exceeding VaR:
     $$\text{CVaR}_\alpha = \mathbb{E}[R \mid R \le \text{VaR}_\alpha]$$

### 3.2 Transaction Cost & Friction Sensitivity Sweep
Evaluates execution decay across fee levels: $F \in [0, 1, 2, 3, 5, 7.5, 10, 15, 20, 25, 30, 40, 50]$ basis points:
* **Critical Break-Even Fee ($F_{\text{crit}}$)**: The friction level where strategy Sharpe decays to $0.0$:
  $$F_{\text{crit}} = \sup \{ F \ge 0 \mid \text{Sharpe}(F) \ge 0 \}$$
* **Friction Elasticity Slope**:
  $$\beta_{\text{friction}} = \frac{d\text{Sharpe}}{d\text{Fee}} \approx \frac{\Delta \text{Sharpe}}{\Delta \text{Fee (bps)}}$$

### 3.3 Purged & Embargoed Walk-Forward Cross-Validation
1. Partitions data into $K$ chronologically sequential rolling folds with an embargo boundary between In-Sample (IS) and Out-Of-Sample (OOS) segments.
2. Computes the **Walk-Forward Efficiency Ratio (WFER)**:
   $$\text{WFER} = \frac{\frac{1}{K}\sum_{k=1}^K \text{Sharpe}_{\text{OOS}}^{(k)}}{\frac{1}{K}\sum_{k=1}^K \text{Sharpe}_{\text{IS}}^{(k)}}$$
   * $\text{WFER} \ge 0.70$: Exceptional generalization robustness.
   * $0.50 \le \text{WFER} < 0.70$: Institutional baseline robustness.
   * $\text{WFER} < 0.30$: Temporal overfitting warning.

### 3.4 Out-of-Distribution (OOD) Zero-Shot Transfer
Evaluates Klint-32M (trained on crypto) on unseen assets across 6 global asset classes without fine-tuning:
* **Token Cross-Entropy Loss & Perplexity**:
  $$\mathcal{L}_{\text{CE}} = -\frac{1}{N}\sum_{i=1}^N \ln P(x_i \mid x_{<i}), \quad \text{Perplexity} = \exp(\mathcal{L}_{\text{CE}})$$
* Evaluates transfer stability across: US Equities (`SPY`, `NVDA`, `AAPL`), Commodities (`GLD`, `USO`), Forex (`EURUSD=X`, `USDJPY=X`), Fixed Income (`TLT`), and Crypto (`BTC`, `ETH`).
