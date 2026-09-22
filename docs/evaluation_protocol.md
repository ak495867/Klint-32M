# Klint Evaluation Protocol & Leakage Controls

## 1. Chronological Partitioning

Financial time-series data exhibits persistent autocorrelation in volatility and volume. Standard random k-fold cross-validation results in severe lookahead bias and must never be used.

Klint enforces strict **chronological splitting** with an **embargo window**:

```text
[ ---- TRAIN SPLIT ---- ] [ EMBARGO ] [ ---- VAL SPLIT ---- ] [ EMBARGO ] [ ---- TEST SPLIT ---- ]
```

* **Embargo Duration**: Minimum 1,440 bars (24 hours on 1-minute data) between any split boundary.
* **No Retrospective Normalization**: Feature statistics (such as volume EMA) are computed strictly causally with online updating.

## 2. Core Benchmark Metrics

Model checkpoints are evaluated against three benchmark dimensions:

### 2.1 Reconstruction & Quantization Fidelity
* **Codebook Utilization**: Percentage of codes in each codebook ($K_p=512, K_r=256, K_a=256$) activated at least once over the test set. Target: $>95\%$.
* **Reconstruction MAE**: Mean absolute error between original OHLCV and reconstructed OHLCV trajectories.

### 2.2 Generative Structural Integrity
* **Candle Invariant Validity Rate**: Fraction of generated synthetic candles satisfying $H \ge \max(O, C)$ and $L \le \min(O, C)$. Target: $100.00\%$.
* **Log-Return Distribution Distance**: 2-Wasserstein distance and Kolmogorov-Smirnov test between empirical and generated return distributions.

### 2.3 Probabilistic Multi-Horizon Forecasting
* **Continuous Ranked Probability Score (CRPS)** evaluated over horizons: $h \in \{1, 5, 15, 60, 240\}$ bars.
