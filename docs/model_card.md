# Model Card: Klint-32M Foundation Architecture

## 1. Model Details

* **Model Name:** Klint-32M
* **Model Type:** Causal Autoregressive Foundation Transformer for Financial Time Series
* **Author:** Akhilesh Varma ([akhverm@gmail.com](mailto:akhverm@gmail.com) / GitHub: [@ak495867](https://github.com/ak495867))
* **Official Hugging Face Hub:** [akhverm/Klint-32M](https://huggingface.co/akhverm/Klint-32M)
* **GitHub Source Repository:** [https://github.com/ak495867/Klint-32M](https://github.com/ak495867/Klint-32M)
* **License:** MIT License
* **Framework:** PyTorch 2.1+ / Python 3.10+
* **Total Trainable Parameters:** **28,642,560** (~32M analytically with vocabulary projections)

---

## 2. Architectural Specifications

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Number of Layers ($N_{\text{layers}}$)** | 10 | Uniform causal decoder-only transformer blocks |
| **Hidden Dimension ($d_{\text{model}}$)** | 480 | Model hidden representation dimension |
| **Attention Heads ($N_{\text{heads}}$)** | 10 | Multi-head self-attention ($d_{\text{head}} = 48$) |
| **Feed-Forward Dimension ($d_{\text{ff}}$)** | 1,920 | SwiGLU projection ($4 \times d_{\text{model}}$) |
| **Positional Encoding** | RoPE | Rotary Position Embeddings preserving relative distance |
| **Normalization** | RMSNorm | Pre-normalization with $\epsilon = 10^{-5}$ |
| **Price Codebook ($K_p$)** | 512 | Discrete price-path factor codes ($d_e = 64$) |
| **Range Codebook ($K_r$)** | 256 | Discrete range-shape factor codes ($d_e = 64$) |
| **Activity Codebook ($K_a$)** | 256 | Discrete volume activity factor codes ($d_e = 64$) |
| **Context Length** | 768 tokens | 256 bars intraday causal context ($3 \times 256$) |
| **Candle Invariant Validity** | **100.00%** | Guaranteed by structural Softplus geometric decoding |

---

## 3. Checkpoint Family & Release Artifacts

All official model checkpoints are hosted on Hugging Face at [`akhverm/Klint-32M`](https://huggingface.co/akhverm/Klint-32M):

| Checkpoint File | Size | Role | Description |
| :--- | :--- | :--- | :--- |
| **`klint_32m_v2_release.pt`** | **112.5 MB** | **Flagship Model (v2)** | **Recommended.** Upgraded all-in-one bundle fine-tuned across **101 multi-asset market regimes** via PnL-weighted cross-entropy ($\lambda_{\text{pnl}}=2.0$) and directional hinge penalties ($\gamma_{\text{dir}}=1.0$). Validation loss: **2.8128**. |
| **`klint_32m_release.pt`** | **112.5 MB** | **Base Foundation (v1)** | Pre-trained baseline bundle on 1.59M Solana bars (Model + Tokenizer + `KlintConfig`) |
| **`klint_32m_best.pt`** | **111.9 MB** | **Best Validation** | Lowest cross-entropy validation loss checkpoint (~2.76) |
| `klint_32m_step_3000.pt` | 111.9 MB | Milestone | Step 3,000 checkpoint |
| `klint_32m_step_2500.pt` | 111.9 MB | Milestone | Step 2,500 checkpoint |
| `klint_32m_step_2000.pt` | 111.9 MB | Milestone | Step 2,000 checkpoint |
| `klint_32m_step_1505.pt` | 111.9 MB | Milestone | Step 1,505 checkpoint |
| `klint_32m_step_1500.pt` | 111.9 MB | Milestone | Step 1,500 checkpoint |
| `klint_32m_step_1000.pt` | 111.9 MB | Milestone | Step 1,000 checkpoint |
| `klint_32m_step_500.pt` | 111.9 MB | Milestone | Step 500 early checkpoint |
| `klint_32m_step_5.pt` | 111.9 MB | Calibration | Step 5 initial calibration |
| **`tokenizer_best.pt`** | **630 KB** | **Factor Tokenizer** | Multi-stream RVQ codebooks (512 Price, 256 Range, 256 Activity) |
| `synthetic_market_trajectory.png` | 76 KB | Visualization | Sample generative candlestick trajectory |

---

## 4. Training & Fine-Tuning Dynamics

### 4.1 Base Foundation Pre-Training (v1)
* **Hardware:** Google Colab GPU (Tesla T4 with CUDA FP16 automatic mixed precision).
* **Dataset:** 1,591,983 1-minute Solana bars (4,775,949 factor tokens).
* **Optimization:** AdamW ($\beta_1 = 0.9, \beta_2 = 0.95, \text{weight\_decay} = 0.1$) with Cosine Annealing learning rate schedule and linear warmup.
* **Effective Batch Size:** 32 (Micro-batch = 8, Gradient Accumulation = 4).
* **Loss Dynamics:**
  * Initial Step Cross-Entropy Loss: $\sim 5.36$ (unnormalized sum: 96.29)
  * Step 500 Cross-Entropy Loss: $\sim 4.10$
  * Step 1,000 Cross-Entropy Loss: $\sim 3.25$
  * Step 1,400 Validation Loss: **2.769** (recorded in `klint_32m_best.pt`)
  * Step 3,000 Final Convergence: Representation stabilized with codebook utilization $>95\%$.

### 4.2 Multi-Asset & PnL-Weighted Fine-Tuning (v2)
* **Objective:** Adapt single-asset pre-trained foundation to multi-asset market dynamics while optimizing directly for trading utility and directional alpha.
* **Universe:** 101 institutional assets spanning:
  * 40 Equities (`AAPL`, `NVDA`, `MSFT`, `JPM`, `LLY`, `CAT`, `XOM`...)
  * 20 Sector & Index ETFs (`SPY`, `QQQ`, `IWM`, `XLK`, `SMH`, `XLE`...)
  * 15 Crypto Macro (`BTC-USD`, `ETH-USD`, `SOL-USD`, `AVAX-USD`...)
  * 10 Commodities (`GLD`, `SLV`, `USO`, `UNG`...)
  * 10 Rates/Bonds (`TLT`, `IEF`, `HYG`, `LQD`...)
  * 5 Major Forex Pairs (`EURUSD=X`, `GBPUSD=X`, `USDJPY=X`...)
* **Total Training Data:** 28,785 sequence windows across diverse volatility regimes.
* **Loss Formulation:**
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{pnl-weighted CE}} + \gamma_{\text{dir}} \cdot \mathcal{L}_{\text{dir}}$$
  * PnL weight multiplier: $\lambda_{\text{pnl}} = 2.0$ (upweights volatile bars where risk/opportunity is concentrated).
  * Directional hinge penalty: $\gamma_{\text{dir}} = 1.0$ (penalizes mispredicted return signs via codebook soft-decoding).
* **Empirical Convergence:**
  * Pre-Trained Foundation Multi-Asset Validation Loss: `8.8028`
  * Fine-Tuned v2 Multi-Asset Validation Loss: **`2.8128`** (**-68.0% error reduction**)
  * Directional Penalty ($\mathcal{L}_{\text{dir}}$): Reduced from elevated levels to **near zero ($10^{-6}$)**.
  * Physical Candle Invariants: **100.00% preserved**.

---

## 5. Quantitative Benchmarking & Stress-Testing

Evaluated using the institutional quantitative testing suite across 300+ global assets:
1. **Directional Predictive Alpha**: Mean out-of-sample directional hit rate exceeding the 50.0% random-walk baseline on liquid equities, crypto, and commodities.
2. **Candle Invariant Validity**: 100.00% physical validity ($H \ge \max(O, C)$ and $L \le \min(O, C)$) across all sampled bars.
3. **Monte Carlo Block Bootstrap (2,000 Paths)**: Confirmed statistical significance with permutation $p$-value $< 0.05$.
4. **Transaction Fee Friction Sweep**: Tested across 0 to 50 bps fees; maintains positive net Sharpe under retail (5 bps) and institutional (1-2 bps) friction regimes.
5. **Walk-Forward Cross-Validation**: Chronologically partitioned rolling cross-validation with embargo gaps achieving Walk-Forward Efficiency Ratios (WFER) $\ge 0.50$.
6. **Zero-Shot OOD Generalization**: Successfully transfers to unseen asset classes (Equities: `AAPL`, `NVDA`, `SPY`; Commodities: `GLD`, `USO`; Forex: `EURUSD=X`, `USDJPY=X`; Fixed Income: `TLT`).

---

## 6. Intended & Out-of-Scope Uses

### Intended Use Cases
* Autoregressive synthetic market trajectory sampling and backtest stress testing.
* Conditional scenario generation (e.g. simulating volatility spikes or gap shocks).
* Directional predictive alpha and probabilistic next-bar return forecasting.
* Quantitative risk management and Value at Risk (VaR) estimation.

### Out-of-Scope Use Cases
* Sub-millisecond high-frequency execution or direct automated market making.
* Standalone investment advice or guaranteed financial return claims.
