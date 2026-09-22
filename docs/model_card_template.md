# Model Card Template: Klint Model Family

## Model Details
* **Model Name:** [e.g. Klint-32M, Klint-100M]
* **Model Type:** Causal Generative Autoregressive Transformer
* **Author:** Akhilesh Varma (akhverm@gmail.com)
* **Architecture:** $L$ layers, $d_{\text{model}}$ hidden dimension, $N_{\text{heads}}$ attention heads, RoPE, RMSNorm, SwiGLU FFN
* **Parameters:** [Exact trainable parameter count]
* **Hugging Face Hub:** [e.g. https://huggingface.co/akhverm/Klint-32M]
* **GitHub Repository:** [https://github.com/ak495867/Klint-32M]
* **License:** MIT

## Intended Use
* Probabilistic multi-horizon market forecasting
* Synthetic market scenario generation and backtest stress testing
* Realized volatility and Value at Risk (VaR) estimation
* Directional alpha signal generation

## Out-of-Scope Use
* High-frequency execution or direct automated market making
* Guaranteed financial return claims or investment advice

## Training Data & Provenance
* Primary Assets: [e.g. Solana 1m bars, Bitcoin, Equities]
* Total Bars & Tokens: [e.g. 1.59M bars = 4.77M tokens]
* Date ranges & partition split: [e.g. 95% Train / 5% Validation]

## Evaluation & Stress Testing Summary
* Codebook utilization rate: [Target > 95%]
* Candle invariant validity rate: 100.00% (Guaranteed by Softplus geometric decoder)
* Monte Carlo Block Bootstrap: [P-value, 95% VaR, CVaR]
* Cost sensitivity break-even fee ($F_{\text{crit}}$): [e.g. > 15 bps]
* Walk-Forward Efficiency Ratio (WFER): [Target >= 0.50]
* Zero-shot OOD Transfer: [Equities, Commodities, FX, Rates]
