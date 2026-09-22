# Model Card: Klint-32M

## Model Details
* **Model Name:** Klint-32M
* **Model Type:** Causal Generative Autoregressive Transformer
* **Author:** Akhilesh Varma (akhverm@gmail.com)
* **Architecture:** 10 layers, 480 hidden dimension, 10 attention heads, RoPE, RMSNorm
* **Parameters:** ~30.6M
* **License:** MIT

## Intended Use
* Probabilistic multi-horizon market forecasting
* Realized volatility estimation
* Synthetic market scenario generation and backtest stress testing

## Out-of-Scope Use
* High-frequency execution or direct automated trading orders
* Guaranteed financial return claims or investment advice

## Training Data & Provenance
* Assets included: 
* Date ranges: 
* Checksum / Manifest: 

## Evaluation Summary
* Codebook utilization:
* Candle validity rate: 100% (guaranteed by structural geometric decoder)
* Multi-horizon CRPS:
