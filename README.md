# Klint - 32M

[![GitHub Repository](https://img.shields.io/badge/GitHub-ak495867%2FKlint--32M-blue?logo=github)](https://github.com/ak495867/Klint-32M)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-akhverm%2FKlint--32M-yellow)](https://huggingface.co/akhverm/Klint-32M)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Parameters](https://img.shields.io/badge/Parameters-28.6M%20Trainable-purple)]()
[![Tests](https://img.shields.io/badge/Tests-15%20Passing-brightgreen)]()

> **Klint** is an open research foundation architecture for generative financial time-series modeling. It represents market bars as separate **price-path**, **range-shape**, and **activity** code streams, models them with a causal Transformer, and decodes them into structurally valid OHLCV trajectories with guaranteed physical geometry.

* **Author:** Akhilesh Varma ([akhverm@gmail.com](mailto:akhverm@gmail.com) / [@akhverm](https://github.com/ak495867))
* **Foundation Checkpoint Family:** [huggingface.co/akhverm/Klint-32M](https://huggingface.co/akhverm/Klint-32M)
* **Code License:** MIT
* **Status:** Active research & production-ready evaluation suite.

> 📦 **Checkpoints Available:** The complete trained model family (Step 5 to Step 3,000, Best checkpoint, and all-in-one Release Bundle) is officially published on Hugging Face at [**akhverm/Klint-32M**](https://huggingface.co/akhverm/Klint-32M).

---

## Why Klint?

Financial candles combine directional movement, intrabar range, and market activity, but those signals are not interchangeable. Klint makes that decomposition explicit. It tokenizes three causal factor streams independently, predicts the next bar’s streams in the order:

$$\text{Price} \longrightarrow \text{Range conditioned on Price} \longrightarrow \text{Activity conditioned on Price \ Range}$$

and reconstructs the candle with rules that preserve its price geometry:

$$\text{High} \ge \max(\text{Open}, \text{Close}) \quad \text{and} \quad \text{Low} \le \min(\text{Open}, \text{Close})$$

Klint is inspired by the general discrete-token/autoregressive paradigm used in financial foundation models such as Kronos. It is an independent design: it does not reuse Kronos source code, checkpoints, data, tokenizer, or brand. Its discrete latent formulation draws on learned Residual Vector Quantization (RVQ), while the causal backbone uses Rotary Position Embeddings (RoPE) and RMSNorm.

---

## Architecture at a Glance

| Component | Klint Design Specification |
|:---|:---|
| **Input Data** | OHLCV candlesticks, trade volume, and factor-presence masks |
| **Causal Factorisation** | Price-path stream $[r_{\text{gap}}, r_{\text{body}}]$; Range-shape $[\log r, u, l]$; Activity stream $[\log v]$ |
| **Discrete Tokenizer** | 3-stream Residual Vector Quantization: $K_p=512$ price, $K_r=256$ range, $K_a=256$ activity codes |
| **Foundation Backbone** | 10-layer, 480-hidden, 10-head causal decoder-only Transformer with RoPE and RMSNorm |
| **Trainable Scale** | **28,642,560 parameters** (~32M analytically with vocab projection) |
| **Context Window** | 256 chronological bars ($T=768$ factor tokens) up to 1,024 bars ($T=3,072$ tokens) |
| **Generation Chain** | Autoregressive Next-Token: Price $\to$ Range $\to$ Activity |
| **Geometric Decoder** | Structural decoder guaranteeing 100.00% physical candle validity |

---

## 📦 Hugging Face Checkpoint Family

All official model weights are hosted on Hugging Face at [**akhverm/Klint-32M**](https://huggingface.co/akhverm/Klint-32M):

| Checkpoint File | Size | Role | Description |
|:---|:---|:---|:---|
| **`klint_32m_release.pt`** | **112.5 MB** | **All-in-One Bundle** | Model weights + Tokenizer codebooks + Config + Training metadata |
| **`klint_32m_best.pt`** | **111.9 MB** | **Best Validation** | Checkpoint with lowest validation loss (~2.76 Cross-Entropy Loss) |
| `klint_32m_step_3000.pt` | 111.9 MB | Milestone | Step 3,000 checkpoint |
| `klint_32m_step_2500.pt` | 111.9 MB | Milestone | Step 2,500 checkpoint |
| `klint_32m_step_2000.pt` | 111.9 MB | Milestone | Step 2,000 checkpoint |
| `klint_32m_step_1505.pt` | 111.9 MB | Milestone | Step 1,505 checkpoint |
| `klint_32m_step_1500.pt` | 111.9 MB | Milestone | Step 1,500 resume checkpoint |
| `klint_32m_step_1000.pt` | 111.9 MB | Milestone | Step 1,000 checkpoint |
| `klint_32m_step_500.pt` | 111.9 MB | Milestone | Step 500 early training checkpoint |
| `klint_32m_step_5.pt` | 111.9 MB | Calibration | Initial step 5 calibration checkpoint |
| **`tokenizer_best.pt`** | **630 KB** | **Factor Tokenizer** | Multi-stream RVQ codebooks (512 Price, 256 Range, 256 Activity) |
| `synthetic_market_trajectory.png` | 76 KB | Output Preview | Sample generative candlestick trajectory |

---

##  Live Inference & Market Forecasting

>  **Comprehensive User Guide:** See [**USAGE.md**](USAGE.md) for full CLI parameter references, data ingestion options, and real-world trading examples.

Use [`inference.py`](file:///d:/Klint/Klint-32M/inference.py) to forecast future candlestick trajectories and generate directional alpha signals:

```bash
# 1. Live market inference on Solana (auto-downloads bundle from Hugging Face if needed):
python inference.py --ticker SOL-USD --horizon 30 --save_plot forecast_sol.png

# 2. Live market inference on Nvidia:
python inference.py --ticker NVDA --horizon 50 --save_plot forecast_nvda.png

# 3. Forecast using local release bundle:
python inference.py --checkpoint checkpoints/klint_32m_release.pt --ticker BTC-USD --horizon 30

# 4. Forecast from custom CSV data and export predicted candles:
python inference.py --data_path my_data.csv --horizon 20 --save_csv predicted_candles.csv
```

### Python SDK Inference:

```python
import torch
from klint.models.klint_32m import Klint32M
from klint.tokenizer.factor_tokenizer import FactorTokenizer
from klint.tokenizer.geometric_decoder import GeometricDecoder

# Load release bundle
bundle = torch.load("checkpoints/klint_32m_release.pt", map_location="cpu", weights_only=False)

model = Klint32M(bundle["config"])
model.load_state_dict(bundle["model_state_dict"])
model.eval()

tokenizer = FactorTokenizer()
tokenizer.load_state_dict(bundle["tokenizer_state_dict"])
decoder = GeometricDecoder()

# Generate 30 future bars conditioned on past prompt tokens
prompt_tokens = torch.randint(0, 256, (1, 90))  # (1, 30 bars * 3 tokens)
new_tokens = model.generate_tokens(prompt_tokens, num_bars=30, temperature=0.8, top_k=40)[:, 90:]

p_tok, r_tok, a_tok = tokenizer.deinterleave(new_tokens)
rec_p, rec_r, rec_a = tokenizer.decode_tokens(p_tok, r_tok, a_tok)
future_candles = decoder(rec_p, rec_r, rec_a, anchor_price=150.0)

print("Generated Candles (Open, High, Low, Close, Volume):", future_candles.shape)
```

---

##  Institutional Stress Testing & Benchmarking

Klint-32M includes an institutional quantitative testing battery to rigorously validate tail-risk, friction decay, and zero-shot transfer:

### 1. Unified Stress Testing Suite ([`scripts/stress_test_klint32m.py`](file:///d:/Klint/Klint-32M/scripts/stress_test_klint32m.py))

```bash
python scripts/stress_test_klint32m.py \
    --checkpoint checkpoints/klint_32m_best.pt \
    --tokenizer_checkpoint checkpoints/tokenizer_best.pt \
    --tests all \
    --monte_carlo_runs 2000 \
    --walkforward_folds 5 \
    --output_dir benchmarks/stress_tests
```

* **Monte Carlo Block Bootstrap (2,000 Paths)**: $P_5 - P_{95}$ confidence ribbons, 99% VaR/CVaR, and permutation $p$-value against luck.
* **Transaction Fee Friction Sweep (0 to 50 bps)**: Pinpoints critical break-even fee ($F_{\text{crit}}$) and Sharpe elasticity.
* **Purged & Embargoed Walk-Forward Validation (5 Folds)**: Strictly chronological cross-validation with embargo gaps measuring the Walk-Forward Efficiency Ratio (WFER).
* **One-Shot / Zero-Shot OOD Generalization**: Evaluates immediate transfer to 10 global assets across Equities (`SPY`, `NVDA`, `AAPL`), Commodities (`GLD`, `USO`), FX (`EURUSD=X`, `USDJPY=X`), Rates (`TLT`), and Crypto (`BTC`, `ETH`).

### 2. 300+ Multi-Asset Quantitative Benchmark ([`scripts/run_multi_asset_benchmark.py`](file:///d:/Klint/Klint-32M/scripts/run_multi_asset_benchmark.py))

```bash
python scripts/run_multi_asset_benchmark.py \
    --checkpoint checkpoints/klint_32m_best.pt \
    --tokenizer_checkpoint checkpoints/tokenizer_best.pt \
    --max_assets 320 \
    --period 1y \
    --output_dir benchmarks
```

---

##  Google Colab Unified Notebook

The repository includes a ready-to-run master notebook: **[`notebooks/Klint-32M.ipynb`](file:///d:/Klint/Klint-32M/notebooks/Klint-32M.ipynb)**.

It seamlessly covers:
1. **Stage 0–3**: Hardware check, tokenizer verification, token cache, and high-throughput training/resumption.
2. **Stage 4–5**: Automatic release bundle packaging and generative candlestick trajectory sampling.
3. **Stage 6–7**: 300+ multi-asset quantitative benchmark with inline visualization of equity curves, Sharpe rankings, and drawdown logs.
4. **Stage 8–9**: One-click institutional stress testing and inline risk scorecards.

---

## Quickstart & Installation

Targeting Python 3.10+ and PyTorch 2.1+.

```bash
# Clone the repository
git clone https://github.com/ak495867/Klint-32M.git
cd Klint-32M

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install package in editable mode with dev dependencies
pip install -e ".[dev]"

# Run full test suite (15 tests)
pytest tests/ -v
```

---

## Repository Structure

```text
configs/
  klint_32m.yaml               Reference model and training configuration
docs/
  architecture.md              Full technical architecture specification
  data_card.md                 Canonical data and source-rights contract
  evaluation_protocol.md       Leakage controls and benchmark plan
notebooks/
  Klint-32M.ipynb              Unified master Google Colab training & testing notebook
scripts/
  train_tokenizer.py           Pre-train RVQ factor codebooks
  cache_tokens.py              Pre-encode 1.59M bars into integer token IDs
  train_klint32m.py            High-speed foundation model trainer (with --resume_from)
  run_multi_asset_benchmark.py 300+ asset quantitative evaluation engine
  stress_test_klint32m.py      Unified institutional stress testing suite
inference.py                   Live market forecasting and trajectory generation engine
HUGGINGFACE_README.md          Official model card for Hugging Face (akhverm/Klint-32M)
src/klint/
  data/                        Validation, factor extraction, and dataset loaders
  tokenizer/                   Residual vector quantization and geometric decoder
  models/                      RoPE, RMSNorm, causal Transformer, and Klint-32M
  training/                    Trainer, loss functions, and evaluation metrics
  benchmark/                   Universe registry, yfinance fetcher, and plotter
  stress_test/                 Monte Carlo, cost sensitivity, walk-forward, OOD engines
tests/                         Comprehensive pytest test suite (15 unit tests)
```

---

## References

1. Shi et al., *Kronos: A Foundation Model for the Language of Financial Markets* (2025)
2. van den Oord, Vinyals & Kavukcuoglu, *Neural Discrete Representation Learning* (2017)
3. Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding* (2021)
4. Zhang & Sennrich, *Root Mean Square Layer Normalization* (2019)

---

## Citation

```bibtex
@misc{varma2026klint32m,
  author = {Akhilesh Varma},
  title = {Klint-32M: A Foundation Architecture for Generative Financial Time-Series Modeling},
  year = {2026},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/akhverm/Klint-32M}},
  note = {GitHub: \url{https://github.com/ak495867/Klint-32M}}
}
```
