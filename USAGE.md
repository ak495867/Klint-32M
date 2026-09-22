# Klint-32M: Inference Engine User Guide (`inference.py`)

This guide provides complete instructions and practical examples for using [`inference.py`](file:///d:/Klint/Klint-32M/inference.py) to generate directional alpha signals and autoregressive candlestick forecasts with **Klint-32M**.

---

##  Table of Contents

1. [Overview](#1-overview)
2. [Prerequisites & Installation](#2-prerequisites--installation)
3. [Model Loading Modes](#3-model-loading-modes)
4. [Data Ingestion Modes](#4-data-ingestion-modes)
5. [CLI Command-Line Reference](#5-cli-command-line-reference)
6. [Practical Real-World Examples](#6-practical-real-world-examples)
7. [Interpreting the Output Scorecard](#7-interpreting-the-output-scorecard)
8. [Programmatic Python SDK Usage](#8-programmatic-python-sdk-usage)
9. [Troubleshooting & FAQs](#9-troubleshooting--faqs)

---

## 1. Overview

`inference.py` is an institutional-grade, zero-configuration CLI script designed to:
- **Condition on Historical Context**: Encodes past OHLCV bars into stationary causal factor streams (**Price-Path**, **Range-Shape**, and **Activity**).
- **Produce Next-Bar Directional Alpha**: Calculates expected return, win probability ($P(\text{Up})$ vs $P(\text{Down})$), and trading signal (`BULLISH` / `BEARISH` / `NEUTRAL`).
- **Autoregressively Sample Future Bars**: Synthesizes future market trajectories across arbitrary time horizons (e.g. 10, 30, 50, 100 bars).
- **Guarantee Candle Geometry**: Reconstructs continuous factors with mathematical guarantees that $High \ge \max(Open, Close)$ and $Low \le \min(Open, Close)$ on **100.00%** of generated bars.
- **Export Data & Visualizations**: Automatically exports future candles to CSV and renders publication-quality matplotlib charts.

---

## 2. Prerequisites & Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/ak495867/Klint-32M.git
cd Klint-32M

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install package in editable mode
pip install -e .

# Optional: Install huggingface_hub for automated weight downloading
pip install huggingface_hub
```

---

## 3. Model Loading Modes

`inference.py` intelligently detects and loads model weights in three distinct ways:

### Mode A: Automatic Hugging Face Download (Default)
If no local files are specified or found, the script automatically downloads the release bundle from Hugging Face Hub under [`akhverm/Klint-32M`](https://huggingface.co/akhverm/Klint-32M):
```bash
python inference.py --ticker SOL-USD
```

### Mode B: Flagship Release Bundle (`klint_32m_v2_release.pt` - Recommended)
If you have the fine-tuned 112.5 MB release bundle containing multi-asset weights, tokenizer codebooks, and configuration:
```bash
python inference.py --checkpoint checkpoints/klint_32m_v2_release.pt --ticker NVDA
```

### Mode C: Base Foundation Bundle (`klint_32m_release.pt`) or Standalone Checkpoints
If you want to use the base pre-trained foundation model or a specific training step checkpoint (e.g., Step 1,500, Step 3,000, or Best):
```bash
# Base Foundation Bundle:
python inference.py --checkpoint checkpoints/klint_32m_release.pt --ticker SOL-USD

# Standalone Checkpoint + Separate Tokenizer:
python inference.py \
    --checkpoint checkpoints/klint_32m_best.pt \
    --tokenizer checkpoints/tokenizer_best.pt \
    --ticker BTC-USD
```

---

## 4. Data Ingestion Modes

You can condition the model on live financial feeds or historical datasets:

| Ingestion Mode | Flag Example | Use Case |
| :--- | :--- | :--- |
| **Live Yahoo Finance** | `--ticker SOL-USD` or `--ticker NVDA` | Real-time paper trading, live forecasting across 300+ global assets. |
| **Local CSV File** | `--data_path path/to/candles.csv` | Backtesting or evaluating proprietary/exchange data. |
| **Local NumPy Array** | `--data_path data/SOL.npy` | Fast binary ingestion of high-frequency datasets. |
| **Synthetic Fallback** | *(Omit `--ticker` and `--data_path`)* | Instant testing with generated sample data. |

> **CSV Format Requirements**: The CSV must contain standard columns: `Open`, `High`, `Low`, `Close`, `Volume` (case-insensitive).

---

## 5. CLI Command-Line Reference

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--ticker` | `str` | `None` | Asset ticker symbol to fetch via `yfinance` (e.g. `SOL-USD`, `NVDA`, `SPY`). |
| `--data_path` | `str` | `None` | Local path to `.csv` or `.npy` file containing OHLCV candles. |
| `--checkpoint` | `str` | `None` *(auto-detects `klint_32m_v2_release.pt`)* | Path to release bundle or model weights checkpoint. |
| `--tokenizer` | `str` | `checkpoints/tokenizer_best.pt` | Path to factor tokenizer codebooks. |
| `--hf_repo` | `str` | `akhverm/Klint-32M` | Hugging Face repository ID to fetch weights from if not found locally. |
| `--horizon` | `int` | `30` | Number of future market bars to forecast autoregressively. |
| `--context_bars`| `int` | `64` | Historical prompt context window length (bars). |
| `--temperature` | `float` | `0.8` | Sampling temperature (`0.2` = conservative/modal, `1.0` = high-entropy). |
| `--top_k` | `int` | `40` | Top-K token truncation for generative sampling. |
| `--save_plot` | `str` | `forecast_trajectory.png` | Output path for publication-grade visualization chart. |
| `--save_csv` | `str` | `None` | Output path to save forecasted OHLCV candles as CSV. |
| `--device` | `str` | `cuda` (if avail) else `cpu` | Execution hardware accelerator (`cuda` or `cpu`). |

---

## 6. Practical Real-World Examples

### Example 1: Live Solana (SOL-USD) 30-Bar Forecast
Fetch live market data, forecast 30 bars into the future, and save a high-resolution chart:
```bash
python inference.py \
    --ticker SOL-USD \
    --horizon 30 \
    --save_plot forecast_sol.png
```

### Example 2: Tech Equity (Nvidia - NVDA) with 50-Bar Horizon
Forecast 50 bars of Nvidia with conservative sampling temperature:
```bash
python inference.py \
    --ticker NVDA \
    --horizon 50 \
    --temperature 0.6 \
    --top_k 20 \
    --save_plot forecast_nvda.png
```

### Example 3: Macro Crypto (Bitcoin - BTC-USD) with CSV Export
Forecast 40 bars of Bitcoin and save the predicted candlestick data for automated order execution:
```bash
python inference.py \
    --ticker BTC-USD \
    --horizon 40 \
    --save_csv btc_forecast.csv \
    --save_plot btc_forecast.png
```

### Example 4: Gold Commodity (GLD) Using Local Step 3,000 Checkpoint
Forecast gold prices using a specific training checkpoint:
```bash
python inference.py \
    --checkpoint checkpoints/klint_32m_step_3000.pt \
    --tokenizer checkpoints/tokenizer_best.pt \
    --ticker GLD \
    --horizon 25 \
    --save_plot forecast_gold.png
```

### Example 5: Custom Historical CSV Ingestion
Condition on a custom proprietary dataset and output forecast:
```bash
python inference.py \
    --data_path my_trades/custom_eth_1h.csv \
    --context_bars 100 \
    --horizon 30 \
    --save_plot forecast_custom.png
```

---

## 7. Interpreting the Output Scorecard

When executed, `inference.py` prints an institutional predictive scorecard:

```text
======================================================================
 QUANTITATIVE PREDICTION & DIRECTIONAL ALPHA SCORECARD
======================================================================
  Asset:                   SOL-USD
  Current Price:           $148.52
  Terminal Forecast:       $154.20 (+3.82%)
  Market Signal:           BULLISH (LONG)
  Expected Next-Bar Alpha: +0.285%
  Directional Probability: P(Up) = 57.3% | P(Down) = 42.7%
  Candle Invariants:       100.00% PASSED (Guaranteed Geometry)
======================================================================

Forecasted Trajectory (First 10 Bars):
               Open        High         Low       Close        Volume
Bar +1   148.520000  149.850000  148.200000  149.400000  125430.120000
Bar +2   149.400000  150.750000  149.100000  150.350000  131200.450000
Bar +3   150.350000  151.200000  149.900000  150.800000  118900.000000
...
```

### Key Metrics Explained:
1. **Market Signal**:
   - `BULLISH (LONG)`: Model assigns expected return $> +5$ bps with $P(\text{Up}) > 51\%$.
   - `BEARISH (SHORT)`: Model assigns expected return $< -5$ bps with $P(\text{Down}) > 51\%$.
   - `NEUTRAL`: Expected edge is within transaction friction boundaries.
2. **Directional Probability**: Normalized softmax probability distribution over the discrete price-codebook body returns.
3. **Candle Invariants**: Structural guarantee that all generated candles satisfy $H \ge \max(O, C)$ and $L \le \min(O, C)$. Zero physical anomalies.
4. **Generated Chart (`forecast_trajectory.png`)**:
   - **Upper Panel**: Shows historical close trajectory (gray), followed by the forecasted path (green/red/amber), with a shaded confidence ribbon representing predicted High/Low intrabar volatility.
   - **Lower Panel**: Historical trading activity compared to the model's forecasted volume flow.

---

## 8. Programmatic Python SDK Usage

You can seamlessly import and call Klint-32M inside your own Python trading bots, backtesting scripts, or notebooks:

```python
import torch
import numpy as np
from inference import load_klint_bundle, generate_forecast, fetch_prompt_data, plot_forecast

# 1. Load model and tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
model, tokenizer, decoder = load_klint_bundle(
    checkpoint_path="checkpoints/klint_32m_v2_release.pt",
    device=device
)

# 2. Fetch live prompt context for any ticker
context_ohlcv, asset_name, last_close = fetch_prompt_data(
    ticker="NVDA",
    context_bars=64
)

# 3. Generate future trajectory & directional signal
forecast_ohlcv, stats = generate_forecast(
    model=model,
    tokenizer=tokenizer,
    decoder=decoder,
    context_ohlcv=context_ohlcv,
    horizon_bars=30,
    temperature=0.8,
    top_k=40,
    device=device
)

print(f"Asset: {asset_name}")
print(f"Current Price: ${last_close:.2f}")
print(f"Signal: {stats['signal']}")
print(f"Next-Bar Return: {stats['expected_next_return_pct']:+.2f}%")
print(f"P(Up): {stats['prob_up_pct']:.1f}% | P(Down): {stats['prob_down_pct']:.1f}%")

# 4. Optional: Save visual chart
plot_forecast(context_ohlcv, forecast_ohlcv, asset_name, stats, save_path="nvda_forecast.png")
```

---

## 9. Troubleshooting & FAQs

#### Q: How do I change the time interval (e.g. 1-hour or 15-minute instead of 1-day)?
You can modify the period and interval when calling `fetch_prompt_data` in Python:
```python
context_ohlcv, name, price = fetch_prompt_data(ticker="BTC-USD", period="7d", interval="1h")
```

#### Q: How can I make forecasts more conservative or more exploratory?
- For **conservative/modal forecasts** (lower variance): set `--temperature 0.4 --top_k 15`.
- For **exploratory forecasts** (higher variance/volatility): set `--temperature 1.0 --top_k 50`.

#### Q: What if I don't have a GPU?
`inference.py` runs smoothly on standard CPUs. A 30-bar forecast typically takes under **1.5 seconds** on a modern multi-core CPU.
