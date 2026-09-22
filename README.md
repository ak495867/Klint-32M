# Klint

> **Klint** is an open research architecture for generative financial time-series modeling. It represents market bars as separate **price-path**, **range-shape**, and **activity** code streams, then models them with a causal Transformer and decodes them into structurally valid OHLCV trajectories.

* **Author:** Akhilesh Varma (akhverm@gmail.com)
* **Planned first checkpoint:** `Klint-32M`
* **Code license:** MIT
* **Status:** Active research and implementation.

---

## Why Klint?

Financial candles combine directional movement, intrabar range, and market activity, but those signals are not interchangeable. Klint makes that decomposition explicit. It tokenizes three causal factor streams independently, predicts the next bar’s streams in the order:
$$\text{Price} \longrightarrow \text{Range conditioned on Price} \longrightarrow \text{Activity conditioned on Price \& Range}$$
and reconstructs the candle with rules that preserve its price geometry.

Klint is inspired by the general discrete-token/autoregressive paradigm used in financial time-series foundation models such as Kronos. It is an independent design: it does not reuse Kronos source code, checkpoints, data, tokenizer, or brand. Its discrete latent formulation draws on learned vector quantization, while the causal backbone uses rotary position embeddings (RoPE) and RMSNorm.

---

## Architecture at a Glance

| Layer | Klint Design |
|---|---|
| **Input** | OHLCV, optional turnover/trade count, and source metadata |
| **Causal Factorisation** | Price path; range shape; activity and feature-presence masks |
| **Tokenizer** | Three residual vector-quantisation codebooks: 512 price, 256 range, 256 activity codes |
| **Foundation Model** | 10-layer, 480-hidden, 10-head decoder-only Transformer with RoPE and RMSNorm |
| **Context** | 1,024 chronological bars (3,072 factor tokens) |
| **Generation Order** | Price code $\to$ Range code $\to$ Activity code |
| **Decoder** | Factor-to-OHLCV decoder that guarantees $\text{High} \ge \max(\text{Open}, \text{Close})$ and $\text{Low} \le \min(\text{Open}, \text{Close})$ |
| **Initial Scale** | `Klint-32M`, analytically estimated at 30,630,934 parameters |

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

# Run test suite
pytest tests/
```

---

## 🚀 Full Scale Model Training on Google Colab

The repository includes a ready-to-run notebook for training the full **32-Million parameter** model: [`notebooks/train_klint32m_colab.ipynb`](file:///d:/Klint/Klint-32M/notebooks/train_klint32m_colab.ipynb).

All checkpoints are saved locally inside the Colab environment (`./checkpoints/`) with **zero Google Drive dependency**.

```bash
# Stage 1: Train the Factor Tokenizer (RVQ)
python scripts/train_tokenizer.py --data_path data/SOL.npy --epochs 15 --batch_size 128

# Stage 2: Cache discrete tokens (50x speedup)
python scripts/cache_tokens.py --data_path data/SOL.npy --tokenizer_path checkpoints/tokenizer_best.pt

# Stage 3: Train the full 32M parameter causal Transformer
python scripts/train_klint32m.py --tokens_path data/sol_tokens.pt --save_dir checkpoints --max_steps 25000 --batch_size 8 --grad_accum_steps 4

# Stage 4: Generate synthetic market trajectories
python examples/generate_synthetic_bars.py
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
  model_card_template.md       Checkpoint documentation template
src/klint/
  data/                        Validation, factor extraction, and dataset loaders
  tokenizer/                   Residual vector quantization and geometric decoder
  models/                      RoPE, RMSNorm, causal Transformer, and Klint-32M
  training/                    Trainer, loss functions, and evaluation metrics
tests/                         Comprehensive pytest test suite
```

## References
1. Shi et al., *Kronos: A Foundation Model for the Language of Financial Markets* (2025)
2. van den Oord, Vinyals & Kavukcuoglu, *Neural Discrete Representation Learning* (2017)
3. Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding* (2021)
4. Zhang & Sennrich, *Root Mean Square Layer Normalization* (2019)
