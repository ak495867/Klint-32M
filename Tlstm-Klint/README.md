# 🚀 TLSTM-Klint: Foundation Knowledge Distillation Engine

[![Teacher Model](https://img.shields.io/badge/Teacher-Klint--32M%20v2%20(28.6M)-orange)](https://huggingface.co/akhverm/Klint-32M)
[![Student Model](https://img.shields.io/badge/Student-TLSTM--Klint%20(0.59M)-blue)]()
[![Compression Ratio](https://img.shields.io/badge/Compression-48.3x%20Reduction-brightgreen)]()
[![Evaluation Universe](https://img.shields.io/badge/Universe-325%20Fresh%20Assets-purple)]()
[![Notebook](https://img.shields.io/badge/Colab%20Notebook-TLSTM__Klint.ipynb-yellow)](TLSTM_Klint.ipynb)

> **TLSTM-Klint** is an ultra-lightweight, high-speed **Temporal LSTM** student model distilled directly from the flagship **Klint-32M v2** foundation Transformer (~28.6M parameters). It achieves **48.3x parameter compression** (~98% smaller) while preserving multi-asset directional alpha, risk-adjusted Sharpe, and low-latency streaming inference.

---

## 🔬 Architectural Compression Overview

| Architectural Dimension | Teacher Model (`Klint-32M v2`) | Student Model (`TLSTM-Klint`) | Relative Compression |
|:---|:---:|:---:|:---:|
| **Backbone Architecture** | 10-Layer Causal Transformer (RoPE + RMSNorm) | 2-Layer Causal LSTM + Temporal Attention | **Recurrent Streaming ($O(1)$ Memory)** |
| **Trainable Parameters** | **28,642,560** (~28.6M) | **592,897** (~0.59M) | **48.3x Reduction (~98% Smaller)** |
| **Hidden Dimension ($d_{\text{model}}$)** | 480 | 128 | **3.75x Compact State** |
| **Attention Heads** | 10 Multi-Head Attention | 2 Causal Temporal Attention Heads | **Low FLOP Footprint** |
| **Checkpoint Bundle Size** | ~112.5 MB | **~2.4 MB** | **Edge & Microcontroller Friendly** |
| **Inference Scaling** | $O(T)$ Attention Sequence Cache | $O(1)$ Hidden State Recurrence $(h_t, c_t)$ | **Zero Latency Accumulation** |

---

## 🧠 Distillation Loss Formulation

The student model is trained using a composite knowledge distillation loss that combines soft dark knowledge transfer, hard factor token cross-entropy, PnL volatility weighting, and directional hinge penalties:

$$\mathcal{L}_{\text{total}} = (1 - \alpha_{\text{KD}}) \mathcal{L}_{\text{CE}} + \alpha_{\text{KD}} \cdot \tau^2 \mathcal{L}_{\text{KL}} + \gamma_{\text{dir}} \mathcal{L}_{\text{dir}}$$

### 1. Soft Logit Distillation ($\mathcal{L}_{\text{KL}}$):
$$\mathcal{L}_{\text{KL}} = \sum_{k} P_{\text{teacher}}(k) \log\left(\frac{P_{\text{teacher}}(k)}{P_{\text{student}}(k)}\right), \quad \text{where } P(k) = \operatorname{softmax}\left(\frac{z_k}{\tau}\right)$$
Transfers the teacher's nuanced probabilistic confidence over non-target price codebook return bins with temperature $\tau = 2.0$ and weight $\alpha_{\text{KD}} = 0.6$.

### 2. PnL-Weighted Loss ($w_t$):
$$w_t = 1.0 + \lambda_{\text{pnl}} \cdot \min(|r_t^{\text{body}}| \cdot 100, 10.0)$$
Dynamically magnifies loss gradients on high-volatility bars where trading profitability and tail risks are concentrated.

### 3. Directional Hinge Penalty ($\mathcal{L}_{\text{dir}}$):
$$\mathcal{L}_{\text{dir}} = \operatorname{ReLU}(-\hat{r}_{\text{pred}} \cdot r_{\text{realized}}) \cdot 100$$
Penalizes wrong-sided directional predictions, ensuring the student learns strict sign alignment.

---

## 🧪 Institutional Multi-Test Battery (325 Fresh Assets)

Just like Klint-32M v2, the distilled TLSTM student is evaluated across **325 completely fresh assets** (0% overlap with training data, zero lookahead leakage):

| Quantitative Test | Evaluated Dynamics | Generated Visualizations (Saved to `TLSTM-multitest/`) |
|:---|:---|:---|
| **1. Multiple Monte Carlo Tests** | 2,000 block bootstrap paths, $P_5 - P_{95}$ ribbons, 99% VaR & CVaR | `monte_carlo_equity_ribbons.png`, `monte_carlo_var_cvar_dist.png` |
| **2. Information Coefficient (IC)** | Multi-horizon Pearson and Spearman Rank IC (1, 3, 5, 10 bars) | `ic_cumulative_trajectory.png`, `ic_cross_sectional_distribution.png` |
| **3. Cross-Sectional Sharpe Tests** | Per-asset Sharpe distribution and rolling 60-day portfolio trajectory | `sharpe_cross_asset_distribution.png`, `sharpe_rolling_trajectory.png` |
| **4. Deflated Sharpe Ratio (DSR)** | Multiple-testing correction ($N \ge 300$) and skewness/kurtosis (López de Prado) | `dsr_selection_bias_curve.png`, `psr_moments_landscape.png` |
| **5. Information Ratio (IR) Tests** | Active alpha generation over equal-weight market benchmark with tracking error | `ir_cumulative_alpha_curve.png`, `ir_rolling_active_risk.png` |
| **6. Extreme Regime Shock Tests** | Resilience under 3x volatility explosion, flash crash gaps, and liquidity droughts | `shock_regime_resilience.png`, `shock_directional_error_shift.png` |
| **7. Placebo & White Noise Tests** | Permuted returns and synthetic Gaussian noise verifying **zero data leakage** | `placebo_true_vs_permuted_dist.png`, `placebo_whitenoise_winrate_qq.png` |
| **8. Multilayer Walk-Forward Tests** | 5 chronological purged & embargoed folds measuring Walk-Forward Efficiency (WFER) | `walkforward_fold_equity_curves.png`, `walkforward_wfer_degradation.png` |
| **9. Noise Injection & Stability** | Graceful degradation curve against escalating factor jitter $\sigma \in [0.1, 2.0]$ | `noise_performance_decay_curve.png`, `noise_token_divergence_snr.png` |
| **10. Friction & Fee Sweep** | 0 to 50 bps fee sensitivity identifying critical breakeven fee $F_{\text{crit}}$ | `friction_sharpe_decay_curve.png`, `friction_cumulative_pnl_sweep.png` |

---

## 🚀 Google Colab 1-Click Interactive Workflow

Run the self-contained Google Colab notebook:
* **Notebook Path:** [`TLSTM_Klint.ipynb`](TLSTM_Klint.ipynb)
* Complete 11-step interactive pipeline:
  1. Hardware verification & environment setup
  2. Teacher model loading (`klint_32m_v2_release.pt` from Hugging Face)
  3. Architectural comparison & parameter plots
  4. Multi-asset distillation data ingestion
  5. Knowledge distillation training loop with inline loss curves
  6. 325 fresh asset universe ingestion
  7. High-throughput GPU evaluation
  8. 10-test institutional quantitative benchmark execution
  9. Inline visual display of all 20 charts
  10. Head-to-head scorecard table (Teacher vs Student)
  11. 1-Click ZIP download of `TLSTM_multitest_results.zip`

---

## 💻 CLI Execution

```bash
# 1. Run complete distillation and 10-test battery from command line:
python Tlstm-Klint/run_all_tests.py \
    --teacher_checkpoint checkpoints/klint_32m_v2_release.pt \
    --student_checkpoint checkpoints/tlstm_klint_distilled.pt \
    --max_assets 325 \
    --output_dir Tlstm-Klint/TLSTM-multitest \
    --epochs 5 \
    --batch_size 64

# 2. Evaluate existing student model without re-training:
python Tlstm-Klint/run_all_tests.py \
    --student_checkpoint checkpoints/tlstm_klint_distilled.pt \
    --skip_training \
    --max_assets 325 \
    --output_dir Tlstm-Klint/TLSTM-multitest
```

---

## 📁 Folder Structure

```text
Tlstm-Klint/
├── TLSTM_Klint.ipynb         # Master Google Colab 1-click interactive notebook
├── README.md                 # Technical documentation & distillation specification
├── tlstm_model.py            # 0.59M Temporal LSTM student architecture
├── distillation_loss.py      # Soft KD + Hard CE + PnL + Directional hinge loss
├── trainer.py                # Distillation training loop & model bundler
├── evaluator.py              # GPU-accelerated student inference across 300+ assets
├── test_battery.py           # 10 institutional quantitative test modules (20 plots)
├── fresh_universe.py         # Curated 325+ unseen assets (0% training overlap)
├── data_loader.py            # High-throughput data loader with dual-cache lookup
├── run_all_tests.py          # Unified CLI benchmark runner
└── TLSTM-multitest/          # Output directory for 20 visual plots and report
```
