# Klint Architecture Specification

## 1. Overview

**Klint** is a causal generative foundation model designed for financial market time series. Rather than treating market bars as arbitrary vectors, Klint decomposes market bars into three stationary causal factor streams:
1. **Price-Path ($F_{\text{price}}$)**: Directional movement and drift.
2. **Range-Shape ($F_{\text{range}}$)**: Intrabar volatility, body location, and wick bounds.
3. **Activity ($F_{\text{activity}}$)**: Market participation, volume, and intensity.

The architecture tokenizes each factor stream with dedicated Residual Vector Quantization (RVQ) codebooks, models the joint distribution autoregressively with a causal Transformer backbone, and reconstructs the trajectories using a structurally constrained geometric decoder.

---

## 2. Causal Factorization

Let a sequence of market bars be denoted by $B = (b_1, b_2, \dots, b_T)$, where each bar $b_t = (O_t, H_t, L_t, C_t, V_t)$ satisfies:
$$H_t \ge \max(O_t, C_t) \quad \text{and} \quad L_t \le \min(O_t, C_t), \quad V_t \ge 0$$

Klint factorizes the joint probability of the bar sequence as:
$$P(B_{1:T}) = \prod_{t=1}^T P(b_t \mid b_{<t})$$

Within each bar $t$, the three factor streams follow a strict causal hierarchy:
$$P(b_t \mid b_{<t}) = P(p_t \mid b_{<t}) \cdot P(r_t \mid p_t, b_{<t}) \cdot P(a_t \mid p_t, r_t, b_{<t})$$
where:
* $p_t \in \{1, \dots, 512\}$ is the discrete price-path code.
* $r_t \in \{1, \dots, 256\}$ is the discrete range-shape code conditioned on $p_t$.
* $a_t \in \{1, \dots, 256\}$ is the discrete activity code conditioned on $(p_t, r_t)$.

---

## 3. Stationary Feature Extraction

To eliminate non-stationarity across orders of magnitude, inputs are mapped into scale-invariant stationary metrics:

### 3.1 Price-Path Stream ($F_{\text{price}}$)
* **Body Return:**
  $$r_t^{\text{body}} = \ln\left(\frac{C_t}{O_t}\right)$$
* **Gap Return:**
  $$r_t^{\text{gap}} = \ln\left(\frac{O_t}{C_{t-1}}\right)$$

### 3.2 Range-Shape Stream ($F_{\text{range}}$)
* **Log Relative Range:**
  $$\rho_t = \ln\left(\frac{H_t}{L_t}\right)$$
* **Upper Wick Ratio:**
  $$u_t = \frac{H_t - \max(O_t, C_t)}{H_t - L_t + \epsilon} \in [0, 1]$$
* **Lower Wick Ratio:**
  $$l_t = \frac{\min(O_t, C_t) - L_t}{H_t - L_t + \epsilon} \in [0, 1]$$

### 3.3 Activity Stream ($F_{\text{activity}}$)
* **Normalized Log Relative Volume:**
  $$v_t = \ln(1 + V_t) - \ln(1 + \bar{V}_t)$$
  where $\bar{V}_t = \text{EMA}(V, \text{span}=60)$ represents the local baseline volume.

---

## 4. Residual Vector Quantization (RVQ)

Continuous factor vectors $x \in \mathbb{R}^d$ are quantized into discrete code indices:
$$\hat{x} = \sum_{m=1}^M e_{k_m}^{(m)}$$
where each stage $m$ quantizes the residual $r_m = x - \sum_{j=1}^{m-1} e_{k_j}^{(j)}$.
Codebook vectors are updated via Exponential Moving Average (EMA) with a dead-code revival mechanism to ensure codebook utilization remains $>95\%$.

* **Price Codebook:** $K_p = 512$ codes.
* **Range Codebook:** $K_r = 256$ codes.
* **Activity Codebook:** $K_a = 256$ codes.

---

## 5. Structural Geometric Decoder

Standard regression heads frequently violate financial invariants. Klint's decoder enforces geometry by structural construction:

$$\hat{O}_t = C_{t-1} \cdot \exp(\hat{r}_t^{\text{gap}})$$
$$\hat{C}_t = \hat{O}_t \cdot \exp(\hat{r}_t^{\text{body}})$$
$$\hat{H}_t = \max(\hat{O}_t, \hat{C}_t) + \text{Softplus}(\hat{w}_{u, t})$$
$$\hat{L}_t = \min(\hat{O}_t, \hat{C}_t) - \text{Softplus}(\hat{w}_{l, t})$$
$$\hat{V}_t = \text{Softplus}(\hat{v}_t)$$

Since $\text{Softplus}(x) = \ln(1 + e^x) > 0$, the invariants:
$$\hat{H}_t \ge \max(\hat{O}_t, \hat{C}_t) \quad \text{and} \quad \hat{L}_t \le \min(\hat{O}_t, \hat{C}_t)$$
are mathematically guaranteed for all possible network latent vectors.

---

## 6. Klint-32M Foundation Backbone

* **Layers ($L$):** 10
* **Hidden Dimension ($d_{\text{model}}$):** 480
* **Attention Heads ($h$):** 10 ($d_{\text{head}} = 48$)
* **FFN Dimension ($d_{\text{ff}}$):** 1920 ($4 \times d_{\text{model}}$)
* **Positional Encoding:** Rotary Position Embedding (RoPE) applied to query and key vectors.
* **Normalization:** Pre-attention and Pre-FFN Root Mean Square Normalization (RMSNorm).
* **Sequence Length:** 1,024 bars decomposed into 3,072 factor tokens.
* **Total Parameters:** Approximately 30.6M parameters.
