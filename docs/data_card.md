# Klint Data Card & Dataset Provenance

## 1. Data Governance Principles

1. **Deterministic Processing**: Raw ticks and vendor candlestick bars are mapped to stationary factor tensors through deterministic transformations with fixed random seeds.
2. **Strict Causal Normalization**: Feature statistics (e.g. volume exponential moving averages) are computed causally with online updating. Retrospective min-max or global z-score normalization is strictly forbidden.
3. **No Proprietary Data Leakage**: Raw proprietary vendor data is never redistributed in the repository. Public demonstration sets and cached tokens are provided under open licenses.

---

## 2. Canonical Bar Schema

Every bar in the Klint pipeline is normalized to the following memory-aligned schema:

| Index | Field | Description | Type | Physical Invariant Constraint |
|:---|:---|:---|:---|:---|
| 0 | `timestamp_open` | Unix epoch in milliseconds | `int64` | Monotonically strictly increasing ($t_i > t_{i-1}$) |
| 1 | `open` ($O$) | First traded price | `float32` | $O > 0$ |
| 2 | `high` ($H$) | Maximum traded price | `float32` | $H \ge \max(O, C)$ |
| 3 | `low` ($L$) | Minimum traded price | `float32` | $L \le \min(O, C)$ |
| 4 | `close` ($C$) | Last traded price | `float32` | $C > 0$ |
| 5 | `volume` ($V$) | Base asset transaction volume | `float32` | $V \ge 0$ |
| 6 | `timestamp_close`| Close epoch in milliseconds | `int64` | $t_{\text{close}} > t_{\text{open}}$ |
| 7 | `turnover` | Quote asset turnover volume | `float32` | Optional ($\ge 0$) |
| 8 | `trade_count` | Number of executed fills | `float32` | Optional ($\ge 0$) |

---

## 3. Primary Pre-Training Dataset: Solana Intraday (`SOL.npy` / `sol_tokens.pt`)

The primary pre-training dataset represents high-frequency, high-volatility cryptocurrency dynamics:

* **Asset:** Solana (SOL / USDT)
* **Resolution:** 1-minute candlestick bars
* **Total Bars:** 1,591,983 bars
* **Total Factor Tokens:** 4,775,949 discrete tokens (3 tokens per bar: Price, Range, Activity)
* **File Size:** ~70.0 MB raw NumPy array (`data/SOL.npy`) $\to$ ~36.4 MB PyTorch token tensor (`data/sol_tokens.pt`)
* **Partitioning:**
  * **Training Split:** 169,021 sequences of length 256 bars (768 factor tokens)
  * **Validation Split:** 7,386 sequences of length 256 bars
  * **Partition Ratio:** ~95% Train / 5% Validation with chronological ordering

---

## 4. Multi-Asset Benchmark Universe (320+ Assets)

For out-of-distribution evaluation, Klint integrates a curated 320+ asset universe across 6 distinct asset classes managed via [`src/klint/benchmark/universe.py`](file:///d:/Klint/Klint-32M/src/klint/benchmark/universe.py):

| Asset Class | Count | Key Representatives | Dynamics Characteristics |
| :--- | :--- | :--- | :--- |
| **US Equities** | 120+ | `AAPL`, `MSFT`, `NVDA`, `TSLA`, `GOOGL`, `AMZN`, `JPM`, `XOM` | High liquidity, overnight gap risk, corporate earnings jumps. |
| **Sector ETFs** | 40+ | `SPY`, `QQQ`, `IWM`, `XLF`, `XLE`, `XLK`, `SOXX`, `ARKK` | Macro market beta, thematic sector rotation. |
| **Cryptocurrencies** | 50+ | `BTC-USD`, `ETH-USD`, `SOL-USD`, `BNB-USD`, `ADA-USD`, `AVAX-USD` | 24/7 continuous trading, heavy tail-risk, regime volatility. |
| **Commodities** | 25+ | `GLD` (Gold), `SLV` (Silver), `USO` (Oil), `UNG` (Gas), `CPER` (Copper) | Geopolitical sensitivity, mean-reverting seasonality. |
| **Foreign Exchange (FX)** | 35+ | `EURUSD=X`, `USDJPY=X`, `GBPUSD=X`, `AUDUSD=X`, `USDCAD=X` | Low intraday drift, macro monetary policy sensitivity. |
| **Fixed Income & Rates** | 20+ | `TLT`, `IEF`, `SHY`, `HYG`, `LQD`, `BND` | Interest rate duration sensitivity, macroeconomic regime shifts. |

Data is retrieved via [`MultiAssetDataFetcher`](file:///d:/Klint/Klint-32M/src/klint/benchmark/data_fetcher.py) using `yfinance` with local disk caching in `data/yfinance_cache/` to ensure deterministic re-evaluation and eliminate network rate-limiting.

---

## 5. Invariant Validation Pipeline

Before any dataset is admitted into factor decomposition or model training, it is verified through [`validate_ohlcv`](file:///d:/Klint/Klint-32M/src/klint/data/validator.py):
1. **Physical Boundary Checks**: Asserts $H \ge \max(O, C)$ and $L \le \min(O, C)$.
2. **Positivity Constraint**: Asserts $O, H, L, C > 0$.
3. **Chronological Continuity**: Asserts strictly monotonic timestamps.
4. **Volume Non-Negativity**: Asserts $V \ge 0$. Any anomalies are flagged and repaired before tokenization.
