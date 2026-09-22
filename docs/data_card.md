# Klint Data Card & Data Contract

## 1. Data Governance Principles

1. **No Proprietary Redistribution**: Raw vendor data will never be committed or redistributed in this repository unless an open data license explicitly permits it.
2. **Deterministic Manifests**: Datasets are specified via cryptographic checksums (SHA-256) and source metadata manifests.
3. **Reproducible Pipeline**: Preprocessing from raw ticks or vendor klines into Klint-compliant tensors must be deterministic and fully reproducible.

## 2. Canonical Bar Schema

Each bar in the pipeline is standardized into the following representation:

| Index | Field | Description | Type | Constraint |
|---|---|---|---|---|
| 0 | `timestamp_open` | Unix epoch in milliseconds | `int64` | Monotonically increasing |
| 1 | `open` | First traded price | `float32` | $> 0$ |
| 2 | `high` | Maximum traded price | `float32` | $\ge \max(\text{open}, \text{close})$ |
| 3 | `low` | Minimum traded price | `float32` | $\le \min(\text{open}, \text{close})$ |
| 4 | `close` | Last traded price | `float32` | $> 0$ |
| 5 | `volume` | Base asset transaction volume | `float32` | $\ge 0$ |
| 6 | `timestamp_close` | Close epoch in milliseconds | `int64` | `> timestamp_open` |
| 7 | `turnover` | Quote asset turnover | `float32` | $\ge 0$ (optional) |
| 8 | `trade_count` | Number of executed trades | `float32` | $\ge 0$ (optional) |

## 3. Preprocessing Invariants & Validation

Before ingestion into the Klint factor pipeline, the data validator enforces:
* Zero non-positive prices ($O, H, L, C > 0$).
* Complete geometric validity: $H \ge \max(O, C)$ and $L \le \min(O, C)$.
* Monotonic chronological ordering without duplicate timestamps.
* Missing bar detection (flagging time gaps greater than the nominal bar interval).
