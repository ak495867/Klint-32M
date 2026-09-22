"""Train-on-Synthetic, Test-on-Real (TSTR) downstream utility benchmark."""

from dataclasses import dataclass
from typing import Dict, Any, Tuple
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

from klint.eval.core_forecasting import calc_ic_rankic


@dataclass
class TSTRResult:
    """Evaluation result comparing downstream model trained on Real vs Synthetic data."""
    trtr_ic: float                     # Train-Real-Test-Real Pearson IC
    trtr_rank_ic: float                # Train-Real-Test-Real Spearman RankIC
    trtr_mse: float                    # TRTR Mean Squared Error
    tstr_ic: float                     # Train-Synthetic-Test-Real Pearson IC
    tstr_rank_ic: float                # Train-Synthetic-Test-Real Spearman RankIC
    tstr_mse: float                    # TSTR Mean Squared Error
    delta_ic: float                    # tstr_ic - trtr_ic
    delta_rank_ic: float               # tstr_rank_ic - trtr_rank_ic
    utility_retention_pct: float       # (tstr_ic / trtr_ic) * 100%


class TSTREvaluator:
    """
    Measures downstream utility of Klint synthetic data by training separate forecasting models
    and testing them on strictly out-of-sample real market data.
    """

    def __init__(self, lookback_lags: int = 10, alpha_reg: float = 1.0):
        self.lags = lookback_lags
        self.alpha = alpha_reg

    def _prepare_features(self, returns: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Creates autoregressive feature matrix (X) and target returns (y)."""
        N = len(returns)
        if N <= self.lags + 1:
            raise ValueError(f"Length {N} is too short for lag {self.lags}.")

        X = []
        y = []
        for i in range(self.lags, N):
            X.append(returns[i - self.lags:i])
            y.append(returns[i])
        return np.array(X), np.array(y)

    def evaluate_tstr(
        self,
        real_train_ohlcv: np.ndarray,
        real_test_ohlcv: np.ndarray,
        synthetic_ohlcv: np.ndarray,
    ) -> TSTRResult:
        """
        Trains downstream model on real vs synthetic data and tests on real test set.
        """
        r_train = np.diff(np.log(real_train_ohlcv[:, 3] + 1e-8))
        r_test = np.diff(np.log(real_test_ohlcv[:, 3] + 1e-8))
        s_train = np.diff(np.log(synthetic_ohlcv[:, 3] + 1e-8))

        X_real_tr, y_real_tr = self._prepare_features(r_train)
        X_test, y_test = self._prepare_features(r_test)
        X_synth_tr, y_synth_tr = self._prepare_features(s_train)

        # 1. Model A: Train on Real, Test on Real (TRTR)
        model_trtr = Ridge(alpha=self.alpha)
        model_trtr.fit(X_real_tr, y_real_tr)
        pred_trtr = model_trtr.predict(X_test)

        ic_trtr, rank_ic_trtr = calc_ic_rankic(pred_trtr, y_test)
        mse_trtr = float(mean_squared_error(y_test, pred_trtr))

        # 2. Model B: Train on Synthetic, Test on Real (TSTR)
        model_tstr = Ridge(alpha=self.alpha)
        model_tstr.fit(X_synth_tr, y_synth_tr)
        pred_tstr = model_tstr.predict(X_test)

        ic_tstr, rank_ic_tstr = calc_ic_rankic(pred_tstr, y_test)
        mse_tstr = float(mean_squared_error(y_test, pred_tstr))

        delta_ic = ic_tstr - ic_trtr
        delta_rank_ic = rank_ic_tstr - rank_ic_trtr
        retention = (ic_tstr / (ic_trtr + 1e-8)) * 100.0 if ic_trtr > 0 else 0.0

        return TSTRResult(
            trtr_ic=float(ic_trtr),
            trtr_rank_ic=float(rank_ic_trtr),
            trtr_mse=mse_trtr,
            tstr_ic=float(ic_tstr),
            tstr_rank_ic=float(rank_ic_tstr),
            tstr_mse=mse_tstr,
            delta_ic=float(delta_ic),
            delta_rank_ic=float(delta_rank_ic),
            utility_retention_pct=float(retention),
        )
