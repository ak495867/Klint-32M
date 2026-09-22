"""Core multi-horizon forecasting benchmark: Price, Return, and Realized Volatility."""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import scipy.stats as stats
import torch

from klint.models.klint_32m import Klint32M
from klint.tokenizer.factor_tokenizer import FactorTokenizer
from klint.tokenizer.geometric_decoder import GeometricDecoder
from klint.data.factors import FactorDecomposer


def calc_ic_rankic(y_pred: np.ndarray, y_true: np.ndarray) -> Tuple[float, float]:
    """Calculates Pearson IC and Spearman RankIC safely."""
    if len(y_pred) < 3:
        return 0.0, 0.0
    # Clean NaN / inf
    mask = np.isfinite(y_pred) & np.isfinite(y_true)
    if np.sum(mask) < 3:
        return 0.0, 0.0
    yp = y_pred[mask]
    yt = y_true[mask]

    std_p = np.std(yp)
    std_t = np.std(yt)
    if std_p < 1e-8 or std_t < 1e-8:
        return 0.0, 0.0

    ic, _ = stats.pearsonr(yp, yt)
    rank_ic, _ = stats.spearmanr(yp, yt)
    return float(np.nan_to_num(ic)), float(np.nan_to_num(rank_ic))


@dataclass
class HorizonForecastMetric:
    """Evaluation metrics for a specific forecasting horizon."""
    horizon: int
    mae: float
    rmse: float
    mape_pct: float
    ic: float
    rank_ic: float
    directional_accuracy_pct: float


@dataclass
class ReturnForecastMetric:
    """Evaluation metrics for cumulative return forecasting."""
    horizon: int
    ic: float
    rank_ic: float
    mae: float
    rmse: float
    directional_hit_rate_pct: float


@dataclass
class VolatilityForecastMetric:
    """Evaluation metrics for realized volatility forecasting."""
    horizon: int
    mae: float
    rmse: float
    r_squared: float
    correlation: float
    rank_correlation: float
    low_vol_mae: float
    normal_vol_mae: float
    high_vol_mae: float


@dataclass
class CoreForecastingResult:
    """Aggregated core forecasting benchmark scorecard."""
    ticker: str
    price_metrics: List[HorizonForecastMetric]
    return_metrics: List[ReturnForecastMetric]
    volatility_metrics: List[VolatilityForecastMetric]


class CoreForecastingEvaluator:
    """
    Evaluates multi-horizon price, return, and realized volatility forecasting fidelity.
    """

    PRICE_HORIZONS = [5, 10, 20, 50, 100]
    RETURN_HORIZONS = [1, 5, 10, 20, 50]
    VOL_HORIZONS = [5, 10, 20, 50]

    def __init__(
        self,
        model: Klint32M,
        tokenizer: FactorTokenizer,
        decoder: GeometricDecoder,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer.to(device)
        self.tokenizer.eval()
        self.decoder = decoder.to(device)
        self.device = device
        self.decomposer = FactorDecomposer()

    @torch.no_grad()
    def evaluate_trajectories(
        self,
        ohlcv: np.ndarray,
        ticker: str = "SOL",
        context_bars: int = 64,
        num_eval_points: int = 30,
        max_horizon: int = 100,
        temperature: float = 0.6,
        top_k: int = 30,
    ) -> CoreForecastingResult:
        """
        Runs multi-step autoregressive trajectory generation across multiple evaluation points,
        comparing predicted vs realized future paths.
        """
        N = len(ohlcv)
        if N < context_bars + max_horizon + 10:
            raise ValueError(f"OHLCV series ({N} bars) is too short for max horizon {max_horizon} + context {context_bars}.")

        factors = self.decomposer.decompose(ohlcv)
        p_tensor = torch.from_numpy(factors.price_path).unsqueeze(0).to(self.device)
        r_tensor = torch.from_numpy(factors.range_shape).unsqueeze(0).to(self.device)
        a_tensor = torch.from_numpy(factors.activity).unsqueeze(0).to(self.device)

        p_tok, r_tok, a_tok, _ = self.tokenizer.encode(p_tensor, r_tensor, a_tensor)
        interleaved = self.tokenizer.interleave(p_tok, r_tok, a_tok).squeeze(0)  # (3N,)

        # Sample evaluation start points uniformly
        valid_range = list(range(context_bars, N - max_horizon, max(1, (N - max_horizon - context_bars) // num_eval_points)))
        eval_points = valid_range[:num_eval_points]

        pred_price_paths = []  # List of arrays of length max_horizon
        true_price_paths = []

        for pt in eval_points:
            # Context window tokens
            c_start = (pt - context_bars) * 3
            c_end = pt * 3
            prompt = interleaved[c_start:c_end].unsqueeze(0)  # (1, 3 * context_bars)

            # Autoregressively generate max_horizon future bars
            gen = self.model.generate_tokens(prompt, num_bars=max_horizon, temperature=temperature, top_k=top_k)
            new_tokens = gen[:, prompt.size(1):]

            new_p, new_r, new_a = self.tokenizer.deinterleave(new_tokens)
            rec_p, rec_r, rec_a = self.tokenizer.decode_tokens(new_p, new_r, new_a)

            anchor_price = float(ohlcv[pt - 1, 3])
            synth_ohlcv = self.decoder(rec_p, rec_r, rec_a, anchor_price=anchor_price).squeeze(0).cpu().numpy()

            pred_closes = synth_ohlcv[:, 3]
            true_closes = ohlcv[pt:pt + max_horizon, 3]

            pred_price_paths.append(pred_closes)
            true_price_paths.append(true_closes)

        pred_price_paths = np.array(pred_price_paths)  # (M, max_horizon)
        true_price_paths = np.array(true_price_paths)  # (M, max_horizon)
        anchor_prices = np.array([ohlcv[pt - 1, 3] for pt in eval_points])  # (M,)

        # 1. Price Forecasting Metrics
        price_results = []
        for H in self.PRICE_HORIZONS:
            if H > max_horizon:
                continue
            pred_sub = pred_price_paths[:, :H]
            true_sub = true_price_paths[:, :H]

            mae = float(np.mean(np.abs(pred_sub - true_sub)))
            rmse = float(np.sqrt(np.mean((pred_sub - true_sub) ** 2)))
            mape = float(np.mean(np.abs((pred_sub - true_sub) / (true_sub + 1e-8))) * 100.0)

            ic, rank_ic = calc_ic_rankic(pred_sub.flatten(), true_sub.flatten())

            # Directional accuracy relative to anchor price at horizon H
            pred_dir = np.sign(pred_sub[:, -1] - anchor_prices)
            true_dir = np.sign(true_sub[:, -1] - anchor_prices)
            valid_dir = (pred_dir != 0) & (true_dir != 0)
            dir_acc = float(np.mean(pred_dir[valid_dir] == true_dir[valid_dir]) * 100.0) if np.any(valid_dir) else 50.0

            price_results.append(HorizonForecastMetric(
                horizon=H,
                mae=mae,
                rmse=rmse,
                mape_pct=mape,
                ic=ic,
                rank_ic=rank_ic,
                directional_accuracy_pct=dir_acc,
            ))

        # 2. Return Forecasting Metrics
        return_results = []
        for H in self.RETURN_HORIZONS:
            if H > max_horizon:
                continue
            pred_h_close = pred_price_paths[:, H - 1]
            true_h_close = true_price_paths[:, H - 1]

            pred_r = (pred_h_close - anchor_prices) / anchor_prices
            true_r = (true_h_close - anchor_prices) / anchor_prices

            ic, rank_ic = calc_ic_rankic(pred_r, true_r)
            mae = float(np.mean(np.abs(pred_r - true_r)))
            rmse = float(np.sqrt(np.mean((pred_r - true_r) ** 2)))

            valid_r = (pred_r != 0) & (true_r != 0)
            hit_rate = float(np.mean(np.sign(pred_r[valid_r]) == np.sign(true_r[valid_r])) * 100.0) if np.any(valid_r) else 50.0

            return_results.append(ReturnForecastMetric(
                horizon=H,
                ic=ic,
                rank_ic=rank_ic,
                mae=mae,
                rmse=rmse,
                directional_hit_rate_pct=hit_rate,
            ))

        # 3. Realized Volatility Metrics
        vol_results = []
        for H in self.VOL_HORIZONS:
            if H > max_horizon or H < 2:
                continue
            pred_log_ret = np.diff(np.log(pred_price_paths[:, :H]), axis=1)  # (M, H-1)
            true_log_ret = np.diff(np.log(true_price_paths[:, :H]), axis=1)  # (M, H-1)

            pred_rv = np.sum(pred_log_ret ** 2, axis=1)  # (M,)
            true_rv = np.sum(true_log_ret ** 2, axis=1)  # (M,)

            mae = float(np.mean(np.abs(pred_rv - true_rv)))
            rmse = float(np.sqrt(np.mean((pred_rv - true_rv) ** 2)))

            # R^2
            ss_tot = np.sum((true_rv - np.mean(true_rv)) ** 2) + 1e-8
            ss_res = np.sum((true_rv - pred_rv) ** 2)
            r2 = float(max(-1.0, 1.0 - (ss_res / ss_tot)))

            corr, rank_corr = calc_ic_rankic(pred_rv, true_rv)

            # Volatility Regimes: Low (bottom 25%), Normal (25-75%), High (top 25%)
            q25 = np.percentile(true_rv, 25)
            q75 = np.percentile(true_rv, 75)

            low_mask = true_rv <= q25
            high_mask = true_rv >= q75
            norm_mask = (true_rv > q25) & (true_rv < q75)

            low_mae = float(np.mean(np.abs(pred_rv[low_mask] - true_rv[low_mask]))) if np.any(low_mask) else mae
            norm_mae = float(np.mean(np.abs(pred_rv[norm_mask] - true_rv[norm_mask]))) if np.any(norm_mask) else mae
            high_mae = float(np.mean(np.abs(pred_rv[high_mask] - true_rv[high_mask]))) if np.any(high_mask) else mae

            vol_results.append(VolatilityForecastMetric(
                horizon=H,
                mae=mae,
                rmse=rmse,
                r_squared=r2,
                correlation=corr,
                rank_correlation=rank_corr,
                low_vol_mae=low_mae,
                normal_vol_mae=norm_mae,
                high_vol_mae=high_mae,
            ))

        return CoreForecastingResult(
            ticker=ticker,
            price_metrics=price_results,
            return_metrics=return_results,
            volatility_metrics=vol_results,
        )
