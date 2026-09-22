"""Distributional testing and financial stylized facts verification."""

from dataclasses import dataclass
from typing import Dict, Any, Tuple
import numpy as np
import scipy.stats as stats


def autocorr(x: np.ndarray, max_lags: int = 20) -> np.ndarray:
    """Computes autocorrelation function up to max_lags."""
    x = np.asarray(x, dtype=np.float64)
    x = x - np.mean(x)
    var = np.var(x)
    if var < 1e-12:
        return np.zeros(max_lags + 1)
    n = len(x)
    r = np.correlate(x, x, mode='full')[-n:]
    return r[:max_lags + 1] / (var * n)


@dataclass
class DistributionalMoments:
    """Statistical moments and stylized facts for a return/volatility distribution."""
    mean: float
    std: float
    skewness: float
    kurtosis: float                   # Excess kurtosis (> 0 indicates fat tails)
    vol_mean: float
    vol_std: float
    vol_p05: float
    vol_p50: float
    vol_p95: float
    acf_raw_lag1: float
    acf_raw_lag5: float
    acf_abs_lag1: float               # Volatility clustering at lag 1
    acf_abs_lag5: float               # Volatility clustering at lag 5
    corr_return_volume: float
    corr_return_future_vol: float     # Leverage effect


@dataclass
class DistributionalComparisonResult:
    """Comparison of stylized facts between real empirical and synthetic generated data."""
    real_moments: DistributionalMoments
    synthetic_moments: DistributionalMoments
    wasserstein_returns: float
    wasserstein_volatility: float
    ks_stat_returns: float
    ks_pvalue_returns: float
    volatility_clustering_captured: bool  # True if ACF(|r|) > 0 and decays slowly


class DistributionalAnalyzer:
    """Analyzes whether generative trajectories reproduce empirical financial stylized facts."""

    def compute_moments(self, ohlcv: np.ndarray) -> DistributionalMoments:
        """Computes statistical moments and stylized facts for an OHLCV trajectory."""
        closes = ohlcv[:, 3]
        volumes = ohlcv[:, 4]
        highs = ohlcv[:, 1]
        lows = ohlcv[:, 2]

        returns = np.diff(np.log(closes + 1e-8))
        vol = np.log(highs[1:] / (lows[1:] + 1e-8) + 1e-8)

        mean_r = float(np.mean(returns))
        std_r = float(np.std(returns))
        skew_r = float(stats.skew(returns))
        kurt_r = float(stats.kurtosis(returns))  # Excess kurtosis

        mean_v = float(np.mean(vol))
        std_v = float(np.std(vol))
        p05_v = float(np.percentile(vol, 5))
        p50_v = float(np.percentile(vol, 50))
        p95_v = float(np.percentile(vol, 95))

        # Autocorrelations
        acf_raw = autocorr(returns, max_lags=10)
        acf_abs = autocorr(np.abs(returns), max_lags=10)

        # Cross-correlations
        valid_len = len(returns)
        vol_aligned = volumes[1:valid_len + 1]
        if np.std(vol_aligned) > 1e-8 and std_r > 1e-8:
            corr_rv, _ = stats.pearsonr(returns, vol_aligned)
        else:
            corr_rv = 0.0

        # Leverage effect: corr(r_t, vol_{t+1})
        if len(returns) > 2 and np.std(vol[1:]) > 1e-8:
            corr_lev, _ = stats.pearsonr(returns[:-1], vol[1:])
        else:
            corr_lev = 0.0

        return DistributionalMoments(
            mean=mean_r,
            std=std_r,
            skewness=skew_r,
            kurtosis=kurt_r,
            vol_mean=mean_v,
            vol_std=std_v,
            vol_p05=p05_v,
            vol_p50=p50_v,
            vol_p95=p95_v,
            acf_raw_lag1=float(acf_raw[1]) if len(acf_raw) > 1 else 0.0,
            acf_raw_lag5=float(acf_raw[5]) if len(acf_raw) > 5 else 0.0,
            acf_abs_lag1=float(acf_abs[1]) if len(acf_abs) > 1 else 0.0,
            acf_abs_lag5=float(acf_abs[5]) if len(acf_abs) > 5 else 0.0,
            corr_return_volume=float(np.nan_to_num(corr_rv)),
            corr_return_future_vol=float(np.nan_to_num(corr_lev)),
        )

    def compare_distributions(
        self,
        real_ohlcv: np.ndarray,
        synthetic_ohlcv: np.ndarray,
    ) -> DistributionalComparisonResult:
        """Compares empirical reality against generative synthesis."""
        real_m = self.compute_moments(real_ohlcv)
        synth_m = self.compute_moments(synthetic_ohlcv)

        r_ret = np.diff(np.log(real_ohlcv[:, 3] + 1e-8))
        s_ret = np.diff(np.log(synthetic_ohlcv[:, 3] + 1e-8))

        r_vol = np.log(real_ohlcv[1:, 1] / (real_ohlcv[1:, 2] + 1e-8) + 1e-8)
        s_vol = np.log(synthetic_ohlcv[1:, 1] / (synthetic_ohlcv[1:, 2] + 1e-8) + 1e-8)

        w_ret = float(stats.wasserstein_distance(r_ret, s_ret))
        w_vol = float(stats.wasserstein_distance(r_vol, s_vol))

        ks_stat, ks_pval = stats.ks_2samp(r_ret, s_ret)

        # Volatility clustering check: ACF(|r|) must be positive at lag 1 and lag 5
        clustered = (synth_m.acf_abs_lag1 > 0.05 and synth_m.acf_abs_lag5 > 0.0)

        return DistributionalComparisonResult(
            real_moments=real_m,
            synthetic_moments=synth_m,
            wasserstein_returns=w_ret,
            wasserstein_volatility=w_vol,
            ks_stat_returns=float(ks_stat),
            ks_pvalue_returns=float(ks_pval),
            volatility_clustering_captured=clustered,
        )
