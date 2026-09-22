"""Quantitative evaluation engine assessing multi-asset forecasting accuracy, risk, and PnL metrics."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import numpy as np
import torch
import torch.nn.functional as F

from klint.data.factors import FactorDecomposer
from klint.data.validator import validate_ohlcv
from klint.tokenizer.factor_tokenizer import FactorTokenizer
from klint.tokenizer.geometric_decoder import GeometricDecoder
from klint.models.klint_32m import Klint32M


@dataclass
class AssetBenchmarkResult:
    """Quantitative performance scorecard for a single asset."""
    ticker: str
    name: str
    asset_class: str
    bars_count: int
    directional_accuracy: float      # % correctly predicted price direction
    annualized_sharpe: float         # Risk-adjusted return ratio (252-day annualized)
    annualized_sortino: float        # Downside risk-adjusted ratio
    max_drawdown_pct: float          # Peak-to-trough decline (%)
    cumulative_return_pct: float     # Total strategy return (%)
    buy_hold_return_pct: float       # Benchmark passive return (%)
    win_rate_pct: float              # % profitable trades
    profit_factor: float             # Gross profits / Gross losses
    volatility_annualized: float     # Annualized realized volatility (%)
    invariant_validity_pct: float    # Structural candle integrity pass rate (%)
    equity_curve: np.ndarray         # Cumulative equity trajectory
    drawdown_curve: np.ndarray       # Underwater drawdown curve


class MultiAssetEvaluator:
    """
    Evaluates a trained Klint model and Tokenizer across hundreds of assets.
    """
    def __init__(
        self,
        model: Klint32M,
        tokenizer: FactorTokenizer,
        decomposer: Optional[FactorDecomposer] = None,
        decoder: Optional[GeometricDecoder] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        fee_bps: float = 5.0,  # 5 basis points transaction fee
    ):
        self.device = device
        self.model = model.to(self.device)
        self.model.eval()
        self.tokenizer = tokenizer.to(self.device)
        self.tokenizer.eval()
        self.decomposer = decomposer or FactorDecomposer()
        self.decoder = (decoder or GeometricDecoder()).to(self.device)
        self.fee = fee_bps / 10000.0

    @torch.no_grad()
    def evaluate_asset(
        self,
        asset_data: Dict[str, Any],
        context_bars: int = 64,
        eval_horizon: int = 150,
    ) -> Optional[AssetBenchmarkResult]:
        """
        Evaluates model on a single asset's historical trajectory.
        """
        ticker = asset_data["ticker"]
        name = asset_data.get("name", ticker)
        asset_class = asset_data.get("asset_class", "Equities")
        ohlcv = asset_data["ohlcv"]
        N = len(ohlcv)

        if N < context_bars + 20:
            return None

        # 1. Structural Invariant Check
        is_valid, _ = validate_ohlcv(ohlcv)
        invariant_pass_rate = 100.0 if is_valid else 0.0

        # 2. Extract Causal Factors
        factors = self.decomposer.decompose(ohlcv)

        # 3. Tokenize factors
        p_tensor = torch.from_numpy(factors.price_path).unsqueeze(0).to(self.device)
        r_tensor = torch.from_numpy(factors.range_shape).unsqueeze(0).to(self.device)
        a_tensor = torch.from_numpy(factors.activity).unsqueeze(0).to(self.device)

        p_tok, r_tok, a_tok, _ = self.tokenizer.encode(p_tensor, r_tensor, a_tensor)
        interleaved = self.tokenizer.interleave(p_tok, r_tok, a_tok).squeeze(0)  # (3N,)

        # Precompute price codebook returns for expectation estimation
        emb_p = self.tokenizer.rvq_price.layers[0].embedding
        dec_p_weights = self.tokenizer.dec_price(emb_p)  # (512, 2)
        code_body_returns = dec_p_weights[:, 1]          # (512,) body returns

        # 4. Out-of-Sample Rolling Simulation
        eval_bars = min(eval_horizon, N - context_bars - 1)
        if eval_bars <= 10:
            return None

        actual_returns = []
        pred_returns = []
        strategy_returns = []

        start_bar = N - eval_bars

        for b in range(start_bar, N):
            # Window of past context_bars
            hist_start = (b - context_bars) * 3
            hist_end = b * 3
            prompt_seq = interleaved[hist_start:hist_end].unsqueeze(0)  # (1, 3 * context_bars)

            # Next token to predict is next bar's Price code (position % 3 == 0)
            out = self.model(prompt_seq)
            last_hidden = out["hidden"][:, -1, :]  # (1, d)
            logits_p = self.model.head_price(last_hidden).squeeze(0)  # (512,)

            probs_p = F.softmax(logits_p, dim=-1)
            # Expected return according to probability distribution
            expected_r = torch.sum(probs_p * code_body_returns).item()

            # Realized body return
            real_r = float(factors.price_path[b, 1])

            actual_returns.append(real_r)
            pred_returns.append(expected_r)

            # Strategy signal: long if expected return > fee, short if < -fee
            signal = 0.0
            if expected_r > self.fee:
                signal = 1.0
            elif expected_r < -self.fee:
                signal = -1.0

            strat_r = signal * real_r - (abs(signal) * self.fee)
            strategy_returns.append(strat_r)

        actual_returns = np.array(actual_returns)
        pred_returns = np.array(pred_returns)
        strategy_returns = np.array(strategy_returns)

        # 5. Performance Metrics Calculation
        # Directional Hit Rate (%)
        correct_dir = np.sum(np.sign(pred_returns) == np.sign(actual_returns))
        directional_acc = (correct_dir / len(actual_returns)) * 100.0

        # Cumulative Return
        equity_curve = np.cumprod(1.0 + strategy_returns)
        cum_return_pct = float(equity_curve[-1] - 1.0) * 100.0

        buy_hold_equity = np.cumprod(1.0 + actual_returns)
        bh_return_pct = float(buy_hold_equity[-1] - 1.0) * 100.0

        # Maximum Drawdown (%)
        peaks = np.maximum.accumulate(equity_curve)
        drawdowns = (equity_curve - peaks) / peaks * 100.0
        max_drawdown = float(np.min(drawdowns))

        # Sharpe & Sortino Ratios (Annualized)
        mean_r = np.mean(strategy_returns)
        std_r = np.std(strategy_returns)
        downside_std = np.std(strategy_returns[strategy_returns < 0])

        ann_sharpe = float((mean_r / (std_r + 1e-8)) * np.sqrt(252))
        ann_sortino = float((mean_r / (downside_std + 1e-8)) * np.sqrt(252))
        ann_volatility = float(std_r * np.sqrt(252) * 100.0)

        # Win Rate & Profit Factor
        trades = strategy_returns[strategy_returns != 0]
        if len(trades) > 0:
            win_rate = (np.sum(trades > 0) / len(trades)) * 100.0
            gains = np.sum(trades[trades > 0])
            losses = np.abs(np.sum(trades[trades < 0]))
            profit_factor = float(gains / (losses + 1e-8))
        else:
            win_rate = 0.0
            profit_factor = 0.0

        return AssetBenchmarkResult(
            ticker=ticker,
            name=name,
            asset_class=asset_class,
            bars_count=N,
            directional_accuracy=directional_acc,
            annualized_sharpe=ann_sharpe,
            annualized_sortino=ann_sortino,
            max_drawdown_pct=max_drawdown,
            cumulative_return_pct=cum_return_pct,
            buy_hold_return_pct=bh_return_pct,
            win_rate_pct=win_rate,
            profit_factor=profit_factor,
            volatility_annualized=ann_volatility,
            invariant_validity_pct=invariant_pass_rate,
            equity_curve=equity_curve,
            drawdown_curve=drawdowns,
        )

    def evaluate_universe(
        self,
        universe_data: Dict[str, Dict[str, Any]],
        context_bars: int = 64,
        eval_horizon: int = 150,
        verbose: bool = True,
    ) -> List[AssetBenchmarkResult]:
        """Evaluates model across the entire fetched multi-asset universe."""
        results = []
        total = len(universe_data)
        if verbose:
            print(f"\nEvaluating Klint-32M across {total} assets...")

        for idx, (ticker, data) in enumerate(universe_data.items(), 1):
            res = self.evaluate_asset(data, context_bars=context_bars, eval_horizon=eval_horizon)
            if res is not None:
                results.append(res)

            if verbose and (idx % 25 == 0 or idx == total):
                print(f"  --> Benchmarked {idx}/{total} assets | Valid results: {len(results)}")

        return results
