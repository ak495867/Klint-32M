"""Zero-Shot & One-Shot Out-Of-Distribution (OOD) generalization evaluation suite."""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import wasserstein_distance

from klint.data.factors import FactorDecomposer
from klint.data.validator import validate_ohlcv
from klint.tokenizer.factor_tokenizer import FactorTokenizer
from klint.tokenizer.geometric_decoder import GeometricDecoder
from klint.models.klint_32m import Klint32M
from klint.benchmark.data_fetcher import DataFetcher


@dataclass
class OODAssetEvaluation:
    """Zero-shot/One-shot generalization result for a single out-of-distribution asset."""
    ticker: str
    name: str
    asset_class: str
    bars_count: int
    ce_loss: float                    # Cross-entropy loss on novel token distribution
    perplexity: float                 # exp(ce_loss) - uncertainty metric
    directional_accuracy: float       # Next-bar directional hit rate (%)
    wasserstein_dist: float           # 2-Wasserstein distance between predicted and actual returns
    annualized_sharpe: float          # Out-of-sample Sharpe ratio
    max_drawdown_pct: float           # Out-of-sample Maximum Drawdown (%)
    win_rate_pct: float
    invariant_validity_pct: float     # Guaranteed geometric validity (%)
    pred_returns: np.ndarray
    actual_returns: np.ndarray


@dataclass
class OneShotGeneralizationResult:
    """Aggregated cross-asset class generalization metrics."""
    total_assets: int
    asset_evaluations: List[OODAssetEvaluation]
    mean_perplexity: float
    mean_directional_accuracy: float
    mean_sharpe: float
    mean_wasserstein_dist: float
    asset_class_breakdown: Dict[str, Dict[str, float]]


class OneShotGeneralizationEngine:
    """
    Evaluates zero-shot transfer of Klint-32M to diverse global asset classes without fine-tuning.
    """

    OOD_DEFAULT_UNIVERSE = [
        {"ticker": "SPY", "name": "SPDR S&P 500 ETF", "asset_class": "Equities"},
        {"ticker": "NVDA", "name": "Nvidia Corp", "asset_class": "Equities"},
        {"ticker": "AAPL", "name": "Apple Inc", "asset_class": "Equities"},
        {"ticker": "GLD", "name": "SPDR Gold Shares", "asset_class": "Commodities"},
        {"ticker": "USO", "name": "United States Oil Fund", "asset_class": "Commodities"},
        {"ticker": "EURUSD=X", "name": "Euro / US Dollar", "asset_class": "Forex"},
        {"ticker": "USDJPY=X", "name": "US Dollar / Japanese Yen", "asset_class": "Forex"},
        {"ticker": "TLT", "name": "iShares 20+ Year Treasury Bond", "asset_class": "Fixed Income"},
        {"ticker": "BTC-USD", "name": "Bitcoin USD", "asset_class": "Crypto"},
        {"ticker": "ETH-USD", "name": "Ethereum USD", "asset_class": "Crypto"},
    ]

    def __init__(
        self,
        model: Klint32M,
        tokenizer: FactorTokenizer,
        decomposer: Optional[FactorDecomposer] = None,
        decoder: Optional[GeometricDecoder] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        fee_bps: float = 5.0,
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
        asset_info: Dict[str, str],
        ohlcv: np.ndarray,
        context_bars: int = 64,
        eval_bars: int = 150,
    ) -> Optional[OODAssetEvaluation]:
        """Evaluates zero-shot prediction metrics on unseen asset time-series."""
        N = len(ohlcv)
        if N < context_bars + 20:
            return None

        # 1. Structural check
        is_valid, _ = validate_ohlcv(ohlcv)
        inv_pct = 100.0 if is_valid else 0.0

        # 2. Factor decomposition
        factors = self.decomposer.decompose(ohlcv)

        # 3. Tokenize
        p_tensor = torch.from_numpy(factors.price_path).unsqueeze(0).to(self.device)
        r_tensor = torch.from_numpy(factors.range_shape).unsqueeze(0).to(self.device)
        a_tensor = torch.from_numpy(factors.activity).unsqueeze(0).to(self.device)

        p_tok, r_tok, a_tok, _ = self.tokenizer.encode(p_tensor, r_tensor, a_tensor)
        interleaved = self.tokenizer.interleave(p_tok, r_tok, a_tok).squeeze(0)  # (3N,)

        # Codebook returns
        emb_p = self.tokenizer.rvq_price.layers[0].embedding
        dec_p_weights = self.tokenizer.dec_price(emb_p)
        code_body_returns = dec_p_weights[:, 1]  # (512,)

        # Rolling inference over evaluation window
        horizon = min(eval_bars, N - context_bars - 1)
        start_bar = N - horizon

        actual_returns = []
        pred_returns = []
        strategy_returns = []
        ce_losses = []

        for b in range(start_bar, N):
            h_start = (b - context_bars) * 3
            h_end = b * 3
            prompt = interleaved[h_start:h_end].unsqueeze(0)  # (1, 3 * context_bars)
            target_token = interleaved[h_end].unsqueeze(0)    # (1,) true next price token

            out = self.model(prompt)
            last_hidden = out["hidden"][:, -1, :]
            logits_p = self.model.head_price(last_hidden)  # (1, 512)

            # Cross-entropy loss on novel token
            loss = F.cross_entropy(logits_p, target_token)
            ce_losses.append(loss.item())

            # Expected return prediction
            probs_p = F.softmax(logits_p.squeeze(0), dim=-1)
            exp_r = torch.sum(probs_p * code_body_returns).item()
            real_r = float(factors.price_path[b, 1])

            actual_returns.append(real_r)
            pred_returns.append(exp_r)

            # Strategy signal
            sig = 0.0
            if exp_r > self.fee:
                sig = 1.0
            elif exp_r < -self.fee:
                sig = -1.0
            strategy_returns.append(sig * real_r - (abs(sig) * self.fee))

        actual_returns = np.array(actual_returns)
        pred_returns = np.array(pred_returns)
        strategy_returns = np.array(strategy_returns)

        # Accuracy & Perplexity
        valid = (pred_returns != 0) & (actual_returns != 0)
        hit_rate = float(np.mean(np.sign(pred_returns[valid]) == np.sign(actual_returns[valid])) * 100.0) if np.any(valid) else 50.0

        mean_ce = float(np.mean(ce_losses))
        perplexity = float(np.exp(np.clip(mean_ce, 0.0, 20.0)))

        # 2-Wasserstein distance between predicted and actual distribution
        w_dist = float(wasserstein_distance(pred_returns, actual_returns))

        # Risk/Return
        equity = np.cumprod(1.0 + strategy_returns)
        peaks = np.maximum.accumulate(equity)
        drawdowns = (equity - peaks) / peaks * 100.0
        mdd = float(np.min(drawdowns))

        mean_r = np.mean(strategy_returns)
        std_r = np.std(strategy_returns) + 1e-8
        sharpe = float((mean_r / std_r) * np.sqrt(252))

        trades = strategy_returns[strategy_returns != 0]
        win_rate = float(np.sum(trades > 0) / len(trades) * 100.0) if len(trades) > 0 else 0.0

        return OODAssetEvaluation(
            ticker=asset_info["ticker"],
            name=asset_info.get("name", asset_info["ticker"]),
            asset_class=asset_info.get("asset_class", "Equities"),
            bars_count=N,
            ce_loss=mean_ce,
            perplexity=perplexity,
            directional_accuracy=hit_rate,
            wasserstein_dist=w_dist,
            annualized_sharpe=sharpe,
            max_drawdown_pct=mdd,
            win_rate_pct=win_rate,
            invariant_validity_pct=inv_pct,
            pred_returns=pred_returns,
            actual_returns=actual_returns,
        )

    def run_generalization_suite(
        self,
        data_fetcher: DataFetcher,
        universe: Optional[List[Dict[str, str]]] = None,
        context_bars: int = 64,
        eval_bars: int = 150,
        period: str = "1y",
        interval: str = "1d",
    ) -> OneShotGeneralizationResult:
        """Runs one-shot generalization across predefined or custom OOD asset universe."""
        target_universe = universe or self.OOD_DEFAULT_UNIVERSE
        evaluations: List[OODAssetEvaluation] = []

        print(f"\nRunning Zero-Shot / One-Shot Generalization across {len(target_universe)} OOD assets...")
        for item in target_universe:
            ticker = item["ticker"]
            asset_data = data_fetcher.fetch_asset_data(ticker, name=item.get("name", ticker), asset_class=item.get("asset_class", "Equities"), period=period, interval=interval)
            if asset_data is None:
                continue

            res = self.evaluate_asset(item, asset_data["ohlcv"], context_bars=context_bars, eval_bars=eval_bars)
            if res is not None:
                evaluations.append(res)
                print(f"  --> {ticker:10s} ({res.asset_class:12s}) | Hit Rate: {res.directional_accuracy:5.1f}% | PPL: {res.perplexity:6.2f} | Sharpe: {res.annualized_sharpe:6.2f}")

        if not evaluations:
            raise RuntimeError("No OOD assets could be successfully evaluated.")

        mean_ppl = float(np.mean([e.perplexity for e in evaluations]))
        mean_acc = float(np.mean([e.directional_accuracy for e in evaluations]))
        mean_sh = float(np.mean([e.annualized_sharpe for e in evaluations]))
        mean_w = float(np.mean([e.wasserstein_dist for e in evaluations]))

        # Group by asset class
        class_breakdown: Dict[str, Dict[str, float]] = {}
        classes = sorted(list(set(e.asset_class for e in evaluations)))
        for ac in classes:
            group = [e for e in evaluations if e.asset_class == ac]
            class_breakdown[ac] = {
                "count": len(group),
                "mean_accuracy": float(np.mean([e.directional_accuracy for e in group])),
                "mean_perplexity": float(np.mean([e.perplexity for e in group])),
                "mean_sharpe": float(np.mean([e.annualized_sharpe for e in group])),
                "mean_wasserstein": float(np.mean([e.wasserstein_dist for e in group])),
            }

        return OneShotGeneralizationResult(
            total_assets=len(evaluations),
            asset_evaluations=evaluations,
            mean_perplexity=mean_ppl,
            mean_directional_accuracy=mean_acc,
            mean_sharpe=mean_sh,
            mean_wasserstein_dist=mean_w,
            asset_class_breakdown=class_breakdown,
        )
