"""Klint-32M Generative Inference Engine.

Generates structurally valid, invariant-preserving future market trajectories
and predictive directional signals conditioned on historical market context.

Usage Examples:
    # 1. Live market inference on Solana via yfinance:
    python inference.py --ticker SOL-USD --horizon 30 --save_plot forecast_sol.png

    # 2. Live market inference on Nvidia:
    python inference.py --ticker NVDA --horizon 50 --save_plot forecast_nvda.png

    # 3. Using local release bundle:
    python inference.py --checkpoint checkpoints/klint_32m_release.pt --ticker BTC-USD

    # 4. Using local checkpoint and tokenizer:
    python inference.py --checkpoint checkpoints/klint_32m_best.pt --tokenizer checkpoints/tokenizer_best.pt --data_path data/SOL.npy
"""

import os
import sys

# Ensure src is in python path
sys.path.insert(0, os.path.abspath("src"))
sys.path.insert(0, os.path.abspath("../src"))
if os.path.exists("/content/Klint-32M/src"):
    sys.path.insert(0, "/content/Klint-32M/src")

import argparse
from typing import Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from klint.data.validator import validate_ohlcv
from klint.data.factors import FactorDecomposer
from klint.tokenizer.factor_tokenizer import FactorTokenizer
from klint.tokenizer.geometric_decoder import GeometricDecoder
from klint.models.klint_32m import Klint32M, KlintConfig


def download_hf_checkpoint(repo_id: str, filename: str) -> str:
    """Downloads checkpoint from Hugging Face Hub if not present locally."""
    try:
        from huggingface_hub import hf_hub_download
        print(f"Downloading {filename} from Hugging Face ({repo_id})...")
        local_path = hf_hub_download(repo_id=repo_id, filename=filename)
        return local_path
    except ImportError:
        raise ImportError(
            "huggingface_hub is not installed. Please install it with: pip install huggingface_hub"
        )


def load_klint_bundle(
    checkpoint_path: Optional[str] = None,
    tokenizer_path: Optional[str] = None,
    hf_repo: str = "akhverm/Klint-32M",
    device: str = "cpu",
) -> Tuple[Klint32M, FactorTokenizer, GeometricDecoder]:
    """
    Loads Klint-32M model and Factor Tokenizer from a release bundle,
    individual checkpoints, or directly from Hugging Face.
    """
    # 1. Release bundle path check
    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        default_release = os.path.join("checkpoints", "klint_32m_release.pt")
        default_best = os.path.join("checkpoints", "klint_32m_best.pt")

        if os.path.exists(default_release):
            checkpoint_path = default_release
        elif os.path.exists(default_best):
            checkpoint_path = default_best
        else:
            print(f"No local checkpoints found. Fetching release bundle from Hugging Face ({hf_repo})...")
            checkpoint_path = download_hf_checkpoint(hf_repo, "klint_32m_release.pt")

    print(f"Loading weights from: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Detect bundle format vs individual checkpoint
    if "tokenizer_state_dict" in state:
        # Full Release Bundle
        print("  --> Detected complete Klint release bundle (Model + Tokenizer + Config).")
        cfg = state.get("config", KlintConfig())
        model = Klint32M(cfg).to(device)
        model.load_state_dict(state["model_state_dict"])

        tokenizer = FactorTokenizer().to(device)
        tokenizer.load_state_dict(state["tokenizer_state_dict"])
    else:
        # Individual Model Checkpoint
        cfg = state.get("config", KlintConfig())
        model = Klint32M(cfg).to(device)
        model_weights = state.get("model_state", state.get("model", state))
        model.load_state_dict(model_weights, strict=False)

        # Load Tokenizer
        if tokenizer_path is None or not os.path.exists(tokenizer_path):
            def_tok = os.path.join("checkpoints", "tokenizer_best.pt")
            if os.path.exists(def_tok):
                tokenizer_path = def_tok
            else:
                print(f"Fetching tokenizer codebooks from Hugging Face ({hf_repo})...")
                tokenizer_path = download_hf_checkpoint(hf_repo, "tokenizer_best.pt")

        print(f"Loading tokenizer codebooks from: {tokenizer_path}")
        tok_state = torch.load(tokenizer_path, map_location=device, weights_only=False)
        tokenizer = FactorTokenizer().to(device)
        tok_weights = tok_state.get("tokenizer_state", tok_state.get("model_state", tok_state))
        tokenizer.load_state_dict(tok_weights, strict=False)

    model.eval()
    tokenizer.eval()
    decoder = GeometricDecoder().to(device)

    print(f"Klint-32M Foundation Model ready on {device.upper()} ({model.count_parameters():,} parameters).")
    return model, tokenizer, decoder


def fetch_prompt_data(
    ticker: Optional[str] = None,
    data_path: Optional[str] = None,
    period: str = "3mo",
    interval: str = "1d",
    context_bars: int = 64,
) -> Tuple[np.ndarray, str, float]:
    """Fetches or loads historical market bars to condition the generation."""
    if ticker is not None:
        import yfinance as yf
        print(f"Fetching live market context for {ticker} (period={period}, interval={interval}) via yfinance...")
        df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=True)
        if df is None or len(df) < context_bars:
            raise ValueError(f"Insufficient bars returned for {ticker} (received {len(df) if df is not None else 0}, needed {context_bars}).")
        cols = ["Open", "High", "Low", "Close", "Volume"]
        df = df[cols].dropna()
        ohlcv = df.values.astype(np.float64)
        asset_name = ticker
    elif data_path is not None and os.path.exists(data_path):
        print(f"Loading context data from file: {data_path}...")
        if data_path.endswith(".npy"):
            raw = np.load(data_path, allow_pickle=True)
            if raw.ndim == 2 and raw.shape[1] >= 5:
                ohlcv = raw[-context_bars * 2:, 1:6].astype(np.float64) if raw.shape[1] >= 6 else raw[-context_bars * 2:, :5].astype(np.float64)
            else:
                raise ValueError("Unsupported shape in .npy file.")
        elif data_path.endswith(".csv"):
            df = pd.read_csv(data_path)
            cols = ["Open", "High", "Low", "Close", "Volume"]
            ohlcv = df[cols].dropna().values.astype(np.float64)
        asset_name = os.path.basename(data_path)
    else:
        # Fallback to SOL.npy or synthetic
        default_sol = os.path.join("data", "SOL.npy")
        if os.path.exists(default_sol):
            return fetch_prompt_data(ticker=None, data_path=default_sol, context_bars=context_bars)
        else:
            print("No ticker or data path specified. Generating sample market context...")
            N = context_bars + 20
            p = np.cumprod(1.0 + np.random.randn(N) * 0.015) * 150.0
            o = p
            c = p * (1.0 + np.random.randn(N) * 0.008)
            h = np.maximum(o, c) + np.random.exponential(0.5, size=N)
            l = np.minimum(o, c) - np.random.exponential(0.5, size=N)
            v = np.random.exponential(50000.0, size=N)
            ohlcv = np.stack([o, h, l, c, v], axis=1)
            asset_name = "Synthetic_Sample"

    # Take the most recent context_bars
    ohlcv = ohlcv[-context_bars:]
    is_valid, report = validate_ohlcv(ohlcv)
    if not is_valid:
        print(f"Warning: Input data had {report['total_violations']} invariant violations. Applying boundary fixes.")
        ohlcv[:, 1] = np.maximum(ohlcv[:, 1], np.maximum(ohlcv[:, 0], ohlcv[:, 3]))
        ohlcv[:, 2] = np.minimum(ohlcv[:, 2], np.minimum(ohlcv[:, 0], ohlcv[:, 3]))

    last_close = float(ohlcv[-1, 3])
    return ohlcv, asset_name, last_close


def generate_forecast(
    model: Klint32M,
    tokenizer: FactorTokenizer,
    decoder: GeometricDecoder,
    context_ohlcv: np.ndarray,
    horizon_bars: int = 30,
    temperature: float = 0.8,
    top_k: int = 40,
    device: str = "cpu",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Generates autoregressive future OHLCV trajectory conditioned on context_ohlcv.
    """
    decomposer = FactorDecomposer()
    factors = decomposer.decompose(context_ohlcv)

    # Encode context into discrete tokens
    p_tensor = torch.from_numpy(factors.price_path).unsqueeze(0).to(device)
    r_tensor = torch.from_numpy(factors.range_shape).unsqueeze(0).to(device)
    a_tensor = torch.from_numpy(factors.activity).unsqueeze(0).to(device)

    with torch.no_grad():
        p_tok, r_tok, a_tok, _ = tokenizer.encode(p_tensor, r_tensor, a_tensor)
        interleaved = tokenizer.interleave(p_tok, r_tok, a_tok)  # (1, 3 * context_bars)
        prompt_tokens_count = interleaved.size(1)

        # 1. Compute Next-Bar Directional Predictive Alpha
        out = model(interleaved)
        last_hidden = out["hidden"][:, -1, :]
        logits_p = model.head_price(last_hidden).squeeze(0)
        probs_p = F.softmax(logits_p, dim=-1)

        emb_p = tokenizer.rvq_price.layers[0].embedding
        dec_p_weights = tokenizer.dec_price(emb_p)
        code_body_returns = dec_p_weights[:, 1]

        expected_return = float(torch.sum(probs_p * code_body_returns).item())
        prob_up = float(torch.sum(probs_p[code_body_returns > 0]).item()) * 100.0
        prob_down = float(torch.sum(probs_p[code_body_returns < 0]).item()) * 100.0

        signal = "NEUTRAL"
        if expected_return > 0.0005 and prob_up > 51.0:
            signal = "BULLISH (LONG)"
        elif expected_return < -0.0005 and prob_down > 51.0:
            signal = "BEARISH (SHORT)"

        # 2. Autoregressive Trajectory Generation
        print(f"Sampling {horizon_bars} future bars (temperature={temperature}, top_k={top_k})...")
        gen_tokens = model.generate_tokens(
            interleaved,
            num_bars=horizon_bars,
            temperature=temperature,
            top_k=top_k,
        )

        # Extract newly sampled tokens (3 tokens per bar)
        new_tokens = gen_tokens[:, prompt_tokens_count:]

        # Deinterleave and decode back to continuous factors
        new_p_tok, new_r_tok, new_a_tok = tokenizer.deinterleave(new_tokens)
        rec_p, rec_r, rec_a = tokenizer.decode_tokens(new_p_tok, new_r_tok, new_a_tok)

        # Reconstruct into geometric OHLCV candles
        anchor_price = float(context_ohlcv[-1, 3])
        synth_bars_tensor = decoder(rec_p, rec_r, rec_a, anchor_price=anchor_price)
        forecast_ohlcv = synth_bars_tensor.squeeze(0).detach().cpu().numpy()

    # Invariant Verification
    is_valid, report = validate_ohlcv(forecast_ohlcv)
    if not is_valid:
        print(f"Warning: {report['total_violations']} candle violations detected. Applying structural enforcement.")
        forecast_ohlcv[:, 1] = np.maximum(forecast_ohlcv[:, 1], np.maximum(forecast_ohlcv[:, 0], forecast_ohlcv[:, 3]))
        forecast_ohlcv[:, 2] = np.minimum(forecast_ohlcv[:, 2], np.minimum(forecast_ohlcv[:, 0], forecast_ohlcv[:, 3]))

    stats = {
        "expected_next_return_pct": expected_return * 100.0,
        "prob_up_pct": prob_up,
        "prob_down_pct": prob_down,
        "signal": signal,
        "anchor_price": anchor_price,
        "terminal_forecast_price": float(forecast_ohlcv[-1, 3]),
        "forecast_change_pct": float((forecast_ohlcv[-1, 3] - anchor_price) / anchor_price * 100.0),
        "invariants_passed": is_valid,
    }

    return forecast_ohlcv, stats


def plot_forecast(
    context_ohlcv: np.ndarray,
    forecast_ohlcv: np.ndarray,
    asset_name: str,
    stats: Dict[str, Any],
    save_path: str = "forecast_trajectory.png",
):
    """Renders high-resolution candlestick/trajectory chart with uncertainty demarcation."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]}, sharex=True)

    C = len(context_ohlcv)
    H = len(forecast_ohlcv)
    total_len = C + H

    time_hist = np.arange(C)
    time_pred = np.arange(C, total_len)

    # 1. Price Panel
    # Historical Close
    ax1.plot(time_hist, context_ohlcv[:, 3], color="#7f8c8d", linewidth=2.0, label=f"Historical Context ({C} bars)")
    # Forecast Close
    # Prepend last historical close for seamless continuity
    traj_time = np.concatenate([[C - 1], time_pred])
    traj_prices = np.concatenate([[context_ohlcv[-1, 3]], forecast_ohlcv[:, 3]])

    color_sig = "#2ecc71" if "BULLISH" in stats["signal"] else ("#e74c3c" if "BEARISH" in stats["signal"] else "#f1c40f")
    ax1.plot(traj_time, traj_prices, color=color_sig, linewidth=2.5, linestyle="-",
             label=f"Klint-32M Generative Forecast ({H} bars: {stats['forecast_change_pct']:+.2f}%)")

    # High / Low Envelopes for forecast
    ax1.fill_between(time_pred, forecast_ohlcv[:, 2], forecast_ohlcv[:, 1], color=color_sig, alpha=0.18, label="Predicted Intrabar Range (High-Low)")

    # Vertical Forecast Demarcation Line
    ax1.axvline(C - 1, color="#ffffff", linestyle="--", linewidth=1.5, alpha=0.8, label="Forecast Horizon Boundary")

    ax1.set_title(
        f"Klint-32M: Generative Market Forecasting - {asset_name}\n"
        f"Signal: {stats['signal']} | Next Bar Return: {stats['expected_next_return_pct']:+.2f}% (P(Up)={stats['prob_up_pct']:.1f}%) | Horizon: {H} bars",
        fontsize=13, fontweight="bold", pad=12
    )
    ax1.set_ylabel("Price ($)", fontsize=11)
    ax1.legend(loc="upper left", framealpha=0.9)

    # 2. Volume Panel
    hist_vol = context_ohlcv[:, 4]
    pred_vol = forecast_ohlcv[:, 4]
    ax2.bar(time_hist, hist_vol, color="#95a5a6", alpha=0.6, label="Historical Volume")
    ax2.bar(time_pred, pred_vol, color=color_sig, alpha=0.7, label="Forecast Activity Stream")
    ax2.axvline(C - 1, color="#ffffff", linestyle="--", linewidth=1.5, alpha=0.8)
    ax2.set_xlabel("Chronological Bars", fontsize=11)
    ax2.set_ylabel("Volume", fontsize=11)
    ax2.legend(loc="upper left", framealpha=0.9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"Publication-grade forecast plot saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Klint-32M Generative Market Inference & Forecasting Engine")
    parser.add_argument("--ticker", type=str, default=None, help="Live asset ticker from yfinance (e.g. SOL-USD, BTC-USD, NVDA, SPY)")
    parser.add_argument("--data_path", type=str, default=None, help="Local path to .npy or .csv OHLCV data")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/klint_32m_release.pt", help="Path to model weights or release bundle")
    parser.add_argument("--tokenizer", type=str, default="checkpoints/tokenizer_best.pt", help="Path to factor tokenizer codebooks")
    parser.add_argument("--hf_repo", type=str, default="akhverm/Klint-32M", help="Hugging Face repo id to fetch weights if missing")
    parser.add_argument("--horizon", type=int, default=30, help="Number of future market bars to forecast")
    parser.add_argument("--context_bars", type=int, default=64, help="Historical context window bars (default: 64)")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature (0.1 = conservative, 1.2 = exploratory)")
    parser.add_argument("--top_k", type=int, default=40, help="Top-K token truncation")
    parser.add_argument("--save_plot", type=str, default="forecast_trajectory.png", help="Path to save output visual plot")
    parser.add_argument("--save_csv", type=str, default=None, help="Optional path to save forecasted OHLCV candles to CSV")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")

    args = parser.parse_args()

    print("=" * 70)
    print(" KLINT-32M: GENERATIVE FINANCIAL FORECASTING INFERENCE")
    print("=" * 70)

    # 1. Load Model Architecture
    model, tokenizer, decoder = load_klint_bundle(
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
        hf_repo=args.hf_repo,
        device=args.device,
    )

    # 2. Ingest Historical Context
    context_ohlcv, asset_name, last_close = fetch_prompt_data(
        ticker=args.ticker,
        data_path=args.data_path,
        context_bars=args.context_bars,
    )
    print(f"Conditioning on {len(context_ohlcv)} bars of {asset_name} (Latest Close: ${last_close:,.2f})")

    # 3. Generate Forecast Trajectory
    forecast_ohlcv, stats = generate_forecast(
        model=model,
        tokenizer=tokenizer,
        decoder=decoder,
        context_ohlcv=context_ohlcv,
        horizon_bars=args.horizon,
        temperature=args.temperature,
        top_k=args.top_k,
        device=args.device,
    )

    # 4. Display Results
    print("\n" + "=" * 70)
    print(" QUANTITATIVE PREDICTION & DIRECTIONAL ALPHA SCORECARD")
    print("=" * 70)
    print(f"  Asset:                   {asset_name}")
    print(f"  Current Price:           ${stats['anchor_price']:,.2f}")
    print(f"  Terminal Forecast:       ${stats['terminal_forecast_price']:,.2f} ({stats['forecast_change_pct']:+.2f}%)")
    print(f"  Market Signal:           {stats['signal']}")
    print(f"  Expected Next-Bar Alpha: {stats['expected_next_return_pct']:+.3f}%")
    print(f"  Directional Probability: P(Up) = {stats['prob_up_pct']:.1f}% | P(Down) = {stats['prob_down_pct']:.1f}%")
    print(f"  Candle Invariants:       {'100.00% PASSED (Guaranteed Geometry)' if stats['invariants_passed'] else 'Violations Repaired'}")
    print("=" * 70)

    # Display preview table
    preview_df = pd.DataFrame(forecast_ohlcv, columns=["Open", "High", "Low", "Close", "Volume"])
    preview_df.index = [f"Bar +{i+1}" for i in range(len(forecast_ohlcv))]
    print("\nForecasted Trajectory (First 10 Bars):")
    print(preview_df.head(10).to_string())

    # Optional CSV Export
    if args.save_csv:
        preview_df.to_csv(args.save_csv)
        print(f"\nSaved full forecasted trajectory to: {args.save_csv}")

    # Visual Plotting
    if args.save_plot:
        plot_forecast(context_ohlcv, forecast_ohlcv, asset_name, stats, save_path=args.save_plot)

    print("\nInference pipeline executed successfully.")


if __name__ == "__main__":
    main()
