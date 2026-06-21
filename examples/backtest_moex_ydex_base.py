import argparse
import datetime as dt
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("NO_PROXY", "*")
os.environ.setdefault("no_proxy", "*")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import Kronos, KronosPredictor, KronosTokenizer
from predict_moex_ydex_base import fetch_moex_candles


FEATURE_COLS = ["open", "high", "low", "close", "volume", "amount"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_windows(df: pd.DataFrame, lookback: int, eval_count: int):
    if len(df) < lookback + eval_count:
        raise RuntimeError(
            f"Need at least lookback + eval_count candles ({lookback + eval_count}), got {len(df)}."
        )

    first_target_idx = len(df) - eval_count
    df_list = []
    x_ts_list = []
    y_ts_list = []
    actual_rows = []
    prev_closes = []

    for target_idx in range(first_target_idx, len(df)):
        context = df.iloc[target_idx - lookback : target_idx].copy()
        target = df.iloc[target_idx].copy()

        df_list.append(context[FEATURE_COLS].reset_index(drop=True))
        x_ts_list.append(context["timestamps"].reset_index(drop=True))
        y_ts_list.append(pd.Series([target["timestamps"]], name="timestamps"))
        actual_rows.append(target)
        prev_closes.append(float(context["close"].iloc[-1]))

    actual_df = pd.DataFrame(actual_rows).reset_index(drop=True)
    actual_df["prev_close"] = prev_closes
    return df_list, x_ts_list, y_ts_list, actual_df


def compute_metrics(result: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    for col in FEATURE_COLS:
        result[f"err_{col}"] = result[f"pred_{col}"] - result[f"actual_{col}"]
        result[f"abs_err_{col}"] = result[f"err_{col}"].abs()
        result[f"ape_{col}"] = result[f"abs_err_{col}"] / result[f"actual_{col}"].abs().replace(0, np.nan)

    result["actual_return"] = result["actual_close"] / result["prev_close"] - 1.0
    result["pred_return"] = result["pred_close"] / result["prev_close"] - 1.0
    result["return_error"] = result["pred_return"] - result["actual_return"]
    result["direction_hit"] = np.sign(result["pred_return"]) == np.sign(result["actual_return"])

    pred_up = result["pred_return"] > 0
    actual_up = result["actual_return"] > 0
    result["signal_long_only_return"] = np.where(pred_up, result["actual_return"], 0.0)
    result["signal_long_short_return"] = np.sign(result["pred_return"]) * result["actual_return"]

    metric_rows = []
    for col in FEATURE_COLS:
        err = result[f"err_{col}"].to_numpy(dtype=float)
        abs_err = np.abs(err)
        actual = result[f"actual_{col}"].to_numpy(dtype=float)
        pred = result[f"pred_{col}"].to_numpy(dtype=float)
        sse = float(np.sum((pred - actual) ** 2))
        sst = float(np.sum((actual - np.mean(actual)) ** 2))
        metric_rows.append(
            {
                "series": col,
                "mae": float(np.mean(abs_err)),
                "rmse": float(np.sqrt(np.mean(err**2))),
                "mape_pct": float(np.nanmean(abs_err / np.where(actual == 0, np.nan, np.abs(actual))) * 100),
                "median_abs_error": float(np.median(abs_err)),
                "p90_abs_error": float(np.quantile(abs_err, 0.90)),
                "bias": float(np.mean(err)),
                "mean_actual": float(np.mean(actual)),
                "mean_pred": float(np.mean(pred)),
                "r2": float(1 - sse / sst) if sst > 0 else np.nan,
                "pearson_level": float(np.corrcoef(actual, pred)[0, 1]) if len(result) > 1 else np.nan,
            }
        )

    metrics = pd.DataFrame(metric_rows)

    actual_return = result["actual_return"].to_numpy(dtype=float)
    pred_return = result["pred_return"].to_numpy(dtype=float)
    return_corr = float(np.corrcoef(actual_return, pred_return)[0, 1]) if len(result) > 1 else np.nan
    return_mae = float(np.mean(np.abs(pred_return - actual_return)))
    return_rmse = float(np.sqrt(np.mean((pred_return - actual_return) ** 2)))
    direction_acc = float(result["direction_hit"].mean())

    signal_summary = pd.DataFrame(
        [
            {"metric": "eval_count", "value": float(len(result))},
            {"metric": "close_return_mae_pct", "value": return_mae * 100},
            {"metric": "close_return_rmse_pct", "value": return_rmse * 100},
            {"metric": "close_return_bias_pct", "value": float(np.mean(pred_return - actual_return)) * 100},
            {"metric": "close_return_pearson", "value": return_corr},
            {"metric": "direction_accuracy", "value": direction_acc},
            {"metric": "actual_up_days", "value": float(actual_up.sum())},
            {"metric": "predicted_up_days", "value": float(pred_up.sum())},
            {"metric": "true_positive_up", "value": float((pred_up & actual_up).sum())},
            {"metric": "true_negative_down", "value": float((~pred_up & ~actual_up).sum())},
            {"metric": "false_positive_up", "value": float((pred_up & ~actual_up).sum())},
            {"metric": "false_negative_down", "value": float((~pred_up & actual_up).sum())},
            {"metric": "buy_hold_return_pct", "value": float(np.prod(1 + actual_return) - 1) * 100},
            {
                "metric": "long_only_signal_return_pct",
                "value": float(np.prod(1 + result["signal_long_only_return"].to_numpy(dtype=float)) - 1) * 100,
            },
            {
                "metric": "long_short_signal_return_pct",
                "value": float(np.prod(1 + result["signal_long_short_return"].to_numpy(dtype=float)) - 1) * 100,
            },
            {
                "metric": "pred_high_violations",
                "value": float((result["pred_high"] < result[["pred_open", "pred_close"]].max(axis=1)).sum()),
            },
            {
                "metric": "pred_low_violations",
                "value": float((result["pred_low"] > result[["pred_open", "pred_close"]].min(axis=1)).sum()),
            },
            {"metric": "pred_low_gt_high_violations", "value": float((result["pred_low"] > result["pred_high"]).sum())},
        ]
    )

    return metrics, signal_summary


def save_plots(result: pd.DataFrame, output_dir: Path, secid: str) -> None:
    x = pd.to_datetime(result["timestamps"])

    fig, (ax_price, ax_err) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax_price.plot(x, result["actual_close"], label="Actual close", linewidth=1.8)
    ax_price.plot(x, result["pred_close"], label="Predicted close", linewidth=1.5)
    ax_price.set_title(f"{secid}: 50 one-day walk-forward forecasts, Kronos-base")
    ax_price.set_ylabel("RUB")
    ax_price.grid(True, alpha=0.25)
    ax_price.legend(loc="best")

    ax_err.bar(x, result["err_close"], label="Predicted - actual close")
    ax_err.axhline(0, color="black", linewidth=0.8)
    ax_err.set_ylabel("RUB error")
    ax_err.grid(True, alpha=0.25)
    ax_err.legend(loc="best")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_dir / f"{secid.lower()}_walkforward_close.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(result["actual_return"] * 100, result["pred_return"] * 100, alpha=0.8)
    lim = max(
        abs(float(result["actual_return"].min() * 100)),
        abs(float(result["actual_return"].max() * 100)),
        abs(float(result["pred_return"].min() * 100)),
        abs(float(result["pred_return"].max() * 100)),
    )
    lim = max(lim, 0.5)
    ax.plot([-lim, lim], [-lim, lim], color="gray", linestyle="--", linewidth=1)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("Actual next-day return, %")
    ax.set_ylabel("Predicted next-day return, %")
    ax.set_title(f"{secid}: predicted vs actual returns")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / f"{secid.lower()}_walkforward_returns_scatter.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward evaluation for Kronos-base on MOEX YDEX.")
    parser.add_argument("--secid", default="YDEX")
    parser.add_argument("--board", default="TQBR")
    parser.add_argument("--days", type=int, default=1200)
    parser.add_argument("--lookback", type=int, default=256)
    parser.add_argument("--eval-count", type=int, default=50)
    parser.add_argument("--sample-count", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--max-context", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--model", default="NeoQuasar/Kronos-base")
    parser.add_argument("--tokenizer", default="NeoQuasar/Kronos-Tokenizer-base")
    args = parser.parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)
    set_seed(args.seed)

    output_dir = ROOT / "outputs" / f"{args.secid.lower()}_kronos_base_walkforward_{dt.datetime.now():%Y%m%d_%H%M%S}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching MOEX daily candles for {args.secid} ({args.board})...")
    df = fetch_moex_candles(args.secid, args.board, args.days)
    raw_csv = output_dir / f"{args.secid.lower()}_moex_daily.csv"
    df.to_csv(raw_csv, index=False)

    df_list, x_ts_list, y_ts_list, actual_df = build_windows(df, args.lookback, args.eval_count)

    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = KronosTokenizer.from_pretrained(args.tokenizer)
    print(f"Loading model: {args.model}")
    model = Kronos.from_pretrained(args.model)

    device = None if args.device == "auto" else args.device
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=args.max_context)

    print(
        f"Running {args.eval_count} hidden-target one-day forecasts "
        f"(lookback={args.lookback}, sample_count={args.sample_count}, device={predictor.device})..."
    )
    pred_dfs = predictor.predict_batch(
        df_list=df_list,
        x_timestamp_list=x_ts_list,
        y_timestamp_list=y_ts_list,
        pred_len=1,
        T=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        sample_count=args.sample_count,
        verbose=True,
    )

    pred_rows = [pred.iloc[0] for pred in pred_dfs]
    pred_df = pd.DataFrame(pred_rows).reset_index(drop=True)

    result = pd.DataFrame({"timestamps": actual_df["timestamps"], "prev_close": actual_df["prev_close"]})
    for col in FEATURE_COLS:
        result[f"actual_{col}"] = actual_df[col].astype(float)
        result[f"pred_{col}"] = pred_df[col].astype(float)

    metrics, signal_summary = compute_metrics(result)

    result_csv = output_dir / f"{args.secid.lower()}_walkforward_predictions.csv"
    metrics_csv = output_dir / f"{args.secid.lower()}_walkforward_metrics.csv"
    signal_csv = output_dir / f"{args.secid.lower()}_walkforward_signal_metrics.csv"
    result.to_csv(result_csv, index=False)
    metrics.to_csv(metrics_csv, index=False)
    signal_summary.to_csv(signal_csv, index=False)
    save_plots(result, output_dir, args.secid)

    summary = {
        "secid": args.secid,
        "board": args.board,
        "model": args.model,
        "tokenizer": args.tokenizer,
        "device": predictor.device,
        "seed": args.seed,
        "candles": len(df),
        "lookback": args.lookback,
        "eval_count": args.eval_count,
        "sample_count": args.sample_count,
        "first_target_timestamp": str(result["timestamps"].iloc[0]),
        "last_target_timestamp": str(result["timestamps"].iloc[-1]),
        "raw_csv": str(raw_csv),
        "predictions_csv": str(result_csv),
        "metrics_csv": str(metrics_csv),
        "signal_metrics_csv": str(signal_csv),
        "close_plot_png": str(output_dir / f"{args.secid.lower()}_walkforward_close.png"),
        "returns_scatter_png": str(output_dir / f"{args.secid.lower()}_walkforward_returns_scatter.png"),
    }
    (output_dir / "summary.txt").write_text(
        "\n".join(f"{key}: {value}" for key, value in summary.items()),
        encoding="utf-8",
    )

    print("\nClose metrics:")
    print(metrics.loc[metrics["series"] == "close"].to_string(index=False))
    print("\nReturn/signal metrics:")
    print(signal_summary.to_string(index=False))
    print("\nArtifacts:")
    print(result_csv)
    print(metrics_csv)
    print(signal_csv)


if __name__ == "__main__":
    main()
