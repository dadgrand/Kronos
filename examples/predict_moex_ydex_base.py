import argparse
import datetime as dt
import os
import sys
from pathlib import Path

os.environ.setdefault("NO_PROXY", "*")
os.environ.setdefault("no_proxy", "*")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import requests
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model import Kronos, KronosPredictor, KronosTokenizer


def fetch_moex_candles(secid: str, board: str, days: int) -> pd.DataFrame:
    end_date = dt.date.today()
    start_date = end_date - dt.timedelta(days=days)
    url = (
        "https://iss.moex.com/iss/engines/stock/markets/shares/"
        f"boards/{board}/securities/{secid}/candles.json"
    )

    session = requests.Session()
    session.trust_env = False

    rows = []
    columns = None
    start = 0
    while True:
        params = {
            "from": start_date.isoformat(),
            "till": end_date.isoformat(),
            "interval": 24,
            "start": start,
            "iss.meta": "off",
        }
        response = session.get(url, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()["candles"]
        columns = payload["columns"]
        batch = payload["data"]
        if not batch:
            break
        rows.extend(batch)
        start += len(batch)
        if len(batch) < 500:
            break

    if not rows:
        raise RuntimeError(f"No candles returned by MOEX for {secid} on {board}.")

    df = pd.DataFrame(rows, columns=columns)
    df = df.rename(columns={"begin": "timestamps", "value": "amount"})
    df["timestamps"] = pd.to_datetime(df["timestamps"])
    df = df[["timestamps", "open", "high", "low", "close", "volume", "amount"]]
    df = df.drop_duplicates("timestamps").sort_values("timestamps").reset_index(drop=True)
    return df


def make_future_daily_timestamps(last_ts: pd.Timestamp, pred_len: int) -> pd.Series:
    first_future_day = last_ts.normalize() + pd.offsets.BDay(1)
    return pd.Series(pd.bdate_range(first_future_day, periods=pred_len), name="timestamps")


def save_plot(history: pd.DataFrame, pred: pd.DataFrame, path: Path, title: str) -> None:
    fig, (ax_price, ax_volume) = plt.subplots(2, 1, figsize=(11, 7), sharex=False)

    ax_price.plot(history["timestamps"], history["close"], label="History close", linewidth=1.6)
    ax_price.plot(pred.index, pred["close"], label="Kronos forecast close", linewidth=1.8)
    ax_price.set_title(title)
    ax_price.set_ylabel("RUB")
    ax_price.grid(True, alpha=0.25)
    ax_price.legend(loc="best")

    ax_volume.plot(history["timestamps"], history["volume"], label="History volume", linewidth=1.2)
    ax_volume.plot(pred.index, pred["volume"], label="Forecast volume", linewidth=1.4)
    ax_volume.set_ylabel("Shares")
    ax_volume.grid(True, alpha=0.25)
    ax_volume.legend(loc="best")

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Kronos-base on live MOEX YDEX daily candles.")
    parser.add_argument("--secid", default="YDEX")
    parser.add_argument("--board", default="TQBR")
    parser.add_argument("--days", type=int, default=900)
    parser.add_argument("--lookback", type=int, default=256)
    parser.add_argument("--pred-len", type=int, default=10)
    parser.add_argument("--sample-count", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--max-context", type=int, default=512)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda:0, mps, etc.")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--model", default="NeoQuasar/Kronos-base")
    parser.add_argument("--tokenizer", default="NeoQuasar/Kronos-Tokenizer-base")
    args = parser.parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)

    output_dir = ROOT / "outputs" / f"{args.secid.lower()}_kronos_base_{dt.datetime.now():%Y%m%d_%H%M%S}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching MOEX daily candles for {args.secid} ({args.board})...")
    df = fetch_moex_candles(args.secid, args.board, args.days)
    if len(df) < args.lookback:
        raise RuntimeError(f"Need at least {args.lookback} candles, got {len(df)}.")

    full_csv = output_dir / f"{args.secid.lower()}_moex_daily.csv"
    df.to_csv(full_csv, index=False)

    x_df = df.tail(args.lookback).reset_index(drop=True)
    x_timestamp = x_df["timestamps"]
    y_timestamp = make_future_daily_timestamps(x_timestamp.iloc[-1], args.pred_len)

    device = None if args.device == "auto" else args.device
    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = KronosTokenizer.from_pretrained(args.tokenizer)
    print(f"Loading model: {args.model}")
    model = Kronos.from_pretrained(args.model)

    print("Running forecast...")
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=args.max_context)
    pred_df = predictor.predict(
        df=x_df[["open", "high", "low", "close", "volume", "amount"]],
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=args.pred_len,
        T=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        sample_count=args.sample_count,
        verbose=True,
    )

    pred_csv = output_dir / f"{args.secid.lower()}_kronos_base_forecast.csv"
    pred_df.to_csv(pred_csv, index_label="timestamps")

    plot_path = output_dir / f"{args.secid.lower()}_kronos_base_forecast.png"
    save_plot(
        history=x_df.tail(min(args.lookback, 120)),
        pred=pred_df,
        path=plot_path,
        title=f"{args.secid} MOEX daily forecast with Kronos-base",
    )

    summary = {
        "secid": args.secid,
        "board": args.board,
        "model": args.model,
        "tokenizer": args.tokenizer,
        "device": predictor.device,
        "candles": len(df),
        "lookback": args.lookback,
        "pred_len": args.pred_len,
        "last_history_timestamp": str(x_timestamp.iloc[-1]),
        "first_forecast_timestamp": str(pred_df.index[0]),
        "last_forecast_timestamp": str(pred_df.index[-1]),
        "history_csv": str(full_csv),
        "forecast_csv": str(pred_csv),
        "plot_png": str(plot_path),
    }
    summary_path = output_dir / "summary.txt"
    summary_path.write_text("\n".join(f"{k}: {v}" for k, v in summary.items()), encoding="utf-8")

    print("\nForecast head:")
    print(pred_df.head())
    print("\nArtifacts:")
    for key in ["history_csv", "forecast_csv", "plot_png"]:
        print(f"{key}: {summary[key]}")


if __name__ == "__main__":
    main()
