"""Run a pinned Kronos hourly OOS replay on cached Binance 1m data.

This is a bounded model-signal smoke run, not a substitute for the live
forward-only 30-day shadow gate. It uses the strict prediction contract and
persists predictions incrementally so interrupted runs can resume.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prod.config import DEFAULT_SYMBOLS
from prod.inference import MODEL_IDS
from prod.ledger import file_checksum, stable_hash
from prod.runtime import current_code_version
from trading.runner import PredictionEvent
from trading.validation import AlphaValidationReport


INITIAL_CAPITAL = 10_000.0
TX_COST_BPS = 10.0
COST_RATE = TX_COST_BPS / 10_000.0
MAX_GROSS = 0.50
MAX_SYMBOL = 0.125
MAX_DAILY_LOSS = 0.02
MAX_DRAWDOWN_HALT = 0.05


def log(message: str) -> None:
    print(message, flush=True)


def load_hourly(path: Path, symbols: tuple[str, ...]) -> pd.DataFrame:
    log(f"loading_1m_csv {path}")
    raw = pd.read_csv(path)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)
    frames = []
    for symbol in symbols:
        group = raw[raw["symbol"] == symbol].set_index("timestamp").sort_index()
        if group.empty:
            raise ValueError(f"No rows for {symbol}.")
        hourly = pd.DataFrame(
            {
                "open": group["open"].resample("1h", label="right", closed="right").first(),
                "high": group["high"].resample("1h", label="right", closed="right").max(),
                "low": group["low"].resample("1h", label="right", closed="right").min(),
                "close": group["close"].resample("1h", label="right", closed="right").last(),
                "volume": group["volume"].resample("1h", label="right", closed="right").sum(),
                "amount": group["quote_volume"].resample("1h", label="right", closed="right").sum(),
            }
        ).dropna()
        hourly["symbol"] = symbol
        frames.append(hourly.reset_index())
    hourly = pd.concat(frames).sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    log(f"hourly_rows {len(hourly)}")
    return hourly


def actual_records(hourly: pd.DataFrame) -> list[dict]:
    return [
        {"symbol": row.symbol, "timestamp": row.timestamp.isoformat(), "close": float(row.close)}
        for row in hourly[["symbol", "timestamp", "close"]].itertuples(index=False)
    ]


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def build_model(model_key: str, model_revision: str, tokenizer_revision: str, device: str):
    from model import Kronos, KronosPredictor, KronosTokenizer

    model_id, tokenizer_id, max_context = MODEL_IDS[model_key]
    log(f"loading_tokenizer {tokenizer_id}@{tokenizer_revision}")
    tokenizer = KronosTokenizer.from_pretrained(tokenizer_id, revision=tokenizer_revision)
    log(f"loading_model {model_id}@{model_revision}")
    model = Kronos.from_pretrained(model_id, revision=model_revision)
    log(f"building_predictor device={device} max_context={max_context}")
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=max_context)
    log("predictor_ready")
    return predictor, model_id, max_context


def target_timestamps(hourly: pd.DataFrame, days: int, end: str | None, context: int) -> list[pd.Timestamp]:
    latest = hourly["timestamp"].max()
    end_ts = pd.Timestamp(end) if end else latest
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize("UTC")
    else:
        end_ts = end_ts.tz_convert("UTC")
    start_ts = end_ts - pd.Timedelta(days=int(days))
    timestamps = sorted(pd.Timestamp(ts) for ts in hourly["timestamp"].unique())
    targets = [ts for ts in timestamps if start_ts < ts <= end_ts]
    min_context_ts = min(timestamps) + pd.Timedelta(hours=context)
    targets = [ts for ts in targets if ts >= min_context_ts]
    return targets


def prediction_batch(
    predictor,
    hourly: pd.DataFrame,
    symbols: tuple[str, ...],
    target: pd.Timestamp,
    *,
    context: int,
    model_version: str,
    model_hash: str,
    sample_count: int,
    temperature: float,
    top_p: float,
) -> list[dict]:
    asof = target - pd.Timedelta(hours=1)
    inputs = []
    x_timestamps = []
    y_timestamps = []
    valid_symbols = []
    for symbol in symbols:
        frame = hourly[(hourly["symbol"] == symbol) & (hourly["timestamp"] <= asof)].tail(context).copy()
        if len(frame) != context:
            raise ValueError(f"{symbol} has {len(frame)} context rows before {asof}, expected {context}.")
        inputs.append(frame[["open", "high", "low", "close", "volume", "amount"]].reset_index(drop=True))
        x_timestamps.append(pd.Series(frame["timestamp"].reset_index(drop=True)))
        y_timestamps.append(pd.Series([target]))
        valid_symbols.append(symbol)

    outputs = predictor.predict_batch(
        inputs,
        x_timestamps,
        y_timestamps,
        pred_len=1,
        T=temperature,
        top_p=top_p,
        sample_count=sample_count,
        verbose=False,
    )
    records = []
    for symbol, output in zip(valid_symbols, outputs):
        event = PredictionEvent(
            symbol=symbol,
            prediction_asof=asof,
            execution_timestamp=target,
            target_timestamp=target,
            features_cutoff=asof,
            horizon=str(target - asof),
            model_version=model_version,
            model_hash=model_hash,
            predicted_close=float(output["close"].iloc[-1]),
        ).validate()
        records.append(asdict(event))
    return records


def validation_report(predictions: list[dict], actuals: list[dict]) -> dict:
    return AlphaValidationReport(
        predictions=predictions,
        actuals=actuals,
        long_threshold=0.0,
        transaction_cost_bps=TX_COST_BPS,
        min_net_excess_return=0.01,
        min_directional_accuracy=0.52,
        min_active_period_fraction=0.10,
        min_observations=100,
        min_symbols=6,
        min_regimes=1,
    ).compute()


def simulate_prod_risk(predictions: list[dict], hourly: pd.DataFrame, symbols: tuple[str, ...]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    prices = hourly.pivot(index="timestamp", columns="symbol", values="close").sort_index()
    frame = pd.DataFrame(predictions)
    for column in ("prediction_asof", "target_timestamp"):
        frame[column] = pd.to_datetime(frame[column], utc=True)
    frame["predicted_close"] = frame["predicted_close"].astype(float)

    equity = INITIAL_CAPITAL
    peak = equity
    day_start_equity = equity
    current_day = None
    previous_weights = {symbol: 0.0 for symbol in symbols}
    daily_halted = False
    drawdown_halted = False
    rows = []
    trades = []
    halt_events = []

    for target, group in frame.groupby("target_timestamp", sort=True):
        asof = target - pd.Timedelta(hours=1)
        if asof not in prices.index or target not in prices.index:
            continue
        day = target.date()
        if day != current_day:
            current_day = day
            day_start_equity = equity
            daily_halted = False

        reference = prices.loc[asof]
        actual = prices.loc[target]
        active = []
        for row in group.itertuples(index=False):
            predicted_return = float(row.predicted_close) / float(reference[row.symbol]) - 1.0
            if predicted_return > 0:
                active.append(row.symbol)
        weights = {symbol: 0.0 for symbol in symbols}
        if active:
            per_symbol = min(MAX_SYMBOL, MAX_GROSS / len(active))
            for symbol in active:
                weights[symbol] = per_symbol

        drawdown = equity / max(peak, 1e-12) - 1.0
        daily_loss = equity / max(day_start_equity, 1e-12) - 1.0
        reason = "signal"
        if drawdown <= -MAX_DRAWDOWN_HALT:
            drawdown_halted = True
        if daily_loss <= -MAX_DAILY_LOSS:
            daily_halted = True
        if drawdown_halted or daily_halted:
            weights = {symbol: 0.0 for symbol in symbols}
            reason = "max_drawdown_halt" if drawdown_halted else "daily_loss_halt"
            if not halt_events or halt_events[-1]["timestamp"][:10] != target.isoformat()[:10]:
                halt_events.append(
                    {
                        "timestamp": target.isoformat(),
                        "reason": reason,
                        "equity": equity,
                        "drawdown": drawdown,
                        "daily_loss": daily_loss,
                    }
                )

        turnover = sum(abs(weights[symbol] - previous_weights[symbol]) for symbol in symbols)
        cost = equity * turnover * COST_RATE
        period_return = sum(weights[symbol] * (float(actual[symbol]) / float(reference[symbol]) - 1.0) for symbol in symbols)
        next_equity = equity + equity * period_return - cost
        for symbol in symbols:
            delta = weights[symbol] - previous_weights[symbol]
            if abs(delta) > 1e-12:
                trades.append(
                    {
                        "timestamp": target.isoformat(),
                        "symbol": symbol,
                        "side": "BUY" if delta > 0 else "SELL",
                        "delta_weight": delta,
                        "notional": equity * abs(delta),
                        "price": float(reference[symbol]),
                        "reason": reason,
                    }
                )
        rows.append(
            {
                "timestamp": target.isoformat(),
                "equity": next_equity,
                "period_return": next_equity / equity - 1.0 if equity else 0.0,
                "gross_exposure": sum(abs(value) for value in weights.values()),
                "active_positions": sum(1 for value in weights.values() if value > 0),
                "turnover": turnover,
                "cost": cost,
                "reason": reason,
            }
        )
        equity = next_equity
        peak = max(peak, equity)
        previous_weights = weights

    curve = pd.DataFrame(rows)
    trades_df = pd.DataFrame(trades)
    returns = curve["period_return"].replace([np.inf, -np.inf], np.nan).dropna()
    sharpe = None
    if len(returns) > 1 and returns.std(ddof=0) > 0:
        sharpe = float(returns.mean() / returns.std(ddof=0) * math.sqrt(365 * 24))
    drawdown_series = curve["equity"] / curve["equity"].cummax() - 1.0
    baseline = prices.loc[curve["timestamp"].map(pd.Timestamp), list(symbols)].pct_change().mean(axis=1).fillna(0.0)
    total_return = float(curve["equity"].iloc[-1] / INITIAL_CAPITAL - 1.0)
    metrics = {
        "initial_capital": INITIAL_CAPITAL,
        "final_capital": float(curve["equity"].iloc[-1]),
        "pnl": float(curve["equity"].iloc[-1] - INITIAL_CAPITAL),
        "total_return": total_return,
        "max_drawdown": float(drawdown_series.min()),
        "annualized_sharpe_hourly": sharpe,
        "active_period_fraction": float((curve["gross_exposure"] > 0).mean()),
        "average_gross_exposure": float(curve["gross_exposure"].mean()),
        "max_gross_exposure": float(curve["gross_exposure"].max()),
        "total_turnover": float(curve["turnover"].sum()),
        "estimated_cost_paid": float(curve["cost"].sum()),
        "trade_events": int(len(trades_df)),
        "halt_events": int(len(halt_events)),
        "halt_event_sample": halt_events[:8],
        "baseline_equal_weight_return": float((1.0 + baseline).prod() - 1.0),
    }
    metrics["net_excess_return"] = metrics["total_return"] - metrics["baseline_equal_weight_return"]
    return curve, trades_df, metrics


def file_entry(path: Path) -> dict:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": file_checksum(path)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="var/kronos_shadow_replay_90d_20260305_20260603/ohlcv_1m.csv.gz")
    parser.add_argument("--out", default="var/kronos_kronos_hourly_oos_7d_gpu0")
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--end", default="2026-06-03T00:00:00Z")
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--model-key", default="kronos-small", choices=sorted(MODEL_IDS))
    parser.add_argument("--model-revision", default="901c26c1332695a2a8f243eb2f37243a37bea320")
    parser.add_argument("--tokenizer-revision", default="0e0117387f39004a9016484a186a908917e22426")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--sample-count", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--limit-targets", type=int, default=0)
    args = parser.parse_args(argv)

    if args.device.startswith("cuda"):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
    symbols = tuple(item.strip().upper() for item in args.symbols.split(",") if item.strip())
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = out_dir / "prediction_results.jsonl"
    actuals_path = out_dir / "actuals.json"
    report_path = out_dir / "validation_report.json"
    summary_path = out_dir / "summary.json"

    hourly = load_hourly(Path(args.data), symbols)
    actuals = actual_records(hourly)
    write_json(actuals_path, {"actuals": actuals, "interval": "1h", "timestamp_semantics": "period_end_exclusive_utc"})

    targets = target_timestamps(hourly, args.days, args.end, args.context)
    if args.limit_targets:
        targets = targets[: args.limit_targets]
    existing = read_jsonl(predictions_path)
    done_targets = {(row["symbol"], str(pd.Timestamp(row["target_timestamp"]))) for row in existing}
    model_hash = stable_hash(
        {
            "model_key": args.model_key,
            "model_revision": args.model_revision,
            "tokenizer_revision": args.tokenizer_revision,
        }
    )
    predictor, model_version, max_context = build_model(
        args.model_key,
        args.model_revision,
        args.tokenizer_revision,
        args.device,
    )
    log(f"targets {len(targets)} existing_predictions {len(existing)}")
    for index, target in enumerate(targets, start=1):
        if all((symbol, str(target)) in done_targets for symbol in symbols):
            continue
        records = prediction_batch(
            predictor,
            hourly,
            symbols,
            target,
            context=min(args.context, max_context),
            model_version=model_version,
            model_hash=model_hash,
            sample_count=args.sample_count,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        append_jsonl(predictions_path, records)
        for record in records:
            done_targets.add((record["symbol"], str(pd.Timestamp(record["target_timestamp"]))))
        if index == 1 or index % 12 == 0 or index == len(targets):
            log(f"predicted_target {index}/{len(targets)} {target.isoformat()} records={len(records)}")

    predictions = read_jsonl(predictions_path)
    report = validation_report(predictions, actuals)
    write_json(report_path, report)
    curve, trades, portfolio = simulate_prod_risk(predictions, hourly, symbols)
    curve_path = out_dir / "equity_curve.csv"
    trades_path = out_dir / "trades.csv"
    curve.to_csv(curve_path, index=False)
    trades.to_csv(trades_path, index=False)
    summary = {
        "schema_version": "kronos.gpu_hourly_oos.v1",
        "generated_at": pd.Timestamp.now("UTC").isoformat(),
        "code_version": current_code_version(),
        "symbols": list(symbols),
        "interval": "1h",
        "days": args.days,
        "targets": len(targets),
        "predictions": len(predictions),
        "model_key": args.model_key,
        "model_version": model_version,
        "model_hash": model_hash,
        "model_revision": args.model_revision,
        "tokenizer_revision": args.tokenizer_revision,
        "device": args.device,
        "cuda_visible_devices": args.cuda_visible_devices,
        "validation_report": report,
        "portfolio": portfolio,
        "files": {
            "prediction_results.jsonl": file_entry(predictions_path),
            "actuals.json": file_entry(actuals_path),
            "validation_report.json": file_entry(report_path),
            "equity_curve.csv": file_entry(curve_path),
            "trades.csv": file_entry(trades_path),
        },
    }
    write_json(summary_path, summary)
    log(json.dumps({"summary": str(summary_path.resolve()), "portfolio": portfolio, "accepted": report["accepted"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
