import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))

from live_paper_alpha_policy import (
    append_marketdata_signal_bar,
    fetch_live_candles,
    json_safe,
    load_protocol_config,
)
from live_paper_moex_policy import append_live_to_matrix, fetch_marketdata
from neural_policy_lab import read_intraday_matrix


def main() -> None:
    parser = argparse.ArgumentParser(description="Check whether live alpha policy inputs are fresh enough.")
    parser.add_argument("--config-json", type=Path, default=Path("configs/strict_alpha_policy_latency_delay3_dd6_20260622.json"))
    parser.add_argument("--board", default="TQBR")
    parser.add_argument("--fetch-days", type=int, default=8)
    parser.add_argument("--fetch-workers", type=int, default=12)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    config = load_protocol_config(args.config_json)
    if args.output_dir is None:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = Path("outputs") / f"live_alpha_readiness_{stamp}"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    _, symbols, base_open, base_close, base_volume = read_intraday_matrix(
        Path(config["dataset_dir"]),
        int(config["interval"]),
    )
    session = requests.Session()
    session.trust_env = False
    started = time.perf_counter()
    now = dt.datetime.now()
    from_date = (now.date() - dt.timedelta(days=args.fetch_days)).isoformat()
    till_date = now.date().isoformat()
    live, errors, fetch_stats = fetch_live_candles(
        symbols,
        board=args.board,
        interval=int(config["interval"]),
        from_date=from_date,
        till_date=till_date,
        workers=max(1, args.fetch_workers),
    )
    open_px, close_px, volume = append_live_to_matrix(base_open, base_close, base_volume, live)
    marketdata = fetch_marketdata(session, args.board, symbols)
    wall_time = dt.datetime.now()

    latest_candle = pd.Timestamp(close_px.index[-1])
    candle_close = latest_candle + pd.Timedelta(minutes=int(config["interval"]))
    candle_start_age_minutes = (pd.Timestamp(wall_time) - latest_candle).total_seconds() / 60.0
    candle_age_minutes = max(0.0, (pd.Timestamp(wall_time) - candle_close).total_seconds() / 60.0)

    max_signal_age_minutes = config.get("live_max_signal_age_minutes")
    max_signal_age_minutes = None if max_signal_age_minutes is None else float(max_signal_age_minutes)
    use_marketdata_signal_bar = bool(config.get("live_use_marketdata_signal_bar", False))
    marketdata_min_coverage = float(config.get("live_marketdata_signal_min_coverage", 0.8))
    marketdata_max_age_minutes = config.get("live_marketdata_signal_max_age_minutes")
    marketdata_max_age_minutes = None if marketdata_max_age_minutes is None else float(marketdata_max_age_minutes)

    market_open_px, market_close_px, market_volume, market_stats = append_marketdata_signal_bar(
        open_px,
        close_px,
        volume,
        marketdata,
        now=wall_time,
        interval=int(config["interval"]),
        min_coverage=marketdata_min_coverage,
        max_age_minutes=marketdata_max_age_minutes,
    )
    signal_source = "marketdata_bar" if use_marketdata_signal_bar and market_stats["marketdata_signal_bar_used"] else "candle"
    signal_age_minutes = 0.0 if signal_source == "marketdata_bar" else candle_age_minutes
    stale_signal_hit = (
        max_signal_age_minutes is not None
        and signal_age_minutes > max_signal_age_minutes
    )
    feed_ready = not stale_signal_hit

    result = {
        "checked_at": wall_time.isoformat(timespec="seconds"),
        "config_json": str(args.config_json),
        "board": args.board,
        "symbols": len(symbols),
        "interval": int(config["interval"]),
        "signal_delay_bars": int(config.get("signal_delay_bars", 0)),
        "latest_candle": str(latest_candle),
        "candle_close": str(candle_close),
        "candle_start_age_minutes": candle_start_age_minutes,
        "candle_age_minutes": candle_age_minutes,
        "max_signal_age_minutes": max_signal_age_minutes,
        "use_marketdata_signal_bar": use_marketdata_signal_bar,
        "marketdata_signal_min_coverage": marketdata_min_coverage,
        "marketdata_signal_max_age_minutes": marketdata_max_age_minutes,
        **market_stats,
        "signal_source": signal_source,
        "signal_age_minutes": signal_age_minutes,
        "stale_signal_hit": stale_signal_hit,
        "feed_ready_for_forward": feed_ready,
        "ready_reason": "ready" if feed_ready else "stale_signal",
        "fetch_elapsed_seconds": fetch_stats.get("fetch_elapsed_seconds"),
        "total_elapsed_seconds": time.perf_counter() - started,
        "fetch_parts": fetch_stats.get("fetch_parts"),
        "fetch_errors": fetch_stats.get("fetch_errors"),
        "fetch_rows": fetch_stats.get("fetch_rows"),
        "errors": errors[:10],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(json_safe(result), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    readme = [
        "# Live Alpha Readiness Check",
        "",
        f"- Config: `{args.config_json}`",
        f"- Checked at: {result['checked_at']}",
        f"- Feed ready for forward: {result['feed_ready_for_forward']}",
        f"- Reason: `{result['ready_reason']}`",
        f"- Signal source: `{result['signal_source']}`",
        f"- Signal age: {result['signal_age_minutes']:.2f} min",
        f"- Latest candle: {result['latest_candle']}",
        f"- Candle close: {result['candle_close']}",
        f"- Raw marketdata prices: {result['marketdata_signal_raw_prices']} / {result['symbols']}",
        f"- Fresh marketdata prices: {result['marketdata_signal_prices']} / {result['symbols']}",
        f"- Stale marketdata prices: {result['marketdata_signal_stale_prices']} / {result['symbols']}",
        "",
    ]
    (args.output_dir / "README.md").write_text("\n".join(readme), encoding="utf-8")
    print(json.dumps(json_safe(result), indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
