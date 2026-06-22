import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from live_paper_alpha_policy import mark_portfolio, trade_to_weights, weights_changed


def parse_json_cell(value, default):
    if value is None or pd.isna(value) or str(value).strip() == "":
        return default
    return json.loads(str(value))


def has_text(value) -> bool:
    if value is None or pd.isna(value):
        return False
    return str(value).strip() != ""


def numeric_or_nan(value) -> float:
    number = pd.to_numeric(value, errors="coerce")
    return float(number) if np.isfinite(number) else np.nan


def price_snapshots(candidate_log: pd.DataFrame, live_log: pd.DataFrame) -> dict[str, dict[str, float]]:
    snapshots: dict[str, dict[str, float]] = {}
    for wall_time, group in candidate_log.groupby("wall_time", sort=False):
        prices = {}
        for _, row in group.iterrows():
            price = pd.to_numeric(row.get("price"), errors="coerce")
            if np.isfinite(price) and price > 0:
                prices[str(row["symbol"])] = float(price)
        snapshots[str(wall_time)] = prices

    for _, row in live_log.iterrows():
        wall_time = str(row["wall_time"])
        prices = snapshots.setdefault(wall_time, {})
        for trade in parse_json_cell(row.get("trades_json"), []):
            price = trade.get("price")
            if price is not None and np.isfinite(float(price)) and float(price) > 0:
                prices[str(trade["symbol"])] = float(price)
        positions = parse_json_cell(row.get("positions_json"), {})
        values = parse_json_cell(row.get("position_values_json"), {})
        for symbol, shares in positions.items():
            value = values.get(symbol)
            if value is None:
                continue
            shares = float(shares)
            if abs(shares) > 1e-12:
                prices[str(symbol)] = abs(float(value) / shares)
    return snapshots


def replay(
    live_log: pd.DataFrame,
    candidate_log: pd.DataFrame,
    *,
    initial_cash: float,
    cost_bps: float,
    interval_minutes: float,
    rebalance_on_target_change: bool,
    target_change_tolerance: float,
    max_session_loss_pct: float | None,
    max_signal_age_minutes: float | None,
) -> tuple[pd.DataFrame, dict]:
    snapshots = price_snapshots(candidate_log, live_log)
    cash = float(initial_cash)
    positions: dict[str, float] = {}
    active_target_weights: dict[str, float] = {}
    total_cost = 0.0
    total_turnover = 0.0
    stopped = False
    rows = []

    for _, row in live_log.iterrows():
        wall_time = str(row["wall_time"])
        target_weights = {
            str(symbol): float(weight)
            for symbol, weight in parse_json_cell(row.get("target_weights_json"), {}).items()
        }
        selected = has_text(row.get("selected_alpha"))
        wall_ts = pd.to_datetime(wall_time, errors="coerce")
        latest_ts = pd.to_datetime(row.get("latest_candle"), errors="coerce")
        signal_source = str(row.get("signal_source") or "candle")
        signal_start_age_minutes = np.nan
        signal_age_minutes = np.nan
        if signal_source == "marketdata_bar":
            signal_start_age_minutes = numeric_or_nan(row.get("signal_start_age_minutes"))
            signal_age_minutes = numeric_or_nan(row.get("signal_age_minutes"))
            if not np.isfinite(signal_age_minutes):
                signal_age_minutes = 0.0
        elif pd.notna(wall_ts) and pd.notna(latest_ts):
            signal_start_age_minutes = (wall_ts - latest_ts).total_seconds() / 60.0
            signal_close_ts = latest_ts + pd.Timedelta(minutes=float(interval_minutes))
            signal_age_minutes = max(0.0, (wall_ts - signal_close_ts).total_seconds() / 60.0)
        stale_signal_hit = (
            max_signal_age_minutes is not None
            and np.isfinite(signal_age_minutes)
            and signal_age_minutes > float(max_signal_age_minutes)
        )
        if stale_signal_hit:
            target_weights = {}
            selected = False
        prices = snapshots.get(wall_time, {})
        pre_equity, position_values = mark_portfolio(cash, positions, prices)
        pre_return_pct = (pre_equity / initial_cash - 1.0) * 100.0
        session_loss_stop_hit = (
            max_session_loss_pct is not None
            and bool(positions)
            and pre_return_pct <= -abs(max_session_loss_pct)
        )
        if session_loss_stop_hit:
            target_weights = {}
            stopped = True
        if stopped:
            target_weights = {}
            selected = False

        target_change_rebalance = (
            rebalance_on_target_change
            and bool(positions)
            and weights_changed(active_target_weights, target_weights, target_change_tolerance)
        )
        cash_gate = not selected and bool(positions)
        opening_trade = not positions and bool(target_weights)
        rebalance_due = (
            cash_gate
            or opening_trade
            or target_change_rebalance
            or session_loss_stop_hit
            or stale_signal_hit
            or (stopped and not positions)
        )

        action = "hold"
        cost = 0.0
        turnover = 0.0
        trades = []
        if rebalance_due:
            cash, positions, cost, turnover, trades = trade_to_weights(
                cash,
                positions,
                target_weights,
                prices,
                cost_bps,
            )
            total_cost += cost
            total_turnover += turnover
            active_target_weights = dict(target_weights)
            if session_loss_stop_hit:
                action = "close_session_loss"
            elif stopped and not positions:
                action = "session_loss_stopped_cash"
            elif stale_signal_hit:
                action = "close_stale_signal" if trades else "stale_signal_cash"
            elif target_change_rebalance:
                action = "rebalance_target_change" if trades else "target_change_no_trade"
            elif cash_gate:
                action = "cash"
            elif opening_trade:
                action = "rebalance"
            else:
                action = "rebalance"

        equity, position_values = mark_portfolio(cash, positions, prices)
        replay_selected_alpha = row.get("selected_alpha") if selected else None
        replay_target_symbols = ",".join(sorted(target_weights))
        rows.append(
            {
                "wall_time": wall_time,
                "latest_candle": row.get("latest_candle"),
                "signal_source": signal_source,
                "source_action": row.get("action"),
                "action": action,
                "source_selected_alpha": row.get("selected_alpha"),
                "source_target_symbols": row.get("target_symbols"),
                "selected_alpha": replay_selected_alpha,
                "target_symbols": replay_target_symbols,
                "position_symbols": ",".join(sorted(positions)),
                "cash": cash,
                "equity": equity,
                "pnl": equity - initial_cash,
                "return_pct": (equity / initial_cash - 1.0) * 100.0,
                "pre_trade_equity": pre_equity,
                "pre_trade_return_pct": pre_return_pct,
                "signal_start_age_minutes": signal_start_age_minutes,
                "signal_age_minutes": signal_age_minutes,
                "stale_signal_hit": stale_signal_hit,
                "session_loss_stop_hit": session_loss_stop_hit,
                "stopped": stopped,
                "target_change_rebalance": target_change_rebalance,
                "trade_cost": cost,
                "turnover": turnover,
                "total_cost": total_cost,
                "total_turnover": total_turnover,
                "positions_json": json.dumps(positions, ensure_ascii=False, sort_keys=True),
                "active_target_weights_json": json.dumps(active_target_weights, ensure_ascii=False, sort_keys=True),
                "trades_json": json.dumps(trades, ensure_ascii=False),
            }
        )

    replay_log = pd.DataFrame(rows)
    final_equity = float(replay_log["equity"].iloc[-1]) if not replay_log.empty else initial_cash
    summary = {
        "initial_cash": initial_cash,
        "final_equity": final_equity,
        "pnl": final_equity - initial_cash,
        "return_pct": (final_equity / initial_cash - 1.0) * 100.0,
        "min_equity": float(replay_log["equity"].min()) if not replay_log.empty else initial_cash,
        "max_equity": float(replay_log["equity"].max()) if not replay_log.empty else initial_cash,
        "min_return_pct": float(replay_log["return_pct"].min()) if not replay_log.empty else 0.0,
        "max_return_pct": float(replay_log["return_pct"].max()) if not replay_log.empty else 0.0,
        "rows": int(len(replay_log)),
        "rebalance_count": int((replay_log["action"] == "rebalance").sum()),
        "target_change_rebalance_count": int((replay_log["action"] == "rebalance_target_change").sum()),
        "session_loss_stop_count": int((replay_log["action"] == "close_session_loss").sum()),
        "stale_signal_count": int((replay_log["stale_signal_hit"]).sum()),
        "stale_signal_cash_count": int((replay_log["action"] == "stale_signal_cash").sum()),
        "close_stale_signal_count": int((replay_log["action"] == "close_stale_signal").sum()),
        "cash_count": int((replay_log["action"] == "cash").sum()),
        "hold_count": int((replay_log["action"] == "hold").sum()),
        "total_cost": total_cost,
        "total_turnover": total_turnover,
        "interval_minutes": interval_minutes,
        "rebalance_on_target_change": rebalance_on_target_change,
        "target_change_tolerance": target_change_tolerance,
        "max_session_loss_pct": max_session_loss_pct,
        "max_signal_age_minutes": max_signal_age_minutes,
    }
    return replay_log, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay live alpha logs with risk overlay variants.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--initial-cash", type=float, default=10000.0)
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--interval-minutes", type=float, default=10.0)
    parser.add_argument("--target-change-tolerance", type=float, default=0.05)
    parser.add_argument("--max-session-loss-pct", type=float, default=1.0)
    parser.add_argument("--max-signal-age-minutes", type=float, default=10.0)
    args = parser.parse_args()

    live_log = pd.read_csv(args.run_dir / "live_log.csv")
    candidate_log = pd.read_csv(args.run_dir / "candidate_log.csv")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    variants = {
        "baseline_like": {
            "rebalance_on_target_change": False,
            "max_session_loss_pct": None,
            "max_signal_age_minutes": None,
        },
        "target_change": {
            "rebalance_on_target_change": True,
            "max_session_loss_pct": None,
            "max_signal_age_minutes": None,
        },
        "loss_stop": {
            "rebalance_on_target_change": False,
            "max_session_loss_pct": args.max_session_loss_pct,
            "max_signal_age_minutes": None,
        },
        "target_change_loss_stop": {
            "rebalance_on_target_change": True,
            "max_session_loss_pct": args.max_session_loss_pct,
            "max_signal_age_minutes": None,
        },
        "signal_age_gate": {
            "rebalance_on_target_change": False,
            "max_session_loss_pct": None,
            "max_signal_age_minutes": args.max_signal_age_minutes,
        },
        "full_live_safety": {
            "rebalance_on_target_change": True,
            "max_session_loss_pct": args.max_session_loss_pct,
            "max_signal_age_minutes": args.max_signal_age_minutes,
        },
    }
    summaries = []
    for name, settings in variants.items():
        replay_log, summary = replay(
            live_log,
            candidate_log,
            initial_cash=args.initial_cash,
            cost_bps=args.cost_bps,
            interval_minutes=args.interval_minutes,
            rebalance_on_target_change=settings["rebalance_on_target_change"],
            target_change_tolerance=args.target_change_tolerance,
            max_session_loss_pct=settings["max_session_loss_pct"],
            max_signal_age_minutes=settings["max_signal_age_minutes"],
        )
        replay_log.to_csv(args.output_dir / f"{name}_replay_log.csv", index=False)
        summaries.append({"variant": name, **summary})

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(args.output_dir / "summary.csv", index=False)
    result = {
        "run_dir": str(args.run_dir),
        "initial_cash": args.initial_cash,
        "cost_bps": args.cost_bps,
        "interval_minutes": args.interval_minutes,
        "target_change_tolerance": args.target_change_tolerance,
        "max_session_loss_pct": args.max_session_loss_pct,
        "max_signal_age_minutes": args.max_signal_age_minutes,
        "summaries": summaries,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
