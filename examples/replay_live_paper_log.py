import argparse
from pathlib import Path

import pandas as pd


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def replay_short_mark_log(
    log: pd.DataFrame,
    *,
    name: str,
    gross: float,
    initial_cash: float,
    cost_bps: float,
    filter_fail_exit_bars: int | None,
    stop_loss_pct: float | None,
    take_profit_pct: float | None,
) -> tuple[dict, pd.DataFrame]:
    cost_rate = cost_bps / 10000.0
    cash = float(initial_cash)
    shares = 0.0
    symbol = None
    entry_equity = None
    filter_fail_count = 0
    trace = []

    for idx, row in log.iterrows():
        price = float(row["mark_price"])
        equity = cash + shares * price
        action = "hold"
        filter_pass = parse_bool(row["filter_pass"])
        if filter_pass:
            filter_fail_count = 0
        else:
            filter_fail_count += 1

        if idx == 0 and filter_pass and pd.notna(row["target_symbol"]):
            notional = gross * equity
            shares = -notional / price
            cash += notional
            trade_cost = abs(notional) * cost_rate
            cash -= trade_cost
            equity = cash + shares * price
            entry_equity = equity
            symbol = str(row["target_symbol"])
            action = "enter_short"
        elif symbol:
            stop_hit = False
            take_hit = False
            if entry_equity is not None:
                position_return_pct = (equity / entry_equity - 1.0) * 100.0
                stop_hit = stop_loss_pct is not None and position_return_pct <= -abs(stop_loss_pct)
                take_hit = take_profit_pct is not None and position_return_pct >= abs(take_profit_pct)
            if stop_hit or take_hit:
                trade_value = abs(shares) * price
                cash += shares * price
                trade_cost = trade_value * cost_rate
                cash -= trade_cost
                shares = 0.0
                symbol = None
                entry_equity = None
                equity = cash
                action = "close_stop_loss" if stop_hit else "close_take_profit"
            elif filter_fail_exit_bars is not None and filter_fail_count >= max(1, filter_fail_exit_bars):
                trade_value = abs(shares) * price
                cash += shares * price
                trade_cost = trade_value * cost_rate
                cash -= trade_cost
                shares = 0.0
                symbol = None
                entry_equity = None
                equity = cash
                action = "exit_filter_fail"

        trace.append(
            {
                "wall_time": row["wall_time"],
                "price": price,
                "equity": equity,
                "pnl": equity - initial_cash,
                "action": action,
                "filter_pass": filter_pass,
                "filter_fail_count": filter_fail_count,
                "symbol": symbol,
                "shares": shares,
            }
        )

    if symbol:
        final_price = float(log.iloc[-1]["mark_price"])
        trade_value = abs(shares) * final_price
        cash += shares * final_price
        trade_cost = trade_value * cost_rate
        cash -= trade_cost
        shares = 0.0
        equity = cash
        trace.append(
            {
                "wall_time": log.iloc[-1]["wall_time"],
                "price": final_price,
                "equity": equity,
                "pnl": equity - initial_cash,
                "action": "final_close",
                "filter_pass": parse_bool(log.iloc[-1]["filter_pass"]),
                "filter_fail_count": filter_fail_count,
                "symbol": None,
                "shares": 0.0,
            }
        )

    trace_df = pd.DataFrame(trace)
    equity_curve = trace_df["equity"]
    summary = {
        "name": name,
        "gross": gross,
        "filter_fail_exit_bars": filter_fail_exit_bars,
        "stop_loss_pct": stop_loss_pct,
        "take_profit_pct": take_profit_pct,
        "final_equity": float(equity_curve.iloc[-1]),
        "pnl": float(equity_curve.iloc[-1] - initial_cash),
        "return_pct": float((equity_curve.iloc[-1] / initial_cash - 1.0) * 100.0),
        "min_equity": float(equity_curve.min()),
        "max_drawdown_pct": float(((equity_curve / equity_curve.cummax()) - 1.0).min() * 100.0),
        "actions": " | ".join(trace_df.loc[trace_df["action"] != "hold", "action"].to_list()),
    }
    return summary, trace_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a saved live paper log under simple execution variants.")
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--initial-cash", type=float, default=10000.0)
    parser.add_argument("--cost-bps", type=float, default=10.0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log = pd.read_csv(args.log, parse_dates=["wall_time"])
    scenarios = [
        ("old_v1_gross4_hold_force_close", 4.0, None, None, None),
        ("v2_default_gross1_stop_no_filter_exit", 1.0, None, 0.75, 0.75),
        ("v2_gross1_stop_exit_after_1_fail", 1.0, 1, 0.75, 0.75),
        ("v2_gross1_stop_exit_after_3_fails", 1.0, 3, 0.75, 0.75),
        ("v2_gross1_stop_exit_after_5_fails", 1.0, 5, 0.75, 0.75),
        ("v2_gross2_stop_no_filter_exit", 2.0, None, 0.75, 0.75),
        ("v2_gross4_stop_no_filter_exit", 4.0, None, 0.75, 0.75),
    ]

    summaries = []
    for name, gross, filter_fail_exit_bars, stop_loss_pct, take_profit_pct in scenarios:
        summary, trace = replay_short_mark_log(
            log,
            name=name,
            gross=gross,
            initial_cash=args.initial_cash,
            cost_bps=args.cost_bps,
            filter_fail_exit_bars=filter_fail_exit_bars,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
        )
        summaries.append(summary)
        trace.to_csv(args.output_dir / f"{name}.csv", index=False)

    comparison = pd.DataFrame(summaries)
    comparison.to_csv(args.output_dir / "comparison.csv", index=False)
    print(comparison.to_string(index=False))
    print(f"Output dir: {args.output_dir}")


if __name__ == "__main__":
    main()
