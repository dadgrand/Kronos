import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StrategySpec:
    name: str
    threshold: float
    beta0: float | None = None
    beta1: float | None = None


def load_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamps"])
    df["actual_return"] = df["actual_close"] / df["prev_close"] - 1.0
    df["pred_return"] = df["pred_close"] / df["prev_close"] - 1.0
    return df


def desired_sign(spec: StrategySpec, pred_return: np.ndarray) -> np.ndarray:
    th = spec.threshold
    if spec.name == "direct_long_short":
        return np.where(pred_return > th, 1, np.where(pred_return < -th, -1, 0))
    if spec.name == "contrarian_long_short":
        return np.where(pred_return > th, -1, np.where(pred_return < -th, 1, 0))
    if spec.name == "direct_long_cash":
        return np.where(pred_return > th, 1, 0)
    if spec.name == "contrarian_long_cash":
        return np.where(pred_return < -th, 1, 0)
    if spec.name == "calibrated_ols_long_short":
        if spec.beta0 is None or spec.beta1 is None:
            raise ValueError("OLS strategy requires beta0 and beta1.")
        expected_return = spec.beta0 + spec.beta1 * pred_return
        return np.where(expected_return > th, 1, np.where(expected_return < -th, -1, 0))
    raise ValueError(f"Unknown strategy: {spec.name}")


def simulate(
    df: pd.DataFrame,
    spec: StrategySpec,
    initial_cash: float,
    cost_bps_per_side: float,
) -> tuple[pd.DataFrame, dict]:
    cash = float(initial_cash)
    position = 0
    rows = []
    cost_rate = cost_bps_per_side / 10000.0
    signals = desired_sign(spec, df["pred_return"].to_numpy(dtype=float))

    for row_idx, (_, row) in enumerate(df.iterrows()):
        entry_price = float(row["prev_close"])
        exit_price = float(row["actual_close"])
        equity_before_trade = cash + position * entry_price

        signal = int(signals[row_idx])
        target_abs_shares = int(np.floor(max(equity_before_trade, 0.0) / entry_price))
        target_position = signal * target_abs_shares
        delta = target_position - position
        trade_value = abs(delta) * entry_price
        trade_cost = trade_value * cost_rate

        if delta > 0:
            cash -= trade_value + trade_cost
        elif delta < 0:
            cash += trade_value - trade_cost

        position = target_position
        equity_after_trade = cash + position * entry_price
        equity_close = cash + position * exit_price
        day_pnl = equity_close - equity_before_trade

        rows.append(
            {
                "timestamps": row["timestamps"],
                "signal": signal,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "shares": position,
                "trade_shares": delta,
                "trade_value": trade_value,
                "trade_cost": trade_cost,
                "equity_before_trade": equity_before_trade,
                "equity_after_trade": equity_after_trade,
                "equity_close": equity_close,
                "day_pnl": day_pnl,
                "actual_return": row["actual_return"],
                "pred_return": row["pred_return"],
                "actual_close": row["actual_close"],
                "pred_close": row["pred_close"],
            }
        )

    if len(df) > 0 and position != 0:
        liquidation_price = float(df["actual_close"].iloc[-1])
        trade_value = abs(position) * liquidation_price
        trade_cost = trade_value * cost_rate
        if position > 0:
            cash += trade_value - trade_cost
        else:
            cash -= trade_value + trade_cost
        rows[-1]["liquidation_trade_value"] = trade_value
        rows[-1]["liquidation_trade_cost"] = trade_cost
        rows[-1]["equity_after_liquidation"] = cash
        position = 0

    trades = pd.DataFrame(rows)
    final_equity = cash if len(trades) and "equity_after_liquidation" in trades.columns else (
        float(trades["equity_close"].iloc[-1]) if len(trades) else initial_cash
    )
    total_cost = float(trades["trade_cost"].sum()) + float(trades.get("liquidation_trade_cost", pd.Series(dtype=float)).fillna(0).sum())
    summary = {
        "strategy": spec.name,
        "threshold": spec.threshold,
        "beta0": spec.beta0,
        "beta1": spec.beta1,
        "initial_cash": initial_cash,
        "final_equity": final_equity,
        "pnl": final_equity - initial_cash,
        "return_pct": (final_equity / initial_cash - 1.0) * 100,
        "trading_costs": total_cost,
        "active_days": int((trades["shares"] != 0).sum()) if len(trades) else 0,
        "long_days": int((trades["shares"] > 0).sum()) if len(trades) else 0,
        "short_days": int((trades["shares"] < 0).sum()) if len(trades) else 0,
        "turnover_rub": float(trades["trade_value"].sum()) + float(trades.get("liquidation_trade_value", pd.Series(dtype=float)).fillna(0).sum()),
        "num_position_changes": int((trades["trade_shares"] != 0).sum()) if len(trades) else 0,
        "max_drawdown_pct": max_drawdown_pct(trades),
    }
    return trades, summary


def max_drawdown_pct(trades: pd.DataFrame) -> float:
    if trades.empty:
        return 0.0
    equity = trades["equity_close"].to_numpy(dtype=float)
    peak = np.maximum.accumulate(equity)
    drawdown = equity / peak - 1.0
    return float(drawdown.min() * 100)


def build_candidates(calibration: pd.DataFrame) -> list[StrategySpec]:
    thresholds = [0.0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.008, 0.010, 0.012, 0.015, 0.020]
    candidates = []
    for name in [
        "direct_long_short",
        "contrarian_long_short",
        "direct_long_cash",
        "contrarian_long_cash",
    ]:
        candidates.extend(StrategySpec(name=name, threshold=th) for th in thresholds)

    x = calibration["pred_return"].to_numpy(dtype=float)
    y = calibration["actual_return"].to_numpy(dtype=float)
    X = np.column_stack([np.ones_like(x), x])
    beta0, beta1 = np.linalg.lstsq(X, y, rcond=None)[0]
    for th in thresholds:
        candidates.append(
            StrategySpec(
                name="calibrated_ols_long_short",
                threshold=th,
                beta0=float(beta0),
                beta1=float(beta1),
            )
        )
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper-trade Kronos-derived YDEX strategies.")
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("outputs/ydex_kronos_base_walkforward_20260621_225559/ydex_walkforward_predictions.csv"),
    )
    parser.add_argument("--calibration-count", type=int, default=100)
    parser.add_argument("--initial-cash", type=float, default=10000.0)
    parser.add_argument("--cost-bps-per-side", type=float, default=10.0)
    args = parser.parse_args()

    predictions = load_predictions(args.predictions)
    calibration = predictions.iloc[: args.calibration_count].copy()
    battle = predictions.iloc[args.calibration_count :].copy()
    if calibration.empty or battle.empty:
        raise RuntimeError("Need both calibration and battle rows.")

    output_dir = args.predictions.parent / "paper_trade_10000"
    output_dir.mkdir(exist_ok=True)

    ranked = []
    candidate_trades = {}
    for spec in build_candidates(calibration):
        trades, summary = simulate(calibration, spec, args.initial_cash, args.cost_bps_per_side)
        ranked.append(summary)
        candidate_trades[(spec.name, spec.threshold, spec.beta0, spec.beta1)] = (spec, trades, summary)

    ranked_df = pd.DataFrame(ranked).sort_values(["final_equity", "active_days"], ascending=[False, False])
    ranked_df.to_csv(output_dir / "calibration_strategy_ranking.csv", index=False)

    best_row = ranked_df.iloc[0]
    best_spec = StrategySpec(
        name=best_row["strategy"],
        threshold=float(best_row["threshold"]),
        beta0=None if pd.isna(best_row["beta0"]) else float(best_row["beta0"]),
        beta1=None if pd.isna(best_row["beta1"]) else float(best_row["beta1"]),
    )

    cal_trades, cal_summary = simulate(calibration, best_spec, args.initial_cash, args.cost_bps_per_side)
    battle_trades, battle_summary = simulate(battle, best_spec, args.initial_cash, args.cost_bps_per_side)
    hold_shares = int(np.floor(args.initial_cash / float(battle["prev_close"].iloc[0])))
    hold_cash = args.initial_cash - hold_shares * float(battle["prev_close"].iloc[0])
    hold_final = hold_cash + hold_shares * float(battle["actual_close"].iloc[-1])
    hold_summary = {
        "strategy": "buy_and_hold_integer_shares",
        "initial_cash": args.initial_cash,
        "shares": hold_shares,
        "start_price": float(battle["prev_close"].iloc[0]),
        "end_price": float(battle["actual_close"].iloc[-1]),
        "final_equity": hold_final,
        "pnl": hold_final - args.initial_cash,
        "return_pct": (hold_final / args.initial_cash - 1.0) * 100,
    }

    cal_trades.to_csv(output_dir / "selected_strategy_calibration_trades.csv", index=False)
    battle_trades.to_csv(output_dir / "selected_strategy_battle_trades.csv", index=False)
    pd.DataFrame([cal_summary]).to_csv(output_dir / "selected_strategy_calibration_summary.csv", index=False)
    pd.DataFrame([battle_summary]).to_csv(output_dir / "selected_strategy_battle_summary.csv", index=False)
    pd.DataFrame([hold_summary]).to_csv(output_dir / "buy_hold_battle_summary.csv", index=False)

    summary_text = [
        "Paper trading setup",
        f"predictions: {args.predictions}",
        f"calibration_rows: {len(calibration)}",
        f"battle_rows: {len(battle)}",
        f"battle_period: {battle['timestamps'].iloc[0]} -> {battle['timestamps'].iloc[-1]}",
        f"initial_cash: {args.initial_cash:.2f}",
        f"cost_bps_per_side: {args.cost_bps_per_side:.2f}",
        "",
        "Selected strategy",
        f"name: {best_spec.name}",
        f"threshold: {best_spec.threshold}",
        f"beta0: {best_spec.beta0}",
        f"beta1: {best_spec.beta1}",
        "",
        "Calibration result",
        *[f"{k}: {v}" for k, v in cal_summary.items()],
        "",
        "Battle result",
        *[f"{k}: {v}" for k, v in battle_summary.items()],
        "",
        "Buy and hold battle baseline",
        *[f"{k}: {v}" for k, v in hold_summary.items()],
    ]
    (output_dir / "paper_trade_summary.txt").write_text("\n".join(summary_text), encoding="utf-8")

    print("Selected strategy:")
    print(best_spec)
    print("\nCalibration summary:")
    print(pd.DataFrame([cal_summary]).to_string(index=False))
    print("\nBattle summary:")
    print(pd.DataFrame([battle_summary]).to_string(index=False))
    print("\nBuy-and-hold battle baseline:")
    print(pd.DataFrame([hold_summary]).to_string(index=False))
    print(f"\nOutput directory: {output_dir}")


if __name__ == "__main__":
    main()
