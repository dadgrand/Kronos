import argparse
import datetime as dt
import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PolicyConfig:
    signal: str
    mode: str
    k: int
    gross: float
    rebalance_every: int
    pred_abs_floor: float
    weight_scheme: str
    max_symbol_weight: float


def load_predictions(path: Path, split: str, model: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamps"])
    required = {"symbol", "timestamps", "prev_close", "actual_close", "pred_close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    df = df.copy()
    df["split"] = split
    df["model"] = model
    return df


def _cs_zscore(s: pd.Series) -> pd.Series:
    std = s.std(ddof=0)
    if not np.isfinite(std) or std == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s.mean()) / std


def prepare_features(parts: list[pd.DataFrame]) -> pd.DataFrame:
    df = pd.concat(parts, ignore_index=True)
    df = df.sort_values(["symbol", "timestamps"]).reset_index(drop=True)
    if "actual_open" in df.columns:
        df["actual_return"] = df["actual_close"] / df["actual_open"] - 1.0
    else:
        df["actual_return"] = df["actual_close"] / df["prev_close"] - 1.0
    df["pred_return"] = df["pred_close"] / df["prev_close"] - 1.0

    by_symbol = df.groupby("symbol", group_keys=False)
    shifted_ret = by_symbol["actual_return"].shift(1)
    df["mom_3"] = shifted_ret.groupby(df["symbol"]).rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
    df["mom_12"] = shifted_ret.groupby(df["symbol"]).rolling(12, min_periods=3).mean().reset_index(level=0, drop=True)
    df["vol_24"] = shifted_ret.groupby(df["symbol"]).rolling(24, min_periods=6).std().reset_index(level=0, drop=True)
    fallback_vol = float(df["actual_return"].std())
    df["vol_24"] = df["vol_24"].replace([np.inf, -np.inf], np.nan).fillna(fallback_vol).clip(lower=1e-6)
    df["mom_3"] = df["mom_3"].fillna(0.0)
    df["mom_12"] = df["mom_12"].fillna(0.0)

    by_time = df.groupby("timestamps", group_keys=False)
    df["pred_center"] = df["pred_return"] - by_time["pred_return"].transform("median")
    df["pred_rank"] = by_time["pred_return"].rank(pct=True) - 0.5
    df["pred_z"] = by_time["pred_center"].transform(_cs_zscore)
    df["mom_3_z"] = by_time["mom_3"].transform(_cs_zscore)
    df["mom_12_z"] = by_time["mom_12"].transform(_cs_zscore)
    df["signal_rank"] = df["pred_rank"]
    df["signal_center"] = df["pred_center"]
    df["signal_vol_adj"] = df["pred_center"] / df["vol_24"]
    df["signal_rank_mom3"] = 0.85 * df["pred_z"] + 0.15 * df["mom_3_z"]
    df["signal_rank_mom12"] = 0.80 * df["pred_z"] + 0.20 * df["mom_12_z"]
    df["signal_conservative"] = 0.90 * df["pred_z"] - 0.10 * df["mom_3_z"].abs()
    return df.sort_values(["timestamps", "symbol"]).reset_index(drop=True)


def allocate_side(
    symbols: list[str],
    side_budget: float,
    sign: float,
    vol: dict[str, float],
    weight_scheme: str,
    max_symbol_weight: float,
) -> dict[str, float]:
    if not symbols or side_budget <= 0:
        return {}
    if weight_scheme == "inv_vol":
        raw = np.array([1.0 / max(vol.get(symbol, 1e-6), 1e-6) for symbol in symbols], dtype=float)
    elif weight_scheme == "equal":
        raw = np.ones(len(symbols), dtype=float)
    else:
        raise ValueError(f"unknown weight_scheme={weight_scheme}")
    raw = raw / raw.sum() * side_budget
    capped = np.minimum(raw, max_symbol_weight)
    return {symbol: float(sign * weight) for symbol, weight in zip(symbols, capped) if weight > 0}


def target_weights(group: pd.DataFrame, config: PolicyConfig) -> dict[str, float]:
    signal_col = f"signal_{config.signal}"
    candidates = group.dropna(subset=[signal_col, "actual_return", "vol_24"]).copy()
    if config.pred_abs_floor > 0:
        candidates = candidates[candidates["pred_center"].abs() >= config.pred_abs_floor]
    if candidates.empty:
        return {}

    vol = dict(zip(candidates["symbol"], candidates["vol_24"]))
    ranked = candidates.sort_values(signal_col)
    weights: dict[str, float] = {}

    if config.mode == "long_short":
        if len(ranked) < 2 * config.k:
            return {}
        longs = list(ranked.tail(config.k)["symbol"])
        shorts = list(ranked.head(config.k)["symbol"])
        weights.update(
            allocate_side(longs, config.gross / 2.0, 1.0, vol, config.weight_scheme, config.max_symbol_weight)
        )
        weights.update(
            allocate_side(shorts, config.gross / 2.0, -1.0, vol, config.weight_scheme, config.max_symbol_weight)
        )
    elif config.mode == "long_only":
        longs = ranked[ranked[signal_col] > 0].tail(config.k)
        if longs.empty:
            return {}
        weights.update(
            allocate_side(
                list(longs["symbol"]),
                config.gross,
                1.0,
                vol,
                config.weight_scheme,
                config.max_symbol_weight,
            )
        )
    elif config.mode == "short_only":
        shorts = ranked[ranked[signal_col] < 0].head(config.k)
        if shorts.empty:
            return {}
        weights.update(
            allocate_side(
                list(shorts["symbol"]),
                config.gross,
                -1.0,
                vol,
                config.weight_scheme,
                config.max_symbol_weight,
            )
        )
    else:
        raise ValueError(f"unknown mode={config.mode}")
    return weights


def max_drawdown_pct(equity: np.ndarray) -> float:
    if equity.size == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    return float(((equity / peak) - 1.0).min() * 100.0)


def backtest_policy(
    df: pd.DataFrame,
    split: str,
    config: PolicyConfig,
    *,
    initial_cash: float,
    cost_bps: float,
) -> tuple[dict, pd.DataFrame]:
    split_df = df[df["split"] == split].sort_values(["timestamps", "symbol"])
    weights: dict[str, float] = {}
    equity = float(initial_cash)
    records = []
    cost_rate = cost_bps / 10000.0

    for bar_idx, (timestamp, group) in enumerate(split_df.groupby("timestamps", sort=True)):
        available_symbols = set(group["symbol"])
        if bar_idx % config.rebalance_every == 0:
            desired = target_weights(group, config)
        else:
            desired = {symbol: weight for symbol, weight in weights.items() if symbol in available_symbols}

        all_symbols = set(weights) | set(desired)
        turnover = sum(abs(desired.get(symbol, 0.0) - weights.get(symbol, 0.0)) for symbol in all_symbols)
        weights = {symbol: weight for symbol, weight in desired.items() if abs(weight) > 1e-12}

        returns = dict(zip(group["symbol"], group["actual_return"]))
        gross = sum(abs(weight) for weight in weights.values())
        net = sum(weights.values())
        gross_return = sum(weight * returns.get(symbol, 0.0) for symbol, weight in weights.items())
        trading_cost = turnover * cost_rate
        net_return = gross_return - trading_cost
        equity *= max(0.0, 1.0 + net_return)

        records.append(
            {
                "timestamp": timestamp,
                "equity": equity,
                "bar_return": net_return,
                "gross_bar_return": gross_return,
                "trading_cost_return": trading_cost,
                "turnover": turnover,
                "gross_exposure": gross,
                "net_exposure": net,
                "n_long": sum(1 for weight in weights.values() if weight > 0),
                "n_short": sum(1 for weight in weights.values() if weight < 0),
            }
        )

    bars = pd.DataFrame(records)
    if bars.empty:
        summary = {
            "split": split,
            "initial_cash": initial_cash,
            "final_equity": initial_cash,
            "pnl": 0.0,
            "return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "sharpe_like": 0.0,
            "bars": 0,
            "avg_turnover_per_bar": 0.0,
            "total_turnover": 0.0,
            "mean_gross_exposure": 0.0,
            "mean_net_exposure": 0.0,
            "active_rate": 0.0,
            "mean_trading_cost_pct": 0.0,
        }
        return summary, bars

    returns = bars["bar_return"].to_numpy()
    sharpe_like = float(returns.mean() / (returns.std(ddof=0) + 1e-12) * np.sqrt(252 * 50))
    active = bars["gross_exposure"] > 1e-9
    summary = {
        "split": split,
        "initial_cash": initial_cash,
        "final_equity": float(bars["equity"].iloc[-1]),
        "pnl": float(bars["equity"].iloc[-1] - initial_cash),
        "return_pct": float((bars["equity"].iloc[-1] / initial_cash - 1.0) * 100.0),
        "max_drawdown_pct": max_drawdown_pct(bars["equity"].to_numpy()),
        "sharpe_like": sharpe_like,
        "bars": int(len(bars)),
        "avg_turnover_per_bar": float(bars["turnover"].mean()),
        "total_turnover": float(bars["turnover"].sum()),
        "mean_gross_exposure": float(bars["gross_exposure"].mean()),
        "mean_net_exposure": float(bars["net_exposure"].mean()),
        "active_rate": float(active.mean()),
        "mean_trading_cost_pct": float(bars["trading_cost_return"].mean() * 100.0),
    }
    return summary, bars


def score_summary(summary: dict, turnover_penalty: float, drawdown_penalty: float) -> float:
    return (
        float(summary["return_pct"])
        + drawdown_penalty * float(summary["max_drawdown_pct"])
        - turnover_penalty * float(summary["avg_turnover_per_bar"])
    )


def policy_grid(max_gross: float, profile: str) -> list[PolicyConfig]:
    if profile == "quick":
        signals = ["rank", "vol_adj"]
        modes = ["long_short", "long_only", "short_only"]
        ks = [1, 3, 5, 10]
        rebalance_every = [1, 6, 12, 24]
        pred_abs_floors = [0.0, 0.0005]
        weight_schemes = ["equal", "inv_vol"]
        max_symbol_weights = [0.20]
    elif profile == "focused":
        signals = ["rank", "vol_adj", "rank_mom12"]
        modes = ["long_short", "long_only", "short_only"]
        ks = [1, 3, 5, 10, 15]
        rebalance_every = [1, 3, 6, 12, 24]
        pred_abs_floors = [0.0, 0.0005, 0.0010]
        weight_schemes = ["equal", "inv_vol"]
        max_symbol_weights = [0.10, 0.20]
    elif profile == "wide":
        signals = ["rank", "center", "vol_adj", "rank_mom3", "rank_mom12", "conservative"]
        modes = ["long_short", "long_only", "short_only"]
        ks = [3, 5, 8, 10, 15]
        rebalance_every = [1, 2, 3, 6, 12, 24]
        pred_abs_floors = [0.0, 0.0002, 0.0005, 0.0010, 0.0015]
        weight_schemes = ["equal", "inv_vol"]
        max_symbol_weights = [0.10, 0.15, 0.20]
    else:
        raise ValueError(f"unknown search profile: {profile}")
    gross_candidates = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
    grosses = [gross for gross in gross_candidates if gross <= max_gross]
    configs = []
    for values in itertools.product(
        signals,
        modes,
        ks,
        grosses,
        rebalance_every,
        pred_abs_floors,
        weight_schemes,
        max_symbol_weights,
    ):
        config = PolicyConfig(*values)
        if config.mode == "long_short" and 2 * config.k > 50:
            continue
        configs.append(config)
    return configs


def run_search(
    df: pd.DataFrame,
    configs: list[PolicyConfig],
    *,
    initial_cash: float,
    cost_bps: float,
    turnover_penalty: float,
    drawdown_penalty: float,
) -> pd.DataFrame:
    rows = []
    for idx, config in enumerate(configs, start=1):
        summary, _ = backtest_policy(df, "validation", config, initial_cash=initial_cash, cost_bps=cost_bps)
        rows.append(
            {
                **asdict(config),
                **{f"validation_{key}": value for key, value in summary.items() if key != "split"},
                "selection_score": score_summary(summary, turnover_penalty, drawdown_penalty),
                "search_order": idx,
            }
        )
    return pd.DataFrame(rows).sort_values(["selection_score", "validation_return_pct"], ascending=False)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation-only portfolio policy lab for Kronos MOEX predictions.")
    parser.add_argument("--experiment-dir", type=Path, default=Path("outputs/lab_grpo_parquet_i10_20260622_010711"))
    parser.add_argument("--validation-file", default="best_validation_predictions.csv")
    parser.add_argument("--test-file", default="tuned_test_predictions.csv")
    parser.add_argument("--baseline-test-file", default="baseline_test_predictions.csv")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--initial-cash", type=float, default=10000.0)
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--max-gross", type=float, default=2.0)
    parser.add_argument("--turnover-penalty", type=float, default=2.0)
    parser.add_argument("--drawdown-penalty", type=float, default=0.25)
    parser.add_argument("--target-return-pct", type=float, default=30.0)
    parser.add_argument("--search-profile", choices=["quick", "focused", "wide"], default="quick")
    args = parser.parse_args()

    if args.output_dir is None:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = Path("outputs") / f"portfolio_policy_lab_{stamp}"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    validation = load_predictions(args.experiment_dir / args.validation_file, "validation", "grpo_lora")
    test = load_predictions(args.experiment_dir / args.test_file, "test", "grpo_lora")
    df = prepare_features([validation, test])

    configs = policy_grid(args.max_gross, args.search_profile)
    search = run_search(
        df,
        configs,
        initial_cash=args.initial_cash,
        cost_bps=args.cost_bps,
        turnover_penalty=args.turnover_penalty,
        drawdown_penalty=args.drawdown_penalty,
    )
    selected = PolicyConfig(
        signal=str(search.iloc[0]["signal"]),
        mode=str(search.iloc[0]["mode"]),
        k=int(search.iloc[0]["k"]),
        gross=float(search.iloc[0]["gross"]),
        rebalance_every=int(search.iloc[0]["rebalance_every"]),
        pred_abs_floor=float(search.iloc[0]["pred_abs_floor"]),
        weight_scheme=str(search.iloc[0]["weight_scheme"]),
        max_symbol_weight=float(search.iloc[0]["max_symbol_weight"]),
    )

    validation_summary, validation_bars = backtest_policy(
        df, "validation", selected, initial_cash=args.initial_cash, cost_bps=args.cost_bps
    )
    test_summary, test_bars = backtest_policy(df, "test", selected, initial_cash=args.initial_cash, cost_bps=args.cost_bps)

    sensitivity_rows = []
    for cost_multiplier in [0.5, 1.0, 2.0, 3.0]:
        cost = args.cost_bps * cost_multiplier
        summary, _ = backtest_policy(df, "test", selected, initial_cash=args.initial_cash, cost_bps=cost)
        sensitivity_rows.append({"cost_bps": cost, **summary})

    search.to_csv(args.output_dir / "validation_search.csv", index=False)
    search.head(50).to_csv(args.output_dir / "top_validation_policies.csv", index=False)
    pd.DataFrame([validation_summary]).to_csv(args.output_dir / "selected_validation_summary.csv", index=False)
    pd.DataFrame([test_summary]).to_csv(args.output_dir / "selected_test_summary.csv", index=False)
    validation_bars.to_csv(args.output_dir / "selected_validation_bars.csv", index=False)
    test_bars.to_csv(args.output_dir / "selected_test_bars.csv", index=False)
    pd.DataFrame(sensitivity_rows).to_csv(args.output_dir / "cost_sensitivity.csv", index=False)

    if (args.experiment_dir / args.baseline_test_file).exists():
        baseline_test = load_predictions(args.experiment_dir / args.baseline_test_file, "test", "baseline")
        baseline_df = prepare_features([validation, baseline_test])
        baseline_summary, baseline_bars = backtest_policy(
            baseline_df, "test", selected, initial_cash=args.initial_cash, cost_bps=args.cost_bps
        )
        pd.DataFrame([baseline_summary]).to_csv(args.output_dir / "baseline_model_selected_policy_test_summary.csv", index=False)
        baseline_bars.to_csv(args.output_dir / "baseline_model_selected_policy_test_bars.csv", index=False)

    metadata = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "protocol": "Policy search uses validation rows only. Test rows are evaluated once with the selected policy.",
        "experiment_dir": str(args.experiment_dir),
        "validation_file": args.validation_file,
        "test_file": args.test_file,
        "initial_cash": args.initial_cash,
        "cost_bps": args.cost_bps,
        "max_gross": args.max_gross,
        "turnover_penalty": args.turnover_penalty,
        "drawdown_penalty": args.drawdown_penalty,
        "target_return_pct": args.target_return_pct,
        "selected_policy": asdict(selected),
        "validation_summary": validation_summary,
        "test_summary": test_summary,
        "target_hit": test_summary["return_pct"] >= args.target_return_pct,
    }
    write_json(args.output_dir / "summary.json", metadata)

    report = [
        "# Portfolio Policy Lab",
        "",
        "Policy search uses validation predictions only. The selected policy is then applied to the locked test split.",
        "",
        "## Selected Policy",
        "",
        "```json",
        json.dumps(asdict(selected), indent=2),
        "```",
        "",
        "## Validation",
        "",
        f"- Final equity: {validation_summary['final_equity']:.2f}",
        f"- Return: {validation_summary['return_pct']:.2f}%",
        f"- Max drawdown: {validation_summary['max_drawdown_pct']:.2f}%",
        f"- Avg turnover/bar: {validation_summary['avg_turnover_per_bar']:.4f}",
        "",
        "## Test",
        "",
        f"- Final equity: {test_summary['final_equity']:.2f}",
        f"- Return: {test_summary['return_pct']:.2f}%",
        f"- PnL: {test_summary['pnl']:.2f}",
        f"- Max drawdown: {test_summary['max_drawdown_pct']:.2f}%",
        f"- Avg turnover/bar: {test_summary['avg_turnover_per_bar']:.4f}",
        f"- Target hit ({args.target_return_pct:.1f}%): {test_summary['return_pct'] >= args.target_return_pct}",
        "",
        "This is a research backtest, not financial advice or a live trading recommendation.",
        "",
    ]
    (args.output_dir / "README.md").write_text("\n".join(report), encoding="utf-8")

    print(f"Output dir: {args.output_dir}")
    print(f"Selected policy: {asdict(selected)}")
    print(f"Validation return: {validation_summary['return_pct']:.2f}%")
    print(f"Test return: {test_summary['return_pct']:.2f}%")
    print(f"Target hit: {test_summary['return_pct'] >= args.target_return_pct}")


if __name__ == "__main__":
    main()
