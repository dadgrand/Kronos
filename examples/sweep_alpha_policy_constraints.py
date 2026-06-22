import argparse
import datetime as dt
import itertools
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from filtered_neural_policy_lab import compute_market_features
from neural_policy_lab import build_matrices, read_intraday_matrix
from stop_aware_filtered_policy_lab import parse_floats, parse_ints, selection_score
from walk_forward_alpha_policy_lab import (
    build_alpha_scores,
    build_candidates,
    delay_market_features,
    delay_matrix,
    run_backtest_pair,
    stop_policy,
)
from walk_forward_stop_aware_policy_lab import make_windows, parse_modes, split_masks_for_window, summarize_compounded


def parse_float_grid(value: str) -> list[float | None]:
    parsed: list[float | None] = []
    for item in value.split(","):
        item = item.strip().lower()
        if not item:
            continue
        parsed.append(None if item in {"none", "null", "off"} else float(item))
    return parsed


def parse_pipe_groups(value: str, item_parser) -> list[tuple]:
    groups = []
    for group in value.split("|"):
        group = group.strip()
        if not group:
            continue
        groups.append(tuple(item_parser(group)))
    if not groups:
        raise ValueError("at least one group is required")
    return groups


def parse_string_items(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def alpha_parts(alpha_name: str) -> tuple[str, int]:
    kind, _, lag = alpha_name.rpartition("_")
    return kind, int(lag)


def group_label(values: tuple) -> str:
    return ",".join(str(value) for value in values)


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.4g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def base_window_results(windows, initial_cash: float) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "window_id": window.window_id,
                "validation_start": window.validation_start.isoformat(),
                "validation_end": window.validation_end.isoformat(),
                "test_start": window.test_start.isoformat(),
                "test_end": window.test_end.isoformat(),
                "selected": "cash",
                "alpha_name": "cash",
                "mode": "cash",
                "k": 0,
                "gross": 0.0,
                "rebalance_every": 0,
                "validation_return_pct": 0.0,
                "validation_max_drawdown_pct": 0.0,
                "worst_validation_segment_return_pct": None,
                "test_return_pct": 0.0,
                "test_final_equity": initial_cash,
                "test_pnl": 0.0,
                "test_max_drawdown_pct": 0.0,
                "test_stress_return_pct": 0.0,
            }
            for window in windows
        ]
    )


def precompute_candidate_metrics(args: argparse.Namespace) -> tuple[pd.DataFrame, list]:
    index, symbols, open_px, close_px, volume = read_intraday_matrix(args.dataset_dir, args.interval)
    matrices = build_matrices(open_px, close_px, volume, args.horizon)
    matrices["bar_return_clean"] = np.nan_to_num(matrices["bar_return"], nan=0.0)
    market_features = compute_market_features(index, symbols, matrices["bar_return"])
    alpha_scores = build_alpha_scores(
        close_px,
        lags=parse_ints(args.alpha_lags),
        kinds=parse_string_items(args.alpha_kinds),
        normalize=args.alpha_normalize,
    )
    if args.signal_delay_bars < 0:
        raise ValueError("--signal-delay-bars must be non-negative")
    if args.signal_delay_bars > 0:
        alpha_scores = {
            alpha_name: delay_matrix(scores, args.signal_delay_bars, np.nan)
            for alpha_name, scores in alpha_scores.items()
        }
        market_features = delay_market_features(market_features, args.signal_delay_bars)

    modes = parse_modes(args.modes)
    candidates = build_candidates(
        list(alpha_scores),
        modes=modes,
        k_grid=parse_ints(args.k_grid),
        gross_grid=parse_floats(args.gross_grid),
        rebalance_grid=parse_ints(args.rebalance_grid),
    )
    windows = make_windows(
        index,
        start_date=args.start_date,
        end_date=args.end_date,
        validation_days=args.validation_days,
        test_days=args.test_days,
        step_days=args.step_days,
        max_windows=args.max_windows,
    )
    if not windows:
        raise ValueError("No complete walk-forward windows fit the requested date range.")

    zero_confidence = np.zeros(matrices["bar_return"].shape[0], dtype=np.float32)
    rows = []
    total = len(windows) * len(candidates)
    done = 0
    for window in windows:
        masks = split_masks_for_window(index, window)
        rows_by_split = {split: np.flatnonzero(mask) for split, mask in masks.items()}
        for candidate in candidates:
            scores = alpha_scores[candidate.alpha_name]
            policy = stop_policy(candidate)
            validation_summary, validation_stress_summary, _, worst_segment = run_backtest_pair(
                scores,
                matrices,
                masks,
                rows_by_split,
                market_features,
                zero_confidence,
                "validation",
                policy,
                initial_cash=args.initial_cash,
                cost_bps=args.cost_bps,
                stress_cost_bps=args.stress_cost_bps,
                segment_count=args.segment_count,
                engine="fast",
            )
            test_summary, test_stress_summary, _, _ = run_backtest_pair(
                scores,
                matrices,
                masks,
                rows_by_split,
                market_features,
                zero_confidence,
                "test",
                policy,
                initial_cash=args.initial_cash,
                cost_bps=args.cost_bps,
                stress_cost_bps=args.stress_cost_bps,
                segment_count=args.segment_count,
                engine="fast",
            )
            alpha_kind, alpha_lag = alpha_parts(candidate.alpha_name)
            rows.append(
                {
                    "window_id": window.window_id,
                    **asdict(candidate),
                    "alpha_kind": alpha_kind,
                    "alpha_lag": alpha_lag,
                    **{f"validation_{key}": value for key, value in validation_summary.items() if key != "split"},
                    "validation_stress_return_pct": validation_stress_summary["return_pct"],
                    "validation_stress_max_drawdown_pct": validation_stress_summary["max_drawdown_pct"],
                    "worst_validation_segment_return_pct": worst_segment,
                    **{f"test_{key}": value for key, value in test_summary.items() if key != "split"},
                    "test_stress_return_pct": test_stress_summary["return_pct"],
                }
            )
            done += 1
        print(f"precomputed window={window.window_id} candidates={done}/{total}", flush=True)

    metrics = pd.DataFrame(rows)
    metrics["raw_selection_score"] = metrics.apply(
        lambda row: selection_score(
            {
                "return_pct": row["validation_return_pct"],
                "max_drawdown_pct": row["validation_max_drawdown_pct"],
                "avg_turnover_per_bar": row["validation_avg_turnover_per_bar"],
            },
            args.drawdown_penalty,
            args.turnover_penalty,
        ),
        axis=1,
    )
    return metrics, windows


def build_windows_for_args(args: argparse.Namespace) -> list:
    index, _, _, _, _ = read_intraday_matrix(args.dataset_dir, args.interval)
    windows = make_windows(
        index,
        start_date=args.start_date,
        end_date=args.end_date,
        validation_days=args.validation_days,
        test_days=args.test_days,
        step_days=args.step_days,
        max_windows=args.max_windows,
    )
    if not windows:
        raise ValueError("No complete walk-forward windows fit the requested date range.")
    return windows


def normalize_metric_columns(metrics: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = [
        "window_id",
        "k",
        "gross",
        "rebalance_every",
        "alpha_lag",
        "validation_return_pct",
        "validation_max_drawdown_pct",
        "validation_avg_turnover_per_bar",
        "validation_stress_return_pct",
        "validation_stress_max_drawdown_pct",
        "worst_validation_segment_return_pct",
        "raw_selection_score",
        "test_return_pct",
        "test_final_equity",
        "test_pnl",
        "test_max_drawdown_pct",
        "test_stress_return_pct",
    ]
    normalized = metrics.copy()
    for column in numeric_columns:
        if column in normalized.columns:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    return normalized


def build_window_results(
    *,
    windows,
    selected_rows: pd.DataFrame,
    initial_cash: float,
) -> pd.DataFrame:
    window_results = base_window_results(windows, initial_cash)
    if selected_rows.empty:
        return window_results

    selected_by_window = selected_rows.set_index("window_id")
    for idx, row in window_results.iterrows():
        window_id = int(row["window_id"])
        if window_id not in selected_by_window.index:
            continue
        selected = selected_by_window.loc[window_id]
        window_results.loc[idx, "selected"] = "alpha_policy"
        for key in ["alpha_name", "mode", "k", "gross", "rebalance_every"]:
            window_results.loc[idx, key] = selected[key]
        for key in [
            "validation_return_pct",
            "validation_max_drawdown_pct",
            "worst_validation_segment_return_pct",
            "test_return_pct",
            "test_final_equity",
            "test_pnl",
            "test_max_drawdown_pct",
            "test_stress_return_pct",
        ]:
            window_results.loc[idx, key] = selected[key]
    return window_results


def evaluate_config(
    metrics: pd.DataFrame,
    windows,
    *,
    config_id: int,
    mode_set: tuple[str, ...],
    alpha_kind_set: tuple[str, ...],
    alpha_lag_set: tuple[int, ...],
    k_set: tuple[int, ...],
    gross_set: tuple[float, ...],
    rebalance_set: tuple[int, ...],
    min_validation_return_pct: float | None,
    min_validation_stress_return_pct: float | None,
    min_validation_segment_return_pct: float | None,
    max_validation_drawdown_pct: float | None,
    segment_selection_penalty: float,
    stress_selection_weight: float,
    initial_cash: float,
    target_return_pct: float,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    mask = (
        metrics["mode"].isin(mode_set)
        & metrics["alpha_kind"].isin(alpha_kind_set)
        & metrics["alpha_lag"].isin(alpha_lag_set)
        & metrics["k"].isin(k_set)
        & metrics["gross"].isin(gross_set)
        & metrics["rebalance_every"].isin(rebalance_set)
    )
    if mask.any():
        filtered = metrics.loc[mask].copy()
        constrained = np.ones(len(filtered), dtype=bool)
        if min_validation_return_pct is not None:
            constrained &= filtered["validation_return_pct"].to_numpy(dtype=float) >= min_validation_return_pct
        if min_validation_stress_return_pct is not None:
            constrained &= (
                filtered["validation_stress_return_pct"].to_numpy(dtype=float)
                >= min_validation_stress_return_pct
            )
        if min_validation_segment_return_pct is not None:
            segment = filtered["worst_validation_segment_return_pct"].to_numpy(dtype=float)
            constrained &= np.isfinite(segment) & (segment >= min_validation_segment_return_pct)
        if max_validation_drawdown_pct is not None:
            constrained &= (
                filtered["validation_max_drawdown_pct"].to_numpy(dtype=float)
                >= -abs(max_validation_drawdown_pct)
            )
        filtered = filtered.loc[constrained].copy()
    else:
        filtered = pd.DataFrame()

    if filtered.empty:
        selected_rows = pd.DataFrame()
    else:
        filtered["selection_score"] = (
            filtered["raw_selection_score"].astype(float)
            + segment_selection_penalty * filtered["worst_validation_segment_return_pct"].astype(float)
            + stress_selection_weight * filtered["validation_stress_return_pct"].astype(float)
        )
        selected_rows = filtered.loc[filtered.groupby("window_id")["selection_score"].idxmax()].copy()

    window_results = build_window_results(windows=windows, selected_rows=selected_rows, initial_cash=initial_cash)
    compounded = summarize_compounded(window_results, initial_cash, target_return_pct)
    stress_window_results = window_results.copy()
    stress_window_results["test_return_pct"] = stress_window_results["test_stress_return_pct"].astype(float)
    compounded_stress = summarize_compounded(stress_window_results, initial_cash, target_return_pct)
    traded = window_results[window_results["selected"] == "alpha_policy"]

    summary = {
        "config_id": config_id,
        "mode_set": group_label(mode_set),
        "alpha_kind_set": group_label(alpha_kind_set),
        "alpha_lag_set": group_label(alpha_lag_set),
        "k_set": group_label(k_set),
        "gross_set": group_label(gross_set),
        "rebalance_set": group_label(rebalance_set),
        "min_validation_return_pct": min_validation_return_pct,
        "min_validation_stress_return_pct": min_validation_stress_return_pct,
        "min_validation_segment_return_pct": min_validation_segment_return_pct,
        "max_validation_drawdown_pct": max_validation_drawdown_pct,
        "segment_selection_penalty": segment_selection_penalty,
        "stress_selection_weight": stress_selection_weight,
        "window_count": int(len(window_results)),
        "trade_window_count": int(len(traded)),
        "return_pct": compounded["return_pct"],
        "final_equity": compounded["final_equity"],
        "pnl": compounded["pnl"],
        "stress_return_pct": compounded_stress["return_pct"],
        "max_drawdown_pct": compounded["max_drawdown_pct"],
        "trade_window_win_rate": float((traded["test_return_pct"].astype(float) > 0.0).mean()) if len(traded) else 0.0,
        "worst_trade_window_test_return_pct": float(traded["test_return_pct"].astype(float).min()) if len(traded) else 0.0,
        "mean_trade_window_test_return_pct": float(traded["test_return_pct"].astype(float).mean()) if len(traded) else 0.0,
        "target_hit": bool(compounded["target_hit"]),
    }
    return summary, window_results, selected_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep strict alpha policy constraints from precomputed candidate paths.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("datasets/moex_intraday_6f8f836cc811"))
    parser.add_argument("--interval", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--start-date", default="2022-09-01")
    parser.add_argument("--end-date", default="2026-06-20")
    parser.add_argument("--validation-days", type=int, default=30)
    parser.add_argument("--test-days", type=int, default=7)
    parser.add_argument("--step-days", type=int, default=30)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--alpha-lags", default="12,48,96")
    parser.add_argument("--alpha-kinds", default="mom,rev,voladj_mom,voladj_rev")
    parser.add_argument("--alpha-normalize", choices=["none", "cs_zscore"], default="none")
    parser.add_argument("--modes", default="short_only,long_only,long_short")
    parser.add_argument("--k-grid", default="1,3")
    parser.add_argument("--gross-grid", default="0.5,1.0,1.5")
    parser.add_argument("--rebalance-grid", default="24")
    parser.add_argument("--signal-delay-bars", type=int, default=3)
    parser.add_argument("--initial-cash", type=float, default=10000.0)
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--stress-cost-bps", type=float, default=20.0)
    parser.add_argument("--drawdown-penalty", type=float, default=0.35)
    parser.add_argument("--turnover-penalty", type=float, default=2.0)
    parser.add_argument("--segment-count", type=int, default=5)
    parser.add_argument("--target-return-pct", type=float, default=10.0)

    parser.add_argument("--mode-sets", default="short_only,long_only,long_short|short_only|long_only|long_short")
    parser.add_argument("--alpha-kind-sets", default="mom,rev,voladj_mom,voladj_rev|mom,rev|mom|rev")
    parser.add_argument("--alpha-lag-sets", default="12,48,96|48,96|96")
    parser.add_argument("--k-sets", default="1,3|1|3")
    parser.add_argument("--gross-sets", default="0.5,1.0,1.5|0.5")
    parser.add_argument("--rebalance-sets", default="24")
    parser.add_argument("--min-validation-return-grid", default="5,8,12,16")
    parser.add_argument("--min-validation-stress-return-grid", default="none,0")
    parser.add_argument("--min-validation-segment-return-grid", default="0,0.5")
    parser.add_argument("--max-validation-drawdown-grid", default="4,6")
    parser.add_argument("--segment-selection-penalty-grid", default="0.5,1.0")
    parser.add_argument("--stress-selection-weight-grid", default="0,0.5")
    parser.add_argument("--max-configs", type=int, default=None)
    parser.add_argument("--metrics-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = Path("outputs") / f"sweep_alpha_policy_constraints_{stamp}"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.metrics_csv is None:
        metrics, windows = precompute_candidate_metrics(args)
        metrics = normalize_metric_columns(metrics)
        metrics.to_csv(args.output_dir / "candidate_metrics.csv", index=False)
        metrics_source = str(args.output_dir / "candidate_metrics.csv")
    else:
        metrics = normalize_metric_columns(pd.read_csv(args.metrics_csv))
        windows = build_windows_for_args(args)
        metrics_source = str(args.metrics_csv)

    mode_sets = parse_pipe_groups(args.mode_sets, parse_string_items)
    alpha_kind_sets = parse_pipe_groups(args.alpha_kind_sets, parse_string_items)
    alpha_lag_sets = parse_pipe_groups(args.alpha_lag_sets, parse_ints)
    k_sets = parse_pipe_groups(args.k_sets, parse_ints)
    gross_sets = parse_pipe_groups(args.gross_sets, parse_floats)
    rebalance_sets = parse_pipe_groups(args.rebalance_sets, parse_ints)
    min_return_grid = parse_float_grid(args.min_validation_return_grid)
    min_stress_grid = parse_float_grid(args.min_validation_stress_return_grid)
    min_segment_grid = parse_float_grid(args.min_validation_segment_return_grid)
    max_drawdown_grid = parse_float_grid(args.max_validation_drawdown_grid)
    segment_penalty_grid = [value for value in parse_float_grid(args.segment_selection_penalty_grid) if value is not None]
    stress_weight_grid = [value for value in parse_float_grid(args.stress_selection_weight_grid) if value is not None]

    summaries = []
    best = None
    config_iter = itertools.product(
        mode_sets,
        alpha_kind_sets,
        alpha_lag_sets,
        k_sets,
        gross_sets,
        rebalance_sets,
        min_return_grid,
        min_stress_grid,
        min_segment_grid,
        max_drawdown_grid,
        segment_penalty_grid,
        stress_weight_grid,
    )
    for config_id, values in enumerate(config_iter, start=1):
        if args.max_configs is not None and config_id > args.max_configs:
            break
        (
            mode_set,
            alpha_kind_set,
            alpha_lag_set,
            k_set,
            gross_set,
            rebalance_set,
            min_return,
            min_stress,
            min_segment,
            max_drawdown,
            segment_penalty,
            stress_weight,
        ) = values
        summary, window_results, selected_rows = evaluate_config(
            metrics,
            windows,
            config_id=config_id,
            mode_set=mode_set,
            alpha_kind_set=alpha_kind_set,
            alpha_lag_set=alpha_lag_set,
            k_set=k_set,
            gross_set=gross_set,
            rebalance_set=rebalance_set,
            min_validation_return_pct=min_return,
            min_validation_stress_return_pct=min_stress,
            min_validation_segment_return_pct=min_segment,
            max_validation_drawdown_pct=max_drawdown,
            segment_selection_penalty=segment_penalty,
            stress_selection_weight=stress_weight,
            initial_cash=args.initial_cash,
            target_return_pct=args.target_return_pct,
        )
        summaries.append(summary)
        if best is None or summary["return_pct"] > best[0]["return_pct"]:
            best = (summary, window_results, selected_rows)
        if config_id % 500 == 0:
            print(f"evaluated configs={config_id} best_return={best[0]['return_pct']:.2f}%", flush=True)

    sweep = pd.DataFrame(summaries).sort_values(
        ["return_pct", "stress_return_pct", "trade_window_count"],
        ascending=[False, False, False],
    )
    sweep.to_csv(args.output_dir / "sweep_results.csv", index=False)
    if best is None:
        raise RuntimeError("No configs evaluated.")

    best_summary, best_window_results, best_selected_rows = best
    best_window_results.to_csv(args.output_dir / "best_window_results.csv", index=False)
    best_selected_rows.to_csv(args.output_dir / "best_selected_candidates.csv", index=False)
    (args.output_dir / "best_config.json").write_text(
        json.dumps(best_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    summary_payload = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "protocol": (
            "Candidate validation/test paths are precomputed once under the same walk-forward windows. "
            "This sweep searches meta-constraints on the full requested history, so the best config is "
            "a research candidate and still needs a future or separately held-out forward test."
        ),
        "metrics_source": metrics_source,
        "input_config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if key != "output_dir"
        },
        "config_count": int(len(sweep)),
        "best": best_summary,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    top = sweep.head(10)
    readme = [
        "# Alpha Policy Constraint Sweep",
        "",
        "Search over policy-selection constraints using precomputed walk-forward candidate paths.",
        "",
        "## Best Result",
        "",
        f"- Config id: {best_summary['config_id']}",
        f"- Return: {best_summary['return_pct']:.2f}%",
        f"- Stress return: {best_summary['stress_return_pct']:.2f}%",
        f"- Trade windows: {best_summary['trade_window_count']} / {best_summary['window_count']}",
        f"- Final equity: {best_summary['final_equity']:.2f}",
        f"- Target hit ({args.target_return_pct:.1f}%): {best_summary['target_hit']}",
        "",
        "## Top 10",
        "",
        dataframe_to_markdown(
            top[
                [
                    "config_id",
                    "return_pct",
                    "stress_return_pct",
                    "trade_window_count",
                    "mode_set",
                    "alpha_kind_set",
                    "alpha_lag_set",
                    "min_validation_return_pct",
                    "min_validation_stress_return_pct",
                    "min_validation_segment_return_pct",
                    "max_validation_drawdown_pct",
                ]
            ]
        ),
        "",
        "This is a research sweep, not financial advice or a production trading recommendation.",
        "",
    ]
    (args.output_dir / "README.md").write_text("\n".join(readme), encoding="utf-8")

    print(f"Output dir: {args.output_dir}")
    print(f"Configs evaluated: {len(sweep)}")
    print(f"Best return: {best_summary['return_pct']:.2f}%")
    print(f"Best stress return: {best_summary['stress_return_pct']:.2f}%")
    print(f"Best target hit: {best_summary['target_hit']}")


if __name__ == "__main__":
    main()
