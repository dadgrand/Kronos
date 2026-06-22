import argparse
import datetime as dt
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from filtered_neural_policy_lab import compute_market_features, finite_quantiles
from neural_policy_lab import build_matrices, read_intraday_matrix
from stop_aware_filtered_policy_lab import (
    StopAwarePolicy,
    backtest_stop_aware,
    build_policy_grid,
    parse_floats,
    parse_ints,
    parse_optional_floats,
    parse_optional_ints,
    robust_selection_score,
    score_confidence,
    segment_diagnostics,
    selection_score,
)


@dataclass(frozen=True)
class WalkForwardWindow:
    window_id: int
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def load_score_ensemble(paths: list[Path]) -> np.ndarray:
    score_parts = [np.load(path) for path in paths]
    first_shape = score_parts[0].shape
    if any(part.shape != first_shape for part in score_parts):
        shapes = [part.shape for part in score_parts]
        raise ValueError(f"all score arrays must have the same shape, got {shapes}")
    return np.mean(score_parts, axis=0).astype(np.float32)


def timestamp_mask(index: pd.DatetimeIndex, start: pd.Timestamp, end: pd.Timestamp) -> np.ndarray:
    return np.asarray((index >= start) & (index < end))


def parse_modes(value: str) -> list[str]:
    modes = [item.strip() for item in value.split(",") if item.strip()]
    allowed = {"short_only", "long_only", "long_short"}
    unknown = sorted(set(modes) - allowed)
    if unknown:
        raise ValueError(f"unknown mode(s): {unknown}; allowed={sorted(allowed)}")
    if not modes:
        raise ValueError("at least one mode is required")
    return modes


def make_windows(
    index: pd.DatetimeIndex,
    *,
    start_date: str,
    end_date: str | None,
    validation_days: int,
    test_days: int,
    step_days: int,
    max_windows: int | None,
) -> list[WalkForwardWindow]:
    data_start = pd.Timestamp(index.min())
    data_end = pd.Timestamp(index.max()) + pd.Timedelta(minutes=1)
    current = max(pd.Timestamp(start_date), data_start)
    final_end = min(pd.Timestamp(end_date), data_end) if end_date else data_end
    windows = []
    window_id = 1
    while True:
        validation_start = current
        validation_end = validation_start + pd.Timedelta(days=validation_days)
        test_start = validation_end
        test_end = test_start + pd.Timedelta(days=test_days)
        if test_end > final_end:
            break
        windows.append(
            WalkForwardWindow(
                window_id=window_id,
                validation_start=validation_start,
                validation_end=validation_end,
                test_start=test_start,
                test_end=test_end,
            )
        )
        if max_windows is not None and len(windows) >= max_windows:
            break
        current += pd.Timedelta(days=step_days)
        window_id += 1
    return windows


def split_masks_for_window(index: pd.DatetimeIndex, window: WalkForwardWindow) -> dict[str, np.ndarray]:
    return {
        "validation": timestamp_mask(index, window.validation_start, window.validation_end),
        "test": timestamp_mask(index, window.test_start, window.test_end),
    }


def build_window_policy_grid(
    *,
    scores: np.ndarray,
    matrices: dict,
    market_features: dict[str, np.ndarray],
    confidence: np.ndarray,
    validation_mask: np.ndarray,
    mode: str,
    k: int,
    max_gross: float,
    rebalance_grid: list[int],
    stop_loss_grid: list[float | None],
    take_profit_grid: list[float | None],
    filter_fail_grid: list[int | None],
    confidence_quantiles: list[float],
    momentum_quantiles: list[float],
) -> list[StopAwarePolicy]:
    confidence_grid = finite_quantiles(confidence[validation_mask], confidence_quantiles)
    mom24_grid = finite_quantiles(market_features["market_mom_24"][validation_mask], momentum_quantiles)
    mom96_grid = finite_quantiles(market_features["market_mom_96"][validation_mask], momentum_quantiles)
    mom24_grid.append(999.0)
    mom96_grid.append(999.0)
    return build_policy_grid(
        mode=mode,
        k=k,
        max_gross=max_gross,
        rebalance_grid=rebalance_grid,
        confidence_grid=confidence_grid,
        mom24_grid=mom24_grid,
        mom96_grid=mom96_grid,
        stop_loss_grid=stop_loss_grid,
        take_profit_grid=take_profit_grid,
        filter_fail_grid=filter_fail_grid,
    )


def policy_from_row(row: pd.Series) -> StopAwarePolicy:
    return StopAwarePolicy(
        mode=str(row["mode"]),
        k=int(row["k"]),
        gross=float(row["gross"]),
        rebalance_every=int(row["rebalance_every"]),
        confidence_min=float(row["confidence_min"]),
        market_mom_24_max=float(row["market_mom_24_max"]),
        market_mom_96_max=float(row["market_mom_96_max"]),
        stop_loss_pct=None if pd.isna(row["stop_loss_pct"]) else float(row["stop_loss_pct"]),
        take_profit_pct=None if pd.isna(row["take_profit_pct"]) else float(row["take_profit_pct"]),
        filter_fail_exit_bars=None
        if pd.isna(row["filter_fail_exit_bars"])
        else int(row["filter_fail_exit_bars"]),
    )


def evaluate_window(
    *,
    scores: np.ndarray,
    matrices: dict,
    masks: dict[str, np.ndarray],
    market_features: dict[str, np.ndarray],
    confidence_by_mode: dict[str, np.ndarray],
    policies: list[StopAwarePolicy],
    initial_cash: float,
    cost_bps: float,
    stress_cost_bps: float,
    drawdown_penalty: float,
    turnover_penalty: float,
    segment_selection_penalty: float,
    stress_selection_weight: float,
    min_validation_return_pct: float | None,
    min_validation_stress_return_pct: float | None,
    min_validation_segment_return_pct: float | None,
    max_validation_drawdown_pct: float | None,
    segment_count: int,
) -> tuple[pd.DataFrame, StopAwarePolicy, dict, dict, pd.DataFrame, pd.DataFrame]:
    rows = []
    for policy in policies:
        confidence = confidence_by_mode[policy.mode]
        validation_summary, validation_bars = backtest_stop_aware(
            scores,
            matrices,
            masks,
            market_features,
            confidence,
            "validation",
            policy,
            initial_cash=initial_cash,
            cost_bps=cost_bps,
        )
        validation_stress_summary, _ = backtest_stop_aware(
            scores,
            matrices,
            masks,
            market_features,
            confidence,
            "validation",
            policy,
            initial_cash=initial_cash,
            cost_bps=stress_cost_bps,
        )
        validation_segments = segment_diagnostics(
            validation_bars,
            split="validation",
            initial_cash=initial_cash,
            segment_count=segment_count,
        )
        worst_segment = (
            float(validation_segments["return_pct"].min()) if not validation_segments.empty else None
        )
        raw_score = selection_score(validation_summary, drawdown_penalty, turnover_penalty)
        robust_score = robust_selection_score(
            validation_summary,
            drawdown_penalty=drawdown_penalty,
            turnover_penalty=turnover_penalty,
            segment_penalty=segment_selection_penalty,
            worst_segment_return_pct=worst_segment,
        )
        selected_score = robust_score + stress_selection_weight * float(validation_stress_summary["return_pct"])
        constraints_ok = True
        if min_validation_return_pct is not None:
            constraints_ok &= float(validation_summary["return_pct"]) >= min_validation_return_pct
        if min_validation_stress_return_pct is not None:
            constraints_ok &= float(validation_stress_summary["return_pct"]) >= min_validation_stress_return_pct
        if min_validation_segment_return_pct is not None:
            constraints_ok &= worst_segment is not None and worst_segment >= min_validation_segment_return_pct
        if max_validation_drawdown_pct is not None:
            constraints_ok &= float(validation_summary["max_drawdown_pct"]) >= -abs(max_validation_drawdown_pct)
        if not constraints_ok:
            selected_score = -np.inf
        rows.append(
            {
                **asdict(policy),
                **{f"validation_{key}": value for key, value in validation_summary.items() if key != "split"},
                "validation_stress_return_pct": validation_stress_summary["return_pct"],
                "validation_stress_max_drawdown_pct": validation_stress_summary["max_drawdown_pct"],
                "worst_validation_segment_return_pct": worst_segment,
                "raw_selection_score": raw_score,
                "constraints_ok": constraints_ok,
                "selection_score": selected_score,
            }
        )

    search = pd.DataFrame(rows).sort_values(["selection_score", "validation_return_pct"], ascending=False)
    if search.empty or not np.isfinite(float(search.iloc[0]["selection_score"])):
        raise ValueError("No policy satisfied the validation constraints for this window.")
    selected = policy_from_row(search.iloc[0])
    selected_confidence = confidence_by_mode[selected.mode]
    validation_summary, validation_bars = backtest_stop_aware(
        scores,
        matrices,
        masks,
        market_features,
        selected_confidence,
        "validation",
        selected,
        initial_cash=initial_cash,
        cost_bps=cost_bps,
    )
    test_summary, test_bars = backtest_stop_aware(
        scores,
        matrices,
        masks,
        market_features,
        selected_confidence,
        "test",
        selected,
        initial_cash=initial_cash,
        cost_bps=cost_bps,
    )
    return search, selected, validation_summary, test_summary, validation_bars, test_bars


def summarize_compounded(window_results: pd.DataFrame, initial_cash: float, target_return_pct: float) -> dict:
    equity = float(initial_cash)
    curve = [equity]
    for _, row in window_results.sort_values("window_id").iterrows():
        equity *= 1.0 + float(row["test_return_pct"]) / 100.0
        curve.append(equity)
    curve_arr = np.asarray(curve, dtype=np.float64)
    if curve_arr.size:
        peak = np.maximum.accumulate(curve_arr)
        max_drawdown = float(((curve_arr / peak) - 1.0).min() * 100.0)
    else:
        max_drawdown = 0.0
    return {
        "initial_cash": initial_cash,
        "final_equity": equity,
        "pnl": equity - initial_cash,
        "return_pct": (equity / initial_cash - 1.0) * 100.0,
        "max_drawdown_pct": max_drawdown,
        "windows": int(len(window_results)),
        "win_rate": float((window_results["test_return_pct"] > 0.0).mean()) if len(window_results) else 0.0,
        "mean_window_test_return_pct": float(window_results["test_return_pct"].mean()) if len(window_results) else 0.0,
        "median_window_test_return_pct": float(window_results["test_return_pct"].median()) if len(window_results) else 0.0,
        "worst_window_test_return_pct": float(window_results["test_return_pct"].min()) if len(window_results) else 0.0,
        "target_hit": bool((equity / initial_cash - 1.0) * 100.0 >= target_return_pct),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward stop-aware policy selection over frozen score files.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("datasets/moex_intraday_6f8f836cc811"))
    parser.add_argument("--scores-path", type=Path, nargs="+", required=True)
    parser.add_argument("--interval", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--start-date", default="2026-01-01")
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--validation-days", type=int, default=30)
    parser.add_argument("--test-days", type=int, default=7)
    parser.add_argument("--step-days", type=int, default=7)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--mode", choices=["short_only", "long_only", "long_short"], default=None)
    parser.add_argument("--modes", default="short_only")
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--max-gross", type=float, default=1.5)
    parser.add_argument("--rebalance-grid", default="24")
    parser.add_argument("--confidence-quantiles", default="0,0.7,0.9")
    parser.add_argument("--momentum-quantiles", default="0.6,1.0")
    parser.add_argument("--stop-loss-grid", default="none")
    parser.add_argument("--take-profit-grid", default="none")
    parser.add_argument("--filter-fail-exit-grid", default="none,8,13")
    parser.add_argument("--initial-cash", type=float, default=10000.0)
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--stress-cost-bps", type=float, default=20.0)
    parser.add_argument("--drawdown-penalty", type=float, default=0.35)
    parser.add_argument("--turnover-penalty", type=float, default=2.0)
    parser.add_argument("--segment-selection-penalty", type=float, default=0.0)
    parser.add_argument("--stress-selection-weight", type=float, default=0.0)
    parser.add_argument("--min-validation-return-pct", type=float, default=None)
    parser.add_argument("--min-validation-stress-return-pct", type=float, default=None)
    parser.add_argument("--min-validation-segment-return-pct", type=float, default=None)
    parser.add_argument("--max-validation-drawdown-pct", type=float, default=None)
    parser.add_argument("--target-return-pct", type=float, default=10.0)
    parser.add_argument("--segment-count", type=int, default=5)
    parser.add_argument(
        "--score-model-protocol",
        default=(
            "frozen_score_model: supplied scores are evaluated as-is; this script walk-forwards "
            "policy selection but does not retrain the score model per fold"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    scores = load_score_ensemble(args.scores_path)
    index, symbols, open_px, close_px, volume = read_intraday_matrix(args.dataset_dir, args.interval)
    matrices = build_matrices(open_px, close_px, volume, args.horizon)
    if scores.shape != matrices["bar_return"].shape:
        raise ValueError(f"scores shape {scores.shape} does not match market matrix {matrices['bar_return'].shape}")

    market_features = compute_market_features(index, symbols, matrices["bar_return"])
    modes = [args.mode] if args.mode else parse_modes(args.modes)
    confidence_by_mode = {mode: score_confidence(scores, matrices["tradable"], mode) for mode in modes}
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

    if args.output_dir is None:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = Path("outputs") / f"walk_forward_stop_aware_policy_{stamp}"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    window_rows = []
    selected_policy_rows = []
    top_rows = []
    validation_segments_all = []
    test_segments_all = []
    for window in windows:
        masks = split_masks_for_window(index, window)
        if int(masks["validation"].sum()) == 0 or int(masks["test"].sum()) == 0:
            continue
        policies = []
        for mode in modes:
            policies.extend(
                build_window_policy_grid(
                    scores=scores,
                    matrices=matrices,
                    market_features=market_features,
                    confidence=confidence_by_mode[mode],
                    validation_mask=masks["validation"],
                    mode=mode,
                    k=args.k,
                    max_gross=args.max_gross,
                    rebalance_grid=parse_ints(args.rebalance_grid),
                    stop_loss_grid=parse_optional_floats(args.stop_loss_grid),
                    take_profit_grid=parse_optional_floats(args.take_profit_grid),
                    filter_fail_grid=parse_optional_ints(args.filter_fail_exit_grid),
                    confidence_quantiles=parse_floats(args.confidence_quantiles),
                    momentum_quantiles=parse_floats(args.momentum_quantiles),
                )
            )
        search, selected, validation_summary, test_summary, validation_bars, test_bars = evaluate_window(
            scores=scores,
            matrices=matrices,
            masks=masks,
            market_features=market_features,
            confidence_by_mode=confidence_by_mode,
            policies=policies,
            initial_cash=args.initial_cash,
            cost_bps=args.cost_bps,
            stress_cost_bps=args.stress_cost_bps,
            drawdown_penalty=args.drawdown_penalty,
            turnover_penalty=args.turnover_penalty,
            segment_selection_penalty=args.segment_selection_penalty,
            stress_selection_weight=args.stress_selection_weight,
            min_validation_return_pct=args.min_validation_return_pct,
            min_validation_stress_return_pct=args.min_validation_stress_return_pct,
            min_validation_segment_return_pct=args.min_validation_segment_return_pct,
            max_validation_drawdown_pct=args.max_validation_drawdown_pct,
            segment_count=args.segment_count,
        )
        validation_segments = segment_diagnostics(
            validation_bars,
            split="validation",
            initial_cash=args.initial_cash,
            segment_count=args.segment_count,
        )
        test_segments = segment_diagnostics(
            test_bars,
            split="test",
            initial_cash=args.initial_cash,
            segment_count=args.segment_count,
        )
        for frame in [validation_segments, test_segments]:
            if not frame.empty:
                frame.insert(0, "window_id", window.window_id)
        validation_segments_all.append(validation_segments)
        test_segments_all.append(test_segments)

        top = search.head(25).copy()
        top.insert(0, "window_id", window.window_id)
        top_rows.append(top)
        selected_policy_rows.append({"window_id": window.window_id, **asdict(selected)})
        window_rows.append(
            {
                "window_id": window.window_id,
                "validation_start": window.validation_start.isoformat(),
                "validation_end": window.validation_end.isoformat(),
                "test_start": window.test_start.isoformat(),
                "test_end": window.test_end.isoformat(),
                "validation_bars": int(masks["validation"].sum()),
                "test_bars": int(masks["test"].sum()),
                "policy_count": len(policies),
                **{f"validation_{key}": value for key, value in validation_summary.items() if key != "split"},
                **{f"test_{key}": value for key, value in test_summary.items() if key != "split"},
                "worst_validation_segment_return_pct": (
                    float(validation_segments["return_pct"].min()) if not validation_segments.empty else None
                ),
                "worst_test_segment_return_pct": (
                    float(test_segments["return_pct"].min()) if not test_segments.empty else None
                ),
            }
        )
        print(
            f"window={window.window_id} validation={validation_summary['return_pct']:.2f}% "
            f"test={test_summary['return_pct']:.2f}% policy={asdict(selected)}",
            flush=True,
        )

    window_results = pd.DataFrame(window_rows)
    if window_results.empty:
        raise ValueError("No windows were evaluated.")
    selected_policies = pd.DataFrame(selected_policy_rows)
    top_search = pd.concat(top_rows, ignore_index=True) if top_rows else pd.DataFrame()
    validation_segments_out = (
        pd.concat(validation_segments_all, ignore_index=True) if validation_segments_all else pd.DataFrame()
    )
    test_segments_out = pd.concat(test_segments_all, ignore_index=True) if test_segments_all else pd.DataFrame()
    compounded = summarize_compounded(window_results, args.initial_cash, args.target_return_pct)

    window_results.to_csv(args.output_dir / "window_results.csv", index=False)
    selected_policies.to_csv(args.output_dir / "selected_policies.csv", index=False)
    top_search.to_csv(args.output_dir / "top_validation_policies_by_window.csv", index=False)
    validation_segments_out.to_csv(args.output_dir / "validation_segments.csv", index=False)
    test_segments_out.to_csv(args.output_dir / "test_segments.csv", index=False)
    summary = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "dataset_dir": str(args.dataset_dir),
        "scores_path": [str(path) for path in args.scores_path],
        "score_model_protocol": args.score_model_protocol,
        "interval": args.interval,
        "horizon": args.horizon,
        "symbols": symbols,
        "window_count": int(len(window_results)),
        "start_date": args.start_date,
        "end_date": args.end_date,
        "validation_days": args.validation_days,
        "test_days": args.test_days,
        "step_days": args.step_days,
        "cost_bps": args.cost_bps,
        "stress_cost_bps": args.stress_cost_bps,
        "target_return_pct": args.target_return_pct,
        "selection": {
            "mode": args.mode,
            "modes": modes,
            "k": args.k,
            "max_gross": args.max_gross,
            "rebalance_grid": args.rebalance_grid,
            "confidence_quantiles": args.confidence_quantiles,
            "momentum_quantiles": args.momentum_quantiles,
            "stop_loss_grid": args.stop_loss_grid,
            "take_profit_grid": args.take_profit_grid,
            "filter_fail_exit_grid": args.filter_fail_exit_grid,
            "drawdown_penalty": args.drawdown_penalty,
            "turnover_penalty": args.turnover_penalty,
            "segment_selection_penalty": args.segment_selection_penalty,
            "stress_selection_weight": args.stress_selection_weight,
            "min_validation_return_pct": args.min_validation_return_pct,
            "min_validation_stress_return_pct": args.min_validation_stress_return_pct,
            "min_validation_segment_return_pct": args.min_validation_segment_return_pct,
            "max_validation_drawdown_pct": args.max_validation_drawdown_pct,
        },
        "compounded_test_summary": compounded,
        "mean_test_max_drawdown_pct": float(window_results["test_max_drawdown_pct"].mean()),
        "worst_test_segment_return_pct": float(window_results["worst_test_segment_return_pct"].min()),
        "protocol": (
            "For each window, policy parameters are selected on that window's validation slice only. "
            "The selected policy is then evaluated on the immediately following test slice. "
            "This script does not retrain the score model per fold; see score_model_protocol."
        ),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    readme = [
        "# Walk-Forward Stop-Aware Policy Lab",
        "",
        "Rolling validation-to-test policy selection over supplied neural score files.",
        "",
        "## Protocol",
        "",
        f"- Dataset: `{args.dataset_dir}`",
        f"- Score model protocol: {args.score_model_protocol}",
        f"- Windows: {len(window_results)}",
        f"- Validation days: {args.validation_days}",
        f"- Test days: {args.test_days}",
        f"- Step days: {args.step_days}",
        f"- Cost: {args.cost_bps:.2f} bps",
        f"- Stress cost: {args.stress_cost_bps:.2f} bps",
        "",
        "## Compounded Test Result",
        "",
        f"- Initial cash: {compounded['initial_cash']:.2f}",
        f"- Final equity: {compounded['final_equity']:.2f}",
        f"- PnL: {compounded['pnl']:.2f}",
        f"- Return: {compounded['return_pct']:.2f}%",
        f"- Max drawdown across window endpoints: {compounded['max_drawdown_pct']:.2f}%",
        f"- Window win rate: {compounded['win_rate'] * 100.0:.2f}%",
        f"- Worst window return: {compounded['worst_window_test_return_pct']:.2f}%",
        f"- Target hit ({args.target_return_pct:.1f}%): {compounded['target_hit']}",
        "",
        "This is a research backtest, not financial advice or a live trading recommendation.",
        "",
    ]
    (args.output_dir / "README.md").write_text("\n".join(readme), encoding="utf-8")
    print(f"Output dir: {args.output_dir}")
    print(f"Windows evaluated: {len(window_results)}")
    print(f"Compounded test return: {compounded['return_pct']:.2f}%")
    print(f"Target hit: {compounded['target_hit']}")


if __name__ == "__main__":
    main()
