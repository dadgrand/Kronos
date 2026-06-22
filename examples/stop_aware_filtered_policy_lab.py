import argparse
import datetime as dt
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from filtered_neural_policy_lab import compute_market_features, finite_quantiles, short_confidence
from neural_policy_lab import build_matrices, read_intraday_matrix, split_masks


@dataclass(frozen=True)
class StopAwarePolicy:
    mode: str
    k: int
    gross: float
    rebalance_every: int
    confidence_min: float
    market_mom_24_max: float
    market_mom_96_max: float
    stop_loss_pct: float | None
    take_profit_pct: float | None
    filter_fail_exit_bars: int | None


def parse_optional_floats(value: str) -> list[float | None]:
    parsed: list[float | None] = []
    for item in value.split(","):
        item = item.strip().lower()
        if item in {"", "none", "null", "off"}:
            parsed.append(None)
        else:
            parsed.append(float(item))
    return parsed


def parse_optional_ints(value: str) -> list[int | None]:
    parsed: list[int | None] = []
    for item in value.split(","):
        item = item.strip().lower()
        if item in {"", "none", "null", "off"}:
            parsed.append(None)
        else:
            parsed.append(int(item))
    return parsed


def parse_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def max_drawdown_pct(equity: np.ndarray) -> float:
    if equity.size == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    return float(((equity / peak) - 1.0).min() * 100.0)


def build_policy_grid(
    *,
    mode: str,
    k: int,
    max_gross: float,
    rebalance_grid: list[int],
    confidence_grid: list[float],
    mom24_grid: list[float],
    mom96_grid: list[float],
    stop_loss_grid: list[float | None],
    take_profit_grid: list[float | None],
    filter_fail_grid: list[int | None],
) -> list[StopAwarePolicy]:
    gross_grid = [gross for gross in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0] if gross <= max_gross]
    policies = []
    for gross in gross_grid:
        for rebalance_every in rebalance_grid:
            for confidence_min in confidence_grid:
                for market_mom_24_max in mom24_grid:
                    for market_mom_96_max in mom96_grid:
                        for stop_loss_pct in stop_loss_grid:
                            for take_profit_pct in take_profit_grid:
                                for filter_fail_exit_bars in filter_fail_grid:
                                    policies.append(
                                        StopAwarePolicy(
                                            mode=mode,
                                            k=k,
                                            gross=gross,
                                            rebalance_every=rebalance_every,
                                            confidence_min=confidence_min,
                                            market_mom_24_max=market_mom_24_max,
                                            market_mom_96_max=market_mom_96_max,
                                            stop_loss_pct=stop_loss_pct,
                                            take_profit_pct=take_profit_pct,
                                            filter_fail_exit_bars=filter_fail_exit_bars,
                                        )
                                    )
    return policies


def select_target(
    scores: np.ndarray,
    tradable: np.ndarray,
    row: int,
    policy: StopAwarePolicy,
) -> np.ndarray:
    target = np.zeros(scores.shape[1], dtype=np.float32)
    valid = np.isfinite(scores[row]) & tradable[row]
    if valid.sum() < policy.k:
        return target

    candidates = np.flatnonzero(valid)
    order = candidates[np.argsort(scores[row, candidates])]
    if policy.mode == "short_only":
        selected = order[: policy.k]
        target[selected] = -policy.gross / policy.k
    elif policy.mode == "long_only":
        selected = order[-policy.k :]
        target[selected] = policy.gross / policy.k
    else:
        raise ValueError(f"unsupported mode: {policy.mode}")
    return target


def backtest_stop_aware(
    scores: np.ndarray,
    matrices: dict,
    masks: dict[str, np.ndarray],
    market_features: dict[str, np.ndarray],
    confidence: np.ndarray,
    split: str,
    policy: StopAwarePolicy,
    *,
    initial_cash: float,
    cost_bps: float,
) -> tuple[dict, pd.DataFrame]:
    bar_return = matrices["bar_return"]
    tradable = matrices["tradable"]
    rows = np.flatnonzero(masks[split])
    weights = np.zeros(scores.shape[1], dtype=np.float32)
    equity = float(initial_cash)
    entry_equity: float | None = None
    filter_fail_count = 0
    cost_rate = cost_bps / 10000.0
    records = []

    for local_idx, row in enumerate(rows):
        action = "hold"
        regime_ok = (
            confidence[row] >= policy.confidence_min
            and market_features["market_mom_24"][row] <= policy.market_mom_24_max
            and market_features["market_mom_96"][row] <= policy.market_mom_96_max
        )
        filter_fail_count = 0 if regime_ok else filter_fail_count + 1

        turnover = 0.0
        if local_idx % policy.rebalance_every == 0:
            target = np.zeros_like(weights)
            if regime_ok:
                target = select_target(scores, tradable, row, policy)
            turnover = float(np.abs(target - weights).sum())
            previous_gross = float(np.abs(weights).sum())
            target_gross = float(np.abs(target).sum())
            weights = target
            if turnover > 0:
                action = "rebalance"
            if target_gross == 0.0:
                entry_equity = None
            elif previous_gross == 0.0 or turnover > 0:
                entry_equity = equity * max(0.0, 1.0 - turnover * cost_rate)

        trading_cost_return = turnover * cost_rate
        gross_bar_return = float(np.nan_to_num(bar_return[row], nan=0.0) @ weights)
        equity *= max(0.0, 1.0 + gross_bar_return - trading_cost_return)

        exit_turnover = 0.0
        gross_exposure = float(np.abs(weights).sum())
        ending_gross_exposure = gross_exposure
        if gross_exposure > 0.0 and entry_equity is not None:
            position_return_pct = (equity / entry_equity - 1.0) * 100.0
            stop_hit = policy.stop_loss_pct is not None and position_return_pct <= -abs(policy.stop_loss_pct)
            take_hit = policy.take_profit_pct is not None and position_return_pct >= abs(policy.take_profit_pct)
            filter_hit = (
                policy.filter_fail_exit_bars is not None
                and filter_fail_count >= max(1, policy.filter_fail_exit_bars)
            )
            if stop_hit or take_hit or filter_hit:
                exit_turnover = gross_exposure
                equity *= max(0.0, 1.0 - exit_turnover * cost_rate)
                weights = np.zeros_like(weights)
                entry_equity = None
                ending_gross_exposure = 0.0
                if stop_hit:
                    action = "close_stop_loss"
                elif take_hit:
                    action = "close_take_profit"
                else:
                    action = "close_filter_fail"

        total_turnover = turnover + exit_turnover
        net_bar_return = gross_bar_return - (turnover + exit_turnover) * cost_rate
        records.append(
            {
                "row": int(row),
                "equity": equity,
                "bar_return": net_bar_return,
                "gross_bar_return": gross_bar_return,
                "turnover": total_turnover,
                "gross_exposure": gross_exposure,
                "ending_gross_exposure": ending_gross_exposure,
                "action": action,
                "confidence": float(confidence[row]),
                "filter_fail_count": int(filter_fail_count),
                "market_mom_24": float(market_features["market_mom_24"][row]),
                "market_mom_96": float(market_features["market_mom_96"][row]),
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
            "active_rate": 0.0,
            "avg_turnover_per_bar": 0.0,
            "total_turnover": 0.0,
            "mean_gross_exposure": 0.0,
            "stop_exits": 0,
            "take_profit_exits": 0,
            "filter_exits": 0,
        }
        return summary, bars

    equity_curve = bars["equity"].to_numpy(dtype=np.float64)
    returns = bars["bar_return"].to_numpy(dtype=np.float64)
    summary = {
        "split": split,
        "initial_cash": initial_cash,
        "final_equity": float(equity_curve[-1]),
        "pnl": float(equity_curve[-1] - initial_cash),
        "return_pct": float((equity_curve[-1] / initial_cash - 1.0) * 100.0),
        "max_drawdown_pct": max_drawdown_pct(equity_curve),
        "sharpe_like": float(returns.mean() / (returns.std(ddof=0) + 1e-12) * np.sqrt(252 * 50)),
        "bars": int(len(bars)),
        "active_rate": float((bars["gross_exposure"] > 0).mean()),
        "avg_turnover_per_bar": float(bars["turnover"].mean()),
        "total_turnover": float(bars["turnover"].sum()),
        "mean_gross_exposure": float(bars["gross_exposure"].mean()),
        "stop_exits": int((bars["action"] == "close_stop_loss").sum()),
        "take_profit_exits": int((bars["action"] == "close_take_profit").sum()),
        "filter_exits": int((bars["action"] == "close_filter_fail").sum()),
    }
    return summary, bars


def selection_score(summary: dict, drawdown_penalty: float, turnover_penalty: float) -> float:
    return (
        float(summary["return_pct"])
        + drawdown_penalty * float(summary["max_drawdown_pct"])
        - turnover_penalty * float(summary["avg_turnover_per_bar"])
    )


def robust_selection_score(
    summary: dict,
    *,
    drawdown_penalty: float,
    turnover_penalty: float,
    segment_penalty: float,
    worst_segment_return_pct: float | None,
) -> float:
    base_score = selection_score(summary, drawdown_penalty, turnover_penalty)
    if worst_segment_return_pct is None:
        return base_score
    return base_score + segment_penalty * float(worst_segment_return_pct)


def segment_diagnostics(
    bars: pd.DataFrame,
    *,
    split: str,
    initial_cash: float,
    segment_count: int,
) -> pd.DataFrame:
    if bars.empty or segment_count <= 0:
        return pd.DataFrame()

    rows = []
    previous_equity = float(initial_cash)
    indices = np.array_split(np.arange(len(bars)), segment_count)
    for segment_id, index_part in enumerate(indices, start=1):
        if index_part.size == 0:
            continue
        segment = bars.iloc[index_part]
        start_equity = previous_equity
        end_equity = float(segment["equity"].iloc[-1])
        equity_curve = np.r_[start_equity, segment["equity"].to_numpy(dtype=np.float64)]
        peak = np.maximum.accumulate(equity_curve)
        rows.append(
            {
                "split": split,
                "segment": segment_id,
                "start_row": int(segment["row"].iloc[0]),
                "end_row": int(segment["row"].iloc[-1]),
                "bars": int(len(segment)),
                "start_equity": start_equity,
                "end_equity": end_equity,
                "return_pct": float((end_equity / start_equity - 1.0) * 100.0),
                "max_drawdown_pct": float(((equity_curve / peak) - 1.0).min() * 100.0),
                "active_rate": float((segment["gross_exposure"] > 0.0).mean()),
                "avg_turnover_per_bar": float(segment["turnover"].mean()),
                "mean_gross_exposure": float(segment["gross_exposure"].mean()),
                "rebalance_count": int((segment["action"] == "rebalance").sum()),
                "stop_exits": int((segment["action"] == "close_stop_loss").sum()),
                "take_profit_exits": int((segment["action"] == "close_take_profit").sum()),
                "filter_exits": int((segment["action"] == "close_filter_fail").sum()),
            }
        )
        previous_equity = end_equity
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stop-aware validation-selected MOEX policy lab.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("datasets/moex_intraday_6f8f836cc811"))
    parser.add_argument("--scores-path", type=Path, nargs="+", required=True)
    parser.add_argument("--interval", type=int, default=10)
    parser.add_argument("--train-end", default="2026-06-01")
    parser.add_argument("--validation-end", default="2026-06-13")
    parser.add_argument("--mode", choices=["short_only", "long_only"], default="short_only")
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--max-gross", type=float, default=1.5)
    parser.add_argument("--rebalance-grid", default="24")
    parser.add_argument("--stop-loss-grid", default="none")
    parser.add_argument("--take-profit-grid", default="none")
    parser.add_argument("--filter-fail-exit-grid", default="none,8,13")
    parser.add_argument("--initial-cash", type=float, default=10000.0)
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--drawdown-penalty", type=float, default=0.35)
    parser.add_argument("--turnover-penalty", type=float, default=2.0)
    parser.add_argument("--segment-selection-penalty", type=float, default=0.0)
    parser.add_argument("--min-validation-segment-return-pct", type=float, default=None)
    parser.add_argument("--target-return-pct", type=float, default=10.0)
    parser.add_argument("--segment-count", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    score_parts = [np.load(path) for path in args.scores_path]
    first_shape = score_parts[0].shape
    if any(part.shape != first_shape for part in score_parts):
        shapes = [part.shape for part in score_parts]
        raise ValueError(f"all score arrays must have the same shape, got {shapes}")
    scores = np.mean(score_parts, axis=0).astype(np.float32)

    index, symbols, open_px, close_px, volume = read_intraday_matrix(args.dataset_dir, args.interval)
    masks = split_masks(index, args.train_end, args.validation_end)
    matrices = build_matrices(open_px, close_px, volume, horizon=12)
    if scores.shape != matrices["bar_return"].shape:
        raise ValueError(f"scores shape {scores.shape} does not match market matrix {matrices['bar_return'].shape}")

    market_features = compute_market_features(index, symbols, matrices["bar_return"])
    confidence = short_confidence(scores, matrices["tradable"])
    validation_mask = masks["validation"]
    confidence_grid = finite_quantiles(confidence[validation_mask], [0.0, 0.50, 0.70, 0.80, 0.90])
    mom24_grid = finite_quantiles(market_features["market_mom_24"][validation_mask], [0.40, 0.60, 0.80, 1.0])
    mom96_grid = finite_quantiles(market_features["market_mom_96"][validation_mask], [0.40, 0.60, 0.80, 1.0])
    mom24_grid.append(999.0)
    mom96_grid.append(999.0)

    policies = build_policy_grid(
        mode=args.mode,
        k=args.k,
        max_gross=args.max_gross,
        rebalance_grid=parse_ints(args.rebalance_grid),
        confidence_grid=confidence_grid,
        mom24_grid=mom24_grid,
        mom96_grid=mom96_grid,
        stop_loss_grid=parse_optional_floats(args.stop_loss_grid),
        take_profit_grid=parse_optional_floats(args.take_profit_grid),
        filter_fail_grid=parse_optional_ints(args.filter_fail_exit_grid),
    )

    rows = []
    for policy in policies:
        summary, validation_bars_for_policy = backtest_stop_aware(
            scores,
            matrices,
            masks,
            market_features,
            confidence,
            "validation",
            policy,
            initial_cash=args.initial_cash,
            cost_bps=args.cost_bps,
        )
        validation_segments_for_policy = segment_diagnostics(
            validation_bars_for_policy,
            split="validation",
            initial_cash=args.initial_cash,
            segment_count=args.segment_count,
        )
        worst_segment_return_pct = (
            float(validation_segments_for_policy["return_pct"].min())
            if not validation_segments_for_policy.empty
            else None
        )
        min_segment_ok = (
            args.min_validation_segment_return_pct is None
            or (
                worst_segment_return_pct is not None
                and worst_segment_return_pct >= args.min_validation_segment_return_pct
            )
        )
        raw_selection_score = selection_score(summary, args.drawdown_penalty, args.turnover_penalty)
        selected_score = (
            robust_selection_score(
                summary,
                drawdown_penalty=args.drawdown_penalty,
                turnover_penalty=args.turnover_penalty,
                segment_penalty=args.segment_selection_penalty,
                worst_segment_return_pct=worst_segment_return_pct,
            )
            if min_segment_ok
            else -np.inf
        )
        rows.append(
            {
                **asdict(policy),
                **{f"validation_{key}": value for key, value in summary.items() if key != "split"},
                "raw_selection_score": raw_selection_score,
                "worst_validation_segment_return_pct": worst_segment_return_pct,
                "min_validation_segment_ok": min_segment_ok,
                "selection_score": selected_score,
            }
        )

    search = pd.DataFrame(rows).sort_values(["selection_score", "validation_return_pct"], ascending=False)
    if not np.isfinite(float(search.iloc[0]["selection_score"])):
        raise ValueError("No policy satisfied the validation segment constraints.")
    selected = StopAwarePolicy(
        mode=str(search.iloc[0]["mode"]),
        k=int(search.iloc[0]["k"]),
        gross=float(search.iloc[0]["gross"]),
        rebalance_every=int(search.iloc[0]["rebalance_every"]),
        confidence_min=float(search.iloc[0]["confidence_min"]),
        market_mom_24_max=float(search.iloc[0]["market_mom_24_max"]),
        market_mom_96_max=float(search.iloc[0]["market_mom_96_max"]),
        stop_loss_pct=None if pd.isna(search.iloc[0]["stop_loss_pct"]) else float(search.iloc[0]["stop_loss_pct"]),
        take_profit_pct=None if pd.isna(search.iloc[0]["take_profit_pct"]) else float(search.iloc[0]["take_profit_pct"]),
        filter_fail_exit_bars=(
            None if pd.isna(search.iloc[0]["filter_fail_exit_bars"]) else int(search.iloc[0]["filter_fail_exit_bars"])
        ),
    )

    validation_summary, validation_bars = backtest_stop_aware(
        scores,
        matrices,
        masks,
        market_features,
        confidence,
        "validation",
        selected,
        initial_cash=args.initial_cash,
        cost_bps=args.cost_bps,
    )
    test_summary, test_bars = backtest_stop_aware(
        scores,
        matrices,
        masks,
        market_features,
        confidence,
        "test",
        selected,
        initial_cash=args.initial_cash,
        cost_bps=args.cost_bps,
    )
    sensitivity = []
    for multiplier in [0.5, 1.0, 2.0, 3.0]:
        cost = args.cost_bps * multiplier
        summary, _ = backtest_stop_aware(
            scores,
            matrices,
            masks,
            market_features,
            confidence,
            "test",
            selected,
            initial_cash=args.initial_cash,
            cost_bps=cost,
        )
        sensitivity.append({"cost_bps": cost, **summary})

    if args.output_dir is None:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = Path("outputs") / f"stop_aware_filtered_policy_{stamp}"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    search.to_csv(args.output_dir / "validation_search.csv", index=False)
    search.head(50).to_csv(args.output_dir / "top_validation_policies.csv", index=False)
    pd.DataFrame([validation_summary]).to_csv(args.output_dir / "selected_validation_summary.csv", index=False)
    pd.DataFrame([test_summary]).to_csv(args.output_dir / "selected_test_summary.csv", index=False)
    validation_bars.to_csv(args.output_dir / "selected_validation_bars.csv", index=False)
    test_bars.to_csv(args.output_dir / "selected_test_bars.csv", index=False)
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
    validation_segments.to_csv(args.output_dir / "selected_validation_segments.csv", index=False)
    test_segments.to_csv(args.output_dir / "selected_test_segments.csv", index=False)
    pd.DataFrame(sensitivity).to_csv(args.output_dir / "cost_sensitivity.csv", index=False)
    (args.output_dir / "selected_policy.json").write_text(
        json.dumps(asdict(selected), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    metadata = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "scores_path": [str(path) for path in args.scores_path],
        "dataset_dir": str(args.dataset_dir),
        "train_end": args.train_end,
        "validation_end": args.validation_end,
        "policy_count": len(policies),
        "initial_cash": args.initial_cash,
        "cost_bps": args.cost_bps,
        "drawdown_penalty": args.drawdown_penalty,
        "turnover_penalty": args.turnover_penalty,
        "segment_selection_penalty": args.segment_selection_penalty,
        "min_validation_segment_return_pct": args.min_validation_segment_return_pct,
        "target_return_pct": args.target_return_pct,
        "segment_count": args.segment_count,
        "selected_policy": asdict(selected),
        "validation_summary": validation_summary,
        "test_summary": test_summary,
        "validation_segments": validation_segments.to_dict(orient="records"),
        "test_segments": test_segments.to_dict(orient="records"),
        "worst_validation_segment_return_pct": (
            float(validation_segments["return_pct"].min()) if not validation_segments.empty else None
        ),
        "worst_test_segment_return_pct": float(test_segments["return_pct"].min()) if not test_segments.empty else None,
        "cost_sensitivity": sensitivity,
        "target_hit": test_summary["return_pct"] >= args.target_return_pct,
        "protocol": (
            "Policy and execution parameters are selected on validation only. "
            "The selected policy is evaluated on the 10 bps test split as the primary result; "
            "cost sensitivity is a post-selection diagnostic, not a selection input."
        ),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    report = [
        "# Stop-Aware Filtered Policy Lab",
        "",
        "Validation-selected execution-aware policy over the frozen neural score ensemble.",
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
        "## Segment Diagnostics",
        "",
        f"- Worst validation segment return: {validation_segments['return_pct'].min():.2f}%",
        f"- Worst test segment return: {test_segments['return_pct'].min():.2f}%",
        "",
        "Cost sensitivity is computed after policy selection and is not used to choose the policy.",
        "",
        "This is a research backtest, not financial advice or a live trading recommendation.",
        "",
    ]
    (args.output_dir / "README.md").write_text("\n".join(report), encoding="utf-8")

    print(f"Output dir: {args.output_dir}")
    print(f"Policies evaluated: {len(policies)}")
    print(f"Selected policy: {asdict(selected)}")
    print(f"Validation return: {validation_summary['return_pct']:.2f}%")
    print(f"Test return: {test_summary['return_pct']:.2f}%")
    print(f"Target hit: {test_summary['return_pct'] >= args.target_return_pct}")


if __name__ == "__main__":
    main()
