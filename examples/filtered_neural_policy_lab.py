import argparse
import datetime as dt
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from neural_policy_lab import build_matrices, read_intraday_matrix, split_masks


@dataclass(frozen=True)
class FilteredPolicy:
    mode: str
    k: int
    gross: float
    rebalance_every: int
    confidence_min: float
    market_mom_24_max: float
    market_mom_96_max: float


def compute_market_features(index: pd.DatetimeIndex, symbols: list[str], bar_return: np.ndarray) -> dict[str, np.ndarray]:
    market_return = pd.DataFrame(bar_return, index=index, columns=symbols).mean(axis=1)
    return {
        "market_mom_24": market_return.shift(1).rolling(24, min_periods=6).sum().fillna(0.0).to_numpy(),
        "market_mom_96": market_return.shift(1).rolling(96, min_periods=24).sum().fillna(0.0).to_numpy(),
    }


def short_confidence(scores: np.ndarray, active: np.ndarray) -> np.ndarray:
    confidence = np.zeros(scores.shape[0], dtype=np.float32)
    for row in range(scores.shape[0]):
        valid = np.isfinite(scores[row]) & active[row]
        if valid.any():
            values = scores[row, valid]
            confidence[row] = float(np.nanmedian(values) - np.nanmin(values))
    return confidence


def backtest_filtered(
    scores: np.ndarray,
    matrices: dict,
    masks: dict[str, np.ndarray],
    market_features: dict[str, np.ndarray],
    confidence: np.ndarray,
    split: str,
    policy: FilteredPolicy,
    *,
    initial_cash: float,
    cost_bps: float,
) -> tuple[dict, pd.DataFrame]:
    bar_return = matrices["bar_return"]
    tradable = matrices["tradable"]
    rows = np.flatnonzero(masks[split])
    weights = np.zeros(scores.shape[1], dtype=np.float32)
    equity = initial_cash
    records = []
    cost_rate = cost_bps / 10000.0

    for local_idx, row in enumerate(rows):
        if local_idx % policy.rebalance_every == 0:
            target = np.zeros_like(weights)
            valid = np.isfinite(scores[row]) & tradable[row]
            regime_ok = (
                confidence[row] >= policy.confidence_min
                and market_features["market_mom_24"][row] <= policy.market_mom_24_max
                and market_features["market_mom_96"][row] <= policy.market_mom_96_max
            )
            if regime_ok and valid.sum() >= policy.k:
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
            turnover = float(np.abs(target - weights).sum())
            weights = target
        else:
            turnover = 0.0

        gross = float(np.abs(weights).sum())
        pnl_return = float(np.nan_to_num(bar_return[row], nan=0.0) @ weights) - turnover * cost_rate
        equity *= max(0.0, 1.0 + pnl_return)
        records.append(
            {
                "row": int(row),
                "equity": equity,
                "bar_return": pnl_return,
                "turnover": turnover,
                "gross_exposure": gross,
                "confidence": float(confidence[row]),
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
        }
        return summary, bars

    eq = bars["equity"].to_numpy(dtype=np.float64)
    peak = np.maximum.accumulate(eq)
    returns = bars["bar_return"].to_numpy(dtype=np.float64)
    summary = {
        "split": split,
        "initial_cash": initial_cash,
        "final_equity": float(eq[-1]),
        "pnl": float(eq[-1] - initial_cash),
        "return_pct": float((eq[-1] / initial_cash - 1.0) * 100.0),
        "max_drawdown_pct": float(((eq / peak) - 1.0).min() * 100.0),
        "sharpe_like": float(returns.mean() / (returns.std(ddof=0) + 1e-12) * np.sqrt(252 * 50)),
        "bars": int(len(bars)),
        "active_rate": float((bars["gross_exposure"] > 0).mean()),
        "avg_turnover_per_bar": float(bars["turnover"].mean()),
        "total_turnover": float(bars["turnover"].sum()),
        "mean_gross_exposure": float(bars["gross_exposure"].mean()),
    }
    return summary, bars


def score(summary: dict, drawdown_penalty: float, turnover_penalty: float) -> float:
    return (
        float(summary["return_pct"])
        + drawdown_penalty * float(summary["max_drawdown_pct"])
        - turnover_penalty * float(summary["avg_turnover_per_bar"])
    )


def finite_quantiles(values: np.ndarray, quantiles: list[float]) -> list[float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return [0.0]
    return [float(value) for value in np.quantile(values, quantiles)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation-selected regime filter for neural MOEX policy scores.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("datasets/moex_intraday_6f8f836cc811"))
    parser.add_argument("--scores-path", type=Path, nargs="+", required=True)
    parser.add_argument("--interval", type=int, default=10)
    parser.add_argument("--train-end", default="2026-06-01")
    parser.add_argument("--validation-end", default="2026-06-13")
    parser.add_argument("--mode", choices=["short_only", "long_only"], default="short_only")
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--gross", type=float, default=4.0)
    parser.add_argument("--rebalance-every", type=int, default=24)
    parser.add_argument("--initial-cash", type=float, default=10000.0)
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--drawdown-penalty", type=float, default=0.25)
    parser.add_argument("--turnover-penalty", type=float, default=2.0)
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

    confidence_grid = finite_quantiles(confidence[validation_mask], [0.0, 0.25, 0.50, 0.70, 0.80, 0.90])
    mom24_grid = finite_quantiles(market_features["market_mom_24"][validation_mask], [0.20, 0.40, 0.60, 0.80, 1.0])
    mom96_grid = finite_quantiles(market_features["market_mom_96"][validation_mask], [0.20, 0.40, 0.60, 0.80, 1.0])
    mom24_grid.append(999.0)
    mom96_grid.append(999.0)

    rows = []
    for confidence_min in confidence_grid:
        for market_mom_24_max in mom24_grid:
            for market_mom_96_max in mom96_grid:
                policy = FilteredPolicy(
                    args.mode,
                    args.k,
                    args.gross,
                    args.rebalance_every,
                    confidence_min,
                    market_mom_24_max,
                    market_mom_96_max,
                )
                summary, _ = backtest_filtered(
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
                rows.append(
                    {
                        **asdict(policy),
                        **{f"validation_{key}": value for key, value in summary.items() if key != "split"},
                        "selection_score": score(summary, args.drawdown_penalty, args.turnover_penalty),
                    }
                )

    search = pd.DataFrame(rows).sort_values(["selection_score", "validation_return_pct"], ascending=False)
    selected = FilteredPolicy(
        mode=str(search.iloc[0]["mode"]),
        k=int(search.iloc[0]["k"]),
        gross=float(search.iloc[0]["gross"]),
        rebalance_every=int(search.iloc[0]["rebalance_every"]),
        confidence_min=float(search.iloc[0]["confidence_min"]),
        market_mom_24_max=float(search.iloc[0]["market_mom_24_max"]),
        market_mom_96_max=float(search.iloc[0]["market_mom_96_max"]),
    )
    validation_summary, validation_bars = backtest_filtered(
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
    test_summary, test_bars = backtest_filtered(
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
        summary, _ = backtest_filtered(
            scores,
            matrices,
            masks,
            market_features,
            confidence,
            "test",
            selected,
            initial_cash=args.initial_cash,
            cost_bps=args.cost_bps * multiplier,
        )
        sensitivity.append({"cost_bps": args.cost_bps * multiplier, **summary})

    if args.output_dir is None:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = Path("outputs") / f"filtered_neural_policy_{stamp}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    search.to_csv(args.output_dir / "validation_filter_search.csv", index=False)
    search.head(25).to_csv(args.output_dir / "top_validation_filters.csv", index=False)
    pd.DataFrame([validation_summary]).to_csv(args.output_dir / "selected_validation_summary.csv", index=False)
    pd.DataFrame([test_summary]).to_csv(args.output_dir / "selected_test_summary.csv", index=False)
    validation_bars.to_csv(args.output_dir / "selected_validation_bars.csv", index=False)
    test_bars.to_csv(args.output_dir / "selected_test_bars.csv", index=False)
    pd.DataFrame(sensitivity).to_csv(args.output_dir / "cost_sensitivity.csv", index=False)
    metadata = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "scores_path": [str(path) for path in args.scores_path],
        "dataset_dir": str(args.dataset_dir),
        "train_end": args.train_end,
        "validation_end": args.validation_end,
        "selected_policy": asdict(selected),
        "validation_summary": validation_summary,
        "test_summary": test_summary,
        "target_30pct_hit": test_summary["return_pct"] >= 30.0,
        "protocol": "Filter thresholds are selected on validation only. Test is evaluated after filter selection.",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    readme = [
        "# Filtered Neural Policy Lab",
        "",
        "Validation-selected confidence and market-regime filter over neural policy scores.",
        "",
        f"- Selected policy: `{asdict(selected)}`",
        f"- Validation return: {validation_summary['return_pct']:.2f}%",
        f"- Validation max drawdown: {validation_summary['max_drawdown_pct']:.2f}%",
        f"- Test return: {test_summary['return_pct']:.2f}%",
        f"- Test final equity: {test_summary['final_equity']:.2f}",
        f"- Test max drawdown: {test_summary['max_drawdown_pct']:.2f}%",
        f"- 30% target hit: {test_summary['return_pct'] >= 30.0}",
        "",
        "This is a research backtest, not financial advice.",
        "",
    ]
    (args.output_dir / "README.md").write_text("\n".join(readme), encoding="utf-8")

    print(f"Output dir: {args.output_dir}")
    print(f"Selected policy: {asdict(selected)}")
    print(f"Validation return: {validation_summary['return_pct']:.2f}%")
    print(f"Test return: {test_summary['return_pct']:.2f}%")
    print(f"30% target hit: {test_summary['return_pct'] >= 30.0}")


if __name__ == "__main__":
    main()
