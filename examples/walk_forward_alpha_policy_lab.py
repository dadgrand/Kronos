import argparse
import datetime as dt
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from filtered_neural_policy_lab import compute_market_features
from neural_policy_lab import build_matrices, read_intraday_matrix
from stop_aware_filtered_policy_lab import (
    StopAwarePolicy,
    backtest_stop_aware,
    parse_floats,
    parse_ints,
    score_confidence,
    segment_diagnostics,
    selection_score,
)
from walk_forward_stop_aware_policy_lab import make_windows, parse_modes, split_masks_for_window, summarize_compounded


@dataclass(frozen=True)
class AlphaCandidate:
    alpha_name: str
    mode: str
    k: int
    gross: float
    rebalance_every: int


CONFIG_FIELDS = [
    "dataset_dir",
    "interval",
    "horizon",
    "start_date",
    "end_date",
    "validation_days",
    "test_days",
    "step_days",
    "max_windows",
    "alpha_lags",
    "alpha_kinds",
    "alpha_normalize",
    "modes",
    "k_grid",
    "gross_grid",
    "rebalance_grid",
    "initial_cash",
    "cost_bps",
    "stress_cost_bps",
    "drawdown_penalty",
    "turnover_penalty",
    "segment_selection_penalty",
    "stress_selection_weight",
    "min_validation_return_pct",
    "min_validation_stress_return_pct",
    "min_validation_segment_return_pct",
    "max_validation_drawdown_pct",
    "segment_count",
    "target_return_pct",
]


def args_to_config(args: argparse.Namespace) -> dict:
    config = {}
    for field in CONFIG_FIELDS:
        value = getattr(args, field)
        config[field] = str(value) if isinstance(value, Path) else value
    return {
        "protocol_name": "strict_walk_forward_alpha_policy_v1",
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "config": config,
        "notes": (
            "Frozen research protocol. Command-line values explicitly different "
            "from parser defaults may override matching config fields for fresh "
            "forward runs; otherwise this config supplies the protocol."
        ),
    }


def apply_config_defaults(parser: argparse.ArgumentParser, args: argparse.Namespace, config_path: Path | None) -> None:
    if config_path is None:
        return
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    config = payload.get("config", payload)
    for field, value in config.items():
        if field not in CONFIG_FIELDS or not hasattr(args, field):
            continue
        current = getattr(args, field)
        default = parser.get_default(field)
        if current != default:
            continue
        if field == "dataset_dir" and value is not None:
            value = Path(value)
        setattr(args, field, value)


def cs_zscore(frame: pd.DataFrame) -> pd.DataFrame:
    mean = frame.mean(axis=1)
    std = frame.std(axis=1).replace(0.0, np.nan)
    return frame.sub(mean, axis=0).div(std, axis=0).replace([np.inf, -np.inf], 0.0).fillna(0.0)


def build_alpha_scores(
    close_px: pd.DataFrame,
    *,
    lags: list[int],
    kinds: list[str],
    normalize: str,
) -> dict[str, np.ndarray]:
    close_ff = close_px.ffill(limit=200)
    ret = close_ff.pct_change(fill_method=None)
    vol = ret.shift(1).rolling(48, min_periods=12).std().replace(0.0, np.nan)
    scores = {}
    for lag in lags:
        mom = close_ff.shift(1) / close_ff.shift(1 + lag) - 1.0
        raw_by_kind = {
            "mom": mom,
            "rev": -mom,
            "voladj_mom": mom / vol,
            "voladj_rev": -mom / vol,
        }
        for kind in kinds:
            if kind not in raw_by_kind:
                raise ValueError(f"unknown alpha kind={kind}; allowed={sorted(raw_by_kind)}")
            frame = raw_by_kind[kind].replace([np.inf, -np.inf], 0.0).fillna(0.0)
            if normalize == "cs_zscore":
                frame = cs_zscore(frame)
            elif normalize == "none":
                pass
            else:
                raise ValueError(f"unknown normalize={normalize}")
            scores[f"{kind}_{lag}"] = frame.to_numpy(dtype=np.float32)
    return scores


def build_candidates(
    alpha_names: list[str],
    *,
    modes: list[str],
    k_grid: list[int],
    gross_grid: list[float],
    rebalance_grid: list[int],
) -> list[AlphaCandidate]:
    candidates = []
    for alpha_name in alpha_names:
        for mode in modes:
            for k in k_grid:
                for gross in gross_grid:
                    for rebalance_every in rebalance_grid:
                        candidates.append(AlphaCandidate(alpha_name, mode, k, gross, rebalance_every))
    return candidates


def stop_policy(candidate: AlphaCandidate) -> StopAwarePolicy:
    return StopAwarePolicy(
        mode=candidate.mode,
        k=candidate.k,
        gross=candidate.gross,
        rebalance_every=candidate.rebalance_every,
        confidence_min=0.0,
        market_mom_24_max=999.0,
        market_mom_96_max=999.0,
        stop_loss_pct=None,
        take_profit_pct=None,
        filter_fail_exit_bars=None,
    )


def constraints_pass(
    validation_summary: dict,
    validation_stress_summary: dict,
    worst_segment_return_pct: float | None,
    *,
    min_validation_return_pct: float | None,
    min_validation_stress_return_pct: float | None,
    min_validation_segment_return_pct: float | None,
    max_validation_drawdown_pct: float | None,
) -> bool:
    ok = True
    if min_validation_return_pct is not None:
        ok &= float(validation_summary["return_pct"]) >= min_validation_return_pct
    if min_validation_stress_return_pct is not None:
        ok &= float(validation_stress_summary["return_pct"]) >= min_validation_stress_return_pct
    if min_validation_segment_return_pct is not None:
        ok &= worst_segment_return_pct is not None and worst_segment_return_pct >= min_validation_segment_return_pct
    if max_validation_drawdown_pct is not None:
        ok &= float(validation_summary["max_drawdown_pct"]) >= -abs(max_validation_drawdown_pct)
    return bool(ok)


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward alpha policy lab with cash fallback.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("datasets/moex_intraday_6f8f836cc811"))
    parser.add_argument("--interval", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--start-date", default="2026-01-01")
    parser.add_argument("--end-date", default="2026-06-20")
    parser.add_argument("--validation-days", type=int, default=30)
    parser.add_argument("--test-days", type=int, default=7)
    parser.add_argument("--step-days", type=int, default=7)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--alpha-lags", default="12,48,96")
    parser.add_argument("--alpha-kinds", default="mom,rev,voladj_mom,voladj_rev")
    parser.add_argument("--alpha-normalize", choices=["none", "cs_zscore"], default="none")
    parser.add_argument("--modes", default="short_only,long_only,long_short")
    parser.add_argument("--k-grid", default="1,3")
    parser.add_argument("--gross-grid", default="0.5,1.0,1.5")
    parser.add_argument("--rebalance-grid", default="24")
    parser.add_argument("--initial-cash", type=float, default=10000.0)
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--stress-cost-bps", type=float, default=20.0)
    parser.add_argument("--drawdown-penalty", type=float, default=0.35)
    parser.add_argument("--turnover-penalty", type=float, default=2.0)
    parser.add_argument("--segment-selection-penalty", type=float, default=0.5)
    parser.add_argument("--stress-selection-weight", type=float, default=0.0)
    parser.add_argument("--min-validation-return-pct", type=float, default=5.0)
    parser.add_argument("--min-validation-stress-return-pct", type=float, default=None)
    parser.add_argument("--min-validation-segment-return-pct", type=float, default=0.0)
    parser.add_argument("--max-validation-drawdown-pct", type=float, default=8.0)
    parser.add_argument("--segment-count", type=int, default=5)
    parser.add_argument("--target-return-pct", type=float, default=10.0)
    parser.add_argument("--config-json", type=Path, default=None)
    parser.add_argument("--write-config-json", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    apply_config_defaults(parser, args, args.config_json)

    if args.write_config_json is not None:
        args.write_config_json.parent.mkdir(parents=True, exist_ok=True)
        args.write_config_json.write_text(
            json.dumps(args_to_config(args), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    index, symbols, open_px, close_px, volume = read_intraday_matrix(args.dataset_dir, args.interval)
    matrices = build_matrices(open_px, close_px, volume, args.horizon)
    market_features = compute_market_features(index, symbols, matrices["bar_return"])
    alpha_scores = build_alpha_scores(
        close_px,
        lags=parse_ints(args.alpha_lags),
        kinds=[item.strip() for item in args.alpha_kinds.split(",") if item.strip()],
        normalize=args.alpha_normalize,
    )
    modes = parse_modes(args.modes)
    confidences = {
        (alpha_name, mode): score_confidence(scores, matrices["tradable"], mode)
        for alpha_name, scores in alpha_scores.items()
        for mode in modes
    }
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

    if args.output_dir is None:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = Path("outputs") / f"walk_forward_alpha_policy_{stamp}"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    search_rows = []
    window_rows = []
    selected_rows = []
    test_segments_all = []
    validation_segments_all = []
    for window in windows:
        masks = split_masks_for_window(index, window)
        best = None
        for candidate in candidates:
            scores = alpha_scores[candidate.alpha_name]
            confidence = confidences[(candidate.alpha_name, candidate.mode)]
            policy = stop_policy(candidate)
            validation_summary, validation_bars = backtest_stop_aware(
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
            validation_stress_summary, _ = backtest_stop_aware(
                scores,
                matrices,
                masks,
                market_features,
                confidence,
                "validation",
                policy,
                initial_cash=args.initial_cash,
                cost_bps=args.stress_cost_bps,
            )
            validation_segments = segment_diagnostics(
                validation_bars,
                split="validation",
                initial_cash=args.initial_cash,
                segment_count=args.segment_count,
            )
            worst_segment = (
                float(validation_segments["return_pct"].min()) if not validation_segments.empty else None
            )
            raw_score = selection_score(validation_summary, args.drawdown_penalty, args.turnover_penalty)
            candidate_score = (
                raw_score
                + args.segment_selection_penalty * float(worst_segment if worst_segment is not None else -999.0)
                + args.stress_selection_weight * float(validation_stress_summary["return_pct"])
            )
            ok = constraints_pass(
                validation_summary,
                validation_stress_summary,
                worst_segment,
                min_validation_return_pct=args.min_validation_return_pct,
                min_validation_stress_return_pct=args.min_validation_stress_return_pct,
                min_validation_segment_return_pct=args.min_validation_segment_return_pct,
                max_validation_drawdown_pct=args.max_validation_drawdown_pct,
            )
            selected_score = candidate_score if ok else -np.inf
            row = {
                "window_id": window.window_id,
                **asdict(candidate),
                **{f"validation_{key}": value for key, value in validation_summary.items() if key != "split"},
                "validation_stress_return_pct": validation_stress_summary["return_pct"],
                "validation_stress_max_drawdown_pct": validation_stress_summary["max_drawdown_pct"],
                "worst_validation_segment_return_pct": worst_segment,
                "raw_selection_score": raw_score,
                "constraints_ok": ok,
                "selection_score": selected_score,
            }
            search_rows.append(row)
            if ok and (best is None or selected_score > best["selection_score"]):
                best = {
                    **row,
                    "policy": policy,
                    "scores": scores,
                    "confidence": confidence,
                    "validation_summary": validation_summary,
                    "validation_bars": validation_bars,
                }

        if best is None:
            window_rows.append(
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
                    "test_final_equity": args.initial_cash,
                    "test_pnl": 0.0,
                    "test_max_drawdown_pct": 0.0,
                    "worst_test_segment_return_pct": 0.0,
                    "test_stress_return_pct": 0.0,
                }
            )
            selected_rows.append({"window_id": window.window_id, "selected": "cash"})
            print(f"window={window.window_id} CASH", flush=True)
            continue

        test_summary, test_bars = backtest_stop_aware(
            best["scores"],
            matrices,
            masks,
            market_features,
            best["confidence"],
            "test",
            best["policy"],
            initial_cash=args.initial_cash,
            cost_bps=args.cost_bps,
        )
        test_stress_summary, _ = backtest_stop_aware(
            best["scores"],
            matrices,
            masks,
            market_features,
            best["confidence"],
            "test",
            best["policy"],
            initial_cash=args.initial_cash,
            cost_bps=args.stress_cost_bps,
        )
        validation_segments = segment_diagnostics(
            best["validation_bars"],
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
        selected_rows.append(
            {
                "window_id": window.window_id,
                "selected": "alpha_policy",
                **{key: best[key] for key in ["alpha_name", "mode", "k", "gross", "rebalance_every"]},
            }
        )
        window_rows.append(
            {
                "window_id": window.window_id,
                "validation_start": window.validation_start.isoformat(),
                "validation_end": window.validation_end.isoformat(),
                "test_start": window.test_start.isoformat(),
                "test_end": window.test_end.isoformat(),
                "selected": "alpha_policy",
                **{key: best[key] for key in ["alpha_name", "mode", "k", "gross", "rebalance_every"]},
                **{f"validation_{key}": value for key, value in best["validation_summary"].items() if key != "split"},
                "worst_validation_segment_return_pct": (
                    float(validation_segments["return_pct"].min()) if not validation_segments.empty else None
                ),
                **{f"test_{key}": value for key, value in test_summary.items() if key != "split"},
                "worst_test_segment_return_pct": (
                    float(test_segments["return_pct"].min()) if not test_segments.empty else None
                ),
                "test_stress_return_pct": test_stress_summary["return_pct"],
            }
        )
        print(
            f"window={window.window_id} validation={best['validation_return_pct']:.2f}% "
            f"worst={best['worst_validation_segment_return_pct']:.2f}% "
            f"test={test_summary['return_pct']:.2f}% "
            f"{best['alpha_name']} {best['mode']} k={best['k']} gross={best['gross']}",
            flush=True,
        )

    search = pd.DataFrame(search_rows)
    window_results = pd.DataFrame(window_rows)
    selected = pd.DataFrame(selected_rows)
    validation_segments_out = (
        pd.concat(validation_segments_all, ignore_index=True) if validation_segments_all else pd.DataFrame()
    )
    test_segments_out = pd.concat(test_segments_all, ignore_index=True) if test_segments_all else pd.DataFrame()
    compounded = summarize_compounded(window_results, args.initial_cash, args.target_return_pct)
    stress_window_results = window_results.copy()
    stress_window_results["test_return_pct"] = stress_window_results["test_stress_return_pct"].astype(float)
    compounded_stress = summarize_compounded(stress_window_results, args.initial_cash, args.target_return_pct)
    traded = window_results[window_results["selected"] == "alpha_policy"]

    search.to_csv(args.output_dir / "validation_search.csv", index=False)
    window_results.to_csv(args.output_dir / "window_results.csv", index=False)
    selected.to_csv(args.output_dir / "selected_candidates.csv", index=False)
    validation_segments_out.to_csv(args.output_dir / "validation_segments.csv", index=False)
    test_segments_out.to_csv(args.output_dir / "test_segments.csv", index=False)
    summary = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "dataset_dir": str(args.dataset_dir),
        "interval": args.interval,
        "horizon": args.horizon,
        "symbols": symbols,
        "window_count": int(len(window_results)),
        "trade_window_count": int(len(traded)),
        "candidate_count_per_window": int(len(candidates)),
        "alpha_lags": parse_ints(args.alpha_lags),
        "alpha_kinds": [item.strip() for item in args.alpha_kinds.split(",") if item.strip()],
        "alpha_normalize": args.alpha_normalize,
        "modes": modes,
        "k_grid": parse_ints(args.k_grid),
        "gross_grid": parse_floats(args.gross_grid),
        "rebalance_grid": parse_ints(args.rebalance_grid),
        "start_date": args.start_date,
        "end_date": args.end_date,
        "validation_days": args.validation_days,
        "test_days": args.test_days,
        "step_days": args.step_days,
        "cost_bps": args.cost_bps,
        "stress_cost_bps": args.stress_cost_bps,
        "constraints": {
            "min_validation_return_pct": args.min_validation_return_pct,
            "min_validation_stress_return_pct": args.min_validation_stress_return_pct,
            "min_validation_segment_return_pct": args.min_validation_segment_return_pct,
            "max_validation_drawdown_pct": args.max_validation_drawdown_pct,
        },
        "selection": {
            "drawdown_penalty": args.drawdown_penalty,
            "turnover_penalty": args.turnover_penalty,
            "segment_selection_penalty": args.segment_selection_penalty,
            "stress_selection_weight": args.stress_selection_weight,
        },
        "compounded_test_summary": compounded,
        "compounded_stress_test_summary": compounded_stress,
        "trade_window_win_rate": float((traded["test_return_pct"] > 0.0).mean()) if len(traded) else 0.0,
        "worst_trade_window_test_return_pct": float(traded["test_return_pct"].min()) if len(traded) else 0.0,
        "mean_trade_window_test_return_pct": float(traded["test_return_pct"].mean()) if len(traded) else 0.0,
        "worst_window_test_max_drawdown_pct": float(window_results["test_max_drawdown_pct"].min()),
        "config_json": str(args.config_json) if args.config_json else None,
        "frozen_protocol_config": args_to_config(args)["config"],
        "protocol": (
            "For each window, alpha and execution parameters are selected on the validation slice only. "
            "A cash fallback is used when no candidate satisfies the validation robustness constraints. "
            "The selected candidate is then evaluated on the immediately following test slice."
        ),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    readme = [
        "# Walk-Forward Alpha Policy Lab",
        "",
        "Rolling alpha and execution-policy selection with a strict cash fallback.",
        "",
        "## Protocol",
        "",
        f"- Dataset: `{args.dataset_dir}`",
        f"- Windows: {len(window_results)}",
        f"- Trade windows: {len(traded)}",
        f"- Validation days: {args.validation_days}",
        f"- Test days: {args.test_days}",
        f"- Candidate count per window: {len(candidates)}",
        f"- Cost: {args.cost_bps:.2f} bps",
        f"- Min validation return: {args.min_validation_return_pct}",
        f"- Min worst validation segment: {args.min_validation_segment_return_pct}",
        f"- Max validation drawdown: {args.max_validation_drawdown_pct}",
        "",
        "## Compounded Test Result",
        "",
        f"- Initial cash: {compounded['initial_cash']:.2f}",
        f"- Final equity: {compounded['final_equity']:.2f}",
        f"- PnL: {compounded['pnl']:.2f}",
        f"- Return: {compounded['return_pct']:.2f}%",
        f"- Stress return at {args.stress_cost_bps:.2f} bps: {compounded_stress['return_pct']:.2f}%",
        f"- Window endpoint max drawdown: {compounded['max_drawdown_pct']:.2f}%",
        f"- Worst per-window max drawdown: {summary['worst_window_test_max_drawdown_pct']:.2f}%",
        f"- Trade-window win rate: {summary['trade_window_win_rate'] * 100.0:.2f}%",
        f"- Worst traded window: {summary['worst_trade_window_test_return_pct']:.2f}%",
        f"- Target hit ({args.target_return_pct:.1f}%): {compounded['target_hit']}",
        "",
        "This is a research backtest, not financial advice or a live trading recommendation.",
        "",
    ]
    (args.output_dir / "README.md").write_text("\n".join(readme), encoding="utf-8")
    print(f"Output dir: {args.output_dir}")
    print(f"Windows evaluated: {len(window_results)}")
    print(f"Trade windows: {len(traded)}")
    print(f"Compounded test return: {compounded['return_pct']:.2f}%")
    print(f"Target hit: {compounded['target_hit']}")


if __name__ == "__main__":
    main()
