from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd


FROZEN_CONFIG = Path("configs/strict_nested_edge_policy_20260622.json")
FRESH_ARTIFACT = Path("artifacts/fresh_diplom_risk_20260623/fresh_kronos_risk_predictions.csv")
OLD_ARTIFACT = Path("D:/AI/diplom_selected/results/diploma_run/predictions.csv")


PERIODS = {
    "default_2026": {
        "start_date": "2026-01-01",
        "end_date": "2026-06-20",
        "validation_days": 30,
        "nested_gate_days": 7,
        "test_days": 7,
        "step_days": 7,
    },
    "broad_2022_2026_monthly": {
        "start_date": "2022-09-01",
        "end_date": "2026-06-20",
        "validation_days": 120,
        "nested_gate_days": 30,
        "test_days": 30,
        "step_days": 30,
    },
    "may_2026": {
        "start_date": "2026-04-01",
        "end_date": "2026-06-01",
        "validation_days": 30,
        "nested_gate_days": 7,
        "test_days": 31,
        "step_days": 31,
    },
    "y2025_full_year": {
        "start_date": "2024-09-01",
        "end_date": "2026-01-01",
        "validation_days": 120,
        "nested_gate_days": 30,
        "test_days": 30,
        "step_days": 30,
    },
    "ydex_transition": {
        "start_date": "2024-03-01",
        "end_date": "2024-10-31",
        "validation_days": 120,
        "nested_gate_days": 30,
        "test_days": 30,
        "step_days": 30,
    },
}


VARIANTS = {
    "baseline": {},
    "old_artifact_target_only": {
        "diplom_risk_enabled": True,
        "diplom_risk_predictions_path": OLD_ARTIFACT,
        "diplom_risk_weight_pct": 25.0,
        "diplom_long_veto_p_high": 0.8,
    },
    "fresh_target_only": {
        "diplom_risk_enabled": True,
        "diplom_risk_predictions_path": FRESH_ARTIFACT,
        "diplom_risk_weight_pct": 25.0,
        "diplom_long_veto_p_high": 0.8,
    },
    "fresh_candidate_only": {
        "diplom_risk_enabled": True,
        "diplom_risk_predictions_path": FRESH_ARTIFACT,
        "diplom_candidate_score_penalty_pct": 6.0,
    },
    "fresh_gate_only": {
        "diplom_risk_enabled": True,
        "diplom_risk_predictions_path": FRESH_ARTIFACT,
        "diplom_gate_penalty_pct": 4.0,
    },
    "fresh_combined": {
        "diplom_risk_enabled": True,
        "diplom_risk_predictions_path": FRESH_ARTIFACT,
        "diplom_risk_weight_pct": 25.0,
        "diplom_long_veto_p_high": 0.8,
        "diplom_long_p_high_cap": 0.60,
        "diplom_candidate_score_penalty_pct": 6.0,
        "diplom_gate_penalty_pct": 4.0,
    },
}


BASE_FIELDS = [
    "dataset_dir",
    "interval",
    "horizon",
    "alpha_lags",
    "alpha_kinds",
    "alpha_normalize",
    "modes",
    "k_grid",
    "gross_grid",
    "rebalance_grid",
    "signal_delay_bars",
    "engine",
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
    "decision_mode",
    "decision_score_threshold",
    "calibration_min_windows",
    "calibration_bias_weight",
    "min_gate_return_pct",
    "min_gate_stress_return_pct",
    "min_gate_worst_month_return_pct",
    "min_gate_month_win_rate",
    "min_edge_stress_pct",
    "edge_uncertainty_weight",
    "edge_regime_risk_weight_pct",
    "max_regime_risk",
    "regime_block_days",
    "regime_bad_return_quantile",
    "regime_bad_drawdown_quantile",
    "min_regime_history_windows",
    "neural_scores_paths",
    "neural_feature_weight",
    "neural_uncertainty_weight",
    "diplom_stale_days",
    "diplom_missing_policy",
    "diplom_enable_yndx_ydex_mapping",
    "diplom_yndx_ydex_effective_date",
    "segment_count",
    "target_return_pct",
]


OPTION_NAMES = {
    "start_date": "--start-date",
    "end_date": "--end-date",
    "validation_days": "--validation-days",
    "test_days": "--test-days",
    "step_days": "--step-days",
    "nested_gate_days": "--nested-gate-days",
    "dataset_dir": "--dataset-dir",
    "interval": "--interval",
    "horizon": "--horizon",
    "alpha_lags": "--alpha-lags",
    "alpha_kinds": "--alpha-kinds",
    "alpha_normalize": "--alpha-normalize",
    "modes": "--modes",
    "k_grid": "--k-grid",
    "gross_grid": "--gross-grid",
    "rebalance_grid": "--rebalance-grid",
    "signal_delay_bars": "--signal-delay-bars",
    "engine": "--engine",
    "initial_cash": "--initial-cash",
    "cost_bps": "--cost-bps",
    "stress_cost_bps": "--stress-cost-bps",
    "drawdown_penalty": "--drawdown-penalty",
    "turnover_penalty": "--turnover-penalty",
    "segment_selection_penalty": "--segment-selection-penalty",
    "stress_selection_weight": "--stress-selection-weight",
    "min_validation_return_pct": "--min-validation-return-pct",
    "min_validation_stress_return_pct": "--min-validation-stress-return-pct",
    "min_validation_segment_return_pct": "--min-validation-segment-return-pct",
    "max_validation_drawdown_pct": "--max-validation-drawdown-pct",
    "decision_mode": "--decision-mode",
    "decision_score_threshold": "--decision-score-threshold",
    "calibration_min_windows": "--calibration-min-windows",
    "calibration_bias_weight": "--calibration-bias-weight",
    "min_gate_return_pct": "--min-gate-return-pct",
    "min_gate_stress_return_pct": "--min-gate-stress-return-pct",
    "min_gate_worst_month_return_pct": "--min-gate-worst-month-return-pct",
    "min_gate_month_win_rate": "--min-gate-month-win-rate",
    "min_edge_stress_pct": "--min-edge-stress-pct",
    "edge_uncertainty_weight": "--edge-uncertainty-weight",
    "edge_regime_risk_weight_pct": "--edge-regime-risk-weight-pct",
    "max_regime_risk": "--max-regime-risk",
    "regime_block_days": "--regime-block-days",
    "regime_bad_return_quantile": "--regime-bad-return-quantile",
    "regime_bad_drawdown_quantile": "--regime-bad-drawdown-quantile",
    "min_regime_history_windows": "--min-regime-history-windows",
    "neural_scores_paths": "--neural-scores-paths",
    "neural_feature_weight": "--neural-feature-weight",
    "neural_uncertainty_weight": "--neural-uncertainty-weight",
    "diplom_risk_predictions_path": "--diplom-risk-predictions-path",
    "diplom_risk_weight_pct": "--diplom-risk-weight-pct",
    "diplom_long_veto_p_high": "--diplom-long-veto-p-high",
    "diplom_long_p_high_cap": "--diplom-long-p-high-cap",
    "diplom_short_bonus_weight_pct": "--diplom-short-bonus-weight-pct",
    "diplom_candidate_score_penalty_pct": "--diplom-candidate-score-penalty-pct",
    "diplom_gate_penalty_pct": "--diplom-gate-penalty-pct",
    "diplom_stale_days": "--diplom-stale-days",
    "diplom_missing_policy": "--diplom-missing-policy",
    "diplom_yndx_ydex_effective_date": "--diplom-yndx-ydex-effective-date",
    "segment_count": "--segment-count",
    "target_return_pct": "--target-return-pct",
}


def load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("config", payload)


def append_option(command: list[str], key: str, value: Any) -> None:
    if value is None:
        return
    if key == "regime_veto":
        command.append("--regime-veto" if bool(value) else "--no-regime-veto")
        return
    if key == "diplom_risk_enabled":
        command.append("--diplom-risk-enabled" if bool(value) else "--no-diplom-risk-enabled")
        return
    if key == "diplom_enable_yndx_ydex_mapping":
        command.append("--diplom-enable-yndx-ydex-mapping" if bool(value) else "--no-diplom-enable-yndx-ydex-mapping")
        return
    option = OPTION_NAMES[key]
    command.extend([option, str(value)])


def build_command(
    base_config: dict[str, Any],
    period_name: str,
    variant_name: str,
    output_dir: Path,
) -> list[str]:
    command = [sys.executable, "examples/walk_forward_alpha_policy_lab.py"]
    for key in BASE_FIELDS:
        append_option(command, key, base_config.get(key))
    append_option(command, "regime_veto", base_config.get("regime_veto", True))
    for key, value in PERIODS[period_name].items():
        append_option(command, key, value)
    variant = VARIANTS[variant_name]
    append_option(command, "diplom_risk_enabled", bool(variant.get("diplom_risk_enabled", False)))
    for key in [
        "diplom_risk_predictions_path",
        "diplom_risk_weight_pct",
        "diplom_long_veto_p_high",
        "diplom_long_p_high_cap",
        "diplom_candidate_score_penalty_pct",
        "diplom_gate_penalty_pct",
    ]:
        append_option(command, key, variant.get(key, 0.0 if key.endswith("_pct") else None))
    command.extend(["--output-dir", str(output_dir)])
    return command


def read_summary(output_dir: Path, period: str, variant: str) -> dict[str, Any]:
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    compounded = summary["compounded_test_summary"]
    stress = summary["compounded_stress_test_summary"]
    overlay = summary["diplom_risk_overlay"]
    return {
        "period": period,
        "variant": variant,
        "return_pct": compounded["return_pct"],
        "final_equity": compounded["final_equity"],
        "max_drawdown_pct": compounded["max_drawdown_pct"],
        "stress_return_pct": stress["return_pct"],
        "windows": summary["window_count"],
        "trade_windows": summary["trade_window_count"],
        "cash_only_rate": summary["cash_only_rate"],
        "trade_window_win_rate": summary["trade_window_win_rate"],
        "worst_trade_window_test_return_pct": summary["worst_trade_window_test_return_pct"],
        "gate_veto_window_count": summary["gate_veto_window_count"],
        "avoided_loss_pct": summary["raw_vs_gated_veto_summary"]["avoided_loss_pct"],
        "missed_profit_pct": summary["raw_vs_gated_veto_summary"]["missed_profit_pct"],
        "diplom_changed_rebalance_decisions": summary["diplom_changed_rebalance_decisions"],
        "diplom_available_rebalance_decisions": summary["diplom_available_rebalance_decisions"],
        "diplom_missing_rebalance_decisions": summary["diplom_missing_rebalance_decisions"],
        "diplom_stale_rebalance_decisions": summary["diplom_stale_rebalance_decisions"],
        "diplom_risk_weight_pct": overlay["risk_weight_pct"],
        "diplom_candidate_score_penalty_pct": overlay.get("candidate_score_penalty_pct", 0.0),
        "diplom_gate_penalty_pct": overlay.get("gate_penalty_pct", 0.0),
        "output_dir": str(output_dir),
    }


def candidate_key(frame: pd.DataFrame) -> pd.Series:
    fields = ["raw_alpha_name", "raw_mode", "raw_k", "raw_gross", "raw_rebalance_every"]
    return frame[fields].astype(str).agg("|".join, axis=1)


def numeric_series(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(default)


def collect_pair_diagnostics(matrix_dir: Path, period: str, variant: str) -> dict[str, Any]:
    if variant == "baseline":
        return {
            "period": period,
            "variant": variant,
            "selected_candidate_changed_windows": 0,
            "gate_pass_changed_windows": 0,
            "executed_state_changed_windows": 0,
            "affected_windows": 0,
            "affected_but_vetoed_windows": 0,
            "affected_and_traded_windows": 0,
            "raw_delta_sum_pct": 0.0,
            "executed_delta_sum_pct": 0.0,
        }
    baseline_path = matrix_dir / period / "baseline" / "window_results.csv"
    variant_path = matrix_dir / period / variant / "window_results.csv"
    if not baseline_path.exists() or not variant_path.exists():
        return {"period": period, "variant": variant}
    base = pd.read_csv(baseline_path)
    other = pd.read_csv(variant_path)
    merged = base.merge(other, on="window_id", suffixes=("_baseline", "_variant"))
    if merged.empty:
        return {"period": period, "variant": variant}
    base_candidate = candidate_key(merged.rename(columns={f"{col}_baseline": col for col in ["raw_alpha_name", "raw_mode", "raw_k", "raw_gross", "raw_rebalance_every"]}))
    variant_candidate = candidate_key(merged.rename(columns={f"{col}_variant": col for col in ["raw_alpha_name", "raw_mode", "raw_k", "raw_gross", "raw_rebalance_every"]}))
    affected = (
        (numeric_series(merged, "raw_test_diplom_risk_changed_count_variant") > 0)
        | (base_candidate != variant_candidate)
        | (merged["gate_pass_baseline"].astype(bool) != merged["gate_pass_variant"].astype(bool))
    )
    traded = merged["gate_pass_variant"].astype(bool)
    return {
        "period": period,
        "variant": variant,
        "selected_candidate_changed_windows": int((base_candidate != variant_candidate).sum()),
        "gate_pass_changed_windows": int((merged["gate_pass_baseline"].astype(bool) != merged["gate_pass_variant"].astype(bool)).sum()),
        "executed_state_changed_windows": int((merged["selected_baseline"].astype(str) != merged["selected_variant"].astype(str)).sum()),
        "affected_windows": int(affected.sum()),
        "affected_but_vetoed_windows": int((affected & ~traded).sum()),
        "affected_and_traded_windows": int((affected & traded).sum()),
        "raw_delta_sum_pct": float(
            (
                numeric_series(merged, "raw_test_return_pct_variant")
                - numeric_series(merged, "raw_test_return_pct_baseline")
            ).sum()
        ),
        "executed_delta_sum_pct": float(
            (
                numeric_series(merged, "test_return_pct_variant")
                - numeric_series(merged, "test_return_pct_baseline")
            ).sum()
        ),
    }


def aggregate_outputs(matrix_dir: Path) -> None:
    rows = []
    pair_rows = []
    for period in PERIODS:
        for variant in VARIANTS:
            out = matrix_dir / period / variant
            if (out / "summary.json").exists():
                rows.append(read_summary(out, period, variant))
                pair_rows.append(collect_pair_diagnostics(matrix_dir, period, variant))
    summary = pd.DataFrame(rows)
    if not summary.empty:
        baseline_cols = ["period", "return_pct", "final_equity", "max_drawdown_pct", "stress_return_pct", "trade_windows", "cash_only_rate"]
        baseline = summary[summary["variant"] == "baseline"][baseline_cols].rename(
            columns={col: f"{col}_baseline" for col in baseline_cols if col != "period"}
        )
        summary = summary.merge(baseline, on="period", how="left")
        for col in ["return_pct", "final_equity", "max_drawdown_pct", "stress_return_pct", "trade_windows", "cash_only_rate"]:
            summary[f"{col}_delta_vs_baseline"] = summary[col] - summary[f"{col}_baseline"]
        summary.to_csv(matrix_dir / "comparison_summary.csv", index=False)
    pair_df = pd.DataFrame(pair_rows)
    if not pair_df.empty:
        pair_df.to_csv(matrix_dir / "pair_diagnostics.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fixed diplom risk overlay backtest matrix.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/fresh_diplom_risk_backtest_matrix_20260623"))
    parser.add_argument("--periods", default=",".join(PERIODS))
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()

    base_config = load_config(FROZEN_CONFIG)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_periods = [item.strip() for item in args.periods.split(",") if item.strip()]
    selected_variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    if args.aggregate_only:
        aggregate_outputs(args.output_dir)
        return

    run_manifest = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "frozen_config": str(FROZEN_CONFIG),
        "fresh_artifact": str(FRESH_ARTIFACT),
        "old_artifact": str(OLD_ARTIFACT),
        "periods": {key: PERIODS[key] for key in selected_periods},
        "variants": {key: {name: str(value) for name, value in VARIANTS[key].items()} for key in selected_variants},
    }
    (args.output_dir / "matrix_manifest.json").write_text(json.dumps(run_manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    for period in selected_periods:
        if period not in PERIODS:
            raise ValueError(f"unknown period: {period}")
        for variant in selected_variants:
            if variant not in VARIANTS:
                raise ValueError(f"unknown variant: {variant}")
            out = args.output_dir / period / variant
            if args.skip_existing and (out / "summary.json").exists():
                print(f"skip {period}/{variant}")
                continue
            out.mkdir(parents=True, exist_ok=True)
            command = build_command(base_config, period, variant, out)
            print(f"run {period}/{variant}", flush=True)
            subprocess.run(command, check=True)
            aggregate_outputs(args.output_dir)
    aggregate_outputs(args.output_dir)
    print(f"Matrix output: {args.output_dir}")


if __name__ == "__main__":
    main()
