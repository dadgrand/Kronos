from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_window_results(matrix_dir: Path, period: str, variant: str) -> pd.DataFrame:
    path = matrix_dir / period / variant / "window_results.csv"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    frame["period"] = period
    frame["variant"] = variant
    return frame


def candidate_key(frame: pd.DataFrame, suffix: str) -> pd.Series:
    fields = [
        f"raw_alpha_name{suffix}",
        f"raw_mode{suffix}",
        f"raw_k{suffix}",
        f"raw_gross{suffix}",
        f"raw_rebalance_every{suffix}",
    ]
    return frame[fields].astype(str).agg("|".join, axis=1)


def numeric(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(default)


def numeric_any(frame: pd.DataFrame, columns: list[str], default: float = 0.0) -> pd.Series:
    for column in columns:
        if column in frame.columns:
            return numeric(frame, column, default)
    return pd.Series(default, index=frame.index, dtype=float)


def bools(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index)
    return frame[column].astype(str).str.lower().isin({"true", "1", "yes"})


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def row_value(row: pd.Series, *columns: str, default: Any = None) -> Any:
    for column in columns:
        if column in row.index:
            return row.get(column)
    return default


def build_pair_diagnostics(matrix_dir: Path, summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for period in sorted(summary["period"].unique()):
        baseline = load_window_results(matrix_dir, period, "baseline")
        if baseline.empty:
            continue
        for variant in sorted(summary.loc[summary["period"] == period, "variant"].unique()):
            if variant == "baseline":
                rows.append(
                    {
                        "period": period,
                        "variant": variant,
                        "selected_candidate_changed_windows": 0,
                        "gate_pass_changed_windows": 0,
                        "affected_windows": 0,
                        "affected_but_vetoed_windows": 0,
                        "affected_and_traded_windows": 0,
                        "raw_delta_sum_pct": 0.0,
                        "executed_delta_sum_pct": 0.0,
                    }
                )
                continue
            other = load_window_results(matrix_dir, period, variant)
            if other.empty:
                continue
            merged = baseline.merge(other, on="window_id", suffixes=("_baseline", "_variant"))
            if merged.empty:
                continue
            base_candidate = candidate_key(merged, "_baseline")
            other_candidate = candidate_key(merged, "_variant")
            gate_changed = bools(merged, "gate_pass_baseline") != bools(merged, "gate_pass_variant")
            raw_delta = numeric(merged, "raw_test_return_pct_variant") - numeric(merged, "raw_test_return_pct_baseline")
            exec_delta = numeric(merged, "test_return_pct_variant") - numeric(merged, "test_return_pct_baseline")
            changed_count = numeric_any(
                merged,
                ["raw_test_diplom_risk_changed_count_variant", "raw_test_diplom_risk_changed_count"],
            )
            affected = (
                (base_candidate != other_candidate)
                | gate_changed
                | (changed_count > 0)
                | (raw_delta.abs() > 1e-12)
                | (exec_delta.abs() > 1e-12)
            )
            variant_traded = bools(merged, "gate_pass_variant")
            rows.append(
                {
                    "period": period,
                    "variant": variant,
                    "selected_candidate_changed_windows": int((base_candidate != other_candidate).sum()),
                    "gate_pass_changed_windows": int(gate_changed.sum()),
                    "affected_windows": int(affected.sum()),
                    "affected_but_vetoed_windows": int((affected & ~variant_traded).sum()),
                    "affected_and_traded_windows": int((affected & variant_traded).sum()),
                    "raw_delta_sum_pct": float(raw_delta.sum()),
                    "executed_delta_sum_pct": float(exec_delta.sum()),
                }
            )
    return pd.DataFrame(rows)


def build_affected_window_audit(matrix_dir: Path, summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for period in sorted(summary["period"].unique()):
        baseline = load_window_results(matrix_dir, period, "baseline")
        if baseline.empty:
            continue
        for variant in sorted(summary.loc[summary["period"] == period, "variant"].unique()):
            if variant == "baseline":
                continue
            other = load_window_results(matrix_dir, period, variant)
            if other.empty:
                continue
            merged = baseline.merge(other, on="window_id", suffixes=("_baseline", "_variant"))
            base_candidate = candidate_key(merged, "_baseline")
            other_candidate = candidate_key(merged, "_variant")
            raw_delta = numeric(merged, "raw_test_return_pct_variant") - numeric(merged, "raw_test_return_pct_baseline")
            exec_delta = numeric(merged, "test_return_pct_variant") - numeric(merged, "test_return_pct_baseline")
            changed_count = numeric_any(
                merged,
                ["raw_test_diplom_risk_changed_count_variant", "raw_test_diplom_risk_changed_count"],
            )
            affected = (
                (base_candidate != other_candidate)
                | (bools(merged, "gate_pass_baseline") != bools(merged, "gate_pass_variant"))
                | (changed_count > 0)
                | (raw_delta.abs() > 1e-12)
                | (exec_delta.abs() > 1e-12)
            )
            for _, row in merged.loc[affected].iterrows():
                rows.append(
                    {
                        "period": period,
                        "variant": variant,
                        "window_id": int(row["window_id"]),
                        "test_start": row.get("test_start_baseline"),
                        "test_end": row.get("test_end_baseline"),
                        "baseline_candidate": "|".join(
                            str(row.get(f"{field}_baseline"))
                            for field in ["raw_alpha_name", "raw_mode", "raw_k", "raw_gross", "raw_rebalance_every"]
                        ),
                        "variant_candidate": "|".join(
                            str(row.get(f"{field}_variant"))
                            for field in ["raw_alpha_name", "raw_mode", "raw_k", "raw_gross", "raw_rebalance_every"]
                        ),
                        "candidate_changed": "|".join(
                            str(row.get(f"{field}_baseline"))
                            for field in ["raw_alpha_name", "raw_mode", "raw_k", "raw_gross", "raw_rebalance_every"]
                        )
                        != "|".join(
                            str(row.get(f"{field}_variant"))
                            for field in ["raw_alpha_name", "raw_mode", "raw_k", "raw_gross", "raw_rebalance_every"]
                        ),
                        "baseline_gate_pass": parse_bool(row.get("gate_pass_baseline")),
                        "variant_gate_pass": parse_bool(row.get("gate_pass_variant")),
                        "baseline_selected": row.get("selected_baseline"),
                        "variant_selected": row.get("selected_variant"),
                        "baseline_raw_test_return_pct": row.get("raw_test_return_pct_baseline"),
                        "variant_raw_test_return_pct": row.get("raw_test_return_pct_variant"),
                        "raw_delta_pct": float(raw_delta.loc[row.name]),
                        "baseline_executed_test_return_pct": row.get("test_return_pct_baseline"),
                        "variant_executed_test_return_pct": row.get("test_return_pct_variant"),
                        "executed_delta_pct": float(exec_delta.loc[row.name]),
                        "variant_diplom_available": row_value(
                            row,
                            "raw_test_diplom_risk_available_count_variant",
                            "raw_test_diplom_risk_available_count",
                            default=0,
                        ),
                        "variant_diplom_missing": row_value(
                            row,
                            "raw_test_diplom_risk_missing_count_variant",
                            "raw_test_diplom_risk_missing_count",
                            default=0,
                        ),
                        "variant_diplom_stale": row_value(
                            row,
                            "raw_test_diplom_risk_stale_count_variant",
                            "raw_test_diplom_risk_stale_count",
                            default=0,
                        ),
                        "variant_diplom_changed": row_value(
                            row,
                            "raw_test_diplom_risk_changed_count_variant",
                            "raw_test_diplom_risk_changed_count",
                            default=0,
                        ),
                        "variant_diplom_p_high": row_value(row, "raw_test_diplom_p_high_variant", "raw_test_diplom_p_high"),
                        "variant_diplom_long_p_high_exposure": row_value(
                            row,
                            "raw_test_diplom_long_p_high_exposure_variant",
                            "raw_test_diplom_long_p_high_exposure",
                        ),
                        "variant_diplom_gross_reduction_pct": row_value(
                            row,
                            "raw_test_diplom_target_gross_reduction_pct_variant",
                            "raw_test_diplom_target_gross_reduction_pct",
                        ),
                        "variant_gate_reject_reasons": row.get("gate_reject_reasons_variant"),
                    }
                )
    return pd.DataFrame(rows)


def build_artifact_ticker_summary(artifact_dir: Path) -> pd.DataFrame:
    predictions = pd.read_csv(artifact_dir / "fresh_kronos_risk_predictions.csv")
    latest = predictions[pd.to_datetime(predictions["availability_timestamp"]).eq(pd.to_datetime(predictions["availability_timestamp"]).max())]
    counts = predictions.groupby("ticker").agg(
        prediction_count=("ticker", "size"),
        first_decision_date=("decision_date", "min"),
        last_decision_date=("decision_date", "max"),
        mean_p_high=("p_high", "mean"),
        max_p_high=("p_high", "max"),
    )
    latest = latest.set_index("ticker")[["predicted_risk_class", "p_low", "p_medium", "p_high", "feature_coverage"]].rename(
        columns={
            "predicted_risk_class": "latest_risk_class",
            "p_low": "latest_p_low",
            "p_medium": "latest_p_medium",
            "p_high": "latest_p_high",
            "feature_coverage": "latest_feature_coverage",
        }
    )
    out = counts.join(latest, how="left").reset_index()
    return out.sort_values(["latest_p_high", "ticker"], ascending=[False, True])


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize diplom risk matrix outputs into report CSVs.")
    parser.add_argument("--matrix-dir", type=Path, default=Path("outputs/fresh_diplom_risk_backtest_matrix_20260623"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/fresh_diplom_risk_20260623"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/fresh_diplom_risk_overlay_20260623"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(args.matrix_dir / "comparison_summary.csv")
    pair = build_pair_diagnostics(args.matrix_dir, summary)
    affected = build_affected_window_audit(args.matrix_dir, summary)
    ticker = build_artifact_ticker_summary(args.artifact_dir)

    summary.to_csv(args.output_dir / "comparison_summary.csv", index=False)
    pair.to_csv(args.output_dir / "pair_diagnostics.csv", index=False)
    affected.to_csv(args.output_dir / "affected_window_audit.csv", index=False)
    ticker.to_csv(args.output_dir / "artifact_ticker_summary.csv", index=False)

    by_period = summary.pivot_table(
        index="period",
        columns="variant",
        values="return_pct_delta_vs_baseline",
        aggfunc="first",
    ).reset_index()
    by_period.to_csv(args.output_dir / "return_delta_pivot.csv", index=False)

    manifest = {
        "matrix_dir": str(args.matrix_dir),
        "artifact_dir": str(args.artifact_dir),
        "summary_rows": int(len(summary)),
        "affected_window_rows": int(len(affected)),
        "ticker_rows": int(len(ticker)),
        "fresh_artifact_manifest": read_json(args.artifact_dir / "artifact_manifest.json"),
    }
    (args.output_dir / "summary_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Report tables: {args.output_dir}")
    print(f"Affected windows: {len(affected)}")


if __name__ == "__main__":
    main()
