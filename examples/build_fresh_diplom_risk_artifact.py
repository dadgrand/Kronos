from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from neural_policy_lab import read_intraday_matrix


FORBIDDEN_COLUMNS = {
    "risk_class",
    "risk_score",
    "future_max_drawdown",
    "future_downside_volatility",
    "future_cvar_95",
    "future_illiquidity",
    "future_return",
}

PREDICTION_COLUMNS = [
    "decision_date",
    "availability_timestamp",
    "source_data_end",
    "ticker",
    "kronos_symbol",
    "sector",
    "predicted_risk_class",
    "p_low",
    "p_medium",
    "p_high",
    "feature_coverage",
    "model_version",
    "final_architecture",
    "used_sector_expert",
]


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_fingerprint(dataset_dir: Path, interval: int) -> str:
    root = dataset_dir / "curated" / f"interval={interval}"
    digest = hashlib.sha256()
    for path in sorted(root.glob("symbol=*/year=*/month=*/candles_curated.parquet")):
        stat = path.stat()
        rel = path.relative_to(dataset_dir).as_posix()
        digest.update(f"{rel}|{stat.st_size}|{int(stat.st_mtime)}\n".encode("utf-8"))
    return digest.hexdigest()


def monthly_source_dates(index: pd.DatetimeIndex) -> list[pd.Timestamp]:
    dates = pd.DatetimeIndex(index).normalize().unique().sort_values()
    if len(dates) == 0:
        return []
    series = pd.Series(dates, index=dates)
    return [pd.Timestamp(value) for value in series.groupby(dates.to_period("M")).max().tolist()]


def last_timestamp_by_day(index: pd.DatetimeIndex) -> pd.Series:
    timestamps = pd.DatetimeIndex(index).sort_values()
    return pd.Series(timestamps, index=timestamps.normalize()).groupby(level=0).max()


def rank01(series: pd.Series, *, ascending: bool = True) -> pd.Series:
    out = pd.Series(0.5, index=series.index, dtype=np.float64)
    finite = series.replace([np.inf, -np.inf], np.nan).dropna()
    if len(finite) >= 2:
        out.loc[finite.index] = finite.rank(pct=True, ascending=ascending).astype(float)
    elif len(finite) == 1:
        out.loc[finite.index] = 0.5
    return out.fillna(0.5).clip(0.0, 1.0)


def softmax3(low_logit: pd.Series, medium_logit: pd.Series, high_logit: pd.Series) -> pd.DataFrame:
    logits = pd.concat([low_logit, medium_logit, high_logit], axis=1)
    logits.columns = ["p_low", "p_medium", "p_high"]
    values = logits.to_numpy(dtype=np.float64)
    values = values - np.nanmax(values, axis=1, keepdims=True)
    exp_values = np.exp(np.nan_to_num(values, nan=-60.0))
    probs = exp_values / np.clip(exp_values.sum(axis=1, keepdims=True), 1e-12, None)
    return pd.DataFrame(probs, index=logits.index, columns=logits.columns)


def compute_month_risk(
    daily_close: pd.DataFrame,
    daily_volume: pd.DataFrame,
    source_date: pd.Timestamp,
    *,
    min_feature_coverage: float,
    lookback_days: int,
) -> pd.DataFrame:
    close_hist = daily_close.loc[:source_date]
    volume_hist = daily_volume.reindex_like(daily_close).fillna(0.0).loc[:source_date]
    if len(close_hist) < 20:
        return pd.DataFrame()

    symbols = list(close_hist.columns)
    returns = close_hist.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    market_returns = returns.mean(axis=1, skipna=True)
    last_close = close_hist.iloc[-1]
    coverage_lookback = close_hist.tail(lookback_days).notna().sum() / float(max(1, min(lookback_days, len(close_hist))))
    volume_coverage = (volume_hist.tail(20) > 0.0).sum() / float(max(1, min(20, len(volume_hist))))
    feature_coverage = pd.concat([coverage_lookback, volume_coverage], axis=1).min(axis=1).reindex(symbols)
    eligible = last_close.notna() & (feature_coverage >= float(min_feature_coverage))
    if not bool(eligible.any()):
        return pd.DataFrame()

    ret_20 = last_close / close_hist.shift(20).iloc[-1] - 1.0
    ret_60 = last_close / close_hist.shift(60).iloc[-1] - 1.0
    vol_20 = returns.tail(20).std(ddof=0) * np.sqrt(252.0)
    downside_60 = returns.tail(60).where(returns.tail(60) < 0.0).std(ddof=0).fillna(0.0) * np.sqrt(252.0)
    window_close = close_hist.tail(60)
    max_drawdown_60 = (window_close / window_close.cummax() - 1.0).min()
    traded_value = (close_hist * volume_hist).replace([np.inf, -np.inf], np.nan)
    value_20 = traded_value.tail(20).median()
    amihud_20 = (returns.abs() / traded_value.replace(0.0, np.nan)).tail(20).median()
    zero_volume_20 = (volume_hist.tail(20) <= 0.0).mean()

    r60 = returns.tail(60)
    m60 = market_returns.reindex(r60.index)
    market_var = float(np.nanvar(m60.to_numpy(dtype=np.float64)))
    beta_60 = pd.Series(0.0, index=symbols, dtype=np.float64)
    if market_var > 1e-12:
        demeaned_market = m60 - m60.mean()
        demeaned_returns = r60.sub(r60.mean(axis=0), axis=1)
        beta_60 = demeaned_returns.mul(demeaned_market, axis=0).mean(axis=0) / market_var
    corr_60 = r60.corrwith(m60).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    score = (
        0.21 * rank01(vol_20)
        + 0.17 * rank01(downside_60)
        + 0.18 * rank01(-max_drawdown_60)
        + 0.13 * rank01(amihud_20)
        + 0.09 * rank01(value_20, ascending=False)
        + 0.08 * rank01(zero_volume_20)
        + 0.10 * rank01(-ret_60)
        + 0.04 * rank01(beta_60.abs() + 0.25 * corr_60.abs())
    ).clip(0.0, 1.0)

    probs = softmax3(
        low_logit=4.0 * (0.35 - score),
        medium_logit=1.25 - 6.0 * (score - 0.50).abs(),
        high_logit=4.0 * (score - 0.65),
    )
    labels = probs.idxmax(axis=1).str.replace("p_", "", regex=False)
    features = pd.DataFrame(
        {
            "feature_coverage": feature_coverage,
            "ret_20": ret_20,
            "ret_60": ret_60,
            "vol_20": vol_20,
            "downside_vol_60": downside_60,
            "max_drawdown_60": max_drawdown_60,
            "median_value_20": value_20,
            "amihud_20": amihud_20,
            "zero_volume_20": zero_volume_20,
            "beta_60": beta_60,
            "corr_60": corr_60,
            "risk_score_internal": score,
        },
        index=symbols,
    )
    out = pd.concat([probs, features], axis=1)
    out["predicted_risk_class"] = labels
    out = out.loc[eligible.reindex(out.index).fillna(False)]
    return out


def build_predictions(
    index: pd.DatetimeIndex,
    close_px: pd.DataFrame,
    volume: pd.DataFrame,
    *,
    min_feature_coverage: float,
    lookback_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily_close = close_px.groupby(close_px.index.normalize()).last()
    daily_volume = volume.groupby(volume.index.normalize()).sum().reindex_like(daily_close).fillna(0.0)
    source_dates = monthly_source_dates(daily_close.index)
    source_end_by_day = last_timestamp_by_day(index)
    rows: list[dict[str, Any]] = []
    diagnostics: list[pd.DataFrame] = []
    for source_date in source_dates:
        month = compute_month_risk(
            daily_close,
            daily_volume,
            source_date,
            min_feature_coverage=min_feature_coverage,
            lookback_days=lookback_days,
        )
        if month.empty:
            continue
        source_data_end = pd.Timestamp(source_end_by_day.get(source_date.normalize(), source_date))
        availability = source_data_end + pd.Timedelta(minutes=1)
        for symbol, row in month.iterrows():
            rows.append(
                {
                    "decision_date": source_date.date().isoformat(),
                    "availability_timestamp": availability.isoformat(),
                    "source_data_end": source_data_end.isoformat(),
                    "ticker": str(symbol).upper(),
                    "kronos_symbol": str(symbol).upper(),
                    "sector": "moex",
                    "predicted_risk_class": str(row["predicted_risk_class"]),
                    "p_low": float(row["p_low"]),
                    "p_medium": float(row["p_medium"]),
                    "p_high": float(row["p_high"]),
                    "feature_coverage": float(row["feature_coverage"]),
                    "model_version": "fresh_kronos_target_free_risk_v1",
                    "final_architecture": "past_only_cross_sectional_vol_drawdown_liquidity_momentum",
                    "used_sector_expert": False,
                }
            )
        diag = month.copy()
        diag.insert(0, "source_date", source_date.date().isoformat())
        diag.insert(1, "ticker", diag.index)
        diagnostics.append(diag.reset_index(drop=True))
    predictions = pd.DataFrame(rows, columns=PREDICTION_COLUMNS)
    feature_diagnostics = pd.concat(diagnostics, ignore_index=True) if diagnostics else pd.DataFrame()
    return predictions, feature_diagnostics


def validate_predictions(predictions: pd.DataFrame) -> dict[str, Any]:
    forbidden_present = sorted(FORBIDDEN_COLUMNS.intersection(predictions.columns))
    duplicates = int(predictions.duplicated(["ticker", "availability_timestamp"]).sum()) if not predictions.empty else 0
    probs = predictions[["p_low", "p_medium", "p_high"]].apply(pd.to_numeric, errors="coerce")
    prob_sum_error = float((probs.sum(axis=1) - 1.0).abs().max()) if len(probs) else 0.0
    prob_bounds_bad = int(((probs < 0.0) | (probs > 1.0)).sum().sum()) if len(probs) else 0
    availability = pd.to_datetime(predictions["availability_timestamp"], errors="coerce") if len(predictions) else pd.Series(dtype="datetime64[ns]")
    source_end = pd.to_datetime(predictions["source_data_end"], errors="coerce") if len(predictions) else pd.Series(dtype="datetime64[ns]")
    availability_before_source_bad = int((availability < source_end).sum()) if len(predictions) else 0
    availability_not_after_source_bad = int((availability <= source_end).sum()) if len(predictions) else 0
    return {
        "target_free": not forbidden_present,
        "forbidden_columns_present": forbidden_present,
        "duplicate_ticker_availability_rows": duplicates,
        "probability_sum_max_abs_error": prob_sum_error,
        "probability_bounds_bad_cells": prob_bounds_bad,
        "availability_before_source_data_end_rows": availability_before_source_bad,
        "availability_not_after_source_data_end_rows": availability_not_after_source_bad,
        "row_count": int(len(predictions)),
        "ticker_count": int(predictions["ticker"].nunique()) if len(predictions) else 0,
    }


def coverage_summary(predictions: pd.DataFrame, symbols: list[str], source_dates: list[pd.Timestamp]) -> pd.DataFrame:
    rows = []
    source_count = max(1, len(source_dates))
    for symbol in symbols:
        part = predictions[predictions["ticker"] == symbol] if len(predictions) else pd.DataFrame()
        rows.append(
            {
                "ticker": symbol,
                "prediction_count": int(len(part)),
                "source_window_count": int(source_count),
                "coverage_pct": float(len(part) / source_count * 100.0),
                "first_decision_date": part["decision_date"].min() if len(part) else None,
                "last_decision_date": part["decision_date"].max() if len(part) else None,
                "latest_p_high": float(part["p_high"].iloc[-1]) if len(part) else np.nan,
                "latest_risk_class": str(part["predicted_risk_class"].iloc[-1]) if len(part) else None,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a target-free monthly Kronos risk artifact.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("datasets/moex_intraday_6f8f836cc811"))
    parser.add_argument("--interval", type=int, default=10)
    parser.add_argument("--min-feature-coverage", type=float, default=0.60)
    parser.add_argument("--lookback-days", type=int, default=60)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = Path("artifacts") / f"fresh_diplom_risk_{stamp}"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    index, symbols, _, close_px, volume = read_intraday_matrix(args.dataset_dir, args.interval)
    predictions, feature_diagnostics = build_predictions(
        index,
        close_px,
        volume,
        min_feature_coverage=args.min_feature_coverage,
        lookback_days=args.lookback_days,
    )
    if predictions.empty:
        raise RuntimeError("fresh risk artifact produced no rows")

    predictions_path = args.output_dir / "fresh_kronos_risk_predictions.csv"
    diagnostics_path = args.output_dir / "fresh_kronos_risk_feature_diagnostics.csv"
    coverage_path = args.output_dir / "fresh_kronos_risk_coverage.csv"
    manifest_path = args.output_dir / "artifact_manifest.json"
    readme_path = args.output_dir / "README.md"

    predictions.to_csv(predictions_path, index=False)
    feature_diagnostics.to_csv(diagnostics_path, index=False)
    source_dates = monthly_source_dates(close_px.groupby(close_px.index.normalize()).last().index)
    coverage = coverage_summary(predictions, symbols, source_dates)
    coverage.to_csv(coverage_path, index=False)
    validation = validate_predictions(predictions)
    manifest = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "artifact_type": "fresh_kronos_monthly_target_free_risk_predictions",
        "model_version": "fresh_kronos_target_free_risk_v1",
        "degraded_model": True,
        "target_free": validation["target_free"],
        "dataset_dir": str(args.dataset_dir),
        "dataset_fingerprint": dataset_fingerprint(args.dataset_dir, args.interval),
        "interval": args.interval,
        "input_start": str(index.min()),
        "input_end": str(index.max()),
        "source_data_end_min": str(pd.to_datetime(predictions["source_data_end"]).min()),
        "source_data_end_max": str(pd.to_datetime(predictions["source_data_end"]).max()),
        "availability_timestamp_min": str(pd.to_datetime(predictions["availability_timestamp"]).min()),
        "availability_timestamp_max": str(pd.to_datetime(predictions["availability_timestamp"]).max()),
        "symbols": symbols,
        "symbol_count": len(symbols),
        "source_months": len(source_dates),
        "min_feature_coverage": args.min_feature_coverage,
        "lookback_days": args.lookback_days,
        "predictions_path": str(predictions_path),
        "predictions_sha256": sha256_file(predictions_path),
        "feature_diagnostics_path": str(diagnostics_path),
        "coverage_path": str(coverage_path),
        "schema": PREDICTION_COLUMNS,
        "forbidden_columns": sorted(FORBIDDEN_COLUMNS),
        "validation": validation,
        "method": {
            "inputs": [
                "past daily close returns",
                "past realized/downside volatility",
                "past max drawdown",
                "past traded value and zero-volume rate",
                "past beta/correlation to equal-weight Kronos market",
                "past 60-day momentum reversal risk",
            ],
            "future_targets_used": False,
            "probability_calibration": "cross_sectional_softmax_from_past_only_risk_score",
            "availability_rule": "monthly source_data_end is the exact last intraday bar used; availability_timestamp is source_data_end + 1 minute, and adapter may only use rows with availability_timestamp <= decision_time",
        },
    }
    manifest_path.write_text(json.dumps(json_safe(manifest), indent=2, ensure_ascii=False), encoding="utf-8")
    readme = [
        "# Fresh Kronos Diplom Risk Artifact",
        "",
        "Monthly target-free risk predictions for the current Kronos universe.",
        "",
        f"- Predictions: `{predictions_path}`",
        f"- Rows: {validation['row_count']}",
        f"- Tickers: {validation['ticker_count']} / {len(symbols)}",
        f"- Input range: {index.min()} .. {index.max()}",
        f"- Target-free schema check: {validation['target_free']}",
        f"- SHA256: `{manifest['predictions_sha256']}`",
        "",
        "The model is intentionally degraded and heuristic. It uses only past close/volume-derived volatility, drawdown, liquidity, beta/correlation, and momentum features available by `source_data_end`.",
        "",
        "This artifact is for leakage-aware overlay testing, not for a standalone claim that risk probabilities are calibrated.",
        "",
    ]
    readme_path.write_text("\n".join(readme), encoding="utf-8")
    print(f"Predictions: {predictions_path}")
    print(f"Rows: {validation['row_count']} tickers={validation['ticker_count']}/{len(symbols)}")
    print(f"Target-free: {validation['target_free']} sha256={manifest['predictions_sha256']}")


if __name__ == "__main__":
    main()
