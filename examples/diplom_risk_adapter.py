from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


FORBIDDEN_INFERENCE_COLUMNS = {
    "risk_class",
    "risk_score",
    "future_max_drawdown",
    "future_downside_volatility",
    "future_cvar_95",
    "future_illiquidity",
    "future_return",
}

SAFE_PREDICTION_COLUMNS = [
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

RISK_CLASS_ORDER = {"low": 0, "medium": 1, "high": 2}


@dataclass(frozen=True)
class DiplomRiskConfig:
    enabled: bool = False
    predictions_path: Path | None = None
    risk_weight_pct: float = 0.0
    long_veto_p_high: float | None = None
    long_p_high_cap: float | None = None
    short_bonus_weight_pct: float = 0.0
    candidate_score_penalty_pct: float = 0.0
    gate_penalty_pct: float = 0.0
    stale_days: int = 45
    missing_policy: str = "neutral"
    enable_yndx_ydex_mapping: bool = False
    yndx_ydex_effective_date: str = "2024-07-24"


def bool_from_config(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "on"}


def config_from_mapping(raw: dict[str, Any]) -> DiplomRiskConfig:
    path_value = raw.get("diplom_risk_predictions_path")
    path = Path(path_value) if path_value not in {None, ""} else None
    veto = raw.get("diplom_long_veto_p_high")
    cap = raw.get("diplom_long_p_high_cap")
    stale_value = raw.get("diplom_stale_days", 45)
    return DiplomRiskConfig(
        enabled=bool_from_config(raw.get("diplom_risk_enabled", False)),
        predictions_path=path,
        risk_weight_pct=float(raw.get("diplom_risk_weight_pct", 0.0) or 0.0),
        long_veto_p_high=None if veto in {None, ""} else float(veto),
        long_p_high_cap=None if cap in {None, ""} else float(cap),
        short_bonus_weight_pct=float(raw.get("diplom_short_bonus_weight_pct", 0.0) or 0.0),
        candidate_score_penalty_pct=float(raw.get("diplom_candidate_score_penalty_pct", 0.0) or 0.0),
        gate_penalty_pct=float(raw.get("diplom_gate_penalty_pct", 0.0) or 0.0),
        stale_days=45 if stale_value in {None, ""} else int(stale_value),
        missing_policy=str(raw.get("diplom_missing_policy", "neutral") or "neutral"),
        enable_yndx_ydex_mapping=bool_from_config(raw.get("diplom_enable_yndx_ydex_mapping", False)),
        yndx_ydex_effective_date=str(raw.get("diplom_yndx_ydex_effective_date", "2024-07-24")),
    )


def empty_diplom_summary(reason: str = "diplom_risk_disabled") -> dict[str, Any]:
    return {
        "diplom_risk_available": False,
        "diplom_risk_class": None,
        "diplom_p_high": np.nan,
        "diplom_risk_penalty_pct": 0.0,
        "diplom_risk_reason": reason,
        "diplom_risk_active_count": 0,
        "diplom_risk_available_count": 0,
        "diplom_risk_missing_count": 0,
        "diplom_risk_stale_count": 0,
        "diplom_risk_changed_count": 0,
        "diplom_target_gross_before": 0.0,
        "diplom_target_gross_after": 0.0,
        "diplom_target_gross_reduction": 0.0,
        "diplom_target_gross_reduction_pct": 0.0,
        "diplom_target_cashout": False,
        "diplom_long_gross_before": 0.0,
        "diplom_long_gross_after": 0.0,
        "diplom_short_gross_before": 0.0,
        "diplom_short_gross_after": 0.0,
        "diplom_long_p_high_exposure": np.nan,
        "diplom_short_p_high_exposure": np.nan,
        "diplom_availability_min_age_days": np.nan,
    }


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def _join_reasons(reasons: list[str]) -> str:
    cleaned = []
    for reason in reasons:
        if reason and reason not in cleaned:
            cleaned.append(reason)
    return ",".join(cleaned) if cleaned else "ok"


def _weighted_average(values: list[float], weights: list[float]) -> float:
    clean_values = []
    clean_weights = []
    for value, weight in zip(values, weights, strict=True):
        if np.isfinite(value) and np.isfinite(weight) and weight > 0.0:
            clean_values.append(value)
            clean_weights.append(weight)
    if not clean_values:
        return np.nan
    return float(np.average(np.asarray(clean_values, dtype=np.float64), weights=np.asarray(clean_weights, dtype=np.float64)))


def summarize_diplom_diagnostics(diags: list[dict[str, Any]], *, disabled_reason: str = "diplom_risk_disabled") -> dict[str, Any]:
    if not diags:
        return empty_diplom_summary(disabled_reason)

    active = [diag for diag in diags if int(diag.get("diplom_risk_active_count", 0) or 0) > 0]
    if not active:
        return empty_diplom_summary("no_position")

    available = [diag for diag in active if bool(diag.get("diplom_risk_available", False))]
    p_high_values = [
        _safe_float(diag.get("diplom_p_high"))
        for diag in available
        if np.isfinite(_safe_float(diag.get("diplom_p_high")))
    ]
    class_values = [
        str(diag.get("diplom_risk_class"))
        for diag in available
        if str(diag.get("diplom_risk_class")) in RISK_CLASS_ORDER
    ]
    worst_class = None
    if class_values:
        worst_class = max(class_values, key=lambda value: RISK_CLASS_ORDER[value])

    penalty_values = [
        _safe_float(diag.get("diplom_risk_penalty_pct"), 0.0)
        for diag in active
        if np.isfinite(_safe_float(diag.get("diplom_risk_penalty_pct"), 0.0))
    ]
    reasons: list[str] = []
    for diag in active:
        reasons.extend(str(diag.get("diplom_risk_reason", "")).split(","))
    long_values = [_safe_float(diag.get("diplom_long_p_high_exposure")) for diag in active]
    long_weights = [_safe_float(diag.get("diplom_long_gross_before"), 0.0) for diag in active]
    short_values = [_safe_float(diag.get("diplom_short_p_high_exposure")) for diag in active]
    short_weights = [_safe_float(diag.get("diplom_short_gross_before"), 0.0) for diag in active]
    age_values = [
        _safe_float(diag.get("diplom_availability_min_age_days"))
        for diag in available
        if np.isfinite(_safe_float(diag.get("diplom_availability_min_age_days")))
    ]

    return {
        "diplom_risk_available": bool(available),
        "diplom_risk_class": worst_class,
        "diplom_p_high": float(max(p_high_values)) if p_high_values else np.nan,
        "diplom_risk_penalty_pct": float(max(penalty_values)) if penalty_values else 0.0,
        "diplom_risk_reason": _join_reasons(reasons),
        "diplom_risk_active_count": int(sum(int(diag.get("diplom_risk_active_count", 0) or 0) for diag in active)),
        "diplom_risk_available_count": int(sum(int(diag.get("diplom_risk_available_count", 0) or 0) for diag in active)),
        "diplom_risk_missing_count": int(sum(int(diag.get("diplom_risk_missing_count", 0) or 0) for diag in active)),
        "diplom_risk_stale_count": int(sum(int(diag.get("diplom_risk_stale_count", 0) or 0) for diag in active)),
        "diplom_risk_changed_count": int(sum(int(diag.get("diplom_risk_changed_count", 0) or 0) for diag in active)),
        "diplom_target_gross_before": float(
            sum(_safe_float(diag.get("diplom_target_gross_before"), 0.0) for diag in active)
        ),
        "diplom_target_gross_after": float(
            sum(_safe_float(diag.get("diplom_target_gross_after"), 0.0) for diag in active)
        ),
        "diplom_target_gross_reduction": float(
            sum(_safe_float(diag.get("diplom_target_gross_reduction"), 0.0) for diag in active)
        ),
        "diplom_target_gross_reduction_pct": float(
            np.average(
                np.asarray(
                    [_safe_float(diag.get("diplom_target_gross_reduction_pct"), 0.0) for diag in active],
                    dtype=np.float64,
                ),
                weights=np.asarray([max(_safe_float(diag.get("diplom_target_gross_before"), 0.0), 0.0) for diag in active])
                if any(_safe_float(diag.get("diplom_target_gross_before"), 0.0) > 0.0 for diag in active)
                else None,
            )
        )
        if active
        else 0.0,
        "diplom_target_cashout": bool(
            sum(_safe_float(diag.get("diplom_target_gross_before"), 0.0) for diag in active) > 0.0
            and sum(_safe_float(diag.get("diplom_target_gross_after"), 0.0) for diag in active) <= 1e-12
        ),
        "diplom_long_gross_before": float(sum(_safe_float(diag.get("diplom_long_gross_before"), 0.0) for diag in active)),
        "diplom_long_gross_after": float(sum(_safe_float(diag.get("diplom_long_gross_after"), 0.0) for diag in active)),
        "diplom_short_gross_before": float(sum(_safe_float(diag.get("diplom_short_gross_before"), 0.0) for diag in active)),
        "diplom_short_gross_after": float(sum(_safe_float(diag.get("diplom_short_gross_after"), 0.0) for diag in active)),
        "diplom_long_p_high_exposure": _weighted_average(long_values, long_weights),
        "diplom_short_p_high_exposure": _weighted_average(short_values, short_weights),
        "diplom_availability_min_age_days": float(min(age_values)) if age_values else np.nan,
    }


class DiplomRiskAdapter:
    def __init__(self, config: DiplomRiskConfig, predictions: pd.DataFrame):
        if config.missing_policy not in {"neutral", "cash"}:
            raise ValueError("diplom_missing_policy must be 'neutral' or 'cash'")
        if config.stale_days < 0:
            raise ValueError("diplom_stale_days must be non-negative")
        if config.risk_weight_pct < 0.0:
            raise ValueError("diplom_risk_weight_pct must be non-negative")
        if config.short_bonus_weight_pct < 0.0:
            raise ValueError("diplom_short_bonus_weight_pct must be non-negative")
        if config.candidate_score_penalty_pct < 0.0:
            raise ValueError("diplom_candidate_score_penalty_pct must be non-negative")
        if config.gate_penalty_pct < 0.0:
            raise ValueError("diplom_gate_penalty_pct must be non-negative")
        if config.long_p_high_cap is not None and not 0.0 <= float(config.long_p_high_cap) <= 1.0:
            raise ValueError("diplom_long_p_high_cap must be within [0, 1]")
        self.config = config
        self.predictions = self._normalize_predictions(predictions)
        self.source_columns = list(predictions.columns)
        self.ignored_forbidden_columns = sorted(FORBIDDEN_INFERENCE_COLUMNS.intersection(predictions.columns))
        self._by_ticker = {
            ticker: group.sort_values("availability_timestamp").reset_index(drop=True)
            for ticker, group in self.predictions.groupby("ticker", sort=False)
        }
        self._mapping_effective_date = pd.Timestamp(config.yndx_ydex_effective_date)

    @classmethod
    def from_config(cls, config: DiplomRiskConfig) -> "DiplomRiskAdapter | None":
        if not config.enabled:
            return None
        if config.predictions_path is None:
            raise ValueError("diplom_risk_enabled requires diplom_risk_predictions_path")
        if not config.predictions_path.exists():
            raise FileNotFoundError(f"diplom risk predictions not found: {config.predictions_path}")
        predictions = pd.read_csv(config.predictions_path)
        return cls(config, predictions)

    def _normalize_predictions(self, predictions: pd.DataFrame) -> pd.DataFrame:
        required = {"decision_date", "ticker", "predicted_risk_class", "p_high"}
        missing = sorted(required.difference(predictions.columns))
        if missing:
            raise ValueError(f"diplom predictions missing required columns: {missing}")

        used = [column for column in SAFE_PREDICTION_COLUMNS if column in predictions.columns]
        data = predictions[used].copy()
        data["decision_date"] = pd.to_datetime(data["decision_date"], errors="coerce")
        if "availability_timestamp" in data.columns:
            data["availability_timestamp"] = pd.to_datetime(data["availability_timestamp"], errors="coerce")
        else:
            data["availability_timestamp"] = data["decision_date"]
        if "source_data_end" in data.columns:
            data["source_data_end"] = pd.to_datetime(data["source_data_end"], errors="coerce")
        else:
            data["source_data_end"] = data["decision_date"]
        data["ticker"] = data["ticker"].astype(str).str.upper().str.strip()
        if "kronos_symbol" in data.columns:
            data["kronos_symbol"] = data["kronos_symbol"].astype(str).str.upper().str.strip()
        data["predicted_risk_class"] = data["predicted_risk_class"].astype(str).str.lower().str.strip()
        for column in ["p_low", "p_medium", "p_high"]:
            if column not in data.columns:
                data[column] = np.nan
            data[column] = pd.to_numeric(data[column], errors="coerce")
            invalid = data[column].notna() & ((data[column] < 0.0) | (data[column] > 1.0))
            if invalid.any():
                raise ValueError(f"diplom prediction probability {column} must be within [0, 1]")
        if "feature_coverage" in data.columns:
            data["feature_coverage"] = pd.to_numeric(data["feature_coverage"], errors="coerce")
        data = data.dropna(subset=["decision_date", "availability_timestamp", "ticker"])
        data = data.sort_values(["ticker", "availability_timestamp"]).drop_duplicates(
            ["ticker", "availability_timestamp"],
            keep="last",
        )
        return data.reset_index(drop=True)

    def _source_ticker(self, symbol: str, timestamp: pd.Timestamp) -> tuple[str, str | None]:
        symbol = str(symbol).upper().strip()
        if symbol in self._by_ticker:
            return symbol, None
        if (
            symbol == "YDEX"
            and self.config.enable_yndx_ydex_mapping
            and timestamp >= self._mapping_effective_date
            and "YNDX" in self._by_ticker
        ):
            return "YNDX", "mapped_yndx_to_ydex"
        if symbol == "YDEX" and not self.config.enable_yndx_ydex_mapping and "YNDX" in self._by_ticker:
            return symbol, "yndx_ydex_mapping_disabled"
        return symbol, None

    def lookup(self, symbol: str, timestamp: Any) -> dict[str, Any]:
        ts = pd.Timestamp(timestamp)
        source_ticker, mapping_reason = self._source_ticker(symbol, ts)
        group = self._by_ticker.get(source_ticker)
        if group is None or group.empty:
            return {
                "symbol": str(symbol),
                "diplom_ticker": source_ticker,
                "status": "missing",
                "reason": mapping_reason or "diplom_risk_missing",
            }

        dates = group["availability_timestamp"].to_numpy(dtype="datetime64[ns]")
        pos = int(np.searchsorted(dates, np.datetime64(ts), side="right") - 1)
        if pos < 0:
            return {
                "symbol": str(symbol),
                "diplom_ticker": source_ticker,
                "status": "missing",
                "reason": mapping_reason or "diplom_risk_missing",
            }

        row = group.iloc[pos]
        availability_ts = pd.Timestamp(row["availability_timestamp"])
        source_data_end = pd.Timestamp(row["source_data_end"]) if pd.notna(row.get("source_data_end")) else pd.Timestamp(row["decision_date"])
        age_days = max(0.0, (ts.normalize() - source_data_end.normalize()).total_seconds() / 86400.0)
        availability_age_days = max(0.0, (ts - availability_ts).total_seconds() / 86400.0)
        status = "ok" if age_days <= float(self.config.stale_days) else "stale"
        reason = "ok" if status == "ok" else "diplom_risk_stale"
        if mapping_reason:
            reason = f"{mapping_reason},{reason}" if reason != "ok" else mapping_reason
        return {
            "symbol": str(symbol),
            "diplom_ticker": source_ticker,
            "decision_date": pd.Timestamp(row["decision_date"]),
            "availability_timestamp": availability_ts,
            "source_data_end": source_data_end,
            "prediction_age_days": float(age_days),
            "availability_age_days": float(availability_age_days),
            "status": status,
            "reason": reason,
            "risk_class": str(row.get("predicted_risk_class", "")).lower(),
            "p_low": _safe_float(row.get("p_low")),
            "p_medium": _safe_float(row.get("p_medium")),
            "p_high": _safe_float(row.get("p_high")),
        }

    def lookup_many(self, symbols: list[str], timestamp: Any) -> pd.DataFrame:
        return pd.DataFrame([self.lookup(symbol, timestamp) for symbol in symbols])

    def apply_to_target(
        self,
        target: np.ndarray,
        *,
        symbols: list[str],
        timestamp: Any,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        adjusted = np.asarray(target, dtype=np.float32).copy()
        active_ids = np.flatnonzero(np.abs(adjusted) > 1e-12)
        target_gross_before = float(np.abs(adjusted).sum())
        long_gross_before = float(np.clip(adjusted, 0.0, None).sum())
        short_gross_before = float(np.abs(np.clip(adjusted, None, 0.0)).sum())
        if active_ids.size == 0:
            return adjusted, empty_diplom_summary("no_position")

        lookups = [self.lookup(symbols[int(idx)], timestamp) for idx in active_ids]
        unavailable = [item for item in lookups if item["status"] != "ok"]
        if unavailable and self.config.missing_policy == "cash":
            adjusted[:] = 0.0
            reasons = [item["reason"] for item in unavailable] + ["diplom_missing_policy_cash"]
            return adjusted, {
                **empty_diplom_summary(_join_reasons(reasons)),
                "diplom_risk_active_count": int(active_ids.size),
                "diplom_risk_missing_count": int(sum(item["status"] == "missing" for item in lookups)),
                "diplom_risk_stale_count": int(sum(item["status"] == "stale" for item in lookups)),
                "diplom_risk_changed_count": int(active_ids.size),
                "diplom_target_gross_before": target_gross_before,
                "diplom_target_gross_after": 0.0,
                "diplom_target_gross_reduction": target_gross_before,
                "diplom_target_gross_reduction_pct": 100.0 if target_gross_before > 0.0 else 0.0,
                "diplom_target_cashout": True,
                "diplom_long_gross_before": long_gross_before,
                "diplom_long_gross_after": 0.0,
                "diplom_short_gross_before": short_gross_before,
                "diplom_short_gross_after": 0.0,
            }

        reasons: list[str] = []
        available_count = 0
        missing_count = 0
        stale_count = 0
        changed_count = 0
        max_p_high = np.nan
        max_penalty_pct = 0.0
        worst_class: str | None = None
        long_p_high_values: list[float] = []
        long_p_high_weights: list[float] = []
        short_p_high_values: list[float] = []
        short_p_high_weights: list[float] = []
        availability_age_values: list[float] = []

        for idx, item in zip(active_ids, lookups, strict=True):
            status = item["status"]
            if status == "missing":
                missing_count += 1
                reasons.append(item["reason"])
                continue
            if status == "stale":
                stale_count += 1
                reasons.append(item["reason"])
                continue

            available_count += 1
            risk_class = str(item.get("risk_class", "")).lower()
            if risk_class in RISK_CLASS_ORDER and (
                worst_class is None or RISK_CLASS_ORDER[risk_class] > RISK_CLASS_ORDER[worst_class]
            ):
                worst_class = risk_class
            p_high = _safe_float(item.get("p_high"), 0.0)
            max_p_high = p_high if not np.isfinite(max_p_high) else max(max_p_high, p_high)
            weight = float(adjusted[int(idx)])
            availability_age = _safe_float(item.get("availability_age_days"))
            if np.isfinite(availability_age):
                availability_age_values.append(availability_age)
            if weight > 0.0:
                long_p_high_values.append(p_high)
                long_p_high_weights.append(abs(weight))
                veto = self.config.long_veto_p_high is not None and p_high >= float(self.config.long_veto_p_high)
                if veto:
                    adjusted[int(idx)] = 0.0
                    max_penalty_pct = 100.0
                    changed_count += 1
                    reasons.append("diplom_long_veto")
                else:
                    penalty_pct = max(0.0, float(self.config.risk_weight_pct) * p_high)
                    max_penalty_pct = max(max_penalty_pct, penalty_pct)
                    if penalty_pct > 0.0:
                        adjusted[int(idx)] = np.float32(weight * max(0.0, 1.0 - penalty_pct / 100.0))
                        if abs(float(adjusted[int(idx)]) - weight) > 1e-12:
                            changed_count += 1
                            reasons.append("diplom_long_risk_penalty")
            elif weight < 0.0:
                short_p_high_values.append(p_high)
                short_p_high_weights.append(abs(weight))
                if self.config.short_bonus_weight_pct > 0.0:
                    bonus_pct = float(self.config.short_bonus_weight_pct) * p_high
                    adjusted[int(idx)] = np.float32(weight * (1.0 + bonus_pct / 100.0))
                    if abs(float(adjusted[int(idx)]) - weight) > 1e-12:
                        changed_count += 1
                        reasons.append("diplom_short_risk_bonus")

        long_p_high_exposure = _weighted_average(long_p_high_values, long_p_high_weights)
        if (
            self.config.long_p_high_cap is not None
            and np.isfinite(long_p_high_exposure)
            and long_p_high_exposure > float(self.config.long_p_high_cap)
            and long_gross_before > 0.0
        ):
            scale = max(0.0, float(self.config.long_p_high_cap) / max(long_p_high_exposure, 1e-12))
            long_ids = np.flatnonzero(adjusted > 1e-12)
            if long_ids.size:
                before = adjusted[long_ids].copy()
                adjusted[long_ids] = (adjusted[long_ids] * scale).astype(np.float32)
                changed_count += int(np.count_nonzero(np.abs(adjusted[long_ids] - before) > 1e-12))
                max_penalty_pct = max(max_penalty_pct, (1.0 - scale) * 100.0)
                reasons.append("diplom_long_p_high_cap")

        target_gross_after = float(np.abs(adjusted).sum())
        long_gross_after = float(np.clip(adjusted, 0.0, None).sum())
        short_gross_after = float(np.abs(np.clip(adjusted, None, 0.0)).sum())
        gross_reduction = max(0.0, target_gross_before - target_gross_after)

        return adjusted, {
            "diplom_risk_available": available_count > 0,
            "diplom_risk_class": worst_class,
            "diplom_p_high": float(max_p_high) if np.isfinite(max_p_high) else np.nan,
            "diplom_risk_penalty_pct": float(max_penalty_pct),
            "diplom_risk_reason": _join_reasons(reasons),
            "diplom_risk_active_count": int(active_ids.size),
            "diplom_risk_available_count": int(available_count),
            "diplom_risk_missing_count": int(missing_count),
            "diplom_risk_stale_count": int(stale_count),
            "diplom_risk_changed_count": int(changed_count),
            "diplom_target_gross_before": target_gross_before,
            "diplom_target_gross_after": target_gross_after,
            "diplom_target_gross_reduction": gross_reduction,
            "diplom_target_gross_reduction_pct": (
                float(gross_reduction / target_gross_before * 100.0) if target_gross_before > 0.0 else 0.0
            ),
            "diplom_target_cashout": bool(target_gross_before > 0.0 and target_gross_after <= 1e-12),
            "diplom_long_gross_before": long_gross_before,
            "diplom_long_gross_after": long_gross_after,
            "diplom_short_gross_before": short_gross_before,
            "diplom_short_gross_after": short_gross_after,
            "diplom_long_p_high_exposure": long_p_high_exposure,
            "diplom_short_p_high_exposure": _weighted_average(short_p_high_values, short_p_high_weights),
            "diplom_availability_min_age_days": float(min(availability_age_values)) if availability_age_values else np.nan,
        }
