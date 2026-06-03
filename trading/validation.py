"""Alpha validation helpers for prediction contracts."""

from __future__ import annotations

import math
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from .runner import PredictionEvent


class AlphaValidationReport:
    """Validate prediction quality against realized closes."""

    def __init__(
        self,
        predictions,
        actuals,
        long_threshold=0.0,
        transaction_cost_bps=0.0,
        min_net_excess_return=None,
        min_directional_accuracy=None,
        min_active_period_fraction=None,
        min_observations=30,
        min_symbols=1,
        min_regimes=1,
    ):
        self.predictions = predictions
        self.actuals = pd.DataFrame(actuals)
        self.long_threshold = float(long_threshold)
        self.transaction_cost_bps = float(transaction_cost_bps)
        self.min_net_excess_return = min_net_excess_return
        self.min_directional_accuracy = min_directional_accuracy
        self.min_active_period_fraction = min_active_period_fraction
        self.min_observations = self._strict_int(min_observations, "min_observations")
        self.min_symbols = self._strict_int(min_symbols, "min_symbols")
        self.min_regimes = self._strict_int(min_regimes, "min_regimes")
        self._validate_parameters()

    def compute(self):
        predictions = self._prediction_frame(self.predictions)
        actuals = self._actual_frame(self.actuals)
        merged = predictions.merge(
            actuals,
            left_on=["symbol", "target_timestamp"],
            right_on=["symbol", "timestamp"],
            how="inner",
        ).sort_values(["target_timestamp", "symbol"])
        merged = merged.merge(
            actuals[["symbol", "timestamp", "close"]].rename(
                columns={"timestamp": "prediction_asof", "close": "reference_close"}
            ),
            on=["symbol", "prediction_asof"],
            how="left",
        )
        if merged.empty:
            raise ValueError("Predictions and actuals do not overlap by symbol/target_timestamp.")
        if merged["reference_close"].isna().any():
            raise ValueError("Missing as-of reference close for one or more predictions.")

        merged["error"] = merged["predicted_close"] - merged["close"]
        merged["absolute_error"] = merged["error"].abs()
        merged["baseline_error"] = (merged["reference_close"] - merged["close"]).abs()
        directional = merged.copy()
        directional["predicted_return"] = directional["predicted_close"] / directional["reference_close"] - 1.0
        directional["actual_return"] = directional["close"] / directional["reference_close"] - 1.0
        directional["direction_hit"] = (
            directional["predicted_return"].gt(0) == directional["actual_return"].gt(0)
        )
        directional["signal"] = directional["predicted_return"].gt(self.long_threshold).astype(int)
        portfolio_periods = self._portfolio_periods(directional)
        strategy_net_return = self._compound_return(portfolio_periods["strategy_period_return"])
        baseline_return = self._compound_return(portfolio_periods["baseline_period_return"])
        net_excess_return = strategy_net_return - baseline_return
        active_period_fraction = float(portfolio_periods["gross_exposure"].gt(0).mean())
        average_gross_exposure = float(portfolio_periods["gross_exposure"].mean())
        directional_accuracy = float(directional["direction_hit"].mean()) if not directional.empty else math.nan
        regime_metrics = self._monthly_regime_metrics(merged)
        failed_criteria = self._failed_criteria(
            net_excess_return,
            directional_accuracy,
            active_period_fraction,
            observations=len(merged),
            symbols=merged["symbol"].nunique(),
            regimes=len(regime_metrics),
        )

        return {
            "observations": int(len(merged)),
            "symbols": int(merged["symbol"].nunique()),
            "start": merged["target_timestamp"].min(),
            "end": merged["target_timestamp"].max(),
            "mae": float(merged["absolute_error"].mean()),
            "baseline_mae": float(merged["baseline_error"].mean()),
            "mae_improvement": float(1.0 - merged["absolute_error"].mean() / merged["baseline_error"].mean())
            if merged["baseline_error"].mean() > 0
            else math.nan,
            "directional_accuracy": directional_accuracy,
            "information_coefficient": self._spearman(directional),
            "strategy_net_return": strategy_net_return,
            "baseline_return": baseline_return,
            "net_excess_return": net_excess_return,
            "active_period_fraction": active_period_fraction,
            "average_gross_exposure": average_gross_exposure,
            "turnover": float(portfolio_periods["turnover"].sum()),
            "accepted": not failed_criteria,
            "failed_criteria": failed_criteria,
            "criteria": {
                "long_threshold": self.long_threshold,
                "transaction_cost_bps": self.transaction_cost_bps,
                "min_net_excess_return": self.min_net_excess_return,
                "min_directional_accuracy": self.min_directional_accuracy,
                "min_active_period_fraction": self.min_active_period_fraction,
                "min_observations": self.min_observations,
                "min_symbols": self.min_symbols,
                "min_regimes": self.min_regimes,
            },
            "regime_metrics": regime_metrics,
        }

    @classmethod
    def _prediction_frame(cls, predictions):
        records = []
        for prediction in predictions:
            event = prediction if isinstance(prediction, PredictionEvent) else PredictionEvent.from_mapping(prediction)
            event.validate()
            records.append(
                {
                    "symbol": event.symbol,
                    "prediction_asof": event.prediction_asof,
                    "execution_timestamp": event.execution_timestamp,
                    "target_timestamp": event.target_timestamp,
                    "predicted_close": event.predicted_close,
                }
            )
        df = pd.DataFrame(records)
        if df.empty:
            raise ValueError("predictions are empty.")
        if df.duplicated(["symbol", "target_timestamp"]).any():
            raise ValueError("Duplicate predictions for symbol/target_timestamp.")
        return df

    @staticmethod
    def _actual_frame(actuals):
        if actuals.empty:
            raise ValueError("actuals are empty.")
        missing = {"symbol", "timestamp", "close"} - set(actuals.columns)
        if missing:
            raise ValueError(f"actuals are missing required columns: {sorted(missing)}")
        df = actuals.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["close"] = df["close"].astype(float)
        if not np.isfinite(df["close"]).all() or not df["close"].gt(0).all():
            raise ValueError("actual close values must be finite positive numbers.")
        if df.duplicated(["symbol", "timestamp"]).any():
            raise ValueError("Duplicate actual bars for symbol/timestamp.")
        return df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    @staticmethod
    def _spearman(directional):
        if directional.empty or len(directional) < 2:
            return math.nan
        if directional["predicted_return"].nunique() < 2 or directional["actual_return"].nunique() < 2:
            return math.nan
        return float(directional["predicted_return"].corr(directional["actual_return"], method="spearman"))

    @staticmethod
    def _monthly_regime_metrics(merged):
        frame = merged.copy()
        frame["month"] = frame["target_timestamp"].dt.to_period("M").astype(str)
        metrics = []
        for month, group in frame.groupby("month", sort=True):
            metrics.append(
                {
                    "month": month,
                    "observations": int(len(group)),
                    "mae": float(group["absolute_error"].mean()),
                    "baseline_mae": float(group["baseline_error"].mean()),
                }
            )
        return metrics

    def _portfolio_periods(self, directional):
        rows = []
        previous_weights: dict[str, float] = {}
        cost_rate = self.transaction_cost_bps / 10000.0
        for timestamp, group in directional.groupby("target_timestamp", sort=True):
            signals = {row.symbol: int(row.signal) for row in group.itertuples()}
            active_symbols = [symbol for symbol, signal in signals.items() if signal > 0]
            if active_symbols:
                weight = 1.0 / len(active_symbols)
                weights = {symbol: weight for symbol in active_symbols}
                gross_return = float(
                    group[group["symbol"].isin(active_symbols)]["actual_return"].mean()
                )
            else:
                weights = {}
                gross_return = 0.0
            symbols = set(previous_weights) | set(weights)
            turnover = sum(abs(weights.get(symbol, 0.0) - previous_weights.get(symbol, 0.0)) for symbol in symbols)
            rows.append(
                {
                    "target_timestamp": timestamp,
                    "strategy_period_return": gross_return - turnover * cost_rate,
                    "baseline_period_return": float(group["actual_return"].mean()),
                    "turnover": turnover,
                    "active_positions": len(active_symbols),
                    "gross_exposure": sum(abs(weight) for weight in weights.values()),
                }
            )
            previous_weights = weights
        return pd.DataFrame(rows)

    @staticmethod
    def _compound_return(period_returns):
        if period_returns.empty:
            return math.nan
        return float((1.0 + period_returns).prod() - 1.0)

    def _failed_criteria(
        self,
        net_excess_return,
        directional_accuracy,
        active_period_fraction,
        observations,
        symbols,
        regimes,
    ):
        failed = []
        if observations < self.min_observations:
            failed.append(
                {
                    "metric": "observations",
                    "minimum": self.min_observations,
                    "actual": int(observations),
                }
            )
        if symbols < self.min_symbols:
            failed.append(
                {
                    "metric": "symbols",
                    "minimum": self.min_symbols,
                    "actual": int(symbols),
                }
            )
        if regimes < self.min_regimes:
            failed.append(
                {
                    "metric": "regimes",
                    "minimum": self.min_regimes,
                    "actual": int(regimes),
                }
            )
        if self.min_net_excess_return is None:
            failed.append(
                {
                    "metric": "min_net_excess_return",
                    "minimum": "explicit threshold",
                    "actual": "not configured",
                }
            )
        if self.min_directional_accuracy is None:
            failed.append(
                {
                    "metric": "min_directional_accuracy",
                    "minimum": "explicit threshold",
                    "actual": "not configured",
                }
            )
        if self.min_active_period_fraction is None:
            failed.append(
                {
                    "metric": "min_active_period_fraction",
                    "minimum": "explicit threshold",
                    "actual": "not configured",
                }
            )
        if self.min_net_excess_return is not None and net_excess_return < self.min_net_excess_return:
            failed.append(
                {
                    "metric": "net_excess_return",
                    "minimum": float(self.min_net_excess_return),
                    "actual": float(net_excess_return),
                }
            )
        if (
            self.min_directional_accuracy is not None
            and directional_accuracy < self.min_directional_accuracy
        ):
            failed.append(
                {
                    "metric": "directional_accuracy",
                    "minimum": float(self.min_directional_accuracy),
                    "actual": float(directional_accuracy),
                }
            )
        if (
            self.min_active_period_fraction is not None
            and active_period_fraction < self.min_active_period_fraction
        ):
            failed.append(
                {
                    "metric": "active_period_fraction",
                    "minimum": float(self.min_active_period_fraction),
                    "actual": float(active_period_fraction),
                }
            )
        return failed

    def _validate_parameters(self):
        if not math.isfinite(self.long_threshold):
            raise ValueError("long_threshold must be finite.")
        if not math.isfinite(self.transaction_cost_bps) or self.transaction_cost_bps < 0:
            raise ValueError("transaction_cost_bps must be a finite non-negative number.")
        for name, value in (
            ("min_net_excess_return", self.min_net_excess_return),
            ("min_directional_accuracy", self.min_directional_accuracy),
            ("min_active_period_fraction", self.min_active_period_fraction),
        ):
            if value is not None and not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite when provided.")
        if self.min_directional_accuracy is not None and not 0 <= float(self.min_directional_accuracy) <= 1:
            raise ValueError("min_directional_accuracy must be in [0, 1].")
        if self.min_active_period_fraction is not None and not 0 <= float(self.min_active_period_fraction) <= 1:
            raise ValueError("min_active_period_fraction must be in [0, 1].")
        if self.min_observations < 1:
            raise ValueError("min_observations must be at least 1.")
        if self.min_symbols < 1:
            raise ValueError("min_symbols must be at least 1.")
        if self.min_regimes < 1:
            raise ValueError("min_regimes must be at least 1.")
        if self.min_net_excess_return is not None:
            self.min_net_excess_return = float(self.min_net_excess_return)
        if self.min_directional_accuracy is not None:
            self.min_directional_accuracy = float(self.min_directional_accuracy)
        if self.min_active_period_fraction is not None:
            self.min_active_period_fraction = float(self.min_active_period_fraction)

    @staticmethod
    def _strict_int(value, name):
        if isinstance(value, bool):
            raise ValueError(f"{name} must be an integer.")
        if isinstance(value, float) and not value.is_integer():
            raise ValueError(f"{name} must be an integer.")
        converted = int(value)
        if str(value).strip() not in {str(converted), f"{converted}.0"}:
            raise ValueError(f"{name} must be an integer.")
        return converted


class ModelApprovalRegistry:
    """Durable model_hash -> accepted validation report registry."""

    PRODUCTION_MIN_NET_EXCESS_RETURN = 0.01
    PRODUCTION_MIN_DIRECTIONAL_ACCURACY = 0.52
    PRODUCTION_MIN_ACTIVE_PERIOD_FRACTION = 0.10
    PRODUCTION_MIN_OBSERVATIONS = 1000
    PRODUCTION_MIN_SYMBOLS = 6
    PRODUCTION_MIN_REGIMES = 12

    CODE_MANIFEST_PATHS = (
        "trading/validation.py",
        "trading/runner.py",
        "trading/paper.py",
        "trading/live.py",
        "trading/engine.py",
        "trading/ops.py",
        "trading/evaluation.py",
    )
    REQUIRED_METADATA = {
        "prediction_file_checksum",
        "prediction_file_path",
        "actuals_file_checksum",
        "actuals_file_path",
        "oos_start",
        "oos_end",
        "universe",
        "model_hash",
        "model_revision",
        "tokenizer_revision",
        "code_version",
        "code_manifest",
        "validation_code_checksum",
    }

    def __init__(
        self,
        path,
        min_net_excess_return=PRODUCTION_MIN_NET_EXCESS_RETURN,
        min_directional_accuracy=PRODUCTION_MIN_DIRECTIONAL_ACCURACY,
        min_active_period_fraction=PRODUCTION_MIN_ACTIVE_PERIOD_FRACTION,
        min_observations=PRODUCTION_MIN_OBSERVATIONS,
        min_symbols=PRODUCTION_MIN_SYMBOLS,
        min_regimes=PRODUCTION_MIN_REGIMES,
        required_metadata=None,
        code_version=None,
        validation_code_path=None,
        code_manifest_paths=None,
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.min_net_excess_return = float(min_net_excess_return)
        self.min_directional_accuracy = float(min_directional_accuracy)
        self.min_active_period_fraction = float(min_active_period_fraction)
        self.min_observations = self._strict_positive_int(min_observations, "min_observations")
        self.min_symbols = self._strict_positive_int(min_symbols, "min_symbols")
        self.min_regimes = self._strict_positive_int(min_regimes, "min_regimes")
        if not math.isfinite(self.min_net_excess_return):
            raise ValueError("min_net_excess_return must be finite.")
        if not math.isfinite(self.min_directional_accuracy) or not 0 <= self.min_directional_accuracy <= 1:
            raise ValueError("min_directional_accuracy must be in [0, 1].")
        if not math.isfinite(self.min_active_period_fraction) or not 0 <= self.min_active_period_fraction <= 1:
            raise ValueError("min_active_period_fraction must be in [0, 1].")
        self.required_metadata = set(required_metadata or self.REQUIRED_METADATA)
        self.code_version = str(code_version or self.current_code_version())
        self.validation_code_path = Path(validation_code_path or __file__)
        self.validation_code_checksum = self.file_checksum(self.validation_code_path)
        self.code_manifest_paths = tuple(code_manifest_paths or self.CODE_MANIFEST_PATHS)
        self.code_manifest = self.current_code_manifest(self.code_manifest_paths)

    def meets_production_floors(self):
        return (
            self.min_net_excess_return >= self.PRODUCTION_MIN_NET_EXCESS_RETURN
            and self.min_directional_accuracy >= self.PRODUCTION_MIN_DIRECTIONAL_ACCURACY
            and self.min_active_period_fraction >= self.PRODUCTION_MIN_ACTIVE_PERIOD_FRACTION
            and self.min_observations >= self.PRODUCTION_MIN_OBSERVATIONS
            and self.min_symbols >= self.PRODUCTION_MIN_SYMBOLS
            and self.min_regimes >= self.PRODUCTION_MIN_REGIMES
        )

    def approve(self, model_hash, report, metadata=None):
        self._validate_report_schema(
            report,
            self.min_net_excess_return,
            self.min_directional_accuracy,
            self.min_active_period_fraction,
            self.min_observations,
            self.min_symbols,
            self.min_regimes,
        )
        if not report.get("accepted"):
            raise ValueError("Only accepted validation reports can be registered.")
        if report.get("failed_criteria"):
            raise ValueError("Validation report with failed criteria cannot be registered.")
        metadata = metadata or {}
        missing_metadata = self.required_metadata - set(metadata)
        if missing_metadata:
            raise ValueError(f"Approval metadata missing required fields: {sorted(missing_metadata)}")
        self._validate_metadata(metadata, expected_model_hash=str(model_hash))
        payload = self._read()
        record = {
            "model_hash": str(model_hash),
            "report": self._json_safe(report),
            "metadata": self._json_safe(metadata),
        }
        record["approval_checksum"] = self._checksum(record)
        payload[str(model_hash)] = record
        self.path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return record

    def accepted_model_hashes(self):
        return set(self._read_verified())

    def require(self, model_hash):
        payload = self._read_verified()
        if model_hash not in payload:
            raise ValueError(f"model_hash {model_hash!r} is not approved.")
        return payload[model_hash]

    def _read(self):
        if not self.path.exists():
            return {}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _read_verified(self):
        payload = self._read()
        for model_hash, record in payload.items():
            if record.get("model_hash") != model_hash:
                raise ValueError("Approval registry model_hash key mismatch.")
            expected = record.get("approval_checksum")
            if not expected:
                raise ValueError("Approval registry record missing approval_checksum.")
            checksum_payload = dict(record)
            checksum_payload.pop("approval_checksum", None)
            actual = self._checksum(checksum_payload)
            if actual != expected:
                raise ValueError("Approval registry checksum mismatch.")
            missing_metadata = self.required_metadata - set(record.get("metadata", {}))
            if missing_metadata:
                raise ValueError(f"Approval metadata missing required fields: {sorted(missing_metadata)}")
            self._validate_metadata(record.get("metadata", {}), expected_model_hash=model_hash)
            self._validate_report_schema(
                record.get("report", {}),
                self.min_net_excess_return,
                self.min_directional_accuracy,
                self.min_active_period_fraction,
                self.min_observations,
                self.min_symbols,
                self.min_regimes,
            )
            if not record["report"].get("accepted") or record["report"].get("failed_criteria"):
                raise ValueError("Approval registry contains an unaccepted report.")
        return payload

    @classmethod
    def _checksum(cls, value):
        canonical = json.dumps(cls._json_safe(value), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @staticmethod
    def _validate_report_schema(
        report,
        min_net_excess_return=0.0,
        min_directional_accuracy=0.5,
        min_active_period_fraction=0.0,
        min_observations=1,
        min_symbols=1,
        min_regimes=1,
    ):
        required = {
            "accepted",
            "failed_criteria",
            "criteria",
            "observations",
            "symbols",
            "start",
            "end",
            "strategy_net_return",
            "baseline_return",
            "net_excess_return",
            "directional_accuracy",
            "active_period_fraction",
            "average_gross_exposure",
            "regime_metrics",
        }
        missing = required - set(report)
        if missing:
            raise ValueError(f"Validation report is missing required fields: {sorted(missing)}")
        if not isinstance(report["accepted"], bool):
            raise ValueError("Validation report accepted must be a boolean.")
        if not isinstance(report["failed_criteria"], list):
            raise ValueError("Validation report failed_criteria must be a list.")
        if not isinstance(report["criteria"], dict):
            raise ValueError("Validation report criteria must be a dict.")
        observations = ModelApprovalRegistry._strict_positive_int(report["observations"], "observations")
        symbols = ModelApprovalRegistry._strict_positive_int(report["symbols"], "symbols")
        if observations <= 0:
            raise ValueError("Validation report observations must be positive.")
        if symbols <= 0:
            raise ValueError("Validation report symbols must be positive.")
        try:
            start = pd.Timestamp(report["start"])
            end = pd.Timestamp(report["end"])
        except (TypeError, ValueError) as exc:
            raise ValueError("Validation report start/end must be valid timestamps.") from exc
        if pd.isna(start) or pd.isna(end):
            raise ValueError("Validation report start/end must be valid timestamps.")
        if start > end:
            raise ValueError("Validation report start must be before or equal to end.")
        if not isinstance(report["regime_metrics"], list) or not report["regime_metrics"]:
            raise ValueError("Validation report must include non-empty regime_metrics.")
        if any(not isinstance(item, dict) for item in report["regime_metrics"]):
            raise ValueError("Validation report regime_metrics must contain dict entries.")
        regimes = len(report["regime_metrics"])
        for field in (
            "strategy_net_return",
            "baseline_return",
            "net_excess_return",
            "directional_accuracy",
            "active_period_fraction",
            "average_gross_exposure",
        ):
            value = float(report[field])
            if not math.isfinite(value):
                raise ValueError(f"Validation report {field} must be finite.")
        if not 0 <= float(report["directional_accuracy"]) <= 1:
            raise ValueError("Validation report directional_accuracy must be in [0, 1].")
        if not 0 <= float(report["active_period_fraction"]) <= 1:
            raise ValueError("Validation report active_period_fraction must be in [0, 1].")
        if not 0 <= float(report["average_gross_exposure"]) <= 1:
            raise ValueError("Validation report average_gross_exposure must be in [0, 1].")
        criteria = report["criteria"]
        for field in (
            "min_net_excess_return",
            "min_directional_accuracy",
            "min_active_period_fraction",
            "min_observations",
            "min_symbols",
            "min_regimes",
        ):
            if criteria.get(field) is None:
                raise ValueError(f"Validation report criteria.{field} is required.")
        criteria_observations = ModelApprovalRegistry._strict_positive_int(
            criteria["min_observations"],
            "criteria.min_observations",
        )
        criteria_symbols = ModelApprovalRegistry._strict_positive_int(
            criteria["min_symbols"],
            "criteria.min_symbols",
        )
        criteria_regimes = ModelApprovalRegistry._strict_positive_int(
            criteria["min_regimes"],
            "criteria.min_regimes",
        )
        criteria_net_excess = float(criteria["min_net_excess_return"])
        criteria_directional_accuracy = float(criteria["min_directional_accuracy"])
        criteria_active_period_fraction = float(criteria["min_active_period_fraction"])
        if not math.isfinite(criteria_net_excess):
            raise ValueError("Validation criteria min_net_excess_return must be finite.")
        if not math.isfinite(criteria_directional_accuracy) or not 0 <= criteria_directional_accuracy <= 1:
            raise ValueError("Validation criteria min_directional_accuracy must be in [0, 1].")
        if not math.isfinite(criteria_active_period_fraction) or not 0 <= criteria_active_period_fraction <= 1:
            raise ValueError("Validation criteria min_active_period_fraction must be in [0, 1].")
        if criteria_net_excess < float(min_net_excess_return):
            raise ValueError("Validation criteria min_net_excess_return is below registry floor.")
        if criteria_directional_accuracy < float(min_directional_accuracy):
            raise ValueError("Validation criteria min_directional_accuracy is below registry floor.")
        if criteria_active_period_fraction < float(min_active_period_fraction):
            raise ValueError("Validation criteria min_active_period_fraction is below registry floor.")
        if criteria_observations < int(min_observations):
            raise ValueError("Validation criteria min_observations is below registry floor.")
        if criteria_symbols < int(min_symbols):
            raise ValueError("Validation criteria min_symbols is below registry floor.")
        if criteria_regimes < int(min_regimes):
            raise ValueError("Validation criteria min_regimes is below registry floor.")
        if float(report["net_excess_return"]) < criteria_net_excess:
            raise ValueError("Validation report net_excess_return does not satisfy criteria.")
        if float(report["directional_accuracy"]) < criteria_directional_accuracy:
            raise ValueError("Validation report directional_accuracy does not satisfy criteria.")
        if float(report["active_period_fraction"]) < criteria_active_period_fraction:
            raise ValueError("Validation report active_period_fraction does not satisfy criteria.")
        if observations < criteria_observations:
            raise ValueError("Validation report observations do not satisfy criteria.")
        if symbols < criteria_symbols:
            raise ValueError("Validation report symbols do not satisfy criteria.")
        if regimes < criteria_regimes:
            raise ValueError("Validation report regimes do not satisfy criteria.")

    @staticmethod
    def _strict_positive_int(value, name):
        if isinstance(value, bool):
            raise ValueError(f"Validation report {name} must be an integer.")
        if isinstance(value, float) and not value.is_integer():
            raise ValueError(f"Validation report {name} must be an integer.")
        try:
            converted = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Validation report {name} must be an integer.") from exc
        if str(value).strip() not in {str(converted), f"{converted}.0"}:
            raise ValueError(f"Validation report {name} must be an integer.")
        return converted

    def _validate_metadata(self, metadata, expected_model_hash=None):
        for key, value in metadata.items():
            if value is None:
                raise ValueError(f"Approval metadata {key} must be populated.")
            if isinstance(value, str) and not value.strip():
                raise ValueError(f"Approval metadata {key} must be populated.")
        universe = metadata.get("universe")
        if not isinstance(universe, list) or not universe:
            raise ValueError("Approval metadata universe must be a non-empty list.")
        if any(not isinstance(symbol, str) or not symbol.strip() for symbol in universe):
            raise ValueError("Approval metadata universe entries must be populated strings.")
        if expected_model_hash is not None and str(metadata.get("model_hash")) != str(expected_model_hash):
            raise ValueError("Approval metadata model_hash does not match approved model_hash.")
        self._validate_metadata_file_checksum(metadata, "prediction_file_path", "prediction_file_checksum")
        self._validate_metadata_file_checksum(metadata, "actuals_file_path", "actuals_file_checksum")
        if "code_version" in self.required_metadata or "code_version" in metadata:
            if str(metadata.get("code_version")) != self.code_version:
                raise ValueError("Approval metadata code_version does not match current code version.")
        if "validation_code_checksum" in self.required_metadata or "validation_code_checksum" in metadata:
            current_validation_checksum = self.file_checksum(self.validation_code_path)
            if str(metadata.get("validation_code_checksum")) != current_validation_checksum:
                raise ValueError("Approval metadata validation_code_checksum does not match current validation code.")
        if "code_manifest" in self.required_metadata or "code_manifest" in metadata:
            current_manifest = self.current_code_manifest(self.code_manifest_paths)
            if metadata.get("code_manifest") != current_manifest:
                raise ValueError("Approval metadata code_manifest does not match current trading code manifest.")

    def _validate_metadata_file_checksum(self, metadata, path_field, checksum_field):
        if path_field not in self.required_metadata and path_field not in metadata and checksum_field not in metadata:
            return
        path_value = metadata.get(path_field)
        checksum_value = metadata.get(checksum_field)
        if not path_value or not checksum_value:
            raise ValueError(f"Approval metadata {path_field}/{checksum_field} must be populated.")
        actual = self.file_checksum(path_value)
        if actual != str(checksum_value):
            raise ValueError(f"Approval metadata {checksum_field} does not match {path_field}.")

    @staticmethod
    def file_checksum(path):
        path = Path(path).expanduser()
        if not path.is_file():
            raise ValueError(f"Approval metadata file does not exist: {path}.")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @classmethod
    def current_code_version(cls):
        env_version = os.environ.get("KRONOS_CODE_VERSION")
        if env_version:
            return env_version
        repo_root = Path(__file__).resolve().parents[1]
        try:
            revision = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            status = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return f"validation:{cls.file_checksum(Path(__file__))}"
        return f"{revision}{'-dirty' if status else ''}"

    @classmethod
    def current_code_manifest(cls, module_paths=None):
        repo_root = Path(__file__).resolve().parents[1]
        manifest = {}
        for module_path in module_paths or cls.CODE_MANIFEST_PATHS:
            normalized = str(module_path).replace("\\", "/")
            manifest[normalized] = cls.file_checksum(repo_root / normalized)
        return manifest

    @classmethod
    def _json_safe(cls, value):
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        if isinstance(value, dict):
            return {key: cls._json_safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [cls._json_safe(item) for item in value]
        if isinstance(value, float) and math.isnan(value):
            return None
        return value
