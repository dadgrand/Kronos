import json

import pandas as pd
import pytest

from trading.validation import AlphaValidationReport, ModelApprovalRegistry


def approval_metadata(tmp_path, registry):
    prediction_file = tmp_path / "predictions.json"
    actuals_file = tmp_path / "actuals.json"
    prediction_file.write_text(json.dumps({"prediction_results": []}), encoding="utf-8")
    actuals_file.write_text(json.dumps({"actuals": []}), encoding="utf-8")
    return {
        "prediction_file_path": str(prediction_file),
        "prediction_file_checksum": ModelApprovalRegistry.file_checksum(prediction_file),
        "actuals_file_path": str(actuals_file),
        "actuals_file_checksum": ModelApprovalRegistry.file_checksum(actuals_file),
        "oos_start": "2024-01-01",
        "oos_end": "2024-01-31",
        "universe": ["AAA"],
        "model_hash": "abc123",
        "model_revision": "model-rev",
        "tokenizer_revision": "tokenizer-rev",
        "code_version": registry.code_version,
        "code_manifest": registry.code_manifest,
        "validation_code_checksum": registry.validation_code_checksum,
    }


def prediction(symbol, asof, target, predicted_close):
    asof = pd.Timestamp(asof)
    target = pd.Timestamp(target)
    return {
        "symbol": symbol,
        "prediction_asof": asof,
        "execution_timestamp": target,
        "target_timestamp": target,
        "features_cutoff": asof,
        "horizon": str(target - asof),
        "model_version": "unit-test-model",
        "model_hash": "abc123",
        "predicted_close": predicted_close,
    }


def test_alpha_validation_report_scores_predictions_against_baseline():
    report = AlphaValidationReport(
        predictions=[
            prediction("AAA", "2024-01-01", "2024-01-02", 101.0),
            prediction("AAA", "2024-01-02", "2024-01-03", 103.0),
            prediction("AAA", "2024-01-03", "2024-01-04", 106.0),
        ],
        actuals=[
            {"symbol": "AAA", "timestamp": "2024-01-01", "close": 100.0},
            {"symbol": "AAA", "timestamp": "2024-01-02", "close": 101.0},
            {"symbol": "AAA", "timestamp": "2024-01-03", "close": 104.0},
            {"symbol": "AAA", "timestamp": "2024-01-04", "close": 105.0},
        ],
        min_net_excess_return=0.0,
        min_directional_accuracy=0.5,
        min_active_period_fraction=0.5,
        min_observations=1,
    ).compute()

    assert report["observations"] == 3
    assert report["symbols"] == 1
    assert report["mae"] < report["baseline_mae"]
    assert report["directional_accuracy"] == pytest.approx(1.0)
    assert report["strategy_net_return"] > 0
    assert report["active_period_fraction"] >= 0.5
    assert report["accepted"] is True
    assert report["regime_metrics"][0]["month"] == "2024-01"


def test_alpha_validation_fails_closed_without_acceptance_thresholds():
    report = AlphaValidationReport(
        predictions=[prediction("AAA", "2024-01-01", "2024-01-02", 90.0)],
        actuals=[
            {"symbol": "AAA", "timestamp": "2024-01-01", "close": 100.0},
            {"symbol": "AAA", "timestamp": "2024-01-02", "close": 101.0},
        ],
        min_observations=1,
    ).compute()

    assert report["accepted"] is False
    assert {item["metric"] for item in report["failed_criteria"]} == {
        "min_net_excess_return",
        "min_directional_accuracy",
        "min_active_period_fraction",
    }


def test_alpha_validation_requires_full_acceptance_threshold_set():
    report = AlphaValidationReport(
        predictions=[prediction("AAA", "2024-01-01", "2024-01-02", 90.0)],
        actuals=[
            {"symbol": "AAA", "timestamp": "2024-01-01", "close": 100.0},
            {"symbol": "AAA", "timestamp": "2024-01-02", "close": 101.0},
        ],
        min_net_excess_return=0.0,
        min_observations=1,
    ).compute()

    assert report["accepted"] is False
    assert any(item["metric"] == "min_directional_accuracy" for item in report["failed_criteria"])
    assert any(item["metric"] == "min_active_period_fraction" for item in report["failed_criteria"])


def test_alpha_validation_rejects_prediction_contract_leakage():
    bad_prediction = prediction("AAA", "2024-01-03", "2024-01-02", 101.0)

    with pytest.raises(ValueError, match="prediction_asof"):
        AlphaValidationReport(
            predictions=[bad_prediction],
            actuals=[{"symbol": "AAA", "timestamp": "2024-01-02", "close": 101.0}],
        ).compute()


def test_alpha_validation_rejects_duplicate_prediction_targets():
    duplicate = prediction("AAA", "2024-01-01", "2024-01-02", 101.0)

    with pytest.raises(ValueError, match="Duplicate predictions"):
        AlphaValidationReport(
            predictions=[duplicate, duplicate],
            actuals=[
                {"symbol": "AAA", "timestamp": "2024-01-01", "close": 100.0},
                {"symbol": "AAA", "timestamp": "2024-01-02", "close": 101.0},
            ],
        ).compute()


def test_alpha_validation_requires_asof_reference_close():
    with pytest.raises(ValueError, match="reference close"):
        AlphaValidationReport(
            predictions=[prediction("AAA", "2024-01-01", "2024-01-03", 103.0)],
            actuals=[{"symbol": "AAA", "timestamp": "2024-01-03", "close": 103.0}],
        ).compute()


def test_alpha_validation_rejects_non_positive_actual_close():
    with pytest.raises(ValueError, match="finite positive"):
        AlphaValidationReport(
            predictions=[prediction("AAA", "2024-01-01", "2024-01-02", 103.0)],
            actuals=[
                {"symbol": "AAA", "timestamp": "2024-01-01", "close": 100.0},
                {"symbol": "AAA", "timestamp": "2024-01-02", "close": 0.0},
            ],
        ).compute()


def test_alpha_validation_applies_net_return_acceptance_thresholds():
    report = AlphaValidationReport(
        predictions=[
            prediction("AAA", "2024-01-01", "2024-01-02", 90.0),
            prediction("AAA", "2024-01-02", "2024-01-03", 90.0),
        ],
        actuals=[
            {"symbol": "AAA", "timestamp": "2024-01-01", "close": 100.0},
            {"symbol": "AAA", "timestamp": "2024-01-02", "close": 101.0},
            {"symbol": "AAA", "timestamp": "2024-01-03", "close": 102.0},
        ],
        min_net_excess_return=0.0,
        min_directional_accuracy=0.5,
        min_active_period_fraction=0.0,
        min_observations=1,
    ).compute()

    assert report["accepted"] is False
    assert {item["metric"] for item in report["failed_criteria"]} == {
        "net_excess_return",
        "directional_accuracy",
    }


def test_alpha_validation_rejects_inactive_cash_alpha_in_falling_market():
    report = AlphaValidationReport(
        predictions=[
            prediction("AAA", "2024-01-01", "2024-01-02", 90.0),
            prediction("AAA", "2024-01-02", "2024-01-03", 80.0),
        ],
        actuals=[
            {"symbol": "AAA", "timestamp": "2024-01-01", "close": 100.0},
            {"symbol": "AAA", "timestamp": "2024-01-02", "close": 95.0},
            {"symbol": "AAA", "timestamp": "2024-01-03", "close": 90.0},
        ],
        long_threshold=0.0,
        min_net_excess_return=0.01,
        min_directional_accuracy=0.5,
        min_active_period_fraction=0.5,
        min_observations=1,
    ).compute()

    assert report["directional_accuracy"] == pytest.approx(1.0)
    assert report["net_excess_return"] > 0
    assert report["active_period_fraction"] == 0.0
    assert report["accepted"] is False
    assert any(item["metric"] == "active_period_fraction" for item in report["failed_criteria"])


def test_alpha_validation_rejects_invalid_acceptance_parameters(tmp_path):
    with pytest.raises(ValueError, match="transaction_cost_bps"):
        AlphaValidationReport(
            predictions=[],
            actuals=[],
            transaction_cost_bps=-1,
            min_net_excess_return=0.0,
            min_directional_accuracy=0.5,
            min_active_period_fraction=0.1,
        )

    with pytest.raises(ValueError, match="min_directional_accuracy"):
        AlphaValidationReport(
            predictions=[],
            actuals=[],
            min_net_excess_return=0.0,
            min_directional_accuracy=2.0,
        )

    with pytest.raises(ValueError, match="min_active_period_fraction"):
        AlphaValidationReport(
            predictions=[],
            actuals=[],
            min_net_excess_return=0.0,
            min_directional_accuracy=0.5,
            min_active_period_fraction=2.0,
        )

    with pytest.raises(ValueError, match="min_observations"):
        AlphaValidationReport(
            predictions=[],
            actuals=[],
            min_observations=1.9,
        )

    with pytest.raises(ValueError, match="min_active_period_fraction"):
        ModelApprovalRegistry(tmp_path / "bad-approvals.json", min_active_period_fraction=1.5)


def test_alpha_validation_compounds_portfolio_periods_not_symbol_rows():
    report = AlphaValidationReport(
        predictions=[
            prediction("AAA", "2024-01-01", "2024-01-02", 110.0),
            prediction("BBB", "2024-01-01", "2024-01-02", 110.0),
        ],
        actuals=[
            {"symbol": "AAA", "timestamp": "2024-01-01", "close": 100.0},
            {"symbol": "BBB", "timestamp": "2024-01-01", "close": 100.0},
            {"symbol": "AAA", "timestamp": "2024-01-02", "close": 110.0},
            {"symbol": "BBB", "timestamp": "2024-01-02", "close": 90.0},
        ],
    ).compute()

    assert report["strategy_net_return"] == pytest.approx(0.0)


def test_model_approval_registry_persists_only_accepted_reports(tmp_path):
    registry = ModelApprovalRegistry(
        tmp_path / "approvals.json",
        min_net_excess_return=0.0,
        min_directional_accuracy=0.5,
        min_active_period_fraction=0.5,
        min_observations=1,
        min_symbols=1,
        min_regimes=1,
    )
    metadata = approval_metadata(tmp_path, registry)
    report = AlphaValidationReport(
        predictions=[
            prediction("AAA", "2024-01-01", "2024-01-02", 101.0),
            prediction("AAA", "2024-01-02", "2024-01-03", 103.0),
        ],
        actuals=[
            {"symbol": "AAA", "timestamp": "2024-01-01", "close": 100.0},
            {"symbol": "AAA", "timestamp": "2024-01-02", "close": 101.0},
            {"symbol": "AAA", "timestamp": "2024-01-03", "close": 103.0},
        ],
        min_net_excess_return=0.0,
        min_directional_accuracy=0.5,
        min_active_period_fraction=0.5,
        min_observations=1,
    ).compute()

    record = registry.approve("abc123", report, metadata=metadata)

    assert "approval_checksum" in record
    assert registry.accepted_model_hashes() == {"abc123"}
    assert registry.require("abc123")["metadata"]["code_version"] == registry.code_version

    with pytest.raises(ValueError, match="not approved"):
        registry.require("missing")

    bad_schema = dict(report)
    bad_schema["accepted"] = "YES"
    with pytest.raises(ValueError, match="accepted"):
        registry.approve("bad-accepted-type", bad_schema, metadata=metadata)

    bad_schema = dict(report)
    bad_schema["failed_criteria"] = ""
    with pytest.raises(ValueError, match="failed_criteria"):
        registry.approve("bad-failed-criteria-type", bad_schema, metadata=metadata)

    bad_schema = dict(report)
    bad_schema["observations"] = "1.5"
    with pytest.raises(ValueError, match="observations"):
        registry.approve("bad-observations-type", bad_schema, metadata=metadata)

    bad_schema = dict(report)
    bad_schema["start"] = "2024-02-01"
    bad_schema["end"] = "2024-01-01"
    with pytest.raises(ValueError, match="start"):
        registry.approve("bad-date-range", bad_schema, metadata=metadata)

    bad_schema = dict(report)
    bad_schema["criteria"] = dict(report["criteria"])
    bad_schema["criteria"]["min_net_excess_return"] = float("nan")
    with pytest.raises(ValueError, match="min_net_excess_return"):
        registry.approve("bad-nan-criteria", bad_schema, metadata=metadata)

    bad_report = dict(report)
    bad_report["accepted"] = False
    with pytest.raises(ValueError, match="accepted"):
        registry.approve("bad", bad_report, metadata=metadata)

    fake = {
        "accepted": True,
        "failed_criteria": [],
        "criteria": {
            "min_net_excess_return": 0.0,
            "min_directional_accuracy": 0.5,
            "min_active_period_fraction": 0.5,
            "min_observations": 1,
            "min_symbols": 1,
            "min_regimes": 1,
        },
        "observations": 0,
        "symbols": 1,
        "start": "2024-01-01",
        "end": "2024-01-02",
        "strategy_net_return": 0.0,
        "baseline_return": 0.0,
        "net_excess_return": 0.0,
        "directional_accuracy": 1.0,
        "active_period_fraction": 1.0,
        "average_gross_exposure": 1.0,
        "regime_metrics": [{"month": "2024-01"}],
    }
    with pytest.raises(ValueError, match="observations"):
        registry.approve("fake", fake, metadata=metadata)

    bad_metadata = dict(metadata)
    bad_metadata["prediction_file_checksum"] = ""
    with pytest.raises(ValueError, match="metadata"):
        registry.approve("abc123", report, metadata=bad_metadata)

    bad_metadata = dict(metadata)
    bad_metadata["universe"] = [""]
    with pytest.raises(ValueError, match="universe"):
        registry.approve("abc123", report, metadata=bad_metadata)

    bad_metadata = dict(metadata)
    bad_metadata["model_hash"] = "other-model"
    with pytest.raises(ValueError, match="model_hash"):
        registry.approve("abc123", report, metadata=bad_metadata)

    bad_metadata = dict(metadata)
    bad_metadata["prediction_file_checksum"] = "0" * 64
    with pytest.raises(ValueError, match="checksum"):
        registry.approve("abc123", report, metadata=bad_metadata)

    bad_metadata = dict(metadata)
    bad_metadata["code_version"] = "stale-code"
    with pytest.raises(ValueError, match="code_version"):
        registry.approve("abc123", report, metadata=bad_metadata)

    bad_metadata = dict(metadata)
    bad_metadata["code_manifest"] = dict(metadata["code_manifest"])
    bad_metadata["code_manifest"]["trading/paper.py"] = "0" * 64
    with pytest.raises(ValueError, match="code_manifest"):
        registry.approve("abc123", report, metadata=bad_metadata)

    payload = registry._read()
    payload["abc123"]["report"]["accepted"] = False
    registry.path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        registry.require("abc123")


def test_model_approval_registry_rejects_underpowered_report_by_default(tmp_path):
    registry = ModelApprovalRegistry(
        tmp_path / "strict-approvals.json",
        min_net_excess_return=0.0,
        min_directional_accuracy=0.5,
        min_active_period_fraction=0.1,
    )
    metadata = approval_metadata(tmp_path, registry)
    report = AlphaValidationReport(
        predictions=[
            prediction("AAA", "2024-01-01", "2024-01-02", 101.0),
            prediction("AAA", "2024-01-02", "2024-01-03", 102.0),
        ],
        actuals=[
            {"symbol": "AAA", "timestamp": "2024-01-01", "close": 100.0},
            {"symbol": "AAA", "timestamp": "2024-01-02", "close": 101.0},
            {"symbol": "AAA", "timestamp": "2024-01-03", "close": 102.0},
        ],
        min_net_excess_return=0.0,
        min_directional_accuracy=0.5,
        min_active_period_fraction=0.1,
        min_observations=1,
        min_symbols=1,
        min_regimes=1,
    ).compute()

    assert report["accepted"] is True
    with pytest.raises(ValueError, match="min_observations"):
        registry.approve("underpowered", report, metadata=metadata)
