import json

import pandas as pd
import pytest

from trading.paper import MarketBar, Position
from trading.runner import PaperTradingRunner, PredictionEvent
from trading.validation import AlphaValidationReport, ModelApprovalRegistry


def approval_metadata(tmp_path, registry, predictions=None, actuals=None, model_hash="abc123"):
    prediction_file = tmp_path / "predictions.json"
    actuals_file = tmp_path / "actuals.json"
    prediction_file.write_text(
        json.dumps({"prediction_results": ModelApprovalRegistry._json_safe(predictions or [])}),
        encoding="utf-8",
    )
    actuals_file.write_text(
        json.dumps({"actuals": ModelApprovalRegistry._json_safe(actuals or [])}),
        encoding="utf-8",
    )
    return {
        "prediction_file_path": str(prediction_file),
        "prediction_file_checksum": ModelApprovalRegistry.file_checksum(prediction_file),
        "actuals_file_path": str(actuals_file),
        "actuals_file_checksum": ModelApprovalRegistry.file_checksum(actuals_file),
        "oos_start": "2024-01-01",
        "oos_end": "2024-01-31",
        "universe": ["TEST"],
        "model_hash": model_hash,
        "model_revision": "model-rev",
        "tokenizer_revision": "tokenizer-rev",
        "code_version": registry.code_version,
        "code_manifest": registry.code_manifest,
        "validation_code_checksum": registry.validation_code_checksum,
    }


def create_runner(**kwargs):
    kwargs.setdefault("accepted_model_hashes", {"abc123"})
    return PaperTradingRunner.create(**kwargs)


def make_bar(timestamp, close=100.0, open_price=None, volume=1000):
    open_price = close if open_price is None else open_price
    return MarketBar(
        symbol="TEST",
        timestamp=pd.Timestamp(timestamp),
        open=open_price,
        high=max(open_price, close),
        low=min(open_price, close),
        close=close,
        volume=volume,
    )


def make_prediction(execution_timestamp, predicted_close=110.0, asof=None):
    execution_timestamp = pd.Timestamp(execution_timestamp)
    asof = pd.Timestamp(asof) if asof is not None else execution_timestamp - pd.Timedelta(days=1)
    return PredictionEvent(
        symbol="TEST",
        prediction_asof=asof,
        execution_timestamp=execution_timestamp,
        target_timestamp=execution_timestamp,
        features_cutoff=asof,
        horizon=str(execution_timestamp - asof),
        model_version="unit-test-model",
        model_hash="abc123",
        predicted_close=predicted_close,
    )


def test_runner_requires_previous_close_before_trading():
    runner = create_runner(initial_cash=10000, commission_rate=0.0, slippage_rate=0.0)

    fills = runner.process_bar(make_bar("2024-01-02", close=100), make_prediction("2024-01-02"))

    assert fills == []
    assert runner.signal_history[-1]["reason"] == "missing_previous_close"
    assert "TEST" not in runner.broker.positions


def test_runner_turns_prediction_into_order_fill_and_report():
    runner = create_runner(
        initial_cash=10000,
        threshold=0.02,
        max_position_fraction=0.25,
        commission_rate=0.0,
        slippage_rate=0.0,
        max_participation_rate=1.0,
    )

    runner.process_bar(make_bar("2024-01-01", close=100))
    fills = runner.process_bar(
        make_bar("2024-01-02", open_price=100, close=105),
        make_prediction("2024-01-02", predicted_close=110),
    )

    assert sum(fill.quantity for fill in fills) == 25
    assert runner.broker.positions["TEST"].quantity == 25
    assert runner.report()["fills"] == 1


def test_runner_validates_prediction_execution_time():
    runner = create_runner(initial_cash=10000)
    runner.process_bar(make_bar("2024-01-01", close=100))

    with pytest.raises(ValueError, match="execution_timestamp"):
        runner.process_bar(make_bar("2024-01-02"), make_prediction("2024-01-03"))


def test_runner_rejects_prediction_asof_that_does_not_match_reference_close():
    runner = create_runner(initial_cash=10000)
    runner.process_bar(make_bar("2024-01-01", close=100))

    with pytest.raises(ValueError, match="prediction_asof"):
        runner.process_bar(
            make_bar("2024-01-02", close=105),
            make_prediction("2024-01-02", asof="2023-12-31"),
        )

    assert runner.broker.orders == {}
    assert "2024-01-02" not in " ".join(runner.broker.processed_bars)


def test_runner_state_round_trip(tmp_path):
    runner = create_runner(
        initial_cash=10000,
        commission_rate=0.0,
        slippage_rate=0.0,
        max_participation_rate=1.0,
    )
    runner.process_bar(make_bar("2024-01-01", close=100))
    runner.process_bar(make_bar("2024-01-02", close=105), make_prediction("2024-01-02"))

    runner.save_state(tmp_path)
    restored = PaperTradingRunner.load_state(tmp_path)

    assert restored.broker.cash == runner.broker.cash
    assert restored.last_close == runner.last_close
    assert restored.last_close_timestamp == runner.last_close_timestamp
    assert restored.max_mark_age == runner.max_mark_age
    assert restored.report() == runner.report()


def test_runner_loads_prediction_contract_json(tmp_path):
    prediction = make_prediction("2024-01-02")
    path = tmp_path / "prediction.json"
    path.write_text(
        json.dumps({"prediction_results": [prediction_to_json(prediction)]}),
        encoding="utf-8",
    )

    events = PaperTradingRunner.load_predictions(path)

    assert len(events) == 1
    assert events[0].symbol == "TEST"


def test_runner_loads_webui_shaped_prediction_json(tmp_path):
    prediction = prediction_to_json(make_prediction("2024-01-02"))
    prediction["close"] = prediction.pop("predicted_close")
    prediction["predicted_close"] = prediction["close"]
    path = tmp_path / "webui_prediction.json"
    path.write_text(
        json.dumps({"contract_schema_version": "kronos-prediction-contract-v1", "prediction_results": [prediction]}),
        encoding="utf-8",
    )

    events = PaperTradingRunner.load_predictions(path)

    assert events[0].predicted_close == prediction["close"]


def test_prediction_event_rejects_placeholder_or_bad_price():
    prediction = make_prediction("2024-01-02")
    prediction.model_hash = "unknown"

    with pytest.raises(ValueError, match="model_hash"):
        prediction.validate()

    prediction = make_prediction("2024-01-02")
    prediction.predicted_close = float("nan")

    with pytest.raises(ValueError, match="predicted_close"):
        prediction.validate()


def test_runner_rejects_late_prediction_for_processed_bar():
    runner = create_runner(initial_cash=10000)
    market_bar = make_bar("2024-01-01", close=100)

    runner.process_bar(market_bar)

    with pytest.raises(ValueError, match="already processed"):
        runner.process_bar(market_bar, make_prediction("2024-01-01", asof="2023-12-31"))


def test_runner_rejects_unapproved_model_hash_when_gate_is_configured():
    runner = create_runner(
        initial_cash=10000,
        accepted_model_hashes={"approved-hash"},
    )
    runner.process_bar(make_bar("2024-01-01", close=100))
    prediction = make_prediction("2024-01-02", predicted_close=110)
    prediction.model_hash = "unapproved-hash"

    with pytest.raises(ValueError, match="not approved"):
        runner.process_bar(make_bar("2024-01-02", close=101), prediction)

    assert runner.broker.orders == {}


def test_runner_accepts_model_hash_from_verified_registry(tmp_path):
    registry = ModelApprovalRegistry(
        tmp_path / "approvals.json",
        min_net_excess_return=0.0,
        min_directional_accuracy=0.5,
        min_active_period_fraction=0.5,
        min_observations=1,
        min_symbols=1,
        min_regimes=1,
    )
    predictions = [
        prediction_to_json(make_prediction("2024-01-02", predicted_close=101.0, asof="2024-01-01")),
    ]
    actuals = [
        {"symbol": "TEST", "timestamp": "2024-01-01", "close": 100.0},
        {"symbol": "TEST", "timestamp": "2024-01-02", "close": 101.0},
    ]
    metadata = approval_metadata(tmp_path, registry, predictions, actuals)
    report = AlphaValidationReport(
        predictions=predictions,
        actuals=actuals,
        min_net_excess_return=0.0,
        min_directional_accuracy=0.5,
        min_active_period_fraction=0.5,
        min_observations=1,
        min_symbols=1,
        min_regimes=1,
    ).compute()
    registry.approve(
        "abc123",
        report,
        metadata=metadata,
    )
    runner = PaperTradingRunner.create(
        initial_cash=10000,
        commission_rate=0.0,
        slippage_rate=0.0,
        max_participation_rate=1.0,
        approval_registry=registry,
    )
    runner.process_bar(make_bar("2024-01-01", close=100))

    fills = runner.process_bar(make_bar("2024-01-02", close=105), make_prediction("2024-01-02"))

    assert fills

    runner.save_state(tmp_path / "state")
    restored = PaperTradingRunner.load_state(tmp_path / "state", approval_registry=registry)
    assert restored.approval_registry.require("abc123")

    registry.path.unlink()
    with pytest.raises(ValueError, match="approval_registry"):
        PaperTradingRunner.load_state(tmp_path / "state")


def test_runner_production_mode_requires_approval_registry():
    with pytest.raises(ValueError, match="approval_registry"):
        PaperTradingRunner.create(initial_cash=10000, require_approval_registry=True)


def test_runner_rejects_invalid_bar_without_mutating_orders():
    runner = create_runner(initial_cash=10000)
    runner.process_bar(make_bar("2024-01-01", close=100))

    invalid_bar = make_bar("2024-01-02", open_price=-100, close=100)
    with pytest.raises(ValueError, match="MarketBar prices"):
        runner.process_bar(invalid_bar, make_prediction("2024-01-02", predicted_close=110))

    assert runner.broker.orders == {}
    assert len(runner.signal_history) == 0


def test_runner_rejects_stale_portfolio_mark_before_mutating_bar_state():
    runner = create_runner(
        initial_cash=10000,
        commission_rate=0.0,
        slippage_rate=0.0,
        max_mark_age=pd.Timedelta(days=1),
    )
    runner.broker.positions["BBB"] = Position("BBB", quantity=10, avg_price=50.0)
    runner.broker.last_prices["BBB"] = 40.0
    runner.broker.last_price_timestamps["BBB"] = pd.Timestamp("2024-01-01")

    with pytest.raises(ValueError, match="Stale market price"):
        runner.process_bar(make_bar("2024-01-03", close=100))

    assert runner.broker.processed_bars == set()
    assert runner.equity_history == []


def test_runner_duplicate_bar_without_prediction_is_noop_for_history():
    runner = create_runner(initial_cash=10000)
    market_bar = make_bar("2024-01-01", close=100)

    runner.process_bar(market_bar)
    runner.process_bar(market_bar)

    assert len(runner.broker.processed_bars) == 1
    assert len(runner.equity_history) == 1


def prediction_to_json(prediction):
    return {
        "symbol": prediction.symbol,
        "prediction_asof": prediction.prediction_asof.isoformat(),
        "execution_timestamp": prediction.execution_timestamp.isoformat(),
        "target_timestamp": prediction.target_timestamp.isoformat(),
        "features_cutoff": prediction.features_cutoff.isoformat(),
        "horizon": prediction.horizon,
        "model_version": prediction.model_version,
        "model_hash": prediction.model_hash,
        "predicted_close": prediction.predicted_close,
    }
