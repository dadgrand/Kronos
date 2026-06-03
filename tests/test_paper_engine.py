import json

import pandas as pd

from trading.engine import PaperTradingEngine
from trading.ops import FileKillSwitch, HeartbeatMonitor, JsonlOrderJournal
from trading.paper import BUY, OPEN, MarketBar, Position
from trading.runner import PaperTradingRunner, PredictionEvent
from trading.validation import ModelApprovalRegistry


def create_runner(**kwargs):
    kwargs.setdefault("accepted_model_hashes", {"abc123"})
    return PaperTradingRunner.create(**kwargs)


def make_bar(timestamp, open_price=100.0, close=100.0, symbol="TEST", volume=1000):
    return MarketBar(
        symbol=symbol,
        timestamp=pd.Timestamp(timestamp),
        open=open_price,
        high=max(open_price, close),
        low=min(open_price, close),
        close=close,
        volume=volume,
    )


def make_prediction(timestamp, predicted_close=110.0, symbol="TEST"):
    timestamp = pd.Timestamp(timestamp)
    asof = timestamp - pd.Timedelta(days=1)
    return PredictionEvent(
        symbol=symbol,
        prediction_asof=asof,
        execution_timestamp=timestamp,
        target_timestamp=timestamp,
        features_cutoff=asof,
        horizon=str(timestamp - asof),
        model_version="unit-test-model",
        model_hash="abc123",
        predicted_close=predicted_close,
    )


def approval_metadata(tmp_path, registry, model_hash="abc123"):
    prediction_file = tmp_path / f"{model_hash}-predictions.json"
    actuals_file = tmp_path / f"{model_hash}-actuals.json"
    prediction_file.write_text(json.dumps({"prediction_results": []}), encoding="utf-8")
    actuals_file.write_text(json.dumps({"actuals": []}), encoding="utf-8")
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


def test_engine_journals_bar_prediction_fill_and_heartbeat(tmp_path):
    runner = create_runner(
        initial_cash=10000,
        commission_rate=0.0,
        slippage_rate=0.0,
        max_participation_rate=1.0,
    )
    journal = JsonlOrderJournal(tmp_path / "orders.jsonl")
    heartbeat = HeartbeatMonitor(tmp_path / "heartbeat.json")
    engine = PaperTradingEngine(runner=runner, journal=journal, heartbeat=heartbeat)

    engine.on_bar(make_bar("2024-01-01", close=100))
    fills = engine.on_bar(make_bar("2024-01-02", close=105), make_prediction("2024-01-02"))

    records = journal.read_all()
    event_types = [record["event_type"] for record in records]

    assert fills
    assert "bar_received" in event_types
    assert "prediction_received" in event_types
    assert "fill" in event_types
    assert heartbeat.path.exists()


def test_engine_production_mode_requires_registry_backed_runner(tmp_path):
    runner = create_runner(initial_cash=10000)
    journal = JsonlOrderJournal(tmp_path / "orders.jsonl")

    try:
        PaperTradingEngine(runner=runner, journal=journal, production_mode=True)
    except ValueError as exc:
        assert "approval_registry" in str(exc)
    else:
        raise AssertionError("Expected production_mode without registry to fail")

    empty_registry = ModelApprovalRegistry(tmp_path / "empty-approvals.json")
    runner = PaperTradingRunner.create(initial_cash=10000, approval_registry=empty_registry)
    try:
        PaperTradingEngine(runner=runner, journal=journal, production_mode=True)
    except ValueError as exc:
        assert "non-empty" in str(exc)
    else:
        raise AssertionError("Expected production_mode with empty registry to fail")


def test_engine_production_mode_rejects_permissive_registry_floors(tmp_path):
    registry = ModelApprovalRegistry(
        tmp_path / "permissive-approvals.json",
        min_net_excess_return=0.0,
        min_directional_accuracy=0.5,
        min_active_period_fraction=0.0,
        min_observations=1,
        min_symbols=1,
        min_regimes=1,
    )
    registry.approve(
        "abc123",
        {
            "accepted": True,
            "failed_criteria": [],
            "criteria": {
                "min_net_excess_return": 0.0,
                "min_directional_accuracy": 0.5,
                "min_active_period_fraction": 0.0,
                "min_observations": 1,
                "min_symbols": 1,
                "min_regimes": 1,
            },
            "observations": 1,
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
        },
        metadata=approval_metadata(tmp_path, registry),
    )
    runner = PaperTradingRunner.create(initial_cash=10000, approval_registry=registry)

    try:
        PaperTradingEngine(runner=runner, journal=JsonlOrderJournal(tmp_path / "orders.jsonl"), production_mode=True)
    except ValueError as exc:
        assert "production floors" in str(exc)
    else:
        raise AssertionError("Expected production_mode with permissive registry floors to fail")


def test_engine_kill_switch_liquidates_position(tmp_path):
    runner = create_runner(
        initial_cash=10000,
        commission_rate=0.0,
        slippage_rate=0.0,
        max_participation_rate=1.0,
    )
    journal = JsonlOrderJournal(tmp_path / "orders.jsonl")
    kill_switch = FileKillSwitch(tmp_path / "KILL")
    engine = PaperTradingEngine(
        runner=runner,
        journal=journal,
        kill_switch=kill_switch,
        max_liquidation_attempts=1,
    )

    engine.on_bar(make_bar("2024-01-01", close=100))
    engine.on_bar(make_bar("2024-01-02", close=105), make_prediction("2024-01-02"))
    assert runner.broker.positions["TEST"].quantity > 0

    kill_switch.activate("risk-off")
    fills = engine.on_bar(make_bar("2024-01-03", open_price=105, close=104))

    assert fills
    assert "TEST" not in runner.broker.positions
    assert any(record["event_type"] == "kill_switch_active" for record in journal.read_all())


def test_engine_kill_switch_liquidates_all_positions_with_current_bars(tmp_path):
    runner = create_runner(
        initial_cash=10000,
        commission_rate=0.0,
        slippage_rate=0.0,
        max_participation_rate=1.0,
    )
    journal = JsonlOrderJournal(tmp_path / "orders.jsonl")
    kill_switch = FileKillSwitch(tmp_path / "KILL")
    engine = PaperTradingEngine(
        runner=runner,
        journal=journal,
        kill_switch=kill_switch,
        max_liquidation_attempts=1,
    )
    runner.broker.positions["AAA"] = Position("AAA", quantity=2, avg_price=100.0)
    runner.broker.positions["BBB"] = Position("BBB", quantity=3, avg_price=50.0)

    kill_switch.activate("risk-off")
    fills = engine.on_bars(
        [
            make_bar("2024-01-03", open_price=99, close=98, symbol="AAA"),
            make_bar("2024-01-03", open_price=49, close=48, symbol="BBB"),
        ]
    )

    assert sorted(fill.symbol for fill in fills) == ["AAA", "BBB"]
    assert runner.broker.positions == {}


def test_engine_kill_switch_journals_missing_position_bar_as_pending(tmp_path):
    runner = create_runner(
        initial_cash=10000,
        commission_rate=0.0,
        slippage_rate=0.0,
        max_participation_rate=1.0,
    )
    journal = JsonlOrderJournal(tmp_path / "orders.jsonl")
    kill_switch = FileKillSwitch(tmp_path / "KILL")
    engine = PaperTradingEngine(runner=runner, journal=journal, kill_switch=kill_switch)
    runner.broker.positions["AAA"] = Position("AAA", quantity=2, avg_price=100.0)
    runner.broker.positions["BBB"] = Position("BBB", quantity=3, avg_price=50.0)
    runner.broker.last_prices["BBB"] = 50.0
    runner.broker.last_price_timestamps["BBB"] = pd.Timestamp("2024-01-03")

    kill_switch.activate("risk-off")
    fills = engine.on_bars([make_bar("2024-01-03", open_price=99, close=98, symbol="AAA")])

    assert [fill.symbol for fill in fills] == ["AAA"]
    assert "AAA" not in runner.broker.positions
    assert runner.broker.positions["BBB"].quantity == 3
    assert any(
        record["event_type"] == "liquidation_pending" and record["payload"]["symbol"] == "BBB"
        for record in journal.read_all()
    )


def test_engine_kill_switch_pending_position_requires_fresh_mark(tmp_path):
    runner = create_runner(initial_cash=10000)
    journal = JsonlOrderJournal(tmp_path / "orders.jsonl")
    kill_switch = FileKillSwitch(tmp_path / "KILL")
    engine = PaperTradingEngine(runner=runner, journal=journal, kill_switch=kill_switch)
    runner.broker.positions["TEST"] = Position("TEST", quantity=1, avg_price=100.0)

    kill_switch.activate("risk-off")
    try:
        engine.on_bars([])
    except ValueError as exc:
        assert "Missing market price" in str(exc)
    else:
        raise AssertionError("Expected missing mark to fail pending liquidation valuation")

    assert any(record["event_type"] == "processing_error" for record in journal.read_all())


def test_engine_kill_switch_journals_partial_liquidation_residual(tmp_path):
    runner = create_runner(
        initial_cash=10000,
        commission_rate=0.0,
        slippage_rate=0.0,
        max_participation_rate=0.1,
    )
    journal = JsonlOrderJournal(tmp_path / "orders.jsonl")
    kill_switch = FileKillSwitch(tmp_path / "KILL")
    engine = PaperTradingEngine(
        runner=runner,
        journal=journal,
        kill_switch=kill_switch,
        max_liquidation_attempts=1,
    )
    runner.broker.positions["TEST"] = Position("TEST", quantity=100, avg_price=100.0)

    kill_switch.activate("risk-off")
    fills = engine.on_bar(make_bar("2024-01-03", open_price=99, close=98, volume=100))

    assert sum(fill.quantity for fill in fills) == 10
    assert runner.broker.positions["TEST"].quantity == 90
    assert any(
        record["event_type"] == "liquidation_residual"
        and record["payload"]["remaining_quantity"] == 90
        for record in journal.read_all()
    )
    assert any(record["event_type"] == "liquidation_escalation" for record in journal.read_all())


def test_engine_kill_switch_invalid_bar_does_not_mutate_orders(tmp_path):
    runner = create_runner(
        initial_cash=10000,
        commission_rate=0.0,
        slippage_rate=0.0,
        max_participation_rate=1.0,
    )
    journal = JsonlOrderJournal(tmp_path / "orders.jsonl")
    kill_switch = FileKillSwitch(tmp_path / "KILL")
    engine = PaperTradingEngine(runner=runner, journal=journal, kill_switch=kill_switch)

    engine.on_bar(make_bar("2024-01-01", close=100))
    engine.on_bar(make_bar("2024-01-02", close=105), make_prediction("2024-01-02"))
    buy = runner.broker.submit_order("TEST", BUY, 1, pd.Timestamp("2024-01-03"))
    kill_switch.activate("risk-off")

    try:
        engine.on_bar(make_bar("2024-01-03", open_price=-1, close=100))
    except ValueError:
        pass

    assert buy.status == OPEN
    assert not any(order.side == "SELL" and order.status == "OPEN" for order in runner.broker.orders.values())
    assert any(record["event_type"] == "processing_error" for record in journal.read_all())


def test_engine_active_bar_batch_validates_before_trading_mutation(tmp_path):
    runner = create_runner(
        initial_cash=10000,
        commission_rate=0.0,
        slippage_rate=0.0,
        max_participation_rate=1.0,
    )
    journal = JsonlOrderJournal(tmp_path / "orders.jsonl")
    engine = PaperTradingEngine(runner=runner, journal=journal)
    runner.process_bar(make_bar("2024-01-01", close=100, symbol="AAA"))

    try:
        engine.on_bars(
            [
                make_bar("2024-01-02", close=105, symbol="AAA"),
                make_bar("2024-01-02", open_price=-1, close=100, symbol="BBB"),
            ],
            {"AAA": make_prediction("2024-01-02", predicted_close=110, symbol="AAA")},
        )
    except ValueError as exc:
        assert "MarketBar prices" in str(exc)
    else:
        raise AssertionError("Expected invalid batch bar to raise ValueError")

    assert runner.broker.orders == {}
    assert runner.broker.positions == {}
    assert len(runner.equity_history) == 1
    assert any(record["event_type"] == "processing_error" for record in journal.read_all())


def test_engine_active_bar_batch_rolls_back_runtime_failure(tmp_path):
    runner = create_runner(
        initial_cash=10000,
        commission_rate=0.0,
        slippage_rate=0.0,
        max_participation_rate=1.0,
    )
    journal = JsonlOrderJournal(tmp_path / "orders.jsonl")
    engine = PaperTradingEngine(runner=runner, journal=journal)
    runner.process_bar(make_bar("2024-01-01", close=100, symbol="AAA"))
    original_process_bar = runner.process_bar

    def flaky_process_bar(bar, prediction=None):
        if bar.symbol == "BBB":
            raise RuntimeError("simulated broker failure")
        return original_process_bar(bar, prediction)

    runner.process_bar = flaky_process_bar

    try:
        engine.on_bars(
            [
                make_bar("2024-01-02", close=105, symbol="AAA"),
                make_bar("2024-01-02", close=50, symbol="BBB"),
            ],
            {"AAA": make_prediction("2024-01-02", predicted_close=110, symbol="AAA")},
        )
    except RuntimeError as exc:
        assert "simulated broker failure" in str(exc)
    else:
        raise AssertionError("Expected runtime batch failure")

    assert runner.broker.positions == {}
    assert not any("2024-01-02" in key for key in runner.broker.processed_bars)
    assert len(runner.equity_history) == 1
    event_types = [record["event_type"] for record in journal.read_all()]
    assert "batch_rollback" in event_types
    assert "fill" not in event_types
    assert "equity" not in event_types


def test_engine_rejects_duplicate_symbols_in_bar_batch(tmp_path):
    runner = create_runner(initial_cash=10000)
    journal = JsonlOrderJournal(tmp_path / "orders.jsonl")
    engine = PaperTradingEngine(runner=runner, journal=journal)

    try:
        engine.on_bars([make_bar("2024-01-01"), make_bar("2024-01-01")])
    except ValueError as exc:
        assert "Duplicate bar" in str(exc)
    else:
        raise AssertionError("Expected duplicate bar batch to raise ValueError")


def test_engine_kill_switch_processed_bar_does_not_leave_stale_order(tmp_path):
    runner = create_runner(
        initial_cash=10000,
        commission_rate=0.0,
        slippage_rate=0.0,
        max_participation_rate=1.0,
    )
    journal = JsonlOrderJournal(tmp_path / "orders.jsonl")
    kill_switch = FileKillSwitch(tmp_path / "KILL")
    engine = PaperTradingEngine(runner=runner, journal=journal, kill_switch=kill_switch)
    market_bar = make_bar("2024-01-01", close=100)

    engine.on_bar(market_bar)
    runner.broker.positions["TEST"] = Position("TEST", quantity=1, avg_price=100.0)
    kill_switch.activate("risk-off")
    fills = engine.on_bar(market_bar)

    assert fills == []
    assert not any(order.side == "SELL" and order.status == "OPEN" for order in runner.broker.orders.values())
    assert any(record["event_type"] == "liquidation_skipped" for record in journal.read_all())


def test_engine_journals_processing_errors(tmp_path):
    runner = create_runner(initial_cash=10000)
    journal = JsonlOrderJournal(tmp_path / "orders.jsonl")
    engine = PaperTradingEngine(runner=runner, journal=journal)

    try:
        engine.on_bar(make_bar("2024-01-01", open_price=-1, close=100))
    except ValueError:
        pass

    assert any(record["event_type"] == "processing_error" for record in journal.read_all())


def test_jsonl_order_journal_writes_sequence_checksum_and_verifies_tampering(tmp_path):
    journal = JsonlOrderJournal(tmp_path / "orders.jsonl")

    first = journal.append("alpha", {"value": 1})
    second = journal.append("beta", {"value": 2})
    records = journal.read_all()

    assert [record["sequence_id"] for record in records] == [0, 1]
    assert records[0]["previous_checksum"] == ""
    assert records[1]["previous_checksum"] == records[0]["checksum"]
    assert records[0]["checksum"] == first["checksum"]
    assert records[1]["checksum"] == second["checksum"]

    text = journal.path.read_text(encoding="utf-8")
    journal.path.write_text(text.replace('"value": 2', '"value": 3'), encoding="utf-8")
    try:
        journal.read_all()
    except ValueError as exc:
        assert "checksum" in str(exc)
    else:
        raise AssertionError("Expected checksum verification to fail after tampering")

    journal.path.write_text("\n".join(reversed(text.strip().splitlines())) + "\n", encoding="utf-8")
    try:
        journal.read_all()
    except ValueError as exc:
        assert "sequence_id" in str(exc)
    else:
        raise AssertionError("Expected sequence verification to fail after reordering")

    first_line = text.strip().splitlines()[0].replace('"checksum": "' + records[0]["checksum"] + '", ', "")
    journal.path.write_text(first_line + "\n", encoding="utf-8")
    try:
        journal.read_all()
    except ValueError as exc:
        assert "missing checksum" in str(exc)
    else:
        raise AssertionError("Expected missing checksum verification to fail")

    legacy = {
        "event_type": "legacy",
        "previous_checksum": "",
        "recorded_at": "2024-01-01T00:00:00+00:00",
        "payload": {"value": 1},
    }
    legacy["checksum"] = JsonlOrderJournal._checksum(legacy)
    journal.path.write_text(json.dumps(legacy, sort_keys=True) + "\n", encoding="utf-8")
    try:
        journal.read_all()
    except ValueError as exc:
        assert "sequence_id" in str(exc)
    else:
        raise AssertionError("Expected missing sequence_id verification to fail")
