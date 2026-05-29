import pandas as pd
import pytest
from decimal import Decimal

from trading.live import (
    AccountSnapshot,
    BrokerReconciler,
    ExternalFill,
    ExternalOrder,
    LiveExecutionLoop,
    LiveOrderRequest,
    LiveReconciliationError,
)
from trading.ops import JsonlOrderJournal
from trading.paper import BUY, CANCELLED, MarketBar, Order, PaperBroker, PaperRiskManager


def test_broker_reconciler_accepts_matching_snapshot():
    broker = PaperBroker(initial_cash=10000, commission_rate=0.0, slippage_rate=0.0)
    broker.submit_order("AAA", BUY, 10, pd.Timestamp("2024-01-01"))
    broker.process_bar(_bar("AAA", "2024-01-02", open_price=100, close=101))
    order = broker.submit_order("BBB", BUY, 5, pd.Timestamp("2024-01-03"))
    snapshot = AccountSnapshot(
        timestamp=pd.Timestamp("2024-01-03"),
        cash=broker.cash,
        positions={"AAA": 10},
        orders=[
            ExternalOrder(
                external_id="broker-1",
                client_order_id=order.id,
                symbol="BBB",
                side="BUY",
                quantity=5,
                filled_quantity=0,
                status="ACCEPTED",
                submitted_at=order.submitted_at,
            )
        ],
    )

    report = BrokerReconciler().reconcile(broker, snapshot)

    assert report.ok is True


def test_broker_reconciler_reports_position_cash_and_order_mismatches():
    broker = PaperBroker(initial_cash=10000, commission_rate=0.0, slippage_rate=0.0)
    order = broker.submit_order("AAA", BUY, 10, pd.Timestamp("2024-01-01"))
    snapshot = AccountSnapshot(
        timestamp=pd.Timestamp("2024-01-02"),
        cash=9999,
        positions={"AAA": 1},
        orders=[
            ExternalOrder(
                external_id="broker-1",
                client_order_id=order.id,
                symbol="AAA",
                side="BUY",
                quantity=10,
                filled_quantity=3,
                status="REJECTED",
                submitted_at=order.submitted_at,
            )
        ],
    )

    report = BrokerReconciler().reconcile(broker, snapshot)

    assert report.ok is False
    assert report.cash_mismatch is not None
    assert any(item["type"] == "position" for item in report.position_mismatches)
    assert any(item["type"] == "order_status" for item in report.order_mismatches)
    assert any(item["type"] == "order_filled_quantity" for item in report.order_mismatches)


def test_broker_reconciler_reports_unexpected_external_order():
    broker = PaperBroker(initial_cash=10000, commission_rate=0.0, slippage_rate=0.0)
    snapshot = AccountSnapshot(
        timestamp=pd.Timestamp("2024-01-02"),
        cash=10000,
        positions={},
        orders=[
            ExternalOrder(
                external_id="broker-rogue",
                client_order_id="unknown-client-order",
                symbol="AAA",
                side="BUY",
                quantity=10,
                filled_quantity=0,
                status="ACCEPTED",
                submitted_at=pd.Timestamp("2024-01-02"),
            )
        ],
    )

    report = BrokerReconciler().reconcile(broker, snapshot)

    assert any(item["type"] == "unexpected_external_order" for item in report.order_mismatches)


def test_broker_reconciler_reports_external_open_after_local_close():
    broker = PaperBroker(initial_cash=10000, commission_rate=0.0, slippage_rate=0.0)
    order = broker.submit_order("AAA", BUY, 10, pd.Timestamp("2024-01-01"))
    order.status = CANCELLED
    snapshot = AccountSnapshot(
        timestamp=pd.Timestamp("2024-01-02"),
        cash=10000,
        positions={},
        orders=[
            ExternalOrder(
                external_id="broker-still-open",
                client_order_id=order.id,
                symbol="AAA",
                side="BUY",
                quantity=10,
                filled_quantity=0,
                status="ACCEPTED",
                submitted_at=pd.Timestamp("2024-01-01"),
            )
        ],
    )

    report = BrokerReconciler().reconcile(broker, snapshot)

    assert any(
        item["type"] == "external_order_open_after_local_close" for item in report.order_mismatches
    )


def test_broker_reconciler_reports_duplicate_external_client_order_id():
    broker = PaperBroker(initial_cash=10000, commission_rate=0.0, slippage_rate=0.0)
    order = broker.submit_order("AAA", BUY, 10, pd.Timestamp("2024-01-01"))
    snapshot = AccountSnapshot(
        timestamp=pd.Timestamp("2024-01-02"),
        cash=10000,
        positions={},
        orders=[
            ExternalOrder(
                external_id="broker-1",
                client_order_id=order.id,
                symbol="AAA",
                side="BUY",
                quantity=10,
                filled_quantity=0,
                status="ACCEPTED",
                submitted_at=pd.Timestamp("2024-01-01"),
            ),
            ExternalOrder(
                external_id="broker-2",
                client_order_id=order.id,
                symbol="AAA",
                side="BUY",
                quantity=10,
                filled_quantity=0,
                status="ACCEPTED",
                submitted_at=pd.Timestamp("2024-01-01"),
            ),
        ],
    )

    report = BrokerReconciler().reconcile(broker, snapshot)

    assert any(
        item["type"] == "duplicate_external_client_order_id" for item in report.order_mismatches
    )


def test_broker_reconciler_reports_order_contract_drift():
    broker = PaperBroker(initial_cash=10000, commission_rate=0.0, slippage_rate=0.0)
    order = Order("AAA", "BUY", 1, pd.Timestamp("2024-01-01"), id="client-1")
    broker.orders[order.id] = order
    snapshot = AccountSnapshot(
        timestamp=pd.Timestamp("2024-01-02"),
        cash=10000,
        positions={},
        orders=[
            ExternalOrder(
                external_id="broker-1",
                client_order_id="client-1",
                symbol="AAA",
                side="BUY",
                quantity=1,
                filled_quantity=0,
                status="ACCEPTED",
                submitted_at=pd.Timestamp("2024-01-01"),
                order_type="LIMIT",
                limit_price="99",
            )
        ],
    )

    report = BrokerReconciler().reconcile(broker, snapshot)

    assert any(item["type"] == "order_order_type" for item in report.order_mismatches)
    assert any(item["type"] == "order_limit_price" for item in report.order_mismatches)


def test_broker_reconciler_preserves_fractional_external_quantities():
    broker = PaperBroker(initial_cash=10000, commission_rate=0.0, slippage_rate=0.0)
    order = broker.submit_order("AAA", BUY, 1, pd.Timestamp("2024-01-01"))
    snapshot = AccountSnapshot(
        timestamp=pd.Timestamp("2024-01-02"),
        cash=10000,
        positions={},
        orders=[
            ExternalOrder(
                external_id="broker-1",
                client_order_id=order.id,
                symbol="AAA",
                side="BUY",
                quantity=Decimal("1.5"),
                filled_quantity=Decimal("0"),
                status="ACCEPTED",
                submitted_at=pd.Timestamp("2024-01-01"),
            )
        ],
    )

    report = BrokerReconciler().reconcile(broker, snapshot)

    assert any(item["type"] == "order_quantity" for item in report.order_mismatches)


def test_live_order_request_validates_market_realistic_fields():
    request = LiveOrderRequest(
        symbol="AAA",
        side="buy",
        quantity="1.5",
        client_order_id="client-1",
        order_type="limit",
        time_in_force="gtc",
        limit_price="10.25",
    )

    assert request.side == "BUY"
    assert request.order_type == "LIMIT"
    assert request.time_in_force == "GTC"

    with pytest.raises(ValueError, match="limit_price"):
        LiveOrderRequest(
            symbol="AAA",
            side="BUY",
            quantity="1",
            client_order_id="client-2",
            order_type="LIMIT",
        )

    with pytest.raises(ValueError, match="client_order_id"):
        LiveOrderRequest(symbol="AAA", side="BUY", quantity="1", client_order_id="")

    with pytest.raises(ValueError, match="finite"):
        LiveOrderRequest(symbol="AAA", side="BUY", quantity="NaN", client_order_id="client-3")

    with pytest.raises(ValueError, match="reduce_only"):
        LiveOrderRequest(symbol="AAA", side="BUY", quantity="1", client_order_id="client-4", reduce_only="False")

    with pytest.raises(ValueError, match="reduce_only"):
        ExternalOrder(
            external_id="broker-1",
            client_order_id="client-1",
            symbol="AAA",
            side="BUY",
            quantity=1,
            filled_quantity=0,
            status="ACCEPTED",
            submitted_at=pd.Timestamp("2024-01-01"),
            reduce_only="False",
        )

    with pytest.raises(ValueError, match="side"):
        ExternalOrder(
            external_id="broker-1",
            client_order_id="client-1",
            symbol="AAA",
            side="HOLD",
            quantity=1,
            filled_quantity=0,
            status="ACCEPTED",
            submitted_at=pd.Timestamp("2024-01-01"),
        )

    with pytest.raises(ValueError, match="limit_price"):
        ExternalOrder(
            external_id="broker-1",
            client_order_id="client-1",
            symbol="AAA",
            side="BUY",
            quantity=1,
            filled_quantity=0,
            status="ACCEPTED",
            submitted_at=pd.Timestamp("2024-01-01"),
            order_type="LIMIT",
        )

    with pytest.raises(ValueError, match="status"):
        ExternalOrder(
            external_id="broker-1",
            client_order_id="client-1",
            symbol="AAA",
            side="BUY",
            quantity=1,
            filled_quantity=0,
            status="BOGUS",
            submitted_at=pd.Timestamp("2024-01-01"),
        )

    with pytest.raises(ValueError, match="FILLED"):
        ExternalOrder(
            external_id="broker-1",
            client_order_id="client-1",
            symbol="AAA",
            side="BUY",
            quantity=1,
            filled_quantity=0,
            status="FILLED",
            submitted_at=pd.Timestamp("2024-01-01"),
        )

    with pytest.raises(ValueError, match="fully filled"):
        ExternalOrder(
            external_id="broker-1",
            client_order_id="client-1",
            symbol="AAA",
            side="BUY",
            quantity=1,
            filled_quantity=1,
            status="CANCELLED",
            submitted_at=pd.Timestamp("2024-01-01"),
        )

    with pytest.raises(ValueError, match="PARTIALLY_FILLED"):
        ExternalOrder(
            external_id="broker-1",
            client_order_id="client-1",
            symbol="AAA",
            side="BUY",
            quantity=1,
            filled_quantity=0,
            status="PARTIALLY_FILLED",
            submitted_at=pd.Timestamp("2024-01-01"),
        )


def test_live_execution_loop_gates_submit_with_reconciliation(tmp_path):
    broker = PaperBroker(initial_cash=10000, commission_rate=0.0, slippage_rate=0.0)
    adapter = FakeBrokerAdapter(cash=10000)
    loop = LiveExecutionLoop(adapter, journal=JsonlOrderJournal(tmp_path / "live.jsonl"))
    request = LiveOrderRequest(symbol="AAA", side="BUY", quantity="1", client_order_id="client-1")

    def record_local_ack(acknowledgement):
        broker.orders[acknowledgement.client_order_id] = Order(
            symbol=acknowledgement.symbol,
            side=acknowledgement.side,
            quantity=acknowledgement.quantity,
            submitted_at=acknowledgement.submitted_at,
            id=acknowledgement.client_order_id,
        )

    acknowledgement = loop.submit_order(broker, request, on_ack=record_local_ack)

    assert acknowledgement.client_order_id == "client-1"
    assert adapter.submitted == [request]


def test_live_execution_loop_requires_local_state_transition_before_post_reconcile(tmp_path):
    broker = PaperBroker(initial_cash=10000, commission_rate=0.0, slippage_rate=0.0)
    adapter = FakeBrokerAdapter(cash=10000)
    loop = LiveExecutionLoop(adapter, journal=JsonlOrderJournal(tmp_path / "live.jsonl"))
    request = LiveOrderRequest(symbol="AAA", side="BUY", quantity="1", client_order_id="client-1")

    with pytest.raises(LiveReconciliationError, match="post_submit"):
        loop.submit_order(broker, request)


def test_live_execution_loop_submit_rolls_back_on_ack_exception(tmp_path):
    broker = PaperBroker(initial_cash=10000, commission_rate=0.0, slippage_rate=0.0)
    adapter = FakeBrokerAdapter(cash=10000)
    loop = LiveExecutionLoop(adapter, journal=JsonlOrderJournal(tmp_path / "live.jsonl"))
    request = LiveOrderRequest(symbol="AAA", side="BUY", quantity="1", client_order_id="client-1")

    def broken_ack(acknowledgement):
        broker.orders[acknowledgement.client_order_id] = Order(
            acknowledgement.symbol,
            acknowledgement.side,
            acknowledgement.quantity,
            acknowledgement.submitted_at,
            id=acknowledgement.client_order_id,
        )
        raise RuntimeError("local transition failed")

    with pytest.raises(RuntimeError, match="local transition failed"):
        loop.submit_order(broker, request, on_ack=broken_ack)

    assert broker.orders == {}
    assert any(record["event_type"] == "live_order_transition_error" for record in loop.journal.read_all())


def test_live_execution_loop_submit_rolls_back_on_failed_post_reconcile(tmp_path):
    broker = PaperBroker(initial_cash=10000, commission_rate=0.0, slippage_rate=0.0)
    adapter = FakeBrokerAdapter(cash=10000)
    loop = LiveExecutionLoop(adapter, journal=JsonlOrderJournal(tmp_path / "live.jsonl"))
    request = LiveOrderRequest(symbol="AAA", side="BUY", quantity="1", client_order_id="client-1")

    def wrong_local_ack(acknowledgement):
        broker.orders[acknowledgement.client_order_id] = Order(
            acknowledgement.symbol,
            acknowledgement.side,
            2,
            acknowledgement.submitted_at,
            id=acknowledgement.client_order_id,
        )

    with pytest.raises(LiveReconciliationError, match="post_submit"):
        loop.submit_order(broker, request, on_ack=wrong_local_ack)

    assert broker.orders == {}
    assert any(record["event_type"] == "live_order_transition_error" for record in loop.journal.read_all())


def test_live_execution_loop_journals_intent_ack_and_transition_error(tmp_path):
    broker = PaperBroker(initial_cash=10000, commission_rate=0.0, slippage_rate=0.0)
    adapter = FakeBrokerAdapter(cash=10000)
    journal = JsonlOrderJournal(tmp_path / "live.jsonl")
    loop = LiveExecutionLoop(adapter, journal=journal)
    request = LiveOrderRequest(symbol="AAA", side="BUY", quantity="1", client_order_id="client-1")

    def broken_local_transition(_acknowledgement):
        raise RuntimeError("local disk unavailable")

    with pytest.raises(RuntimeError, match="local disk unavailable"):
        loop.submit_order(broker, request, on_ack=broken_local_transition)

    assert adapter.orders
    assert broker.orders == {}
    event_types = [record["event_type"] for record in journal.read_all()]
    assert "live_order_intent" in event_types
    assert "live_order_ack" in event_types
    assert "live_order_transition_error" in event_types


def test_live_execution_loop_runs_local_transition_even_if_ack_journal_fails():
    broker = PaperBroker(initial_cash=10000, commission_rate=0.0, slippage_rate=0.0)
    adapter = FakeBrokerAdapter(cash=10000)
    loop = LiveExecutionLoop(adapter, journal=FailingAckJournal())
    request = LiveOrderRequest(symbol="AAA", side="BUY", quantity="1", client_order_id="client-1")

    def record_local_ack(acknowledgement):
        broker.orders[acknowledgement.client_order_id] = Order(
            symbol=acknowledgement.symbol,
            side=acknowledgement.side,
            quantity=acknowledgement.quantity,
            submitted_at=acknowledgement.submitted_at,
            id=acknowledgement.client_order_id,
        )

    with pytest.raises(OSError, match="disk full"):
        loop.submit_order(broker, request, on_ack=record_local_ack)

    assert "client-1" in broker.orders
    assert adapter.orders
    retry_ack = loop.submit_order(broker, request, on_ack=record_local_ack)
    assert retry_ack.external_id == "broker-ack"
    assert adapter.submitted == [request]

    conflicting = LiveOrderRequest(symbol="BBB", side="BUY", quantity="2", client_order_id="client-1")
    with pytest.raises(ValueError, match="idempotency_conflict"):
        loop.submit_order(broker, conflicting, on_ack=record_local_ack)

    conflicting_contract = LiveOrderRequest(
        symbol="AAA",
        side="BUY",
        quantity="1",
        client_order_id="client-1",
        order_type="LIMIT",
        limit_price="99",
    )
    with pytest.raises(ValueError, match="idempotency_conflict"):
        loop.submit_order(broker, conflicting_contract, on_ack=record_local_ack)


def test_live_execution_loop_recovery_returns_reconciliation_report(tmp_path):
    broker = PaperBroker(initial_cash=10000, commission_rate=0.0, slippage_rate=0.0)
    adapter = FakeBrokerAdapter(cash=9999)
    journal = JsonlOrderJournal(tmp_path / "live.jsonl")
    loop = LiveExecutionLoop(adapter, journal=journal)

    report = loop.recover(broker)

    assert report.ok is False
    assert any(action["action"] == "reconcile_cash_to_broker_snapshot" for action in report.recovery_actions)
    assert any(record["event_type"] == "live_recovery_reconcile" for record in journal.read_all())


def test_live_execution_loop_cancel_dedup_updates_local_state_when_external_absent(tmp_path):
    broker = PaperBroker(initial_cash=10000, commission_rate=0.0, slippage_rate=0.0)
    broker.orders["client-1"] = Order("AAA", "BUY", 1, pd.Timestamp("2024-01-01"), id="client-1")
    adapter = FakeBrokerAdapter(cash=10000)
    loop = LiveExecutionLoop(adapter, journal=JsonlOrderJournal(tmp_path / "live.jsonl"))

    def mark_cancelled(_ack):
        broker.orders["client-1"].status = CANCELLED

    result = loop.cancel_order(broker, "already-gone", on_ack=mark_cancelled)

    assert result["status"] == "already_absent"
    assert adapter.cancelled == []
    assert broker.orders["client-1"].status == CANCELLED


def test_live_execution_loop_cancel_dedup_rolls_back_failed_transition(tmp_path):
    broker = PaperBroker(initial_cash=10000, commission_rate=0.0, slippage_rate=0.0)
    broker.orders["client-1"] = Order("AAA", "BUY", 1, pd.Timestamp("2024-01-01"), id="client-1")
    adapter = FakeBrokerAdapter(cash=10000)
    adapter.orders.append(
        ExternalOrder(
            external_id="actual-open",
            client_order_id="client-1",
            symbol="AAA",
            side="BUY",
            quantity=1,
            filled_quantity=0,
            status="ACCEPTED",
            submitted_at=pd.Timestamp("2024-01-01"),
        )
    )
    loop = LiveExecutionLoop(adapter, journal=JsonlOrderJournal(tmp_path / "live.jsonl"))

    def wrongly_mark_cancelled(_ack):
        broker.orders["client-1"].status = CANCELLED

    with pytest.raises(LiveReconciliationError, match="post_cancel_deduplicated"):
        loop.cancel_order(broker, "wrong-external-id", on_ack=wrongly_mark_cancelled)

    assert broker.orders["client-1"].status == "OPEN"
    assert any(record["event_type"] == "live_cancel_transition_error" for record in loop.journal.read_all())


def test_live_execution_loop_cancel_dedup_rolls_back_on_ack_exception(tmp_path):
    broker = PaperBroker(initial_cash=10000, commission_rate=0.0, slippage_rate=0.0)
    broker.orders["client-1"] = Order("AAA", "BUY", 1, pd.Timestamp("2024-01-01"), id="client-1")
    adapter = FakeBrokerAdapter(cash=10000)
    loop = LiveExecutionLoop(adapter, journal=JsonlOrderJournal(tmp_path / "live.jsonl"))

    def broken_ack(_ack):
        broker.orders["client-1"].status = CANCELLED
        raise RuntimeError("local transition failed")

    with pytest.raises(RuntimeError, match="local transition failed"):
        loop.cancel_order(broker, "already-gone", on_ack=broken_ack)

    assert broker.orders["client-1"].status == "OPEN"
    assert any(record["event_type"] == "live_cancel_transition_error" for record in loop.journal.read_all())


def test_live_execution_loop_normal_cancel_rolls_back_on_failed_post_reconcile(tmp_path):
    broker = PaperBroker(initial_cash=10000, commission_rate=0.0, slippage_rate=0.0)
    broker.orders["client-1"] = Order("AAA", "BUY", 1, pd.Timestamp("2024-01-01"), id="client-1")
    adapter = FakeBrokerAdapter(cash=10000)
    adapter.orders.append(
        ExternalOrder(
            external_id="actual-open",
            client_order_id="client-1",
            symbol="AAA",
            side="BUY",
            quantity=1,
            filled_quantity=0,
            status="ACCEPTED",
            submitted_at=pd.Timestamp("2024-01-01"),
        )
    )
    loop = LiveExecutionLoop(adapter, journal=JsonlOrderJournal(tmp_path / "live.jsonl"))

    with pytest.raises(LiveReconciliationError, match="post_cancel"):
        loop.cancel_order(broker, "actual-open")

    assert broker.orders["client-1"].status == "OPEN"
    assert any(record["event_type"] == "live_cancel_transition_error" for record in loop.journal.read_all())


def test_live_execution_loop_recovery_suggests_cancel_for_local_closed_external_open(tmp_path):
    broker = PaperBroker(initial_cash=10000, commission_rate=0.0, slippage_rate=0.0)
    order = Order("AAA", "BUY", Decimal("1"), pd.Timestamp("2024-01-01"), id="client-1")
    order.status = CANCELLED
    broker.orders[order.id] = order
    adapter = FakeBrokerAdapter(cash=10000)
    adapter.orders.append(
        ExternalOrder(
            external_id="broker-open",
            client_order_id="client-1",
            symbol="AAA",
            side="BUY",
            quantity=Decimal("1"),
            filled_quantity=Decimal("0"),
            status="ACCEPTED",
            submitted_at=pd.Timestamp("2024-01-01"),
        )
    )
    loop = LiveExecutionLoop(adapter, journal=JsonlOrderJournal(tmp_path / "live.jsonl"))

    report = loop.recover(broker)

    assert any(action["action"] == "cancel_external_order" for action in report.recovery_actions)


def test_live_execution_loop_recovery_suggests_action_for_missing_external_order(tmp_path):
    broker = PaperBroker(initial_cash=10000, commission_rate=0.0, slippage_rate=0.0)
    broker.orders["client-1"] = Order("AAA", "BUY", 1, pd.Timestamp("2024-01-01"), id="client-1")
    adapter = FakeBrokerAdapter(cash=10000)
    loop = LiveExecutionLoop(adapter, journal=JsonlOrderJournal(tmp_path / "live.jsonl"))

    report = loop.recover(broker)

    assert any(
        action["action"] == "mark_local_order_closed_or_resubmit_missing_external_order"
        for action in report.recovery_actions
    )


def test_live_execution_loop_recovery_suggests_halt_for_order_contract_drift(tmp_path):
    broker = PaperBroker(initial_cash=10000, commission_rate=0.0, slippage_rate=0.0)
    broker.orders["client-1"] = Order("AAA", "BUY", 1, pd.Timestamp("2024-01-01"), id="client-1")
    adapter = FakeBrokerAdapter(cash=10000)
    adapter.orders.append(
        ExternalOrder(
            external_id="broker-1",
            client_order_id="client-1",
            symbol="AAA",
            side="BUY",
            quantity=1,
            filled_quantity=0,
            status="ACCEPTED",
            submitted_at=pd.Timestamp("2024-01-01"),
            order_type="LIMIT",
            limit_price="99",
        )
    )
    loop = LiveExecutionLoop(adapter, journal=JsonlOrderJournal(tmp_path / "live.jsonl"))

    report = loop.recover(broker)

    assert any(action["action"] == "halt_and_reconcile_order_contract" for action in report.recovery_actions)


def test_live_execution_loop_applies_fill_stream_idempotently(tmp_path):
    broker = PaperBroker(initial_cash=10000, commission_rate=0.0, slippage_rate=0.0)
    adapter = FakeBrokerAdapter(cash=10000)
    loop = LiveExecutionLoop(adapter, journal=JsonlOrderJournal(tmp_path / "live.jsonl"))
    order = Order("AAA", "BUY", Decimal("1"), pd.Timestamp("2024-01-01"), id="client-1")
    broker.orders[order.id] = order
    adapter.fills.append(
        ExternalFill(
            external_fill_id="fill-1",
            external_order_id="broker-order",
            client_order_id="client-1",
            symbol="AAA",
            side="BUY",
            quantity=Decimal("1"),
            price=Decimal("10"),
            commission=Decimal("0"),
            timestamp=pd.Timestamp("2024-01-02"),
        )
    )

    first = loop.process_fills(broker)
    second = loop.process_fills(broker)

    assert len(first) == 1
    assert second == []
    assert broker.positions["AAA"].quantity == Decimal("1")
    assert broker.cash == 9990
    assert order.status == "FILLED"


def test_live_execution_loop_preserves_decimal_fill_precision(tmp_path):
    broker = PaperBroker(initial_cash=10000, commission_rate=0.0, slippage_rate=0.0)
    adapter = FakeBrokerAdapter(cash=10000)
    loop = LiveExecutionLoop(adapter, journal=JsonlOrderJournal(tmp_path / "live.jsonl"))
    quantity = Decimal("0.123456789123456789")
    price = Decimal("123.456789123456789")
    commission = Decimal("0.000000001")
    order = Order("AAA", "BUY", quantity, pd.Timestamp("2024-01-01"), id="client-1")
    broker.orders[order.id] = order
    adapter.fills.append(
        ExternalFill(
            external_fill_id="fill-precise",
            external_order_id="broker-order",
            client_order_id="client-1",
            symbol="AAA",
            side="BUY",
            quantity=quantity,
            price=price,
            commission=commission,
            timestamp=pd.Timestamp("2024-01-02"),
        )
    )

    loop.process_fills(broker)

    assert broker.positions["AAA"].quantity == quantity
    assert broker.positions["AAA"].avg_price == price
    assert broker.fills[-1].quantity == quantity
    assert broker.fills[-1].price == price
    assert order.filled_quantity == quantity
    assert order.avg_fill_price == price
    assert broker.cash == Decimal("10000.0") - quantity * price - commission

    state_path = tmp_path / "broker-state.json"
    broker.save_state(state_path)
    restored = PaperBroker.load_state(state_path)
    assert restored.cash == broker.cash
    assert restored.positions["AAA"].quantity == quantity
    assert restored.orders["client-1"].avg_fill_price == price
    assert restored.fills[-1].price == price
    assert "fill-precise" in restored.external_fill_ids
    assert restored.reconcile({"AAA": quantity}, expected_cash=restored.cash) == []

    decimal_snapshot = AccountSnapshot(
        timestamp=pd.Timestamp("2024-01-03"),
        cash=restored.cash,
        positions={"AAA": quantity},
        orders=[],
    )
    assert BrokerReconciler().reconcile(restored, decimal_snapshot).ok is True
    drift_snapshot = AccountSnapshot(
        timestamp=pd.Timestamp("2024-01-03"),
        cash=restored.cash + Decimal("0.01"),
        positions={"AAA": quantity},
        orders=[],
    )
    drift_report = BrokerReconciler().reconcile(restored, drift_snapshot)
    assert drift_report.cash_mismatch["difference"] == Decimal("0.01")

    risk = PaperRiskManager(initial_equity=10000, max_drawdown=0.9, min_cash_fraction=0.01)
    allowed, reason = risk.validate_order(restored, Order("BBB", "BUY", 1, pd.Timestamp("2024-01-03")), {"AAA": price})
    assert allowed, reason
    restored.risk_manager = risk
    restored.orders["client-2"] = Order("BBB", "BUY", 1, pd.Timestamp("2024-01-03"), id="client-2")
    fills = restored.process_bar(
        _bar("BBB", "2024-01-04", open_price=10, close=11),
        reference_prices={"AAA": price},
    )
    assert len(fills) == 1
    assert restored.orders["client-2"].status == "FILLED"


def test_live_execution_loop_rolls_back_fill_when_applied_journal_fails():
    broker = PaperBroker(initial_cash=10000, commission_rate=0.0, slippage_rate=0.0)
    broker.orders["client-1"] = Order("AAA", "BUY", 1, pd.Timestamp("2024-01-01"), id="client-1")
    adapter = FakeBrokerAdapter(cash=10000)
    adapter.fills.append(_external_fill("fill-1", "client-1"))
    loop = LiveExecutionLoop(adapter, journal=FailingFillAppliedJournal())

    with pytest.raises(OSError, match="disk full"):
        loop.process_fills(broker)

    assert broker.cash == 10000
    assert broker.positions == {}
    assert "fill-1" not in loop.processed_external_fills


def test_live_execution_loop_rejects_custom_on_fill_hook(tmp_path):
    broker = PaperBroker(initial_cash=10000, commission_rate=0.0, slippage_rate=0.0)
    adapter = FakeBrokerAdapter(cash=10000)
    loop = LiveExecutionLoop(adapter, journal=JsonlOrderJournal(tmp_path / "live.jsonl"))

    with pytest.raises(ValueError, match="Custom on_fill"):
        loop.process_fills(broker, on_fill=lambda _fill: None)


def test_live_execution_loop_does_not_lose_unmanaged_fill_before_attach(tmp_path):
    broker = PaperBroker(initial_cash=10000, commission_rate=0.0, slippage_rate=0.0)
    adapter = FakeBrokerAdapter(cash=10000)
    loop = LiveExecutionLoop(adapter, journal=JsonlOrderJournal(tmp_path / "live.jsonl"))
    adapter.fills.append(_external_fill("fill-1", "client-1"))

    first = loop.process_fills(broker)
    broker.orders["client-1"] = Order("AAA", "BUY", 1, pd.Timestamp("2024-01-01"), id="client-1")
    second = loop.process_fills(broker)

    assert first == []
    assert len(second) == 1
    assert broker.positions["AAA"].quantity == 1.0


def test_live_execution_loop_rejects_fill_mismatches(tmp_path):
    broker = PaperBroker(initial_cash=10000, commission_rate=0.0, slippage_rate=0.0)
    adapter = FakeBrokerAdapter(cash=10000)
    loop = LiveExecutionLoop(adapter, journal=JsonlOrderJournal(tmp_path / "live.jsonl"))
    broker.orders["client-1"] = Order("AAA", "BUY", 1, pd.Timestamp("2024-01-01"), id="client-1")
    adapter.fills.append(_external_fill("fill-1", "client-1", side="SELL"))

    applied = loop.process_fills(broker)

    assert applied == []
    assert "AAA" not in broker.positions
    assert any(
        record["event_type"] == "live_fill_rejected"
        for record in loop.journal.read_all()
    )


def test_live_execution_loop_rejects_sell_fill_without_inventory(tmp_path):
    broker = PaperBroker(initial_cash=10000, commission_rate=0.0, slippage_rate=0.0)
    adapter = FakeBrokerAdapter(cash=10000)
    loop = LiveExecutionLoop(adapter, journal=JsonlOrderJournal(tmp_path / "live.jsonl"))
    broker.orders["client-1"] = Order("AAA", "SELL", 1, pd.Timestamp("2024-01-01"), id="client-1")
    adapter.fills.append(_external_fill("fill-1", "client-1", side="SELL"))

    applied = loop.process_fills(broker)

    assert applied == []
    assert broker.cash == 10000
    assert "AAA" not in broker.positions
    assert any(
        "inventory" in record["payload"]["reason"]
        for record in loop.journal.read_all()
        if record["event_type"] == "live_fill_rejected"
    )


def test_live_execution_loop_restores_fill_idempotency_from_journal(tmp_path):
    journal = JsonlOrderJournal(tmp_path / "live.jsonl")
    broker = PaperBroker(initial_cash=10000, commission_rate=0.0, slippage_rate=0.0)
    broker.orders["client-1"] = Order("AAA", "BUY", 1, pd.Timestamp("2024-01-01"), id="client-1")
    adapter = FakeBrokerAdapter(cash=10000)
    adapter.fills.append(_external_fill("fill-1", "client-1"))
    first_loop = LiveExecutionLoop(adapter, journal=journal)
    first_loop.process_fills(broker)
    cash_after_first = broker.cash

    second_loop = LiveExecutionLoop(adapter, journal=journal)
    applied = second_loop.process_fills(broker)

    assert applied == []
    assert broker.cash == cash_after_first


def test_live_execution_loop_replays_applied_fills_from_journal(tmp_path):
    journal = JsonlOrderJournal(tmp_path / "live.jsonl")
    adapter = FakeBrokerAdapter(cash=10000)
    adapter.fills.append(_external_fill("fill-1", "client-1"))
    source = PaperBroker(initial_cash=10000, commission_rate=0.0, slippage_rate=0.0)
    source.orders["client-1"] = Order("AAA", "BUY", 1, pd.Timestamp("2024-01-01"), id="client-1")
    LiveExecutionLoop(adapter, journal=journal).process_fills(source)

    restored = PaperBroker(initial_cash=10000, commission_rate=0.0, slippage_rate=0.0)
    restored.orders["client-1"] = Order("AAA", "BUY", 1, pd.Timestamp("2024-01-01"), id="client-1")
    replayed = LiveExecutionLoop(adapter, journal=journal).replay_applied_fills(restored)

    assert len(replayed) == 1
    assert restored.positions["AAA"].quantity == 1.0
    assert restored.cash == 9990


def test_live_execution_loop_preflight_replays_applied_fills_before_reconcile(tmp_path):
    journal = JsonlOrderJournal(tmp_path / "live.jsonl")
    adapter = FakeBrokerAdapter(cash=10000)
    adapter.fills.append(_external_fill("fill-1", "client-1"))
    source = PaperBroker(initial_cash=10000, commission_rate=0.0, slippage_rate=0.0)
    source.orders["client-1"] = Order("AAA", "BUY", 1, pd.Timestamp("2024-01-01"), id="client-1")
    LiveExecutionLoop(adapter, journal=journal).process_fills(source)

    restored = PaperBroker(initial_cash=10000, commission_rate=0.0, slippage_rate=0.0)
    restored.orders["client-1"] = Order("AAA", "BUY", 1, pd.Timestamp("2024-01-01"), id="client-1")
    adapter.cash = 9990
    adapter.positions = {"AAA": Decimal("1")}
    LiveExecutionLoop(adapter, journal=journal).preflight(restored)

    assert restored.positions["AAA"].quantity == 1.0
    assert restored.cash == 9990


def test_external_fill_validates_required_fields():
    with pytest.raises(ValueError, match="quantity"):
        _external_fill("fill-1", "client-1", quantity="NaN")


def test_live_execution_loop_blocks_submit_on_reconciliation_mismatch(tmp_path):
    broker = PaperBroker(initial_cash=10000, commission_rate=0.0, slippage_rate=0.0)
    adapter = FakeBrokerAdapter(cash=9000)
    loop = LiveExecutionLoop(adapter, journal=JsonlOrderJournal(tmp_path / "live.jsonl"))
    request = LiveOrderRequest(symbol="AAA", side="BUY", quantity="1", client_order_id="client-1")

    with pytest.raises(LiveReconciliationError, match="pre_submit"):
        loop.submit_order(broker, request)

    assert adapter.submitted == []


def test_live_execution_loop_requires_durable_journal():
    with pytest.raises(ValueError, match="durable journal"):
        LiveExecutionLoop(FakeBrokerAdapter(cash=10000))


class FakeBrokerAdapter:
    def __init__(self, cash=10000):
        self.cash = cash
        self.positions = {}
        self.orders = []
        self.fills = []
        self.submitted = []
        self.cancelled = []

    def submit_order(self, request):
        self.submitted.append(request)
        acknowledgement = ExternalOrder(
            external_id="broker-ack",
            client_order_id=request.client_order_id,
            symbol=request.symbol,
            side=request.side,
            quantity=request.quantity,
            filled_quantity=Decimal("0"),
            status="ACCEPTED",
            submitted_at=pd.Timestamp("2024-01-02"),
            order_type=request.order_type,
            time_in_force=request.time_in_force,
            limit_price=request.limit_price,
            stop_price=request.stop_price,
            reduce_only=request.reduce_only,
        )
        self.orders.append(acknowledgement)
        return acknowledgement

    def cancel_order(self, external_id):
        self.cancelled.append(external_id)
        self.orders = [order for order in self.orders if order.external_id != external_id]
        return {"external_id": external_id, "status": "cancel_requested"}

    def account_snapshot(self):
        return AccountSnapshot(
            timestamp=pd.Timestamp("2024-01-02"),
            cash=self.cash,
            positions=dict(self.positions),
            orders=list(self.orders),
        )

    def stream_fills(self, since=None):
        return list(self.fills)


class FailingAckJournal:
    def __init__(self):
        self.records = []

    def append(self, event_type, payload):
        if event_type == "live_order_ack":
            raise OSError("disk full")
        self.records.append({"event_type": event_type, "payload": payload})
        return self.records[-1]

    def read_all(self):
        return list(self.records)


class FailingFillAppliedJournal(FailingAckJournal):
    def append(self, event_type, payload):
        if event_type == "live_fill_applied":
            raise OSError("disk full")
        self.records.append({"event_type": event_type, "payload": payload})
        return self.records[-1]


def _external_fill(fill_id, client_order_id, side="BUY", quantity="1"):
    return ExternalFill(
        external_fill_id=fill_id,
        external_order_id="broker-order",
        client_order_id=client_order_id,
        symbol="AAA",
        side=side,
        quantity=Decimal(quantity),
        price=Decimal("10"),
        commission=Decimal("0"),
        timestamp=pd.Timestamp("2024-01-02"),
    )


def _bar(symbol, timestamp, open_price, close):
    return MarketBar(
        symbol=symbol,
        timestamp=pd.Timestamp(timestamp),
        open=open_price,
        high=max(open_price, close),
        low=min(open_price, close),
        close=close,
        volume=1000,
    )
