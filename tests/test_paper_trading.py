import pandas as pd
from decimal import Decimal

from trading.paper import (
    BUY,
    CANCELLED,
    FILLED,
    PARTIALLY_FILLED,
    REJECTED,
    SELL,
    MarketBar,
    Order,
    PaperBroker,
    PaperRiskManager,
    Position,
)


def bar(symbol="TEST", timestamp="2024-01-02", open_price=100.0, close_price=101.0, volume=1000):
    return MarketBar(
        symbol=symbol,
        timestamp=pd.Timestamp(timestamp),
        open=open_price,
        high=max(open_price, close_price),
        low=min(open_price, close_price),
        close=close_price,
        volume=volume,
    )


def test_paper_broker_fills_order_and_updates_cash_position():
    broker = PaperBroker(initial_cash=10000, commission_rate=0.0, slippage_rate=0.0)

    order = broker.submit_order("TEST", BUY, 10, pd.Timestamp("2024-01-01"))
    fills = broker.process_bar(bar())

    assert order.status == FILLED
    assert len(fills) == 1
    assert broker.positions["TEST"].quantity == 10
    assert broker.cash == 9000


def test_order_validates_order_contract_fields():
    try:
        Order("TEST", BUY, 1, pd.Timestamp("2024-01-01"), order_type="LIMIT")
    except ValueError as exc:
        assert "limit_price" in str(exc)
    else:
        raise AssertionError("Expected missing limit_price to raise ValueError")

    try:
        Order("TEST", BUY, 1, pd.Timestamp("2024-01-01"), reduce_only="False")
    except ValueError as exc:
        assert "reduce_only" in str(exc)
    else:
        raise AssertionError("Expected non-bool reduce_only to raise ValueError")

    try:
        Order("TEST", BUY, 1, pd.Timestamp("2024-01-01"), filled_quantity=2)
    except ValueError as exc:
        assert "filled_quantity" in str(exc)
    else:
        raise AssertionError("Expected overfilled order to raise ValueError")

    try:
        Order("TEST", BUY, 1, pd.Timestamp("2024-01-01"), status="GHOST")
    except ValueError as exc:
        assert "status" in str(exc)
    else:
        raise AssertionError("Expected invalid status to raise ValueError")

    try:
        Order("TEST", BUY, 1, pd.Timestamp("2024-01-01"), status="FILLED", filled_quantity=0)
    except ValueError as exc:
        assert "FILLED" in str(exc)
    else:
        raise AssertionError("Expected inconsistent FILLED order to raise ValueError")

    try:
        Order("TEST", BUY, 1, pd.Timestamp("2024-01-01"), order_type="LIMIT", limit_price=float("nan"))
    except ValueError as exc:
        assert "limit_price" in str(exc)
    else:
        raise AssertionError("Expected NaN limit_price to raise ValueError")

    try:
        Order("TEST", BUY, float("nan"), pd.Timestamp("2024-01-01"))
    except ValueError as exc:
        assert "quantity" in str(exc)
    else:
        raise AssertionError("Expected NaN quantity to raise ValueError")


def test_paper_broker_partial_fill_respects_participation_limit():
    broker = PaperBroker(
        initial_cash=100000,
        commission_rate=0.0,
        slippage_rate=0.0,
        max_participation_rate=0.1,
    )

    order = broker.submit_order("TEST", BUY, 100, pd.Timestamp("2024-01-01"))
    broker.process_bar(bar(volume=50))

    assert order.status == PARTIALLY_FILLED
    assert order.filled_quantity == 5
    assert order.remaining_quantity == 95


def test_paper_broker_does_not_fill_orders_before_submission_time():
    broker = PaperBroker(initial_cash=10000, commission_rate=0.0, slippage_rate=0.0)

    order = broker.submit_order("TEST", BUY, 10, pd.Timestamp("2024-01-05"))
    fills = broker.process_bar(bar())

    assert fills == []
    assert order.status == "OPEN"
    assert "TEST" not in broker.positions
    assert broker.cash == 10000


def test_paper_broker_applies_participation_limit_across_all_orders():
    broker = PaperBroker(
        initial_cash=100000,
        commission_rate=0.0,
        slippage_rate=0.0,
        max_participation_rate=0.1,
    )

    broker.submit_order("TEST", BUY, 100, pd.Timestamp("2024-01-01"))
    broker.submit_order("TEST", BUY, 100, pd.Timestamp("2024-01-01"))
    fills = broker.process_bar(bar(volume=100))

    assert sum(fill.quantity for fill in fills) == 10
    assert broker.positions["TEST"].quantity == 10


def test_paper_broker_process_bar_is_idempotent():
    broker = PaperBroker(initial_cash=10000, commission_rate=0.0, slippage_rate=0.0)
    market_bar = bar()

    order = broker.submit_order("TEST", BUY, 20, pd.Timestamp("2024-01-01"))
    first_fills = broker.process_bar(market_bar)
    second_fills = broker.process_bar(market_bar)

    assert sum(fill.quantity for fill in first_fills) == 20
    assert second_fills == []
    assert order.filled_quantity == 20
    assert broker.positions["TEST"].quantity == 20


def test_paper_broker_rejects_short_sale_without_inventory():
    broker = PaperBroker(initial_cash=10000)

    order = broker.submit_order("TEST", SELL, 10, pd.Timestamp("2024-01-01"))
    broker.process_bar(bar())

    assert order.status == REJECTED
    assert "does not short" in order.reject_reason


def test_paper_broker_respects_projected_cash_reserve_on_fill():
    risk = PaperRiskManager(initial_equity=1000, max_drawdown=0.5, min_cash_fraction=0.1)
    broker = PaperBroker(
        initial_cash=1000,
        commission_rate=0.0,
        slippage_rate=0.0,
        max_participation_rate=1.0,
        risk_manager=risk,
    )

    broker.submit_order("TEST", BUY, 100, pd.Timestamp("2024-01-01"))
    broker.process_bar(bar(open_price=100.0, close_price=100.0, volume=100))

    assert broker.positions["TEST"].quantity == 9
    assert broker.cash == 100
    assert broker.cash >= broker.equity({"TEST": 100.0}) * risk.min_cash_fraction


def test_paper_risk_manager_rejects_after_drawdown_halt():
    broker = PaperBroker(initial_cash=10000, commission_rate=0.0, slippage_rate=0.0)
    risk = PaperRiskManager(initial_equity=10000, max_drawdown=0.1)

    risk.update(8000)
    order = broker.submit_order(
        "TEST",
        BUY,
        1,
        pd.Timestamp("2024-01-01"),
        risk_manager=risk,
        reference_prices={"TEST": 100.0},
    )

    assert order.status == REJECTED
    assert "max_drawdown" in order.reject_reason


def test_paper_broker_cancels_open_orders_when_risk_halts_on_bar():
    risk = PaperRiskManager(initial_equity=10000, max_drawdown=0.1)
    broker = PaperBroker(initial_cash=10000)

    order = broker.submit_order("TEST", BUY, 10, pd.Timestamp("2024-01-01"))
    broker.risk_manager = risk
    risk.update(8000)
    broker.process_bar(bar())

    assert order.status == CANCELLED


def test_paper_broker_allows_liquidation_sell_during_risk_halt():
    risk = PaperRiskManager(initial_equity=10000, max_drawdown=0.1)
    broker = PaperBroker(
        initial_cash=10000,
        commission_rate=0.0,
        slippage_rate=0.0,
        max_participation_rate=1.0,
        risk_manager=risk,
    )

    broker.submit_order("TEST", BUY, 10, pd.Timestamp("2024-01-01"))
    broker.process_bar(bar(open_price=100.0, close_price=100.0))
    risk.update(8000)

    sell = broker.submit_order("TEST", SELL, 10, pd.Timestamp("2024-01-02"))
    broker.process_bar(bar(timestamp="2024-01-03", open_price=90.0, close_price=90.0))

    assert sell.status == FILLED
    assert "TEST" not in broker.positions


def test_paper_broker_state_round_trip_and_reconciliation(tmp_path):
    risk = PaperRiskManager(initial_equity=10000, max_drawdown=0.2)
    risk.update(10500)
    broker = PaperBroker(initial_cash=10000, commission_rate=0.0, slippage_rate=0.0, risk_manager=risk)
    broker.submit_order("TEST", BUY, 10, pd.Timestamp("2024-01-01"))
    broker.process_bar(bar())
    risk.update(8000)

    state_path = tmp_path / "paper_state.json"
    broker.save_state(state_path)

    restored = PaperBroker.load_state(state_path)

    assert restored.cash == broker.cash
    assert restored.positions["TEST"].quantity == 10
    assert restored.risk_manager.peak_equity == risk.peak_equity
    assert restored.risk_manager.halted is True
    assert restored.processed_bars == broker.processed_bars
    assert restored.reconcile({"TEST": 10}, expected_cash=9000) == []


def test_paper_broker_equity_marks_unreferenced_positions_with_last_prices():
    broker = PaperBroker(initial_cash=10000, commission_rate=0.0, slippage_rate=0.0)
    broker.positions["AAA"] = Position("AAA", quantity=10, avg_price=100.0)
    broker.positions["BBB"] = Position("BBB", quantity=5, avg_price=50.0)
    broker.last_prices["BBB"] = 40.0

    equity = broker.equity({"AAA": 110.0})

    assert equity == 10000 + 10 * 110.0 + 5 * 40.0


def test_paper_broker_equity_rejects_stale_or_future_marks_when_enforced():
    broker = PaperBroker(initial_cash=10000, commission_rate=0.0, slippage_rate=0.0)
    broker.positions["AAA"] = Position("AAA", quantity=10, avg_price=100.0)
    broker.last_prices["AAA"] = 90.0
    broker.last_price_timestamps["AAA"] = pd.Timestamp("2024-01-01")

    try:
        broker.equity({}, asof=pd.Timestamp("2024-01-03"), max_price_age=pd.Timedelta(days=1))
    except ValueError as exc:
        assert "Stale market price" in str(exc)
    else:
        raise AssertionError("Expected stale mark to raise ValueError")

    try:
        broker.equity({}, asof=pd.Timestamp("2023-12-31"), max_price_age=pd.Timedelta(days=1))
    except ValueError as exc:
        assert "Future market price" in str(exc)
    else:
        raise AssertionError("Expected future mark to raise ValueError")


def test_paper_broker_equity_normalizes_naive_and_aware_mark_timestamps():
    broker = PaperBroker(initial_cash=10000, commission_rate=0.0, slippage_rate=0.0)
    broker.positions["AAA"] = Position("AAA", quantity=10, avg_price=100.0)
    broker.last_prices["AAA"] = 90.0
    broker.last_price_timestamps["AAA"] = pd.Timestamp("2024-01-01")

    equity = broker.equity(
        {},
        asof=pd.Timestamp("2024-01-01T12:00:00Z"),
        max_price_age=pd.Timedelta(days=1),
    )

    assert equity == 10000 + 10 * 90.0


def test_paper_broker_state_round_trip_preserves_last_prices(tmp_path):
    broker = PaperBroker(initial_cash=10000, commission_rate=0.0, slippage_rate=0.0)
    broker.submit_order("TEST", BUY, 10, pd.Timestamp("2024-01-01"))
    broker.process_bar(bar(open_price=100.0, close_price=98.0))

    state_path = tmp_path / "paper_state.json"
    broker.save_state(state_path)
    restored = PaperBroker.load_state(state_path)

    assert restored.last_prices == {"TEST": 98.0}
    assert restored.last_price_timestamps == {"TEST": pd.Timestamp("2024-01-02")}


def test_paper_broker_state_round_trip_preserves_external_fill_ids(tmp_path):
    broker = PaperBroker(initial_cash=10000, commission_rate=0.0, slippage_rate=0.0)
    broker.external_fill_ids.add("external-fill-1")

    state_path = tmp_path / "paper_state.json"
    broker.save_state(state_path)
    restored = PaperBroker.load_state(state_path)

    assert restored.external_fill_ids == {"external-fill-1"}


def test_paper_broker_state_round_trip_preserves_decimal_positions(tmp_path):
    broker = PaperBroker(initial_cash=10000, commission_rate=0.0, slippage_rate=0.0)
    broker.positions["AAA"] = Position("AAA", quantity=Decimal("1.5"), avg_price=Decimal("10.25"))

    state_path = tmp_path / "paper_state.json"
    broker.save_state(state_path)
    restored = PaperBroker.load_state(state_path)

    assert restored.positions["AAA"].quantity == Decimal("1.5")
    assert restored.positions["AAA"].avg_price == Decimal("10.25")
