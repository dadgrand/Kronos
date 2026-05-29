"""Paper trading primitives with explicit order lifecycle and reconciliation.

The paper broker is deliberately conservative: no short selling, no synthetic
liquidity beyond a configurable bar participation limit, and no hidden cash.
It is not a live broker adapter, but it gives the project a realistic execution
boundary for strategy dry-runs before any live integration is attempted.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal
import json
import math
from pathlib import Path
from typing import Iterable
from uuid import uuid4

import pandas as pd


OPEN = "OPEN"
PARTIALLY_FILLED = "PARTIALLY_FILLED"
FILLED = "FILLED"
CANCELLED = "CANCELLED"
REJECTED = "REJECTED"
BUY = "BUY"
SELL = "SELL"
ORDER_STATUSES = {OPEN, PARTIALLY_FILLED, FILLED, CANCELLED, REJECTED}
ORDER_TYPES = {"MARKET", "LIMIT", "STOP", "STOP_LIMIT"}
TIME_IN_FORCE = {"DAY", "GTC", "IOC", "FOK"}


@dataclass
class MarketBar:
    symbol: str
    timestamp: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class Position:
    symbol: str
    quantity: int = 0
    avg_price: float = 0.0


@dataclass
class Order:
    symbol: str
    side: str
    quantity: int
    submitted_at: pd.Timestamp
    id: str = ""
    status: str = OPEN
    filled_quantity: int = 0
    avg_fill_price: float = 0.0
    reject_reason: str = ""
    order_type: str = "MARKET"
    time_in_force: str = "DAY"
    limit_price: float | None = None
    stop_price: float | None = None
    reduce_only: bool = False

    def __post_init__(self):
        self.symbol = str(self.symbol)
        self.side = self.side.upper()
        self.order_type = self.order_type.upper()
        self.time_in_force = self.time_in_force.upper()
        self.submitted_at = pd.Timestamp(self.submitted_at)
        if not self.id:
            self.id = uuid4().hex
        if self.side not in {BUY, SELL}:
            raise ValueError("Order side must be BUY or SELL.")
        if self.status not in ORDER_STATUSES:
            raise ValueError(f"Order status must be one of {sorted(ORDER_STATUSES)}.")
        if not math.isfinite(float(self.quantity)) or self.quantity <= 0:
            raise ValueError("Order quantity must be positive.")
        if not math.isfinite(float(self.filled_quantity)):
            raise ValueError("filled_quantity must be finite.")
        if self.filled_quantity < 0:
            raise ValueError("filled_quantity cannot be negative.")
        if self.filled_quantity > self.quantity:
            raise ValueError("filled_quantity cannot exceed quantity.")
        if self.status == OPEN and self.filled_quantity != 0:
            raise ValueError("OPEN orders cannot have filled_quantity.")
        if self.status == PARTIALLY_FILLED and not 0 < self.filled_quantity < self.quantity:
            raise ValueError("PARTIALLY_FILLED orders require 0 < filled_quantity < quantity.")
        if self.status == FILLED and self.filled_quantity != self.quantity:
            raise ValueError("FILLED orders require filled_quantity to equal quantity.")
        if self.status in {CANCELLED, REJECTED} and self.filled_quantity == self.quantity:
            raise ValueError("Closed unfilled statuses cannot be fully filled.")
        if self.order_type not in ORDER_TYPES:
            raise ValueError(f"order_type must be one of {sorted(ORDER_TYPES)}.")
        if self.time_in_force not in TIME_IN_FORCE:
            raise ValueError(f"time_in_force must be one of {sorted(TIME_IN_FORCE)}.")
        if not isinstance(self.reduce_only, bool):
            raise ValueError("reduce_only must be a boolean.")
        if self.order_type in {"LIMIT", "STOP_LIMIT"} and self.limit_price is None:
            raise ValueError("limit_price is required for limit orders.")
        if self.order_type in {"STOP", "STOP_LIMIT"} and self.stop_price is None:
            raise ValueError("stop_price is required for stop orders.")
        if self.limit_price is not None and (not math.isfinite(float(self.limit_price)) or self.limit_price <= 0):
            raise ValueError("limit_price must be positive.")
        if self.stop_price is not None and (not math.isfinite(float(self.stop_price)) or self.stop_price <= 0):
            raise ValueError("stop_price must be positive.")

    @property
    def remaining_quantity(self):
        return max(self.quantity - self.filled_quantity, 0)


@dataclass
class Fill:
    order_id: str
    symbol: str
    side: str
    quantity: int
    price: float
    commission: float
    timestamp: pd.Timestamp


class PaperRiskManager:
    """Minimal account-level risk controls for paper trading."""

    def __init__(self, initial_equity, max_drawdown=0.2, min_cash_fraction=0.01):
        if initial_equity <= 0:
            raise ValueError("initial_equity must be positive.")
        if not 0 < max_drawdown < 1:
            raise ValueError("max_drawdown must be in (0, 1).")
        if not 0 <= min_cash_fraction < 1:
            raise ValueError("min_cash_fraction must be in [0, 1).")

        self.initial_equity = float(initial_equity)
        self.peak_equity = float(initial_equity)
        self.max_drawdown = max_drawdown
        self.min_cash_fraction = min_cash_fraction
        self.halted = False
        self.halt_reason = ""

    def update(self, equity):
        equity = float(equity)
        self.peak_equity = max(self.peak_equity, equity)
        drawdown = equity / self.peak_equity - 1.0
        if drawdown <= -self.max_drawdown:
            self.halted = True
            self.halt_reason = f"max_drawdown breached: {drawdown:.2%}"
        return not self.halted

    def validate_order(self, broker: "PaperBroker", order: Order, reference_prices: dict[str, float]):
        equity = broker.equity(reference_prices)
        self.update(equity)
        if self.halted:
            if order.side == SELL:
                return True, ""
            return False, self.halt_reason

        if order.side == BUY and equity > 0:
            if isinstance(equity, Decimal) or isinstance(broker.cash, Decimal):
                min_cash = PaperBroker._decimal(equity) * PaperBroker._decimal(self.min_cash_fraction)
                broker_cash = PaperBroker._decimal(broker.cash)
            else:
                min_cash = equity * self.min_cash_fraction
                broker_cash = broker.cash
            if broker_cash <= min_cash:
                return False, "minimum cash reserve breached"

        return True, ""


class PaperBroker:
    """A deterministic paper broker with partial fills and JSON state."""

    def __init__(
        self,
        initial_cash,
        commission_rate=0.001,
        slippage_rate=0.0005,
        max_participation_rate=0.1,
        risk_manager: PaperRiskManager | None = None,
    ):
        if initial_cash <= 0:
            raise ValueError("initial_cash must be positive.")
        if not 0 <= commission_rate < 1:
            raise ValueError("commission_rate must be in [0, 1).")
        if not 0 <= slippage_rate < 1:
            raise ValueError("slippage_rate must be in [0, 1).")
        if not 0 < max_participation_rate <= 1:
            raise ValueError("max_participation_rate must be in (0, 1].")

        self.initial_cash = float(initial_cash)
        self.cash = float(initial_cash)
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.max_participation_rate = max_participation_rate
        self.risk_manager = risk_manager
        self.positions: dict[str, Position] = {}
        self.orders: dict[str, Order] = {}
        self.fills: list[Fill] = []
        self.processed_bars: set[str] = set()
        self.last_prices: dict[str, float] = {}
        self.last_price_timestamps: dict[str, pd.Timestamp] = {}
        self.external_fill_ids: set[str] = set()

    def submit_order(self, symbol, side, quantity, submitted_at, risk_manager=None, reference_prices=None):
        order = Order(symbol=symbol, side=side, quantity=int(quantity), submitted_at=pd.Timestamp(submitted_at))
        reference_prices = reference_prices or {}
        risk_manager = risk_manager or self.risk_manager

        if risk_manager is not None:
            allowed, reason = risk_manager.validate_order(self, order, reference_prices)
            if not allowed:
                order.status = REJECTED
                order.reject_reason = reason
                self.orders[order.id] = order
                return order

        self.orders[order.id] = order
        return order

    def cancel_order(self, order_id):
        order = self.orders[order_id]
        if order.status in {OPEN, PARTIALLY_FILLED}:
            order.status = CANCELLED
        return order

    def process_bar(self, bar: MarketBar, risk_manager=None, reference_prices=None):
        bar.timestamp = pd.Timestamp(bar.timestamp)
        if bar.open <= 0 or bar.close <= 0 or bar.volume < 0:
            raise ValueError("MarketBar prices must be positive and volume cannot be negative.")

        fills = []
        bar_key = f"{bar.symbol}|{bar.timestamp.isoformat()}"
        if bar_key in self.processed_bars:
            return fills
        self.processed_bars.add(bar_key)
        self.last_prices[bar.symbol] = float(bar.close)
        self.last_price_timestamps[bar.symbol] = pd.Timestamp(bar.timestamp)

        remaining_bar_capacity = int(bar.volume * self.max_participation_rate)
        if remaining_bar_capacity <= 0:
            return fills

        risk_manager = risk_manager or self.risk_manager
        reference_prices = dict(reference_prices or {})
        reference_prices.setdefault(bar.symbol, bar.open)
        risk_halted = False
        if risk_manager is not None:
            risk_manager.update(self.equity(reference_prices))
            if risk_manager.halted:
                risk_halted = True
                self.cancel_open_orders(side=BUY, reason=risk_manager.halt_reason)

        for order in list(self.orders.values()):
            if order.symbol != bar.symbol or order.status not in {OPEN, PARTIALLY_FILLED}:
                continue
            if risk_halted and order.side != SELL:
                continue
            if order.submitted_at > bar.timestamp:
                continue
            if remaining_bar_capacity <= 0:
                break

            available_quantity = min(order.remaining_quantity, remaining_bar_capacity)
            if available_quantity <= 0:
                continue

            if order.side == BUY:
                decimal_path = self._uses_decimal_order_path(order)
                if decimal_path:
                    fill_price = self._decimal(bar.open) * (Decimal("1") + self._decimal(self.slippage_rate))
                else:
                    fill_price = bar.open * (1 + self.slippage_rate)
                available_cash = self.cash
                if risk_manager is not None:
                    equity = self.equity(reference_prices)
                    if decimal_path or isinstance(equity, Decimal) or isinstance(self.cash, Decimal):
                        available_cash = max(
                            self._decimal(self.cash)
                            - self._decimal(equity) * self._decimal(risk_manager.min_cash_fraction),
                            Decimal("0"),
                        )
                    else:
                        available_cash = max(self.cash - equity * risk_manager.min_cash_fraction, 0.0)
                if decimal_path or isinstance(available_cash, Decimal):
                    price_with_commission = self._decimal(fill_price) * (
                        Decimal("1") + self._decimal(self.commission_rate)
                    )
                else:
                    price_with_commission = fill_price * (1 + self.commission_rate)
                affordable_quantity = int(available_cash / price_with_commission)
                fill_quantity = min(available_quantity, affordable_quantity)
                if fill_quantity <= 0:
                    order.status = REJECTED
                    order.reject_reason = "insufficient cash"
                    continue
            else:
                decimal_path = self._uses_decimal_order_path(order)
                if decimal_path:
                    fill_price = self._decimal(bar.open) * (Decimal("1") - self._decimal(self.slippage_rate))
                else:
                    fill_price = bar.open * (1 - self.slippage_rate)
                held_quantity = self.positions.get(order.symbol, Position(order.symbol)).quantity
                fill_quantity = min(available_quantity, held_quantity)
                if fill_quantity <= 0:
                    order.status = REJECTED
                    order.reject_reason = "insufficient long position; paper broker does not short"
                    continue

            if decimal_path:
                commission = self._decimal(fill_quantity) * self._decimal(fill_price) * self._decimal(self.commission_rate)
            else:
                commission = fill_quantity * fill_price * self.commission_rate
            self._apply_fill(order, fill_quantity, fill_price, commission, bar.timestamp)
            remaining_bar_capacity -= fill_quantity
            fills.append(self.fills[-1])

            if order.remaining_quantity == 0:
                order.status = FILLED
            else:
                order.status = PARTIALLY_FILLED

        return fills

    def cancel_open_orders(self, symbol=None, side=None, reason=""):
        cancelled = []
        for order in self.orders.values():
            if symbol is not None and order.symbol != symbol:
                continue
            if side is not None and order.side != side:
                continue
            if order.status in {OPEN, PARTIALLY_FILLED}:
                order.status = CANCELLED
                order.reject_reason = reason
                cancelled.append(order)
        return cancelled

    def _apply_fill(self, order, quantity, price, commission, timestamp):
        position = self.positions.get(order.symbol, Position(order.symbol))
        use_decimal = any(
            isinstance(value, Decimal)
            for value in (
                self.cash,
                quantity,
                price,
                commission,
                order.quantity,
                order.filled_quantity,
                order.avg_fill_price,
                position.quantity,
                position.avg_price,
            )
        )
        if use_decimal:
            quantity = self._decimal(quantity)
            price = self._decimal(price)
            commission = self._decimal(commission)
            self.cash = self._decimal(self.cash)
            position.quantity = self._decimal(position.quantity)
            position.avg_price = self._decimal(position.avg_price)
            order.quantity = self._decimal(order.quantity)
            order.filled_quantity = self._decimal(order.filled_quantity)
            order.avg_fill_price = self._decimal(order.avg_fill_price)

        if order.side == BUY:
            notional = quantity * price
            new_quantity = position.quantity + quantity
            position.avg_price = (
                (position.quantity * position.avg_price + notional) / new_quantity
                if new_quantity > 0
                else 0.0
            )
            position.quantity = new_quantity
            self.cash -= notional + commission
        else:
            notional = quantity * price
            position.quantity -= quantity
            self.cash += notional - commission
            if position.quantity == 0:
                position.avg_price = 0.0

        if position.quantity > 0:
            self.positions[order.symbol] = position
        else:
            self.positions.pop(order.symbol, None)

        previous_qty = order.filled_quantity
        order.filled_quantity += quantity
        order.avg_fill_price = (
            (previous_qty * order.avg_fill_price + quantity * price) / order.filled_quantity
        )

        self.fills.append(
            Fill(
                order_id=order.id,
                symbol=order.symbol,
                side=order.side,
                quantity=quantity,
                price=price,
                commission=commission,
                timestamp=pd.Timestamp(timestamp),
            )
        )

    def equity(self, reference_prices: dict[str, float], asof=None, max_price_age=None):
        total = self.cash
        for symbol, position in self.positions.items():
            if symbol in reference_prices:
                price = reference_prices[symbol]
            elif symbol in self.last_prices:
                if max_price_age is not None:
                    if asof is None:
                        raise ValueError("asof is required when max_price_age is enforced.")
                    mark_timestamp = self.last_price_timestamps.get(symbol)
                    if mark_timestamp is None:
                        raise ValueError(f"Missing mark timestamp for {symbol}.")
                    age = self._timestamp_age(asof, mark_timestamp)
                    if age < pd.Timedelta(0):
                        raise ValueError(f"Future market price for {symbol}: mark is after asof.")
                    if age > pd.Timedelta(max_price_age):
                        raise ValueError(f"Stale market price for {symbol}: age {age}.")
                price = self.last_prices[symbol]
            elif max_price_age is not None:
                raise ValueError(f"Missing market price for {symbol}.")
            else:
                price = position.avg_price
            if isinstance(total, Decimal) or isinstance(position.quantity, Decimal):
                total = self._decimal(total) + self._decimal(position.quantity) * self._decimal(price)
            else:
                total += position.quantity * float(price)
        return total

    @staticmethod
    def _decimal(value):
        return value if isinstance(value, Decimal) else Decimal(str(value))

    @staticmethod
    def _timestamp_age(asof, mark_timestamp):
        asof_ts = pd.Timestamp(asof)
        mark_ts = pd.Timestamp(mark_timestamp)
        if asof_ts.tzinfo is None:
            asof_ts = asof_ts.tz_localize("UTC")
        else:
            asof_ts = asof_ts.tz_convert("UTC")
        if mark_ts.tzinfo is None:
            mark_ts = mark_ts.tz_localize("UTC")
        else:
            mark_ts = mark_ts.tz_convert("UTC")
        return asof_ts - mark_ts

    def reconcile(self, expected_positions: dict[str, int], expected_cash=None, tolerance=1e-6):
        mismatches = []
        symbols = set(expected_positions) | set(self.positions)
        for symbol in sorted(symbols):
            expected_value = expected_positions.get(symbol, 0)
            actual_qty = self.positions.get(symbol, Position(symbol)).quantity
            if isinstance(expected_value, Decimal) or isinstance(actual_qty, Decimal):
                expected_qty = self._decimal(expected_value)
                actual_qty = self._decimal(actual_qty)
            else:
                expected_qty = int(expected_value)
            if actual_qty != expected_qty:
                mismatches.append(
                    {
                        "type": "position",
                        "symbol": symbol,
                        "expected": expected_qty,
                        "actual": actual_qty,
                    }
                )

        if expected_cash is not None:
            if isinstance(self.cash, Decimal) or isinstance(expected_cash, Decimal):
                cash_difference = abs(self._decimal(self.cash) - self._decimal(expected_cash))
                cash_tolerance = self._decimal(tolerance)
            else:
                cash_difference = abs(self.cash - float(expected_cash))
                cash_tolerance = tolerance
            if cash_difference > cash_tolerance:
                mismatches.append({"type": "cash", "expected": expected_cash, "actual": self.cash})

        return mismatches

    def save_state(self, path):
        path = Path(path)
        payload = {
            "initial_cash": self._serialize_number(self.initial_cash),
            "cash": self._serialize_number(self.cash),
            "commission_rate": self.commission_rate,
            "slippage_rate": self.slippage_rate,
            "max_participation_rate": self.max_participation_rate,
            "positions": {symbol: self._serialize_position(position) for symbol, position in self.positions.items()},
            "orders": {order_id: self._serialize_order(order) for order_id, order in self.orders.items()},
            "fills": [self._serialize_fill(fill) for fill in self.fills],
            "risk_manager": self._serialize_risk_manager(self.risk_manager) if self.risk_manager else None,
            "processed_bars": sorted(self.processed_bars),
            "last_prices": {symbol: self._serialize_number(price) for symbol, price in self.last_prices.items()},
            "last_price_timestamps": {
                symbol: pd.Timestamp(timestamp).isoformat()
                for symbol, timestamp in self.last_price_timestamps.items()
            },
            "external_fill_ids": sorted(self.external_fill_ids),
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load_state(cls, path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        broker = cls(
            initial_cash=cls._deserialize_number(payload["initial_cash"]),
            commission_rate=payload["commission_rate"],
            slippage_rate=payload["slippage_rate"],
            max_participation_rate=payload["max_participation_rate"],
        )
        broker.initial_cash = cls._deserialize_number(payload["initial_cash"])
        broker.cash = cls._deserialize_number(payload["cash"])
        broker.positions = {
            symbol: cls._deserialize_position(position_payload)
            for symbol, position_payload in payload.get("positions", {}).items()
        }
        broker.orders = {
            order_id: cls._deserialize_order(order_payload)
            for order_id, order_payload in payload.get("orders", {}).items()
        }
        broker.fills = [cls._deserialize_fill(fill_payload) for fill_payload in payload.get("fills", [])]
        if payload.get("risk_manager"):
            broker.risk_manager = cls._deserialize_risk_manager(payload["risk_manager"])
        broker.processed_bars = set(payload.get("processed_bars", []))
        broker.last_prices = {
            symbol: cls._deserialize_number(price)
            for symbol, price in payload.get("last_prices", {}).items()
        }
        broker.last_price_timestamps = {
            symbol: pd.Timestamp(timestamp)
            for symbol, timestamp in payload.get("last_price_timestamps", {}).items()
        }
        broker.external_fill_ids = set(payload.get("external_fill_ids", []))
        return broker

    def _uses_decimal_order_path(self, order):
        position = self.positions.get(order.symbol)
        values = [self.cash, order.quantity, order.filled_quantity, order.avg_fill_price]
        if position is not None:
            values.extend([position.quantity, position.avg_price])
        return any(isinstance(value, Decimal) for value in values)

    @staticmethod
    def _serialize_number(value):
        return str(value) if isinstance(value, Decimal) else value

    @staticmethod
    def _deserialize_number(value):
        return Decimal(value) if isinstance(value, str) else value

    @staticmethod
    def _serialize_order(order):
        payload = asdict(order)
        payload["submitted_at"] = pd.Timestamp(order.submitted_at).isoformat()
        for key in ("quantity", "filled_quantity", "avg_fill_price", "limit_price", "stop_price"):
            if isinstance(payload.get(key), Decimal):
                payload[key] = str(payload[key])
        return payload

    @staticmethod
    def _serialize_position(position):
        payload = asdict(position)
        for key in ("quantity", "avg_price"):
            if isinstance(payload.get(key), Decimal):
                payload[key] = str(payload[key])
        return payload

    @staticmethod
    def _deserialize_position(payload):
        payload = dict(payload)
        for key in ("quantity", "avg_price"):
            if isinstance(payload.get(key), str):
                payload[key] = Decimal(payload[key])
        return Position(**payload)

    @staticmethod
    def _deserialize_order(payload):
        payload = dict(payload)
        payload["submitted_at"] = pd.Timestamp(payload["submitted_at"])
        for key in ("quantity", "filled_quantity", "avg_fill_price", "limit_price", "stop_price"):
            if isinstance(payload.get(key), str):
                payload[key] = Decimal(payload[key])
        return Order(**payload)

    @staticmethod
    def _serialize_fill(fill):
        payload = asdict(fill)
        payload["timestamp"] = pd.Timestamp(fill.timestamp).isoformat()
        for key in ("quantity", "price", "commission"):
            if isinstance(payload.get(key), Decimal):
                payload[key] = str(payload[key])
        return payload

    @staticmethod
    def _deserialize_fill(payload):
        payload = dict(payload)
        payload["timestamp"] = pd.Timestamp(payload["timestamp"])
        for key in ("quantity", "price", "commission"):
            if isinstance(payload.get(key), str):
                payload[key] = Decimal(payload[key])
        return Fill(**payload)

    @staticmethod
    def _serialize_risk_manager(risk_manager):
        return {
            "initial_equity": risk_manager.initial_equity,
            "peak_equity": risk_manager.peak_equity,
            "max_drawdown": risk_manager.max_drawdown,
            "min_cash_fraction": risk_manager.min_cash_fraction,
            "halted": risk_manager.halted,
            "halt_reason": risk_manager.halt_reason,
        }

    @staticmethod
    def _deserialize_risk_manager(payload):
        risk_manager = PaperRiskManager(
            initial_equity=payload["initial_equity"],
            max_drawdown=payload["max_drawdown"],
            min_cash_fraction=payload["min_cash_fraction"],
        )
        risk_manager.peak_equity = payload["peak_equity"]
        risk_manager.halted = payload["halted"]
        risk_manager.halt_reason = payload["halt_reason"]
        return risk_manager

    def open_orders(self) -> Iterable[Order]:
        return [order for order in self.orders.values() if order.status in {OPEN, PARTIALLY_FILLED}]
