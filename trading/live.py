"""Live-trading adapter contracts and reconciliation helpers.

This module does not place live orders by itself. It defines the boundary a
real broker or exchange adapter must satisfy before the paper engine can be
promoted into an execution process.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Protocol

import pandas as pd

from .paper import FILLED, OPEN, PARTIALLY_FILLED, Position


LIVE_OPEN_STATUSES = {OPEN, PARTIALLY_FILLED, "ACCEPTED", "PENDING", "NEW"}
LIVE_TERMINAL_STATUSES = {FILLED, "CANCELLED", "REJECTED", "EXPIRED"}
LIVE_STATUSES = LIVE_OPEN_STATUSES | LIVE_TERMINAL_STATUSES
ORDER_TYPES = {"MARKET", "LIMIT", "STOP", "STOP_LIMIT"}
TIME_IN_FORCE = {"DAY", "GTC", "IOC", "FOK"}


@dataclass(frozen=True)
class LiveOrderRequest:
    symbol: str
    side: str
    quantity: Decimal
    client_order_id: str
    order_type: str = "MARKET"
    time_in_force: str = "DAY"
    limit_price: Decimal | None = None
    stop_price: Decimal | None = None
    reduce_only: bool = False

    def __post_init__(self):
        quantity = self._decimal(self.quantity, "quantity")
        object.__setattr__(self, "quantity", quantity)
        object.__setattr__(self, "side", self.side.upper())
        object.__setattr__(self, "order_type", self.order_type.upper())
        object.__setattr__(self, "time_in_force", self.time_in_force.upper())
        if self.limit_price is not None:
            object.__setattr__(self, "limit_price", self._decimal(self.limit_price, "limit_price"))
        if self.stop_price is not None:
            object.__setattr__(self, "stop_price", self._decimal(self.stop_price, "stop_price"))
        if not isinstance(self.reduce_only, bool):
            raise ValueError("reduce_only must be a boolean.")
        if not self.symbol.strip():
            raise ValueError("symbol must be populated.")
        if not self.client_order_id.strip():
            raise ValueError("client_order_id must be populated.")
        if self.side not in {"BUY", "SELL"}:
            raise ValueError("side must be BUY or SELL.")
        if quantity <= 0:
            raise ValueError("quantity must be positive.")
        if self.order_type not in ORDER_TYPES:
            raise ValueError(f"order_type must be one of {sorted(ORDER_TYPES)}.")
        if self.time_in_force not in TIME_IN_FORCE:
            raise ValueError(f"time_in_force must be one of {sorted(TIME_IN_FORCE)}.")
        if self.order_type in {"LIMIT", "STOP_LIMIT"} and self.limit_price is None:
            raise ValueError("limit_price is required for limit orders.")
        if self.order_type in {"STOP", "STOP_LIMIT"} and self.stop_price is None:
            raise ValueError("stop_price is required for stop orders.")
        if self.limit_price is not None and self.limit_price <= 0:
            raise ValueError("limit_price must be positive.")
        if self.stop_price is not None and self.stop_price <= 0:
            raise ValueError("stop_price must be positive.")

    @staticmethod
    def _decimal(value, field):
        try:
            converted = Decimal(str(value))
        except (InvalidOperation, ValueError) as exc:
            raise ValueError(f"{field} must be a valid decimal number.") from exc
        if not converted.is_finite():
            raise ValueError(f"{field} must be finite.")
        return converted


@dataclass(frozen=True)
class ExternalOrder:
    external_id: str
    client_order_id: str
    symbol: str
    side: str
    quantity: Decimal
    filled_quantity: Decimal
    status: str
    submitted_at: pd.Timestamp
    avg_fill_price: float = 0.0
    order_type: str = "MARKET"
    time_in_force: str = "DAY"
    limit_price: Decimal | None = None
    stop_price: Decimal | None = None
    reduce_only: bool = False

    def __post_init__(self):
        if not self.external_id.strip():
            raise ValueError("external_id must be populated.")
        if not self.client_order_id.strip():
            raise ValueError("client_order_id must be populated.")
        if not self.symbol.strip():
            raise ValueError("symbol must be populated.")
        object.__setattr__(self, "side", self.side.upper())
        object.__setattr__(self, "status", self.status.upper())
        object.__setattr__(self, "order_type", self.order_type.upper())
        object.__setattr__(self, "time_in_force", self.time_in_force.upper())
        if not isinstance(self.reduce_only, bool):
            raise ValueError("reduce_only must be a boolean.")
        if self.side not in {"BUY", "SELL"}:
            raise ValueError("side must be BUY or SELL.")
        if self.order_type not in ORDER_TYPES:
            raise ValueError(f"order_type must be one of {sorted(ORDER_TYPES)}.")
        if self.time_in_force not in TIME_IN_FORCE:
            raise ValueError(f"time_in_force must be one of {sorted(TIME_IN_FORCE)}.")
        if self.status not in LIVE_STATUSES:
            raise ValueError(f"status must be one of {sorted(LIVE_STATUSES)}.")
        object.__setattr__(self, "quantity", LiveOrderRequest._decimal(self.quantity, "quantity"))
        object.__setattr__(self, "filled_quantity", LiveOrderRequest._decimal(self.filled_quantity, "filled_quantity"))
        if self.quantity <= 0:
            raise ValueError("quantity must be positive.")
        if self.filled_quantity < 0:
            raise ValueError("filled_quantity cannot be negative.")
        if self.filled_quantity > self.quantity:
            raise ValueError("filled_quantity cannot exceed quantity.")
        if self.status == FILLED and self.filled_quantity != self.quantity:
            raise ValueError("FILLED status requires filled_quantity to equal quantity.")
        if self.status in {"CANCELLED", "REJECTED", "EXPIRED"} and self.filled_quantity == self.quantity:
            raise ValueError("Closed unfilled statuses cannot be fully filled.")
        if self.status in {OPEN, "ACCEPTED", "PENDING", "NEW"} and self.filled_quantity != 0:
            raise ValueError("Unfilled open statuses require filled_quantity to be zero.")
        if self.status == PARTIALLY_FILLED and not 0 < self.filled_quantity < self.quantity:
            raise ValueError("PARTIALLY_FILLED status requires 0 < filled_quantity < quantity.")
        if self.order_type in {"LIMIT", "STOP_LIMIT"} and self.limit_price is None:
            raise ValueError("limit_price is required for limit orders.")
        if self.order_type in {"STOP", "STOP_LIMIT"} and self.stop_price is None:
            raise ValueError("stop_price is required for stop orders.")
        if self.limit_price is not None:
            object.__setattr__(self, "limit_price", LiveOrderRequest._decimal(self.limit_price, "limit_price"))
        if self.stop_price is not None:
            object.__setattr__(self, "stop_price", LiveOrderRequest._decimal(self.stop_price, "stop_price"))
        if self.limit_price is not None and self.limit_price <= 0:
            raise ValueError("limit_price must be positive.")
        if self.stop_price is not None and self.stop_price <= 0:
            raise ValueError("stop_price must be positive.")
        object.__setattr__(self, "submitted_at", pd.Timestamp(self.submitted_at))


@dataclass(frozen=True)
class ExternalFill:
    external_fill_id: str
    external_order_id: str
    client_order_id: str
    symbol: str
    side: str
    quantity: Decimal
    price: Decimal
    commission: Decimal
    timestamp: pd.Timestamp

    def __post_init__(self):
        if not self.external_fill_id.strip():
            raise ValueError("external_fill_id must be populated.")
        if not self.external_order_id.strip():
            raise ValueError("external_order_id must be populated.")
        if not self.client_order_id.strip():
            raise ValueError("client_order_id must be populated.")
        if not self.symbol.strip():
            raise ValueError("symbol must be populated.")
        object.__setattr__(self, "side", self.side.upper())
        if self.side not in {"BUY", "SELL"}:
            raise ValueError("side must be BUY or SELL.")
        quantity = LiveOrderRequest._decimal(self.quantity, "quantity")
        price = LiveOrderRequest._decimal(self.price, "price")
        commission = LiveOrderRequest._decimal(self.commission, "commission")
        if quantity <= 0:
            raise ValueError("quantity must be positive.")
        if price <= 0:
            raise ValueError("price must be positive.")
        if commission < 0:
            raise ValueError("commission cannot be negative.")
        object.__setattr__(self, "quantity", quantity)
        object.__setattr__(self, "price", price)
        object.__setattr__(self, "commission", commission)
        object.__setattr__(self, "timestamp", pd.Timestamp(self.timestamp))


@dataclass(frozen=True)
class AccountSnapshot:
    timestamp: pd.Timestamp
    cash: float
    positions: dict[str, Decimal]
    orders: list[ExternalOrder]


@dataclass(frozen=True)
class ReconciliationReport:
    cash_mismatch: dict | None
    position_mismatches: list[dict]
    order_mismatches: list[dict]
    recovery_actions: list[dict] | None = None

    @property
    def ok(self):
        return not self.cash_mismatch and not self.position_mismatches and not self.order_mismatches


class LiveReconciliationError(RuntimeError):
    """Raised when broker state does not match local intended state."""

    def __init__(self, stage, report: ReconciliationReport):
        self.stage = stage
        self.report = report
        summary = {
            "cash": report.cash_mismatch is not None,
            "positions": [item["type"] for item in report.position_mismatches],
            "orders": [item["type"] for item in report.order_mismatches],
        }
        super().__init__(f"Live reconciliation failed at {stage}: {summary}.")


class BrokerAdapter(Protocol):
    """Required surface for a real execution adapter."""

    def submit_order(self, request: LiveOrderRequest):
        ...

    def cancel_order(self, external_id: str):
        ...

    def account_snapshot(self) -> AccountSnapshot:
        ...

    def stream_fills(self, since=None) -> list[ExternalFill]:
        ...


class MarketDataAdapter(Protocol):
    """Required surface for a real market-data adapter."""

    def latest_bars(self, symbols: list[str]):
        ...


class BrokerReconciler:
    """Compare local intended state with a broker/account snapshot."""

    def __init__(self, cash_tolerance=1e-6):
        self.cash_tolerance = self._decimal(cash_tolerance, "cash_tolerance")

    def reconcile(self, broker, snapshot: AccountSnapshot):
        cash_mismatch = None
        expected_cash = self._decimal(broker.cash, "local cash")
        actual_cash = self._decimal(snapshot.cash, "snapshot cash")
        difference = actual_cash - expected_cash
        if abs(difference) > self.cash_tolerance:
            cash_mismatch = {
                "expected": expected_cash,
                "actual": actual_cash,
                "difference": difference,
            }

        position_mismatches = self._position_mismatches(broker.positions, snapshot.positions)
        order_mismatches = self._order_mismatches(broker.orders, snapshot.orders)
        return ReconciliationReport(
            cash_mismatch=cash_mismatch,
            position_mismatches=position_mismatches,
            order_mismatches=order_mismatches,
        )

    @staticmethod
    def _position_mismatches(local_positions: dict[str, Position], external_positions: dict[str, int]):
        mismatches = []
        symbols = set(local_positions) | set(external_positions)
        for symbol in sorted(symbols):
            expected = local_positions.get(symbol, Position(symbol)).quantity
            actual = BrokerReconciler._decimal(external_positions.get(symbol, 0), "external position")
            if expected != actual:
                mismatches.append(
                    {
                        "type": "position",
                        "symbol": symbol,
                        "expected": expected,
                        "actual": actual,
                    }
                )
        return mismatches

    @staticmethod
    def _order_mismatches(local_orders, external_orders: list[ExternalOrder]):
        mismatches = []
        external_by_client_id: dict[str, list[ExternalOrder]] = {}
        for external in external_orders:
            external_by_client_id.setdefault(external.client_order_id, []).append(external)
        for client_order_id, orders in sorted(external_by_client_id.items()):
            if len(orders) > 1:
                mismatches.append(
                    {
                        "type": "duplicate_external_client_order_id",
                        "client_order_id": client_order_id,
                        "external_ids": sorted(order.external_id for order in orders),
                    }
                )
        for order_id, local_order in sorted(local_orders.items()):
            local_is_open = local_order.status in {OPEN, PARTIALLY_FILLED}
            external_orders_for_id = external_by_client_id.get(order_id, [])
            external = external_orders_for_id[0] if external_orders_for_id else None
            if local_is_open and external is None:
                mismatches.append(
                    {
                        "type": "missing_external_order",
                        "client_order_id": order_id,
                        "symbol": local_order.symbol,
                    }
                )
                continue
            if external is None:
                continue
            external_status = external.status.upper()
            if not local_is_open:
                if external_status in LIVE_OPEN_STATUSES:
                    mismatches.append(
                        {
                            "type": "external_order_open_after_local_close",
                            "client_order_id": order_id,
                            "external_id": external.external_id,
                            "local_status": local_order.status,
                            "external_status": external.status,
                        }
                    )
                continue
            if external_status not in LIVE_OPEN_STATUSES:
                mismatches.append(
                    {
                        "type": "order_status",
                        "client_order_id": order_id,
                        "external_id": external.external_id,
                        "expected": "open",
                        "actual": external.status,
                    }
                )
            if local_order.symbol != external.symbol:
                mismatches.append(
                    {
                        "type": "order_symbol",
                        "client_order_id": order_id,
                        "external_id": external.external_id,
                        "expected": local_order.symbol,
                        "actual": external.symbol,
                    }
                )
            if local_order.side != external.side.upper():
                mismatches.append(
                    {
                        "type": "order_side",
                        "client_order_id": order_id,
                        "external_id": external.external_id,
                        "expected": local_order.side,
                        "actual": external.side,
                    }
                )
            if BrokerReconciler._decimal(local_order.quantity, "local order quantity") != BrokerReconciler._decimal(
                external.quantity,
                "external order quantity",
            ):
                mismatches.append(
                    {
                        "type": "order_quantity",
                        "client_order_id": order_id,
                        "external_id": external.external_id,
                        "expected": BrokerReconciler._decimal(local_order.quantity, "local order quantity"),
                        "actual": BrokerReconciler._decimal(external.quantity, "external order quantity"),
                    }
                )
            if BrokerReconciler._decimal(local_order.filled_quantity, "local filled quantity") != BrokerReconciler._decimal(
                external.filled_quantity,
                "external filled quantity",
            ):
                mismatches.append(
                    {
                        "type": "order_filled_quantity",
                        "client_order_id": order_id,
                        "external_id": external.external_id,
                        "expected": BrokerReconciler._decimal(local_order.filled_quantity, "local filled quantity"),
                        "actual": BrokerReconciler._decimal(external.filled_quantity, "external filled quantity"),
                    }
                )
            contract_fields = (
                ("order_type", local_order.order_type, external.order_type),
                ("time_in_force", local_order.time_in_force, external.time_in_force),
                ("limit_price", local_order.limit_price or 0, external.limit_price or 0),
                ("stop_price", local_order.stop_price or 0, external.stop_price or 0),
                ("reduce_only", bool(local_order.reduce_only), bool(external.reduce_only)),
            )
            for field, expected, actual in contract_fields:
                if field in {"limit_price", "stop_price"}:
                    expected_value = BrokerReconciler._decimal(expected, f"local {field}")
                    actual_value = BrokerReconciler._decimal(actual, f"external {field}")
                else:
                    expected_value = expected
                    actual_value = actual
                if expected_value != actual_value:
                    mismatches.append(
                        {
                            "type": f"order_{field}",
                            "client_order_id": order_id,
                            "external_id": external.external_id,
                            "expected": expected_value,
                            "actual": actual_value,
                        }
                    )
        for external in external_orders:
            if external.client_order_id in local_orders:
                continue
            if external.status.upper() not in LIVE_OPEN_STATUSES:
                continue
            mismatches.append(
                {
                    "type": "unexpected_external_order",
                    "client_order_id": external.client_order_id,
                    "external_id": external.external_id,
                    "symbol": external.symbol,
                }
            )
        return mismatches

    @staticmethod
    def _decimal(value, field):
        try:
            converted = Decimal(str(value))
        except (InvalidOperation, ValueError) as exc:
            raise ValueError(f"{field} must be a valid decimal number.") from exc
        if not converted.is_finite():
            raise ValueError(f"{field} must be finite.")
        return converted


class LiveExecutionLoop:
    """Small live-execution guard that wraps adapters with reconciliation gates."""

    def __init__(self, broker_adapter: BrokerAdapter, reconciler: BrokerReconciler | None = None, journal=None):
        if journal is None:
            raise ValueError("LiveExecutionLoop requires a durable journal.")
        self.broker_adapter = broker_adapter
        self.reconciler = reconciler or BrokerReconciler()
        self.journal = journal
        self.processed_external_fills: set[str] = self._load_processed_fill_ids()
        self.pending_unmanaged_fills: set[str] = set()

    def preflight(self, local_broker):
        return self._reconcile_or_raise(local_broker, "preflight")

    def submit_order(self, local_broker, request: LiveOrderRequest, on_ack=None):
        self._reconcile_or_raise(local_broker, "pre_submit")
        existing = self._existing_external_order(request.client_order_id)
        if existing is not None and request.client_order_id in local_broker.orders:
            conflict = self._idempotency_conflict(local_broker.orders[request.client_order_id], existing, request)
            if conflict:
                raise ValueError(f"idempotency_conflict: {conflict}")
            self._journal("live_order_submit_deduplicated", {"request": request, "acknowledgement": existing})
            return existing
        self._journal("live_order_intent", {"request": request})
        try:
            acknowledgement = self.broker_adapter.submit_order(request)
        except Exception as exc:
            self._journal("live_order_submit_error", {"request": request, "error": str(exc)})
            raise
        ack_journal_error = None
        try:
            self._journal("live_order_ack", {"request": request, "acknowledgement": acknowledgement})
        except Exception as exc:
            ack_journal_error = exc
        snapshot = copy.deepcopy(local_broker)
        try:
            if on_ack is not None:
                on_ack(acknowledgement)
            self._reconcile_or_raise(local_broker, "post_submit")
        except Exception as exc:
            local_broker.__dict__.clear()
            local_broker.__dict__.update(snapshot.__dict__)
            self._journal(
                "live_order_transition_error",
                {"request": request, "acknowledgement": acknowledgement, "error": str(exc)},
            )
            raise
        if ack_journal_error is not None:
            raise ack_journal_error
        return acknowledgement

    def cancel_order(self, local_broker, external_id: str, on_ack=None):
        if self._existing_external_order_by_external_id(external_id) is None:
            result = {"external_id": external_id, "status": "already_absent"}
            self._journal("live_cancel_deduplicated", result)
            snapshot = copy.deepcopy(local_broker)
            try:
                if on_ack is not None:
                    on_ack(result)
                self._reconcile_or_raise(local_broker, "post_cancel_deduplicated")
            except Exception as exc:
                local_broker.__dict__.clear()
                local_broker.__dict__.update(snapshot.__dict__)
                self._journal(
                    "live_cancel_transition_error",
                    {"external_id": external_id, "acknowledgement": result, "error": str(exc)},
                )
                raise
            return result
        self._reconcile_or_raise(local_broker, "pre_cancel")
        self._journal("live_cancel_intent", {"external_id": external_id})
        try:
            result = self.broker_adapter.cancel_order(external_id)
        except Exception as exc:
            self._journal("live_cancel_error", {"external_id": external_id, "error": str(exc)})
            raise
        ack_journal_error = None
        try:
            self._journal("live_cancel_ack", {"external_id": external_id, "acknowledgement": result})
        except Exception as exc:
            ack_journal_error = exc
        snapshot = copy.deepcopy(local_broker)
        try:
            if on_ack is not None:
                on_ack(result)
            self._reconcile_or_raise(local_broker, "post_cancel")
        except Exception as exc:
            local_broker.__dict__.clear()
            local_broker.__dict__.update(snapshot.__dict__)
            self._journal(
                "live_cancel_transition_error",
                {"external_id": external_id, "acknowledgement": result, "error": str(exc)},
            )
            raise
        if ack_journal_error is not None:
            raise ack_journal_error
        return result

    def recover(self, local_broker):
        self.replay_applied_fills(local_broker)
        snapshot = self.broker_adapter.account_snapshot()
        report = self.reconciler.reconcile(local_broker, snapshot)
        recovery_actions = self._recovery_actions(local_broker, snapshot, report)
        report = ReconciliationReport(
            cash_mismatch=report.cash_mismatch,
            position_mismatches=report.position_mismatches,
            order_mismatches=report.order_mismatches,
            recovery_actions=recovery_actions,
        )
        self._journal(
            "live_recovery_reconcile",
            {
                "ok": report.ok,
                "cash_mismatch": report.cash_mismatch,
                "position_mismatches": report.position_mismatches,
                "order_mismatches": report.order_mismatches,
                "recovery_actions": report.recovery_actions,
            },
        )
        return report

    def process_fills(self, local_broker, since=None, on_fill=None):
        if on_fill is not None:
            raise ValueError("Custom on_fill hooks are not allowed in durable live fill processing.")
        self.replay_applied_fills(local_broker)
        applied = []
        for fill in self.broker_adapter.stream_fills(since=since):
            if fill.external_fill_id in self.processed_external_fills:
                continue
            if fill.external_fill_id in getattr(local_broker, "external_fill_ids", set()):
                self.processed_external_fills.add(fill.external_fill_id)
                continue
            self._journal("live_fill_received", {"fill": fill})
            order = local_broker.orders.get(fill.client_order_id)
            if order is None:
                if fill.external_fill_id not in self.pending_unmanaged_fills:
                    self._journal(
                        "live_unmanaged_fill",
                        {
                            "fill": fill,
                            "action_required": "attach order, reconcile position, or flatten externally",
                        },
                    )
                    self.pending_unmanaged_fills.add(fill.external_fill_id)
                continue
            rejection = self._fill_rejection_reason(local_broker, order, fill)
            if rejection:
                self._journal(
                    "live_fill_rejected",
                    {"fill": fill, "reason": rejection, "action_required": "manual reconciliation"},
                )
                self.processed_external_fills.add(fill.external_fill_id)
                continue
            self._journal("live_fill_intent", {"fill": fill})
            snapshot = copy.deepcopy(local_broker)
            try:
                self._apply_external_fill(local_broker, order, fill)
                self._journal("live_fill_applied", {"fill": fill})
            except Exception:
                local_broker.__dict__.clear()
                local_broker.__dict__.update(snapshot.__dict__)
                raise
            self.processed_external_fills.add(fill.external_fill_id)
            applied.append(fill)
        return applied

    def replay_applied_fills(self, local_broker):
        replayed = []
        for record in self.journal.read_all():
            if record.get("event_type") != "live_fill_applied":
                continue
            fill_payload = record.get("payload", {}).get("fill", {})
            fill = ExternalFill(**fill_payload)
            if fill.external_fill_id in getattr(local_broker, "external_fill_ids", set()):
                continue
            order = local_broker.orders.get(fill.client_order_id)
            if order is None:
                self._journal(
                    "live_fill_replay_pending",
                    {"fill": fill, "reason": "missing local order for replay"},
                )
                continue
            rejection = self._fill_rejection_reason(local_broker, order, fill)
            if rejection:
                self._journal("live_fill_replay_rejected", {"fill": fill, "reason": rejection})
                continue
            self._apply_external_fill(local_broker, order, fill)
            replayed.append(fill)
        return replayed

    def _reconcile_or_raise(self, local_broker, stage):
        self.replay_applied_fills(local_broker)
        snapshot = self.broker_adapter.account_snapshot()
        report = self.reconciler.reconcile(local_broker, snapshot)
        if not report.ok:
            raise LiveReconciliationError(stage, report)
        return report

    def _journal(self, event_type, payload):
        return self.journal.append(event_type, payload)

    def _recovery_actions(self, local_broker, snapshot: AccountSnapshot, report: ReconciliationReport):
        local_order_ids = set(local_broker.orders)
        actions = []
        for mismatch in report.order_mismatches:
            if mismatch["type"] == "external_order_open_after_local_close":
                actions.append(
                    {
                        "action": "cancel_external_order",
                        "client_order_id": mismatch["client_order_id"],
                        "external_id": mismatch["external_id"],
                    }
                )
            elif mismatch["type"] == "missing_external_order":
                actions.append(
                    {
                        "action": "mark_local_order_closed_or_resubmit_missing_external_order",
                        "client_order_id": mismatch["client_order_id"],
                        "symbol": mismatch["symbol"],
                    }
                )
            elif mismatch["type"] == "duplicate_external_client_order_id":
                actions.append(
                    {
                        "action": "halt_and_investigate_duplicate_external_orders",
                        "client_order_id": mismatch["client_order_id"],
                        "external_ids": mismatch["external_ids"],
                    }
                )
            elif mismatch["type"].startswith("order_"):
                actions.append(
                    {
                        "action": "halt_and_reconcile_order_contract",
                        "type": mismatch["type"],
                        "client_order_id": mismatch["client_order_id"],
                        "external_id": mismatch["external_id"],
                    }
                )
        for mismatch in report.position_mismatches:
            actions.append(
                {
                    "action": "reconcile_position_to_broker_snapshot",
                    "symbol": mismatch["symbol"],
                    "expected": mismatch["expected"],
                    "actual": mismatch["actual"],
                }
            )
        if report.cash_mismatch is not None:
            actions.append(
                {
                    "action": "reconcile_cash_to_broker_snapshot",
                    "expected": report.cash_mismatch["expected"],
                    "actual": report.cash_mismatch["actual"],
                }
            )
        for external in snapshot.orders:
            if external.status.upper() not in LIVE_OPEN_STATUSES:
                continue
            if external.client_order_id in local_order_ids:
                continue
            actions.append(
                {
                    "action": "attach_or_cancel_unmanaged_order",
                    "client_order_id": external.client_order_id,
                    "external_id": external.external_id,
                    "symbol": external.symbol,
                }
            )
        return actions

    def _fill_rejection_reason(self, local_broker, order, fill: ExternalFill):
        if order.symbol != fill.symbol:
            return "fill symbol does not match local order"
        if order.side != fill.side:
            return "fill side does not match local order"
        remaining = BrokerReconciler._decimal(order.remaining_quantity, "remaining quantity")
        quantity = BrokerReconciler._decimal(fill.quantity, "fill quantity")
        if quantity > remaining:
            return "fill quantity exceeds local remaining quantity"
        if fill.side == "SELL":
            position = local_broker.positions.get(order.symbol, Position(order.symbol))
            inventory = BrokerReconciler._decimal(position.quantity, "position quantity")
            if quantity > inventory:
                return "sell fill exceeds local inventory"
        return ""

    def _apply_external_fill(self, local_broker, order, fill: ExternalFill):
        local_broker._apply_fill(
            order,
            BrokerReconciler._decimal(fill.quantity, "fill quantity"),
            BrokerReconciler._decimal(fill.price, "fill price"),
            BrokerReconciler._decimal(fill.commission, "fill commission"),
            pd.Timestamp(fill.timestamp),
        )
        order.status = FILLED if order.remaining_quantity == 0 else PARTIALLY_FILLED
        local_broker.external_fill_ids.add(fill.external_fill_id)

    def _existing_external_order(self, client_order_id):
        snapshot = self.broker_adapter.account_snapshot()
        for order in snapshot.orders:
            if order.client_order_id == client_order_id and order.status.upper() in LIVE_OPEN_STATUSES:
                return order
        return None

    def _existing_external_order_by_external_id(self, external_id):
        snapshot = self.broker_adapter.account_snapshot()
        for order in snapshot.orders:
            if order.external_id == external_id and order.status.upper() in LIVE_OPEN_STATUSES:
                return order
        return None

    def _idempotency_conflict(self, local_order, external_order: ExternalOrder, request: LiveOrderRequest):
        if local_order.symbol != request.symbol or external_order.symbol != request.symbol:
            return "symbol mismatch"
        if local_order.side != request.side or external_order.side.upper() != request.side:
            return "side mismatch"
        local_qty = BrokerReconciler._decimal(local_order.quantity, "local order quantity")
        external_qty = BrokerReconciler._decimal(external_order.quantity, "external order quantity")
        request_qty = BrokerReconciler._decimal(request.quantity, "request quantity")
        if local_qty != request_qty or external_qty != request_qty:
            return "quantity mismatch"
        if external_order.order_type.upper() != request.order_type:
            return "order_type mismatch"
        if external_order.time_in_force.upper() != request.time_in_force:
            return "time_in_force mismatch"
        if BrokerReconciler._decimal(external_order.limit_price or 0, "external limit price") != BrokerReconciler._decimal(
            request.limit_price or 0,
            "request limit price",
        ):
            return "limit_price mismatch"
        if BrokerReconciler._decimal(external_order.stop_price or 0, "external stop price") != BrokerReconciler._decimal(
            request.stop_price or 0,
            "request stop price",
        ):
            return "stop_price mismatch"
        if bool(external_order.reduce_only) != bool(request.reduce_only):
            return "reduce_only mismatch"
        return ""

    def _load_processed_fill_ids(self):
        processed = set()
        if self.journal is None:
            return processed
        if not hasattr(self.journal, "read_all"):
            return processed
        try:
            records = self.journal.read_all()
        except FileNotFoundError:
            return processed
        for record in records:
            if record.get("event_type") not in {"live_fill_applied", "live_fill_rejected"}:
                continue
            fill = record.get("payload", {}).get("fill", {})
            fill_id = fill.get("external_fill_id") if isinstance(fill, dict) else None
            if fill_id:
                processed.add(fill_id)
        return processed
