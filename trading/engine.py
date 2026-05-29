"""Paper trading engine wiring predictions, bars, broker state, and ops."""

from __future__ import annotations

import copy

import pandas as pd

from .ops import FileKillSwitch, HeartbeatMonitor, JsonlOrderJournal
from .paper import SELL, MarketBar
from .runner import PaperTradingRunner, PredictionEvent


class PaperTradingEngine:
    """Operational wrapper around PaperTradingRunner.

    It provides the missing process-level pieces for paper trading: journaled
    events, heartbeat, kill-switch handling, and liquidation attempts when the
    process is halted.
    """

    def __init__(
        self,
        runner: PaperTradingRunner,
        journal: JsonlOrderJournal,
        kill_switch: FileKillSwitch | None = None,
        heartbeat: HeartbeatMonitor | None = None,
        max_liquidation_attempts: int = 3,
        max_mark_age="1D",
        production_mode: bool = False,
    ):
        if max_liquidation_attempts < 1:
            raise ValueError("max_liquidation_attempts must be at least 1.")
        if production_mode and runner.approval_registry is None:
            raise ValueError("production_mode requires a runner with approval_registry.")
        if production_mode and not runner.approval_registry.accepted_model_hashes():
            raise ValueError("production_mode requires a verified non-empty approval_registry.")
        self.runner = runner
        self.journal = journal
        self.kill_switch = kill_switch
        self.heartbeat = heartbeat
        self.max_liquidation_attempts = max_liquidation_attempts
        self.max_mark_age = pd.Timedelta(max_mark_age) if max_mark_age is not None else None
        self.production_mode = production_mode
        self.liquidation_attempts: dict[str, int] = {}
        self._journal_buffer: list[tuple[str, object]] | None = None

    def on_bar(self, bar: MarketBar, prediction: PredictionEvent | None = None):
        return self.on_bars([bar], {bar.symbol: prediction} if prediction is not None else {})

    def on_bars(self, bars, predictions=None):
        bars = list(bars)
        predictions = predictions or {}
        bars_by_symbol = {}
        for bar in bars:
            if bar.symbol in bars_by_symbol:
                raise ValueError(f"Duplicate bar for symbol {bar.symbol!r} in one engine batch.")
            bars_by_symbol[bar.symbol] = bar
        extra_predictions = sorted(set(predictions) - set(bars_by_symbol))
        if extra_predictions:
            raise ValueError(f"Predictions supplied without matching bars: {extra_predictions}")
        try:
            self._validate_batch(bars_by_symbol, predictions)
        except Exception as exc:
            self._journal(
                "processing_error",
                {"symbols": sorted(bars_by_symbol), "error": str(exc)},
            )
            self._heartbeat("error", bars[0] if bars else None, [])
            raise

        if self.kill_switch is not None and self.kill_switch.is_active():
            reason = self.kill_switch.reason() or "kill switch active"
            self._journal("kill_switch_active", {"reason": reason, "symbols": sorted(bars_by_symbol)})
            try:
                fills = self._liquidate_account(bars_by_symbol, reason)
            except Exception as exc:
                self._journal(
                    "processing_error",
                    {"symbols": sorted(bars_by_symbol), "error": str(exc)},
                )
                self._heartbeat("error", bars[0] if bars else None, [])
                raise
            self._heartbeat("halted", bars[0] if bars else None, fills)
            return fills

        snapshot = self._snapshot_runner()
        all_fills = []
        self._begin_journal_transaction()
        try:
            for bar in bars:
                all_fills.extend(self._process_active_bar(bar, predictions.get(bar.symbol)))
        except Exception as exc:
            self._discard_journal_transaction()
            self._restore_runner(snapshot)
            self._journal(
                "batch_rollback",
                {"symbols": sorted(bars_by_symbol), "error": str(exc)},
            )
            self._heartbeat("error", bars[0] if bars else None, [])
            raise
        self._commit_journal_transaction()
        self._heartbeat("ok", bars[-1] if bars else None, all_fills)
        return all_fills

    def _validate_batch(self, bars_by_symbol, predictions):
        for bar in bars_by_symbol.values():
            self.runner._validate_bar(bar)
            bar.timestamp = pd.Timestamp(bar.timestamp)

        for symbol, prediction in predictions.items():
            if prediction is None:
                continue
            prediction.validate()
            bar = bars_by_symbol[symbol]
            if prediction.symbol != bar.symbol:
                raise ValueError("Prediction symbol must match bar symbol.")
            if prediction.execution_timestamp != bar.timestamp:
                raise ValueError("Prediction execution_timestamp must match bar timestamp.")
            bar_key = f"{bar.symbol}|{bar.timestamp.isoformat()}"
            if bar_key in self.runner.broker.processed_bars:
                raise ValueError("Cannot process a prediction for a bar that was already processed.")

    def _process_active_bar(self, bar: MarketBar, prediction: PredictionEvent | None = None):
        self._journal(
            "bar_received",
            {
                "symbol": bar.symbol,
                "timestamp": pd.Timestamp(bar.timestamp),
                "has_prediction": prediction is not None,
            },
        )
        if prediction is not None:
            self._journal(
                "prediction_received",
                {
                    "symbol": prediction.symbol,
                    "execution_timestamp": prediction.execution_timestamp,
                    "model_version": prediction.model_version,
                    "model_hash": prediction.model_hash,
                },
            )

        try:
            fills = self.runner.process_bar(bar, prediction)
        except Exception as exc:
            self._journal(
                "processing_error",
                {"symbol": bar.symbol, "timestamp": pd.Timestamp(bar.timestamp), "error": str(exc)},
            )
            self._heartbeat("error", bar, [])
            raise

        for fill in fills:
            self._journal("fill", fill)
        self._journal("equity", self.runner.equity_history[-1] if self.runner.equity_history else {})
        self._heartbeat("ok", bar, fills)
        return fills

    def _liquidate_account(self, bars_by_symbol, reason):
        for bar in bars_by_symbol.values():
            self.runner._validate_bar(bar)
            bar.timestamp = pd.Timestamp(bar.timestamp)

        self.runner.broker.cancel_open_orders(reason=reason)
        fills = []
        for symbol in sorted(list(self.runner.broker.positions)):
            bar = bars_by_symbol.get(symbol)
            if bar is None:
                attempt = self._record_liquidation_attempt(symbol)
                self._journal(
                    "liquidation_pending",
                    {
                        "symbol": symbol,
                        "reason": "missing current bar",
                        "attempt": attempt,
                        "max_attempts": self.max_liquidation_attempts,
                    },
                )
                self._journal_liquidation_escalation(symbol, attempt, "missing current bar")
                continue
            fills.extend(
                self._liquidate_symbol(
                    bar,
                    reason,
                    cancel_orders=False,
                    validate_bar=False,
                    order_reference_prices={symbol: item.open for symbol, item in bars_by_symbol.items()},
                    equity_reference_prices={symbol: item.close for symbol, item in bars_by_symbol.items()},
                )
            )
        residuals = {
            symbol: position.quantity
            for symbol, position in sorted(self.runner.broker.positions.items())
            if position.quantity > 0
        }
        self._validate_account_marks(bars_by_symbol)
        self._journal(
            "liquidation_incomplete" if residuals else "liquidation_complete",
            {"residual_positions": residuals},
        )
        return fills

    def _liquidate_symbol(
        self,
        bar: MarketBar,
        reason,
        cancel_orders=True,
        validate_bar=True,
        order_reference_prices=None,
        equity_reference_prices=None,
    ):
        if validate_bar:
            self.runner._validate_bar(bar)
            bar.timestamp = pd.Timestamp(bar.timestamp)
        bar_key = f"{bar.symbol}|{bar.timestamp.isoformat()}"
        if bar_key in self.runner.broker.processed_bars:
            self._journal(
                "liquidation_skipped",
                {"symbol": bar.symbol, "timestamp": bar.timestamp, "reason": "bar already processed"},
            )
            return []

        position = self.runner.broker.positions.get(bar.symbol)
        if not position or position.quantity <= 0:
            return []

        if cancel_orders:
            self.runner.broker.cancel_open_orders(symbol=bar.symbol, reason=reason)
        order = self.runner.broker.submit_order(
            bar.symbol,
            SELL,
            position.quantity,
            pd.Timestamp(bar.timestamp),
            reference_prices=order_reference_prices or {bar.symbol: bar.open},
        )
        self._journal(
            "liquidation_order",
            {"order_id": order.id, "symbol": bar.symbol, "quantity": position.quantity, "reason": reason},
        )
        fills = self.runner.broker.process_bar(
            bar,
            reference_prices=order_reference_prices or {bar.symbol: bar.open},
        )
        for fill in fills:
            self._journal("fill", fill)
        residual = self.runner.broker.positions.get(bar.symbol)
        if residual and residual.quantity > 0:
            attempt = self._record_liquidation_attempt(bar.symbol)
            self._journal(
                "liquidation_residual",
                {
                    "order_id": order.id,
                    "symbol": bar.symbol,
                    "remaining_quantity": residual.quantity,
                    "reason": "partial liquidation fill",
                    "attempt": attempt,
                    "max_attempts": self.max_liquidation_attempts,
                },
            )
            self._journal_liquidation_escalation(bar.symbol, attempt, "partial liquidation fill")
        else:
            self.liquidation_attempts.pop(bar.symbol, None)
        equity = self.runner.broker.equity(
            equity_reference_prices or {bar.symbol: bar.close},
            asof=bar.timestamp,
            max_price_age=self.max_mark_age,
        )
        self.runner.broker.risk_manager.update(equity)
        self.runner.equity_history.append(
            {
                "timestamp": pd.Timestamp(bar.timestamp),
                "symbol": bar.symbol,
                "equity": equity,
                "cash": self.runner.broker.cash,
                "position": self.runner.broker.positions.get(bar.symbol).quantity
                if self.runner.broker.positions.get(bar.symbol)
                else 0,
                "close": bar.close,
                "halted": True,
            }
        )
        return fills

    def _snapshot_runner(self):
        return {
            "broker": copy.deepcopy(self.runner.broker),
            "last_close": copy.deepcopy(self.runner.last_close),
            "equity_history": copy.deepcopy(self.runner.equity_history),
            "signal_history": copy.deepcopy(self.runner.signal_history),
        }

    def _restore_runner(self, snapshot):
        self.runner.broker = snapshot["broker"]
        self.runner.last_close = snapshot["last_close"]
        self.runner.equity_history = snapshot["equity_history"]
        self.runner.signal_history = snapshot["signal_history"]

    def _record_liquidation_attempt(self, symbol):
        attempt = self.liquidation_attempts.get(symbol, 0) + 1
        self.liquidation_attempts[symbol] = attempt
        return attempt

    def _journal_liquidation_escalation(self, symbol, attempt, reason):
        if attempt < self.max_liquidation_attempts:
            return None
        return self._journal(
            "liquidation_escalation",
            {
                "symbol": symbol,
                "attempt": attempt,
                "max_attempts": self.max_liquidation_attempts,
                "reason": reason,
                "action_required": "manual broker intervention or additional liquidity source",
            },
        )

    def _validate_account_marks(self, bars_by_symbol):
        if not self.runner.broker.positions:
            return self.runner.broker.cash
        reference_prices = {symbol: bar.close for symbol, bar in bars_by_symbol.items()}
        if bars_by_symbol:
            asof = max(pd.Timestamp(bar.timestamp) for bar in bars_by_symbol.values())
        else:
            asof = pd.Timestamp.now("UTC")
        return self.runner.broker.equity(
            reference_prices,
            asof=asof,
            max_price_age=self.max_mark_age,
        )

    def _journal(self, event_type, payload):
        if self._journal_buffer is not None:
            self._journal_buffer.append((event_type, payload))
            return {"event_type": event_type, "payload": payload}
        return self.journal.append(event_type, payload)

    def _begin_journal_transaction(self):
        self._journal_buffer = []

    def _commit_journal_transaction(self):
        buffered = self._journal_buffer or []
        self._journal_buffer = None
        for event_type, payload in buffered:
            self.journal.append(event_type, payload)

    def _discard_journal_transaction(self):
        self._journal_buffer = None

    def _heartbeat(self, status, bar, fills):
        if self.heartbeat is None:
            return None
        if self._journal_buffer is not None:
            return None
        payload = {
            "symbol": bar.symbol if bar is not None else None,
            "timestamp": pd.Timestamp(bar.timestamp).isoformat() if bar is not None else None,
            "fills": len(fills),
            "report": self.runner.report(),
        }
        return self.heartbeat.beat(
            status=status,
            payload=payload,
        )
