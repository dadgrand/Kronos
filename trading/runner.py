"""Event-driven paper trading runner for Kronos prediction contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .paper import BUY, SELL, MarketBar, PaperBroker, PaperRiskManager


REQUIRED_PREDICTION_COLUMNS = {
    "symbol",
    "prediction_asof",
    "execution_timestamp",
    "target_timestamp",
    "features_cutoff",
    "horizon",
    "model_version",
    "model_hash",
    "predicted_close",
}


@dataclass
class PredictionEvent:
    symbol: str
    prediction_asof: pd.Timestamp
    execution_timestamp: pd.Timestamp
    target_timestamp: pd.Timestamp
    features_cutoff: pd.Timestamp
    horizon: str
    model_version: str
    model_hash: str
    predicted_close: float

    @classmethod
    def from_mapping(cls, payload):
        return cls(
            symbol=str(payload["symbol"]),
            prediction_asof=pd.Timestamp(payload["prediction_asof"]),
            execution_timestamp=pd.Timestamp(payload["execution_timestamp"]),
            target_timestamp=pd.Timestamp(payload["target_timestamp"]),
            features_cutoff=pd.Timestamp(payload["features_cutoff"]),
            horizon=str(payload["horizon"]),
            model_version=str(payload["model_version"]),
            model_hash=str(payload["model_hash"]),
            predicted_close=float(payload["predicted_close"]),
        )

    def validate(self):
        if not self.symbol.strip():
            raise ValueError("symbol must be populated.")
        if self.prediction_asof >= self.execution_timestamp:
            raise ValueError("prediction_asof must be strictly before execution_timestamp.")
        if self.execution_timestamp > self.target_timestamp:
            raise ValueError("execution_timestamp must be less than or equal to target_timestamp.")
        if self.features_cutoff > self.prediction_asof:
            raise ValueError("features_cutoff must be less than or equal to prediction_asof.")
        if self.features_cutoff >= self.execution_timestamp:
            raise ValueError("features_cutoff must be strictly before execution_timestamp.")
        placeholders = {"", "unknown", "none", "nan", "unversioned", "model-not-recorded"}
        if self.model_version.strip().lower() in placeholders:
            raise ValueError("model_version must be populated with a non-placeholder value.")
        if self.model_hash.strip().lower() in placeholders:
            raise ValueError("model_hash must be populated with a non-placeholder value.")
        if not np.isfinite(self.predicted_close) or self.predicted_close <= 0:
            raise ValueError("predicted_close must be a finite positive number.")

        horizon_delta = pd.to_timedelta(self.horizon, errors="coerce")
        if pd.isna(horizon_delta) or horizon_delta <= pd.Timedelta(0):
            raise ValueError("horizon must be a positive pandas-compatible Timedelta string.")
        if horizon_delta != self.target_timestamp - self.prediction_asof:
            raise ValueError("horizon must equal target_timestamp - prediction_asof.")

        return self


class PaperTradingRunner:
    """Turn prediction events and bars into paper orders, fills, and reports."""

    def __init__(
        self,
        broker: PaperBroker,
        threshold=0.02,
        max_position_fraction=0.25,
        max_mark_age="1D",
        accepted_model_hashes=(),
        approval_registry=None,
        require_approval_registry=False,
    ):
        if threshold < 0:
            raise ValueError("threshold must be non-negative.")
        if not 0 < max_position_fraction <= 1:
            raise ValueError("max_position_fraction must be in (0, 1].")
        if broker.risk_manager is None:
            raise ValueError("PaperTradingRunner requires a broker with a PaperRiskManager.")

        self.broker = broker
        self.threshold = threshold
        self.max_position_fraction = max_position_fraction
        self.max_mark_age = pd.Timedelta(max_mark_age) if max_mark_age is not None else None
        self.accepted_model_hashes = set(accepted_model_hashes)
        self.approval_registry = approval_registry
        self.require_approval_registry = bool(require_approval_registry)
        if self.require_approval_registry and self.approval_registry is None:
            raise ValueError("approval_registry is required when require_approval_registry=True.")
        self.last_close: dict[str, float] = {}
        self.last_close_timestamp: dict[str, pd.Timestamp] = {}
        self.equity_history: list[dict] = []
        self.signal_history: list[dict] = []

    @classmethod
    def create(
        cls,
        initial_cash,
        threshold=0.02,
        max_position_fraction=0.25,
        commission_rate=0.001,
        slippage_rate=0.0005,
        max_participation_rate=0.1,
        max_drawdown=0.2,
        min_cash_fraction=0.01,
        max_mark_age="1D",
        accepted_model_hashes=(),
        approval_registry=None,
        require_approval_registry=False,
    ):
        risk_manager = PaperRiskManager(
            initial_equity=initial_cash,
            max_drawdown=max_drawdown,
            min_cash_fraction=min_cash_fraction,
        )
        broker = PaperBroker(
            initial_cash=initial_cash,
            commission_rate=commission_rate,
            slippage_rate=slippage_rate,
            max_participation_rate=max_participation_rate,
            risk_manager=risk_manager,
        )
        return cls(
            broker=broker,
            threshold=threshold,
            max_position_fraction=max_position_fraction,
            max_mark_age=max_mark_age,
            accepted_model_hashes=accepted_model_hashes,
            approval_registry=approval_registry,
            require_approval_registry=require_approval_registry,
        )

    def process_bar(self, bar: MarketBar, prediction: PredictionEvent | None = None):
        bar.timestamp = pd.Timestamp(bar.timestamp)
        self._validate_bar(bar)
        bar_key = f"{bar.symbol}|{bar.timestamp.isoformat()}"
        if bar_key in self.broker.processed_bars:
            if prediction is not None:
                raise ValueError("Cannot process a prediction for a bar that was already processed.")
            return []

        self.broker.equity(
            {bar.symbol: bar.open},
            asof=bar.timestamp,
            max_price_age=self.max_mark_age,
        )

        if prediction is not None:
            prediction.validate()
            if prediction.symbol != bar.symbol:
                raise ValueError("Prediction symbol must match bar symbol.")
            if prediction.execution_timestamp != bar.timestamp:
                raise ValueError("Prediction execution_timestamp must match bar timestamp.")
            if self.approval_registry is not None:
                self.approval_registry.require(prediction.model_hash)
            elif prediction.model_hash not in self.accepted_model_hashes:
                raise ValueError("Prediction model_hash is not approved by the validation gate.")
            self._submit_target_order(bar, prediction)

        fills = self.broker.process_bar(bar, reference_prices={bar.symbol: bar.open})
        equity = self.broker.equity(
            {bar.symbol: bar.close},
            asof=bar.timestamp,
            max_price_age=self.max_mark_age,
        )
        self.broker.risk_manager.update(equity)
        position = self.broker.positions.get(bar.symbol)
        self.last_close[bar.symbol] = float(bar.close)
        self.last_close_timestamp[bar.symbol] = pd.Timestamp(bar.timestamp)
        self.equity_history.append(
            {
                "timestamp": bar.timestamp,
                "symbol": bar.symbol,
                "equity": equity,
                "cash": self.broker.cash,
                "position": position.quantity if position else 0,
                "close": bar.close,
                "halted": self.broker.risk_manager.halted,
            }
        )
        return fills

    @staticmethod
    def _validate_bar(bar: MarketBar):
        if bar.open <= 0 or bar.high <= 0 or bar.low <= 0 or bar.close <= 0:
            raise ValueError("MarketBar prices must be positive.")
        if bar.volume < 0:
            raise ValueError("MarketBar volume cannot be negative.")
        if bar.low > min(bar.open, bar.close) or bar.high < max(bar.open, bar.close):
            raise ValueError("MarketBar high/low are inconsistent with open/close.")

    def _submit_target_order(self, bar: MarketBar, prediction: PredictionEvent):
        previous_close = self.last_close.get(bar.symbol)
        previous_close_timestamp = self.last_close_timestamp.get(bar.symbol)
        if previous_close is None or previous_close_timestamp is None:
            self.signal_history.append(
                {
                    "timestamp": bar.timestamp,
                    "symbol": bar.symbol,
                    "signal": 0,
                    "reason": "missing_previous_close",
                }
            )
            return None
        if pd.Timestamp(previous_close_timestamp) != pd.Timestamp(prediction.prediction_asof):
            raise ValueError(
                "prediction_asof must match the latest processed close timestamp for the symbol."
            )

        predicted_return = prediction.predicted_close / previous_close - 1.0
        signal = 1 if predicted_return > self.threshold else 0
        if self.broker.risk_manager.halted:
            signal = 0

        current_position = self.broker.positions.get(bar.symbol)
        current_shares = current_position.quantity if current_position else 0
        self.broker.cancel_open_orders(symbol=bar.symbol, reason="replaced by paper runner target")

        if signal > 0:
            equity = self.broker.equity(
                {bar.symbol: bar.open},
                asof=bar.timestamp,
                max_price_age=self.max_mark_age,
            )
            target_notional = equity * self.max_position_fraction
            fill_price_estimate = bar.open * (1 + self.broker.slippage_rate)
            target_shares = int(
                target_notional / (fill_price_estimate * (1 + self.broker.commission_rate))
            )
        else:
            target_shares = 0

        delta = target_shares - current_shares
        order = None
        if delta > 0:
            order = self.broker.submit_order(
                bar.symbol,
                BUY,
                delta,
                bar.timestamp,
                reference_prices={bar.symbol: bar.open},
            )
        elif delta < 0:
            order = self.broker.submit_order(
                bar.symbol,
                SELL,
                abs(delta),
                bar.timestamp,
                reference_prices={bar.symbol: bar.open},
            )

        self.signal_history.append(
            {
                "timestamp": bar.timestamp,
                "symbol": bar.symbol,
                "predicted_return": predicted_return,
                "signal": signal,
                "target_shares": target_shares,
                "current_shares": current_shares,
                "order_id": order.id if order else None,
            }
        )
        return order

    def report(self):
        equity_df = pd.DataFrame(self.equity_history)
        if equity_df.empty:
            return {
                "final_equity": self.broker.cash,
                "total_return": 0.0,
                "max_drawdown": 0.0,
                "fills": len(self.broker.fills),
                "halted": self.broker.risk_manager.halted,
            }

        equity = equity_df["equity"].astype(float)
        total_return = equity.iloc[-1] / self.broker.initial_cash - 1.0
        peak = equity.expanding().max()
        max_drawdown = ((equity - peak) / peak).min()
        return {
            "final_equity": equity.iloc[-1],
            "total_return": total_return,
            "max_drawdown": max_drawdown,
            "fills": len(self.broker.fills),
            "halted": self.broker.risk_manager.halted,
        }

    def save_state(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.broker.save_state(directory / "broker_state.json")
        runner_payload = {
            "threshold": self.threshold,
            "max_position_fraction": self.max_position_fraction,
            "max_mark_age": str(self.max_mark_age) if self.max_mark_age is not None else None,
            "approval_registry_path": str(getattr(self.approval_registry, "path", "")),
            "require_approval_registry": self.require_approval_registry,
            "last_close": self.last_close,
            "last_close_timestamp": {
                symbol: pd.Timestamp(timestamp).isoformat()
                for symbol, timestamp in self.last_close_timestamp.items()
            },
            "equity_history": self._serialize_records(self.equity_history),
            "signal_history": self._serialize_records(self.signal_history),
        }
        (directory / "runner_state.json").write_text(
            json.dumps(runner_payload, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load_state(cls, directory, approval_registry=None):
        directory = Path(directory)
        broker = PaperBroker.load_state(directory / "broker_state.json")
        payload = json.loads((directory / "runner_state.json").read_text(encoding="utf-8"))
        registry_path = payload.get("approval_registry_path")
        if approval_registry is None and registry_path:
            from .validation import ModelApprovalRegistry

            if not Path(registry_path).exists():
                raise ValueError("approval_registry file does not exist.")
            approval_registry = ModelApprovalRegistry(registry_path)
        if approval_registry is not None and not approval_registry.accepted_model_hashes():
            raise ValueError("approval_registry does not contain approved models.")
        if payload.get("require_approval_registry") and approval_registry is None:
            raise ValueError("approval_registry is required to load this runner state.")
        runner = cls(
            broker=broker,
            threshold=payload["threshold"],
            max_position_fraction=payload["max_position_fraction"],
            max_mark_age=pd.Timedelta(payload["max_mark_age"]) if payload.get("max_mark_age") else None,
            accepted_model_hashes=(),
            approval_registry=approval_registry,
            require_approval_registry=payload.get("require_approval_registry", False),
        )
        runner.last_close = {symbol: float(value) for symbol, value in payload["last_close"].items()}
        runner.last_close_timestamp = {
            symbol: pd.Timestamp(value)
            for symbol, value in payload.get("last_close_timestamp", {}).items()
        }
        runner.equity_history = cls._deserialize_records(payload["equity_history"])
        runner.signal_history = cls._deserialize_records(payload["signal_history"])
        return runner

    @staticmethod
    def load_predictions(path):
        path = Path(path)
        if path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            records = payload.get("prediction_results", payload)
            df = pd.DataFrame(records)
        else:
            df = pd.read_csv(path)

        missing = REQUIRED_PREDICTION_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"Prediction file is missing required columns: {sorted(missing)}")

        events = [PredictionEvent.from_mapping(row).validate() for row in df.to_dict("records")]
        return sorted(events, key=lambda event: (event.execution_timestamp, event.symbol))

    @staticmethod
    def _serialize_records(records):
        serialized = []
        for record in records:
            converted = {}
            for key, value in record.items():
                converted[key] = value.isoformat() if isinstance(value, pd.Timestamp) else value
            serialized.append(converted)
        return serialized

    @staticmethod
    def _deserialize_records(records):
        deserialized = []
        for record in records:
            converted = dict(record)
            if "timestamp" in converted:
                converted["timestamp"] = pd.Timestamp(converted["timestamp"])
            deserialized.append(converted)
        return deserialized
