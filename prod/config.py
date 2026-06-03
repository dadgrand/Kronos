"""Strict production configuration for the Kronos local runtime."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path


DEFAULT_SYMBOLS = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT")
PROD_MODES = {"shadow", "canary_live", "live"}


@dataclass(frozen=True)
class ProdConfig:
    mode: str = "shadow"
    venue: str = "binance_spot"
    symbols: tuple[str, ...] = DEFAULT_SYMBOLS
    interval: str = "1m"
    run_dir: Path = Path("var/kronos_prod")
    model_key: str = "kronos-small"
    model_revision: str = "901c26c1332695a2a8f243eb2f37243a37bea320"
    tokenizer_revision: str = "0e0117387f39004a9016484a186a908917e22426"
    device: str = "cpu"
    cuda_visible_devices: str | None = None
    max_gross_exposure: float = 0.50
    max_symbol_exposure: float = 0.125
    min_cash_fraction: float = 0.40
    max_daily_loss: float = 0.02
    max_drawdown_halt: float = 0.05
    max_order_notional: float | None = None
    shadow_gate_days: int = 30
    long_threshold: float = 0.0
    transaction_cost_bps: float = 10.0
    min_net_excess_return: float = 0.01
    min_directional_accuracy: float = 0.52
    min_active_period_fraction: float = 0.10
    prediction_ledger_name: str = "prediction_results.jsonl"
    actuals_name: str = "actuals.json"
    approval_registry_name: str = "approvals.json"
    journal_name: str = "orders.jsonl"
    heartbeat_name: str = "heartbeat.json"
    kill_switch_name: str = "KILL_SWITCH"
    live_state_name: str = "live_state.json"
    binance_base_url: str = "https://api.binance.com"
    recv_window_ms: int = 5000

    def __post_init__(self):
        object.__setattr__(self, "mode", str(self.mode).strip().lower())
        object.__setattr__(self, "venue", str(self.venue).strip().lower())
        object.__setattr__(self, "symbols", tuple(str(symbol).strip().upper() for symbol in self.symbols))
        object.__setattr__(self, "run_dir", Path(self.run_dir))
        self.validate()

    @property
    def live_requested(self) -> bool:
        return self.mode in {"canary_live", "live"}

    @property
    def live_trading_enabled(self) -> bool:
        return os.environ.get("KRONOS_ENABLE_LIVE_TRADING") == "1"

    @property
    def prediction_ledger_path(self) -> Path:
        return self.run_dir / self.prediction_ledger_name

    @property
    def actuals_path(self) -> Path:
        return self.run_dir / self.actuals_name

    @property
    def approval_registry_path(self) -> Path:
        return self.run_dir / self.approval_registry_name

    @property
    def journal_path(self) -> Path:
        return self.run_dir / self.journal_name

    @property
    def heartbeat_path(self) -> Path:
        return self.run_dir / self.heartbeat_name

    @property
    def kill_switch_path(self) -> Path:
        return self.run_dir / self.kill_switch_name

    @property
    def live_state_path(self) -> Path:
        return self.run_dir / self.live_state_name

    def validate(self) -> None:
        if self.mode not in PROD_MODES:
            raise ValueError(f"mode must be one of {sorted(PROD_MODES)}.")
        if self.venue != "binance_spot":
            raise ValueError("venue must be binance_spot for the v1 production runtime.")
        if self.symbols != DEFAULT_SYMBOLS:
            raise ValueError(f"symbols must exactly match the v1 universe: {list(DEFAULT_SYMBOLS)}.")
        if self.interval != "1m":
            raise ValueError("interval must be 1m for the v1 production runtime.")
        if not self.model_key.strip():
            raise ValueError("model_key must be populated.")
        if not self.model_revision.strip() or not self.tokenizer_revision.strip():
            raise ValueError("model_revision and tokenizer_revision must be populated.")
        self._fraction(self.max_gross_exposure, "max_gross_exposure", lower_open=True)
        self._fraction(self.max_symbol_exposure, "max_symbol_exposure", lower_open=True)
        self._fraction(self.min_cash_fraction, "min_cash_fraction")
        self._fraction(self.max_daily_loss, "max_daily_loss", lower_open=True)
        self._fraction(self.max_drawdown_halt, "max_drawdown_halt", lower_open=True)
        if self.max_symbol_exposure > self.max_gross_exposure:
            raise ValueError("max_symbol_exposure cannot exceed max_gross_exposure.")
        if self.live_requested and self.max_order_notional is None:
            raise ValueError("max_order_notional is required before canary_live/live modes.")
        if self.max_order_notional is not None and float(self.max_order_notional) <= 0:
            raise ValueError("max_order_notional must be positive when provided.")
        if int(self.shadow_gate_days) < 30:
            raise ValueError("shadow_gate_days must be at least 30.")
        if self.recv_window_ms <= 0 or self.recv_window_ms > 60_000:
            raise ValueError("recv_window_ms must be in (0, 60000].")

    @staticmethod
    def _fraction(value, name, *, lower_open=False):
        value = float(value)
        if lower_open and not 0 < value <= 1:
            raise ValueError(f"{name} must be in (0, 1].")
        if not lower_open and not 0 <= value <= 1:
            raise ValueError(f"{name} must be in [0, 1].")

    def to_record(self) -> dict:
        payload = asdict(self)
        payload["run_dir"] = str(self.run_dir)
        return payload

    def write_snapshot(self) -> Path:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        path = self.run_dir / "config.snapshot.json"
        path.write_text(json.dumps(self.to_record(), indent=2, sort_keys=True), encoding="utf-8")
        return path


def load_config(path: str | Path | None = None, env: dict | None = None) -> ProdConfig:
    env = env or os.environ
    payload = {}
    if path:
        config_path = Path(path)
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload = _apply_env_overrides(payload, env)
    return ProdConfig(**payload)


def _apply_env_overrides(payload: dict, env: dict) -> dict:
    result = dict(payload)
    mapping = {
        "KRONOS_PROD_MODE": ("mode", str),
        "KRONOS_RUN_DIR": ("run_dir", Path),
        "KRONOS_MODEL_KEY": ("model_key", str),
        "KRONOS_MODEL_REVISION": ("model_revision", str),
        "KRONOS_TOKENIZER_REVISION": ("tokenizer_revision", str),
        "KRONOS_DEVICE": ("device", str),
        "KRONOS_CUDA_VISIBLE_DEVICES": ("cuda_visible_devices", str),
        "KRONOS_MAX_ORDER_NOTIONAL": ("max_order_notional", float),
        "KRONOS_BINANCE_BASE_URL": ("binance_base_url", str),
    }
    for env_name, (field_name, converter) in mapping.items():
        value = env.get(env_name)
        if value not in {None, ""}:
            result[field_name] = converter(value)
    symbols = env.get("KRONOS_SYMBOLS")
    if symbols:
        result["symbols"] = tuple(item.strip().upper() for item in symbols.split(",") if item.strip())
    return result
