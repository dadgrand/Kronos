"""Production runtime orchestration and fail-closed preflight gates."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess

import pandas as pd

from trading.live import LiveExecutionLoop, LiveOrderRequest
from trading.market_data import OhlcvBar, interval_to_ms
from trading.ops import FileKillSwitch, HeartbeatMonitor, JsonlOrderJournal
from trading.runner import PredictionEvent
from trading.validation import ModelApprovalRegistry

from .config import DEFAULT_SYMBOLS, ProdConfig
from .ledger import PredictionLedger, stable_hash


@dataclass(frozen=True)
class PreflightIssue:
    severity: str
    code: str
    message: str


@dataclass(frozen=True)
class PreflightReport:
    ok: bool
    mode: str
    issues: list[PreflightIssue]
    details: dict

    def to_record(self) -> dict:
        return {
            "ok": self.ok,
            "mode": self.mode,
            "issues": [issue.__dict__ for issue in self.issues],
            "details": self.details,
        }


class ProdRiskPolicy:
    def __init__(self, config: ProdConfig):
        self.config = config

    def validate_order_request(self, request: LiveOrderRequest, *, reference_price) -> None:
        if request.symbol not in self.config.symbols:
            raise ValueError("Order symbol is outside the configured production universe.")
        if request.side not in {"BUY", "SELL"}:
            raise ValueError("Only BUY/SELL spot orders are allowed.")
        if request.order_type not in {"MARKET", "LIMIT"}:
            raise ValueError("Production v1 allows only MARKET or LIMIT spot orders.")
        notional = float(request.quantity) * float(reference_price)
        if self.config.max_order_notional is None:
            raise ValueError("max_order_notional is required before live order submission.")
        if notional > float(self.config.max_order_notional):
            raise ValueError("Order notional exceeds configured max_order_notional.")


class HoldLastClosePredictor:
    """Safe shadow-only predictor used for scheduler smoke tests."""

    model_version = "shadow-hold-last-close"
    model_hash = "shadow-hold-last-close-not-live"

    def predict_close(self, bars: list[OhlcvBar]) -> float:
        if not bars:
            raise ValueError("Cannot predict without at least one input bar.")
        return float(bars[-1].close)


class ShadowScheduler:
    def __init__(self, config: ProdConfig, ledger: PredictionLedger | None = None, predictor=None):
        self.config = config
        self.ledger = ledger or PredictionLedger(config.prediction_ledger_path)
        self.predictor = predictor or HoldLastClosePredictor()

    def write_predictions_for_closed_bars(self, bars: list[OhlcvBar], *, code_version=None) -> list[dict]:
        by_symbol: dict[str, list[OhlcvBar]] = {}
        for bar in sorted((bar for bar in bars if bar.closed), key=lambda item: (item.symbol, item.period_end_ms)):
            if bar.symbol in self.config.symbols:
                by_symbol.setdefault(bar.symbol, []).append(bar)
        records = []
        for symbol in self.config.symbols:
            symbol_bars = by_symbol.get(symbol, [])
            if not symbol_bars:
                continue
            latest = symbol_bars[-1]
            event = self._prediction_event(symbol, latest, symbol_bars)
            source_checksums = {
                "latest_bar_raw_response_sha256": latest.raw_response_sha256,
                "latest_bar_record_key": latest.record_key,
            }
            input_window_hash = stable_hash([bar.to_record() for bar in symbol_bars[-512:]])
            records.append(
                self.ledger.append(
                    event,
                    source_checksums=source_checksums,
                    input_window_hash=input_window_hash,
                    code_version=code_version or current_code_version(),
                )
            )
        return records

    def _prediction_event(self, symbol: str, latest: OhlcvBar, bars: list[OhlcvBar]) -> PredictionEvent:
        interval_ms = interval_to_ms(self.config.interval)
        prediction_asof = pd.Timestamp(latest.period_end)
        execution_timestamp = pd.Timestamp(latest.period_end_ms + interval_ms, unit="ms", tz="UTC")
        predicted_close = float(self.predictor.predict_close(bars))
        return PredictionEvent(
            symbol=symbol,
            prediction_asof=prediction_asof,
            execution_timestamp=execution_timestamp,
            target_timestamp=execution_timestamp,
            features_cutoff=prediction_asof,
            horizon=str(execution_timestamp - prediction_asof),
            model_version=getattr(self.predictor, "model_version", self.config.model_key),
            model_hash=getattr(self.predictor, "model_hash", stable_hash(self.config.to_record())),
            predicted_close=predicted_close,
        ).validate()


class ProdPreflight:
    def __init__(self, config: ProdConfig):
        self.config = config

    def run(self, *, broker_adapter=None, local_broker=None, require_live=False) -> PreflightReport:
        issues: list[PreflightIssue] = []
        details = {
            "config": self.config.to_record(),
            "device": self._device_status(),
            "shadow_gate": self._shadow_gate_status(),
            "approval": self._approval_status(),
            "kill_switch_active": FileKillSwitch(self.config.kill_switch_path).is_active(),
        }
        if details["kill_switch_active"]:
            issues.append(PreflightIssue("P0", "kill_switch", "Kill switch is active."))
        if not details["device"]["ok"]:
            issues.append(PreflightIssue("P0", "device", details["device"]["message"]))
        if self.config.live_requested or require_live:
            if not self.config.live_trading_enabled:
                issues.append(
                    PreflightIssue("P0", "live_env", "KRONOS_ENABLE_LIVE_TRADING=1 is required for live modes.")
                )
            if self.config.max_order_notional is None:
                issues.append(PreflightIssue("P0", "max_order_notional", "max_order_notional must be configured."))
            if not details["shadow_gate"]["ok"]:
                issues.append(PreflightIssue("P0", "shadow_gate", details["shadow_gate"]["message"]))
            if not details["approval"]["ok"]:
                issues.append(PreflightIssue("P0", "approval", details["approval"]["message"]))
            if broker_adapter is None or local_broker is None:
                issues.append(
                    PreflightIssue("P0", "broker_preflight", "Broker adapter and local broker are required.")
                )
            else:
                broker_status = self._broker_status(broker_adapter, local_broker)
                details["broker"] = broker_status
                if not broker_status["ok"]:
                    issues.append(PreflightIssue("P0", "broker_reconciliation", broker_status["message"]))
        ok = not issues
        return PreflightReport(ok=ok, mode=self.config.mode, issues=issues, details=details)

    def _device_status(self) -> dict:
        if not str(self.config.device).lower().startswith("cuda"):
            return {"ok": True, "device": self.config.device, "message": "CPU/shadow mode device accepted."}
        if not self.config.cuda_visible_devices:
            return {
                "ok": False,
                "device": self.config.device,
                "message": "CUDA device requires explicit cuda_visible_devices to avoid touching unrelated GPUs.",
            }
        try:
            import torch

            available = bool(torch.cuda.is_available())
        except Exception as exc:
            return {"ok": False, "device": self.config.device, "message": f"CUDA preflight failed: {exc}"}
        return {"ok": available, "device": self.config.device, "message": "CUDA available." if available else "CUDA unavailable."}

    def _shadow_gate_status(self) -> dict:
        ledger = PredictionLedger(self.config.prediction_ledger_path)
        try:
            records = ledger.read_all()
        except ValueError as exc:
            return {"ok": False, "message": str(exc), "records": 0}
        predictions = [record["prediction"] for record in records]
        if not predictions:
            return {"ok": False, "message": "No shadow prediction ledger records found.", "records": 0}
        frame = pd.DataFrame(predictions)
        for column in ("prediction_asof", "execution_timestamp", "target_timestamp", "features_cutoff"):
            frame[column] = pd.to_datetime(frame[column], utc=True)
        symbols = set(frame["symbol"])
        window_days = (frame["target_timestamp"].max() - frame["target_timestamp"].min()).days + 1
        leakage = not (
            (frame["prediction_asof"] < frame["execution_timestamp"]).all()
            and (frame["execution_timestamp"] <= frame["target_timestamp"]).all()
            and (frame["features_cutoff"] <= frame["prediction_asof"]).all()
            and (frame["features_cutoff"] < frame["execution_timestamp"]).all()
        )
        ok = symbols == set(DEFAULT_SYMBOLS) and window_days >= self.config.shadow_gate_days and not leakage
        message = "Shadow gate passed." if ok else "Shadow ledger lacks 30-day full-universe forward-only evidence."
        return {
            "ok": ok,
            "message": message,
            "records": len(frame),
            "symbols": sorted(symbols),
            "window_days": int(window_days),
            "leakage": leakage,
        }

    def _approval_status(self) -> dict:
        registry = ModelApprovalRegistry(
            self.config.approval_registry_path,
            code_manifest_paths=production_code_manifest_paths(),
        )
        try:
            accepted = sorted(registry.accepted_model_hashes())
        except ValueError as exc:
            return {"ok": False, "message": str(exc), "accepted_model_hashes": []}
        if not accepted:
            return {"ok": False, "message": "No accepted model hashes in approval registry.", "accepted_model_hashes": []}
        return {"ok": True, "message": "Approval registry contains accepted model hashes.", "accepted_model_hashes": accepted}

    def _broker_status(self, broker_adapter, local_broker) -> dict:
        try:
            journal = JsonlOrderJournal(self.config.journal_path)
            loop = LiveExecutionLoop(broker_adapter, journal=journal)
            report = loop.preflight(local_broker)
        except Exception as exc:
            return {"ok": False, "message": str(exc)}
        return {"ok": report.ok, "message": "Broker reconciled." if report.ok else "Broker reconciliation failed."}


def status_record(config: ProdConfig) -> dict:
    heartbeat = _read_json(config.heartbeat_path)
    live_state = _read_json(config.live_state_path)
    ledger_records = []
    ledger_error = ""
    try:
        ledger_records = PredictionLedger(config.prediction_ledger_path).read_all()
    except ValueError as exc:
        ledger_error = str(exc)
    return {
        "mode": config.mode,
        "venue": config.venue,
        "symbols": list(config.symbols),
        "run_dir": str(config.run_dir.resolve()),
        "kill_switch_active": FileKillSwitch(config.kill_switch_path).is_active(),
        "heartbeat": heartbeat,
        "live_state": live_state,
        "prediction_ledger_records": len(ledger_records),
        "prediction_ledger_error": ledger_error,
    }


def write_dashboard(config: ProdConfig, path=None) -> Path:
    path = Path(path or config.run_dir / "dashboard.html")
    record = status_record(config)
    preflight = ProdPreflight(config).run().to_record()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_dashboard_html(record, preflight), encoding="utf-8")
    return path


def enable_canary_state(config: ProdConfig, *, operator: str) -> Path:
    gate = ProdPreflight(config)._shadow_gate_status()
    if not gate["ok"]:
        raise ValueError(gate["message"])
    payload = {
        "state": "canary_enabled",
        "operator": operator,
        "enabled_at": pd.Timestamp.now("UTC").isoformat(),
        "shadow_gate": gate,
    }
    config.run_dir.mkdir(parents=True, exist_ok=True)
    config.live_state_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return config.live_state_path


def disable_live_state(config: ProdConfig, *, reason: str) -> Path:
    FileKillSwitch(config.kill_switch_path).activate(reason)
    payload = {
        "state": "disabled",
        "reason": reason,
        "disabled_at": pd.Timestamp.now("UTC").isoformat(),
    }
    config.run_dir.mkdir(parents=True, exist_ok=True)
    config.live_state_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return config.live_state_path


def beat(config: ProdConfig, status="ok", payload=None) -> dict:
    return HeartbeatMonitor(config.heartbeat_path).beat(status=status, payload=payload or {})


def current_code_version() -> str:
    root = Path(__file__).resolve().parents[1]
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{revision}{'-dirty' if dirty else ''}"


def production_code_manifest_paths() -> tuple[str, ...]:
    return ModelApprovalRegistry.CODE_MANIFEST_PATHS + (
        "trading/market_data.py",
        "prod/binance.py",
        "prod/config.py",
        "prod/inference.py",
        "prod/ledger.py",
        "prod/runtime.py",
        "prod/cli.py",
    )


def _read_json(path: Path):
    if not Path(path).is_file():
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _dashboard_html(status: dict, preflight: dict) -> str:
    issue_rows = "".join(
        f"<tr><td>{item['severity']}</td><td>{item['code']}</td><td>{item['message']}</td></tr>"
        for item in preflight["issues"]
    ) or "<tr><td colspan='3'>No blocking issues in this view.</td></tr>"
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Kronos Prod Status</title>
<style>
body {{ margin:0; font-family:Segoe UI, Arial, sans-serif; background:#f6f8fb; color:#172033; }}
main {{ width:min(1120px, calc(100vw - 32px)); margin:24px auto; }}
.grid {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; }}
.card, section {{ background:white; border:1px solid #d8e0ea; border-radius:8px; padding:14px; }}
.label {{ color:#617089; font-size:12px; text-transform:uppercase; letter-spacing:.04em; }}
.value {{ margin-top:8px; font-size:22px; font-weight:700; }}
table {{ width:100%; border-collapse:collapse; margin-top:10px; }}
th,td {{ text-align:left; border-bottom:1px solid #e7ecf3; padding:8px; }}
code,pre {{ overflow-wrap:anywhere; white-space:pre-wrap; }}
@media(max-width:840px) {{ .grid {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} }}
</style>
</head>
<body><main>
<h1>Kronos Prod Status</h1>
<div class="grid">
<div class="card"><div class="label">Mode</div><div class="value">{status['mode']}</div></div>
<div class="card"><div class="label">Venue</div><div class="value">{status['venue']}</div></div>
<div class="card"><div class="label">Kill switch</div><div class="value">{'ACTIVE' if status['kill_switch_active'] else 'clear'}</div></div>
<div class="card"><div class="label">Predictions</div><div class="value">{status['prediction_ledger_records']}</div></div>
</div>
<section><h2>Preflight</h2><table><thead><tr><th>Severity</th><th>Code</th><th>Message</th></tr></thead><tbody>{issue_rows}</tbody></table></section>
<section><h2>Status JSON</h2><pre>{json.dumps(status, indent=2, sort_keys=True)}</pre></section>
</main></body></html>"""
