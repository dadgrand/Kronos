"""Append-only prediction ledgers for forward-only production validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import pandas as pd

from trading.runner import PredictionEvent


@dataclass(frozen=True)
class PredictionLedgerRecord:
    sequence_id: int
    previous_checksum: str
    recorded_at: str
    event_type: str
    prediction: dict
    source_checksums: dict
    input_window_hash: str
    code_version: str
    checksum: str


class PredictionLedger:
    """Checksum-chained JSONL ledger for immutable prediction events."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, prediction, *, source_checksums: dict, input_window_hash: str, code_version: str) -> dict:
        event = prediction if isinstance(prediction, PredictionEvent) else PredictionEvent.from_mapping(prediction)
        event.validate()
        if not source_checksums:
            raise ValueError("source_checksums must be populated.")
        if not str(input_window_hash).strip():
            raise ValueError("input_window_hash must be populated.")
        if not str(code_version).strip():
            raise ValueError("code_version must be populated.")
        sequence_id, previous_checksum = self._next_state()
        record = {
            "sequence_id": sequence_id,
            "previous_checksum": previous_checksum,
            "recorded_at": pd.Timestamp.now("UTC").isoformat(),
            "event_type": "prediction",
            "prediction": _json_safe(asdict(event)),
            "source_checksums": _json_safe(source_checksums),
            "input_window_hash": str(input_window_hash),
            "code_version": str(code_version),
        }
        record["checksum"] = self._checksum(record)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        return record

    def read_all(self) -> list[dict]:
        if not self.path.exists():
            return []
        records = []
        expected_sequence = 0
        previous_checksum = ""
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                self._verify(record)
                if record["sequence_id"] != expected_sequence:
                    raise ValueError("Prediction ledger sequence_id continuity mismatch.")
                if record["previous_checksum"] != previous_checksum:
                    raise ValueError("Prediction ledger checksum chain mismatch.")
                expected_sequence += 1
                previous_checksum = record["checksum"]
                records.append(record)
        return records

    def prediction_events(self) -> list[dict]:
        return [record["prediction"] for record in self.read_all() if record.get("event_type") == "prediction"]

    def export_prediction_json(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "contract_schema_version": "kronos-prediction-contract-v1",
            "prediction_results": self.prediction_events(),
            "source_ledger": str(self.path.resolve()),
            "exported_at": pd.Timestamp.now("UTC").isoformat(),
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return path

    def _next_state(self) -> tuple[int, str]:
        records = self.read_all()
        if not records:
            return 0, ""
        last = records[-1]
        return int(last["sequence_id"]) + 1, last["checksum"]

    @classmethod
    def _verify(cls, record) -> None:
        for field in (
            "sequence_id",
            "previous_checksum",
            "recorded_at",
            "event_type",
            "prediction",
            "source_checksums",
            "input_window_hash",
            "code_version",
            "checksum",
        ):
            if field not in record:
                raise ValueError(f"Prediction ledger record missing {field}.")
        expected = record["checksum"]
        actual = cls._checksum(record)
        if actual != expected:
            raise ValueError("Prediction ledger checksum mismatch.")
        PredictionEvent.from_mapping(record["prediction"]).validate()

    @staticmethod
    def _checksum(record) -> str:
        payload = dict(record)
        payload.pop("checksum", None)
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def stable_hash(value) -> str:
    canonical = json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def file_checksum(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value):
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value
