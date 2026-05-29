"""Operational primitives for paper/live trading loops."""

from __future__ import annotations

from contextlib import contextmanager
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import threading
import time

import pandas as pd

_JOURNAL_LOCKS: dict[Path, threading.Lock] = {}


class JsonlOrderJournal:
    """Append-only JSONL audit journal for trading events."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._thread_lock = _JOURNAL_LOCKS.setdefault(self.path.resolve(), threading.Lock())

    def append(self, event_type, payload):
        with self._thread_lock:
            with self.path.open("a+", encoding="utf-8") as f:
                with self._file_lock(f):
                    sequence_id, previous_checksum = self._next_journal_state(f)
                    record = {
                        "sequence_id": sequence_id,
                        "previous_checksum": previous_checksum,
                        "event_type": event_type,
                        "recorded_at": pd.Timestamp.now("UTC").isoformat(),
                        "payload": self._json_safe(payload),
                    }
                    record["checksum"] = self._checksum(record)
                    f.seek(0, os.SEEK_END)
                    f.write(json.dumps(record, sort_keys=True) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
        return record

    def read_all(self):
        if not self.path.exists():
            return []
        records = []
        with self.path.open("r", encoding="utf-8") as f:
            expected_sequence = 0
            previous_checksum = ""
            for line in f:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    self._verify_checksum(record)
                    if "sequence_id" in record:
                        if int(record["sequence_id"]) != expected_sequence:
                            raise ValueError("Journal sequence_id continuity mismatch.")
                        if record.get("previous_checksum", "") != previous_checksum:
                            raise ValueError("Journal checksum chain mismatch.")
                        expected_sequence += 1
                        previous_checksum = record.get("checksum", "")
                    records.append(record)
        return records

    @staticmethod
    @contextmanager
    def _file_lock(file_obj):
        if os.name == "nt":
            import msvcrt

            file_obj.seek(0)
            msvcrt.locking(file_obj.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                file_obj.seek(0)
                msvcrt.locking(file_obj.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(file_obj.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(file_obj.fileno(), fcntl.LOCK_UN)

    @classmethod
    def _next_journal_state(cls, file_obj):
        file_obj.seek(0)
        expected_sequence = 0
        previous_checksum = ""
        for line in file_obj:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            cls._verify_checksum(record)
            if "sequence_id" in record:
                if int(record["sequence_id"]) != expected_sequence:
                    raise ValueError("Journal sequence_id continuity mismatch.")
                if record.get("previous_checksum", "") != previous_checksum:
                    raise ValueError("Journal checksum chain mismatch.")
                expected_sequence += 1
                previous_checksum = record.get("checksum", "")
            else:
                expected_sequence += 1
                previous_checksum = record.get("checksum", "")
        return expected_sequence, previous_checksum

    @classmethod
    def _checksum(cls, record):
        checksum_payload = dict(record)
        checksum_payload.pop("checksum", None)
        canonical = json.dumps(checksum_payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @classmethod
    def _verify_checksum(cls, record):
        for field in ("sequence_id", "previous_checksum", "checksum"):
            if field not in record:
                raise ValueError(f"Journal record missing {field}.")
        expected = record.get("checksum")
        actual = cls._checksum(record)
        if actual != expected:
            raise ValueError("Journal record checksum mismatch.")

    @classmethod
    def _json_safe(cls, value):
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        if isinstance(value, Decimal):
            return str(value)
        if isinstance(value, dict):
            return {key: cls._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._json_safe(item) for item in value]
        if hasattr(value, "__dict__"):
            return cls._json_safe(vars(value))
        return value


class FileKillSwitch:
    """File-backed kill switch for deterministic local control."""

    def __init__(self, path):
        self.path = Path(path)

    def activate(self, reason="manual"):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(reason, encoding="utf-8")

    def clear(self):
        if self.path.exists():
            self.path.unlink()

    def is_active(self):
        return self.path.exists()

    def reason(self):
        if not self.path.exists():
            return ""
        return self.path.read_text(encoding="utf-8").strip()


class HeartbeatMonitor:
    """Simple heartbeat writer for process monitoring."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def beat(self, status="ok", payload=None):
        record = {
            "timestamp": pd.Timestamp.now("UTC").isoformat(),
            "status": status,
            "payload": payload or {},
            "epoch": time.time(),
        }
        self.path.write_text(json.dumps(record, sort_keys=True), encoding="utf-8")
        return record
