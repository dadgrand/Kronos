"""Public market-data ingestion with validation-grade provenance.

The collector deliberately separates committed OHLCV bars from running candles.
Only committed primary-source bars are written to the append-only actuals ledger
that can later be joined to forward-only prediction events.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import html
import json
import math
from pathlib import Path
import time
from typing import Iterable
import urllib.error
import urllib.parse
import urllib.request


BINANCE_REST_DOC = "https://github.com/binance/binance-spot-api-docs/blob/master/rest-api.md#klinecandlestick-data"
BINANCE_WS_DOC = "https://github.com/binance/binance-spot-api-docs/blob/master/web-socket-streams.md#klinecandlestick-streams-for-utc"
KRAKEN_REST_DOC = "https://docs.kraken.com/api/docs/rest-api/get-ohlc-data/"
KRAKEN_WS_DOC = "https://docs.kraken.com/api/docs/websocket-v1/ohlc/"

BINANCE_WS_BASE_URL = "wss://data-stream.binance.vision"
BINANCE_MARKET_DATA_BASES = (
    "https://data-api.binance.vision",
    "https://api.binance.com",
)
KRAKEN_REST_BASE = "https://api.kraken.com"

INTERVAL_MS = {
    "1s": 1_000,
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "8h": 28_800_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
    "3d": 259_200_000,
    "1w": 604_800_000,
}
KRAKEN_INTERVAL_MINUTES = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "4h": 240,
    "1d": 1440,
    "1w": 10080,
    "15d": 21600,
}
KRAKEN_SYMBOL_MAP = {
    "BTCUSDT": "XBTUSDT",
    "ETHUSDT": "ETHUSDT",
    "SOLUSDT": "SOLUSDT",
    "ADAUSDT": "ADAUSDT",
    "XRPUSDT": "XRPUSDT",
}


class MarketDataError(RuntimeError):
    """Raised when a market-data request fails with captured provenance."""

    def __init__(self, message: str, request: "SourceRequest | None" = None):
        super().__init__(message)
        self.request = request


@dataclass(frozen=True)
class SourceRequest:
    source: str
    endpoint: str
    params: dict
    requested_at: str
    received_at: str
    latency_ms: float
    status_code: int | None
    response_sha256: str | None
    response_bytes: int
    headers: dict
    error: str | None = None

    @property
    def request_id(self) -> str:
        payload = {
            "source": self.source,
            "endpoint": self.endpoint,
            "params": self.params,
            "requested_at": self.requested_at,
            "response_sha256": self.response_sha256,
            "error": self.error,
        }
        return stable_sha256(payload)[:24]

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["request_id"] = self.request_id
        return payload


@dataclass(frozen=True)
class OhlcvBar:
    source: str
    symbol: str
    exchange_symbol: str
    interval: str
    open_time: str
    close_time: str
    period_end: str
    open_time_ms: int
    close_time_ms: int
    period_end_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: float | None
    trade_count: int | None
    closed: bool
    collected_at: str
    request_id: str
    raw_response_sha256: str
    open_raw: str
    high_raw: str
    low_raw: str
    close_raw: str
    volume_raw: str
    quote_volume_raw: str | None = None
    vwap_raw: str | None = None

    def __post_init__(self):
        self.validate()

    @property
    def record_key(self) -> str:
        return "|".join(
            [
                self.source,
                self.symbol,
                self.interval,
                str(self.open_time_ms),
                str(self.period_end_ms),
            ]
        )

    def validate(self) -> "OhlcvBar":
        if self.source not in {"binance", "kraken"}:
            raise ValueError("source must be binance or kraken.")
        if not self.symbol.strip() or not self.exchange_symbol.strip():
            raise ValueError("symbol and exchange_symbol must be populated.")
        interval_ms = interval_to_ms(self.interval)
        if self.period_end_ms - self.open_time_ms != interval_ms:
            raise ValueError("bar period length does not match interval.")
        if self.close_time_ms != self.period_end_ms - 1:
            raise ValueError("close_time_ms must be period_end_ms - 1.")
        for name in ("open", "high", "low", "close"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite positive number.")
        if self.low > min(self.open, self.close) or self.high < max(self.open, self.close):
            raise ValueError("OHLC high/low are inconsistent with open/close.")
        if not math.isfinite(float(self.volume)) or self.volume < 0:
            raise ValueError("volume must be finite and non-negative.")
        if self.quote_volume is not None and (
            not math.isfinite(float(self.quote_volume)) or self.quote_volume < 0
        ):
            raise ValueError("quote_volume must be finite and non-negative.")
        return self

    def to_dict(self) -> dict:
        return asdict(self) | {"record_key": self.record_key}

    def to_record(self) -> dict:
        return {
            "symbol": self.symbol,
            "exchange_symbol": self.exchange_symbol,
            "exchange": "BINANCE_SPOT" if self.source == "binance" else "KRAKEN_SPOT",
            "source": self.source,
            "interval": self.interval,
            "open_time": self.open_time,
            "close_time": self.close_time,
            "period_end": self.period_end,
            "timestamp_semantics": "period_end_exclusive_utc",
            "open": self.open_raw,
            "high": self.high_raw,
            "low": self.low_raw,
            "close": self.close_raw,
            "volume": self.volume_raw,
            "quote_volume": self.quote_volume_raw,
            "trade_count": self.trade_count,
            "is_closed": self.closed,
            "request_id": self.request_id,
            "raw_response_sha256": self.raw_response_sha256,
            "record_key": self.record_key,
            "ingested_at": self.collected_at,
        }

    def to_actual_record(self) -> dict:
        if not self.closed:
            raise ValueError("Running candles cannot be exported as validation actuals.")
        return {
            "symbol": self.symbol,
            "timestamp": self.period_end,
            "close": self.close,
            "source": self.source,
            "interval": self.interval,
            "bar_open_time": self.open_time,
            "bar_close_time": self.close_time,
            "timestamp_semantics": "period_end_exclusive_utc",
            "source_request_id": self.request_id,
            "source_bar_key": self.record_key,
        }


@dataclass(frozen=True)
class MarketDataSnapshot:
    source: str
    symbol: str
    exchange_symbol: str
    interval: str
    bars: list[OhlcvBar]
    requests: list[SourceRequest]
    raw_payload: object


@dataclass(frozen=True)
class LedgerUpdate:
    path: str
    new_records: int
    duplicate_records: int
    conflicts: list[dict]


@dataclass(frozen=True)
class MarketDataQualityReport:
    rows: int
    closed_rows: int
    symbols: list[str]
    interval: str
    start: str | None
    end: str | None
    duplicate_keys: int
    gap_count: int
    running_candle_rows: int
    non_positive_ohlc_rows: int
    inconsistent_ohlc_rows: int
    symbol_summary: list[dict]
    warnings: list[str]
    generated_at: str

    @property
    def passed(self) -> bool:
        return not self.warnings

    def to_record(self) -> dict:
        payload = asdict(self)
        payload["passed"] = self.passed
        return payload


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def iso_from_ms(ms: int) -> str:
    return utc_iso(datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc))


def ms_from_iso(value: str) -> int:
    normalized = value.replace("Z", "+00:00")
    return int(datetime.fromisoformat(normalized).timestamp() * 1000)


def stable_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def stable_sha256(value) -> str:
    if isinstance(value, bytes):
        data = value
    else:
        data = stable_json(value).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def interval_to_ms(interval: str) -> int:
    if interval not in INTERVAL_MS:
        raise ValueError(f"Unsupported interval: {interval}")
    return INTERVAL_MS[interval]


def parse_number(value, name: str, *, positive=False, non_negative=False) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric.") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite.")
    if positive and parsed <= 0:
        raise ValueError(f"{name} must be positive.")
    if non_negative and parsed < 0:
        raise ValueError(f"{name} cannot be negative.")
    return parsed


def http_get_json(
    source: str,
    endpoint: str,
    params: dict | None = None,
    *,
    timeout: float = 10.0,
    max_retries: int = 2,
) -> tuple[object, SourceRequest]:
    params = {key: value for key, value in (params or {}).items() if value is not None}
    query = urllib.parse.urlencode(params)
    url = endpoint if not query else f"{endpoint}?{query}"
    last_request: SourceRequest | None = None
    for attempt in range(max_retries + 1):
        requested_at = utc_now()
        started = time.perf_counter()
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "KronosMarketDataValidation/1.0"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read()
                received_at = utc_now()
                request_record = SourceRequest(
                    source=source,
                    endpoint=endpoint,
                    params=params,
                    requested_at=utc_iso(requested_at),
                    received_at=utc_iso(received_at),
                    latency_ms=round((time.perf_counter() - started) * 1000, 3),
                    status_code=int(response.status),
                    response_sha256=hashlib.sha256(body).hexdigest(),
                    response_bytes=len(body),
                    headers=interesting_headers(dict(response.headers)),
                )
                payload = json.loads(body.decode("utf-8"))
                return payload, request_record
        except urllib.error.HTTPError as exc:
            body = exc.read()
            received_at = utc_now()
            retry_after = exc.headers.get("Retry-After")
            last_request = SourceRequest(
                source=source,
                endpoint=endpoint,
                params=params,
                requested_at=utc_iso(requested_at),
                received_at=utc_iso(received_at),
                latency_ms=round((time.perf_counter() - started) * 1000, 3),
                status_code=int(exc.code),
                response_sha256=hashlib.sha256(body).hexdigest() if body else None,
                response_bytes=len(body),
                headers=interesting_headers(dict(exc.headers)),
                error=f"HTTP {exc.code}: {body[:256].decode('utf-8', 'replace')}",
            )
            if exc.code in {418, 429, 500, 502, 503, 504} and attempt < max_retries:
                sleep_for = float(retry_after) if retry_after else min(2.0 * (attempt + 1), 5.0)
                time.sleep(sleep_for)
                continue
            raise MarketDataError(last_request.error or "HTTP request failed.", last_request) from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            received_at = utc_now()
            last_request = SourceRequest(
                source=source,
                endpoint=endpoint,
                params=params,
                requested_at=utc_iso(requested_at),
                received_at=utc_iso(received_at),
                latency_ms=round((time.perf_counter() - started) * 1000, 3),
                status_code=None,
                response_sha256=None,
                response_bytes=0,
                headers={},
                error=f"{type(exc).__name__}: {exc}",
            )
            if attempt < max_retries:
                time.sleep(min(2.0 * (attempt + 1), 5.0))
                continue
            raise MarketDataError(last_request.error or "Request failed.", last_request) from exc
    raise MarketDataError("Request failed.", last_request)


def interesting_headers(headers: dict) -> dict:
    prefixes = (
        "x-mbx-used-weight",
        "retry-after",
        "content-type",
        "date",
        "cf-cache-status",
        "cache-control",
    )
    result = {}
    for key, value in headers.items():
        lowered = key.lower()
        if lowered.startswith(prefixes):
            result[key] = value
    return result


class BinanceMarketDataClient:
    """Binance public market-data client for REST klines."""

    def __init__(
        self,
        bases: Iterable[str] = BINANCE_MARKET_DATA_BASES,
        *,
        base_url: str | None = None,
        timeout: float = 10.0,
        close_grace_ms=1_000,
    ):
        self.bases = (str(base_url).rstrip("/"),) if base_url else tuple(str(base).rstrip("/") for base in bases)
        self.timeout = float(timeout)
        self.close_grace_ms = int(close_grace_ms)

    def fetch_klines(self, symbol: str, interval: str, limit=500) -> MarketDataSnapshot:
        interval_to_ms(interval)
        errors = []
        for base in self.bases:
            try:
                return self._fetch_from_base(base, symbol, interval, limit)
            except MarketDataError as exc:
                errors.append(str(exc))
        raise MarketDataError(f"Binance klines failed for {symbol}: {'; '.join(errors)}")

    def _fetch_from_base(self, base: str, symbol: str, interval: str, limit: int) -> MarketDataSnapshot:
        server_payload, time_request = http_get_json("binance", f"{base}/api/v3/time", timeout=self.timeout)
        server_time_ms = int(server_payload["serverTime"])
        params = {"symbol": symbol.upper(), "interval": interval, "limit": int(limit)}
        payload, klines_request = http_get_json(
            "binance",
            f"{base}/api/v3/klines",
            params,
            timeout=self.timeout,
        )
        if not isinstance(payload, list):
            raise MarketDataError("Binance kline response must be a list.", klines_request)
        collected_at = klines_request.received_at
        bars = parse_binance_klines(
            payload,
            symbol=symbol.upper(),
            interval=interval,
            collected_at=collected_at,
            server_time_ms=server_time_ms,
            close_grace_ms=self.close_grace_ms,
            request_id=klines_request.request_id,
            response_sha256=klines_request.response_sha256 or "",
        )
        return MarketDataSnapshot(
            source="binance",
            symbol=symbol.upper(),
            exchange_symbol=symbol.upper(),
            interval=interval,
            bars=bars,
            requests=[time_request, klines_request],
            raw_payload=payload,
        )

    def fetch_closed_bars(self, symbols: Iterable[str], *, interval="1m", limit=500):
        bars: list[OhlcvBar] = []
        requests: list[SourceRequest] = []
        for symbol in symbols:
            snapshot = self.fetch_klines(str(symbol).upper(), interval, limit=limit)
            bars.extend(bar for bar in snapshot.bars if bar.closed)
            requests.extend(snapshot.requests)
        bars = sorted(bars, key=lambda item: (item.symbol, item.open_time_ms))
        return bars, requests


class KrakenMarketDataClient:
    """Kraken public market-data client for REST OHLC."""

    def __init__(self, base: str = KRAKEN_REST_BASE, *, close_grace_ms=1_000):
        self.base = base.rstrip("/")
        self.close_grace_ms = int(close_grace_ms)

    def fetch_ohlc(self, symbol: str, interval: str, limit=500) -> MarketDataSnapshot:
        if interval not in KRAKEN_INTERVAL_MINUTES:
            raise MarketDataError(f"Kraken does not support interval {interval}.")
        exchange_symbol = KRAKEN_SYMBOL_MAP.get(symbol.upper(), symbol.upper())
        params = {"pair": exchange_symbol, "interval": KRAKEN_INTERVAL_MINUTES[interval]}
        payload, request = http_get_json("kraken", f"{self.base}/0/public/OHLC", params)
        if not isinstance(payload, dict):
            raise MarketDataError("Kraken OHLC response must be an object.", request)
        if payload.get("error"):
            raise MarketDataError(f"Kraken OHLC error: {payload['error']}", request)
        result = payload.get("result", {})
        pair_keys = [key for key in result if key != "last"]
        if not pair_keys:
            raise MarketDataError("Kraken OHLC response did not include a pair result.", request)
        pair_key = pair_keys[0]
        rows = result[pair_key]
        if limit:
            rows = rows[-int(limit) :]
        bars = parse_kraken_ohlc(
            rows,
            symbol=symbol.upper(),
            exchange_symbol=pair_key,
            interval=interval,
            collected_at=request.received_at,
            close_grace_ms=self.close_grace_ms,
            request_id=request.request_id,
            response_sha256=request.response_sha256 or "",
        )
        return MarketDataSnapshot(
            source="kraken",
            symbol=symbol.upper(),
            exchange_symbol=pair_key,
            interval=interval,
            bars=bars,
            requests=[request],
            raw_payload=payload,
        )


def parse_binance_klines(
    payload,
    *,
    symbol: str,
    interval: str,
    collected_at: str,
    server_time_ms: int,
    close_grace_ms: int,
    request_id: str,
    response_sha256: str,
) -> list[OhlcvBar]:
    bars = []
    for row in payload:
        if len(row) < 11:
            raise ValueError("Binance kline row is missing fields.")
        open_time_ms = int(row[0])
        close_time_ms = int(row[6])
        period_end_ms = close_time_ms + 1
        closed = period_end_ms <= server_time_ms - close_grace_ms
        bars.append(
            OhlcvBar(
                source="binance",
                symbol=symbol.upper(),
                exchange_symbol=symbol.upper(),
                interval=interval,
                open_time=iso_from_ms(open_time_ms),
                close_time=iso_from_ms(close_time_ms),
                period_end=iso_from_ms(period_end_ms),
                open_time_ms=open_time_ms,
                close_time_ms=close_time_ms,
                period_end_ms=period_end_ms,
                open=parse_number(row[1], "open", positive=True),
                high=parse_number(row[2], "high", positive=True),
                low=parse_number(row[3], "low", positive=True),
                close=parse_number(row[4], "close", positive=True),
                volume=parse_number(row[5], "volume", non_negative=True),
                quote_volume=parse_number(row[7], "quote_volume", non_negative=True),
                trade_count=int(row[8]),
                closed=closed,
                collected_at=collected_at,
                request_id=request_id,
                raw_response_sha256=response_sha256,
                open_raw=str(row[1]),
                high_raw=str(row[2]),
                low_raw=str(row[3]),
                close_raw=str(row[4]),
                volume_raw=str(row[5]),
                quote_volume_raw=str(row[7]),
            )
        )
    return bars


def parse_kraken_ohlc(
    rows,
    *,
    symbol: str,
    exchange_symbol: str,
    interval: str,
    collected_at: str,
    close_grace_ms: int,
    request_id: str,
    response_sha256: str,
) -> list[OhlcvBar]:
    interval_ms = interval_to_ms(interval)
    collected_ms = ms_from_iso(collected_at)
    bars = []
    for index, row in enumerate(rows):
        if len(row) < 8:
            raise ValueError("Kraken OHLC row is missing fields.")
        open_time_ms = int(float(row[0]) * 1000)
        period_end_ms = open_time_ms + interval_ms
        close_time_ms = period_end_ms - 1
        closed = index < len(rows) - 1 and period_end_ms <= collected_ms - close_grace_ms
        bars.append(
            OhlcvBar(
                source="kraken",
                symbol=symbol.upper(),
                exchange_symbol=exchange_symbol,
                interval=interval,
                open_time=iso_from_ms(open_time_ms),
                close_time=iso_from_ms(close_time_ms),
                period_end=iso_from_ms(period_end_ms),
                open_time_ms=open_time_ms,
                close_time_ms=close_time_ms,
                period_end_ms=period_end_ms,
                open=parse_number(row[1], "open", positive=True),
                high=parse_number(row[2], "high", positive=True),
                low=parse_number(row[3], "low", positive=True),
                close=parse_number(row[4], "close", positive=True),
                volume=parse_number(row[6], "volume", non_negative=True),
                quote_volume=None,
                trade_count=int(row[7]),
                closed=closed,
                collected_at=collected_at,
                request_id=request_id,
                raw_response_sha256=response_sha256,
                open_raw=str(row[1]),
                high_raw=str(row[2]),
                low_raw=str(row[3]),
                close_raw=str(row[4]),
                volume_raw=str(row[6]),
                quote_volume_raw=None,
                vwap_raw=str(row[5]),
            )
        )
    return bars


def collect_snapshots(
    symbols: Iterable[str],
    *,
    interval="1m",
    limit=240,
    sources=("binance", "kraken"),
) -> tuple[list[MarketDataSnapshot], list[dict]]:
    snapshots: list[MarketDataSnapshot] = []
    errors: list[dict] = []
    binance = BinanceMarketDataClient()
    kraken = KrakenMarketDataClient()
    for symbol in symbols:
        normalized = symbol.upper()
        if "binance" in sources:
            try:
                snapshots.append(binance.fetch_klines(normalized, interval, limit=limit))
            except MarketDataError as exc:
                errors.append(error_record("binance", normalized, interval, exc))
        if "kraken" in sources:
            try:
                snapshots.append(kraken.fetch_ohlc(normalized, interval, limit=limit))
            except MarketDataError as exc:
                errors.append(error_record("kraken", normalized, interval, exc))
    return snapshots, errors


def error_record(source: str, symbol: str, interval: str, exc: MarketDataError) -> dict:
    record = {
        "source": source,
        "symbol": symbol,
        "interval": interval,
        "error": str(exc),
        "captured_at": utc_iso(utc_now()),
    }
    if exc.request is not None:
        record["request"] = exc.request.to_dict()
    return record


def build_quality_report(
    bars: list[OhlcvBar],
    requests: list[SourceRequest],
    errors: list[dict],
    *,
    primary_source: str,
    ledger_update: LedgerUpdate | None = None,
) -> dict:
    duplicates = duplicate_bar_keys(bars)
    invalid_rows = []
    for bar in bars:
        try:
            bar.validate()
        except ValueError as exc:
            invalid_rows.append({"record_key": bar.record_key, "error": str(exc)})
    gaps, overlaps = gap_report(bars)
    freshness = freshness_report(bars)
    source_summaries = summarize_sources(bars)
    primary_closed = [
        bar for bar in bars if bar.source == primary_source and bar.closed
    ]
    ledger_conflicts = ledger_update.conflicts if ledger_update else []
    p0 = []
    p1 = []
    if invalid_rows:
        p0.append("Invalid OHLCV rows are present.")
    if any(item["closed_duplicate_count"] for item in duplicates):
        p0.append("Duplicate closed bars are present.")
    if ledger_conflicts:
        p0.append("Existing actuals ledger contains conflicting records.")
    if not primary_closed:
        p0.append("No closed primary-source bars are available for validation actuals.")
    if gaps:
        p1.append("Closed-bar gaps are present; downstream predictions must not assume continuous coverage.")
    if errors:
        p1.append("One or more source requests failed; fallback/cross-check coverage is incomplete.")
    if freshness.get("latest_closed_lag_seconds") is not None and freshness["latest_closed_lag_seconds"] > 300:
        p1.append("Latest closed primary bar is more than 5 minutes old.")

    return {
        "generated_at": utc_iso(utc_now()),
        "primary_source": primary_source,
        "timestamp_semantics": {
            "open_time": "exchange kline open time in UTC",
            "close_time": "source close timestamp normalized to period_end - 1 millisecond",
            "period_end": "exclusive UTC period end; use this as validation actual timestamp",
            "running_candles": "closed=false rows are for viewing only and must not score predictions",
        },
        "source_docs": source_docs(),
        "row_counts": {
            "bars_total": len(bars),
            "closed_bars_total": sum(1 for bar in bars if bar.closed),
            "running_bars_total": sum(1 for bar in bars if not bar.closed),
            "primary_closed_bars": len(primary_closed),
            "requests_total": len(requests),
            "request_errors": len(errors),
        },
        "source_summaries": source_summaries,
        "freshness": freshness,
        "duplicates": duplicates,
        "gaps": gaps,
        "overlaps": overlaps,
        "invalid_rows": invalid_rows,
        "request_errors": errors,
        "ledger_update": asdict(ledger_update) if ledger_update else None,
        "readiness": {
            "actuals_ledger_ready": not p0,
            "approval_grade_forward_oos_ready": False,
            "approval_grade_forward_oos_blocker": (
                "Market actuals are ready for future scoring, but approval-grade OOS also requires "
                "immutable prediction events generated before each target period_end with model_hash, "
                "code_version, feature cutoff, and checksums."
            ),
            "live_chart_is_alpha_evidence": False,
        },
        "review_findings": {"p0": p0, "p1": p1},
    }


def duplicate_bar_keys(bars: list[OhlcvBar]) -> list[dict]:
    seen = {}
    for bar in bars:
        seen.setdefault(bar.record_key, []).append(bar)
    rows = []
    for key, group in sorted(seen.items()):
        if len(group) > 1:
            rows.append(
                {
                    "record_key": key,
                    "count": len(group),
                    "closed_duplicate_count": sum(1 for bar in group if bar.closed),
                }
            )
    return rows


def gap_report(bars: list[OhlcvBar]) -> tuple[list[dict], list[dict]]:
    gaps = []
    overlaps = []
    groups: dict[tuple[str, str, str], list[OhlcvBar]] = {}
    for bar in bars:
        if bar.closed:
            groups.setdefault((bar.source, bar.symbol, bar.interval), []).append(bar)
    for (source, symbol, interval), group in sorted(groups.items()):
        expected = interval_to_ms(interval)
        ordered = sorted(group, key=lambda item: item.open_time_ms)
        for previous, current in zip(ordered, ordered[1:]):
            delta = current.open_time_ms - previous.open_time_ms
            if delta > expected:
                missing = max(int(delta / expected) - 1, 1)
                gaps.append(
                    {
                        "source": source,
                        "symbol": symbol,
                        "interval": interval,
                        "after_period_end": previous.period_end,
                        "before_open_time": current.open_time,
                        "missing_intervals": missing,
                    }
                )
            elif delta < expected:
                overlaps.append(
                    {
                        "source": source,
                        "symbol": symbol,
                        "interval": interval,
                        "previous_open_time": previous.open_time,
                        "current_open_time": current.open_time,
                        "delta_ms": delta,
                    }
                )
    return gaps, overlaps


def freshness_report(bars: list[OhlcvBar]) -> dict:
    closed = [bar for bar in bars if bar.closed]
    if not closed:
        return {"latest_closed_period_end": None, "latest_closed_lag_seconds": None}
    latest = max(closed, key=lambda bar: bar.period_end_ms)
    latest_collected = max(ms_from_iso(bar.collected_at) for bar in bars)
    lag = max((latest_collected - latest.period_end_ms) / 1000.0, 0.0)
    return {
        "latest_closed_source": latest.source,
        "latest_closed_symbol": latest.symbol,
        "latest_closed_period_end": latest.period_end,
        "latest_closed_lag_seconds": round(lag, 3),
    }


def summarize_sources(bars: list[OhlcvBar]) -> list[dict]:
    groups: dict[tuple[str, str, str], list[OhlcvBar]] = {}
    for bar in bars:
        groups.setdefault((bar.source, bar.symbol, bar.interval), []).append(bar)
    rows = []
    for (source, symbol, interval), group in sorted(groups.items()):
        closed = [bar for bar in group if bar.closed]
        rows.append(
            {
                "source": source,
                "symbol": symbol,
                "interval": interval,
                "rows": len(group),
                "closed_rows": len(closed),
                "running_rows": len(group) - len(closed),
                "first_period_end": min((bar.period_end for bar in group), default=None),
                "latest_closed_period_end": max((bar.period_end for bar in closed), default=None),
                "latest_close": closed[-1].close if closed else None,
            }
        )
    return rows


def source_docs() -> dict:
    return {
        "binance_rest_klines": BINANCE_REST_DOC,
        "binance_websocket_klines": BINANCE_WS_DOC,
        "kraken_rest_ohlc": KRAKEN_REST_DOC,
        "kraken_websocket_ohlc": KRAKEN_WS_DOC,
    }


def update_closed_actuals_ledger(path: Path, bars: list[OhlcvBar]) -> LedgerUpdate:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, str] = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                existing[str(record["record_key"])] = str(record["record_checksum"])
    new_records = []
    duplicate_records = 0
    conflicts = []
    for bar in sorted((bar for bar in bars if bar.closed), key=lambda item: (item.symbol, item.period_end_ms)):
        actual = bar.to_actual_record()
        record = {
            "record_key": bar.record_key,
            "ingested_at": utc_iso(utc_now()),
            "actual": actual,
        }
        checksum = stable_sha256(record["actual"])
        record["record_checksum"] = checksum
        old_checksum = existing.get(bar.record_key)
        if old_checksum is None:
            new_records.append(record)
            existing[bar.record_key] = checksum
        elif old_checksum == checksum:
            duplicate_records += 1
        else:
            conflicts.append(
                {
                    "record_key": bar.record_key,
                    "existing_checksum": old_checksum,
                    "new_checksum": checksum,
                }
            )
    if new_records:
        with path.open("a", encoding="utf-8") as handle:
            for record in new_records:
                handle.write(stable_json(record) + "\n")
    return LedgerUpdate(
        path=str(path),
        new_records=len(new_records),
        duplicate_records=duplicate_records,
        conflicts=conflicts,
    )


def append_new_closed_bars(existing_bars: list[OhlcvBar], new_bars: list[OhlcvBar]):
    combined = list(existing_bars)
    seen = {bar.record_key: stable_sha256(bar.to_record()) for bar in combined}
    added = []
    for bar in sorted((bar for bar in new_bars if bar.closed), key=lambda item: (item.symbol, item.open_time_ms)):
        checksum = stable_sha256(bar.to_record())
        existing_checksum = seen.get(bar.record_key)
        if existing_checksum is None:
            combined.append(bar)
            added.append(bar)
            seen[bar.record_key] = checksum
        elif existing_checksum != checksum:
            raise ValueError(f"Closed bar conflict for {bar.record_key}.")
    combined.sort(key=lambda item: (item.symbol, item.open_time_ms))
    return combined, added


def build_market_data_quality_report(bars: list[OhlcvBar], *, interval: str) -> MarketDataQualityReport:
    ordered = sorted(bars, key=lambda item: (item.symbol, item.open_time_ms))
    symbols = sorted({bar.symbol for bar in ordered})
    duplicates = duplicate_bar_keys(ordered)
    gaps, overlaps = gap_report(ordered)
    non_positive = 0
    inconsistent = 0
    warnings = []
    for bar in ordered:
        if any(float(getattr(bar, field)) <= 0 for field in ("open", "high", "low", "close")):
            non_positive += 1
        if bar.low > min(bar.open, bar.close) or bar.high < max(bar.open, bar.close):
            inconsistent += 1
    running = sum(1 for bar in ordered if not bar.closed)
    if duplicates:
        warnings.append("duplicate bar keys detected")
    if gaps:
        warnings.append("closed-bar gaps detected")
    if overlaps:
        warnings.append("closed-bar overlaps detected")
    if non_positive:
        warnings.append("non-positive OHLC values detected")
    if inconsistent:
        warnings.append("OHLC high/low inconsistencies detected")
    if running:
        warnings.append("running candle rows included")

    summary = []
    for symbol in symbols:
        group = [bar for bar in ordered if bar.symbol == symbol]
        closed = [bar for bar in group if bar.closed]
        symbol_gaps, _ = gap_report(closed)
        summary.append(
            {
                "symbol": symbol,
                "rows": len(group),
                "closed_rows": len(closed),
                "start": min((bar.open_time for bar in group), default=None),
                "end": max((bar.period_end for bar in group), default=None),
                "gaps": len(symbol_gaps),
            }
        )
    return MarketDataQualityReport(
        rows=len(ordered),
        closed_rows=sum(1 for bar in ordered if bar.closed),
        symbols=symbols,
        interval=interval,
        start=min((bar.open_time for bar in ordered), default=None),
        end=max((bar.period_end for bar in ordered), default=None),
        duplicate_keys=len(duplicates),
        gap_count=len(gaps),
        running_candle_rows=running,
        non_positive_ohlc_rows=non_positive,
        inconsistent_ohlc_rows=inconsistent,
        symbol_summary=summary,
        warnings=warnings,
        generated_at=utc_iso(utc_now()),
    )


def write_market_data_package(
    bars: list[OhlcvBar],
    output_dir: Path | str,
    *,
    request_log=None,
    interval="1m",
    source_notes=None,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bars = sorted((bar for bar in bars if bar.closed), key=lambda item: (item.symbol, item.open_time_ms))
    quality = build_market_data_quality_report(bars, interval=interval)
    if not quality.passed:
        raise ValueError(f"Market-data quality failed: {quality.warnings}")

    paths = {
        "ohlcv_jsonl": output_dir / "ohlcv.jsonl",
        "ohlcv_csv": output_dir / "ohlcv.csv",
        "actuals": output_dir / "actuals.json",
        "quality_report": output_dir / "quality_report.json",
        "source_manifest": output_dir / "source_manifest.json",
    }
    records = [bar.to_record() for bar in bars]
    write_jsonl(paths["ohlcv_jsonl"], records)
    write_csv(paths["ohlcv_csv"], records)
    write_json(
        paths["actuals"],
        {
            "actuals": [bar.to_actual_record() for bar in bars],
            "timestamp_semantics": "period_end_exclusive_utc",
            "closed_only": True,
            "generated_at": utc_iso(utc_now()),
        },
    )
    write_json(paths["quality_report"], quality.to_record())
    manifest = {
        "schema_version": "kronos.market_data_package.v1",
        "generated_at": utc_iso(utc_now()),
        "exchange": "BINANCE_SPOT" if all(bar.source == "binance" for bar in bars) else "MIXED_SPOT",
        "interval": interval,
        "symbols": quality.symbols,
        "quality_passed": quality.passed,
        "actuals_contract": {
            "file": "actuals.json",
            "join_key": ["symbol", "timestamp"],
            "timestamp_field": "timestamp",
            "timestamp_semantics": "period_end_exclusive_utc",
            "closed_only": True,
            "prediction_requirement": (
                "Predictions must be written before each target timestamp and include model_hash, "
                "code_version, feature cutoff, and immutable prediction/actuals checksums."
            ),
        },
        "timestamp_semantics": {
            "open_time": "exchange kline start time in UTC",
            "close_time": "source close timestamp normalized to period_end - 1 millisecond",
            "actuals.timestamp": "exclusive UTC period_end used by Kronos validation joins",
            "is_closed": "true only after source/server time has passed the candle close with grace",
        },
        "source_docs": source_docs(),
        "source_notes": list(source_notes or []),
        "requests": [request.to_dict() if hasattr(request, "to_dict") else request for request in (request_log or [])],
        "files": {
            "ohlcv.jsonl": file_entry(paths["ohlcv_jsonl"]),
            "ohlcv.csv": file_entry(paths["ohlcv_csv"]),
            "actuals.json": file_entry(paths["actuals"]),
            "quality_report.json": file_entry(paths["quality_report"]),
        },
    }
    write_json(paths["source_manifest"], manifest)
    return {
        "bars": bars,
        "quality": quality,
        "manifest": manifest,
        "paths": paths,
    }


def file_entry(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def write_package(
    snapshots: list[MarketDataSnapshot],
    errors: list[dict],
    out_dir: Path,
    *,
    primary_source="binance",
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(exist_ok=True)
    bars = [bar for snapshot in snapshots for bar in snapshot.bars]
    requests = [request for snapshot in snapshots for request in snapshot.requests]
    raw_files = []
    for snapshot in snapshots:
        raw_path = raw_dir / f"{snapshot.source}_{snapshot.symbol}_{snapshot.interval}.json"
        write_json(raw_path, snapshot.raw_payload)
        raw_files.append(str(raw_path))

    primary_closed = [
        bar for bar in bars if bar.source == primary_source and bar.closed
    ]
    ledger_update = update_closed_actuals_ledger(out_dir / "closed_actuals_ledger.jsonl", primary_closed)
    write_jsonl(out_dir / "bars.jsonl", [bar.to_dict() for bar in bars])
    write_json(
        out_dir / "actuals_for_validation.json",
        {
            "actuals": [bar.to_actual_record() for bar in primary_closed],
            "source": primary_source,
            "timestamp_semantics": "period_end_exclusive_utc",
            "closed_only": True,
            "generated_at": utc_iso(utc_now()),
        },
    )
    provenance = {
        "schema_version": "kronos.market_data_provenance.v1",
        "generated_at": utc_iso(utc_now()),
        "sources": sorted({snapshot.source for snapshot in snapshots}),
        "primary_source": primary_source,
        "requests": [request.to_dict() for request in requests],
        "request_errors": errors,
        "raw_files": raw_files,
        "source_docs": source_docs(),
    }
    write_json(out_dir / "provenance.json", provenance)
    quality = build_quality_report(
        bars,
        requests,
        errors,
        primary_source=primary_source,
        ledger_update=ledger_update,
    )
    write_json(out_dir / "quality_report.json", quality)
    manifest = build_manifest(out_dir)
    dashboard_html = render_dashboard_html(bars, quality, manifest)
    write_text(out_dir / "dashboard.html", dashboard_html)
    manifest = build_manifest(out_dir)
    write_json(out_dir / "manifest.json", manifest)
    return {
        "out_dir": str(out_dir),
        "bars": len(bars),
        "primary_closed_bars": len(primary_closed),
        "requests": len(requests),
        "request_errors": len(errors),
        "dashboard": str(out_dir / "dashboard.html"),
        "quality_report": str(out_dir / "quality_report.json"),
        "manifest": str(out_dir / "manifest.json"),
    }


def build_manifest(out_dir: Path) -> dict:
    files = []
    for path in sorted(out_dir.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            files.append(
                {
                    "path": str(path.relative_to(out_dir)).replace("\\", "/"),
                    "bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
            )
    return {
        "schema_version": "kronos.market_data_manifest.v1",
        "generated_at": utc_iso(utc_now()),
        "files": files,
    }


def write_json(path: Path, payload) -> None:
    write_text(path, json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True))


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    write_text(path, "".join(stable_json(row) + "\n" for row in rows))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        write_text(path, "")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    fieldnames = list(rows[0].keys())
    with tmp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)


def render_dashboard_html(bars: list[OhlcvBar], quality: dict, manifest: dict) -> str:
    primary = quality["primary_source"]
    primary_closed = [bar for bar in bars if bar.source == primary and bar.closed]
    chart = line_chart_svg(primary_closed)
    latest_rows = sorted(primary_closed, key=lambda item: item.period_end_ms, reverse=True)[:18]
    cards = [
        ("Primary closed bars", quality["row_counts"]["primary_closed_bars"]),
        ("Running candles held out", quality["row_counts"]["running_bars_total"]),
        ("Request errors", quality["row_counts"]["request_errors"]),
        ("Gaps", len(quality["gaps"])),
        ("Duplicate keys", len(quality["duplicates"])),
        ("Ledger conflicts", len((quality.get("ledger_update") or {}).get("conflicts", []))),
    ]
    status = "READY" if quality["readiness"]["actuals_ledger_ready"] else "BLOCKED"
    status_class = "good" if status == "READY" else "bad"
    p0 = quality["review_findings"]["p0"]
    p1 = quality["review_findings"]["p1"]
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Kronos Real Market Data Viewer</title>
  <style>
    :root {{
      --bg: #f6f7f9;
      --ink: #141821;
      --muted: #5d6677;
      --line: #d8dde7;
      --panel: #ffffff;
      --green: #0f8b6f;
      --red: #b42318;
      --blue: #2457c5;
      --amber: #a15c00;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, Segoe UI, Arial, sans-serif;
      background: var(--bg);
      color: var(--ink);
      line-height: 1.45;
    }}
    header {{
      padding: 26px 28px 18px;
      border-bottom: 1px solid var(--line);
      background: #ffffff;
    }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 24px 18px 40px; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; letter-spacing: 0; }}
    h2 {{ margin: 28px 0 12px; font-size: 18px; letter-spacing: 0; }}
    p {{ margin: 6px 0; color: var(--muted); }}
    .status {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 4px 10px;
      border-radius: 6px;
      font-weight: 700;
      font-size: 13px;
    }}
    .good {{ color: var(--green); background: #e7f5ef; }}
    .bad {{ color: var(--red); background: #feecea; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 12px;
      margin-top: 16px;
    }}
    .card, .section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    .card {{ padding: 14px; min-height: 82px; }}
    .label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }}
    .value {{ margin-top: 8px; font-size: 24px; font-weight: 760; }}
    .section {{ padding: 18px; margin-top: 14px; overflow: auto; }}
    table {{ width: 100%; border-collapse: collapse; min-width: 720px; }}
    th, td {{ text-align: left; border-bottom: 1px solid var(--line); padding: 8px 7px; font-size: 13px; }}
    th {{ color: var(--muted); font-weight: 700; background: #fbfcfe; }}
    code {{ background: #eef1f6; padding: 1px 4px; border-radius: 4px; }}
    .warning {{ border-left: 4px solid var(--amber); padding: 10px 12px; background: #fff7e8; color: #442b00; }}
    .findings li {{ margin: 6px 0; }}
    svg {{ width: 100%; height: auto; display: block; }}
    @media (max-width: 720px) {{
      header {{ padding: 20px 16px 14px; }}
      main {{ padding: 16px 12px 30px; }}
      h1 {{ font-size: 23px; }}
      .value {{ font-size: 21px; }}
      .section {{ padding: 12px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Kronos Real Market Data Viewer</h1>
    <p><span class="status {status_class}">{status}</span> Closed primary bars are separated from running candles. Generated at {escape(quality["generated_at"])}.</p>
  </header>
  <main>
    <div class="warning">
      Live market data is not alpha evidence. This package can support future forward-only OOS scoring only after Kronos writes prediction events before each target <code>period_end</code>.
    </div>
    <div class="grid">
      {''.join(render_card(label, value) for label, value in cards)}
    </div>

    <h2>Primary Close Trend</h2>
    <div class="section">{chart}</div>

    <h2>Quality Findings</h2>
    <div class="section findings">
      {render_findings("P0", p0)}
      {render_findings("P1", p1)}
    </div>

    <h2>Source Coverage</h2>
    <div class="section">{render_source_table(quality["source_summaries"])}</div>

    <h2>Latest Primary Actuals</h2>
    <div class="section">{render_latest_table(latest_rows)}</div>

    <h2>Provenance</h2>
    <div class="section">
      <p>Primary source: <code>{escape(primary)}</code>. Validation timestamp: <code>period_end_exclusive_utc</code>.</p>
      <p>Files are checksummed in <code>manifest.json</code>. Current manifest contains {len(manifest["files"])} files.</p>
      <p>Docs: <a href="{BINANCE_REST_DOC}">Binance REST klines</a>, <a href="{BINANCE_WS_DOC}">Binance WebSocket klines</a>, <a href="{KRAKEN_REST_DOC}">Kraken REST OHLC</a>, <a href="{KRAKEN_WS_DOC}">Kraken WebSocket OHLC</a>.</p>
    </div>
  </main>
</body>
</html>"""


def render_card(label, value) -> str:
    return f'<div class="card"><div class="label">{escape(str(label))}</div><div class="value">{escape(str(value))}</div></div>'


def render_findings(title: str, findings: list[str]) -> str:
    if findings:
        items = "".join(f"<li>{escape(item)}</li>" for item in findings)
    else:
        items = "<li>None.</li>"
    return f"<p><strong>{escape(title)}</strong></p><ul>{items}</ul>"


def render_source_table(rows: list[dict]) -> str:
    body = "".join(
        "<tr>"
        f"<td>{escape(row['source'])}</td>"
        f"<td>{escape(row['symbol'])}</td>"
        f"<td>{escape(row['interval'])}</td>"
        f"<td>{row['rows']}</td>"
        f"<td>{row['closed_rows']}</td>"
        f"<td>{row['running_rows']}</td>"
        f"<td>{escape(str(row['latest_closed_period_end']))}</td>"
        f"<td>{format_number(row['latest_close'])}</td>"
        "</tr>"
        for row in rows
    )
    return (
        "<table><thead><tr><th>Source</th><th>Symbol</th><th>Interval</th><th>Rows</th>"
        "<th>Closed</th><th>Running</th><th>Latest closed</th><th>Latest close</th></tr></thead>"
        f"<tbody>{body}</tbody></table>"
    )


def render_latest_table(rows: list[OhlcvBar]) -> str:
    body = "".join(
        "<tr>"
        f"<td>{escape(row.symbol)}</td>"
        f"<td>{escape(row.period_end)}</td>"
        f"<td>{format_number(row.open)}</td>"
        f"<td>{format_number(row.high)}</td>"
        f"<td>{format_number(row.low)}</td>"
        f"<td>{format_number(row.close)}</td>"
        f"<td>{format_number(row.volume)}</td>"
        "</tr>"
        for row in rows
    )
    return (
        "<table><thead><tr><th>Symbol</th><th>Period end</th><th>Open</th><th>High</th>"
        "<th>Low</th><th>Close</th><th>Volume</th></tr></thead>"
        f"<tbody>{body}</tbody></table>"
    )


def line_chart_svg(bars: list[OhlcvBar]) -> str:
    width, height = 1040, 320
    pad_l, pad_r, pad_t, pad_b = 56, 22, 22, 38
    if not bars:
        return f'<svg viewBox="0 0 {width} {height}" role="img"><text x="24" y="48">No closed primary bars.</text></svg>'
    groups: dict[str, list[OhlcvBar]] = {}
    for bar in sorted(bars, key=lambda item: item.period_end_ms):
        groups.setdefault(bar.symbol, []).append(bar)
    xs = [bar.period_end_ms for bar in bars]
    ys = [bar.close for bar in bars]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    if min_y == max_y:
        min_y *= 0.999
        max_y *= 1.001
    span_x = max(max_x - min_x, 1)
    span_y = max(max_y - min_y, 1e-9)

    def project(bar: OhlcvBar) -> tuple[float, float]:
        x = pad_l + (bar.period_end_ms - min_x) / span_x * (width - pad_l - pad_r)
        y = pad_t + (max_y - bar.close) / span_y * (height - pad_t - pad_b)
        return x, y

    colors = ["#2457c5", "#0f8b6f", "#b42318", "#7b4ab8", "#a15c00", "#117c8b"]
    lines = []
    legend = []
    for index, (symbol, group) in enumerate(sorted(groups.items())):
        color = colors[index % len(colors)]
        points = " ".join(f"{x:.1f},{y:.1f}" for x, y in (project(bar) for bar in group))
        lines.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.2"/>')
        legend.append(
            f'<g transform="translate({pad_l + index * 138}, {height - 10})">'
            f'<line x1="0" y1="-4" x2="18" y2="-4" stroke="{color}" stroke-width="3"/>'
            f'<text x="24" y="0" font-size="12" fill="#141821">{escape(symbol)}</text></g>'
        )
    y_labels = []
    for frac in (0, 0.25, 0.5, 0.75, 1.0):
        value = min_y + (max_y - min_y) * (1 - frac)
        y = pad_t + frac * (height - pad_t - pad_b)
        y_labels.append(
            f'<line x1="{pad_l}" y1="{y:.1f}" x2="{width-pad_r}" y2="{y:.1f}" stroke="#e7ebf1"/>'
            f'<text x="8" y="{y+4:.1f}" font-size="11" fill="#5d6677">{format_number(value)}</text>'
        )
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Primary-source close trend">'
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>'
        + "".join(y_labels)
        + f'<line x1="{pad_l}" y1="{height-pad_b}" x2="{width-pad_r}" y2="{height-pad_b}" stroke="#9aa4b2"/>'
        + f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{height-pad_b}" stroke="#9aa4b2"/>'
        + "".join(lines)
        + "".join(legend)
        + "</svg>"
    )


def format_number(value) -> str:
    if value is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return escape(str(value))
    if abs(number) >= 1000:
        return f"{number:,.2f}"
    if abs(number) >= 1:
        return f"{number:.4f}".rstrip("0").rstrip(".")
    return f"{number:.8f}".rstrip("0").rstrip(".")


def escape(value: str) -> str:
    return html.escape(value, quote=True)


def collect_market_data_package(
    out_dir: Path,
    *,
    symbols: list[str],
    interval="1m",
    limit=240,
    sources=("binance", "kraken"),
    primary_source="binance",
) -> dict:
    snapshots, errors = collect_snapshots(symbols, interval=interval, limit=limit, sources=sources)
    if not snapshots:
        raise MarketDataError("No market-data snapshots were collected.")
    return write_package(snapshots, errors, out_dir, primary_source=primary_source)


def run_watch(args) -> dict:
    result = {}
    iterations = int(args.iterations or 1)
    for index in range(iterations):
        result = collect_market_data_package(
            Path(args.out),
            symbols=args.symbols,
            interval=args.interval,
            limit=args.limit,
            sources=tuple(args.sources),
            primary_source=args.primary_source,
        )
        print(stable_json(result))
        if args.watch_seconds and index < iterations - 1:
            time.sleep(float(args.watch_seconds))
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect public OHLCV data for Kronos validation.")
    parser.add_argument("--out", default="tmp/kronos_market_data_real", help="Output package directory.")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    parser.add_argument("--interval", default="1m", choices=sorted(INTERVAL_MS))
    parser.add_argument("--limit", type=int, default=240)
    parser.add_argument("--sources", nargs="+", default=["binance", "kraken"], choices=["binance", "kraken"])
    parser.add_argument("--primary-source", default="binance", choices=["binance", "kraken"])
    parser.add_argument("--watch-seconds", type=float, default=0.0)
    parser.add_argument("--iterations", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    run_watch(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
