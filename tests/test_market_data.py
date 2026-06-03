from dataclasses import replace
import json

import pytest

from trading.market_data import (
    MarketDataSnapshot,
    OhlcvBar,
    SourceRequest,
    build_quality_report,
    parse_binance_klines,
    parse_kraken_ohlc,
    update_closed_actuals_ledger,
    write_market_data_package,
    write_package,
)


def request_record(source="binance"):
    return SourceRequest(
        source=source,
        endpoint=f"https://example.com/{source}",
        params={"symbol": "BTCUSDT"},
        requested_at="2026-06-03T00:00:00Z",
        received_at="2026-06-03T00:00:01Z",
        latency_ms=1.0,
        status_code=200,
        response_sha256="a" * 64,
        response_bytes=100,
        headers={},
    )


def binance_rows():
    return [
        [
            1_700_000_000_000,
            "100.0",
            "101.0",
            "99.0",
            "100.5",
            "10.0",
            1_700_000_059_999,
            "1000.0",
            20,
            "5.0",
            "500.0",
            "0",
        ],
        [
            1_700_000_060_000,
            "100.5",
            "102.0",
            "100.0",
            "101.5",
            "12.0",
            1_700_000_119_999,
            "1200.0",
            24,
            "6.0",
            "600.0",
            "0",
        ],
    ]


def closed_bar(symbol="BTCUSDT", source="binance", open_time_ms=1_700_000_000_000):
    rows = binance_rows()
    rows[0][0] = open_time_ms
    rows[0][6] = open_time_ms + 59_999
    return parse_binance_klines(
        [rows[0]],
        symbol=symbol,
        interval="1m",
        collected_at="2023-11-14T22:15:00Z",
        server_time_ms=open_time_ms + 120_000,
        close_grace_ms=1_000,
        request_id="req",
        response_sha256="a" * 64,
    )[0]


def test_binance_parser_marks_closed_and_running_candles():
    bars = parse_binance_klines(
        binance_rows(),
        symbol="BTCUSDT",
        interval="1m",
        collected_at="2023-11-14T22:15:00Z",
        server_time_ms=1_700_000_061_000,
        close_grace_ms=1_000,
        request_id="req",
        response_sha256="a" * 64,
    )

    assert bars[0].closed is True
    assert bars[1].closed is False
    assert bars[0].period_end == "2023-11-14T22:14:20Z"
    assert bars[0].to_actual_record()["timestamp_semantics"] == "period_end_exclusive_utc"


def test_kraken_parser_never_exports_last_rest_row_as_committed():
    rows = [
        [1_700_000_000, "100.0", "101.0", "99.0", "100.5", "100.1", "10.0", 20],
        [1_700_000_060, "100.5", "102.0", "100.0", "101.5", "101.1", "12.0", 24],
    ]

    bars = parse_kraken_ohlc(
        rows,
        symbol="BTCUSDT",
        exchange_symbol="XBTUSDT",
        interval="1m",
        collected_at="2023-11-14T22:16:00Z",
        close_grace_ms=1_000,
        request_id="req",
        response_sha256="b" * 64,
    )

    assert bars[0].closed is True
    assert bars[1].closed is False
    with pytest.raises(ValueError, match="Running candles"):
        bars[1].to_actual_record()


def test_quality_report_flags_closed_duplicates_and_gaps():
    first = closed_bar()
    duplicate = closed_bar()
    gap = closed_bar(open_time_ms=first.open_time_ms + 180_000)

    quality = build_quality_report(
        [first, duplicate, gap],
        [request_record()],
        [],
        primary_source="binance",
    )

    assert quality["readiness"]["actuals_ledger_ready"] is False
    assert "Duplicate closed bars are present." in quality["review_findings"]["p0"]
    assert quality["gaps"][0]["missing_intervals"] == 2


def test_closed_actuals_ledger_skips_duplicates_and_detects_conflicts(tmp_path):
    ledger = tmp_path / "closed_actuals_ledger.jsonl"
    first = closed_bar()
    duplicate = closed_bar()
    changed = replace(
        first,
        high=first.high + 2.0,
        close=first.close + 1.0,
        high_raw=str(first.high + 2.0),
        close_raw=str(first.close + 1.0),
    )

    initial = update_closed_actuals_ledger(ledger, [first])
    repeated = update_closed_actuals_ledger(ledger, [duplicate])
    conflict = update_closed_actuals_ledger(ledger, [changed])

    assert initial.new_records == 1
    assert repeated.duplicate_records == 1
    assert conflict.conflicts[0]["record_key"] == first.record_key


def test_write_package_outputs_closed_actuals_and_dashboard(tmp_path):
    bars = parse_binance_klines(
        binance_rows(),
        symbol="BTCUSDT",
        interval="1m",
        collected_at="2023-11-14T22:15:00Z",
        server_time_ms=1_700_000_061_000,
        close_grace_ms=1_000,
        request_id="req",
        response_sha256="a" * 64,
    )
    snapshot = MarketDataSnapshot(
        source="binance",
        symbol="BTCUSDT",
        exchange_symbol="BTCUSDT",
        interval="1m",
        bars=bars,
        requests=[request_record()],
        raw_payload=binance_rows(),
    )

    result = write_package([snapshot], [], tmp_path, primary_source="binance")
    actuals = json.loads((tmp_path / "actuals_for_validation.json").read_text(encoding="utf-8"))
    quality = json.loads((tmp_path / "quality_report.json").read_text(encoding="utf-8"))

    assert result["primary_closed_bars"] == 1
    assert len(actuals["actuals"]) == 1
    assert quality["readiness"]["live_chart_is_alpha_evidence"] is False
    assert (tmp_path / "dashboard.html").is_file()
    assert (tmp_path / "manifest.json").is_file()


def test_write_market_data_package_declares_actuals_contract(tmp_path):
    bar = closed_bar()
    package = write_market_data_package([bar], tmp_path, request_log=[request_record()])
    manifest = json.loads((tmp_path / "source_manifest.json").read_text(encoding="utf-8"))
    actuals = json.loads((tmp_path / "actuals.json").read_text(encoding="utf-8"))

    assert package["quality"].passed is True
    assert manifest["schema_version"] == "kronos.market_data_package.v1"
    assert manifest["actuals_contract"]["join_key"] == ["symbol", "timestamp"]
    assert manifest["actuals_contract"]["timestamp_semantics"] == "period_end_exclusive_utc"
    assert manifest["source_docs"]["binance_rest_klines"].startswith("https://")
    assert actuals["actuals"][0]["timestamp"] == bar.period_end
