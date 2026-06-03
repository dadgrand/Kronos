import json
from pathlib import Path

import pandas as pd
import pytest

from prod.config import DEFAULT_SYMBOLS, ProdConfig
from prod.ledger import PredictionLedger
from prod.runtime import ProdPreflight, ShadowScheduler
from trading.market_data import OhlcvBar, iso_from_ms
from trading.runner import PredictionEvent


def prediction(symbol="BTCUSDT", asof="2026-01-01T00:00:00Z"):
    asof_ts = pd.Timestamp(asof)
    target = asof_ts + pd.Timedelta(minutes=1)
    return PredictionEvent(
        symbol=symbol,
        prediction_asof=asof_ts,
        execution_timestamp=target,
        target_timestamp=target,
        features_cutoff=asof_ts,
        horizon=str(target - asof_ts),
        model_version="model-v1",
        model_hash="hash-v1",
        predicted_close=100.0,
    )


def bar(symbol="BTCUSDT", open_time_ms=1_700_000_000_000, close=100.0):
    period_end_ms = open_time_ms + 60_000
    return OhlcvBar(
        source="binance",
        symbol=symbol,
        exchange_symbol=symbol,
        interval="1m",
        open_time=iso_from_ms(open_time_ms),
        close_time=iso_from_ms(period_end_ms - 1),
        period_end=iso_from_ms(period_end_ms),
        open_time_ms=open_time_ms,
        close_time_ms=period_end_ms - 1,
        period_end_ms=period_end_ms,
        open=close,
        high=close + 1,
        low=close - 1,
        close=close,
        volume=1000,
        quote_volume=1000 * close,
        trade_count=10,
        closed=True,
        collected_at=iso_from_ms(period_end_ms + 1_000),
        request_id="req",
        raw_response_sha256="a" * 64,
        open_raw=str(close),
        high_raw=str(close + 1),
        low_raw=str(close - 1),
        close_raw=str(close),
        volume_raw="1000",
        quote_volume_raw=str(1000 * close),
    )


def test_prediction_ledger_appends_and_detects_tampering(tmp_path):
    ledger = PredictionLedger(tmp_path / "prediction_results.jsonl")
    record = ledger.append(
        prediction(),
        source_checksums={"bars": "a" * 64},
        input_window_hash="b" * 64,
        code_version="code-v1",
    )

    assert record["sequence_id"] == 0
    assert ledger.prediction_events()[0]["symbol"] == "BTCUSDT"

    path = tmp_path / "prediction_results.jsonl"
    tampered = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    tampered["prediction"]["predicted_close"] = 101
    path.write_text(json.dumps(tampered, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="checksum"):
        ledger.read_all()


def test_shadow_scheduler_writes_forward_only_predictions(tmp_path):
    config = ProdConfig(run_dir=tmp_path)
    scheduler = ShadowScheduler(config)
    records = scheduler.write_predictions_for_closed_bars([bar(symbol) for symbol in DEFAULT_SYMBOLS], code_version="code")

    assert len(records) == len(DEFAULT_SYMBOLS)
    events = PredictionLedger(config.prediction_ledger_path).prediction_events()
    for event in events:
        assert pd.Timestamp(event["prediction_asof"]) < pd.Timestamp(event["execution_timestamp"])
        assert pd.Timestamp(event["execution_timestamp"]) == pd.Timestamp(event["target_timestamp"])


def test_preflight_blocks_live_without_shadow_gate_and_env(tmp_path):
    config = ProdConfig(mode="canary_live", run_dir=tmp_path, max_order_notional=10)
    report = ProdPreflight(config).run(require_live=True)

    assert report.ok is False
    codes = {issue.code for issue in report.issues}
    assert {"live_env", "shadow_gate", "approval", "broker_preflight"}.issubset(codes)


def test_shadow_gate_passes_full_universe_thirty_day_ledger(tmp_path):
    config = ProdConfig(run_dir=tmp_path)
    ledger = PredictionLedger(config.prediction_ledger_path)
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    for day in range(30):
        for symbol in DEFAULT_SYMBOLS:
            ledger.append(
                prediction(symbol=symbol, asof=(start + pd.Timedelta(days=day)).isoformat()),
                source_checksums={"bars": "a" * 64},
                input_window_hash="b" * 64,
                code_version="code-v1",
            )

    status = ProdPreflight(config)._shadow_gate_status()

    assert status["ok"] is True
    assert status["window_days"] == 30
