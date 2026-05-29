from trading.evaluation import WalkForwardReport
import pytest


def test_walk_forward_report_compares_strategy_to_baseline():
    report = WalkForwardReport(
        equity_history=[
            {"timestamp": "2024-01-01", "equity": 10000},
            {"timestamp": "2024-01-02", "equity": 10100},
            {"timestamp": "2024-01-03", "equity": 10200},
        ],
        price_history=[
            {"timestamp": "2024-01-01", "close": 100},
            {"timestamp": "2024-01-02", "close": 99},
            {"timestamp": "2024-01-03", "close": 101},
        ],
    ).compute()

    assert report["strategy_return"] > report["baseline_return"]
    assert report["observations"] == 3


def test_walk_forward_report_requires_timestamp_overlap():
    try:
        WalkForwardReport(
            equity_history=[{"timestamp": "2024-01-01", "equity": 10000}],
            price_history=[{"timestamp": "2024-02-01", "close": 100}],
        ).compute()
    except ValueError as exc:
        assert "overlap" in str(exc)
    else:
        raise AssertionError("Expected ValueError for non-overlapping histories")


def test_walk_forward_report_sorts_aligned_timestamps_before_return_math():
    report = WalkForwardReport(
        equity_history=[
            {"timestamp": "2024-01-03", "equity": 10200},
            {"timestamp": "2024-01-01", "equity": 10000},
            {"timestamp": "2024-01-02", "equity": 10100},
        ],
        price_history=[
            {"timestamp": "2024-01-02", "close": 99},
            {"timestamp": "2024-01-03", "close": 101},
            {"timestamp": "2024-01-01", "close": 100},
        ],
    ).compute()

    assert report["strategy_return"] == pytest.approx(0.02)
    assert report["baseline_return"] == pytest.approx(0.01)


def test_walk_forward_report_rejects_positional_fallback_without_timestamps():
    try:
        WalkForwardReport(
            equity_history=[{"equity": 10000}, {"equity": 10100}],
            price_history=[{"close": 100}, {"close": 101}],
        ).compute()
    except ValueError as exc:
        assert "timestamp" in str(exc)
    else:
        raise AssertionError("Expected ValueError when timestamps are missing")
