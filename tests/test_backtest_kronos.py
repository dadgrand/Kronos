import pandas as pd
import pytest

from examples.run_backtest_kronos import KronosBacktester


def make_prediction_frame(targets, predicted_close, asofs=None, executions=None, horizon="1D"):
    targets = pd.to_datetime(targets)
    if asofs is None:
        asofs = targets - pd.Timedelta(days=1)
    asofs = pd.to_datetime(asofs)
    if executions is None:
        executions = targets
    executions = pd.to_datetime(executions)
    pred = pd.DataFrame(
        {
            "symbol": "TEST",
            "prediction_asof": asofs,
            "execution_timestamp": executions,
            "target_timestamp": targets,
            "features_cutoff": asofs,
            "horizon": horizon,
            "model_version": "unit-test",
            "model_hash": "abc123",
            "predicted_close": predicted_close,
        }
    )
    return pred.set_index("execution_timestamp", drop=False)


def test_backtest_rejects_future_only_predictions():
    hist = pd.DataFrame(
        {
            "open": [100.0, 101.0],
            "close": [100.0, 101.0],
        },
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
    )
    pred = make_prediction_frame(["2024-01-03"], [110.0])

    backtester = KronosBacktester(".", ".")

    with pytest.raises(ValueError, match="overlapping actual and prediction dates"):
        backtester.calculate_trading_signals(hist, pred)


def test_backtest_uses_realized_prices_and_flat_signals_without_ffill():
    hist = pd.DataFrame(
        {
            "open": [100.0, 102.0, 104.0, 106.0],
            "close": [100.0, 102.0, 104.0, 106.0],
        },
        index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]),
    )
    pred = make_prediction_frame(hist.index[1:], [110.0, 90.0, 120.0])

    backtester = KronosBacktester(
        ".",
        ".",
        initial_capital=100000,
        commission_rate=0.0,
        slippage_rate=0.0,
    )

    signals = backtester.calculate_trading_signals(hist, pred, threshold=0.02)

    assert signals["position"].tolist() == [1, 0, 1]

    results, trades = backtester.run_backtest(signals)

    assert results["price"].tolist() == [102.0, 104.0, 106.0]
    assert [trade["action"] for trade in trades] == ["BUY", "SELL", "BUY"]
    assert trades[1]["price"] == 104.0
    assert results["capital"].iloc[-1] > 0


def test_backtest_uses_prediction_asof_close_for_sparse_predictions():
    hist = pd.DataFrame(
        {
            "open": [100.0, 50.0, 50.0],
            "close": [100.0, 50.0, 50.0],
        },
        index=pd.to_datetime(["2024-01-01", "2024-01-03", "2024-01-04"]),
    )
    pred = make_prediction_frame(
        ["2024-01-04"],
        [110.0],
        asofs=pd.to_datetime(["2024-01-01"]),
        horizon="3D",
    )

    backtester = KronosBacktester(".", ".")
    signals = backtester.calculate_trading_signals(hist, pred, threshold=0.5)

    assert signals["reference_close"].tolist() == [100.0]
    assert signals["pred_return"].tolist() == pytest.approx([0.1])
    assert signals["position"].tolist() == [0]


def test_backtest_rejects_missing_prediction_asof_reference_close():
    hist = pd.DataFrame(
        {
            "open": [100.0],
            "close": [100.0],
        },
        index=pd.to_datetime(["2024-01-02"]),
    )
    pred = make_prediction_frame(
        ["2024-01-02"],
        [110.0],
        asofs=pd.to_datetime(["2024-01-01"]),
    )

    backtester = KronosBacktester(".", ".")

    with pytest.raises(ValueError, match="reference close"):
        backtester.calculate_trading_signals(hist, pred)


def test_backtest_rejects_prediction_available_at_or_after_target():
    hist = pd.DataFrame(
        {
            "open": [100.0, 101.0],
            "close": [100.0, 101.0],
        },
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
    )
    pred = make_prediction_frame(
        hist.index,
        [100.0, 110.0],
        asofs=pd.to_datetime(["2023-12-31", "2024-01-02"]),
    )

    backtester = KronosBacktester(".", ".")

    with pytest.raises(ValueError, match="prediction_asof"):
        backtester.calculate_trading_signals(hist, pred)


def test_backtest_rejects_ambiguous_execution_timestamp():
    hist = pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0],
            "close": [100.0, 101.0, 102.0],
        },
        index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
    )
    pred = make_prediction_frame(
        hist.index[:2],
        [100.0, 110.0],
        executions=pd.to_datetime(["2024-01-01", "2024-01-03"]),
    )

    backtester = KronosBacktester(".", ".")

    with pytest.raises(ValueError, match="execution_timestamp"):
        backtester.calculate_trading_signals(hist, pred)


def test_backtest_rejects_placeholder_model_hash_and_bad_horizon():
    hist = pd.DataFrame(
        {
            "open": [100.0, 101.0],
            "close": [100.0, 101.0],
        },
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
    )
    pred = make_prediction_frame(hist.index, [100.0, 110.0], horizon="not-a-horizon")
    pred["model_hash"] = "unknown"

    backtester = KronosBacktester(".", ".")

    with pytest.raises(ValueError, match="model_hash|horizon"):
        backtester.calculate_trading_signals(hist, pred)


def test_backtest_keeps_cash_non_negative_with_costs():
    hist = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0],
            "close": [100.0, 100.0, 100.0],
        },
        index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
    )
    pred = make_prediction_frame(hist.index[1:], [110.0, 110.0])

    backtester = KronosBacktester(
        ".",
        ".",
        initial_capital=10000,
        max_position_fraction=1.0,
        min_cash_fraction=0.0,
        commission_rate=0.01,
        slippage_rate=0.01,
    )

    signals = backtester.calculate_trading_signals(hist, pred, threshold=0.02)
    results, _ = backtester.run_backtest(signals)

    assert (results["cash"] >= -1e-9).all()


def test_short_mode_requires_explicit_margin_model():
    with pytest.raises(ValueError, match="Short backtests require"):
        KronosBacktester(".", ".", allow_short=True)


def test_backtest_replaces_partial_orders_without_exceeding_target():
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    combined = pd.DataFrame(
        {
            "symbol": "TEST",
            "open": 100.0,
            "actual": 100.0,
            "volume": 10.0,
            "position": 1,
        },
        index=dates,
    )
    backtester = KronosBacktester(
        ".",
        ".",
        initial_capital=10000,
        commission_rate=0.0,
        slippage_rate=0.0,
        max_position_fraction=0.25,
        max_participation_rate=0.1,
    )

    results, _ = backtester.run_backtest(combined)

    assert results["position"].max() <= 25
