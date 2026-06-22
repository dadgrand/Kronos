# Walk-Forward Alpha Policy Lab

Rolling alpha and execution-policy selection with a strict cash fallback.

## Protocol

- Dataset: `datasets\moex_intraday_6f8f836cc811`
- Windows: 1
- Trade windows: 0
- Validation days: 30
- Nested gate days: 7
- Test days: 31
- Signal delay bars: 3
- Engine: fast
- Decision mode: balanced
- Decision score threshold: 0.0
- Candidate count per window: 234
- Cost: 10.00 bps
- Selection min return: 5.0
- Selection min worst segment: 0.0
- Gate min stress return: 0.0
- Gate min stress edge: 0.0
- Max May-like regime risk: 0.7
- Gate veto windows: 1
- Diplom risk enabled: True
- Diplom changed rebalance decisions: 0
- Diplom available/missing/stale rebalances: 0/89/17
- Avoided loss: 24.27 pct-points
- Missed profit: 0.00 pct-points

## Compounded Test Result

- Initial cash: 10000.00
- Final equity: 10000.00
- PnL: 0.00
- Return: 0.00%
- Stress return at 20.00 bps: 0.00%
- Window endpoint max drawdown: 0.00%
- Worst per-window max drawdown: 0.00%
- Trade-window win rate: 0.00%
- Worst traded window: 0.00%
- Executed test month win rate: 0.00%
- Worst executed test month: 0.0
- Target hit (10.0%): False

The nested gate is a veto only. If the selection winner fails stress edge, month stability, or May-like regime risk, the window is recorded as cash and the candidate's raw test result is kept under `raw_test_*` fields for audit.

This is a research backtest, not financial advice or a live trading recommendation.
