# Walk-Forward Alpha Policy Lab

Rolling alpha and execution-policy selection with a strict cash fallback.

## Protocol

- Dataset: `datasets\moex_intraday_6f8f836cc811`
- Windows: 19
- Trade windows: 1
- Validation days: 30
- Nested gate days: 7
- Test days: 7
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
- Gate veto windows: 18
- Diplom risk enabled: False
- Diplom changed rebalance decisions: 0
- Diplom available/missing/stale rebalances: 0/0/0
- Avoided loss: 64.54 pct-points
- Missed profit: 11.15 pct-points

## Compounded Test Result

- Initial cash: 10000.00
- Final equity: 9580.25
- PnL: -419.75
- Return: -4.20%
- Stress return at 20.00 bps: -9.12%
- Window endpoint max drawdown: -4.20%
- Worst per-window max drawdown: -12.31%
- Trade-window win rate: 0.00%
- Worst traded window: -4.20%
- Executed test month win rate: 0.00%
- Worst executed test month: -4.197525005726
- Target hit (10.0%): False

The nested gate is a veto only. If the selection winner fails stress edge, month stability, or May-like regime risk, the window is recorded as cash and the candidate's raw test result is kept under `raw_test_*` fields for audit.

This is a research backtest, not financial advice or a live trading recommendation.
