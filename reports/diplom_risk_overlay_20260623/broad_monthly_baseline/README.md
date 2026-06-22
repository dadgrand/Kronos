# Walk-Forward Alpha Policy Lab

Rolling alpha and execution-policy selection with a strict cash fallback.

## Protocol

- Dataset: `datasets\moex_intraday_6f8f836cc811`
- Windows: 42
- Trade windows: 1
- Validation days: 120
- Nested gate days: 30
- Test days: 30
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
- Gate veto windows: 41
- Diplom risk enabled: False
- Diplom changed rebalance decisions: 0
- Diplom available/missing/stale rebalances: 0/0/0
- Avoided loss: 322.55 pct-points
- Missed profit: 62.09 pct-points

## Compounded Test Result

- Initial cash: 10000.00
- Final equity: 9950.46
- PnL: -49.54
- Return: -0.50%
- Stress return at 20.00 bps: -5.40%
- Window endpoint max drawdown: -0.50%
- Worst per-window max drawdown: -5.17%
- Trade-window win rate: 0.00%
- Worst traded window: -0.50%
- Executed test month win rate: 0.00%
- Worst executed test month: -0.49542126463127945
- Target hit (10.0%): False

The nested gate is a veto only. If the selection winner fails stress edge, month stability, or May-like regime risk, the window is recorded as cash and the candidate's raw test result is kept under `raw_test_*` fields for audit.

This is a research backtest, not financial advice or a live trading recommendation.
