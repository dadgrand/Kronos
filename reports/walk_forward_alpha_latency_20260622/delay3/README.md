# Walk-Forward Alpha Policy Lab

Rolling alpha and execution-policy selection with a strict cash fallback.

## Protocol

- Dataset: `datasets\moex_intraday_6f8f836cc811`
- Windows: 19
- Trade windows: 3
- Validation days: 30
- Test days: 7
- Signal delay bars: 3
- Candidate count per window: 216
- Cost: 10.00 bps
- Min validation return: 5.0
- Min worst validation segment: 0.0
- Max validation drawdown: 8.0

## Compounded Test Result

- Initial cash: 10000.00
- Final equity: 10959.38
- PnL: 959.38
- Return: 9.59%
- Stress return at 20.00 bps: 1.36%
- Window endpoint max drawdown: -2.32%
- Worst per-window max drawdown: -9.45%
- Trade-window win rate: 66.67%
- Worst traded window: -2.32%
- Target hit (10.0%): False

This is a research backtest, not financial advice or a live trading recommendation.
