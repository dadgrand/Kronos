# Walk-Forward Alpha Policy Lab

Rolling alpha and execution-policy selection with a strict cash fallback.

## Protocol

- Dataset: `datasets\moex_intraday_6f8f836cc811`
- Windows: 19
- Trade windows: 2
- Validation days: 30
- Test days: 7
- Signal delay bars: 0
- Candidate count per window: 216
- Cost: 10.00 bps
- Min validation return: 5.0
- Min worst validation segment: 0.0
- Max validation drawdown: 8.0

## Compounded Test Result

- Initial cash: 10000.00
- Final equity: 11849.31
- PnL: 1849.31
- Return: 18.49%
- Stress return at 20.00 bps: 10.13%
- Window endpoint max drawdown: -0.50%
- Worst per-window max drawdown: -10.80%
- Trade-window win rate: 50.00%
- Worst traded window: -0.50%
- Target hit (10.0%): True

This is a research backtest, not financial advice or a live trading recommendation.
