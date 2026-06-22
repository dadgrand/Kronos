# Walk-Forward Alpha Policy Lab

Rolling alpha and execution-policy selection with a strict cash fallback.

## Protocol

- Dataset: `datasets\moex_intraday_6f8f836cc811`
- Windows: 19
- Trade windows: 2
- Validation days: 30
- Test days: 7
- Signal delay bars: 2
- Candidate count per window: 216
- Cost: 10.00 bps
- Min validation return: 5.0
- Min worst validation segment: 0.0
- Max validation drawdown: 8.0

## Compounded Test Result

- Initial cash: 10000.00
- Final equity: 10032.47
- PnL: 32.47
- Return: 0.32%
- Stress return at 20.00 bps: -3.03%
- Window endpoint max drawdown: -0.33%
- Worst per-window max drawdown: -3.08%
- Trade-window win rate: 50.00%
- Worst traded window: -0.33%
- Target hit (10.0%): False

This is a research backtest, not financial advice or a live trading recommendation.
