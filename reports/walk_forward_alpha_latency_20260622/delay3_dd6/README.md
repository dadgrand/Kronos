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
- Max validation drawdown: 6.0

## Compounded Test Result

- Initial cash: 10000.00
- Final equity: 11089.55
- PnL: 1089.55
- Return: 10.90%
- Stress return at 20.00 bps: 3.93%
- Window endpoint max drawdown: -1.16%
- Worst per-window max drawdown: -9.45%
- Trade-window win rate: 66.67%
- Worst traded window: -1.16%
- Target hit (10.0%): True

This is a research backtest, not financial advice or a live trading recommendation.

Important caveat: this is a post-hoc latency-aware candidate from the latency
sweep, not a forward-live proof. The 6% validation drawdown gate must be frozen
before any fresh paper run that is used to judge whether this result generalizes.
