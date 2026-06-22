# Live Latency Candidate Smoke

This one-shot run verifies that the live alpha runner can load and apply
`configs/strict_alpha_policy_latency_delay3_dd6_20260622.json`.

## Command

```powershell
.\.venv\Scripts\python.exe examples\live_paper_alpha_policy.py `
  --config-json configs\strict_alpha_policy_latency_delay3_dd6_20260622.json `
  --once `
  --fetch-workers 12 `
  --candidate-log-top-n 10 `
  --validation-engine fast `
  --output-dir outputs\live_alpha_latency_candidate_once_20260622
```

## Result

| Metric | Value |
| --- | ---: |
| Wall time | 2026-06-22 19:18:21 |
| Signal delay | 3 bars |
| Signal source | `candle` |
| Latest candle | 2026-06-22 18:50:00 |
| Candle close | 2026-06-22 19:00:00 |
| Signal age after close | 18.36 min |
| Raw marketdata prices | 50 / 50 |
| Fresh marketdata prices | 0 / 50 |
| Stale marketdata prices | 50 / 50 |
| Action | `stale_signal_cash` |
| Initial equity | 10,000.00 RUB |
| Final equity | 10,000.00 RUB |
| PnL | 0.00 RUB |
| Return | 0.00% |

## Interpretation

The live runner now honors `signal_delay_bars=3` and the live freshness overlay
from the latency candidate config. On the current public MOEX feed, all
marketdata prices were stale under the 2-minute rule and the candle fallback was
older than the 10-minute post-close limit, so the paper book stayed in cash.

This verifies config plumbing and safety behavior. It is not a forward-profit
validation of the `+10.90%` research backtest.

