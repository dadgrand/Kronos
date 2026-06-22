# Frozen Latency Candidate Replay

This report verifies that the latency-aware candidate protocol can be reproduced
from `configs/strict_alpha_policy_latency_delay3_dd6_20260622.json`.

## Command

```powershell
.\.venv\Scripts\python.exe examples\walk_forward_alpha_policy_lab.py `
  --config-json configs\strict_alpha_policy_latency_delay3_dd6_20260622.json `
  --output-dir outputs\frozen_alpha_latency_delay3_dd6_replay_20260622
```

## Result

| Metric | Value |
| --- | ---: |
| Signal delay | 3 bars |
| Validation max drawdown gate | 6.0% |
| Windows | 19 |
| Trade windows | 3 |
| Initial equity | 10,000.00 RUB |
| Final equity | 11,089.55 RUB |
| PnL | 1,089.55 RUB |
| Return at 10 bps | 10.90% |
| Return at 20 bps stress | 3.93% |
| Worst traded window | -1.16% |
| Target hit | true |

## Caveat

This replay freezes and reproduces an exploratory latency-aware candidate. It is
not yet a forward-live proof. The 6% drawdown gate was introduced after the
latency sweep, so the next validation step must be a fresh forward paper run
without changing this config.

