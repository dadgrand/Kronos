# Frozen Alpha Policy Replay

This report verifies that the strict alpha benchmark can be reproduced from a
frozen JSON protocol, not just from ad hoc command-line defaults.

## Frozen Config

- Config: `configs/strict_alpha_policy_20260622.json`
- Script: `examples/walk_forward_alpha_policy_lab.py`
- Replay output: `outputs/frozen_alpha_policy_replay_20260622`

## Command

```powershell
.\.venv\Scripts\python.exe examples\walk_forward_alpha_policy_lab.py `
  --config-json configs\strict_alpha_policy_20260622.json `
  --output-dir outputs\frozen_alpha_policy_replay_20260622
```

## Replay Result

| Metric | Value |
| --- | ---: |
| Windows | 19 |
| Trade windows | 2 |
| Final equity at 10 bps | 11,849.31 RUB |
| Return at 10 bps | 18.49% |
| Final equity at 20 bps stress | 11,012.81 RUB |
| Return at 20 bps stress | 10.13% |
| Worst traded window | -0.50% |
| Target hit at 10 bps | yes |
| Target hit at 20 bps stress | yes |

The replay matches `reports/walk_forward_alpha_policy_strict_20260622` on the
key compounded and per-window results.

## Why This Matters

The previous report established a promising sparse walk-forward research result.
This replay makes the next forward step cleaner: future paper runs can point to a
single frozen config and override only the new date range/output path. That
reduces the chance of quietly changing gates after seeing new data.

This still does not make the strategy production-ready. It only turns the current
research benchmark into a pre-registered protocol that can be tested forward.
