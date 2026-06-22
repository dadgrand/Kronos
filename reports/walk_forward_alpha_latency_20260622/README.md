# Walk-Forward Alpha Latency Study

This report re-evaluates the strict alpha policy under delayed signal execution.
The motivation is the live paper finding that public MOEX ISS candles and
marketdata snapshots were delayed enough to make the original intraday live test
unsafe.

## Protocol

- Base config: `configs/strict_alpha_policy_20260622.json`
- Dataset: `datasets/moex_intraday_6f8f836cc811`
- Interval: 10 minutes
- Windows: 19
- Candidate count per window: 216
- Costs: 10 bps primary, 20 bps stress
- Signal delay: alpha scores and regime features are shifted by N bars before
  validation selection and test execution.
- Execution assumption: after the N-bar signal delay, the signal is available
  before the open of the evaluated bar, and PnL is computed on that bar's
  open-to-close return.

## Commands

```powershell
.\.venv\Scripts\python.exe examples\walk_forward_alpha_policy_lab.py `
  --config-json configs\strict_alpha_policy_20260622.json `
  --signal-delay-bars 3 `
  --max-validation-drawdown-pct 6.0 `
  --output-dir outputs\walk_forward_alpha_latency_delay3_dd6_20260622
```

## Summary

| Run | Delay | Validation DD gate | Trade windows | Return 10 bps | Return 20 bps | Worst traded window | Target hit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `delay0` | 0 bars | 8.0% | 2 | 18.49% | 10.13% | -0.50% | true |
| `delay1` | 1 bar | 8.0% | 1 | 0.89% | -1.61% | 0.89% | false |
| `delay2` | 2 bars | 8.0% | 2 | 0.32% | -3.03% | -0.33% | false |
| `delay3` | 3 bars | 8.0% | 3 | 9.59% | 1.36% | -2.32% | false |
| `delay3_dd6` | 3 bars | 6.0% | 3 | 10.90% | 3.93% | -1.16% | true |

## Selected Windows For `delay3_dd6`

| Window | Alpha | Mode | k | Gross | Validation return | Validation max DD | Test return | Stress test return |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | `rev_96` | `short_only` | 1 | 0.5 | 7.49% | -5.03% | 2.57% | 0.69% |
| 11 | `rev_96` | `short_only` | 1 | 1.0 | 19.32% | -5.49% | 9.39% | 5.82% |
| 15 | `mom_96` | `short_only` | 3 | 0.5 | 6.65% | -3.57% | -1.16% | -2.45% |

## Interpretation

The old no-delay benchmark was fragile for the observed live environment:
shifting the signal by one or two 10-minute bars reduced the result from
`+18.49%` to `+0.89%` and `+0.32%`. That explains why the previous 30-minute live
test failed despite a strong offline headline.

The `delay3_dd6` run is the first latency-aware candidate to cross the 10% target
in this study. It delays signals by 30 minutes and tightens the validation max
drawdown gate from 8% to 6%, reaching `+10.90%` at 10 bps.

This is not yet a production proof. The stricter 6% drawdown gate was introduced
after inspecting latency results, so it must be treated as an exploratory
candidate requiring a fresh frozen replay and forward paper run. The stress-cost
result is only `+3.93%`, so transaction-cost robustness is materially weaker
than the original no-delay headline.

## Artifacts

- `delay0/`
- `delay1/`
- `delay2/`
- `delay3/`
- `delay3_dd6/`
