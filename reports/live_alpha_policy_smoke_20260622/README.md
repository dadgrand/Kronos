# Live Alpha Policy Smoke

This smoke run verifies that the frozen strict alpha protocol can drive a live
paper decision using current MOEX candles and marketdata.

## Command

```powershell
.\.venv\Scripts\python.exe examples\live_paper_alpha_policy.py `
  --config-json configs\strict_alpha_policy_20260622.json `
  --once `
  --candidate-log-top-n 10 `
  --output-dir outputs\live_paper_alpha_policy_once_20260622
```

Candidate-log analysis:

```powershell
.\.venv\Scripts\python.exe examples\analyze_live_candidate_log.py `
  --run-dir outputs\live_paper_alpha_policy_once_20260622 `
  --output-dir outputs\live_alpha_candidate_analysis_once_20260622
```

## Result

| Metric | Value |
| --- | ---: |
| Latest candle | 2026-06-22 16:30:00 |
| Validation window start | 2026-05-23 16:30:00 |
| Validation window end | 2026-06-22 16:30:00 |
| Candidate count | 216 |
| Candidates passing strict gate | 2 |
| Selected alpha | `mom_48` |
| Selected mode | `short_only` |
| Selected symbol | `SPBE` |
| Gross | 0.5 |
| Validation return | 20.94% |
| Worst validation segment | 1.26% |
| 20 bps validation stress return | 12.92% |
| Final equity after immediate smoke close | 9,990.00 RUB |
| Final return after immediate smoke close | -0.10% |

The `-0.10%` result is the expected two-sided commission cost from opening and
immediately closing a 0.5x paper short in a one-shot smoke test. It is not a
strategy forward-performance measurement.

## Candidate Log Check

- Candidate polls: 1
- Candidate rows: 20
- Price coverage: 100%
- Tradable rate: 100%
- Selected candidate rows: 1
- One-step opportunity rows: 0

## Interpretation

This is meaningful progress because the strict alpha benchmark is no longer only
an offline report. The same frozen protocol now produces a live paper decision,
logs its candidate universe, and records the validation evidence that caused the
trade.

The next real gate is a multi-poll forward paper run without changing
`configs/strict_alpha_policy_20260622.json`.
