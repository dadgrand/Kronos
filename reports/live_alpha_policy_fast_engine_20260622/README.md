# Live Alpha Policy Fast Engine

This report verifies the optimized validation engine for the live strict-alpha
paper runner.

## What Changed

- The live runner now supports `--validation-engine fast`.
- Fast mode skips full per-bar DataFrame construction during validation search.
- Fast mode also skips `score_confidence` generation because the frozen alpha
  protocol uses `confidence_min = 0`.
- Full mode remains available with `--validation-engine full`.

## Compatibility Check

On a fixed historical validation slice from the local dataset, fast and full
selection matched exactly:

| Engine | Elapsed | Passed gates | Selected | Validation return | Worst segment | Selection score |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| fast | 5.497s | 3 | `mom_96 short_only k=3 gross=1.0` | 19.36% | 1.50% | 17.478 |
| full | 7.134s | 3 | `mom_96 short_only k=3 gross=1.0` | 19.36% | 1.50% | 17.478 |

## Live Smoke Command

```powershell
.\.venv\Scripts\python.exe examples\live_paper_alpha_policy.py `
  --config-json configs\strict_alpha_policy_20260622.json `
  --once `
  --fetch-workers 12 `
  --candidate-log-top-n 10 `
  --validation-engine fast `
  --output-dir outputs\live_paper_alpha_policy_fast_engine_once_20260622
```

## Live Smoke Result

| Metric | Value |
| --- | ---: |
| Latest candle | 2026-06-22 17:00:00 |
| Validation engine | `fast` |
| Fetch seconds | 5.08 |
| Decision seconds | 6.98 |
| Total poll seconds | 16.98 |
| Candidates | 216 |
| Candidates passing gate | 3 |
| Selected alpha | `voladj_mom_96` |
| Selected mode | `short_only` |
| Selected symbols | `CNRU,RNFT,SMLT` |
| Gross | 1.5 |
| Validation return | 28.38% |
| Worst validation segment | 0.83% |
| 20 bps validation stress | 8.77% |
| Final equity after immediate close | 9,970.00 RUB |
| Final return after immediate close | -0.30% |

The negative one-shot return is the modeled two-sided commission from opening and
immediately closing a 1.5x paper short basket. It is not a forward-performance
claim.

## Decision

The live runner is now fast enough for 10-minute candle decisions. The next
meaningful gate remains a longer frozen forward run without changing
`configs/strict_alpha_policy_20260622.json`.
