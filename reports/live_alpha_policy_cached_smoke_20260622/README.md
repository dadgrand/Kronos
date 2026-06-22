# Live Alpha Policy Cached Multi-Poll Smoke

This run verifies the live strict-alpha runner after adding bounded parallel
MOEX candle fetching and per-candle decision caching.

## Command

```powershell
.\.venv\Scripts\python.exe examples\live_paper_alpha_policy.py `
  --config-json configs\strict_alpha_policy_20260622.json `
  --duration-minutes 2.2 `
  --poll-seconds 5 `
  --fetch-workers 12 `
  --candidate-log-top-n 10 `
  --output-dir outputs\live_paper_alpha_policy_cached_smoke_20260622
```

Candidate-log analysis:

```powershell
.\.venv\Scripts\python.exe examples\analyze_live_candidate_log.py `
  --run-dir outputs\live_paper_alpha_policy_cached_smoke_20260622 `
  --output-dir outputs\live_alpha_candidate_analysis_cached_smoke_20260622
```

## Result

| Event | Candle | Action | Selected alpha | Target | Equity | Return |
| --- | --- | --- | --- | --- | ---: | ---: |
| 1 | 2026-06-22 16:40 | rebalance | `mom_48` | short `SPBE` 0.5x | 9,995.00 | -0.050% |
| 2 | 2026-06-22 16:50 | hold | `voladj_mom_96` | short basket candidate | 10,021.16 | 0.212% |
| final | 2026-06-22 16:50 | close | n/a | cash | 10,016.19 | 0.162% |

The run ended with `+16.19 RUB` after modeled commission and final close.

## Timing

| Poll | Fetch seconds | Decision seconds | Total poll seconds | Cache hit |
| ---: | ---: | ---: | ---: | --- |
| 1 | 18.01 | 71.83 | 93.85 | no |
| 2 | 5.01 | 73.28 | 82.63 | no |

Both polls landed on different 10-minute candles, so the decision cache did not
activate. The fetch phase is now fast enough; the remaining bottleneck is the
validation search. This is acceptable for 10-minute candle decisions but should
not be treated as a sub-minute trading loop.

## Candidate Log Check

- Candidate polls: 2
- Candidate rows: 40
- Price coverage: 100%
- Tradable rate: 100%
- Selected candidate rows: 4
- Final return after close: 0.1619%

## Interpretation

This is not a 10% proof by itself; it is a live-forward plumbing check. The
important improvement is that the frozen strict alpha protocol can now run across
multiple live polls, log candidates, preserve positions, mark PnL, and close the
paper book cleanly.
