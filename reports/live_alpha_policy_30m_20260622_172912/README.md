# Live Alpha Policy 30m Forward Run

This is a 30-minute live paper run of the frozen strict alpha protocol using the
fast validation engine.

## Command

```powershell
.\.venv\Scripts\python.exe examples\live_paper_alpha_policy.py `
  --config-json configs\strict_alpha_policy_20260622.json `
  --duration-minutes 30 `
  --poll-seconds 60 `
  --fetch-workers 12 `
  --candidate-log-top-n 10 `
  --validation-engine fast `
  --output-dir outputs\live_paper_alpha_policy_30m_20260622_172912
```

Candidate-log analysis:

```powershell
.\.venv\Scripts\python.exe examples\analyze_live_candidate_log.py `
  --run-dir outputs\live_paper_alpha_policy_30m_20260622_172912 `
  --output-dir outputs\live_alpha_candidate_analysis_30m_20260622_172912
```

## Result

| Metric | Value |
| --- | ---: |
| Start | 2026-06-22 17:29:25 |
| End | 2026-06-22 18:00:09 |
| Live rows | 26 |
| Candidate polls | 26 |
| Candidate rows | 520 |
| Initial equity | 10,000.00 RUB |
| Final equity | 9,814.94 RUB |
| Final return | -1.85% |
| Best marked return | 0.48% |
| Worst marked return | -1.98% |
| Rebalance actions | 2 |
| Hold actions | 20 |
| Cash actions | 4 |
| Average fetch seconds | 7.87 |
| Average decision seconds | 1.05 |
| Decision cache hit rate | 84.62% |

## Timeline

| Time | Candle | Action | Selected | Position | Equity | Return |
| --- | --- | --- | --- | --- | ---: | ---: |
| 17:29:25 | 17:00 | rebalance | `voladj_mom_96`, short basket 1.5x | `CNRU,RNFT,SMLT` | 9,985.00 | -0.15% |
| 17:39:02 | 17:10 | hold | same | `CNRU,RNFT,SMLT` | 10,047.80 | 0.48% |
| 17:44:52 | 17:20 | hold | short basket 1.0x candidate | still old 1.5x basket | 9,979.27 | -0.21% |
| 17:51:14 | 17:20 | hold | same | still old 1.5x basket | 9,801.58 | -1.98% |
| 17:55:08 | 17:30 | rebalance | cash gate | cash | 9,814.94 | -1.85% |
| 18:00:09 | 17:30 | cash | cash gate | cash | 9,814.94 | -1.85% |

## Interpretation

This run is a negative forward result. The frozen strict alpha protocol did not
move toward the 10% target in this live slice; it lost 185.06 RUB.

The main diagnostic finding is live execution inertia. At the 17:20 candle, the
selector still liked the same alpha family but reduced the preferred gross from
1.5x to 1.0x. The runner held the older 1.5x basket because the frozen policy's
`rebalance_every=24` rule had not elapsed. By 17:30 the strict gate moved to
cash, and the runner closed the paper book, but most of the loss had already
appeared.

The next improvement should keep the offline frozen protocol intact while adding
a live risk overlay:

1. Rebalance when target weights materially change on a new candle.
2. Close the paper book on a configurable max session loss.
3. Keep logging both the raw frozen candidate and the live risk-adjusted action.

This report is not a production proof; it is a useful forward failure that points
to the next engineering fix.
