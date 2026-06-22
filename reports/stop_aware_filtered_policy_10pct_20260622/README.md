# Stop-Aware Filtered Policy, 10% Target Check

This report evaluates a lower-gross execution-aware variant of the frozen MOEX neural score ensemble.

## Protocol

- Dataset: `datasets/moex_intraday_6f8f836cc811`
- Interval: 10 minute MOEX candles
- Training period used by the score models: before 2026-06-01
- Validation period: 2026-06-01 through 2026-06-13
- Test period: after 2026-06-13
- Initial cash: 10,000 RUB
- Trading cost: 10 bps per unit turnover
- Policy search: validation only
- Test: primary 10 bps result evaluated after selecting the policy
- Target: at least 10% return on the locked test split

The new lab supports bar-close stop-loss, take-profit, and persistent filter-fail exits. For this run, the selected validation policy did not use those exits; the improvement came from lower gross exposure and lower turnover versus the old 4x research policy.

## Command

```powershell
.\.venv\Scripts\python.exe examples\stop_aware_filtered_policy_lab.py `
  --scores-path `
    outputs\neural_policy_lab_h12_seed20260622_20260622_132851\scores.npy `
    outputs\neural_policy_lab_h12_seed20260623_20260622_133133\scores.npy `
    outputs\neural_policy_lab_h12_seed20260624_20260622_133615\scores.npy `
  --train-end 2026-06-01 `
  --validation-end 2026-06-13 `
  --max-gross 1.5 `
  --target-return-pct 10 `
  --cost-bps 10 `
  --rebalance-grid 24 `
  --stop-loss-grid none `
  --take-profit-grid none `
  --filter-fail-exit-grid none,8,13 `
  --output-dir outputs\stop_aware_filtered_policy_20260622_10pct
```

## Selected Policy

```json
{
  "mode": "short_only",
  "k": 1,
  "gross": 1.5,
  "rebalance_every": 24,
  "confidence_min": 0.035875335335731506,
  "market_mom_24_max": 0.002567113547411282,
  "market_mom_96_max": 0.000038395675483115644,
  "stop_loss_pct": null,
  "take_profit_pct": null,
  "filter_fail_exit_bars": null
}
```

The same policy is also saved as `selected_policy.json` for live-paper runs.

## Live Paper Command

```powershell
.\.venv\Scripts\python.exe examples\live_paper_moex_policy_v2.py `
  --policy-json reports\stop_aware_filtered_policy_10pct_20260622\selected_policy.json `
  --duration-minutes 30 `
  --poll-seconds 60 `
  --initial-cash 10000 `
  --cost-bps 10 `
  --device cuda:0
```

Smoke-test output:

- Output dir: `outputs/live_paper_policy_candidate_log_smoke_20260622`
- Loaded mode: `short_only`
- Loaded gross: `1.5` from `selected_policy.json`
- Loaded confidence gate: `0.035875335335731506`
- Loaded stop/take: `null` / `null`
- Candidate log rows: `50`
- Selected smoke candidate: `PIKK`, short-rank 1
- One-shot result: 9,970.00 RUB after immediate final close, which is the expected two-sided 1.5x commission cost for a smoke test.

## Results

| Split | Final equity | PnL | Return | Max drawdown | Active rate | Avg turnover/bar |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | 12,440.09 | 2,440.09 | 24.40% | -5.49% | 57.62% | 0.0693 |
| Test | 11,131.61 | 1,131.61 | 11.32% | -7.14% | 57.42% | 0.0702 |

The 10% target is reached on this research test split: 10,000 RUB becomes 11,131.61 RUB.

## Segment Stability

The selected policy was also split into five chronological segments per split. This is a diagnostic only; it is not used for selection.

| Split | Segment | Return | Max drawdown | Active rate | Avg turnover/bar |
| --- | ---: | ---: | ---: | ---: | ---: |
| Validation | 1 | 2.45% | -2.35% | 22.12% | 0.0276 |
| Validation | 2 | 1.00% | -3.14% | 78.34% | 0.1037 |
| Validation | 3 | -2.78% | -3.14% | 88.94% | 0.0968 |
| Validation | 4 | 27.39% | -2.87% | 54.17% | 0.0625 |
| Validation | 5 | -2.92% | -4.54% | 44.44% | 0.0556 |
| Test | 1 | -2.35% | -3.01% | 38.71% | 0.0484 |
| Test | 2 | 0.00% | 0.00% | 0.00% | 0.0000 |
| Test | 3 | 7.24% | -2.45% | 67.74% | 0.0847 |
| Test | 4 | 7.46% | -3.26% | 100.00% | 0.1210 |
| Test | 5 | -1.08% | -7.14% | 80.65% | 0.0968 |

Worst segment returns: validation -2.92%, test -2.35%. The total test return is positive, but it is not uniformly distributed; most of the gain comes from the middle of the test split.

## Post-Selection Cost Sensitivity

| Cost bps | Test return |
| ---: | ---: |
| 5 | 13.77% |
| 10 | 11.32% |
| 20 | 6.55% |
| 30 | 1.98% |

The result is still cost-sensitive. It clears 10% at 10 bps, but not at 20 bps. This sensitivity table is a diagnostic after selecting the policy; it is not used as a selection input.

## Interpretation

This is a meaningful improvement over the live 30 minute slice because it targets a portfolio-level test result, not a single paper-trade outcome. It is also more realistic than the old 4x research policy because gross exposure is capped at 1.5x and average turnover is lower.

It is not yet a production proof. The policy remains short-only and one-symbol concentrated, borrow constraints are not modeled, and the test period is short. The next gating step should be a forward live-paper run with full candidate price logging and this exact frozen policy.

## Artifacts

- `summary.json`: full metadata and selected policy.
- `validation_search.csv`: all validation-scored policies.
- `top_validation_policies.csv`: top 50 validation policies.
- `selected_validation_summary.csv` and `selected_test_summary.csv`: split summaries.
- `selected_validation_bars.csv` and `selected_test_bars.csv`: equity/action traces.
- `selected_validation_segments.csv` and `selected_test_segments.csv`: chronological segment diagnostics.
- `cost_sensitivity.csv`: test sensitivity to execution costs.

Forward live-paper runs now also produce `candidate_log.csv`, which records every symbol's score, rank, live/candle price, tradability, and selected-candidate flag at each poll. That file is the required artifact for evaluating target switches and missed alternatives after a live run.
