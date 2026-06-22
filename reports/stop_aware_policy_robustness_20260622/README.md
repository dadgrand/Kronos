# Stop-Aware Policy Robustness Check

This report checks whether the 10% research-test policy remains attractive when the
validation selector is made segment-aware.

## Question

The original selected policy reached `+11.32%` on the locked research test split,
but the segment table showed that gains were concentrated in the middle of the
test period. This run asks a stricter question:

Can the selector choose a policy that is still profitable while avoiding weak
validation segments?

## Protocol

- Dataset: `datasets/moex_intraday_6f8f836cc811`
- Interval: 10 minute MOEX candles
- Train end: `2026-06-01`
- Validation end: `2026-06-13`
- Test: bars after `2026-06-13`
- Initial cash: 10,000 RUB
- Trading cost: 10 bps per unit turnover
- Max gross exposure: 1.5x
- Segment count: 5 chronological segments per split
- Policy search: validation only

The code change adds two validation-selection controls:

- `--segment-selection-penalty`: penalizes policies with a weak worst validation
  segment.
- `--min-validation-segment-return-pct`: rejects policies whose worst validation
  segment is below the configured floor.

## Results

| Scenario | Segment penalty | Min validation segment | Validation return | Worst validation segment | Test return | Worst test segment | Active rate test | Hit 10% target |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| baseline | 0.0 | n/a | 24.40% | -2.92% | 11.32% | -2.35% | 57.42% | yes |
| segment penalty 0.5 | 0.5 | n/a | 23.21% | -1.61% | 3.76% | -0.47% | 26.45% | no |
| segment penalty 2.0 | 2.0 | n/a | 23.21% | -1.61% | 3.76% | -0.47% | 26.45% | no |
| min validation segment -2% | 0.0 | -2.0% | 23.21% | -1.61% | 3.76% | -0.47% | 26.45% | no |
| min validation segment 0% | 0.0 | 0.0% | 8.56% | 0.00% | 1.17% | -0.23% | 7.74% | no |

Full machine-readable results are in `comparison.csv`.

## Interpretation

The original `+11.32%` research-test result does not survive stricter validation
robustness selection. The segment-aware variants improve smoothness and reduce
activity, but they also give up most of the test profit.

This is a negative but useful result. It means the current edge is not yet robust
enough to trust as a production policy. The best-looking policy is likely tied to
a specific market regime and should be treated as a research candidate, not a
deployable trading system.

## Decision

Do not promote the `+11.32%` policy as solved. Keep it as a benchmark, but gate
future policies on robustness:

1. Walk-forward validation across multiple time slices.
2. Cost sensitivity that remains profitable at 20 bps, not only 10 bps.
3. Live-forward paper runs with full candidate logs before any production use.

The next productive step is to optimize for stable forward behavior rather than
for a single high test-window return.

## Reviewer Check

An independent reviewer checked the current methodology and found no direct
future leakage or explicit test-label selection in the inspected code. The review
did flag high process-overfit risk:

- The test window is short and the profit is concentrated in the middle of the
  split.
- The search covers 1,125 policy candidates on a small validation period.
- The original 10% result fails under robustness-oriented selection.
- The policy is still cost-sensitive and execution realism is incomplete.

The reviewer recommendation matches the decision above: make walk-forward
validation, robust selection, and stricter execution modeling mandatory before
promoting any policy.
