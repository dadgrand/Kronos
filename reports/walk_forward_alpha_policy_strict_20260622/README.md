# Walk-Forward Alpha Policy, Strict Gate

This report evaluates a non-neural causal alpha layer with a strict cash fallback.
It is meant to test whether the project can clear a 10% research target without
relying on the fragile single-window neural-score result.

## Protocol

- Dataset: `datasets/moex_intraday_6f8f836cc811`
- Date coverage used by the run: 2026-01-01 through 2026-06-20 request range
- Completed walk-forward windows: 19
- Validation window: 30 calendar days
- Test window: 7 calendar days
- Step: 7 calendar days
- Initial cash: 10,000 RUB
- Base trading cost: 10 bps
- Stress trading cost: 20 bps
- Candidate count per window: 216
- Candidates: causal momentum/reversal alphas from prior closes only
- Selection: alpha, mode, `k`, gross, and rebalance interval are selected on
  validation only
- Cash fallback: if no candidate passes the strict validation gate, the strategy
  does not trade in the next test window

## Command

```powershell
.\.venv\Scripts\python.exe examples\walk_forward_alpha_policy_lab.py `
  --output-dir outputs\walk_forward_alpha_policy_strict_20260622
```

The protocol is now frozen in `configs/strict_alpha_policy_20260622.json`.
It can be replayed with:

```powershell
.\.venv\Scripts\python.exe examples\walk_forward_alpha_policy_lab.py `
  --config-json configs\strict_alpha_policy_20260622.json `
  --output-dir outputs\frozen_alpha_policy_replay_20260622
```

Replay artifacts are saved in `reports/frozen_alpha_policy_replay_20260622`.

Default strict gate:

- `min_validation_return_pct = 5.0`
- `min_validation_segment_return_pct = 0.0`
- `max_validation_drawdown_pct = 8.0`
- `segment_selection_penalty = 0.5`

## Result

| Metric | Value |
| --- | ---: |
| Initial cash | 10,000.00 RUB |
| Final equity at 10 bps | 11,849.31 RUB |
| PnL at 10 bps | 1,849.31 RUB |
| Return at 10 bps | 18.49% |
| Final equity at 20 bps stress | 11,012.81 RUB |
| Return at 20 bps stress | 10.13% |
| Trade windows | 2 / 19 |
| Trade-window win rate | 50.00% |
| Worst traded window | -0.50% |
| Worst per-window max drawdown | -10.80% |
| Target hit at 10 bps | yes |
| Target hit at 20 bps stress | yes |

## Selected Test Windows

| Window | Alpha | Mode | k | Gross | Validation return | Worst validation segment | Test return | 20 bps stress return |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 11 | `rev_96` | `short_only` | 1 | 1.5 | 31.13% | 1.57% | 19.08% | 13.31% |
| 14 | `rev_12` | `short_only` | 1 | 0.5 | 13.15% | 0.05% | -0.50% | -2.81% |

All other windows were cash because no candidate passed the strict validation
gate.

## Interpretation

This is materially better than the prior neural-score policy claim:

- It is walk-forward: each test window follows its own validation window.
- It uses a cash fallback rather than forcing a trade.
- It clears 10% after doubling the modeled cost from 10 bps to 20 bps.
- It does not use the supplied neural score files, avoiding the frozen-score
  leakage caveat for pre-June windows.

It is still not a production proof:

- The alpha/gate family was introduced after earlier failed experiments, so it
  needs a fresh untouched period or live-forward paper run.
- The return depends mostly on one large winning window.
- The strategy is short-only in the traded windows, so borrow availability,
  short restrictions, spread, impact, and partial fills still need stronger
  modeling.
- The worst intra-window drawdown is -10.80%, even though endpoint drawdown is
  small because most windows are cash.

## Reviewer Check

An independent reviewer found no explicit future leakage or direct test
selection in the inspected code. The reviewer also did not see evidence of
malicious gate manipulation in the implementation.

The reviewer flagged the following risks:

- Effective out-of-sample trade observations are only 2, not 19.
- Most profit comes from window 11, where the test return was 19.08%.
- The strict gate is plausible but may still be process-fit because it was added
  after earlier failed experiments.
- Out of 4,104 candidate-window evaluations, only 7 passed the constraints, all
  in two windows.
- Execution realism remains incomplete for short-only trading.

The honest positioning is therefore: promising strict walk-forward research
signal with sparse trading and cash fallback; not a production-proof trading bot.

## Decision

Promote this to the new research benchmark, not to production. The old
single-window neural result should remain marked as fragile. The next gate is a
fresh forward test using this exact frozen alpha/gate protocol, plus stricter
execution modeling for short availability and slippage.
