# May 2026 Variant Sweep

This report tests the main alpha-policy variants on May 2026 test windows.

## Protocol

- Test period: weekly test windows inside May 2026.
- Window construction: 30 validation days before each 7-day test window.
- Date command envelope: `start_date=2026-04-01`, `end_date=2026-06-01`, `validation_days=30`, `test_days=7`, `step_days=7`.
- Costs: 10 bps base, 20 bps stress.
- Initial cash: 10000 RUB.
- Engine: `fast`.

The four evaluated May test windows are:

- 2026-05-01 to 2026-05-08
- 2026-05-08 to 2026-05-15
- 2026-05-15 to 2026-05-22
- 2026-05-22 to 2026-05-29

## Summary

| Variant | Trade windows | Return | PnL | Stress return | Worst test DD | Result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `fullgrid_delay2_dd8` | 1 / 4 | 0.20% | 19.65 | -2.15% | -3.66% | best May result |
| `fullgrid_delay0_dd8` | 0 / 4 | 0.00% | 0.00 | 0.00% | 0.00% | cash |
| `fullgrid_delay1_dd8` | 0 / 4 | 0.00% | 0.00 | 0.00% | 0.00% | cash |
| `fullgrid_delay3_dd8` | 1 / 4 | -3.76% | -376.17 | -4.86% | -5.41% | failed May |
| `fullgrid_delay3_dd6` | 1 / 4 | -3.76% | -376.17 | -4.86% | -5.41% | failed May |
| `rev96_delay3_dd10` | 1 / 4 | -7.44% | -744.20 | -9.55% | -10.57% | failed May |

## Trades

| Variant | Test window | Selected policy | Validation return | Test return | Stress return | Test max DD |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `fullgrid_delay2_dd8` | 2026-05-01..2026-05-08 | `mom_96 short_only k=3 gross=1.0` | 10.65% | 0.20% | -2.15% | -3.66% |
| `fullgrid_delay3_dd8` | 2026-05-08..2026-05-15 | `mom_96 short_only k=1 gross=0.5` | 11.83% | -3.76% | -4.86% | -5.41% |
| `fullgrid_delay3_dd6` | 2026-05-08..2026-05-15 | `mom_96 short_only k=1 gross=0.5` | 11.83% | -3.76% | -4.86% | -5.41% |
| `rev96_delay3_dd10` | 2026-05-08..2026-05-15 | `mom_96 short_only k=1 gross=1.0` | 24.37% | -7.44% | -9.55% | -10.57% |

## Interpretation

May 2026 is a weak local holdout for the current policies. The newly selected broad-history candidate (`rev96_delay3_dd10`) did not generalize to this month and produced the worst result in the sweep. The only positive May result was a tiny +0.20% from `fullgrid_delay2_dd8`, and even that trade became negative under the 20 bps stress cost.

The practical conclusion is that the strategy needs an additional regime filter or no-trade veto for May-like conditions. The broad 2022-09..2026-06 result is still useful research evidence, but May 2026 argues against treating the current candidate as production-ready.

## Commands

```powershell
.\.venv\Scripts\python.exe examples\walk_forward_alpha_policy_lab.py --start-date 2026-04-01 --end-date 2026-06-01 --validation-days 30 --test-days 7 --step-days 7 --alpha-lags 12,48,96 --alpha-kinds mom,rev,voladj_mom,voladj_rev --k-grid 1,3 --gross-grid 0.5,1.0,1.5 --rebalance-grid 24 --signal-delay-bars 0 --max-validation-drawdown-pct 8 --engine fast --output-dir outputs\may2026_fullgrid_delay0_dd8
.\.venv\Scripts\python.exe examples\walk_forward_alpha_policy_lab.py --start-date 2026-04-01 --end-date 2026-06-01 --validation-days 30 --test-days 7 --step-days 7 --alpha-lags 12,48,96 --alpha-kinds mom,rev,voladj_mom,voladj_rev --k-grid 1,3 --gross-grid 0.5,1.0,1.5 --rebalance-grid 24 --signal-delay-bars 1 --max-validation-drawdown-pct 8 --engine fast --output-dir outputs\may2026_fullgrid_delay1_dd8
.\.venv\Scripts\python.exe examples\walk_forward_alpha_policy_lab.py --start-date 2026-04-01 --end-date 2026-06-01 --validation-days 30 --test-days 7 --step-days 7 --alpha-lags 12,48,96 --alpha-kinds mom,rev,voladj_mom,voladj_rev --k-grid 1,3 --gross-grid 0.5,1.0,1.5 --rebalance-grid 24 --signal-delay-bars 2 --max-validation-drawdown-pct 8 --engine fast --output-dir outputs\may2026_fullgrid_delay2_dd8
.\.venv\Scripts\python.exe examples\walk_forward_alpha_policy_lab.py --start-date 2026-04-01 --end-date 2026-06-01 --validation-days 30 --test-days 7 --step-days 7 --alpha-lags 12,48,96 --alpha-kinds mom,rev,voladj_mom,voladj_rev --k-grid 1,3 --gross-grid 0.5,1.0,1.5 --rebalance-grid 24 --signal-delay-bars 3 --max-validation-drawdown-pct 8 --engine fast --output-dir outputs\may2026_fullgrid_delay3_dd8
.\.venv\Scripts\python.exe examples\walk_forward_alpha_policy_lab.py --start-date 2026-04-01 --end-date 2026-06-01 --validation-days 30 --test-days 7 --step-days 7 --alpha-lags 12,48,96 --alpha-kinds mom,rev,voladj_mom,voladj_rev --k-grid 1,3 --gross-grid 0.5,1.0,1.5 --rebalance-grid 24 --signal-delay-bars 3 --max-validation-drawdown-pct 6 --engine fast --output-dir outputs\may2026_fullgrid_delay3_dd6
.\.venv\Scripts\python.exe examples\walk_forward_alpha_policy_lab.py --start-date 2026-04-01 --end-date 2026-06-01 --validation-days 30 --test-days 7 --step-days 7 --alpha-kinds mom,rev --alpha-lags 96 --k-grid 1 --gross-grid 0.5,1.0,1.5 --rebalance-grid 24 --signal-delay-bars 3 --min-validation-stress-return-pct 0 --max-validation-drawdown-pct 10 --engine fast --output-dir outputs\may2026_rev96_delay3_dd10
```

