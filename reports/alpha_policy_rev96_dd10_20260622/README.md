# Alpha Policy Rev96 DD10 Research Candidate

This report records the first broad-history candidate in this branch that clears the 10% compounded-return target under the walk-forward alpha-policy protocol.

## Candidate

- Config: `configs/strict_alpha_policy_latency_rev96_dd10_20260622.json`
- Dataset: `datasets/moex_intraday_6f8f836cc811`, 50 MOEX symbols, 10-minute bars, from 2022-09-01
- Signal delay: 3 bars
- Alpha universe: `mom_96`, `rev_96`
- Modes: `short_only`, `long_only`, `long_short`
- Selection grid: `k=1`, `gross in {0.5, 1.0, 1.5}`, rebalance every 24 bars
- Gates: validation return >= 5%, validation stress return >= 0%, worst validation segment >= 0%, validation max drawdown no worse than -10%
- Costs: 10 bps base, 20 bps stress

## Results

| Run | Windows | Trade windows | Return | Stress return | Worst trade | Worst per-window DD | Target |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Monthly frozen, fast engine | 46 | 2 | 14.25% | 8.16% | 0.47% | -19.38% | hit |
| Monthly frozen, full engine | 46 | 2 | 14.25% | 8.16% | 0.47% | -19.38% | hit |
| Weekly rolling, fast engine | 193 | 6 | 28.22% | 13.84% | -6.80% | -13.62% | hit |

The monthly full/fast equality check matched selected candidates and all compared return/equity fields exactly.

## Context

The previous latency-aware candidate (`strict_alpha_policy_latency_delay3_dd6_20260622`) reproduced +10.90% on the narrow 2026 slice, but failed the broad monthly history from 2022-09 with -13.73%. A stricter meta-sweep that kept validation drawdown at 6% improved the broad monthly result to +9.76%, just under target. Expanding the validation drawdown gate to 10% selected a simple `rev_96` short-only policy in two monthly windows and reached +14.25%.

## Professional Read

This is meaningful progress, not a production proof. The per-window selection still uses validation-only data before each test window, but the meta-configuration itself was chosen after sweeping the full historical period. That makes it a research candidate requiring future forward validation.

The good signs are simplicity, latency delay, positive stress filter, identical fast/full reproduction, and a stronger weekly rolling check. The weak signs are sparse monthly trading, one weekly losing trade, monthly stress below 10%, and a large intrawindow drawdown in the best monthly trade.

## Verification Commands

```powershell
.\.venv\Scripts\python.exe -m py_compile examples\walk_forward_alpha_policy_lab.py examples\sweep_alpha_policy_constraints.py
git diff --check
.\.venv\Scripts\python.exe examples\walk_forward_alpha_policy_lab.py --config-json configs\strict_alpha_policy_latency_rev96_dd10_20260622.json --engine full --output-dir outputs\walk_forward_alpha_latency_rev96_dd10_monthly_202209_202606_full
.\.venv\Scripts\python.exe examples\walk_forward_alpha_policy_lab.py --start-date 2022-09-01 --end-date 2026-06-20 --validation-days 30 --test-days 7 --step-days 7 --alpha-kinds mom,rev --alpha-lags 96 --k-grid 1 --gross-grid 0.5,1.0,1.5 --signal-delay-bars 3 --min-validation-stress-return-pct 0 --max-validation-drawdown-pct 10 --engine fast --output-dir outputs\walk_forward_alpha_latency_rev96_dd10_weekly_202209_202606_fast_real
```

