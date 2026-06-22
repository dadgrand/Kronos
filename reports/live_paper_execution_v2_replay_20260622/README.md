# Live Paper 30 Minute Replay, 2026-06-22

This report replays the original 30 minute live paper run from:

`outputs/live_paper_moex_policy_30m_20260622_140906/live_log.csv`

The goal is to evaluate the execution-aware policy changes on the same observed live slice, starting with 10,000 RUB and 10 bps transaction cost. V2 scenarios include the live runner's default 0.75% stop-loss and 0.75% take-profit rules.

## Scope

The saved live log contains exact mark prices only for the actually held instrument, HEAD. The original runner later selected HYDR as the target, but did not log HYDR marks at every step. This replay is therefore a strict same-log replay of the observed HEAD position path, not a full multi-symbol market replay.

That limitation is important: it lets us compare execution rules on the same known position path, but it cannot fairly score a target switch into HYDR.

## Results

| Scenario | Gross | Filter exit | Final equity | PnL | Return | Max drawdown |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| old_v1_gross4_hold_force_close | 4.0x | No | 10007.61 | 7.61 | 0.0761% | -0.8788% |
| v2_default_gross1_stop_no_filter_exit | 1.0x | No | 10001.90 | 1.90 | 0.0190% | -0.2190% |
| v2_gross1_stop_exit_after_1_fail | 1.0x | 1 fail | 9958.10 | -41.90 | -0.4190% | -0.3194% |
| v2_gross1_stop_exit_after_3_fails | 1.0x | 3 fails | 9958.10 | -41.90 | -0.4190% | -0.3194% |
| v2_gross1_stop_exit_after_5_fails | 1.0x | 5 fails | 9980.00 | -20.00 | -0.2000% | -0.2190% |
| v2_gross2_stop_no_filter_exit | 2.0x | No | 10003.81 | 3.81 | 0.0381% | -0.4385% |
| v2_gross4_stop_no_filter_exit | 4.0x | No | 9832.39 | -167.61 | -1.6761% | -1.2813% |

Report artifacts:

- `comparison.csv`: scenario-level replay metrics.
- `old_v1_gross4_hold_force_close.csv`: old aggressive behavior with forced close.
- `v2_default_gross1_stop_no_filter_exit.csv`: new default conservative behavior.
- `v2_gross1_stop_exit_after_1_fail.csv`: immediate filter-fail exit.
- `v2_gross1_stop_exit_after_3_fails.csv`: three-bar filter-fail patience.
- `v2_gross1_stop_exit_after_5_fails.csv`: five-bar filter-fail patience.
- `v2_gross2_stop_no_filter_exit.csv`: intermediate gross with no filter-fail exit.
- `v2_gross4_stop_no_filter_exit.csv`: aggressive gross with stop-loss enabled.

## Interpretation

On this exact 30 minute slice, immediate exit on filter failure was harmful. The filter failed while the short HEAD position was near the adverse part of the move, around 2748. Holding the conservative short until the end benefited from the later decline to 2736.

The stop-aware replay also shows why the old 4x gross should not be treated as a v2 result: with the live runner's default 0.75% stop-loss, the 4x short exits at the adverse move and finishes at 9832.39 RUB. The old v1 profit only appears when the position is allowed to keep running through that drawdown.

The execution-aware v2 work is still valuable because it adds controls that the v1 live runner lacked: lower gross exposure, explicit closeout, stop/take-profit hooks, target-switch handling, and a fixed cash-account sizing path. But this replay does not support enabling a filter-fail exit as the default.

## Policy Change Recommended By This Slice

The live runner now treats the market filter primarily as an entry gate by default. It no longer exits an existing position just because a single filter/gate signal blocks new entries.

For research runs, `examples/live_paper_moex_policy_v2.py` supports:

- `--exit-on-filter-fail` to enable filter/gate exits.
- `--filter-fail-exit-bars N` to require N consecutive blocked bars before closing.

On this slice, 1 and 3 failed bars both exited near the adverse price area. Five failed bars reduced the loss to about two commissions. No filter-fail exit produced the best conservative gross-1 result, while the independent stop-loss still protects higher gross settings.

For the next live run, the logger should also store all candidate symbol marks and scores on every step. Without that, replay cannot evaluate whether switching from HEAD to HYDR would have improved the outcome.

## Reproduction

```powershell
.\.venv\Scripts\python.exe examples\replay_live_paper_log.py `
  --log outputs\live_paper_moex_policy_30m_20260622_140906\live_log.csv `
  --output-dir outputs\replay_30m_execution_v2_stopaware_20260622_140906 `
  --initial-cash 10000 `
  --cost-bps 10
```
