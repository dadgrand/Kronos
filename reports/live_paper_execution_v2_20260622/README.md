# Live paper execution v2

`examples/live_paper_moex_policy_v2.py` is the execution-aware live paper runner.

It keeps the neural score ensemble from the research pipeline, but changes the live execution layer:

- supports `short_only`, `long_only`, and `long_short` candidate selection;
- defaults to safer `max_gross=1.0` instead of the research backtest's 4x gross;
- gates entries by a cost-aware confidence proxy;
- exits early on stop-loss, take-profit, or target switch;
- can optionally close after N consecutive weak/blocked filter signals with `--exit-on-filter-fail --filter-fail-exit-bars N`;
- closes any remaining paper position at the end of the run and reports realized final equity;
- logs every action with candidate, edge estimate, turnover, costs, and equity.
- can load a frozen research policy with `--policy-json`, including gross, mode, confidence, and market-regime gates.
- writes `candidate_log.csv` with per-symbol scores, ranks, prices, tradability, and selected-candidate flags for replayable forward analysis.

Post-run candidate analysis:

```powershell
python examples/analyze_live_candidate_log.py --run-dir <live-output-dir> --output-dir <analysis-output-dir>
```

## Conservative live paper command

```powershell
python examples/live_paper_moex_policy_v2.py --duration-minutes 30 --poll-seconds 60 --initial-cash 10000 --cost-bps 10 --device cuda:0 --max-gross 1.0 --mode long_short
```

## Aggressive research-style command

```powershell
python examples/live_paper_moex_policy_v2.py --duration-minutes 30 --poll-seconds 60 --initial-cash 10000 --cost-bps 10 --device cuda:0 --max-gross 4.0 --mode short_only
```

## One-shot sanity check

Command:

```powershell
python examples/live_paper_moex_policy_v2.py --once --duration-minutes 0 --poll-seconds 10 --initial-cash 10000 --cost-bps 10 --device cuda:0 --max-gross 1.0 --mode long_short
```

Observed output:

```text
FINAL equity_after_close=9980.02 pnl=-19.98 return=-0.1998%
```

This one-shot intentionally enters and immediately closes, so the loss is almost exactly two 10 bps commissions on a 1x gross paper trade. It verifies that end-of-run forced close and cost accounting work.

## Status

This is still paper trading infrastructure, not broker execution. It should be run for multi-day forward tests before any production claim.
