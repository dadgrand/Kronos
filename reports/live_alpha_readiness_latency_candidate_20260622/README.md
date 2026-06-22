# Live Alpha Readiness Check

- Config: `configs\strict_alpha_policy_latency_delay3_dd6_20260622.json`
- Checked at: 2026-06-22T19:20:46
- Feed ready for forward: False
- Reason: `stale_signal`
- Signal source: `candle`
- Signal age: 20.78 min
- Latest candle: 2026-06-22 18:50:00
- Candle close: 2026-06-22 19:00:00
- Raw marketdata prices: 50 / 50
- Fresh marketdata prices: 0 / 50
- Stale marketdata prices: 50 / 50

## Interpretation

The latency candidate was not ready for a meaningful 30-minute forward paper run
at this timestamp. The public MOEX feed had full raw price coverage, but every
marketdata price was stale under the configured 2-minute freshness rule, and the
candle fallback was 20.78 minutes older than its close. The live runner would
therefore stay in cash via `stale_signal_cash`.

This is a useful pre-flight check: it prevents counting a no-trade cash run as
evidence for or against the `+10.90%` latency research candidate.
