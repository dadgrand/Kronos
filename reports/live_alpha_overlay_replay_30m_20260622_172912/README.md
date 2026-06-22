# Live Alpha Overlay Replay With Signal-Age Gate

This report replays the failed 30-minute forward paper run from
`reports/live_alpha_policy_30m_20260622_172912` with several live-risk overlays.

## Command

```powershell
.\.venv\Scripts\python.exe examples\replay_live_alpha_overlay.py `
  --run-dir reports\live_alpha_policy_30m_20260622_172912 `
  --output-dir outputs\live_alpha_overlay_replay_30m_20260622_172912 `
  --max-session-loss-pct 1.0 `
  --target-change-tolerance 0.05 `
  --max-signal-age-minutes 10.0
```

## Result

| Variant | Final equity | PnL | Return | Interpretation |
| --- | ---: | ---: | ---: | --- |
| `baseline_like` | 9,814.94 | -185.06 | -1.85% | Original behavior reconstructed from logs. |
| `target_change` | 9,864.79 | -135.21 | -1.35% | Earlier rebalance helped, but still lost money. |
| `loss_stop` | 9,843.45 | -156.55 | -1.57% | Session stop helped only after damage had accumulated. |
| `target_change_loss_stop` | 9,883.74 | -116.26 | -1.16% | Best risk overlay without signal freshness. |
| `signal_age_gate` | 10,000.00 | 0.00 | 0.00% | Blocked all trades because signal candles were stale. |
| `full_live_safety` | 10,000.00 | 0.00 | 0.00% | Full overlay also stayed in cash. |

## Signal Freshness Finding

The 30-minute run was not using fresh intraday candles:

| Metric | Value |
| --- | ---: |
| Minimum signal age from candle open | 24.87 min |
| Average signal age from candle open | 29.62 min |
| Maximum signal age from candle open | 34.20 min |
| Minimum signal age from candle close | 14.87 min |
| Average signal age from candle close | 19.62 min |
| Maximum signal age from candle close | 24.20 min |
| Polls older than one 10m bar after close | 26 / 26 |

The original runner entered at `17:29:25` using a latest candle of `17:00:00`.
That is not a robust live trading signal for a 30-minute paper test.

## Interpretation

The previous risk overlay was directionally useful but insufficient: it reduced
the same-slice loss from -1.85% to -1.16%, but still traded a stale signal.

The new signal-age gate changes the live contract: if the latest signal candle is
older than the configured threshold, the runner stays in cash or closes the paper
book. In this replay the threshold is one full 10-minute bar after candle close,
so the candle-only fallback avoids the loss entirely. This is not a profitability
proof; it is a production-safety fix.

The current live runner also supports a `marketdata_bar` signal source, but it
uses only prices whose `UPDATETIME/TIME` is fresh under the configured marketdata
age limit. The old 30-minute logs cannot reconstruct that mode because they only
retained a top-N candidate subset, not full-universe marketdata snapshots with
per-symbol update timestamps.
