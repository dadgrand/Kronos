# Live Alpha Policy Freshness Smoke

This one-shot run verifies the live runner after adding strict freshness checks
for both candle fallback signals and marketdata signal bars.

## Command

```powershell
.\.venv\Scripts\python.exe examples\live_paper_alpha_policy.py `
  --config-json configs\strict_alpha_policy_live_risk_overlay_20260622.json `
  --once `
  --fetch-workers 12 `
  --candidate-log-top-n 10 `
  --validation-engine fast `
  --output-dir outputs\live_paper_alpha_policy_freshness_once_20260622
```

## Result

| Metric | Value |
| --- | ---: |
| Wall time | 2026-06-22 18:38:13 |
| Signal source | `candle` |
| Latest candle | 2026-06-22 18:10:00 |
| Candle close | 2026-06-22 18:20:00 |
| Signal age after close | 18.23 min |
| Max allowed candle age | 10.00 min |
| Raw marketdata prices | 50 / 50 |
| Fresh marketdata prices | 0 / 50 |
| Stale marketdata prices | 50 / 50 |
| Marketdata max age allowed | 2.00 min |
| Marketdata signal bar used | false |
| Stale-signal gate hit | true |
| Action | `stale_signal_cash` |
| Initial equity | 10,000.00 RUB |
| Final equity | 10,000.00 RUB |
| PnL | 0.00 RUB |
| Return | 0.00% |

## Interpretation

The free MOEX ISS marketdata snapshot contained prices for the whole universe,
but every price was stale under the 2-minute marketdata freshness rule. The
runner therefore refused to synthesize a fresh signal bar, fell back to the
latest candle, and then blocked execution because that candle was more than one
full 10-minute bar old after close.

An in-memory regression check also verifies the partial-freshness case: if one
symbol is fresh and another is stale, only the fresh symbol remains eligible for
latest-row selection and paper execution.

This is a safety result, not a profitability result. The current public data feed
is too delayed for a short live paper trading loop unless the strategy horizon is
changed or a genuinely fresh market-data source is added.
