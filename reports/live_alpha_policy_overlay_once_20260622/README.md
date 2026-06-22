# Live Alpha Policy Risk Overlay Smoke

This smoke run verifies the live risk overlay added after the negative 30-minute
forward run.

## Config

- Base alpha protocol: `configs/strict_alpha_policy_20260622.json`
- Live overlay config: `configs/strict_alpha_policy_live_risk_overlay_20260622.json`
- Overlay features:
  - `live_rebalance_on_target_change = true`
  - `live_target_change_tolerance = 0.05`
  - `live_max_session_loss_pct = 1.0`

## Command

```powershell
.\.venv\Scripts\python.exe examples\live_paper_alpha_policy.py `
  --config-json configs\strict_alpha_policy_live_risk_overlay_20260622.json `
  --once `
  --fetch-workers 12 `
  --candidate-log-top-n 10 `
  --validation-engine fast `
  --output-dir outputs\live_paper_alpha_policy_overlay_once_20260622
```

## Result

| Metric | Value |
| --- | ---: |
| Latest candle | 2026-06-22 17:30:00 |
| Candidates passing gate | 0 |
| Action | cash |
| Equity | 10,000.00 RUB |
| Return | 0.00% |
| Candidate rows | 20 |
| Price coverage | 100% |
| Tradable rate | 100% |
| Decision seconds | 5.03 |

The overlay did not need to intervene in this smoke because the strict gate was
already cash. The run verifies that the overlay config is loaded, logged, and
keeps the paper book flat when no candidate passes the gate.
