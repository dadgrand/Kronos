# Fresh Kronos Diplom Risk Artifact

Monthly target-free risk predictions for the current Kronos universe.

- Predictions: `artifacts\fresh_diplom_risk_20260623\fresh_kronos_risk_predictions.csv`
- Rows: 2059
- Tickers: 50 / 50
- Input range: 2022-09-01 09:50:00 .. 2026-06-19 23:40:00
- Target-free schema check: True
- SHA256: `320fd18dc8bb47ef383e7399387009991cee48742cfe1b1e0aac7a540e843a11`

The model is intentionally degraded and heuristic. It uses only past close/volume-derived volatility, drawdown, liquidity, beta/correlation, and momentum features available by `source_data_end`.

This artifact is for leakage-aware overlay testing, not for a standalone claim that risk probabilities are calibrated.
