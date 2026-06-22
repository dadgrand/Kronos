# Fresh Diplom Risk Overlay Report

Дата: 2026-06-23.

Цель: довести Kronos + diplom risk overlay от старого неполного artifact к практически проверяемому состоянию: свежий target-free risk artifact для текущей Kronos universe, pre-gate risk-aware hooks, backtest matrix, leakage checks и честный вывод по utility.

## Итог

Свежий overlay стал практически используемым как risk layer, но не как доказанный генератор прибыли.

- Fresh artifact покрывает 50/50 Kronos тикеров, включая прямой `YDEX`, без `YNDX` mapping.
- Adapter теперь использует `availability_timestamp` для point-in-time as-of lookup, а не только `decision_date`.
- Overlay может влиять на target sizing, candidate score и gate score. Все новые ручки выключены по умолчанию.
- В backtests есть измеримый realized downside reduction:
  - `default_2026`: fresh target-only/combined улучшили executed return на `+0.1515` pct-points.
  - `broad_2022_2026_monthly`: fresh target-only/combined улучшили executed return на `+0.4954` pct-points, уйдя из единственного убыточного trade window.
  - `y2025_full_year`: fresh gate-only/combined улучшили executed return на `+0.6190` pct-points, тоже risk-off через cash.
- Profit improvement не доказан: лучшие realized deltas в основном получены через уменьшение/отключение убыточных trades, при очень высокой cash-only rate.

## Artifacts

- Predictions: `artifacts/fresh_diplom_risk_20260623/fresh_kronos_risk_predictions.csv`
- Manifest: `artifacts/fresh_diplom_risk_20260623/artifact_manifest.json`
- SHA256: `320fd18dc8bb47ef383e7399387009991cee48742cfe1b1e0aac7a540e843a11`
- Rows: `2059`
- Tickers: `50/50`
- Source data: `2022-09-01 09:50:00` to `2026-06-19 23:40:00`
- Latest availability: `2026-06-19 23:41:00`
- Target-free validation: forbidden columns absent, duplicate key count `0`, probability bounds errors `0`, probability sum max error `2.22e-16`

The artifact is intentionally degraded and target-free: it uses only past close/volume-derived volatility, downside volatility, drawdown, liquidity, beta/correlation and momentum features. It does not export `risk_class`, `risk_score`, `future_*`, realized PnL, future drawdown, or future liquidity.

## Fresh Configs

- Practical profile: `configs/strict_nested_edge_policy_fresh_diplom_target_only_20260623.json`
  - `diplom_risk_weight_pct=25`
  - `diplom_long_veto_p_high=0.8`
  - score/gate penalties disabled
- Conservative risk-off profile: `configs/strict_nested_edge_policy_fresh_diplom_combined_20260623.json`
  - target sizing + veto
  - `diplom_long_p_high_cap=0.60`
  - `diplom_candidate_score_penalty_pct=6`
  - `diplom_gate_penalty_pct=4`

## Backtest Matrix

Outputs: `outputs/fresh_diplom_risk_backtest_matrix_20260623`

Report tables:

- `comparison_summary.csv`
- `pair_diagnostics.csv`
- `affected_window_audit.csv`
- `artifact_ticker_summary.csv`
- `return_delta_pivot.csv`

Executed return deltas vs baseline:

| Period | old artifact | fresh target-only | fresh candidate-only | fresh gate-only | fresh combined |
|---|---:|---:|---:|---:|---:|
| default_2026 | 0.0000 | +0.1515 | 0.0000 | 0.0000 | +0.1515 |
| broad_2022_2026_monthly | 0.0000 | +0.4954 | 0.0000 | 0.0000 | +0.4954 |
| may_2026 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| y2025_full_year | 0.0000 | 0.0000 | 0.0000 | +0.6190 | +0.6190 |
| ydex_transition | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

Affected-window diagnostics:

| Period / variant | candidate changes | gate changes | affected windows | affected and traded | raw delta sum | executed delta sum |
|---|---:|---:|---:|---:|---:|---:|
| default_2026 / fresh_target_only | 3 | 0 | 7 | 1 | +3.6286 | +0.1515 |
| default_2026 / fresh_combined | 6 | 0 | 9 | 1 | +8.8895 | +0.1515 |
| broad_2022_2026 / fresh_target_only | 4 | 1 | 15 | 0 | +18.4787 | +0.4954 |
| broad_2022_2026 / fresh_combined | 7 | 1 | 18 | 0 | +2.6357 | +0.4954 |
| y2025_full_year / fresh_gate_only | 0 | 1 | 1 | 0 | 0.0000 | +0.6190 |
| y2025_full_year / fresh_combined | 2 | 1 | 5 | 0 | +11.3671 | +0.6190 |
| ydex_transition / fresh_combined | 1 | 0 | 3 | 0 | +10.0566 | 0.0000 |

Interpretation:

- `fresh_target_only` is the least intrusive practical candidate. It improves the only traded `default_2026` loss by sizing and improves broad monthly by avoiding a loss through the existing gate after risk-adjusted validation/gate paths.
- `fresh_combined` is more defensive. It can change candidate selection and gate score, but it tends to reduce trade frequency.
- `fresh_gate_only` only showed realized value in the 2025 slice by vetoing the sole losing trade. This is risk-off utility, not alpha.
- `fresh_candidate_only` is not robust enough: it helped broad raw deltas but hurt default and YDEX-transition raw deltas.
- The old saved diplom artifact remains a negative control: executed deltas were zero in all required periods.

## YDEX

The fresh artifact has direct `YDEX` rows:

- first decision date: `2024-09-30`
- latest decision date: `2026-06-19`
- prediction count: `22`
- latest class: `low`
- latest `p_high`: `0.0738`
- latest feature coverage: `1.0`

The YDEX transition backtest had zero executed trades for every scenario because the nested gate sent all windows to cash. Therefore, YDEX mapping/trading utility is not proven by executed PnL, but the fresh artifact removes the old blocker where no `YDEX` row existed.

## Leakage Checklist

- Artifact export is target-free: pass.
- `availability_timestamp <= decision_time` enforced by adapter search key: pass.
- Lookup just before latest availability still sees the prior month; lookup after latest availability returns `50/50 ok`: pass.
- `YDEX` is direct, no silent `YNDX` mapping: pass.
- Overlay parameters were fixed before final matrix runs: pass.
- Old artifact is treated only as negative control: pass.
- Live runner is updated for config parity, but live-paper output is not used as validation evidence.

## Commands

```powershell
.\.venv\Scripts\python.exe examples\build_fresh_diplom_risk_artifact.py --dataset-dir datasets\moex_intraday_6f8f836cc811 --interval 10 --output-dir artifacts\fresh_diplom_risk_20260623
.\.venv\Scripts\python.exe tests\run_direct_overlay_tests.py
.\.venv\Scripts\python.exe examples\run_diplom_risk_backtest_matrix.py --output-dir outputs\fresh_diplom_risk_backtest_matrix_20260623 --skip-existing
.\.venv\Scripts\python.exe examples\summarize_diplom_risk_matrix.py --matrix-dir outputs\fresh_diplom_risk_backtest_matrix_20260623 --artifact-dir artifacts\fresh_diplom_risk_20260623 --output-dir reports\fresh_diplom_risk_overlay_20260623
```

Direct overlay tests: `17 passed`.

Backtest matrix: `30/30` summary files.

## Limitations

- The fresh model is a degraded heuristic risk model, not a retrained diploma classifier with validated probability calibration.
- The current backtest policy still has very high cash-only behavior; many raw improvements are latent because the nested gate already vetoes trades.
- The most recent available bars in the frozen-compatible dataset end at `2026-06-19 23:40:00`, not at the manifest `till_date`.
- No claim is made that the overlay improves live trading PnL. The honest claim is narrower: the integration is point-in-time safe, the fresh artifact covers the Kronos universe, and fixed fresh overlay profiles show measurable downside/risk-off utility in the required historical matrix.

This is research infrastructure, not financial advice.
