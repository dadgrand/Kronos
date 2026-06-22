# Diplom Risk Overlay Integration

## Scope

This run integrates the diploma monthly risk-screening artifact as a Kronos-side point-in-time overlay. The overlay is disabled by default in the frozen strict config and must be enabled explicitly for research/live paper runs.

Implemented behavior:

- Load only safe prediction-time columns from the diploma `predictions.csv`.
- Ignore `risk_class`, `risk_score`, and all `future_*` columns on inference.
- Use backward as-of matching only: `prediction_date <= decision_time`.
- Mark missing and stale predictions explicitly as `diplom_risk_missing` / `diplom_risk_stale`.
- For long weights, reduce gross by `diplom_risk_weight_pct * p_high`, with optional long veto.
- For short weights, do nothing by default. High risk is not a direct short signal.
- Keep `YNDX -> YDEX` mapping disabled by default. If enabled, it is only active after `diplom_yndx_ydex_effective_date`.

## Files

- Adapter: `examples/diplom_risk_adapter.py`
- Walk-forward integration: `examples/walk_forward_alpha_policy_lab.py`
- Live paper integration: `examples/live_paper_alpha_policy.py`
- Shared stop-aware hook: `examples/stop_aware_filtered_policy_lab.py`
- Config fields: `configs/strict_nested_edge_policy_20260622.json`
- Tests: `tests/test_diplom_risk_adapter.py`
- Metrics: `comparison_metrics.csv`, `comparison_deltas.csv`

## Data Contract

The saved diploma artifact used here is:

`D:/AI/diplom_selected/results/diploma_run/predictions.csv`

Coverage facts:

- Saved predictions: 2025-03-31 through 2025-08-29, 17 tickers.
- Kronos universe: 50 tickers.
- Direct overlap: 17 tickers.
- `YNDX` appears in the diploma monthly panel, while Kronos uses `YDEX`.
- The checked saved prediction artifact itself has no `YNDX` rows, so enabling mapping would not improve this artifact.
- For 2026 and May 2026 runs, the saved artifact is stale or missing for all active decisions.

YDEX dating note: MOEX lists YDEX with listing date 2024-07-08, and public news reported regular trading scheduled for 2024-07-24. The default mapping date is therefore 2024-07-24, but mapping remains disabled unless configured.

## Commands

Adapter checks:

```powershell
@'
import importlib.util
from pathlib import Path
module_path = Path('tests/test_diplom_risk_adapter.py')
spec = importlib.util.spec_from_file_location('test_diplom_risk_adapter', module_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
for name in sorted(dir(mod)):
    if name.startswith('test_'):
        getattr(mod, name)()
'@ | .\.venv\Scripts\python.exe -
```

Compile:

```powershell
.\.venv\Scripts\python.exe -m py_compile examples\diplom_risk_adapter.py examples\stop_aware_filtered_policy_lab.py examples\walk_forward_stop_aware_policy_lab.py examples\walk_forward_alpha_policy_lab.py examples\live_paper_alpha_policy.py examples\check_live_alpha_readiness.py tests\test_diplom_risk_adapter.py
```

Backtests:

```powershell
.\.venv\Scripts\python.exe examples\walk_forward_alpha_policy_lab.py --config-json configs\strict_nested_edge_policy_20260622.json --output-dir reports\diplom_risk_overlay_20260623\default_2026_baseline
.\.venv\Scripts\python.exe examples\walk_forward_alpha_policy_lab.py --config-json configs\strict_nested_edge_policy_20260622.json --diplom-risk-enabled --diplom-risk-predictions-path D:\AI\diplom_selected\results\diploma_run\predictions.csv --output-dir reports\diplom_risk_overlay_20260623\default_2026_overlay

.\.venv\Scripts\python.exe examples\walk_forward_alpha_policy_lab.py --config-json configs\strict_nested_edge_policy_20260622.json --start-date 2022-09-01 --end-date 2026-06-20 --validation-days 120 --nested-gate-days 30 --test-days 30 --step-days 30 --output-dir reports\diplom_risk_overlay_20260623\broad_monthly_baseline
.\.venv\Scripts\python.exe examples\walk_forward_alpha_policy_lab.py --config-json configs\strict_nested_edge_policy_20260622.json --start-date 2022-09-01 --end-date 2026-06-20 --validation-days 120 --nested-gate-days 30 --test-days 30 --step-days 30 --diplom-risk-enabled --diplom-risk-predictions-path D:\AI\diplom_selected\results\diploma_run\predictions.csv --output-dir reports\diplom_risk_overlay_20260623\broad_monthly_overlay

.\.venv\Scripts\python.exe examples\walk_forward_alpha_policy_lab.py --config-json configs\strict_nested_edge_policy_20260622.json --start-date 2026-04-01 --end-date 2026-06-01 --validation-days 30 --nested-gate-days 7 --test-days 31 --step-days 31 --output-dir reports\diplom_risk_overlay_20260623\may2026_baseline
.\.venv\Scripts\python.exe examples\walk_forward_alpha_policy_lab.py --config-json configs\strict_nested_edge_policy_20260622.json --start-date 2026-04-01 --end-date 2026-06-01 --validation-days 30 --nested-gate-days 7 --test-days 31 --step-days 31 --diplom-risk-enabled --diplom-risk-predictions-path D:\AI\diplom_selected\results\diploma_run\predictions.csv --output-dir reports\diplom_risk_overlay_20260623\may2026_overlay
```

## Metrics

| Run | Return % | Final equity | Max DD % | Trade windows | Cash-only rate | Avoided loss | Missed profit | Raw loss recall | Diplom changed decisions | Available | Missing | Stale |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| default 2026 baseline | -4.20 | 9580.25 | -4.20 | 1 / 19 | 94.74% | 64.54 | 11.15 | 93.33% | 0 | 0 | 0 | 0 |
| default 2026 overlay | -4.20 | 9580.25 | -4.20 | 1 / 19 | 94.74% | 64.54 | 11.15 | 93.33% | 0 | 0 | 562 | 232 |
| broad monthly baseline | -0.50 | 9950.46 | -0.50 | 1 / 42 | 97.62% | 322.55 | 62.09 | 96.67% | 0 | 0 | 0 | 0 |
| broad monthly overlay | -0.50 | 9950.46 | -0.50 | 1 / 42 | 97.62% | 320.02 | 62.09 | 96.67% | 35 | 222 | 4864 | 432 |
| May 2026 baseline | 0.00 | 10000.00 | 0.00 | 0 / 1 | 100.00% | 24.27 | 0.00 | 100.00% | 0 | 0 | 0 | 0 |
| May 2026 overlay | 0.00 | 10000.00 | 0.00 | 0 / 1 | 100.00% | 24.27 | 0.00 | 100.00% | 0 | 0 | 89 | 17 |

Overlay deltas:

| Scenario | Return delta | Equity delta | Trade-window delta | Changed decisions |
| --- | ---: | ---: | ---: | ---: |
| default 2026 | 0.00 | 0.00 | 0 | 0 |
| broad monthly | 0.00 | 0.00 | 0 | 35 |
| May 2026 | 0.00 | 0.00 | 0 | 0 |

## Interpretation

The adapter architecture works, but this saved diploma prediction artifact does not improve executed results.

Default 2026 and May 2026 have no available fresh diploma predictions. The overlay correctly reports missing/stale status and leaves execution unchanged under the default neutral missing policy.

The broad monthly run has some available 2025 risk coverage and the overlay changes 35 raw rebalance decisions. Those changes do not affect executed equity because the strict nested gate already sends the affected windows to cash. In other words, the risk overlay is reusable, but the current artifact is not current enough to help the live 2026 strategy.

May 2026 was already protected by the nested gate: the raw selected candidate would have lost -24.27%, but the baseline and overlay both stayed cash.

## Reviews

Researcher:

- Recommended a Kronos-side adapter with safe columns, backward as-of matching, explicit mapping, and no target columns.
- Noted that a production diploma export should be target-free and based on `risk_pipeline/predict.py`.

Leakage reviewer:

- Verdict: no malicious fitting / no test leakage.
- Caveat: the CSV `decision_date` must mean the prediction was available as of that date. If the prediction was produced after close, a production export should include an availability timestamp or shift dates forward.

Code reviewer:

- No evidence of same-window test leakage or profit hacking.
- Adapter bugs found and fixed: `diplom_stale_days=0` preservation and probability range validation.
- Residual protocol issues not introduced by the adapter: live paper still does not fully mirror walk-forward calibration history, neural uncertainty, and monthly-period gate semantics. Treat live paper decisions as a forward smoke runner until those parity gaps are closed.

Backtest analyst:

- Verdict: no measurable realized robustness improvement from the diploma overlay in these outputs.
- The existing nested gate/cash behavior, not the diploma overlay, explains the avoided losses.
- Broad monthly changed 35 raw rebalance decisions, but those changes occurred inside cash-vetoed windows and did not affect executed equity.
- Main trade-off: very high cash-only rate avoids many raw losses but misses large upside, especially broad monthly missed profit of 62.09%.

## Conclusion

Do not claim a profit improvement from this integration. The valuable result is the reusable, leakage-aware risk overlay with explicit diagnostics. To make it useful for May 2026 and forward live runs, the next required artifact is a fresh target-free diploma `current_predictions.csv` generated point-in-time with availability timestamps and coverage for the Kronos universe, especially `YDEX`.
