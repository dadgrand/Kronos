# Kronos Production Runbook

This runtime is fail-closed. Real Binance Spot orders are blocked unless the
operator explicitly configures live mode, provides credentials, clears the kill
switch, and passes the 30-day shadow gate plus model approval checks.

## Setup

1. Copy `prod/config.example.json` to a private run config outside commits.
2. Keep `mode` as `shadow` until the shadow gate passes.
3. Set `KRONOS_RUN_DIR` or `run_dir` to a durable local disk path.
4. For CUDA inference, set both `device` and `cuda_visible_devices`; otherwise
   the runtime refuses CUDA to avoid touching unrelated GPU processes.
5. For Binance live preflight, set `BINANCE_API_KEY` and `BINANCE_SECRET_KEY`.
   Use least-privilege keys and revoke immediately on abnormal activity.

## Shadow Period

Run shadow ticks on the local machine:

```powershell
python -m prod.cli --config path\to\prod.json shadow-run
python -m prod.cli --config path\to\prod.json status
```

The default shadow command uses a hold-last-close smoke predictor. Use real
Kronos inference explicitly:

```powershell
python -m prod.cli --config path\to\prod.json shadow-run --predictor kronos
```

The generated `prediction_results.jsonl` is checksum-chained. Do not edit it.
Promotion requires at least 30 days of forward-only evidence across all six v1
symbols.

## Canary And Live Gates

Before changing `mode` to `canary_live` or `live`:

1. Configure `max_order_notional`.
2. Ensure the approval registry accepts the exact `model_hash`.
3. Run `python -m prod.cli --config path\to\prod.json live-preflight`.
4. Set `KRONOS_ENABLE_LIVE_TRADING=1` only for the operator shell/service that
   should be able to trade.

## Incident Actions

Disable live trading:

```powershell
python -m prod.cli --config path\to\prod.json disable-live --reason "incident text"
```

This activates the file kill switch and writes `live_state.json`. Do not clear
the kill switch until account state, open orders, fills, and the local journal
have been reconciled.

## Restore From Journal

1. Verify `orders.jsonl` checksum chain by running status/preflight.
2. Fetch Binance account snapshot with `reconcile`.
3. Compare open orders and positions against dashboard state.
4. If unmanaged external orders exist, cancel them manually or through a
   reviewed adapter path before resuming.
