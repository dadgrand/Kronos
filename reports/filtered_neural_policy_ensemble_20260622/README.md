# Filtered neural policy ensemble report

This report is the current best strict research result after the reviewer-flagged issues were addressed.

## Protocol

- Dataset: `datasets/moex_intraday_6f8f836cc811`
- Interval: 10 minute MOEX candles
- Training period: 2022-09-01 through 2026-06-01
- Validation period: 2026-06-01 through 2026-06-13
- Test period: after 2026-06-13
- Initial cash: 10,000 RUB
- Trading cost: 10 bps per unit turnover
- Neural policy model: causal feature MLP with symbol embeddings
- Ensemble: equal average of three independently trained seed score matrices
- Selection: filter thresholds are selected on validation only; test is evaluated once after selection

Anti-leakage fixes included in this version:

- Training targets are masked so horizon labels do not cross train/validation/test boundaries.
- Trade eligibility uses previous-bar tradability instead of knowing the current bar's volume/close.
- Features use shifted causal inputs such as prior momentum, prior volatility, and prior volume.

Commands:

```powershell
python examples/neural_policy_lab.py --device cuda:0 --horizon 12 --train-end 2026-06-01 --validation-end 2026-06-13 --epochs 8 --steps-per-epoch 250 --batch-size 8192 --max-gross 4.0 --cost-bps 10 --seed 20260622
python examples/neural_policy_lab.py --device cuda:0 --horizon 12 --train-end 2026-06-01 --validation-end 2026-06-13 --epochs 8 --steps-per-epoch 250 --batch-size 8192 --max-gross 4.0 --cost-bps 10 --seed 20260623
python examples/neural_policy_lab.py --device cuda:0 --horizon 12 --train-end 2026-06-01 --validation-end 2026-06-13 --epochs 8 --steps-per-epoch 250 --batch-size 8192 --max-gross 4.0 --cost-bps 10 --seed 20260624

python examples/filtered_neural_policy_lab.py --scores-path outputs/neural_policy_lab_h12_seed20260622_20260622_132851/scores.npy outputs/neural_policy_lab_h12_seed20260623_20260622_133133/scores.npy outputs/neural_policy_lab_h12_seed20260624_20260622_133615/scores.npy --train-end 2026-06-01 --validation-end 2026-06-13 --mode short_only --k 1 --gross 4.0 --rebalance-every 24 --cost-bps 10
```

## Selected Policy

```json
{
  "mode": "short_only",
  "k": 1,
  "gross": 4.0,
  "rebalance_every": 24,
  "confidence_min": 0.07741670683026314,
  "market_mom_24_max": 0.002567113547411282,
  "market_mom_96_max": 0.000038395675483115644
}
```

## Result

| Split | Final equity | PnL | Return | Max drawdown | Active rate | Avg turnover/bar |
|---|---:|---:|---:|---:|---:|---:|
| Validation | 18,757.80 | 8,757.80 | 87.58% | -8.23% | 39.89% | 0.1256 |
| Test | 13,483.02 | 3,483.02 | 34.83% | -18.54% | 46.45% | 0.1548 |

The 30% target was reached in this research holdout: 10,000 RUB became 13,483.02 RUB.

## Cost Sensitivity

| Cost bps | Test return |
|---:|---:|
| 5 | 41.54% |
| 10 | 34.83% |
| 20 | 22.30% |
| 30 | 10.86% |

The result clears 30% at 10 bps, but not at 20 bps. This remains a major execution-risk caveat.

## Limitations

- This is a research backtest, not a live trading recommendation.
- The strategy is aggressive: short-only, one-symbol concentration, and 4x gross exposure.
- Borrow availability, borrow cost, hard-to-borrow constraints, margin liquidation, order-book spread, market impact, and intrabar execution slippage are not modeled.
- The 50-symbol universe may still contain survivorship/liquidity lookahead because it comes from the local dataset manifest.
- The filter family was developed during the broader research cycle, so the June test should not be treated as a pristine independent lockbox. A true forward test after freezing this protocol is still required before any production claim.
