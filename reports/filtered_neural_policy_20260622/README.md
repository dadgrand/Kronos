# Filtered neural policy report

Superseded: this single-seed report was generated before the stricter ensemble report in
`reports/filtered_neural_policy_ensemble_20260622`. Keep it as exploratory history only.
The ensemble report is the current reference result.

This report summarizes the best research result from the overnight MOEX/Kronos trading-bot improvement cycle.

## Protocol

- Dataset: `datasets/moex_intraday_6f8f836cc811`
- Interval: 10 minute candles
- Universe: 50 TQBR symbols from the local MOEX dataset
- Training period: 2022-09-01 through 2026-06-01
- Validation period: 2026-06-01 through 2026-06-13
- Test period: after 2026-06-13
- Initial cash: 10,000 RUB
- Trading cost: 10 bps per unit turnover
- Neural policy model: causal feature MLP with symbol embeddings
- Selection rule: checkpoint, base policy, and regime-filter thresholds are selected on validation only

Commands:

```powershell
python examples/neural_policy_lab.py --device cuda:0 --horizon 12 --train-end 2026-06-01 --validation-end 2026-06-13 --epochs 8 --steps-per-epoch 250 --batch-size 8192 --max-gross 4.0 --cost-bps 10

python examples/filtered_neural_policy_lab.py --scores-path outputs/neural_policy_lab_h12_20260622_031344/scores.npy --train-end 2026-06-01 --validation-end 2026-06-13 --mode short_only --k 1 --gross 4.0 --rebalance-every 24 --cost-bps 10
```

## Selected Policy

```json
{
  "mode": "short_only",
  "k": 1,
  "gross": 4.0,
  "rebalance_every": 24,
  "confidence_min": 0.09412878006696701,
  "market_mom_24_max": 0.002567113547411282,
  "market_mom_96_max": 0.000038395675483115644
}
```

The policy shorts the single lowest-scored symbol with 4x gross exposure, but only when validation-selected confidence and market-regime filters pass.

## Result

| Split | Final equity | PnL | Return | Max drawdown | Active rate | Avg turnover/bar |
|---|---:|---:|---:|---:|---:|---:|
| Validation | 18,537.74 | 8,537.74 | 85.38% | -8.07% | 42.11% | 0.1404 |
| Test | 13,967.83 | 3,967.83 | 39.68% | -18.54% | 42.58% | 0.1419 |

The 30% target was reached on this holdout test: 10,000 RUB became 13,967.83 RUB.

## Cost Sensitivity

| Cost bps | Test return |
|---:|---:|
| 5 | 46.04% |
| 10 | 39.68% |
| 20 | 27.72% |
| 30 | 16.71% |

The result is sensitive to transaction-cost assumptions. At 20 bps, the strategy no longer reaches 30%.

## Honest Limitations

- This is not a production trading claim. It is a research backtest on known historical data.
- The strategy is aggressive: `short_only`, one-symbol concentration, 4x gross exposure.
- The universe still has survivorship/liquidity lookahead risk because it uses the local 50-symbol MOEX dataset manifest.
- The filter family was introduced during the research cycle after earlier weaker test results. Thresholds are selected by validation only, but a future untouched lockbox is still required before treating the result as robust.
- Short availability, borrow cost, market-impact, order-book spread, and exchange-specific execution constraints are not modeled.

This is research output, not financial advice or a live trading recommendation.
