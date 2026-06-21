# MOEX RL fine-tuning report

This report summarizes the local Kronos-base RL fine-tuning experiment on MOEX intraday candles.

## Dataset

- Source: MOEX ISS candles
- Period: from 2022-09-01 to 2026-06-21
- Symbols: 50 liquid TQBR names, including YDEX
- Intervals downloaded: 1 minute and 10 minutes
- Training interval used here: 10 minutes
- Local dataset path: `datasets/moex_intraday_6f8f836cc811`
- Total downloaded rows: 41,993,682
- Rows at 10 minute interval: 4,114,588

The raw/curated dataset is intentionally not tracked by git. It is reproducible through:

```powershell
python examples/build_moex_intraday_partitions.py --from 2022-09-01 --to 2026-06-21 --intervals 1,10
```

## Training setup

- Base model: `kronos-base`
- Device: CUDA GPU
- Fine-tuning method: LoRA on the last 4 transformer blocks and output heads
- RL style: grouped reward optimization with KL/reference and supervised regularization
- Trainable parameters: 572,416
- Train samples: 749,833
- Validation samples: 15,000
- Test samples: 30,000
- Training steps: 600
- Best validation checkpoint: step 200

Command:

```powershell
python examples/lab_grpo_parquet_moex.py --device cuda:0 --interval 10 --top-n 50 --lookback 256 --val-count 300 --test-count 600 --max-train-per-symbol 15000 --steps 600 --batch-size 32 --group-size 4 --eval-interval 200 --eval-sample-count 2 --eval-batch-size 256 --last-n-layers 4 --lora-rank 8 --learning-rate 6e-5
```

## Forecast test results

| Model | Close MAE | Close RMSE | Close MAPE | Return MAE | Return Pearson | Direction accuracy |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 1.118567 | 3.208933 | 0.182182% | 0.182086% | 0.040295 | 47.8633% |
| grpo_lora | 1.104379 | 3.184731 | 0.179869% | 0.179769% | 0.036020 | 48.0033% |

The tuned model improved price regression metrics slightly, but did not create a strong directional edge.

## Trading test

Initial capital: 10,000 RUB.

The naive per-bar strategy loses more than 43% because it pays trading costs on every 10 minute bar. A more realistic persistent-position analysis pays costs mainly when the position changes.

Persistent-position result:

| Model | Threshold | Final equity | PnL | Return | No-cost return | Buy-and-hold | Avg turnover/bar |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.0005 | 6,821.51 | -3,178.49 | -31.78% | 4.17% | -3.31% | 0.6740 |
| grpo_lora | 0.0005 | 6,833.60 | -3,166.40 | -31.66% | 3.84% | -3.31% | 0.6660 |
| baseline | 0.0010 | 7,762.65 | -2,237.35 | -22.37% | 2.72% | -3.31% | 0.4460 |
| grpo_lora | 0.0010 | 7,831.28 | -2,168.72 | -21.69% | 2.16% | -3.31% | 0.4232 |
| baseline | 0.0020 | 9,018.82 | -981.18 | -9.81% | 1.08% | -3.31% | 0.1816 |
| grpo_lora | 0.0020 | 9,078.79 | -921.21 | -9.21% | 0.58% | -3.31% | 0.1631 |

## Conclusion

The RL run produced a measurable but small improvement in forecast accuracy. It did not produce a profitable trading strategy after costs. The main failure mode is excessive turnover: the gross signal is weakly positive in some configurations, but transaction costs consume it.

The next research step should train a true policy rather than a next-price forecaster: include current position in the state, use explicit hold/flat/long/short actions, optimize a turnover-aware net portfolio reward, and select checkpoints by validation Sharpe/Calmar or drawdown-adjusted return rather than price error.

This is a research result, not financial advice and not a live trading recommendation.
