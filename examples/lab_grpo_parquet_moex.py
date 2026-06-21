import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import Kronos, KronosTokenizer
from lab_grpo_moex import (
    build_splits,
    evaluate_samples,
    freeze_and_lora,
    grpo_step,
    load_trainable_state,
    metric_score,
    set_seed,
    trainable_state_dict,
)


FEATURE_COLS = ["open", "high", "low", "close", "volume", "amount"]


def choose_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def read_symbols(dataset_dir: Path, requested_symbols: str | None, top_n: int) -> list[str]:
    if requested_symbols:
        symbols = [item.strip().upper() for item in requested_symbols.split(",") if item.strip()]
    else:
        manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
        symbols = [item.upper() for item in manifest["symbols"]]
    if top_n > 0:
        symbols = symbols[:top_n]
    return symbols


def load_interval_frames(dataset_dir: Path, interval: int, symbols: list[str], min_rows: int) -> dict[str, pd.DataFrame]:
    frames = {}
    interval_dir = dataset_dir / "curated" / f"interval={interval}"
    columns = ["symbol", "timestamp", *FEATURE_COLS]
    for idx, symbol in enumerate(symbols, start=1):
        symbol_dir = interval_dir / f"symbol={symbol}"
        files = sorted(symbol_dir.rglob("candles_curated.parquet"))
        if not files:
            print(f"skip {symbol}: no parquet files")
            continue
        parts = []
        for file in files:
            try:
                part = pd.read_parquet(file, columns=columns)
            except Exception:
                part = pd.read_parquet(file)
                if not {"symbol", "timestamp", *FEATURE_COLS}.issubset(part.columns):
                    continue
                part = part[columns]
            if not part.empty:
                parts.append(part)
        if not parts:
            print(f"skip {symbol}: all chunks empty")
            continue
        df = pd.concat(parts, ignore_index=True)
        df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
        df = df.rename(columns={"timestamp": "timestamps"})
        df = df.dropna(subset=FEATURE_COLS + ["timestamps"])
        if len(df) < min_rows:
            print(f"skip {symbol}: {len(df)} rows < {min_rows}")
            continue
        frames[symbol] = df
        print(f"[{idx}/{len(symbols)}] {symbol}: {len(df):,} rows {df['timestamps'].iloc[0]} -> {df['timestamps'].iloc[-1]}")
    if not frames:
        raise RuntimeError("No usable frames loaded.")
    return frames


def make_batch_from_samples(frames, samples, lookback, clip, tokenizer, device):
    from lab_grpo_moex import make_batch

    return make_batch(frames, samples, lookback, clip, tokenizer, device)


def summarize_model(label: str, metrics: pd.DataFrame, signal: pd.DataFrame, portfolio: dict) -> dict:
    close = metrics.loc[metrics["series"] == "close"].iloc[0].to_dict()
    signal_dict = dict(zip(signal["metric"], signal["value"]))
    return {
        "model": label,
        "close_mae": close["mae"],
        "close_rmse": close["rmse"],
        "close_mape_pct": close["mape_pct"],
        "close_bias": close["bias"],
        "return_mae_pct": signal_dict["close_return_mae_pct"],
        "return_pearson": signal_dict["close_return_pearson"],
        "direction_accuracy": signal_dict["direction_accuracy"],
        "portfolio_final_equity": portfolio["final_equity"],
        "portfolio_pnl": portfolio["pnl"],
        "portfolio_return_pct": portfolio["return_pct"],
        "portfolio_buy_hold_return_pct": portfolio["buy_hold_return_pct"],
        "portfolio_max_drawdown_pct": portfolio["max_drawdown_pct"],
        "portfolio_sharpe_like": portfolio["daily_sharpe_like"],
        "active_rate": portfolio["active_rate"],
        "long_rate": portfolio["long_rate"],
        "short_rate": portfolio["short_rate"],
    }


def by_symbol_report(result: pd.DataFrame) -> pd.DataFrame:
    rows = []
    df = result.copy()
    df["actual_return"] = df["actual_close"] / df["prev_close"] - 1.0
    df["pred_return"] = df["pred_close"] / df["prev_close"] - 1.0
    for symbol, group in df.groupby("symbol"):
        err = group["pred_close"] - group["actual_close"]
        ret_err = group["pred_return"] - group["actual_return"]
        pos = np.where(group["pred_return"] > 0, 1.0, np.where(group["pred_return"] < 0, -1.0, 0.0))
        trade_ret = pos * group["actual_return"] - np.abs(pos) * 0.001
        rows.append(
            {
                "symbol": symbol,
                "rows": len(group),
                "close_mae": float(err.abs().mean()),
                "return_mae_pct": float(ret_err.abs().mean() * 100),
                "direction_accuracy": float((np.sign(group["pred_return"]) == np.sign(group["actual_return"])).mean()),
                "signal_return_pct": float((np.prod(1 + trade_ret) - 1) * 100),
                "buy_hold_pct": float((np.prod(1 + group["actual_return"]) - 1) * 100),
                "pred_up_rate": float((group["pred_return"] > 0).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("symbol")


def main() -> None:
    parser = argparse.ArgumentParser(description="GRPO-LoRA training on partitioned MOEX Parquet intraday dataset.")
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "datasets" / "moex_intraday_6f8f836cc811")
    parser.add_argument("--interval", type=int, default=10)
    parser.add_argument("--symbols", default=None)
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--lookback", type=int, default=256)
    parser.add_argument("--val-count", type=int, default=200)
    parser.add_argument("--test-count", type=int, default=300)
    parser.add_argument("--max-train-per-symbol", type=int, default=12000)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=6e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--last-n-layers", type=int, default=4)
    parser.add_argument("--clip", type=float, default=5.0)
    parser.add_argument("--reward-trade-weight", type=float, default=120.0)
    parser.add_argument("--reward-error-weight", type=float, default=20.0)
    parser.add_argument("--reward-direction-bonus", type=float, default=0.08)
    parser.add_argument("--signal-threshold", type=float, default=0.0005)
    parser.add_argument("--round-trip-cost-bps", type=float, default=10.0)
    parser.add_argument("--kl-coef", type=float, default=0.04)
    parser.add_argument("--entropy-coef", type=float, default=0.001)
    parser.add_argument("--ce-coef", type=float, default=0.02)
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--eval-sample-count", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--validation-objective", default="portfolio_return", choices=["portfolio_return", "negative_return_mae"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260622)
    parser.add_argument("--model", default="NeoQuasar/Kronos-base")
    parser.add_argument("--tokenizer", default="NeoQuasar/Kronos-Tokenizer-base")
    args = parser.parse_args()

    set_seed(args.seed)
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    device = choose_device(args.device)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")

    output_dir = ROOT / "outputs" / f"lab_grpo_parquet_i{args.interval}_{dt.datetime.now():%Y%m%d_%H%M%S}"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")

    print(f"Device: {device}")
    if device.startswith("cuda"):
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    symbols = read_symbols(args.dataset_dir, args.symbols, args.top_n)
    min_rows = args.lookback + args.val_count + args.test_count + 20
    frames = load_interval_frames(args.dataset_dir, args.interval, symbols, min_rows)

    train_samples, val_samples, test_samples, split_df = build_splits(
        frames,
        lookback=args.lookback,
        val_count=args.val_count,
        test_count=args.test_count,
        max_train_per_symbol=args.max_train_per_symbol,
        seed=args.seed,
    )
    split_df.to_csv(output_dir / "splits.csv", index=False)
    print(split_df.to_string(index=False))
    print(f"Samples: train={len(train_samples):,}, val={len(val_samples):,}, test={len(test_samples):,}")

    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = KronosTokenizer.from_pretrained(args.tokenizer).to(device).eval()
    print(f"Loading reference model: {args.model}")
    reference_model = Kronos.from_pretrained(args.model).to(device).eval()
    for param in reference_model.parameters():
        param.requires_grad = False
    print(f"Loading policy model: {args.model}")
    policy_model = Kronos.from_pretrained(args.model).to(device)
    trainable_params, wrapped_modules = freeze_and_lora(
        policy_model,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        last_n_layers=args.last_n_layers,
    )
    policy_model.to(device)
    (output_dir / "lora_wrapped_modules.txt").write_text("\n".join(wrapped_modules), encoding="utf-8")
    print(f"LoRA modules={len(wrapped_modules)}, trainable params={trainable_params:,}")

    optimizer = torch.optim.AdamW(
        [param for param in policy_model.parameters() if param.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=args.learning_rate * 0.1)
    rng = np.random.default_rng(args.seed)
    train_log = []
    val_log = []
    best_score = -float("inf")
    best_step = 0
    best_state = trainable_state_dict(policy_model)

    policy_model.train()
    for step in range(1, args.steps + 1):
        chosen = rng.choice(len(train_samples), size=args.batch_size, replace=True)
        batch_refs = [train_samples[int(i)] for i in chosen]
        batch = make_batch_from_samples(frames, batch_refs, args.lookback, args.clip, tokenizer, device)
        loss, stats = grpo_step(
            model=policy_model,
            reference_model=reference_model,
            tokenizer=tokenizer,
            batch=batch,
            group_size=args.group_size,
            reward_trade_weight=args.reward_trade_weight,
            reward_error_weight=args.reward_error_weight,
            reward_direction_bonus=args.reward_direction_bonus,
            signal_threshold=args.signal_threshold,
            round_trip_cost_bps=args.round_trip_cost_bps,
            kl_coef=args.kl_coef,
            entropy_coef=args.entropy_coef,
            ce_coef=args.ce_coef,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in policy_model.parameters() if p.requires_grad], args.grad_clip)
        optimizer.step()
        scheduler.step()

        stats["step"] = step
        stats["lr"] = scheduler.get_last_lr()[0]
        train_log.append(stats)

        if step == 1 or step % 50 == 0:
            print(
                f"step {step:04d}/{args.steps} loss={stats['loss']:.4f} reward={stats['reward_mean']:.4f} "
                f"trade_bps={stats['trade_reward_mean_bps']:.2f} err={stats['ret_error_mean_pct']:.3f}% "
                f"dir={stats['direction_acc']:.2f} kl={stats['kl']:.4f}"
            )

        if step % args.eval_interval == 0 or step == args.steps:
            print(f"Validation step {step}...")
            val_result, val_metrics, _, val_portfolio = evaluate_samples(
                policy_model,
                tokenizer,
                frames,
                val_samples,
                args.lookback,
                device,
                args.eval_sample_count,
                args.eval_batch_size,
                seed=args.seed + step,
            )
            score = metric_score(val_metrics, val_portfolio, args.validation_objective)
            val_row = {"step": step, "score": score, **val_portfolio}
            val_log.append(val_row)
            print(
                f"  val score={score:.4f} return={val_portfolio['return_pct']:.2f}% "
                f"bh={val_portfolio['buy_hold_return_pct']:.2f}% dd={val_portfolio['max_drawdown_pct']:.2f}%"
            )
            if score > best_score:
                best_score = score
                best_step = step
                best_state = trainable_state_dict(policy_model)
                val_result.to_csv(output_dir / "best_validation_predictions.csv", index=False)
                print(f"  new best step {best_step}")
            policy_model.train()

    pd.DataFrame(train_log).to_csv(output_dir / "train_log.csv", index=False)
    pd.DataFrame(val_log).to_csv(output_dir / "validation_log.csv", index=False)
    torch.save(best_state, output_dir / "best_lora_adapter.pt")
    load_trainable_state(policy_model, best_state)

    print(f"Best validation step={best_step}, score={best_score:.4f}")
    print("Evaluating baseline on test...")
    baseline_result, baseline_metrics, baseline_signal, baseline_portfolio = evaluate_samples(
        reference_model,
        tokenizer,
        frames,
        test_samples,
        args.lookback,
        device,
        args.eval_sample_count,
        args.eval_batch_size,
        seed=args.seed + 1000,
    )
    print("Evaluating tuned policy on test...")
    tuned_result, tuned_metrics, tuned_signal, tuned_portfolio = evaluate_samples(
        policy_model,
        tokenizer,
        frames,
        test_samples,
        args.lookback,
        device,
        args.eval_sample_count,
        args.eval_batch_size,
        seed=args.seed + 1000,
    )

    baseline_result.to_csv(output_dir / "baseline_test_predictions.csv", index=False)
    tuned_result.to_csv(output_dir / "tuned_test_predictions.csv", index=False)
    baseline_metrics.to_csv(output_dir / "baseline_test_metrics.csv", index=False)
    tuned_metrics.to_csv(output_dir / "tuned_test_metrics.csv", index=False)
    baseline_signal.to_csv(output_dir / "baseline_test_signal_metrics.csv", index=False)
    tuned_signal.to_csv(output_dir / "tuned_test_signal_metrics.csv", index=False)
    pd.DataFrame([baseline_portfolio]).to_csv(output_dir / "baseline_test_portfolio.csv", index=False)
    pd.DataFrame([tuned_portfolio]).to_csv(output_dir / "tuned_test_portfolio.csv", index=False)

    baseline_by_symbol = by_symbol_report(baseline_result)
    tuned_by_symbol = by_symbol_report(tuned_result)
    baseline_by_symbol.insert(0, "model", "baseline")
    tuned_by_symbol.insert(0, "model", "grpo_lora")
    pd.concat([baseline_by_symbol, tuned_by_symbol], ignore_index=True).to_csv(output_dir / "test_by_symbol.csv", index=False)

    comparison = pd.DataFrame(
        [
            summarize_model("baseline", baseline_metrics, baseline_signal, baseline_portfolio),
            summarize_model("grpo_lora", tuned_metrics, tuned_signal, tuned_portfolio),
        ]
    )
    comparison.to_csv(output_dir / "test_comparison.csv", index=False)
    summary = {
        "device": device,
        "dataset_dir": str(args.dataset_dir),
        "interval": args.interval,
        "symbols": list(frames.keys()),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "test_samples": len(test_samples),
        "best_step": best_step,
        "best_score": best_score,
        "trainable_params": trainable_params,
        "output_dir": str(output_dir),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "summary.txt").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n\n" + comparison.to_string(index=False),
        encoding="utf-8",
    )
    print("\nTest comparison:")
    print(comparison.to_string(index=False))
    print(f"\nOutput directory: {output_dir}")


if __name__ == "__main__":
    main()
