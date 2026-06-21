import argparse
import datetime as dt
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("NO_PROXY", "*")
os.environ.setdefault("no_proxy", "*")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import Kronos, KronosPredictor, KronosTokenizer
from model.kronos import calc_time_stamps
from backtest_moex_ydex_base import compute_metrics
from predict_moex_ydex_base import fetch_moex_candles


FEATURE_COLS = ["open", "high", "low", "close", "volume", "amount"]
CLOSE_IDX = FEATURE_COLS.index("close")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def freeze_for_policy_head_tuning(model: Kronos) -> int:
    for param in model.parameters():
        param.requires_grad = False

    for module in [model.head, model.dep_layer]:
        for param in module.parameters():
            param.requires_grad = True

    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def build_target_indices(df: pd.DataFrame, lookback: int, train_count: int, holdout_count: int):
    needed = lookback + train_count + holdout_count
    if len(df) < needed:
        raise RuntimeError(f"Need at least {needed} candles, got {len(df)}.")
    first_target = len(df) - train_count - holdout_count
    train_indices = list(range(first_target, first_target + train_count))
    holdout_indices = list(range(first_target + train_count, len(df)))
    return train_indices, holdout_indices


def make_training_batch(
    df: pd.DataFrame,
    target_indices,
    lookback: int,
    clip: float,
    tokenizer: KronosTokenizer,
    device: str,
):
    contexts = []
    contexts_plus_target = []
    stamps = []
    means = []
    stds = []
    actuals = []
    prev_closes = []

    for target_idx in target_indices:
        context = df.iloc[target_idx - lookback : target_idx]
        target = df.iloc[target_idx]

        x = context[FEATURE_COLS].to_numpy(dtype=np.float32)
        actual = target[FEATURE_COLS].to_numpy(dtype=np.float32)

        mean = x.mean(axis=0)
        std = x.std(axis=0)
        x_norm = np.clip((x - mean) / (std + 1e-5), -clip, clip)
        target_norm = np.clip((actual - mean) / (std + 1e-5), -clip, clip)

        contexts.append(x_norm)
        contexts_plus_target.append(np.vstack([x_norm, target_norm[None, :]]))
        stamps.append(calc_time_stamps(context["timestamps"]).to_numpy(dtype=np.float32))
        means.append(mean)
        stds.append(std)
        actuals.append(actual)
        prev_closes.append(float(x[-1, CLOSE_IDX]))

    context_tensor = torch.tensor(np.stack(contexts), dtype=torch.float32, device=device)
    full_tensor = torch.tensor(np.stack(contexts_plus_target), dtype=torch.float32, device=device)
    stamp_tensor = torch.tensor(np.stack(stamps), dtype=torch.float32, device=device)
    mean_tensor = torch.tensor(np.stack(means), dtype=torch.float32, device=device)
    std_tensor = torch.tensor(np.stack(stds), dtype=torch.float32, device=device)
    actual_tensor = torch.tensor(np.stack(actuals), dtype=torch.float32, device=device)
    prev_close_tensor = torch.tensor(prev_closes, dtype=torch.float32, device=device)

    with torch.no_grad():
        context_tokens = tokenizer.encode(context_tensor, half=True)
        full_tokens = tokenizer.encode(full_tensor, half=True)
        target_s1 = full_tokens[0][:, -1]
        target_s2 = full_tokens[1][:, -1]

    return {
        "s1": context_tokens[0],
        "s2": context_tokens[1],
        "stamp": stamp_tensor,
        "mean": mean_tensor,
        "std": std_tensor,
        "actual": actual_tensor,
        "prev_close": prev_close_tensor,
        "target_s1": target_s1,
        "target_s2": target_s2,
    }


def policy_step(
    model: Kronos,
    reference_model: Kronos,
    tokenizer: KronosTokenizer,
    batch: dict,
    return_error_weight: float,
    direction_bonus: float,
    kl_coef: float,
    entropy_coef: float,
    ce_coef: float,
):
    s1_logits, context = model.decode_s1(batch["s1"], batch["s2"], batch["stamp"])
    s1_last = s1_logits[:, -1, :]
    dist_s1 = Categorical(logits=s1_last)
    sampled_s1 = dist_s1.sample()
    logp_s1 = dist_s1.log_prob(sampled_s1)
    entropy_s1 = dist_s1.entropy()

    s2_logits = model.decode_s2(context, sampled_s1.unsqueeze(-1))[:, -1, :]
    dist_s2 = Categorical(logits=s2_logits)
    sampled_s2 = dist_s2.sample()
    logp_s2 = dist_s2.log_prob(sampled_s2)
    entropy_s2 = dist_s2.entropy()

    with torch.no_grad():
        generated_s1 = torch.cat([batch["s1"], sampled_s1.unsqueeze(-1)], dim=1)
        generated_s2 = torch.cat([batch["s2"], sampled_s2.unsqueeze(-1)], dim=1)
        decoded = tokenizer.decode([generated_s1, generated_s2], half=True)[:, -1, :]
        pred = decoded * (batch["std"] + 1e-5) + batch["mean"]

        pred_close = pred[:, CLOSE_IDX]
        actual_close = batch["actual"][:, CLOSE_IDX]
        actual_ret = actual_close / batch["prev_close"] - 1.0
        pred_ret = pred_close / batch["prev_close"] - 1.0
        return_error = torch.abs(pred_ret - actual_ret)
        sign_match = torch.sign(pred_ret) == torch.sign(actual_ret)
        direction_reward = torch.where(
            sign_match,
            torch.full_like(return_error, direction_bonus),
            torch.full_like(return_error, -direction_bonus),
        )
        reward = direction_reward - return_error_weight * return_error
        advantage = (reward - reward.mean()) / (reward.std(unbiased=False) + 1e-6)

    with torch.no_grad():
        ref_s1_logits, ref_context = reference_model.decode_s1(batch["s1"], batch["s2"], batch["stamp"])
        ref_s1_last = ref_s1_logits[:, -1, :]
        ref_s2_logits = reference_model.decode_s2(ref_context, sampled_s1.unsqueeze(-1))[:, -1, :]

    logp_all_s1 = F.log_softmax(s1_last, dim=-1)
    logp_ref_s1 = F.log_softmax(ref_s1_last, dim=-1)
    prob_s1 = logp_all_s1.exp()
    kl_s1 = (prob_s1 * (logp_all_s1 - logp_ref_s1)).sum(dim=-1)

    logp_all_s2 = F.log_softmax(s2_logits, dim=-1)
    logp_ref_s2 = F.log_softmax(ref_s2_logits, dim=-1)
    prob_s2 = logp_all_s2.exp()
    kl_s2 = (prob_s2 * (logp_all_s2 - logp_ref_s2)).sum(dim=-1)

    policy_loss = -(advantage.detach() * (logp_s1 + logp_s2)).mean()
    kl_loss = (kl_s1 + kl_s2).mean()
    entropy = (entropy_s1 + entropy_s2).mean()

    ce_loss = torch.zeros((), dtype=torch.float32, device=s1_last.device)
    if ce_coef > 0:
        teacher_s2_logits = model.decode_s2(context, batch["target_s1"].unsqueeze(-1))[:, -1, :]
        ce_loss = 0.5 * (
            F.cross_entropy(s1_last, batch["target_s1"])
            + F.cross_entropy(teacher_s2_logits, batch["target_s2"])
        )

    loss = policy_loss + kl_coef * kl_loss - entropy_coef * entropy + ce_coef * ce_loss

    stats = {
        "loss": float(loss.detach().cpu()),
        "policy_loss": float(policy_loss.detach().cpu()),
        "kl": float(kl_loss.detach().cpu()),
        "entropy": float(entropy.detach().cpu()),
        "ce": float(ce_loss.detach().cpu()),
        "reward_mean": float(reward.mean().detach().cpu()),
        "reward_std": float(reward.std(unbiased=False).detach().cpu()),
        "ret_error_mean_pct": float((return_error.mean() * 100).detach().cpu()),
        "direction_acc": float(sign_match.float().mean().detach().cpu()),
        "pred_ret_mean_pct": float((pred_ret.mean() * 100).detach().cpu()),
        "actual_ret_mean_pct": float((actual_ret.mean() * 100).detach().cpu()),
    }
    return loss, stats


def windows_from_indices(df: pd.DataFrame, target_indices, lookback: int):
    df_list = []
    x_ts_list = []
    y_ts_list = []
    actual_rows = []
    prev_closes = []
    for target_idx in target_indices:
        context = df.iloc[target_idx - lookback : target_idx].copy()
        target = df.iloc[target_idx].copy()
        df_list.append(context[FEATURE_COLS].reset_index(drop=True))
        x_ts_list.append(context["timestamps"].reset_index(drop=True))
        y_ts_list.append(pd.Series([target["timestamps"]], name="timestamps"))
        actual_rows.append(target)
        prev_closes.append(float(context["close"].iloc[-1]))

    actual_df = pd.DataFrame(actual_rows).reset_index(drop=True)
    actual_df["prev_close"] = prev_closes
    return df_list, x_ts_list, y_ts_list, actual_df


def evaluate_one_step(
    model: Kronos,
    tokenizer: KronosTokenizer,
    df: pd.DataFrame,
    target_indices,
    lookback: int,
    sample_count: int,
    device: str,
    seed: int,
    verbose: bool,
):
    set_seed(seed)
    df_list, x_ts_list, y_ts_list, actual_df = windows_from_indices(df, target_indices, lookback)
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=lookback)
    model.eval()
    pred_dfs = predictor.predict_batch(
        df_list=df_list,
        x_timestamp_list=x_ts_list,
        y_timestamp_list=y_ts_list,
        pred_len=1,
        T=1.0,
        top_k=0,
        top_p=0.9,
        sample_count=sample_count,
        verbose=verbose,
    )
    pred_rows = [pred.iloc[0] for pred in pred_dfs]
    pred_df = pd.DataFrame(pred_rows).reset_index(drop=True)

    result = pd.DataFrame({"timestamps": actual_df["timestamps"], "prev_close": actual_df["prev_close"]})
    for col in FEATURE_COLS:
        result[f"actual_{col}"] = actual_df[col].astype(float)
        result[f"pred_{col}"] = pred_df[col].astype(float)

    metrics, signal_metrics = compute_metrics(result)
    return result, metrics, signal_metrics


def train_rl(args):
    set_seed(args.seed)
    device = choose_device(args.device)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but torch.cuda.is_available() is False.")
    if args.threads > 0:
        torch.set_num_threads(args.threads)

    output_dir = ROOT / "outputs" / f"{args.secid.lower()}_rl_finetune_{dt.datetime.now():%Y%m%d_%H%M%S}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    if device.startswith("cuda"):
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"Fetching MOEX daily candles for {args.secid} ({args.board})...")
    df = fetch_moex_candles(args.secid, args.board, args.days)
    df.to_csv(output_dir / f"{args.secid.lower()}_moex_daily.csv", index=False)

    train_indices, holdout_indices = build_target_indices(df, args.lookback, args.train_count, args.holdout_count)
    print(
        f"Train targets: {df.iloc[train_indices[0]]['timestamps']} -> {df.iloc[train_indices[-1]]['timestamps']} "
        f"({len(train_indices)} rows)"
    )
    print(
        f"Holdout targets: {df.iloc[holdout_indices[0]]['timestamps']} -> {df.iloc[holdout_indices[-1]]['timestamps']} "
        f"({len(holdout_indices)} rows)"
    )

    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = KronosTokenizer.from_pretrained(args.tokenizer).to(device).eval()

    print(f"Loading policy model: {args.model}")
    model = Kronos.from_pretrained(args.model).to(device)
    print(f"Loading frozen reference model: {args.model}")
    reference_model = Kronos.from_pretrained(args.model).to(device).eval()
    for param in reference_model.parameters():
        param.requires_grad = False

    trainable_params = freeze_for_policy_head_tuning(model)
    print(f"Trainable parameters: {trainable_params:,}")
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    rng = np.random.default_rng(args.seed)
    train_log = []
    model.train()
    for step in range(1, args.steps + 1):
        batch_indices = rng.choice(train_indices, size=args.batch_size, replace=True)
        batch = make_training_batch(
            df=df,
            target_indices=batch_indices,
            lookback=args.lookback,
            clip=args.clip,
            tokenizer=tokenizer,
            device=device,
        )

        loss, stats = policy_step(
            model=model,
            reference_model=reference_model,
            tokenizer=tokenizer,
            batch=batch,
            return_error_weight=args.return_error_weight,
            direction_bonus=args.direction_bonus,
            kl_coef=args.kl_coef,
            entropy_coef=args.entropy_coef,
            ce_coef=args.ce_coef,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.grad_clip)
        optimizer.step()

        stats["step"] = step
        train_log.append(stats)
        if step == 1 or step % args.log_interval == 0 or step == args.steps:
            print(
                f"step {step:04d}/{args.steps} "
                f"loss={stats['loss']:.4f} reward={stats['reward_mean']:.4f} "
                f"ret_err={stats['ret_error_mean_pct']:.3f}% dir={stats['direction_acc']:.2f} "
                f"kl={stats['kl']:.5f} ce={stats['ce']:.4f}"
            )

    pd.DataFrame(train_log).to_csv(output_dir / "rl_train_log.csv", index=False)

    checkpoint_dir = output_dir / "kronos_base_ydex_rl_head"
    model.save_pretrained(checkpoint_dir)
    print(f"Saved RL checkpoint: {checkpoint_dir}")

    print("Evaluating baseline on holdout...")
    baseline_result, baseline_metrics, baseline_signal = evaluate_one_step(
        model=reference_model,
        tokenizer=tokenizer,
        df=df,
        target_indices=holdout_indices,
        lookback=args.lookback,
        sample_count=args.eval_sample_count,
        device=device,
        seed=args.seed + 100,
        verbose=True,
    )

    print("Evaluating RL-finetuned model on holdout...")
    rl_result, rl_metrics, rl_signal = evaluate_one_step(
        model=model,
        tokenizer=tokenizer,
        df=df,
        target_indices=holdout_indices,
        lookback=args.lookback,
        sample_count=args.eval_sample_count,
        device=device,
        seed=args.seed + 100,
        verbose=True,
    )

    baseline_result.to_csv(output_dir / "baseline_holdout_predictions.csv", index=False)
    baseline_metrics.to_csv(output_dir / "baseline_holdout_metrics.csv", index=False)
    baseline_signal.to_csv(output_dir / "baseline_holdout_signal_metrics.csv", index=False)
    rl_result.to_csv(output_dir / "rl_holdout_predictions.csv", index=False)
    rl_metrics.to_csv(output_dir / "rl_holdout_metrics.csv", index=False)
    rl_signal.to_csv(output_dir / "rl_holdout_signal_metrics.csv", index=False)

    summary_rows = []
    for label, metrics, signal in [
        ("baseline", baseline_metrics, baseline_signal),
        ("rl", rl_metrics, rl_signal),
    ]:
        close_row = metrics.loc[metrics["series"] == "close"].iloc[0].to_dict()
        signal_dict = dict(zip(signal["metric"], signal["value"]))
        summary_rows.append(
            {
                "model": label,
                "close_mae": close_row["mae"],
                "close_rmse": close_row["rmse"],
                "close_mape_pct": close_row["mape_pct"],
                "close_bias": close_row["bias"],
                "close_r2": close_row["r2"],
                "return_mae_pct": signal_dict["close_return_mae_pct"],
                "return_rmse_pct": signal_dict["close_return_rmse_pct"],
                "return_bias_pct": signal_dict["close_return_bias_pct"],
                "return_pearson": signal_dict["close_return_pearson"],
                "direction_accuracy": signal_dict["direction_accuracy"],
                "long_short_signal_return_pct": signal_dict["long_short_signal_return_pct"],
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "holdout_comparison_summary.csv", index=False)

    summary_txt = [
        f"device: {device}",
        f"model: {args.model}",
        f"tokenizer: {args.tokenizer}",
        f"train_count: {args.train_count}",
        f"holdout_count: {args.holdout_count}",
        f"lookback: {args.lookback}",
        f"steps: {args.steps}",
        f"batch_size: {args.batch_size}",
        f"trainable_params: {trainable_params}",
        f"checkpoint: {checkpoint_dir}",
        "",
        summary_df.to_string(index=False),
    ]
    (output_dir / "summary.txt").write_text("\n".join(summary_txt), encoding="utf-8")

    print("\nHoldout comparison:")
    print(summary_df.to_string(index=False))
    print(f"\nOutput directory: {output_dir}")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="RL-style policy-gradient fine-tuning for Kronos on MOEX YDEX.")
    parser.add_argument("--secid", default="YDEX")
    parser.add_argument("--board", default="TQBR")
    parser.add_argument("--days", type=int, default=1200)
    parser.add_argument("--lookback", type=int, default=256)
    parser.add_argument("--train-count", type=int, default=100)
    parser.add_argument("--holdout-count", type=int, default=50)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--clip", type=float, default=5.0)
    parser.add_argument("--return-error-weight", type=float, default=100.0)
    parser.add_argument("--direction-bonus", type=float, default=0.25)
    parser.add_argument("--kl-coef", type=float, default=0.05)
    parser.add_argument("--entropy-coef", type=float, default=0.001)
    parser.add_argument("--ce-coef", type=float, default=0.02)
    parser.add_argument("--eval-sample-count", type=int, default=5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--model", default="NeoQuasar/Kronos-base")
    parser.add_argument("--tokenizer", default="NeoQuasar/Kronos-Tokenizer-base")
    args = parser.parse_args()
    train_rl(args)


if __name__ == "__main__":
    main()
