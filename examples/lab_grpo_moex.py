import argparse
import copy
import datetime as dt
import json
import math
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("NO_PROXY", "*")
os.environ.setdefault("no_proxy", "*")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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


@dataclass(frozen=True)
class SampleRef:
    symbol: str
    target_idx: int


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        self.base = base
        for param in self.base.parameters():
            param.requires_grad = False

        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scale = alpha / rank

        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x):
        return self.base(x) + self.lora_b(self.dropout(self.lora_a(x))) * self.scale


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


def should_lora_wrap(module_name: str, model: Kronos, last_n_layers: int) -> bool:
    if module_name.startswith("head.") or module_name.startswith("dep_layer."):
        return True
    if module_name.startswith("transformer."):
        parts = module_name.split(".")
        if len(parts) >= 2 and parts[1].isdigit():
            layer_idx = int(parts[1])
            return layer_idx >= max(0, model.n_layers - last_n_layers)
    return False


def inject_lora(module: nn.Module, root_model: Kronos, rank: int, alpha: float, dropout: float, last_n_layers: int, prefix: str = "") -> list[str]:
    wrapped = []
    for child_name, child in list(module.named_children()):
        full_name = f"{prefix}.{child_name}" if prefix else child_name
        if isinstance(child, nn.Linear) and should_lora_wrap(full_name, root_model, last_n_layers):
            setattr(module, child_name, LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout))
            wrapped.append(full_name)
        else:
            wrapped.extend(inject_lora(child, root_model, rank, alpha, dropout, last_n_layers, full_name))
    return wrapped


def freeze_and_lora(model: Kronos, rank: int, alpha: float, dropout: float, last_n_layers: int) -> tuple[int, list[str]]:
    for param in model.parameters():
        param.requires_grad = False
    wrapped = inject_lora(model, model, rank, alpha, dropout, last_n_layers)
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return trainable, wrapped


def trainable_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    trainable_names = {name for name, param in model.named_parameters() if param.requires_grad}
    state = model.state_dict()
    return {name: tensor.detach().cpu().clone() for name, tensor in state.items() if name in trainable_names}


def load_trainable_state(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    current = model.state_dict()
    current.update({name: tensor.to(current[name].device, dtype=current[name].dtype) for name, tensor in state.items()})
    model.load_state_dict(current)


def parse_symbols(symbols_arg: str) -> list[str]:
    return [symbol.strip().upper() for symbol in symbols_arg.split(",") if symbol.strip()]


def fetch_universe(symbols: list[str], board: str, days: int, min_rows: int, output_dir: Path) -> dict[str, pd.DataFrame]:
    frames = {}
    raw_dir = output_dir / "raw_moex"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for symbol in symbols:
        try:
            print(f"Fetching {symbol}...")
            df = fetch_moex_candles(symbol, board, days)
            if len(df) < min_rows:
                print(f"  skip {symbol}: only {len(df)} rows, need {min_rows}")
                continue
            df.to_csv(raw_dir / f"{symbol.lower()}_daily.csv", index=False)
            frames[symbol] = df
            print(f"  ok {symbol}: {len(df)} rows, {df['timestamps'].iloc[0]} -> {df['timestamps'].iloc[-1]}")
        except Exception as exc:
            print(f"  skip {symbol}: {exc}")
    if not frames:
        raise RuntimeError("No usable symbols fetched.")
    return frames


def build_splits(
    frames: dict[str, pd.DataFrame],
    lookback: int,
    val_count: int,
    test_count: int,
    max_train_per_symbol: int,
    seed: int,
) -> tuple[list[SampleRef], list[SampleRef], list[SampleRef], pd.DataFrame]:
    rng = np.random.default_rng(seed)
    train, val, test = [], [], []
    rows = []
    for symbol, df in frames.items():
        targets = list(range(lookback, len(df)))
        if len(targets) <= val_count + test_count + 5:
            continue
        train_targets = targets[: -(val_count + test_count)]
        val_targets = targets[-(val_count + test_count) : -test_count]
        test_targets = targets[-test_count:]
        if max_train_per_symbol > 0 and len(train_targets) > max_train_per_symbol:
            train_targets = sorted(rng.choice(train_targets, size=max_train_per_symbol, replace=False).tolist())

        train.extend(SampleRef(symbol, idx) for idx in train_targets)
        val.extend(SampleRef(symbol, idx) for idx in val_targets)
        test.extend(SampleRef(symbol, idx) for idx in test_targets)
        rows.append(
            {
                "symbol": symbol,
                "rows": len(df),
                "train": len(train_targets),
                "val": len(val_targets),
                "test": len(test_targets),
                "train_start": str(df.iloc[train_targets[0]]["timestamps"]),
                "train_end": str(df.iloc[train_targets[-1]]["timestamps"]),
                "val_start": str(df.iloc[val_targets[0]]["timestamps"]),
                "val_end": str(df.iloc[val_targets[-1]]["timestamps"]),
                "test_start": str(df.iloc[test_targets[0]]["timestamps"]),
                "test_end": str(df.iloc[test_targets[-1]]["timestamps"]),
            }
        )

    split_df = pd.DataFrame(rows)
    if not train or not val or not test:
        raise RuntimeError("Empty split. Reduce lookback/val/test or add more data.")
    return train, val, test, split_df


def make_batch(
    frames: dict[str, pd.DataFrame],
    samples: list[SampleRef],
    lookback: int,
    clip: float,
    tokenizer: KronosTokenizer,
    device: str,
) -> dict[str, torch.Tensor | list[str]]:
    contexts = []
    contexts_plus_target = []
    stamps = []
    means = []
    stds = []
    actuals = []
    prev_closes = []
    symbols = []
    timestamps = []

    for sample in samples:
        df = frames[sample.symbol]
        target_idx = sample.target_idx
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
        symbols.append(sample.symbol)
        timestamps.append(str(target["timestamps"]))

    context_tensor = torch.tensor(np.stack(contexts), dtype=torch.float32, device=device)
    full_tensor = torch.tensor(np.stack(contexts_plus_target), dtype=torch.float32, device=device)

    with torch.no_grad():
        context_tokens = tokenizer.encode(context_tensor, half=True)
        full_tokens = tokenizer.encode(full_tensor, half=True)

    return {
        "s1": context_tokens[0],
        "s2": context_tokens[1],
        "stamp": torch.tensor(np.stack(stamps), dtype=torch.float32, device=device),
        "mean": torch.tensor(np.stack(means), dtype=torch.float32, device=device),
        "std": torch.tensor(np.stack(stds), dtype=torch.float32, device=device),
        "actual": torch.tensor(np.stack(actuals), dtype=torch.float32, device=device),
        "prev_close": torch.tensor(prev_closes, dtype=torch.float32, device=device),
        "target_s1": full_tokens[0][:, -1],
        "target_s2": full_tokens[1][:, -1],
        "symbols": symbols,
        "timestamps": timestamps,
    }


def full_kl(policy_logits: torch.Tensor, ref_logits: torch.Tensor) -> torch.Tensor:
    policy_logp = F.log_softmax(policy_logits, dim=-1)
    ref_logp = F.log_softmax(ref_logits, dim=-1)
    policy_prob = policy_logp.exp()
    return (policy_prob * (policy_logp - ref_logp)).sum(dim=-1)


def grpo_step(
    model: Kronos,
    reference_model: Kronos,
    tokenizer: KronosTokenizer,
    batch: dict,
    group_size: int,
    reward_trade_weight: float,
    reward_error_weight: float,
    reward_direction_bonus: float,
    signal_threshold: float,
    round_trip_cost_bps: float,
    kl_coef: float,
    entropy_coef: float,
    ce_coef: float,
):
    bsz = batch["s1"].shape[0]

    s1_logits, context = model.decode_s1(batch["s1"], batch["s2"], batch["stamp"])
    s1_last = s1_logits[:, -1, :]
    dist_s1 = Categorical(logits=s1_last)
    sampled_s1 = dist_s1.sample((group_size,)).transpose(0, 1).reshape(-1)
    logp_s1 = dist_s1.log_prob(sampled_s1.view(bsz, group_size).transpose(0, 1)).transpose(0, 1).reshape(-1)
    entropy_s1 = dist_s1.entropy().repeat_interleave(group_size)

    context_g = context.repeat_interleave(group_size, dim=0)
    s2_logits = model.decode_s2(context_g, sampled_s1.unsqueeze(-1))[:, -1, :]
    dist_s2 = Categorical(logits=s2_logits)
    sampled_s2 = dist_s2.sample()
    logp_s2 = dist_s2.log_prob(sampled_s2)
    entropy_s2 = dist_s2.entropy()

    s1_g = batch["s1"].repeat_interleave(group_size, dim=0)
    s2_g = batch["s2"].repeat_interleave(group_size, dim=0)
    generated_s1 = torch.cat([s1_g, sampled_s1.unsqueeze(-1)], dim=1)
    generated_s2 = torch.cat([s2_g, sampled_s2.unsqueeze(-1)], dim=1)

    with torch.no_grad():
        decoded = tokenizer.decode([generated_s1, generated_s2], half=True)[:, -1, :]
        mean_g = batch["mean"].repeat_interleave(group_size, dim=0)
        std_g = batch["std"].repeat_interleave(group_size, dim=0)
        actual_g = batch["actual"].repeat_interleave(group_size, dim=0)
        prev_close_g = batch["prev_close"].repeat_interleave(group_size, dim=0)

        pred = decoded * (std_g + 1e-5) + mean_g
        pred_close = pred[:, CLOSE_IDX]
        actual_close = actual_g[:, CLOSE_IDX]
        pred_ret = pred_close / prev_close_g - 1.0
        actual_ret = actual_close / prev_close_g - 1.0
        ret_error = torch.abs(pred_ret - actual_ret)

        position = torch.where(
            pred_ret > signal_threshold,
            torch.ones_like(pred_ret),
            torch.where(pred_ret < -signal_threshold, -torch.ones_like(pred_ret), torch.zeros_like(pred_ret)),
        )
        active = position.abs()
        trade_reward = position * actual_ret - active * (round_trip_cost_bps / 10000.0)
        direction_reward = torch.where(
            torch.sign(pred_ret) == torch.sign(actual_ret),
            torch.full_like(ret_error, reward_direction_bonus),
            torch.full_like(ret_error, -reward_direction_bonus),
        )
        reward = reward_trade_weight * trade_reward - reward_error_weight * ret_error + direction_reward
        reward_grouped = reward.view(bsz, group_size)
        advantage = (reward_grouped - reward_grouped.mean(dim=1, keepdim=True)) / (
            reward_grouped.std(dim=1, keepdim=True, unbiased=False) + 1e-6
        )
        advantage = advantage.reshape(-1)

    with torch.no_grad():
        ref_s1_logits, ref_context = reference_model.decode_s1(batch["s1"], batch["s2"], batch["stamp"])
        ref_s1_last = ref_s1_logits[:, -1, :]
        ref_context_g = ref_context.repeat_interleave(group_size, dim=0)
        ref_s2_logits = reference_model.decode_s2(ref_context_g, sampled_s1.unsqueeze(-1))[:, -1, :]

    kl_s1 = full_kl(s1_last, ref_s1_last).repeat_interleave(group_size)
    kl_s2 = full_kl(s2_logits, ref_s2_logits)
    kl = kl_s1 + kl_s2

    logp = logp_s1 + logp_s2
    policy_loss = -(advantage.detach() * logp).mean()
    kl_loss = kl.mean()
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
        "trade_reward_mean_bps": float((trade_reward.mean() * 10000).detach().cpu()),
        "ret_error_mean_pct": float((ret_error.mean() * 100).detach().cpu()),
        "direction_acc": float((torch.sign(pred_ret) == torch.sign(actual_ret)).float().mean().detach().cpu()),
        "active_rate": float(active.mean().detach().cpu()),
        "pred_ret_mean_pct": float((pred_ret.mean() * 100).detach().cpu()),
        "actual_ret_mean_pct": float((actual_ret.mean() * 100).detach().cpu()),
    }
    return loss, stats


def sample_to_prediction_inputs(frames: dict[str, pd.DataFrame], samples: list[SampleRef], lookback: int):
    df_list, x_ts_list, y_ts_list, actual_rows, symbols, prev_closes = [], [], [], [], [], []
    for sample in samples:
        df = frames[sample.symbol]
        context = df.iloc[sample.target_idx - lookback : sample.target_idx].copy()
        target = df.iloc[sample.target_idx].copy()
        df_list.append(context[FEATURE_COLS].reset_index(drop=True))
        x_ts_list.append(context["timestamps"].reset_index(drop=True))
        y_ts_list.append(pd.Series([target["timestamps"]], name="timestamps"))
        actual_rows.append(target)
        symbols.append(sample.symbol)
        prev_closes.append(float(context["close"].iloc[-1]))
    actual_df = pd.DataFrame(actual_rows).reset_index(drop=True)
    actual_df["symbol"] = symbols
    actual_df["prev_close"] = prev_closes
    return df_list, x_ts_list, y_ts_list, actual_df


def evaluate_samples(
    model: Kronos,
    tokenizer: KronosTokenizer,
    frames: dict[str, pd.DataFrame],
    samples: list[SampleRef],
    lookback: int,
    device: str,
    sample_count: int,
    eval_batch_size: int,
    seed: int,
    verbose: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    set_seed(seed)
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=lookback)
    model.eval()

    result_parts = []
    for start in range(0, len(samples), eval_batch_size):
        chunk = samples[start : start + eval_batch_size]
        df_list, x_ts_list, y_ts_list, actual_df = sample_to_prediction_inputs(frames, chunk, lookback)
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
        result = pd.DataFrame(
            {
                "symbol": actual_df["symbol"],
                "timestamps": actual_df["timestamps"],
                "prev_close": actual_df["prev_close"],
            }
        )
        for col in FEATURE_COLS:
            result[f"actual_{col}"] = actual_df[col].astype(float)
            result[f"pred_{col}"] = pred_df[col].astype(float)
        result_parts.append(result)

    all_results = pd.concat(result_parts, ignore_index=True)
    metrics, signal_metrics = compute_metrics(all_results.copy())
    portfolio = portfolio_summary(all_results)
    return all_results, metrics, signal_metrics, portfolio


def portfolio_summary(result: pd.DataFrame, initial_cash: float = 10000.0, threshold: float = 0.0, cost_bps: float = 10.0) -> dict:
    df = result.copy()
    df["actual_return"] = df["actual_close"] / df["prev_close"] - 1.0
    df["pred_return"] = df["pred_close"] / df["prev_close"] - 1.0
    df["position"] = np.where(df["pred_return"] > threshold, 1.0, np.where(df["pred_return"] < -threshold, -1.0, 0.0))
    df["trade_return"] = df["position"] * df["actual_return"] - np.abs(df["position"]) * (cost_bps / 10000.0)
    daily = df.groupby("timestamps", sort=True)["trade_return"].mean()
    equity = initial_cash * (1.0 + daily).cumprod()
    buy_hold_daily = df.groupby("timestamps", sort=True)["actual_return"].mean()
    buy_hold_equity = initial_cash * (1.0 + buy_hold_daily).cumprod()
    if len(equity) == 0:
        return {}
    drawdown = equity / equity.cummax() - 1.0
    return {
        "initial_cash": initial_cash,
        "final_equity": float(equity.iloc[-1]),
        "pnl": float(equity.iloc[-1] - initial_cash),
        "return_pct": float((equity.iloc[-1] / initial_cash - 1.0) * 100),
        "buy_hold_final_equity": float(buy_hold_equity.iloc[-1]),
        "buy_hold_return_pct": float((buy_hold_equity.iloc[-1] / initial_cash - 1.0) * 100),
        "max_drawdown_pct": float(drawdown.min() * 100),
        "mean_daily_return_pct": float(daily.mean() * 100),
        "daily_sharpe_like": float(daily.mean() / (daily.std(ddof=1) + 1e-12) * np.sqrt(252)) if len(daily) > 1 else np.nan,
        "active_rate": float((df["position"] != 0).mean()),
        "long_rate": float((df["position"] > 0).mean()),
        "short_rate": float((df["position"] < 0).mean()),
        "rows": int(len(df)),
        "days": int(len(daily)),
    }


def metric_score(metrics: pd.DataFrame, portfolio: dict, objective: str) -> float:
    close = metrics.loc[metrics["series"] == "close"].iloc[0]
    if objective == "portfolio_return":
        return float(portfolio["return_pct"])
    if objective == "negative_return_mae":
        return -float(close["mape_pct"])
    raise ValueError(f"Unknown objective: {objective}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Lab-style GRPO/LoRA fine-tuning for Kronos on MOEX daily candles.")
    parser.add_argument("--symbols", default="YDEX,SBER,GAZP,LKOH,ROSN,NVTK,GMKN,MOEX,TATN,MTSS")
    parser.add_argument("--board", default="TQBR")
    parser.add_argument("--days", type=int, default=1600)
    parser.add_argument("--lookback", type=int, default=256)
    parser.add_argument("--val-count", type=int, default=40)
    parser.add_argument("--test-count", type=int, default=50)
    parser.add_argument("--max-train-per-symbol", type=int, default=260)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=8e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--last-n-layers", type=int, default=4)
    parser.add_argument("--clip", type=float, default=5.0)
    parser.add_argument("--reward-trade-weight", type=float, default=80.0)
    parser.add_argument("--reward-error-weight", type=float, default=25.0)
    parser.add_argument("--reward-direction-bonus", type=float, default=0.10)
    parser.add_argument("--signal-threshold", type=float, default=0.001)
    parser.add_argument("--round-trip-cost-bps", type=float, default=20.0)
    parser.add_argument("--kl-coef", type=float, default=0.03)
    parser.add_argument("--entropy-coef", type=float, default=0.001)
    parser.add_argument("--ce-coef", type=float, default=0.03)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-sample-count", type=int, default=3)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--validation-objective", default="portfolio_return", choices=["portfolio_return", "negative_return_mae"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--model", default="NeoQuasar/Kronos-base")
    parser.add_argument("--tokenizer", default="NeoQuasar/Kronos-Tokenizer-base")
    args = parser.parse_args()

    set_seed(args.seed)
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    device = choose_device(args.device)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but not available.")

    output_dir = ROOT / "outputs" / f"lab_grpo_moex_{dt.datetime.now():%Y%m%d_%H%M%S}"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    print(f"Device: {device}")
    if device.startswith("cuda"):
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    symbols = parse_symbols(args.symbols)
    min_rows = args.lookback + args.val_count + args.test_count + 20
    frames = fetch_universe(symbols, args.board, args.days, min_rows, output_dir)
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
    print(f"Train samples: {len(train_samples)}, val: {len(val_samples)}, test: {len(test_samples)}")

    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = KronosTokenizer.from_pretrained(args.tokenizer).to(device).eval()
    print(f"Loading baseline/reference model: {args.model}")
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
    print(f"LoRA modules: {len(wrapped_modules)}, trainable params: {trainable_params:,}")

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
    best_state = trainable_state_dict(policy_model)
    best_step = 0

    policy_model.train()
    for step in range(1, args.steps + 1):
        chosen = rng.choice(len(train_samples), size=args.batch_size, replace=True)
        batch_refs = [train_samples[int(i)] for i in chosen]
        batch = make_batch(frames, batch_refs, args.lookback, args.clip, tokenizer, device)
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

        if step == 1 or step % 25 == 0:
            print(
                f"step {step:04d}/{args.steps} "
                f"loss={stats['loss']:.4f} reward={stats['reward_mean']:.4f} "
                f"trade_bps={stats['trade_reward_mean_bps']:.2f} "
                f"err={stats['ret_error_mean_pct']:.3f}% dir={stats['direction_acc']:.2f} "
                f"kl={stats['kl']:.4f}"
            )

        if step % args.eval_interval == 0 or step == args.steps:
            print(f"Validation at step {step}...")
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

    print(f"Best validation step: {best_step}, score={best_score:.4f}")
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
    baseline_metrics.to_csv(output_dir / "baseline_test_metrics.csv", index=False)
    baseline_signal.to_csv(output_dir / "baseline_test_signal_metrics.csv", index=False)
    tuned_result.to_csv(output_dir / "tuned_test_predictions.csv", index=False)
    tuned_metrics.to_csv(output_dir / "tuned_test_metrics.csv", index=False)
    tuned_signal.to_csv(output_dir / "tuned_test_signal_metrics.csv", index=False)
    pd.DataFrame([baseline_portfolio]).to_csv(output_dir / "baseline_test_portfolio.csv", index=False)
    pd.DataFrame([tuned_portfolio]).to_csv(output_dir / "tuned_test_portfolio.csv", index=False)

    def summarize_model(label: str, metrics: pd.DataFrame, signal: pd.DataFrame, portfolio: dict):
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

    comparison = pd.DataFrame(
        [
            summarize_model("baseline", baseline_metrics, baseline_signal, baseline_portfolio),
            summarize_model("grpo_lora", tuned_metrics, tuned_signal, tuned_portfolio),
        ]
    )
    comparison.to_csv(output_dir / "test_comparison.csv", index=False)

    summary = {
        "device": device,
        "symbols": list(frames.keys()),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "test_samples": len(test_samples),
        "best_step": best_step,
        "best_score": best_score,
        "trainable_params": trainable_params,
        "wrapped_modules": len(wrapped_modules),
        "output_dir": str(output_dir),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "summary.txt").write_text(
        json.dumps(summary, indent=2) + "\n\n" + comparison.to_string(index=False),
        encoding="utf-8",
    )

    print("\nTest comparison:")
    print(comparison.to_string(index=False))
    print(f"\nOutput directory: {output_dir}")


if __name__ == "__main__":
    main()
