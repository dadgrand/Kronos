import argparse
import datetime as dt
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class NeuralPolicyConfig:
    mode: str
    k: int
    gross: float
    rebalance_every: int
    weight_scheme: str


class PolicyNet(nn.Module):
    def __init__(self, n_features: int, n_symbols: int, embedding_dim: int = 8, hidden_dim: int = 96):
        super().__init__()
        self.symbol_embedding = nn.Embedding(n_symbols, embedding_dim)
        self.net = nn.Sequential(
            nn.Linear(n_features + embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.05),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor, symbol_id: torch.Tensor) -> torch.Tensor:
        emb = self.symbol_embedding(symbol_id)
        return self.net(torch.cat([x, emb], dim=-1)).squeeze(-1)


def read_intraday_matrix(dataset_dir: Path, interval: int) -> tuple[pd.DatetimeIndex, list[str], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root = dataset_dir / "curated" / f"interval={interval}"
    files = list(root.glob("symbol=*/year=*/month=*/candles_curated.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {root}")
    parts = []
    for path in files:
        frame = pd.read_parquet(path)
        symbol = path.parts[-4].split("=", 1)[1]
        timestamp = pd.to_datetime(frame["timestamp"] if "timestamp" in frame.columns else frame["begin"])
        parts.append(
            pd.DataFrame(
                {
                    "timestamp": timestamp.to_numpy(),
                    "symbol": symbol,
                    "open": frame["open"].astype(float).to_numpy(),
                    "close": frame["close"].astype(float).to_numpy(),
                    "volume": frame["volume"].astype(float).to_numpy(),
                }
            )
        )
    raw = pd.concat(parts, ignore_index=True).drop_duplicates(["timestamp", "symbol"])
    open_px = raw.pivot(index="timestamp", columns="symbol", values="open").sort_index()
    close_px = raw.pivot(index="timestamp", columns="symbol", values="close").sort_index()
    volume = raw.pivot(index="timestamp", columns="symbol", values="volume").reindex_like(close_px).fillna(0.0)
    close_px = close_px.ffill(limit=200)
    open_px = open_px.reindex_like(close_px)
    volume = volume.reindex_like(close_px).fillna(0.0)
    return close_px.index, list(close_px.columns), open_px, close_px, volume


def cs_zscore(values: np.ndarray) -> np.ndarray:
    mean = np.nanmean(values, axis=1, keepdims=True)
    std = np.nanstd(values, axis=1, keepdims=True)
    return (values - mean) / np.clip(std, 1e-6, None)


def build_matrices(open_px: pd.DataFrame, close_px: pd.DataFrame, volume: pd.DataFrame, horizon: int) -> dict:
    close = close_px.to_numpy(dtype=np.float64)
    open_ = open_px.to_numpy(dtype=np.float64)
    volume_arr = volume.to_numpy(dtype=np.float64)
    active = np.isfinite(open_) & np.isfinite(close) & (volume_arr > 0)
    tradable = np.zeros_like(active, dtype=bool)
    tradable[1:, :] = active[:-1, :]
    bar_return = np.where(active, close / open_ - 1.0, 0.0)

    close_ff = close_px.ffill(limit=200)
    ret1 = close_ff.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float64)
    feature_frames = []
    feature_names = []

    for lag in [1, 3, 6, 12, 24, 48, 96]:
        mom = close_ff.shift(1) / close_ff.shift(1 + lag) - 1.0
        feature_frames.append(mom.to_numpy(dtype=np.float64))
        feature_names.append(f"mom_{lag}")
        feature_frames.append(cs_zscore(mom.to_numpy(dtype=np.float64)))
        feature_names.append(f"mom_{lag}_csz")

    ret_frame = pd.DataFrame(ret1, index=close_px.index, columns=close_px.columns)
    for window in [12, 48, 96]:
        vol = ret_frame.shift(1).rolling(window, min_periods=max(3, window // 4)).std()
        vol_arr = vol.to_numpy(dtype=np.float64)
        feature_frames.append(np.log1p(np.nan_to_num(vol_arr, nan=0.0) * 10000.0))
        feature_names.append(f"vol_{window}")

    log_volume = np.log1p(volume_arr)
    log_volume_shift = pd.DataFrame(log_volume, index=close_px.index, columns=close_px.columns).shift(1)
    feature_frames.append(log_volume_shift.to_numpy(dtype=np.float64))
    feature_names.append("log_volume_1")
    feature_frames.append(cs_zscore(log_volume_shift.to_numpy(dtype=np.float64)))
    feature_names.append("log_volume_csz")

    timestamps = close_px.index
    minute_of_day = np.array([ts.hour * 60 + ts.minute for ts in timestamps], dtype=np.float64)
    weekday = np.array([ts.weekday() for ts in timestamps], dtype=np.float64)
    time_features = np.stack(
        [
            np.sin(2.0 * np.pi * minute_of_day / 1440.0),
            np.cos(2.0 * np.pi * minute_of_day / 1440.0),
            np.sin(2.0 * np.pi * weekday / 7.0),
            np.cos(2.0 * np.pi * weekday / 7.0),
        ],
        axis=1,
    )
    for idx in range(time_features.shape[1]):
        feature_frames.append(np.repeat(time_features[:, [idx]], close.shape[1], axis=1))
        feature_names.append(f"time_{idx}")

    target = np.full_like(bar_return, np.nan, dtype=np.float64)
    for step in range(horizon):
        shifted = np.roll(bar_return, -step, axis=0)
        if step > 0:
            shifted[-step:, :] = np.nan
        target = np.where(np.isfinite(target), target, 0.0)
        target += np.nan_to_num(shifted, nan=0.0)
    if horizon > 0:
        target[-horizon:, :] = np.nan

    features = np.stack(feature_frames, axis=-1).astype(np.float32)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return {
        "features": features,
        "target": target.astype(np.float32),
        "bar_return": bar_return.astype(np.float32),
        "active": active,
        "tradable": tradable,
        "volatility": np.nan_to_num(
            ret_frame.shift(1).rolling(48, min_periods=12).std().to_numpy(dtype=np.float64),
            nan=np.nanstd(ret1),
            posinf=np.nanstd(ret1),
            neginf=np.nanstd(ret1),
        ).astype(np.float32),
        "feature_names": feature_names,
    }


def split_masks(index: pd.DatetimeIndex, train_end: str, validation_end: str) -> dict[str, np.ndarray]:
    train_end_ts = pd.Timestamp(train_end)
    validation_end_ts = pd.Timestamp(validation_end)
    return {
        "train": np.asarray(index < train_end_ts),
        "validation": np.asarray((index >= train_end_ts) & (index < validation_end_ts)),
        "test": np.asarray(index >= validation_end_ts),
    }


def horizon_safe_mask(mask: np.ndarray, horizon: int) -> np.ndarray:
    safe = mask.copy()
    for step in range(1, max(horizon, 1)):
        shifted = np.zeros_like(mask, dtype=bool)
        shifted[:-step] = mask[step:]
        safe &= shifted
    return safe


def flatten_split(features: np.ndarray, target: np.ndarray, active: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    row_ids = np.flatnonzero(mask)
    x = features[row_ids].reshape(-1, features.shape[-1])
    y = target[row_ids].reshape(-1)
    a = active[row_ids].reshape(-1)
    symbol_id = np.tile(np.arange(features.shape[1], dtype=np.int64), len(row_ids))
    valid = np.isfinite(y) & a
    return x[valid], y[valid], symbol_id[valid]


def policy_grid(max_gross: float) -> list[NeuralPolicyConfig]:
    configs = [NeuralPolicyConfig("cash", 0, 0.0, 1, "equal")]
    for mode in ["long_short", "long_only", "short_only"]:
        for k in [1, 3, 5, 10, 15]:
            if mode == "long_short" and 2 * k > 50:
                continue
            for gross in [0.5, 1.0, 2.0, 3.0, 4.0]:
                if gross > max_gross:
                    continue
                for rebalance_every in [1, 3, 6, 12, 24, 48]:
                    for weight_scheme in ["equal", "inv_vol"]:
                        configs.append(NeuralPolicyConfig(mode, k, gross, rebalance_every, weight_scheme))
    return configs


def allocate(ids: np.ndarray, budget: float, sign: float, vol: np.ndarray, scheme: str) -> tuple[np.ndarray, np.ndarray]:
    if ids.size == 0 or budget <= 0:
        return ids, np.zeros(ids.size, dtype=np.float32)
    if scheme == "inv_vol":
        raw = 1.0 / np.clip(vol[ids], 1e-6, None)
        raw = np.nan_to_num(raw, nan=1.0, posinf=1.0, neginf=1.0)
    else:
        raw = np.ones(ids.size, dtype=np.float32)
    weights = raw / raw.sum() * budget * sign
    return ids, weights.astype(np.float32)


def backtest_scores(
    scores: np.ndarray,
    bar_return: np.ndarray,
    active: np.ndarray,
    volatility: np.ndarray,
    mask: np.ndarray,
    config: NeuralPolicyConfig,
    *,
    initial_cash: float,
    cost_bps: float,
) -> dict:
    row_ids = np.flatnonzero(mask)
    weights = np.zeros(scores.shape[1], dtype=np.float32)
    equity = initial_cash
    equity_curve = []
    turnovers = []
    gross_exposures = []
    cost_rate = cost_bps / 10000.0

    for local_idx, row in enumerate(row_ids):
        if local_idx % config.rebalance_every == 0:
            row_score = scores[row]
            valid = np.isfinite(row_score) & active[row]
            candidates = np.flatnonzero(valid)
            target = np.zeros_like(weights)
            if candidates.size >= max(2 * config.k if config.mode == "long_short" else config.k, 1):
                order = candidates[np.argsort(row_score[candidates])]
                if config.mode == "long_short":
                    longs = order[-config.k :]
                    shorts = order[: config.k]
                    ids, vals = allocate(longs, config.gross / 2.0, 1.0, volatility[row], config.weight_scheme)
                    target[ids] = vals
                    ids, vals = allocate(shorts, config.gross / 2.0, -1.0, volatility[row], config.weight_scheme)
                    target[ids] = vals
                elif config.mode == "long_only":
                    longs = order[-config.k :]
                    ids, vals = allocate(longs, config.gross, 1.0, volatility[row], config.weight_scheme)
                    target[ids] = vals
                elif config.mode == "short_only":
                    shorts = order[: config.k]
                    ids, vals = allocate(shorts, config.gross, -1.0, volatility[row], config.weight_scheme)
                    target[ids] = vals
                elif config.mode == "cash":
                    target = np.zeros_like(weights)
                else:
                    raise ValueError(f"unknown mode: {config.mode}")
            turnover = float(np.abs(target - weights).sum())
            weights = target
        else:
            turnover = 0.0

        pnl_return = float(np.nan_to_num(bar_return[row], nan=0.0) @ weights) - turnover * cost_rate
        equity *= max(0.0, 1.0 + pnl_return)
        equity_curve.append(equity)
        turnovers.append(turnover)
        gross_exposures.append(float(np.abs(weights).sum()))

    eq = np.asarray(equity_curve, dtype=np.float64)
    bar_rets = np.diff(np.r_[initial_cash, eq]) / np.r_[initial_cash, eq[:-1]] if eq.size else np.array([])
    max_dd = float(((eq / np.maximum.accumulate(eq)) - 1.0).min() * 100.0) if eq.size else 0.0
    return {
        "initial_cash": initial_cash,
        "final_equity": float(equity),
        "pnl": float(equity - initial_cash),
        "return_pct": float((equity / initial_cash - 1.0) * 100.0),
        "max_drawdown_pct": max_dd,
        "sharpe_like": float(bar_rets.mean() / (bar_rets.std(ddof=0) + 1e-12) * np.sqrt(252 * 50)) if bar_rets.size else 0.0,
        "bars": int(len(row_ids)),
        "avg_turnover_per_bar": float(np.mean(turnovers)) if turnovers else 0.0,
        "total_turnover": float(np.sum(turnovers)) if turnovers else 0.0,
        "mean_gross_exposure": float(np.mean(gross_exposures)) if gross_exposures else 0.0,
    }


def selection_score(summary: dict, drawdown_penalty: float, turnover_penalty: float) -> float:
    return (
        float(summary["return_pct"])
        + drawdown_penalty * float(summary["max_drawdown_pct"])
        - turnover_penalty * float(summary["avg_turnover_per_bar"])
    )


@torch.no_grad()
def predict_matrix(model: PolicyNet, features: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    rows, symbols, n_features = features.shape
    flat = features.reshape(-1, n_features)
    symbol_ids = np.tile(np.arange(symbols, dtype=np.int64), rows)
    preds = np.empty(flat.shape[0], dtype=np.float32)
    for start in range(0, flat.shape[0], batch_size):
        end = min(start + batch_size, flat.shape[0])
        xb = torch.from_numpy(flat[start:end]).to(device)
        sb = torch.from_numpy(symbol_ids[start:end]).to(device)
        preds[start:end] = model(xb, sb).detach().cpu().numpy()
    return preds.reshape(rows, symbols)


def train_model(
    features: np.ndarray,
    target: np.ndarray,
    active: np.ndarray,
    masks: dict[str, np.ndarray],
    matrices: dict,
    *,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[PolicyNet, dict, np.ndarray]:
    train_target_mask = masks["train"] & horizon_safe_mask(masks["train"], args.horizon)
    validation_target_mask = masks["validation"] & horizon_safe_mask(masks["validation"], args.horizon)
    x_train, y_train, s_train = flatten_split(features, target, active, train_target_mask)
    x_val, y_val, s_val = flatten_split(features, target, active, validation_target_mask)
    y_scale = float(np.nanstd(y_train))
    if not np.isfinite(y_scale) or y_scale <= 0:
        y_scale = 1.0
    y_train_scaled = (y_train / y_scale).astype(np.float32)
    y_val_scaled = (y_val / y_scale).astype(np.float32)

    model = PolicyNet(features.shape[-1], features.shape[1], args.embedding_dim, args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    configs = policy_grid(args.max_gross)
    best = {"score": -1e18, "epoch": 0, "state": None, "policy": None, "summary": None}
    rng = np.random.default_rng(args.seed)
    n_train = len(y_train_scaled)

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for _ in range(args.steps_per_epoch):
            batch_idx = rng.integers(0, n_train, size=args.batch_size)
            xb = torch.from_numpy(x_train[batch_idx]).to(device)
            yb = torch.from_numpy(y_train_scaled[batch_idx]).to(device)
            sb = torch.from_numpy(s_train[batch_idx]).to(device)
            pred = model(xb, sb)
            loss = F.huber_loss(pred, yb, delta=1.0)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            sample = min(args.validation_metric_rows, len(y_val_scaled))
            metric_idx = rng.choice(len(y_val_scaled), size=sample, replace=False)
            xb = torch.from_numpy(x_val[metric_idx]).to(device)
            sb = torch.from_numpy(s_val[metric_idx]).to(device)
            pred = model(xb, sb).detach().cpu().numpy()
            corr = float(np.corrcoef(pred, y_val_scaled[metric_idx])[0, 1]) if np.std(pred) > 0 else 0.0

        scores = predict_matrix(model, features, device, args.predict_batch_size)
        rows = []
        for config in configs:
            summary = backtest_scores(
                scores,
                matrices["bar_return"],
                matrices["tradable"],
                matrices["volatility"],
                masks["validation"],
                config,
                initial_cash=args.initial_cash,
                cost_bps=args.cost_bps,
            )
            score = selection_score(summary, args.drawdown_penalty, args.turnover_penalty)
            rows.append((score, config, summary))
        score, config, summary = max(rows, key=lambda row: row[0])
        print(
            f"epoch={epoch} loss={np.mean(losses):.5f} val_corr={corr:.5f} "
            f"val_return={summary['return_pct']:.2f}% score={score:.3f} policy={asdict(config)}",
            flush=True,
        )
        if score > best["score"]:
            best = {
                "score": score,
                "epoch": epoch,
                "state": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                "policy": config,
                "summary": summary,
                "corr": corr,
            }

    model.load_state_dict(best["state"])
    scores = predict_matrix(model, features, device, args.predict_batch_size)
    return model, best, scores


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Neural causal policy lab for MOEX intraday candles.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("datasets/moex_intraday_6f8f836cc811"))
    parser.add_argument("--interval", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--train-end", default="2026-01-01")
    parser.add_argument("--validation-end", default="2026-06-01")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--steps-per-epoch", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--embedding-dim", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--initial-cash", type=float, default=10000.0)
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--max-gross", type=float, default=4.0)
    parser.add_argument("--turnover-penalty", type=float, default=2.0)
    parser.add_argument("--drawdown-penalty", type=float, default=0.25)
    parser.add_argument("--validation-metric-rows", type=int, default=200000)
    parser.add_argument("--predict-batch-size", type=int, default=262144)
    parser.add_argument("--seed", type=int, default=20260622)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    if args.output_dir is None:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = Path("outputs") / f"neural_policy_lab_h{args.horizon}_seed{args.seed}_{stamp}"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    index, symbols, open_px, close_px, volume = read_intraday_matrix(args.dataset_dir, args.interval)
    masks = split_masks(index, args.train_end, args.validation_end)
    matrices = build_matrices(open_px, close_px, volume, args.horizon)
    model, best, scores = train_model(
        matrices["features"], matrices["target"], matrices["active"], masks, matrices, args=args, device=device
    )

    selected_policy = best["policy"]
    validation_summary = backtest_scores(
        scores,
        matrices["bar_return"],
        matrices["tradable"],
        matrices["volatility"],
        masks["validation"],
        selected_policy,
        initial_cash=args.initial_cash,
        cost_bps=args.cost_bps,
    )
    test_summary = backtest_scores(
        scores,
        matrices["bar_return"],
        matrices["tradable"],
        matrices["volatility"],
        masks["test"],
        selected_policy,
        initial_cash=args.initial_cash,
        cost_bps=args.cost_bps,
    )
    sensitivity = []
    for multiplier in [0.5, 1.0, 2.0, 3.0]:
        sensitivity.append(
            {
                "cost_bps": args.cost_bps * multiplier,
                **backtest_scores(
                    scores,
                    matrices["bar_return"],
                    matrices["tradable"],
                    matrices["volatility"],
                    masks["test"],
                    selected_policy,
                    initial_cash=args.initial_cash,
                    cost_bps=args.cost_bps * multiplier,
                ),
            }
        )

    torch.save({"model": model.state_dict(), "policy": asdict(selected_policy), "args": vars(args)}, args.output_dir / "policy_model.pt")
    np.save(args.output_dir / "scores.npy", scores.astype(np.float32))
    pd.DataFrame([validation_summary]).to_csv(args.output_dir / "selected_validation_summary.csv", index=False)
    pd.DataFrame([test_summary]).to_csv(args.output_dir / "selected_test_summary.csv", index=False)
    pd.DataFrame(sensitivity).to_csv(args.output_dir / "cost_sensitivity.csv", index=False)
    write_json(
        args.output_dir / "summary.json",
        {
            "created_at": dt.datetime.now().isoformat(timespec="seconds"),
            "dataset_dir": str(args.dataset_dir),
            "interval": args.interval,
            "horizon": args.horizon,
            "symbols": symbols,
            "train_end": args.train_end,
            "validation_end": args.validation_end,
            "split_rows": {name: int(mask.sum()) for name, mask in masks.items()},
            "feature_names": matrices["feature_names"],
            "best_epoch": best["epoch"],
            "best_validation_score": best["score"],
            "best_validation_corr": best.get("corr"),
            "selected_policy": asdict(selected_policy),
            "validation_summary": validation_summary,
            "test_summary": test_summary,
            "target_30pct_hit": test_summary["return_pct"] >= 30.0,
            "protocol": "Features are shifted/causal. Checkpoint and policy are selected on validation only; test is evaluated after selection.",
        },
    )
    readme = [
        "# Neural Policy Lab",
        "",
        "Causal neural policy model trained on MOEX 10m candles.",
        "",
        f"- Horizon: {args.horizon}",
        f"- Best epoch: {best['epoch']}",
        f"- Selected policy: `{asdict(selected_policy)}`",
        f"- Validation return: {validation_summary['return_pct']:.2f}%",
        f"- Test return: {test_summary['return_pct']:.2f}%",
        f"- Test final equity: {test_summary['final_equity']:.2f}",
        f"- Test max drawdown: {test_summary['max_drawdown_pct']:.2f}%",
        f"- 30% target hit: {test_summary['return_pct'] >= 30.0}",
        "",
        "This is a research backtest, not financial advice.",
        "",
    ]
    (args.output_dir / "README.md").write_text("\n".join(readme), encoding="utf-8")
    print(f"Output dir: {args.output_dir}")
    print(f"Selected policy: {asdict(selected_policy)}")
    print(f"Validation return: {validation_summary['return_pct']:.2f}%")
    print(f"Test return: {test_summary['return_pct']:.2f}%")
    print(f"30% target hit: {test_summary['return_pct'] >= 30.0}")


if __name__ == "__main__":
    main()
