import argparse
import datetime as dt
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch

from filtered_neural_policy_lab import compute_market_features, short_confidence
from neural_policy_lab import PolicyNet, build_matrices, read_intraday_matrix


FINAL_POLICY = {
    "mode": "short_only",
    "k": 1,
    "gross": 4.0,
    "rebalance_every": 24,
    "confidence_min": 0.07741670683026314,
    "market_mom_24_max": 0.002567113547411282,
    "market_mom_96_max": 3.8395675483115644e-05,
}


def block_to_df(payload: dict, block: str) -> pd.DataFrame:
    data = payload.get(block, {})
    columns = data.get("columns", [])
    rows = data.get("data", [])
    return pd.DataFrame(rows, columns=columns)


def fetch_candles(
    session: requests.Session,
    symbol: str,
    board: str,
    interval: int,
    from_date: str,
    till_date: str,
) -> pd.DataFrame:
    rows = []
    start = 0
    while True:
        url = (
            f"https://iss.moex.com/iss/engines/stock/markets/shares/boards/{board}"
            f"/securities/{symbol}/candles.json"
        )
        params = {
            "from": from_date,
            "till": till_date,
            "interval": interval,
            "start": start,
            "iss.meta": "off",
        }
        response = session.get(url, params=params, timeout=20)
        response.raise_for_status()
        block = block_to_df(response.json(), "candles")
        if block.empty:
            break
        rows.append(block)
        if len(block) < 500:
            break
        start += len(block)
    if not rows:
        return pd.DataFrame(columns=["timestamp", "symbol", "open", "close", "volume"])
    out = pd.concat(rows, ignore_index=True)
    timestamp_col = "begin" if "begin" in out.columns else "timestamp"
    out["timestamp"] = pd.to_datetime(out[timestamp_col])
    out["symbol"] = symbol
    for col in ["open", "close", "volume"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out[["timestamp", "symbol", "open", "close", "volume"]].dropna(subset=["timestamp", "close"])


def fetch_marketdata(session: requests.Session, board: str, symbols: list[str]) -> dict[str, dict]:
    url = f"https://iss.moex.com/iss/engines/stock/markets/shares/boards/{board}/securities.json"
    params = {
        "iss.meta": "off",
        "marketdata.columns": "SECID,BOARDID,LAST,LCURRENTPRICE,MARKETPRICE2,BID,OFFER,TIME,UPDATETIME,SYSTIME",
        "securities.columns": "SECID,BOARDID,SHORTNAME",
    }
    response = session.get(url, params=params, timeout=20)
    response.raise_for_status()
    marketdata = block_to_df(response.json(), "marketdata")
    out = {}
    if marketdata.empty:
        return out
    marketdata = marketdata[marketdata["SECID"].isin(symbols)]
    for _, row in marketdata.iterrows():
        symbol = str(row["SECID"])
        price = None
        for col in ["LAST", "LCURRENTPRICE", "MARKETPRICE2", "BID", "OFFER"]:
            value = pd.to_numeric(row.get(col), errors="coerce")
            if np.isfinite(value) and value > 0:
                price = float(value)
                break
        out[symbol] = {
            "price": price,
            "time": row.get("TIME"),
            "update_time": row.get("UPDATETIME"),
            "system_time": row.get("SYSTIME"),
        }
    return out


def append_live_to_matrix(
    base_open: pd.DataFrame,
    base_close: pd.DataFrame,
    base_volume: pd.DataFrame,
    live: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if live.empty:
        return base_open, base_close, base_volume
    live = live.drop_duplicates(["timestamp", "symbol"], keep="last")
    live_open = live.pivot(index="timestamp", columns="symbol", values="open")
    live_close = live.pivot(index="timestamp", columns="symbol", values="close")
    live_volume = live.pivot(index="timestamp", columns="symbol", values="volume")
    open_px = pd.concat([base_open, live_open]).sort_index()
    close_px = pd.concat([base_close, live_close]).sort_index()
    volume = pd.concat([base_volume, live_volume]).sort_index()
    open_px = open_px[~open_px.index.duplicated(keep="last")]
    close_px = close_px[~close_px.index.duplicated(keep="last")]
    volume = volume[~volume.index.duplicated(keep="last")]
    open_px = open_px.reindex(columns=base_close.columns)
    close_px = close_px.reindex(columns=base_close.columns)
    volume = volume.reindex(columns=base_close.columns).fillna(0.0)
    return open_px, close_px, volume


def load_models(checkpoints: list[Path], n_features: int, n_symbols: int, device: torch.device) -> list[PolicyNet]:
    models = []
    for path in checkpoints:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        args = checkpoint.get("args", {})
        model = PolicyNet(
            n_features,
            n_symbols,
            int(args.get("embedding_dim", 8)),
            int(args.get("hidden_dim", 96)),
        ).to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        models.append(model)
    return models


@torch.no_grad()
def score_latest(models: list[PolicyNet], latest_features: np.ndarray, device: torch.device) -> np.ndarray:
    x = torch.from_numpy(latest_features.astype(np.float32)).to(device)
    symbol_id = torch.arange(latest_features.shape[0], dtype=torch.long, device=device)
    preds = [model(x, symbol_id).detach().cpu().numpy() for model in models]
    return np.mean(preds, axis=0).astype(np.float32)


def execute_target(
    cash: float,
    position_symbol: str | None,
    shares: float,
    target_symbol: str | None,
    target_weight: float,
    prices: dict[str, float],
    cost_bps: float,
) -> tuple[float, str | None, float, float, float]:
    current_price = prices.get(position_symbol) if position_symbol else None
    current_value = shares * current_price if position_symbol and current_price else 0.0
    equity = cash + current_value
    cost_rate = cost_bps / 10000.0
    total_cost = 0.0
    turnover_value = 0.0

    if position_symbol and (target_symbol != position_symbol):
        trade_value = abs(shares) * current_price
        cash += shares * current_price
        cost = trade_value * cost_rate
        cash -= cost
        total_cost += cost
        turnover_value += trade_value
        shares = 0.0
        position_symbol = None

    if target_symbol is not None:
        price = prices[target_symbol]
        target_value = equity * target_weight
        current_target_value = shares * price if position_symbol == target_symbol else 0.0
        delta_value = target_value - current_target_value
        delta_shares = delta_value / price
        trade_value = abs(delta_value)
        cash -= delta_shares * price
        cost = trade_value * cost_rate
        cash -= cost
        total_cost += cost
        turnover_value += trade_value
        shares = shares + delta_shares
        position_symbol = target_symbol

    latest_price = prices.get(position_symbol) if position_symbol else None
    new_equity = cash + (shares * latest_price if position_symbol and latest_price else 0.0)
    return cash, position_symbol, shares, total_cost, turnover_value


def main() -> None:
    parser = argparse.ArgumentParser(description="30-minute live paper run for the filtered neural MOEX policy.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("datasets/moex_intraday_6f8f836cc811"))
    parser.add_argument("--board", default="TQBR")
    parser.add_argument("--interval", type=int, default=10)
    parser.add_argument("--duration-minutes", type=float, default=30.0)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--initial-cash", type=float, default=10000.0)
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        action="append",
        default=[
            Path("outputs/neural_policy_lab_h12_seed20260622_20260622_132851/policy_model.pt"),
            Path("outputs/neural_policy_lab_h12_seed20260623_20260622_133133/policy_model.pt"),
            Path("outputs/neural_policy_lab_h12_seed20260624_20260622_133615/policy_model.pt"),
        ],
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    if args.output_dir is None:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = Path("outputs") / f"live_paper_moex_policy_{stamp}"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    index, symbols, base_open, base_close, base_volume = read_intraday_matrix(args.dataset_dir, args.interval)
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")

    # Build once to infer feature size and symbol count.
    initial_matrices = build_matrices(base_open, base_close, base_volume, horizon=12)
    models = load_models(args.checkpoint, initial_matrices["features"].shape[-1], len(symbols), device)

    session = requests.Session()
    session.trust_env = False
    cash = float(args.initial_cash)
    position_symbol = None
    shares = 0.0
    last_rebalance_ts = None
    total_cost = 0.0
    total_turnover = 0.0
    rows = []
    start_time = time.time()
    end_time = start_time + args.duration_minutes * 60.0

    while True:
        now = dt.datetime.now()
        from_date = (now.date() - dt.timedelta(days=8)).isoformat()
        till_date = now.date().isoformat()
        live_parts = []
        errors = []
        for symbol in symbols:
            try:
                live_parts.append(fetch_candles(session, symbol, args.board, args.interval, from_date, till_date))
            except Exception as exc:  # noqa: BLE001 - live runner should log and continue.
                errors.append(f"{symbol}: {exc}")
        live = pd.concat(live_parts, ignore_index=True) if live_parts else pd.DataFrame()
        open_px, close_px, volume = append_live_to_matrix(base_open, base_close, base_volume, live)
        matrices = build_matrices(open_px, close_px, volume, horizon=12)
        market_features = compute_market_features(close_px.index, list(close_px.columns), matrices["bar_return"])

        latest_row = len(close_px.index) - 1
        latest_ts = close_px.index[latest_row]
        latest_features = matrices["features"][latest_row]
        latest_scores = score_latest(models, latest_features, device)
        tradable = matrices["tradable"][latest_row] & np.isfinite(latest_scores)
        confidence = short_confidence(latest_scores.reshape(1, -1), tradable.reshape(1, -1))[0]
        market_mom_24 = float(market_features["market_mom_24"][latest_row])
        market_mom_96 = float(market_features["market_mom_96"][latest_row])

        candle_prices = close_px.iloc[latest_row].to_dict()
        marketdata = fetch_marketdata(session, args.board, symbols)
        prices = {}
        for symbol in symbols:
            md_price = marketdata.get(symbol, {}).get("price")
            candle_price = candle_prices.get(symbol)
            if md_price and np.isfinite(md_price):
                prices[symbol] = float(md_price)
            elif candle_price and np.isfinite(candle_price):
                prices[symbol] = float(candle_price)

        target_symbol = None
        target_weight = 0.0
        filter_pass = (
            confidence >= FINAL_POLICY["confidence_min"]
            and market_mom_24 <= FINAL_POLICY["market_mom_24_max"]
            and market_mom_96 <= FINAL_POLICY["market_mom_96_max"]
        )
        if filter_pass and tradable.any():
            candidates = np.flatnonzero(tradable)
            selected_idx = int(candidates[np.argmin(latest_scores[candidates])])
            candidate = symbols[selected_idx]
            if candidate in prices:
                target_symbol = candidate
                target_weight = -float(FINAL_POLICY["gross"])

        new_bar = last_rebalance_ts is None or latest_ts != last_rebalance_ts
        should_rebalance = last_rebalance_ts is None
        if should_rebalance:
            cash, position_symbol, shares, cost, turnover = execute_target(
                cash,
                position_symbol,
                shares,
                target_symbol,
                target_weight,
                prices,
                args.cost_bps,
            )
            last_rebalance_ts = latest_ts
            total_cost += cost
            total_turnover += turnover
        else:
            cost = 0.0
            turnover = 0.0

        mark_price = prices.get(position_symbol) if position_symbol else None
        equity = cash + (shares * mark_price if position_symbol and mark_price else 0.0)
        row = {
            "wall_time": now.isoformat(timespec="seconds"),
            "latest_candle": str(latest_ts),
            "cash": cash,
            "position_symbol": position_symbol,
            "shares": shares,
            "mark_price": mark_price,
            "equity": equity,
            "pnl": equity - args.initial_cash,
            "return_pct": (equity / args.initial_cash - 1.0) * 100.0,
            "target_symbol": target_symbol,
            "target_weight": target_weight,
            "filter_pass": filter_pass,
            "confidence": float(confidence),
            "market_mom_24": market_mom_24,
            "market_mom_96": market_mom_96,
            "trade_cost": cost,
            "turnover": turnover,
            "total_cost": total_cost,
            "total_turnover": total_turnover,
            "errors": " | ".join(errors[:5]),
        }
        rows.append(row)
        pd.DataFrame(rows).to_csv(args.output_dir / "live_log.csv", index=False)
        summary = {
            "started_at": dt.datetime.fromtimestamp(start_time).isoformat(timespec="seconds"),
            "updated_at": now.isoformat(timespec="seconds"),
            "initial_cash": args.initial_cash,
            "equity": equity,
            "pnl": equity - args.initial_cash,
            "return_pct": (equity / args.initial_cash - 1.0) * 100.0,
            "position_symbol": position_symbol,
            "shares": shares,
            "mark_price": mark_price,
            "total_cost": total_cost,
            "total_turnover": total_turnover,
            "latest_candle": str(latest_ts),
            "log_rows": len(rows),
            "output_dir": str(args.output_dir),
            "policy": FINAL_POLICY,
        }
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(
            f"{row['wall_time']} candle={row['latest_candle']} equity={equity:.2f} "
            f"pnl={row['pnl']:.2f} pos={position_symbol} shares={shares:.4f} "
            f"filter={filter_pass} target={target_symbol} conf={confidence:.4f}",
            flush=True,
        )

        if args.once or time.time() >= end_time:
            break
        time.sleep(max(1.0, args.poll_seconds))


if __name__ == "__main__":
    main()
