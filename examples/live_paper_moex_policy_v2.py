import argparse
import datetime as dt
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch

from filtered_neural_policy_lab import compute_market_features
from live_paper_moex_policy import (
    FINAL_POLICY,
    append_live_to_matrix,
    fetch_candles,
    fetch_marketdata,
    load_models,
    score_latest,
)
from neural_policy_lab import build_matrices, read_intraday_matrix


@dataclass
class PositionState:
    symbol: str | None = None
    shares: float = 0.0
    entry_price: float | None = None
    entry_equity: float | None = None
    entry_time: str | None = None
    opened_by: str | None = None


@dataclass(frozen=True)
class Candidate:
    symbol: str | None
    weight: float
    confidence: float
    confidence_edge_bps: float
    score: float | None
    reason: str


BLOCKING_CANDIDATE_REASONS = {"filter_blocked", "edge_below_cost_gate", "no_tradable_price"}


def finite_price(value: object) -> float | None:
    try:
        price = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(price) or price <= 0:
        return None
    return price


def side_confidence(scores: np.ndarray, tradable: np.ndarray, side: str) -> float:
    valid_scores = scores[tradable]
    if valid_scores.size == 0:
        return 0.0
    median = float(np.nanmedian(valid_scores))
    if side == "short":
        return max(0.0, median - float(np.nanmin(valid_scores)))
    if side == "long":
        return max(0.0, float(np.nanmax(valid_scores)) - median)
    raise ValueError(f"unknown side: {side}")


def select_candidate(
    symbols: list[str],
    scores: np.ndarray,
    tradable: np.ndarray,
    prices: dict[str, float],
    *,
    mode: str,
    gross: float,
    confidence_to_bps: float,
    required_edge_bps: float,
    filter_pass: bool,
) -> Candidate:
    if not filter_pass:
        return Candidate(None, 0.0, 0.0, 0.0, None, "filter_blocked")

    candidates = np.flatnonzero(tradable)
    candidates = np.array([idx for idx in candidates if symbols[int(idx)] in prices], dtype=int)
    if candidates.size == 0:
        return Candidate(None, 0.0, 0.0, 0.0, None, "no_tradable_price")

    side = "short"
    if mode == "long_only":
        side = "long"
        selected_idx = int(candidates[np.argmax(scores[candidates])])
        weight = gross
    elif mode == "short_only":
        selected_idx = int(candidates[np.argmin(scores[candidates])])
        weight = -gross
    elif mode == "long_short":
        short_conf = side_confidence(scores, tradable, "short")
        long_conf = side_confidence(scores, tradable, "long")
        if long_conf > short_conf:
            side = "long"
            selected_idx = int(candidates[np.argmax(scores[candidates])])
            weight = gross
        else:
            side = "short"
            selected_idx = int(candidates[np.argmin(scores[candidates])])
            weight = -gross
    else:
        raise ValueError(f"unknown mode: {mode}")

    confidence = side_confidence(scores, tradable, side)
    edge_bps = confidence * confidence_to_bps
    if edge_bps < required_edge_bps:
        return Candidate(None, 0.0, confidence, edge_bps, None, "edge_below_cost_gate")

    symbol = symbols[selected_idx]
    return Candidate(symbol, weight, confidence, edge_bps, float(scores[selected_idx]), f"{side}_candidate")


def mark_equity(cash: float, state: PositionState, prices: dict[str, float]) -> tuple[float, float | None]:
    mark_price = prices.get(state.symbol) if state.symbol else None
    position_value = state.shares * mark_price if state.symbol and mark_price else 0.0
    return cash + position_value, mark_price


def trade_to_target(
    cash: float,
    state: PositionState,
    target_symbol: str | None,
    target_weight: float,
    prices: dict[str, float],
    cost_bps: float,
) -> tuple[float, PositionState, float, float]:
    cost_rate = cost_bps / 10000.0
    total_cost = 0.0
    turnover = 0.0
    equity_before, _ = mark_equity(cash, state, prices)

    if state.symbol and state.symbol != target_symbol:
        current_price = prices.get(state.symbol)
        if current_price:
            trade_value = abs(state.shares) * current_price
            cash += state.shares * current_price
            cost = trade_value * cost_rate
            cash -= cost
            total_cost += cost
            turnover += trade_value
        state = PositionState()

    if target_symbol is None:
        return cash, state, total_cost, turnover

    target_price = prices[target_symbol]
    current_value = state.shares * target_price if state.symbol == target_symbol else 0.0
    target_value = equity_before * target_weight
    if target_value > 0:
        target_value = min(target_value, equity_before / (1.0 + cost_rate))
    delta_value = target_value - current_value
    if abs(delta_value) > 1e-9:
        delta_shares = delta_value / target_price
        cash -= delta_shares * target_price
        cost = abs(delta_value) * cost_rate
        cash -= cost
        total_cost += cost
        turnover += abs(delta_value)
        state.shares += delta_shares
        state.symbol = target_symbol
        if state.entry_price is None:
            state.entry_price = target_price

    return cash, state, total_cost, turnover


def build_live_prices(
    symbols: list[str],
    candle_prices: dict[str, float],
    marketdata: dict[str, dict],
) -> dict[str, float]:
    prices = {}
    for symbol in symbols:
        md_price = finite_price(marketdata.get(symbol, {}).get("price"))
        candle_price = finite_price(candle_prices.get(symbol))
        if md_price is not None:
            prices[symbol] = md_price
        elif candle_price is not None:
            prices[symbol] = candle_price
    return prices


def decide_action(
    state: PositionState,
    candidate: Candidate,
    equity: float,
    mark_price: float | None,
    *,
    stop_loss_pct: float,
    take_profit_pct: float,
    exit_on_filter_fail: bool,
    filter_fail_count: int,
    filter_fail_exit_bars: int,
    switch_target: bool,
) -> tuple[str, str | None, float]:
    if state.symbol is None:
        if candidate.symbol is None:
            return "flat", None, 0.0
        return f"enter_{candidate.reason}", candidate.symbol, candidate.weight

    if mark_price is None or state.entry_equity is None:
        return "hold_no_mark", state.symbol, 0.0

    position_return_pct = (equity / state.entry_equity - 1.0) * 100.0
    if position_return_pct <= -abs(stop_loss_pct):
        return "close_stop_loss", None, 0.0
    if position_return_pct >= abs(take_profit_pct):
        return "close_take_profit", None, 0.0
    if (
        exit_on_filter_fail
        and candidate.reason in BLOCKING_CANDIDATE_REASONS
        and filter_fail_count >= max(1, filter_fail_exit_bars)
    ):
        return f"close_{candidate.reason}_{filter_fail_count}bars", None, 0.0
    if switch_target and candidate.symbol and candidate.symbol != state.symbol:
        return f"switch_to_{candidate.reason}", candidate.symbol, candidate.weight

    return "hold", state.symbol, 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Execution-aware MOEX live paper trading runner.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("datasets/moex_intraday_6f8f836cc811"))
    parser.add_argument("--board", default="TQBR")
    parser.add_argument("--interval", type=int, default=10)
    parser.add_argument("--duration-minutes", type=float, default=30.0)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--initial-cash", type=float, default=10000.0)
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--mode", choices=["short_only", "long_only", "long_short"], default="long_short")
    parser.add_argument("--max-gross", type=float, default=1.0)
    parser.add_argument("--confidence-to-bps", type=float, default=1000.0)
    parser.add_argument("--min-expected-edge-bps", type=float, default=10.0)
    parser.add_argument("--stop-loss-pct", type=float, default=0.75)
    parser.add_argument("--take-profit-pct", type=float, default=0.75)
    parser.add_argument("--exit-on-filter-fail", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--filter-fail-exit-bars", type=int, default=3)
    parser.add_argument("--switch-target", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--close-on-exit", action=argparse.BooleanOptionalAction, default=True)
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
        args.output_dir = Path("outputs") / f"live_paper_moex_policy_v2_{stamp}"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    _, symbols, base_open, base_close, base_volume = read_intraday_matrix(args.dataset_dir, args.interval)
    initial_matrices = build_matrices(base_open, base_close, base_volume, horizon=12)
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    models = load_models(args.checkpoint, initial_matrices["features"].shape[-1], len(symbols), device)

    session = requests.Session()
    session.trust_env = False
    cash = float(args.initial_cash)
    state = PositionState()
    total_cost = 0.0
    total_turnover = 0.0
    filter_fail_count = 0
    rows = []
    start_time = time.time()
    end_time = start_time + args.duration_minutes * 60.0
    effective_gross = min(float(args.max_gross), float(FINAL_POLICY["gross"]))
    required_edge_bps = 2.0 * args.cost_bps + args.min_expected_edge_bps

    while True:
        now = dt.datetime.now()
        from_date = (now.date() - dt.timedelta(days=8)).isoformat()
        till_date = now.date().isoformat()
        live_parts = []
        errors = []
        for symbol in symbols:
            try:
                live_parts.append(fetch_candles(session, symbol, args.board, args.interval, from_date, till_date))
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{symbol}: {exc}")
        live = pd.concat(live_parts, ignore_index=True) if live_parts else pd.DataFrame()
        open_px, close_px, volume = append_live_to_matrix(base_open, base_close, base_volume, live)
        matrices = build_matrices(open_px, close_px, volume, horizon=12)
        market_features = compute_market_features(close_px.index, list(close_px.columns), matrices["bar_return"])

        latest_row = len(close_px.index) - 1
        latest_ts = close_px.index[latest_row]
        latest_scores = score_latest(models, matrices["features"][latest_row], device)
        tradable = matrices["tradable"][latest_row] & np.isfinite(latest_scores)
        market_mom_24 = float(market_features["market_mom_24"][latest_row])
        market_mom_96 = float(market_features["market_mom_96"][latest_row])
        filter_pass = (
            market_mom_24 <= FINAL_POLICY["market_mom_24_max"]
            and market_mom_96 <= FINAL_POLICY["market_mom_96_max"]
        )

        candle_prices = close_px.iloc[latest_row].to_dict()
        marketdata = fetch_marketdata(session, args.board, symbols)
        prices = build_live_prices(symbols, candle_prices, marketdata)
        candidate = select_candidate(
            symbols,
            latest_scores,
            tradable,
            prices,
            mode=args.mode,
            gross=effective_gross,
            confidence_to_bps=args.confidence_to_bps,
            required_edge_bps=required_edge_bps,
            filter_pass=filter_pass,
        )
        if candidate.reason in BLOCKING_CANDIDATE_REASONS:
            filter_fail_count += 1
        else:
            filter_fail_count = 0

        equity_before, mark_price_before = mark_equity(cash, state, prices)
        action, target_symbol, target_weight = decide_action(
            state,
            candidate,
            equity_before,
            mark_price_before,
            stop_loss_pct=args.stop_loss_pct,
            take_profit_pct=args.take_profit_pct,
            exit_on_filter_fail=args.exit_on_filter_fail,
            filter_fail_count=filter_fail_count,
            filter_fail_exit_bars=args.filter_fail_exit_bars,
            switch_target=args.switch_target,
        )

        previous_symbol = state.symbol
        previous_shares = state.shares
        cost = 0.0
        turnover = 0.0
        if action not in {"flat", "hold", "hold_no_mark"}:
            cash, state, cost, turnover = trade_to_target(
                cash,
                state,
                target_symbol,
                target_weight,
                prices,
                args.cost_bps,
            )
            total_cost += cost
            total_turnover += turnover
            if state.symbol and (state.symbol != previous_symbol or previous_shares == 0):
                state.entry_price = prices[state.symbol]
                state.entry_equity = mark_equity(cash, state, prices)[0]
                state.entry_time = now.isoformat(timespec="seconds")
                state.opened_by = action
            if state.symbol is None:
                state = PositionState()

        equity, mark_price = mark_equity(cash, state, prices)
        position_return_pct = None
        if state.symbol and state.entry_equity:
            position_return_pct = (equity / state.entry_equity - 1.0) * 100.0

        row = {
            "wall_time": now.isoformat(timespec="seconds"),
            "latest_candle": str(latest_ts),
            "cash": cash,
            "position_symbol": state.symbol,
            "shares": state.shares,
            "mark_price": mark_price,
            "equity": equity,
            "pnl": equity - args.initial_cash,
            "return_pct": (equity / args.initial_cash - 1.0) * 100.0,
            "position_return_pct": position_return_pct,
            "action": action,
            "candidate_symbol": candidate.symbol,
            "candidate_weight": candidate.weight,
            "candidate_confidence": candidate.confidence,
            "candidate_edge_bps": candidate.confidence_edge_bps,
            "candidate_reason": candidate.reason,
            "filter_pass": filter_pass,
            "filter_fail_count": filter_fail_count,
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

        if args.once or time.time() >= end_time:
            break
        print(
            f"{row['wall_time']} equity={equity:.2f} pnl={row['pnl']:.2f} "
            f"action={action} pos={state.symbol} candidate={candidate.symbol} "
            f"edge={candidate.confidence_edge_bps:.1f}bps",
            flush=True,
        )
        time.sleep(max(1.0, args.poll_seconds))

    final_close_cost = 0.0
    final_close_turnover = 0.0
    final_close_equity = equity
    if args.close_on_exit and state.symbol:
        now = dt.datetime.now()
        cash, state, final_close_cost, final_close_turnover = trade_to_target(
            cash,
            state,
            None,
            0.0,
            prices,
            args.cost_bps,
        )
        total_cost += final_close_cost
        total_turnover += final_close_turnover
        final_close_equity, _ = mark_equity(cash, state, prices)
        rows.append(
            {
                "wall_time": now.isoformat(timespec="seconds"),
                "latest_candle": str(latest_ts),
                "cash": cash,
                "position_symbol": None,
                "shares": 0.0,
                "mark_price": None,
                "equity": final_close_equity,
                "pnl": final_close_equity - args.initial_cash,
                "return_pct": (final_close_equity / args.initial_cash - 1.0) * 100.0,
                "position_return_pct": None,
                "action": "final_close",
                "candidate_symbol": candidate.symbol,
                "candidate_weight": candidate.weight,
                "candidate_confidence": candidate.confidence,
                "candidate_edge_bps": candidate.confidence_edge_bps,
                "candidate_reason": candidate.reason,
                "filter_pass": filter_pass,
                "filter_fail_count": filter_fail_count,
                "market_mom_24": market_mom_24,
                "market_mom_96": market_mom_96,
                "trade_cost": final_close_cost,
                "turnover": final_close_turnover,
                "total_cost": total_cost,
                "total_turnover": total_turnover,
                "errors": "",
            }
        )
        pd.DataFrame(rows).to_csv(args.output_dir / "live_log.csv", index=False)

    summary = {
        "started_at": dt.datetime.fromtimestamp(start_time).isoformat(timespec="seconds"),
        "updated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "initial_cash": args.initial_cash,
        "mark_to_market_equity": equity,
        "mark_to_market_pnl": equity - args.initial_cash,
        "mark_to_market_return_pct": (equity / args.initial_cash - 1.0) * 100.0,
        "final_equity_after_close": final_close_equity,
        "final_pnl_after_close": final_close_equity - args.initial_cash,
        "final_return_pct_after_close": (final_close_equity / args.initial_cash - 1.0) * 100.0,
        "total_cost": total_cost,
        "total_turnover": total_turnover,
        "final_close_cost": final_close_cost,
        "log_rows": len(rows),
        "output_dir": str(args.output_dir),
        "execution_config": {
            "mode": args.mode,
            "max_gross": effective_gross,
            "required_edge_bps": required_edge_bps,
            "stop_loss_pct": args.stop_loss_pct,
            "take_profit_pct": args.take_profit_pct,
            "exit_on_filter_fail": args.exit_on_filter_fail,
            "filter_fail_exit_bars": args.filter_fail_exit_bars,
            "switch_target": args.switch_target,
            "close_on_exit": args.close_on_exit,
        },
        "base_policy": FINAL_POLICY,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"FINAL equity_after_close={final_close_equity:.2f} "
        f"pnl={final_close_equity - args.initial_cash:.2f} "
        f"return={(final_close_equity / args.initial_cash - 1.0) * 100.0:.4f}% "
        f"rows={len(rows)} output={args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
