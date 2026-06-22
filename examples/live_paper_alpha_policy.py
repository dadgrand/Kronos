import argparse
import datetime as dt
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))

from filtered_neural_policy_lab import compute_market_features
from live_paper_moex_policy import append_live_to_matrix, fetch_candles, fetch_marketdata
from live_paper_moex_policy_v2 import build_live_prices, finite_price
from neural_policy_lab import build_matrices, read_intraday_matrix
from stop_aware_filtered_policy_lab import backtest_stop_aware, score_confidence, segment_diagnostics, selection_score, select_target
from walk_forward_alpha_policy_lab import (
    build_alpha_scores,
    build_candidates,
    constraints_pass,
    stop_policy,
)
from walk_forward_stop_aware_policy_lab import parse_modes


def load_protocol_config(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("config", payload)


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, Path):
        return str(value)
    return value


def parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_csv_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_csv_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def mark_portfolio(cash: float, positions: dict[str, float], prices: dict[str, float]) -> tuple[float, dict[str, float]]:
    values = {}
    equity = float(cash)
    for symbol, shares in positions.items():
        price = prices.get(symbol)
        if price is None:
            values[symbol] = np.nan
            continue
        value = float(shares) * float(price)
        values[symbol] = value
        equity += value
    return equity, values


def trade_to_weights(
    cash: float,
    positions: dict[str, float],
    target_weights: dict[str, float],
    prices: dict[str, float],
    cost_bps: float,
) -> tuple[float, dict[str, float], float, float, list[dict]]:
    cost_rate = cost_bps / 10000.0
    equity, _ = mark_portfolio(cash, positions, prices)
    total_cost = 0.0
    turnover = 0.0
    trades = []
    updated = dict(positions)

    symbols = sorted(set(updated) | set(target_weights))
    for symbol in symbols:
        price = prices.get(symbol)
        if price is None or not np.isfinite(price) or price <= 0:
            continue
        current_shares = float(updated.get(symbol, 0.0))
        current_value = current_shares * price
        target_value = equity * float(target_weights.get(symbol, 0.0))
        delta_value = target_value - current_value
        if abs(delta_value) <= 1e-9:
            continue
        delta_shares = delta_value / price
        trade_cost = abs(delta_value) * cost_rate
        cash -= delta_shares * price
        cash -= trade_cost
        total_cost += trade_cost
        turnover += abs(delta_value)
        new_shares = current_shares + delta_shares
        if abs(new_shares) <= 1e-10:
            updated.pop(symbol, None)
        else:
            updated[symbol] = new_shares
        trades.append(
            {
                "symbol": symbol,
                "price": price,
                "delta_value": delta_value,
                "delta_shares": delta_shares,
                "target_weight": float(target_weights.get(symbol, 0.0)),
                "trade_cost": trade_cost,
            }
        )
    return cash, updated, total_cost, turnover, trades


def select_validation_candidate(
    *,
    alpha_scores: dict[str, np.ndarray],
    confidences: dict[tuple[str, str], np.ndarray],
    matrices: dict,
    masks: dict[str, np.ndarray],
    market_features: dict[str, np.ndarray],
    candidates: list,
    config: dict,
) -> tuple[dict | None, dict | None, list[dict]]:
    rows = []
    best = None
    best_any = None
    for candidate in candidates:
        scores = alpha_scores[candidate.alpha_name]
        confidence = confidences[(candidate.alpha_name, candidate.mode)]
        policy = stop_policy(candidate)
        validation_summary, validation_bars = backtest_stop_aware(
            scores,
            matrices,
            masks,
            market_features,
            confidence,
            "validation",
            policy,
            initial_cash=float(config["initial_cash"]),
            cost_bps=float(config["cost_bps"]),
        )
        validation_stress_summary, _ = backtest_stop_aware(
            scores,
            matrices,
            masks,
            market_features,
            confidence,
            "validation",
            policy,
            initial_cash=float(config["initial_cash"]),
            cost_bps=float(config["stress_cost_bps"]),
        )
        validation_segments = segment_diagnostics(
            validation_bars,
            split="validation",
            initial_cash=float(config["initial_cash"]),
            segment_count=int(config["segment_count"]),
        )
        worst_segment = float(validation_segments["return_pct"].min()) if not validation_segments.empty else None
        raw_score = selection_score(
            validation_summary,
            float(config["drawdown_penalty"]),
            float(config["turnover_penalty"]),
        )
        selection = (
            raw_score
            + float(config["segment_selection_penalty"]) * float(worst_segment if worst_segment is not None else -999.0)
            + float(config["stress_selection_weight"]) * float(validation_stress_summary["return_pct"])
        )
        ok = constraints_pass(
            validation_summary,
            validation_stress_summary,
            worst_segment,
            min_validation_return_pct=config.get("min_validation_return_pct"),
            min_validation_stress_return_pct=config.get("min_validation_stress_return_pct"),
            min_validation_segment_return_pct=config.get("min_validation_segment_return_pct"),
            max_validation_drawdown_pct=config.get("max_validation_drawdown_pct"),
        )
        row = {
            **asdict(candidate),
            "validation_return_pct": validation_summary["return_pct"],
            "validation_max_drawdown_pct": validation_summary["max_drawdown_pct"],
            "validation_stress_return_pct": validation_stress_summary["return_pct"],
            "worst_validation_segment_return_pct": worst_segment,
            "raw_selection_score": raw_score,
            "constraints_ok": ok,
            "selection_score": selection if ok else -np.inf,
        }
        rows.append(row)
        if best_any is None or selection > best_any["raw_candidate_score"]:
            best_any = {**row, "raw_candidate_score": selection, "policy": policy}
        if ok and (best is None or selection > best["selection_score"]):
            best = {**row, "policy": policy}
    return best, best_any, rows


def target_weights_from_policy(
    *,
    symbols: list[str],
    scores: np.ndarray,
    tradable: np.ndarray,
    prices: dict[str, float],
    latest_row: int,
    policy,
) -> dict[str, float]:
    price_mask = np.array([symbol in prices for symbol in symbols], dtype=bool)
    tradable_for_policy = tradable.copy()
    tradable_for_policy[latest_row] &= price_mask
    target = select_target(scores, tradable_for_policy, latest_row, policy)
    return {symbols[idx]: float(weight) for idx, weight in enumerate(target) if abs(float(weight)) > 1e-12}


def build_candidate_log_rows(
    *,
    wall_time: str,
    latest_candle: str,
    symbols: list[str],
    scores: np.ndarray,
    tradable: np.ndarray,
    prices: dict[str, float],
    candle_prices: dict[str, float],
    marketdata: dict[str, dict],
    target_weights: dict[str, float],
    selected: dict | None,
    best_any: dict | None,
    top_n: int,
) -> list[dict]:
    valid = np.isfinite(scores) & tradable & np.array([symbol in prices for symbol in symbols], dtype=bool)
    valid_ids = np.flatnonzero(valid)
    short_rank = np.full(len(symbols), np.nan, dtype=np.float64)
    long_rank = np.full(len(symbols), np.nan, dtype=np.float64)
    median_score = np.nan
    if valid_ids.size:
        short_order = valid_ids[np.argsort(scores[valid_ids])]
        long_order = valid_ids[np.argsort(scores[valid_ids])[::-1]]
        short_rank[short_order] = np.arange(1, len(short_order) + 1)
        long_rank[long_order] = np.arange(1, len(long_order) + 1)
        median_score = float(np.nanmedian(scores[valid_ids]))

    rows = []
    selected_symbols = set(target_weights)
    reason = "validation_gate_passed" if selected is not None else "validation_gate_blocked"
    source = selected or best_any or {}
    for idx, symbol in enumerate(symbols):
        if top_n > 0:
            in_top = (
                (np.isfinite(short_rank[idx]) and short_rank[idx] <= top_n)
                or (np.isfinite(long_rank[idx]) and long_rank[idx] <= top_n)
                or symbol in selected_symbols
            )
            if not in_top:
                continue
        score = float(scores[idx]) if np.isfinite(scores[idx]) else np.nan
        md_price = finite_price(marketdata.get(symbol, {}).get("price"))
        candle_price = finite_price(candle_prices.get(symbol))
        rows.append(
            {
                "wall_time": wall_time,
                "latest_candle": latest_candle,
                "symbol": symbol,
                "score": score,
                "score_minus_median": score - median_score if np.isfinite(score) and np.isfinite(median_score) else np.nan,
                "short_rank": short_rank[idx],
                "long_rank": long_rank[idx],
                "tradable": bool(tradable[idx]),
                "has_live_price": symbol in prices,
                "price": prices.get(symbol),
                "marketdata_price": md_price,
                "candle_price": candle_price,
                "is_selected_candidate": symbol in selected_symbols,
                "candidate_reason": reason,
                "candidate_weight": float(target_weights.get(symbol, 0.0)),
                "filter_pass": selected is not None,
                "trade_mode": source.get("mode"),
                "alpha_name": source.get("alpha_name"),
                "policy_gross": source.get("gross"),
                "policy_confidence_min": 0.0,
                "policy_rebalance_every": source.get("rebalance_every"),
                "validation_return_pct": source.get("validation_return_pct"),
                "worst_validation_segment_return_pct": source.get("worst_validation_segment_return_pct"),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Live paper runner for the frozen strict alpha protocol.")
    parser.add_argument("--config-json", type=Path, default=Path("configs/strict_alpha_policy_20260622.json"))
    parser.add_argument("--board", default="TQBR")
    parser.add_argument("--duration-minutes", type=float, default=30.0)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--fetch-days", type=int, default=8)
    parser.add_argument("--candidate-log-top-n", type=int, default=0)
    parser.add_argument("--close-on-exit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    config = load_protocol_config(args.config_json)
    if args.output_dir is None:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = Path("outputs") / f"live_paper_alpha_policy_{stamp}"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    _, symbols, base_open, base_close, base_volume = read_intraday_matrix(Path(config["dataset_dir"]), int(config["interval"]))
    session = requests.Session()
    session.trust_env = False
    cash = float(config["initial_cash"])
    positions: dict[str, float] = {}
    total_cost = 0.0
    total_turnover = 0.0
    last_rebalance_row: int | None = None
    live_rows = []
    candidate_rows = []
    start_time = time.time()
    end_time = start_time + args.duration_minutes * 60.0

    while True:
        now = dt.datetime.now()
        from_date = (now.date() - dt.timedelta(days=args.fetch_days)).isoformat()
        till_date = now.date().isoformat()
        live_parts = []
        errors = []
        for symbol in symbols:
            try:
                live_parts.append(fetch_candles(session, symbol, args.board, int(config["interval"]), from_date, till_date))
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{symbol}: {exc}")
        live = pd.concat(live_parts, ignore_index=True) if live_parts else pd.DataFrame()
        open_px, close_px, volume = append_live_to_matrix(base_open, base_close, base_volume, live)
        matrices = build_matrices(open_px, close_px, volume, int(config["horizon"]))
        market_features = compute_market_features(close_px.index, list(close_px.columns), matrices["bar_return"])

        latest_row = len(close_px.index) - 1
        latest_ts = pd.Timestamp(close_px.index[latest_row])
        validation_end = latest_ts
        validation_start = validation_end - pd.Timedelta(days=int(config["validation_days"]))
        validation_mask = np.asarray((close_px.index >= validation_start) & (close_px.index < validation_end))
        masks = {"validation": validation_mask}

        alpha_scores = build_alpha_scores(
            close_px,
            lags=parse_csv_ints(config["alpha_lags"]),
            kinds=parse_csv_strings(config["alpha_kinds"]),
            normalize=config["alpha_normalize"],
        )
        modes = parse_modes(config["modes"])
        confidences = {
            (alpha_name, mode): score_confidence(scores, matrices["tradable"], mode)
            for alpha_name, scores in alpha_scores.items()
            for mode in modes
        }
        candidates = build_candidates(
            list(alpha_scores),
            modes=modes,
            k_grid=parse_csv_ints(config["k_grid"]),
            gross_grid=parse_csv_floats(config["gross_grid"]),
            rebalance_grid=parse_csv_ints(config["rebalance_grid"]),
        )
        selected, best_any, search_rows = select_validation_candidate(
            alpha_scores=alpha_scores,
            confidences=confidences,
            matrices=matrices,
            masks=masks,
            market_features=market_features,
            candidates=candidates,
            config=config,
        )

        candle_prices = close_px.iloc[latest_row].to_dict()
        marketdata = fetch_marketdata(session, args.board, symbols)
        prices = build_live_prices(symbols, candle_prices, marketdata)

        target_weights = {}
        score_source = selected or best_any
        if selected is not None:
            selected_policy = selected["policy"]
            selected_scores = alpha_scores[selected["alpha_name"]]
            target_weights = target_weights_from_policy(
                symbols=symbols,
                scores=selected_scores,
                tradable=matrices["tradable"] & np.isfinite(selected_scores),
                prices=prices,
                latest_row=latest_row,
                policy=selected_policy,
            )
        selected_scores_for_log = (
            alpha_scores[score_source["alpha_name"]]
            if score_source is not None
            else np.zeros_like(matrices["bar_return"], dtype=np.float32)
        )

        rebalance_due = (
            selected is None
            or last_rebalance_row is None
            or (latest_row - last_rebalance_row) >= int(selected.get("rebalance_every", 1))
            or (not positions and bool(target_weights))
        )
        cost = 0.0
        turnover = 0.0
        trades = []
        action = "hold"
        if rebalance_due:
            cash, positions, cost, turnover, trades = trade_to_weights(
                cash,
                positions,
                target_weights,
                prices,
                float(config["cost_bps"]),
            )
            total_cost += cost
            total_turnover += turnover
            last_rebalance_row = latest_row
            action = "rebalance" if trades else ("cash" if not target_weights else "no_trade")

        equity, position_values = mark_portfolio(cash, positions, prices)
        wall_time = now.isoformat(timespec="seconds")
        target_symbols = ",".join(sorted(target_weights))
        position_symbols = ",".join(sorted(positions))
        candidate_rows.extend(
            build_candidate_log_rows(
                wall_time=wall_time,
                latest_candle=str(latest_ts),
                symbols=symbols,
                scores=selected_scores_for_log[latest_row],
                tradable=matrices["tradable"][latest_row] & np.isfinite(selected_scores_for_log[latest_row]),
                prices=prices,
                candle_prices=candle_prices,
                marketdata=marketdata,
                target_weights=target_weights,
                selected=selected,
                best_any=best_any,
                top_n=args.candidate_log_top_n,
            )
        )
        live_rows.append(
            {
                "wall_time": wall_time,
                "latest_candle": str(latest_ts),
                "validation_start": str(validation_start),
                "validation_end": str(validation_end),
                "action": action,
                "cash": cash,
                "equity": equity,
                "pnl": equity - float(config["initial_cash"]),
                "return_pct": (equity / float(config["initial_cash"]) - 1.0) * 100.0,
                "position_symbols": position_symbols,
                "positions_json": json.dumps(positions, ensure_ascii=False, sort_keys=True),
                "position_values_json": json.dumps(position_values, ensure_ascii=False, sort_keys=True),
                "target_symbols": target_symbols,
                "target_weights_json": json.dumps(target_weights, ensure_ascii=False, sort_keys=True),
                "selected_alpha": selected.get("alpha_name") if selected else None,
                "selected_mode": selected.get("mode") if selected else None,
                "selected_k": selected.get("k") if selected else None,
                "selected_gross": selected.get("gross") if selected else None,
                "selected_rebalance_every": selected.get("rebalance_every") if selected else None,
                "validation_return_pct": selected.get("validation_return_pct") if selected else None,
                "worst_validation_segment_return_pct": (
                    selected.get("worst_validation_segment_return_pct") if selected else None
                ),
                "best_blocked_alpha": best_any.get("alpha_name") if best_any else None,
                "best_blocked_mode": best_any.get("mode") if best_any else None,
                "trade_cost": cost,
                "turnover": turnover,
                "total_cost": total_cost,
                "total_turnover": total_turnover,
                "trade_count": len(trades),
                "trades_json": json.dumps(trades, ensure_ascii=False),
                "candidate_count": len(search_rows),
                "constraints_passed_count": int(sum(bool(row["constraints_ok"]) for row in search_rows)),
                "errors": " | ".join(errors[:5]),
            }
        )
        pd.DataFrame(live_rows).to_csv(args.output_dir / "live_log.csv", index=False)
        pd.DataFrame(candidate_rows).to_csv(args.output_dir / "candidate_log.csv", index=False)
        pd.DataFrame(search_rows).to_csv(args.output_dir / "validation_search_latest.csv", index=False)

        summary = {
            "started_at": dt.datetime.fromtimestamp(start_time).isoformat(timespec="seconds"),
            "updated_at": wall_time,
            "config_json": str(args.config_json),
            "initial_cash": float(config["initial_cash"]),
            "equity": equity,
            "pnl": equity - float(config["initial_cash"]),
            "return_pct": (equity / float(config["initial_cash"]) - 1.0) * 100.0,
            "cash": cash,
            "positions": positions,
            "total_cost": total_cost,
            "total_turnover": total_turnover,
            "latest_candle": str(latest_ts),
            "validation_start": str(validation_start),
            "validation_end": str(validation_end),
            "log_rows": len(live_rows),
            "candidate_log_rows": len(candidate_rows),
            "output_dir": str(args.output_dir),
            "selected": {key: value for key, value in (selected or {}).items() if key != "policy"},
            "best_any": {key: value for key, value in (best_any or {}).items() if key != "policy"},
            "constraints_passed_count": int(sum(bool(row["constraints_ok"]) for row in search_rows)),
        }
        (args.output_dir / "summary.json").write_text(
            json.dumps(json_safe(summary), indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
        print(
            f"{wall_time} candle={latest_ts} equity={equity:.2f} pnl={summary['pnl']:.2f} "
            f"action={action} selected={summary['selected'].get('alpha_name')} targets={target_symbols or 'cash'} "
            f"passed={summary['constraints_passed_count']}",
            flush=True,
        )

        if args.once or time.time() >= end_time:
            break
        time.sleep(max(1.0, args.poll_seconds))

    if args.close_on_exit and positions:
        now = dt.datetime.now().isoformat(timespec="seconds")
        cash, positions, cost, turnover, trades = trade_to_weights(cash, positions, {}, prices, float(config["cost_bps"]))
        total_cost += cost
        total_turnover += turnover
        equity, position_values = mark_portfolio(cash, positions, prices)
        live_rows.append(
            {
                "wall_time": now,
                "latest_candle": str(latest_ts),
                "validation_start": str(validation_start),
                "validation_end": str(validation_end),
                "action": "final_close",
                "cash": cash,
                "equity": equity,
                "pnl": equity - float(config["initial_cash"]),
                "return_pct": (equity / float(config["initial_cash"]) - 1.0) * 100.0,
                "position_symbols": "",
                "positions_json": "{}",
                "position_values_json": json.dumps(position_values, ensure_ascii=False, sort_keys=True),
                "target_symbols": "",
                "target_weights_json": "{}",
                "selected_alpha": None,
                "selected_mode": None,
                "selected_k": None,
                "selected_gross": None,
                "selected_rebalance_every": None,
                "validation_return_pct": None,
                "worst_validation_segment_return_pct": None,
                "best_blocked_alpha": None,
                "best_blocked_mode": None,
                "trade_cost": cost,
                "turnover": turnover,
                "total_cost": total_cost,
                "total_turnover": total_turnover,
                "trade_count": len(trades),
                "trades_json": json.dumps(trades, ensure_ascii=False),
                "candidate_count": 0,
                "constraints_passed_count": 0,
                "errors": "",
            }
        )
        pd.DataFrame(live_rows).to_csv(args.output_dir / "live_log.csv", index=False)
        summary.update(
            {
                "updated_at": now,
                "cash": cash,
                "equity": equity,
                "pnl": equity - float(config["initial_cash"]),
                "return_pct": (equity / float(config["initial_cash"]) - 1.0) * 100.0,
                "final_equity_after_close": equity,
                "final_return_pct_after_close": (equity / float(config["initial_cash"]) - 1.0) * 100.0,
                "positions": positions,
                "total_cost": total_cost,
                "total_turnover": total_turnover,
            }
        )
        (args.output_dir / "summary.json").write_text(
            json.dumps(json_safe(summary), indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
