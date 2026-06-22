import argparse
import datetime as dt
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))

from diplom_risk_adapter import DiplomRiskAdapter, config_from_mapping, empty_diplom_summary, summarize_diplom_diagnostics
from filtered_neural_policy_lab import compute_market_features
from live_paper_moex_policy import append_live_to_matrix, fetch_candles, fetch_marketdata
from live_paper_moex_policy_v2 import build_live_prices, finite_price
from neural_policy_lab import build_matrices, read_intraday_matrix
from stop_aware_filtered_policy_lab import backtest_stop_aware, score_confidence, segment_diagnostics, selection_score, select_target
from walk_forward_alpha_policy_lab import (
    build_alpha_scores,
    build_candidates,
    build_regime_history,
    build_trade_probability_model,
    calibration_stats_from_history,
    delay_market_features,
    delay_matrix,
    apply_diplom_gate_penalty,
    diplom_score_penalty,
    gate_trade_passes,
    make_diplom_target_adjuster,
    market_regime_profile,
    mode_default_threshold,
    score_may_like_regime,
    selection_allowed_by_mode,
    split_nested_masks,
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


def max_drawdown_pct(equity: np.ndarray) -> float:
    if equity.size == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    return float(((equity / peak) - 1.0).min() * 100.0)


def fast_select_target(
    scores: np.ndarray,
    tradable: np.ndarray,
    row: int,
    policy,
) -> np.ndarray:
    target = np.zeros(scores.shape[1], dtype=np.float32)
    valid = np.isfinite(scores[row]) & tradable[row]
    required = 2 * policy.k if policy.mode == "long_short" else policy.k
    if valid.sum() < required:
        return target

    candidates = np.flatnonzero(valid)
    order = candidates[np.argsort(scores[row, candidates])]
    if policy.mode == "short_only":
        target[order[: policy.k]] = -policy.gross / policy.k
    elif policy.mode == "long_only":
        target[order[-policy.k :]] = policy.gross / policy.k
    elif policy.mode == "long_short":
        target[order[: policy.k]] = -(policy.gross / 2.0) / policy.k
        target[order[-policy.k :]] = (policy.gross / 2.0) / policy.k
    else:
        raise ValueError(f"unsupported mode: {policy.mode}")
    return target


def fast_backtest_stopless(
    scores: np.ndarray,
    matrices: dict,
    masks: dict[str, np.ndarray],
    split: str,
    policy,
    *,
    initial_cash: float,
    cost_bps: float,
    target_adjuster=None,
) -> tuple[dict, np.ndarray]:
    rows = np.flatnonzero(masks[split])
    weights = np.zeros(scores.shape[1], dtype=np.float32)
    equity = float(initial_cash)
    cost_rate = cost_bps / 10000.0
    equity_curve = np.empty(len(rows), dtype=np.float64)
    bar_returns = np.empty(len(rows), dtype=np.float64)
    turnovers = np.empty(len(rows), dtype=np.float64)
    gross_exposures = np.empty(len(rows), dtype=np.float64)
    bar_return = matrices["bar_return"]
    tradable = matrices["tradable"]
    diplom_risk_diags = []

    for local_idx, row in enumerate(rows):
        if local_idx % policy.rebalance_every == 0:
            target = fast_select_target(scores, tradable, row, policy)
            if target_adjuster is not None:
                target, diplom_diag = target_adjuster(target, row)
                diplom_risk_diags.append(diplom_diag)
            turnover = float(np.abs(target - weights).sum())
            weights = target
        else:
            turnover = 0.0
        gross_bar_return = float(np.nan_to_num(bar_return[row], nan=0.0) @ weights)
        net_return = gross_bar_return - turnover * cost_rate
        equity *= max(0.0, 1.0 + net_return)
        equity_curve[local_idx] = equity
        bar_returns[local_idx] = net_return
        turnovers[local_idx] = turnover
        gross_exposures[local_idx] = float(np.abs(weights).sum())

    if rows.size == 0:
        summary = {
            "split": split,
            "initial_cash": initial_cash,
            "final_equity": initial_cash,
            "pnl": 0.0,
            "return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "sharpe_like": 0.0,
            "bars": 0,
            "active_rate": 0.0,
            "avg_turnover_per_bar": 0.0,
            "total_turnover": 0.0,
            "mean_gross_exposure": 0.0,
            "stop_exits": 0,
            "take_profit_exits": 0,
            "filter_exits": 0,
        }
        if target_adjuster is not None:
            summary.update(summarize_diplom_diagnostics(diplom_risk_diags, disabled_reason="no_position"))
        return summary, equity_curve

    summary = {
        "split": split,
        "initial_cash": initial_cash,
        "final_equity": float(equity_curve[-1]),
        "pnl": float(equity_curve[-1] - initial_cash),
        "return_pct": float((equity_curve[-1] / initial_cash - 1.0) * 100.0),
        "max_drawdown_pct": max_drawdown_pct(equity_curve),
        "sharpe_like": float(bar_returns.mean() / (bar_returns.std(ddof=0) + 1e-12) * np.sqrt(252 * 50)),
        "bars": int(len(rows)),
        "active_rate": float((gross_exposures > 0.0).mean()),
        "avg_turnover_per_bar": float(turnovers.mean()),
        "total_turnover": float(turnovers.sum()),
        "mean_gross_exposure": float(gross_exposures.mean()),
        "stop_exits": 0,
        "take_profit_exits": 0,
        "filter_exits": 0,
    }
    if target_adjuster is not None:
        summary.update(summarize_diplom_diagnostics(diplom_risk_diags, disabled_reason="no_position"))
    return summary, equity_curve


def fast_worst_segment_return(equity_curve: np.ndarray, initial_cash: float, segment_count: int) -> float | None:
    if equity_curve.size == 0 or segment_count <= 0:
        return None
    previous_equity = float(initial_cash)
    worst = None
    for index_part in np.array_split(np.arange(len(equity_curve)), segment_count):
        if index_part.size == 0:
            continue
        end_equity = float(equity_curve[index_part[-1]])
        segment_return = (end_equity / previous_equity - 1.0) * 100.0
        worst = segment_return if worst is None else min(worst, segment_return)
        previous_equity = end_equity
    return worst


def current_alpha_score_vector(close_px: pd.DataFrame, alpha_name: str, normalize: str) -> np.ndarray:
    kind, lag_text = alpha_name.rsplit("_", 1)
    lag = int(lag_text)
    close_ff = close_px.ffill(limit=200)
    current = close_ff.iloc[-1]
    lagged = close_ff.shift(lag).iloc[-1]
    mom = current / lagged - 1.0
    ret = close_ff.pct_change(fill_method=None)
    vol = ret.shift(1).rolling(48, min_periods=12).std().iloc[-1].replace(0.0, np.nan)
    raw_by_kind = {
        "mom": mom,
        "rev": -mom,
        "voladj_mom": mom / vol,
        "voladj_rev": -mom / vol,
    }
    if kind not in raw_by_kind:
        raise ValueError(f"unknown alpha kind={kind}; parsed from {alpha_name}")
    score = raw_by_kind[kind].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    if normalize == "cs_zscore":
        mean = float(score.mean())
        std = float(score.std())
        score = (score - mean) / max(std, 1e-6)
    elif normalize != "none":
        raise ValueError(f"unknown normalize={normalize}")
    return score.to_numpy(dtype=np.float32)


def fetch_symbol_candles(symbol: str, board: str, interval: int, from_date: str, till_date: str) -> pd.DataFrame:
    session = requests.Session()
    session.trust_env = False
    return fetch_candles(session, symbol, board, interval, from_date, till_date)


def fetch_live_candles(
    symbols: list[str],
    *,
    board: str,
    interval: int,
    from_date: str,
    till_date: str,
    workers: int,
) -> tuple[pd.DataFrame, list[str], dict]:
    started = time.perf_counter()
    live_parts = []
    errors = []
    if workers <= 1:
        for symbol in symbols:
            try:
                live_parts.append(fetch_symbol_candles(symbol, board, interval, from_date, till_date))
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{symbol}: {exc}")
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(fetch_symbol_candles, symbol, board, interval, from_date, till_date): symbol
                for symbol in symbols
            }
            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    live_parts.append(future.result())
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"{symbol}: {exc}")
    live = pd.concat(live_parts, ignore_index=True) if live_parts else pd.DataFrame()
    stats = {
        "fetch_elapsed_seconds": time.perf_counter() - started,
        "fetch_parts": len(live_parts),
        "fetch_errors": len(errors),
        "fetch_rows": int(len(live)),
    }
    return live, errors, stats


def parse_marketdata_timestamp(value: object, reference: pd.Timestamp) -> pd.Timestamp | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    if " " in text or "-" in text:
        timestamp = pd.to_datetime(text, errors="coerce")
    else:
        timestamp = pd.to_datetime(f"{reference.date()} {text}", errors="coerce")
    if pd.isna(timestamp):
        return None
    timestamp = pd.Timestamp(timestamp)
    if timestamp > reference + pd.Timedelta(minutes=1):
        timestamp -= pd.Timedelta(days=1)
    return timestamp


def marketdata_update_timestamp(row: dict, reference: pd.Timestamp) -> pd.Timestamp | None:
    system_time = parse_marketdata_timestamp(row.get("system_time"), reference) or reference
    for key in ["update_time", "time"]:
        timestamp = parse_marketdata_timestamp(row.get(key), system_time)
        if timestamp is not None:
            return timestamp
    return None


def append_marketdata_signal_bar(
    open_px: pd.DataFrame,
    close_px: pd.DataFrame,
    volume: pd.DataFrame,
    marketdata: dict[str, dict],
    *,
    now: dt.datetime,
    interval: int,
    min_coverage: float,
    max_age_minutes: float | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    symbols = list(close_px.columns)
    reference = pd.Timestamp(now)
    raw_prices = 0
    live_prices = {}
    price_ages = []
    stale_prices = 0
    missing_update_times = 0
    for symbol in symbols:
        row = marketdata.get(symbol, {})
        price = finite_price(row.get("price"))
        if price is None:
            continue
        raw_prices += 1
        update_ts = marketdata_update_timestamp(row, reference)
        if update_ts is None:
            missing_update_times += 1
            continue
        age_minutes = (reference - update_ts).total_seconds() / 60.0
        if age_minutes < -1.0:
            stale_prices += 1
            continue
        if max_age_minutes is not None and age_minutes > float(max_age_minutes):
            stale_prices += 1
            continue
        live_prices[symbol] = price
        price_ages.append(max(0.0, age_minutes))
    coverage = len(live_prices) / max(1, len(symbols))
    raw_coverage = raw_prices / max(1, len(symbols))
    stats = {
        "marketdata_signal_bar_used": False,
        "marketdata_signal_coverage": coverage,
        "marketdata_signal_raw_coverage": raw_coverage,
        "marketdata_signal_prices": len(live_prices),
        "marketdata_signal_symbols": ",".join(sorted(live_prices)),
        "marketdata_signal_raw_prices": raw_prices,
        "marketdata_signal_stale_prices": stale_prices,
        "marketdata_signal_missing_update_times": missing_update_times,
        "marketdata_signal_max_age_minutes": max_age_minutes,
        "marketdata_signal_mean_price_age_minutes": float(np.mean(price_ages)) if price_ages else None,
        "marketdata_signal_worst_price_age_minutes": float(np.max(price_ages)) if price_ages else None,
        "marketdata_signal_timestamp": None,
    }
    if coverage < min_coverage or not live_prices:
        return open_px, close_px, volume, stats

    timestamp = pd.Timestamp(now).floor(f"{int(interval)}min")
    previous_close = close_px.ffill(limit=200).iloc[-1].reindex(symbols)
    open_row = previous_close.copy()
    close_row = previous_close.copy()
    volume_row = pd.Series(0.0, index=symbols, dtype=np.float64)
    for symbol, price in live_prices.items():
        close_row.loc[symbol] = price
        volume_row.loc[symbol] = 1.0

    open_px = open_px.copy()
    close_px = close_px.copy()
    volume = volume.copy()
    open_px.loc[timestamp, symbols] = open_row
    close_px.loc[timestamp, symbols] = close_row
    volume.loc[timestamp, symbols] = volume_row
    open_px = open_px.sort_index()
    close_px = close_px.sort_index()
    volume = volume.sort_index()
    open_px = open_px[~open_px.index.duplicated(keep="last")]
    close_px = close_px[~close_px.index.duplicated(keep="last")]
    volume = volume[~volume.index.duplicated(keep="last")]
    stats.update(
        {
            "marketdata_signal_bar_used": True,
            "marketdata_signal_timestamp": str(timestamp),
            "marketdata_signal_symbols": ",".join(sorted(live_prices)),
        }
    )
    return open_px, close_px, volume, stats


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


def weights_changed(previous: dict[str, float], current: dict[str, float], tolerance: float) -> bool:
    symbols = set(previous) | set(current)
    return any(abs(float(previous.get(symbol, 0.0)) - float(current.get(symbol, 0.0))) > tolerance for symbol in symbols)


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
    validation_engine: str,
    regime_risk: float,
    target_adjuster=None,
) -> tuple[dict | None, dict | None, list[dict]]:
    rows = []
    best = None
    best_any = None
    decision_mode = str(config.get("decision_mode", "balanced"))
    decision_score_threshold = mode_default_threshold(decision_mode, config.get("decision_score_threshold"))
    selection_split = "selection" if "selection" in masks else "validation"
    gate_split = "gate" if "gate" in masks else selection_split
    needs_selection_stress = (
        config.get("min_validation_stress_return_pct") is not None
        or abs(float(config.get("stress_selection_weight", 0.0))) > 0.0
    )
    needs_selection_segment = (
        config.get("min_validation_segment_return_pct") is not None
        or abs(float(config.get("segment_selection_penalty", 0.0))) > 0.0
    )
    for candidate in candidates:
        scores = alpha_scores[candidate.alpha_name]
        confidence = confidences.get((candidate.alpha_name, candidate.mode))
        policy = stop_policy(candidate)
        if validation_engine == "fast":
            selection_summary, selection_curve = fast_backtest_stopless(
                scores,
                matrices,
                masks,
                selection_split,
                policy,
                initial_cash=float(config["initial_cash"]),
                cost_bps=float(config["cost_bps"]),
                target_adjuster=target_adjuster,
            )
            selection_bars = None
        elif validation_engine == "full":
            if confidence is None:
                raise ValueError(f"missing confidence for {(candidate.alpha_name, candidate.mode)}")
            selection_summary, selection_bars = backtest_stop_aware(
                scores,
                matrices,
                masks,
                market_features,
                confidence,
                selection_split,
                policy,
                initial_cash=float(config["initial_cash"]),
                cost_bps=float(config["cost_bps"]),
                target_adjuster=target_adjuster,
            )
            selection_curve = None
        else:
            raise ValueError(f"unknown validation_engine={validation_engine}")
        raw_score = selection_score(
            selection_summary,
            float(config["drawdown_penalty"]),
            float(config["turnover_penalty"]),
        )
        preliminary_ok = True
        if config.get("min_validation_return_pct") is not None:
            preliminary_ok &= float(selection_summary["return_pct"]) >= float(config["min_validation_return_pct"])
        if config.get("max_validation_drawdown_pct") is not None:
            preliminary_ok &= float(selection_summary["max_drawdown_pct"]) >= -abs(
                float(config["max_validation_drawdown_pct"])
            )

        worst_segment = None
        segment_ok = preliminary_ok
        if preliminary_ok and needs_selection_segment:
            if validation_engine == "fast":
                worst_segment = fast_worst_segment_return(
                    selection_curve,
                    float(config["initial_cash"]),
                    int(config["segment_count"]),
                )
            else:
                selection_segments = segment_diagnostics(
                    selection_bars,
                    split=selection_split,
                    initial_cash=float(config["initial_cash"]),
                    segment_count=int(config["segment_count"]),
                )
                worst_segment = float(selection_segments["return_pct"].min()) if not selection_segments.empty else None
            if config.get("min_validation_segment_return_pct") is not None:
                segment_ok &= worst_segment is not None and worst_segment >= float(
                    config["min_validation_segment_return_pct"]
                )

        selection_stress_summary = {"return_pct": np.nan, "max_drawdown_pct": np.nan}
        stress_ok = preliminary_ok and segment_ok
        if preliminary_ok and segment_ok and needs_selection_stress:
            if validation_engine == "fast":
                selection_stress_summary, _ = fast_backtest_stopless(
                    scores,
                    matrices,
                    masks,
                    selection_split,
                    policy,
                    initial_cash=float(config["initial_cash"]),
                    cost_bps=float(config["stress_cost_bps"]),
                    target_adjuster=target_adjuster,
                )
            else:
                if confidence is None:
                    raise ValueError(f"missing confidence for {(candidate.alpha_name, candidate.mode)}")
                selection_stress_summary, _ = backtest_stop_aware(
                    scores,
                    matrices,
                    masks,
                    market_features,
                    confidence,
                    selection_split,
                    policy,
                    initial_cash=float(config["initial_cash"]),
                    cost_bps=float(config["stress_cost_bps"]),
                    target_adjuster=target_adjuster,
                )
            if config.get("min_validation_stress_return_pct") is not None:
                stress_ok &= float(selection_stress_summary["return_pct"]) >= float(
                    config["min_validation_stress_return_pct"]
                )

        stress_component = (
            float(config.get("stress_selection_weight", 0.0)) * float(selection_stress_summary["return_pct"])
            if needs_selection_stress
            else 0.0
        )
        selection = (
            raw_score
            + float(config.get("segment_selection_penalty", 0.0))
            * float(worst_segment if worst_segment is not None else -999.0)
            + stress_component
        )
        selection_score_before_diplom = selection
        diplom_candidate_penalty = diplom_score_penalty(
            selection_summary,
            float(config.get("diplom_candidate_score_penalty_pct", 0.0) or 0.0),
        )
        selection -= diplom_candidate_penalty
        ok = bool(preliminary_ok and segment_ok and stress_ok)
        selection_allowed = selection_allowed_by_mode(ok, decision_mode)
        row = {
            **asdict(candidate),
            "selection_return_pct": selection_summary["return_pct"],
            "selection_max_drawdown_pct": selection_summary["max_drawdown_pct"],
            "selection_stress_return_pct": selection_stress_summary["return_pct"],
            "validation_return_pct": selection_summary["return_pct"],
            "validation_max_drawdown_pct": selection_summary["max_drawdown_pct"],
            "validation_stress_return_pct": selection_stress_summary["return_pct"],
            "worst_validation_segment_return_pct": worst_segment,
            "worst_selection_segment_return_pct": worst_segment,
            "raw_selection_score": raw_score,
            "selection_score_before_diplom": selection_score_before_diplom,
            "diplom_candidate_penalty_score_pct": diplom_candidate_penalty,
            "constraints_ok": ok,
            "selection_allowed": selection_allowed,
            "selection_score": selection if selection_allowed else -np.inf,
        }
        rows.append(row)
        raw_record = {
            **row,
            "raw_candidate_score": selection,
            "policy": policy,
            "_scores": scores,
            "_confidence": confidence,
            "_selection_summary": selection_summary,
            "_selection_stress_summary": selection_stress_summary,
        }
        if best_any is None or selection > best_any["raw_candidate_score"]:
            best_any = raw_record
        if selection_allowed and (best is None or selection > best["selection_score"]):
            best = raw_record

    if best is None:
        return None, best_any, rows

    scores = best["_scores"]
    confidence = best["_confidence"]
    if validation_engine == "fast":
        gate_summary, gate_curve = fast_backtest_stopless(
            scores,
            matrices,
            masks,
            gate_split,
            best["policy"],
            initial_cash=float(config["initial_cash"]),
            cost_bps=float(config["cost_bps"]),
            target_adjuster=target_adjuster,
        )
        gate_stress_summary, _ = fast_backtest_stopless(
            scores,
            matrices,
            masks,
            gate_split,
            best["policy"],
            initial_cash=float(config["initial_cash"]),
            cost_bps=float(config["stress_cost_bps"]),
            target_adjuster=target_adjuster,
        )
        gate_worst_segment = fast_worst_segment_return(
            gate_curve,
            float(config["initial_cash"]),
            int(config["segment_count"]),
        )
    else:
        if confidence is None:
            raise ValueError(f"missing confidence for {(best['alpha_name'], best['mode'])}")
        gate_summary, gate_bars = backtest_stop_aware(
            scores,
            matrices,
            masks,
            market_features,
            confidence,
            gate_split,
            best["policy"],
            initial_cash=float(config["initial_cash"]),
            cost_bps=float(config["cost_bps"]),
            target_adjuster=target_adjuster,
        )
        gate_stress_summary, _ = backtest_stop_aware(
            scores,
            matrices,
            masks,
            market_features,
            confidence,
            gate_split,
            best["policy"],
            initial_cash=float(config["initial_cash"]),
            cost_bps=float(config["stress_cost_bps"]),
            target_adjuster=target_adjuster,
        )
        gate_segments = segment_diagnostics(
            gate_bars,
            split=gate_split,
            initial_cash=float(config["initial_cash"]),
            segment_count=int(config["segment_count"]),
        )
        gate_worst_segment = float(gate_segments["return_pct"].min()) if not gate_segments.empty else None

    selection_periods = pd.DataFrame([{"return_pct": float(best["_selection_summary"]["return_pct"])}])
    gate_periods = pd.DataFrame([{"return_pct": float(gate_summary["return_pct"])}])
    calibration_stats = calibration_stats_from_history(
        [],
        min_windows=int(config.get("calibration_min_windows", 4)),
    )
    trade_model = build_trade_probability_model(
        selection_summary=best["_selection_summary"],
        selection_stress_summary=best["_selection_stress_summary"],
        selection_periods=selection_periods,
        gate_summary=gate_summary,
        gate_stress_summary=gate_stress_summary,
        gate_periods=gate_periods,
        regime_risk=regime_risk,
        edge_uncertainty_weight=float(config.get("edge_uncertainty_weight", 1.0)),
        edge_regime_risk_weight_pct=float(config.get("edge_regime_risk_weight_pct", 5.0)),
        decision_mode=decision_mode,
        calibration_stats=calibration_stats,
        calibration_bias_weight=float(config.get("calibration_bias_weight", 0.5)),
        neural_feature_weight=float(config.get("neural_feature_weight", 0.0)),
        neural_uncertainty_weight=float(config.get("neural_uncertainty_weight", 0.5)),
        neural_uncertainty_pct=0.0,
        uses_neural_feature=best["alpha_name"] == "neural_ensemble",
    )
    trade_model.update(
        {
            **empty_diplom_summary(),
            **{key: value for key, value in gate_summary.items() if key.startswith("diplom_")},
        }
    )
    apply_diplom_gate_penalty(
        trade_model,
        gate_summary,
        float(config.get("diplom_gate_penalty_pct", 0.0) or 0.0),
    )
    gate_pass, gate_reasons = gate_trade_passes(
        trade_model,
        gate_summary,
        gate_stress_summary,
        decision_mode=decision_mode,
        decision_score_threshold=decision_score_threshold,
        min_gate_return_pct=None
        if config.get("min_gate_return_pct") is None
        else float(config.get("min_gate_return_pct")),
        min_gate_stress_return_pct=None
        if config.get("min_gate_stress_return_pct", 0.0) is None
        else float(config.get("min_gate_stress_return_pct", 0.0)),
        min_gate_worst_month_return_pct=None
        if config.get("min_gate_worst_month_return_pct") is None
        else float(config.get("min_gate_worst_month_return_pct")),
        min_gate_month_win_rate=None
        if config.get("min_gate_month_win_rate", 0.5) is None
        else float(config.get("min_gate_month_win_rate", 0.5)),
        min_edge_stress_pct=None
        if config.get("min_edge_stress_pct", 0.0) is None
        else float(config.get("min_edge_stress_pct", 0.0)),
        max_regime_risk=None
        if config.get("max_regime_risk", 0.70) is None
        else float(config.get("max_regime_risk", 0.70)),
        regime_veto=bool(config.get("regime_veto", True)),
    )
    best.update(
        {
            "gate_pass": gate_pass,
            "gate_reject_reasons": gate_reasons,
            "gate_return_pct": gate_summary["return_pct"],
            "gate_max_drawdown_pct": gate_summary["max_drawdown_pct"],
            "gate_stress_return_pct": gate_stress_summary["return_pct"],
            "gate_stress_max_drawdown_pct": gate_stress_summary["max_drawdown_pct"],
            "worst_gate_segment_return_pct": gate_worst_segment,
            "validation_return_pct": gate_summary["return_pct"],
            "validation_max_drawdown_pct": gate_summary["max_drawdown_pct"],
            "validation_stress_return_pct": gate_stress_summary["return_pct"],
            "worst_validation_segment_return_pct": gate_worst_segment,
            **trade_model,
        }
    )
    for row in rows:
        same = all(row.get(key) == best.get(key) for key in ["alpha_name", "mode", "k", "gross", "rebalance_every"])
        if same:
            row.update(
                {
                    "gate_pass": gate_pass,
                    "gate_reject_reasons": gate_reasons,
                    "gate_return_pct": gate_summary["return_pct"],
                    "gate_stress_return_pct": gate_stress_summary["return_pct"],
                    "worst_gate_segment_return_pct": gate_worst_segment,
                    **trade_model,
                }
            )
            break

    clean_best = {key: value for key, value in best.items() if not key.startswith("_")}
    return (clean_best if gate_pass else None), clean_best, rows


def target_weights_from_policy(
    *,
    symbols: list[str],
    scores: np.ndarray,
    tradable: np.ndarray,
    prices: dict[str, float],
    latest_row: int,
    policy,
    diplom_adapter: DiplomRiskAdapter | None = None,
    decision_time: pd.Timestamp | None = None,
) -> tuple[dict[str, float], dict]:
    price_mask = np.array([symbol in prices for symbol in symbols], dtype=bool)
    tradable_for_policy = tradable.copy()
    tradable_for_policy[latest_row] &= price_mask
    target = select_target(scores, tradable_for_policy, latest_row, policy)
    if diplom_adapter is not None:
        target, risk_diag = diplom_adapter.apply_to_target(
            target,
            symbols=symbols,
            timestamp=decision_time if decision_time is not None else pd.Timestamp.utcnow(),
        )
    else:
        risk_diag = empty_diplom_summary()
    return {symbols[idx]: float(weight) for idx, weight in enumerate(target) if abs(float(weight)) > 1e-12}, risk_diag


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
    live_gate_reason: str,
    top_n: int,
    diplom_adapter: DiplomRiskAdapter | None = None,
    decision_time: pd.Timestamp | None = None,
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
    if selected is not None:
        reason = "validation_gate_passed"
    elif live_gate_reason:
        reason = f"live_gate_blocked:{live_gate_reason}"
    else:
        reason = "validation_gate_blocked"
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
        diplom_lookup = diplom_adapter.lookup(symbol, decision_time or latest_candle) if diplom_adapter is not None else None
        diplom_status = diplom_lookup.get("status") if diplom_lookup else None
        diplom_p_high = diplom_lookup.get("p_high") if diplom_lookup else np.nan
        diplom_penalty_pct = 0.0
        if diplom_lookup and diplom_status == "ok" and np.isfinite(float(diplom_p_high)):
            diplom_penalty_pct = float(diplom_adapter.config.risk_weight_pct) * float(diplom_p_high)
            if (
                diplom_adapter.config.long_veto_p_high is not None
                and float(diplom_p_high) >= float(diplom_adapter.config.long_veto_p_high)
            ):
                diplom_penalty_pct = 100.0
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
                "gate_pass": source.get("gate_pass"),
                "gate_reject_reasons": source.get("gate_reject_reasons"),
                "gate_return_pct": source.get("gate_return_pct"),
                "gate_stress_return_pct": source.get("gate_stress_return_pct"),
                "expected_return_pct": source.get("expected_return_pct"),
                "uncertainty_pct": source.get("uncertainty_pct"),
                "regime_risk": source.get("regime_risk"),
                "stress_edge_pct": source.get("stress_edge_pct"),
                "decision_score": source.get("decision_score"),
                "soft_risk_flags": source.get("soft_risk_flags"),
                "trade_probability": source.get("trade_probability"),
                "diplom_ticker": diplom_lookup.get("diplom_ticker") if diplom_lookup else None,
                "diplom_prediction_date": (
                    str(diplom_lookup.get("decision_date")) if diplom_lookup and diplom_lookup.get("decision_date") is not None else None
                ),
                "diplom_prediction_age_days": diplom_lookup.get("prediction_age_days") if diplom_lookup else None,
                "diplom_risk_available": bool(diplom_status == "ok") if diplom_lookup else False,
                "diplom_risk_class": diplom_lookup.get("risk_class") if diplom_lookup else None,
                "diplom_p_high": diplom_p_high,
                "diplom_risk_penalty_pct": diplom_penalty_pct,
                "diplom_risk_reason": diplom_lookup.get("reason") if diplom_lookup else "diplom_risk_disabled",
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Live paper runner for the frozen strict alpha protocol.")
    parser.add_argument("--config-json", type=Path, default=Path("configs/strict_nested_edge_policy_20260622.json"))
    parser.add_argument("--board", default="TQBR")
    parser.add_argument("--duration-minutes", type=float, default=30.0)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--fetch-days", type=int, default=8)
    parser.add_argument("--fetch-workers", type=int, default=12)
    parser.add_argument("--candidate-log-top-n", type=int, default=0)
    parser.add_argument("--validation-engine", choices=["fast", "full"], default="fast")
    parser.add_argument("--rebalance-on-target-change", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--target-change-tolerance", type=float, default=None)
    parser.add_argument("--max-session-loss-pct", type=float, default=None)
    parser.add_argument("--max-signal-age-minutes", type=float, default=None)
    parser.add_argument("--use-marketdata-signal-bar", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--marketdata-signal-min-coverage", type=float, default=None)
    parser.add_argument("--marketdata-signal-max-age-minutes", type=float, default=None)
    parser.add_argument("--close-on-exit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    config = load_protocol_config(args.config_json)
    diplom_config = config_from_mapping(config)
    diplom_adapter = DiplomRiskAdapter.from_config(diplom_config)
    if args.output_dir is None:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = Path("outputs") / f"live_paper_alpha_policy_{stamp}"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rebalance_on_target_change = (
        bool(args.rebalance_on_target_change)
        if args.rebalance_on_target_change is not None
        else bool(config.get("live_rebalance_on_target_change", False))
    )
    target_change_tolerance = (
        float(args.target_change_tolerance)
        if args.target_change_tolerance is not None
        else float(config.get("live_target_change_tolerance", 0.05))
    )
    max_session_loss_pct = (
        float(args.max_session_loss_pct)
        if args.max_session_loss_pct is not None
        else (
            None
            if config.get("live_max_session_loss_pct") is None
            else float(config.get("live_max_session_loss_pct"))
        )
    )
    signal_delay_bars = int(config.get("signal_delay_bars", 0))
    if signal_delay_bars < 0:
        raise ValueError("signal_delay_bars must be non-negative")
    max_signal_age_minutes = (
        float(args.max_signal_age_minutes)
        if args.max_signal_age_minutes is not None
        else (
            None
            if config.get("live_max_signal_age_minutes") is None
            else float(config.get("live_max_signal_age_minutes"))
        )
    )
    use_marketdata_signal_bar = (
        bool(args.use_marketdata_signal_bar)
        if args.use_marketdata_signal_bar is not None
        else bool(config.get("live_use_marketdata_signal_bar", False))
    )
    marketdata_signal_min_coverage = (
        float(args.marketdata_signal_min_coverage)
        if args.marketdata_signal_min_coverage is not None
        else float(config.get("live_marketdata_signal_min_coverage", 0.8))
    )
    marketdata_signal_max_age_minutes = (
        float(args.marketdata_signal_max_age_minutes)
        if args.marketdata_signal_max_age_minutes is not None
        else (
            None
            if config.get("live_marketdata_signal_max_age_minutes") is None
            else float(config.get("live_marketdata_signal_max_age_minutes"))
        )
    )

    _, symbols, base_open, base_close, base_volume = read_intraday_matrix(Path(config["dataset_dir"]), int(config["interval"]))
    session = requests.Session()
    session.trust_env = False
    modes = parse_modes(config["modes"])
    candidates = build_candidates(
        [
            f"{kind}_{lag}"
            for lag in parse_csv_ints(config["alpha_lags"])
            for kind in parse_csv_strings(config["alpha_kinds"])
        ],
        modes=modes,
        k_grid=parse_csv_ints(config["k_grid"]),
        gross_grid=parse_csv_floats(config["gross_grid"]),
        rebalance_grid=parse_csv_ints(config["rebalance_grid"]),
    )
    cash = float(config["initial_cash"])
    positions: dict[str, float] = {}
    total_cost = 0.0
    total_turnover = 0.0
    last_rebalance_row: int | None = None
    active_target_weights: dict[str, float] = {}
    session_stopped = False
    cached_decision_candle: pd.Timestamp | None = None
    cached_alpha_scores: dict[str, np.ndarray] | None = None
    cached_selected: dict | None = None
    cached_best_any: dict | None = None
    cached_search_rows: list[dict] | None = None
    live_rows = []
    candidate_rows = []
    start_time = time.time()
    end_time = start_time + args.duration_minutes * 60.0

    while True:
        poll_started = time.perf_counter()
        now = dt.datetime.now()
        from_date = (now.date() - dt.timedelta(days=args.fetch_days)).isoformat()
        till_date = now.date().isoformat()
        live, errors, fetch_stats = fetch_live_candles(
            symbols,
            board=args.board,
            interval=int(config["interval"]),
            from_date=from_date,
            till_date=till_date,
            workers=max(1, args.fetch_workers),
        )
        open_px, close_px, volume = append_live_to_matrix(base_open, base_close, base_volume, live)
        marketdata = fetch_marketdata(session, args.board, symbols)
        wall_time_dt = dt.datetime.now()
        signal_bar_stats = {
            "marketdata_signal_bar_used": False,
            "marketdata_signal_coverage": 0.0,
            "marketdata_signal_prices": 0,
            "marketdata_signal_timestamp": None,
        }
        if use_marketdata_signal_bar:
            open_px, close_px, volume, signal_bar_stats = append_marketdata_signal_bar(
                open_px,
                close_px,
                volume,
                marketdata,
                now=wall_time_dt,
                interval=int(config["interval"]),
                min_coverage=marketdata_signal_min_coverage,
                max_age_minutes=marketdata_signal_max_age_minutes,
            )
        matrices = build_matrices(open_px, close_px, volume, int(config["horizon"]))
        market_features = compute_market_features(close_px.index, list(close_px.columns), matrices["bar_return"])
        if signal_delay_bars > 0:
            market_features = delay_market_features(market_features, signal_delay_bars)

        latest_row = len(close_px.index) - 1
        latest_ts = pd.Timestamp(close_px.index[latest_row])
        diplom_target_adjuster = make_diplom_target_adjuster(diplom_adapter, close_px.index, symbols)
        signal_source = "marketdata_bar" if signal_bar_stats["marketdata_signal_bar_used"] else "candle"
        validation_end = latest_ts
        validation_start = validation_end - pd.Timedelta(days=int(config["validation_days"]))
        masks, nested_spans = split_nested_masks(
            close_px.index,
            validation_start=validation_start,
            validation_end=validation_end,
            test_start=None,
            test_end=None,
            gate_days=int(config.get("nested_gate_days", 7)),
        )
        regime_history = build_regime_history(
            close_px.index,
            matrices,
            market_features,
            end=nested_spans["selection_end"],
            block_days=int(config.get("regime_block_days", 7)),
        )
        gate_regime_profile = market_regime_profile(close_px.index, matrices, market_features, masks["gate"])
        regime = score_may_like_regime(
            gate_regime_profile,
            regime_history,
            bad_return_quantile=float(config.get("regime_bad_return_quantile", 0.25)),
            bad_drawdown_quantile=float(config.get("regime_bad_drawdown_quantile", 0.25)),
            min_history_windows=int(config.get("min_regime_history_windows", 4)),
        )

        decision_started = time.perf_counter()
        decision_cache_hit = (
            signal_source == "candle"
            and cached_decision_candle is not None
            and latest_ts == cached_decision_candle
            and cached_alpha_scores is not None
            and cached_search_rows is not None
        )
        if decision_cache_hit:
            alpha_scores = cached_alpha_scores
            selected = cached_selected
            best_any = cached_best_any
            search_rows = cached_search_rows
        else:
            alpha_scores = build_alpha_scores(
                close_px,
                lags=parse_csv_ints(config["alpha_lags"]),
                kinds=parse_csv_strings(config["alpha_kinds"]),
                normalize=config["alpha_normalize"],
            )
            if signal_delay_bars > 0:
                alpha_scores = {
                    alpha_name: delay_matrix(scores, signal_delay_bars, np.nan)
                    for alpha_name, scores in alpha_scores.items()
                }
            confidences = (
                {
                    (alpha_name, mode): score_confidence(scores, matrices["tradable"], mode)
                    for alpha_name, scores in alpha_scores.items()
                    for mode in modes
                }
                if args.validation_engine == "full"
                else {}
            )
            selected, best_any, search_rows = select_validation_candidate(
                alpha_scores=alpha_scores,
                confidences=confidences,
                matrices=matrices,
                masks=masks,
                market_features=market_features,
                candidates=candidates,
                config=config,
                validation_engine=args.validation_engine,
                regime_risk=float(regime["regime_risk"]),
                target_adjuster=diplom_target_adjuster,
            )
            if signal_source == "candle":
                cached_decision_candle = latest_ts
                cached_alpha_scores = alpha_scores
                cached_selected = selected
                cached_best_any = best_any
                cached_search_rows = search_rows
        decision_elapsed_seconds = time.perf_counter() - decision_started

        candle_prices = close_px.iloc[latest_row].to_dict()
        prices = build_live_prices(symbols, candle_prices, marketdata)
        if signal_source == "marketdata_bar":
            fresh_symbols = set(str(signal_bar_stats.get("marketdata_signal_symbols") or "").split(","))
            fresh_symbols.discard("")
            prices = {symbol: price for symbol, price in prices.items() if symbol in fresh_symbols}
        signal_start_age_minutes = (pd.Timestamp(wall_time_dt) - latest_ts).total_seconds() / 60.0
        if signal_source == "marketdata_bar":
            signal_close_ts = pd.Timestamp(wall_time_dt)
            signal_age_minutes = 0.0
        else:
            signal_close_ts = latest_ts + pd.Timedelta(minutes=int(config["interval"]))
            signal_age_minutes = (pd.Timestamp(wall_time_dt) - signal_close_ts).total_seconds() / 60.0
            signal_age_minutes = max(0.0, signal_age_minutes)
        stale_signal_hit = (
            max_signal_age_minutes is not None
            and signal_age_minutes > float(max_signal_age_minutes)
        )

        target_weights = {}
        raw_selected = selected or best_any
        live_score_uses_marketdata = False
        live_gate_reasons = []
        if stale_signal_hit:
            live_gate_reasons.append("stale_signal")
        if session_stopped:
            live_gate_reasons.append("session_loss_stopped")
        live_gate_reason = ",".join(live_gate_reasons)
        if live_gate_reasons:
            selected = None
        score_source = selected or best_any
        selected_scores_for_execution = None
        target_risk_diag = empty_diplom_summary("no_selected_policy")
        if selected is not None:
            selected_policy = selected["policy"]
            selected_scores = alpha_scores[selected["alpha_name"]]
            if signal_source == "marketdata_bar" and signal_delay_bars == 0 and selected["alpha_name"] != "neural_ensemble":
                selected_scores = selected_scores.copy()
                selected_scores[latest_row] = current_alpha_score_vector(
                    close_px,
                    selected["alpha_name"],
                    config["alpha_normalize"],
                )
                live_score_uses_marketdata = True
            selected_scores_for_execution = selected_scores
            execution_tradable = matrices["tradable"] & np.isfinite(selected_scores_for_execution)
            if signal_source == "marketdata_bar":
                fresh_mask = np.array([symbol in prices for symbol in symbols], dtype=bool)
                execution_tradable = execution_tradable.copy()
                execution_tradable[latest_row] &= fresh_mask
            target_weights, target_risk_diag = target_weights_from_policy(
                symbols=symbols,
                scores=selected_scores_for_execution,
                tradable=execution_tradable,
                prices=prices,
                latest_row=latest_row,
                policy=selected_policy,
                diplom_adapter=diplom_adapter,
                decision_time=latest_ts,
            )
        pre_trade_equity, _ = mark_portfolio(cash, positions, prices)
        session_return_pct = (pre_trade_equity / float(config["initial_cash"]) - 1.0) * 100.0
        risk_reasons = []
        if stale_signal_hit:
            risk_reasons.append("stale_signal")
        if session_stopped:
            risk_reasons.append("session_loss_stopped")
        session_loss_stop_hit = (
            max_session_loss_pct is not None
            and not session_stopped
            and bool(positions)
            and session_return_pct <= -abs(max_session_loss_pct)
        )
        if session_loss_stop_hit:
            target_weights = {}
            session_stopped = True
            selected = None
            live_gate_reasons.append("session_loss_stop")
            live_gate_reason = ",".join(live_gate_reasons)
            risk_reasons.append("session_loss_stop")
        risk_action = ",".join(risk_reasons)
        if score_source is None:
            selected_scores_for_log = np.zeros_like(matrices["bar_return"], dtype=np.float32)
        elif (
            selected is not None
            and selected_scores_for_execution is not None
            and score_source["alpha_name"] == selected["alpha_name"]
        ):
            selected_scores_for_log = selected_scores_for_execution
        else:
            selected_scores_for_log = alpha_scores[score_source["alpha_name"]]
            if signal_source == "marketdata_bar" and signal_delay_bars == 0 and score_source["alpha_name"] != "neural_ensemble":
                selected_scores_for_log = selected_scores_for_log.copy()
                selected_scores_for_log[latest_row] = current_alpha_score_vector(
                    close_px,
                    score_source["alpha_name"],
                    config["alpha_normalize"],
                )
                live_score_uses_marketdata = True
        log_tradable_latest = matrices["tradable"][latest_row] & np.isfinite(selected_scores_for_log[latest_row])
        if signal_source == "marketdata_bar":
            log_tradable_latest = log_tradable_latest & np.array([symbol in prices for symbol in symbols], dtype=bool)

        target_change_rebalance = (
            rebalance_on_target_change
            and bool(positions)
            and weights_changed(active_target_weights, target_weights, target_change_tolerance)
        )
        rebalance_due = (
            selected is None
            or last_rebalance_row is None
            or (latest_row - last_rebalance_row) >= int(selected.get("rebalance_every", 1))
            or (not positions and bool(target_weights))
            or target_change_rebalance
            or session_loss_stop_hit
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
            active_target_weights = dict(target_weights)
            if session_loss_stop_hit:
                action = "close_session_loss"
            elif session_stopped:
                action = "session_loss_stopped_cash"
            elif stale_signal_hit:
                action = "close_stale_signal" if trades else "stale_signal_cash"
            elif target_change_rebalance:
                action = "rebalance_target_change" if trades else "target_change_no_trade"
            else:
                action = "rebalance" if trades else ("cash" if not target_weights else "no_trade")

        equity, position_values = mark_portfolio(cash, positions, prices)
        wall_time = wall_time_dt.isoformat(timespec="seconds")
        target_symbols = ",".join(sorted(target_weights))
        position_symbols = ",".join(sorted(positions))
        candidate_rows.extend(
            build_candidate_log_rows(
                wall_time=wall_time,
                latest_candle=str(latest_ts),
                symbols=symbols,
                scores=selected_scores_for_log[latest_row],
                tradable=log_tradable_latest,
                prices=prices,
                candle_prices=candle_prices,
                marketdata=marketdata,
                target_weights=target_weights,
                selected=selected,
                best_any=best_any,
                live_gate_reason=live_gate_reason if raw_selected is not None else "",
                top_n=args.candidate_log_top_n,
                diplom_adapter=diplom_adapter,
                decision_time=latest_ts,
            )
        )
        live_rows.append(
            {
                "wall_time": wall_time,
                "latest_candle": str(latest_ts),
                "validation_start": str(validation_start),
                "validation_end": str(validation_end),
                "selection_start": str(nested_spans["selection_start"]),
                "selection_end": str(nested_spans["selection_end"]),
                "gate_start": str(nested_spans["gate_start"]),
                "gate_end": str(nested_spans["gate_end"]),
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
                "raw_selected_alpha": raw_selected.get("alpha_name") if raw_selected else None,
                "raw_selected_mode": raw_selected.get("mode") if raw_selected else None,
                "raw_selected_k": raw_selected.get("k") if raw_selected else None,
                "raw_selected_gross": raw_selected.get("gross") if raw_selected else None,
                "raw_selected_rebalance_every": raw_selected.get("rebalance_every") if raw_selected else None,
                "raw_validation_return_pct": raw_selected.get("validation_return_pct") if raw_selected else None,
                "raw_worst_validation_segment_return_pct": (
                    raw_selected.get("worst_validation_segment_return_pct") if raw_selected else None
                ),
                "raw_gate_pass": raw_selected.get("gate_pass") if raw_selected else None,
                "raw_gate_reject_reasons": raw_selected.get("gate_reject_reasons") if raw_selected else None,
                "raw_expected_return_pct": raw_selected.get("expected_return_pct") if raw_selected else None,
                "raw_uncertainty_pct": raw_selected.get("uncertainty_pct") if raw_selected else None,
                "raw_regime_risk": raw_selected.get("regime_risk") if raw_selected else None,
                "raw_stress_edge_pct": raw_selected.get("stress_edge_pct") if raw_selected else None,
                "raw_decision_score": raw_selected.get("decision_score") if raw_selected else None,
                "raw_trade_probability": raw_selected.get("trade_probability") if raw_selected else None,
                "validation_return_pct": selected.get("validation_return_pct") if selected else None,
                "worst_validation_segment_return_pct": (
                    selected.get("worst_validation_segment_return_pct") if selected else None
                ),
                "gate_pass": selected.get("gate_pass") if selected else False,
                "gate_reject_reasons": selected.get("gate_reject_reasons") if selected else (
                    raw_selected.get("gate_reject_reasons") if raw_selected else None
                ),
                "expected_return_pct": selected.get("expected_return_pct") if selected else None,
                "uncertainty_pct": selected.get("uncertainty_pct") if selected else None,
                "regime_risk": regime.get("regime_risk"),
                "stress_edge_pct": selected.get("stress_edge_pct") if selected else None,
                "decision_score": selected.get("decision_score") if selected else None,
                "soft_risk_flags": selected.get("soft_risk_flags") if selected else (
                    raw_selected.get("soft_risk_flags") if raw_selected else None
                ),
                "trade_probability": selected.get("trade_probability") if selected else None,
                **target_risk_diag,
                "regime_reason": regime.get("regime_reason"),
                "regime_history_windows": regime.get("regime_history_windows"),
                "regime_bad_windows": regime.get("regime_bad_windows"),
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
                "validation_engine": args.validation_engine,
                "rebalance_on_target_change": rebalance_on_target_change,
                "target_change_rebalance": target_change_rebalance,
                "target_change_tolerance": target_change_tolerance,
                "max_session_loss_pct": max_session_loss_pct,
                "signal_delay_bars": signal_delay_bars,
                "max_signal_age_minutes": max_signal_age_minutes,
                "signal_source": signal_source,
                "live_score_uses_marketdata": live_score_uses_marketdata,
                "signal_start_age_minutes": signal_start_age_minutes,
                "signal_age_minutes": signal_age_minutes,
                "signal_close": str(signal_close_ts),
                "use_marketdata_signal_bar": use_marketdata_signal_bar,
                "marketdata_signal_min_coverage": marketdata_signal_min_coverage,
                "marketdata_signal_max_age_minutes": marketdata_signal_max_age_minutes,
                **signal_bar_stats,
                "stale_signal_hit": stale_signal_hit,
                "session_stopped": session_stopped,
                "live_gate_reason": live_gate_reason,
                "session_loss_stop_hit": session_loss_stop_hit,
                "risk_action": risk_action,
                "pre_trade_equity": pre_trade_equity,
                "pre_trade_return_pct": session_return_pct,
                "decision_cache_hit": decision_cache_hit,
                "decision_elapsed_seconds": decision_elapsed_seconds,
                "poll_elapsed_seconds": time.perf_counter() - poll_started,
                **fetch_stats,
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
            "selection_start": str(nested_spans["selection_start"]),
            "selection_end": str(nested_spans["selection_end"]),
            "gate_start": str(nested_spans["gate_start"]),
            "gate_end": str(nested_spans["gate_end"]),
            "log_rows": len(live_rows),
            "candidate_log_rows": len(candidate_rows),
            "output_dir": str(args.output_dir),
            "selected": {key: value for key, value in (selected or {}).items() if key != "policy"},
            "raw_selected": {key: value for key, value in (raw_selected or {}).items() if key != "policy"},
            "best_any": {key: value for key, value in (best_any or {}).items() if key != "policy"},
            "diplom_risk_overlay": {
                "enabled": bool(diplom_config.enabled),
                "predictions_path": str(diplom_config.predictions_path) if diplom_config.predictions_path else None,
                "risk_weight_pct": diplom_config.risk_weight_pct,
                "long_veto_p_high": diplom_config.long_veto_p_high,
                "long_p_high_cap": diplom_config.long_p_high_cap,
                "short_bonus_weight_pct": diplom_config.short_bonus_weight_pct,
                "candidate_score_penalty_pct": diplom_config.candidate_score_penalty_pct,
                "gate_penalty_pct": diplom_config.gate_penalty_pct,
                "stale_days": diplom_config.stale_days,
                "missing_policy": diplom_config.missing_policy,
                "enable_yndx_ydex_mapping": diplom_config.enable_yndx_ydex_mapping,
                "yndx_ydex_effective_date": diplom_config.yndx_ydex_effective_date,
                "ignored_forbidden_columns": diplom_adapter.ignored_forbidden_columns if diplom_adapter else [],
                "latest_target_risk": target_risk_diag,
            },
            "constraints_passed_count": int(sum(bool(row["constraints_ok"]) for row in search_rows)),
            "validation_engine": args.validation_engine,
            "rebalance_on_target_change": rebalance_on_target_change,
            "target_change_tolerance": target_change_tolerance,
            "max_session_loss_pct": max_session_loss_pct,
            "signal_delay_bars": signal_delay_bars,
            "max_signal_age_minutes": max_signal_age_minutes,
            "signal_source": signal_source,
            "live_score_uses_marketdata": live_score_uses_marketdata,
            "signal_start_age_minutes": signal_start_age_minutes,
            "signal_age_minutes": signal_age_minutes,
            "signal_close": str(signal_close_ts),
            "use_marketdata_signal_bar": use_marketdata_signal_bar,
            "marketdata_signal_min_coverage": marketdata_signal_min_coverage,
            "marketdata_signal_max_age_minutes": marketdata_signal_max_age_minutes,
            **signal_bar_stats,
            "stale_signal_hit": stale_signal_hit,
            "session_stopped": session_stopped,
            "live_gate_reason": live_gate_reason,
            "regime": regime,
            "gate_regime_profile": gate_regime_profile,
            "session_loss_stop_hit": session_loss_stop_hit,
            "risk_action": risk_action,
            "pre_trade_equity": pre_trade_equity,
            "pre_trade_return_pct": session_return_pct,
            "decision_cache_hit": decision_cache_hit,
            "decision_elapsed_seconds": decision_elapsed_seconds,
            "poll_elapsed_seconds": time.perf_counter() - poll_started,
            **fetch_stats,
        }
        (args.output_dir / "summary.json").write_text(
            json.dumps(json_safe(summary), indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
        print(
            f"{wall_time} candle={latest_ts} equity={equity:.2f} pnl={summary['pnl']:.2f} "
            f"action={action} selected={summary['selected'].get('alpha_name')} targets={target_symbols or 'cash'} "
            f"passed={summary['constraints_passed_count']} signal_age={signal_age_minutes:.1f}m "
            f"score={summary['selected'].get('decision_score')} edge={summary['selected'].get('stress_edge_pct')} "
            f"regime={regime.get('regime_risk')} "
            f"diplom={target_risk_diag.get('diplom_risk_reason')} "
            f"risk={risk_action or 'none'}",
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
                "raw_selected_alpha": None,
                "raw_selected_mode": None,
                "raw_selected_k": None,
                "raw_selected_gross": None,
                "raw_selected_rebalance_every": None,
                "raw_validation_return_pct": None,
                "raw_worst_validation_segment_return_pct": None,
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
                "validation_engine": args.validation_engine,
                "rebalance_on_target_change": rebalance_on_target_change,
                "target_change_rebalance": False,
                "target_change_tolerance": target_change_tolerance,
                "max_session_loss_pct": max_session_loss_pct,
                "signal_delay_bars": signal_delay_bars,
                "max_signal_age_minutes": max_signal_age_minutes,
                "signal_source": signal_source,
                "live_score_uses_marketdata": live_score_uses_marketdata,
                "signal_start_age_minutes": signal_start_age_minutes,
                "signal_age_minutes": signal_age_minutes,
                "signal_close": str(signal_close_ts),
                "use_marketdata_signal_bar": use_marketdata_signal_bar,
                "marketdata_signal_min_coverage": marketdata_signal_min_coverage,
                "marketdata_signal_max_age_minutes": marketdata_signal_max_age_minutes,
                **signal_bar_stats,
                "stale_signal_hit": stale_signal_hit,
                "session_stopped": session_stopped,
                "live_gate_reason": live_gate_reason,
                "session_loss_stop_hit": session_loss_stop_hit,
                "risk_action": "final_close",
                "pre_trade_equity": pre_trade_equity,
                "pre_trade_return_pct": session_return_pct,
                "decision_cache_hit": decision_cache_hit,
                "decision_elapsed_seconds": decision_elapsed_seconds,
                "poll_elapsed_seconds": None,
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
