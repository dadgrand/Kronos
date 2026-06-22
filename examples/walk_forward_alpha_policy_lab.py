import argparse
import datetime as dt
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from diplom_risk_adapter import DiplomRiskAdapter, config_from_mapping, empty_diplom_summary, summarize_diplom_diagnostics
from filtered_neural_policy_lab import compute_market_features
from neural_policy_lab import build_matrices, read_intraday_matrix
from stop_aware_filtered_policy_lab import (
    StopAwarePolicy,
    backtest_stop_aware,
    parse_floats,
    parse_ints,
    score_confidence,
    segment_diagnostics,
    selection_score,
    select_target,
)
from walk_forward_stop_aware_policy_lab import make_windows, parse_modes, summarize_compounded


@dataclass(frozen=True)
class AlphaCandidate:
    alpha_name: str
    mode: str
    k: int
    gross: float
    rebalance_every: int


CONFIG_FIELDS = [
    "dataset_dir",
    "interval",
    "horizon",
    "start_date",
    "end_date",
    "validation_days",
    "test_days",
    "step_days",
    "max_windows",
    "alpha_lags",
    "alpha_kinds",
    "alpha_normalize",
    "modes",
    "k_grid",
    "gross_grid",
    "rebalance_grid",
    "signal_delay_bars",
    "engine",
    "initial_cash",
    "cost_bps",
    "stress_cost_bps",
    "drawdown_penalty",
    "turnover_penalty",
    "segment_selection_penalty",
    "stress_selection_weight",
    "min_validation_return_pct",
    "min_validation_stress_return_pct",
    "min_validation_segment_return_pct",
    "max_validation_drawdown_pct",
    "decision_mode",
    "decision_score_threshold",
    "calibration_min_windows",
    "calibration_bias_weight",
    "nested_gate_days",
    "min_gate_return_pct",
    "min_gate_stress_return_pct",
    "min_gate_worst_month_return_pct",
    "min_gate_month_win_rate",
    "min_edge_stress_pct",
    "edge_uncertainty_weight",
    "edge_regime_risk_weight_pct",
    "max_regime_risk",
    "regime_veto",
    "regime_block_days",
    "regime_bad_return_quantile",
    "regime_bad_drawdown_quantile",
    "min_regime_history_windows",
    "neural_scores_paths",
    "neural_feature_weight",
    "neural_uncertainty_weight",
    "diplom_risk_enabled",
    "diplom_risk_predictions_path",
    "diplom_risk_weight_pct",
    "diplom_long_veto_p_high",
    "diplom_short_bonus_weight_pct",
    "diplom_stale_days",
    "diplom_missing_policy",
    "diplom_enable_yndx_ydex_mapping",
    "diplom_yndx_ydex_effective_date",
    "segment_count",
    "target_return_pct",
]


def args_to_config(args: argparse.Namespace) -> dict:
    config = {}
    for field in CONFIG_FIELDS:
        value = getattr(args, field)
        config[field] = str(value) if isinstance(value, Path) else value
    return {
        "protocol_name": "strict_walk_forward_alpha_policy_v1",
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "config": config,
        "notes": (
            "Frozen research protocol. Command-line values explicitly different "
            "from parser defaults may override matching config fields for fresh "
            "forward runs; otherwise this config supplies the protocol."
        ),
    }


def apply_config_defaults(parser: argparse.ArgumentParser, args: argparse.Namespace, config_path: Path | None) -> None:
    if config_path is None:
        return
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    config = payload.get("config", payload)
    for field, value in config.items():
        if field not in CONFIG_FIELDS or not hasattr(args, field):
            continue
        current = getattr(args, field)
        default = parser.get_default(field)
        if current != default:
            continue
        if field in {"dataset_dir", "diplom_risk_predictions_path"} and value is not None:
            value = Path(value)
        setattr(args, field, value)


def cs_zscore(frame: pd.DataFrame) -> pd.DataFrame:
    mean = frame.mean(axis=1)
    std = frame.std(axis=1).replace(0.0, np.nan)
    return frame.sub(mean, axis=0).div(std, axis=0).replace([np.inf, -np.inf], 0.0).fillna(0.0)


def build_alpha_scores(
    close_px: pd.DataFrame,
    *,
    lags: list[int],
    kinds: list[str],
    normalize: str,
) -> dict[str, np.ndarray]:
    close_ff = close_px.ffill(limit=200)
    ret = close_ff.pct_change(fill_method=None)
    vol = ret.shift(1).rolling(48, min_periods=12).std().replace(0.0, np.nan)
    scores = {}
    for lag in lags:
        mom = close_ff.shift(1) / close_ff.shift(1 + lag) - 1.0
        raw_by_kind = {
            "mom": mom,
            "rev": -mom,
            "voladj_mom": mom / vol,
            "voladj_rev": -mom / vol,
        }
        for kind in kinds:
            if kind not in raw_by_kind:
                raise ValueError(f"unknown alpha kind={kind}; allowed={sorted(raw_by_kind)}")
            frame = raw_by_kind[kind].replace([np.inf, -np.inf], 0.0).fillna(0.0)
            if normalize == "cs_zscore":
                frame = cs_zscore(frame)
            elif normalize == "none":
                pass
            else:
                raise ValueError(f"unknown normalize={normalize}")
            scores[f"{kind}_{lag}"] = frame.to_numpy(dtype=np.float32)
    return scores


def delay_matrix(values: np.ndarray, delay_bars: int, fill_value: float = np.nan) -> np.ndarray:
    if delay_bars <= 0:
        return values
    delayed = np.full(values.shape, fill_value, dtype=values.dtype)
    delayed[delay_bars:] = values[:-delay_bars]
    return delayed


def delay_vector(values: np.ndarray, delay_bars: int, fill_value: float = 0.0) -> np.ndarray:
    if delay_bars <= 0:
        return values
    delayed = np.full(values.shape, fill_value, dtype=values.dtype)
    delayed[delay_bars:] = values[:-delay_bars]
    return delayed


def delay_market_features(market_features: dict[str, np.ndarray], delay_bars: int) -> dict[str, np.ndarray]:
    if delay_bars <= 0:
        return market_features
    return {key: delay_vector(value, delay_bars, 0.0) for key, value in market_features.items()}


def max_drawdown_pct(equity: np.ndarray) -> float:
    if equity.size == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    return float(((equity / peak) - 1.0).min() * 100.0)


def empty_summary(split: str, initial_cash: float) -> dict:
    return {
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


def summarize_stopless_path(
    *,
    split: str,
    initial_cash: float,
    gross_bar_returns: np.ndarray,
    turnover: np.ndarray,
    gross_exposure: np.ndarray,
    cost_bps: float,
) -> tuple[dict, np.ndarray, np.ndarray]:
    if gross_bar_returns.size == 0:
        return empty_summary(split, initial_cash), np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    cost_rate = cost_bps / 10000.0
    net_bar_returns = gross_bar_returns - turnover * cost_rate
    equity_curve = initial_cash * np.cumprod(np.maximum(0.0, 1.0 + net_bar_returns))
    summary = {
        "split": split,
        "initial_cash": initial_cash,
        "final_equity": float(equity_curve[-1]),
        "pnl": float(equity_curve[-1] - initial_cash),
        "return_pct": float((equity_curve[-1] / initial_cash - 1.0) * 100.0),
        "max_drawdown_pct": max_drawdown_pct(equity_curve),
        "sharpe_like": float(
            net_bar_returns.mean() / (net_bar_returns.std(ddof=0) + 1e-12) * np.sqrt(252 * 50)
        ),
        "bars": int(gross_bar_returns.size),
        "active_rate": float((gross_exposure > 0.0).mean()),
        "avg_turnover_per_bar": float(turnover.mean()),
        "total_turnover": float(turnover.sum()),
        "mean_gross_exposure": float(gross_exposure.mean()),
        "stop_exits": 0,
        "take_profit_exits": 0,
        "filter_exits": 0,
    }
    return summary, equity_curve.astype(np.float64, copy=False), net_bar_returns.astype(np.float64, copy=False)


def build_stopless_bars(
    rows: np.ndarray,
    equity_curve: np.ndarray,
    net_bar_returns: np.ndarray,
    gross_bar_returns: np.ndarray,
    turnover: np.ndarray,
    gross_exposure: np.ndarray,
) -> pd.DataFrame:
    if rows.size == 0:
        return pd.DataFrame()
    actions = np.where(turnover > 0.0, "rebalance", "hold")
    return pd.DataFrame(
        {
            "row": rows.astype(np.int64, copy=False),
            "equity": equity_curve,
            "bar_return": net_bar_returns,
            "gross_bar_return": gross_bar_returns,
            "turnover": turnover,
            "gross_exposure": gross_exposure,
            "ending_gross_exposure": gross_exposure,
            "action": actions,
            "confidence": np.zeros(rows.size, dtype=np.float64),
            "filter_fail_count": np.zeros(rows.size, dtype=np.int32),
            "market_mom_24": np.zeros(rows.size, dtype=np.float64),
            "market_mom_96": np.zeros(rows.size, dtype=np.float64),
        }
    )


def worst_segment_return_pct_from_equity(
    equity_curve: np.ndarray,
    *,
    initial_cash: float,
    segment_count: int,
) -> float | None:
    if equity_curve.size == 0 or segment_count <= 0:
        return None
    worst = None
    previous_equity = float(initial_cash)
    for index_part in np.array_split(np.arange(equity_curve.size), segment_count):
        if index_part.size == 0:
            continue
        end_equity = float(equity_curve[index_part[-1]])
        if previous_equity == 0.0:
            segment_return = 0.0 if end_equity == 0.0 else np.inf
        else:
            segment_return = (end_equity / previous_equity - 1.0) * 100.0
        worst = segment_return if worst is None else min(worst, segment_return)
        previous_equity = end_equity
    return None if worst is None else float(worst)


def make_diplom_target_adjuster(
    diplom_adapter: DiplomRiskAdapter | None,
    index: pd.DatetimeIndex,
    symbols: list[str],
):
    if diplom_adapter is None:
        return None

    def adjust(target: np.ndarray, row: int) -> tuple[np.ndarray, dict]:
        return diplom_adapter.apply_to_target(target, symbols=symbols, timestamp=index[int(row)])

    return adjust


def fast_stopless_path(
    scores: np.ndarray,
    bar_return: np.ndarray,
    tradable: np.ndarray,
    rows: np.ndarray,
    policy: StopAwarePolicy,
    *,
    target_adjuster=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    gross_bar_returns = np.zeros(rows.size, dtype=np.float64)
    turnover = np.zeros(rows.size, dtype=np.float64)
    gross_exposure = np.zeros(rows.size, dtype=np.float64)
    weights = np.zeros(scores.shape[1], dtype=np.float32)
    if rows.size == 0:
        return gross_bar_returns, turnover, gross_exposure, empty_diplom_summary("no_position")

    rebalance_every = max(1, int(policy.rebalance_every))
    diplom_risk_diags = []
    for segment_start in range(0, rows.size, rebalance_every):
        row = int(rows[segment_start])
        target = select_target(scores, tradable, row, policy)
        if target_adjuster is not None:
            target, diplom_diag = target_adjuster(target, row)
            diplom_risk_diags.append(diplom_diag)
        turnover[segment_start] = float(np.abs(target - weights).sum())
        weights = target
        segment_end = min(rows.size, segment_start + rebalance_every)
        segment_rows = rows[segment_start:segment_end]
        gross_bar_returns[segment_start:segment_end] = bar_return[segment_rows] @ weights
        gross_exposure[segment_start:segment_end] = float(np.abs(weights).sum())

    return (
        gross_bar_returns,
        turnover,
        gross_exposure,
        summarize_diplom_diagnostics(diplom_risk_diags, disabled_reason="no_position")
        if target_adjuster is not None
        else empty_diplom_summary(),
    )


def fast_backtest_stopless_dual_cost(
    scores: np.ndarray,
    matrices: dict,
    rows: np.ndarray,
    split: str,
    policy: StopAwarePolicy,
    *,
    initial_cash: float,
    cost_bps: float,
    stress_cost_bps: float,
    return_bars: bool,
    target_adjuster=None,
) -> tuple[dict, dict, pd.DataFrame | None, np.ndarray]:
    bar_return = matrices.get("bar_return_clean", matrices["bar_return"])
    tradable = matrices["tradable"]
    gross_bar_returns, turnover, gross_exposure, diplom_summary = fast_stopless_path(
        scores,
        bar_return,
        tradable,
        rows,
        policy,
        target_adjuster=target_adjuster,
    )
    summary, equity_curve, net_bar_returns = summarize_stopless_path(
        split=split,
        initial_cash=initial_cash,
        gross_bar_returns=gross_bar_returns,
        turnover=turnover,
        gross_exposure=gross_exposure,
        cost_bps=cost_bps,
    )
    if target_adjuster is not None:
        summary.update(diplom_summary)
    stress_summary, _, _ = summarize_stopless_path(
        split=split,
        initial_cash=initial_cash,
        gross_bar_returns=gross_bar_returns,
        turnover=turnover,
        gross_exposure=gross_exposure,
        cost_bps=stress_cost_bps,
    )
    if target_adjuster is not None:
        stress_summary.update(diplom_summary)
    bars = (
        build_stopless_bars(rows, equity_curve, net_bar_returns, gross_bar_returns, turnover, gross_exposure)
        if return_bars
        else None
    )
    return summary, stress_summary, bars, equity_curve


def run_backtest_pair(
    scores: np.ndarray,
    matrices: dict,
    masks: dict[str, np.ndarray],
    rows_by_split: dict[str, np.ndarray],
    market_features: dict[str, np.ndarray],
    confidence: np.ndarray,
    split: str,
    policy: StopAwarePolicy,
    *,
    initial_cash: float,
    cost_bps: float,
    stress_cost_bps: float,
    segment_count: int,
    engine: str,
    target_adjuster=None,
) -> tuple[dict, dict, pd.DataFrame | None, float | None]:
    if engine == "fast":
        summary, stress_summary, bars, equity_curve = fast_backtest_stopless_dual_cost(
            scores,
            matrices,
            rows_by_split[split],
            split,
            policy,
            initial_cash=initial_cash,
            cost_bps=cost_bps,
            stress_cost_bps=stress_cost_bps,
            return_bars=False,
            target_adjuster=target_adjuster,
        )
        worst_segment = worst_segment_return_pct_from_equity(
            equity_curve,
            initial_cash=initial_cash,
            segment_count=segment_count,
        )
        return summary, stress_summary, bars, worst_segment
    if engine == "full":
        summary, bars = backtest_stop_aware(
            scores,
            matrices,
            masks,
            market_features,
            confidence,
            split,
            policy,
            initial_cash=initial_cash,
            cost_bps=cost_bps,
            target_adjuster=target_adjuster,
        )
        stress_summary, _ = backtest_stop_aware(
            scores,
            matrices,
            masks,
            market_features,
            confidence,
            split,
            policy,
            initial_cash=initial_cash,
            cost_bps=stress_cost_bps,
            target_adjuster=target_adjuster,
        )
        segments = segment_diagnostics(
            bars,
            split=split,
            initial_cash=initial_cash,
            segment_count=segment_count,
        )
        worst_segment = float(segments["return_pct"].min()) if not segments.empty else None
        return summary, stress_summary, bars, worst_segment
    raise ValueError(f"unknown engine={engine}")


def run_backtest_with_bars(
    scores: np.ndarray,
    matrices: dict,
    masks: dict[str, np.ndarray],
    rows_by_split: dict[str, np.ndarray],
    market_features: dict[str, np.ndarray],
    confidence: np.ndarray,
    split: str,
    policy: StopAwarePolicy,
    *,
    initial_cash: float,
    cost_bps: float,
    stress_cost_bps: float,
    engine: str,
    target_adjuster=None,
) -> tuple[dict, dict, pd.DataFrame]:
    if engine == "fast":
        summary, stress_summary, bars, _ = fast_backtest_stopless_dual_cost(
            scores,
            matrices,
            rows_by_split[split],
            split,
            policy,
            initial_cash=initial_cash,
            cost_bps=cost_bps,
            stress_cost_bps=stress_cost_bps,
            return_bars=True,
            target_adjuster=target_adjuster,
        )
        return summary, stress_summary, bars if bars is not None else pd.DataFrame()
    if engine == "full":
        summary, bars = backtest_stop_aware(
            scores,
            matrices,
            masks,
            market_features,
            confidence,
            split,
            policy,
            initial_cash=initial_cash,
            cost_bps=cost_bps,
            target_adjuster=target_adjuster,
        )
        stress_summary, _ = backtest_stop_aware(
            scores,
            matrices,
            masks,
            market_features,
            confidence,
            split,
            policy,
            initial_cash=initial_cash,
            cost_bps=stress_cost_bps,
            target_adjuster=target_adjuster,
        )
        return summary, stress_summary, bars
    raise ValueError(f"unknown engine={engine}")


def build_candidates(
    alpha_names: list[str],
    *,
    modes: list[str],
    k_grid: list[int],
    gross_grid: list[float],
    rebalance_grid: list[int],
) -> list[AlphaCandidate]:
    candidates = []
    for alpha_name in alpha_names:
        for mode in modes:
            for k in k_grid:
                for gross in gross_grid:
                    for rebalance_every in rebalance_grid:
                        candidates.append(AlphaCandidate(alpha_name, mode, k, gross, rebalance_every))
    return candidates


def stop_policy(candidate: AlphaCandidate) -> StopAwarePolicy:
    return StopAwarePolicy(
        mode=candidate.mode,
        k=candidate.k,
        gross=candidate.gross,
        rebalance_every=candidate.rebalance_every,
        confidence_min=0.0,
        market_mom_24_max=999.0,
        market_mom_96_max=999.0,
        stop_loss_pct=None,
        take_profit_pct=None,
        filter_fail_exit_bars=None,
    )


def constraints_pass(
    validation_summary: dict,
    validation_stress_summary: dict,
    worst_segment_return_pct: float | None,
    *,
    min_validation_return_pct: float | None,
    min_validation_stress_return_pct: float | None,
    min_validation_segment_return_pct: float | None,
    max_validation_drawdown_pct: float | None,
) -> bool:
    ok = True
    if min_validation_return_pct is not None:
        ok &= float(validation_summary["return_pct"]) >= min_validation_return_pct
    if min_validation_stress_return_pct is not None:
        ok &= float(validation_stress_summary["return_pct"]) >= min_validation_stress_return_pct
    if min_validation_segment_return_pct is not None:
        ok &= worst_segment_return_pct is not None and worst_segment_return_pct >= min_validation_segment_return_pct
    if max_validation_drawdown_pct is not None:
        ok &= float(validation_summary["max_drawdown_pct"]) >= -abs(max_validation_drawdown_pct)
    return bool(ok)


def parse_path_list(value: str | list[str] | None) -> list[Path]:
    if value is None:
        return []
    if isinstance(value, list):
        return [Path(item) for item in value if str(item).strip()]
    return [Path(item.strip()) for item in str(value).split(",") if item.strip()]


def mode_default_threshold(decision_mode: str, explicit_threshold: float | None) -> float:
    if explicit_threshold is not None:
        return float(explicit_threshold)
    if decision_mode == "strict":
        return 0.0
    if decision_mode == "balanced":
        return 0.0
    if decision_mode == "exploratory":
        return -5.0
    raise ValueError(f"unknown decision_mode={decision_mode}")


def selection_allowed_by_mode(selection_constraints_ok: bool, decision_mode: str) -> bool:
    if decision_mode == "strict":
        return bool(selection_constraints_ok)
    if decision_mode in {"balanced", "exploratory"}:
        return True
    raise ValueError(f"unknown decision_mode={decision_mode}")


def load_neural_ensemble_scores(paths: list[Path], expected_shape: tuple[int, int]) -> tuple[np.ndarray | None, np.ndarray | None, list[str]]:
    if not paths:
        return None, None, []
    parts = []
    used_paths = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"missing neural score file: {path}")
        scores = np.load(path)
        if scores.shape != expected_shape:
            raise ValueError(f"neural score shape {scores.shape} from {path} does not match {expected_shape}")
        parts.append(scores.astype(np.float32, copy=False))
        used_paths.append(str(path))
    stack = np.stack(parts, axis=0)
    return np.nanmean(stack, axis=0).astype(np.float32), np.nanstd(stack, axis=0).astype(np.float32), used_paths


def candidate_neural_uncertainty_pct(
    neural_std: np.ndarray | None,
    scores: np.ndarray,
    tradable: np.ndarray,
    rows: np.ndarray,
    policy: StopAwarePolicy,
) -> float:
    if neural_std is None or rows.size == 0:
        return 0.0
    samples = []
    rebalance_every = max(1, int(policy.rebalance_every))
    for local_idx in range(0, rows.size, rebalance_every):
        row = int(rows[local_idx])
        target = select_target(scores, tradable, row, policy)
        active = np.abs(target) > 1e-12
        if active.any():
            samples.append(float(np.average(neural_std[row, active], weights=np.abs(target[active]))))
    return float(np.nanmean(samples) * 100.0) if samples else 0.0


def split_nested_masks(
    index: pd.DatetimeIndex,
    *,
    validation_start: pd.Timestamp,
    validation_end: pd.Timestamp,
    test_start: pd.Timestamp | None,
    test_end: pd.Timestamp | None,
    gate_days: int,
) -> tuple[dict[str, np.ndarray], dict[str, pd.Timestamp]]:
    if gate_days <= 0:
        raise ValueError("--nested-gate-days must be positive; strict nested walk-forward needs a holdout gate")
    selection_end = validation_end - pd.Timedelta(days=gate_days)
    if selection_end <= validation_start:
        raise ValueError(
            f"--nested-gate-days={gate_days} leaves no selection slice inside "
            f"{validation_start}..{validation_end}"
        )
    masks = {
        "selection": np.asarray((index >= validation_start) & (index < selection_end)),
        "gate": np.asarray((index >= selection_end) & (index < validation_end)),
        "validation": np.asarray((index >= validation_start) & (index < validation_end)),
    }
    if test_start is not None and test_end is not None:
        masks["test"] = np.asarray((index >= test_start) & (index < test_end))
    spans = {
        "selection_start": validation_start,
        "selection_end": selection_end,
        "gate_start": selection_end,
        "gate_end": validation_end,
    }
    return masks, spans


def split_nested_masks_for_window(
    index: pd.DatetimeIndex,
    window,
    *,
    gate_days: int,
) -> tuple[dict[str, np.ndarray], dict[str, pd.Timestamp]]:
    return split_nested_masks(
        index,
        validation_start=window.validation_start,
        validation_end=window.validation_end,
        test_start=window.test_start,
        test_end=window.test_end,
        gate_days=gate_days,
    )


def calendar_period_diagnostics(
    index: pd.DatetimeIndex,
    bars: pd.DataFrame,
    *,
    split: str,
    initial_cash: float,
    frequency: str = "M",
) -> pd.DataFrame:
    if bars.empty:
        return pd.DataFrame()

    enriched = bars.copy()
    row_ids = enriched["row"].to_numpy(dtype=np.int64)
    enriched["timestamp"] = index[row_ids]
    enriched["period"] = pd.to_datetime(enriched["timestamp"]).dt.to_period(frequency).astype(str)

    rows = []
    previous_equity = float(initial_cash)
    for period, segment in enriched.groupby("period", sort=True):
        start_equity = previous_equity
        end_equity = float(segment["equity"].iloc[-1])
        equity_curve = np.r_[start_equity, segment["equity"].to_numpy(dtype=np.float64)]
        peak = np.maximum.accumulate(equity_curve)
        rows.append(
            {
                "split": split,
                "period": period,
                "start_timestamp": str(segment["timestamp"].iloc[0]),
                "end_timestamp": str(segment["timestamp"].iloc[-1]),
                "bars": int(len(segment)),
                "start_equity": start_equity,
                "end_equity": end_equity,
                "return_pct": float((end_equity / start_equity - 1.0) * 100.0) if start_equity else 0.0,
                "max_drawdown_pct": float(((equity_curve / peak) - 1.0).min() * 100.0),
                "active_rate": float((segment["gross_exposure"] > 0.0).mean()),
                "avg_turnover_per_bar": float(segment["turnover"].mean()),
                "mean_gross_exposure": float(segment["gross_exposure"].mean()),
                "rebalance_count": int((segment["action"] == "rebalance").sum()) if "action" in segment else 0,
                "stop_exits": int((segment["action"] == "close_stop_loss").sum()) if "action" in segment else 0,
                "take_profit_exits": int((segment["action"] == "close_take_profit").sum()) if "action" in segment else 0,
                "filter_exits": int((segment["action"] == "close_filter_fail").sum()) if "action" in segment else 0,
            }
        )
        previous_equity = end_equity
    return pd.DataFrame(rows)


def period_stability_summary(periods: pd.DataFrame) -> dict:
    if periods.empty:
        return {
            "period_count": 0,
            "period_win_rate": 0.0,
            "worst_period_return_pct": None,
            "mean_period_return_pct": None,
            "period_return_std_pct": None,
        }
    returns = periods["return_pct"].astype(float)
    return {
        "period_count": int(len(returns)),
        "period_win_rate": float((returns > 0.0).mean()),
        "worst_period_return_pct": float(returns.min()),
        "mean_period_return_pct": float(returns.mean()),
        "period_return_std_pct": float(returns.std(ddof=0)),
    }


def executed_window_month_results(window_results: pd.DataFrame) -> pd.DataFrame:
    if window_results.empty:
        return pd.DataFrame()
    frame = window_results.copy()
    frame["test_month"] = pd.to_datetime(frame["test_start"]).dt.to_period("M").astype(str)
    rows = []
    for month, group in frame.groupby("test_month", sort=True):
        returns = group["test_return_pct"].astype(float).to_numpy(dtype=np.float64) / 100.0
        compounded = float((np.prod(1.0 + returns) - 1.0) * 100.0) if returns.size else 0.0
        rows.append(
            {
                "test_month": month,
                "windows": int(len(group)),
                "trade_windows": int((group["selected"] == "alpha_policy").sum()),
                "return_pct": compounded,
                "mean_window_return_pct": float(group["test_return_pct"].astype(float).mean()),
                "worst_window_return_pct": float(group["test_return_pct"].astype(float).min()),
            }
        )
    return pd.DataFrame(rows)


def raw_vs_gated_month_results(window_results: pd.DataFrame) -> pd.DataFrame:
    if window_results.empty:
        return pd.DataFrame()
    frame = window_results.copy()
    frame["test_month"] = pd.to_datetime(frame["test_start"]).dt.to_period("M").astype(str)
    rows = []
    for month, group in frame.groupby("test_month", sort=True):
        gated_returns = group["test_return_pct"].astype(float).to_numpy(dtype=np.float64) / 100.0
        raw_col = "raw_test_return_pct" if "raw_test_return_pct" in group else "test_return_pct"
        raw_returns = pd.to_numeric(group[raw_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64) / 100.0
        rows.append(
            {
                "test_month": month,
                "windows": int(len(group)),
                "raw_candidate_windows": int(group.get("raw_alpha_name", pd.Series(index=group.index)).notna().sum()),
                "gated_trade_windows": int((group["selected"] == "alpha_policy").sum()),
                "raw_return_pct": float((np.prod(1.0 + raw_returns) - 1.0) * 100.0),
                "gated_return_pct": float((np.prod(1.0 + gated_returns) - 1.0) * 100.0),
                "raw_win_rate": float((raw_returns > 0.0).mean()) if raw_returns.size else 0.0,
                "gated_win_rate": float((gated_returns > 0.0).mean()) if gated_returns.size else 0.0,
                "raw_worst_window_return_pct": float(raw_returns.min() * 100.0) if raw_returns.size else 0.0,
                "gated_worst_window_return_pct": float(gated_returns.min() * 100.0) if gated_returns.size else 0.0,
            }
        )
    return pd.DataFrame(rows)


def veto_diagnostics(window_results: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    if window_results.empty or "raw_test_return_pct" not in window_results:
        empty = {
            "raw_candidate_windows": 0,
            "gated_trade_windows": 0,
            "vetoed_windows": 0,
            "avoided_loss_pct": 0.0,
            "missed_profit_pct": 0.0,
            "veto_precision_loss_rate": 0.0,
            "raw_losing_windows": 0,
            "raw_loss_recall": 0.0,
        }
        return empty, pd.DataFrame()

    raw = window_results[pd.to_numeric(window_results["raw_test_return_pct"], errors="coerce").notna()].copy()
    raw["raw_test_return_pct"] = pd.to_numeric(raw["raw_test_return_pct"], errors="coerce")
    raw["is_vetoed"] = raw["selected"] == "cash_gate_veto"
    raw["raw_was_loss"] = raw["raw_test_return_pct"] < 0.0
    raw["raw_was_profit"] = raw["raw_test_return_pct"] > 0.0
    vetoed = raw[raw["is_vetoed"]]
    losing = raw[raw["raw_was_loss"]]
    avoided_loss = float((-vetoed.loc[vetoed["raw_was_loss"], "raw_test_return_pct"]).sum())
    missed_profit = float(vetoed.loc[vetoed["raw_was_profit"], "raw_test_return_pct"].sum())

    reason_rows = []
    for reason_text, group in vetoed.groupby("gate_reject_reasons", dropna=False):
        reasons = [item for item in str(reason_text).split(",") if item]
        if not reasons:
            reasons = ["unknown"]
        for reason in reasons:
            reason_rows.append(
                {
                    "reject_reason": reason,
                    "windows": int(len(group)),
                    "raw_profitable_windows": int((group["raw_test_return_pct"] > 0.0).sum()),
                    "raw_losing_windows": int((group["raw_test_return_pct"] < 0.0).sum()),
                    "avoided_loss_pct": float((-group.loc[group["raw_test_return_pct"] < 0.0, "raw_test_return_pct"]).sum()),
                    "missed_profit_pct": float(group.loc[group["raw_test_return_pct"] > 0.0, "raw_test_return_pct"].sum()),
                    "mean_raw_test_return_pct": float(group["raw_test_return_pct"].mean()),
                }
            )
    by_reason = pd.DataFrame(reason_rows)
    if not by_reason.empty:
        weighted_sum = by_reason["mean_raw_test_return_pct"] * by_reason["windows"].clip(lower=1)
        by_reason = (
            by_reason.assign(_weighted_raw_sum=weighted_sum)
            .groupby("reject_reason", as_index=False)
            .agg(
                windows=("windows", "sum"),
                raw_profitable_windows=("raw_profitable_windows", "sum"),
                raw_losing_windows=("raw_losing_windows", "sum"),
                avoided_loss_pct=("avoided_loss_pct", "sum"),
                missed_profit_pct=("missed_profit_pct", "sum"),
                _weighted_raw_sum=("_weighted_raw_sum", "sum"),
            )
        )
        by_reason["mean_raw_test_return_pct"] = by_reason["_weighted_raw_sum"] / by_reason["windows"].clip(lower=1)
        by_reason = by_reason.drop(columns=["_weighted_raw_sum"]).sort_values("windows", ascending=False)
    summary = {
        "raw_candidate_windows": int(len(raw)),
        "gated_trade_windows": int((raw["selected"] == "alpha_policy").sum()),
        "vetoed_windows": int(len(vetoed)),
        "avoided_loss_pct": avoided_loss,
        "missed_profit_pct": missed_profit,
        "veto_precision_loss_rate": float((vetoed["raw_test_return_pct"] < 0.0).mean()) if len(vetoed) else 0.0,
        "raw_losing_windows": int(len(losing)),
        "raw_loss_recall": float(((losing["selected"] == "cash_gate_veto").mean())) if len(losing) else 0.0,
        "vetoed_profitable_windows": int((vetoed["raw_test_return_pct"] > 0.0).sum()),
        "vetoed_losing_windows": int((vetoed["raw_test_return_pct"] < 0.0).sum()),
    }
    return summary, by_reason


def market_regime_profile(
    index: pd.DatetimeIndex,
    matrices: dict,
    market_features: dict[str, np.ndarray],
    mask: np.ndarray,
) -> dict:
    rows = np.flatnonzero(mask)
    if rows.size == 0:
        return {
            "bars": 0,
            "market_return_pct": 0.0,
            "market_max_drawdown_pct": 0.0,
            "market_volatility_pct": 0.0,
            "market_dispersion_pct": 0.0,
            "market_mom_24_mean_pct": 0.0,
            "market_mom_96_mean_pct": 0.0,
        }
    bar_return = np.nan_to_num(matrices["bar_return"][rows], nan=0.0)
    market_returns = bar_return.mean(axis=1)
    equity = np.cumprod(np.maximum(0.0, 1.0 + market_returns))
    dispersion = np.nanstd(bar_return, axis=1)
    return {
        "bars": int(rows.size),
        "start_timestamp": str(index[rows[0]]),
        "end_timestamp": str(index[rows[-1]]),
        "market_return_pct": float((equity[-1] - 1.0) * 100.0),
        "market_max_drawdown_pct": max_drawdown_pct(equity),
        "market_volatility_pct": float(np.std(market_returns, ddof=0) * 100.0),
        "market_dispersion_pct": float(np.mean(dispersion) * 100.0),
        "market_mom_24_mean_pct": float(np.nanmean(market_features["market_mom_24"][rows]) * 100.0),
        "market_mom_96_mean_pct": float(np.nanmean(market_features["market_mom_96"][rows]) * 100.0),
    }


REGIME_FEATURE_COLUMNS = [
    "market_return_pct",
    "market_max_drawdown_pct",
    "market_volatility_pct",
    "market_dispersion_pct",
    "market_mom_24_mean_pct",
    "market_mom_96_mean_pct",
]


def build_regime_history(
    index: pd.DatetimeIndex,
    matrices: dict,
    market_features: dict[str, np.ndarray],
    *,
    end: pd.Timestamp,
    block_days: int,
) -> pd.DataFrame:
    if block_days <= 0:
        raise ValueError("--regime-block-days must be positive")
    rows = []
    start = pd.Timestamp(index.min())
    block = pd.Timedelta(days=block_days)
    current = start
    block_id = 1
    while current + block <= end:
        block_end = current + block
        mask = np.asarray((index >= current) & (index < block_end))
        profile = market_regime_profile(index, matrices, market_features, mask)
        if profile["bars"] > 0:
            rows.append(
                {
                    "regime_block_id": block_id,
                    "block_start": current.isoformat(),
                    "block_end": block_end.isoformat(),
                    **profile,
                }
            )
            block_id += 1
        current = block_end
    return pd.DataFrame(rows)


def score_may_like_regime(
    current_profile: dict,
    history: pd.DataFrame,
    *,
    bad_return_quantile: float,
    bad_drawdown_quantile: float,
    min_history_windows: int,
) -> dict:
    if history.empty or len(history) < min_history_windows:
        return {
            "regime_risk": 0.5,
            "regime_reason": "insufficient_prior_history",
            "regime_history_windows": int(len(history)),
            "regime_bad_windows": 0,
            "regime_similarity": None,
        }

    features = history[REGIME_FEATURE_COLUMNS].astype(float)
    return_cutoff = float(history["market_return_pct"].quantile(bad_return_quantile))
    drawdown_cutoff = float(history["market_max_drawdown_pct"].quantile(bad_drawdown_quantile))
    bad_mask = (history["market_return_pct"] <= return_cutoff) | (
        history["market_max_drawdown_pct"] <= drawdown_cutoff
    )
    bad_features = features[bad_mask]
    if bad_features.empty:
        return {
            "regime_risk": 0.0,
            "regime_reason": "no_prior_bad_regime",
            "regime_history_windows": int(len(history)),
            "regime_bad_windows": 0,
            "regime_similarity": 0.0,
        }

    mean = features.mean(axis=0).to_numpy(dtype=np.float64)
    std = np.clip(features.std(axis=0, ddof=0).to_numpy(dtype=np.float64), 1e-6, None)
    current = np.array([float(current_profile.get(key, 0.0)) for key in REGIME_FEATURE_COLUMNS], dtype=np.float64)
    current_z = (current - mean) / std
    bad_z = (bad_features.to_numpy(dtype=np.float64) - mean) / std
    distances = np.linalg.norm(bad_z - current_z, axis=1) / np.sqrt(len(REGIME_FEATURE_COLUMNS))
    similarity = float(np.exp(-0.5 * float(np.min(distances)) ** 2))
    return_gap = max(0.0, return_cutoff - float(current_profile["market_return_pct"]))
    drawdown_gap = max(0.0, drawdown_cutoff - float(current_profile["market_max_drawdown_pct"]))
    weakness = np.clip(
        0.5 * return_gap / (abs(return_cutoff) + 1.0)
        + 0.5 * drawdown_gap / (abs(drawdown_cutoff) + 1.0),
        0.0,
        1.0,
    )
    risk = float(np.clip(max(similarity, weakness), 0.0, 1.0))
    return {
        "regime_risk": risk,
        "regime_reason": "prior_bad_regime_similarity",
        "regime_history_windows": int(len(history)),
        "regime_bad_windows": int(bad_mask.sum()),
        "regime_similarity": similarity,
        "regime_bad_return_cutoff_pct": return_cutoff,
        "regime_bad_drawdown_cutoff_pct": drawdown_cutoff,
    }


def sigmoid(value: float) -> float:
    value = float(np.clip(value, -60.0, 60.0))
    return float(1.0 / (1.0 + np.exp(-value)))


def build_trade_probability_model(
    *,
    selection_summary: dict,
    selection_stress_summary: dict,
    selection_periods: pd.DataFrame,
    gate_summary: dict,
    gate_stress_summary: dict,
    gate_periods: pd.DataFrame,
    regime_risk: float,
    edge_uncertainty_weight: float,
    edge_regime_risk_weight_pct: float,
    decision_mode: str,
    calibration_stats: dict,
    calibration_bias_weight: float,
    neural_feature_weight: float,
    neural_uncertainty_weight: float,
    neural_uncertainty_pct: float,
    uses_neural_feature: bool,
) -> dict:
    selection_stability = period_stability_summary(selection_periods)
    gate_stability = period_stability_summary(gate_periods)
    samples = []
    if not selection_periods.empty:
        samples.extend(selection_periods["return_pct"].astype(float).tolist())
    else:
        samples.append(float(selection_summary["return_pct"]))
    if not gate_periods.empty:
        samples.extend(gate_periods["return_pct"].astype(float).tolist())
    else:
        samples.append(float(gate_summary["return_pct"]))

    sample_array = np.asarray(samples, dtype=np.float64)
    stress_gap = max(0.0, float(gate_summary["return_pct"]) - float(gate_stress_summary["return_pct"]))
    worst_sample = float(np.min(sample_array)) if sample_array.size else 0.0
    uncertainty_pct = float(
        np.std(sample_array, ddof=0)
        + 0.5 * stress_gap
        + 0.25 * max(0.0, -worst_sample)
        + 0.1 * abs(float(gate_summary["max_drawdown_pct"]))
        + neural_uncertainty_weight * max(0.0, neural_uncertainty_pct)
    )
    expected_return_pct = float(min(float(gate_summary["return_pct"]), float(gate_stress_summary["return_pct"])))
    calibration_bias_pct = float(calibration_stats.get("calibration_bias_pct", 0.0))
    calibration_count = int(calibration_stats.get("calibration_count", 0))
    calibrated_expected_return_pct = float(expected_return_pct + calibration_bias_weight * calibration_bias_pct)
    stress_edge_pct = float(
        calibrated_expected_return_pct
        - edge_uncertainty_weight * uncertainty_pct
        - edge_regime_risk_weight_pct * float(regime_risk)
    )
    neural_feature_bonus_pct = neural_feature_weight * max(0.0, calibrated_expected_return_pct) if uses_neural_feature else 0.0
    decision_score = float(stress_edge_pct + neural_feature_bonus_pct)
    probability = sigmoid(decision_score / max(1.0, uncertainty_pct))
    return {
        "expected_return_pct": expected_return_pct,
        "calibrated_expected_return_pct": calibrated_expected_return_pct,
        "uncertainty_pct": uncertainty_pct,
        "neural_uncertainty_pct": float(neural_uncertainty_pct),
        "uses_neural_feature": bool(uses_neural_feature),
        "neural_feature_bonus_pct": float(neural_feature_bonus_pct),
        "regime_risk": float(regime_risk),
        "stress_edge_pct": stress_edge_pct,
        "decision_score": decision_score,
        "trade_probability": probability,
        "decision_mode": decision_mode,
        "calibration_count": calibration_count,
        "calibration_bias_pct": calibration_bias_pct,
        "calibration_win_rate": calibration_stats.get("calibration_win_rate"),
        "calibration_mean_raw_test_return_pct": calibration_stats.get("calibration_mean_raw_test_return_pct"),
        "selection_period_count": selection_stability["period_count"],
        "selection_period_win_rate": selection_stability["period_win_rate"],
        "selection_worst_period_return_pct": selection_stability["worst_period_return_pct"],
        "gate_period_count": gate_stability["period_count"],
        "gate_period_win_rate": gate_stability["period_win_rate"],
        "gate_worst_period_return_pct": gate_stability["worst_period_return_pct"],
        "selection_stress_return_pct": float(selection_stress_summary["return_pct"]),
        "gate_stress_return_pct": float(gate_stress_summary["return_pct"]),
    }


def calibration_stats_from_history(history: list[dict], *, min_windows: int) -> dict:
    usable = [
        row
        for row in history
        if row.get("raw_test_return_pct") is not None
        and row.get("expected_return_pct") is not None
        and np.isfinite(float(row["raw_test_return_pct"]))
        and np.isfinite(float(row["expected_return_pct"]))
    ]
    if len(usable) < min_windows:
        return {
            "calibration_count": len(usable),
            "calibration_bias_pct": 0.0,
            "calibration_win_rate": None,
            "calibration_mean_raw_test_return_pct": None,
            "calibration_reason": "insufficient_prior_windows",
        }
    raw = np.asarray([float(row["raw_test_return_pct"]) for row in usable], dtype=np.float64)
    expected = np.asarray([float(row["expected_return_pct"]) for row in usable], dtype=np.float64)
    return {
        "calibration_count": int(len(usable)),
        "calibration_bias_pct": float(np.mean(raw - expected)),
        "calibration_win_rate": float((raw > 0.0).mean()),
        "calibration_mean_raw_test_return_pct": float(np.mean(raw)),
        "calibration_mean_expected_return_pct": float(np.mean(expected)),
        "calibration_reason": "prior_window_outcomes",
    }


def gate_trade_passes(
    model: dict,
    gate_summary: dict,
    gate_stress_summary: dict,
    *,
    decision_mode: str,
    decision_score_threshold: float,
    min_gate_return_pct: float | None,
    min_gate_stress_return_pct: float | None,
    min_gate_worst_month_return_pct: float | None,
    min_gate_month_win_rate: float | None,
    min_edge_stress_pct: float | None,
    max_regime_risk: float | None,
    regime_veto: bool,
) -> tuple[bool, str]:
    reasons = []
    soft_flags = []
    if min_gate_return_pct is not None and float(gate_summary["return_pct"]) < min_gate_return_pct:
        soft_flags.append("gate_return")
    if min_gate_stress_return_pct is not None and float(gate_stress_summary["return_pct"]) < min_gate_stress_return_pct:
        soft_flags.append("gate_stress")
    worst_gate = model.get("gate_worst_period_return_pct")
    if (
        min_gate_worst_month_return_pct is not None
        and (worst_gate is None or float(worst_gate) < min_gate_worst_month_return_pct)
    ):
        soft_flags.append("gate_month_worst")
    if (
        min_gate_month_win_rate is not None
        and float(model.get("gate_period_win_rate", 0.0)) < min_gate_month_win_rate
    ):
        soft_flags.append("gate_month_win_rate")
    if min_edge_stress_pct is not None and float(model["stress_edge_pct"]) < min_edge_stress_pct:
        soft_flags.append("stress_edge")
    if regime_veto and max_regime_risk is not None and float(model["regime_risk"]) > max_regime_risk:
        soft_flags.append("may_like_regime")

    if decision_mode == "strict":
        reasons.extend(soft_flags)
    elif decision_mode == "balanced":
        if float(model["decision_score"]) < decision_score_threshold:
            reasons.append("decision_score")
        if regime_veto and max_regime_risk is not None and float(model["regime_risk"]) > max(0.95, max_regime_risk):
            reasons.append("catastrophic_regime_risk")
    elif decision_mode == "exploratory":
        if float(model["decision_score"]) < decision_score_threshold:
            reasons.append("decision_score")
    else:
        raise ValueError(f"unknown decision_mode={decision_mode}")

    if soft_flags:
        model["soft_risk_flags"] = ",".join(soft_flags)
    return not reasons, ",".join(reasons)


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward alpha policy lab with cash fallback.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("datasets/moex_intraday_6f8f836cc811"))
    parser.add_argument("--interval", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--start-date", default="2026-01-01")
    parser.add_argument("--end-date", default="2026-06-20")
    parser.add_argument("--validation-days", type=int, default=30)
    parser.add_argument("--test-days", type=int, default=7)
    parser.add_argument("--step-days", type=int, default=7)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--alpha-lags", default="12,48,96")
    parser.add_argument("--alpha-kinds", default="mom,rev,voladj_mom,voladj_rev")
    parser.add_argument("--alpha-normalize", choices=["none", "cs_zscore"], default="none")
    parser.add_argument("--modes", default="short_only,long_only,long_short")
    parser.add_argument("--k-grid", default="1,3")
    parser.add_argument("--gross-grid", default="0.5,1.0,1.5")
    parser.add_argument("--rebalance-grid", default="24")
    parser.add_argument("--signal-delay-bars", type=int, default=0)
    parser.add_argument("--engine", choices=["full", "fast"], default="full")
    parser.add_argument("--initial-cash", type=float, default=10000.0)
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--stress-cost-bps", type=float, default=20.0)
    parser.add_argument("--drawdown-penalty", type=float, default=0.35)
    parser.add_argument("--turnover-penalty", type=float, default=2.0)
    parser.add_argument("--segment-selection-penalty", type=float, default=0.5)
    parser.add_argument("--stress-selection-weight", type=float, default=0.0)
    parser.add_argument("--min-validation-return-pct", type=float, default=5.0)
    parser.add_argument("--min-validation-stress-return-pct", type=float, default=None)
    parser.add_argument("--min-validation-segment-return-pct", type=float, default=0.0)
    parser.add_argument("--max-validation-drawdown-pct", type=float, default=8.0)
    parser.add_argument("--decision-mode", choices=["strict", "balanced", "exploratory"], default="balanced")
    parser.add_argument("--decision-score-threshold", type=float, default=None)
    parser.add_argument("--calibration-min-windows", type=int, default=4)
    parser.add_argument("--calibration-bias-weight", type=float, default=0.5)
    parser.add_argument("--nested-gate-days", type=int, default=7)
    parser.add_argument("--min-gate-return-pct", type=float, default=None)
    parser.add_argument("--min-gate-stress-return-pct", type=float, default=0.0)
    parser.add_argument("--min-gate-worst-month-return-pct", type=float, default=None)
    parser.add_argument("--min-gate-month-win-rate", type=float, default=0.5)
    parser.add_argument("--min-edge-stress-pct", type=float, default=0.0)
    parser.add_argument("--edge-uncertainty-weight", type=float, default=1.0)
    parser.add_argument("--edge-regime-risk-weight-pct", type=float, default=5.0)
    parser.add_argument("--max-regime-risk", type=float, default=0.70)
    parser.add_argument("--regime-veto", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--regime-block-days", type=int, default=7)
    parser.add_argument("--regime-bad-return-quantile", type=float, default=0.25)
    parser.add_argument("--regime-bad-drawdown-quantile", type=float, default=0.25)
    parser.add_argument("--min-regime-history-windows", type=int, default=4)
    parser.add_argument(
        "--neural-scores-paths",
        default=(
            "outputs/neural_policy_lab_h12_seed20260622_20260622_132851/scores.npy,"
            "outputs/neural_policy_lab_h12_seed20260623_20260622_133133/scores.npy,"
            "outputs/neural_policy_lab_h12_seed20260624_20260622_133615/scores.npy"
        ),
    )
    parser.add_argument("--neural-feature-weight", type=float, default=0.0)
    parser.add_argument("--neural-uncertainty-weight", type=float, default=0.5)
    parser.add_argument("--diplom-risk-enabled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--diplom-risk-predictions-path", type=Path, default=None)
    parser.add_argument("--diplom-risk-weight-pct", type=float, default=0.0)
    parser.add_argument("--diplom-long-veto-p-high", type=float, default=None)
    parser.add_argument("--diplom-short-bonus-weight-pct", type=float, default=0.0)
    parser.add_argument("--diplom-stale-days", type=int, default=45)
    parser.add_argument("--diplom-missing-policy", choices=["neutral", "cash"], default="neutral")
    parser.add_argument("--diplom-enable-yndx-ydex-mapping", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--diplom-yndx-ydex-effective-date", default="2024-07-24")
    parser.add_argument("--segment-count", type=int, default=5)
    parser.add_argument("--target-return-pct", type=float, default=10.0)
    parser.add_argument("--config-json", type=Path, default=None)
    parser.add_argument("--write-config-json", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    apply_config_defaults(parser, args, args.config_json)
    decision_score_threshold = mode_default_threshold(args.decision_mode, args.decision_score_threshold)

    if args.write_config_json is not None:
        args.write_config_json.parent.mkdir(parents=True, exist_ok=True)
        args.write_config_json.write_text(
            json.dumps(args_to_config(args), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    index, symbols, open_px, close_px, volume = read_intraday_matrix(args.dataset_dir, args.interval)
    matrices = build_matrices(open_px, close_px, volume, args.horizon)
    if args.engine == "fast":
        matrices["bar_return_clean"] = np.nan_to_num(matrices["bar_return"], nan=0.0)
    diplom_config = config_from_mapping(vars(args))
    diplom_adapter = DiplomRiskAdapter.from_config(diplom_config)
    diplom_target_adjuster = make_diplom_target_adjuster(diplom_adapter, index, symbols)
    market_features = compute_market_features(index, symbols, matrices["bar_return"])
    alpha_scores = build_alpha_scores(
        close_px,
        lags=parse_ints(args.alpha_lags),
        kinds=[item.strip() for item in args.alpha_kinds.split(",") if item.strip()],
        normalize=args.alpha_normalize,
    )
    neural_mean_scores, neural_std_scores, neural_used_paths = load_neural_ensemble_scores(
        parse_path_list(args.neural_scores_paths),
        matrices["bar_return"].shape,
    )
    if neural_mean_scores is not None:
        alpha_scores["neural_ensemble"] = neural_mean_scores
    if args.signal_delay_bars < 0:
        raise ValueError("--signal-delay-bars must be non-negative")
    if args.signal_delay_bars > 0:
        alpha_scores = {
            alpha_name: delay_matrix(scores, args.signal_delay_bars, np.nan)
            for alpha_name, scores in alpha_scores.items()
        }
        if neural_std_scores is not None:
            neural_std_scores = delay_matrix(neural_std_scores, args.signal_delay_bars, np.nan)
        market_features = delay_market_features(market_features, args.signal_delay_bars)
    modes = parse_modes(args.modes)
    if args.engine == "full":
        confidences = {
            (alpha_name, mode): score_confidence(scores, matrices["tradable"], mode)
            for alpha_name, scores in alpha_scores.items()
            for mode in modes
        }
    else:
        zero_confidence = np.zeros(matrices["bar_return"].shape[0], dtype=np.float32)
        confidences = {
            (alpha_name, mode): zero_confidence
            for alpha_name in alpha_scores
            for mode in modes
        }
    candidates = build_candidates(
        list(alpha_scores),
        modes=modes,
        k_grid=parse_ints(args.k_grid),
        gross_grid=parse_floats(args.gross_grid),
        rebalance_grid=parse_ints(args.rebalance_grid),
    )
    windows = make_windows(
        index,
        start_date=args.start_date,
        end_date=args.end_date,
        validation_days=args.validation_days,
        test_days=args.test_days,
        step_days=args.step_days,
        max_windows=args.max_windows,
    )
    if not windows:
        raise ValueError("No complete walk-forward windows fit the requested date range.")

    if args.output_dir is None:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = Path("outputs") / f"walk_forward_alpha_policy_{stamp}"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    search_rows = []
    window_rows = []
    selected_rows = []
    test_segments_all = []
    validation_segments_all = []
    selection_segments_all = []
    gate_segments_all = []
    selection_months_all = []
    gate_months_all = []
    test_months_all = []
    regime_rows = []
    calibration_history = []
    for window in windows:
        masks, spans = split_nested_masks_for_window(index, window, gate_days=args.nested_gate_days)
        rows_by_split = {split: np.flatnonzero(mask) for split, mask in masks.items()}
        regime_history = build_regime_history(
            index,
            matrices,
            market_features,
            end=spans["selection_end"],
            block_days=args.regime_block_days,
        )
        gate_regime_profile = market_regime_profile(index, matrices, market_features, masks["gate"])
        regime = score_may_like_regime(
            gate_regime_profile,
            regime_history,
            bad_return_quantile=args.regime_bad_return_quantile,
            bad_drawdown_quantile=args.regime_bad_drawdown_quantile,
            min_history_windows=args.min_regime_history_windows,
        )
        regime_rows.append(
            {
                "window_id": window.window_id,
                "selection_end": spans["selection_end"].isoformat(),
                "gate_start": spans["gate_start"].isoformat(),
                "gate_end": spans["gate_end"].isoformat(),
                **gate_regime_profile,
                **regime,
            }
        )

        best = None
        for candidate in candidates:
            scores = alpha_scores[candidate.alpha_name]
            confidence = confidences[(candidate.alpha_name, candidate.mode)]
            policy = stop_policy(candidate)
            selection_summary, selection_stress_summary, _, worst_segment = run_backtest_pair(
                scores,
                matrices,
                masks,
                rows_by_split,
                market_features,
                confidence,
                "selection",
                policy,
                initial_cash=args.initial_cash,
                cost_bps=args.cost_bps,
                stress_cost_bps=args.stress_cost_bps,
                segment_count=args.segment_count,
                engine=args.engine,
                target_adjuster=diplom_target_adjuster,
            )
            raw_score = selection_score(selection_summary, args.drawdown_penalty, args.turnover_penalty)
            candidate_score = (
                raw_score
                + args.segment_selection_penalty * float(worst_segment if worst_segment is not None else -999.0)
                + args.stress_selection_weight * float(selection_stress_summary["return_pct"])
            )
            ok = constraints_pass(
                selection_summary,
                selection_stress_summary,
                worst_segment,
                min_validation_return_pct=args.min_validation_return_pct,
                min_validation_stress_return_pct=args.min_validation_stress_return_pct,
                min_validation_segment_return_pct=args.min_validation_segment_return_pct,
                max_validation_drawdown_pct=args.max_validation_drawdown_pct,
            )
            selection_allowed = selection_allowed_by_mode(ok, args.decision_mode)
            selected_score = candidate_score if selection_allowed else -np.inf
            row = {
                "window_id": window.window_id,
                **asdict(candidate),
                **{f"selection_{key}": value for key, value in selection_summary.items() if key != "split"},
                "selection_stress_return_pct": selection_stress_summary["return_pct"],
                "selection_stress_max_drawdown_pct": selection_stress_summary["max_drawdown_pct"],
                "worst_selection_segment_return_pct": worst_segment,
                **{f"validation_{key}": value for key, value in selection_summary.items() if key != "split"},
                "validation_stress_return_pct": selection_stress_summary["return_pct"],
                "validation_stress_max_drawdown_pct": selection_stress_summary["max_drawdown_pct"],
                "worst_validation_segment_return_pct": worst_segment,
                "raw_selection_score": raw_score,
                "constraints_ok": ok,
                "selection_allowed": selection_allowed,
                "selection_score": selected_score,
            }
            search_rows.append(row)
            if selection_allowed and (best is None or selected_score > best["selection_score"]):
                best = {
                    **row,
                    "policy": policy,
                    "scores": scores,
                    "confidence": confidence,
                    "selection_summary": selection_summary,
                    "selection_stress_summary": selection_stress_summary,
                }

        if best is None:
            cash_test = empty_summary("test", args.initial_cash)
            window_rows.append(
                {
                    "window_id": window.window_id,
                    "validation_start": window.validation_start.isoformat(),
                    "validation_end": window.validation_end.isoformat(),
                    **{key: value.isoformat() for key, value in spans.items()},
                    "test_start": window.test_start.isoformat(),
                    "test_end": window.test_end.isoformat(),
                    "selected": "cash_selection",
                    "gate_pass": False,
                    "gate_reject_reasons": "selection_constraints",
                    "alpha_name": "cash",
                    "mode": "cash",
                    "k": 0,
                    "gross": 0.0,
                    "rebalance_every": 0,
                    "expected_return_pct": 0.0,
                    "uncertainty_pct": 0.0,
                    "regime_risk": regime["regime_risk"],
                    "stress_edge_pct": 0.0,
                    "trade_probability": 0.0,
                    **empty_diplom_summary("cash_selection"),
                    **{f"test_{key}": value for key, value in cash_test.items() if key != "split"},
                    "worst_test_segment_return_pct": 0.0,
                    "test_stress_return_pct": 0.0,
                }
            )
            selected_rows.append(
                {
                    "window_id": window.window_id,
                    "selected": "cash_selection",
                    "gate_pass": False,
                    "gate_reject_reasons": "selection_constraints",
                    **empty_diplom_summary("cash_selection"),
                }
            )
            print(f"window={window.window_id} CASH selection_constraints", flush=True)
            continue

        selection_summary_with_bars, selection_stress_summary, selection_bars = run_backtest_with_bars(
            best["scores"],
            matrices,
            masks,
            rows_by_split,
            market_features,
            best["confidence"],
            "selection",
            best["policy"],
            initial_cash=args.initial_cash,
            cost_bps=args.cost_bps,
            stress_cost_bps=args.stress_cost_bps,
            engine=args.engine,
            target_adjuster=diplom_target_adjuster,
        )
        gate_summary, gate_stress_summary, gate_bars = run_backtest_with_bars(
            best["scores"],
            matrices,
            masks,
            rows_by_split,
            market_features,
            best["confidence"],
            "gate",
            best["policy"],
            initial_cash=args.initial_cash,
            cost_bps=args.cost_bps,
            stress_cost_bps=args.stress_cost_bps,
            engine=args.engine,
            target_adjuster=diplom_target_adjuster,
        )
        raw_test_summary, raw_test_stress_summary, raw_test_bars = run_backtest_with_bars(
            best["scores"],
            matrices,
            masks,
            rows_by_split,
            market_features,
            best["confidence"],
            "test",
            best["policy"],
            initial_cash=args.initial_cash,
            cost_bps=args.cost_bps,
            stress_cost_bps=args.stress_cost_bps,
            engine=args.engine,
            target_adjuster=diplom_target_adjuster,
        )

        selection_segments = segment_diagnostics(
            selection_bars,
            split="selection",
            initial_cash=args.initial_cash,
            segment_count=args.segment_count,
        )
        gate_segments = segment_diagnostics(
            gate_bars,
            split="gate",
            initial_cash=args.initial_cash,
            segment_count=args.segment_count,
        )
        test_segments = segment_diagnostics(
            raw_test_bars,
            split="test",
            initial_cash=args.initial_cash,
            segment_count=args.segment_count,
        )
        selection_months = calendar_period_diagnostics(
            index,
            selection_bars,
            split="selection",
            initial_cash=args.initial_cash,
        )
        gate_months = calendar_period_diagnostics(
            index,
            gate_bars,
            split="gate",
            initial_cash=args.initial_cash,
        )
        test_months = calendar_period_diagnostics(
            index,
            raw_test_bars,
            split="test",
            initial_cash=args.initial_cash,
        )
        calibration_stats = calibration_stats_from_history(
            calibration_history,
            min_windows=args.calibration_min_windows,
        )
        neural_uncertainty_pct = candidate_neural_uncertainty_pct(
            neural_std_scores,
            best["scores"],
            matrices["tradable"],
            rows_by_split["gate"],
            best["policy"],
        )

        trade_model = build_trade_probability_model(
            selection_summary=selection_summary_with_bars,
            selection_stress_summary=selection_stress_summary,
            selection_periods=selection_months,
            gate_summary=gate_summary,
            gate_stress_summary=gate_stress_summary,
            gate_periods=gate_months,
            regime_risk=float(regime["regime_risk"]),
            edge_uncertainty_weight=args.edge_uncertainty_weight,
            edge_regime_risk_weight_pct=args.edge_regime_risk_weight_pct,
            decision_mode=args.decision_mode,
            calibration_stats=calibration_stats,
            calibration_bias_weight=args.calibration_bias_weight,
            neural_feature_weight=args.neural_feature_weight,
            neural_uncertainty_weight=args.neural_uncertainty_weight,
            neural_uncertainty_pct=neural_uncertainty_pct,
            uses_neural_feature=best["alpha_name"] == "neural_ensemble",
        )
        trade_model.update(
            {
                **empty_diplom_summary(),
                **{key: value for key, value in gate_summary.items() if key.startswith("diplom_")},
            }
        )
        gate_pass, gate_reasons = gate_trade_passes(
            trade_model,
            gate_summary,
            gate_stress_summary,
            decision_mode=args.decision_mode,
            decision_score_threshold=decision_score_threshold,
            min_gate_return_pct=args.min_gate_return_pct,
            min_gate_stress_return_pct=args.min_gate_stress_return_pct,
            min_gate_worst_month_return_pct=args.min_gate_worst_month_return_pct,
            min_gate_month_win_rate=args.min_gate_month_win_rate,
            min_edge_stress_pct=args.min_edge_stress_pct,
            max_regime_risk=args.max_regime_risk,
            regime_veto=args.regime_veto,
        )

        for frame in [selection_segments, gate_segments, test_segments]:
            if not frame.empty:
                frame.insert(0, "window_id", window.window_id)
        for frame in [selection_months, gate_months, test_months]:
            if not frame.empty:
                frame.insert(0, "window_id", window.window_id)
        if not test_segments.empty:
            test_segments.insert(1, "executed", bool(gate_pass))
        if not test_months.empty:
            test_months.insert(1, "executed", bool(gate_pass))
        selection_segments_all.append(selection_segments)
        gate_segments_all.append(gate_segments)
        test_segments_all.append(test_segments)
        validation_segments_all.append(gate_segments)
        selection_months_all.append(selection_months)
        gate_months_all.append(gate_months)
        test_months_all.append(test_months)

        executed_test_summary = raw_test_summary if gate_pass else empty_summary("test", args.initial_cash)
        executed_test_stress_summary = raw_test_stress_summary if gate_pass else empty_summary("test", args.initial_cash)
        worst_test_segment = float(test_segments["return_pct"].min()) if gate_pass and not test_segments.empty else 0.0
        selected_state = "alpha_policy" if gate_pass else "cash_gate_veto"
        selected_rows.append(
            {
                "window_id": window.window_id,
                "selected": selected_state,
                "gate_pass": gate_pass,
                "gate_reject_reasons": gate_reasons,
                **{key: best[key] for key in ["alpha_name", "mode", "k", "gross", "rebalance_every"]},
                **trade_model,
            }
        )
        window_rows.append(
            {
                "window_id": window.window_id,
                "validation_start": window.validation_start.isoformat(),
                "validation_end": window.validation_end.isoformat(),
                **{key: value.isoformat() for key, value in spans.items()},
                "test_start": window.test_start.isoformat(),
                "test_end": window.test_end.isoformat(),
                "selected": selected_state,
                "gate_pass": gate_pass,
                "gate_reject_reasons": gate_reasons,
                "alpha_name": best["alpha_name"] if gate_pass else "cash",
                "mode": best["mode"] if gate_pass else "cash",
                "k": best["k"] if gate_pass else 0,
                "gross": best["gross"] if gate_pass else 0.0,
                "rebalance_every": best["rebalance_every"] if gate_pass else 0,
                "raw_alpha_name": best["alpha_name"],
                "raw_mode": best["mode"],
                "raw_k": best["k"],
                "raw_gross": best["gross"],
                "raw_rebalance_every": best["rebalance_every"],
                **{f"selection_{key}": value for key, value in selection_summary_with_bars.items() if key != "split"},
                "selection_stress_return_pct": selection_stress_summary["return_pct"],
                "worst_selection_segment_return_pct": (
                    float(selection_segments["return_pct"].min()) if not selection_segments.empty else None
                ),
                **{f"validation_{key}": value for key, value in gate_summary.items() if key != "split"},
                "validation_stress_return_pct": gate_stress_summary["return_pct"],
                "worst_validation_segment_return_pct": (
                    float(gate_segments["return_pct"].min()) if not gate_segments.empty else None
                ),
                **{f"gate_{key}": value for key, value in gate_summary.items() if key != "split"},
                "gate_stress_return_pct": gate_stress_summary["return_pct"],
                "worst_gate_segment_return_pct": (
                    float(gate_segments["return_pct"].min()) if not gate_segments.empty else None
                ),
                **trade_model,
                **{f"test_{key}": value for key, value in executed_test_summary.items() if key != "split"},
                "worst_test_segment_return_pct": worst_test_segment,
                "test_stress_return_pct": executed_test_stress_summary["return_pct"],
                **{f"raw_test_{key}": value for key, value in raw_test_summary.items() if key != "split"},
                "raw_test_stress_return_pct": raw_test_stress_summary["return_pct"],
            }
        )
        calibration_history.append(
            {
                "window_id": window.window_id,
                "test_start": window.test_start.isoformat(),
                "selected": selected_state,
                "gate_pass": gate_pass,
                "gate_reject_reasons": gate_reasons,
                "alpha_name": best["alpha_name"],
                "mode": best["mode"],
                "expected_return_pct": trade_model["expected_return_pct"],
                "calibrated_expected_return_pct": trade_model["calibrated_expected_return_pct"],
                "uncertainty_pct": trade_model["uncertainty_pct"],
                "regime_risk": trade_model["regime_risk"],
                "stress_edge_pct": trade_model["stress_edge_pct"],
                "decision_score": trade_model["decision_score"],
                "trade_probability": trade_model["trade_probability"],
                "raw_test_return_pct": raw_test_summary["return_pct"],
                "raw_test_stress_return_pct": raw_test_stress_summary["return_pct"],
            }
        )
        print(
            f"window={window.window_id} selection={best['selection_return_pct']:.2f}% "
            f"gate={gate_summary['return_pct']:.2f}% edge={trade_model['stress_edge_pct']:.2f}% "
            f"risk={trade_model['regime_risk']:.2f} selected={selected_state} "
            f"test={executed_test_summary['return_pct']:.2f}% raw_test={raw_test_summary['return_pct']:.2f}% "
            f"{best['alpha_name']} {best['mode']} k={best['k']} gross={best['gross']}",
            flush=True,
        )

    search = pd.DataFrame(search_rows)
    window_results = pd.DataFrame(window_rows)
    selected = pd.DataFrame(selected_rows)
    selection_segments_out = (
        pd.concat(selection_segments_all, ignore_index=True) if selection_segments_all else pd.DataFrame()
    )
    gate_segments_out = pd.concat(gate_segments_all, ignore_index=True) if gate_segments_all else pd.DataFrame()
    validation_segments_out = (
        pd.concat(validation_segments_all, ignore_index=True) if validation_segments_all else pd.DataFrame()
    )
    test_segments_out = pd.concat(test_segments_all, ignore_index=True) if test_segments_all else pd.DataFrame()
    selection_months_out = pd.concat(selection_months_all, ignore_index=True) if selection_months_all else pd.DataFrame()
    gate_months_out = pd.concat(gate_months_all, ignore_index=True) if gate_months_all else pd.DataFrame()
    test_months_out = pd.concat(test_months_all, ignore_index=True) if test_months_all else pd.DataFrame()
    regime_out = pd.DataFrame(regime_rows)
    window_month_results = executed_window_month_results(window_results)
    raw_vs_gated_months = raw_vs_gated_month_results(window_results)
    veto_summary, veto_by_reason = veto_diagnostics(window_results)
    window_month_stability = period_stability_summary(
        window_month_results.rename(columns={"test_month": "period"}) if not window_month_results.empty else pd.DataFrame()
    )
    compounded = summarize_compounded(window_results, args.initial_cash, args.target_return_pct)
    stress_window_results = window_results.copy()
    stress_window_results["test_return_pct"] = stress_window_results["test_stress_return_pct"].astype(float)
    compounded_stress = summarize_compounded(stress_window_results, args.initial_cash, args.target_return_pct)
    traded = window_results[window_results["selected"] == "alpha_policy"]
    gate_vetoed = window_results[window_results["selected"] == "cash_gate_veto"]
    raw_changed_col = window_results.get("raw_test_diplom_risk_changed_count", pd.Series(0, index=window_results.index))
    raw_available_col = window_results.get("raw_test_diplom_risk_available_count", pd.Series(0, index=window_results.index))
    raw_missing_col = window_results.get("raw_test_diplom_risk_missing_count", pd.Series(0, index=window_results.index))
    raw_stale_col = window_results.get("raw_test_diplom_risk_stale_count", pd.Series(0, index=window_results.index))
    diplom_changed_decisions = int(pd.to_numeric(raw_changed_col, errors="coerce").fillna(0).sum())
    diplom_available_decisions = int(pd.to_numeric(raw_available_col, errors="coerce").fillna(0).sum())
    diplom_missing_decisions = int(pd.to_numeric(raw_missing_col, errors="coerce").fillna(0).sum())
    diplom_stale_decisions = int(pd.to_numeric(raw_stale_col, errors="coerce").fillna(0).sum())
    cash_only_rate = float((window_results["selected"] != "alpha_policy").mean()) if len(window_results) else 0.0

    search.to_csv(args.output_dir / "validation_search.csv", index=False)
    window_results.to_csv(args.output_dir / "window_results.csv", index=False)
    selected.to_csv(args.output_dir / "selected_candidates.csv", index=False)
    selection_segments_out.to_csv(args.output_dir / "selection_segments.csv", index=False)
    gate_segments_out.to_csv(args.output_dir / "gate_segments.csv", index=False)
    validation_segments_out.to_csv(args.output_dir / "validation_segments.csv", index=False)
    test_segments_out.to_csv(args.output_dir / "test_segments.csv", index=False)
    selection_months_out.to_csv(args.output_dir / "selection_months.csv", index=False)
    gate_months_out.to_csv(args.output_dir / "gate_months.csv", index=False)
    test_months_out.to_csv(args.output_dir / "test_months.csv", index=False)
    regime_out.to_csv(args.output_dir / "regime_risk.csv", index=False)
    window_month_results.to_csv(args.output_dir / "window_month_results.csv", index=False)
    raw_vs_gated_months.to_csv(args.output_dir / "raw_vs_gated_month_results.csv", index=False)
    veto_by_reason.to_csv(args.output_dir / "veto_by_reason.csv", index=False)
    summary = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "dataset_dir": str(args.dataset_dir),
        "interval": args.interval,
        "horizon": args.horizon,
        "symbols": symbols,
        "window_count": int(len(window_results)),
        "trade_window_count": int(len(traded)),
        "candidate_count_per_window": int(len(candidates)),
        "alpha_lags": parse_ints(args.alpha_lags),
        "alpha_kinds": [item.strip() for item in args.alpha_kinds.split(",") if item.strip()],
        "alpha_normalize": args.alpha_normalize,
        "modes": modes,
        "k_grid": parse_ints(args.k_grid),
        "gross_grid": parse_floats(args.gross_grid),
        "rebalance_grid": parse_ints(args.rebalance_grid),
        "signal_delay_bars": args.signal_delay_bars,
        "engine": args.engine,
        "decision_mode": args.decision_mode,
        "decision_score_threshold": decision_score_threshold,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "validation_days": args.validation_days,
        "nested_gate_days": args.nested_gate_days,
        "test_days": args.test_days,
        "step_days": args.step_days,
        "cost_bps": args.cost_bps,
        "stress_cost_bps": args.stress_cost_bps,
        "selection_constraints": {
            "min_validation_return_pct": args.min_validation_return_pct,
            "min_validation_stress_return_pct": args.min_validation_stress_return_pct,
            "min_validation_segment_return_pct": args.min_validation_segment_return_pct,
            "max_validation_drawdown_pct": args.max_validation_drawdown_pct,
        },
        "gate_constraints": {
            "min_gate_return_pct": args.min_gate_return_pct,
            "min_gate_stress_return_pct": args.min_gate_stress_return_pct,
            "min_gate_worst_month_return_pct": args.min_gate_worst_month_return_pct,
            "min_gate_month_win_rate": args.min_gate_month_win_rate,
            "min_edge_stress_pct": args.min_edge_stress_pct,
            "edge_uncertainty_weight": args.edge_uncertainty_weight,
            "edge_regime_risk_weight_pct": args.edge_regime_risk_weight_pct,
            "max_regime_risk": args.max_regime_risk,
            "regime_veto": args.regime_veto,
        },
        "calibration": {
            "min_windows": args.calibration_min_windows,
            "bias_weight": args.calibration_bias_weight,
        },
        "feature_model": {
            "name": "alpha_neural_feature_model",
            "neural_scores_paths": neural_used_paths,
            "neural_feature": "ensemble_mean_score",
            "neural_uncertainty": "ensemble_std_dispersion",
            "neural_feature_weight": args.neural_feature_weight,
            "neural_uncertainty_weight": args.neural_uncertainty_weight,
            "note": "Neural/Kronos-family scores are feature and uncertainty inputs, not price-oracle forecasts.",
        },
        "diplom_risk_overlay": {
            "enabled": bool(diplom_config.enabled),
            "predictions_path": str(diplom_config.predictions_path) if diplom_config.predictions_path else None,
            "risk_weight_pct": diplom_config.risk_weight_pct,
            "long_veto_p_high": diplom_config.long_veto_p_high,
            "short_bonus_weight_pct": diplom_config.short_bonus_weight_pct,
            "stale_days": diplom_config.stale_days,
            "missing_policy": diplom_config.missing_policy,
            "enable_yndx_ydex_mapping": diplom_config.enable_yndx_ydex_mapping,
            "yndx_ydex_effective_date": diplom_config.yndx_ydex_effective_date,
            "ignored_forbidden_columns": diplom_adapter.ignored_forbidden_columns if diplom_adapter else [],
            "changed_rebalance_decisions": diplom_changed_decisions,
            "available_rebalance_decisions": diplom_available_decisions,
            "missing_rebalance_decisions": diplom_missing_decisions,
            "stale_rebalance_decisions": diplom_stale_decisions,
            "cash_only_rate": cash_only_rate,
        },
        "regime_classifier": {
            "type": "prior_only_bad_block_similarity",
            "block_days": args.regime_block_days,
            "bad_return_quantile": args.regime_bad_return_quantile,
            "bad_drawdown_quantile": args.regime_bad_drawdown_quantile,
            "min_history_windows": args.min_regime_history_windows,
        },
        "selection": {
            "drawdown_penalty": args.drawdown_penalty,
            "turnover_penalty": args.turnover_penalty,
            "segment_selection_penalty": args.segment_selection_penalty,
            "stress_selection_weight": args.stress_selection_weight,
        },
        "compounded_test_summary": compounded,
        "compounded_stress_test_summary": compounded_stress,
        "executed_test_month_stability": window_month_stability,
        "raw_vs_gated_veto_summary": veto_summary,
        "trade_window_win_rate": float((traded["test_return_pct"] > 0.0).mean()) if len(traded) else 0.0,
        "worst_trade_window_test_return_pct": float(traded["test_return_pct"].min()) if len(traded) else 0.0,
        "mean_trade_window_test_return_pct": float(traded["test_return_pct"].mean()) if len(traded) else 0.0,
        "gate_veto_window_count": int(len(gate_vetoed)),
        "cash_only_rate": cash_only_rate,
        "diplom_changed_rebalance_decisions": diplom_changed_decisions,
        "diplom_available_rebalance_decisions": diplom_available_decisions,
        "diplom_missing_rebalance_decisions": diplom_missing_decisions,
        "diplom_stale_rebalance_decisions": diplom_stale_decisions,
        "mean_trade_probability": float(selected["trade_probability"].dropna().mean()) if "trade_probability" in selected else 0.0,
        "mean_stress_edge_pct": float(selected["stress_edge_pct"].dropna().mean()) if "stress_edge_pct" in selected else 0.0,
        "mean_regime_risk": float(selected["regime_risk"].dropna().mean()) if "regime_risk" in selected else 0.0,
        "worst_window_test_max_drawdown_pct": float(window_results["test_max_drawdown_pct"].min()),
        "config_json": str(args.config_json) if args.config_json else None,
        "frozen_protocol_config": args_to_config(args)["config"],
        "protocol": (
            "For each window, alpha and execution parameters are selected only on the earlier selection slice. "
            "The later gate slice is a nested holdout used strictly as a veto/stress test, not to choose an "
            "alternate meta-configuration. The selected candidate trades the test slice only when the gate "
            "trade-probability model passes the selected decision-mode risk budget. Alpha and neural ensemble "
            "scores are treated as feature inputs and uncertainty signals, not as price-oracle forecasts. "
            f"Alpha scores and regime features are delayed by {args.signal_delay_bars} bar(s) before selection."
        ),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    readme = [
        "# Walk-Forward Alpha Policy Lab",
        "",
        "Rolling alpha and execution-policy selection with a strict cash fallback.",
        "",
        "## Protocol",
        "",
        f"- Dataset: `{args.dataset_dir}`",
        f"- Windows: {len(window_results)}",
        f"- Trade windows: {len(traded)}",
        f"- Validation days: {args.validation_days}",
        f"- Nested gate days: {args.nested_gate_days}",
        f"- Test days: {args.test_days}",
        f"- Signal delay bars: {args.signal_delay_bars}",
        f"- Engine: {args.engine}",
        f"- Decision mode: {args.decision_mode}",
        f"- Decision score threshold: {decision_score_threshold}",
        f"- Candidate count per window: {len(candidates)}",
        f"- Cost: {args.cost_bps:.2f} bps",
        f"- Selection min return: {args.min_validation_return_pct}",
        f"- Selection min worst segment: {args.min_validation_segment_return_pct}",
        f"- Gate min stress return: {args.min_gate_stress_return_pct}",
        f"- Gate min stress edge: {args.min_edge_stress_pct}",
        f"- Max May-like regime risk: {args.max_regime_risk}",
        f"- Gate veto windows: {len(gate_vetoed)}",
        f"- Diplom risk enabled: {diplom_config.enabled}",
        f"- Diplom changed rebalance decisions: {diplom_changed_decisions}",
        f"- Diplom available/missing/stale rebalances: {diplom_available_decisions}/{diplom_missing_decisions}/{diplom_stale_decisions}",
        f"- Avoided loss: {veto_summary['avoided_loss_pct']:.2f} pct-points",
        f"- Missed profit: {veto_summary['missed_profit_pct']:.2f} pct-points",
        "",
        "## Compounded Test Result",
        "",
        f"- Initial cash: {compounded['initial_cash']:.2f}",
        f"- Final equity: {compounded['final_equity']:.2f}",
        f"- PnL: {compounded['pnl']:.2f}",
        f"- Return: {compounded['return_pct']:.2f}%",
        f"- Stress return at {args.stress_cost_bps:.2f} bps: {compounded_stress['return_pct']:.2f}%",
        f"- Window endpoint max drawdown: {compounded['max_drawdown_pct']:.2f}%",
        f"- Worst per-window max drawdown: {summary['worst_window_test_max_drawdown_pct']:.2f}%",
        f"- Trade-window win rate: {summary['trade_window_win_rate'] * 100.0:.2f}%",
        f"- Worst traded window: {summary['worst_trade_window_test_return_pct']:.2f}%",
        f"- Executed test month win rate: {window_month_stability['period_win_rate'] * 100.0:.2f}%",
        f"- Worst executed test month: {window_month_stability['worst_period_return_pct']}",
        f"- Target hit ({args.target_return_pct:.1f}%): {compounded['target_hit']}",
        "",
        "The nested gate is a veto only. If the selection winner fails stress edge, month stability, or "
        "May-like regime risk, the window is recorded as cash and the candidate's raw test result is kept "
        "under `raw_test_*` fields for audit.",
        "",
        "This is a research backtest, not financial advice or a live trading recommendation.",
        "",
    ]
    (args.output_dir / "README.md").write_text("\n".join(readme), encoding="utf-8")
    print(f"Output dir: {args.output_dir}")
    print(f"Windows evaluated: {len(window_results)}")
    print(f"Trade windows: {len(traded)}")
    print(f"Compounded test return: {compounded['return_pct']:.2f}%")
    print(f"Target hit: {compounded['target_hit']}")


if __name__ == "__main__":
    main()
