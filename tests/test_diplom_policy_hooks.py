from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from walk_forward_alpha_policy_lab import apply_diplom_gate_penalty, diplom_score_penalty, gate_trade_passes, sigmoid


def test_candidate_penalty_uses_long_risk_exposure_before_max_p_high() -> None:
    summary = {
        "diplom_long_p_high_exposure": 0.25,
        "diplom_p_high": 0.90,
    }

    assert diplom_score_penalty(summary, 10.0) == 2.5


def test_candidate_penalty_falls_back_to_max_p_high() -> None:
    summary = {
        "diplom_long_p_high_exposure": np.nan,
        "diplom_p_high": 0.90,
    }

    assert diplom_score_penalty(summary, 10.0) == 9.0


def test_gate_penalty_updates_decision_score_and_probability() -> None:
    model = {
        "stress_edge_pct": 1.0,
        "decision_score": 2.0,
        "uncertainty_pct": 2.0,
        "trade_probability": sigmoid(1.0),
    }
    gate_summary = {
        "diplom_long_p_high_exposure": 0.5,
        "diplom_p_high": 0.9,
    }

    updated = apply_diplom_gate_penalty(model, gate_summary, 4.0)

    assert updated["stress_edge_before_diplom_pct"] == 1.0
    assert updated["decision_score_before_diplom"] == 2.0
    assert updated["diplom_gate_penalty_score_pct"] == 2.0
    assert updated["stress_edge_pct"] == -1.0
    assert updated["decision_score"] == 0.0
    assert updated["trade_probability"] == 0.5
    assert "diplom_gate_penalty" in updated["soft_risk_flags"]


def test_gate_trade_passes_preserves_existing_diplom_soft_flag() -> None:
    model = {
        "stress_edge_pct": -1.0,
        "decision_score": -1.0,
        "regime_risk": 0.1,
        "gate_period_win_rate": 1.0,
        "soft_risk_flags": "diplom_gate_penalty",
    }
    gate_summary = {"return_pct": 0.0}
    gate_stress_summary = {"return_pct": -1.0}

    passed, reasons = gate_trade_passes(
        model,
        gate_summary,
        gate_stress_summary,
        decision_mode="balanced",
        decision_score_threshold=0.0,
        min_gate_return_pct=None,
        min_gate_stress_return_pct=0.0,
        min_gate_worst_month_return_pct=None,
        min_gate_month_win_rate=0.5,
        min_edge_stress_pct=0.0,
        max_regime_risk=0.7,
        regime_veto=True,
    )

    assert passed is False
    assert reasons == "decision_score"
    assert "diplom_gate_penalty" in model["soft_risk_flags"]
    assert "stress_edge" in model["soft_risk_flags"]
