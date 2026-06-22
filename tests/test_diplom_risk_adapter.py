from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from diplom_risk_adapter import DiplomRiskAdapter, DiplomRiskConfig, config_from_mapping


def make_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "decision_date": "2025-12-31",
                "ticker": "AAA",
                "sector": "test",
                "predicted_risk_class": "low",
                "p_low": 0.8,
                "p_medium": 0.1,
                "p_high": 0.1,
                "risk_class": "high",
                "future_max_drawdown": 0.99,
            },
            {
                "decision_date": "2026-01-31",
                "ticker": "AAA",
                "sector": "test",
                "predicted_risk_class": "high",
                "p_low": 0.1,
                "p_medium": 0.2,
                "p_high": 0.7,
                "risk_class": "low",
                "future_cvar_95": 0.01,
            },
            {
                "decision_date": "2026-01-31",
                "ticker": "BBB",
                "sector": "test",
                "predicted_risk_class": "high",
                "p_low": 0.05,
                "p_medium": 0.1,
                "p_high": 0.85,
            },
            {
                "decision_date": "2024-08-31",
                "ticker": "YNDX",
                "sector": "tech",
                "predicted_risk_class": "medium",
                "p_low": 0.2,
                "p_medium": 0.5,
                "p_high": 0.3,
            },
        ]
    )


def test_forbidden_columns_are_ignored_on_inference() -> None:
    adapter = DiplomRiskAdapter(DiplomRiskConfig(enabled=True, stale_days=999), make_predictions())

    assert "future_max_drawdown" in adapter.ignored_forbidden_columns
    assert "risk_class" in adapter.ignored_forbidden_columns
    assert "future_max_drawdown" not in adapter.predictions.columns
    assert "risk_class" not in adapter.predictions.columns


def test_asof_lookup_never_uses_future_prediction() -> None:
    adapter = DiplomRiskAdapter(DiplomRiskConfig(enabled=True, stale_days=999), make_predictions())

    lookup = adapter.lookup("AAA", "2026-01-15")
    assert lookup["status"] == "ok"
    assert str(lookup["decision_date"].date()) == "2025-12-31"
    assert lookup["p_high"] == 0.1

    missing = adapter.lookup("AAA", "2025-12-01")
    assert missing["status"] == "missing"


def test_stale_and_missing_policy_cash_zeroes_target() -> None:
    adapter = DiplomRiskAdapter(
        DiplomRiskConfig(enabled=True, stale_days=5, missing_policy="cash"),
        make_predictions(),
    )

    target = np.array([1.0, -1.0], dtype=np.float32)
    adjusted, diag = adapter.apply_to_target(target, symbols=["AAA", "MISSING"], timestamp="2026-02-20")

    assert np.allclose(adjusted, np.zeros_like(target))
    assert diag["diplom_risk_stale_count"] == 1
    assert diag["diplom_risk_missing_count"] == 1
    assert "diplom_missing_policy_cash" in diag["diplom_risk_reason"]


def test_stale_days_zero_is_preserved_from_config() -> None:
    config = config_from_mapping({"diplom_risk_enabled": True, "diplom_stale_days": 0})

    assert config.stale_days == 0


def test_invalid_probability_bounds_are_rejected() -> None:
    bad = make_predictions()
    bad.loc[0, "p_high"] = 1.5

    try:
        DiplomRiskAdapter(DiplomRiskConfig(enabled=True), bad)
    except ValueError as exc:
        assert "within [0, 1]" in str(exc)
    else:
        raise AssertionError("invalid p_high was not rejected")


def test_yndx_to_ydex_mapping_requires_explicit_flag_and_effective_date() -> None:
    disabled = DiplomRiskAdapter(DiplomRiskConfig(enabled=True, stale_days=999), make_predictions())
    assert disabled.lookup("YDEX", "2024-08-31")["status"] == "missing"

    enabled = DiplomRiskAdapter(
        DiplomRiskConfig(enabled=True, stale_days=999, enable_yndx_ydex_mapping=True),
        make_predictions(),
    )
    before_effective = enabled.lookup("YDEX", "2024-07-23")
    assert before_effective["status"] == "missing"

    mapped = enabled.lookup("YDEX", "2024-08-31")
    assert mapped["status"] == "ok"
    assert mapped["diplom_ticker"] == "YNDX"
    assert "mapped_yndx_to_ydex" in mapped["reason"]


def test_long_penalty_does_not_turn_high_risk_into_default_short_signal() -> None:
    adapter = DiplomRiskAdapter(
        DiplomRiskConfig(enabled=True, stale_days=999, risk_weight_pct=20.0, short_bonus_weight_pct=0.0),
        make_predictions(),
    )

    target = np.array([1.0, -1.0], dtype=np.float32)
    adjusted, diag = adapter.apply_to_target(target, symbols=["AAA", "BBB"], timestamp="2026-01-31")

    assert adjusted[0] < target[0]
    assert adjusted[1] == target[1]
    assert diag["diplom_risk_changed_count"] == 1
    assert "diplom_long_risk_penalty" in diag["diplom_risk_reason"]


def test_long_veto_zeros_only_long_leg_by_default() -> None:
    adapter = DiplomRiskAdapter(
        DiplomRiskConfig(enabled=True, stale_days=999, long_veto_p_high=0.6),
        make_predictions(),
    )

    target = np.array([1.0, -1.0], dtype=np.float32)
    adjusted, diag = adapter.apply_to_target(target, symbols=["AAA", "BBB"], timestamp="2026-01-31")

    assert adjusted[0] == 0.0
    assert adjusted[1] == target[1]
    assert diag["diplom_risk_penalty_pct"] == 100.0
    assert "diplom_long_veto" in diag["diplom_risk_reason"]
