from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from build_fresh_diplom_risk_artifact import validate_predictions


def make_prediction_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "decision_date": "2026-01-31",
                "availability_timestamp": "2026-01-31 23:41:00",
                "source_data_end": "2026-01-31 23:40:00",
                "ticker": "AAA",
                "kronos_symbol": "AAA",
                "sector": "moex",
                "predicted_risk_class": "medium",
                "p_low": 0.2,
                "p_medium": 0.5,
                "p_high": 0.3,
                "feature_coverage": 1.0,
                "model_version": "test",
                "final_architecture": "past_only",
                "used_sector_expert": False,
            }
        ]
    )


def test_fresh_artifact_validation_accepts_target_free_schema() -> None:
    validation = validate_predictions(make_prediction_frame())

    assert validation["target_free"] is True
    assert validation["forbidden_columns_present"] == []
    assert validation["duplicate_ticker_availability_rows"] == 0
    assert validation["probability_bounds_bad_cells"] == 0
    assert validation["probability_sum_max_abs_error"] < 1e-12
    assert validation["availability_not_after_source_data_end_rows"] == 0


def test_fresh_artifact_validation_rejects_forbidden_columns() -> None:
    frame = make_prediction_frame()
    frame["future_max_drawdown"] = -0.4

    validation = validate_predictions(frame)

    assert validation["target_free"] is False
    assert validation["forbidden_columns_present"] == ["future_max_drawdown"]
