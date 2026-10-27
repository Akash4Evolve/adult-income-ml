import pandas as pd
import pytest

from src.error_analysis import group_error_rates


def test_group_error_rates_computes_miss_and_false_alarm():
    df = pd.DataFrame(
        {
            "g": ["a"] * 4 + ["b"] * 4,
            "y_true": [1, 1, 0, 0, 1, 1, 0, 0],
            "y_pred": [1, 0, 0, 1, 1, 1, 0, 0],
        }
    )
    out = group_error_rates(df, "g", min_size=1).set_index("group")
    assert out.loc["a", "miss_rate"] == 0.5
    assert out.loc["a", "false_alarm_rate"] == 0.5
    assert out.loc["b", "miss_rate"] == 0.0


def test_small_groups_are_skipped():
    df = pd.DataFrame({"g": ["a", "a"], "y_true": [1, 0], "y_pred": [1, 0]})
    assert group_error_rates(df, "g", min_size=100).empty


def test_missing_column_raises():
    with pytest.raises(ValueError):
        group_error_rates(pd.DataFrame({"y_true": [1], "y_pred": [1]}), "g")
