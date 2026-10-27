import pandas as pd
import pytest

from src.error_section import build_section


def sample():
    return pd.DataFrame(
        {
            "feature": ["sex", "sex", "occupation", "occupation"],
            "group": ["Female", "Male", "Sales", "Tech-support"],
            "n": [3000, 6000, 1000, 300],
            "share_positive": [0.11, 0.30, 0.26, 0.30],
            "miss_rate": [0.38, 0.33, 0.45, 0.30],
            "false_alarm_rate": [0.02, 0.07, 0.05, 0.06],
        }
    )


def test_section_has_header_and_groups():
    text = build_section(sample())
    assert text.startswith("## Error analysis")
    assert "Female" in text and "Sales" in text


def test_missing_column_raises():
    with pytest.raises(ValueError):
        build_section(sample().drop(columns=["miss_rate"]))
