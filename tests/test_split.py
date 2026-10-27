import pandas as pd
import pytest

from src.split import make_splits


def make_df(n=1000):
    return pd.DataFrame(
        {
            "x": range(n),
            "class": ["<=50K"] * (n * 3 // 4) + [">50K"] * (n - n * 3 // 4),
        }
    )


def test_splits_do_not_overlap():
    train, val, test = make_splits(make_df())
    assert set(train.index).isdisjoint(val.index)
    assert set(train.index).isdisjoint(test.index)
    assert set(val.index).isdisjoint(test.index)


def test_splits_cover_all_rows_with_expected_sizes():
    df = make_df()
    train, val, test = make_splits(df)
    assert len(train) + len(val) + len(test) == len(df)
    assert abs(len(train) / len(df) - 0.6) < 0.01


def test_class_ratio_is_preserved():
    df = make_df()
    base = (df["class"] == ">50K").mean()
    for part in make_splits(df):
        assert abs((part["class"] == ">50K").mean() - base) < 0.02


def test_missing_target_raises():
    with pytest.raises(ValueError):
        make_splits(pd.DataFrame({"x": [1, 2, 3]}))
