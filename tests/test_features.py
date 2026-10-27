import pandas as pd
import pytest

from src.features import build_preprocessor, split_xy


def sample_df():
    return pd.DataFrame(
        {
            "age": [25, 40, 35, 50],
            "workclass": ["Private", None, "State-gov", "Private"],
            "fnlwgt": [1, 2, 3, 4],
            "education": ["HS-grad", "Bachelors", "Masters", "HS-grad"],
            "education-num": [9, 13, 14, 9],
            "marital-status": ["Never-married", "Married-civ-spouse", "Divorced", "Never-married"],
            "occupation": ["Sales", None, "Tech-support", "Sales"],
            "relationship": ["Own-child", "Husband", "Unmarried", "Own-child"],
            "race": ["White", "Black", "White", "Asian-Pac-Islander"],
            "sex": ["Male", "Female", "Male", "Female"],
            "capital-gain": [0, 0, 5000, 0],
            "capital-loss": [0, 0, 0, 0],
            "hours-per-week": [40, 50, 45, 30],
            "native-country": ["United-States", None, "India", "United-States"],
            "class": ["<=50K", ">50K", ">50K", "<=50K"],
        }
    )


def test_split_xy_encodes_target():
    X, y = split_xy(sample_df())
    assert list(y) == [0, 1, 1, 0]
    assert "class" not in X.columns


def test_split_xy_missing_target_raises():
    with pytest.raises(ValueError):
        split_xy(pd.DataFrame({"age": [1, 2]}))


def test_preprocessor_handles_missing_and_unseen_values():
    X, _ = split_xy(sample_df())
    prep = build_preprocessor()
    out = prep.fit_transform(X)
    assert not pd.DataFrame(out).isna().any().any()
    new_row = X.iloc[[0]].copy()
    new_row["native-country"] = "Atlantis"
    assert prep.transform(new_row).shape[1] == out.shape[1]


def test_sensitive_columns_excluded_by_default():
    X, _ = split_xy(sample_df())
    default_names = build_preprocessor().fit(X).get_feature_names_out()
    with_names = build_preprocessor(include_sensitive=True).fit(X).get_feature_names_out()
    assert not any("race" in n or "sex" in n for n in default_names)
    assert any("race" in n for n in with_names)
