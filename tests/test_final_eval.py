import pandas as pd
import pytest

from src.final_eval import pick_best


def test_pick_best_ignores_baseline():
    df = pd.DataFrame(
        {"model": ["baseline_prior", "a", "b"], "val_auc": [0.99, 0.80, 0.90]}
    )
    assert pick_best(df) == "b"


def test_pick_best_requires_a_real_model():
    df = pd.DataFrame({"model": ["baseline_prior"], "val_auc": [0.5]})
    with pytest.raises(ValueError):
        pick_best(df)
