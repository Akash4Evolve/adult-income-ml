import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from src.features import build_preprocessor, split_xy
from src.train import get_models

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS = ROOT / "reports" / "results_validation.csv"
OUT = ROOT / "reports" / "results_test.json"


def pick_best(results):
    """Best model by validation AUC, ignoring the baseline."""
    candidates = results[results["model"] != "baseline_prior"]
    if candidates.empty:
        raise ValueError("No non-baseline models in results.")
    return candidates.sort_values("val_auc", ascending=False).iloc[0]["model"]


if __name__ == "__main__":
    if OUT.exists():
        raise SystemExit(
            f"{OUT.name} already exists. The test set is scored once. "
            "Delete it only if you changed the code, and say so in the README."
        )
    for path in (RESULTS, DATA_DIR / "train.csv", DATA_DIR / "val.csv", DATA_DIR / "test.csv"):
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run split.py and train.py first.")
    best = pick_best(pd.read_csv(RESULTS))
    train = pd.read_csv(DATA_DIR / "train.csv")
    val = pd.read_csv(DATA_DIR / "val.csv")
    X_fit, y_fit = split_xy(pd.concat([train, val], ignore_index=True))
    X_test, y_test = split_xy(pd.read_csv(DATA_DIR / "test.csv"))
    pipe = Pipeline(
        [("prep", build_preprocessor()), ("model", get_models()[best])]
    )
    pipe.fit(X_fit, y_fit)
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    result = {
        "model": best,
        "trained_on": "train+val",
        "test_f1": round(f1_score(y_test, pred), 4),
        "test_auc": round(roc_auc_score(y_test, proba), 4),
        "test_precision": round(precision_score(y_test, pred), 4),
        "test_recall": round(recall_score(y_test, pred), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
