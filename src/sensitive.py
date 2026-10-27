from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline

from src.features import build_preprocessor, split_xy

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT = ROOT / "reports" / "sensitive_comparison.csv"
SEED = 42


def run(include_sensitive, X_train, y_train, X_val, y_val):
    pipe = Pipeline(
        [
            ("prep", build_preprocessor(include_sensitive=include_sensitive)),
            ("model", HistGradientBoostingClassifier(random_state=SEED)),
        ]
    )
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_val)
    proba = pipe.predict_proba(X_val)[:, 1]
    row = {
        "include_race_sex": include_sensitive,
        "val_f1": round(f1_score(y_val, pred), 4),
        "val_auc": round(roc_auc_score(y_val, proba), 4),
    }
    for sex in ("Female", "Male"):
        mask = (X_val["sex"] == sex).to_numpy()
        row[f"{sex}_recall"] = round(recall_score(y_val[mask], pred[mask]), 4)
        row[f"{sex}_positive_rate"] = round(float(pred[mask].mean()), 4)
    return row


if __name__ == "__main__":
    for name in ("train", "val"):
        if not (DATA_DIR / f"{name}.csv").exists():
            raise FileNotFoundError(f"{name}.csv not found. Run: python src\\split.py")
    X_train, y_train = split_xy(pd.read_csv(DATA_DIR / "train.csv"))
    X_val, y_val = split_xy(pd.read_csv(DATA_DIR / "val.csv"))
    results = pd.DataFrame(
        [run(flag, X_train, y_train, X_val, y_val) for flag in (False, True)]
    )
    OUT.parent.mkdir(exist_ok=True)
    results.to_csv(OUT, index=False)
    print(results.to_string(index=False))
