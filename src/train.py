from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from src.features import build_preprocessor, split_xy

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT = ROOT / "reports" / "results_validation.csv"
SEED = 42


def get_models():
    return {
        "baseline_prior": DummyClassifier(strategy="prior"),
        "logreg": LogisticRegression(max_iter=1000),
        "logreg_balanced": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "hist_gb": HistGradientBoostingClassifier(random_state=SEED),
    }


def evaluate(models, X_train, y_train, X_val, y_val):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows = []
    for name, model in models.items():
        pipe = Pipeline([("prep", build_preprocessor()), ("model", model)])
        scores = cross_validate(
            pipe, X_train, y_train, cv=cv, scoring={"f1": "f1", "auc": "roc_auc"}
        )
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_val)
        proba = pipe.predict_proba(X_val)[:, 1]
        rows.append(
            {
                "model": name,
                "cv_f1_mean": round(scores["test_f1"].mean(), 4),
                "cv_f1_std": round(scores["test_f1"].std(), 4),
                "cv_auc_mean": round(scores["test_auc"].mean(), 4),
                "cv_auc_std": round(scores["test_auc"].std(), 4),
                "val_f1": round(f1_score(y_val, pred, zero_division=0), 4),
                "val_auc": round(roc_auc_score(y_val, proba), 4),
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    for name in ("train", "val"):
        if not (DATA_DIR / f"{name}.csv").exists():
            raise FileNotFoundError(f"{name}.csv not found. Run: python src\\split.py")
    X_train, y_train = split_xy(pd.read_csv(DATA_DIR / "train.csv"))
    X_val, y_val = split_xy(pd.read_csv(DATA_DIR / "val.csv"))
    results = evaluate(get_models(), X_train, y_train, X_val, y_val)
    OUT.parent.mkdir(exist_ok=True)
    results.to_csv(OUT, index=False)
    print(results.to_string(index=False))
