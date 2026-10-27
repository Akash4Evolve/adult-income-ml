from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline

from src.features import build_preprocessor, split_xy
from src.final_eval import pick_best
from src.train import get_models

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS = ROOT / "reports" / "results_validation.csv"
OUT = ROOT / "reports" / "error_analysis.csv"
GROUP_COLS = ["occupation", "sex", "marital-status"]


def group_error_rates(df, group_col, min_size=100):
    """df needs y_true, y_pred (0/1) and group_col. Returns one row per group."""
    for col in ("y_true", "y_pred", group_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' missing.")
    groups = df[group_col].fillna("Unknown")
    rows = []
    for name, g in df.groupby(groups):
        if len(g) < min_size:
            continue
        pos, neg = g[g["y_true"] == 1], g[g["y_true"] == 0]
        rows.append(
            {
                "feature": group_col,
                "group": name,
                "n": len(g),
                "share_positive": round(g["y_true"].mean(), 3),
                "miss_rate": round((pos["y_pred"] == 0).mean(), 3) if len(pos) else None,
                "false_alarm_rate": round((neg["y_pred"] == 1).mean(), 3) if len(neg) else None,
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    for path in (RESULTS, DATA_DIR / "train.csv", DATA_DIR / "val.csv"):
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run split.py and train.py first.")
    best = pick_best(pd.read_csv(RESULTS))
    X_train, y_train = split_xy(pd.read_csv(DATA_DIR / "train.csv"))
    X_val, y_val = split_xy(pd.read_csv(DATA_DIR / "val.csv"))
    pipe = Pipeline([("prep", build_preprocessor()), ("model", get_models()[best])])
    pipe.fit(X_train, y_train)
    frame = X_val.copy()
    frame["y_true"] = y_val.to_numpy()
    frame["y_pred"] = pipe.predict(X_val)
    parts = [group_error_rates(frame, col) for col in GROUP_COLS]
    report = pd.concat(parts, ignore_index=True)
    OUT.parent.mkdir(exist_ok=True)
    report.to_csv(OUT, index=False)
    print(f"Model: {best}")
    print(report.to_string(index=False))
