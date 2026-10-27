from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW = DATA_DIR / "raw_adult.csv"


def make_splits(df, target="class", seed=42):
    """Stratified 60/20/20 train/val/test split."""
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found.")
    train, temp = train_test_split(
        df, test_size=0.4, stratify=df[target], random_state=seed
    )
    val, test = train_test_split(
        temp, test_size=0.5, stratify=temp[target], random_state=seed
    )
    return train, val, test


if __name__ == "__main__":
    if not RAW.exists():
        raise FileNotFoundError(f"{RAW} not found. Run: python src\\data.py")
    df = pd.read_csv(RAW).drop_duplicates().reset_index(drop=True)
    train, val, test = make_splits(df)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        part.to_csv(DATA_DIR / f"{name}.csv", index=False)
        print(name, part.shape, part["class"].value_counts(normalize=True).round(3).to_dict())
