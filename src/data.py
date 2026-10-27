from pathlib import Path

from sklearn.datasets import fetch_openml

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def load_raw():
    DATA_DIR.mkdir(exist_ok=True)
    try:
        ds = fetch_openml("adult", version=2, as_frame=True)
    except Exception as exc:
        raise RuntimeError(f"Could not download the Adult dataset: {exc}") from exc
    df = ds.frame
    df.to_csv(DATA_DIR / "raw_adult.csv", index=False)
    return df


if __name__ == "__main__":
    print(load_raw().shape)
