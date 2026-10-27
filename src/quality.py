from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw_adult.csv"
OUT = ROOT / "reports" / "data_quality.txt"


def build_report(df, target="class"):
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found. Columns: {list(df.columns)}")
    lines = [
        f"Shape: {df.shape}",
        f"Duplicate rows: {int(df.duplicated().sum())}",
        "",
        "Missing values per column:",
        df.isna().sum().to_string(),
        "",
        f"Class balance ({target}):",
        df[target].value_counts(normalize=True).round(3).to_string(),
        "",
        "Categorical columns (unique values):",
    ]
    for col in [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]:
        values = sorted(df[col].dropna().astype(str).unique())
        lines.append(f"- {col}: {len(values)} unique -> {values[:20]}")
    return "\n".join(lines)


if __name__ == "__main__":
    if not RAW.exists():
        raise FileNotFoundError(f"{RAW} not found. Run: python src\\data.py")
    report = build_report(pd.read_csv(RAW))
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(report, encoding="utf-8")
    print(report)

