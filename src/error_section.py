from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "reports" / "error_analysis.csv"
README = ROOT / "README.md"
HEADER = "## Error analysis (validation split)"
REQUIRED = ["feature", "group", "n", "share_positive", "miss_rate", "false_alarm_rate"]


def to_table(df):
    lines = [
        "| Group | n | Share >50K | Miss rate | False-alarm rate |",
        "|---|---|---|---|---|",
    ]
    for r in df.itertuples(index=False):
        lines.append(
            f"| {r.group} | {r.n} | {r.share_positive} | {r.miss_rate} | {r.false_alarm_rate} |"
        )
    return "\n".join(lines)


def build_section(report):
    for col in REQUIRED:
        if col not in report.columns:
            raise ValueError(f"Column '{col}' missing from error analysis.")
    sex = report[report["feature"] == "sex"]
    occ = (
        report[report["feature"] == "occupation"]
        .sort_values("miss_rate", ascending=False)
        .head(3)
    )
    parts = [
        HEADER,
        "Miss rate is the share of true >50K people predicted <=50K; false-alarm rate is the share of true <=50K people predicted >50K. Groups under 100 rows are omitted.",
        "",
        "**By sex**",
        to_table(sex),
        "",
        "**Occupations with the highest miss rate**",
        to_table(occ),
        "",
    ]
    return "\n".join(parts)


if __name__ == "__main__":
    if not CSV.exists():
        raise FileNotFoundError(f"{CSV} not found. Run: python -m src.error_analysis")
    text = README.read_text(encoding="utf-8")
    if HEADER in text:
        raise SystemExit("Error analysis section already in README.")
    section = build_section(pd.read_csv(CSV))
    README.write_text(text.rstrip() + "\n\n" + section, encoding="utf-8")
    print(section)
