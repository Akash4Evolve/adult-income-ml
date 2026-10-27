import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "class"
POSITIVE = ">50K"
# fnlwgt and education are dropped (see reports/decisions.md): they are simply
# not listed below, and the ColumnTransformer drops unlisted columns.
NUMERIC = ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
CATEGORICAL = [
    "workclass",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
SENSITIVE = ["race", "sex"]


def split_xy(df):
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found.")
    y = (df[TARGET] == POSITIVE).astype(int)
    X = df.drop(columns=[TARGET])
    return X, y


def build_preprocessor(include_sensitive=False):
    cat_cols = [c for c in CATEGORICAL if include_sensitive or c not in SENSITIVE]
    num_pipe = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    cat_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [("num", num_pipe, NUMERIC), ("cat", cat_pipe, cat_cols)],
        remainder="drop",
    )
