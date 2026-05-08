from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _ensure_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for c in columns:
        if c not in df.columns:
            df[c] = None
    return df[columns]


def build_demo_model(*, model_kind: str = "combined", months: int = 6) -> Path:
    stem = f"recent_{months}m_{model_kind}"
    meta_path = ARTIFACTS / f"model_{stem}.meta.json"
    examples_path = ARTIFACTS / f"examples_{stem}.test.json"
    out_path = ARTIFACTS / f"model_{stem}.joblib"

    meta = _read_json(meta_path)
    raw_cols: list[str] = list(meta.get("raw_columns") or [])
    if not raw_cols:
        raise ValueError(f"Missing raw_columns in {meta_path}")

    examples = _read_json(examples_path)
    if not examples:
        raise ValueError(f"No examples in {examples_path}")

    X = pd.DataFrame([e["x"] for e in examples])
    y = pd.Series([float(e["y_true"]) for e in examples], name="y")
    X = _ensure_columns(X, raw_cols).copy()

    numeric_cols = [
        "district_lon",
        "district_lat",
        "no_rooms",
        "m",
        "map_lon",
        "map_lat",
        "param_build_year",
        "param_building_floors_num",
        "param_remote_services",
        "param_rooms_num",
        "_days",
    ]
    numeric_cols = [c for c in numeric_cols if c in raw_cols]
    text_col = "__text__" if "__text__" in raw_cols else None

    cat_cols = [c for c in raw_cols if c not in set(numeric_cols) | ({text_col} if text_col else set())]

    for c in numeric_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    if text_col:
        X[text_col] = X[text_col].fillna("").astype(str)

    numeric_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_pipe, numeric_cols))
    if cat_cols:
        transformers.append(("cat", cat_pipe, cat_cols))
    if text_col:
        transformers.append(
            (
                "txt",
                TfidfVectorizer(max_features=4000, ngram_range=(1, 2)),
                text_col,
            )
        )

    pre = ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.3)
    model = Pipeline(steps=[("pre", pre), ("reg", Ridge(alpha=1.0))])
    model.fit(X, y)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_path)
    return out_path


if __name__ == "__main__":
    p = build_demo_model(model_kind="combined", months=6)
    print(f"Wrote demo model: {p}")

