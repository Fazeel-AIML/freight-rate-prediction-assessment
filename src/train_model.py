from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from features import (
    FEATURE_COLUMNS,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    city_coordinates,
    october_medians,
    prepare_features,
    save_metadata,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
TRAIN_PATH = DATA_DIR / "train_test.csv"
MODEL_PATH = MODEL_DIR / "model.joblib"
METADATA_PATH = MODEL_DIR / "metadata.json"
METRICS_PATH = MODEL_DIR / "metrics.json"

TRAIN_END = "2025-09-30"
VALID_START = "2025-10-01"
VALID_END = "2025-10-31"


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
            (
                "equipment",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    model = HistGradientBoostingRegressor(
        max_iter=200,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def split_data(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.to_datetime(frame["date"])
    train = frame[dates <= TRAIN_END].copy()
    valid = frame[(dates >= VALID_START) & (dates <= VALID_END)].copy()
    return train, valid


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train_frame = pd.read_csv(TRAIN_PATH)
    train_split, valid_split = split_data(train_frame)

    pipeline = build_pipeline()
    x_train = prepare_features(train_split)
    y_train = train_split["posted_rate"]
    x_valid = prepare_features(valid_split)
    y_valid = valid_split["posted_rate"]

    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_valid)

    mae = float(mean_absolute_error(y_valid, predictions))
    rmse = float(np.sqrt(mean_squared_error(y_valid, predictions)))

    joblib.dump(pipeline, MODEL_PATH)
    save_metadata(METADATA_PATH, october_medians(train_frame), city_coordinates(train_frame))

    metrics = {
        "train_rows": int(len(train_split)),
        "validation_rows": int(len(valid_split)),
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "train_date_range": ["2025-01-01", TRAIN_END],
        "validation_date_range": [VALID_START, VALID_END],
        "model": "HistGradientBoostingRegressor",
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Saved model to {MODEL_PATH}")
    print(f"October holdout MAE:  ${mae:,.2f}")
    print(f"October holdout RMSE: ${rmse:,.2f}")


if __name__ == "__main__":
    main()
