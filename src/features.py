from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

NUMERIC_FEATURES = [
    "distance",
    "weight",
    "market_index",
    "quote_signal",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "month",
    "day_of_week",
]
CATEGORICAL_FEATURES = ["equipment"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def add_date_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    dates = pd.to_datetime(result["date"])
    result["month"] = dates.dt.month
    result["day_of_week"] = dates.dt.dayofweek
    return result


def prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = add_date_features(frame)
    return enriched[FEATURE_COLUMNS]


def city_coordinates(train: pd.DataFrame) -> dict[str, dict[str, float]]:
    pickup_coords = (
        train.groupby("pickup")[["pickup_lat", "pickup_lon"]]
        .median()
        .rename(columns={"pickup_lat": "lat", "pickup_lon": "lon"})
    )
    delivery_coords = (
        train.groupby("delivery")[["delivery_lat", "delivery_lon"]]
        .median()
        .rename(columns={"delivery_lat": "lat", "delivery_lon": "lon"})
    )

    coordinates: dict[str, dict[str, float]] = {}
    for city, row in pickup_coords.iterrows():
        coordinates[city] = {"lat": float(row["lat"]), "lon": float(row["lon"])}
    for city, row in delivery_coords.iterrows():
        coordinates[city] = {"lat": float(row["lat"]), "lon": float(row["lon"])}
    return coordinates


def october_medians(train: pd.DataFrame) -> dict[str, float]:
    october = train[pd.to_datetime(train["date"]).dt.to_period("M") == "2025-10"]
    return {
        "weight": float(october["weight"].median()),
        "market_index": float(october["market_index"].median()),
        "quote_signal": float(october["quote_signal"].median()),
    }


def december_market_by_date(validation_path: Path) -> pd.DataFrame:
    validation = pd.read_csv(validation_path)
    validation["date"] = pd.to_datetime(validation["date"])
    december = validation[validation["date"].dt.month == 12]
    daily = december.groupby("date", as_index=False)[["market_index", "quote_signal"]].median()
    return daily


def enrich_december_rows(
    frame: pd.DataFrame,
    coordinates: dict[str, dict[str, float]],
    medians: dict[str, float],
    daily_market: pd.DataFrame | None = None,
) -> pd.DataFrame:
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"])
    pickup = coordinates[result["pickup"].iloc[0]]
    delivery = coordinates[result["delivery"].iloc[0]]

    result["pickup_lat"] = pickup["lat"]
    result["pickup_lon"] = pickup["lon"]
    result["delivery_lat"] = delivery["lat"]
    result["delivery_lon"] = delivery["lon"]
    result["weight"] = result["weight"].fillna(medians["weight"])

    if daily_market is not None:
        result = result.merge(daily_market, on="date", how="left")
        result["market_index"] = result["market_index"].fillna(medians["market_index"])
        result["quote_signal"] = result["quote_signal"].fillna(medians["quote_signal"])
    else:
        result["market_index"] = medians["market_index"]
        result["quote_signal"] = medians["quote_signal"]

    result["date"] = result["date"].dt.strftime("%Y-%m-%d")
    return result


def save_metadata(path: Path, medians: dict[str, float], coordinates: dict[str, dict[str, float]]) -> None:
    payload = {
        "october_medians": medians,
        "city_coordinates": coordinates,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_metadata(path: Path) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["october_medians"], payload["city_coordinates"]
