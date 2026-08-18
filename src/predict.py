from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from features import december_market_by_date, enrich_december_rows, load_metadata, prepare_features

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_PATH = ROOT / "models" / "model.joblib"
METADATA_PATH = ROOT / "models" / "metadata.json"
VALIDATION_PATH = DATA_DIR / "validation.csv"
TEMPLATE_PATH = DATA_DIR / "validation_predictions_template.csv"
DECEMBER_PATH = DATA_DIR / "december_chart_inputs.csv"
OUTPUT_PREDICTIONS = ROOT / "validation_predictions.csv"


def predict_rates(model, frame: pd.DataFrame) -> pd.Series:
    features = prepare_features(frame)
    rates = model.predict(features)
    return pd.Series(rates, index=frame.index).clip(lower=1.0)


def main() -> None:
    model = joblib.load(MODEL_PATH)
    medians, coordinates = load_metadata(METADATA_PATH)

    validation = pd.read_csv(VALIDATION_PATH)
    validation_rates = predict_rates(model, validation)

    template = pd.read_csv(TEMPLATE_PATH)
    prediction_map = dict(zip(validation["load_id"], validation_rates, strict=True))
    predictions = template.copy()
    predictions["predicted_rate"] = predictions["load_id"].map(prediction_map)
    predictions.to_csv(OUTPUT_PREDICTIONS, index=False)

    december = pd.read_csv(DECEMBER_PATH)
    daily_market = december_market_by_date(VALIDATION_PATH)
    december_ready = enrich_december_rows(december, coordinates, medians, daily_market)
    december["predicted_rate"] = predict_rates(model, december_ready)
    december.to_csv(DECEMBER_PATH, index=False)

    print(f"Wrote {len(predictions):,} rows to {OUTPUT_PREDICTIONS}")
    print(f"Updated December predictions in {DECEMBER_PATH}")


if __name__ == "__main__":
    main()
