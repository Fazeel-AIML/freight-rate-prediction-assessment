# Freight Rate Prediction Challenge

Simple scikit-learn solution for the Spotter freight rate assessment.

## Setup

```bash
python -m venv fr_venv
fr_venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Run

```bash
python src/train_model.py
python src/predict.py
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
python generate_december_chart.py
python generate_report_docx.py
```

## Outputs

- `models/model.joblib` — trained pipeline
- `models/metrics.json` — October holdout MAE/RMSE
- `validation_predictions.csv` — 12,000 submission predictions
- `data/december_chart_inputs.csv` — completed December daily predictions
- `scorer_results/candidate_december.png` — December chart for the report
- `report.docx` — submission report

## Approach

- **Split:** Jan–Sep 2025 train, Oct 2025 validation
- **Model:** `HistGradientBoostingRegressor`
- **Missing values:** median imputation for `weight` and `market_index`
- **December rows:** city coordinates from training medians; daily market signals from December validation medians