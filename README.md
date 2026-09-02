# Electricity Load Forecasting

Machine-learning research project for short-term electricity load forecasting.

## Project overview

This repository contains the code, notebooks, figures and experiment outputs for an electricity load forecasting research project. The project investigates whether historical electricity demand, calendar/time variables and weather-related features can improve forecasting accuracy while keeping the workflow reproducible.

> **Status:** Design Artefact / early implementation stage. The exact electricity field, weather station, sampling interval, forecasting horizon and final model set will be confirmed as the project progresses.

## Research question

**How effectively can machine-learning models forecast electricity load using historical demand, temporal features and weather-related variables?**

Supporting questions:
- Which features contribute most to forecast performance?
- How do different machine-learning models compare under the same time-series split?
- How well does the selected model generalise to later unseen time periods and seasons?

## Data sources

The project is planned to use:
- public historical electricity demand/load data;
- public weather observations matched to the electricity time series;
- derived calendar variables such as hour, weekday, weekend and holiday indicators.

The exact electricity region/field, weather station, sampling interval and source URLs will be recorded after verification.

Large or licence-restricted raw datasets are **not** committed to this repository. Place local raw files under `data/raw/`.

## Project structure

```text
load-forecasting/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_data_acquisition.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_baseline_model.ipynb
│   ├── 06_model_training.ipynb
│   ├── 07_model_evaluation.ipynb
│   ├── 08_shap_analysis.ipynb
│   └── 09_temporal_validation.ipynb
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── models.py
│   ├── evaluation.py
│   └── visualisation.py
├── figures/
├── results/
├── .gitignore
├── README.md
└── requirements.txt
```

## Planned workflow

1. **Data acquisition** — obtain public electricity load and weather datasets.
2. **Data cleaning** — handle missing values, detect abnormal values and align timestamps.
3. **Exploratory Data Analysis (EDA)** — inspect load curves, seasonal patterns and correlations.
4. **Feature engineering** — create lag features such as `t-1`, `t-24`, `t-168`, calendar variables, weather features and rolling statistics.
5. **Time-series split** — split chronologically into training (70%), validation (15%) and test (15%) sets; do not randomly shuffle.
6. **Scaling** — fit any scaler on the training set only, then transform validation/test data.
7. **Model training** — train candidate models and tune hyperparameters using the validation set.
8. **Evaluation** — evaluate final models on the test set using MAE, RMSE and sMAPE, plus prediction-vs-actual and error plots.
9. **Explainability** — use SHAP or suitable feature-importance methods to examine influential predictors.
10. **Temporal generalisation** — test performance on later unseen periods/seasons, for example training on earlier years and predicting a later year.

## Planned models

Initial candidates include:
- Persistence / naive baseline
- Linear Regression
- Random Forest Regressor
- XGBoost Regressor
- LightGBM Regressor (if retained after initial experiments)

The final model set will be based on literature review, data suitability and supervisor feedback.

## Evaluation metrics

- **MAE** — Mean Absolute Error
- **RMSE** — Root Mean Squared Error
- **sMAPE** — Symmetric Mean Absolute Percentage Error

## Reproducibility rules

- Split time-series data chronologically.
- Fit scalers and preprocessing parameters using training data only.
- Keep raw data unchanged; cleaned/merged outputs go to `data/processed/`.
- Save generated figures to `figures/` and experiment summaries to `results/`.
- Fix random seeds where relevant.

## How to run

Create and activate a Python environment, then install dependencies:

```bash
pip install -r requirements.txt
```

Run the notebooks in numerical order from `01_` to `09_` as implementation progresses.

## Current Design Artefact evidence

At this stage, this repository demonstrates:
- a reproducible project structure;
- separation between raw and processed data;
- a documented forecasting workflow;
- planned model comparison and evaluation metrics;
- placeholders for EDA, model results and explainability outputs.

As implementation progresses, genuine data-quality summaries, EDA figures, baseline results and model-comparison outputs will be added.
