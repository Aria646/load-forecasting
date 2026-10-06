# Load Forecasting Using Machine Learning

## Project Overview

This project investigates machine learning-based load forecasting models using Australian electricity demand data from AEMO (Australian Energy Market Operator). The focus is on short-term forecasting (1 hour to 1 week ahead). The project also investigates cross-state generalisation and transfer learning, where models trained on one state are adapted to other states.

## Research Question

How accurately can machine learning and deep learning models forecast short-term electricity demand using historical load and weather data, and how well do they generalise across states?

## Data Source

- **Source**: AEMO (Australian Energy Market Operator)
- **Tables**: DISPATCHREGIONSUM (load), DISPATCHPRICE (price)
- **Weather API**: Weather.com API
- **Period**: Configurable via `config.py` (default: 2015-2019)
- **Regions**: NSW1, QLD1, SA1, TAS1, VIC1
- **Default Training Region**: Configurable via `config.py` (default: NSW1)
- **Resolution**: Hourly (aggregated from 5-minute AEMO data)

## Configuration

All configurable parameters are defined in `config.py`:

```python
# Time range for data download and training
YEARS = [2021, 2022, 2023]

# Default region for model training and EDA analysis
SELECT_REGION = "NSW1"  # Options: 'NSW1', 'QLD1', 'SA1', 'TAS1', 'VIC1'
```

- **YEARS**: Specifies which years of data to download from AEMO and weather API, and which years to use for training/testing (all years except the last are used for training, the last year is used for testing)
- **SELECT_REGION**: Specifies the default region used for EDA detailed analysis and model training (default NSW1)

## Project Structure

```
|-- config.py                            # Configuration (years, default region)
|-- step1_download_aemo_weather_data.py  # Download AEMO + weather data
|-- step2_process_aemo_weather_data.py   # Process and merge datasets
|-- step3_eda_aemo_load_data.py          # Exploratory data analysis
|-- step4_feature_engineering.py         # Feature engineering pipeline
|-- step5-9_model_training.py            # Model training and comparison
|-- step10_validate_regions.py           # Cross-region validation
|-- step11_transfer_learning.py          # Transfer learning to target states
|-- processed_data/                      # Processed hourly dataset
|-- weather_data/raw/                    # Raw weather data
|-- aemo_data/                           # Raw AEMO data
|-- results/                             # EDA and model results
|   |-- models/                          # Saved models
|   |-- validation/                      # Validation results
|   |-- transfer/                        # Transfer learning results
|   |-- transfer_models/                 # Fine-tuned transfer models
|-- requirements.txt                     # Dependencies
```

## File Descriptions

### step1_download_aemo_weather_data.py
Downloads AEMO and weather data for the years specified in `config.py`.

**What it does:**
- Downloads DISPATCHREGIONSUM (load) and DISPATCHPRICE (price) tables for the configured years
- Fetches weather data from Weather.com API for each region
- Saves raw data to `aemo_data/` and `weather_data/raw/`

**Output:** Raw CSV files in `aemo_data/` and `weather_data/raw/`

### step2_process_aemo_weather_data.py
Processes and merges AEMO and weather data into a unified hourly dataset.

**What it does:**
- Loads AEMO 5-minute data, cleans duplicates, merges demand & price
- Aggregates AEMO data to hourly resolution
- Loads weather data and aggregates to hourly resolution
- Merges AEMO and weather data by region and settlement date
- Filters data to the years specified in `config.py`

**Output:** `processed_data/aemo_weather_hourly_{YEARS[0]}_{YEARS[-1]}.csv`

### step3_eda_aemo_load_data.py
Performs comprehensive exploratory data analysis on the merged dataset, focusing on the region specified in `config.py` for detailed analysis.

**Generated plots:**
- **eda_01_time_series_overview.png**: Load, temperature, humidity, precipitation time series
- **eda_02_load_patterns.png**: 24-hour load curve, weekly/monthly/seasonal patterns
- **eda_03_weather_patterns.png**: Weather distributions and correlations
- **eda_04_temperature_load_relationship.png**: Temperature-load U-shape analysis (for `SELECT_REGION`)
- **eda_05_weather_load_correlations.png**: Correlation heatmaps across regions
- **eda_06_missing_data.png**: Missing data distribution

### step4_feature_engineering.py
Generates comprehensive features for forecasting models.

**Features generated:**
- **Time features**: Hour of day, day of week, month, cyclic encoding
- **Lag features**: Load and price lags (1, 2, 3, 6, 12, 24, 48, 168 hours)
- **Rolling statistics**: Mean, std, min, max over 12h, 24h, 48h, 168h windows
- **Change features**: Hourly, daily, weekly differences and percentage changes

**Output:** `aemo_features_{YEARS[0]}_{YEARS[-1]}.csv`

### step5-9_model_training.py
Trains and compares multiple forecasting models for short-term load forecasting on the region specified in `config.py`.

**Models:**
- **XGBoost** - Gradient boosting baseline
- **LightGBM** - Gradient boosting baseline
- **LSTM** - Long Short-Term Memory
- **GRU** - Gated Recurrent Unit
- **BiLSTM** - Bidirectional LSTM
- **BiGRU** - Bidirectional GRU
- **ANN** - Feedforward neural network

**Prediction Horizons:** 1h, 24h, 168h (1 week)

**Training/Test Split:**
- Training: All years except the last year in `YEARS`
- Test: The last year in `YEARS`

**Metrics:** MAE, MAPE, RMSE, R²

**Output:**
- `results/models/*.pkl` - Saved models
- `results/performance_comparison.png` - Performance bar charts
- `results/mape_heatmap.png` - MAPE heatmap by model and horizon
- `results/metrics_summary.csv` - Performance metrics
- `results/model_report.txt` - Detailed evaluation report

### step10_validate_regions.py
Validates model generalization across different regions.

**What it does:**
- Evaluates saved models on unseen regions (QLD1, SA1, TAS1, VIC1)
- Uses the test year specified in `config.py` (the last year in `YEARS`)
- Computes MAE, MAPE, RMSE, R² for each region and horizon
- Generates validation reports and visualization

**Output:**
- `results/validation/validation_metrics.csv` - Validation metrics
- `results/validation/validation_report.txt` - Summary report
- `results/validation/validation_mape_heatmap.png` - Cross-region heatmap
- `results/validation/validation_*.png` - Per-region performance plots

### step11_transfer_learning.py
Applies transfer learning to adapt NSW-trained models to target states (QLD1, SA1, TAS1, VIC1), and evaluates whether fine-tuning can recover performance lost in direct transfer.

**What it does:**
- Loads models trained on the source region (default NSW1) from `results/models/`
- Adapts them to each target state using four strategies:
  - **scaler**: Refit `target_scaler` on target-state data only; model parameters unchanged
  - **partial**: Freeze early layers (RNN layer for recurrent models; fc1/fc2 for ANN; existing trees for tree models) and retrain output layers on target data
  - **full**: Fine-tune all parameters on target data with a small learning rate
  - **fewshot**: Run partial and full fine-tuning with varying amounts of target data (e.g., 7, 30, 90, 180 days) to study data requirements
- Evaluates on the same test set (last year in `YEARS`) as `step10` for fair comparison
- Saves fine-tuned models to a separate directory (`results/transfer_models/`) to keep them distinct from source models

**Models supported:** XGBoost, LightGBM, ANN, LSTM, GRU, BiLSTM, BiGRU

**Prediction Horizons:** 1h, 24h, 168h

**Metrics:** MAE, MAPE, RMSE, R²

**Output:**
- `results/transfer/transfer_results.csv` - Full results per (region, model, horizon, strategy)
- `results/transfer/transfer_summary_mape.csv` - MAPE pivot table
- `results/transfer/transfer_heatmap_{region}.png` - MAPE heatmap per target state
- `results/transfer/fewshot_curve_{region}.png` - Few-shot learning curves
- `results/transfer_models/*.pkl` - Fine-tuned transfer models

## Installation

```bash
# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Step 1: Download Data
```bash
python step1_download_aemo_weather_data.py
```

### Step 2: Process Data
```bash
python step2_process_aemo_weather_data.py
```

### Step 3: Exploratory Data Analysis
```bash
python step3_eda_aemo_load_data.py
```

### Step 4: Feature Engineering
```bash
python step4_feature_engineering.py
```

### Step 5-9: Model Training
```bash
# Train all models with default parameters (uses SELECT_REGION from config.py)
python step5-9_model_training.py

# Custom training with specific region
python step5-9_model_training.py --region NSW1 --horizons 1,24,168 --models xgboost,lightgbm,lstm --epochs 150
```

### Step 10: Cross-Region Validation
```bash
# Validate on all regions using models trained on SELECT_REGION
python step10_validate_regions.py

# Validate specific regions
python step10_validate_regions.py --regions NSW1,QLD1,VIC1
```

### Step 11: Transfer Learning
```bash
# Full transfer learning (all models, all target states, all strategies)
python step11_transfer_learning.py

# Only run scaler, partial, and full strategies
python step11_transfer_learning.py --strategies scaler,partial,full

# Run on specific models and target regions
python step11_transfer_learning.py --models xgboost,lightgbm,lstm --target-regions QLD1,TAS1

# Few-shot learning (varying target data size)
python step11_transfer_learning.py --strategies fewshot --fewshot-days 7,30,90,180

# Skip saving fine-tuned models (evaluation only, faster)
python step11_transfer_learning.py --strategies scaler,partial,full --no-save-models

# Custom save directory for transfer models
python step11_transfer_learning.py --transfer-model-dir ./results/transfer_models_pilot

# Extended fine-tuning for harder target states (more epochs, smaller learning rate)
python step11_transfer_learning.py --models lstm --target-regions SA1 --strategies scaler,partial,full --finetune-epochs 50 --finetune-lr 1e-4
```

## Methodology

1. Literature review on load forecasting techniques
2. Data acquisition and preprocessing
3. Exploratory data analysis and feature engineering
4. Model development (machine learning and deep learning)
5. Training and validation (time-based split)
6. Performance evaluation and comparison
7. Cross-region generalization testing
8. Transfer learning to adapt source-region models to target states

## Key Findings

- Temperature exhibits a **U-shaped relationship** with load: high demand during both extreme cold and extreme heat
- **Seasonal correlations** vary significantly (positive in summer for cooling, negative in winter for heating)
- **RNN-based models** (LSTM, BiLSTM) generally outperform tree-based models for longer horizons (24h, 168h)
- **Cyclic encoding** of temporal features improves model performance
- **Direct cross-state transfer** (without adaptation) yields poor accuracy: MAPE increases by roughly 5–10x compared to in-region performance, mainly due to target-scale mismatch and climate/demand pattern differences
- **Fine-tuning** substantially recovers performance: partial and full strategies reduce MAPE to within 1–3 percentage points of in-region baselines across all model families
- **Scaler-only adaptation** already recovers much of the gap for target states with similar demand scale, but is insufficient for states with large scale differences (e.g., TAS1)
- **Tree-based models** (XGBoost, LightGBM) tend to generalise better after fine-tuning than feedforward neural networks, while RNNs require full fine-tuning to reach comparable performance

## Project Objectives

1. Conduct literature review on load forecasting techniques
2. Analyse electricity demand and weather datasets
3. Investigate effects of weather, temporal factors, and historical patterns
4. Compare forecasting models
5. Identify key variables influencing accuracy
6. Assess suitability for smart grid applications
7. Evaluate cross-state generalisation and transfer learning potential

## Expected Outcomes

- Machine learning-based load forecasting framework
- Comparative evaluation of forecasting algorithms
- Insights into demand-influencing factors
- Recommendations for smart grid implementation
- Characterisation of cross-state generalisation limits and effectiveness of transfer learning strategies

## License

Academic research purposes only.
