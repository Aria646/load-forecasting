# Load Forecasting Using Machine Learning

## Project Overview

This project investigates machine learning-based load forecasting models using Australian electricity demand data from AEMO (Australian Energy Market Operator). The focus is on short-term forecasting (1 hour to 1 week ahead).

## Research Question

How accurately can machine learning and deep learning models forecast short-term electricity demand using historical load and weather data?

## Data Source

- **Source**: AEMO (Australian Energy Market Operator)
- **Tables**: DISPATCHREGIONSUM (load), DISPATCHPRICE (price)
- **Period**: 2021-2023
- **Regions**: NSW1, QLD1, SA1, TAS1, VIC1
- **Resolution**: Hourly (aggregated from 5/30-minute data)

## Files

### step1_download_build_aemo_data.py
Downloads AEMO data and prepares hourly dataset.

**What it does:**
- Downloads load and price data for 2021-2023
- Merges data by settlement date and region
- Aggregates to hourly resolution

**Output:** `aemo_merged_hourly_2021_2023.csv`

### step2_eda_aemo_load_data.py
Performs exploratory data analysis on the merged dataset.

**Generated plots:**
- Load time series (2021-2023)
- 24-hour average load curve
- Average load by day of week
- Average load by month and season
- Load distribution by region
- Missing data distribution

**Output:** `results/aemo_eda_combined.png`, `results/eda_summary_stats.csv`

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
python step1_download_build_aemo_data.py
```

### Step 2: Exploratory Data Analysis
```bash
python step2_eda_aemo_load_data.py
```

## Methodology

1. Literature review on load forecasting techniques
2. Data acquisition and preprocessing
3. Exploratory data analysis and feature engineering
4. Model development (machine learning and deep learning)
5. Training and validation
6. Performance evaluation and comparison

## Project Objectives

1. Conduct literature review on load forecasting techniques
2. Analyse electricity demand and weather datasets
3. Investigate effects of weather, temporal factors, and historical patterns
4. Compare forecasting models
5. Identify key variables influencing accuracy
6. Assess suitability for smart grid applications

## Expected Outcomes

- Machine learning-based load forecasting framework
- Comparative evaluation of forecasting algorithms
- Insights into demand-influencing factors
- Recommendations for smart grid implementation

## License

Academic research purposes only.
