import pandas as pd
import numpy as np
from pathlib import Path
from config import *

# ==================== Configuration ====================
INPUT_FILE = Path(f"./processed_data/aemo_weather_hourly_{YEARS[0]}_{YEARS[-1]}.csv")
OUTPUT_FILE = Path(f"./aemo_features_{YEARS[0]}_{YEARS[-1]}.csv")

# ==================== Read Data ====================
print("=" * 70)
print("Feature Engineering Pipeline")
print("=" * 70)
print("\nReading data...")

if not INPUT_FILE.exists():
    print(f"ERROR: File {INPUT_FILE} not found!")
    print("Please run process_aemo_weather_data.py first.")
    exit()

df = pd.read_csv(INPUT_FILE)
df['SETTLEMENTDATE'] = pd.to_datetime(df['SETTLEMENTDATE'])

# Rename SETTLEMENTDATE to HOUR for consistency with original feature engineering code
df = df.rename(columns={'SETTLEMENTDATE': 'HOUR'})

print(f"  Original data: {len(df):,} records")
print(f"  Date range: {df['HOUR'].min()} -> {df['HOUR'].max()}")
print(f"  Regions: {df['REGIONID'].unique().tolist()}")
print(f"  Features: {len(df.columns)} columns")

# ==================== Fill missing values in original data ====================
print("\nFilling missing values in original data with 0...")
before_fill = df.isnull().sum().sum()
df = df.fillna(0)
after_fill = df.isnull().sum().sum()
print(f"  Filled {before_fill:,} missing values with 0")
print(f"  No missing values remaining: {after_fill == 0}")

# ==================== Process by Region ====================
all_regions = []

for region in df['REGIONID'].unique():
    print(f"\nProcessing region: {region}")
    
    region_df = df[df['REGIONID'] == region].copy()
    region_df = region_df.sort_values('HOUR').reset_index(drop=True)
    
    # ==================== 1. Time Features ====================
    print("  Generating time features...")
    region_df['HOUR_OF_DAY'] = region_df['HOUR'].dt.hour
    region_df['DAY_OF_WEEK'] = region_df['HOUR'].dt.dayofweek
    region_df['MONTH'] = region_df['HOUR'].dt.month
    region_df['DAY_OF_YEAR'] = region_df['HOUR'].dt.dayofyear
    region_df['IS_WEEKEND'] = (region_df['DAY_OF_WEEK'] >= 5).astype(int)
    region_df['IS_WORKING_HOUR'] = ((region_df['HOUR_OF_DAY'] >= 7) & (region_df['HOUR_OF_DAY'] <= 19)).astype(int)
    
    region_df['HOUR_SIN'] = np.sin(2 * np.pi * region_df['HOUR_OF_DAY'] / 24)
    region_df['HOUR_COS'] = np.cos(2 * np.pi * region_df['HOUR_OF_DAY'] / 24)
    region_df['MONTH_SIN'] = np.sin(2 * np.pi * region_df['MONTH'] / 12)
    region_df['MONTH_COS'] = np.cos(2 * np.pi * region_df['MONTH'] / 12)
    
    # ==================== 2. Lag Features ====================
    print("  Generating lag features...")
    
    lag_steps = [1, 2, 3, 6, 12, 24, 48, 168]
    
    for lag in lag_steps:
        region_df[f'LOAD_LAG_{lag}'] = region_df['TOTALDEMAND'].shift(lag)
        region_df[f'PRICE_LAG_{lag}'] = region_df['RRP'].shift(lag)
    
    # ==================== 3. Rolling Statistics ====================
    print("  Generating rolling statistics...")
    
    region_df['LOAD_MEAN_12H'] = region_df['TOTALDEMAND'].shift(1).rolling(window=12, min_periods=1).mean()
    region_df['LOAD_MEAN_24H'] = region_df['TOTALDEMAND'].shift(1).rolling(window=24, min_periods=1).mean()
    region_df['LOAD_MEAN_48H'] = region_df['TOTALDEMAND'].shift(1).rolling(window=48, min_periods=1).mean()
    region_df['LOAD_MEAN_168H'] = region_df['TOTALDEMAND'].shift(1).rolling(window=168, min_periods=1).mean()
    
    region_df['LOAD_STD_24H'] = region_df['TOTALDEMAND'].shift(1).rolling(window=24, min_periods=1).std()
    region_df['LOAD_MIN_24H'] = region_df['TOTALDEMAND'].shift(1).rolling(window=24, min_periods=1).min()
    region_df['LOAD_MAX_24H'] = region_df['TOTALDEMAND'].shift(1).rolling(window=24, min_periods=1).max()
    
    region_df['PRICE_MEAN_24H'] = region_df['RRP'].shift(1).rolling(window=24, min_periods=1).mean()
    region_df['PRICE_STD_24H'] = region_df['RRP'].shift(1).rolling(window=24, min_periods=1).std()
    
    # ==================== 4. Change Features ====================
    print("  Generating change features...")
    
    region_df['LOAD_CHANGE_1H'] = region_df['LOAD_LAG_1'] - region_df['LOAD_LAG_2']
    region_df['LOAD_CHANGE_24H'] = region_df['LOAD_LAG_1'] - region_df['LOAD_LAG_24']
    region_df['LOAD_CHANGE_168H'] = region_df['LOAD_LAG_1'] - region_df['LOAD_LAG_168']
    region_df['LOAD_PCT_CHANGE_24H'] = (region_df['LOAD_LAG_1'] - region_df['LOAD_LAG_24']) / (region_df['LOAD_LAG_24'] + 1e-6) * 100
    
    # ==================== 5. Interaction Features ====================
    # print("  Generating interaction features...")
    
    # region_df['LOAD_PRICE_LAG_1'] = region_df['LOAD_LAG_1'] * region_df['PRICE_LAG_1']
    # region_df['LOAD_PRICE_LAG_24'] = region_df['LOAD_LAG_24'] * region_df['PRICE_LAG_24']
    # region_df['LOAD_PRICE_RATIO_LAG_1'] = region_df['LOAD_LAG_1'] / (region_df['PRICE_LAG_1'] + 1e-6)
    
    # ==================== 6. Drop first 168 rows (due to max lag of 168) ====================
    print("  Dropping first 168 rows (max lag = 168 hours)...")
    before_len = len(region_df)
    region_df = region_df.iloc[168:].reset_index(drop=True)
    after_len = len(region_df)
    print(f"  Removed 168 rows, remaining {after_len:,} records")
    
    all_regions.append(region_df)

# ==================== Merge All Regions ====================
print("\nMerging all regions...")
final_df = pd.concat(all_regions, ignore_index=True)

# ==================== Save ====================
print(f"\nSaving to: {OUTPUT_FILE}")
final_df.to_csv(OUTPUT_FILE, index=False)

print(f"\n" + "=" * 70)
print("Feature Engineering Complete!")
print("=" * 70)
print(f"\n  Output file: {OUTPUT_FILE}")
print(f"  Total records: {len(final_df):,}")
print(f"  Total features: {len(final_df.columns)}")

print("\nAll feature columns:")
for i, col in enumerate(final_df.columns, 1):
    print(f"  {i:2d}. {col}")

print("\nData preview (first 5 rows):")
print(final_df.head())

print("\nMissing value check:")
missing = final_df.isnull().sum()
if missing.sum() > 0:
    print(missing[missing > 0])
else:
    print("  No missing values found!")

print("\nFeature correlation diagnostics:")
numeric_cols = final_df.select_dtypes(include=[np.number]).columns
correlations = final_df[numeric_cols].corr()['TOTALDEMAND'].sort_values(ascending=False)

print("\nTop 10 features with highest correlation to TOTALDEMAND:")
print(correlations.head(10))

print("\nTop 10 features with lowest correlation to TOTALDEMAND:")
print(correlations.tail(10))

print("\n" + "=" * 70)
print("Feature engineering complete! Data ready for modeling.")
print("=" * 70)
