import pandas as pd
from pathlib import Path
import numpy as np
from tqdm import tqdm
from config import *

# ==================== Configuration ====================

AEMO_DATA_DIR = Path("./aemo_data")
WEATHER_DATA_DIR = Path("./weather_data/raw")
OUTPUT_DIR = Path("./processed_data")
OUTPUT_DIR.mkdir(exist_ok=True)

# ==================== Load AEMO Data ====================

def load_aemo_data():
    """Load all AEMO CSV files"""
    print("=" * 70)
    print("Loading AEMO data...")
    print("=" * 70)
    
    region_files = list(AEMO_DATA_DIR.glob("*DISPATCHREGIONSUM*.csv"))
    price_files = list(AEMO_DATA_DIR.glob("*DISPATCHPRICE*.csv"))
    
    if not region_files or not price_files:
        print("ERROR: AEMO data files not found. Please run download script first.")
        return None, None
    
    print(f"Found {len(region_files)} DISPATCHREGIONSUM files")
    print(f"Found {len(price_files)} DISPATCHPRICE files")
    
    # Load region data
    print("\nLoading DISPATCHREGIONSUM files...")
    region_dfs = []
    for file in tqdm(region_files, desc="Loading region data"):
        try:
            df = pd.read_csv(file, skiprows=1, low_memory=False)
            region_dfs.append(df)
        except Exception as e:
            print(f"  Warning: Failed to load {file.name}: {e}")
    
    if not region_dfs:
        print("ERROR: No region data loaded")
        return None, None
    
    region_df = pd.concat(region_dfs, ignore_index=True)
    print(f"  Loaded {len(region_df):,} records (5-minute intervals)")
    
    # Load price data
    print("\nLoading DISPATCHPRICE files...")
    price_dfs = []
    for file in tqdm(price_files, desc="Loading price data"):
        try:
            df = pd.read_csv(file, skiprows=1, low_memory=False)
            price_dfs.append(df)
        except Exception as e:
            print(f"  Warning: Failed to load {file.name}: {e}")
    
    if not price_dfs:
        print("ERROR: No price data loaded")
        return None, None
    
    price_df = pd.concat(price_dfs, ignore_index=True)
    print(f"  Loaded {len(price_df):,} records (5-minute intervals)")
    
    return region_df, price_df

# ==================== Clean and Merge AEMO Data ====================

def clean_and_merge_aemo(region_df, price_df):
    """Clean and merge AEMO demand and price data at 5-minute level"""
    print("\n" + "=" * 70)
    print("Cleaning and merging AEMO data...")
    print("=" * 70)
    
    # Convert datetime
    print("Converting datetime columns...")
    region_df['SETTLEMENTDATE'] = pd.to_datetime(region_df['SETTLEMENTDATE'])
    price_df['SETTLEMENTDATE'] = pd.to_datetime(price_df['SETTLEMENTDATE'])
    
    # Keep only required columns
    region_cols = ['SETTLEMENTDATE', 'REGIONID', 'TOTALDEMAND']
    price_cols = ['SETTLEMENTDATE', 'REGIONID', 'RRP']
    
    # Check if columns exist
    for col in region_cols:
        if col not in region_df.columns:
            if col == 'TOTALDEMAND' and 'TOTALDEMAND' not in region_df.columns:
                possible_cols = [c for c in region_df.columns if 'DEMAND' in c.upper()]
                if possible_cols:
                    print(f"  Using '{possible_cols[0]}' instead of 'TOTALDEMAND'")
                    region_cols[region_cols.index(col)] = possible_cols[0]
    
    for col in price_cols:
        if col not in price_df.columns:
            if col == 'RRP' and 'RRP' not in price_df.columns:
                possible_cols = [c for c in price_df.columns if 'PRICE' in c.upper() or 'RRP' in c.upper()]
                if possible_cols:
                    print(f"  Using '{possible_cols[0]}' instead of 'RRP'")
                    price_cols[price_cols.index(col)] = possible_cols[0]
    
    region_df = region_df[region_cols]
    price_df = price_df[price_cols]
    
    # Rename columns for consistency
    region_df.columns = ['SETTLEMENTDATE', 'REGIONID', 'TOTALDEMAND']
    price_df.columns = ['SETTLEMENTDATE', 'REGIONID', 'RRP']
    
    # Remove duplicates
    print("Removing duplicates...")
    region_df = region_df.drop_duplicates(subset=['SETTLEMENTDATE', 'REGIONID'])
    price_df = price_df.drop_duplicates(subset=['SETTLEMENTDATE', 'REGIONID'])
    print(f"  Demand after dedup: {len(region_df):,} records")
    print(f"  Price after dedup: {len(price_df):,} records")
    
    # Merge demand and price at 5-minute level
    print("\nMerging demand and price...")
    aemo_5min = pd.merge(
        region_df,
        price_df,
        on=['SETTLEMENTDATE', 'REGIONID'],
        how='inner'
    )
    print(f"  Merged: {len(aemo_5min):,} records (5-minute intervals)")
    
    # Filter to specified years
    aemo_5min = aemo_5min[aemo_5min['SETTLEMENTDATE'].dt.year.isin(YEARS)]
    print(f"  Filtered to {YEARS}: {len(aemo_5min):,} records")
    
    return aemo_5min

# ==================== Aggregate AEMO to Hourly ====================

def aggregate_aemo_to_hourly(aemo_5min):
    """Aggregate 5-minute AEMO data to hourly level"""
    print("\n" + "=" * 70)
    print("Aggregating AEMO data to hourly...")
    print("=" * 70)
    
    # Create hourly floor
    aemo_5min['HOUR'] = aemo_5min['SETTLEMENTDATE'].dt.floor('h')
    
    # Aggregate by region and hour
    aemo_hourly = aemo_5min.groupby(['REGIONID', 'HOUR']).agg({
        'TOTALDEMAND': 'mean',   # Average demand over the hour
        'RRP': 'mean'            # Average price over the hour
    }).reset_index()
    
    # Rename HOUR to SETTLEMENTDATE
    aemo_hourly = aemo_hourly.rename(columns={'HOUR': 'SETTLEMENTDATE'})
    
    print(f"  Hourly aggregated: {len(aemo_hourly):,} records")
    print(f"  Date range: {aemo_hourly['SETTLEMENTDATE'].min()} to {aemo_hourly['SETTLEMENTDATE'].max()}")
    print(f"  Reduction: {len(aemo_5min):,} -> {len(aemo_hourly):,} records ({(1 - len(aemo_hourly)/len(aemo_5min))*100:.1f}% reduction)")
    
    return aemo_hourly

# ==================== Load and Merge Weather Data ====================

def load_and_merge_weather():
    """Load and merge weather data for all regions"""
    print("\n" + "=" * 70)
    print("Loading and merging weather data...")
    print("=" * 70)
    
    weather_files = list(WEATHER_DATA_DIR.glob("weather_*.csv"))
    
    if not weather_files:
        print("ERROR: Weather data files not found. Please run download script first.")
        return None
    
    # Filter out the summary file if it exists
    weather_files = [f for f in weather_files if "summary" not in f.name.lower()]
    
    print(f"Found {len(weather_files)} weather files")
    
    weather_dfs = []
    for file in tqdm(weather_files, desc="Loading weather data"):
        try:
            df = pd.read_csv(file)
            weather_dfs.append(df)
        except Exception as e:
            print(f"  Warning: Failed to load {file.name}: {e}")
    
    if not weather_dfs:
        print("ERROR: No weather data loaded")
        return None
    
    # Merge all regions into a single DataFrame
    all_weather = pd.concat(weather_dfs, ignore_index=True)
    print(f"  Loaded and merged: {len(all_weather):,} records")
    
    # Display records per region
    if 'REGIONID' in all_weather.columns:
        print("\n  Records per region:")
        region_counts = all_weather['REGIONID'].value_counts()
        for region, count in region_counts.items():
            print(f"    {region}: {count:,} records")
    
    return all_weather

# ==================== Clean and Aggregate Weather to Hourly ====================

def clean_and_aggregate_weather(weather_df):
    """Clean weather data and aggregate to hourly level"""
    print("\n" + "=" * 70)
    print("Cleaning and aggregating weather data to hourly...")
    print("=" * 70)
    
    # Convert datetime
    print("Converting datetime...")
    weather_df['DATETIME'] = pd.to_datetime(weather_df['DATETIME'])
    
    # Filter to specified years
    weather_df = weather_df[weather_df['DATETIME'].dt.year.isin(YEARS)]
    print(f"  Filtered to {YEARS}: {len(weather_df):,} records")
    
    # Create hourly floor
    print("Aggregating to hourly level...")
    weather_df['HOUR'] = weather_df['DATETIME'].dt.floor('h')
    
    # Aggregate by region and hour
    weather_hourly = weather_df.groupby(['REGIONID', 'HOUR']).agg({
        'TEMPERATURE': 'mean',
        'DEW_POINT': 'mean',
        'HUMIDITY': 'mean',
        'PRESSURE': 'mean',
        'WIND_SPEED': 'mean',
        'VISIBILITY': 'mean',
        'PRECIPITATION': 'max',  # Max indicates if there was any precipitation
    }).reset_index()
    
    # Rename HOUR to SETTLEMENTDATE for consistency
    weather_hourly = weather_hourly.rename(columns={'HOUR': 'SETTLEMENTDATE'})
    
    print(f"  Hourly aggregated: {len(weather_hourly):,} records")
    print(f"  Date range: {weather_hourly['SETTLEMENTDATE'].min()} to {weather_hourly['SETTLEMENTDATE'].max()}")
    
    return weather_hourly

# ==================== Merge AEMO and Weather Data ====================

def merge_aemo_weather(aemo_hourly, weather_hourly):
    """Merge hourly AEMO data with hourly weather data"""
    print("\n" + "=" * 70)
    print("Merging AEMO and weather data...")
    print("=" * 70)
    
    # Both datasets are now hourly, merge directly on REGIONID and SETTLEMENTDATE
    merged_df = pd.merge(
        aemo_hourly,
        weather_hourly,
        on=['REGIONID', 'SETTLEMENTDATE'],
        how='inner'
    )
    
    print(f"  Merged records: {len(merged_df):,}")
    
    # Sort by region and datetime
    merged_df = merged_df.sort_values(['REGIONID', 'SETTLEMENTDATE']).reset_index(drop=True)
    
    return merged_df

# ==================== Save Processed Data ====================

def save_processed_data(df):
    """Save processed data to CSV file"""
    print("\n" + "=" * 70)
    print("Saving processed data...")
    print("=" * 70)
    
    output_filename = OUTPUT_DIR / f"aemo_weather_hourly_{YEARS[0]}_{YEARS[-1]}.csv"
    df.to_csv(output_filename, index=False)
    
    print(f"  Output file: {output_filename}")
    print(f"    Size: {output_filename.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"    Records: {len(df):,}")
    print(f"    Features: {len(df.columns)}")
    print(f"    Regions: {df['REGIONID'].unique().tolist()}")
    print(f"    Date range: {df['SETTLEMENTDATE'].min()} to {df['SETTLEMENTDATE'].max()}")

# ==================== Main Execution ====================

def main():
    """Main function to process and merge all data"""
    print("=" * 70)
    print("AEMO & Weather Data Processing Pipeline")
    print("=" * 70)
    print(f"\nProcessing years: {YEARS}")
    print(f"Output directory: {OUTPUT_DIR.absolute()}")
    print()
    
    # Load AEMO data (5-minute)
    region_df, price_df = load_aemo_data()
    if region_df is None or price_df is None:
        print("ERROR: Failed to load AEMO data. Exiting.")
        return
    
    # Clean and merge AEMO data (5-minute)
    aemo_5min = clean_and_merge_aemo(region_df, price_df)
    if aemo_5min is None or len(aemo_5min) == 0:
        print("ERROR: AEMO data cleaning failed. Exiting.")
        return
    
    # Aggregate AEMO to hourly
    aemo_hourly = aggregate_aemo_to_hourly(aemo_5min)
    if aemo_hourly is None or len(aemo_hourly) == 0:
        print("ERROR: AEMO aggregation failed. Exiting.")
        return
    
    # Load and merge weather data
    weather_raw = load_and_merge_weather()
    if weather_raw is None:
        print("ERROR: Failed to load weather data. Exiting.")
        return
    
    # Clean and aggregate weather to hourly
    weather_hourly = clean_and_aggregate_weather(weather_raw)
    if weather_hourly is None or len(weather_hourly) == 0:
        print("ERROR: Weather data processing failed. Exiting.")
        return
    
    # Merge AEMO and weather (both hourly)
    merged_df = merge_aemo_weather(aemo_hourly, weather_hourly)
    if merged_df is None or len(merged_df) == 0:
        print("ERROR: Data merging failed. Exiting.")
        return
    
    # Save processed data
    save_processed_data(merged_df)
    
    # Final summary
    print("\n" + "=" * 70)
    print("Processing completed successfully!")
    print("=" * 70)
    print(f"\nData processing pipeline completed:")
    print(f"  1. Loaded AEMO 5-minute data")
    print(f"  2. Merged demand and price (5-minute)")
    print(f"  3. Aggregated AEMO to hourly")
    print(f"  4. Loaded and merged weather data across regions")
    print(f"  5. Aggregated weather to hourly")
    print(f"  6. Merged AEMO + weather (hourly)")
    print(f"  7. Saved to: {OUTPUT_DIR / f'aemo_weather_hourly_{YEARS[0]}_{YEARS[-1]}.csv'}")
    print("=" * 70)

if __name__ == "__main__":
    main()
