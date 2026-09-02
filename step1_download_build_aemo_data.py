import mms_monthly_cli.mms_monthly as mms
from pathlib import Path
import pandas as pd
import time
from tqdm import tqdm

# ==================== Configuration ====================
CACHE_DIR = Path("./aemo_data")
CACHE_DIR.mkdir(exist_ok=True)

# Download all months from 2021-2023
YEARS = [2021, 2022, 2023]
MONTHS = list(range(1, 13))

# Tables to download
TABLES = ["DISPATCHREGIONSUM", "DISPATCHPRICE"]

# Record download results
download_log = []

# ==================== Download function ====================
def download_table(year, month, table):
    """Download a single table for a given month, return success status"""
    try:
        mms.get_and_unzip_table_csv(
            year=year,
            month=month,
            data_dir="DATA",
            table=table,
            cache=CACHE_DIR
        )
        return True
    except Exception as e:
        error_msg = str(e)
        if "404" in error_msg or "Not Found" in error_msg or "does not exist" in error_msg:
            return False
        else:
            print(f"      Warning: Other error: {error_msg[:100]}...")
            return False

# ==================== Batch download ====================
print("=" * 70)
print("Starting AEMO data download (2021-2023)")
print("=" * 70)

total_attempts = 0
success_count = 0

for year in YEARS:
    for month in MONTHS:
        for table in TABLES:
            total_attempts += 1

            print(f"[{year}-{month:02d}] | {table:20s}...", end=" ", flush=True)

            success = download_table(year, month, table)

            if success:
                success_count += 1
                print("SUCCESS")
                download_log.append({"year": year, "month": month, "table": table, "status": "success"})
            else:
                print("FAILED")
                download_log.append({"year": year, "month": month, "table": table, "status": "failed"})

            time.sleep(0.5)

        print()

print("\n" + "=" * 70)
print(f"Download statistics: {success_count}/{total_attempts} successful")
print("=" * 70)

# ==================== Check successfully downloaded files ====================
print("\nList of successfully downloaded files:")
successful_files = list(CACHE_DIR.glob("*.csv")) + list(CACHE_DIR.glob("*.CSV"))
for f in sorted(successful_files):
    size_mb = f.stat().st_size / 1024 / 1024
    print(f"  [OK] {f.name} ({size_mb:.1f} MB)")

print(f"\nTotal {len(successful_files)} CSV files")

# ==================== Merge function ====================
def load_and_merge_table(table_name):
    """Load all monthly CSV files for a given table and merge them"""
    csv_files = (
        list(CACHE_DIR.glob(f"*{table_name}*.csv"))
        + list(CACHE_DIR.glob(f"*{table_name}*.CSV"))
    )

    if not csv_files:
        print(f"Warning: No CSV files found for {table_name}")
        return None

    print(f"\nLoading {table_name}: Found {len(csv_files)} files")

    df_list = []
    for file in tqdm(csv_files, desc=f"Loading {table_name}"):
        try:
            df = pd.read_csv(file, skiprows=1, low_memory=False)
            df_list.append(df)
        except Exception as e:
            print(f"  Warning: Failed to load: {file.name} - {e}")

    if not df_list:
        return None

    df_merged = pd.concat(df_list, ignore_index=True)
    print(f"  [OK] {table_name} merge complete: {len(df_merged):,} records")
    return df_merged

# ==================== Merge both tables ====================
print("\n" + "=" * 70)
print("Starting data merge...")
print("=" * 70)

region_df = load_and_merge_table("DISPATCHREGIONSUM")
price_df = load_and_merge_table("DISPATCHPRICE")

if region_df is None or price_df is None:
    print("ERROR: Data loading failed, please check if download is complete")
    raise SystemExit(1)

# ==================== Data cleaning ====================
print("\nData cleaning...")

region_df['SETTLEMENTDATE'] = pd.to_datetime(region_df['SETTLEMENTDATE'])
price_df['SETTLEMENTDATE'] = pd.to_datetime(price_df['SETTLEMENTDATE'])

# Keep only required columns
region_cols = ['SETTLEMENTDATE', 'REGIONID', 'TOTALDEMAND']
price_cols = ['SETTLEMENTDATE', 'REGIONID', 'RRP']

region_df = region_df[region_cols]
price_df = price_df[price_cols]

print(f"  Demand data: {len(region_df):,} records")
print(f"  Price data: {len(price_df):,} records")

# ==================== Merge ====================
print("\nMerging demand and price data...")
merged_df = pd.merge(
    region_df,
    price_df,
    on=['SETTLEMENTDATE', 'REGIONID'],
    how='inner'
)
print(f"  [OK] Merge complete: {len(merged_df):,} records")

# ==================== Aggregate to hourly level ====================
print("\nAggregating to hourly data...")

merged_df['HOUR'] = merged_df['SETTLEMENTDATE'].dt.floor('h')

hourly_df = merged_df.groupby(['REGIONID', 'HOUR']).agg({
    'TOTALDEMAND': 'mean',
    'RRP': 'mean'
}).reset_index()

print(f"  [OK] Aggregation complete: {len(hourly_df):,} records")
print(f"  Date range: {hourly_df['HOUR'].min()} -> {hourly_df['HOUR'].max()}")

hourly_df = hourly_df.sort_values(['REGIONID', 'HOUR']).reset_index(drop=True)

# ==================== Save ====================
output_file = Path("./aemo_merged_hourly_2021_2023.csv")
hourly_df.to_csv(output_file, index=False)
print(f"\n[OK] Data saved: {output_file}")
print(f"   File size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")

# ==================== Preview ====================
print("\nData preview (first 10 rows):")
print(hourly_df.head(10))

print("\nData count by region:")
print(hourly_df['REGIONID'].value_counts())

print("\nData count by year:")
print(hourly_df['HOUR'].dt.year.value_counts().sort_index())

# ==================== Save download log ====================
log_df = pd.DataFrame(download_log)
log_df.to_csv("./download_log.csv", index=False)
print("\n[OK] Download log saved: download_log.csv")
