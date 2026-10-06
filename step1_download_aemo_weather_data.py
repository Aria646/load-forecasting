import mms_monthly_cli.mms_monthly as mms
from pathlib import Path
import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import os
import pytz
from config import *

# ==================== Configuration ====================

# AEMO data configuration
AEMO_CACHE_DIR = Path("./aemo_data")
AEMO_CACHE_DIR.mkdir(exist_ok=True)

MONTHS = list(range(1, 13))
AEMO_TABLES = ["DISPATCHREGIONSUM", "DISPATCHPRICE"]

# Weather data configuration
WEATHER_API_KEY = "f6d2efe5720d47ea92efe5720df7eaa8"
WEATHER_START_DATE = f"{YEARS[0]}-01-01"
WEATHER_END_DATE = f"{YEARS[-1]}-12-31"

# AEMO region to weather station mapping
REGION_STATIONS = {
    "NSW1": "YSBK",   # Sydney Airport
    "QLD1": "YBBN",   # Brisbane Airport
    "SA1": "YPAD",    # Adelaide Airport
    "TAS1": "YMHB",   # Hobart Airport
    "VIC1": "YMML",   # Melbourne Airport
}

WEATHER_RAW_DIR = Path("./weather_data/raw")
WEATHER_RAW_DIR.mkdir(parents=True, exist_ok=True)

# ==================== AEMO Download Functions ====================

def download_aemo_table(year, month, table):
    """
    Download a single AEMO table for a given month
    Returns: (success: bool, file_path: Path or None)
    """
    try:
        mms.get_and_unzip_table_csv(
            year=year,
            month=month,
            data_dir="DATA",
            table=table,
            cache=AEMO_CACHE_DIR
        )
        # Find the downloaded CSV file
        csv_files = list(AEMO_CACHE_DIR.glob(f"*{table}*{year}{month:02d}*.csv"))
        if csv_files:
            return True, csv_files[0]
        return True, None
    except Exception as e:
        error_msg = str(e)
        if "404" in error_msg or "Not Found" in error_msg or "does not exist" in error_msg:
            return False, None
        else:
            print(f"      Warning: Other error: {error_msg[:100]}...")
            return False, None

def download_all_aemo_data():
    """Download all AEMO data"""
    print("\n" + "=" * 70)
    print(f"Starting AEMO data download ({YEARS[0]}-{YEARS[-1]})")
    print("=" * 70)
    
    total_attempts = 0
    success_count = 0
    downloaded_files = []
    download_log = []
    
    for year in YEARS:
        for month in MONTHS:
            for table in AEMO_TABLES:
                total_attempts += 1
                
                print(f"[{year}-{month:02d}] | {table:20s}...", end=" ", flush=True)
                
                success, file_path = download_aemo_table(year, month, table)
                
                if success:
                    success_count += 1
                    print("SUCCESS")
                    download_log.append({
                        "year": year,
                        "month": month,
                        "table": table,
                        "status": "success",
                        "file": str(file_path) if file_path else "unknown"
                    })
                    if file_path:
                        downloaded_files.append(file_path)
                else:
                    print("FAILED")
                    download_log.append({
                        "year": year,
                        "month": month,
                        "table": table,
                        "status": "failed",
                        "file": None
                    })
                
                time.sleep(0.5)
            
            print()
    
    print("\n" + "=" * 70)
    print(f"AEMO download statistics: {success_count}/{total_attempts} successful")
    print("=" * 70)
    
    # List downloaded files
    print("\nAEMO downloaded files:")
    if downloaded_files:
        for f in sorted(downloaded_files):
            size_mb = f.stat().st_size / 1024 / 1024
            print(f"  [OK] {f.name} ({size_mb:.1f} MB)")
    else:
        csv_files = list(AEMO_CACHE_DIR.glob("*.csv"))
        if csv_files:
            for f in sorted(csv_files):
                size_mb = f.stat().st_size / 1024 / 1024
                print(f"  [OK] {f.name} ({size_mb:.1f} MB)")
        else:
            print("  No CSV files found")
    
    print(f"\nTotal CSV files: {len(list(AEMO_CACHE_DIR.glob('*.csv')))}")
    
    # Save download log
    log_df = pd.DataFrame(download_log)
    log_df.to_csv(AEMO_CACHE_DIR / "aemo_download_log.csv", index=False)
    print(f"[OK] Download log saved: {AEMO_CACHE_DIR / 'aemo_download_log.csv'}")
    
    return downloaded_files

# ==================== Weather Download Functions ====================

def convert_to_australian_time(timestamp_gmt):
    """
    Convert GMT timestamp to Australian Eastern Time (AEST/AEDT)
    Returns a naive datetime object without timezone info
    """
    gmt_time = datetime.fromtimestamp(timestamp_gmt, tz=pytz.UTC)
    australia_tz = pytz.timezone('Australia/Sydney')
    australia_time = gmt_time.astimezone(australia_tz)
    return australia_time.replace(tzinfo=None)

def fetch_weather_data(station_id, start_date, end_date):
    """
    Fetch historical weather data from Weather.com API
    """
    url = f"https://api.weather.com/v1/location/{station_id}:9:AU/observations/historical.json"
    
    params = {
        "apiKey": WEATHER_API_KEY,
        "units": "m",
        "startDate": start_date.replace("-", ""),
        "endDate": end_date.replace("-", "")
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            observations = data.get("observations", [])
            return observations
        else:
            print(f"Download failed: HTTP {response.status_code}")
            return []
    except Exception as e:
        print(f"Error: {e}")
        return []

def parse_observation(obs, station_id):
    """Parse a single observation record"""
    gmt_timestamp = obs["valid_time_gmt"]
    australia_time = convert_to_australian_time(gmt_timestamp)
    
    return {
        "STATION": station_id,
        "DATETIME": australia_time,
        "TEMPERATURE": obs.get("temp"),
        "DEW_POINT": obs.get("dewPt"),
        "HUMIDITY": obs.get("rh"),
        "PRESSURE": obs.get("pressure"),
        "WIND_SPEED": obs.get("wspd"),
        "VISIBILITY": obs.get("vis"),
        "PRECIPITATION": obs.get("precip_total"),
        "WX_PHRASE": obs.get("wx_phrase"),
    }

def download_weather_for_station(station_id, start_date, end_date):
    """
    Download weather data month by month for a single station
    """
    all_data = []
    
    current = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    while current <= end:
        # Calculate month end
        if current.month == 12:
            next_month = current.replace(year=current.year + 1, month=1, day=1)
        else:
            next_month = current.replace(month=current.month + 1, day=1)
        
        month_end = min(next_month - timedelta(days=1), end)
        
        start_str = current.strftime("%Y%m%d")
        end_str = month_end.strftime("%Y%m%d")
        
        print(f"Downloading {current.strftime('%Y-%m')}...", end=" ", flush=True)
        
        obs = fetch_weather_data(station_id, start_str, end_str)
        
        if obs:
            for o in obs:
                parsed = parse_observation(o, station_id)
                all_data.append(parsed)
            print(f"SUCCESS: {len(obs)} records")
        else:
            print("FAILED")
        
        current = next_month
        time.sleep(1)
    
    return pd.DataFrame(all_data)

def download_all_weather_data():
    """Download weather data for all regions (each region saved separately)"""
    print("\n" + "=" * 60)
    print(f"Starting weather data download ({YEARS[0]}-{YEARS[-1]})")
    print("=" * 60)
    
    download_summary = []
    
    for region, station in REGION_STATIONS.items():
        print(f"\nProcessing region: {region} ({station})")
        
        df = download_weather_for_station(station, WEATHER_START_DATE, WEATHER_END_DATE)
        
        if not df.empty:
            df["REGIONID"] = region
            
            print(f"Date range: {df['DATETIME'].min()} to {df['DATETIME'].max()}")
            print(f"Total records: {len(df)}")
            
            # Save each region's raw data separately
            raw_filename = WEATHER_RAW_DIR / f"weather_{region}_{WEATHER_START_DATE[:4]}_{WEATHER_END_DATE[:4]}.csv"
            df.to_csv(raw_filename, index=False)
            print(f"Saved: {raw_filename}")
            
            download_summary.append({
                "region": region,
                "station": station,
                "records": len(df),
                "file": str(raw_filename),
                "status": "success"
            })
        else:
            print(f"No data for {region}")
            download_summary.append({
                "region": region,
                "station": station,
                "records": 0,
                "file": None,
                "status": "failed"
            })
    
    # Save download summary
    summary_df = pd.DataFrame(download_summary)
    summary_filename = WEATHER_RAW_DIR / "weather_download_summary.csv"
    summary_df.to_csv(summary_filename, index=False)
    print(f"\nDownload summary saved: {summary_filename}")
    
    # List all downloaded weather files
    print("\nWeather downloaded files:")
    for f in sorted(WEATHER_RAW_DIR.glob("*.csv")):
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"   - {f.name} ({size_mb:.2f} MB)")
    
    return download_summary

# ==================== Main Execution ====================

def main():
    """Main function to download both AEMO and weather data"""
    print("=" * 70)
    print(f"AEMO & Weather Data Downloader ({YEARS[0]}-{YEARS[-1]})")
    print("=" * 70)
    print("\nThis script downloads raw data only. No processing or merging is performed.")
    print("Data will be saved to:")
    print(f"  - AEMO data: {AEMO_CACHE_DIR.absolute()}")
    print(f"  - Weather data: {WEATHER_RAW_DIR.absolute()}")
    print()
    
    # Download AEMO data
    aemo_files = download_all_aemo_data()
    
    # Download Weather data
    weather_summary = download_all_weather_data()
    
    # Final summary
    print("\n" + "=" * 70)
    print("Download completed successfully!")
    print("=" * 70)
    print(f"\nData directories:")
    print(f"   AEMO data   : {AEMO_CACHE_DIR.absolute()}")
    print(f"   Weather data: {WEATHER_RAW_DIR.absolute()}")
    print(f"\nSummary:")
    print(f"   AEMO files  : {len(list(AEMO_CACHE_DIR.glob('*.csv')))} CSV files")
    print(f"   Weather files: {len(list(WEATHER_RAW_DIR.glob('*.csv')))} CSV files")
    print("\nData is ready for processing in later steps.")
    print("=" * 70)

if __name__ == "__main__":
    main()
