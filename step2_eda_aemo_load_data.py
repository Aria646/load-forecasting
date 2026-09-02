import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ==================== Configuration ====================
RESULTS_DIR = Path("./results")
RESULTS_DIR.mkdir(exist_ok=True)

# Set plot style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['font.size'] = 10

# ==================== Load data ====================
print("Loading data...")
data_file = Path("./aemo_merged_hourly_2021_2023.csv")

if not data_file.exists():
    print(f"Error: File {data_file} not found!")
    print("Please run the download script first.")
    exit()

df = pd.read_csv(data_file)
df['HOUR'] = pd.to_datetime(df['HOUR'])

print(f"Data loaded: {len(df):,} records")
print(f"Date range: {df['HOUR'].min()} to {df['HOUR'].max()}")
print(f"Regions: {df['REGIONID'].unique().tolist()}")
print("\n" + "="*70)

# ==================== Prepare data ====================
print("\nPreparing data for EDA plots...")

df['YEAR'] = df['HOUR'].dt.year
df['MONTH'] = df['HOUR'].dt.month
df['HOUR_OF_DAY'] = df['HOUR'].dt.hour
df['DAY_OF_WEEK'] = df['HOUR'].dt.dayofweek
df['DAY_NAME'] = df['HOUR'].dt.day_name()
df['MONTH_NAME'] = df['HOUR'].dt.month_name()
df['QUARTER'] = df['HOUR'].dt.quarter

def get_season(month):
    if month in [12, 1, 2]:
        return 'Summer'
    elif month in [3, 4, 5]:
        return 'Autumn'
    elif month in [6, 7, 8]:
        return 'Winter'
    else:
        return 'Spring'

df['SEASON'] = df['MONTH'].apply(get_season)

hourly_avg = df.groupby(['REGIONID', 'HOUR_OF_DAY'])['TOTALDEMAND'].mean().reset_index()
weekly_avg = df.groupby(['REGIONID', 'DAY_OF_WEEK', 'DAY_NAME'])['TOTALDEMAND'].mean().reset_index()
weekly_pivot = weekly_avg.pivot(index='DAY_NAME', columns='REGIONID', values='TOTALDEMAND')
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
weekly_pivot = weekly_pivot.reindex(day_order)

monthly_avg = df.groupby(['REGIONID', 'MONTH', 'MONTH_NAME'])['TOTALDEMAND'].mean().reset_index()
monthly_pivot = monthly_avg.pivot(index='MONTH_NAME', columns='REGIONID', values='TOTALDEMAND')
month_order = ['January', 'February', 'March', 'April', 'May', 'June',
               'July', 'August', 'September', 'October', 'November', 'December']
monthly_pivot = monthly_pivot.reindex(month_order)

seasonal_avg = df.groupby(['REGIONID', 'SEASON'])['TOTALDEMAND'].mean().reset_index()
seasonal_pivot = seasonal_avg.pivot(index='SEASON', columns='REGIONID', values='TOTALDEMAND')
season_order = ['Summer', 'Autumn', 'Winter', 'Spring']
seasonal_pivot = seasonal_pivot.reindex(season_order)

missing_df = df.isnull().sum().to_frame('Missing_Count')
missing_df['Missing_Percentage'] = (missing_df['Missing_Count'] / len(df)) * 100
missing_df = missing_df[missing_df['Missing_Count'] > 0]

print("\nCreating combined EDA figure...")

fig = plt.figure(figsize=(20, 24))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

ax1 = fig.add_subplot(gs[0, 0])
for region in df['REGIONID'].unique():
    region_data = df[df['REGIONID'] == region]
    ax1.plot(region_data['HOUR'], region_data['TOTALDEMAND'],
             label=region, alpha=0.6, linewidth=0.8)
ax1.set_title('(a) Load Time Series (2021-2023)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Date')
ax1.set_ylabel('Total Demand (MW)')
ax1.legend(loc='upper right', fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='x', rotation=45)

ax2 = fig.add_subplot(gs[0, 1])
for region in df['REGIONID'].unique():
    region_hourly = hourly_avg[hourly_avg['REGIONID'] == region]
    ax2.plot(region_hourly['HOUR_OF_DAY'], region_hourly['TOTALDEMAND'],
             marker='o', label=region, linewidth=2, markersize=5)
ax2.set_title('(b) 24-Hour Average Load Curve', fontsize=12, fontweight='bold')
ax2.set_xlabel('Hour of Day')
ax2.set_ylabel('Average Demand (MW)')
ax2.set_xticks(range(0, 24, 3))
ax2.legend(loc='best', fontsize=9)
ax2.grid(True, alpha=0.3)

ax3 = fig.add_subplot(gs[1, 0])
weekly_pivot.plot(kind='bar', ax=ax3)
ax3.set_title('(c) Average Load by Day of Week', fontsize=12, fontweight='bold')
ax3.set_xlabel('Day of Week')
ax3.set_ylabel('Average Demand (MW)')
ax3.legend(title='Region', fontsize=9)
ax3.grid(True, alpha=0.3)
ax3.tick_params(axis='x', rotation=45)

ax4 = fig.add_subplot(gs[1, 1])
monthly_pivot.plot(kind='bar', ax=ax4, alpha=0.8)
ax4.set_title('(d) Average Load by Month (bars) and Season (lines)', fontsize=12, fontweight='bold')
ax4.set_xlabel('Month')
ax4.set_ylabel('Average Demand (MW)')
ax4.legend(title='Region', fontsize=9)
ax4.grid(True, alpha=0.3)
ax4.tick_params(axis='x', rotation=45)

season_boundaries = {
    'Summer': [0, 2],
    'Autumn': [2, 5],
    'Winter': [5, 8],
    'Spring': [8, 11]
}

for season, (start, end) in season_boundaries.items():
    ax4.axvspan(start - 0.5, end - 0.5, alpha=0.1, color='gray')
    mid_point = (start + end) / 2 - 0.5
    if mid_point >= 0 and mid_point < 12:
        ax4.text(mid_point, ax4.get_ylim()[1] * 0.95, season,
                ha='center', va='top', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

ax5 = fig.add_subplot(gs[2, 0])
df.boxplot(column='TOTALDEMAND', by='REGIONID', ax=ax5, grid=True)
ax5.set_title('(e) Load Distribution by Region', fontsize=12, fontweight='bold')
ax5.set_xlabel('Region')
ax5.set_ylabel('Total Demand (MW)')
ax5.set_xticklabels(ax5.get_xticklabels(), rotation=0)

ax6 = fig.add_subplot(gs[2, 1])

if len(missing_df) > 0:
    missing_df_sorted = missing_df.sort_values('Missing_Percentage', ascending=True)
    bars = ax6.barh(missing_df_sorted.index, missing_df_sorted['Missing_Percentage'],
                    color='coral', edgecolor='darkred', alpha=0.7)
    ax6.set_title('(f) Missing Data Distribution', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Missing Percentage (%)')
    ax6.set_ylabel('Column')
    ax6.grid(True, alpha=0.3)
    for i, (idx, row) in enumerate(missing_df_sorted.iterrows()):
        ax6.text(row['Missing_Percentage'] + 0.5, i,
                f"{row['Missing_Percentage']:.1f}%",
                va='center', fontsize=9)
else:
    ax6.text(0.5, 0.5, 'No Missing Data Found!',
             ha='center', va='center', fontsize=16, transform=ax6.transAxes,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7))
    ax6.set_title('(f) Missing Data Distribution', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Missing Percentage (%)')
    ax6.set_ylabel('Column')
    ax6.grid(True, alpha=0.3)

fig.suptitle('AEMO Electricity Load Data - Exploratory Data Analysis (2021-2023)',
             fontsize=16, fontweight='bold', y=0.98)

plt.tight_layout()
plt.subplots_adjust(top=0.96)

output_file = RESULTS_DIR / "aemo_eda_combined.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"  Combined EDA figure saved: {output_file}")

plt.close()

print("\nSaving summary statistics...")

summary_stats = df.groupby('REGIONID')['TOTALDEMAND'].describe()
summary_stats.to_csv(RESULTS_DIR / 'eda_summary_stats.csv')

print(f"  Summary statistics saved: {RESULTS_DIR / 'eda_summary_stats.csv'}")

print("\n" + "="*70)
print("EDA COMPLETE - SUMMARY REPORT")
print("="*70)

print(f"\nDataset Overview:")
print(f"  - Total records: {len(df):,}")
print(f"  - Date range: {df['HOUR'].min()} to {df['HOUR'].max()}")
print(f"  - Number of regions: {df['REGIONID'].nunique()}")
print(f"  - Regions: {df['REGIONID'].unique().tolist()}")

print(f"\nLoad Statistics (MW):")
print(df.groupby('REGIONID')['TOTALDEMAND'].describe())

print(f"\nMissing Values:")
if df.isnull().sum().sum() > 0:
    print(df.isnull().sum())
else:
    print("  No missing values found!")

print(f"\nData Completeness:")
print(f"  - Total missing values: {df.isnull().sum().sum()}")
print(f"  - Complete rows: {len(df) - df.isnull().any(axis=1).sum():,}")

print(f"\nOutput files saved to '{RESULTS_DIR}/' directory:")
print(f"  1. aemo_eda_combined.png")
print(f"  2. eda_summary_stats.csv")

print(f"\n" + "="*70)
print("EDA COMPLETE!")
print("="*70)
