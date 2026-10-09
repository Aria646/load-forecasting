import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from pathlib import Path
from config import *
import warnings
warnings.filterwarnings('ignore')

# ==================== Configuration ====================
RESULTS_DIR = Path("./results")
RESULTS_DIR.mkdir(exist_ok=True)

# Set plot style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 100

# ==================== Load Data ====================
print("=" * 80)
print("AEMO Electricity Load & Weather Data - Exploratory Data Analysis")
print("=" * 80)

data_file = Path(f"./processed_data/aemo_weather_hourly_{YEARS[0]}_{YEARS[-1]}.csv")

if not data_file.exists():
    print(f"Error: File {data_file} not found!")
    print("Please run process_aemo_weather_data.py first.")
    exit()

df = pd.read_csv(data_file)
df['SETTLEMENTDATE'] = pd.to_datetime(df['SETTLEMENTDATE'])

print(f"\nData loaded: {len(df):,} records")
print(f"Date range: {df['SETTLEMENTDATE'].min()} to {df['SETTLEMENTDATE'].max()}")
print(f"Regions: {df['REGIONID'].unique().tolist()}")
print(f"Detailed analysis region: {SELECT_REGION}")

# ==================== Prepare Data ====================
print("\nPreparing data for EDA...")

# Time features
df['YEAR'] = df['SETTLEMENTDATE'].dt.year
df['MONTH'] = df['SETTLEMENTDATE'].dt.month
df['HOUR_OF_DAY'] = df['SETTLEMENTDATE'].dt.hour
df['DAY_OF_WEEK'] = df['SETTLEMENTDATE'].dt.dayofweek
df['DAY_NAME'] = df['SETTLEMENTDATE'].dt.day_name()
df['MONTH_NAME'] = df['SETTLEMENTDATE'].dt.month_name()
df['QUARTER'] = df['SETTLEMENTDATE'].dt.quarter

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

# Extract region-specific data for detailed analysis
detail_df = df[df['REGIONID'] == SELECT_REGION].dropna(subset=['TEMPERATURE', 'TOTALDEMAND'])

# ==================== Figure 1: Load and Weather Time Series ====================
print("\nCreating Figure 1: Time Series Overview...")

fig1, axes = plt.subplots(4, 1, figsize=(16, 12))

# Plot 1: Load time series
ax = axes[0]
for region in df['REGIONID'].unique():
    region_data = df[df['REGIONID'] == region]
    ax.plot(region_data['SETTLEMENTDATE'], region_data['TOTALDEMAND'], 
            label=region, alpha=0.6, linewidth=0.8)
ax.set_title(f'(a) Electricity Load Time Series ({YEARS[0]}-{YEARS[-1]})', fontsize=12, fontweight='bold')
ax.set_ylabel('Total Demand (MW)')
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, alpha=0.3)

# Plot 2: Temperature time series
ax = axes[1]
for region in df['REGIONID'].unique():
    region_data = df[df['REGIONID'] == region]
    ax.plot(region_data['SETTLEMENTDATE'], region_data['TEMPERATURE'], 
            label=region, alpha=0.6, linewidth=0.8)
ax.set_title(f'(b) Temperature Time Series ({YEARS[0]}-{YEARS[-1]})', fontsize=12, fontweight='bold')
ax.set_ylabel('Temperature (C)')
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, alpha=0.3)

# Plot 3: Humidity time series
ax = axes[2]
for region in df['REGIONID'].unique():
    region_data = df[df['REGIONID'] == region]
    ax.plot(region_data['SETTLEMENTDATE'], region_data['HUMIDITY'], 
            label=region, alpha=0.6, linewidth=0.8)
ax.set_title(f'(c) Humidity Time Series ({YEARS[0]}-{YEARS[-1]})', fontsize=12, fontweight='bold')
ax.set_ylabel('Humidity (%)')
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, alpha=0.3)

# Plot 4: Precipitation time series
ax = axes[3]
for region in df['REGIONID'].unique():
    region_data = df[df['REGIONID'] == region]
    ax.plot(region_data['SETTLEMENTDATE'], region_data['PRECIPITATION'], 
            label=region, alpha=0.6, linewidth=0.8)
ax.set_title(f'(d) Precipitation Time Series ({YEARS[0]}-{YEARS[-1]})', fontsize=12, fontweight='bold')
ax.set_ylabel('Precipitation (mm)')
ax.set_xlabel('Date')
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig1.savefig(RESULTS_DIR / 'eda_01_time_series_overview.png', dpi=300, bbox_inches='tight')
plt.close(fig1)
print(f"  Saved: eda_01_time_series_overview.png")

# ==================== Figure 2: Load Patterns (Daily, Weekly, Monthly, Seasonal) ====================
print("\nCreating Figure 2: Load Patterns...")

fig2, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: 24-hour average load curve
ax = axes[0, 0]
hourly_avg = df.groupby(['REGIONID', 'HOUR_OF_DAY'])['TOTALDEMAND'].mean().reset_index()
for region in df['REGIONID'].unique():
    region_hourly = hourly_avg[hourly_avg['REGIONID'] == region]
    ax.plot(region_hourly['HOUR_OF_DAY'], region_hourly['TOTALDEMAND'], 
            marker='o', label=region, linewidth=2, markersize=4)
ax.set_title('(a) 24-Hour Average Load Curve', fontsize=12, fontweight='bold')
ax.set_xlabel('Hour of Day')
ax.set_ylabel('Average Demand (MW)')
ax.set_xticks(range(0, 24, 3))
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Plot 2: Load by day of week
ax = axes[0, 1]
weekly_avg = df.groupby(['REGIONID', 'DAY_OF_WEEK', 'DAY_NAME'])['TOTALDEMAND'].mean().reset_index()
weekly_pivot = weekly_avg.pivot(index='DAY_NAME', columns='REGIONID', values='TOTALDEMAND')
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
weekly_pivot = weekly_pivot.reindex(day_order)
weekly_pivot.plot(kind='bar', ax=ax)
ax.set_title('(b) Average Load by Day of Week', fontsize=12, fontweight='bold')
ax.set_xlabel('Day of Week')
ax.set_ylabel('Average Demand (MW)')
ax.legend(title='Region', fontsize=9)
ax.grid(True, alpha=0.3)
ax.tick_params(axis='x', rotation=45)

# Plot 3: Load by month
ax = axes[1, 0]
monthly_avg = df.groupby(['REGIONID', 'MONTH', 'MONTH_NAME'])['TOTALDEMAND'].mean().reset_index()
monthly_pivot = monthly_avg.pivot(index='MONTH_NAME', columns='REGIONID', values='TOTALDEMAND')
month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
               'July', 'August', 'September', 'October', 'November', 'December']
monthly_pivot = monthly_pivot.reindex(month_order)
monthly_pivot.plot(kind='bar', ax=ax)
ax.set_title('(c) Average Load by Month', fontsize=12, fontweight='bold')
ax.set_xlabel('Month')
ax.set_ylabel('Average Demand (MW)')
ax.legend(title='Region', fontsize=9)
ax.grid(True, alpha=0.3)
ax.tick_params(axis='x', rotation=45)

# Plot 4: Load by season
ax = axes[1, 1]
seasonal_avg = df.groupby(['REGIONID', 'SEASON'])['TOTALDEMAND'].mean().reset_index()
seasonal_pivot = seasonal_avg.pivot(index='SEASON', columns='REGIONID', values='TOTALDEMAND')
season_order = ['Summer', 'Autumn', 'Winter', 'Spring']
seasonal_pivot = seasonal_pivot.reindex(season_order)
seasonal_pivot.plot(kind='bar', ax=ax)
ax.set_title('(d) Average Load by Season', fontsize=12, fontweight='bold')
ax.set_xlabel('Season')
ax.set_ylabel('Average Demand (MW)')
ax.legend(title='Region', fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig2.savefig(RESULTS_DIR / 'eda_02_load_patterns.png', dpi=300, bbox_inches='tight')
plt.close(fig2)
print(f"  Saved: eda_02_load_patterns.png")

# ==================== Figure 3: Weather Patterns ====================
print("\nCreating Figure 3: Weather Patterns...")

fig3, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Temperature distribution by region
ax = axes[0, 0]
for region in df['REGIONID'].unique():
    region_data = df[df['REGIONID'] == region]
    ax.hist(region_data['TEMPERATURE'], bins=30, alpha=0.5, label=region, density=True)
ax.set_title('(a) Temperature Distribution by Region', fontsize=12, fontweight='bold')
ax.set_xlabel('Temperature (C)')
ax.set_ylabel('Density')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Plot 2: Temperature vs Humidity scatter
ax = axes[0, 1]
sample = df.sample(min(10000, len(df)))
scatter = ax.scatter(sample['TEMPERATURE'], sample['HUMIDITY'], 
                     c=sample['TOTALDEMAND'], cmap='viridis', alpha=0.5, s=10)
ax.set_title('(b) Temperature vs Humidity (colored by Load)', fontsize=12, fontweight='bold')
ax.set_xlabel('Temperature (C)')
ax.set_ylabel('Humidity (%)')
plt.colorbar(scatter, ax=ax, label='Demand (MW)')
ax.grid(True, alpha=0.3)

# Plot 3: Weather variable correlations
ax = axes[1, 0]
weather_vars = ['TEMPERATURE', 'HUMIDITY', 'PRESSURE', 'WIND_SPEED', 'PRECIPITATION']
weather_corr = df[weather_vars].corr(method='spearman')
sns.heatmap(weather_corr, annot=True, fmt='.2f', cmap='RdBu_r', 
            center=0, vmin=-1, vmax=1, square=True, linewidths=0.5, ax=ax)
ax.set_title('(c) Weather Variable Correlations (Spearman)', fontsize=12, fontweight='bold')

# Plot 4: Monthly weather patterns
ax = axes[1, 1]
monthly_weather = df.groupby('MONTH').agg({
    'TEMPERATURE': 'mean',
    'HUMIDITY': 'mean',
    'PRECIPITATION': 'mean',
    'WIND_SPEED': 'mean'
}).reset_index()

ax.plot(monthly_weather['MONTH'], monthly_weather['TEMPERATURE'], 
        'ro-', linewidth=2, markersize=8, label='Temperature')
ax2 = ax.twinx()
ax2.plot(monthly_weather['MONTH'], monthly_weather['HUMIDITY'], 
         'bs-', linewidth=2, markersize=8, label='Humidity')
ax2.plot(monthly_weather['MONTH'], monthly_weather['PRECIPITATION'], 
         'g^-', linewidth=2, markersize=8, label='Precipitation')

ax.set_title('(d) Monthly Weather Patterns', fontsize=12, fontweight='bold')
ax.set_xlabel('Month')
ax.set_ylabel('Temperature (C)')
ax2.set_ylabel('Humidity (%) / Precipitation (mm)')
ax.legend(loc='upper left')
ax2.legend(loc='upper right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig3.savefig(RESULTS_DIR / 'eda_03_weather_patterns.png', dpi=300, bbox_inches='tight')
plt.close(fig3)
print(f"  Saved: eda_03_weather_patterns.png")

# ==================== Figure 4: Temperature-Load Relationship ====================
print(f"\nCreating Figure 4: Temperature-Load Relationship (Region: {SELECT_REGION})...")

fig4, axes = plt.subplots(2, 3, figsize=(18, 12))

# Plot 1: Raw scatter (misleading)
ax = axes[0, 0]
sample_detail = detail_df.sample(min(5000, len(detail_df)))
ax.scatter(sample_detail['TEMPERATURE'], sample_detail['TOTALDEMAND'], alpha=0.1, s=5)
z = np.polyfit(sample_detail['TEMPERATURE'], sample_detail['TOTALDEMAND'], 1)
p = np.poly1d(z)
x_line = np.linspace(sample_detail['TEMPERATURE'].min(), sample_detail['TEMPERATURE'].max(), 100)
ax.plot(x_line, p(x_line), 'r--', linewidth=2, 
        label=f'Linear (r={pearsonr(sample_detail["TEMPERATURE"], sample_detail["TOTALDEMAND"])[0]:.2f})')
ax.set_xlabel('Temperature (C)')
ax.set_ylabel('Load (MW)')
ax.set_title(f'(a) Raw Scatter - {SELECT_REGION}', fontsize=11, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Binned means (U-shape)
ax = axes[0, 1]
temp_bins = pd.cut(detail_df['TEMPERATURE'], bins=np.arange(-5, 46, 2))
bin_stats = detail_df.groupby(temp_bins, observed=True)['TOTALDEMAND'].agg(['mean', 'std', 'count']).reset_index()
bin_centers = []
for interval in bin_stats['TEMPERATURE']:
    if pd.notna(interval):
        bin_centers.append((interval.left + interval.right) / 2)
    else:
        bin_centers.append(np.nan)
bin_stats['TEMP_MID'] = bin_centers
bin_stats_valid = bin_stats[bin_stats['count'] > 50]

ax.plot(bin_stats_valid['TEMP_MID'], bin_stats_valid['mean'], 
        'ro-', linewidth=2, markersize=8, label='Mean Load')
ax.fill_between(bin_stats_valid['TEMP_MID'],
                bin_stats_valid['mean'] - bin_stats_valid['std'],
                bin_stats_valid['mean'] + bin_stats_valid['std'],
                alpha=0.2, color='red', label='+/-1 Std')
ax.axvline(20, color='gray', linestyle='--', alpha=0.5, label='Comfort (~20C)')
ax.set_xlabel('Temperature (C)')
ax.set_ylabel('Average Load (MW)')
ax.set_title(f'(b) Binned Means - U-Shape ({SELECT_REGION})', fontsize=11, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Seasonal scatter
ax = axes[0, 2]
colors = {'Summer': 'red', 'Winter': 'blue', 'Autumn': 'orange', 'Spring': 'green'}
for season, color in colors.items():
    subset = detail_df[detail_df['SEASON'] == season]
    if len(subset) > 100:
        subset_sample = subset.sample(min(2000, len(subset)))
        ax.scatter(subset_sample['TEMPERATURE'], subset_sample['TOTALDEMAND'],
                   alpha=0.3, s=5, color=color, label=season)
ax.set_xlabel('Temperature (C)')
ax.set_ylabel('Load (MW)')
ax.set_title(f'(c) Seasonal Scatter - {SELECT_REGION}', fontsize=11, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Seasonal correlations
ax = axes[1, 0]
seasons = ['Summer', 'Autumn', 'Winter', 'Spring']
seasonal_corrs = []
for season in seasons:
    subset = detail_df[detail_df['SEASON'] == season]
    if len(subset) > 100:
        corr = pearsonr(subset['TEMPERATURE'], subset['TOTALDEMAND'])[0]
    else:
        corr = np.nan
    seasonal_corrs.append(corr)

bars = ax.bar(seasons, seasonal_corrs, color=['red', 'orange', 'blue', 'green'])
ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
ax.set_xlabel('Season')
ax.set_ylabel('Pearson Correlation')
ax.set_title(f'(d) Seasonal Correlations ({SELECT_REGION})', fontsize=11, fontweight='bold')
for i, (bar, val) in enumerate(zip(bars, seasonal_corrs)):
    if not np.isnan(val):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', va='bottom')
ax.grid(True, alpha=0.3)

# Plot 5: Spearman vs Pearson comparison
ax = axes[1, 1]
corr_methods = ['Pearson', 'Spearman', 'Kendall']
corr_values = [
    pearsonr(detail_df['TEMPERATURE'], detail_df['TOTALDEMAND'])[0],
    spearmanr(detail_df['TEMPERATURE'], detail_df['TOTALDEMAND'])[0],
    kendalltau(detail_df['TEMPERATURE'], detail_df['TOTALDEMAND'])[0]
]

bars = ax.bar(corr_methods, corr_values, color=['lightblue', 'salmon', 'lightgreen'])
ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
ax.set_ylabel('Correlation Coefficient')
ax.set_title(f'(e) Correlation Method Comparison ({SELECT_REGION})', fontsize=11, fontweight='bold')
for i, (bar, val) in enumerate(zip(bars, corr_values)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{val:.3f}', ha='center', va='bottom')
ax.grid(True, alpha=0.3)

# Plot 6: Correlation by temperature range
ax = axes[1, 2]
temp_ranges = [(-5, 10), (10, 18), (18, 25), (25, 35), (35, 45)]
range_labels = ['<-5~10C', '10~18C', '18~25C', '25~35C', '35~45C']
range_corrs = []

for low, high in temp_ranges:
    subset = detail_df[(detail_df['TEMPERATURE'] >= low) & (detail_df['TEMPERATURE'] < high)]
    if len(subset) > 100:
        corr = pearsonr(subset['TEMPERATURE'], subset['TOTALDEMAND'])[0]
    else:
        corr = np.nan
    range_corrs.append(corr)

bars = ax.bar(range_labels, range_corrs, color=['darkblue', 'blue', 'lightblue', 'orange', 'red'])
ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
ax.set_xlabel('Temperature Range')
ax.set_ylabel('Correlation')
ax.set_title(f'(f) Correlation by Temperature Range ({SELECT_REGION})', fontsize=11, fontweight='bold')
for i, (bar, corr) in enumerate(zip(bars, range_corrs)):
    if not np.isnan(corr):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{corr:.2f}', ha='center', va='bottom', fontweight='bold')
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig4.savefig(RESULTS_DIR / f'eda_04_temperature_load_relationship_{SELECT_REGION}.png', dpi=300, bbox_inches='tight')
plt.close(fig4)
print(f"  Saved: eda_04_temperature_load_relationship_{SELECT_REGION}.png")

# ==================== Figure 5: Weather-Load Correlations ====================
print("\nCreating Figure 5: Weather-Load Correlations...")

fig5, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Correlation heatmap
ax = axes[0, 0]
features = ['TOTALDEMAND', 'TEMPERATURE', 'HUMIDITY', 'PRESSURE',
            'WIND_SPEED', 'HOUR_OF_DAY', 'MONTH']
corr_matrix = df[features].corr(method='spearman')
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, square=True, linewidths=0.5, ax=ax)
ax.set_title('(a) All Variables Correlation (Spearman)', fontsize=12, fontweight='bold')

# Plot 2: Correlation by region
ax = axes[0, 1]
weather_cols = ['TEMPERATURE', 'HUMIDITY', 'PRESSURE', 'WIND_SPEED']
regions_order = ['NSW1', 'QLD1', 'SA1', 'TAS1', 'VIC1']
corr_by_region = {}
for region in regions_order:
    region_df = df[df['REGIONID'] == region]
    corrs = []
    for col in weather_cols:
        valid = region_df[['TOTALDEMAND', col]].dropna()
        if len(valid) > 100:
            c = pearsonr(valid['TOTALDEMAND'], valid[col])[0]
            corrs.append(c if not np.isnan(c) else 0)
        else:
            corrs.append(0)
    corr_by_region[region] = corrs

corr_df = pd.DataFrame(corr_by_region, index=weather_cols).T
corr_df.plot(kind='bar', ax=ax, width=0.75)
ax.set_title('(b) Weather-Load Correlation by Region', fontsize=12, fontweight='bold')
ax.set_xlabel('Region')
ax.set_ylabel('Pearson Correlation')
ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
ax.legend(title='Weather Variable', fontsize=9)
ax.grid(True, alpha=0.3)
ax.tick_params(axis='x', rotation=0)
for container in ax.containers:
    ax.bar_label(container, fmt='%.2f', fontsize=7, padding=2)

# Plot 3: Load vs Humidity
ax = axes[1, 0]
sample = df.sample(min(10000, len(df)))
scatter = ax.scatter(sample['HUMIDITY'], sample['TOTALDEMAND'],
                     c=sample['TEMPERATURE'], cmap='coolwarm', alpha=0.5, s=10)
ax.set_xlabel('Humidity (%)')
ax.set_ylabel('Load (MW)')
ax.set_title('(c) Load vs Humidity (colored by Temperature)', fontsize=12, fontweight='bold')
plt.colorbar(scatter, ax=ax, label='Temperature (C)')
ax.grid(True, alpha=0.3)

# Plot 4: Load vs Wind Speed
ax = axes[1, 1]
sample = df.sample(min(10000, len(df)))
scatter = ax.scatter(sample['WIND_SPEED'], sample['TOTALDEMAND'],
                     c=sample['TEMPERATURE'], cmap='viridis', alpha=0.5, s=10)
ax.set_xlabel('Wind Speed (m/s)')
ax.set_ylabel('Load (MW)')
ax.set_title('(d) Load vs Wind Speed (colored by Temperature)', fontsize=12, fontweight='bold')
plt.colorbar(scatter, ax=ax, label='Temperature (C)')
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig5.savefig(RESULTS_DIR / 'eda_05_weather_load_correlations.png', dpi=300, bbox_inches='tight')
plt.close(fig5)
print(f"  Saved: eda_05_weather_load_correlations.png")

# ==================== Figure 6: Missing Data Distribution ====================
print("\nCreating Figure 6: Missing Data...")

fig6, ax = plt.subplots(figsize=(10, 8))

missing_df = df.isnull().sum().to_frame('Missing_Count')
missing_df['Missing_Percentage'] = (missing_df['Missing_Count'] / len(df)) * 100
missing_df = missing_df[missing_df['Missing_Count'] > 0]

if len(missing_df) > 0:
    missing_df_sorted = missing_df.sort_values('Missing_Percentage', ascending=True)
    bars = ax.barh(missing_df_sorted.index, missing_df_sorted['Missing_Percentage'], 
                   color='coral', edgecolor='darkred', alpha=0.7)
    ax.set_title('Missing Data Distribution', fontsize=12, fontweight='bold')
    ax.set_xlabel('Missing Percentage (%)')
    ax.set_ylabel('Column')
    ax.grid(True, alpha=0.3)
    for i, (idx, row) in enumerate(missing_df_sorted.iterrows()):
        ax.text(row['Missing_Percentage'] + 0.5, i, 
                f"{row['Missing_Percentage']:.1f}%", 
                va='center', fontsize=9)
else:
    ax.text(0.5, 0.5, 'No Missing Data Found!', 
            ha='center', va='center', fontsize=16, transform=ax.transAxes,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7))
    ax.set_title('Missing Data Distribution', fontsize=12, fontweight='bold')
    ax.set_xlabel('Missing Percentage (%)')
    ax.set_ylabel('Column')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
fig6.savefig(RESULTS_DIR / 'eda_06_missing_data.png', dpi=300, bbox_inches='tight')
plt.close(fig6)
print(f"  Saved: eda_06_missing_data.png")

# ==================== Summary Report ====================
print("\n" + "=" * 80)
print("EDA COMPLETE - SUMMARY REPORT")
print("=" * 80)

print(f"\nDataset Overview:")
print(f"  - Total records: {len(df):,}")
print(f"  - Date range: {df['SETTLEMENTDATE'].min()} to {df['SETTLEMENTDATE'].max()}")
print(f"  - Number of regions: {df['REGIONID'].nunique()}")
print(f"  - Regions: {df['REGIONID'].unique().tolist()}")

print(f"\nLoad Statistics (MW):")
print(df.groupby('REGIONID')['TOTALDEMAND'].describe().round(2))

print(f"\nWeather Statistics:")
weather_cols = ['TEMPERATURE', 'HUMIDITY', 'PRESSURE', 'WIND_SPEED', 'PRECIPITATION']
print(df[weather_cols].describe().round(2))

print(f"\nTemperature-Load Correlation ({SELECT_REGION}):")
print(f"  Pearson:  {pearsonr(detail_df['TEMPERATURE'], detail_df['TOTALDEMAND'])[0]:.3f}")
print(f"  Spearman: {spearmanr(detail_df['TEMPERATURE'], detail_df['TOTALDEMAND'])[0]:.3f}")

print(f"\nSeasonal Temperature-Load Correlations ({SELECT_REGION}):")
for season in ['Summer', 'Autumn', 'Winter', 'Spring']:
    subset = detail_df[detail_df['SEASON'] == season]
    if len(subset) > 100:
        corr = pearsonr(subset['TEMPERATURE'], subset['TOTALDEMAND'])[0]
        print(f"  {season:6s}: {corr:.3f}")

print(f"\nMissing Values:")
missing_count = df.isnull().sum().sum()
if missing_count > 0:
    print(f"  Total missing values: {missing_count}")
    print(df.isnull().sum())
else:
    print("  No missing values found!")

print(f"\nOutput files saved to '{RESULTS_DIR}/' directory:")
eda_files = sorted(RESULTS_DIR.glob("eda_*.png"))
for f in eda_files:
    size_mb = f.stat().st_size / 1024 / 1024
    print(f"  - {f.name} ({size_mb:.2f} MB)")

print("\n" + "=" * 80)
print("EDA COMPLETE!")
print("=" * 80)
