"""
v3.0-Alpha: USD/TRY Exchange Rate Integration

This script:
1. Fetches USD/TRY exchange rate from CBRT EVDS API
2. Creates 1-month and 3-month lagged features
3. Merges with existing macro data
4. Prepares data for LSTM retraining
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import requests
import warnings
warnings.filterwarnings('ignore')

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from load_env import get_evds_api_key
from dotenv import load_dotenv

print("="*70)
print("v3.0-ALPHA: USD/TRY Exchange Rate Integration")
print("="*70)

# Load API key
print("\n[1/7] Loading EVDS API key...")
env_file = project_root / ".env"
if env_file.exists():
    load_dotenv(env_file)

EVDS_API_KEY = get_evds_api_key()

if not EVDS_API_KEY:
    print("[ERROR] EVDS_API_KEY not found!")
    print("   Please ensure your .env file contains: EVDS_API_KEY=your_key_here")
    sys.exit(1)

print(f"[OK] API key loaded: {EVDS_API_KEY[:10]}...{EVDS_API_KEY[-5:]}")

# Setup paths
data_raw_dir = project_root / "data" / "raw"
data_processed_dir = project_root / "data" / "processed"
data_processed_dir.mkdir(parents=True, exist_ok=True)

# Fetch USD/TRY exchange rate - Try multiple sources
print("\n[2/7] Fetching USD/TRY exchange rate...")

# Method 1: Try yfinance (most reliable for FX data)
print("   Method 1: Trying yfinance (USDTRY=X)...")
try:
    import yfinance as yf
    usd_try_ticker = yf.Ticker("USDTRY=X")
    
    # Get historical data (last 10 years)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*10)
    
    hist = usd_try_ticker.history(start=start_date, end=end_date)
    
    if not hist.empty:
        usd_try_df = pd.DataFrame({
            'Date': hist.index,
            'USD_TRY': hist['Close'].values
        })
        # Convert to timezone-naive datetime
        usd_try_df['Date'] = pd.to_datetime(usd_try_df['Date'])
        if usd_try_df['Date'].dtype.tz is not None:
            usd_try_df['Date'] = usd_try_df['Date'].dt.tz_localize(None)
        else:
            # Convert to date first then back to datetime to remove timezone
            usd_try_df['Date'] = pd.to_datetime(usd_try_df['Date'].dt.date)
        usd_try_df = usd_try_df.sort_values('Date').reset_index(drop=True)
        usd_try_df = usd_try_df.dropna()
        
        print(f"   [OK] Successfully fetched from yfinance")
        print(f"   Records: {len(usd_try_df):,}")
        print(f"   Date range: {usd_try_df['Date'].min().date()} to {usd_try_df['Date'].max().date()}")
        print(f"   USD/TRY range: {usd_try_df['USD_TRY'].min():.2f} to {usd_try_df['USD_TRY'].max():.2f}")
        method_used = "yfinance"
    else:
        raise ValueError("No data from yfinance")
        
except Exception as e:
    print(f"   [WARNING] yfinance failed: {str(e)}")
    method_used = None
    usd_try_df = pd.DataFrame()
    
    # Method 2: Try EVDS API as fallback
    if usd_try_df.empty:
        print("   Method 2: Trying EVDS API...")
        EVDS_BASE_URL = "https://evds2.tcmb.gov.tr/service/evds"
        
        # Try different series codes
        series_codes_to_try = [
            "TP.DK.USD.S.YTL",
            "TP.DK.USD.A.YTL", 
            "TP.DK.USD.S",
        ]
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*10)
        start_date_str = start_date.strftime("%d-%m-%Y")
        end_date_str = end_date.strftime("%d-%m-%Y")
        
        for series_code in series_codes_to_try:
            try:
                url = f"{EVDS_BASE_URL}/dataseries/{series_code}"
                params = {
                    "key": EVDS_API_KEY,
                    "startDate": start_date_str,
                    "endDate": end_date_str,
                    "type": "json",
                    "formulas": "0"
                }
                
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                if 'items' in data and len(data['items']) > 0:
                    df = pd.DataFrame(data['items'])
                    date_col = None
                    value_col = None
                    
                    for col in df.columns:
                        if 'tarih' in col.lower():
                            date_col = col
                        elif col != date_col and pd.api.types.is_numeric_dtype(df[col]):
                            if value_col is None:
                                value_col = col
                    
                    if date_col and value_col:
                        usd_try_df = pd.DataFrame({
                            'Date': pd.to_datetime(df[date_col], format='%d-%m-%Y', errors='coerce'),
                            'USD_TRY': pd.to_numeric(df[value_col], errors='coerce')
                        })
                        usd_try_df = usd_try_df.dropna().sort_values('Date').reset_index(drop=True)
                        
                        if len(usd_try_df) > 0:
                            method_used = f"EVDS ({series_code})"
                            print(f"   [OK] Successfully fetched from EVDS using {series_code}")
                            print(f"   Records: {len(usd_try_df):,}")
                            break
                
                time.sleep(0.5)
            except:
                continue

if usd_try_df.empty:
    print("\n[ERROR] Could not fetch USD/TRY data. Exiting.")
    sys.exit(1)

# Load existing macro data
print("\n[3/7] Loading existing macroeconomic data...")
macro_file = data_processed_dir / "bist_macro_merged.csv"

if macro_file.exists():
    macro_df = pd.read_csv(macro_file)
    macro_df['Date'] = pd.to_datetime(macro_df['Date']).astype('datetime64[ns]')
    macro_df = macro_df.sort_values('Date').reset_index(drop=True)
    print(f"[OK] Loaded existing macro data: {len(macro_df):,} records")
    print(f"   Columns: {macro_df.columns.tolist()}")
else:
    print("[INFO] No existing macro data found, creating new dataset")
    macro_df = pd.DataFrame()

# Load BIST stock data for date alignment
print("\n[4/7] Loading BIST stock data for date alignment...")
stock_file = data_raw_dir / "bist_stock_prices.csv"
full_features_file = data_processed_dir / "bist_features_full.csv"

if full_features_file.exists():
    bist_df = pd.read_csv(full_features_file)
    bist_df['Date'] = pd.to_datetime(bist_df['Date'])
    if pd.api.types.is_datetime64_any_dtype(bist_df['Date']) and hasattr(bist_df['Date'].dtype, 'tz') and bist_df['Date'].dtype.tz is not None:
        bist_df['Date'] = bist_df['Date'].dt.tz_localize(None)
    bist_df = bist_df.sort_values('Date').reset_index(drop=True)
    print(f"[OK] Loaded BIST features: {len(bist_df):,} records")
    date_range = (bist_df['Date'].min(), bist_df['Date'].max())
elif stock_file.exists():
    bist_df = pd.read_csv(stock_file)
    bist_df['Date'] = pd.to_datetime(bist_df['Date'])
    if pd.api.types.is_datetime64_any_dtype(bist_df['Date']) and hasattr(bist_df['Date'].dtype, 'tz') and bist_df['Date'].dtype.tz is not None:
        bist_df['Date'] = bist_df['Date'].dt.tz_localize(None)
    bist_df = bist_df.sort_values('Date').reset_index(drop=True)
    print(f"[OK] Loaded BIST stock data: {len(bist_df):,} records")
    date_range = (bist_df['Date'].min(), bist_df['Date'].max())
else:
    print("[WARNING] No BIST data found, using USD/TRY date range")
    date_range = (usd_try_df['Date'].min(), usd_try_df['Date'].max())

# Resample USD/TRY to daily frequency
print("\n[5/7] Resampling USD/TRY to daily frequency...")

# Ensure all dates are timezone-naive (convert to date then back to datetime)
usd_try_df['Date'] = pd.to_datetime(usd_try_df['Date'].dt.date) if hasattr(usd_try_df['Date'].dtype, 'tz') else pd.to_datetime(usd_try_df['Date'])

# Get date range from BIST data (ensure timezone-naive)
bist_min_date = pd.to_datetime(date_range[0])
bist_max_date = pd.to_datetime(date_range[1])
if hasattr(bist_min_date, 'tz') and bist_min_date.tz is not None:
    bist_min_date = bist_min_date.tz_localize(None)
if hasattr(bist_max_date, 'tz') and bist_max_date.tz is not None:
    bist_max_date = bist_max_date.tz_localize(None)

# Get USD/TRY date range
usd_min_date = usd_try_df['Date'].min()
usd_max_date = usd_try_df['Date'].max()

# Create daily date range (use the overlap)
min_date = max(bist_min_date, usd_min_date)
max_date = min(bist_max_date, usd_max_date)

print(f"   BIST date range: {bist_min_date.date()} to {bist_max_date.date()}")
print(f"   USD/TRY date range: {usd_min_date.date()} to {usd_max_date.date()}")
print(f"   Overlap range: {min_date.date()} to {max_date.date()}")

usd_try_df_indexed = usd_try_df.set_index('Date').copy()
daily_dates = pd.date_range(start=min_date, end=max_date, freq='D')

# Reindex to daily and forward fill
usd_try_daily = usd_try_df_indexed.reindex(daily_dates).ffill().bfill()
usd_try_daily = usd_try_daily.reset_index()
usd_try_daily = usd_try_daily.rename(columns={'index': 'Date'})

print(f"[OK] Resampled to daily frequency")
print(f"   Daily records: {len(usd_try_daily):,}")
print(f"   Date range: {usd_try_daily['Date'].min().date()} to {usd_try_daily['Date'].max().date()}")

# Create lagged features
print("\n[6/7] Creating lagged USD/TRY features...")

# 1-month lag (approximately 30 days)
usd_try_daily['USD_TRY_Lag_1M'] = usd_try_daily['USD_TRY'].shift(30)
# 3-month lag (approximately 90 days)
usd_try_daily['USD_TRY_Lag_3M'] = usd_try_daily['USD_TRY'].shift(90)

# Additional features
usd_try_daily['USD_TRY_Change'] = usd_try_daily['USD_TRY'].pct_change() * 100  # Daily % change
usd_try_daily['USD_TRY_Volatility'] = usd_try_daily['USD_TRY_Change'].rolling(window=30).std()  # 30-day volatility

# Forward fill lagged features
lag_cols = [col for col in usd_try_daily.columns if 'Lag' in col]
if lag_cols:
    usd_try_daily[lag_cols] = usd_try_daily[lag_cols].ffill().bfill()

print(f"[OK] Created features:")
print(f"   - USD_TRY (current)")
print(f"   - USD_TRY_Lag_1M (1-month lag)")
print(f"   - USD_TRY_Lag_3M (3-month lag)")
print(f"   - USD_TRY_Change (daily % change)")
print(f"   - USD_TRY_Volatility (30-day rolling std)")

# Merge with existing macro data
print("\n[7/7] Merging USD/TRY with existing macro data...")

if not macro_df.empty:
    # Ensure Date columns are timezone-naive
    macro_df['Date'] = pd.to_datetime(macro_df['Date'])
    if pd.api.types.is_datetime64_any_dtype(macro_df['Date']) and hasattr(macro_df['Date'].dtype, 'tz') and macro_df['Date'].dtype.tz is not None:
        macro_df['Date'] = macro_df['Date'].dt.tz_localize(None)
    
    usd_try_daily['Date'] = pd.to_datetime(usd_try_daily['Date'])
    if pd.api.types.is_datetime64_any_dtype(usd_try_daily['Date']) and hasattr(usd_try_daily['Date'].dtype, 'tz') and usd_try_daily['Date'].dtype.tz is not None:
        usd_try_daily['Date'] = usd_try_daily['Date'].dt.tz_localize(None)
    
    # Merge with existing macro data
    merged_df = macro_df.merge(
        usd_try_daily[['Date', 'USD_TRY', 'USD_TRY_Lag_1M', 'USD_TRY_Lag_3M', 
                      'USD_TRY_Change', 'USD_TRY_Volatility']],
        on='Date',
        how='outer'
    )
    print(f"[OK] Merged with existing macro data")
else:
    # Create new merged dataset
    merged_df = usd_try_daily.copy()
    print(f"[OK] Created new macro dataset with USD/TRY")

# Ensure Date is timezone-naive
merged_df['Date'] = pd.to_datetime(merged_df['Date'])
if pd.api.types.is_datetime64_any_dtype(merged_df['Date']) and hasattr(merged_df['Date'].dtype, 'tz') and merged_df['Date'].dtype.tz is not None:
    merged_df['Date'] = merged_df['Date'].dt.tz_localize(None)

# Sort by date
merged_df = merged_df.sort_values('Date').reset_index(drop=True)

# Handle missing values
merged_df = merged_df.ffill().bfill()

# Save merged data
output_file = data_processed_dir / "bist_macro_merged_v3.csv"
merged_df.to_csv(output_file, index=False)

print(f"\n[OK] Saved merged dataset: {output_file.name}")
print(f"   Total records: {len(merged_df):,}")
print(f"   Date range: {merged_df['Date'].min().date()} to {merged_df['Date'].max().date()}")
print(f"   Columns: {len(merged_df.columns)}")
print(f"   USD/TRY features: {[col for col in merged_df.columns if 'USD' in col]}")

# Summary statistics
print(f"\n{'='*70}")
print("USD/TRY FEATURE SUMMARY")
print("="*70)
usd_cols = [col for col in merged_df.columns if 'USD' in col]
if usd_cols:
    print(merged_df[usd_cols].describe())

print(f"\n{'='*70}")
print("[SUCCESS] USD/TRY Integration Complete!")
print("="*70)
print(f"\n[OK] Ready for LSTM retraining with USD/TRY features")
print(f"   Next step: Run LSTM training with new features")
