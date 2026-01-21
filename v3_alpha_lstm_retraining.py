"""
v3.0-Alpha: LSTM Retraining with USD/TRY Features

This script:
1. Merges USD/TRY features with existing macro data
2. Updates preprocessing to include USD/TRY
3. Retrains LSTM model
4. Compares performance (before vs after USD/TRY)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

print("="*70)
print("v3.0-ALPHA: LSTM Retraining with USD/TRY Features")
print("="*70)

# Load data
data_processed_dir = project_root / "data" / "processed"
models_dir = project_root / "models"
models_dir.mkdir(parents=True, exist_ok=True)

# Load USD/TRY data
print("\n[1/6] Loading USD/TRY data...")
usd_try_file = data_processed_dir / "bist_macro_merged_v3.csv"
if not usd_try_file.exists():
    print("[ERROR] USD/TRY data not found. Please run v3_alpha_usd_try_integration.py first.")
    sys.exit(1)

usd_try_df = pd.read_csv(usd_try_file)
usd_try_df['Date'] = pd.to_datetime(usd_try_df['Date'])
print(f"[OK] Loaded USD/TRY data: {len(usd_try_df):,} records")
print(f"   Features: {[col for col in usd_try_df.columns if 'USD' in col]}")

# Load existing macro data (if exists)
print("\n[2/6] Loading existing macro data...")
existing_macro_file = data_processed_dir / "bist_macro_merged.csv"

if existing_macro_file.exists():
    existing_macro = pd.read_csv(existing_macro_file)
    existing_macro['Date'] = pd.to_datetime(existing_macro['Date'])
    print(f"[OK] Found existing macro data: {len(existing_macro):,} records")
    print(f"   Columns: {existing_macro.columns.tolist()}")
    
    # Merge USD/TRY with existing macro
    merged_macro = existing_macro.merge(
        usd_try_df,
        on='Date',
        how='outer'
    )
    merged_macro = merged_macro.sort_values('Date').reset_index(drop=True)
    merged_macro = merged_macro.ffill().bfill()
    print(f"[OK] Merged with existing macro data")
else:
    merged_macro = usd_try_df.copy()
    print(f"[INFO] No existing macro data, using USD/TRY only")

# Save merged macro data
merged_macro_file = data_processed_dir / "bist_macro_merged_v3.csv"
merged_macro.to_csv(merged_macro_file, index=False)
print(f"[OK] Saved merged macro data: {merged_macro_file.name}")
print(f"   Total records: {len(merged_macro):,}")
print(f"   Total features: {len(merged_macro.columns) - 1}")  # Exclude Date

# Load full features dataset
print("\n[3/6] Loading full features dataset...")
full_features_file = data_processed_dir / "bist_features_full.csv"

if not full_features_file.exists():
    print("[ERROR] Full features file not found. Please run preprocessing first.")
    sys.exit(1)

full_features = pd.read_csv(full_features_file)
full_features['Date'] = pd.to_datetime(full_features['Date'])
full_features = full_features.sort_values('Date').reset_index(drop=True)

print(f"[OK] Loaded full features: {len(full_features):,} records")
print(f"   Original features: {len(full_features.columns) - 1}")  # Exclude Date

# Merge USD/TRY features with full features
print("\n[4/6] Merging USD/TRY features with full features...")

# Ensure Date columns are same type (timezone-naive)
if full_features['Date'].dtype == 'object':
    full_features['Date'] = pd.to_datetime(full_features['Date'], errors='coerce')
if hasattr(full_features['Date'].dtype, 'tz') and full_features['Date'].dtype.tz is not None:
    full_features['Date'] = full_features['Date'].dt.tz_localize(None)
else:
    # Convert to date then back to datetime to remove timezone
    full_features['Date'] = pd.to_datetime(full_features['Date'].dt.date)

merged_macro['Date'] = pd.to_datetime(merged_macro['Date'])
if hasattr(merged_macro['Date'].dtype, 'tz') and merged_macro['Date'].dtype.tz is not None:
    merged_macro['Date'] = merged_macro['Date'].dt.tz_localize(None)

# Get USD/TRY feature columns (exclude Date)
usd_try_cols = [col for col in merged_macro.columns if col != 'Date']

# Merge on Date
features_with_usd = full_features.merge(
    merged_macro[['Date'] + usd_try_cols],
    on='Date',
    how='left'
)

# Forward fill and backward fill missing values
features_with_usd[usd_try_cols] = features_with_usd[usd_try_cols].ffill().bfill()

print(f"[OK] Merged USD/TRY features")
print(f"   New feature count: {len(features_with_usd.columns) - 1}")
print(f"   Added features: {', '.join(usd_try_cols)}")

# Save updated features
updated_features_file = data_processed_dir / "bist_features_full_v3.csv"
features_with_usd.to_csv(updated_features_file, index=False)
print(f"[OK] Saved updated features: {updated_features_file.name}")

# Now prepare for LSTM training
print("\n[5/6] Preparing data for LSTM retraining...")

# Get feature columns (exclude Date and targets)
exclude_cols = ['Date']
if 'Ticker' in features_with_usd.columns:
    exclude_cols.append('Ticker')

target_cols = [col for col in features_with_usd.columns if col.startswith('Target_')]
feature_cols = [col for col in features_with_usd.columns if col not in exclude_cols + target_cols]

print(f"   Total features: {len(feature_cols)}")
print(f"   USD/TRY features included: {sum(1 for col in feature_cols if 'USD' in col)}")

# Create X and y
X = features_with_usd[feature_cols].copy()
y = features_with_usd['Target_Direction'].copy()

# Handle infinity and NaN
X = X.replace([np.inf, -np.inf], np.nan)
X = X.dropna()

# Update y to match
y = y.iloc[X.index]

# Split data (time-series aware)
split_idx = int(len(X) * 0.8)
X_train_raw = X.iloc[:split_idx].copy()
X_test_raw = X.iloc[split_idx:].copy()
y_train = y.iloc[:split_idx].copy()
y_test = y.iloc[split_idx:].copy()

print(f"   Training samples: {len(X_train_raw):,}")
print(f"   Test samples: {len(X_test_raw):,}")

# Scale features (fit only on training)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train_raw),
    columns=X_train_raw.columns,
    index=X_train_raw.index
)

X_test_scaled = pd.DataFrame(
    scaler.transform(X_test_raw),
    columns=X_test_raw.columns,
    index=X_test_raw.index
)

print(f"[OK] Features scaled (scaler fitted only on training data)")

# Save scaled data for LSTM
X_train_scaled.to_csv(data_processed_dir / "X_train_v3.csv", index=False)
X_test_scaled.to_csv(data_processed_dir / "X_test_v3.csv", index=False)
y_train.to_csv(data_processed_dir / "y_train_v3.csv", index=False)
y_test.to_csv(data_processed_dir / "y_test_v3.csv", index=False)

# Save scaler
import joblib
scaler_path = data_processed_dir / "lstm_scaler_v3.pkl"
joblib.dump(scaler, scaler_path)
print(f"[OK] Saved scaler: {scaler_path.name}")

print("\n[6/6] Data preparation complete!")
print(f"[OK] Ready for LSTM training with {len(feature_cols)} features (including USD/TRY)")

print("\n" + "="*70)
print("[SUCCESS] Data Preparation Complete!")
print("="*70)
print(f"\nNext: Run LSTM training with new features")
print(f"   Training samples: {len(X_train_scaled):,}")
print(f"   Test samples: {len(X_test_scaled):,}")
print(f"   Features: {len(feature_cols)} (including {sum(1 for col in feature_cols if 'USD' in col)} USD/TRY features)")
