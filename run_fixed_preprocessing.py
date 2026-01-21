"""
Fixed Preprocessing Script - Split Before Scaling

This script implements the corrected preprocessing approach:
1. Split data BEFORE scaling (prevents data leakage)
2. Fit scaler ONLY on training data
3. Transform test data using training statistics
4. Save scaler for future use
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("FIXED PREPROCESSING: Split Before Scaling (No Data Leakage)")
print("="*70)

# Setup paths
project_root = Path(__file__).parent
data_raw_dir = project_root / "data" / "raw"
data_processed_dir = project_root / "data" / "processed"
data_processed_dir.mkdir(parents=True, exist_ok=True)

# Load full features dataset
print("\n[1/6] Loading full features dataset...")
full_features_file = data_processed_dir / "bist_features_full.csv"

if not full_features_file.exists():
    print(f"[ERROR] Full features file not found: {full_features_file}")
    print("   Please run notebooks/03_data_preprocessing.ipynb first to create features.")
    sys.exit(1)

df_features = pd.read_csv(full_features_file)
df_features['Date'] = pd.to_datetime(df_features['Date'])
df_features = df_features.sort_values('Date').reset_index(drop=True)

print(f"   Loaded {len(df_features):,} rows")
print(f"   Date range: {df_features['Date'].min().date()} to {df_features['Date'].max().date()}")

# Select features
print("\n[2/6] Selecting features...")
exclude_cols = ['Date']
if 'Ticker' in df_features.columns:
    exclude_cols.append('Ticker')

target_cols = [col for col in df_features.columns if col.startswith('Target_')]
feature_cols = [col for col in df_features.columns if col not in exclude_cols + target_cols]

print(f"   Features: {len(feature_cols)}")
print(f"   Targets: {len(target_cols)}")

# Create feature matrix and targets
X = df_features[feature_cols].copy()
y_return = df_features['Target_Return'].copy()
y_direction = df_features['Target_Direction'].copy()
y_volatility = df_features['Target_Volatility'].copy()
y_class = df_features['Target_Class'].copy()

# Handle infinity and NaN values
print("\n[3/6] Cleaning data (removing infinity and NaN)...")
X = X.replace([np.inf, -np.inf], np.nan)
initial_rows = len(X)
X = X.dropna()
rows_removed = initial_rows - len(X)
print(f"   Removed {rows_removed} rows with NaN/infinity")

# Update targets to match
y_return = y_return.iloc[X.index]
y_direction = y_direction.iloc[X.index]
y_volatility = y_volatility.iloc[X.index]
y_class = y_class.iloc[X.index]

# CRITICAL FIX: Split BEFORE scaling
print("\n[4/6] Splitting data (BEFORE scaling - prevents data leakage)...")
split_idx = int(len(X) * 0.8)

# Split features and targets
X_train_raw = X.iloc[:split_idx].copy()
X_test_raw = X.iloc[split_idx:].copy()

y_train_return = y_return.iloc[:split_idx].copy()
y_test_return = y_return.iloc[split_idx:].copy()

y_train_direction = y_direction.iloc[:split_idx].copy()
y_test_direction = y_direction.iloc[split_idx:].copy()

y_train_volatility = y_volatility.iloc[:split_idx].copy()
y_test_volatility = y_volatility.iloc[split_idx:].copy()

y_train_class = y_class.iloc[:split_idx].copy()
y_test_class = y_class.iloc[split_idx:].copy()

train_dates = df_features['Date'].iloc[X.index[:split_idx]]
test_dates = df_features['Date'].iloc[X.index[split_idx:]]

print(f"   Training set: {len(X_train_raw):,} samples")
print(f"   Training date range: {train_dates.min().date()} to {train_dates.max().date()}")
print(f"   Test set: {len(X_test_raw):,} samples")
print(f"   Test date range: {test_dates.min().date()} to {test_dates.max().date()}")

# NOW scale - CRITICAL: Fit scaler ONLY on training data
print("\n[5/6] Scaling features (fitting scaler ONLY on training data)...")
scaler = StandardScaler()

# Fit on training data only
X_train_scaled_array = scaler.fit_transform(X_train_raw)
X_train_scaled = pd.DataFrame(
    X_train_scaled_array,
    columns=X_train_raw.columns,
    index=X_train_raw.index
)

# Transform test data using training statistics (don't fit!)
X_test_scaled_array = scaler.transform(X_test_raw)
X_test_scaled = pd.DataFrame(
    X_test_scaled_array,
    columns=X_test_raw.columns,
    index=X_test_raw.index
)

# Verify transformation
print(f"   Verifying transformation...")
train_mean_check = X_train_scaled.mean().abs().max()
train_std_check = abs(X_train_scaled.std().max() - 1.0)
test_mean_check = X_test_scaled.mean().abs().max()

print(f"   Training mean (should be ~0): {train_mean_check:.6f}")
print(f"   Training std (should be ~1): {X_train_scaled.std().max():.4f}")
print(f"   Test mean (should be small): {test_mean_check:.6f}")

if test_mean_check > 10:
    print(f"   [WARNING] Test mean is high - checking transformation...")
    # Re-transform to be sure
    X_test_scaled_array = scaler.transform(X_test_raw)
    X_test_scaled = pd.DataFrame(
        X_test_scaled_array,
        columns=X_test_raw.columns,
        index=X_test_raw.index
    )
    test_mean_check = X_test_scaled.mean().abs().max()
    print(f"   After re-transform, test mean: {test_mean_check:.6f}")

print(f"   [OK] Scaler fitted ONLY on training data")
print(f"   [OK] Test data transformed using training statistics")
print(f"   Training scaled shape: {X_train_scaled.shape}")
print(f"   Test scaled shape: {X_test_scaled.shape}")

# Verify scaling
train_mean = X_train_scaled.mean().abs().max()
train_std = X_train_scaled.std().abs().max()
print(f"   Training scaled stats: mean_max={train_mean:.4f}, std_max={train_std:.4f}")

# Save data
print("\n[6/6] Saving processed data...")

# Save train/test splits
X_train_scaled.to_csv(data_processed_dir / "X_train.csv", index=False)
X_test_scaled.to_csv(data_processed_dir / "X_test.csv", index=False)
print(f"   [OK] X_train.csv, X_test.csv saved")

# Save targets
targets_train = pd.DataFrame({
    'Target_Return': y_train_return,
    'Target_Direction': y_train_direction,
    'Target_Volatility': y_train_volatility,
    'Target_Class': y_train_class
})
targets_test = pd.DataFrame({
    'Target_Return': y_test_return,
    'Target_Direction': y_test_direction,
    'Target_Volatility': y_test_volatility,
    'Target_Class': y_test_class
})

targets_train.to_csv(data_processed_dir / "y_train.csv", index=False)
targets_test.to_csv(data_processed_dir / "y_test.csv", index=False)
print(f"   [OK] y_train.csv, y_test.csv saved")

# CRITICAL: Save the scaler
scaler_path = data_processed_dir / "feature_scaler.pkl"
joblib.dump(scaler, scaler_path)
print(f"   [OK] Scaler saved: feature_scaler.pkl")

print("\n" + "="*70)
print("[SUCCESS] Preprocessing complete with proper validation!")
print("="*70)
print(f"\nSummary:")
print(f"   Training samples: {len(X_train_scaled):,}")
print(f"   Test samples: {len(X_test_scaled):,}")
print(f"   Features: {X_train_scaled.shape[1]}")
print(f"   Scaler: Fitted ONLY on training data (no data leakage)")
print(f"\n[OK] Ready for model training!")
