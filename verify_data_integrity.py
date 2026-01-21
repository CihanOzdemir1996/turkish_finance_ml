"""
Verify Data Integrity - Check if data leakage is fixed
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.preprocessing import StandardScaler

print("="*70)
print("DATA INTEGRITY VERIFICATION")
print("="*70)

project_root = Path(__file__).parent
data_processed_dir = project_root / "data" / "processed"

# Load data
X_train = pd.read_csv(data_processed_dir / "X_train.csv")
X_test = pd.read_csv(data_processed_dir / "X_test.csv")

# Load scaler
scaler_path = data_processed_dir / "feature_scaler.pkl"
scaler = joblib.load(scaler_path)

print("\n[1] Checking scaler statistics...")
print(f"   Scaler mean shape: {scaler.mean_.shape}")
print(f"   Scaler scale shape: {scaler.scale_.shape}")

# Check if data is already scaled
print("\n[2] Checking if data is already scaled...")
train_mean = X_train.mean().abs().max()
train_std = X_train.std().abs().max()
test_mean = X_test.mean().abs().max()
test_std = X_test.std().abs().max()

print(f"   Training mean (max abs): {train_mean:.6f}")
print(f"   Training std (max abs): {train_std:.6f}")
print(f"   Test mean (max abs): {test_mean:.6f}")
print(f"   Test std (max abs): {test_std:.6f}")

if train_mean < 0.01 and abs(train_std - 1.0) < 0.1:
    print("   [OK] Training data appears to be scaled (mean~0, std~1)")
else:
    print("   [WARNING] Training data may not be properly scaled")

# Check if test data was transformed with training statistics
print("\n[3] Verifying test data transformation...")
# If test data was transformed with training scaler, its mean won't be exactly 0
# But it should be close to 0
if test_mean < 1.0:  # Reasonable threshold
    print(f"   [OK] Test data mean is reasonable ({test_mean:.4f})")
else:
    print(f"   [WARNING] Test data mean is high ({test_mean:.4f}) - may indicate leakage")

# Check feature ranges
print("\n[4] Checking feature value ranges...")
train_max = X_train.max().max()
train_min = X_train.min().min()
test_max = X_test.max().max()
test_min = X_test.min().min()

print(f"   Training range: [{train_min:.2f}, {train_max:.2f}]")
print(f"   Test range: [{test_min:.2f}, {test_max:.2f}]")

if abs(train_max) < 10 and abs(test_max) < 10:
    print("   [OK] Feature values are in reasonable range for scaled data")
else:
    print("   [WARNING] Feature values are outside expected range")

print("\n" + "="*70)
print("VERIFICATION COMPLETE")
print("="*70)
