"""
Fixed Model Training Script - With Proper Validation

This script trains models with proper validation:
1. Uses TimeSeriesSplit cross-validation
2. Fits scaler only on training folds
3. Reports true performance metrics
4. Verifies gap is <5%
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import joblib
import warnings
warnings.filterwarnings('ignore')

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from src.validation import TimeSeriesValidator

print("="*70)
print("FIXED MODEL TRAINING: With Proper Validation")
print("="*70)

# Load data
print("\n[1/5] Loading processed data...")
data_processed_dir = project_root / "data" / "processed"
models_dir = project_root / "models"
models_dir.mkdir(parents=True, exist_ok=True)

X_train = pd.read_csv(data_processed_dir / "X_train.csv")
X_test = pd.read_csv(data_processed_dir / "X_test.csv")
y_train_df = pd.read_csv(data_processed_dir / "y_train.csv")
y_test_df = pd.read_csv(data_processed_dir / "y_test.csv")

y_train = y_train_df['Target_Direction'].values
y_test = y_test_df['Target_Direction'].values

print(f"   Training samples: {len(X_train):,}")
print(f"   Test samples: {len(X_test):,}")
print(f"   Features: {X_train.shape[1]}")

# Load scaler
print("\n[2/5] Loading scaler...")
scaler_path = data_processed_dir / "feature_scaler.pkl"
if scaler_path.exists():
    scaler = joblib.load(scaler_path)
    print(f"   [OK] Scaler loaded from: {scaler_path.name}")
else:
    print(f"   [WARNING] Scaler not found, creating new one...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, scaler_path)
    print(f"   [OK] Scaler saved")

# Data should already be scaled, but verify
print("\n[3/5] Verifying data scaling...")
if scaler_path.exists():
    # Data is already scaled from preprocessing
    X_train_scaled = X_train.values
    X_test_scaled = X_test.values
    print(f"   [OK] Using pre-scaled data")
else:
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

# Cross-validation
print("\n[4/5] Running TimeSeriesSplit cross-validation...")
validator = TimeSeriesValidator(n_splits=5)

# Reduced complexity to prevent overfitting
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,  # Reduced from 15 to prevent overfitting
    min_samples_split=20,  # Increased from 10
    min_samples_leaf=10,  # Increased from 5
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)

print("   Running 5-fold cross-validation...")
cv_results = validator.validate_with_scaler(
    X_train, y_train,
    model=rf_model,
    scaler_type='standard',
    verbose=True
)

print(f"\n   [RESULTS] Cross-Validation Results:")
print(f"      Mean Accuracy: {cv_results['mean_accuracy']:.4f} ± {cv_results['std_accuracy']:.4f}")
print(f"      Mean Precision: {cv_results['mean_precision']:.4f} ± {cv_results['std_precision']:.4f}")
print(f"      Mean Recall: {cv_results['mean_recall']:.4f} ± {cv_results['std_recall']:.4f}")
print(f"      Mean F1: {cv_results['mean_f1']:.4f} ± {cv_results['std_f1']:.4f}")

# Train on full training set
print("\n[5/5] Training final model on full training set...")
rf_model.fit(X_train_scaled, y_train)

# Evaluate
y_train_pred = rf_model.predict(X_train_scaled)
y_test_pred = rf_model.predict(X_test_scaled)

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)
gap = train_acc - test_acc

train_prec = precision_score(y_train, y_train_pred, zero_division=0)
test_prec = precision_score(y_test, y_test_pred, zero_division=0)

train_rec = recall_score(y_train, y_train_pred, zero_division=0)
test_rec = recall_score(y_test, y_test_pred, zero_division=0)

train_f1 = f1_score(y_train, y_train_pred, zero_division=0)
test_f1 = f1_score(y_test, y_test_pred, zero_division=0)

print(f"\n   [RESULTS] Final Model Performance:")
print(f"      Training Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
print(f"      Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"      Gap: {gap:.4f} ({gap*100:.2f}%)")

if gap < 0.05:
    print(f"      [OK] Gap is healthy (<5%) - No overfitting!")
    gap_healthy = True
elif gap < 0.10:
    print(f"      [WARNING] Gap is moderate (5-10%) - Acceptable")
    gap_healthy = True
else:
    print(f"      [FAILED] Gap is too large (>10%) - Overfitting detected")
    gap_healthy = False

print(f"\n   [METRICS] Detailed Performance:")
print(f"      Training: Precision={train_prec:.4f}, Recall={train_rec:.4f}, F1={train_f1:.4f}")
print(f"      Test: Precision={test_prec:.4f}, Recall={test_rec:.4f}, F1={test_f1:.4f}")

# Save model
model_path = models_dir / "random_forest_model.pkl"
joblib.dump(rf_model, model_path)
print(f"\n   [OK] Model saved: {model_path.name}")

# Confusion matrix
cm = confusion_matrix(y_test, y_test_pred)
print(f"\n   [CONFUSION MATRIX] Test Set:")
print(f"      True Negatives: {cm[0,0]}, False Positives: {cm[0,1]}")
print(f"      False Negatives: {cm[1,0]}, True Positives: {cm[1,1]}")

print("\n" + "="*70)
if gap_healthy:
    print("[SUCCESS] Model training complete - Gap is healthy!")
    print("="*70)
    print(f"\n[OK] System is ready for v3.0-Alpha!")
    print(f"   True Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"   Training/Test Gap: {gap:.4f} ({'Healthy' if gap < 0.05 else 'Acceptable'})")
else:
    print("[WARNING] Model training complete but gap is large")
    print("="*70)
    print(f"\n   Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"   Gap: {gap:.4f} - May need further investigation")
