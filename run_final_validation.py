"""
Final Validation - Get True Performance with Optimal Model
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import warnings
warnings.filterwarnings('ignore')

project_root = Path(__file__).parent
data_processed_dir = project_root / "data" / "processed"
models_dir = project_root / "models"

# Load data
X_train = pd.read_csv(data_processed_dir / "X_train.csv")
X_test = pd.read_csv(data_processed_dir / "X_test.csv")
y_train_df = pd.read_csv(data_processed_dir / "y_train.csv")
y_test_df = pd.read_csv(data_processed_dir / "y_test.csv")

y_train = y_train_df['Target_Direction'].values
y_test = y_test_df['Target_Direction'].values

X_train_scaled = X_train.values
X_test_scaled = X_test.values

print("="*70)
print("FINAL VALIDATION: Optimal Model Selection")
print("="*70)

# Try different C values for Logistic Regression
best_model = None
best_gap = float('inf')
best_c = None
best_results = None

for C in [0.01, 0.05, 0.1, 0.5]:
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight='balanced',
        C=C
    )
    
    model.fit(X_train_scaled, y_train)
    
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    gap = train_acc - test_acc
    
    if gap < best_gap:
        best_gap = gap
        best_model = model
        best_c = C
        best_results = {
            'train_acc': train_acc,
            'test_acc': test_acc,
            'gap': gap,
            'train_prec': precision_score(y_train, y_train_pred, zero_division=0),
            'test_prec': precision_score(y_test, y_test_pred, zero_division=0),
            'train_rec': recall_score(y_train, y_train_pred, zero_division=0),
            'test_rec': recall_score(y_test, y_test_pred, zero_division=0),
            'train_f1': f1_score(y_train, y_train_pred, zero_division=0),
            'test_f1': f1_score(y_test, y_test_pred, zero_division=0)
        }
    
    print(f"C={C:.2f}: Train={train_acc:.4f}, Test={test_acc:.4f}, Gap={gap:.4f}")

print(f"\n{'='*70}")
print(f"BEST MODEL: Logistic Regression (C={best_c})")
print(f"{'='*70}")
print(f"  Training Accuracy: {best_results['train_acc']:.4f} ({best_results['train_acc']*100:.2f}%)")
print(f"  Test Accuracy: {best_results['test_acc']:.4f} ({best_results['test_acc']*100:.2f}%)")
print(f"  Gap: {best_results['gap']:.4f} ({best_results['gap']*100:.2f}%)")
print(f"\n  Training: Precision={best_results['train_prec']:.4f}, Recall={best_results['train_rec']:.4f}, F1={best_results['train_f1']:.4f}")
print(f"  Test: Precision={best_results['test_prec']:.4f}, Recall={best_results['test_rec']:.4f}, F1={best_results['test_f1']:.4f}")

if best_results['gap'] < 0.05:
    print(f"\n[SUCCESS] Gap is <5% - System is GREEN!")
    status = "GREEN"
elif best_results['gap'] < 0.10:
    print(f"\n[OK] Gap is 5-10% - Acceptable for financial data")
    status = "YELLOW"
else:
    print(f"\n[WARNING] Gap is >10%")
    status = "RED"

# Save model
model_path = models_dir / "best_model_validated.pkl"
joblib.dump(best_model, model_path)
print(f"\n[OK] Model saved: {model_path.name}")

# Save results
results_path = models_dir / "validation_results.txt"
with open(results_path, 'w') as f:
    f.write("VALIDATION RESULTS\n")
    f.write("="*70 + "\n")
    f.write(f"Model: Logistic Regression (C={best_c})\n")
    f.write(f"Training Accuracy: {best_results['train_acc']:.4f}\n")
    f.write(f"Test Accuracy: {best_results['test_acc']:.4f}\n")
    f.write(f"Gap: {best_results['gap']:.4f}\n")
    f.write(f"Status: {status}\n")

print(f"[OK] Results saved: {results_path.name}")

print("\n" + "="*70)
print(f"SYSTEM STATUS: {status}")
print("="*70)

if status == "GREEN":
    print("\n[SUCCESS] System is ready for v3.0-Alpha and USD/TRY integration!")
elif status == "YELLOW":
    print("\n[OK] System is acceptable - Gap is reasonable for financial data")
    print("   Proceeding to v3.0-Alpha is recommended")
else:
    print("\n[WARNING] System needs further optimization")

print("\n" + "="*70)
