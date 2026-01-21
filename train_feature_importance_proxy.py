"""
Train Feature Importance Proxy Model

Trains a lightweight XGBoost Regressor using the exact same 75 features
as LSTM v3.0 to provide interpretable feature importance.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("Training Feature Importance Proxy Model (XGBoost)")
print("="*70)

# Setup paths
project_root = Path(__file__).parent
data_processed_dir = project_root / "data" / "processed"
models_dir = project_root / "models"
models_dir.mkdir(parents=True, exist_ok=True)

# Load v3.0 data
print("\n[1/5] Loading v3.0 data...")
X_train_file = data_processed_dir / "X_train_v3.csv"
X_test_file = data_processed_dir / "X_test_v3.csv"
y_train_file = data_processed_dir / "y_train_v3.csv"
y_test_file = data_processed_dir / "y_test_v3.csv"

if not all([f.exists() for f in [X_train_file, X_test_file, y_train_file, y_test_file]]):
    print("[ERROR] v3.0 data files not found. Please run v3_alpha_lstm_retraining.py first.")
    sys.exit(1)

X_train = pd.read_csv(X_train_file)
X_test = pd.read_csv(X_test_file)
y_train_df = pd.read_csv(y_train_file)
y_test_df = pd.read_csv(y_test_file)

# Get target (use Target_Direction for classification, but we'll train as regressor for importance)
y_train = y_train_df['Target_Direction'].values
y_test = y_test_df['Target_Direction'].values

print(f"[OK] Loaded data")
print(f"   Training samples: {len(X_train):,}")
print(f"   Test samples: {len(X_test):,}")
print(f"   Features: {X_train.shape[1]}")

# Get feature names
feature_names = list(X_train.columns)
print(f"\n[2/5] Feature names extracted: {len(feature_names)} features")
print(f"   USD/TRY features: {sum(1 for f in feature_names if 'USD' in f)}")
print(f"   Macro features: {sum(1 for f in feature_names if any(x in f for x in ['Inflation', 'Interest']))}")

# Train XGBoost Regressor (using regression for feature importance)
print("\n[3/5] Training XGBoost Regressor...")
print("   Note: Using regression to predict target values for feature importance analysis")

proxy_model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    objective='reg:squarederror'
)

proxy_model.fit(X_train, y_train)

print("[OK] Model trained")

# Evaluate proxy model
print("\n[4/5] Evaluating proxy model...")
y_train_pred = proxy_model.predict(X_train)
y_test_pred = proxy_model.predict(X_test)

train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"   Training MSE: {train_mse:.4f}, R²: {train_r2:.4f}")
print(f"   Test MSE: {test_mse:.4f}, R²: {test_r2:.4f}")

# Get feature importance
feature_importance = proxy_model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
}).sort_values('Importance', ascending=False)

print(f"\n[5/5] Feature importance calculated")
print(f"   Top 5 features:")
for idx, row in importance_df.head(5).iterrows():
    print(f"      {row['Feature']:30s}: {row['Importance']:.4f}")

# Save model and feature names
print("\n[Saving] Saving proxy model and feature names...")
model_path = models_dir / "feature_importance_proxy.pkl"
feature_names_path = models_dir / "feature_names_v3.pkl"

joblib.dump(proxy_model, model_path)
joblib.dump(feature_names, feature_names_path)

print(f"[OK] Model saved: {model_path.name}")
print(f"[OK] Feature names saved: {feature_names_path.name}")

# Save importance dataframe for reference
importance_path = models_dir / "feature_importance_v3.csv"
importance_df.to_csv(importance_path, index=False)
print(f"[OK] Feature importance saved: {importance_path.name}")

print("\n" + "="*70)
print("[SUCCESS] Feature Importance Proxy Model Training Complete!")
print("="*70)
print(f"\nModel Details:")
print(f"   Features: {len(feature_names)}")
print(f"   Top Feature: {importance_df.iloc[0]['Feature']} ({importance_df.iloc[0]['Importance']:.4f})")
print(f"   Model saved: {model_path}")
print(f"   Feature names saved: {feature_names_path}")
print(f"\n[OK] Ready for use in Streamlit app!")
