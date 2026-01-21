"""
v3.0-Alpha: LSTM Training with USD/TRY Features

Retrains LSTM model with USD/TRY features and compares performance
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("v3.0-ALPHA: LSTM Training with USD/TRY Features")
print("="*70)

# Setup
project_root = Path(__file__).parent
data_processed_dir = project_root / "data" / "processed"
models_dir = project_root / "models"
models_dir.mkdir(parents=True, exist_ok=True)

# Check if v3 data exists
print("\n[1/8] Checking v3 data files...")
X_train_file = data_processed_dir / "X_train_v3.csv"
X_test_file = data_processed_dir / "X_test_v3.csv"
y_train_file = data_processed_dir / "y_train_v3.csv"
y_test_file = data_processed_dir / "y_test_v3.csv"

if not all([f.exists() for f in [X_train_file, X_test_file, y_train_file, y_test_file]]):
    print("[ERROR] v3 data files not found. Please run v3_alpha_lstm_retraining.py first.")
    sys.exit(1)

# Load data
print("\n[2/8] Loading v3 data...")
X_train = pd.read_csv(X_train_file)
X_test = pd.read_csv(X_test_file)
y_train_df = pd.read_csv(y_train_file)
y_test_df = pd.read_csv(y_test_file)

y_train = y_train_df['Target_Direction'].values
y_test = y_test_df['Target_Direction'].values

print(f"[OK] Loaded data")
print(f"   Training: {len(X_train):,} samples, {X_train.shape[1]} features")
print(f"   Test: {len(X_test):,} samples")
print(f"   USD/TRY features: {sum(1 for col in X_train.columns if 'USD' in col)}")

# Load scaler
scaler_path = data_processed_dir / "lstm_scaler_v3.pkl"
scaler = joblib.load(scaler_path)
print(f"[OK] Loaded scaler")

# LSTM Model Class (same as before)
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size1=128, hidden_size2=64, hidden_size3=32, num_layers=1, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size1, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden_size1, hidden_size2, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout2 = nn.Dropout(dropout)
        self.lstm3 = nn.LSTM(hidden_size2, hidden_size3, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout3 = nn.Dropout(dropout)
        self.bn = nn.BatchNorm1d(hidden_size3)
        self.fc1 = nn.Linear(hidden_size3, 32)
        self.dropout_fc = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        lstm_out1, _ = self.lstm1(x)
        lstm_out1 = self.dropout1(lstm_out1)
        lstm_out2, _ = self.lstm2(lstm_out1)
        lstm_out2 = self.dropout2(lstm_out2)
        lstm_out3, _ = self.lstm3(lstm_out2)
        lstm_out3 = lstm_out3[:, -1, :]
        lstm_out3 = self.dropout3(lstm_out3)
        lstm_out3 = self.bn(lstm_out3)
        out = torch.relu(self.fc1(lstm_out3))
        out = self.dropout_fc(out)
        out = torch.relu(self.fc2(out))
        out = self.sigmoid(self.fc3(out))
        return out

# Create sequences for LSTM
print("\n[3/8] Creating sequences for LSTM...")
sequence_length = 30

def create_sequences(X, y, seq_length):
    sequences = []
    targets = []
    for i in range(len(X) - seq_length):
        sequences.append(X.iloc[i:i+seq_length].values)
        targets.append(y.iloc[i+seq_length])
    return np.array(sequences), np.array(targets)

# Combine train and test for sequence creation, then split
X_combined = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)
y_combined = pd.concat([pd.Series(y_train), pd.Series(y_test)], axis=0).reset_index(drop=True)

X_seq, y_seq = create_sequences(X_combined, y_combined, sequence_length)

# Split sequences back to train/test
train_seq_end = len(X_train) - sequence_length
X_train_seq = X_seq[:train_seq_end]
y_train_seq = y_seq[:train_seq_end]
X_test_seq = X_seq[train_seq_end:]
y_test_seq = y_seq[train_seq_end:]

print(f"[OK] Created sequences")
print(f"   Training sequences: {len(X_train_seq):,}")
print(f"   Test sequences: {len(X_test_seq):,}")
print(f"   Sequence length: {sequence_length} days")
print(f"   Features per timestep: {X_train_seq.shape[2]}")

# Convert to tensors
print("\n[4/8] Converting to PyTorch tensors...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"   Device: {device}")

X_train_tensor = torch.FloatTensor(X_train_seq).to(device)
y_train_tensor = torch.FloatTensor(y_train_seq).to(device)
X_test_tensor = torch.FloatTensor(X_test_seq).to(device)
y_test_tensor = torch.FloatTensor(y_test_seq).to(device)

# Create model
print("\n[5/8] Creating LSTM model...")
input_size = X_train_seq.shape[2]
model = LSTMModel(
    input_size=input_size,
    hidden_size1=128,
    hidden_size2=64,
    hidden_size3=32,
    dropout=0.2
).to(device)

print(f"[OK] Model created")
print(f"   Input size: {input_size}")
print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# Training setup
print("\n[6/8] Setting up training...")
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

epochs = 50
batch_size = 32
patience = 10
best_val_loss = float('inf')
patience_counter = 0
best_model_state = None

# Create validation split (20% of training data)
val_split = int(len(X_train_seq) * 0.8)
X_train_fold = X_train_tensor[:val_split]
y_train_fold = y_train_tensor[:val_split]
X_val_fold = X_train_tensor[val_split:]
y_val_fold = y_train_tensor[val_split:]

print(f"   Training samples: {len(X_train_fold):,}")
print(f"   Validation samples: {len(X_val_fold):,}")
print(f"   Test samples: {len(X_test_tensor):,}")

# Training loop
print("\n[7/8] Training LSTM model...")
history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}

for epoch in range(epochs):
    # Training
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    
    for i in range(0, len(X_train_fold), batch_size):
        batch_X = X_train_fold[i:i+batch_size]
        batch_y = y_train_fold[i:i+batch_size]
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs.squeeze(), batch_y)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        predicted = (outputs.squeeze() > 0.5).float()
        train_total += batch_y.size(0)
        train_correct += (predicted == batch_y).sum().item()
    
    # Validation
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for i in range(0, len(X_val_fold), batch_size):
            batch_X = X_val_fold[i:i+batch_size]
            batch_y = y_val_fold[i:i+batch_size]
            
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            
            val_loss += loss.item()
            predicted = (outputs.squeeze() > 0.5).float()
            val_total += batch_y.size(0)
            val_correct += (predicted == batch_y).sum().item()
    
    train_loss /= (len(X_train_fold) // batch_size + 1)
    val_loss /= (len(X_val_fold) // batch_size + 1)
    train_acc = train_correct / train_total
    val_acc = val_correct / val_total
    
    history['loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['accuracy'].append(train_acc)
    history['val_accuracy'].append(val_acc)
    
    scheduler.step(val_loss)
    
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"   Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        best_model_state = model.state_dict().copy()
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"   Early stopping at epoch {epoch+1}")
            model.load_state_dict(best_model_state)
            break

if best_model_state:
    model.load_state_dict(best_model_state)

print("[OK] Training complete!")

# Evaluate on test set
print("\n[8/8] Evaluating on test set...")
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    test_pred = (test_outputs.squeeze() > 0.5).float().cpu().numpy()
    test_proba = test_outputs.squeeze().cpu().numpy()

# Calculate metrics
test_acc = accuracy_score(y_test_seq, test_pred)
test_prec = precision_score(y_test_seq, test_pred, zero_division=0)
test_rec = recall_score(y_test_seq, test_pred, zero_division=0)
test_f1 = f1_score(y_test_seq, test_pred, zero_division=0)
test_auc = roc_auc_score(y_test_seq, test_proba)

print(f"\n{'='*70}")
print("LSTM v3.0-ALPHA PERFORMANCE (WITH USD/TRY)")
print("="*70)
print(f"   Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"   Test Precision: {test_prec:.4f}")
print(f"   Test Recall: {test_rec:.4f}")
print(f"   Test F1-Score: {test_f1:.4f}")
print(f"   Test AUC-ROC: {test_auc:.4f}")

# Load previous LSTM results for comparison
print("\n[COMPARISON] Loading previous LSTM results...")
prev_model_info_file = models_dir / "lstm_model_info.json"
if prev_model_info_file.exists():
    with open(prev_model_info_file, 'r') as f:
        prev_info = json.load(f)
    
    prev_acc = prev_info.get('test_accuracy', 0)
    improvement = test_acc - prev_acc
    
    print(f"\n{'='*70}")
    print("PERFORMANCE COMPARISON")
    print("="*70)
    print(f"   Previous LSTM (v2.0): {prev_acc:.4f} ({prev_acc*100:.2f}%)")
    print(f"   New LSTM (v3.0-Alpha): {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"   Improvement: {improvement:.4f} ({improvement*100:.2f}%)")
    
    if improvement > 0:
        print(f"   [SUCCESS] Accuracy improved by {improvement*100:.2f}%!")
    else:
        print(f"   [INFO] Accuracy change: {improvement*100:.2f}%")
else:
    print("[INFO] Previous model info not found for comparison")

# Save model
print("\n[Saving] Saving v3.0-Alpha model...")
model_path = models_dir / "lstm_model_v3.pth"
torch.save(model.state_dict(), model_path)

# Save model info
model_info = {
    'model_type': 'lstm',
    'model_path': str(model_path),
    'input_size': input_size,
    'hidden_size1': 128,
    'hidden_size2': 64,
    'hidden_size3': 32,
    'dropout': 0.2,
    'sequence_length': sequence_length,
    'test_accuracy': float(test_acc),
    'test_precision': float(test_prec),
    'test_recall': float(test_rec),
    'test_f1': float(test_f1),
    'test_auc': float(test_auc),
    'features_count': input_size,
    'usd_try_features': sum(1 for col in X_train.columns if 'USD' in col),
    'version': 'v3.0-Alpha',
    'trained_date': datetime.now().isoformat()
}

model_info_path = models_dir / "lstm_model_info_v3.json"
with open(model_info_path, 'w') as f:
    json.dump(model_info, f, indent=2)

# Save best model reference
best_model_v3_path = models_dir / "best_model_v3.pkl"
with open(best_model_v3_path, 'w') as f:
    json.dump({
        'model_type': 'lstm',
        'model_path': str(model_path),
        'version': 'v3.0-Alpha',
        'accuracy': float(test_acc)
    }, f, indent=2)

print(f"[OK] Model saved: {model_path.name}")
print(f"[OK] Model info saved: {model_info_path.name}")

print("\n" + "="*70)
print("[SUCCESS] LSTM v3.0-Alpha Training Complete!")
print("="*70)
print(f"\nFinal Performance:")
print(f"   Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"   AUC-ROC: {test_auc:.4f}")
print(f"   Features: {input_size} (including USD/TRY)")
print(f"\n[OK] Model ready for deployment!")
