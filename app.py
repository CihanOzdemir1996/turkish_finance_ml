"""
BIST-100 Price Direction Prediction - Streamlit Web Application v2.0

A clean dashboard for predicting next-day stock price direction using LSTM (PyTorch) deep learning model.

Features:
- LSTM v2.0 model with 52.46% accuracy (outperforms XGBoost by 3.68%)
- 30-day sequence lookback for temporal pattern recognition
- 70+ technical indicators + lagged macroeconomic features (Inflation, Interest Rates with 1M/3M lags)
- Real-time predictions with confidence scores
- Automatic fallback to XGBoost if LSTM unavailable
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import json
warnings.filterwarnings('ignore')

# PyTorch for LSTM model
try:
    import torch
    import torch.nn as nn
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="BIST-100 AI Prediction Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #1f77b4;
        margin: 1rem 0;
    }
    .up-prediction {
        background-color: #d4edda;
        border-color: #28a745;
    }
    .down-prediction {
        background-color: #f8d7da;
        border-color: #dc3545;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = None
if 'prediction_data' not in st.session_state:
    st.session_state.prediction_data = None

# LSTM Model class (must match training)
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

@st.cache_data
def load_model():
    """Load the trained model (LSTM v2.0 or fallback to XGBoost)"""
    project_root = Path(__file__).parent
    models_dir = project_root / "models"
    
    # Try to load LSTM v2.0 model first
    best_model_v2_path = models_dir / "best_model_v2.pkl"
    if best_model_v2_path.exists():
        try:
            with open(best_model_v2_path, 'r') as f:
                model_info = json.load(f)
            
            if model_info.get('model_type') == 'lstm' and PYTORCH_AVAILABLE:
                lstm_model_path = models_dir / Path(model_info['model_path']).name
                model_info_path = models_dir / "lstm_model_info.json"
                
                if lstm_model_path.exists() and model_info_path.exists():
                    # Load model info
                    with open(model_info_path, 'r') as f:
                        lstm_info = json.load(f)
                    
                    # Create model
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    model = LSTMModel(
                        input_size=lstm_info['input_size'],
                        hidden_size1=lstm_info['hidden_size1'],
                        hidden_size2=lstm_info['hidden_size2'],
                        hidden_size3=lstm_info['hidden_size3'],
                        dropout=lstm_info['dropout']
                    ).to(device)
                    
                    # Load weights
                    model.load_state_dict(torch.load(lstm_model_path, map_location=device))
                    model.eval()
                    
                    return {
                        'model': model,
                        'model_name': 'LSTM v2.0 (PyTorch)',
                        'model_type': 'lstm',
                        'device': device,
                        'sequence_length': lstm_info['sequence_length']
                    }
        except Exception as e:
            st.warning(f"Could not load LSTM model: {e}. Falling back to XGBoost.")
    
    # Fallback to XGBoost
    xgb_path = models_dir / "xgboost_model.pkl"
    if xgb_path.exists():
        model = joblib.load(xgb_path)
        return {
            'model': model,
            'model_name': 'XGBoost',
            'model_type': 'xgboost'
        }
    
    # Fallback to best model
    best_model_path = models_dir / "best_model.pkl"
    if best_model_path.exists():
        model = joblib.load(best_model_path)
        return {
            'model': model,
            'model_name': 'Best Model',
            'model_type': 'xgboost'
        }
    
    # Fallback to Random Forest
    rf_path = models_dir / "random_forest_model.pkl"
    if rf_path.exists():
        model = joblib.load(rf_path)
        return {
            'model': model,
            'model_name': 'Random Forest',
            'model_type': 'xgboost'
        }
    
    raise FileNotFoundError("No trained model found. Please run model training first.")

@st.cache_data
def load_latest_data():
    """Load the latest processed data for prediction"""
    project_root = Path(__file__).parent
    data_processed_dir = project_root / "data" / "processed"
    
    X_test = pd.read_csv(data_processed_dir / "X_test.csv")
    y_test_df = pd.read_csv(data_processed_dir / "y_test.csv")
    
    # Also load full features for LSTM sequence creation
    full_features_file = data_processed_dir / "bist_features_full.csv"
    full_features = None
    if full_features_file.exists():
        full_features = pd.read_csv(full_features_file)
        full_features['Date'] = pd.to_datetime(full_features['Date'])
        full_features = full_features.sort_values('Date').reset_index(drop=True)
    
    # Get the most recent sample (last row)
    latest_features = X_test.iloc[-1:].copy()
    
    return latest_features, X_test, y_test_df, full_features

@st.cache_data
def get_feature_importance(_model_dict, feature_names):
    """Get feature importance from the model"""
    model = _model_dict['model']
    model_type = _model_dict.get('model_type', 'xgboost')
    
    if model_type == 'lstm':
        # LSTM doesn't have direct feature importance, return None
        return None
    
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        return importance_df
    return None

@st.cache_data
def load_scaler():
    """Load the scaler used for LSTM"""
    project_root = Path(__file__).parent
    models_dir = project_root / "models"
    scaler_path = models_dir / "lstm_scaler.pkl"
    if scaler_path.exists():
        return joblib.load(scaler_path)
    return None

def create_sequence_for_prediction(full_features, scaler, sequence_length=30):
    """Create a sequence from the latest data for LSTM prediction"""
    if full_features is None or scaler is None:
        return None
    
    # Get feature columns (exclude Date, Ticker, targets)
    exclude_cols = ['Date']
    if 'Ticker' in full_features.columns:
        exclude_cols.append('Ticker')
    target_cols = [col for col in full_features.columns if col.startswith('Target_')]
    feature_cols = [col for col in full_features.columns if col not in exclude_cols + target_cols]
    
    # Get last sequence_length rows
    X = full_features[feature_cols].iloc[-sequence_length:].copy()
    
    # Scale
    X_scaled = pd.DataFrame(
        scaler.transform(X),
        columns=X.columns,
        index=X.index
    )
    
    # Create sequence
    sequence = X_scaled.values.reshape(1, sequence_length, len(feature_cols))
    return sequence

def make_prediction(model_dict, latest_features, full_features=None):
    """Make prediction using the model (supports both LSTM and XGBoost)"""
    model = model_dict['model']
    model_type = model_dict.get('model_type', 'xgboost')
    
    if model_type == 'lstm':
        # LSTM prediction
        device = model_dict['device']
        sequence_length = model_dict['sequence_length']
        scaler = load_scaler()
        
        if scaler is None or full_features is None:
            raise ValueError("Scaler or full features not available for LSTM prediction")
        
        # Create sequence
        sequence = create_sequence_for_prediction(full_features, scaler, sequence_length)
        if sequence is None:
            raise ValueError("Could not create sequence for LSTM prediction")
        
        # Predict
        with torch.no_grad():
            sequence_tensor = torch.FloatTensor(sequence).to(device)
            output = model(sequence_tensor).cpu().numpy()[0][0]
        
        # Convert to binary prediction
        prediction = 1 if output > 0.5 else 0
        prob_up = output * 100
        prob_down = (1 - output) * 100
        confidence = max(prob_up, prob_down)
        
        return prediction, prob_up, prob_down, confidence
    else:
        # XGBoost/Random Forest prediction
        prediction = model.predict(latest_features)[0]
        prediction_proba = model.predict_proba(latest_features)[0]
        
        prob_down = prediction_proba[0] * 100
        prob_up = prediction_proba[1] * 100
        confidence = max(prob_up, prob_down)
        
        return prediction, prob_up, prob_down, confidence

def plot_feature_importance(importance_df, top_n=10):
    """Create feature importance visualization"""
    top_features = importance_df.head(top_n)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
    bars = ax.barh(range(len(top_features)), top_features['Importance'], color=colors)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['Feature'])
    ax.set_xlabel('Feature Importance', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {top_n} Most Important Features for Price Direction Prediction', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels on bars
    for i, (idx, row) in enumerate(top_features.iterrows()):
        ax.text(row['Importance'] + 0.001, i, f'{row["Importance"]:.4f}', 
                va='center', fontsize=9)
    
    plt.tight_layout()
    return fig

def load_feature_importance_chart():
    """Load feature importance chart from reports if available"""
    project_root = Path(__file__).parent
    reports_dir = project_root / "reports"
    
    chart_path = reports_dir / "feature_importance.png"
    if chart_path.exists():
        return chart_path
    return None

# Main application
def main():
    # Header
    st.markdown('<h1 class="main-header">📈 BIST-100 Price Direction Prediction Dashboard</h1>', 
                unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Dashboard Settings")
        st.markdown("### Model Information")
        
        # Model info will be shown after model is loaded in main
        st.info("Model information will be displayed after loading.")
        
        st.markdown("---")
        st.markdown("### About")
        st.info("""
        This dashboard uses a trained machine learning model to predict 
        the next-day price direction (UP/DOWN) for BIST-100 stocks.
        
        **Model Performance (v2.0):**
        - Model: LSTM (PyTorch) Deep Learning
        - Accuracy: 52.46% (outperforms XGBoost by 3.68%)
        - Features: 70+ technical indicators + lagged macro features
        - Sequence Length: 30 days lookback
        """)
        
        st.markdown("---")
        st.markdown("### ⚠️ Disclaimer")
        st.warning("""
        This is for educational purposes only. 
        Not financial advice. Always do your own research.
        """)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🔮 Latest Prediction")
        
        # Refresh button
        if st.button("🔄 Refresh Data", type="primary", use_container_width=True):
            st.cache_data.clear()
            st.session_state.last_refresh = datetime.now()
            st.rerun()
        
        if st.session_state.last_refresh:
            st.caption(f"Last refreshed: {st.session_state.last_refresh.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data and make prediction
    try:
        latest_features, X_test, y_test_df, full_features = load_latest_data()
        prediction, prob_up, prob_down, confidence = make_prediction(model_dict, latest_features, full_features)
        
        # Determine confidence level
        if confidence >= 70:
            confidence_level = "HIGH"
            confidence_color = "green"
        elif confidence >= 60:
            confidence_level = "MEDIUM"
            confidence_color = "orange"
        else:
            confidence_level = "LOW"
            confidence_color = "red"
        
        # Display prediction
        prediction_class = "up-prediction" if prediction == 1 else "down-prediction"
        direction_emoji = "⬆️" if prediction == 1 else "⬇️"
        direction_text = "UP" if prediction == 1 else "DOWN"
        
        st.markdown(f"""
        <div class="prediction-box {prediction_class}">
            <h2 style="text-align: center; margin: 0;">
                {direction_emoji} Predicted Direction: <strong>{direction_text}</strong>
            </h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Probability of UP",
                value=f"{prob_up:.2f}%",
                delta=f"{prob_up - 50:.2f}%" if prob_up > 50 else None
            )
        
        with col2:
            st.metric(
                label="Probability of DOWN",
                value=f"{prob_down:.2f}%",
                delta=f"{prob_down - 50:.2f}%" if prob_down > 50 else None
            )
        
        with col3:
            st.metric(
                label="Confidence Level",
                value=confidence_level,
                delta=f"{confidence:.2f}%"
            )
        
        # Prediction details
        with st.expander("📊 Prediction Details", expanded=False):
            st.write(f"**Model Used:** {model_dict['model_name']}")
            st.write(f"**Model Type:** {model_dict.get('model_type', 'xgboost').upper()}")
            st.write(f"**Prediction:** {direction_text} ({direction_emoji})")
            st.write(f"**Confidence:** {confidence:.2f}% ({confidence_level})")
            st.write(f"**Total Features:** {latest_features.shape[1]}")
            st.write(f"**Test Samples Available:** {len(X_test):,}")
            if model_dict.get('model_type') == 'lstm':
                st.write(f"**Sequence Length:** {model_dict.get('sequence_length', 30)} days")
            
            if prediction == 1:
                st.success(f"💡 The model predicts the price will **GO UP** tomorrow with {prob_up:.2f}% confidence.")
            else:
                st.error(f"💡 The model predicts the price will **GO DOWN** tomorrow with {prob_down:.2f}% confidence.")
        
        st.markdown("---")
        
        # Feature Importance Section
        st.header("🏆 Feature Importance Analysis")
        
        # Get feature names
        feature_names = X_test.columns.tolist()
        importance_df = get_feature_importance(model_dict, feature_names)
        
        if importance_df is not None:
            # Display top 10 features
            st.subheader("Top 10 Most Important Features")
            
            # Create visualization
            fig = plot_feature_importance(importance_df, top_n=10)
            st.pyplot(fig)
            
            # Display as table
            with st.expander("📋 View All Feature Importances", expanded=False):
                st.dataframe(
                    importance_df.style.background_gradient(cmap='viridis', subset=['Importance']),
                    use_container_width=True,
                    height=400
                )
        else:
            st.warning("Feature importance not available for this model type.")
        
        # Try to load saved chart
        chart_path = load_feature_importance_chart()
        if chart_path:
            st.markdown("---")
            st.subheader("📊 Saved Feature Importance Chart")
            st.image(str(chart_path), use_container_width=True)
        
        st.markdown("---")
        
        # Model Performance Section
        st.header("📈 Model Performance Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if model_dict.get('model_type') == 'lstm':
                st.metric("Test Accuracy", "52.46%", delta="+3.68% vs XGBoost")
            else:
                st.metric("Test Accuracy", "~49%", delta="Baseline")
        
        with col2:
            st.metric("Features Used", "70+", delta="Technical indicators")
        
        with col3:
            st.metric("Model Type", model_dict['model_name'], delta=model_dict.get('model_type', 'xgboost').upper())
        
        with col4:
            st.metric("Test Samples", f"{len(X_test):,}", delta="Evaluated")
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #666; padding: 2rem 0;">
            <p><strong>BIST-100 AI Prediction System v2.0</strong></p>
            <p>Built with Streamlit | Powered by LSTM (PyTorch) & XGBoost</p>
            <p style="font-size: 0.9rem;">⚠️ For educational purposes only. Not financial advice.</p>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"❌ Error loading data or making prediction: {str(e)}")
        st.info("Please ensure you have run the preprocessing and model training notebooks first.")

if __name__ == "__main__":
    main()
