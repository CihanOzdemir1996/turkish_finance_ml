"""
BIST-100 Price Direction Prediction - Streamlit Web Application v3.0-Alpha

A clean dashboard for predicting next-day stock price direction using LSTM (PyTorch) deep learning model.

Features:
- LSTM v3.0-Alpha model with USD/TRY exchange rate features (52.00% accuracy)
- 30-day sequence lookback for temporal pattern recognition
- 75 features: 70+ technical indicators + lagged macroeconomic features (Inflation, Interest Rates, USD/TRY with 1M/3M lags)
- Real-time predictions with confidence scores
- USD/TRY trend visualization
- Automatic fallback to v2.0 or XGBoost if v3.0 unavailable
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import json
warnings.filterwarnings('ignore')

# Plotly for interactive charts
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.warning("Plotly not available. Install with: pip install plotly")

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
    """Load the trained model (LSTM v3.0-Alpha, v2.0, or fallback to XGBoost)"""
    project_root = Path(__file__).parent
    models_dir = project_root / "models"
    
    # Try to load LSTM v3.0-Alpha model first
    if PYTORCH_AVAILABLE:
        lstm_info_v3_path = models_dir / "lstm_model_info_v3.json"
        lstm_model_v3_path = models_dir / "lstm_model_v3.pth"
        
        if lstm_info_v3_path.exists() and lstm_model_v3_path.exists():
            try:
                with open(str(lstm_info_v3_path), 'r', encoding='utf-8') as f:
                    model_info = json.load(f)
                
                lstm_model = LSTMModel(
                    input_size=model_info['input_size'],
                    hidden_size1=model_info['hidden_size1'],
                    hidden_size2=model_info['hidden_size2'],
                    hidden_size3=model_info['hidden_size3'],
                    dropout=model_info['dropout']
                )
                
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                lstm_model.load_state_dict(torch.load(str(lstm_model_v3_path), map_location=device))
                lstm_model.eval()
                
                return {
                    'model': lstm_model,
                    'model_name': 'LSTM v3.0-Alpha (PyTorch)',
                    'model_type': 'lstm',
                    'device': device,
                    'sequence_length': model_info['sequence_length'],
                    'version': 'v3.0-Alpha',
                    'accuracy': model_info.get('test_accuracy', 0.52),
                    'features_count': model_info['features_count'],
                    'usd_try_features': model_info.get('usd_try_features', 5)
                }
            except Exception as e:
                st.warning(f"Could not load LSTM v3.0: {str(e)}")
    
    # Try to load LSTM v2.0 model as fallback
    if PYTORCH_AVAILABLE:
        best_model_v2_path = models_dir / "best_model_v2.pkl"
    if best_model_v2_path.exists():
        try:
            with open(best_model_v2_path, 'r', encoding='utf-8') as f:
                model_info = json.load(f)
            
            if model_info.get('model_type') == 'lstm' and PYTORCH_AVAILABLE:
                # Handle both absolute and relative paths for cross-platform compatibility
                model_path_str = model_info.get('model_path', '')
                if model_path_str:
                    # Extract just the filename if it's a full path
                    model_filename = Path(model_path_str).name
                    lstm_model_path = models_dir / model_filename
                else:
                    # Default fallback
                    lstm_model_path = models_dir / "lstm_model_v2.pth"
                
                model_info_path = models_dir / "lstm_model_info.json"
                
                # Check if files exist (cross-platform path handling)
                if lstm_model_path.exists() and model_info_path.exists():
                    # Load model info
                    with open(model_info_path, 'r', encoding='utf-8') as f:
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
                    
                    # Load weights (map_location ensures it works on CPU even if trained on GPU)
                    model.load_state_dict(torch.load(str(lstm_model_path), map_location=device))
                    model.eval()
                    
                    return {
                        'model': model,
                        'model_name': 'LSTM v2.0 (PyTorch)',
                        'model_type': 'lstm',
                        'device': device,
                        'sequence_length': lstm_info.get('sequence_length', 30)
                    }
        except Exception as e:
            # Silently fall back to XGBoost if LSTM loading fails
            pass
    
    # Fallback to XGBoost
    xgb_path = models_dir / "xgboost_model.pkl"
    if xgb_path.exists():
        model = joblib.load(str(xgb_path))  # Use str() for cross-platform compatibility
        return {
            'model': model,
            'model_name': 'XGBoost',
            'model_type': 'xgboost'
        }
    
    # Fallback to best model
    best_model_path = models_dir / "best_model.pkl"
    if best_model_path.exists():
        model = joblib.load(str(best_model_path))
        return {
            'model': model,
            'model_name': 'Best Model',
            'model_type': 'xgboost'
        }
    
    # Fallback to Random Forest
    rf_path = models_dir / "random_forest_model.pkl"
    if rf_path.exists():
        model = joblib.load(str(rf_path))
        return {
            'model': model,
            'model_name': 'Random Forest',
            'model_type': 'xgboost'
        }
    
    raise FileNotFoundError("No trained model found. Please run model training first.")

@st.cache_data
def load_latest_data():
    """Load the latest processed data for prediction with error handling"""
    try:
        project_root = Path(__file__).parent
        data_processed_dir = project_root / "data" / "processed"
        
        # Use str() for cross-platform path compatibility
        X_test = pd.read_csv(str(data_processed_dir / "X_test.csv"))
        y_test_df = pd.read_csv(str(data_processed_dir / "y_test.csv"))
    except FileNotFoundError as e:
        st.error(f"Data files not found: {str(e)}. Please ensure data preprocessing is complete.")
        return None, None, None, None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None, None
    
    # Also load full features for LSTM sequence creation (prioritize v3.0)
    full_features = None
    try:
        full_features_file_v3 = data_processed_dir / "bist_features_full_v3.csv"
        if full_features_file_v3.exists():
            full_features = pd.read_csv(str(full_features_file_v3))
            full_features['Date'] = pd.to_datetime(full_features['Date'])
            full_features = full_features.sort_values('Date').reset_index(drop=True)
        else:
            full_features_file = data_processed_dir / "bist_features_full.csv"
            if full_features_file.exists():
                full_features = pd.read_csv(str(full_features_file))
                full_features['Date'] = pd.to_datetime(full_features['Date'])
                full_features = full_features.sort_values('Date').reset_index(drop=True)
    except Exception as e:
        st.warning(f"Could not load full features: {str(e)}. LSTM predictions may be unavailable.")
    
    # Get the most recent sample (last row)
    latest_features = X_test.iloc[-1:].copy()
    
    return latest_features, X_test, y_test_df, full_features

@st.cache_data
def load_macro_data():
    """Load macroeconomic data for visualization (prioritize v3.0) with error handling"""
    try:
        project_root = Path(__file__).parent
        data_processed_dir = project_root / "data" / "processed"
        
        # Try v3.0 first (includes USD/TRY)
        macro_file_v3 = data_processed_dir / "bist_macro_merged_v3.csv"
        if macro_file_v3.exists():
            macro_df = pd.read_csv(str(macro_file_v3))
            macro_df['Date'] = pd.to_datetime(macro_df['Date'])
            macro_df = macro_df.sort_values('Date').reset_index(drop=True)
            return macro_df
        
        # Fallback to v2.0
        macro_file = data_processed_dir / "bist_macro_merged.csv"
        if macro_file.exists():
            macro_df = pd.read_csv(str(macro_file))
            macro_df['Date'] = pd.to_datetime(macro_df['Date'])
            macro_df = macro_df.sort_values('Date').reset_index(drop=True)
            return macro_df
    except Exception as e:
        st.warning(f"Could not load macroeconomic data: {str(e)}")
    return None

@st.cache_data
def load_price_data():
    """Load BIST-100 price data for technical analysis with error handling"""
    try:
        project_root = Path(__file__).parent
        data_processed_dir = project_root / "data" / "processed"
        
        # Try to load from full features first (has Date and Close)
        full_features_file = data_processed_dir / "bist_features_full.csv"
        if full_features_file.exists():
            df = pd.read_csv(str(full_features_file))
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                # Try to find Close price column
                close_cols = [col for col in df.columns if 'Close' in col or 'close' in col]
                if close_cols:
                    price_df = df[['Date', close_cols[0]]].copy()
                    price_df.columns = ['Date', 'Close']
                    price_df = price_df.sort_values('Date').reset_index(drop=True)
                    return price_df
        
        # Fallback to raw data
        data_raw_dir = project_root / "data" / "raw"
        raw_file = data_raw_dir / "bist_stock_prices.csv"
        if raw_file.exists():
            df = pd.read_csv(str(raw_file))
            if 'Date' in df.columns and 'Close' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                price_df = df[['Date', 'Close']].copy()
                price_df = price_df.sort_values('Date').reset_index(drop=True)
                return price_df
    except Exception as e:
        st.warning(f"Could not load price data: {str(e)}")
    return None


@st.cache_data
def load_scaler():
    """Load the scaler used for LSTM (prioritize v3.0) with error handling"""
    try:
        project_root = Path(__file__).parent
        data_processed_dir = project_root / "data" / "processed"
        
        # Try v3.0 scaler first
        scaler_path_v3 = data_processed_dir / "lstm_scaler_v3.pkl"
        if scaler_path_v3.exists():
            return joblib.load(str(scaler_path_v3))
        
        # Fallback to v2.0
        models_dir = project_root / "models"
        scaler_path = models_dir / "lstm_scaler.pkl"
        if scaler_path.exists():
            return joblib.load(str(scaler_path))
    except Exception as e:
        st.warning(f"Could not load scaler: {str(e)}")
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


# Main application
def main():
    # Header
    st.markdown('<h1 class="main-header">📈 BIST-100 Price Direction Prediction Dashboard</h1>', 
                unsafe_allow_html=True)
    st.markdown("---")
    
    # Load model first (must be before using model_dict)
    try:
        model_dict = load_model()
    except FileNotFoundError as e:
        st.error(f"❌ {str(e)}")
        st.info("Please ensure you have run the model training notebooks first.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.info("Please check that model files exist in the models/ directory.")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Dashboard Settings")
        st.markdown("### Model Information")
        
        model_name = model_dict['model_name']
        model_type = model_dict.get('model_type', 'xgboost')
        st.success(f"✅ Model Loaded: {model_name}")
        if model_type == 'lstm':
            st.info(f"Model Type: LSTM (PyTorch)")
            device_str = str(model_dict.get('device', 'CPU'))
            st.info(f"Device: {device_str}")
            st.info(f"Sequence Length: {model_dict.get('sequence_length', 30)} days")
        else:
            st.info(f"Model Type: {type(model_dict['model']).__name__}")
        
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
        
        # ============================================
        # NEW PROFESSIONAL DASHBOARD SECTIONS
        # ============================================
        
        # 1. Model Performance Showcase
        st.header("📊 Model Performance Comparison")
        # Check if v3.0 is loaded
        model_version = model_dict.get('version', 'v2.0')
        if model_version == 'v3.0-Alpha':
            st.markdown("### LSTM v3.0-Alpha vs v2.0 vs XGBoost")
            # Model comparison data
            model_comparison_data = {
                'Model': ['LSTM v3.0-Alpha (with USD/TRY)', 'LSTM v2.0', 'XGBoost Baseline'],
                'Accuracy': [52.00, 52.46, 48.79],
                'Improvement': [3.21, 3.68, 0.0]
            }
        else:
            st.markdown("### LSTM v2.0 vs XGBoost Baseline")
            # Model comparison data
            model_comparison_data = {
                'Model': ['LSTM v2.0 (PyTorch)', 'XGBoost Baseline'],
                'Accuracy': [52.46, 48.79],
                'Improvement': [3.68, 0.0]
            }
        comparison_df = pd.DataFrame(model_comparison_data)
        
        if PLOTLY_AVAILABLE:
            # Interactive horizontal bar chart
            fig_comparison = go.Figure()
            
            fig_comparison.add_trace(go.Bar(
                y=comparison_df['Model'],
                x=comparison_df['Accuracy'],
                orientation='h',
                marker=dict(
                    color=['#2E86AB', '#A23B72'],
                    line=dict(color='white', width=1)
                ),
                text=[f"{acc:.2f}%" for acc in comparison_df['Accuracy']],
                textposition='outside',
                name='Accuracy'
            ))
            
            fig_comparison.update_layout(
                title='Model Accuracy Comparison',
                xaxis_title='Accuracy (%)',
                yaxis_title='Model',
                height=300,
                showlegend=False,
                margin=dict(l=20, r=20, t=50, b=20),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig_comparison, use_container_width=True)
        else:
            # Fallback to matplotlib
            fig, ax = plt.subplots(figsize=(10, 4))
            colors = ['#2E86AB', '#A23B72']
            bars = ax.barh(comparison_df['Model'], comparison_df['Accuracy'], color=colors)
            ax.set_xlabel('Accuracy (%)', fontsize=12, fontweight='bold')
            ax.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold', pad=20)
            ax.set_xlim(0, 60)
            
            # Add value labels
            for i, (idx, row) in enumerate(comparison_df.iterrows()):
                ax.text(row['Accuracy'] + 0.5, i, f"{row['Accuracy']:.2f}%", 
                       va='center', fontsize=11, fontweight='bold')
            
            # Add improvement annotation
            ax.annotate(f"+{comparison_df.iloc[0]['Improvement']:.2f}% improvement",
                       xy=(52.46, 0), xytext=(55, -0.3),
                       arrowprops=dict(arrowstyle='->', color='green', lw=2),
                       fontsize=10, color='green', fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
        
        # Metrics row
        if model_version == 'v3.0-Alpha':
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("LSTM v3.0 Accuracy", "52.00%", delta="+3.21%", delta_color="normal")
            with col2:
                st.metric("LSTM v2.0 Accuracy", "52.46%", delta="+3.68%", delta_color="normal")
            with col3:
                st.metric("XGBoost Accuracy", "48.79%", delta="Baseline", delta_color="off")
            with col4:
                st.metric("USD/TRY Features", f"{model_dict.get('usd_try_features', 5)}", delta="v3.0", delta_color="normal")
        else:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("LSTM Accuracy", "52.46%", delta="+3.68%", delta_color="normal")
            with col2:
                st.metric("XGBoost Accuracy", "48.79%", delta="Baseline", delta_color="off")
            with col3:
                st.metric("Improvement", "+3.68%", delta="Significant", delta_color="normal")
        
        st.markdown("---")
        
        # 2. Macroeconomic Context
        st.header("🌍 Macroeconomic Context")
        st.markdown("### Inflation & Interest Rate Trends with Lagged Features")
        
        macro_df = load_macro_data()
        if macro_df is not None:
            # Get last 12 months of data
            recent_macro = macro_df.tail(365).copy()  # Last year
            
            if PLOTLY_AVAILABLE:
                # Dual-axis line chart - ensure we're using Plotly
                try:
                    fig_macro = make_subplots(specs=[[{"secondary_y": True}]])
                    
                    # Primary axis: Inflation
                    if 'Inflation_TUFE' in recent_macro.columns:
                        fig_macro.add_trace(
                            go.Scatter(
                                x=recent_macro['Date'],
                                y=recent_macro['Inflation_TUFE'],
                                name='Inflation (TUFE)',
                                line=dict(color='#FF6B6B', width=2),
                                mode='lines+markers'
                            ),
                            secondary_y=False
                        )
                    
                    # Lagged inflation features
                    if 'Inflation_TUFE_Lag_1M' in recent_macro.columns:
                        fig_macro.add_trace(
                            go.Scatter(
                                x=recent_macro['Date'],
                                y=recent_macro['Inflation_TUFE_Lag_1M'],
                                name='Inflation Lag 1M',
                                line=dict(color='#FF6B6B', width=1, dash='dash'),
                                opacity=0.6
                            ),
                            secondary_y=False
                        )
                    
                    if 'Inflation_TUFE_Lag_3M' in recent_macro.columns:
                        fig_macro.add_trace(
                            go.Scatter(
                                x=recent_macro['Date'],
                                y=recent_macro['Inflation_TUFE_Lag_3M'],
                                name='Inflation Lag 3M',
                                line=dict(color='#FF6B6B', width=1, dash='dot'),
                                opacity=0.4
                            ),
                            secondary_y=False
                        )
                
                    # Secondary axis: Interest Rate
                    if 'Interest_Rate' in recent_macro.columns:
                        fig_macro.add_trace(
                            go.Scatter(
                                x=recent_macro['Date'],
                                y=recent_macro['Interest_Rate'],
                                name='Interest Rate',
                                line=dict(color='#4ECDC4', width=2),
                                mode='lines+markers'
                            ),
                            secondary_y=True
                        )
                    
                    # Lagged interest rate features
                    if 'Interest_Rate_Lag_1M' in recent_macro.columns:
                        fig_macro.add_trace(
                            go.Scatter(
                                x=recent_macro['Date'],
                                y=recent_macro['Interest_Rate_Lag_1M'],
                                name='Interest Rate Lag 1M',
                                line=dict(color='#4ECDC4', width=1, dash='dash'),
                                opacity=0.6
                            ),
                            secondary_y=True
                        )
                    
                    if 'Interest_Rate_Lag_3M' in recent_macro.columns:
                        fig_macro.add_trace(
                            go.Scatter(
                                x=recent_macro['Date'],
                                y=recent_macro['Interest_Rate_Lag_3M'],
                                name='Interest Rate Lag 3M',
                                line=dict(color='#4ECDC4', width=1, dash='dot'),
                                opacity=0.4
                            ),
                            secondary_y=True
                        )
                
                    # Update axes - use correct Plotly syntax (update_xaxes, not update_xaxis)
                    fig_macro.update_xaxes(title_text="Date")
                    fig_macro.update_yaxes(title_text="Inflation (TUFE) %", secondary_y=False)
                    fig_macro.update_yaxes(title_text="Interest Rate %", secondary_y=True)
                    fig_macro.update_layout(
                        title='Macroeconomic Indicators: Inflation & Interest Rates with Lagged Features',
                        height=400,
                        hovermode='x unified',
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)'
                    )
                    
                    st.plotly_chart(fig_macro, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating macroeconomic chart: {str(e)}")
                    st.info("Falling back to data table view.")
                    st.dataframe(recent_macro.tail(20), use_container_width=True)
                
                # Latest values table
                st.markdown("#### Latest Macroeconomic Values")
                latest_macro_cols = ['Date']
                if 'Inflation_TUFE' in recent_macro.columns:
                    latest_macro_cols.append('Inflation_TUFE')
                if 'Interest_Rate' in recent_macro.columns:
                    latest_macro_cols.append('Interest_Rate')
                if 'Inflation_TUFE_Lag_1M' in recent_macro.columns:
                    latest_macro_cols.extend(['Inflation_TUFE_Lag_1M', 'Inflation_TUFE_Lag_3M'])
                if 'Interest_Rate_Lag_1M' in recent_macro.columns:
                    latest_macro_cols.extend(['Interest_Rate_Lag_1M', 'Interest_Rate_Lag_3M'])
                
                latest_macro_table = recent_macro[latest_macro_cols].tail(10).copy()
                latest_macro_table['Date'] = latest_macro_table['Date'].dt.strftime('%Y-%m-%d')
                st.dataframe(latest_macro_table.style.format({
                    col: '{:.2f}' for col in latest_macro_table.columns if col != 'Date'
                }), use_container_width=True, hide_index=True)
            else:
                st.info("Plotly not available. Showing data table instead.")
                st.dataframe(recent_macro.tail(20), use_container_width=True)
        else:
            st.warning("Macroeconomic data not available. Run notebook 06_macro_data_integration.ipynb to generate this data.")
        
        st.markdown("---")
        
        # 2.5. USD/TRY Exchange Rate Trend (NEW in v3.0)
        st.header("💵 USD/TRY Exchange Rate Trend (v3.0-Alpha)")
        st.markdown("### Currency Impact on BIST-100")
        
        macro_df = load_macro_data()
        if macro_df is not None and 'USD_TRY' in macro_df.columns:
            # Get last 12 months
            recent_usd = macro_df.tail(365).copy()
            
            if PLOTLY_AVAILABLE:
                fig_usd = make_subplots(specs=[[{"secondary_y": True}]])
                
                # Primary axis: USD/TRY
                fig_usd.add_trace(
                    go.Scatter(
                        x=recent_usd['Date'],
                        y=recent_usd['USD_TRY'],
                        name='USD/TRY (Current)',
                        line=dict(color='#E63946', width=2),
                        mode='lines+markers'
                    ),
                    secondary_y=False
                )
                
                # Lagged features
                if 'USD_TRY_Lag_1M' in recent_usd.columns:
                    fig_usd.add_trace(
                        go.Scatter(
                            x=recent_usd['Date'],
                            y=recent_usd['USD_TRY_Lag_1M'],
                            name='USD/TRY Lag 1M',
                            line=dict(color='#E63946', width=1, dash='dash'),
                            opacity=0.6
                        ),
                        secondary_y=False
                    )
                
                if 'USD_TRY_Lag_3M' in recent_usd.columns:
                    fig_usd.add_trace(
                        go.Scatter(
                            x=recent_usd['Date'],
                            y=recent_usd['USD_TRY_Lag_3M'],
                            name='USD/TRY Lag 3M',
                            line=dict(color='#E63946', width=1, dash='dot'),
                            opacity=0.4
                        ),
                        secondary_y=False
                    )
                
                # Secondary axis: Volatility
                if 'USD_TRY_Volatility' in recent_usd.columns:
                    fig_usd.add_trace(
                        go.Scatter(
                            x=recent_usd['Date'],
                            y=recent_usd['USD_TRY_Volatility'],
                            name='USD/TRY Volatility (30-day)',
                            line=dict(color='#FFB703', width=2),
                            mode='lines',
                            fill='tonexty',
                            fillcolor='rgba(255, 183, 3, 0.1)'
                        ),
                        secondary_y=True
                    )
                
                fig_usd.update_xaxes(title_text="Date")
                fig_usd.update_yaxes(title_text="USD/TRY Exchange Rate", secondary_y=False)
                fig_usd.update_yaxes(title_text="Volatility (%)", secondary_y=True)
                fig_usd.update_layout(
                    title='USD/TRY Exchange Rate with Lagged Features and Volatility',
                    height=400,
                    hovermode='x unified',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                
                st.plotly_chart(fig_usd, use_container_width=True)
                
                # Latest USD/TRY values
                st.markdown("#### Latest USD/TRY Values")
                usd_cols = ['Date', 'USD_TRY']
                if 'USD_TRY_Lag_1M' in recent_usd.columns:
                    usd_cols.extend(['USD_TRY_Lag_1M', 'USD_TRY_Lag_3M'])
                if 'USD_TRY_Change' in recent_usd.columns:
                    usd_cols.append('USD_TRY_Change')
                if 'USD_TRY_Volatility' in recent_usd.columns:
                    usd_cols.append('USD_TRY_Volatility')
                
                latest_usd_table = recent_usd[usd_cols].tail(10).copy()
                latest_usd_table['Date'] = latest_usd_table['Date'].dt.strftime('%Y-%m-%d')
                st.dataframe(latest_usd_table.style.format({
                    col: '{:.2f}' if 'USD_TRY' in col else '{:.4f}' if 'Volatility' in col or 'Change' in col else '{:.2f}'
                    for col in latest_usd_table.columns if col != 'Date'
                }), use_container_width=True, hide_index=True)
            else:
                st.info("Plotly not available. Showing data table instead.")
                st.dataframe(recent_usd[['Date', 'USD_TRY', 'USD_TRY_Lag_1M', 'USD_TRY_Lag_3M', 'USD_TRY_Volatility']].tail(20), use_container_width=True)
        else:
            st.info("USD/TRY data not available. This feature requires v3.0-Alpha data.")
        
        st.markdown("---")
        
        # 3. Technical Analysis Visuals
        st.header("📈 Technical Analysis: BIST-100 Price Trend")
        st.markdown("### Last 60 Days with Moving Average (SMA 20)")
        
        price_df = load_price_data()
        if price_df is not None:
            # Get last 60 days
            recent_prices = price_df.tail(60).copy()
            
            # Calculate SMA 20
            recent_prices['SMA_20'] = recent_prices['Close'].rolling(window=20, min_periods=1).mean()
            
            if PLOTLY_AVAILABLE:
                fig_price = go.Figure()
                
                # Price line
                fig_price.add_trace(go.Scatter(
                    x=recent_prices['Date'],
                    y=recent_prices['Close'],
                    name='BIST-100 Close Price',
                    line=dict(color='#1f77b4', width=2),
                    mode='lines+markers',
                    marker=dict(size=4)
                ))
                
                # SMA 20 line
                fig_price.add_trace(go.Scatter(
                    x=recent_prices['Date'],
                    y=recent_prices['SMA_20'],
                    name='SMA 20',
                    line=dict(color='#ff7f0e', width=2, dash='dash'),
                    mode='lines'
                ))
                
                fig_price.update_layout(
                    title='BIST-100 Closing Price (Last 60 Days) with 20-Day Moving Average',
                    xaxis_title='Date',
                    yaxis_title='Price (TRY)',
                    height=400,
                    hovermode='x unified',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                
                st.plotly_chart(fig_price, use_container_width=True)
            else:
                # Fallback to matplotlib
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(recent_prices['Date'], recent_prices['Close'], 
                       label='BIST-100 Close Price', linewidth=2, color='#1f77b4')
                ax.plot(recent_prices['Date'], recent_prices['SMA_20'], 
                       label='SMA 20', linewidth=2, linestyle='--', color='#ff7f0e')
                ax.set_xlabel('Date', fontsize=12, fontweight='bold')
                ax.set_ylabel('Price (TRY)', fontsize=12, fontweight='bold')
                ax.set_title('BIST-100 Closing Price (Last 60 Days) with 20-Day Moving Average', 
                           fontsize=14, fontweight='bold', pad=20)
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
            
            # Price statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Current Price", f"{recent_prices['Close'].iloc[-1]:.2f} TRY")
            with col2:
                price_change = recent_prices['Close'].iloc[-1] - recent_prices['Close'].iloc[0]
                st.metric("60-Day Change", f"{price_change:.2f} TRY", 
                         delta=f"{(price_change/recent_prices['Close'].iloc[0]*100):.2f}%")
            with col3:
                st.metric("SMA 20", f"{recent_prices['SMA_20'].iloc[-1]:.2f} TRY")
            with col4:
                sma_signal = "Above SMA" if recent_prices['Close'].iloc[-1] > recent_prices['SMA_20'].iloc[-1] else "Below SMA"
                st.metric("Price vs SMA", sma_signal)
        else:
            st.warning("Price data not available. Please ensure data files exist.")
        
        st.markdown("---")
        
        # 4. Feature Insights
        st.header("🔍 Model Architecture & Feature Insights")
        
        with st.expander("📚 How the LSTM Model Works", expanded=False):
            st.markdown("""
            ### Model Architecture
            
            Our **LSTM v2.0** model uses a sophisticated deep learning architecture to capture temporal patterns in financial data:
            
            **Neural Network Structure:**
            - **3-Layer LSTM**: 128 → 64 → 32 hidden units
            - **Sequence Length**: 30 days lookback window
            - **Dropout**: 20% for regularization
            - **Batch Normalization**: For stable training
            - **Total Parameters**: 166,273 trainable weights
            
            **Why LSTM?**
            - **Temporal Memory**: Remembers patterns across 30-day sequences
            - **Non-linear Patterns**: Captures complex relationships traditional ML misses
            - **Feature Learning**: Automatically discovers relevant patterns
            """)
        
        with st.expander("📊 Feature Engineering: 70+ Technical Indicators", expanded=False):
            st.markdown("""
            ### Technical Indicators Used
            
            **Price-Based Features:**
            - Moving Averages: SMA (5, 10, 20, 50, 100, 200 days), EMA (12, 26 days)
            - Price Changes: Daily returns, momentum, rate of change
            - Price Patterns: High-Low spreads, price positions
            
            **Momentum Indicators:**
            - **RSI (Relative Strength Index)**: Overbought/oversold conditions
            - **MACD**: Trend-following momentum indicator
            - **Stochastic Oscillator**: Momentum comparison
            
            **Volatility Measures:**
            - **ATR (Average True Range)**: Market volatility
            - **Bollinger Bands**: Price volatility bands
            - **Rolling Standard Deviation**: Price volatility over time
            
            **Volume Indicators:**
            - Volume ratios and trends
            - Volume-price relationships
            - Volume moving averages
            
            **Lag Features:**
            - Previous day/week values
            - Rolling statistics (mean, std, min, max)
            """)
        
        with st.expander("🌍 Macroeconomic Features with Lagged Effects", expanded=False):
            st.markdown("""
            ### Economic Context Integration
            
            **Primary Macro Features:**
            - **Inflation (TUFE)**: Consumer Price Index - measures purchasing power
            - **Interest Rate**: Central Bank policy rate - affects investment decisions
            
            **Lagged Features (Key Innovation):**
            - **1-Month Lag**: Captures delayed economic impacts (30-day delay)
            - **3-Month Lag**: Captures longer-term economic cycles (90-day delay)
            
            **Why Lags Matter:**
            - Economic policies take time to affect markets
            - Inflation changes don't immediately impact stock prices
            - Interest rate changes have delayed effects on investment
            - Historical economic conditions influence current market sentiment
            
            **Example:**
            - If inflation was high 3 months ago, it may still affect current market behavior
            - Interest rate changes from last month influence today's investment decisions
            """)
        
        with st.expander("🧠 How Features Combine for Predictions", expanded=False):
            st.markdown("""
            ### Non-Linear Pattern Recognition
            
            **The LSTM Advantage:**
            
            1. **Temporal Sequences**: 
               - Analyzes 30-day patterns, not just single-day snapshots
               - Learns how technical indicators evolve over time
               
            2. **Feature Interactions**:
               - Discovers relationships between RSI, MACD, and price movements
               - Combines volume patterns with price trends
               - Links macroeconomic conditions to technical signals
               
            3. **Lagged Economic Effects**:
               - Understands that high inflation 3 months ago affects current prices
               - Recognizes that interest rate changes have delayed market impacts
               
            4. **Multi-Scale Patterns**:
               - Short-term: Daily price movements and technical signals
               - Medium-term: 30-day trends and momentum
               - Long-term: Macroeconomic cycles and lagged effects
            
            **Result**: The model captures complex, non-linear relationships that traditional 
            machine learning models (like XGBoost) might miss, leading to the **3.68% accuracy improvement**.
            """)
        
        st.markdown("---")
        
        # Original Model Performance Section (kept for compatibility)
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
    try:
        main()
    except Exception as e:
        st.error(f"⚠️ Application error: {str(e)}")
        st.info("The application encountered an unexpected error. Please refresh the page or contact support if the issue persists.")
