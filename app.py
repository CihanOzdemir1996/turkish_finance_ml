"""
BIST-100 Price Direction Prediction - Streamlit Web Application
A clean dashboard for predicting next-day stock price direction using XGBoost model.
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
warnings.filterwarnings('ignore')

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

@st.cache_data
def load_model():
    """Load the trained XGBoost model"""
    project_root = Path(__file__).parent
    models_dir = project_root / "models"
    
    # Try to load XGBoost model
    xgb_path = models_dir / "xgboost_model.pkl"
    if xgb_path.exists():
        model = joblib.load(xgb_path)
        model_name = "XGBoost"
        return model, model_name
    
    # Fallback to best model
    best_model_path = models_dir / "best_model.pkl"
    if best_model_path.exists():
        model = joblib.load(best_model_path)
        model_name = "Best Model"
        return model, model_name
    
    # Fallback to Random Forest
    rf_path = models_dir / "random_forest_model.pkl"
    if rf_path.exists():
        model = joblib.load(rf_path)
        model_name = "Random Forest"
        return model, model_name
    
    raise FileNotFoundError("No trained model found. Please run notebook 04_model_training.ipynb first.")

@st.cache_data
def load_latest_data():
    """Load the latest processed data for prediction"""
    project_root = Path(__file__).parent
    data_processed_dir = project_root / "data" / "processed"
    
    X_test = pd.read_csv(data_processed_dir / "X_test.csv")
    y_test_df = pd.read_csv(data_processed_dir / "y_test.csv")
    
    # Get the most recent sample (last row)
    latest_features = X_test.iloc[-1:].copy()
    
    return latest_features, X_test, y_test_df

@st.cache_data
def get_feature_importance(_model, feature_names):
    """Get feature importance from the model"""
    if hasattr(_model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': _model.feature_importances_
        }).sort_values('Importance', ascending=False)
        return importance_df
    return None

def make_prediction(model, latest_features):
    """Make prediction using the model"""
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
        
        try:
            model, model_name = load_model()
            st.success(f"✅ Model Loaded: {model_name}")
            st.info(f"Model Type: {type(model).__name__}")
        except FileNotFoundError as e:
            st.error(f"❌ {str(e)}")
            st.stop()
        
        st.markdown("---")
        st.markdown("### About")
        st.info("""
        This dashboard uses a trained machine learning model to predict 
        the next-day price direction (UP/DOWN) for BIST-100 stocks.
        
        **Model Performance:**
        - Accuracy: ~49%
        - Features: 70+ technical indicators
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
        latest_features, X_test, y_test_df = load_latest_data()
        prediction, prob_up, prob_down, confidence = make_prediction(model, latest_features)
        
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
            st.write(f"**Model Used:** {model_name}")
            st.write(f"**Prediction:** {direction_text} ({direction_emoji})")
            st.write(f"**Confidence:** {confidence:.2f}% ({confidence_level})")
            st.write(f"**Total Features:** {latest_features.shape[1]}")
            st.write(f"**Test Samples Available:** {len(X_test):,}")
            
            if prediction == 1:
                st.success(f"💡 The model predicts the price will **GO UP** tomorrow with {prob_up:.2f}% confidence.")
            else:
                st.error(f"💡 The model predicts the price will **GO DOWN** tomorrow with {prob_down:.2f}% confidence.")
        
        st.markdown("---")
        
        # Feature Importance Section
        st.header("🏆 Feature Importance Analysis")
        
        # Get feature names
        feature_names = X_test.columns.tolist()
        importance_df = get_feature_importance(model, feature_names)
        
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
            st.metric("Test Accuracy", "~49%", delta="Close to random")
        
        with col2:
            st.metric("Features Used", "70+", delta="Technical indicators")
        
        with col3:
            st.metric("Model Type", model_name, delta="Ensemble")
        
        with col4:
            st.metric("Test Samples", f"{len(X_test):,}", delta="Evaluated")
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #666; padding: 2rem 0;">
            <p><strong>BIST-100 AI Prediction System</strong></p>
            <p>Built with Streamlit | Powered by XGBoost & Random Forest</p>
            <p style="font-size: 0.9rem;">⚠️ For educational purposes only. Not financial advice.</p>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"❌ Error loading data or making prediction: {str(e)}")
        st.info("Please ensure you have run the preprocessing and model training notebooks first.")

if __name__ == "__main__":
    main()
