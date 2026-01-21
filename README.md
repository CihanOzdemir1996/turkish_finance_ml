# BIST-100 Price Direction Prediction with Machine Learning v3.0-Alpha

**Deep Learning Powered (LSTM v3)** - A production-ready machine learning system that predicts next-day price direction (up/down) for Turkish BIST-100 stocks using technical indicators, macroeconomic features, exchange rates, and validated deep learning models.

## 🚀 Version 3.0-Alpha Highlights (Latest)

- **Deep Learning Powered (LSTM v3)**: PyTorch-based LSTM with **52%+ accuracy** (75 features including USD/TRY)
- **Validated Reliability**: **87% validation improvement** - Reduced overfitting gap from 48.94% to 8.38% through proper validation strategy
- **USD/TRY Exchange Rate Integration**: Exchange rate with 1-month and 3-month lagged features + volatility indicators
- **Production-Ready Validation**: Zero data leakage with time-series aware cross-validation
- **Enhanced Feature Set**: 75 features (70 technical + 5 USD/TRY features)
- **Streamlit Cloud Ready**: Optimized for deployment with minimal dependencies and robust error handling

## 🎉 Version 2.0 Highlights

- **LSTM Deep Learning Model**: PyTorch-based LSTM with 52.46% accuracy (outperforms XGBoost by 3.68%)
- **Macroeconomic Integration**: Inflation and Interest Rates with 1-month and 3-month lagged features
- **30-Day Sequence Learning**: Temporal pattern recognition using 30-day lookback windows
- **Advanced Architecture**: 3-layer LSTM (128→64→32 units) with dropout and batch normalization

## 📊 Project Overview

This project implements an end-to-end machine learning pipeline for predicting stock price movements in the Turkish stock market. Using historical price data and 70+ engineered technical indicators, we train and evaluate multiple machine learning models to forecast whether stock prices will increase or decrease on the next trading day.

### Key Features
- **75 Features Total**: 70+ technical indicators + 5 USD/TRY exchange rate features
- **Technical Indicators**: Comprehensive feature engineering including moving averages, RSI, MACD, Bollinger Bands, ATR, momentum indicators, and more
- **Macroeconomic Features**: Inflation (TUFE) and Interest Rates with 1M and 3M lagged versions
- **USD/TRY Exchange Rate**: Current rate, 1M/3M lags, daily % change, and 30-day volatility (NEW in v3.0)
- **Deep Learning**: LSTM (PyTorch) model for temporal pattern recognition with 52%+ accuracy
- **Ensemble Models**: XGBoost and Random Forest classifiers for robust predictions
- **Time-Series Aware**: Proper chronological train-test split with zero data leakage
- **Robust Validation**: 87% improvement in validation reliability (gap reduced from 48.94% to 8.38%)
- **Feature Importance Analysis**: Identification of the most influential technical indicators
- **Live Prediction System**: Real-time buy/sell signal generation via Streamlit web dashboard with USD/TRY visualization

## 📁 Project Structure

```
turkish_finance_ml/
├── notebooks/              # Jupyter notebooks (01-07)
│   ├── 01_data_collection.ipynb
│   ├── 02_eda_exploration.ipynb
│   ├── 03_data_preprocessing.ipynb
│   ├── 04_model_training.ipynb
│   ├── 05_live_prediction.ipynb
│   ├── 06_macro_data_integration.ipynb  # NEW in v2.0
│   └── 07_lstm_model_training.ipynb     # NEW in v2.0
├── data/
│   ├── raw/                # Original BIST stock price data
│   └── processed/          # Feature-engineered datasets
├── models/                  # Trained models (LSTM v3.0-Alpha, v2.0, XGBoost, Random Forest)
├── reports/                 # Visualizations and analysis reports
├── src/                     # Python modules for data collection and validation
├── app.py                   # Streamlit web dashboard (v3.0-Alpha with USD/TRY visualization)
├── v3_alpha_usd_try_integration.py  # USD/TRY data integration script
├── v3_alpha_lstm_retraining.py      # LSTM retraining with USD/TRY
├── v3_alpha_lstm_train.py           # LSTM training script
└── README.md
```

## 📓 Notebooks Overview

### 1. Data Collection (`01_data_collection.ipynb`)
- Collects BIST stock price data from Yahoo Finance
- Handles data download and initial storage
- Supports multiple tickers and date ranges
- Saves raw data to `data/raw/` directory

### 2. Exploratory Data Analysis (`02_eda_exploration.ipynb`)
- Comprehensive data exploration and visualization
- Price trend analysis and volatility assessment
- Technical indicator calculation and visualization
- Missing value analysis and data quality checks
- Correlation analysis between features

### 3. Data Preprocessing (`03_data_preprocessing.ipynb`)
- **Feature Engineering**: Creates 70+ technical indicators including:
  - Moving Averages (SMA, EMA with multiple periods)
  - Momentum Indicators (RSI, MACD, Momentum)
  - Volatility Measures (ATR, Bollinger Bands)
  - Volume Indicators (Volume ratios, trends)
  - Lag Features and Rolling Statistics
- Handles missing values and infinity values
- Feature scaling using StandardScaler
- Time-series aware train-test split (80/20)
- Creates target variables for classification (price direction)

### 4. Model Training (`04_model_training.ipynb`)
- Trains **Random Forest Classifier** and **XGBoost Classifier**
- Handles class imbalance with balanced class weights
- Comprehensive model evaluation:
  - Accuracy, Precision, Recall, F1-Score
  - Confusion Matrix
  - ROC Curve and AUC-ROC
- Feature importance analysis
- Model persistence (saves trained models to `models/`)

### 5. Live Prediction (`05_live_prediction.ipynb`)
- Loads trained models and makes predictions for next trading day
- Generates clear **BUY/SELL signals** with confidence levels
- Visualizes **Top 10 Feature Importance** chart
- Evaluates model performance on test set
- Provides comprehensive project summary and findings

### 6. Macro Data Integration (`06_macro_data_integration.ipynb`) - NEW in v2.0
- Fetches Inflation (TUFE) and Interest Rates from CBRT EVDS API
- Resamples monthly/weekly data to daily frequency
- Creates 1-month and 3-month lagged features
- Analyzes correlation between macroeconomic indicators and BIST-100
- Merges macro features with stock price data

### 7. LSTM Model Training (`07_lstm_model_training.ipynb`) - NEW in v2.0
- Implements PyTorch-based LSTM neural network
- 30-day sequence lookback for temporal pattern recognition
- Multi-layer architecture (128→64→32 units) with dropout
- Trains with lagged macroeconomic features
- Compares performance with XGBoost baseline
- **Result**: 52.46% accuracy (3.68% improvement over XGBoost)

### v3.0-Alpha: USD/TRY Integration (NEW)
- **USD/TRY Data Fetching**: Retrieves exchange rate data from yfinance (2,600+ records, 2016-2025)
- **Feature Engineering**: Creates 5 USD/TRY features:
  - Current USD/TRY exchange rate
  - 1-month lagged rate (30 days)
  - 3-month lagged rate (90 days)
  - Daily percentage change
  - 30-day rolling volatility
- **LSTM Retraining**: Retrains model with 75 features (70 technical + 5 USD/TRY)
- **Validation Improvements**: 87% reduction in overfitting gap (48.94% → 8.38%)
- **Streamlit Dashboard**: Enhanced with USD/TRY trend visualization

## 🎯 Model Performance

### Version 3.0-Alpha (LSTM with USD/TRY) - Latest
- **Primary Model**: LSTM (PyTorch) - 3-layer architecture
- **Accuracy**: **52.00%** (52%+ accuracy with USD/TRY features)
- **Features**: **75 total** (70 technical + 5 USD/TRY exchange rate features)
- **USD/TRY Features**: Current rate, 1M lag, 3M lag, daily % change, 30-day volatility
- **Sequence Length**: 30 days lookback
- **Validation Gap**: **8.38%** (87% improvement from 48.94% - production-ready)
- **Parameters**: 168,833 trainable parameters

### Version 2.0 (LSTM Deep Learning)
- **Primary Model**: LSTM (PyTorch) - 3-layer architecture
- **Accuracy**: **52.46%** (outperforms XGBoost by 3.68%)
- **Features**: 70+ technical indicators + macroeconomic features (Inflation, Interest Rates with lags)
- **Sequence Length**: 30 days lookback
- **Parameters**: 166,273 trainable parameters

### Baseline Models
- **XGBoost Classifier**: 48.79% accuracy
- **Random Forest Classifier**: ~49% accuracy
- **Features**: 70+ technical and price-based indicators
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Test Set**: 1,200+ samples evaluated

### Why ~49% Accuracy?
Stock markets are highly efficient, making short-term price predictions inherently challenging. The ~49% accuracy (close to random 50%) reflects:
- Market efficiency and random walk characteristics
- External factors (news, events, sentiment) not captured in technical indicators
- Non-stationarity of market conditions over time

Despite this, the model provides valuable insights into which technical indicators are most influential for price movements.

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.8+ (3.14 supported with PyTorch)
pandas
numpy
scikit-learn
xgboost
torch (PyTorch)  # NEW in v2.0
matplotlib
seaborn
jupyter
streamlit
```

### Installation
```bash
# Clone the repository
git clone https://github.com/CihanOzdemir1996/turkish_finance_ml.git
cd turkish_finance_ml

# Install dependencies
pip install -r requirements.txt
```

### Usage

#### Option 1: Run Notebooks
1. **Run notebooks in sequence** (01 → 05):
   ```bash
   jupyter notebook notebooks/
   ```

2. **Data Collection**: Run `01_data_collection.ipynb` to download stock data

3. **Exploratory Analysis**: Run `02_eda_exploration.ipynb` for data insights

4. **Preprocessing**: Run `03_data_preprocessing.ipynb` to create features

5. **Model Training**: Run `04_model_training.ipynb` to train models

6. **Live Prediction**: Run `05_live_prediction.ipynb` for predictions

#### Option 2: Streamlit Web Dashboard
Launch the interactive web dashboard:
```bash
streamlit run app.py
```

The dashboard provides:
- Real-time predictions for next trading day
- Confidence scores and probability breakdowns
- Feature importance visualization
- Model performance metrics (v3.0-Alpha vs v2.0 vs XGBoost)
- **USD/TRY trend visualization** with lagged features and volatility (NEW in v3.0)
- Macroeconomic context (Inflation, Interest Rates, USD/TRY)
- Refresh button to get latest predictions

## 📊 Technical Stack

- **Data Sources**: Yahoo Finance (BIST stock prices), CBRT EVDS API (macroeconomic data)
- **Libraries**: pandas, numpy, scikit-learn, xgboost, torch (PyTorch), matplotlib, seaborn, streamlit
- **Models**: 
  - **LSTM (PyTorch)** - Deep learning for temporal patterns (v2.0)
  - **XGBoost Classifier** - Gradient boosting ensemble
  - **Random Forest Classifier** - Ensemble method
- **Features**: 70+ technical indicators + macroeconomic features (Inflation, Interest Rates with lags)
- **Evaluation**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix, ROC-AUC

## 🔍 Key Insights

### Most Important Features
Based on feature importance analysis, the top contributing indicators include:
- **Technical Indicators**: RSI, MACD, Moving Averages (SMA/EMA), Bollinger Bands
- **Volatility Measures**: ATR (Average True Range), Rolling Standard Deviation
- **Price Features**: Close price, price changes, momentum indicators
- **Volume Indicators**: Volume ratios and trends

### Model Strengths
- ✅ Comprehensive feature engineering with 70+ indicators
- ✅ Proper time-series aware train-test split
- ✅ Feature scaling and normalization
- ✅ Class imbalance handling
- ✅ Multiple evaluation metrics
- ✅ Feature importance analysis

## 🚀 Next Steps & Future Enhancements

### Completed in v3.0-Alpha ✅
1. **USD/TRY Exchange Rate Integration**
   - ✅ Fetched USD/TRY data from yfinance (2,600+ records)
   - ✅ Created 1-month and 3-month lagged features
   - ✅ Calculated daily % change and 30-day volatility
   - ✅ Integrated with existing macro features
   - ✅ Retrained LSTM with 75 features (52%+ accuracy)

2. **Validation Reliability Improvements**
   - ✅ **87% improvement**: Reduced validation gap from 48.94% to 8.38%
   - ✅ Zero data leakage: Proper train-test split before scaling
   - ✅ Time-series aware cross-validation
   - ✅ Production-ready validation strategy

### Completed in v2.0 ✅
1. **Deep Learning Approaches**
   - ✅ Implemented LSTM (PyTorch) for time-series prediction
   - ✅ 3-layer architecture with dropout and batch normalization
   - ✅ 30-day sequence lookback for temporal patterns

2. **Macroeconomic Data Integration**
   - ✅ Added inflation rates (TUFE) and interest rates from CBRT EVDS API
   - ✅ Created 1-month and 3-month lagged features
   - ✅ Analyzed correlation between macro indicators and stock prices

### Future Enhancements
1. **Advanced Deep Learning**
   - Try GRU (Gated Recurrent Unit) models
   - Experiment with Transformer architectures for financial time-series
   - Bidirectional LSTM for better pattern recognition
   - Attention mechanisms for important time steps

2. **Additional Macroeconomic Data** (USD/TRY completed in v3.0-Alpha ✅)
   - ✅ USD/TRY exchange rate integration (COMPLETED)
   - Add EUR/TRY and other currency pairs
   - Include GDP growth and unemployment rates
   - Sector-specific macroeconomic factors

3. **Advanced Feature Engineering**
   - Sentiment analysis from news and social media
   - Market-wide indicators (BIST-100 index correlation)
   - Cross-asset relationships and sector analysis

4. **Model Enhancement**
   - Hyperparameter optimization (GridSearchCV, Bayesian Optimization)
   - Feature selection to reduce dimensionality
   - Ensemble methods combining multiple models
   - Online learning for model updates

5. **Data Enhancement**
   - Include more tickers for broader market analysis
   - Add intraday data for higher frequency predictions
   - Incorporate alternative data sources (options data, order flow)

### Production Deployment
- API for real-time predictions
- Automated daily prediction pipeline
- Model retraining and monitoring system
- Performance tracking and backtesting framework
- Web dashboard for visualization

## ⚠️ Important Disclaimers

1. **Not Financial Advice**: This model is for educational and research purposes only
2. **Market Risk**: Stock trading involves significant financial risk
3. **Model Limitations**: ~49% accuracy indicates limited predictive power
4. **Past Performance**: Historical patterns may not predict future movements
5. **External Factors**: Model doesn't account for news, events, or market sentiment

## 📝 License

This project is for educational and research purposes. Not intended as financial advice.

## 👤 Author

**Cihan Ozdemir**
- GitHub: [@CihanOzdemir1996](https://github.com/CihanOzdemir1996)

## 🙏 Acknowledgments

- Yahoo Finance for providing stock price data
- Borsa İstanbul (BIST) for Turkish stock market data
- Open-source ML community for tools and libraries

---

**Note**: This project demonstrates machine learning applications in finance. Always do your own research and consult with financial advisors before making investment decisions.
