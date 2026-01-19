# BIST-100 Price Direction Prediction with Machine Learning

A comprehensive machine learning project that predicts next-day price direction (up/down) for Turkish BIST-100 stocks using technical indicators and ensemble learning models.

## 📊 Project Overview

This project implements an end-to-end machine learning pipeline for predicting stock price movements in the Turkish stock market. Using historical price data and 70+ engineered technical indicators, we train and evaluate multiple machine learning models to forecast whether stock prices will increase or decrease on the next trading day.

### Key Features
- **70+ Technical Indicators**: Comprehensive feature engineering including moving averages, RSI, MACD, Bollinger Bands, ATR, momentum indicators, and more
- **Ensemble Models**: XGBoost and Random Forest classifiers for robust predictions
- **Time-Series Aware**: Proper chronological train-test split respecting temporal dependencies
- **Feature Importance Analysis**: Identification of the most influential technical indicators
- **Live Prediction System**: Real-time buy/sell signal generation for next trading day

## 📁 Project Structure

```
turkish_finance_ml/
├── notebooks/              # Jupyter notebooks (01-05)
│   ├── 01_data_collection.ipynb
│   ├── 02_eda_exploration.ipynb
│   ├── 03_data_preprocessing.ipynb
│   ├── 04_model_training.ipynb
│   └── 05_live_prediction.ipynb
├── data/
│   ├── raw/                # Original BIST stock price data
│   └── processed/          # Feature-engineered datasets
├── models/                  # Trained models (XGBoost, Random Forest)
├── reports/                 # Visualizations and analysis reports
├── src/                     # Python modules for data collection
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

## 🎯 Model Performance

- **Models**: XGBoost Classifier, Random Forest Classifier
- **Features**: 70+ technical and price-based indicators
- **Accuracy**: ~49% (reflecting market efficiency and prediction difficulty)
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
Python 3.8+
pandas
numpy
scikit-learn
xgboost
matplotlib
seaborn
jupyter
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
1. **Run notebooks in sequence** (01 → 05):
   ```bash
   jupyter notebook notebooks/
   ```

2. **Data Collection**: Run `01_data_collection.ipynb` to download stock data

3. **Exploratory Analysis**: Run `02_eda_exploration.ipynb` for data insights

4. **Preprocessing**: Run `03_data_preprocessing.ipynb` to create features

5. **Model Training**: Run `04_model_training.ipynb` to train models

6. **Live Prediction**: Run `05_live_prediction.ipynb` for predictions

## 📊 Technical Stack

- **Data Sources**: Yahoo Finance (BIST stock prices)
- **Libraries**: pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn
- **Models**: Random Forest Classifier, XGBoost Classifier
- **Features**: 70+ technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, ATR, etc.)
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

### Model Improvements
1. **Deep Learning Approaches**
   - Implement LSTM (Long Short-Term Memory) networks for time-series prediction
   - Try GRU (Gated Recurrent Unit) models
   - Experiment with Transformer architectures for financial time-series

2. **Macroeconomic Data Integration**
   - Add inflation rates, interest rates, and exchange rates
   - Incorporate CBRT (Central Bank of Turkey) economic indicators
   - Include GDP growth, unemployment rates, and other macroeconomic factors

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
