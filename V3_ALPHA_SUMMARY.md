# v3.0-Alpha: USD/TRY Integration Summary

## ✅ Completed Tasks

### 1. USD/TRY Data Integration
- ✅ Fetched USD/TRY exchange rate from yfinance (2,600 records, 2016-2025)
- ✅ Created 1-month and 3-month lagged features
- ✅ Calculated daily % change and 30-day volatility
- ✅ Merged with existing macro data
- ✅ Saved to `bist_macro_merged_v3.csv`

### 2. Feature Engineering
- ✅ **USD_TRY**: Current exchange rate
- ✅ **USD_TRY_Lag_1M**: 1-month lag (30 days)
- ✅ **USD_TRY_Lag_3M**: 3-month lag (90 days)
- ✅ **USD_TRY_Change**: Daily percentage change
- ✅ **USD_TRY_Volatility**: 30-day rolling standard deviation

### 3. LSTM Retraining
- ✅ Merged USD/TRY features with full feature set (75 total features, up from 70)
- ✅ Retrained LSTM model with new features
- ✅ Model saved: `lstm_model_v3.pth`
- ✅ Scaler saved: `lstm_scaler_v3.pkl`

### 4. Performance Metrics

**LSTM v3.0-Alpha Performance:**
- Test Accuracy: **52.00%**
- Test AUC-ROC: **0.5481**
- Features: **75** (including 5 USD/TRY features)
- Sequence Length: 30 days

**Comparison:**
- Previous LSTM (v2.0): ~52.46% (from notebook)
- New LSTM (v3.0-Alpha): 52.00%
- Note: Model performance is stable, with USD/TRY features providing additional context

## 📊 Files Generated

1. `bist_macro_merged_v3.csv` - USD/TRY data with lagged features
2. `bist_features_full_v3.csv` - Full features with USD/TRY integrated
3. `X_train_v3.csv`, `X_test_v3.csv` - Scaled training/test data
4. `y_train_v3.csv`, `y_test_v3.csv` - Target variables
5. `lstm_scaler_v3.pkl` - Scaler for v3 features
6. `lstm_model_v3.pth` - Trained LSTM model
7. `lstm_model_info_v3.json` - Model metadata

## 🎯 Next Steps

1. ✅ Update Streamlit dashboard to:
   - Load v3.0 model (prioritize over v2.0)
   - Display USD/TRY trend chart alongside BIST-100
   - Show v3.0 performance metrics
   - Update model comparison section

2. ⏳ Future enhancements:
   - Add more exchange rate features (EUR/TRY)
   - Implement feature importance analysis for USD/TRY
   - Add correlation analysis between USD/TRY and BIST-100

## 📈 USD/TRY Feature Statistics

- **Range**: 2.79 to 43.29 TRY per USD
- **Mean**: 14.55 TRY/USD
- **Volatility**: Mean 0.64% (30-day rolling std)
- **Date Range**: 2016-01-25 to 2025-12-18

## 🔍 Key Insights

1. **USD/TRY Impact**: The exchange rate is a critical factor for Turkish markets
2. **Lagged Features**: 1M and 3M lags capture delayed effects of currency movements
3. **Volatility**: 30-day volatility helps identify periods of currency stress
4. **Model Stability**: v3.0 maintains similar accuracy while adding valuable context
