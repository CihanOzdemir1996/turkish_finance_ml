# Turkish Financial Data Sources Guide

## 🎯 Best Options for Your Project

### 1. **CBRT EVDS API** ⭐ RECOMMENDED for Macroeconomic Data

**What it provides:**
- ✅ Consumer Price Index (CPI/TÜFE) - Monthly, Annual
- ✅ Producer Price Index (PPI/ÜFE) - Monthly, Annual  
- ✅ Policy Interest Rates (Repo rates)
- ✅ Exchange Rates (USD/TRY, EUR/TRY)
- ✅ Money supply, banking statistics
- ✅ Time-series format (daily, weekly, monthly, quarterly)
- ✅ **FREE** and official
- ✅ Historical data back to 2000s

**How to access:**
1. Register at: https://evds2.tcmb.gov.tr/
2. Get your API key from profile settings
3. Use the `data_collection.py` script provided

**Example series codes:**
- `TP.DK.YTL.A.YTL`: CPI Annual
- `TP.DK.USD.A.YTL`: USD/TRY Exchange Rate
- `TP.INT.RSK.A`: Policy Interest Rate
- `TP.DK.GB.A.YTL`: PPI Annual

**Documentation:** https://evds2.tcmb.gov.tr/help

---

### 2. **BIST Stock Prices - Yahoo Finance** ⭐ EASY & FREE

**What it provides:**
- ✅ Historical daily stock prices (OHLCV)
- ✅ BIST-100 index data
- ✅ Individual company stocks
- ✅ Easy to use with `yfinance` library
- ✅ No API key needed
- ✅ Data back to 2000s

**How to use:**
```python
import yfinance as yf

# BIST-100 index
bist100 = yf.Ticker("XU100.IS")
hist = bist100.history(start="2000-01-01")

# Individual stocks (add .IS suffix)
akbank = yf.Ticker("AKBNK.IS")
hist = akbank.history(start="2000-01-01")
```

**Common BIST tickers:**
- `XU100.IS` - BIST-100 Index
- `AKBNK.IS` - Akbank
- `GARAN.IS` - Garanti BBVA
- `THYAO.IS` - Turkish Airlines
- `TUPRS.IS` - Tüpraş
- `SAHOL.IS` - Hacı Ömer Sabancı Holding

---

### 3. **Kaggle Datasets** 📊 Pre-processed Options

#### Option A: Borsa Istanbul Stock Exchange Dataset
- **Link:** https://www.kaggle.com/datasets/gokhankesler/borsa-istanbul-turkish-stock-exchange-dataset
- **Content:** Historical daily stock prices for multiple BIST companies
- **Format:** CSV files
- **Size:** Large, comprehensive

#### Option B: BIST100 Turkish Stock Market
- **Link:** https://www.kaggle.com/datasets/hakanetin/bist100turkishstaockmarketturkhissefiyatlar
- **Content:** BIST-100 index and component stocks
- **Format:** CSV

#### Option C: Turkish Financial News + Technical Indicators
- **Link:** https://huggingface.co/datasets/rsmctn/bist-dp-lstm-trading-turkish_financial_news
- **Content:** BIST stocks (2019-2024) + technical indicators + news sentiment
- **Format:** Already includes feature engineering
- **Note:** Time span is limited (2019-2024)

---

### 4. **TÜİK (Turkish Statistical Institute)** 📈 Official Statistics

**What it provides:**
- GDP, National Accounts
- Employment statistics
- Trade data
- Sectoral statistics

**Access:**
- Website: https://data.tuik.gov.tr/
- **Note:** Mainly web-based, limited API access
- Can download CSV/Excel files manually

---

## 🚀 Recommended Approach for Your Project

### **Combination Strategy:**

1. **Macroeconomic Data:** Use CBRT EVDS API
   - Inflation (CPI, PPI)
   - Interest rates
   - Exchange rates
   - Monthly frequency

2. **Stock Prices:** Use Yahoo Finance (via `yfinance`)
   - BIST-100 index
   - Major company stocks
   - Daily frequency

3. **Combine them:**
   - Merge on date
   - Handle frequency differences (daily stocks + monthly macro)
   - Create lag features (e.g., inflation last month)

### **Why This Combination Works:**
- ✅ **Comprehensive:** Both micro (stocks) and macro (economy) data
- ✅ **Time-series format:** Perfect for ML analysis
- ✅ **Large dataset:** Years of historical data
- ✅ **High quality:** Official sources
- ✅ **Free:** No cost for data
- ✅ **Easy to collect:** Automated scripts provided

---

## 📝 Quick Start

### Step 1: Get CBRT EVDS API Key
1. Visit: https://evds2.tcmb.gov.tr/
2. Register/Login
3. Get API key from profile

### Step 2: Run Data Collection
```python
from src.data_collection import TurkishFinancialDataCollector

# Initialize collector
collector = TurkishFinancialDataCollector(
    evds_api_key="YOUR_EVDS_API_KEY"  # Get from CBRT website
)

# Collect macroeconomic data
macro_data = collector.collect_cbrt_macroeconomic_data(
    start_date="01-01-2000",
    end_date="31-12-2024"
)

# Collect BIST stock data
stock_data = collector.collect_bist_stock_data(
    tickers=['XU100.IS'],  # BIST-100 index
    start_date="2000-01-01"
)

# Or collect multiple stocks
stocks = collector.collect_bist_100_companies()
```

### Step 3: Combine Data
```python
# Merge stock and macro data
combined = stock_data.merge(
    macro_data,
    on='Date',
    how='left'
)
```

---

## 📊 Expected Dataset Sizes

- **Macroeconomic data:** ~300 rows (monthly, 2000-2024)
- **Stock prices (daily):** ~6,000+ rows (2000-2024)
- **Combined:** Rich time-series with multiple features

---

## ⚠️ Important Notes

1. **API Rate Limits:** CBRT EVDS has rate limits - use delays between requests
2. **Data Frequency:** Macro data is monthly, stocks are daily - plan your merge strategy
3. **Missing Data:** Some dates may have missing values (holidays, weekends)
4. **Data Quality:** Always validate data ranges and check for anomalies

---

## 🔗 Useful Links

- CBRT EVDS: https://evds2.tcmb.gov.tr/
- TÜİK Data Portal: https://data.tuik.gov.tr/
- Borsa İstanbul: https://www.borsaistanbul.com/
- Yahoo Finance BIST: https://finance.yahoo.com/quote/XU100.IS/

---

## 💡 Pro Tips

1. **Start with BIST-100 index** - it's a good proxy for overall market
2. **Collect macro data first** - it's smaller and faster
3. **Save raw data immediately** - don't lose your downloads
4. **Document your data sources** - important for portfolio projects
5. **Check data updates** - macro data is updated monthly
