# ✅ Pre-Run Checklist for Data Collection

Before running `01_data_collection.ipynb`, please verify the following:

## 🔐 1. API Key Setup

### Check .env File
- [ ] `.env` file exists in `turkish_finance_ml/` folder (same level as `notebooks/`, `src/`, etc.)
- [ ] `.env` file contains exactly this format:
  ```
  EVDS_API_KEY=your_actual_api_key_here
  ```
- [ ] No quotes around the API key value
- [ ] No spaces before or after the `=` sign
- [ ] API key is the complete key (usually 30-40 characters)

### Verify API Key Format
Your `.env` file should look like this:
```
EVDS_API_KEY=abc123def456ghi789jkl012mno345pqr678
```

**NOT like this:**
```
EVDS_API_KEY="abc123..."  ❌ (no quotes)
EVDS_API_KEY = abc123...  ❌ (no spaces)
EVDS_API_KEY=abc123...    ✅ (correct)
```

### Test API Key Loading
You can test if the .env file is being read correctly:
```python
# Run this in a notebook cell or Python script
from src.load_env import get_evds_api_key
api_key = get_evds_api_key()
if api_key:
    print(f"✅ API key loaded: {api_key[:10]}...{api_key[-5:]}")
else:
    print("❌ API key not found in .env file")
```

---

## 📦 2. Dependencies Installation

### Check Required Packages
- [ ] All packages from `requirements.txt` are installed
- [ ] Run: `pip install -r requirements.txt` if not already done

### Key Packages Needed:
- [x] `pandas` - Data manipulation
- [x] `numpy` - Numerical operations
- [x] `requests` - API calls
- [x] `yfinance` - Stock data collection
- [x] `matplotlib` - Visualizations (for EDA later)
- [x] `seaborn` - Visualizations (for EDA later)

### Verify Installation
Run this in a Python cell to check:
```python
import pandas as pd
import numpy as np
import requests
import yfinance as yf
print("✅ All packages installed successfully!")
```

---

## 📁 3. Folder Structure

### Verify Folders Exist
- [ ] `turkish_finance_ml/data/raw/` folder exists
- [ ] `turkish_finance_ml/src/` folder exists
- [ ] `turkish_finance_ml/notebooks/` folder exists

### Check File Locations
- [ ] `.env` file is in `turkish_finance_ml/` (project root)
- [ ] `src/data_collection.py` exists
- [ ] `src/load_env.py` exists
- [ ] `notebooks/01_data_collection.ipynb` exists

---

## 🌐 4. Internet Connection

- [ ] Internet connection is active
- [ ] Can access: https://evds2.tcmb.gov.tr/ (CBRT EVDS API)
- [ ] Can access: https://finance.yahoo.com/ (Yahoo Finance)

### Test Connectivity
```python
import requests
try:
    response = requests.get("https://evds2.tcmb.gov.tr/", timeout=5)
    print("✅ Can reach CBRT EVDS website")
except:
    print("❌ Cannot reach CBRT EVDS website - check internet connection")
```

---

## ⚙️ 5. Notebook Settings

### Jupyter Notebook Setup
- [ ] Jupyter Notebook or JupyterLab is installed
- [ ] Kernel is selected (Python 3.x)
- [ ] You're running from the correct directory

### Path Configuration
The notebook should be run from the `notebooks/` folder, or ensure the path setup in the first cell works correctly.

---

## 🔍 6. API Key Validation (Optional but Recommended)

Before collecting full dataset, test with a small date range:

```python
from src.data_collection import TurkishFinancialDataCollector
from pathlib import Path

# Test with small date range first
collector = TurkishFinancialDataCollector(
    data_dir=Path("../data/raw"),
    evds_api_key="your_key_here"  # Or it will load from .env
)

# Test with just 2023 data
test_data = collector.collect_cbrt_macroeconomic_data(
    start_date="01-01-2023",
    end_date="31-12-2023"
)

if not test_data.empty:
    print("✅ API key works! Ready for full data collection.")
else:
    print("❌ API key test failed. Check your key.")
```

---

## 📊 7. Expected Data Collection Time

### Macroeconomic Data (CBRT EVDS)
- **Time:** ~30-60 seconds
- **Size:** ~300 rows (monthly data, 2000-2024)
- **Files:** `cbrt_macroeconomic_data.csv`

### Stock Data (Yahoo Finance)
- **BIST-100 Index:** ~10-20 seconds
- **Multiple Stocks:** ~1-2 minutes (10 stocks)
- **Size:** ~6,000+ rows per stock (daily data, 2000-2024)
- **Files:** `bist_stock_prices.csv`

### Combined Data
- **Time:** ~5 seconds (merging)
- **Files:** `combined_stock_macro_data.csv`

**Total Expected Time:** ~2-3 minutes for full collection

---

## ⚠️ 8. Common Issues & Solutions

### Issue: "ModuleNotFoundError: No module named 'load_env'"
**Solution:** Make sure you're running from the `notebooks/` folder, or the path setup in the first cell is correct.

### Issue: "API key not found"
**Solution:** 
- Check `.env` file location (should be in project root)
- Check `.env` file format (no quotes, no spaces)
- Verify API key is complete

### Issue: "Connection timeout" or "Request failed"
**Solution:**
- Check internet connection
- CBRT EVDS might be temporarily down - wait and retry
- Check if you're behind a firewall/proxy

### Issue: "No data returned"
**Solution:**
- Verify API key is valid (test with small date range first)
- Check date range (some series might not have data for all dates)
- Series codes might have changed - check CBRT EVDS website

### Issue: "Rate limit exceeded"
**Solution:**
- Wait 1-2 minutes and try again
- The script includes automatic delays, but if you run multiple times quickly, you might hit limits

---

## ✅ Final Verification

Before running the full notebook, verify:

1. [ ] `.env` file exists with correct API key
2. [ ] All packages installed (`pip install -r requirements.txt`)
3. [ ] Internet connection active
4. [ ] Folders exist (`data/raw/`, `src/`, etc.)
5. [ ] Test API key works (optional but recommended)

---

## 🚀 Ready to Run!

Once all checks pass:
1. Open `notebooks/01_data_collection.ipynb`
2. Run cells in order
3. The notebook will:
   - Load your API key from `.env`
   - Collect macroeconomic data
   - Collect stock data
   - Combine datasets
   - Save everything to `data/raw/`

**Good luck! 🎉**

---

## 📞 Need Help?

- Check `CBRT_API_SETUP.md` for API registration help
- Check `DATA_SOURCES.md` for data source information
- Review error messages - they usually indicate the issue
- Test with small date ranges first before full collection
