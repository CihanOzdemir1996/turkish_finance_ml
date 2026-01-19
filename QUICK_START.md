# 🚀 Quick Start Guide - CBRT API Registration

## ⚡ 5-Minute Setup

### Step 1: Register at CBRT EVDS (2 minutes)
1. Go to: **https://evds2.tcmb.gov.tr/**
2. Click **"Kayıt Ol"** (Register) - top right corner
3. Fill the form:
   - Email
   - Password
   - Name
   - Phone (optional)
4. Verify your email (check inbox)
5. Login with your credentials

### Step 2: Get Your API Key (1 minute)
1. After login, click your **profile/name** (top right)
2. Find **"API Anahtarı"** (API Key) section
3. **Copy your API key** (it's a long string)

### Step 3: Use Your API Key (2 minutes)

**Option A: In Notebook (Quick Test)**
1. Open `notebooks/01_data_collection.ipynb`
2. Find: `EVDS_API_KEY = "YOUR_API_KEY"`
3. Replace with: `EVDS_API_KEY = "your_actual_key_here"`
4. Run the cells!

**Option B: Use .env File (Recommended)**
1. Create file: `turkish_finance_ml/.env`
2. Add this line:
   ```
   EVDS_API_KEY=your_actual_key_here
   ```
3. The notebook will automatically load it!

### Step 4: Test It Works
Run the test cell in the notebook - you should see:
```
✅ API Key works! You can collect data.
```

## 🎯 That's It!

Now you can collect real Turkish macroeconomic data:
- Inflation rates (CPI)
- Interest rates
- Exchange rates (USD/TRY)
- Producer Price Index (PPI)

All data is **FREE** and **OFFICIAL** from Turkish Central Bank!

## ❓ Need Help?

- **Detailed guide**: See `CBRT_API_SETUP.md`
- **Test script**: Run `python src/test_api_key.py`
- **Website issues**: Use browser translation (right-click → Translate)

## 🔗 Direct Links

- **EVDS Homepage**: https://evds2.tcmb.gov.tr/
- **Registration**: https://evds2.tcmb.gov.tr/ (click "Kayıt Ol")
- **Login**: https://evds2.tcmb.gov.tr/ (click "Giriş Yap")

---

**Turkish Translations:**
- Kayıt Ol = Register
- Giriş Yap = Login  
- API Anahtarı = API Key
- Profil = Profile

Good luck! 🎉
