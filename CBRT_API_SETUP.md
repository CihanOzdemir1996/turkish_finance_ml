# CBRT EVDS API Key Registration Guide

## 🎯 Step-by-Step Instructions

### Step 1: Visit the EVDS Website
1. Open your browser and go to: **https://evds2.tcmb.gov.tr/**
2. You'll see the EVDS (Electronic Data Distribution System) homepage

### Step 2: Register a New Account
1. Click on **"Kayıt Ol"** (Register) or **"Register"** button (usually in the top right)
2. Fill in the registration form:
   - **E-posta (Email)**: Your email address
   - **Şifre (Password)**: Create a strong password
   - **Şifre Tekrar (Confirm Password)**: Re-enter your password
   - **Ad Soyad (Name Surname)**: Your full name
   - **Kurum (Institution)**: You can enter "Bireysel" (Individual) or your organization
   - **Telefon (Phone)**: Your phone number
3. Accept the terms and conditions
4. Click **"Kayıt Ol"** (Register) button

### Step 3: Verify Your Email
1. Check your email inbox for a verification email from CBRT
2. Click the verification link in the email
3. Your account will be activated

### Step 4: Login
1. Go back to: **https://evds2.tcmb.gov.tr/**
2. Click **"Giriş Yap"** (Login)
3. Enter your email and password
4. Click **"Giriş Yap"** (Login)

### Step 5: Get Your API Key
1. After logging in, look for your **profile/user menu** (usually top right corner)
2. Click on your name/email or profile icon
3. Look for **"API Anahtarı"** (API Key) or **"API Key"** section
4. You may see:
   - An existing API key (copy it)
   - Or a button to **"Yeni API Anahtarı Oluştur"** (Generate New API Key)
5. **Copy your API key** - you'll need it for the data collection script

### Step 6: Save Your API Key Securely
⚠️ **Important**: Save your API key in a safe place. You'll need it to collect data.

**Recommended**: Create a `.env` file (see below) or save it in a secure location.

---

## 🔐 Secure API Key Storage

### Option 1: Use .env file (Recommended)
1. Create a file named `.env` in your project root (`turkish_finance_ml/`)
2. Add this line:
   ```
   EVDS_API_KEY=your_actual_api_key_here
   ```
3. **Important**: Add `.env` to `.gitignore` to keep it private!

### Option 2: Direct in Notebook (Quick Test)
- You can paste it directly in the notebook for testing
- **But remember**: Don't commit it to Git!

---

## ✅ Test Your API Key

After getting your API key, run the test script to verify it works:

```python
# In your notebook or Python script
from src.data_collection import TurkishFinancialDataCollector

# Replace with your actual API key
API_KEY = "your_api_key_here"

collector = TurkishFinancialDataCollector(evds_api_key=API_KEY)

# Test with a simple request
test_data = collector.collect_cbrt_macroeconomic_data(
    start_date="01-01-2023",
    end_date="31-12-2023"
)

if not test_data.empty:
    print("✅ API Key works! You can collect data.")
else:
    print("❌ API Key test failed. Check your key.")
```

---

## 🐛 Troubleshooting

### Problem: "Invalid API Key" error
- **Solution**: Double-check you copied the entire key (no spaces, no extra characters)
- Make sure you're using the key from the EVDS website, not from another source

### Problem: "Rate limit exceeded"
- **Solution**: The API has rate limits. Wait a few minutes and try again
- Our script includes automatic delays between requests

### Problem: "No data returned"
- **Solution**: 
  - Check the date range (some series may not have data for all dates)
  - Verify the series code exists
  - Check if you're logged into EVDS website

### Problem: Website is in Turkish
- **Solution**: 
  - Use browser translation (Chrome/Edge: Right-click → Translate)
  - Or use these translations:
    - Kayıt Ol = Register
    - Giriş Yap = Login
    - API Anahtarı = API Key
    - Profil = Profile

### Problem: Can't find API Key section
- **Solution**:
  - Make sure you're logged in
  - Look in "Hesap Ayarları" (Account Settings) or "Profil" (Profile)
  - Some accounts may need to request API access - contact CBRT support if needed

---

## 📞 Need Help?

- **EVDS Support**: Check the help section on https://evds2.tcmb.gov.tr/
- **Documentation**: The website has documentation (may be in Turkish)
- **Contact**: If you have issues, you can contact CBRT through their website

---

## 🎉 Once You Have Your API Key

1. Open `notebooks/01_data_collection.ipynb`
2. Find the cell with `EVDS_API_KEY = "YOUR_API_KEY"`
3. Replace `"YOUR_API_KEY"` with your actual key
4. Run the cells to collect real Turkish macroeconomic data!

---

## 💡 Pro Tips

1. **API Key is FREE**: No cost, just registration required
2. **No expiration**: Your API key doesn't expire (unless you regenerate it)
3. **Rate limits**: Be respectful - don't make too many requests too quickly
4. **Data updates**: Macro data is typically updated monthly
5. **Historical data**: You can get data back to 2000s for most series

---

## 🔗 Quick Links

- **EVDS Homepage**: https://evds2.tcmb.gov.tr/
- **Registration**: https://evds2.tcmb.gov.tr/ (click "Kayıt Ol")
- **Login**: https://evds2.tcmb.gov.tr/ (click "Giriş Yap")
- **Data Series Browser**: Available after login

Good luck! 🚀
