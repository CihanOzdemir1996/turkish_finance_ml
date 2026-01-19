# 📸 Visual Guide: Getting Your CBRT EVDS API Key

## 🎯 Step-by-Step with Visual Cues

### Step 1: Go to EVDS Website
**URL:** https://evds2.tcmb.gov.tr/

You'll see a page that looks like this:
```
┌─────────────────────────────────────────┐
│  EVDS - Electronic Data Distribution  │
│                                         │
│  [Login]  [Register]  [Help]           │
└─────────────────────────────────────────┘
```

### Step 2: Click "Register" / "Kayıt Ol"
- Look for **"Kayıt Ol"** button (usually top right)
- Or **"Register"** if page is in English

### Step 3: Fill Registration Form
You'll see a form like this:
```
┌─────────────────────────────────────┐
│  Registration Form                   │
├─────────────────────────────────────┤
│  Email:        [____________]       │
│  Password:     [____________]       │
│  Confirm:      [____________]       │
│  Name:         [____________]       │
│  Institution:  [____________]       │
│  Phone:        [____________]       │
│                                     │
│  [ ] I accept terms                 │
│                                     │
│  [Register Button]                  │
└─────────────────────────────────────┘
```

**Fill in:**
- ✅ Your email address
- ✅ Create a password
- ✅ Your full name
- ✅ Institution: "Bireysel" (Individual) or your company
- ✅ Check the terms box
- ✅ Click Register

### Step 4: Verify Email
1. Check your email inbox
2. Look for email from CBRT/EVDS
3. Click the verification link
4. You'll be redirected back to EVDS

### Step 5: Login
1. Go back to: https://evds2.tcmb.gov.tr/
2. Click **"Giriş Yap"** (Login)
3. Enter your email and password
4. Click Login

### Step 6: Find Your API Key
After logging in, you'll see the main page. Look for:

**Option A: Profile Menu (Most Common)**
```
Top Right Corner:
┌──────────────┐
│ [Your Name] ▼│  ← Click here
└──────────────┘
   ↓
┌─────────────────────┐
│ Profile             │
│ Settings            │
│ API Anahtarı ← HERE │
│ Logout              │
└─────────────────────┘
```

**Option B: Direct Link**
- Sometimes there's a direct link: **"API Anahtarı"** or **"API Key"**
- Usually in the top menu or sidebar

**Option C: Account Settings**
- Click **"Hesap Ayarları"** (Account Settings)
- Look for **"API Anahtarı"** section

### Step 7: Copy Your API Key
When you find the API Key section, you'll see something like:
```
┌─────────────────────────────────────────────┐
│  API Anahtarı (API Key)                     │
├─────────────────────────────────────────────┤
│  Your API Key:                              │
│  ┌───────────────────────────────────────┐ │
│  │ abc123def456ghi789jkl012mno345pqr678  │ │ ← Copy this
│  └───────────────────────────────────────┘ │
│                                             │
│  [Copy] [Regenerate]                        │
└─────────────────────────────────────────────┘
```

**⚠️ Important:**
- Copy the ENTIRE key (it's long, usually 30-40 characters)
- No spaces before or after
- It might be all on one line or split across lines

### Step 8: Save Your API Key

**Method 1: .env File (Recommended)**
1. Create file: `turkish_finance_ml/.env`
2. Add this line:
   ```
   EVDS_API_KEY=abc123def456ghi789jkl012mno345pqr678
   ```
3. Replace with YOUR actual key

**Method 2: In Notebook (Quick Test)**
1. Open `notebooks/01_data_collection.ipynb`
2. Find: `EVDS_API_KEY = "YOUR_API_KEY"`
3. Replace with: `EVDS_API_KEY = "your_actual_key_here"`

---

## 🔍 What to Look For

### Turkish Terms (if page is in Turkish):
- **Kayıt Ol** = Register
- **Giriş Yap** = Login
- **API Anahtarı** = API Key
- **Profil** = Profile
- **Hesap Ayarları** = Account Settings
- **Kopyala** = Copy

### English Terms:
- Register
- Login
- API Key
- Profile
- Account Settings
- Copy

---

## ✅ Verification Checklist

After getting your API key, verify:
- [ ] Key is 30-40 characters long
- [ ] No spaces in the key
- [ ] Copied completely (check beginning and end)
- [ ] Saved securely (.env file or notebook)
- [ ] Tested in notebook (see test cell)

---

## 🧪 Test Your Key

After copying your key, test it:

**In Python/Notebook:**
```python
from src.data_collection import TurkishFinancialDataCollector

collector = TurkishFinancialDataCollector(
    evds_api_key="your_key_here"
)

# Test with small date range
test = collector.collect_cbrt_macroeconomic_data(
    start_date="01-01-2023",
    end_date="31-12-2023"
)

if not test.empty:
    print("✅ API Key works!")
else:
    print("❌ Check your key")
```

**Or use the test script:**
```bash
python src/test_api_key.py
```

---

## 🆘 Still Can't Find It?

1. **Check if you're logged in** - API key is only visible when logged in
2. **Look in different menus** - Try "Settings", "Profile", "Account"
3. **Use browser search** - Press Ctrl+F and search for "API"
4. **Check help section** - EVDS website has documentation
5. **Contact support** - CBRT has support if you're stuck

---

## 🎉 Success!

Once you have your API key working, you can:
- ✅ Collect real Turkish macroeconomic data
- ✅ Get inflation, interest rates, exchange rates
- ✅ Build your ML project with official data
- ✅ All for FREE!

Good luck! 🚀
