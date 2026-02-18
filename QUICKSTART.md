# 🚀 QUICKSTART: Multi-User Version

## ❗ Important: Two Separate Files

You now have **TWO different apps**:

| File | Login? | Data Saved? | When to Use |
|------|--------|-------------|-------------|
| `ai_stock_screener.py` | ❌ No | ❌ No | Quick demo, testing |
| `ai_stock_screener_multi_user.py` | ✅ Yes | ✅ Yes | Persistent data, multiple users |

**⚠️ They are NOT the same!** If you run `ai_stock_screener.py`, you won't see a login screen.

---

## Step 1: Choose Which Version to Run

### Option A: Original (No Login)
```bash
streamlit run ai_stock_screener.py
```
→ Starts immediately, no login needed

### Option B: Multi-User (With Login) ⭐
```bash
streamlit run ai_stock_screener_multi_user.py
```
→ Shows login screen first

---

## Step 2: What You'll See (Multi-User Version)

### 🔐 Login Screen
When you run the **multi-user version**, you'll see:

```
┌─────────────────────────────────────────┐
│                                         │
│      📈 AI Stock Screener              │
│                                         │
│  Multi-user trading simulator with     │
│  persistent data storage                │
│                                         │
├─────────────────────────────────────────┤
│                                         │
│     👤 Enter Your Username              │
│                                         │
│  ┌───────────────────────────────────┐ │
│  │ Enter username (e.g., john_doe)   │ │
│  └───────────────────────────────────┘ │
│                                         │
│  No password needed — just a simple    │
│  name to keep your data separate       │
│                                         │
│  ┌──────────┐  ┌──────────────────┐   │
│  │🆕 Create │  │ 🔓 Login Existing│   │
│  │ New User │  │                  │   │
│  └──────────┘  └──────────────────┘   │
│                                         │
│  Registered users (0):                 │
│  (none yet)                             │
│                                         │
└─────────────────────────────────────────┘
```

### ✅ What to Do

**First Time:**
1. Type a username (e.g., `john`)
2. Click **🆕 Create New User**
3. App loads with empty portfolio

**Returning User:**
1. Type your username
2. Click **🔓 Login Existing**
3. Your saved data loads automatically

---

## Step 3: Setting Up API Keys

After login, you'll see the main app with a sidebar:

```
Sidebar:
┌────────────────────────────┐
│ Logged in as               │
│ 👤 john                    │
├────────────────────────────┤
│ 🚪 Logout                  │
├────────────────────────────┤
│ 🎨 DISPLAY                 │
│ 🌙  Dark Mode  [ON]        │
├────────────────────────────┤
│ 🔑 API CONFIGURATION       │
│ ✅ Provider: Finnhub       │
│ [🔄 Reset API Keys]        │
└────────────────────────────┘
```

**To add/change API:**
1. Click **🔄 Reset API Keys**
2. Select provider (Finnhub, Alpha Vantage, or Twelve Data)
3. Paste your API key
4. Click **🔒 Secure & Save Keys**

---

## Step 4: Using the App

### Tab 1: 🔍 Screener
- **Two sub-tabs**: 🇺🇸 US Markets | 🇮🇳 Indian Markets
- Run each screener independently
- Results saved separately per market

### Tab 2: 💹 Simulator
- **Two sub-tabs**: 🇺🇸 US Markets | 🇮🇳 Indian Markets
- Separate portfolios (USD for US, INR for India)
- Independent trading per market

### Tab 3: ⚙️ Settings
- **Two sub-tabs**: 🇺🇸 US Strategy | 🇮🇳 India Strategy
- Different parameters per market
- US uses 10% stop loss, India uses 12%

### Tab 4: 📈 Performance
- **Two sub-tabs**: 🇺🇸 US Performance | 🇮🇳 India Performance
- Separate analytics per market

---

## Troubleshooting

### Problem: No login screen appears
**Solution:** Make sure you're running the right file:
```bash
# WRONG - this is the old version
streamlit run ai_stock_screener.py

# CORRECT - this shows login
streamlit run ai_stock_screener_multi_user.py
```

### Problem: Can't find my username
**Check the folder:**
```bash
ls user_data/
```
Your username should be there. If not, create a new user.

### Problem: Data disappeared
**Two possible causes:**
1. You logged in with a different username
2. The `user_data/` folder was deleted

**Solution:** Always use the same username.

### Problem: API provider dropdown missing
**Solution:** Click **🔄 Reset API Keys** in the sidebar to see the provider selection.

---

## File Structure After First Login

```
your-project/
├── ai_stock_screener.py                 # Original (no login)
├── ai_stock_screener_multi_user.py      # Multi-user (with login)
├── requirements.txt
├── test_multi_user.py                   # Test script
└── user_data/                           # Created after first login
    └── john/                            # Your username
        └── session_data.json            # Your saved data
```

---

## Quick Test

Run the test script:
```bash
python3 test_multi_user.py
```

It will:
- ✅ Check file exists
- ✅ Validate syntax
- ✅ Check dependencies
- ✅ Launch the app

---

## Still Having Issues?

### Check which file is running:
Look at the **browser tab title**:
- "AI Stock Screener + Simulator" = old version (no login)
- "AI Stock Screener + Simulator (Multi-User)" = new version (with login)

### Force restart:
```bash
# Stop streamlit
Ctrl+C

# Clear cache
rm -rf ~/.streamlit/cache

# Run again
streamlit run ai_stock_screener_multi_user.py
```

---

## Summary

| You Want | Run This Command |
|----------|------------------|
| Quick demo, no login | `streamlit run ai_stock_screener.py` |
| Login + saved data | `streamlit run ai_stock_screener_multi_user.py` |
| Test if it works | `python3 test_multi_user.py` |

**The multi-user version ALWAYS shows a login screen first.**

If you don't see a login screen, you're running the wrong file!
