# 🚀 Upgrade Guide: Multi-User Version

## What Changed

### 📦 Two Files Now Available

| File | Use Case |
|------|----------|
| **`ai_stock_screener.py`** | Original single-user version with combined US+India screening |
| **`ai_stock_screener_multi_user.py`** | **NEW** — Multi-user with persistent storage and separate US/India strategies |

---

## ✨ New Features in Multi-User Version

### 1. 👤 Simple Username Login
- No passwords required
- Each user gets isolated data storage
- Data persists across app restarts
- Shows list of registered users

### 2. 🇺🇸 🇮🇳 Separate Strategies
- **US Market Strategy** — optimized for NYSE/NASDAQ (higher liquidity, tighter spreads)
- **India Market Strategy** — optimized for NSE (higher volatility, wider stops)
- Each market has its own:
  - Strategy parameters (tech/fund weights, stop loss %, etc.)
  - Paper trading portfolio
  - Daily P&L ledger
  - Screener results cache

### 3. 💾 Persistent Storage
Everything auto-saves to `user_data/<username>/session_data.json`:
- ✅ Strategy parameters (US & India)
- ✅ Screener results
- ✅ Portfolio state (cash, positions, closed trades)
- ✅ Daily P&L ledger
- ✅ Trade logs
- ✅ All preferences (dark mode, price ranges, etc.)

### 4. 📊 Separate Screener Tabs
- **US Screener** — scans ~200 US stocks
- **India Screener** — scans ~85 NSE stocks
- Run independently, cached separately
- Last scan time tracked per market

### 5. 💹 Separate Simulators
- **US Simulator** — trades in USD ($)
- **India Simulator** — trades in INR (₹)
- Independent portfolios
- Separate daily ledgers

---

## 📁 File Structure After First Use

```
your-project/
├── ai_stock_screener.py              # Original (keep in git)
├── ai_stock_screener_multi_user.py   # New multi-user version
├── requirements.txt
├── user_data/                         # Auto-created (add to .gitignore!)
│   ├── john_doe/
│   │   └── session_data.json
│   ├── alice/
│   │   └── session_data.json
│   └── bob/
│       └── session_data.json
└── README.md
```

---

## 🔄 Migration Steps

### Option A: Fresh Start (Recommended)
1. Keep your current `ai_stock_screener.py` as-is in git
2. Add `ai_stock_screener_multi_user.py` to your repo
3. Run the multi-user version locally
4. Create a username (e.g., your name)
5. Reconfigure strategies in the Settings tab

### Option B: Deploy Both
Deploy both files to Streamlit Cloud:
1. **Main app** (single-user): `streamlit run ai_stock_screener.py`
2. **Multi-user app**: `streamlit run ai_stock_screener_multi_user.py`

Each gets its own URL. Keep both for different use cases.

---

## ⚙️ Updated `.gitignore`

Add this to your `.gitignore`:

```gitignore
# User data (contains personal portfolios and strategies)
user_data/
*.json

# Python cache
__pycache__/
*.pyc
.pytest_cache/

# Streamlit cache
.streamlit/
```

---

## 🎯 When to Use Which Version

| Scenario | Use This Version |
|----------|------------------|
| Quick demo / testing | `ai_stock_screener.py` (original) |
| Multiple people sharing one deployment | `ai_stock_screener_multi_user.py` |
| Want data to persist across restarts | `ai_stock_screener_multi_user.py` |
| Want separate US/India strategies | `ai_stock_screener_multi_user.py` |
| Deploy on Streamlit Cloud | Either — both work! |

---

## 🆕 What Stays the Same

- ✅ API key encryption
- ✅ Dark/light mode toggle
- ✅ All AI scoring logic
- ✅ Pattern detection
- ✅ Candlestick charts
- ✅ Full trading ledger
- ✅ Performance analytics
- ✅ Self-improvement suggestions

---

## 🐛 Troubleshooting

**Q: I lost my data!**  
A: Data is saved in `user_data/<your_username>/`. Make sure you log in with the same username.

**Q: Can I export my data?**  
A: Yes! Your `session_data.json` file contains everything. Just copy it.

**Q: Can I delete a user?**  
A: Yes, just delete the folder in `user_data/<username>/`.

**Q: Does this work on Streamlit Cloud?**  
A: Yes! The `user_data/` folder persists as long as the app stays running. If the app restarts, you may lose data. For production, consider using a database (MongoDB, PostgreSQL, etc.).

---

## 🔮 Future Enhancements (Coming Soon)

- [ ] Password protection
- [ ] Export portfolio to CSV
- [ ] Real-time price updates
- [ ] Email alerts on trade executions
- [ ] Database backend (PostgreSQL/MongoDB)
- [ ] Trade copying between users
- [ ] Leaderboard

---

**Questions?** Open an issue on GitHub or check the inline code comments.
