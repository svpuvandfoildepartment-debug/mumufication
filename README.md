# 📈 AI Stock Screener + Paper Trading Simulator

Professional-grade stock screening and paper trading simulator for US (NYSE/NASDAQ) and Indian (NSE) markets.

## 🎯 Two Versions Available

### Version 1: Single-User (Original)
**File:** `ai_stock_screener.py`

✅ Best for:
- Quick demos
- Personal use
- Testing strategies
- Learning how the screener works

Features:
- Combined US+India screening
- Single portfolio
- Session-based (resets on close)
- Simpler codebase

### Version 2: Multi-User with Persistent Storage ⭐ NEW
**File:** `ai_stock_screener_multi_user.py`

✅ Best for:
- Multiple users sharing one deployment
- Data persistence across restarts
- Separate US/India strategies
- Production use

Features:
- 👤 Simple username login (no passwords)
- 💾 Auto-saves everything to JSON
- 🇺🇸 🇮🇳 Separate strategies for each market
- 📊 Independent screeners per market
- 💹 Separate portfolios (USD for US, INR for India)
- 📅 Daily P&L ledgers per market
- 🔄 Data persists across app restarts

---

## 🚀 Quick Start

### Prerequisites
```bash
pip install streamlit requests pandas numpy plotly cryptography python-dateutil pytz ta
```

### Run Single-User Version
```bash
streamlit run ai_stock_screener.py
```

### Run Multi-User Version
```bash
streamlit run ai_stock_screener_multi_user.py
```

---

## 📦 What's Included

| File | Description |
|------|-------------|
| `ai_stock_screener.py` | Original single-user version |
| `ai_stock_screener_multi_user.py` | Multi-user with persistence |
| `requirements.txt` | Python dependencies |
| `UPGRADE_GUIDE.md` | Detailed migration guide |
| `.gitignore` | Git ignore rules (excludes user_data/) |

---

## 🎨 Features (Both Versions)

### AI Stock Screener
- **40% Technical Analysis** — RSI, MACD, SMA, patterns, momentum, volume
- **40% Fundamental Analysis** — P/E, EPS growth, ROE, Debt/Equity
- **20% Market Context** — Sector strength, volatility regime
- **Pattern Detection** — Double bottom, breakout, cup & handle, golden cross, etc.

### Paper Trading Simulator
- Virtual capital management
- AI-powered trade selection
- Stop-loss and take-profit automation
- Position sizing (2% risk per trade)
- Full transaction ledger
- Performance analytics
- Self-improvement engine

### UI/UX
- 🌓 Dark/Light mode toggle
- 🇺🇸 🇮🇳 Market toggle switches
- 📊 Interactive Plotly charts
- 📈 Candlestick + volume + RSI charts
- 💹 Real-time portfolio tracking
- 📅 Daily P&L ledger with bar charts

---

## 🔑 API Keys

Get your free Finnhub API key:
1. Visit [finnhub.io](https://finnhub.io)
2. Sign up (30 seconds)
3. Copy your API key
4. Paste into the app sidebar

**Without API keys:** App runs in demo mode with synthetic data.

---

## 📊 Data Coverage

| Market | Exchange | Stocks | Sectors |
|--------|----------|--------|---------|
| 🇺🇸 US | NYSE / NASDAQ | ~200 | Tech, Finance, Healthcare, Consumer, Energy, Industrials, Materials, REITs, Utilities |
| 🇮🇳 India | NSE | ~85 | Nifty 50, Nifty Next 50, Mid-cap IT, Banking, Pharma, Auto, FMCG |

**Total: ~285 pure equity stocks** — No ETFs, no crypto, no commodities.

---

## 🎓 How It Works

### 1. Screening Process
```
User Sets Params → Fetch Live Prices → Compute Indicators → 
Detect Patterns → Score Fundamentals → Calculate Composite AI Score → 
Filter & Rank → Return Top 20
```

### 2. AI Scoring Formula
```
AI Score = (Tech Score × 40%) + (Fund Score × 40%) + (Market Context × 20%)
```

### 3. Paper Trading Loop
```
Screen Stocks → Select High Scores → Enter Positions → 
Monitor Price → Check Stop/Target → Close Trades → 
Record P&L → Analyze Performance → Suggest Improvements → Repeat
```

---

## ⚙️ Deployment

### Local
```bash
git clone <your-repo>
cd <your-repo>
pip install -r requirements.txt
streamlit run ai_stock_screener_multi_user.py
```

### Streamlit Cloud
1. Push both `.py` files to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app**
4. Select your repo
5. Choose main file: `ai_stock_screener_multi_user.py`
6. Deploy

---

## 🔒 Security

- API keys encrypted with `cryptography.fernet`
- Keys stored in session memory only
- Never logged or saved to disk
- User data isolated per username
- All data stored locally in `user_data/` folder

---

## ⚠️ Disclaimer

**FOR SIMULATION PURPOSES ONLY — NOT FINANCIAL ADVICE**

This tool is for educational and paper trading simulation. It does not provide financial, investment, or trading advice. Past performance of simulated trades does not guarantee future real-world results.

Stock markets involve substantial risk including total loss of capital. Always consult a licensed financial advisor before making investment decisions.

---

## 📝 License

MIT License — See LICENSE file for details.

---

## 🤝 Contributing

PRs welcome! Please:
1. Fork the repo
2. Create a feature branch
3. Test both versions
4. Submit PR with description

---

## 📧 Support

- **Issues:** Open a GitHub issue
- **Questions:** Check `UPGRADE_GUIDE.md`
- **Updates:** Watch the repo for releases

---

Built with ❤️ using Streamlit, Plotly, and the Finnhub API.
