"""
╔══════════════════════════════════════════════════════════════════════╗
║     AI STOCK SCREENER + PAPER TRADING SIMULATOR                      ║
║     US Stocks (NYSE/NASDAQ) + Indian Stocks (NSE)                    ║
║     Pure Share Market Only — No ETFs, No Crypto, No Commodities      ║
║     Disclaimer: Simulation only — NOT financial advice               ║
╚══════════════════════════════════════════════════════════════════════╝

Required packages:
  pip install streamlit requests pandas numpy plotly cryptography python-dateutil pytz

Optional (for advanced TA):
  pip install ta
"""

# ─── IMPORTS ────────────────────────────────────────────────────────────────
import streamlit as st
import requests
import pandas as pd
import numpy as np
import json
import os
import time
import math
import pickle
import hashlib
import threading
import random
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
import warnings
warnings.filterwarnings("ignore")

# Cryptography
try:
    from cryptography.fernet import Fernet
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False

# Plotly
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# TA library (pure Python alternative to TA-Lib)
try:
    import ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False

# ─── CONSTANTS ───────────────────────────────────────────────────────────────
INR_TO_USD = 1 / 83.5
USER_DATA_DIR = "user_data"
os.makedirs(USER_DATA_DIR, exist_ok=True)

# ─── US STOCKS — NYSE / NASDAQ (pure equities only) ──────────────────────────
US_TICKERS = [
    # Technology
    "AAPL","MSFT","GOOGL","AMZN","NVDA","META","TSLA","AVGO","ORCL","ADBE",
    "CRM","AMD","INTC","QCOM","TXN","MU","AMAT","LRCX","KLAC","MRVL",
    "SHOP","NET","SNOW","DDOG","CRWD","ZS","PANW","FTNT","OKTA","PLTR",
    "RBLX","UBER","ABNB","DASH","RIVN","SOFI","HOOD","SQ","PYPL","TWLO",
    # Finance / Banking
    "JPM","BAC","WFC","GS","MS","C","BLK","AXP","V","MA",
    "COF","USB","TFC","PNC","SCHW","ICE","CME","SPGI","MCO","ALLY",
    # Healthcare / Pharma / Biotech
    "JNJ","UNH","PFE","ABBV","LLY","MRK","TMO","ABT","DHR","BMY",
    "AMGN","GILD","ISRG","REGN","VRTX","MDT","BSX","EW","ZTS","DXCM",
    "MRNA","BIIB","ILMN","IDXX","BAX","BDX","IQV","CNC","HUM","CI",
    # Consumer Discretionary
    "WMT","HD","COST","MCD","SBUX","NKE","TGT","LOW","BKNG",
    "MAR","HLT","YUM","CMG","DG","DLTR","KR","CVS","TSCO","ROST",
    # Consumer Staples
    "KO","PEP","PG","CL","EL","GIS","CPB","MO","PM","BTI","KHC","STZ",
    # Energy (oil & gas companies — not commodity ETFs)
    "XOM","CVX","COP","EOG","SLB","MPC","PSX","VLO","HAL","BKR",
    "DVN","PXD","FANG","MRO","OXY","HES","APA","RRC","AR","CHK",
    # Industrials / Aerospace / Defense
    "CAT","DE","GE","HON","MMM","RTX","LMT","NOC","BA","UPS",
    "FDX","CSX","NSC","UNP","EMR","ITW","ROK","PH","GWW","CMI",
    # Materials & Chemicals (stocks not commodity ETFs)
    "LIN","APD","SHW","FCX","NEM","NUE","CF","MOS","ALB","CE",
    # Real Estate (REITs — share market listed companies)
    "AMT","PLD","EQIX","CCI","SPG","PSA","EXR","AVB","EQR","DLR",
    # Utilities
    "NEE","DUK","SO","D","AEP","EXC","SRE","PEG","XEL","ED",
    # Communication
    "T","VZ","CMCSA","NFLX","DIS","CHTR","PARA","WBD","FOX","OMC",
    # Growth / Small-mid cap
    "LCID","RIVN","NKLA","WKHS","XPEV","NIO","LI","GRAB","SE","DKNG",
]

# ─── INDIA STOCKS — NSE (pure equities only) ─────────────────────────────────
INDIA_TICKERS = [
    # Nifty 50
    "RELIANCE.NS","TCS.NS","HDFCBANK.NS","INFY.NS","ICICIBANK.NS",
    "HINDUNILVR.NS","KOTAKBANK.NS","SBIN.NS","BAJFINANCE.NS","BHARTIARTL.NS",
    "ITC.NS","ASIANPAINT.NS","AXISBANK.NS","MARUTI.NS","SUNPHARMA.NS",
    "TITAN.NS","WIPRO.NS","ULTRACEMCO.NS","NESTLEIND.NS","TECHM.NS",
    "HCLTECH.NS","ONGC.NS","POWERGRID.NS","NTPC.NS","DIVISLAB.NS",
    "DRREDDY.NS","CIPLA.NS","EICHERMOT.NS","HINDALCO.NS","JSWSTEEL.NS",
    "TATASTEEL.NS","TATAMOTORS.NS","M&M.NS","BAJAJFINSV.NS","INDUSINDBK.NS",
    "BPCL.NS","ADANIENT.NS","ADANIPORTS.NS","GRASIM.NS","HEROMOTOCO.NS",
    "COALINDIA.NS","BRITANNIA.NS","SHREECEM.NS","SBILIFE.NS","HDFCLIFE.NS",
    "APOLLOHOSP.NS","TATACONSUM.NS","LT.NS","PIDILITIND.NS","HAVELLS.NS",
    # Nifty Next 50 / Mid-cap
    "DMART.NS","BERGEPAINT.NS","GODREJCP.NS","MUTHOOTFIN.NS","SIEMENS.NS",
    "BANDHANBNK.NS","FEDERALBNK.NS","IDFCFIRSTB.NS","RBLBANK.NS","AUBANK.NS",
    "TRENT.NS","NAUKRI.NS","INDIGO.NS","IRCTC.NS","ZOMATO.NS",
    "PAYTM.NS","NYKAA.NS","POLICYBZR.NS","DELHIVERY.NS","CARTRADE.NS",
    "VEDL.NS","SAIL.NS","NMDC.NS","MOIL.NS","NATIONALUM.NS",
    "TORNTPHARM.NS","LUPIN.NS","ALKEM.NS","AUROPHARMA.NS","BIOCON.NS",
    "BALKRISIND.NS","MOTHERSON.NS","BHARAT FORGE.NS","CUMMINSIND.NS","SCHAEFFLER.NS",
    "ICICIPRULI.NS","ICICIGI.NS","BAJAJHLDNG.NS","CHOLAFIN.NS","MANAPPURAM.NS",
    "VOLTAS.NS","BLUESTARCO.NS","WHIRLPOOL.NS","CROMPTON.NS","POLYCAB.NS",
    "PHOENIXLTD.NS","DLF.NS","GODREJPROP.NS","OBEROIRLTY.NS","PRESTIGE.NS",
]

ALL_TICKERS = US_TICKERS + INDIA_TICKERS

# ─── SECURITY HELPERS ────────────────────────────────────────────────────────
def get_or_create_fernet_key() -> bytes:
    """Derive/store a Fernet encryption key (session-scoped)."""
    if "fernet_key" not in st.session_state:
        st.session_state.fernet_key = Fernet.generate_key()
    return st.session_state.fernet_key


def encrypt_api_key(api_key: str) -> bytes:
    """Encrypt an API key string."""
    if not CRYPTO_AVAILABLE or not api_key:
        return api_key.encode() if api_key else b""
    fernet = Fernet(get_or_create_fernet_key())
    return fernet.encrypt(api_key.encode())


def decrypt_api_key(encrypted: bytes) -> str:
    """Decrypt an API key."""
    if not CRYPTO_AVAILABLE or not encrypted:
        return encrypted.decode() if encrypted else ""
    try:
        fernet = Fernet(get_or_create_fernet_key())
        return fernet.decrypt(encrypted).decode()
    except Exception:
        return ""


def mask_key(key: str) -> str:
    """Return masked version of key for display."""
    if not key or len(key) < 8:
        return "****"
    return key[:4] + "****" + key[-4:]


# ─── DATA FETCHING ────────────────────────────────────────────────────────────
class DataFetcher:
    """Handles all market data retrieval with rate-limit handling and caching."""

    def __init__(self, finnhub_key: str, av_key: str = "", td_key: str = ""):
        self.finnhub_key = finnhub_key
        self.av_key = av_key
        self.td_key = td_key
        self._cache: Dict = {}
        self._cache_ttl = 300  # 5 min
        self._last_call: Dict[str, float] = {}
        self._rate_limits = {
            "finnhub": 0.5,   # 60 calls/min → ~1 per sec (safe)
            "alphavantage": 13.0,  # 5 calls/min free tier
            "twelvedata": 1.0,
        }

    def _rate_limit(self, source: str):
        """Enforce per-source rate limiting."""
        now = time.time()
        min_gap = self._rate_limits.get(source, 1.0)
        elapsed = now - self._last_call.get(source, 0)
        if elapsed < min_gap:
            time.sleep(min_gap - elapsed)
        self._last_call[source] = time.time()

    def _cached(self, key: str):
        """Return cached value if not expired."""
        if key in self._cache:
            val, ts = self._cache[key]
            if time.time() - ts < self._cache_ttl:
                return val
        return None

    def _store(self, key: str, val):
        self._cache[key] = (val, time.time())

    def get_quote_finnhub(self, symbol: str) -> Optional[Dict]:
        """Fetch real-time quote from Finnhub."""
        cache_key = f"quote_fh_{symbol}"
        cached = self._cached(cache_key)
        if cached:
            return cached
        if not self.finnhub_key:
            return None
        self._rate_limit("finnhub")
        try:
            url = f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={self.finnhub_key}"
            r = requests.get(url, timeout=8)
            if r.status_code == 200:
                data = r.json()
                if data.get("c", 0) > 0:
                    self._store(cache_key, data)
                    return data
        except Exception:
            pass
        return None

    def get_candles_finnhub(self, symbol: str, resolution: str = "D", count: int = 90) -> Optional[pd.DataFrame]:
        """Fetch OHLCV candle data from Finnhub."""
        cache_key = f"candles_fh_{symbol}_{resolution}_{count}"
        cached = self._cached(cache_key)
        if cached is not None:
            return cached
        if not self.finnhub_key:
            return None
        self._rate_limit("finnhub")
        try:
            end = int(time.time())
            # For 'D' resolution, go back ~130 trading days to get ~90 bars
            if resolution == "D":
                start = end - count * 86400 * 2
            else:
                start = end - count * 3600 * 24
            url = (
                f"https://finnhub.io/api/v1/stock/candle"
                f"?symbol={symbol}&resolution={resolution}"
                f"&from={start}&to={end}&token={self.finnhub_key}"
            )
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                d = r.json()
                if d.get("s") == "ok" and len(d.get("c", [])) >= 5:
                    df = pd.DataFrame({
                        "open": d["o"], "high": d["h"], "low": d["l"],
                        "close": d["c"], "volume": d["v"],
                        "timestamp": pd.to_datetime(d["t"], unit="s"),
                    }).set_index("timestamp")
                    df = df.tail(count)
                    self._store(cache_key, df)
                    return df
        except Exception:
            pass
        return None

    def get_fundamentals_finnhub(self, symbol: str) -> Optional[Dict]:
        """Fetch company fundamentals from Finnhub."""
        cache_key = f"fundamentals_fh_{symbol}"
        cached = self._cached(cache_key)
        if cached:
            return cached
        if not self.finnhub_key:
            return None
        self._rate_limit("finnhub")
        try:
            url = f"https://finnhub.io/api/v1/stock/metric?symbol={symbol}&metric=all&token={self.finnhub_key}"
            r = requests.get(url, timeout=8)
            if r.status_code == 200:
                data = r.json()
                if data.get("metric"):
                    self._store(cache_key, data)
                    return data
        except Exception:
            pass
        return None

    def get_company_profile_finnhub(self, symbol: str) -> Optional[Dict]:
        """Fetch company profile (sector, industry, etc.)."""
        cache_key = f"profile_fh_{symbol}"
        cached = self._cached(cache_key)
        if cached:
            return cached
        if not self.finnhub_key:
            return None
        self._rate_limit("finnhub")
        try:
            url = f"https://finnhub.io/api/v1/stock/profile2?symbol={symbol}&token={self.finnhub_key}"
            r = requests.get(url, timeout=8)
            if r.status_code == 200:
                data = r.json()
                if data.get("ticker"):
                    self._store(cache_key, data)
                    return data
        except Exception:
            pass
        return None

    def get_quote_av(self, symbol: str) -> Optional[Dict]:
        """Fallback: Alpha Vantage global quote."""
        if not self.av_key:
            return None
        cache_key = f"quote_av_{symbol}"
        cached = self._cached(cache_key)
        if cached:
            return cached
        self._rate_limit("alphavantage")
        try:
            url = (
                f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE"
                f"&symbol={symbol}&apikey={self.av_key}"
            )
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                data = r.json()
                gq = data.get("Global Quote", {})
                if gq.get("05. price"):
                    result = {
                        "c": float(gq["05. price"]),
                        "o": float(gq["02. open"]),
                        "h": float(gq["03. high"]),
                        "l": float(gq["04. low"]),
                        "pc": float(gq["08. previous close"]),
                        "pct_change": float(gq["10. change percent"].replace("%", "")),
                    }
                    self._store(cache_key, result)
                    return result
        except Exception:
            pass
        return None

    def get_quote(self, symbol: str) -> Optional[Dict]:
        """Get quote with auto-fallback."""
        q = self.get_quote_finnhub(symbol)
        if q:
            return q
        return self.get_quote_av(symbol)

    def get_candles(self, symbol: str, resolution: str = "D", count: int = 90) -> Optional[pd.DataFrame]:
        """Get candles with fallback to synthetic demo data if no API."""
        df = self.get_candles_finnhub(symbol, resolution, count)
        if df is not None and len(df) >= 10:
            return df
        # Generate synthetic OHLCV for demo/testing
        return self._synthetic_candles(symbol, count)

    def _synthetic_candles(self, symbol: str, count: int = 90) -> pd.DataFrame:
        """Generate realistic synthetic OHLCV data for demo purposes."""
        seed = int(hashlib.md5(symbol.encode()).hexdigest(), 16) % (2**32)
        rng = np.random.default_rng(seed)
        base = rng.uniform(10, 500)
        returns = rng.normal(0.0005, 0.018, count)
        prices = base * np.cumprod(1 + returns)
        highs = prices * (1 + np.abs(rng.normal(0, 0.01, count)))
        lows = prices * (1 - np.abs(rng.normal(0, 0.01, count)))
        opens = np.roll(prices, 1)
        opens[0] = prices[0]
        volumes = rng.integers(500_000, 10_000_000, count).astype(float)
        # Occasionally add volume spikes
        spike_idx = rng.integers(0, count, 5)
        volumes[spike_idx] *= rng.uniform(2, 5, 5)
        dates = pd.date_range(end=datetime.now(), periods=count, freq="B")
        return pd.DataFrame({
            "open": opens, "high": highs, "low": lows,
            "close": prices, "volume": volumes,
        }, index=dates)


# ─── TECHNICAL ANALYSIS ──────────────────────────────────────────────────────
class TechnicalAnalyzer:
    """Compute technical indicators and detect patterns."""

    @staticmethod
    def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def compute_macd(series: pd.Series, fast=12, slow=26, signal=9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        sig = macd.ewm(span=signal, adjust=False).mean()
        hist = macd - sig
        return macd, sig, hist

    @staticmethod
    def compute_bollinger(series: pd.Series, window=20, std_dev=2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        mid = series.rolling(window).mean()
        std = series.rolling(window).std()
        return mid + std_dev * std, mid, mid - std_dev * std

    @staticmethod
    def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        high, low, close = df["high"], df["low"], df["close"]
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    @staticmethod
    def compute_sma(series: pd.Series, period: int) -> pd.Series:
        return series.rolling(period).mean()

    @staticmethod
    def compute_ema(series: pd.Series, period: int) -> pd.Series:
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def compute_stochastic(df: pd.DataFrame, k=14, d=3) -> Tuple[pd.Series, pd.Series]:
        low_min = df["low"].rolling(k).min()
        high_max = df["high"].rolling(k).max()
        pct_k = 100 * (df["close"] - low_min) / (high_max - low_min + 1e-10)
        pct_d = pct_k.rolling(d).mean()
        return pct_k, pct_d

    @staticmethod
    def detect_patterns(df: pd.DataFrame) -> Dict[str, bool]:
        """Detect common chart patterns via rule-based logic."""
        patterns = {}
        close = df["close"].values
        volume = df["volume"].values
        n = len(close)
        if n < 20:
            return patterns

        # ── Double Bottom ─────────────────────────
        # Look for two troughs of similar level separated by a peak
        try:
            window = min(20, n // 2)
            lows_idx = []
            for i in range(1, n - 1):
                if close[i] < close[i-1] and close[i] < close[i+1]:
                    lows_idx.append(i)
            if len(lows_idx) >= 2:
                l1, l2 = lows_idx[-2], lows_idx[-1]
                if l2 - l1 >= 5:
                    diff = abs(close[l1] - close[l2]) / (close[l1] + 1e-10)
                    patterns["double_bottom"] = diff < 0.05
        except Exception:
            patterns["double_bottom"] = False

        # ── Breakout (price above 52-week high) ───
        try:
            high_52 = np.max(close[:-1]) if n > 1 else close[-1]
            patterns["breakout"] = float(close[-1]) > float(high_52) * 0.98
        except Exception:
            patterns["breakout"] = False

        # ── Cup & Handle ──────────────────────────
        try:
            if n >= 30:
                first_third = close[:n // 3]
                mid_third = close[n // 3 : 2 * n // 3]
                last_third = close[2 * n // 3:]
                cup = (np.mean(first_third) > np.mean(mid_third) * 0.97 and
                       np.mean(last_third) > np.mean(mid_third) * 0.97 and
                       close[-1] > np.mean(first_third))
                patterns["cup_handle"] = cup
            else:
                patterns["cup_handle"] = False
        except Exception:
            patterns["cup_handle"] = False

        # ── Golden Cross (50 SMA > 200 SMA) ───────
        try:
            if n >= 50:
                sma50 = np.mean(close[-50:])
                sma200 = np.mean(close[-min(200, n):])
                patterns["golden_cross"] = sma50 > sma200
            else:
                patterns["golden_cross"] = False
        except Exception:
            patterns["golden_cross"] = False

        # ── Volume Surge ──────────────────────────
        try:
            avg_vol = np.mean(volume[-20:-1]) if n >= 20 else np.mean(volume[:-1])
            patterns["volume_surge"] = volume[-1] > avg_vol * 1.5
        except Exception:
            patterns["volume_surge"] = False

        # ── Oversold Bounce ───────────────────────
        try:
            close_series = pd.Series(close)
            rsi = TechnicalAnalyzer.compute_rsi(close_series)
            rsi_vals = rsi.dropna().values
            if len(rsi_vals) >= 3:
                patterns["oversold_bounce"] = (rsi_vals[-3] < 30 and rsi_vals[-1] > 35)
            else:
                patterns["oversold_bounce"] = False
        except Exception:
            patterns["oversold_bounce"] = False

        return patterns

    @staticmethod
    def compute_all_indicators(df: pd.DataFrame) -> Dict:
        """Return dict of all key technical indicator values."""
        close = df["close"]
        volume = df["volume"]
        result = {}

        try:
            rsi = TechnicalAnalyzer.compute_rsi(close)
            result["rsi"] = float(rsi.iloc[-1]) if not rsi.empty else 50.0
        except Exception:
            result["rsi"] = 50.0

        try:
            macd, sig, hist = TechnicalAnalyzer.compute_macd(close)
            result["macd"] = float(macd.iloc[-1]) if not macd.empty else 0.0
            result["macd_signal"] = float(sig.iloc[-1]) if not sig.empty else 0.0
            result["macd_hist"] = float(hist.iloc[-1]) if not hist.empty else 0.0
            result["macd_bullish"] = result["macd"] > result["macd_signal"]
        except Exception:
            result["macd"] = 0.0
            result["macd_bullish"] = False

        try:
            sma20 = TechnicalAnalyzer.compute_sma(close, 20)
            sma50 = TechnicalAnalyzer.compute_sma(close, min(50, len(close) - 1))
            result["sma20"] = float(sma20.iloc[-1]) if not sma20.empty else float(close.iloc[-1])
            result["sma50"] = float(sma50.iloc[-1]) if not sma50.empty else float(close.iloc[-1])
            result["above_sma20"] = float(close.iloc[-1]) > result["sma20"]
            result["above_sma50"] = float(close.iloc[-1]) > result["sma50"]
        except Exception:
            result["sma20"] = float(close.iloc[-1])
            result["sma50"] = float(close.iloc[-1])
            result["above_sma20"] = False
            result["above_sma50"] = False

        try:
            bb_upper, bb_mid, bb_lower = TechnicalAnalyzer.compute_bollinger(close)
            result["bb_upper"] = float(bb_upper.iloc[-1]) if not bb_upper.empty else 0.0
            result["bb_lower"] = float(bb_lower.iloc[-1]) if not bb_lower.empty else 0.0
            result["bb_position"] = (float(close.iloc[-1]) - result["bb_lower"]) / \
                                    (result["bb_upper"] - result["bb_lower"] + 1e-10)
        except Exception:
            result["bb_position"] = 0.5

        try:
            # Momentum: 20-day return
            if len(close) >= 21:
                result["momentum_20d"] = float(close.iloc[-1] / close.iloc[-21] - 1) * 100
            else:
                result["momentum_20d"] = 0.0
        except Exception:
            result["momentum_20d"] = 0.0

        try:
            # Volume ratio
            avg_vol = float(volume.iloc[-20:].mean()) if len(volume) >= 20 else float(volume.mean())
            result["volume_ratio"] = float(volume.iloc[-1]) / (avg_vol + 1)
        except Exception:
            result["volume_ratio"] = 1.0

        try:
            atr = TechnicalAnalyzer.compute_atr(df)
            result["atr"] = float(atr.iloc[-1]) if not atr.empty else 0.0
            result["atr_pct"] = result["atr"] / (float(close.iloc[-1]) + 1e-10) * 100
        except Exception:
            result["atr"] = 0.0
            result["atr_pct"] = 1.0

        try:
            k, d = TechnicalAnalyzer.compute_stochastic(df)
            result["stoch_k"] = float(k.iloc[-1]) if not k.empty else 50.0
            result["stoch_d"] = float(d.iloc[-1]) if not d.empty else 50.0
        except Exception:
            result["stoch_k"] = 50.0
            result["stoch_d"] = 50.0

        return result


# ─── FUNDAMENTAL SCORER ──────────────────────────────────────────────────────
class FundamentalScorer:
    """Score stocks based on fundamental metrics."""

    @staticmethod
    def score(metrics: Dict, weights: Dict) -> Tuple[float, List[str]]:
        """Return fundamental score 0–100 and list of reasons."""
        score = 0.0
        reasons = []
        m = metrics.get("metric", {})
        if not m:
            return 30.0, ["Limited fundamental data"]

        def safe_get(key):
            v = m.get(key)
            if v is None:
                return None
            try:
                return float(v)
            except Exception:
                return None

        # P/E ratio (lower is better for value, but too low could mean trouble)
        pe = safe_get("peBasicExclExtraTTM") or safe_get("peNormalizedAnnual")
        if pe is not None:
            if 5 < pe < 15:
                score += 15; reasons.append(f"Attractive P/E: {pe:.1f}")
            elif 15 <= pe < 25:
                score += 10; reasons.append(f"Fair P/E: {pe:.1f}")
            elif 25 <= pe < 40:
                score += 5
            elif pe < 0:
                score += 0; reasons.append("Negative earnings (P/E N/A)")
        else:
            score += 5

        # EPS Growth
        eps_growth = safe_get("epsGrowth3Y") or safe_get("epsGrowthTTMYoy")
        if eps_growth is not None:
            if eps_growth > 20:
                score += 15; reasons.append(f"Strong EPS growth: {eps_growth:.1f}%")
            elif eps_growth > 10:
                score += 10
            elif eps_growth > 0:
                score += 5
            else:
                reasons.append("Declining EPS")

        # ROE
        roe = safe_get("roeTTM") or safe_get("roeRfy")
        if roe is not None:
            if roe > 20:
                score += 15; reasons.append(f"High ROE: {roe:.1f}%")
            elif roe > 12:
                score += 10
            elif roe > 5:
                score += 5

        # Debt/Equity
        de = safe_get("totalDebt/totalEquityAnnual") or safe_get("longTermDebt/equityAnnual")
        if de is not None:
            if de < 0.3:
                score += 15; reasons.append("Low leverage")
            elif de < 0.8:
                score += 10
            elif de < 1.5:
                score += 5
            else:
                reasons.append(f"High D/E: {de:.2f}")

        # Revenue Growth
        rev_growth = safe_get("revenueGrowth3Y") or safe_get("revenueGrowthTTMYoy")
        if rev_growth is not None:
            if rev_growth > 15:
                score += 10; reasons.append(f"Revenue growth: {rev_growth:.1f}%")
            elif rev_growth > 5:
                score += 7
            elif rev_growth > 0:
                score += 3

        # Profit Margin
        margin = safe_get("netProfitMarginTTM") or safe_get("netProfitMarginAnnual")
        if margin is not None:
            if margin > 20:
                score += 10; reasons.append(f"Fat margins: {margin:.1f}%")
            elif margin > 10:
                score += 7
            elif margin > 0:
                score += 3

        # Current Ratio (liquidity)
        cr = safe_get("currentRatioAnnual") or safe_get("currentRatioQuarterly")
        if cr is not None:
            if cr > 2:
                score += 5; reasons.append("Strong liquidity")
            elif cr > 1.2:
                score += 3
            elif cr < 1:
                reasons.append("Tight liquidity")

        # Beta (market context)
        beta = safe_get("beta")
        if beta is not None:
            if 0.7 < beta < 1.5:
                score += 5
            elif beta < 0:
                score -= 5

        return min(float(score), 100.0), reasons


# ─── TECHNICAL SCORER ────────────────────────────────────────────────────────
class TechnicalScorer:
    """Score stocks based on technical indicators and patterns."""

    @staticmethod
    def score(indicators: Dict, patterns: Dict, weights: Dict) -> Tuple[float, List[str]]:
        """Return technical score 0–100 and list of reasons."""
        score = 0.0
        reasons = []

        # RSI score
        rsi = indicators.get("rsi", 50)
        rsi_w = weights.get("rsi_weight", 1.0)
        if 40 <= rsi <= 60:
            score += 10 * rsi_w; reasons.append(f"Neutral RSI: {rsi:.0f}")
        elif 30 <= rsi < 40:
            score += 18 * rsi_w; reasons.append(f"Oversold RSI: {rsi:.0f} ↑")
        elif 60 < rsi <= 70:
            score += 12 * rsi_w; reasons.append(f"Bullish RSI: {rsi:.0f}")
        elif rsi < 30:
            score += 8 * rsi_w; reasons.append(f"Deeply oversold: {rsi:.0f}")
        elif rsi > 80:
            score += 3 * rsi_w; reasons.append(f"Overbought RSI: {rsi:.0f}")

        # MACD
        if indicators.get("macd_bullish", False):
            score += 15; reasons.append("Bullish MACD crossover")
        else:
            score += 3

        # SMA alignment
        if indicators.get("above_sma20") and indicators.get("above_sma50"):
            score += 15; reasons.append("Price above SMA20 + SMA50")
        elif indicators.get("above_sma50"):
            score += 8
        elif indicators.get("above_sma20"):
            score += 5

        # Momentum
        mom = indicators.get("momentum_20d", 0)
        if mom > 10:
            score += 12; reasons.append(f"Strong 20d momentum: +{mom:.1f}%")
        elif mom > 3:
            score += 8
        elif mom > 0:
            score += 5
        elif mom < -10:
            score += 0; reasons.append(f"Weak momentum: {mom:.1f}%")
        else:
            score += 2

        # Volume ratio
        vol_ratio = indicators.get("volume_ratio", 1)
        if vol_ratio > 2:
            score += 8; reasons.append(f"Volume surge: {vol_ratio:.1f}x avg")
        elif vol_ratio > 1.3:
            score += 4

        # Patterns bonus
        pattern_scores = {
            "double_bottom": 8,
            "breakout": 10,
            "cup_handle": 9,
            "golden_cross": 8,
            "volume_surge": 5,
            "oversold_bounce": 7,
        }
        for pat, val in patterns.items():
            if val:
                bonus = pattern_scores.get(pat, 3)
                score += bonus
                pat_name = pat.replace("_", " ").title()
                reasons.append(f"Pattern: {pat_name}")

        # BB position
        bb_pos = indicators.get("bb_position", 0.5)
        if bb_pos < 0.2:
            score += 5; reasons.append("Near Bollinger lower band")
        elif bb_pos > 0.85:
            score -= 3

        # Stochastic
        stk = indicators.get("stoch_k", 50)
        std = indicators.get("stoch_d", 50)
        if stk < 20 and std < 20:
            score += 5; reasons.append("Stochastic oversold")
        elif stk > stk and stk < 40:
            score += 3

        return min(float(score), 100.0), reasons


# ─── COMPOSITE SCORER ────────────────────────────────────────────────────────
class CompositeScorer:
    """Combine technical + fundamental + market context into AI Score."""

    def __init__(self, weights: Dict):
        self.weights = weights

    def compute_score(
        self,
        tech_score: float,
        fund_score: float,
        market_score: float,
        tech_reasons: List[str],
        fund_reasons: List[str],
    ) -> Tuple[float, float, float, float, List[str]]:
        """
        Returns: (composite, tech_contrib, fund_contrib, mkt_contrib, reasons)
        Weights: 40% technical, 40% fundamental, 20% market context
        """
        tech_w = self.weights.get("tech_weight", 0.40)
        fund_w = self.weights.get("fund_weight", 0.40)
        mkt_w = self.weights.get("market_weight", 0.20)

        tech_contrib = tech_score * tech_w
        fund_contrib = fund_score * fund_w
        mkt_contrib = market_score * mkt_w
        composite = tech_contrib + fund_contrib + mkt_contrib

        all_reasons = tech_reasons[:3] + fund_reasons[:2]
        return composite, tech_contrib, fund_contrib, mkt_contrib, all_reasons

    def estimate_target_price(
        self,
        current_price: float,
        indicators: Dict,
        patterns: Dict,
        min_profit_pct: float,
    ) -> float:
        """Estimate a realistic target price based on technicals."""
        targets = []

        # Bollinger upper band target
        bb_upper = indicators.get("bb_upper", current_price * 1.05)
        if bb_upper > current_price:
            targets.append(bb_upper)

        # Momentum extrapolation
        mom = indicators.get("momentum_20d", 0)
        if mom > 0:
            targets.append(current_price * (1 + min(mom / 100 * 2, 0.5)))

        # Pattern targets
        if patterns.get("breakout"):
            targets.append(current_price * 1.10)
        if patterns.get("cup_handle"):
            targets.append(current_price * 1.15)
        if patterns.get("double_bottom"):
            targets.append(current_price * 1.12)

        # ATR-based target (2x ATR)
        atr = indicators.get("atr", current_price * 0.02)
        targets.append(current_price + 2 * atr)

        # Minimum target based on user preference
        min_target = current_price * (1 + min_profit_pct / 100)
        targets.append(min_target)

        if targets:
            target = float(np.median(targets))
            # Sanity check: target within 5% to 80% of current price
            target = max(current_price * 1.05, min(target, current_price * 1.8))
            return round(target, 2)
        return round(current_price * (1 + min_profit_pct / 100), 2)


# ─── STOCK SCREENER ──────────────────────────────────────────────────────────
class StockScreener:
    """Main screening engine — US stocks (NYSE/NASDAQ) + India stocks (NSE)."""

    def __init__(
        self,
        fetcher: DataFetcher,
        strategy_params: Dict,
        min_price_usd: float,
        max_price_usd: float,
        min_profit_pct: float,
        markets: List[str],
    ):
        self.fetcher = fetcher
        self.params = strategy_params
        self.min_price = min_price_usd
        self.max_price = max_price_usd
        self.min_profit_pct = min_profit_pct
        self.markets = markets
        self.ta = TechnicalAnalyzer()
        self.fs = FundamentalScorer()
        self.ts = TechnicalScorer()
        self.cs = CompositeScorer(strategy_params)

    def _get_ticker_list(self) -> List[str]:
        tickers = []
        if "US" in self.markets:
            tickers += US_TICKERS
        if "India" in self.markets:
            tickers += INDIA_TICKERS
        return tickers

    def _to_usd(self, price: float, symbol: str) -> float:
        """Convert INR to USD for Indian stocks."""
        if symbol.endswith(".NS") or symbol.endswith(".BO"):
            return price * INR_TO_USD
        return price

    def screen_single(self, symbol: str) -> Optional[Dict]:
        """Screen a single stock and return its data dict."""
        try:
            # 1. Get quote
            quote = self.fetcher.get_quote(symbol)
            if not quote:
                return None
            price_raw = float(quote.get("c", 0))
            if price_raw <= 0:
                return None

            # Convert to USD for unified price filtering
            price_usd = self._to_usd(price_raw, symbol)
            if price_usd < self.min_price or price_usd > self.max_price:
                return None

            # 3. Get OHLCV candles
            df = self.fetcher.get_candles(symbol)
            if df is None or len(df) < 10:
                return None

            # 4. Technical analysis
            indicators = TechnicalAnalyzer.compute_all_indicators(df)
            patterns = TechnicalAnalyzer.detect_patterns(df)

            # 5. Technical score
            tech_score, tech_reasons = self.ts.score(
                indicators, patterns, self.params
            )

            # 6. Fundamentals
            fund_score, fund_reasons = 40.0, ["Using default fundamentals"]
            fundamentals = {}
            if self.fetcher.finnhub_key:
                fund_data = self.fetcher.get_fundamentals_finnhub(symbol)
                if fund_data:
                    fundamentals = fund_data
                    fund_score, fund_reasons = self.fs.score(fund_data, self.params)

            # 7. Market context score (simplified)
            market_score = self._market_context_score(indicators, symbol)

            # 8. Composite AI Score
            composite, tech_c, fund_c, mkt_c, all_reasons = self.cs.compute_score(
                tech_score, fund_score, market_score, tech_reasons, fund_reasons
            )

            # 9. Target price
            target = self.cs.estimate_target_price(
                price_usd, indicators, patterns, self.min_profit_pct
            )
            upside_pct = (target / price_usd - 1) * 100

            # 10. Filter by minimum upside
            if upside_pct < self.min_profit_pct:
                return None

            # 11. Get company info
            profile = self.fetcher.get_company_profile_finnhub(symbol) or {}
            company_name = profile.get("name", symbol.split(".")[0])
            sector = profile.get("finnhubIndustry", "Unknown")

            # 12. Recent daily change
            prev_close = float(quote.get("pc", price_raw))
            daily_change_pct = (price_raw / prev_close - 1) * 100 if prev_close > 0 else 0.0

            return {
                "symbol": symbol,
                "name": company_name,
                "sector": sector,
                "price_usd": round(price_usd, 4),
                "price_raw": price_raw,
                "daily_change_pct": round(daily_change_pct, 2),
                "ai_score": round(composite, 1),
                "tech_score": round(tech_score, 1),
                "fund_score": round(fund_score, 1),
                "market_score": round(market_score, 1),
                "target_price": target,
                "upside_pct": round(upside_pct, 1),
                "reasons": all_reasons[:5],
                "patterns": [k for k, v in patterns.items() if v],
                "rsi": round(indicators.get("rsi", 50), 1),
                "macd_bullish": indicators.get("macd_bullish", False),
                "volume_ratio": round(indicators.get("volume_ratio", 1), 2),
                "momentum_20d": round(indicators.get("momentum_20d", 0), 2),
                "atr_pct": round(indicators.get("atr_pct", 1), 2),
                "df": df,  # Store OHLCV for charts
            }
        except Exception:
            return None

    def _market_context_score(self, indicators: Dict, symbol: str) -> float:
        """Simple market context scoring based on volatility and trend."""
        score = 50.0
        atr_pct = indicators.get("atr_pct", 2)
        if atr_pct < 1.5:
            score += 20
        elif atr_pct < 3:
            score += 10
        elif atr_pct > 5:
            score -= 10
        # Small currency-risk discount for Indian stocks
        if symbol.endswith(".NS") or symbol.endswith(".BO"):
            score -= 3
        return min(max(score, 0), 100)

    def run(
        self,
        tickers: Optional[List[str]] = None,
        top_n: int = 20,
        progress_callback=None,
    ) -> pd.DataFrame:
        """Run the full screening pipeline."""
        tickers = tickers or self._get_ticker_list()
        results = []
        total = len(tickers)
        for i, sym in enumerate(tickers):
            if progress_callback:
                progress_callback(i / total, sym)
            result = self.screen_single(sym)
            if result:
                results.append(result)
            # Early exit if we have enough high-score candidates
            if len(results) >= top_n * 3:
                break

        if not results:
            return pd.DataFrame()

        df_results = pd.DataFrame(results).sort_values("ai_score", ascending=False)
        return df_results.head(top_n)


# ─── PAPER TRADING SIMULATOR ─────────────────────────────────────────────────
class PaperTrader:
    """Autonomous paper trading engine with risk management."""

    def __init__(self, initial_capital: float, strategy_params: Dict):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.params = strategy_params
        self.positions: Dict[str, Dict] = {}
        self.closed_trades: List[Dict] = []
        self.equity_curve: List[Tuple[datetime, float]] = [(datetime.now(), initial_capital)]
        self.trade_log: List[str] = []

    @property
    def portfolio_value(self) -> float:
        pos_value = sum(
            p["qty"] * p["current_price"] for p in self.positions.values()
        )
        return self.cash + pos_value

    @property
    def unrealized_pnl(self) -> float:
        return sum(
            p["qty"] * (p["current_price"] - p["entry_price"])
            for p in self.positions.values()
        )

    @property
    def total_pnl(self) -> float:
        realized = sum(t["pnl"] for t in self.closed_trades)
        return realized + self.unrealized_pnl

    @property
    def win_rate(self) -> float:
        if not self.closed_trades:
            return 0.0
        wins = sum(1 for t in self.closed_trades if t["pnl"] > 0)
        return wins / len(self.closed_trades) * 100

    @property
    def max_drawdown(self) -> float:
        if len(self.equity_curve) < 2:
            return 0.0
        values = [v for _, v in self.equity_curve]
        peak = values[0]
        max_dd = 0.0
        for v in values[1:]:
            peak = max(peak, v)
            dd = (peak - v) / peak * 100
            max_dd = max(max_dd, dd)
        return max_dd

    def enter_position(
        self,
        symbol: str,
        current_price: float,
        target_price: float,
        ai_score: float,
    ) -> bool:
        """Enter a new long position with position sizing."""
        if symbol in self.positions:
            return False
        if len(self.positions) >= self.params.get("max_positions", 5):
            return False

        # Position sizing: risk 1–2% of portfolio per trade
        risk_pct = self.params.get("position_risk_pct", 0.02)
        stop_pct = self.params.get("stop_loss_pct", 0.10)
        position_value = self.portfolio_value * risk_pct / stop_pct
        position_value = min(position_value, self.cash * 0.25)  # max 25% of cash

        if position_value < current_price:
            return False

        # Slippage + commission
        slippage = current_price * self.params.get("slippage_pct", 0.002)
        commission = position_value * self.params.get("commission_pct", 0.001)
        fill_price = current_price + slippage
        qty = math.floor(position_value / fill_price)
        if qty <= 0:
            return False

        cost = qty * fill_price + commission
        if cost > self.cash:
            return False

        self.cash -= cost
        self.positions[symbol] = {
            "symbol": symbol,
            "qty": qty,
            "entry_price": fill_price,
            "current_price": fill_price,
            "target_price": target_price,
            "stop_price": fill_price * (1 - stop_pct),
            "ai_score": ai_score,
            "entry_time": datetime.now(),
            "commission_paid": commission,
        }
        self.trade_log.append(
            f"[{datetime.now().strftime('%H:%M:%S')}] BUY {qty} {symbol} @ ${fill_price:.2f}"
        )
        return True

    def update_and_check_exits(self, fetcher: DataFetcher):
        """Update prices and check stop-loss / take-profit conditions."""
        to_exit = []
        for sym, pos in self.positions.items():
            quote = fetcher.get_quote(sym)
            if quote:
                new_price = float(quote.get("c", pos["current_price"]))
                if new_price > 0:
                    pos["current_price"] = new_price

            cp = pos["current_price"]
            ep = pos["entry_price"]

            # Trailing stop: tighten stop as price rises
            trail_pct = self.params.get("trailing_stop_pct", 0.07)
            new_stop = cp * (1 - trail_pct)
            if new_stop > pos["stop_price"]:
                pos["stop_price"] = new_stop

            # Check exits
            if cp <= pos["stop_price"]:
                to_exit.append((sym, cp, "Stop Loss"))
            elif cp >= pos["target_price"]:
                to_exit.append((sym, cp, "Take Profit"))

        for sym, exit_price, reason in to_exit:
            self._close_position(sym, exit_price, reason)

    def _close_position(self, symbol: str, exit_price: float, reason: str):
        """Close a position and record the trade."""
        if symbol not in self.positions:
            return
        pos = self.positions.pop(symbol)
        slippage = exit_price * self.params.get("slippage_pct", 0.002)
        commission = pos["qty"] * exit_price * self.params.get("commission_pct", 0.001)
        fill_price = exit_price - slippage
        proceeds = pos["qty"] * fill_price - commission
        self.cash += proceeds

        pnl = proceeds - (pos["qty"] * pos["entry_price"] + pos["commission_paid"])
        pnl_pct = pnl / (pos["qty"] * pos["entry_price"]) * 100

        trade_record = {
            "symbol": symbol,
            "entry_price": pos["entry_price"],
            "exit_price": fill_price,
            "qty": pos["qty"],
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "reason": reason,
            "duration": (datetime.now() - pos["entry_time"]).total_seconds() / 3600,
            "ai_score": pos["ai_score"],
        }
        self.closed_trades.append(trade_record)
        emoji = "✅" if pnl > 0 else "❌"
        self.trade_log.append(
            f"[{datetime.now().strftime('%H:%M:%S')}] {emoji} SELL {pos['qty']} {symbol} "
            f"@ ${fill_price:.2f} | P&L: ${pnl:+.2f} ({pnl_pct:+.1f}%) | {reason}"
        )
        self.equity_curve.append((datetime.now(), self.portfolio_value))

    def auto_trade(self, screener_results: pd.DataFrame, fetcher: DataFetcher):
        """AI autonomously selects positions from screener output."""
        # First update existing positions
        self.update_and_check_exits(fetcher)

        # Then open new positions from top-scored candidates
        min_score = self.params.get("min_score_to_trade", 60)
        candidates = screener_results[
            screener_results["ai_score"] >= min_score
        ].head(self.params.get("max_positions", 5))

        for _, row in candidates.iterrows():
            if len(self.positions) >= self.params.get("max_positions", 5):
                break
            if row["symbol"] not in self.positions:
                self.enter_position(
                    row["symbol"],
                    row["price_usd"],
                    row["target_price"],
                    row["ai_score"],
                )

    def get_performance_summary(self) -> Dict:
        """Compile performance metrics."""
        if not self.closed_trades:
            return {}
        wins = [t for t in self.closed_trades if t["pnl"] > 0]
        losses = [t for t in self.closed_trades if t["pnl"] <= 0]
        avg_win = np.mean([t["pnl_pct"] for t in wins]) if wins else 0
        avg_loss = np.mean([t["pnl_pct"] for t in losses]) if losses else 0
        profit_factor = (
            abs(sum(t["pnl"] for t in wins)) / (abs(sum(t["pnl"] for t in losses)) + 1e-10)
        )
        return {
            "total_trades": len(self.closed_trades),
            "win_rate": self.win_rate,
            "avg_win_pct": avg_win,
            "avg_loss_pct": avg_loss,
            "profit_factor": profit_factor,
            "max_drawdown": self.max_drawdown,
            "total_return_pct": (self.portfolio_value / self.initial_capital - 1) * 100,
            "total_pnl": self.total_pnl,
        }

    def record_daily_snapshot(self):
        """Record end-of-day P&L snapshot into session state ledger."""
        today = datetime.now().date()
        last = st.session_state.get("last_ledger_date")
        if last == today:
            return  # already recorded today

        realized_today = sum(
            t["pnl"] for t in self.closed_trades
            if datetime.fromisoformat(str(t.get("exit_time", datetime.now()))).date() == today
            if "exit_time" in t
        )
        entry = {
            "date": today.strftime("%Y-%m-%d"),
            "portfolio_value": round(self.portfolio_value, 2),
            "daily_pnl": round(self.portfolio_value - (
                st.session_state.daily_pnl_ledger[-1]["portfolio_value"]
                if st.session_state.daily_pnl_ledger else self.initial_capital
            ), 2),
            "unrealized_pnl": round(self.unrealized_pnl, 2),
            "realized_pnl": round(sum(t["pnl"] for t in self.closed_trades), 2),
            "open_positions": len(self.positions),
            "total_trades": len(self.closed_trades),
            "win_rate": round(self.win_rate, 1),
            "cash": round(self.cash, 2),
        }
        st.session_state.daily_pnl_ledger.append(entry)
        st.session_state.last_ledger_date = today

    def get_todays_pnl(self) -> float:
        """Return today's P&L vs yesterday's portfolio value."""
        ledger = st.session_state.get("daily_pnl_ledger", [])
        prev_value = ledger[-1]["portfolio_value"] if ledger else self.initial_capital
        return self.portfolio_value - prev_value


        """Analyze closed trades and suggest strategy improvements."""
        suggestions = []
        if len(self.closed_trades) < 5:
            return ["Insufficient trade history for analysis."]

        summary = self.get_performance_summary()

        if summary["win_rate"] < 45:
            suggestions.append("📉 Win rate below 45% — consider increasing minimum AI Score threshold from 60 → 70.")
        if summary["avg_loss_pct"] < -12:
            suggestions.append("🔴 Average loss too large — tighten stop-loss from 10% → 7%.")
        if summary["profit_factor"] < 1.2:
            suggestions.append("⚠️ Low profit factor — filter out stocks with D/E ratio > 1.0.")
        if summary["max_drawdown"] > 15:
            suggestions.append("📉 High drawdown — reduce max simultaneous positions from 5 → 3.")

        # Score-performance correlation
        high_score_trades = [t for t in self.closed_trades if t.get("ai_score", 0) > 75]
        if high_score_trades:
            hs_win_rate = sum(1 for t in high_score_trades if t["pnl"] > 0) / len(high_score_trades)
            if hs_win_rate > 0.6:
                suggestions.append("✅ High-score stocks (>75) outperforming — increase tech_weight to 0.45.")
        
        low_score_trades = [t for t in self.closed_trades if t.get("ai_score", 0) < 65]
        if low_score_trades:
            ls_win_rate = sum(1 for t in low_score_trades if t["pnl"] > 0) / len(low_score_trades) if low_score_trades else 0
            if ls_win_rate < 0.4:
                suggestions.append("🔄 Low-score stocks (<65) underperforming — raise min_score_to_trade to 70.")

        if summary["win_rate"] > 60 and summary["profit_factor"] > 1.5:
            suggestions.append("🌟 Strategy performing well! Consider adding momentum filter (RSI 40–65 only).")

        if not suggestions:
            suggestions.append("✅ Strategy parameters look balanced. Continue monitoring.")
        return suggestions


# ─── USER MANAGEMENT & PERSISTENT STORAGE ────────────────────────────────────
def get_user_dir(username: str) -> str:
    """Get or create user-specific data directory."""
    user_dir = os.path.join(USER_DATA_DIR, username)
    os.makedirs(user_dir, exist_ok=True)
    return user_dir


def save_user_data(username: str):
    """Save all session state to user's JSON file."""
    if not username:
        return
    user_dir = get_user_dir(username)
    data = {
        "strategy_params_us": st.session_state.get("strategy_params_us", {}),
        "strategy_params_india": st.session_state.get("strategy_params_india", {}),
        "screener_results_us": st.session_state.get("screener_results_us", pd.DataFrame()).to_dict("records"),
        "screener_results_india": st.session_state.get("screener_results_india", pd.DataFrame()).to_dict("records"),
        "last_screen_time_us": str(st.session_state.get("last_screen_time_us")) if st.session_state.get("last_screen_time_us") else None,
        "last_screen_time_india": str(st.session_state.get("last_screen_time_india")) if st.session_state.get("last_screen_time_india") else None,
        "daily_pnl_ledger_us": st.session_state.get("daily_pnl_ledger_us", []),
        "daily_pnl_ledger_india": st.session_state.get("daily_pnl_ledger_india", []),
        "last_ledger_date_us": str(st.session_state.get("last_ledger_date_us")) if st.session_state.get("last_ledger_date_us") else None,
        "last_ledger_date_india": str(st.session_state.get("last_ledger_date_india")) if st.session_state.get("last_ledger_date_india") else None,
        "initial_capital_us": st.session_state.get("initial_capital_us", 50000.0),
        "initial_capital_india": st.session_state.get("initial_capital_india", 4200000.0),
        "min_price": st.session_state.get("min_price", 5.0),
        "max_price": st.session_state.get("max_price", 500.0),
        "min_profit_pct_us": st.session_state.get("min_profit_pct_us", 15.0),
        "min_profit_pct_india": st.session_state.get("min_profit_pct_india", 15.0),
        "dark_mode": st.session_state.get("dark_mode", True),
    }
    
    # Save trader state
    trader_us = st.session_state.get("trader_us")
    trader_india = st.session_state.get("trader_india")
    
    if trader_us:
        data["trader_us"] = {
            "cash": trader_us.cash,
            "initial_capital": trader_us.initial_capital,
            "positions": trader_us.positions,
            "closed_trades": trader_us.closed_trades,
            "equity_curve": [(str(ts), v) for ts, v in trader_us.equity_curve],
            "trade_log": trader_us.trade_log,
        }
    if trader_india:
        data["trader_india"] = {
            "cash": trader_india.cash,
            "initial_capital": trader_india.initial_capital,
            "positions": trader_india.positions,
            "closed_trades": trader_india.closed_trades,
            "equity_curve": [(str(ts), v) for ts, v in trader_india.equity_curve],
            "trade_log": trader_india.trade_log,
        }
    
    with open(os.path.join(user_dir, "session_data.json"), "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_user_data(username: str):
    """Load user's saved data into session state."""
    if not username:
        return
    user_dir = get_user_dir(username)
    filepath = os.path.join(user_dir, "session_data.json")
    
    if not os.path.exists(filepath):
        return
    
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
        
        st.session_state.strategy_params_us = data.get("strategy_params_us", load_default_strategy("US"))
        st.session_state.strategy_params_india = data.get("strategy_params_india", load_default_strategy("India"))
        
        # Load screener results
        results_us = data.get("screener_results_us", [])
        results_india = data.get("screener_results_india", [])
        st.session_state.screener_results_us = pd.DataFrame(results_us) if results_us else pd.DataFrame()
        st.session_state.screener_results_india = pd.DataFrame(results_india) if results_india else pd.DataFrame()
        
        # Load timestamps
        if data.get("last_screen_time_us"):
            try:
                st.session_state.last_screen_time_us = datetime.fromisoformat(data["last_screen_time_us"])
            except Exception:
                st.session_state.last_screen_time_us = None
        if data.get("last_screen_time_india"):
            try:
                st.session_state.last_screen_time_india = datetime.fromisoformat(data["last_screen_time_india"])
            except Exception:
                st.session_state.last_screen_time_india = None
        
        # Load ledgers
        st.session_state.daily_pnl_ledger_us = data.get("daily_pnl_ledger_us", [])
        st.session_state.daily_pnl_ledger_india = data.get("daily_pnl_ledger_india", [])
        
        # Load other preferences
        st.session_state.min_price = data.get("min_price", 5.0)
        st.session_state.max_price = data.get("max_price", 500.0)
        st.session_state.min_profit_pct_us = data.get("min_profit_pct_us", 15.0)
        st.session_state.min_profit_pct_india = data.get("min_profit_pct_india", 15.0)
        st.session_state.initial_capital_us = data.get("initial_capital_us", 50000.0)
        st.session_state.initial_capital_india = data.get("initial_capital_india", 4200000.0)
        st.session_state.dark_mode = data.get("dark_mode", True)
        
        # Restore traders
        if data.get("trader_us") and not st.session_state.get("trader_us"):
            td = data["trader_us"]
            trader = PaperTrader(td["initial_capital"], st.session_state.strategy_params_us)
            trader.cash = td["cash"]
            trader.positions = td.get("positions", {})
            trader.closed_trades = td.get("closed_trades", [])
            trader.trade_log = td.get("trade_log", [])
            eq = td.get("equity_curve", [])
            trader.equity_curve = [(datetime.fromisoformat(ts) if isinstance(ts, str) else ts, v) for ts, v in eq]
            st.session_state.trader_us = trader
            
        if data.get("trader_india") and not st.session_state.get("trader_india"):
            td = data["trader_india"]
            trader = PaperTrader(td["initial_capital"], st.session_state.strategy_params_india)
            trader.cash = td["cash"]
            trader.positions = td.get("positions", {})
            trader.closed_trades = td.get("closed_trades", [])
            trader.trade_log = td.get("trade_log", [])
            eq = td.get("equity_curve", [])
            trader.equity_curve = [(datetime.fromisoformat(ts) if isinstance(ts, str) else ts, v) for ts, v in eq]
            st.session_state.trader_india = trader
            
    except Exception as e:
        st.error(f"Error loading user data: {e}")


def load_default_strategy(market: str) -> Dict:
    """Return default strategy params for US or India market."""
    base = {
        "tech_weight": 0.40,
        "fund_weight": 0.40,
        "market_weight": 0.20,
        "rsi_weight": 1.0,
        "min_score_to_trade": 60,
        "stop_loss_pct": 0.10,
        "trailing_stop_pct": 0.07,
        "take_profit_pct": 0.20,
        "position_risk_pct": 0.02,
        "max_positions": 5,
        "slippage_pct": 0.002,
        "commission_pct": 0.001,
        "version": 1,
    }
    
    if market == "India":
        # India-specific adjustments
        base.update({
            "tech_weight": 0.35,
            "fund_weight": 0.45,
            "market_weight": 0.20,
            "stop_loss_pct": 0.12,  # Wider stops for higher volatility
            "slippage_pct": 0.003,  # Slightly higher slippage
            "min_score_to_trade": 65,  # More conservative
        })
    
    return base


def get_all_usernames() -> List[str]:
    """Get list of all registered usernames."""
    if not os.path.exists(USER_DATA_DIR):
        return []
    return [d for d in os.listdir(USER_DATA_DIR) if os.path.isdir(os.path.join(USER_DATA_DIR, d))]


# ─── STRATEGY PARAMETER MANAGER (DEPRECATED - now per-market) ────────────────
def load_strategy_params() -> Dict:
    """Load strategy params from file or return defaults."""
    defaults = {
        "tech_weight": 0.40,
        "fund_weight": 0.40,
        "market_weight": 0.20,
        "rsi_weight": 1.0,
        "min_score_to_trade": 60,
        "stop_loss_pct": 0.10,
        "trailing_stop_pct": 0.07,
        "take_profit_pct": 0.20,
        "position_risk_pct": 0.02,
        "max_positions": 5,
        "slippage_pct": 0.002,
        "commission_pct": 0.001,
        "version": 1,
    }
    if os.path.exists(STRATEGY_FILE):
        try:
            with open(STRATEGY_FILE, "r") as f:
                saved = json.load(f)
            defaults.update(saved)
        except Exception:
            pass
    return defaults


def save_strategy_params(params: Dict):
    """Save updated strategy params to file."""
    try:
        with open(STRATEGY_FILE, "w") as f:
            json.dump(params, f, indent=2)
    except Exception:
        pass


# ─── CHART HELPERS ───────────────────────────────────────────────────────────
def make_candlestick_chart(df: pd.DataFrame, symbol: str) -> go.Figure:
    """Create a candlestick + volume + RSI chart."""
    if df is None or len(df) < 2:
        return go.Figure()

    t = get_theme()
    close = df["close"]
    rsi = TechnicalAnalyzer.compute_rsi(close)
    macd, sig, hist = TechnicalAnalyzer.compute_macd(close)
    sma20 = TechnicalAnalyzer.compute_sma(close, min(20, len(close) - 1))
    sma50 = TechnicalAnalyzer.compute_sma(close, min(50, len(close) - 1))

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.55, 0.22, 0.23],
        subplot_titles=[f"{symbol} — Price", "Volume", "RSI"],
    )

    fig.add_trace(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"],
        low=df["low"], close=df["close"], name="Price",
        increasing_line_color=t["up_color"],
        decreasing_line_color=t["down_color"],
    ), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=sma20, name="SMA20",
        line=dict(color="#ff9800", width=1.5, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=sma50, name="SMA50",
        line=dict(color=t["accent"], width=1.5, dash="dot")), row=1, col=1)

    vol_colors = [t["up_color"] if c >= o else t["down_color"]
                  for c, o in zip(df["close"], df["open"])]
    fig.add_trace(go.Bar(x=df.index, y=df["volume"], name="Volume",
        marker_color=vol_colors, showlegend=False), row=2, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=rsi, name="RSI",
        line=dict(color="#e040fb", width=2)), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="#ff4444", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#44bb44", row=3, col=1)

    fig.update_layout(
        template=t["plotly_template"],
        paper_bgcolor=t["chart_bg"],
        plot_bgcolor=t["chart_bg"],
        height=520,
        margin=dict(l=0, r=0, t=30, b=0),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0,
                    font=dict(color=t["text"])),
        xaxis_rangeslider_visible=False,
        font=dict(color=t["text"]),
    )
    for i in range(1, 4):
        fig.update_xaxes(gridcolor=t["border"], row=i, col=1)
        fig.update_yaxes(gridcolor=t["border"], row=i, col=1)
    return fig


def make_equity_curve(equity_curve: List) -> go.Figure:
    """Create equity curve chart."""
    if len(equity_curve) < 2:
        return go.Figure()
    t = get_theme()
    timestamps = [ts for ts, _ in equity_curve]
    values = [v for _, v in equity_curve]
    baseline = values[0]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timestamps, y=values,
        fill="tozeroy",
        line=dict(color=t["accent2"], width=2),
        fillcolor=t["accent2"] + "22",
        name="Portfolio Value",
    ))
    fig.add_hline(y=baseline, line_dash="dash",
                  line_color=t["text_muted"],
                  annotation_text="Initial Capital",
                  annotation_font_color=t["text_muted"])
    fig.update_layout(
        **get_plotly_layout(t, height=350, title="Portfolio Equity Curve"),
        yaxis_title="Portfolio Value (USD)",
    )
    return fig


def make_score_breakdown_chart(results_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of top stocks by AI score."""
    if results_df.empty:
        return go.Figure()
    t = get_theme()
    top = results_df.head(15).sort_values("ai_score")
    fig = go.Figure()
    fig.add_trace(go.Bar(y=top["symbol"], x=top["tech_score"] * 0.4,
        name="Technical (40%)", orientation="h", marker_color="#2196f3"))
    fig.add_trace(go.Bar(y=top["symbol"], x=top["fund_score"] * 0.4,
        name="Fundamental (40%)", orientation="h", marker_color=t["accent2"]))
    fig.add_trace(go.Bar(y=top["symbol"], x=top["market_score"] * 0.2,
        name="Market (20%)", orientation="h", marker_color="#ff9800"))
    fig.update_layout(
        barmode="stack",
        **get_plotly_layout(t, height=420, title="AI Score Breakdown"),
        xaxis_title="Score Contribution",
        legend=dict(orientation="h", font=dict(color=t["text"])),
    )
    return fig


# ─── STREAMLIT APP ────────────────────────────────────────────────────────────
def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        # User auth
        "logged_in": False,
        "username": "",
        
        # API keys
        "api_keys_set": False,
        "finnhub_key_enc": b"",
        "av_key_enc": b"",
        "td_key_enc": b"",
        
        # Separate strategies for US and India
        "strategy_params_us": load_default_strategy("US"),
        "strategy_params_india": load_default_strategy("India"),
        
        # Separate screener results
        "screener_results_us": pd.DataFrame(),
        "screener_results_india": pd.DataFrame(),
        "last_screen_time_us": None,
        "last_screen_time_india": None,
        
        # Separate traders
        "trader_us": None,
        "trader_india": None,
        "trading_active_us": False,
        "trading_active_india": False,
        
        # Separate capital and params
        "initial_capital_us": 50000.0,
        "initial_capital_india": 4200000.0,  # ~$50k in INR
        "min_profit_pct_us": 15.0,
        "min_profit_pct_india": 15.0,
        
        # Shared params
        "min_price": 5.0,
        "max_price": 500.0,
        "markets": ["US", "India"],
        "screen_ticker_subset": "Top 100",
        "dark_mode": True,
        "selected_stock": None,
        "improvements_us": [],
        "improvements_india": [],
        "auto_refresh": False,
        
        # Separate ledgers
        "daily_pnl_ledger_us": [],
        "daily_pnl_ledger_india": [],
        "last_ledger_date_us": None,
        "last_ledger_date_india": None,
        
        # Active market tab
        "active_market": "US",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def get_fetcher() -> DataFetcher:
    """Instantiate DataFetcher from session state keys."""
    fh = decrypt_api_key(st.session_state.get("finnhub_key_enc", b""))
    av = decrypt_api_key(st.session_state.get("av_key_enc", b""))
    td = decrypt_api_key(st.session_state.get("td_key_enc", b""))
    return DataFetcher(fh, av, td)


def render_login_screen():
    """Render simple username login screen."""
    t = get_theme()
    
    st.markdown(f"""
    <div style="text-align:center;padding-top:80px;">
        <div style="font-size:3rem;font-family:'JetBrains Mono',monospace;font-weight:800;
                    background:linear-gradient(135deg,{t['accent']},{t['accent2']});
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                    margin-bottom:12px;">
            📈 AI Stock Screener
        </div>
        <p style="color:{t['text_muted']};font-size:1rem;margin-bottom:40px;">
            Multi-user trading simulator with persistent data storage
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"""
        <div style="background:{t['surface']};border:1px solid {t['border']};
                    border-radius:16px;padding:32px 40px;box-shadow:0 4px 20px {t['accent_glow']};">
            <h3 style="text-align:center;color:{t['text']};margin-bottom:24px;">
                👤 Enter Your Username
            </h3>
        </div>
        """, unsafe_allow_html=True)
        
        username = st.text_input(
            "",
            placeholder="Enter username (e.g., john_doe)",
            max_chars=30,
            label_visibility="collapsed",
            key="login_username_input",
        )
        
        st.caption("No password needed — just a simple name to keep your data separate from others.")
        
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("🆕 Create New User", use_container_width=True):
                if username and len(username) >= 3:
                    if username in get_all_usernames():
                        st.error(f"❌ Username '{username}' already exists! Choose another or login.")
                    else:
                        st.session_state.username = username
                        st.session_state.logged_in = True
                        load_user_data(username)  # Will be empty for new user
                        st.success(f"✅ Welcome, {username}!")
                        time.sleep(0.5)
                        st.rerun()
                else:
                    st.warning("Username must be at least 3 characters.")
        
        with col_b:
            if st.button("🔓 Login Existing", use_container_width=True, type="primary"):
                if username:
                    if username not in get_all_usernames():
                        st.error(f"❌ User '{username}' not found! Create a new account first.")
                    else:
                        st.session_state.username = username
                        st.session_state.logged_in = True
                        load_user_data(username)
                        st.success(f"✅ Welcome back, {username}!")
                        time.sleep(0.5)
                        st.rerun()
                else:
                    st.warning("Please enter your username.")
        
        # Show existing users
        existing = get_all_usernames()
        if existing:
            st.markdown("---")
            st.caption(f"**Registered users ({len(existing)})**: {', '.join(existing[:10])}" + 
                      (f" and {len(existing)-10} more..." if len(existing) > 10 else ""))


def render_api_key_setup():
    """Render API key input section in sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔑 API Keys")

    if st.session_state.api_keys_set:
        fh = decrypt_api_key(st.session_state.finnhub_key_enc)
        st.sidebar.success(f"Finnhub: {mask_key(fh)}")
        if st.sidebar.button("🔄 Reset API Keys"):
            st.session_state.api_keys_set = False
            st.session_state.finnhub_key_enc = b""
            st.rerun()
        return

    with st.sidebar.form("api_key_form"):
        st.caption("Keys are encrypted in session memory. Never logged or stored.")
        fh_key = st.text_input(
            "Finnhub API Key *",
            type="password",
            placeholder="Enter Finnhub key...",
            help="Get free key at finnhub.io",
        )
        av_key = st.text_input(
            "Alpha Vantage Key (optional)",
            type="password",
            placeholder="Fallback data source",
        )
        submitted = st.form_submit_button("🔒 Secure & Save Keys", type="primary")
        if submitted:
            if fh_key:
                st.session_state.finnhub_key_enc = encrypt_api_key(fh_key)
                st.session_state.av_key_enc = encrypt_api_key(av_key) if av_key else b""
                st.session_state.api_keys_set = True
                st.success("✅ Keys encrypted and saved!")
                st.rerun()
            else:
                st.warning("Finnhub key is required. You can still use demo mode.")
                st.session_state.api_keys_set = True
                st.rerun()

    st.sidebar.info("💡 **Demo Mode**: Without API keys, the screener uses synthetic data to demonstrate functionality.")


def render_screening_controls():
    """Render screener control panel in sidebar."""
    t = get_theme()
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"<p style='font-weight:700;font-size:0.85rem;color:{t['text_muted']};text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px'>⚙️ Screening Parameters</p>", unsafe_allow_html=True)

    # ── Market toggle switches ──────────────────────────────────────────────
    st.sidebar.markdown(f"<p style='font-size:0.82rem;font-weight:600;color:{t['text']}'>Markets</p>", unsafe_allow_html=True)
    col_us, col_in = st.sidebar.columns(2)
    with col_us:
        us_on = st.toggle("🇺🇸 US", value=("US" in st.session_state.markets), key="mkt_us")
    with col_in:
        in_on = st.toggle("🇮🇳 India", value=("India" in st.session_state.markets), key="mkt_in")

    new_markets = []
    if us_on:
        new_markets.append("US")
    if in_on:
        new_markets.append("India")
    if not new_markets:          # prevent empty selection
        new_markets = ["US"]
    st.session_state.markets = new_markets

    st.sidebar.markdown("")

    # ── Price range ─────────────────────────────────────────────────────────
    st.session_state.min_price, st.session_state.max_price = st.sidebar.slider(
        "Price Range (USD)",
        min_value=0.5, max_value=5000.0,
        value=(st.session_state.min_price, st.session_state.max_price),
        step=0.5,
        help="Indian stock prices auto-converted from INR at ~83.5 rate",
    )

    # ── Profit target ────────────────────────────────────────────────────────
    st.session_state.min_profit_pct = st.sidebar.slider(
        "Min Profit Target (%)",
        min_value=5, max_value=150,
        value=int(st.session_state.min_profit_pct),
        step=5,
        help="Minimum projected upside % for a stock to appear in results",
    )

    # ── Ticker subset ────────────────────────────────────────────────────────
    st.session_state.screen_ticker_subset = st.sidebar.selectbox(
        "Scan Depth",
        ["Top 50", "Top 100", "Top 200", "All (~330)"],
        index=1,
    )


def get_ticker_subset(subset_label: str) -> List[str]:
    tickers = []
    if "US" in st.session_state.get("markets", ["US"]):
        tickers += US_TICKERS
    if "India" in st.session_state.get("markets", ["US"]):
        tickers += INDIA_TICKERS
    n_map = {"Top 50": 50, "Top 100": 100, "Top 200": 200, "All (~330)": len(tickers)}
    n = n_map.get(subset_label, 100)
    return tickers[:n]


def render_screener_tab():
    """Render the stock screener tab with separate US/India sub-tabs."""
    st.header("🔍 AI Stock Screener")
    
    # Sub-tabs for US and India
    tab_us, tab_india = st.tabs(["🇺🇸 US Markets", "🇮🇳 Indian Markets"])
    
    with tab_us:
        render_market_screener("US")
    
    with tab_india:
        render_market_screener("India")


def render_market_screener(market: str):
    """Render screener for a specific market."""
    is_us = (market == "US")
    flag = "🇺🇸" if is_us else "🇮🇳"
    exchange = "NYSE / NASDAQ" if is_us else "NSE"
    cur = "$" if is_us else "₹"
    
    results_key = f"screener_results_{market.lower()}"
    last_time_key = f"last_screen_time_{market.lower()}"
    min_profit_key = f"min_profit_pct_{market.lower()}"
    
    st.caption(
        f"{flag} Scanning {exchange} pure stocks • "
        f"No ETFs • No Crypto • "
        f"Price: ${st.session_state.min_price:.0f}–${st.session_state.max_price:.0f} USD • "
        f"Min upside: {st.session_state.get(min_profit_key, 15):.0f}%"
    )
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        run_screen = st.button(f"🚀 Run {market} Screener", type="primary", 
                               use_container_width=True, key=f"run_{market}")
    with col2:
        auto_refresh = st.toggle("⏱ Auto (5min)", value=st.session_state.auto_refresh, 
                                key=f"auto_{market}")
        st.session_state.auto_refresh = auto_refresh
    with col3:
        last_time = st.session_state.get(last_time_key)
        if last_time:
            st.caption(f"Last: {last_time.strftime('%H:%M:%S')}")
    
    # Run screening
    if run_screen or (auto_refresh and last_time and (datetime.now() - last_time).seconds > 300):
        fetcher = get_fetcher()
        tickers = US_TICKERS if is_us else INDIA_TICKERS
        tickers = tickers[:st.session_state.get("screen_ticker_subset_limit", 100)]
        
        strategy_key = f"strategy_params_{market.lower()}"
        screener = StockScreener(
            fetcher=fetcher,
            strategy_params=st.session_state.get(strategy_key, load_default_strategy(market)),
            min_price_usd=st.session_state.min_price,
            max_price_usd=st.session_state.max_price,
            min_profit_pct=st.session_state.get(min_profit_key, 15),
            markets=[market],
        )
        
        progress_bar = st.progress(0.0)
        status_text = st.empty()
        
        def progress_callback(pct, sym):
            progress_bar.progress(min(pct, 1.0))
            status_text.caption(f"⏳ Scanning {sym}...")
        
        with st.spinner(f"🔍 Running {market} screening..."):
            results = screener.run(tickers=tickers, top_n=20, progress_callback=progress_callback)
        
        progress_bar.empty()
        status_text.empty()
        st.session_state[results_key] = results
        st.session_state[last_time_key] = datetime.now()
        
        # Auto-save
        if st.session_state.get("logged_in"):
            save_user_data(st.session_state.username)
    
    # Display results
    results = st.session_state.get(results_key, pd.DataFrame())
    if results.empty:
        st.info(f"👆 Click **Run {market} Screener** to scan. Without API keys, demo data will be used.")
        return
    
    st.subheader(f"📊 Top {len(results)} {market} Opportunities")
    display_cols = ["symbol", "name", "price_usd", "daily_change_pct",
                    "ai_score", "tech_score", "fund_score",
                    "upside_pct", "target_price", "rsi", "patterns"]
    
    display_df = results[display_cols].copy()
    display_df.columns = ["Ticker", "Company", f"Price ({cur})", "Day Chg %",
                          "AI Score", "Tech", "Fund",
                          "Upside %", f"Target ({cur})", "RSI", "Patterns"]
    display_df["Patterns"] = display_df["Patterns"].apply(lambda x: ", ".join(x[:2]) if x else "–")
    
    def color_score(val):
        if val >= 70: return "background-color: #1b5e20; color: #a5d6a7"
        if val >= 55: return "background-color: #1a237e; color: #90caf9"
        if val >= 40: return "background-color: #f57f17; color: #fff9c4"
        return ""
    
    def color_upside(val):
        if val >= 30: return "color: #69f0ae; font-weight: bold"
        if val >= 15: return "color: #40c4ff"
        return "color: #ef9a9a"
    
    styled = (
        display_df.style
        .applymap(color_score, subset=["AI Score", "Tech", "Fund"])
        .applymap(color_upside, subset=["Upside %"])
        .format({
            f"Price ({cur})": f"{cur}{{:.2f}}",
            "Day Chg %": "{:+.1f}%",
            "Upside %": "{:.1f}%",
            f"Target ({cur})": f"{cur}{{:.2f}}",
            "RSI": "{:.0f}",
        })
    )
    st.dataframe(styled, use_container_width=True, height=500)
    
    # Charts
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(make_score_breakdown_chart(results), use_container_width=True)
    with col2:
        if "sector" in results.columns:
            sector_counts = results["sector"].value_counts().head(8)
            t = get_theme()
            fig_sector = px.pie(
                values=sector_counts.values,
                names=sector_counts.index,
                title=f"{market} Stocks by Sector",
                template=t["plotly_template"],
                color_discrete_sequence=px.colors.qualitative.Dark24,
            )
            fig_sector.update_layout(
                paper_bgcolor=t["chart_bg"],
                height=380,
                margin=dict(l=0, r=0, t=40, b=0),
                font=dict(color=t["text"]),
            )
            st.plotly_chart(fig_sector, use_container_width=True)
    
    # Stock detail
    st.subheader("📈 Stock Detail")
    ticker_options = results["symbol"].tolist()
    selected = st.selectbox(f"Select a {market} stock:", ticker_options, key=f"stock_{market}")
    if selected:
        row = results[results["symbol"] == selected].iloc[0]
        df_chart = row.get("df")
        
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("AI Score", f"{row['ai_score']:.0f}/100")
        m2.metric("Price", f"{cur}{row['price_usd']:.2f}", delta=f"{row['daily_change_pct']:+.1f}%")
        m3.metric("Target", f"{cur}{row['target_price']:.2f}")
        m4.metric("Upside", f"{row['upside_pct']:.1f}%")
        m5.metric("RSI", f"{row['rsi']:.0f}")
        
        if isinstance(df_chart, pd.DataFrame) and len(df_chart) > 5:
            st.plotly_chart(make_candlestick_chart(df_chart, selected), use_container_width=True)
        
        st.markdown("**🧠 AI Reasoning:**")
        reasons = row.get("reasons", [])
        for r in reasons:
            st.markdown(f"  • {r}")
        patterns = row.get("patterns", [])
        if patterns:
            st.markdown(f"**📐 Patterns detected:** {', '.join(patterns)}")


def render_simulator_tab():
    """Render simulator with separate US/India tabs."""
    st.header("💹 Paper Trading Simulator")
    tab_us, tab_india = st.tabs(["🇺🇸 US Markets", "🇮🇳 Indian Markets"])
    
    with tab_us:
        render_market_simulator("US")
    with tab_india:
        render_market_simulator("India")


def render_market_simulator(market: str):
    """Render simulator for specific market with persistent ledger."""
    tc = get_theme()
    active = st.session_state.get("active_market", "US")
    flag   = "🇺🇸" if active == "US" else "🇮🇳"
    cur    = "$" if active == "US" else "₹"

    st.header(f"💹 Paper Trading Simulator — {flag} {active}")
    st.caption("AI autonomously manages a virtual portfolio. Full ledger tracks every transaction.")

    # ── Capital setup ──────────────────────────────────────────────────────────
    if st.session_state.trader is None:
        col1, col2 = st.columns(2)
        with col1:
            capital = st.number_input(
                f"Initial Virtual Capital ({cur})",
                min_value=1000.0, max_value=10_000_000.0,
                value=st.session_state.initial_capital,
                step=1000.0, format="%.0f",
            )
            st.session_state.initial_capital = capital
        with col2:
            st.info("💡 Capital is locked once trading starts. Demo mode uses synthetic data without API key.")

    # ── Controls ───────────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if not st.session_state.trading_active:
            if st.button("▶ START Trading", type="primary", use_container_width=True):
                if st.session_state.screener_results.empty:
                    st.error("❌ Run the Screener tab first!")
                else:
                    st.session_state.trader = PaperTrader(
                        initial_capital=st.session_state.initial_capital,
                        strategy_params=st.session_state.strategy_params,
                    )
                    st.session_state.trading_active = True
                    st.success("✅ Paper trading started!")
                    st.rerun()
        else:
            if st.button("⏹ STOP Trading", type="secondary", use_container_width=True):
                st.session_state.trading_active = False
                if st.session_state.trader:
                    st.session_state.improvements = st.session_state.trader.suggest_improvements()
                st.info("Trading stopped. View Performance tab for analysis.")
                st.rerun()
    with col2:
        if st.session_state.trading_active and st.button("🔄 Run Cycle", use_container_width=True):
            trader = st.session_state.trader
            fetcher = get_fetcher()
            with st.spinner("Executing trading cycle..."):
                trader.auto_trade(st.session_state.screener_results, fetcher)
            st.session_state.trader = trader
            st.success("Cycle complete!")
            st.rerun()

    trader = st.session_state.trader
    if trader is None:
        st.info("📌 Run the Screener first, then press START to begin paper trading.")
        return

    # ── Portfolio summary ──────────────────────────────────────────────────────
    pv        = trader.portfolio_value
    initial   = trader.initial_capital
    total_ret = (pv / initial - 1) * 100
    realized  = sum(t_["pnl"] for t_ in trader.closed_trades)
    st.markdown("---")
    m1,m2,m3,m4,m5,m6 = st.columns(6)
    m1.metric("Portfolio Value",  f"{cur}{pv:,.0f}",                delta=f"{total_ret:+.1f}%")
    m2.metric("Cash Available",   f"{cur}{trader.cash:,.0f}")
    m3.metric("Unrealized P&L",   f"{cur}{trader.unrealized_pnl:+,.0f}")
    m4.metric("Realized P&L",     f"{cur}{realized:+,.0f}")
    m5.metric("Win Rate",         f"{trader.win_rate:.0f}%")
    m6.metric("Max Drawdown",     f"{trader.max_drawdown:.1f}%")

    # ── Open positions ─────────────────────────────────────────────────────────
    st.markdown("---")
    if trader.positions:
        st.subheader(f"📂 Open Positions ({len(trader.positions)})")
        pos_rows = []
        for sym, pos in trader.positions.items():
            pnl     = pos["qty"] * (pos["current_price"] - pos["entry_price"])
            pnl_pct = (pos["current_price"] / pos["entry_price"] - 1) * 100
            pos_rows.append({
                "Symbol":        sym,
                "Qty":           pos["qty"],
                "Entry Price":   f"{cur}{pos['entry_price']:.2f}",
                "Current Price": f"{cur}{pos['current_price']:.2f}",
                "Market Value":  f"{cur}{pos['qty']*pos['current_price']:,.0f}",
                "Cost Basis":    f"{cur}{pos['qty']*pos['entry_price']:,.0f}",
                "Target":        f"{cur}{pos['target_price']:.2f}",
                "Stop Loss":     f"{cur}{pos['stop_price']:.2f}",
                "P&L":           f"{cur}{pnl:+,.0f}",
                "P&L %":         f"{pnl_pct:+.1f}%",
                "AI Score":      f"{pos['ai_score']:.0f}",
                "Since":         pos["entry_time"].strftime("%H:%M:%S"),
            })
        def color_pos(val):
            t2 = get_theme()
            if isinstance(val, str) and "+" in val and val not in ("+0",):
                return f"color:{t2['positive']};font-weight:600"
            elif isinstance(val, str) and val.startswith(("-", cur+"-")):
                return f"color:{t2['negative']}"
            return ""
        styled_pos = pd.DataFrame(pos_rows).style.applymap(color_pos, subset=["P&L","P&L %"])
        st.dataframe(styled_pos, use_container_width=True, height=230)
    else:
        st.info("No open positions currently.")

    # ── Equity curve ───────────────────────────────────────────────────────────
    if len(trader.equity_curve) > 1:
        st.plotly_chart(make_equity_curve(trader.equity_curve), use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # ── FULL TRADING LEDGER ────────────────────────────────────────────────────
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("📒 Full Trading Ledger")
    st.caption("Complete record of every BUY / SELL transaction with cost, commission, and net P&L.")

    ledger_rows = []
    for i, tr in enumerate(trader.closed_trades):
        tid = f"T{i+1:04d}"
        comm = abs(tr.get("commission_paid", tr["qty"] * tr["entry_price"] * 0.001))
        ledger_rows.append({
            "Trade ID":    tid, "Side": "BUY",
            "Symbol":      tr["symbol"], "Qty": tr["qty"],
            "Price":       f"{cur}{tr['entry_price']:.2f}",
            "Gross Amt":   f"-{cur}{tr['qty']*tr['entry_price']:,.2f}",
            "Commission":  f"-{cur}{comm:.2f}",
            "Net P&L":     "—",
            "Exit Reason": "—",
            "Status":      "CLOSED",
        })
        ledger_rows.append({
            "Trade ID":    tid, "Side": "SELL",
            "Symbol":      tr["symbol"], "Qty": tr["qty"],
            "Price":       f"{cur}{tr['exit_price']:.2f}",
            "Gross Amt":   f"+{cur}{tr['qty']*tr['exit_price']:,.2f}",
            "Commission":  f"-{cur}{comm:.2f}",
            "Net P&L":     f"{cur}{tr['pnl']:+,.2f}",
            "Exit Reason": tr["reason"],
            "Status":      "CLOSED",
        })

    for sym, pos in trader.positions.items():
        unreal = pos["qty"] * (pos["current_price"] - pos["entry_price"])
        comm   = abs(pos.get("commission_paid", pos["qty"] * pos["entry_price"] * 0.001))
        ledger_rows.append({
            "Trade ID":    "OPEN", "Side": "BUY (HELD)",
            "Symbol":      sym, "Qty": pos["qty"],
            "Price":       f"{cur}{pos['entry_price']:.2f}",
            "Gross Amt":   f"-{cur}{pos['qty']*pos['entry_price']:,.2f}",
            "Commission":  f"-{cur}{comm:.2f}",
            "Net P&L":     f"{cur}{unreal:+,.2f}  (unrealized)",
            "Exit Reason": "Still open",
            "Status":      "OPEN",
        })

    if ledger_rows:
        ldf = pd.DataFrame(ledger_rows)

        def style_ledger(row):
            t2 = get_theme()
            if row["Status"] == "OPEN":
                bg = t2["accent"] + "18"
            elif row["Side"] == "SELL" and row["Net P&L"].startswith(cur+"+"):
                bg = t2["positive"] + "18"
            elif row["Side"] == "SELL" and "-" in row["Net P&L"]:
                bg = t2["negative"] + "15"
            else:
                bg = ""
            return [f"background-color:{bg}" if bg else "" for _ in row]

        def color_cell(val):
            t2 = get_theme()
            s = str(val)
            if s.startswith(cur+"+") or s.startswith("+"):
                return f"color:{t2['positive']};font-weight:600"
            if "-" in s and s not in ("—","Still open","CLOSED","OPEN"):
                return f"color:{t2['negative']}"
            if s in ("BUY","BUY (HELD)"):
                return f"color:{t2['accent']};font-weight:600"
            if s == "SELL":
                return f"color:{t2['accent2']};font-weight:600"
            return ""

        styled_ledger = (
            ldf.style
            .apply(style_ledger, axis=1)
            .applymap(color_cell, subset=["Side","Net P&L","Gross Amt"])
        )
        st.dataframe(styled_ledger, use_container_width=True, height=420)

        # Ledger totals row
        st.markdown("##### 📊 Ledger Summary")
        total_commission = sum(
            abs(tr.get("commission_paid", tr["qty"]*tr["entry_price"]*0.001))
            for tr in trader.closed_trades
        ) * 2  # buy + sell
        open_commission  = sum(
            abs(p.get("commission_paid", p["qty"]*p["entry_price"]*0.001))
            for p in trader.positions.values()
        )
        total_realized   = sum(tr["pnl"] for tr in trader.closed_trades)
        total_unrealized = sum(
            p["qty"]*(p["current_price"]-p["entry_price"])
            for p in trader.positions.values()
        )
        total_invested   = sum(
            tr["qty"]*tr["entry_price"] for tr in trader.closed_trades
        ) + sum(
            p["qty"]*p["entry_price"] for p in trader.positions.values()
        )

        sc1,sc2,sc3,sc4,sc5 = st.columns(5)
        sc1.metric("Total Trades",       len(trader.closed_trades))
        sc2.metric("Total Invested",     f"{cur}{total_invested:,.0f}")
        sc3.metric("Total Commission",   f"{cur}{total_commission+open_commission:,.2f}")
        sc4.metric("Realized P&L",       f"{cur}{total_realized:+,.2f}")
        sc5.metric("Unrealized P&L",     f"{cur}{total_unrealized:+,.2f}")
    else:
        st.info("📌 No ledger entries yet — run a trading cycle first.")

    # ── Raw trade log ──────────────────────────────────────────────────────────
    st.markdown("---")
    with st.expander("📜 Raw Trade Log (last 30 entries)", expanded=False):
        if trader.trade_log:
            st.code("\n".join(reversed(trader.trade_log[-30:])), language=None)
        else:
            st.caption("No log entries yet.")

    # ── Daily P&L Ledger ───────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📅 Daily P&L Ledger")
    st.caption("Snapshot of portfolio value, daily gain/loss, and trade stats — recorded each session day.")

    # Record today's snapshot
    if trader:
        try:
            trader.record_daily_snapshot()
        except Exception:
            pass

    ledger = st.session_state.get("daily_pnl_ledger", [])

    if ledger:
        t2 = get_theme()
        dl_df = pd.DataFrame(ledger)
        dl_df.columns = ["Date", "Portfolio Value", "Daily P&L", "Unrealized P&L",
                         "Realized P&L", "Open Pos.", "Total Trades", "Win Rate %", "Cash"]

        # Style daily P&L column
        def color_daily(val):
            try:
                v = float(str(val).replace(",",""))
                if v > 0:   return f"color:{t2['positive']};font-weight:700"
                if v < 0:   return f"color:{t2['negative']};font-weight:700"
            except Exception:
                pass
            return f"color:{t2['text_muted']}"

        styled_daily = (
            dl_df.style
            .applymap(color_daily, subset=["Daily P&L", "Unrealized P&L", "Realized P&L"])
            .format({
                "Portfolio Value":  "${:,.2f}",
                "Daily P&L":       "${:+,.2f}",
                "Unrealized P&L":  "${:+,.2f}",
                "Realized P&L":    "${:+,.2f}",
                "Cash":            "${:,.2f}",
                "Win Rate %":      "{:.1f}%",
            })
        )
        st.dataframe(styled_daily, use_container_width=True, height=280)

        # Daily P&L bar chart
        dates   = [e["date"] for e in ledger]
        dpnls   = [e["daily_pnl"] for e in ledger]
        bar_colors = [t2["positive"] if v >= 0 else t2["negative"] for v in dpnls]

        fig_daily = go.Figure(go.Bar(
            x=dates, y=dpnls,
            marker_color=bar_colors,
            text=[f"${v:+,.0f}" for v in dpnls],
            textposition="outside",
            textfont=dict(size=11, color=t2["text"]),
        ))
        fig_daily.update_layout(
            **get_plotly_layout(t2, height=280, title="Daily P&L"),
            yaxis_title="Daily Gain / Loss (USD)",
            showlegend=False,
        )
        st.plotly_chart(fig_daily, use_container_width=True)

        # Cumulative P&L line
        cum_pnl = [sum(dpnls[:i+1]) for i in range(len(dpnls))]
        fig_cum = go.Figure(go.Scatter(
            x=dates, y=cum_pnl,
            fill="tozeroy",
            line=dict(color=t2["accent2"] if cum_pnl[-1] >= 0 else t2["negative"], width=2.5),
            fillcolor=(t2["accent2"] if cum_pnl[-1] >= 0 else t2["negative"]) + "22",
            mode="lines+markers",
            marker=dict(size=6, color=t2["accent2"]),
            name="Cumulative P&L",
        ))
        fig_cum.add_hline(y=0, line_dash="dash", line_color=t2["text_muted"])
        fig_cum.update_layout(
            **get_plotly_layout(t2, height=260, title="Cumulative P&L"),
            yaxis_title="Cumulative Gain / Loss (USD)",
        )
        st.plotly_chart(fig_cum, use_container_width=True)

        # Clear ledger button
        if st.button("🗑️ Clear Daily Ledger", type="secondary"):
            st.session_state.daily_pnl_ledger = []
            st.session_state.last_ledger_date = None
            st.rerun()
    else:
        st.info("📌 Daily snapshots will appear here. Run a trading cycle to record your first entry.")
        st.markdown(f"""
        <div style='background:{get_theme()['surface']};border:1px solid {get_theme()['border']};
        border-radius:10px;padding:16px 20px;margin-top:8px;'>
        <span class='section-label'>What the ledger tracks</span>
        <ul style='color:{get_theme()['text_secondary']};font-size:0.875rem;line-height:1.9;margin:0;padding-left:18px;'>
            <li>📅 <b>Date</b> — each calendar day</li>
            <li>💼 <b>Portfolio Value</b> — total cash + open positions</li>
            <li>📈 <b>Daily P&L</b> — gain or loss vs. previous day</li>
            <li>🔒 <b>Realized P&L</b> — booked profit/loss from closed trades</li>
            <li>⏳ <b>Unrealized P&L</b> — floating P&L on open positions</li>
            <li>🎯 <b>Win Rate</b> — % of profitable trades to date</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)


def render_settings_tab():
    """Render settings with separate US/India strategy params."""
    st.header("⚙️ Strategy Settings")
    st.caption("Separate strategies for US and Indian markets.")
    
    tab_us, tab_india = st.tabs(["🇺🇸 US Strategy", "🇮🇳 India Strategy"])
    
    with tab_us:
        render_strategy_settings("US")
    with tab_india:
        render_strategy_settings("India")


def render_strategy_settings(market: str):
    """Render strategy settings for a specific market."""
    st.subheader(f"{market} Market Strategy")
    st.caption("Tune the AI scoring weights and trading parameters.")
    params = st.session_state.strategy_params

    with st.form("strategy_form"):
        st.subheader("📊 Scoring Weights")
        col1, col2, col3 = st.columns(3)
        with col1:
            tech_w = st.slider("Technical Weight", 0.1, 0.8, float(params["tech_weight"]), 0.05)
        with col2:
            fund_w = st.slider("Fundamental Weight", 0.1, 0.8, float(params["fund_weight"]), 0.05)
        with col3:
            mkt_w = st.slider("Market Context Weight", 0.05, 0.5, float(params["market_weight"]), 0.05)

        total_w = tech_w + fund_w + mkt_w
        if abs(total_w - 1.0) > 0.01:
            st.warning(f"⚠️ Weights sum to {total_w:.2f} (should be 1.0) — will be auto-normalized.")

        st.subheader("📉 Technical Indicator Weights")
        rsi_w = st.slider("RSI Weight (multiplier)", 0.5, 2.0, float(params.get("rsi_weight", 1.0)), 0.1)

        st.subheader("💰 Trade Management")
        col1, col2 = st.columns(2)
        with col1:
            stop_loss = st.slider("Stop Loss %", 3, 20, int(params["stop_loss_pct"] * 100), 1)
            trailing_stop = st.slider("Trailing Stop %", 3, 15, int(params["trailing_stop_pct"] * 100), 1)
            min_score = st.slider("Min AI Score to Trade", 40, 85, int(params["min_score_to_trade"]), 5)
        with col2:
            max_pos = st.slider("Max Open Positions", 1, 10, int(params["max_positions"]), 1)
            slippage = st.slider("Slippage %", 0.1, 1.0, float(params["slippage_pct"] * 100), 0.05)
            commission = st.slider("Commission %", 0.0, 0.5, float(params["commission_pct"] * 100), 0.025)

        submitted = st.form_submit_button("💾 Save Strategy Params", type="primary")
        if submitted:
            total = tech_w + fund_w + mkt_w
            new_params = {
                "tech_weight": round(tech_w / total, 3),
                "fund_weight": round(fund_w / total, 3),
                "market_weight": round(mkt_w / total, 3),
                "rsi_weight": rsi_w,
                "stop_loss_pct": stop_loss / 100,
                "trailing_stop_pct": trailing_stop / 100,
                "take_profit_pct": params.get("take_profit_pct", 0.20),
                "min_score_to_trade": min_score,
                "max_positions": max_pos,
                "slippage_pct": slippage / 100,
                "commission_pct": commission / 100,
                "position_risk_pct": params.get("position_risk_pct", 0.02),
                "version": params.get("version", 1) + 1,
            }
            st.session_state.strategy_params = new_params
            save_strategy_params(new_params)
            st.success("✅ Strategy saved!")
            st.rerun()

    # Current params display
    st.subheader("📋 Current Active Parameters")
    col1, col2 = st.columns(2)
    with col1:
        st.json({k: v for k, v in params.items() if "weight" in k or "pct" in k})
    with col2:
        st.json({k: v for k, v in params.items() if "weight" not in k and "pct" not in k})


def render_performance_tab():
    """Render performance with separate US/India analysis."""
    st.header("📈 Performance & Self-Improvement")
    
    tab_us, tab_india = st.tabs(["🇺🇸 US Performance", "🇮🇳 India Performance"])
    
    with tab_us:
        render_market_performance("US")
    with tab_india:
        render_market_performance("India")


def render_market_performance(market: str):
    """Render performance analysis for a specific market."""
    st.subheader(f"{market} Trading Performance")

    trader = st.session_state.trader
    if trader is None or not trader.closed_trades:
        st.info("📌 No completed trades yet. Start the simulator and run some trading cycles.")
        # Show placeholder analysis cards
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            ### 🧠 Self-Improvement Engine
            After 10+ simulated trades, the AI will:
            - Analyze win/loss patterns
            - Identify which indicators predicted correctly
            - Suggest parameter adjustments
            - Auto-update scoring weights
            """)
        with col2:
            st.markdown("""
            ### 📊 Metrics Tracked
            - **Win Rate**: % of trades profitable
            - **Profit Factor**: Gross profit / Gross loss
            - **Max Drawdown**: Peak-to-trough decline
            - **Sharpe Ratio**: Risk-adjusted return
            - **Score Correlation**: AI score vs trade outcome
            """)
        return

    summary = trader.get_performance_summary()

    # Metrics
    st.subheader("📊 Performance Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Trades", summary["total_trades"])
    col2.metric("Win Rate", f"{summary['win_rate']:.0f}%")
    col3.metric("Profit Factor", f"{summary['profit_factor']:.2f}")
    col4.metric("Total Return", f"{summary['total_return_pct']:+.1f}%")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg Win", f"{summary['avg_win_pct']:+.1f}%")
    col2.metric("Avg Loss", f"{summary['avg_loss_pct']:+.1f}%")
    col3.metric("Max Drawdown", f"{summary['max_drawdown']:.1f}%")
    col4.metric("Net P&L", f"${summary['total_pnl']:+,.0f}")

    # Trade outcome distribution
    if len(trader.closed_trades) >= 3:
        st.subheader("📉 Trade Outcome Distribution")
        pnl_pcts = [t["pnl_pct"] for t in trader.closed_trades]
        colors = ["#69f0ae" if p > 0 else "#ff5252" for p in pnl_pcts]
        symbols = [t["symbol"] for t in trader.closed_trades]

        t = get_theme()
        fig_dist = go.Figure(go.Bar(
            x=symbols, y=pnl_pcts,
            marker_color=[t["positive"] if p > 0 else t["negative"] for p in pnl_pcts],
            text=[f"{p:+.1f}%" for p in pnl_pcts],
            textposition="outside",
            textfont=dict(color=t["text"]),
        ))
        fig_dist.update_layout(
            **get_plotly_layout(t, height=350, title="P&L % by Trade"),
            yaxis_title="P&L %",
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    # AI Score vs Outcome
    if len(trader.closed_trades) >= 5:
        scores = [t.get("ai_score", 0) for t in trader.closed_trades]
        pnls = [t["pnl_pct"] for t in trader.closed_trades]
        t2 = get_theme()
        fig_corr = px.scatter(
            x=scores, y=pnls,
            labels={"x": "AI Score at Entry", "y": "P&L %"},
            title="AI Score vs Trade Outcome",
            trendline="ols",
            color=[("Win" if p > 0 else "Loss") for p in pnls],
            color_discrete_map={"Win": t2["positive"], "Loss": t2["negative"]},
            template=t2["plotly_template"],
        )
        fig_corr.update_layout(
            **get_plotly_layout(t2, height=350, title="AI Score vs Trade Outcome"),
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    # Self-improvement suggestions
    st.subheader("🤖 AI Self-Improvement Suggestions")
    suggestions = trader.suggest_improvements()
    if suggestions:
        for s in suggestions:
            st.markdown(f"- {s}")

    # Apply suggestions button
    if len(trader.closed_trades) >= 5:
        if st.button("🔄 Auto-Apply Suggested Improvements", type="primary"):
            params = st.session_state.strategy_params.copy()
            summary = trader.get_performance_summary()

            # Auto-adjust based on performance
            if summary["win_rate"] < 45:
                params["min_score_to_trade"] = min(params["min_score_to_trade"] + 5, 80)
            if summary["avg_loss_pct"] < -12:
                params["stop_loss_pct"] = max(params["stop_loss_pct"] - 0.02, 0.05)
            if summary["profit_factor"] < 1.2:
                params["version"] = params.get("version", 1) + 1

            # High-score stock performance
            high_score_trades = [t for t in trader.closed_trades if t.get("ai_score", 0) > 75]
            if high_score_trades:
                hs_wr = sum(1 for t in high_score_trades if t["pnl"] > 0) / len(high_score_trades)
                if hs_wr > 0.6:
                    params["tech_weight"] = min(params["tech_weight"] + 0.05, 0.6)
                    # Renormalize
                    total = params["tech_weight"] + params["fund_weight"] + params["market_weight"]
                    params["fund_weight"] = params["fund_weight"] / total
                    params["market_weight"] = params["market_weight"] / total
                    params["tech_weight"] = params["tech_weight"] / total

            st.session_state.strategy_params = params
            save_strategy_params(params)
            st.success("✅ Strategy parameters updated based on performance analysis!")
            st.rerun()


# ─── THEME HELPERS ────────────────────────────────────────────────────────────
def get_theme() -> Dict:
    """Return colour tokens for the current mode."""
    dark = st.session_state.get("dark_mode", True)
    if dark:
        return {
            "bg":                "#0a0e1a",
            "surface":           "#111827",
            "surface2":          "#1a2236",
            "border":            "#1f2d45",
            "text":              "#e8edf5",
            "text_muted":        "#64748b",
            "text_secondary":    "#94a3b8",
            "accent":            "#3b82f6",
            "accent_glow":       "rgba(59,130,246,0.15)",
            "accent2":           "#10b981",
            "accent2_glow":      "rgba(16,185,129,0.15)",
            "warning":           "#f59e0b",
            "danger":            "#ef4444",
            "sidebar_bg":        "#0d1421",
            "disclaimer_bg":     "#1a1010",
            "disclaimer_border": "#7f1d1d",
            "disclaimer_text":   "#fca5a5",
            "tab_bg":            "#111827",
            "tab_selected":      "#3b82f6",
            "plotly_template":   "plotly_dark",
            "chart_bg":          "#0a0e1a",
            "chart_surface":     "#111827",
            "up_color":          "#10b981",
            "down_color":        "#ef4444",
            "positive":          "#10b981",
            "negative":          "#ef4444",
            "grid":              "#1f2d45",
        }
    else:
        return {
            "bg":                "#f8fafc",
            "surface":           "#ffffff",
            "surface2":          "#f1f5f9",
            "border":            "#e2e8f0",
            "text":              "#0f172a",
            "text_muted":        "#94a3b8",
            "text_secondary":    "#64748b",
            "accent":            "#2563eb",
            "accent_glow":       "rgba(37,99,235,0.08)",
            "accent2":           "#059669",
            "accent2_glow":      "rgba(5,150,105,0.08)",
            "warning":           "#d97706",
            "danger":            "#dc2626",
            "sidebar_bg":        "#ffffff",
            "disclaimer_bg":     "#fef2f2",
            "disclaimer_border": "#fca5a5",
            "disclaimer_text":   "#991b1b",
            "tab_bg":            "#f1f5f9",
            "tab_selected":      "#2563eb",
            "plotly_template":   "plotly_white",
            "chart_bg":          "#ffffff",
            "chart_surface":     "#f8fafc",
            "up_color":          "#059669",
            "down_color":        "#dc2626",
            "positive":          "#059669",
            "negative":          "#dc2626",
            "grid":              "#e2e8f0",
        }


def apply_theme_css():
    """Inject dynamic CSS based on current dark/light mode."""
    t = get_theme()
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"], .stApp {{
        font-family: 'Inter', -apple-system, sans-serif !important;
        background-color: {t['bg']} !important;
        color: {t['text']} !important;
    }}
    .stApp {{ background-color: {t['bg']} !important; }}

    section[data-testid="stSidebar"] > div,
    div[data-testid="stSidebarContent"] {{
        background-color: {t['sidebar_bg']} !important;
        border-right: 1px solid {t['border']};
    }}

    div[data-testid="metric-container"] {{
        background: {t['surface']};
        border: 1px solid {t['border']};
        border-radius: 12px;
        padding: 16px 18px;
        transition: border-color 0.2s;
    }}
    div[data-testid="metric-container"]:hover {{ border-color: {t['accent']}; }}
    div[data-testid="metric-container"] label,
    div[data-testid="metric-container"] [data-testid="stMetricLabel"] {{
        color: {t['text_muted']} !important;
        font-size: 0.72rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.07em;
    }}
    div[data-testid="metric-container"] [data-testid="stMetricValue"] {{
        color: {t['text']} !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 1.4rem !important;
        font-weight: 700 !important;
    }}
    div[data-testid="metric-container"] [data-testid="stMetricDelta"] {{
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.78rem !important;
        font-weight: 600;
    }}

    .stButton > button {{
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.875rem !important;
        transition: all 0.18s ease !important;
    }}
    .stButton > button[kind="primary"] {{
        background: {t['accent']} !important;
        border: none !important;
        color: #fff !important;
    }}
    .stButton > button[kind="primary"]:hover {{
        filter: brightness(1.15) !important;
        box-shadow: 0 4px 14px {t['accent_glow']} !important;
        transform: translateY(-1px) !important;
    }}
    .stButton > button[kind="secondary"] {{
        background: {t['surface2']} !important;
        border: 1px solid {t['border']} !important;
        color: {t['text']} !important;
    }}
    .stButton > button[kind="secondary"]:hover {{
        border-color: {t['accent']} !important;
        color: {t['accent']} !important;
    }}

    .stTabs [data-baseweb="tab-list"] {{
        background: {t['tab_bg']};
        border-radius: 10px;
        padding: 4px;
        gap: 3px;
        border: 1px solid {t['border']};
    }}
    .stTabs [data-baseweb="tab"] {{
        background: transparent !important;
        color: {t['text_muted']} !important;
        border-radius: 7px !important;
        font-weight: 500 !important;
        font-size: 0.875rem !important;
        padding: 6px 16px !important;
        transition: all 0.15s !important;
    }}
    .stTabs [data-baseweb="tab"]:hover {{
        color: {t['text']} !important;
        background: {t['border']} !important;
    }}
    .stTabs [aria-selected="true"] {{
        background: {t['tab_selected']} !important;
        color: #fff !important;
        font-weight: 600 !important;
    }}

    .stTextInput > div > div > input,
    .stNumberInput > div > div > input {{
        background: {t['surface']} !important;
        color: {t['text']} !important;
        border: 1px solid {t['border']} !important;
        border-radius: 8px !important;
    }}
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus {{
        border-color: {t['accent']} !important;
        box-shadow: 0 0 0 3px {t['accent_glow']} !important;
    }}
    .stSelectbox > div > div,
    .stMultiSelect > div > div {{
        background: {t['surface']} !important;
        color: {t['text']} !important;
        border: 1px solid {t['border']} !important;
        border-radius: 8px !important;
    }}

    .stDataFrame {{ border-radius: 12px; overflow: hidden; border: 1px solid {t['border']}; }}

    .stCode, code, pre {{
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.8rem !important;
        background: {t['surface']} !important;
        color: {t['text']} !important;
        border: 1px solid {t['border']} !important;
        border-radius: 8px !important;
    }}

    div[data-testid="stInfo"] {{
        background: {t['accent_glow']};
        border-left: 3px solid {t['accent']};
        border-radius: 8px;
        color: {t['text']} !important;
    }}
    div[data-testid="stSuccess"] {{
        background: {t['accent2_glow']};
        border-left: 3px solid {t['accent2']};
        border-radius: 8px;
    }}

    .stProgress > div > div > div > div {{
        background: linear-gradient(90deg, {t['accent']}, {t['accent2']}) !important;
        border-radius: 99px;
    }}
    .stProgress > div > div > div {{
        background: {t['surface2']} !important;
        border-radius: 99px;
    }}

    .stSpinner > div {{ border-top-color: {t['accent']} !important; }}
    .stToggle label {{ color: {t['text']} !important; font-size: 0.875rem !important; }}

    h1, h2, h3, h4 {{ color: {t['text']} !important; font-weight: 700 !important; letter-spacing: -0.02em; }}
    h2 {{ border-bottom: 1px solid {t['border']}; padding-bottom: 8px; margin-bottom: 16px; }}

    .stCaption, [data-testid="stCaptionContainer"], small {{
        color: {t['text_muted']} !important;
        font-size: 0.78rem !important;
    }}
    hr {{ border-color: {t['border']} !important; margin: 20px 0 !important; }}

    .disclaimer-box {{
        background: {t['disclaimer_bg']};
        border: 1px solid {t['disclaimer_border']};
        border-radius: 10px;
        padding: 12px 18px;
        font-size: 0.8rem;
        color: {t['disclaimer_text']};
        margin-bottom: 16px;
        line-height: 1.6;
    }}
    .stat-card {{
        background: {t['surface']};
        border: 1px solid {t['border']};
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 8px;
    }}
    .ledger-pos {{ color: {t['positive']}; font-weight: 700; font-family: 'JetBrains Mono', monospace; }}
    .ledger-neg {{ color: {t['negative']}; font-weight: 700; font-family: 'JetBrains Mono', monospace; }}
    .section-label {{
        font-size: 0.7rem; font-weight: 700; text-transform: uppercase;
        letter-spacing: 0.09em; color: {t['text_muted']}; display: block; margin-bottom: 10px;
    }}
    .badge {{
        display: inline-block; padding: 2px 10px; border-radius: 99px;
        font-size: 0.7rem; font-weight: 700; letter-spacing: 0.05em;
        background: {t['accent_glow']}; color: {t['accent']};
        border: 1px solid {t['accent']}; margin-right: 4px;
    }}
    .badge-green {{
        background: {t['accent2_glow']}; color: {t['accent2']}; border-color: {t['accent2']};
    }}
    ::-webkit-scrollbar {{ width: 5px; height: 5px; }}
    ::-webkit-scrollbar-track {{ background: {t['bg']}; }}
    ::-webkit-scrollbar-thumb {{ background: {t['border']}; border-radius: 99px; }}
    ::-webkit-scrollbar-thumb:hover {{ background: {t['text_muted']}; }}
    </style>
    """, unsafe_allow_html=True)

def get_plotly_layout(t: Dict, height: int = 400, title: str = "") -> Dict:
    """Return consistent Plotly layout kwargs based on current theme."""
    return dict(
        template=t["plotly_template"],
        paper_bgcolor=t["chart_bg"],
        plot_bgcolor=t["chart_bg"],
        height=height,
        margin=dict(l=0, r=0, t=40 if title else 10, b=0),
        title=title,
        font=dict(color=t["text"], family="Inter, sans-serif"),
        xaxis=dict(gridcolor=t["border"], zerolinecolor=t["border"]),
        yaxis=dict(gridcolor=t["border"], zerolinecolor=t["border"]),
    )


# ─── MAIN APP ─────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="AI Stock Screener + Simulator (Multi-User)",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    # Initialize state
    init_session_state()
    
    # Check login
    if not st.session_state.get("logged_in"):
        apply_theme_css()
        render_login_screen()
        return
    
    # Apply theme
    apply_theme_css()
    t = get_theme()
    
    # Auto-save on every interaction
    username = st.session_state.get("username")
    
    # ── Sidebar — User info ────────────────────────────────────────────────────
    st.sidebar.markdown(f"""
    <div style='background:{t["surface"]};border:1px solid {t["border"]};border-radius:10px;
    padding:12px 16px;margin-bottom:12px;text-align:center;'>
        <div style='font-size:0.7rem;color:{t["text_muted"]};text-transform:uppercase;
        letter-spacing:0.08em;margin-bottom:4px;'>Logged in as</div>
        <div style='font-size:1.1rem;font-weight:700;color:{t["accent"]};'>
            👤 {username}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.sidebar.button("🚪 Logout", use_container_width=True):
        save_user_data(username)
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.rerun()
    
    # ── Sidebar — Display / Theme ──────────────────────────────────────────────
    st.sidebar.markdown(f"<span class='section-label'>🎨 Display</span>", unsafe_allow_html=True)
    mode_icon  = "🌙" if st.session_state.dark_mode else "☀️"
    mode_label = "Dark Mode" if st.session_state.dark_mode else "Light Mode"
    toggled = st.sidebar.toggle(
        f"{mode_icon}  {mode_label}",
        value=st.session_state.dark_mode,
        key="theme_toggle",
    )
    if toggled != st.session_state.dark_mode:
        st.session_state.dark_mode = toggled
        save_user_data(username)
        st.rerun()
    
    # ── Sidebar — API Keys + Controls ─────────────────────────────────────────
    render_api_key_setup()
    render_screening_controls()
    
    # ── Sidebar — Session Status ───────────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"<span class='section-label'>📊 Status</span>", unsafe_allow_html=True)
    
    us_active = st.session_state.get("trading_active_us", False)
    india_active = st.session_state.get("trading_active_india", False)
    
    if us_active or india_active:
        st.sidebar.success("🟢 Trading: ACTIVE")
    else:
        st.sidebar.info("⚫ Trading: Stopped")
    
    trader_us = st.session_state.get("trader_us")
    trader_india = st.session_state.get("trader_india")
    
    if trader_us:
        pv = trader_us.portfolio_value
        initial = trader_us.initial_capital
        ret = (pv / initial - 1) * 100
        st.sidebar.metric("🇺🇸 US Portfolio", f"${pv:,.0f}", delta=f"{ret:+.1f}%")
    
    if trader_india:
        pv = trader_india.portfolio_value
        initial = trader_india.initial_capital
        ret = (pv / initial - 1) * 100
        st.sidebar.metric("🇮🇳 India Portfolio", f"₹{pv:,.0f}", delta=f"{ret:+.1f}%")
    
    st.sidebar.markdown("---")
    st.sidebar.caption(f"📦 cryptography: {'✅' if CRYPTO_AVAILABLE else '❌'}")
    st.sidebar.caption(f"📦 plotly: {'✅' if PLOTLY_AVAILABLE else '❌'}")
    st.sidebar.caption(f"📦 ta: {'✅' if TA_AVAILABLE else '⚪'}")
    
    # ── App header ─────────────────────────────────────────────────────────────
    markets_active = st.session_state.get("markets", ["US", "India"])
    badges_html = "".join([
        f"<span class='badge'>🇺🇸 US</span>" if m == "US"
        else f"<span class='badge badge-green'>🇮🇳 India</span>"
        for m in markets_active
    ])
    
    st.markdown(f"""
    <div style="margin-bottom:18px;">
        <div style="display:flex;align-items:flex-end;gap:14px;flex-wrap:wrap;margin-bottom:6px;">
            <div style="font-size:2rem;font-family:'JetBrains Mono',monospace;font-weight:800;
                        background:linear-gradient(135deg,{t['accent']},{t['accent2']});
                        -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
                📈 AI Stock Screener
            </div>
            <div style="display:flex;gap:6px;align-items:center;padding-bottom:4px;">
                {badges_html}
                <span style="font-size:0.72rem;color:{t['text_muted']};font-weight:500;">
                    &nbsp;Multi-User &nbsp;·&nbsp; Persistent Storage &nbsp;·&nbsp; Separate Strategies
                </span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # ── Main tabs ───────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍  Screener",
        "💹  Simulator",
        "⚙️  Settings",
        "📈  Performance",
    ])
    
    with tab1:
        render_screener_tab()
    with tab2:
        render_simulator_tab()
    with tab3:
        render_settings_tab()
    with tab4:
        render_performance_tab()
    
    # Auto-save periodically
    save_user_data(username)
    
    # Auto-refresh
    if st.session_state.auto_refresh and (us_active or india_active):
        time.sleep(1)
        st.rerun()


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()

