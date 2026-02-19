# 🔧 MANUAL CLEANUP INSTRUCTIONS

The automated removal has syntax issues. Here's how to fix it manually in 5 minutes:

## Step 1: Open `ai_stock_screener.py` in a text editor

Use VS Code, Notepad++, Sublime, or any editor.

## Step 2: Delete These Sections

### A. Delete the INDIA_TICKERS list (lines ~100-185)
Find this:
```python
# ─── INDIA STOCKS — NSE (pure equities only) ─────────────────────────────────
INDIA_TICKERS = [
    "RELIANCE.NS","TCS.NS","HDFCBANK.NS",
    ... (many lines)
]
```

Delete everything from `# ─── INDIA STOCKS` down to (but NOT including) the line that says `ALL_TICKERS =`

### B. Delete the INR constant (line ~58)
Find and delete:
```python
INR_TO_USD = 1 / 83.5
```

### C. Fix the ALL_TICKERS line (line ~216)  
Change this:
```python
ALL_TICKERS = US_TICKERS + INDIA_TICKERS
```

To this:
```python
ALL_TICKERS = US_TICKERS
```

### D. Delete the _normalize_price_to_usd method (lines ~867-872)
Find and delete:
```python
    def _normalize_price_to_usd(self, price: float, symbol: str) -> float:
        """Convert INR to USD for Indian stocks."""
        if symbol.endswith(".NS") or symbol.endswith(".BO"):
            return price * INR_TO_USD
        return price
```

### E. Remove the India market option (lines ~1540)
Find this:
```python
    st.session_state.markets = st.sidebar.multiselect(
        "Markets",
        ["US", "India"],  # ← Change this line
        default=st.session_state.markets,
    )
```

Change `["US", "India"]` to `["US"]`

### F. Update the header (line ~4)
Change:
```python
║     US Stocks (NYSE/NASDAQ) + Indian Stocks (NSE)                    ║
```

To:
```python
║     US Stocks Only — NYSE / NASDAQ                                   ║
```

## Step 3: Save the File

Save as `ai_stock_screener_us_only.py`

## Step 4: Test It

```bash
python3 -m py_compile ai_stock_screener_us_only.py
```

If no errors, you're good!

## Step 5: Upload to GitHub

```bash
git add ai_stock_screener_us_only.py
git commit -m "Add US-only version"
git push origin main
```

---

## Or Use Find & Replace

If your editor has find & replace:

1. Find: `INDIA_TICKERS` → Replace: `# INDIA_TICKERS` (comment it out)
2. Find: `INR_TO_USD = 1 / 83.5` → Replace: `# INR_TO_USD = 1 / 83.5`
3. Find: `+ INDIA_TICKERS` → Replace: `` (empty - remove it)
4. Find: `["US", "India"]` → Replace: `["US"]`
5. Find: `Indian` → Replace: `US` (in comments/docs only)

Save and test!

---

## Why Manual?

The automated script breaks Python syntax when removing code blocks. Manual editing is safer and takes just 5 minutes.

## Final Check

After editing, run:
```bash
python3 ai_stock_screener_us_only.py
```

If it starts without errors, you're done! ✅
