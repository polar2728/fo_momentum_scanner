"""
Institutional F&O Positional Futures Scanner (Stable 2026 version)
Focus: Long Buildup (OI ↑ + Price ↑ + RS vs Nifty + ATR stop)

Improvements:
- Correct NSE bhavcopy URL for 2025–2026 format
- Exclude index futures (NIFTY, BANKNIFTY, etc.)
- Proper relative strength (total return over lookback)
- Caching on price data
- ThreadPool with error handling
- Ban-list check (optional)
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import zipfile
import io
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta

st.set_page_config(page_title="F&O Positional Scanner", layout="wide")

# =========================================================
# CONFIG
# =========================================================

RS_LOOKBACK = 60
ATR_PERIOD = 14
MIN_OI_PCT_CHANGE = 5.0
MAX_WORKERS = 6
USE_BAN_FILTER = True

# =========================================================
# UTILITY FUNCTIONS
# =========================================================

def get_last_trading_day():
    return date.today() - timedelta(days=1)

# ─────────────────────────────────────────────
# FETCH F&O BHAVCOPY (2026 format)
# ─────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fo_bhavcopy():
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0"})
    try:
        s.get("https://www.nseindia.com", timeout=5)  # session priming
    except:
        pass

    dt = get_last_trading_day()
    for i in range(7):  # try last 7 days
        d = dt - timedelta(days=i)
        date_str = d.strftime("%Y%m%d")
        url = f"https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{date_str}_F_0000.csv.zip"
        try:
            r = s.get(url, timeout=10)
            if r.status_code == 200:
                with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                    df = pd.read_csv(z.open(z.namelist()[0]))
                    df.columns = df.columns.str.strip().str.upper()
                    return df
        except:
            continue
    return pd.DataFrame()

# ─────────────────────────────────────────────
# EXTRACT STOCK FUTURES ONLY (exclude indices)
# ─────────────────────────────────────────────
def extract_stock_universe(df):
    if df.empty:
        return []

    # Filter to stock futures (FUTSTK)
    df = df[df["INSTRUMENT"] == "FUTSTK"]

    # Exclude index-like symbols
    exclude = ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTYNXT50", "SENSEX"]
    symbols = [s for s in df["SYMBOL"].unique() if s not in exclude]

    return sorted(symbols)

# ─────────────────────────────────────────────
# OPTIONAL: FETCH BAN LIST
# ─────────────────────────────────────────────
@st.cache_data(ttl=3600)
def get_banned_symbols():
    try:
        url = "https://nsearchives.nseindia.com/content/fo/fo_secban.csv"
        r = requests.get(url, timeout=8)
        if r.status_code == 200:
            df = pd.read_csv(io.StringIO(r.text))
            return set(df.iloc[:,0].str.strip().str.upper())
    except:
        pass
    return set()

# ─────────────────────────────────────────────
# PRICE DATA FETCH (cached per symbol)
# ─────────────────────────────────────────────
@st.cache_data(ttl=3600 * 2, show_spinner=False)
def fetch_price_data(symbol):
    try:
        df = yf.download(f"{symbol}.NS", period="4mo", progress=False)
        if len(df) < RS_LOOKBACK + 10:
            return None

        # ATR
        high_low = df["High"] - df["Low"]
        high_close = np.abs(df["High"] - df["Close"].shift())
        low_close = np.abs(df["Low"] - df["Close"].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(ATR_PERIOD).mean().iloc[-1]

        return df, df["Close"].iloc[-1], atr
    except:
        return None

# ─────────────────────────────────────────────
# RELATIVE STRENGTH (correct total return)
# ─────────────────────────────────────────────
def calculate_rs(stock_close, nifty_close):
    if len(stock_close) < RS_LOOKBACK or len(nifty_close) < RS_LOOKBACK:
        return np.nan
    stock_ret = (stock_close.iloc[-1] / stock_close.iloc[-RS_LOOKBACK]) - 1
    nifty_ret = (nifty_close.iloc[-1] / nifty_close.iloc[-RS_LOOKBACK]) - 1
    return stock_ret - nifty_ret

# =========================================================
# UI & SCAN LOGIC
# =========================================================

st.title("🔥 Institutional F&O Positional Futures Scanner")
st.caption("Long Buildup Focus • OI ↑ + Price ↑ + RS vs Nifty • Stable 2026 version")

capital = st.number_input("Capital (₹)", min_value=100000, value=1000000, step=100000)
risk_pct = st.slider("Risk per Trade (%)", 0.5, 5.0, 1.0, 0.1)

if st.button("Run Scan", type="primary"):

    with st.spinner("Loading F&O universe & price data..."):

        fo_df = fetch_fo_bhavcopy()
        if fo_df.empty:
            st.error("Failed to fetch F&O bhavcopy. Try again later.")
            st.stop()

        symbols = extract_stock_universe(fo_df)
        if not symbols:
            st.error("No stock futures found in bhavcopy.")
            st.stop()

        # Optional ban filter
        banned = get_banned_symbols() if USE_BAN_FILTER else set()
        symbols = [s for s in symbols if s not in banned]

        st.caption(f"Scanning {len(symbols)} stock futures (after ban filter)")

        nifty = yf.download("^NSEI", period="4mo", progress=False)
        if nifty.empty:
            st.warning("Nifty data unavailable — RS skipped.")
            nifty_close = pd.Series()
        else:
            nifty_close = nifty["Close"]

        results = []

        progress_bar = st.progress(0)
        status_text = st.empty()

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_sym = {executor.submit(fetch_price_data, sym): sym for sym in symbols}
            total = len(symbols)

            for idx, future in enumerate(as_completed(future_to_sym)):
                sym = future_to_sym[future]
                try:
                    data = future.result()
                    if data is None:
                        continue

                    df_price, close, atr = data

                    # Get OI & price change from bhavcopy
                    row = fo_df[(fo_df["SYMBOL"] == sym) & (fo_df["INSTRUMENT"] == "FUTSTK")]
                    if row.empty:
                        continue

                    oi_change_pct = row["CHG_IN_OI"].iloc[0] / row["OPEN_INT"].iloc[0] * 100 if row["OPEN_INT"].iloc[0] > 0 else 0
                    price_change_pct = (row["SETTLE_PR"].iloc[0] / row["OPEN"].iloc[0]) - 1

                    # Core Long Buildup condition
                    if oi_change_pct > MIN_OI_PCT_CHANGE and price_change_pct > 0:

                        rs = calculate_rs(df_price["Close"], nifty_close) if not nifty_close.empty else 0

                        if rs > 0:  # relative strength positive
                            stop = close - (2 * atr)
                            risk_per_share = close - stop
                            if risk_per_share <= 0:
                                continue

                            risk_amt = capital * (risk_pct / 100)
                            qty = int(risk_amt / risk_per_share)

                            results.append({
                                "Symbol": sym,
                                "Close": round(close, 2),
                                "OI % Chg": round(oi_change_pct, 2),
                                "Price % Chg": round(price_change_pct * 100, 2),
                                "RS vs Nifty": round(rs, 3),
                                "ATR": round(atr, 2),
                                "Stop": round(stop, 2),
                                "Qty": qty,
                                "Risk Amt": round(qty * risk_per_share, 0),
                            })

                except Exception as e:
                    pass  # silent fail per symbol

                progress_bar.progress((idx + 1) / total)
                status_text.text(f"Processed {idx+1}/{total} symbols...")

        progress_bar.empty()
        status_text.empty()

        if not results:
            st.warning("No high-probability long buildup setups found today.")
        else:
            df_final = pd.DataFrame(results)
            df_final = df_final.sort_values("RS vs Nifty", ascending=False).reset_index(drop=True)
            df_final.index += 1

            st.success(f"Found {len(df_final)} high-probability futures setups")
            st.dataframe(
                df_final.style.format({
                    "Close": "₹{:,.2f}",
                    "OI % Chg": "{:+.2f}%",
                    "Price % Chg": "{:+.2f}%",
                    "ATR": "₹{:,.2f}",
                    "Stop": "₹{:,.2f}",
                    "Risk Amt": "₹{:,.0f}",
                }),
                use_container_width=True
            )

            csv = df_final.to_csv(index=True).encode('utf-8')
            st.download_button(
                "⬇️ Download CSV",
                csv,
                f"futures_scanner_{datetime.date.today().strftime('%Y%m%d')}.csv",
                "text/csv"
            )