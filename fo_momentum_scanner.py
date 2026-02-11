import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import ta
import requests
import zipfile
import io
import time
import datetime
from datetime import date, timedelta

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="F&O Positional Futures Scanner",
    page_icon="📈",
    layout="wide"
)

st.title("📈 F&O Positional Futures Scanner")
st.caption("2–4 Week Institutional Long Setup Engine")

# ─────────────────────────────────────────────
# SIDEBAR SETTINGS
# ─────────────────────────────────────────────
st.sidebar.header("⚙️ Risk Settings")

capital = st.sidebar.number_input("Total Capital (₹)", value=1000000)
risk_pct = st.sidebar.slider("Risk Per Trade (%)", 0.5, 3.0, 1.0)
risk_amount = capital * (risk_pct / 100)

st.sidebar.markdown(f"**Risk per Trade: ₹{risk_amount:,.0f}**")

st.sidebar.header("📋 Stock Universe")

DEFAULT_SYMBOLS = [
    "RELIANCE","TCS","HDFCBANK","ICICIBANK","INFY","SBIN",
    "LT","AXISBANK","BAJFINANCE","TATAMOTORS","SUNPHARMA",
    "MARUTI","ADANIENT","HAL","BEL","IRFC","PFC","RECLTD"
]

universe_type = st.sidebar.radio(
    "Universe Type",
    ["Default F&O List", "Custom Symbols"]
)

if universe_type == "Default F&O List":
    symbols = DEFAULT_SYMBOLS
else:
    custom = st.sidebar.text_area("Enter symbols (comma separated)")
    symbols = [x.strip().upper() for x in custom.split(",") if x.strip()]

# ─────────────────────────────────────────────
# NSE SESSION
# ─────────────────────────────────────────────
def get_nse_session():
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://www.nseindia.com"
    })
    try:
        s.get("https://www.nseindia.com", timeout=5)
    except:
        pass
    return s

# ─────────────────────────────────────────────
# EXPIRY WEEK FILTER
# ─────────────────────────────────────────────
def is_expiry_week():
    today = datetime.date.today()
    last_day = today.replace(day=28) + datetime.timedelta(days=4)
    last_thursday = last_day - datetime.timedelta(days=last_day.weekday() - 3)
    return (last_thursday - today).days <= 3

# ─────────────────────────────────────────────
# FETCH BHAVCOPIES
# ─────────────────────────────────────────────
@st.cache_data(ttl=14400)
def get_recent_bhavcopies(days=6):
    s = get_nse_session()
    bhav = {}
    today = date.today()

    for i in range(days + 10):
        d = today - timedelta(days=i)
        url = f"https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{d.strftime('%Y%m%d')}_F_0000.csv.zip"
        try:
            r = s.get(url, timeout=10)
            if r.status_code == 200:
                with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                    bhav[d] = pd.read_csv(z.open(z.namelist()[0]), low_memory=False)
                if len(bhav) >= days:
                    break
        except:
            continue

    return bhav

# ─────────────────────────────────────────────
# OI % CHANGE (NEAR MONTH)
# ─────────────────────────────────────────────
def get_oi_percent_change(symbol, bhavcopies, lookback=4):

    oi_values = []

    for d, df in sorted(bhavcopies.items()):
        fut = df[
            (df.get('TckrSymb', '') == symbol) &
            (df.get('FinInstrmTp', '') == 'STF')
        ]

        if fut.empty or 'XpryDt' not in fut.columns:
            continue

        nearest_expiry = fut['XpryDt'].min()
        fut = fut[fut['XpryDt'] == nearest_expiry]

        if 'OpnIntrst' in fut.columns:
            oi_values.append(fut['OpnIntrst'].sum())

    if len(oi_values) < lookback:
        return False, 0

    recent = oi_values[-lookback:]
    if recent[0] == 0:
        return False, 0

    pct_change = ((recent[-1] - recent[0]) / recent[0]) * 100
    positive_days = sum(recent[i] > recent[i-1] for i in range(1, len(recent)))

    return (pct_change > 5 and positive_days >= 2), pct_change

# ─────────────────────────────────────────────
# FETCH PRICE DATA
# ─────────────────────────────────────────────
@st.cache_data(ttl=21600)
def fetch_ohlcv(symbol):
    ticker = yf.Ticker(symbol + ".NS")
    df = ticker.history(period="14mo", auto_adjust=True)
    if df.empty:
        return pd.DataFrame()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df

@st.cache_data(ttl=21600)
def get_nifty():
    df = yf.Ticker("^NSEI").history(period="6mo", auto_adjust=True)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df

# ─────────────────────────────────────────────
# SCAN LOGIC
# ─────────────────────────────────────────────
def scan_stock(symbol, bhavcopies, nifty_df):

    df = fetch_ohlcv(symbol)
    if df.empty or len(df) < 260:
        return None

    close = df["Close"]
    volume = df["Volume"]

    latest = close.iloc[-1]
    prev = close.iloc[-2]

    # Stage 2 Trend
    ma20 = close.rolling(20).mean().iloc[-1]
    ma50 = close.rolling(50).mean().iloc[-1]
    ma200 = close.rolling(200).mean().iloc[-1]

    if not (ma20 > ma50 > ma200 and latest > ma20):
        return None

    # Momentum
    ret_1m = (latest / close.iloc[-22] - 1) * 100
    ret_6m = (latest / close.iloc[-130] - 1) * 100

    if not (ret_1m > 8 and ret_1m > ret_6m):
        return None

    # Relative Strength vs Nifty
    nifty_1m = (nifty_df["Close"].iloc[-1] /
                nifty_df["Close"].iloc[-22] - 1) * 100

    if ret_1m <= nifty_1m:
        return None

    rs_outperformance = ret_1m - nifty_1m

    # Expiry distortion filter
    if is_expiry_week():
        return None

    # OI %
    oi_ok, oi_pct = get_oi_percent_change(symbol, bhavcopies)
    if not oi_ok or latest <= prev:
        return None

    # RSI
    rsi = ta.momentum.RSIIndicator(close, 14).rsi().iloc[-1]
    if not (60 <= rsi <= 75):
        return None

    # Volume median filter
    vol_median = volume.iloc[-21:-1].median()
    vol_ratio = volume.iloc[-1] / vol_median if vol_median > 0 else 0
    if vol_ratio < 1.5:
        return None

    # Quality filter
    hi_52w = close.rolling(252).max().iloc[-1]
    pct_from_high = (latest / hi_52w - 1) * 100
    if pct_from_high < -10:
        return None

    # ATR Stop
    atr = ta.volatility.AverageTrueRange(
        df["High"], df["Low"], close, 14
    ).average_true_range().iloc[-1]

    stop = latest - (1.5 * atr)
    risk = latest - stop

    # Composite Score
    score = (
        rs_outperformance * 0.4 +
        oi_pct * 0.3 +
        vol_ratio * 10 * 0.2 +
        (ret_1m - ret_6m) * 0.1
    )

    return {
        "Symbol": symbol,
        "LTP": round(latest, 2),
        "1M %": round(ret_1m, 2),
        "RS Outperformance %": round(rs_outperformance, 2),
        "OI % Change": round(oi_pct, 2),
        "Volume Ratio": round(vol_ratio, 2),
        "RSI": round(rsi, 2),
        "ATR": round(atr, 2),
        "Suggested Stop": round(stop, 2),
        "Risk per Share (₹)": round(risk, 2),
        "Score": round(score, 2)
    }

# ─────────────────────────────────────────────
# RUN SCAN
# ─────────────────────────────────────────────
if st.button("🚀 Run Scan"):

    if not symbols:
        st.warning("No symbols selected.")
        st.stop()

    bhavcopies = get_recent_bhavcopies()
    nifty_df = get_nifty()

    results = []
    progress = st.progress(0)

    for i, sym in enumerate(symbols):
        out = scan_stock(sym, bhavcopies, nifty_df)
        if out:
            results.append(out)
        progress.progress((i + 1) / len(symbols))
        time.sleep(0.1)

    progress.empty()

    if results:
        df = pd.DataFrame(results)
        df.sort_values("Score", ascending=False, inplace=True)
        df.reset_index(drop=True, inplace=True)
        df.index += 1

        df["Position Size (Qty)"] = (risk_amount / df["Risk per Share (₹)"]).astype(int)

        st.success(f"{len(df)} High Probability Futures Setups Found")
        st.dataframe(df, use_container_width=True)
    else:
        st.warning("No high-probability setups found.")
