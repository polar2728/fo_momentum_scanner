import streamlit as st
import pandas as pd
import numpy as np
import requests
import zipfile
import io
import datetime
from datetime import date, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import ta
import time

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(layout="wide", page_title="Institutional F&O Scanner")

st.title("🔥 Institutional F&O Positional Scanner")
st.caption("NSE Only | Multi-threaded | 2–4 Week Futures Long Engine")

# ─────────────────────────────────────────────
# SIDEBAR RISK SETTINGS
# ─────────────────────────────────────────────
st.sidebar.header("Risk Settings")

capital = st.sidebar.number_input("Capital (₹)", value=1000000)
risk_pct = st.sidebar.slider("Risk Per Trade (%)", 0.5, 3.0, 1.0)
risk_amt = capital * (risk_pct / 100)

st.sidebar.markdown(f"Risk per trade: ₹{risk_amt:,.0f}")

# ─────────────────────────────────────────────
# NSE SESSION
# ─────────────────────────────────────────────
def get_session():
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://www.nseindia.com"
    })
    s.get("https://www.nseindia.com")
    return s

# ─────────────────────────────────────────────
# AUTO FETCH F&O UNIVERSE
# ─────────────────────────────────────────────
@st.cache_data(ttl=86400)
def get_fo_universe():
    """
    Fetch F&O eligible securities using NSE official CSV.
    This is stable and cloud-safe.
    """

    session = get_session()

    # Official NSE F&O securities list (public CSV)
    url = "https://archives.nseindia.com/content/fo/fo_mktlots.csv"

    try:
        r = session.get(url, timeout=10)

        if r.status_code != 200:
            st.error("Unable to fetch F&O list from NSE.")
            return []

        df = pd.read_csv(io.StringIO(r.text))

        # Clean column names
        df.columns = df.columns.str.strip()

        # Remove index rows
        df = df[~df['SYMBOL'].str.contains("NIFTY|BANKNIFTY", na=False)]

        symbols = sorted(df['SYMBOL'].unique())

        return symbols

    except Exception as e:
        st.error("F&O universe fetch failed.")
        return []


# ─────────────────────────────────────────────
# GET CASH BHAVCOPY
# ─────────────────────────────────────────────
def get_cash_bhav(date_obj, session):
    url = f"https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_{date_obj.strftime('%Y%m%d')}_F_0000.csv.zip"
    r = session.get(url, timeout=10)
    if r.status_code != 200:
        return None
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        df = pd.read_csv(z.open(z.namelist()[0]))
    return df

# ─────────────────────────────────────────────
# GET FO BHAVCOPY
# ─────────────────────────────────────────────
def get_fo_bhav(date_obj, session):
    url = f"https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{date_obj.strftime('%Y%m%d')}_F_0000.csv.zip"
    r = session.get(url, timeout=10)
    if r.status_code != 200:
        return None
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        df = pd.read_csv(z.open(z.namelist()[0])
)
    return df

# ─────────────────────────────────────────────
# GET LAST N DAYS DATA
# ─────────────────────────────────────────────
@st.cache_data(ttl=14400)
def get_recent_data(days=20):
    session = get_session()
    cash_data = {}
    fo_data = {}

    today = date.today()
    count = 0
    i = 0

    while count < days and i < days + 10:
        d = today - timedelta(days=i)
        cash = get_cash_bhav(d, session)
        fo = get_fo_bhav(d, session)

        if cash is not None and fo is not None:
            cash_data[d] = cash
            fo_data[d] = fo
            count += 1

        i += 1

    return cash_data, fo_data

# ─────────────────────────────────────────────
# EXPIRY WEEK FILTER
# ─────────────────────────────────────────────
def is_expiry_week():
    today = datetime.date.today()
    last_day = today.replace(day=28) + datetime.timedelta(days=4)
    last_thursday = last_day - datetime.timedelta(days=last_day.weekday() - 3)
    return (last_thursday - today).days <= 3

# ─────────────────────────────────────────────
# SCAN STOCK LOGIC
# ─────────────────────────────────────────────
def scan_symbol(symbol, cash_data, fo_data, nifty_returns):

    closes = []
    volumes = []
    delivery_ratios = []
    oi_vals = []

    for d in sorted(cash_data.keys()):
        cash = cash_data[d]
        fo = fo_data[d]

        row = cash[cash['SYMBOL'] == symbol]
        if row.empty:
            continue

        closes.append(row['CLOSE_PRICE'].values[0])
        volumes.append(row['TTL_TRD_QNTY'].values[0])
        delivery = row['DELIV_QTY'].values[0]
        delivery_ratios.append(delivery / row['TTL_TRD_QNTY'].values[0])

        fut = fo[(fo['TckrSymb'] == symbol) & (fo['FinInstrmTp'] == 'STF')]
        if not fut.empty:
            nearest = fut['XpryDt'].min()
            fut = fut[fut['XpryDt'] == nearest]
            oi_vals.append(fut['OpnIntrst'].sum())

    if len(closes) < 15:
        return None

    close_series = pd.Series(closes)
    vol_series = pd.Series(volumes)

    ma20 = close_series.rolling(20).mean().iloc[-1]
    ma50 = close_series.rolling(15).mean().iloc[-1]

    if not (close_series.iloc[-1] > ma20 > ma50):
        return None

    ret_1m = (close_series.iloc[-1] / close_series.iloc[-15] - 1) * 100
    if ret_1m < nifty_returns:
        return None

    # OI %
    if len(oi_vals) < 5:
        return None

    oi_pct = ((oi_vals[-1] - oi_vals[-5]) / oi_vals[-5]) * 100
    if oi_pct < 5:
        return None

    # FII/DII proxy (delivery spike)
    delivery_avg = np.mean(delivery_ratios[:-5])
    delivery_now = np.mean(delivery_ratios[-5:])
    if delivery_now <= delivery_avg:
        return None

    # ATR
    atr = ta.volatility.AverageTrueRange(
        high=close_series,
        low=close_series,
        close=close_series,
        window=14
    ).average_true_range().iloc[-1]

    stop = close_series.iloc[-1] - 1.5 * atr
    risk = close_series.iloc[-1] - stop

    score = ret_1m * 0.4 + oi_pct * 0.4 + (delivery_now - delivery_avg) * 100 * 0.2

    return {
        "Symbol": symbol,
        "LTP": round(close_series.iloc[-1],2),
        "1M %": round(ret_1m,2),
        "OI %": round(oi_pct,2),
        "Delivery Spike %": round((delivery_now - delivery_avg)*100,2),
        "Stop": round(stop,2),
        "Risk ₹": round(risk,2),
        "Score": round(score,2)
    }

# ─────────────────────────────────────────────
# RUN SCAN
# ─────────────────────────────────────────────
if st.button("🚀 Run Institutional Scan"):

    if is_expiry_week():
        st.warning("Expiry week distortion. Scan paused.")
        st.stop()

    symbols = get_fo_universe()
    cash_data, fo_data = get_recent_data()

    # Nifty return proxy
    nifty_symbol = "NIFTY"
    nifty_closes = []
    for d in sorted(cash_data.keys()):
        df = cash_data[d]
        row = df[df['SYMBOL'] == nifty_symbol]
        if not row.empty:
            nifty_closes.append(row['CLOSE_PRICE'].values[0])

    if len(nifty_closes) >= 15:
        nifty_returns = (nifty_closes[-1]/nifty_closes[-15]-1)*100
    else:
        nifty_returns = 0

    results = []
    progress = st.progress(0)

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(scan_symbol, sym, cash_data, fo_data, nifty_returns): sym
            for sym in symbols
        }

        for i, future in enumerate(as_completed(futures)):
            res = future.result()
            if res:
                results.append(res)
            progress.progress((i+1)/len(futures))

    progress.empty()

    if results:
        df = pd.DataFrame(results).sort_values("Score", ascending=False)
        df.reset_index(drop=True, inplace=True)
        df.index += 1

        df["Position Qty"] = (risk_amt / df["Risk ₹"]).astype(int)

        st.success(f"{len(df)} Institutional Long Setups Found")
        st.dataframe(df, use_container_width=True)
    else:
        st.warning("No strong setups today.")
