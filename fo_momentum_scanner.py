# fo_positional_scanner.py
# Institutional Positional F&O Futures Scanner
# Stable Version – No NSE JSON scraping

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import zipfile
import io
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

st.set_page_config(page_title="F&O Positional Scanner", layout="wide")

# =========================================================
# CONFIG
# =========================================================

RS_LOOKBACK = 60
ATR_PERIOD = 14
MIN_OI_PCT = 5
MAX_WORKERS = 6

# =========================================================
# UTILITY FUNCTIONS
# =========================================================

def get_last_trading_day():
    today = datetime.date.today()
    return today - datetime.timedelta(days=1)

# ---------------------------------------------------------

@st.cache_data(ttl=3600)
def fetch_fo_bhavcopy():
    """
    Fetch F&O Bhavcopy ZIP (safe method, no scraping).
    """
    date = get_last_trading_day()
    date_str = date.strftime("%d%b%Y").upper()

    url = f"https://archives.nseindia.com/content/historical/DERIVATIVES/{date.year}/{date.strftime('%b').upper()}/fo{date.strftime('%d%b%Y').upper()}bhav.csv.zip"

    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return pd.DataFrame()

        z = zipfile.ZipFile(io.BytesIO(r.content))
        df = pd.read_csv(z.open(z.namelist()[0]))

        df.columns = df.columns.str.strip().str.upper()
        return df

    except:
        return pd.DataFrame()

# ---------------------------------------------------------

def extract_stock_universe(df):
    """
    Extract only stock futures (exclude index)
    """
    if df.empty:
        return []

    df = df[df["INSTRUMENT"] == "FUTSTK"]
    symbols = df["SYMBOL"].unique().tolist()
    return symbols

# ---------------------------------------------------------

def calculate_rs(stock_close, nifty_close):
    stock_return = stock_close.pct_change(RS_LOOKBACK)
    nifty_return = nifty_close.pct_change(RS_LOOKBACK)
    return stock_return.iloc[-1] - nifty_return.iloc[-1]

# ---------------------------------------------------------

def calculate_atr(df):
    df["H-L"] = df["High"] - df["Low"]
    df["H-PC"] = abs(df["High"] - df["Close"].shift())
    df["L-PC"] = abs(df["Low"] - df["Close"].shift())
    tr = df[["H-L","H-PC","L-PC"]].max(axis=1)
    atr = tr.rolling(ATR_PERIOD).mean()
    return atr.iloc[-1]

# ---------------------------------------------------------

def fetch_price_data(symbol):
    try:
        df = yf.download(symbol + ".NS", period="4mo", progress=False)
        if len(df) < RS_LOOKBACK + 5:
            return None

        atr = calculate_atr(df)
        last_close = df["Close"].iloc[-1]
        return df, last_close, atr
    except:
        return None

# =========================================================
# UI
# =========================================================

st.title("🔥 Institutional F&O Positional Futures Scanner")

capital = st.number_input("Capital (₹)", value=1000000)
risk_pct = st.slider("Risk per Trade (%)", 0.5, 5.0, 1.0)

if st.button("Run Scan"):

    with st.spinner("Fetching F&O Data..."):

        fo_df = fetch_fo_bhavcopy()

        if fo_df.empty:
            st.error("F&O universe fetch failed.")
            st.stop()

        symbols = extract_stock_universe(fo_df)

        if not symbols:
            st.error("No F&O stock futures found.")
            st.stop()

        nifty = yf.download("^NSEI", period="4mo", progress=False)

        results = []

        # Multi-threaded price fetch
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(fetch_price_data, sym): sym for sym in symbols}

            for future in as_completed(futures):
                sym = futures[future]
                data = future.result()

                if data is None:
                    continue

                df_price, close, atr = data

                # Get OI data
                row = fo_df[(fo_df["SYMBOL"] == sym) & (fo_df["INSTRUMENT"] == "FUTSTK")]

                if row.empty:
                    continue

                oi_change_pct = row["CHG_IN_OI"].values[0] / row["OPEN_INT"].values[0] * 100
                price_change_pct = row["SETTLE_PR"].values[0] / row["OPEN"].values[0] - 1

                # Long buildup condition
                if oi_change_pct > MIN_OI_PCT and price_change_pct > 0:

                    rs = calculate_rs(df_price["Close"], nifty["Close"])

                    if rs > 0:
                        stop = close - (2 * atr)
                        risk_amt = capital * (risk_pct / 100)
                        qty = int(risk_amt / (close - stop)) if (close - stop) > 0 else 0

                        results.append({
                            "Symbol": sym,
                            "Close": round(close,2),
                            "OI %": round(oi_change_pct,2),
                            "RS vs Nifty": round(rs,3),
                            "ATR": round(atr,2),
                            "Suggested Stop": round(stop,2),
                            "Qty": qty
                        })

        if not results:
            st.warning("No high-probability long setups found.")
        else:
            df_final = pd.DataFrame(results)
            df_final = df_final.sort_values("RS vs Nifty", ascending=False)

            st.success(f"{len(df_final)} High-Probability Futures Found")
            st.dataframe(df_final, use_container_width=True)
