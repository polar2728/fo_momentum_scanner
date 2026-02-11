import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
import zipfile
import datetime
import concurrent.futures
import ta

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="F&O Positional Long Scanner",
    page_icon="📈",
    layout="wide"
)

RISK_PER_TRADE = 10000  # ₹ risk per trade default

# ─────────────────────────────────────────────
# SESSION
# ─────────────────────────────────────────────

def get_session():
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://www.nseindia.com"
    })
    return s

# ─────────────────────────────────────────────
# 1️⃣ F&O UNIVERSE (ROBUST)
# ─────────────────────────────────────────────

@st.cache_data(ttl=86400)
def get_fo_universe():
    """
    Cloud-safe F&O universe builder.
    Uses NSE equity master + F&O lot file fallback.
    """

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0"
    })

    try:
        # Step 1: Get all NSE equity symbols (very stable endpoint)
        eq_url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
        eq_r = session.get(eq_url, timeout=10)

        if eq_r.status_code != 200:
            return [], {}

        eq_df = pd.read_csv(io.StringIO(eq_r.text))
        eq_df.columns = eq_df.columns.str.strip().str.upper()

        symbol_col = next((c for c in eq_df.columns if "SYMBOL" in c), None)
        if symbol_col is None:
            return [], {}

        all_symbols = eq_df[symbol_col].astype(str).str.strip().str.upper().tolist()

        # Step 2: Try fetching F&O lot file
        lot_url = "https://archives.nseindia.com/content/fo/fo_mktlots.csv"
        lot_r = session.get(lot_url, timeout=10)

        lot_map = {}
        fo_symbols = []

        if lot_r.status_code == 200:
            lot_df = pd.read_csv(io.StringIO(lot_r.text))
            lot_df.columns = lot_df.columns.str.strip().str.upper()

            sym_col = next((c for c in lot_df.columns if "SYMB" in c), None)
            lot_col = next((c for c in lot_df.columns if "LOT" in c), None)

            if sym_col:
                fo_symbols = (
                    lot_df[sym_col]
                    .astype(str)
                    .str.strip()
                    .str.upper()
                    .unique()
                    .tolist()
                )

                if lot_col:
                    for _, row in lot_df.iterrows():
                        sym = str(row[sym_col]).strip().upper()
                        try:
                            lot_map[sym] = int(row[lot_col])
                        except:
                            continue

        # If lot file blocked → fallback to top 150 liquid stocks
        if not fo_symbols:
            fo_symbols = all_symbols[:150]

        # Remove indices
        exclude = ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"]
        fo_symbols = [s for s in fo_symbols if s not in exclude]

        return sorted(fo_symbols), lot_map

    except Exception:
        return [], {}


# ─────────────────────────────────────────────
# 2️⃣ PRICE FETCH (NSE HISTORICAL API)
# ─────────────────────────────────────────────

def fetch_price(symbol):
    try:
        url = f"https://www.nseindia.com/api/historical/cm/equity?symbol={symbol}&series=[%22EQ%22]&from=01-01-2023&to={datetime.date.today().strftime('%d-%m-%Y')}"
        session = get_session()
        r = session.get(url, timeout=10)

        if r.status_code != 200:
            return None

        data = r.json()
        if "data" not in data:
            return None

        df = pd.DataFrame(data["data"])
        df["CH_TIMESTAMP"] = pd.to_datetime(df["CH_TIMESTAMP"])
        df.sort_values("CH_TIMESTAMP", inplace=True)

        df.rename(columns={
            "CH_CLOSE_PRICE": "Close",
            "CH_OPEN_PRICE": "Open",
            "CH_TRADE_HIGH_PRICE": "High",
            "CH_TRADE_LOW_PRICE": "Low",
            "CH_TOT_TRADED_QTY": "Volume"
        }, inplace=True)

        return df[["CH_TIMESTAMP","Open","High","Low","Close","Volume"]]

    except:
        return None

# ─────────────────────────────────────────────
# 3️⃣ OI DATA (Bhavcopy)
# ─────────────────────────────────────────────

def get_recent_bhavcopies(days=5):
    session = get_session()
    bhav = {}
    today = datetime.date.today()

    for i in range(days):
        d = today - datetime.timedelta(days=i)
        url = f"https://archives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{d.strftime('%Y%m%d')}_F_0000.csv.zip"
        try:
            r = session.get(url, timeout=10)
            if r.status_code == 200:
                with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                    df = pd.read_csv(z.open(z.namelist()[0]))
                    bhav[d] = df
        except:
            continue

    return bhav

def get_oi_pct_change(symbol, bhavcopies):
    values = []

    for d in sorted(bhavcopies.keys(), reverse=True)[:3]:
        df = bhavcopies[d]
        fut = df[df['SYMBOL'] == symbol]
        if 'OPENINTEREST' in df.columns:
            values.append(fut['OPENINTEREST'].sum())

    if len(values) < 2:
        return 0

    return ((values[0] - values[-1]) / values[-1]) * 100 if values[-1] != 0 else 0

# ─────────────────────────────────────────────
# 4️⃣ SCAN LOGIC
# ─────────────────────────────────────────────

def scan_stock(symbol, lot_map, bhavcopies):
    df = fetch_price(symbol)
    if df is None or len(df) < 200:
        return None

    close = df["Close"]

    ma200 = close.rolling(200).mean().iloc[-1]
    if close.iloc[-1] <= ma200:
        return None

    ret_1m = close.pct_change(22).iloc[-1] * 100
    ret_3m = close.pct_change(66).iloc[-1] * 100

    if ret_1m <= ret_3m:
        return None

    rsi = ta.momentum.RSIIndicator(close).rsi().iloc[-1]
    if not 55 <= rsi <= 70:
        return None

    oi_pct = get_oi_pct_change(symbol, bhavcopies)
    if oi_pct < 5:
        return None

    atr = ta.volatility.AverageTrueRange(
        high=df["High"],
        low=df["Low"],
        close=df["Close"]
    ).average_true_range().iloc[-1]

    stop = close.iloc[-1] - 1.5 * atr
    risk = close.iloc[-1] - stop

    lot = lot_map.get(symbol, 1)
    position_size = int((RISK_PER_TRADE / risk) / lot) * lot if risk > 0 else 0

    return {
        "Symbol": symbol,
        "LTP": round(close.iloc[-1],2),
        "1M %": round(ret_1m,2),
        "3M %": round(ret_3m,2),
        "RSI": round(rsi,1),
        "OI % Change": round(oi_pct,2),
        "ATR Stop": round(stop,2),
        "Lot Size": lot,
        "Qty (₹10k risk)": position_size
    }

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

st.title("📈 F&O Positional Long Scanner (2–4 Weeks)")

symbols, lot_map = get_fo_universe()

if not symbols:
    st.error("F&O universe fetch failed.")
    st.stop()

bhavcopies = get_recent_bhavcopies(4)

if st.button("Run Scan"):

    results = []

    progress = st.progress(0)

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(scan_stock, s, lot_map, bhavcopies): s for s in symbols[:120]}

        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            res = future.result()
            if res:
                results.append(res)
            progress.progress((i+1)/len(futures))

    if results:
        df = pd.DataFrame(results)
        df.sort_values("1M %", ascending=False, inplace=True)
        st.success(f"{len(df)} Long Candidates Found")
        st.dataframe(df)
    else:
        st.warning("No high-probability setups today.")
