"""
F&O Momentum Scanner — 5-Filter System (OI-based version)
Uses recent NSE bhavcopies for OI trend + yfinance for price/volume

Filters:
① Price > 200 DMA
② 1M return > 6M return
③ OI rising over last N sessions AND price rising today
④ RSI(14) 55–70
⑤ Volume > multiplier × 20-day avg AND up-day

Install:
  pip install streamlit yfinance pandas numpy ta requests

Run after market close.
"""

import time
from datetime import date, timedelta
import zipfile
import io
import requests
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import ta
import datetime

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="F&O Momentum Scanner (OI Version)",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stSidebar"] { background: #0f1117; }
  .block-container { padding-top: 1.5rem; }
  div[data-testid="metric-container"] {
    background: #1e2130; border: 1px solid #2d3250; border-radius: 10px; padding: 14px 20px;
  }
  .header-bar {
    background: linear-gradient(90deg, #1e40af, #7c3aed);
    border-radius: 10px; padding: 18px 24px; margin-bottom: 1.5rem;
  }
  .header-bar h1 { color: white; margin: 0; font-size: 1.6rem; }
  .header-bar p  { color: #cbd5e1; margin: 4px 0 0; font-size: 0.9rem; }
  .results-table { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
  .results-table th { background: #1e2130; color: #94a3b8; padding: 10px 14px; text-align: left; border-bottom: 1px solid #2d3250; }
  .results-table td { padding: 10px 14px; border-bottom: 1px solid #1e2130; color: #e2e8f0; }
  .results-table tr:hover td { background: #1e2130; }
  .badge-green { background: #16a34a22; color: #4ade80; border: 1px solid #16a34a55; padding: 2px 8px; border-radius: 4px; font-size: 0.82rem; }
  .badge-blue  { background: #2563eb22; color: #93c5fd; border: 1px solid #2563eb55; padding: 2px 8px; border-radius: 4px; font-size: 0.82rem; }
  .badge-gold  { background: #ca8a0422; color: #fbbf24; border: 1px solid #ca8a0455; padding: 2px 8px; border-radius: 4px; font-size: 0.82rem; }
  .chip { display: inline-block; background: #1e2130; border: 1px solid #2d3250; border-radius: 20px; padding: 4px 12px; font-size: 0.8rem; color: #94a3b8; margin: 2px; }
  .chip-active { border-color: #4f46e5; color: #a5b4fc; background: #1e1b4b; }
  .info-box { background: #1e2130; border: 1px solid #2d3250; border-radius: 10px; padding: 16px 20px; margin: 12px 0; font-size: 0.85rem; color: #94a3b8; line-height: 1.8; }
  .obv-note { background: #1e1b4b; border: 1px solid #4f46e5; border-radius: 8px; padding: 12px 16px; font-size: 0.82rem; color: #a5b4fc; margin-bottom: 16px; }
  .scan-label { font-size: 0.82rem; color: #64748b; margin-bottom: 2px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
for key, default in {
    "scan_results": None,
    "last_scan_time": None,
    "scan_running": False,
    "errors": [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ─────────────────────────────────────────────
# NSE HELPERS
# ─────────────────────────────────────────────
def get_nse_session():
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0", "Referer": "https://www.nseindia.com"})
    try:
        s.get("https://www.nseindia.com", timeout=5)
    except:
        pass
    return s

@st.cache_data(ttl=14400, show_spinner=False)
def get_recent_bhavcopies(num_days=5):
    s = get_nse_session()
    bhav = {}
    today = date.today()
    count = 0
    i = 0
    while count < num_days and i < 15:
        d = today - timedelta(days=i)
        url = f"https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{d.strftime('%Y%m%d')}_F_0000.csv.zip"
        try:
            r = s.get(url, timeout=10)
            if r.status_code == 200:
                with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                    bhav[d] = pd.read_csv(z.open(z.namelist()[0]), low_memory=False)
                    count += 1
        except:
            pass
        i += 1
    return bhav

def get_oi_history(symbol, bhavcopies):
    oi = {}
    for d, df in sorted(bhavcopies.items()):
        fut = df[(df.get('TckrSymb', '') == symbol) & (df.get('FinInstrmTp', '') == 'STF')]
        oi[d] = fut['OpnIntrst'].sum() if 'OpnIntrst' in fut.columns else 0
    return oi

def oi_rising_consecutive(oi_hist, lookback=3):
    dates = sorted(oi_hist.keys(), reverse=True)[:lookback]
    if len(dates) < lookback:
        return False
    values = [oi_hist[d] for d in dates]
    return all(values[i] > values[i+1] for i in range(lookback-1))

@st.cache_data(ttl=14400, show_spinner=False)
def get_fo_symbols():
    s = get_nse_session()
    today = date.today()
    df = pd.DataFrame()
    for i in range(7):
        d = today - timedelta(days=i)
        url = f"https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{d.strftime('%Y%m%d')}_F_0000.csv.zip"
        try:
            r = s.get(url, timeout=10)
            if r.status_code == 200:
                with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                    df = pd.read_csv(z.open(z.namelist()[0]), low_memory=False)
                    break
        except:
            continue

    if df.empty:
        fallback = [
            "RELIANCE","TCS","HDFCBANK","ICICIBANK","INFY","SBIN","BHARTIARTL","ITC",
            "LT","KOTAKBANK","AXISBANK","BAJFINANCE","TATAMOTORS","SUNPHARMA","TITAN",
            "MARUTI","ULTRACEMCO","ADANIENT","ZOMATO","HAL","BEL","IRFC","PFC","RECLTD","SAIL"
        ]
        return sorted(fallback)

    sym_col = next((c for c in df.columns if 'symbol' in c.lower() or 'tckr' in c.lower()), None)
    if sym_col is None:
        return sorted(["RELIANCE","TCS","HDFCBANK"])

    stocks = df[df.get('FinInstrmTp','') == 'STF'][sym_col].dropna().str.strip().str.upper().unique().tolist()

    exclude = ["NIFTY","BANKNIFTY","FINNIFTY","MIDCPNIFTY","NIFTYNXT50","SENSEX"]
    filtered = [s for s in stocks if not any(e in s for e in exclude)]

    return sorted(filtered) if filtered else sorted(fallback)

ALL_FO_SYMBOLS = get_fo_symbols()

def to_yf(sym: str) -> str:
    return sym + ".NS"

# ─────────────────────────────────────────────
# DATA FETCH
# ─────────────────────────────────────────────
@st.cache_data(ttl=21600, show_spinner=False)
def fetch_ohlcv(symbol: str) -> pd.DataFrame:
    ticker = yf.Ticker(to_yf(symbol))
    df = ticker.history(period="14mo", auto_adjust=True)
    if df.empty:
        return pd.DataFrame()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.sort_index(inplace=True)
    return df

# ─────────────────────────────────────────────
# SCAN SINGLE STOCK
# ─────────────────────────────────────────────
def scan_single(symbol: str, rsi_low: float, rsi_high: float,
                vol_mult: float, oi_lookback: int, bhavcopies: dict):
    try:
        df = fetch_ohlcv(symbol)
        if df.empty or len(df) < 210:
            return None

        close  = df["Close"]
        volume = df["Volume"]
        latest = float(close.iloc[-1])

        # ① Price > 200 DMA
        ma200 = float(close.rolling(200).mean().iloc[-1])
        if pd.isna(ma200) or latest <= ma200:
            return None

        # ② 1M return > 6M return
        if len(close) < 130:
            return None
        ret_1m = (latest / float(close.iloc[-22]) - 1) * 100
        ret_6m = (latest / float(close.iloc[-130]) - 1) * 100
        if ret_1m <= ret_6m:
            return None

        # ③ OI rising consecutive + price up today
        oi_hist = get_oi_history(symbol, bhavcopies)
        oi_rising_ok = oi_rising_consecutive(oi_hist, oi_lookback)
        price_up_today = latest > float(close.iloc[-2])
        if not (oi_rising_ok and price_up_today):
            return None

        # ④ RSI 55–70
        rsi = ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-1]
        if pd.isna(rsi) or not (rsi_low <= rsi <= rsi_high):
            return None

        # ⑤ Volume > mult × avg AND up-day
        vol_20avg = float(volume.iloc[-21:-1].mean())
        latest_vol = float(volume.iloc[-1])
        vol_ratio = latest_vol / vol_20avg if vol_20avg > 0 else 0
        if vol_ratio < vol_mult or not price_up_today:
            return None

        # Stats
        ma20 = float(close.rolling(20).mean().iloc[-1])
        ma50 = float(close.rolling(50).mean().iloc[-1])
        hi_52w = float(close.rolling(252).max().iloc[-1])

        oi_dates = sorted(oi_hist.keys(), reverse=True)[:oi_lookback]
        oi_vals = [oi_hist[d] for d in oi_dates]
        oi_slope = round(np.polyfit(range(len(oi_vals)), oi_vals, 1)[0], 0) if len(oi_vals) >= 2 else 0

        return {
            "Symbol": symbol,
            "LTP": round(latest, 2),
            "20 DMA": round(ma20, 2),
            "50 DMA": round(ma50, 2),
            "200 DMA": round(ma200, 2),
            "% vs 200 DMA": round((latest / ma200 - 1) * 100, 2),
            "1M Return %": round(ret_1m, 2),
            "6M Return %": round(ret_6m, 2),
            "Momentum Gap": round(ret_1m - ret_6m, 2),
            "RSI (14)": round(rsi, 2),
            "OI Slope": oi_slope,
            "Vol Ratio": round(vol_ratio, 2),
            "52W High": round(hi_52w, 2),
            "% from 52W High": round((latest / hi_52w - 1) * 100, 2),
        }

    except Exception as e:
        return {"_error": symbol, "_msg": str(e)}

# ─────────────────────────────────────────────
# RUN SCAN
# ─────────────────────────────────────────────
def run_scan(symbols, rsi_low, rsi_high, vol_mult, oi_lookback, bhavcopies, progress_cb=None):
    results, errors = [], []
    total = len(symbols)

    for idx, sym in enumerate(symbols):
        if progress_cb:
            progress_cb(idx + 1, total, sym)

        out = scan_single(sym, rsi_low, rsi_high, vol_mult, oi_lookback, bhavcopies)

        if out is None:
            pass
        elif "_error" in out:
            errors.append(f"{out['_error']}: {out['_msg']}")
        else:
            results.append(out)

        time.sleep(0.12)  # gentle delay

    if not results:
        return pd.DataFrame(), errors

    df = pd.DataFrame(results)
    df.sort_values("Momentum Gap", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.index += 1
    return df, errors

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
def render_sidebar():
    st.sidebar.markdown("## ⚙️ Settings")

    st.sidebar.markdown("### 🎛️ Filter Parameters")

    rsi_range = st.sidebar.slider("RSI Range (Filter ④)", 30, 90, (55, 70))
    vol_mult = st.sidebar.slider("Volume Multiplier (Filter ⑤)", 1.0, 4.0, 1.5, 0.1)
    oi_lookback = st.sidebar.slider("OI Consecutive Rising Days (Filter ③)", 2, 5, 3)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 Stock Universe")
    universe = st.sidebar.radio("Scan which stocks?", ["All NSE F&O", "Custom List"], index=0)

    custom_syms = []
    if universe == "Custom List":
        raw = st.sidebar.text_area("Symbols (comma-separated, NO .NS)", placeholder="RELIANCE,TCS,HDFCBANK", height=100)
        custom_syms = [s.strip().upper() for s in raw.split(",") if s.strip()]

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ✅ Active Filters")
    for f in [
        "① Price > 200 DMA",
        "② 1M return > 6M return",
        f"③ OI rising ({oi_lookback} consecutive sessions) + Price ↑",
        f"④ RSI {rsi_range[0]}–{rsi_range[1]}",
        f"⑤ Volume > {vol_mult}× 20-day avg on up-day",
    ]:
        st.sidebar.markdown(f'<span class="chip chip-active">{f}</span>', unsafe_allow_html=True)

    return rsi_range, vol_mult, oi_lookback, universe, custom_syms

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    st.markdown("""
    <div class="header-bar">
      <h1>📈 F&O Momentum Scanner — OI Version</h1>
      <p>Fresh long money detection via consecutive OI rise + price action</p>
    </div>
    """, unsafe_allow_html=True)

    rsi_range, vol_mult, oi_lookback, universe_opt, custom_syms = render_sidebar()

    bhavcopies = get_recent_bhavcopies(oi_lookback + 2)  # extra days for safety

    symbols = ALL_FO_SYMBOLS if universe_opt == "All NSE F&O" else custom_syms

    if not symbols:
        st.error("No symbols. Use Custom List or restart.")
        st.stop()

    est_secs = round(len(symbols) * 0.15)
    est_label = f"~{est_secs}s" if est_secs < 60 else f"~{est_secs//60}m {est_secs%60}s"

    col_btn, col_info = st.columns([2, 5])
    with col_btn:
        label = "🔄 Re-scan" if st.session_state.scan_results is not None else "🚀 Run Scan"
        run_now = st.button(label, type="primary", use_container_width=True,
                            disabled=st.session_state.scan_running)
    with col_info:
        st.markdown(f"Scanning <b>{len(symbols)} symbols</b> — est. {est_label}", unsafe_allow_html=True)

    if run_now:
        st.session_state.scan_running = True
        fetch_ohlcv.clear()

        progress_bar = st.progress(0)
        status_text = st.empty()

        def progress_cb(done, total, sym):
            progress_bar.progress(done / total)
            status_text.markdown(f"Scanning {done}/{total} — {sym}")

        results, errors = run_scan(
            symbols, rsi_range[0], rsi_range[1],
            vol_mult, oi_lookback, bhavcopies, progress_cb
        )

        st.session_state.scan_results = results
        st.session_state.last_scan_time = datetime.datetime.now()
        st.session_state.errors = errors
        st.session_state.scan_running = False
        progress_bar.empty()
        status_text.empty()
        st.rerun()

    # Display results (copy your existing display code here)
    if st.session_state.scan_results is not None:
        results = st.session_state.scan_results
        if results.empty:
            st.warning("No stocks passed all filters today.")
        else:
            st.success(f"{len(results)} stocks passed all filters.")
            st.dataframe(results)

        if st.session_state.errors:
            st.expander("Errors").write(st.session_state.errors)

if __name__ == "__main__":
    main()