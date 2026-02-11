"""
F&O Momentum Scanner — 5-Filter System
Updated: Use OI data instead of OBV for Filter 3
Dynamic NSE F&O symbols from official bhavcopy ZIP
Uses yfinance for OHLCV + requests for NSE bhavcopy

Install:
  pip install streamlit yfinance pandas numpy ta requests

Run after market close for fresh data.
"""

import time
import datetime
import zipfile
import io
import requests
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import ta

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="F&O Momentum Scanner",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CSS (same as your original)
# ─────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stSidebar"] { background: #0f1117; }
  .block-container { padding-top: 1.5rem; }

  div[data-testid="metric-container"] {
    background: #1e2130; border: 1px solid #2d3250;
    border-radius: 10px; padding: 14px 20px;
  }

  .header-bar {
    background: linear-gradient(90deg, #1e40af, #7c3aed);
    border-radius: 10px; padding: 18px 24px; margin-bottom: 1.5rem;
  }
  .header-bar h1 { color: white; margin: 0; font-size: 1.6rem; }
  .header-bar p  { color: #cbd5e1; margin: 4px 0 0; font-size: 0.9rem; }

  .results-table { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
  .results-table th {
    background: #1e2130; color: #94a3b8; padding: 10px 14px;
    text-align: left; border-bottom: 1px solid #2d3250;
  }
  .results-table td { padding: 10px 14px; border-bottom: 1px solid #1e2130; color: #e2e8f0; }
  .results-table tr:hover td { background: #1e2130; }

  .badge-green { background: #16a34a22; color: #4ade80; border: 1px solid #16a34a55;
                 padding: 2px 8px; border-radius: 4px; font-size: 0.82rem; }
  .badge-blue  { background: #2563eb22; color: #93c5fd; border: 1px solid #2563eb55;
                 padding: 2px 8px; border-radius: 4px; font-size: 0.82rem; }
  .badge-gold  { background: #ca8a0422; color: #fbbf24; border: 1px solid #ca8a0455;
                 padding: 2px 8px; border-radius: 4px; font-size: 0.82rem; }

  .chip { display: inline-block; background: #1e2130; border: 1px solid #2d3250;
          border-radius: 20px; padding: 4px 12px; font-size: 0.8rem;
          color: #94a3b8; margin: 2px; }
  .chip-active { border-color: #4f46e5; color: #a5b4fc; background: #1e1b4b; }

  .info-box {
    background: #1e2130; border: 1px solid #2d3250; border-radius: 10px;
    padding: 16px 20px; margin: 12px 0; font-size: 0.85rem; color: #94a3b8;
    line-height: 1.8;
  }

  .obv-note {
    background: #1e1b4b; border: 1px solid #4f46e5; border-radius: 8px;
    padding: 12px 16px; font-size: 0.82rem; color: #a5b4fc; margin-bottom: 16px;
  }

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
# NSE SESSION
# ─────────────────────────────────────────────
def get_nse_session():
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0", "Referer": "https://www.nseindia.com"})
    try:
        s.get("https://www.nseindia.com", timeout=5)
    except:
        pass
    return s

# ─────────────────────────────────────────────
# FETCH RECENT BHAVCOPIES FOR OI (cached - run once)
# ─────────────────────────────────────────────
@st.cache_data(ttl=3600 * 4, show_spinner=False)
def get_recent_bhavcopies(num_days=4):
    s = get_nse_session()
    bhavcopies = {}
    today = date.today()
    count = 0
    i = 0
    while count < num_days and i < 10:  # max 10 attempts to skip non-trading days
        d = today - timedelta(days=i)
        url = f"https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{d.strftime('%Y%m%d')}_F_0000.csv.zip"
        try:
            r = s.get(url, timeout=10)
            if r.status_code == 200:
                with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                    bhavcopies[d] = pd.read_csv(z.open(z.namelist()[0]))
                    count += 1
        except:
            pass
        i += 1
    if len(bhavcopies) < 3:
        st.warning("Not enough recent bhavcopies for OI analysis. Using fallback symbols.")
    return bhavcopies

# ─────────────────────────────────────────────
# GET OI FOR A STOCK OVER DAYS
# ─────────────────────────────────────────────
def get_oi_history(symbol, bhavcopies):
    oi_hist = {}
    for d, df in bhavcopies.items():
        fut_df = df[(df['TckrSymb'] == symbol) & (df['FinInstrmTp'] == 'STF')]
        total_oi = fut_df['OpnIntrst'].sum() if 'OpnIntrst' in fut_df.columns else 0
        oi_hist[d] = total_oi
    return oi_hist

# ─────────────────────────────────────────────
# CHECK IF OI RISING 3 CONSECUTIVE SESSIONS
# ─────────────────────────────────────────────
def oi_rising(oi_hist, lookback=3):
    dates = sorted(oi_hist.keys())
    if len(dates) < lookback:
        return False
    last_oi = [oi_hist[dates[-i-1]] for i in range(lookback)]
    # Check if strictly rising: last > prev > prev_prev
    return all(last_oi[i] > last_oi[i+1] for i in range(lookback-1))

# ─────────────────────────────────────────────
# FETCH EXACT NSE F&O SYMBOLS
# ─────────────────────────────────────────────
@st.cache_data(ttl=3600 * 4, show_spinner=False)
def get_fo_symbols():
    fo_df = download_fo()
    if fo_df.empty:
        st.warning("Fallback to core + common F&O symbols (~50).")
        fallback = [
            "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "SBIN", "BHARTIARTL", "ITC",
            "LT", "KOTAKBANK", "AXISBANK", "BAJFINANCE", "TATAMOTORS", "SUNPHARMA", "TITAN",
            "MARUTI", "ULTRACEMCO", "ADANIENT", "ZOMATO", "HAL", "BEL", "IRFC", "PFC", "RECLTD", "SAIL"
        ]
        return sorted(fallback)

    sym_col = next((c for c in fo_df.columns if 'symbol' in c.lower() or 'tckrsymb' in c.lower()), None)
    if sym_col is None:
        return sorted(["RELIANCE", "TCS", "HDFCBANK"])

    stocks = fo_df[(fo_df['FinInstrmTp'] == 'STF')][sym_col].dropna().str.strip().str.upper().unique().tolist()

    # Exclude index futures
    exclude = ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTYNXT50"]
    filtered = [s for s in stocks if s not in exclude]

    return sorted(filtered)

ALL_FO_SYMBOLS = get_fo_symbols()

def to_yf(sym: str) -> str:
    return sym + ".NS"

# ─────────────────────────────────────────────
# DATA FETCH (OHLCV via yfinance)
# ─────────────────────────────────────────────
@st.cache_data(ttl=3600 * 6, show_spinner=False)
def fetch_ohlcv(symbol: str) -> pd.DataFrame:
    ticker = yf.Ticker(to_yf(symbol))
    df = ticker.history(period="14mo", auto_adjust=True)
    if df.empty:
        return pd.DataFrame()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.sort_index(inplace=True)
    return df

# ─────────────────────────────────────────────
# INDICATORS (RSI only, since OBV removed)
# ─────────────────────────────────────────────
def compute_rsi(close: pd.Series, period: int = 14) -> float:
    rsi_series = ta.momentum.RSIIndicator(close, window=period).rsi()
    if pd.isna(rsi_series.iloc[-1]):
        return np.nan
    return round(float(rsi_series.iloc[-1]), 2)

# ─────────────────────────────────────────────
# SCAN SINGLE STOCK (updated Filter 3: OI instead of OBV)
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

        # Filter 1: Price > 200 DMA
        ma200 = float(close.rolling(200).mean().iloc[-1])
        if pd.isna(ma200) or latest <= ma200:
            return None

        # Filter 2: 1M return > 6M return
        if len(close) < 130:
            return None
        ret_1m = (latest / float(close.iloc[-22]) - 1) * 100
        ret_6m = (latest / float(close.iloc[-130]) - 1) * 100
        if ret_1m <= ret_6m:
            return None

        # Filter 3: OI rising over lookback sessions + price rising
        oi_hist = get_oi_history(symbol, bhavcopies)
        rising_oi = oi_rising(oi_hist, lookback=oi_lookback)
        price_up = latest > float(close.iloc[-2])
        if not (rising_oi and price_up):
            return None

        # Filter 4: RSI in range
        rsi = compute_rsi(close)
        if pd.isna(rsi) or not (rsi_low <= rsi <= rsi_high):
            return None

        # Filter 5: Volume > vol_mult × 20-day avg on up-day
        vol_20avg  = float(volume.iloc[-21:-1].mean())
        latest_vol = float(volume.iloc[-1])
        vol_ratio  = round(latest_vol / vol_20avg, 2) if vol_20avg > 0 else 0.0
        if vol_ratio < vol_mult or not price_up:  # up-day already checked, but reinforce
            return None

        # Collect stats (updated: OI slope instead of OBV slope)
        ma20   = float(close.rolling(20).mean().iloc[-1])
        ma50   = float(close.rolling(50).mean().iloc[-1])
        hi_52w = float(close.rolling(252).max().iloc[-1])

        # OI slope (for display, similar to original OBV slope)
        dates = sorted(oi_hist.keys())
        if len(dates) >= oi_lookback:
            recent_oi = [oi_hist[dates[-i-1]] for i in range(oi_lookback)]
            x = np.arange(len(recent_oi))
            oi_slope = round(float(np.polyfit(x, recent_oi, 1)[0]), 0)
        else:
            oi_slope = 0

        return {
            "Symbol":           symbol,
            "LTP":              round(latest, 2),
            "20 DMA":           round(ma20, 2),
            "50 DMA":           round(ma50, 2),
            "200 DMA":          round(ma200, 2),
            "% vs 200 DMA":     round((latest / ma200 - 1) * 100, 2),
            "1M Return %":      round(ret_1m, 2),
            "6M Return %":      round(ret_6m, 2),
            "Momentum Gap":     round(ret_1m - ret_6m, 2),
            "RSI (14)":         rsi,
            "OI Slope":         oi_slope,  # new
            "Vol Ratio":        vol_ratio,
            "52W High":         round(hi_52w, 2),
            "% from 52W High":  round((latest / hi_52w - 1) * 100, 2),
        }

    except Exception as e:
        return {"_error": symbol, "_msg": str(e)}

# ─────────────────────────────────────────────
# RUN FULL SCAN (updated to pass bhavcopies)
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

        time.sleep(0.08)

    if not results:
        return pd.DataFrame(), errors

    df = pd.DataFrame(results)
    df.sort_values("Momentum Gap", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.index += 1
    return df, errors

# ─────────────────────────────────────────────
# SIDEBAR (updated for OI lookback)
# ─────────────────────────────────────────────
def render_sidebar():
    st.sidebar.markdown("## ⚙️ Settings")

    st.sidebar.markdown("### 🎛️ Filter Parameters")

    rsi_range = st.sidebar.slider(
        "RSI Range (Filter 4)",
        min_value=30, max_value=90, value=(55, 70),
        help="Only include stocks with RSI within this band — avoids overbought entries",
    )
    vol_mult = st.sidebar.slider(
        "Volume Multiplier (Filter 5)",
        min_value=1.0, max_value=4.0, value=1.5, step=0.1,
        help="Latest session volume must exceed this multiple of the 20-day average on an up-day",
    )
    oi_lookback = st.sidebar.slider(
        "OI Lookback Sessions (Filter 3)",
        min_value=3, max_value=5, value=3,
        help="Number of consecutive sessions over which OI must be rising. Higher = stricter.",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 Stock Universe")
    universe = st.sidebar.radio(
        "Scan which stocks?",
        ["All NSE F&O (from latest bhavcopy)", "Custom List"],
        index=0,
    )

    custom_syms = []
    if universe == "Custom List":
        raw = st.sidebar.text_area(
            "Symbols (comma-separated, NO .NS suffix)",
            placeholder="RELIANCE,TCS,HDFCBANK",
            height=100,
        )
        custom_syms = [s.strip().upper() for s in raw.split(",") if s.strip()]

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ✅ Active Filters")
    for f in [
        "① Price > 200 DMA",
        "② 1M return > 6M return",
        f"③ OI rising ({oi_lookback} consecutive sessions) + Price rising",
        f"④ RSI {rsi_range[0]}–{rsi_range[1]}",
        f"⑤ Volume > {vol_mult}× 20-day avg on up-day",
    ]:
        st.sidebar.markdown(
            f'<span class="chip chip-active">{f}</span>',
            unsafe_allow_html=True,
        )

    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style='font-size:0.75rem; color:#475569; line-height:1.7'>
    <b>ℹ️ Why OI?</b><br>
    Open Interest rising over consecutive sessions = fresh long money entering.<br>
    Fetched from recent NSE bhavcopies (aggregated futures OI).<br>
    No login, computed locally.
    </div>
    """, unsafe_allow_html=True)

    return rsi_range, vol_mult, oi_lookback, universe, custom_syms

# ─────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────
def main():
    st.markdown("""
    <div class="header-bar">
      <h1>📈 F&amp;O Momentum Scanner — 5-Filter System</h1>
      <p>Long bias · NSE Futures · yfinance · No login · No downloads · Run after market close</p>
    </div>
    """, unsafe_allow_html=True)

    rsi_range, vol_mult, oi_lookback, universe_opt, custom_syms = render_sidebar()

    bhavcopies = get_recent_bhavcopies(oi_lookback + 1)  # fetch one extra for safety

    st.markdown("---")

    symbols = ALL_FO_SYMBOLS if universe_opt == "All NSE F&O (from latest bhavcopy)" else custom_syms

    if not symbols:
        st.error("No symbols selected. Please enter a custom list or choose a preset.")
        st.stop()

    est_secs  = round(len(symbols) * 0.15)
    est_label = f"~{est_secs}s" if est_secs < 60 else f"~{est_secs//60}m {est_secs%60}s"

    col_btn, col_info = st.columns([2, 5])
    with col_btn:
        label   = "🔄 Re-scan" if st.session_state.scan_results is not None else "🚀 Run Scan"
        run_now = st.button(label, type="primary", use_container_width=True,
                            disabled=st.session_state.scan_running)
    with col_info:
        st.markdown(f"""
        <p style='color:#64748b; font-size:0.85rem; margin-top:10px;'>
        Scanning <b>{len(symbols)} symbols</b> — estimated time: <b>{est_label}</b>.
        Fetches EOD data from Yahoo Finance. Best run after <b>3:30 PM IST</b>.
        </p>""", unsafe_allow_html=True)

    if run_now:
        st.session_state.scan_running = True
        fetch_ohlcv.clear()

        progress_bar = st.progress(0)
        status_text  = st.empty()

        def progress_cb(done, total, sym):
            progress_bar.progress(done / total)
            status_text.markdown(
                f'<p class="scan-label">Scanning {done}/{total} — <b>{sym}</b></p>',
                unsafe_allow_html=True,
            )

        results, errors = run_scan(
            symbols, rsi_range[0], rsi_range[1],
            vol_mult, oi_lookback, bhavcopies, progress_cb
        )

        st.session_state.scan_results   = results
        st.session_state.last_scan_time = datetime.datetime.now()
        st.session_state.errors         = errors
        st.session_state.scan_running   = False
        progress_bar.empty()
        status_text.empty()
        st.rerun()

    # (rest of main — display results, tabs, etc. unchanged)
    # ... copy the display part from your original code ...

if __name__ == "__main__":
    main()