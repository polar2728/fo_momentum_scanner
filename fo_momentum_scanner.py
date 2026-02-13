"""
F&O Momentum Scanner — 5-Filter System
Built with Streamlit + yfinance + NSE F&O Bhavcopy (real OI data)

Run AFTER market close (post 5:30 PM IST when NSE uploads bhavcopy).

Filters:
  1. Price > 200 DMA
  2. 1-month return > 6-month return (momentum acceleration)
  3. Open Interest rising over last 3 trading sessions + Price rising
     (Real F&O OI from NSE bhavcopy — no proxies, no approximations)
  4. RSI(14) between 55 and 70
  5. Volume > 1.5× 20-day average on latest session

Install:
  pip install streamlit yfinance pandas numpy ta requests
Run:
  streamlit run fo_momentum_scanner.py
"""

import time
import datetime
import zipfile
from io import BytesIO
from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import ta
import requests

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="F&O Momentum Scanner",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CSS
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

  .oi-note {
    background: #1e1b4b; border: 1px solid #4f46e5; border-radius: 8px;
    padding: 12px 16px; font-size: 0.82rem; color: #a5b4fc; margin-bottom: 16px;
  }

  .scan-label { font-size: 0.82rem; color: #64748b; margin-bottom: 2px; }
  
  .debug-box {
    background: #0f172a; border: 1px solid #334155; border-radius: 8px;
    padding: 12px; font-size: 0.75rem; color: #94a3b8; font-family: monospace;
  }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────
for key, default in {
    "scan_results": None,
    "last_scan_time": None,
    "scan_running": False,
    "errors": [],
    "oi_data": None,
    "bhavcopy_dates": [],
    "debug_info": [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ─────────────────────────────────────────────
#  NSE F&O SYMBOL LISTS
# ─────────────────────────────────────────────
NIFTY50 = [
    "RELIANCE","TCS","HDFCBANK","ICICIBANK","INFY","SBIN","HINDUNILVR","ITC",
    "KOTAKBANK","LT","AXISBANK","BAJFINANCE","BHARTIARTL","ASIANPAINT","MARUTI",
    "SUNPHARMA","TITAN","WIPRO","ULTRACEMCO","NESTLEIND","TECHM","HCLTECH",
    "TATAMOTORS","M&M","POWERGRID","NTPC","TATASTEEL","JSWSTEEL","BAJAJFINSV",
    "COALINDIA","ADANIPORTS","HINDALCO","ONGC","DIVISLAB","DRREDDY","CIPLA",
    "EICHERMOT","BAJAJ-AUTO","HEROMOTOCO","BRITANNIA","GRASIM","INDUSINDBK",
    "BPCL","IOC","SBILIFE","HDFCLIFE","APOLLOHOSP","TRENT","ADANIENT","SHRIRAMFIN",
]

NIFTY100_FO = NIFTY50 + [
    "HAL","BEL","BHEL","SAIL","NMDC","VEDL","FORTIS","MAXHEALTH",
    "ZOMATO","IRCTC","DMART","JUBLFOOD","BATAINDIA","PAGEIND",
    "ABCAPITAL","SBICARD","PNB","BANKBARODA","CANBK","UNIONBANK",
    "FEDERALBNK","IDFCFIRSTB","RECLTD","PFC","IRFC",
    "MOTHERSON","BALKRISIND","MRF","APOLLOTYRE","CEAT",
    "PIDILITIND","BERGEPAINT","HAVELLS","VOLTAS",
    "LUPIN","TORNTPHARM","AUROPHARMA","BIOCON","GLENMARK",
    "ICICIPRULI","MANAPPURAM","CHOLAFIN","MUTHOOTFIN","LICHSGFIN",
    "NAUKRI","TATACOMM","LTIM","MPHASIS","COFORGE","PERSISTENT",
    "ATGL","GAIL","INDIGO","AMBUJACEM","ACC",
    "DABUR","MARICO","COLPAL","GODREJCP",
    "PHOENIXLTD","PRESTIGE","LODHA","OBEROIRLTY","DLF","GODREJPROP",
    "POLYCAB","SIEMENS","ABB","CONCOR","GMRINFRA",
]

ALL_FO_SYMBOLS = list(dict.fromkeys(NIFTY100_FO + [
    "AARTIIND","ALKEM","AMBER","ANGELONE","ASTRAL","AUBANK",
    "BANDHANBNK","BAYERCROP","CANFINHOME","CDSL","CROMPTON","CYIENT",
    "DEEPAKNTR","DELHIVERY","DEVYANI","EASEMYTRIP","ESCORTS","EXIDEIND",
    "FINEORG","GNFC","HAPPSTMNDS","HFCL","HONAUT","IEX","IGL",
    "INDHOTEL","INTELLECT","IPCALAB","ISEC","JBCHEPHARM","JKCEMENT",
    "JKTYRE","JUSTDIAL","KAJARIACER","KEC","KFINTECH","KRBL",
    "LATENTVIEW","LAXMIMACH","LEMONTREE","LUXIND","MCX","METROPOLIS",
    "MINDTREE","MOIL","NATCOPHARM","NAVINFLUOR","NILKAMAL","OFSS",
    "PNBHOUSING","POLYPLEX","PRAJIND","PRINCEPIPE","QUESS","RADICO",
    "RBLBANK","REDINGTON","RELAXO","ROUTE","SCHAEFFLER","SJVN",
    "SKFINDIA","SOBHA","SOLARINDS","SONACOMS","STLTECH","SUDARSCHEM",
    "SUMICHEM","SUNDRMFAST","SUPREMEIND","SWSOLAR","TANLA","TATACHEM",
    "TATAELXSI","TATAPOWER","TV18BRDCST","UCOBANK","UPL",
    "VINATIORGA","WELCORP","WESTLIFE","WHIRLPOOL","ZEEL",
    "ZENSARTECH","ZYDUSLIFE",
]))


# ─────────────────────────────────────────────
#  NSE SESSION & BHAVCOPY DOWNLOAD
# ─────────────────────────────────────────────
def get_nse_session():
    """Create requests session with NSE-friendly headers."""
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate",
        "DNT": "1",
        "Connection": "keep-alive",
    })
    return s


def download_fo_bhavcopy(trading_date: date) -> pd.DataFrame:
    """Download F&O bhavcopy for a specific date."""
    s = get_nse_session()
    url = (
        f"https://nsearchives.nseindia.com/content/fo/"
        f"BhavCopy_NSE_FO_0_0_0_{trading_date.strftime('%Y%m%d')}_F_0000.csv.zip"
    )
    try:
        r = s.get(url, timeout=10)
        if r.status_code == 200:
            with zipfile.ZipFile(BytesIO(r.content)) as z:
                df = pd.read_csv(z.open(z.namelist()[0]))
                return df
    except Exception:
        pass
    return pd.DataFrame()


def download_last_n_bhavcopy(n_days: int = 5) -> dict:
    """
    Download F&O bhavcopy for the last n_days, skipping weekends/holidays.
    Returns dict: {date: DataFrame}
    """
    bhavcopy_data = {}
    for i in range(n_days * 3):  # go back further to skip weekends/holidays
        d = date.today() - timedelta(days=i)
        if d.weekday() >= 5:  # skip Sat/Sun
            continue
        df = download_fo_bhavcopy(d)
        if not df.empty:
            bhavcopy_data[d] = df
            if len(bhavcopy_data) >= n_days:
                break
    return bhavcopy_data


def extract_oi_timeseries(bhavcopy_data: dict, symbol: str) -> pd.DataFrame:
    """
    Extract OI timeseries for a given symbol from multiple bhavcopy files.
    Returns DataFrame with columns: [date, OI]
    Uses nearest expiry futures contract.
    """
    records = []
    
    for dt, df in sorted(bhavcopy_data.items()):
        # The NSE F&O bhavcopy has these key columns:
        # TckrSymb = ticker symbol
        # FinInstrmTp = instrument type (we want futures, usually blank or "XX" for futures)
        # OptnTp = option type (should be blank/XX for futures)
        # XpryDt = expiry date
        # OpnIntrst = open interest
        
        # Filter for this symbol's futures (not options)
        # Futures have blank/XX in OptnTp column
        fut = df[df["TckrSymb"] == symbol].copy()
        
        if fut.empty:
            continue
            
        # Remove options - keep only futures (OptnTp should be blank, 'XX', or NaN)
        if "OptnTp" in fut.columns:
            fut = fut[
                (fut["OptnTp"].isna()) | 
                (fut["OptnTp"] == "XX") | 
                (fut["OptnTp"] == "")
            ].copy()
        
        if fut.empty:
            continue
        
        # Ensure we have OI data
        if "OpnIntrst" not in fut.columns or fut["OpnIntrst"].isna().all():
            continue
            
        # Pick nearest expiry
        fut["XpryDt"] = pd.to_datetime(fut["XpryDt"])
        fut = fut[fut["XpryDt"] >= pd.Timestamp(dt)]
        
        if fut.empty:
            continue

        nearest = fut.sort_values("XpryDt").iloc[0]
        
        records.append({
            "date": dt,
            "OI": int(nearest["OpnIntrst"]) if pd.notna(nearest["OpnIntrst"]) else 0,
        })

    if not records:
        return pd.DataFrame()

    oi_df = pd.DataFrame(records).sort_values("date")
    return oi_df


# ─────────────────────────────────────────────
#  YFINANCE DATA FETCH
# ─────────────────────────────────────────────
def to_yf(sym: str) -> str:
    return sym + ".NS"


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_ohlcv(symbol: str) -> pd.DataFrame:
    """Fetch 14 months of daily OHLCV from yfinance."""
    ticker = yf.Ticker(to_yf(symbol))
    df = ticker.history(period="14mo", auto_adjust=True)
    if df.empty:
        return pd.DataFrame()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.sort_index(inplace=True)
    return df


# ─────────────────────────────────────────────
#  INDICATORS
# ─────────────────────────────────────────────
def compute_rsi(close: pd.Series, period: int = 14) -> float:
    val = ta.momentum.RSIIndicator(close, window=period).rsi().iloc[-1]
    return round(float(val), 2)


# ─────────────────────────────────────────────
#  5-FILTER SCAN LOGIC
# ─────────────────────────────────────────────
def scan_single(symbol: str, rsi_low: float, rsi_high: float,
                vol_mult: float, bhavcopy_data: dict):
    try:
        df = fetch_ohlcv(symbol)
        if df.empty or len(df) < 210:
            return None

        close  = df["Close"]
        volume = df["Volume"]
        latest = float(close.iloc[-1])

        # ── Filter 1: Price > 200 DMA ─────────────────────────────
        ma200 = float(close.rolling(200).mean().iloc[-1])
        if pd.isna(ma200) or latest <= ma200:
            return None

        # ── Filter 2: 1-month return > 6-month return ─────────────
        if len(close) < 130:
            return None
        ret_1m = (latest / float(close.iloc[-22]) - 1) * 100
        ret_6m = (latest / float(close.iloc[-130]) - 1) * 100
        if ret_1m <= ret_6m:
            return None

        # ── Filter 3: OI rising (last 3 sessions) + Price rising ───
        oi_ts = extract_oi_timeseries(bhavcopy_data, symbol)
        if oi_ts.empty or len(oi_ts) < 3:
            return None

        oi_vals = oi_ts["OI"].values[-3:]
        oi_rising = bool(oi_vals[0] < oi_vals[1] < oi_vals[2])
        price_rising = float(close.iloc[-1]) > float(close.iloc[-2])

        if not (oi_rising and price_rising):
            return None

        # ── Filter 4: RSI between rsi_low and rsi_high ────────────
        rsi = compute_rsi(close)
        if not (rsi_low <= rsi <= rsi_high):
            return None

        # ── Filter 5: Volume > multiplier × 20-day average ────────
        vol_20avg  = float(volume.iloc[-21:-1].mean())
        latest_vol = float(volume.iloc[-1])
        vol_ratio  = round(latest_vol / vol_20avg, 2) if vol_20avg > 0 else 0.0
        if vol_ratio < vol_mult:
            return None

        # ── All 5 filters passed ──────────────────────────────────
        ma20   = float(close.rolling(20).mean().iloc[-1])
        ma50   = float(close.rolling(50).mean().iloc[-1])
        hi_52w = float(close.rolling(252).max().iloc[-1])

        oi_change_pct = round((oi_vals[-1] / oi_vals[0] - 1) * 100, 2) if oi_vals[0] > 0 else 0
        latest_oi     = int(oi_vals[-1])

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
            "Latest OI":        latest_oi,
            "OI Change (3d) %": oi_change_pct,
            "Vol Ratio":        vol_ratio,
            "52W High":         round(hi_52w, 2),
            "% from 52W High":  round((latest / hi_52w - 1) * 100, 2),
        }

    except Exception as e:
        return {"_error": symbol, "_msg": str(e)}


def run_scan(symbols, rsi_low, rsi_high, vol_mult, bhavcopy_data, progress_cb=None):
    results, errors = [], []
    total = len(symbols)

    for idx, sym in enumerate(symbols):
        if progress_cb:
            progress_cb(idx + 1, total, sym)

        out = scan_single(sym, rsi_low, rsi_high, vol_mult, bhavcopy_data)

        if out is None:
            pass
        elif "_error" in out:
            errors.append(f"{out['_error']}: {out['_msg']}")
        else:
            results.append(out)

        time.sleep(0.1)

    if not results:
        return pd.DataFrame(), errors

    df = pd.DataFrame(results)
    df.sort_values("Momentum Gap", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.index += 1
    return df, errors


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
def render_sidebar():
    st.sidebar.markdown("## ⚙️ Settings")

    st.sidebar.markdown("### 🎛️ Filter Parameters")
    rsi_range = st.sidebar.slider(
        "RSI Range (Filter 4)",
        min_value=30, max_value=90, value=(55, 70),
    )
    vol_mult = st.sidebar.slider(
        "Volume Multiplier (Filter 5)",
        min_value=1.0, max_value=4.0, value=1.5, step=0.1,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 Stock Universe")
    universe = st.sidebar.radio(
        "Scan which stocks?",
        ["Nifty 50", "Nifty 100 F&O", "Full F&O List (~200 stocks)", "Custom List"],
        index=1,
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
        "③ OI rising (last 3 days) + Price ↑",
        f"④ RSI {rsi_range[0]}–{rsi_range[1]}",
        f"⑤ Volume > {vol_mult}× 20-day avg",
    ]:
        st.sidebar.markdown(
            f'<span class="chip chip-active">{f}</span>',
            unsafe_allow_html=True,
        )

    if st.session_state.bhavcopy_dates:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📅 Bhavcopy Loaded")
        for d in sorted(st.session_state.bhavcopy_dates, reverse=True):
            st.sidebar.markdown(f"✅ {d.strftime('%d-%b-%Y')}")

    return rsi_range, vol_mult, universe, custom_syms


# ─────────────────────────────────────────────
#  MAIN APP
# ─────────────────────────────────────────────
def main():
    st.markdown("""
    <div class="header-bar">
      <h1>📈 F&amp;O Momentum Scanner — 5-Filter System</h1>
      <p>Long bias · NSE Futures · Real OI from NSE Bhavcopy · Run after 5:30 PM IST</p>
    </div>
    """, unsafe_allow_html=True)

    rsi_range, vol_mult, universe, custom_syms = render_sidebar()

    # OI explanation banner
    st.markdown("""
    <div class="oi-note">
      <b>🎯 Filter ③ — Real Open Interest from NSE F&amp;O Bhavcopy</b><br>
      This scanner downloads the official NSE F&amp;O settlement data (bhav copy) for the last 5 trading days
      and extracts actual Open Interest for each stock's nearest expiry futures contract.
      Rising OI over 3 consecutive sessions = fresh long money entering = institutional accumulation.
      <b>No proxies, no approximations — this is the real deal.</b>
    </div>
    """, unsafe_allow_html=True)

    # Top metrics
    c1, c2, c3, c4 = st.columns(4)
    n_res = len(st.session_state.scan_results) if st.session_state.scan_results is not None else "—"
    ts    = st.session_state.last_scan_time
    with c1: st.metric("Stocks Passing All 5 Filters", n_res)
    with c2: st.metric("Last Scan", ts.strftime("%H:%M  %d-%b") if ts else "Not run yet")
    with c3: st.metric("RSI Band", f"{rsi_range[0]} – {rsi_range[1]}")
    with c4: st.metric("Vol Threshold", f"> {vol_mult}× avg")

    st.markdown("---")

    universe_map = {
        "Nifty 50":                    NIFTY50,
        "Nifty 100 F&O":               NIFTY100_FO,
        "Full F&O List (~200 stocks)":  ALL_FO_SYMBOLS,
        "Custom List":                 custom_syms,
    }
    symbols = universe_map.get(universe, NIFTY100_FO)

    col_btn, col_info = st.columns([2, 5])
    with col_btn:
        label   = "🔄 Re-scan" if st.session_state.scan_results is not None else "🚀 Run Scan"
        run_now = st.button(label, type="primary", use_container_width=True,
                            disabled=st.session_state.scan_running)
    with col_info:
        st.markdown(f"""
        <p style='color:#64748b; font-size:0.85rem; margin-top:10px;'>
        Scanning <b>{len(symbols)} symbols</b>.
        Downloads NSE bhavcopy first (takes ~10s), then scans stocks (~2 min for Nifty 100).
        <b>Best run after 5:30 PM IST</b> when NSE uploads the EOD bhavcopy.
        </p>""", unsafe_allow_html=True)

    # ── Run scan ──────────────────────────────────────────────
    if run_now:
        if not symbols:
            st.error("No symbols selected. Please choose a universe or enter custom symbols.")
            st.stop()

        st.session_state.scan_running = True
        st.session_state.debug_info = []

        # Step 1: Download bhavcopy
        with st.spinner("📥 Downloading NSE F&O bhavcopy (last 5 trading days)..."):
            bhavcopy_data = download_last_n_bhavcopy(n_days=5)

        if not bhavcopy_data:
            st.error("""
            ❌ Could not download NSE F&O bhavcopy. This usually means:
            - It's too early (NSE uploads bhavcopy after 5:30 PM IST)
            - Today is a market holiday
            - NSE server is temporarily down
            
            Try running the scanner after 6:00 PM IST on a trading day.
            """)
            st.session_state.scan_running = False
            st.stop()

        st.session_state.bhavcopy_dates = list(bhavcopy_data.keys())
        st.success(f"✅ Loaded bhavcopy for {len(bhavcopy_data)} trading days: "
                   f"{', '.join(d.strftime('%d-%b') for d in sorted(bhavcopy_data.keys(), reverse=True))}")

        # Debug: show sample data from first bhavcopy
        first_date = sorted(bhavcopy_data.keys())[0]
        sample_df = bhavcopy_data[first_date]
        
        with st.expander("🔍 Debug — Sample Bhavcopy Data (click to expand)"):
            st.markdown(f"**Columns in bhavcopy:** {', '.join(sample_df.columns.tolist())}")
            st.markdown(f"**Total rows:** {len(sample_df):,}")
            
            # Show unique values for key columns
            if "TckrSymb" in sample_df.columns:
                unique_symbols = sample_df["TckrSymb"].unique()[:20]
                st.markdown(f"**Sample symbols (first 20):** {', '.join(unique_symbols)}")
            
            if "OptnTp" in sample_df.columns:
                st.markdown(f"**Unique OptnTp values:** {sample_df['OptnTp'].unique().tolist()}")
            
            # Show a few rows for RELIANCE as example
            if "TckrSymb" in sample_df.columns:
                rel_sample = sample_df[sample_df["TckrSymb"] == "RELIANCE"].head(3)
                if not rel_sample.empty:
                    st.markdown("**Sample rows for RELIANCE:**")
                    st.dataframe(rel_sample[["TckrSymb", "XpryDt", "OpnIntrst"] + 
                                           (["OptnTp"] if "OptnTp" in rel_sample.columns else [])],
                               use_container_width=True)

        # Step 2: Clear yfinance cache and run scan
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
            vol_mult, bhavcopy_data, progress_cb,
        )

        st.session_state.scan_results   = results
        st.session_state.last_scan_time = datetime.datetime.now()
        st.session_state.errors         = errors
        st.session_state.scan_running   = False
        progress_bar.empty()
        status_text.empty()
        st.rerun()

    # ── Display results ───────────────────────────────────────
    if st.session_state.scan_results is not None:
        results = st.session_state.scan_results

        if results.empty:
            st.warning("""
            ⚠️ No stocks passed all 5 filters today.
            **Try:** widening the RSI range, lowering the volume multiplier,
            or running later in the evening if bhavcopy data is incomplete.
            """)
        else:
            st.success(f"✅ **{len(results)} stock(s)** passed all 5 momentum filters with rising real OI.")

            sc1, sc2, sc3, sc4 = st.columns(4)
            with sc1:
                st.metric("Top Pick", results.iloc[0]["Symbol"],
                          delta=f"+{results.iloc[0]['Momentum Gap']:.1f}% mom gap")
            with sc2:
                st.metric("Avg OI Change", f"+{results['OI Change (3d) %'].mean():.1f}%")
            with sc3:
                st.metric("Avg RSI", round(results["RSI (14)"].mean(), 1))
            with sc4:
                st.metric("Avg Vol Ratio", f"{results['Vol Ratio'].mean():.2f}×")

            st.markdown("---")

            tab1, tab2, tab3, tab4 = st.tabs(
                ["📋 All Results", "🏆 Top 10 Picks", "📊 Charts", "📥 Export"])

            # ── All Results ──────────────────────────────────
            with tab1:
                cols = [
                    "Symbol", "LTP", "200 DMA", "% vs 200 DMA",
                    "1M Return %", "6M Return %", "Momentum Gap",
                    "RSI (14)", "Latest OI", "OI Change (3d) %",
                    "Vol Ratio", "% from 52W High",
                ]
                styled = (
                    results[cols].style
                    .background_gradient(subset=["Momentum Gap"], cmap="Greens")
                    .background_gradient(subset=["RSI (14)"],     cmap="Blues")
                    .background_gradient(subset=["OI Change (3d) %"], cmap="Purples")
                    .format({
                        "LTP":              "₹{:.2f}",
                        "200 DMA":          "₹{:.2f}",
                        "% vs 200 DMA":     "{:+.2f}%",
                        "1M Return %":      "{:+.2f}%",
                        "6M Return %":      "{:+.2f}%",
                        "Momentum Gap":     "{:+.2f}%",
                        "RSI (14)":         "{:.1f}",
                        "Latest OI":        "{:,.0f}",
                        "OI Change (3d) %": "{:+.1f}%",
                        "Vol Ratio":        "{:.2f}×",
                        "% from 52W High":  "{:+.2f}%",
                    })
                )
                st.dataframe(styled, use_container_width=True, height=480)

            # ── Top 10 ───────────────────────────────────────
            with tab2:
                st.markdown("### 🏆 Top 10 — Ranked by Momentum Gap")
                top10 = results.head(10)
                rows_html = ""
                for i, (_, row) in enumerate(top10.iterrows(), 1):
                    gap   = row["Momentum Gap"]
                    color = "#4ade80" if gap > 10 else "#facc15" if gap > 5 else "#e2e8f0"
                    pct52 = row["% from 52W High"]
                    b52   = (
                        f'<span class="badge-green">{pct52:+.1f}%</span>'
                        if pct52 > -10
                        else f'<span style="color:#f87171">{pct52:+.1f}%</span>'
                    )
                    oi_chg = row["OI Change (3d) %"]
                    rows_html += f"""
                    <tr>
                      <td><b>#{i}</b></td>
                      <td><b>{row['Symbol']}</b></td>
                      <td>₹{row['LTP']:.2f}</td>
                      <td><span class="badge-green">{row['% vs 200 DMA']:+.1f}%</span></td>
                      <td style='color:{color}'>{row['1M Return %']:+.1f}%</td>
                      <td>{row['6M Return %']:+.1f}%</td>
                      <td style='color:{color}; font-weight:700'>{gap:+.1f}%</td>
                      <td><span class="badge-blue">{row['RSI (14)']:.1f}</span></td>
                      <td>{row['Latest OI']:,.0f}</td>
                      <td><span class="badge-gold">{oi_chg:+.1f}%</span></td>
                      <td>{row['Vol Ratio']:.2f}×</td>
                      <td>{b52}</td>
                    </tr>"""

                st.markdown(f"""
                <table class="results-table">
                  <thead>
                    <tr>
                      <th>#</th><th>Symbol</th><th>LTP</th><th>vs 200DMA</th>
                      <th>1M Ret</th><th>6M Ret</th><th>Mom Gap ↑</th>
                      <th>RSI</th><th>OI (lots)</th><th>OI Δ 3d</th>
                      <th>Vol Ratio</th><th>vs 52W High</th>
                    </tr>
                  </thead>
                  <tbody>{rows_html}</tbody>
                </table>
                """, unsafe_allow_html=True)

                st.markdown("""
                <div class="info-box" style='margin-top:20px'>
                  🎯 <b>Trade Setup for Next Session</b><br><br>
                  &nbsp;&nbsp;📌 <b>Entry:</b> Wait for a pullback to the 20 EMA — enter on bounce with confirming volume<br>
                  &nbsp;&nbsp;🛑 <b>Stop Loss:</b> Below the most recent swing low or below the 20 EMA (whichever tighter)<br>
                  &nbsp;&nbsp;💰 <b>Target 1 (50% qty):</b> 1.5× the risk taken — book partial and move SL to breakeven<br>
                  &nbsp;&nbsp;🚀 <b>Target 2 (remaining):</b> Trail SL below successive 5-day lows until stopped out<br>
                  &nbsp;&nbsp;⚠️ <b>Ban list check:</b> Always verify stock isn't on
                  <a href="https://www.nseindia.com/regulations/member-regulation-fo-participants-ban"
                  target="_blank" style="color:#818cf8">NSE F&amp;O ban list</a> before placing orders
                </div>
                """, unsafe_allow_html=True)

            # ── Charts ───────────────────────────────────────
            with tab3:
                c_l, c_r = st.columns(2)
                with c_l:
                    st.markdown("**📊 Momentum Gap — Top 15**")
                    st.bar_chart(results[["Symbol","Momentum Gap"]].head(15).set_index("Symbol"))
                with c_r:
                    st.markdown("**📊 OI Change (3d) — Top 15**")
                    st.bar_chart(results[["Symbol","OI Change (3d) %"]].head(15).set_index("Symbol"))

                c_l2, c_r2 = st.columns(2)
                with c_l2:
                    st.markdown("**📊 RSI Values — Top 15**")
                    st.bar_chart(results[["Symbol","RSI (14)"]].head(15).set_index("Symbol"))
                with c_r2:
                    st.markdown("**📊 Volume Ratio — Top 15**")
                    st.bar_chart(results[["Symbol","Vol Ratio"]].head(15).set_index("Symbol"))

            # ── Export ───────────────────────────────────────
            with tab4:
                st.markdown("### 📥 Export Results")
                csv = results.to_csv(index=True).encode("utf-8")
                st.download_button(
                    "⬇️ Download Full Results as CSV",
                    data=csv,
                    file_name=f"fo_momentum_{datetime.date.today()}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
                st.markdown("#### Symbol List — copy into broker terminal")
                st.code(", ".join(results["Symbol"].tolist()), language="text")

        # Errors
        if st.session_state.errors:
            with st.expander(f"⚠️ {len(st.session_state.errors)} symbols had errors"):
                for e in st.session_state.errors[:30]:
                    st.text(e)


if __name__ == "__main__":
    main()