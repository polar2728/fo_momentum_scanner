"""
F&O Momentum Scanner — 5-Filter System
Dynamic NSE F&O symbols from official bhavcopy ZIP (no hard-coded lists)
Uses yfinance for OHLCV + requests for NSE bhavcopy

Install:
  pip install streamlit yfinance pandas numpy ta requests

Run after market close for fresh data.
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

USE_ALL_FNO = True

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
# FETCH EXACT NSE F&O SYMBOLS FROM BHAVCOPY ZIP
# ─────────────────────────────────────────────
def get_nse_session():
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0", "Referer": "https://www.nseindia.com"})
    try:
        s.get("https://www.nseindia.com", timeout=5)
    except:
        pass
    return s

def fetch_ban_list():
    url = "https://nsearchives.nseindia.com/content/fo/fo_secban.csv"
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        if r.status_code != 200:
            return set()
        text = r.text.strip()
        if "Securities in Ban" not in text:
            return set()
        header = text.splitlines()[0]
        ban_part = header.split(":", 1)[1].strip() if ':' in header else ""
        banned = set()
        for item in ban_part.split():
            if ',' in item:
                sym = item.split(",", 1)[1].strip().upper()
                if sym.isalpha():
                    banned.add(sym)
        print(f"🚫 Banned symbols: {len(banned)}")
        return banned
    except Exception:
        print("⚠️ Ban list unavailable")
        return set()

def download_fo():
    s = get_nse_session()
    for i in range(7):
        d = date.today() - timedelta(days=i)
        url = f"https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{d.strftime('%Y%m%d')}_F_0000.csv.zip"
        try:
            r = s.get(url, timeout=10)
            if r.status_code == 200:
                with zipfile.ZipFile(BytesIO(r.content)) as z:
                    print(f"✅ F&O bhavcopy loaded for {d}")
                    return pd.read_csv(z.open(z.namelist()[0]))
        except Exception:
            continue
    print("⚠️ No F&O bhavcopy found")
    return pd.DataFrame()

# ==========================================================
# BUILD TICKER UNIVERSE
# ==========================================================
def build_ticker_universe():
    print("⏳ Building ticker universe...")
    if not USE_ALL_FNO:
        tickers = sorted(set(CORE_TICKERS))
    else:
        fo = download_fo()
        if fo.empty or "FinInstrmTp" not in fo.columns or "TckrSymb" not in fo.columns:
            print("⚠️ Fallback to core tickers")
            tickers = sorted(set(CORE_TICKERS))
        else:
            banned = fetch_ban_list()
            stocks = fo[fo["FinInstrmTp"] == "STF"]["TckrSymb"].str.upper().unique()
            valid = set(stocks) - banned
            tickers = sorted(valid | INDEX_SYMBOLS)
            print(f"✅ Loaded {len(tickers)} F&O symbols")
    symbol_map = {
        t: "^NSEI" if t == "NIFTY" else "^NSEBANK" if t == "BANKNIFTY" else f"{t}.NS"
        for t in tickers
    }
    return symbol_map

ALL_FO_SYMBOLS = build_ticker_universe()

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
# INDICATORS
# ─────────────────────────────────────────────
def compute_rsi(close: pd.Series, period: int = 14) -> float:
    rsi_series = ta.momentum.RSIIndicator(close, window=period).rsi()
    if pd.isna(rsi_series.iloc[-1]):
        return np.nan
    return round(float(rsi_series.iloc[-1]), 2)

def compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    return ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()

def obv_trending_up(obv: pd.Series, lookback: int = 5) -> bool:
    recent = obv.iloc[-lookback:].values.astype(float)
    if len(recent) < lookback:
        return False
    x = np.arange(len(recent))
    slope = np.polyfit(x, recent, 1)[0]
    return slope > 0

# ─────────────────────────────────────────────
# SCAN SINGLE STOCK
# ─────────────────────────────────────────────
def scan_single(symbol: str, rsi_low: float, rsi_high: float,
                vol_mult: float, obv_lookback: int):
    try:
        df = fetch_ohlcv(symbol)
        if df.empty or len(df) < 210:
            return None

        close  = df["Close"]
        volume = df["Volume"]
        latest = float(close.iloc[-1])

        ma200 = float(close.rolling(200).mean().iloc[-1])
        if pd.isna(ma200) or latest <= ma200:
            return None

        if len(close) < 130:
            return None
        ret_1m = (latest / float(close.iloc[-22]) - 1) * 100
        ret_6m = (latest / float(close.iloc[-130]) - 1) * 100
        if ret_1m <= ret_6m:
            return None

        obv    = compute_obv(close, volume)
        rising = obv_trending_up(obv, lookback=obv_lookback)
        price_up = float(close.iloc[-1]) > float(close.iloc[-2])
        if not (rising and price_up):
            return None

        rsi = compute_rsi(close)
        if pd.isna(rsi) or not (rsi_low <= rsi <= rsi_high):
            return None

        vol_20avg  = float(volume.iloc[-21:-1].mean())
        latest_vol = float(volume.iloc[-1])
        vol_ratio  = round(latest_vol / vol_20avg, 2) if vol_20avg > 0 else 0.0
        if vol_ratio < vol_mult:
            return None

        ma20   = float(close.rolling(20).mean().iloc[-1])
        ma50   = float(close.rolling(50).mean().iloc[-1])
        hi_52w = float(close.rolling(252).max().iloc[-1])

        obv_recent = obv.iloc[-obv_lookback:].values.astype(float)
        x          = np.arange(len(obv_recent))
        obv_slope  = round(float(np.polyfit(x, obv_recent, 1)[0]), 0)

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
            "OBV Slope":        obv_slope,
            "Vol Ratio":        vol_ratio,
            "52W High":         round(hi_52w, 2),
            "% from 52W High":  round((latest / hi_52w - 1) * 100, 2),
        }

    except Exception as e:
        return {"_error": symbol, "_msg": str(e)}

# ─────────────────────────────────────────────
# RUN FULL SCAN
# ─────────────────────────────────────────────
def run_scan(symbols, rsi_low, rsi_high, vol_mult, obv_lookback, progress_cb=None):
    results, errors = [], []
    total = len(symbols)

    for idx, sym in enumerate(symbols):
        if progress_cb:
            progress_cb(idx + 1, total, sym)

        out = scan_single(sym, rsi_low, rsi_high, vol_mult, obv_lookback)

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
# SIDEBAR
# ─────────────────────────────────────────────
def render_sidebar():
    st.sidebar.markdown("## ⚙️ Settings")

    st.sidebar.markdown("### 🎛️ Filter Parameters")

    rsi_range = st.sidebar.slider(
        "RSI Range (Filter 4)", 30, 90, (55, 70),
        help="Avoid overbought entries"
    )
    vol_mult = st.sidebar.slider(
        "Volume Multiplier (Filter 5)", 1.0, 4.0, 1.5, 0.1
    )
    obv_lookback = st.sidebar.slider(
        "OBV Lookback (Filter 3)", 3, 10, 5
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
            "Symbols (comma-separated, NO .NS)",
            placeholder="RELIANCE,TCS,HDFCBANK",
            height=100,
        )
        custom_syms = [s.strip().upper() for s in raw.split(",") if s.strip()]

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ✅ Active Filters")
    for f in [
        "① Price > 200 DMA",
        "② 1M return > 6M return",
        f"③ OBV rising ({obv_lookback}-day) + Price ↑",
        f"④ RSI {rsi_range[0]}–{rsi_range[1]}",
        f"⑤ Volume > {vol_mult}× 20-day avg",
    ]:
        st.sidebar.markdown(f'<span class="chip chip-active">{f}</span>', unsafe_allow_html=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style='font-size:0.75rem; color:#475569'>
    Universe auto-fetched from NSE F&amp;O bhavcopy (~180–220 symbols).<br>
    Cached 12h — fresh on restart if needed.
    </div>
    """, unsafe_allow_html=True)

    return rsi_range, vol_mult, obv_lookback, universe, custom_syms

# ─────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────
def main():
    st.markdown("""
    <div class="header-bar">
      <h1>📈 F&amp;O Momentum Scanner — 5-Filter System</h1>
      <p>Exact NSE F&amp;O universe from bhavcopy · yfinance prices · Run after close</p>
    </div>
    """, unsafe_allow_html=True)

    rsi_range, vol_mult, obv_lookback, universe_opt, custom_syms = render_sidebar()

    st.markdown("""
    <div class="obv-note">
      <b>Filter ③ — OBV:</b> Rising slope over N days + close up = institutional accumulation proxy.<br>
      Purely from price/volume — no external OI needed.
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    n_res = len(st.session_state.scan_results) if st.session_state.scan_results is not None else "—"
    ts    = st.session_state.last_scan_time
    with c1: st.metric("Stocks Passing", n_res)
    with c2: st.metric("Last Scan", ts.strftime("%H:%M  %d-%b") if ts else "Not run")
    with c3: st.metric("RSI Band", f"{rsi_range[0]} – {rsi_range[1]}")
    with c4: st.metric("Vol >", f"{vol_mult}× avg")

    st.markdown("---")

    symbols = ALL_FO_SYMBOLS if universe_opt == "All NSE F&O (from latest bhavcopy)" else custom_syms

    if not symbols:
        st.error("No symbols available. Use Custom List or check NSE connectivity.")
        st.stop()

    est_secs  = round(len(symbols) * 0.10)
    est_label = f"~{est_secs}s" if est_secs < 60 else f"~{est_secs//60}m {est_secs%60}s"

    col_btn, col_info = st.columns([2, 5])
    with col_btn:
        label = "🔄 Re-scan" if st.session_state.scan_results is not None else "🚀 Run Scan"
        run_now = st.button(label, type="primary", use_container_width=True,
                            disabled=st.session_state.scan_running)
    with col_info:
        st.markdown(f"""
        <p style='color:#64748b; font-size:0.85rem; margin-top:10px;'>
        Scanning <b>{len(symbols)} F&amp;O symbols</b> — est. {est_label}<br>
        Best after 15:30–16:00 IST.
        </p>""", unsafe_allow_html=True)

    if run_now:
        st.session_state.scan_running = True
        fetch_ohlcv.clear()

        progress_bar = st.progress(0)
        status_text  = st.empty()

        def progress_cb(done, total, sym):
            progress_bar.progress(done / total)
            status_text.markdown(f'<p class="scan-label">Scanning {done}/{total} — {sym}</p>', unsafe_allow_html=True)

        results, errors = run_scan(
            symbols, rsi_range[0], rsi_range[1], vol_mult, obv_lookback, progress_cb
        )

        st.session_state.scan_results   = results
        st.session_state.last_scan_time = datetime.datetime.now()
        st.session_state.errors         = errors
        st.session_state.scan_running   = False
        progress_bar.empty()
        status_text.empty()
        st.rerun()

    if st.session_state.scan_results is not None:
        results = st.session_state.scan_results

        if results.empty:
            st.warning("No stocks passed all filters. Try wider RSI or lower vol multiplier.")
        else:
            st.success(f"✅ **{len(results)} stocks** passed momentum filters.")
            st.caption(f"Data as of {st.session_state.last_scan_time.strftime('%d %b %Y %H:%M IST')} | Universe: {len(symbols)} F&O symbols")

            sc1, sc2, sc3, sc4 = st.columns(4)
            with sc1: st.metric("Top Pick", results.iloc[0]["Symbol"], f"+{results.iloc[0]['Momentum Gap']:.1f}% gap")
            with sc2: st.metric("Avg RSI", round(results["RSI (14)"].mean(), 1))
            with sc3: st.metric("Avg Vol Ratio", f"{results['Vol Ratio'].mean():.2f}×")
            with sc4: st.metric("Avg Gap", f"+{results['Momentum Gap'].mean():.1f}%")

            st.markdown("---")

            tab1, tab2, tab3, tab4 = st.tabs(["📋 All Results", "🏆 Top 10", "📊 Charts", "📥 Export"])

            with tab1:
                cols = [
                    "Symbol", "LTP", "200 DMA", "% vs 200 DMA",
                    "1M Return %", "6M Return %", "Momentum Gap",
                    "RSI (14)", "OBV Slope", "Vol Ratio", "% from 52W High",
                ]
                styled = results[cols].style\
                    .background_gradient(subset=["Momentum Gap"], cmap="Greens")\
                    .background_gradient(subset=["RSI (14)"], cmap="Blues")\
                    .background_gradient(subset=["OBV Slope"], cmap="Purples")\
                    .format({
                        "LTP": "₹{:.2f}", "200 DMA": "₹{:.2f}",
                        "% vs 200 DMA": "{:+.2f}%", "1M Return %": "{:+.2f}%",
                        "6M Return %": "{:+.2f}%", "Momentum Gap": "{:+.2f}%",
                        "RSI (14)": "{:.1f}", "OBV Slope": "{:,.0f}",
                        "Vol Ratio": "{:.2f}×", "% from 52W High": "{:+.2f}%"
                    })
                st.dataframe(styled, use_container_width=True, height=480)

            with tab2:
                st.markdown("### 🏆 Top 10 by Momentum Gap")
                top10 = results.head(10)
                rows_html = ""
                for i, (_, row) in enumerate(top10.iterrows(), 1):
                    gap = row["Momentum Gap"]
                    color = "#4ade80" if gap > 10 else "#facc15" if gap > 5 else "#e2e8f0"
                    pct52 = row["% from 52W High"]
                    badge52 = f'<span class="badge-green">{pct52:+.1f}%</span>' if pct52 > -10 else f'<span style="color:#f87171">{pct52:+.1f}%</span>'
                    rows_html += f"""
                    <tr>
                      <td><b>#{i}</b></td>
                      <td><b>{row['Symbol']}</b></td>
                      <td>₹{row['LTP']:.2f}</td>
                      <td><span class="badge-green">{row['% vs 200 DMA']:+.1f}%</span></td>
                      <td style="color:{color}">{row['1M Return %']:+.1f}%</td>
                      <td>{row['6M Return %']:+.1f}%</td>
                      <td style="color:{color};font-weight:700">{gap:+.1f}%</td>
                      <td><span class="badge-blue">{row['RSI (14)']:.1f}</span></td>
                      <td><span class="badge-gold">{row['OBV Slope']:,.0f}</span></td>
                      <td>{row['Vol Ratio']:.2f}×</td>
                      <td>{badge52}</td>
                    </tr>"""
                st.markdown(f"""
                <table class="results-table">
                  <thead><tr><th>#</th><th>Symbol</th><th>LTP</th><th>vs 200DMA</th><th>1M Ret</th><th>6M Ret</th><th>Mom Gap</th><th>RSI</th><th>OBV Slope</th><th>Vol Ratio</th><th>vs 52W High</th></tr></thead>
                  <tbody>{rows_html}</tbody>
                </table>
                """, unsafe_allow_html=True)

                st.markdown("""
                <div class="info-box" style="margin-top:20px">
                  🎯 <b>Setup Idea</b><br>
                  • Entry: Pullback to 20 EMA + volume<br>
                  • Stop: Below swing low / 20 EMA<br>
                  • Targets: 1.5× risk (50%), trail rest<br>
                  ⚠️ Always check <a href="https://www.nseindia.com/regulations/member-regulation-fo-participants-ban" target="_blank" style="color:#818cf8">NSE F&amp;O Ban List</a> before trading futures.
                </div>
                """, unsafe_allow_html=True)

            with tab3:
                c1, c2 = st.columns(2)
                with c1: st.bar_chart(results.head(15)[["Symbol", "Momentum Gap"]].set_index("Symbol"))
                with c2: st.bar_chart(results.head(15)[["Symbol", "RSI (14)"]].set_index("Symbol"))

            with tab4:
                today_str = datetime.date.today().strftime("%Y-%m-%d")
                csv = results.to_csv(index=True).encode("utf-8")
                st.download_button(
                    "⬇️ Download Results CSV",
                    csv,
                    f"fo_momentum_{today_str}.csv",
                    "text/csv",
                    use_container_width=True
                )
                st.markdown("#### Copy symbols:")
                st.code(", ".join(results["Symbol"].tolist()))

        if st.session_state.errors:
            with st.expander(f"⚠️ {len(st.session_state.errors)} errors"):
                for e in st.session_state.errors[:30]:
                    st.text(e)

if __name__ == "__main__":
    main()