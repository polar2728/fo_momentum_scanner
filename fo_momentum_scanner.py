"""
F&O Momentum Scanner — 5-Filter System
Built with Streamlit + KiteConnect API

Filters:
  1. Price > 200 DMA
  2. 1-month return > 6-month return (momentum acceleration)
  3. Rising OI (last 3 sessions) + Rising Price
  4. RSI between 55 and 70
  5. Volume > 1.5x the 20-day average on the latest up-day

Run:
  pip install streamlit kiteconnect pandas numpy ta requests
  streamlit run fo_momentum_scanner.py
"""

import time
import math
import datetime
import threading
import webbrowser
from urllib.parse import urlparse, parse_qs

import numpy as np
import pandas as pd
import streamlit as st
from kiteconnect import KiteConnect
import ta  # technical analysis library


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
#  CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
  /* Dark card feel */
  [data-testid="stSidebar"] { background: #0f1117; }
  .block-container { padding-top: 1.5rem; }

  /* Metric tiles */
  div[data-testid="metric-container"] {
    background: #1e2130;
    border: 1px solid #2d3250;
    border-radius: 10px;
    padding: 14px 20px;
  }

  /* Green / red badge helper classes (used in HTML tables) */
  .badge-green { background:#16a34a22; color:#4ade80; border:1px solid #16a34a55;
                 padding:2px 8px; border-radius:4px; font-size:0.82rem; }
  .badge-red   { background:#dc262622; color:#f87171; border:1px solid #dc262655;
                 padding:2px 8px; border-radius:4px; font-size:0.82rem; }
  .badge-blue  { background:#2563eb22; color:#93c5fd; border:1px solid #2563eb55;
                 padding:2px 8px; border-radius:4px; font-size:0.82rem; }
  .header-bar  { background: linear-gradient(90deg,#1e40af,#7c3aed);
                 border-radius:10px; padding:18px 24px; margin-bottom:1.5rem; }
  .header-bar h1 { color:white; margin:0; font-size:1.6rem; }
  .header-bar p  { color:#cbd5e1; margin:4px 0 0; font-size:0.9rem; }

  /* Result table */
  .results-table { width:100%; border-collapse:collapse; font-size:0.88rem; }
  .results-table th { background:#1e2130; color:#94a3b8; padding:10px 14px;
                      text-align:left; border-bottom:1px solid #2d3250; }
  .results-table td { padding:10px 14px; border-bottom:1px solid #1e2130; color:#e2e8f0; }
  .results-table tr:hover td { background:#1e2130; }

  /* Filter chips */
  .chip { display:inline-block; background:#1e2130; border:1px solid #2d3250;
          border-radius:20px; padding:4px 12px; font-size:0.8rem; color:#94a3b8;
          margin:2px; }
  .chip-active { border-color:#4f46e5; color:#a5b4fc; background:#1e1b4b; }

  /* Login card */
  .login-card { background:#1e2130; border:1px solid #2d3250; border-radius:16px;
                padding:32px; max-width:480px; margin:40px auto; }
  .login-card h2 { color:#e2e8f0; margin-top:0; }
  .login-card p  { color:#94a3b8; font-size:0.9rem; }

  /* Progress label */
  .scan-label { font-size:0.82rem; color:#64748b; margin-bottom:2px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  SESSION STATE DEFAULTS
# ─────────────────────────────────────────────
for key, default in {
    "access_token": None,
    "kite": None,
    "api_key": "",
    "api_secret": "",
    "scan_results": None,
    "last_scan_time": None,
    "scan_running": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────

def get_fo_symbols(kite) -> list[str]:
    """Return all stock symbols currently in the F&O segment (NFO)."""
    instruments = kite.instruments("NFO")
    df = pd.DataFrame(instruments)
    # Keep only FUTSTK (stock futures), unique underlying symbols
    futures = df[df["instrument_type"] == "FUT"].copy()
    futures = futures[futures["segment"] == "NFO-FUT"]
    symbols = futures["name"].dropna().unique().tolist()
    return sorted(symbols)


def fetch_ohlcv(kite, symbol: str, exchange: str = "NSE", days: int = 250) -> pd.DataFrame:
    """Fetch daily OHLCV data for a symbol."""
    to_date   = datetime.date.today()
    from_date = to_date - datetime.timedelta(days=days + 50)  # buffer for weekends/holidays

    instrument_token = None
    instruments = kite.instruments(exchange)
    for inst in instruments:
        if inst["tradingsymbol"] == symbol and inst["instrument_type"] == "EQ":
            instrument_token = inst["instrument_token"]
            break

    if instrument_token is None:
        return pd.DataFrame()

    data = kite.historical_data(
        instrument_token,
        from_date.strftime("%Y-%m-%d"),
        to_date.strftime("%Y-%m-%d"),
        "day"
    )
    df = pd.DataFrame(data)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    df.sort_index(inplace=True)
    return df


def fetch_oi_history(kite, symbol: str, days: int = 10) -> pd.DataFrame:
    """Fetch recent OI data from the nearest expiry futures contract."""
    to_date   = datetime.date.today()
    from_date = to_date - datetime.timedelta(days=days + 10)

    instruments = kite.instruments("NFO")
    df_inst = pd.DataFrame(instruments)
    futures = df_inst[
        (df_inst["name"] == symbol) &
        (df_inst["instrument_type"] == "FUT")
    ].copy()

    if futures.empty:
        return pd.DataFrame()

    # Pick the nearest expiry
    futures["expiry"] = pd.to_datetime(futures["expiry"])
    futures = futures[futures["expiry"] >= pd.Timestamp(to_date)]
    if futures.empty:
        return pd.DataFrame()

    nearest = futures.sort_values("expiry").iloc[0]
    token   = nearest["instrument_token"]

    data = kite.historical_data(
        token,
        from_date.strftime("%Y-%m-%d"),
        to_date.strftime("%Y-%m-%d"),
        "day"
    )
    df = pd.DataFrame(data)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    df.sort_index(inplace=True)
    return df


def compute_rsi(series: pd.Series, period: int = 14) -> float:
    """Return the latest RSI value."""
    rsi_series = ta.momentum.RSIIndicator(series, window=period).rsi()
    return round(float(rsi_series.iloc[-1]), 2)


def run_5_filter_scan(kite, symbols: list[str],
                      rsi_low=55, rsi_high=70,
                      vol_multiplier=1.5,
                      progress_cb=None) -> pd.DataFrame:
    """
    Apply all 5 momentum filters to each symbol.
    Returns a DataFrame of stocks that pass all filters.
    """
    results = []
    total   = len(symbols)

    for idx, symbol in enumerate(symbols):
        if progress_cb:
            progress_cb(idx + 1, total, symbol)

        try:
            df = fetch_ohlcv(kite, symbol, days=260)
            if len(df) < 130:
                continue

            close   = df["close"]
            volume  = df["volume"]
            latest  = close.iloc[-1]

            # ── Filter 1: Price > 200 DMA ─────────────────
            ma200 = close.rolling(200).mean().iloc[-1]
            if pd.isna(ma200) or latest <= ma200:
                continue

            # ── Filter 2: 1-month return > 6-month return ─
            ret_1m = (latest / close.iloc[-22] - 1) * 100  if len(close) >= 22  else None
            ret_6m = (latest / close.iloc[-130] - 1) * 100 if len(close) >= 130 else None
            if ret_1m is None or ret_6m is None or ret_1m <= ret_6m:
                continue

            # ── Filter 3: Rising OI + Rising Price ────────
            oi_df = fetch_oi_history(kite, symbol, days=10)
            if oi_df.empty or len(oi_df) < 3 or "oi" not in oi_df.columns:
                continue

            oi_vals    = oi_df["oi"].dropna().values
            price_vals = close.iloc[-3:].values

            oi_rising    = all(oi_vals[-3] < oi_vals[-2] < oi_vals[-1])
            price_rising = price_vals[-1] > price_vals[-2]

            if not (oi_rising and price_rising):
                continue

            # ── Filter 4: RSI between rsi_low and rsi_high ─
            rsi = compute_rsi(close)
            if not (rsi_low <= rsi <= rsi_high):
                continue

            # ── Filter 5: Volume > multiplier × 20-day avg (on up-day) ──
            is_up_day = close.iloc[-1] > close.iloc[-2]
            vol_20avg = volume.iloc[-21:-1].mean()
            latest_vol = volume.iloc[-1]
            vol_ratio  = round(latest_vol / vol_20avg, 2) if vol_20avg > 0 else 0

            if not (is_up_day and vol_ratio >= vol_multiplier):
                continue

            # ── All 5 filters passed ──
            results.append({
                "Symbol":         symbol,
                "LTP":            round(latest, 2),
                "200 DMA":        round(ma200, 2),
                "% vs 200 DMA":   round((latest / ma200 - 1) * 100, 2),
                "1M Return %":    round(ret_1m, 2),
                "6M Return %":    round(ret_6m, 2),
                "Momentum Gap":   round(ret_1m - ret_6m, 2),
                "RSI (14)":       rsi,
                "Vol Ratio":      vol_ratio,
                "OI Rising":      "✅ Yes",
                "Price Rising":   "✅ Yes",
            })

        except Exception as e:
            # Silently skip stocks that throw errors (e.g. illiquid, delisted)
            continue

        # Zerodha rate limit: ~3 req/sec
        time.sleep(0.35)

    if not results:
        return pd.DataFrame()

    out = pd.DataFrame(results)
    out.sort_values("Momentum Gap", ascending=False, inplace=True)
    out.reset_index(drop=True, inplace=True)
    out.index += 1
    return out


# ─────────────────────────────────────────────
#  LOGIN SECTION
# ─────────────────────────────────────────────

def render_login():
    st.markdown("""
    <div class="login-card">
      <h2>🔐 Connect to Zerodha Kite</h2>
      <p>Enter your Kite Connect API credentials to start the scanner.
         Generate them from <a href="https://developers.kite.trade" target="_blank"
         style="color:#818cf8">developers.kite.trade</a>.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        api_key    = st.text_input("API Key",    value=st.session_state.api_key,
                                   placeholder="your_api_key")
        api_secret = st.text_input("API Secret", value=st.session_state.api_secret,
                                   type="password", placeholder="your_api_secret")

        st.markdown("---")
        st.markdown("#### Step 1 — Generate Login URL")
        if st.button("🔗 Open Kite Login", use_container_width=True, type="primary"):
            if not api_key:
                st.error("Please enter your API Key first.")
            else:
                st.session_state.api_key    = api_key
                st.session_state.api_secret = api_secret
                kite = KiteConnect(api_key=api_key)
                login_url = kite.login_url()
                st.markdown(f"""
                <div style='background:#1e2130;border:1px solid #2d3250;border-radius:8px;padding:12px;margin-top:8px;'>
                  <p style='color:#94a3b8;font-size:0.85rem;margin:0 0 8px'>
                    Click the link below, log in to Zerodha, and paste the
                    <code>request_token</code> from the redirect URL back here:
                  </p>
                  <a href="{login_url}" target="_blank"
                     style='color:#818cf8;word-break:break-all;font-size:0.85rem'>{login_url}</a>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("#### Step 2 — Paste Request Token")
        request_token = st.text_input(
            "Request Token",
            placeholder="Paste request_token from redirect URL",
            help="After login, copy the `request_token` param from your browser's address bar"
        )

        if st.button("✅ Generate Access Token & Connect", use_container_width=True):
            if not api_key or not api_secret or not request_token:
                st.error("All three fields (API Key, API Secret, Request Token) are required.")
            else:
                with st.spinner("Authenticating..."):
                    try:
                        kite = KiteConnect(api_key=api_key)
                        data = kite.generate_session(request_token.strip(), api_secret=api_secret)
                        kite.set_access_token(data["access_token"])

                        st.session_state.kite          = kite
                        st.session_state.access_token  = data["access_token"]
                        st.session_state.api_key       = api_key
                        st.session_state.api_secret    = api_secret
                        st.success("✅ Connected to Kite! Refreshing...")
                        time.sleep(1)
                        st.rerun()

                    except Exception as e:
                        st.error(f"Authentication failed: {e}")


# ─────────────────────────────────────────────
#  SIDEBAR — FILTER CONTROLS
# ─────────────────────────────────────────────

def render_sidebar():
    st.sidebar.markdown("## ⚙️ Scanner Settings")

    st.sidebar.markdown("### 👤 Session")
    if st.sidebar.button("🚪 Logout", use_container_width=True):
        for key in ["access_token", "kite", "scan_results"]:
            st.session_state[key] = None
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎛️ Filter Parameters")

    rsi_range = st.sidebar.slider(
        "RSI Range (Filter 4)",
        min_value=30, max_value=90,
        value=(55, 70),
        help="Only include stocks whose RSI falls within this band"
    )

    vol_multiplier = st.sidebar.slider(
        "Volume Multiplier vs 20-day avg (Filter 5)",
        min_value=1.0, max_value=3.0,
        value=1.5, step=0.1,
        help="Stock volume on the latest session must exceed this × 20-day average"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 Stock Universe")
    universe_option = st.sidebar.radio(
        "Scan which stocks?",
        ["All F&O Stocks (Full Scan)", "Nifty 100 F&O Stocks", "Custom List"],
        index=1,
    )

    custom_symbols = []
    if universe_option == "Custom List":
        raw = st.sidebar.text_area(
            "Enter symbols (comma-separated)",
            placeholder="RELIANCE,TCS,HDFCBANK,INFY",
            height=100
        )
        custom_symbols = [s.strip().upper() for s in raw.split(",") if s.strip()]

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Active Filters")
    filters = [
        "Price > 200 DMA",
        "1M return > 6M return",
        "Rising OI + Rising Price",
        f"RSI {rsi_range[0]}–{rsi_range[1]}",
        f"Volume > {vol_multiplier}× 20-day avg",
    ]
    for f in filters:
        st.sidebar.markdown(f'<span class="chip chip-active">✓ {f}</span>', unsafe_allow_html=True)

    return rsi_range, vol_multiplier, universe_option, custom_symbols


# ─────────────────────────────────────────────
#  NIFTY 100 F&O SYMBOL LIST (subset)
# ─────────────────────────────────────────────

NIFTY100_FO_SYMBOLS = [
    "RELIANCE","TCS","HDFCBANK","ICICIBANK","INFY","SBIN","HINDUNILVR","ITC",
    "KOTAKBANK","LT","AXISBANK","BAJFINANCE","BHARTIARTL","ASIANPAINT","MARUTI",
    "SUNPHARMA","TITAN","WIPRO","ULTRACEMCO","NESTLEIND","TECHM","HCLTECH",
    "TATAMOTORS","M&M","POWERGRID","NTPC","TATASTEEL","JSWSTEEL","BAJAJFINSV",
    "COALINDIA","ADANIPORTS","HINDALCO","ONGC","DIVISLAB","DRREDDY","CIPLA",
    "EICHERMOT","BAJAJ-AUTO","HEROMOTOCO","BRITANNIA","GRASIM","INDUSINDBK",
    "BPCL","IOC","SBILIFE","HDFCLIFE","DABUR","PIDILITIND","BERGEPAINT",
    "MCDOWELL-N","GODREJCP","MARICO","COLPAL","HAVELLS","VOLTAS","WHIRLPOOL",
    "LUPIN","TORNTPHARM","AUROPHARMA","BIOCON","GLENMARK","CADILAHC",
    "ICICIPRULI","MANAPPURAM","CHOLAFIN","MUTHOOTFIN","PEL","LICHSGFIN",
    "HAL","BEL","BHEL","SAIL","NMDC","VEDL","APOLLOHOSP","FORTIS","MAXHEALTH",
    "NAUKRI","ZOMATO","PAYTM","POLICYBZR","IRCTC","DMART","TRENT","JUBLFOOD",
    "BATAINDIA","PAGEIND","RAJESHEXPO","ABCAPITAL","ICICIGI","SBICARD",
    "PNB","BANKBARODA","CANBK","UNIONBANK","FEDERALBNK","IDFCFIRSTB",
    "RECLTD","PFC","IRFC","GMRINFRA","ADANIENT","ADANIGREEN","ADANITRANS",
    "MOTHERSON","BALKRISIND","MRF","APOLLOTYRE","CEAT",
]


# ─────────────────────────────────────────────
#  MAIN SCANNER DASHBOARD
# ─────────────────────────────────────────────

def render_scanner():
    # Header
    st.markdown("""
    <div class="header-bar">
      <h1>📈 F&O Momentum Scanner — 5-Filter System</h1>
      <p>Long bias · NSE Futures · Zerodha Kite Connect</p>
    </div>
    """, unsafe_allow_html=True)

    rsi_range, vol_multiplier, universe_option, custom_symbols = render_sidebar()

    kite = st.session_state.kite

    # ── Top metrics ──
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        n_results = len(st.session_state.scan_results) if st.session_state.scan_results is not None else "—"
        st.metric("Stocks Passing All Filters", n_results)
    with col2:
        ts = st.session_state.last_scan_time
        st.metric("Last Scan", ts.strftime("%H:%M:%S %d-%b") if ts else "Not run yet")
    with col3:
        st.metric("RSI Band", f"{rsi_range[0]} – {rsi_range[1]}")
    with col4:
        st.metric("Volume Filter", f">{vol_multiplier}× avg")

    st.markdown("---")

    # ── Scan button ──
    col_btn, col_info = st.columns([2, 5])
    with col_btn:
        scan_label = "🔄 Re-scan" if st.session_state.scan_results is not None else "🚀 Run 5-Filter Scan"
        run_scan = st.button(scan_label, type="primary", use_container_width=True,
                             disabled=st.session_state.scan_running)
    with col_info:
        st.markdown("""
        <p style='color:#64748b; font-size:0.85rem; margin-top:10px;'>
        ⚡ Full scan can take 5–15 minutes depending on universe size due to Zerodha rate limits (3 req/s).
        Nifty 100 scan takes ~3–5 min.
        </p>
        """, unsafe_allow_html=True)

    # ── Run the scan ──
    if run_scan:
        st.session_state.scan_running = True

        # Determine symbol list
        if universe_option == "Custom List":
            symbols = custom_symbols if custom_symbols else []
        elif universe_option == "Nifty 100 F&O Stocks":
            symbols = NIFTY100_FO_SYMBOLS
        else:
            with st.spinner("Fetching full F&O instrument list..."):
                symbols = get_fo_symbols(kite)

        if not symbols:
            st.error("No symbols selected. Please choose a universe or enter custom symbols.")
            st.session_state.scan_running = False
            st.stop()

        st.info(f"Scanning **{len(symbols)} symbols** — this window will update in real time.")

        progress_bar = st.progress(0)
        status_text  = st.empty()
        result_placeholder = st.empty()

        def progress_cb(done, total, current_symbol):
            pct = done / total
            progress_bar.progress(pct)
            status_text.markdown(
                f'<p class="scan-label">Scanning {done}/{total} — Current: <b>{current_symbol}</b></p>',
                unsafe_allow_html=True
            )

        results = run_5_filter_scan(
            kite, symbols,
            rsi_low=rsi_range[0],
            rsi_high=rsi_range[1],
            vol_multiplier=vol_multiplier,
            progress_cb=progress_cb,
        )

        st.session_state.scan_results    = results
        st.session_state.last_scan_time  = datetime.datetime.now()
        st.session_state.scan_running    = False
        progress_bar.empty()
        status_text.empty()
        st.rerun()

    # ── Display results ──
    if st.session_state.scan_results is not None:
        results = st.session_state.scan_results

        if results.empty:
            st.warning("⚠️ No stocks passed all 5 filters in this scan. Try relaxing the RSI or volume thresholds.")
        else:
            st.success(f"✅ **{len(results)} stocks** passed all 5 momentum filters.")

            # Summary row
            c1, c2, c3 = st.columns(3)
            with c1:
                best = results.iloc[0]["Symbol"]
                st.metric("Top Momentum Stock", best,
                          delta=f"+{results.iloc[0]['Momentum Gap']:.1f}% gap")
            with c2:
                avg_rsi = results["RSI (14)"].mean()
                st.metric("Avg RSI of Results", round(avg_rsi, 1))
            with c3:
                avg_vol = results["Vol Ratio"].mean()
                st.metric("Avg Volume Ratio", f"{avg_vol:.2f}×")

            st.markdown("---")

            # Tabs
            tab1, tab2, tab3 = st.tabs(["📋 Full Results", "🏆 Top 10 Picks", "📥 Export"])

            with tab1:
                st.markdown("### All Qualifying Stocks")
                # Colour-code the momentum gap column
                def highlight_row(val):
                    if isinstance(val, float) and val > 10:
                        return "color: #4ade80"
                    elif isinstance(val, float) and val > 5:
                        return "color: #facc15"
                    return ""

                styled = results.style.applymap(
                    highlight_row, subset=["Momentum Gap", "1M Return %"]
                ).format({
                    "LTP": "₹{:.2f}",
                    "200 DMA": "₹{:.2f}",
                    "% vs 200 DMA": "{:+.2f}%",
                    "1M Return %": "{:+.2f}%",
                    "6M Return %": "{:+.2f}%",
                    "Momentum Gap": "{:+.2f}%",
                    "RSI (14)": "{:.1f}",
                    "Vol Ratio": "{:.2f}×",
                })
                st.dataframe(styled, use_container_width=True, height=450)

            with tab2:
                st.markdown("### 🏆 Top 10 — Ranked by Momentum Gap")
                top10 = results.head(10).copy()

                # Render as rich HTML table
                rows_html = ""
                for _, row in top10.iterrows():
                    gap   = row["Momentum Gap"]
                    color = "#4ade80" if gap > 10 else "#facc15" if gap > 5 else "#e2e8f0"
                    rows_html += f"""
                    <tr>
                      <td><b>{row['Symbol']}</b></td>
                      <td>₹{row['LTP']:.2f}</td>
                      <td>₹{row['200 DMA']:.2f}</td>
                      <td><span class="badge-green">{row['% vs 200 DMA']:+.1f}%</span></td>
                      <td style="color:{color}">{row['1M Return %']:+.1f}%</td>
                      <td>{row['6M Return %']:+.1f}%</td>
                      <td style="color:{color}; font-weight:600">{gap:+.1f}%</td>
                      <td><span class="badge-blue">{row['RSI (14)']:.1f}</span></td>
                      <td>{row['Vol Ratio']:.2f}×</td>
                    </tr>"""

                st.markdown(f"""
                <table class="results-table">
                  <thead>
                    <tr>
                      <th>Symbol</th><th>LTP</th><th>200 DMA</th><th>vs DMA</th>
                      <th>1M Ret</th><th>6M Ret</th><th>Mom Gap</th>
                      <th>RSI</th><th>Vol Ratio</th>
                    </tr>
                  </thead>
                  <tbody>{rows_html}</tbody>
                </table>
                """, unsafe_allow_html=True)

                st.markdown("---")
                st.markdown("""
                <p style='color:#64748b; font-size:0.82rem;'>
                🎯 <b>How to use these picks:</b> Enter on pullback to 10/20 EMA.
                Set stop below the recent swing low. Target 1.5× risk for T1,
                trail stop to breakeven after T1 and ride with a 5-day low trail for T2.
                Check F&O ban list daily before trading.
                </p>
                """, unsafe_allow_html=True)

            with tab3:
                st.markdown("### 📥 Export Results")
                csv_data = results.to_csv(index=True).encode("utf-8")
                st.download_button(
                    label="⬇️ Download CSV",
                    data=csv_data,
                    file_name=f"fo_momentum_scan_{datetime.date.today()}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

                st.markdown("#### Quick Copy — Symbol List")
                symbol_list = ", ".join(results["Symbol"].tolist())
                st.code(symbol_list, language="text")


# ─────────────────────────────────────────────
#  APP ENTRY POINT
# ─────────────────────────────────────────────

def main():
    if st.session_state.access_token is None or st.session_state.kite is None:
        render_login()
    else:
        render_scanner()


if __name__ == "__main__":
    main()