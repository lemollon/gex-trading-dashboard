"""
GEX Trading Dashboard - Multi-Symbol Gamma Exposure Analysis Platform
Author: GEX Trading System
Version: 3.1.0
Description: Comprehensive multi-symbol dashboard for gamma exposure analysis,
             trade setup detection across entire watchlist, and position management
             with Databricks SQL live feed toggle.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta, time
import pytz
import warnings
import json
import time as time_module
from typing import Dict, List, Tuple, Optional
import logging
import random

# ===== Logging & Warnings =====
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# ===== Page Config =====
st.set_page_config(
    page_title="GEX Trading Dashboard Pro",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== NEW: Databricks SQL Setup =====
DBSQL_AVAILABLE = True
try:
    from databricks import sql
except Exception:
    DBSQL_AVAILABLE = False

CATALOG = os.getenv("GEX_CATALOG", "analytics")
SCHEMA = os.getenv("GEX_SCHEMA", "default")
T_OPPS = f"{CATALOG}.{SCHEMA}.gex_opportunities"
T_LATEST = f"{CATALOG}.{SCHEMA}.gex_latest"
T_UNIV = f"{CATALOG}.{SCHEMA}.gex_universe"

@st.cache_data(ttl=60)
def load_live_opportunities(min_conf: int = 65) -> pd.DataFrame:
    if not DBSQL_AVAILABLE:
        raise RuntimeError("Databricks SQL connector not installed")
    conn = sql.connect(
        server_hostname=os.getenv("DBSQL_HOST"),
        http_path=os.getenv("DBSQL_HTTP_PATH"),
        access_token=os.getenv("DBSQL_TOKEN"),
    )
    q = f"""
      WITH latest AS (
        SELECT symbol, price, gex_flip_price, distance_to_flip, distance_to_flip_pct, analysis_timestamp
        FROM {T_LATEST}
      )
      SELECT o.symbol, o.category, o.priority, o.structure_type, o.confidence_score,
             o.recommendation, o.spot_price, o.gamma_flip_point, o.distance_to_flip,
             o.distance_to_flip_pct, o.analysis_timestamp
      FROM {T_OPPS} o
      INNER JOIN latest l ON l.symbol = o.symbol
      QUALIFY ROW_NUMBER() OVER (PARTITION BY o.symbol ORDER BY o.analysis_timestamp DESC) = 1
      WHERE o.confidence_score >= {int(min_conf)}
      ORDER BY o.confidence_score DESC, o.analysis_timestamp DESC
      LIMIT 200
    """
    with conn.cursor() as c:
        c.execute(q)
        rows = c.fetchall()
        cols = [d[0] for d in c.description]
    conn.close()
    return pd.DataFrame(rows, columns=cols)

@st.cache_data(ttl=60)
def load_universe() -> pd.DataFrame:
    if not DBSQL_AVAILABLE:
        raise RuntimeError("Databricks SQL connector not installed")
    conn = sql.connect(
        server_hostname=os.getenv("DBSQL_HOST"),
        http_path=os.getenv("DBSQL_HTTP_PATH"),
        access_token=os.getenv("DBSQL_TOKEN"),
    )
    q = f"SELECT symbol, sector, categories, priority, last_updated FROM {T_UNIV}"
    with conn.cursor() as c:
        c.execute(q)
        rows = c.fetchall()
        cols = [d[0] for d in c.description]
    conn.close()
    return pd.DataFrame(rows, columns=cols)

# ===== KEEP: All your CSS (unchanged, ~600 lines) =====
# (paste your full CSS block here — already in your v3.0.0 file)

# ===== Session State Initialization =====
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = {
        'positions': [], 'cash': 100000, 'total_value': 100000,
        'daily_pnl': 0, 'trade_history': []
    }

if 'all_gex_data' not in st.session_state:
    st.session_state.all_gex_data = {}

if 'all_setups' not in st.session_state:
    st.session_state.all_setups = []

if 'last_update' not in st.session_state:
    st.session_state.last_update = None

if 'alerts' not in st.session_state:
    st.session_state.alerts = []

if 'watchlist' not in st.session_state:
    st.session_state.watchlist = ["SPY", "QQQ", "IWM", "DIA"]

# ===== Core Classes (GEXCalculator + TradeSetupDetector) =====
# (paste your full GEXCalculator and TradeSetupDetector classes here — unchanged)
# ================= MAIN DASHBOARD =================
def main():
    # Animated Header with live indicator
    col1, col2, col3 = st.columns([1, 6, 1])
    with col2:
        st.markdown("""
        <h1 style='text-align: center; font-size: 48px; margin-bottom: 0;'>
            <span class='live-indicator'></span>
            GEX Trading Dashboard Pro
        </h1>
        <p style='text-align: center; color: rgba(255,255,255,0.7); font-size: 18px; margin-top: 10px;'>
            Real-time Multi-Symbol Gamma Exposure Analysis & Trade Detection
        </p>
        """, unsafe_allow_html=True)

    # Sidebar configuration
    with st.sidebar:
        st.markdown("<h2 style='text-align: center;'>⚙️ Control Panel</h2>", unsafe_allow_html=True)

        # Watchlist
        st.markdown("### 📊 Watchlist Management")
        popular_symbols = ["SPY", "QQQ", "IWM", "DIA", "AAPL", "TSLA", "NVDA", 
                          "AMD", "META", "AMZN", "GOOGL", "MSFT", "NFLX", "BA",
                          "JPM", "GS", "XOM", "CVX", "PFE", "JNJ"]

        selected_symbols = st.multiselect(
            "Select Symbols to Monitor",
            options=popular_symbols,
            default=st.session_state.watchlist
        )
        custom_symbols = st.text_input("Add Custom Symbols", placeholder="SYMBOL1, SYMBOL2, ...")
        if custom_symbols:
            selected_symbols.extend([s.strip().upper() for s in custom_symbols.split(",") if s.strip()])
        st.session_state.watchlist = list(set(selected_symbols))

        # Show watchlist
        st.markdown("### 👁️ Active Watchlist")
        cols = st.columns(3)
        for i, symbol in enumerate(st.session_state.watchlist):
            with cols[i % 3]:
                st.markdown(f"<div class='symbol-card'><div style='font-weight:600;color:#00D2FF;'>{symbol}</div></div>", unsafe_allow_html=True)

        st.divider()

        # NEW: Databricks toggle
        use_live = st.checkbox("Use Databricks SQL live feed", value=False)

        # Auto refresh
        st.markdown("### 🔄 Auto Refresh")
        auto_refresh = st.checkbox("Enable Auto Refresh", value=False)
        if auto_refresh:
            refresh_interval = st.slider("Interval (minutes)", 1, 30, 5)

        st.divider()

        # Portfolio
        st.markdown("### 💼 Portfolio Overview")
        st.markdown(f"""
        <div class='info-box'>
            <div style='display:flex;justify-content:space-between;'><span>Cash</span><span>${st.session_state.portfolio['cash']:,.0f}</span></div>
            <div style='display:flex;justify-content:space-between;'><span>Total Value</span><span>${st.session_state.portfolio['total_value']:,.0f}</span></div>
            <div style='display:flex;justify-content:space-between;'><span>Daily P&L</span>
            <span style='color:{"#38ef7d" if st.session_state.portfolio["daily_pnl"]>=0 else "#f45c43"};'>${st.session_state.portfolio['daily_pnl']:+,.0f}</span></div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # Risk
        st.markdown("### 🎚️ Risk Management")
        max_position = st.slider("Max Position Size %", 1, 10, 5)
        max_loss = st.slider("Max Loss per Trade %", 1, 5, 3)
        confidence_threshold = st.slider("Min Confidence %", 50, 90, 65)

        st.divider()
        if st.button("🚀 Scan All Symbols", type="primary", use_container_width=True):
            st.session_state.last_update = datetime.now()
            st.rerun()

    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🎯 Top Opportunities", "📊 GEX Analysis", "💎 Trade Setups",
        "📈 Positions", "⚠️ Alerts", "📉 Performance", "🔍 Strategy Guide"
    ])

    # ==== Tab 1 ====
    with tab1:
        st.markdown("## 🏆 Top Trading Opportunities")

        if use_live:
            try:
                live_df = load_live_opportunities(confidence_threshold)
                if not live_df.empty:
                    st.success(f"✅ Showing {len(live_df)} live opportunities from Databricks")
                    for _, r in live_df.head(10).iterrows():
                        with st.expander(f"{r['symbol']} — {r['structure_type']} — {int(r['confidence_score'])}%"):
                            c1,c2,c3,c4 = st.columns(4)
                            c1.metric("Spot", f"${r['spot_price']:.2f}")
                            c2.metric("Flip", f"${r['gamma_flip_point']:.2f}", delta=f"{r['distance_to_flip_pct']:+.2f}%")
                            c3.metric("Priority", f"P{int(r['priority'])}")
                            c4.metric("Confidence", f"{int(r['confidence_score'])}%")
                            st.write(f"**Category**: {r['category']}  |  **Recommendation**: {r['recommendation']}")
                else:
                    st.info("No live Databricks results available.")
            except Exception as e:
                st.error(f"Live feed unavailable: {e}")
        else:
            st.warning("📡 Running local scan with yfinance...")

            all_setups = []
            all_gex_data = {}
            scan_results = []
            progress_bar = st.progress(0)
            status_text = st.empty()

            for idx, symbol in enumerate(st.session_state.watchlist):
                status_text.text(f"Analyzing {symbol}...")
                progress_bar.progress((idx+1)/len(st.session_state.watchlist))
                gex_calc = GEXCalculator(symbol)
                if gex_calc.fetch_options_data():
                    gex_calc.calculate_gamma_exposure()
                    all_gex_data[symbol] = gex_calc
                    detector = TradeSetupDetector(gex_calc)
                    setups = detector.detect_all_setups()
                    for s in setups: s['symbol'] = symbol
                    all_setups.extend(setups)
                    scan_results.append({
                        'symbol': symbol, 'spot': gex_calc.spot_price,
                        'net_gex': gex_calc.net_gex, 'gamma_flip': gex_calc.gamma_flip,
                        'setup_count': len(setups)
                    })
            progress_bar.empty()
            status_text.empty()

            all_setups.sort(key=lambda x: x['confidence'], reverse=True)
            st.session_state.all_setups = all_setups
            st.session_state.all_gex_data = all_gex_data

            # Show results (your existing metrics + expander cards — unchanged)
            # ... (retain your v3.0.0 Tab1 display code here)

    # ==== Tab 2 ====
    with tab2:
        st.markdown("## 📊 Detailed GEX Analysis")
        if st.session_state.all_gex_data:
            selected_symbol = st.selectbox("Select Symbol", options=list(st.session_state.all_gex_data.keys()))
            if selected_symbol in st.session_state.all_gex_data:
                gex = st.session_state.all_gex_data[selected_symbol]
                # ... (retain your full v3.0.0 tab2 chart + regime analysis code here)

    # ==== Tab 3 ====
    with tab3:
        st.markdown("## 💎 All Trade Setups")
        if st.session_state.all_setups:
            strategy_groups = {}
            for setup in st.session_state.all_setups:
                strategy_groups.setdefault(setup['strategy'], []).append(setup)
            for strategy, setups in strategy_groups.items():
                st.markdown(f"### {strategy}")
                for setup in setups[:5]:
                    if setup['confidence'] >= confidence_threshold:
                        st.markdown(f"""
                        <div class='trade-setup-card'>
                            <div style='display:flex;justify-content:space-between;'>
                                <h4>{setup['symbol']}</h4>
                                <span class='confidence-badge {"confidence-high" if setup["confidence"] > 80 else "confidence-medium" if setup["confidence"] > 70 else "confidence-low"}'>
                                    {setup['confidence']:.0f}% Confidence
                                </span>
                            </div>
                            <p>{setup['description']}</p>
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.info("No setups found yet.")
              # ==== Tab 4 ====
    with tab4:
        st.markdown("## 📈 Portfolio & Position Management")
        col1,col2,col3,col4 = st.columns(4)
        st.metric("Open Positions", len(st.session_state.portfolio['positions']))
        total_exposure = sum([p['value'] for p in st.session_state.portfolio['positions']]) if st.session_state.portfolio['positions'] else 0
        st.metric("Total Exposure", f"${total_exposure:,.0f}")
        utilization = (total_exposure / st.session_state.portfolio['total_value'] * 100) if st.session_state.portfolio['total_value'] > 0 else 0
        st.metric("Capital Utilization", f"{utilization:.1f}%")
        st.metric("Daily P&L", f"${st.session_state.portfolio['daily_pnl']:+,.0f}")
        if st.session_state.portfolio['positions']:
            st.markdown("### Active Positions")
            st.dataframe(pd.DataFrame(st.session_state.portfolio['positions']), use_container_width=True)
        else:
            st.info("No active positions.")

    # ==== Tab 5 ====
    with tab5:
        st.markdown("## ⚠️ Trading Alerts")
        alerts = []
        for symbol,gex in st.session_state.all_gex_data.items():
            if gex.net_gex and gex.net_gex < -1e9:
                alerts.append({'priority':'HIGH','symbol':symbol,'type':'NEGATIVE_GEX','message':f'{symbol}: Net GEX {gex.net_gex/1e9:.2f}B','action':'Consider long vol'})
            if gex.gamma_flip and gex.spot_price:
                distance = abs(gex.spot_price - gex.gamma_flip)/gex.spot_price*100
                if distance < 0.5:
                    alerts.append({'priority':'HIGH','symbol':symbol,'type':'NEAR_FLIP','message':f'{symbol}: Within {distance:.2f}% of flip','action':'Volatility regime change'})
        if alerts:
            for alert in alerts:
                st.markdown(f"<div class='alert-high'><b>{alert['symbol']} {alert['type']}</b><br>{alert['message']}<br><i>{alert['action']}</i></div>", unsafe_allow_html=True)
        else:
            st.success("✅ No active alerts.")

    # ==== Tab 6 ====
    with tab6:
        st.markdown("## 📉 Performance Analytics")
        if st.session_state.portfolio['trade_history']:
            trades_df = pd.DataFrame(st.session_state.portfolio['trade_history'])
            total_trades = len(trades_df)
            wins = len(trades_df[trades_df['pnl']>0]) if 'pnl' in trades_df.columns else 0
            win_rate = wins/total_trades*100 if total_trades>0 else 0
            st.metric("Total Trades", total_trades)
            st.metric("Win Rate", f"{win_rate:.1f}%")
            if 'pnl' in trades_df.columns:
                trades_df['cumulative_pnl']=trades_df['pnl'].cumsum()
                st.line_chart(trades_df['cumulative_pnl'])
        else:
            st.info("No trading history yet.")

    # ==== Tab 7 ====
    with tab7:
        st.markdown("## 🔍 Strategy Guide & Education")
        option = st.selectbox("Select Topic", ["Quick Start","Squeeze Plays","Premium Selling","Iron Condors","Risk Management","GEX Fundamentals","Market Regimes"])
        if option=="Quick Start":
            st.write("How to use the dashboard...")
        elif option=="Squeeze Plays":
            st.write("Squeeze play strategies...")
        elif option=="Premium Selling":
            st.write("Premium selling strategies...")
        elif option=="Iron Condors":
            st.write("Iron condor strategies...")
        elif option=="Risk Management":
            st.write("Risk management rules...")
        elif option=="GEX Fundamentals":
            st.write("Gamma exposure basics...")
        elif option=="Market Regimes":
            st.write("Market regime playbook...")

    # ==== Footer ====
    st.divider()
    col1,col2,col3 = st.columns([1,2,1])
    with col2:
        update_time = st.session_state.last_update.strftime('%Y-%m-%d %H:%M:%S') if st.session_state.last_update else "Never"
        st.markdown(f"<div style='text-align:center;color:rgba(255,255,255,0.6)'>Last Updated: {update_time}</div>", unsafe_allow_html=True)

    # Auto refresh
    if auto_refresh:
        time_module.sleep(refresh_interval*60)
        st.rerun()

if __name__ == "__main__":
    main()


