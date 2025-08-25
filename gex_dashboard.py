"""
GEX Trading Dashboard - Multi-Symbol Gamma Exposure Analysis Platform
Author: GEX Trading System
Version: 3.1.1
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime
import warnings
import time as time_module
import logging

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

if 'watchlist' not in st.session_state:
    st.session_state.watchlist = ["SPY", "QQQ", "IWM", "DIA"]

# ===== Core Classes =====
class GEXCalculator:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.spot_price = None
        self.options_chain = None
        self.gex_profile = None
        self.gamma_flip = None
        self.net_gex = None
    def fetch_options_data(self):
        try:
            ticker = yf.Ticker(self.symbol)
            hist = ticker.history(period='1d')
            if not hist.empty:
                self.spot_price = hist['Close'].iloc[-1]
            return True
        except:
            return False
    def calculate_gamma_exposure(self):
        self.net_gex = np.random.randint(-2e9, 2e9)
        self.gamma_flip = self.spot_price * (1 + np.random.uniform(-0.02,0.02))
        return pd.DataFrame()

class TradeSetupDetector:
    def __init__(self, gex_calc: GEXCalculator):
        self.gex = gex_calc
    def detect_all_setups(self):
        return [{
            'symbol': self.gex.symbol,
            'strategy': 'Sample Strategy',
            'description': 'Placeholder setup',
            'confidence': 75
        }]

# ================= MAIN DASHBOARD =================
def main():
    col1,col2,col3 = st.columns([1,6,1])
    with col2:
        st.markdown("""
        <h1 style='text-align:center;'>🚀 GEX Trading Dashboard Pro</h1>
        <p style='text-align:center;color:rgba(255,255,255,0.7);'>Real-time Multi-Symbol Gamma Exposure Analysis & Trade Detection</p>
        """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### ⚙️ Control Panel")

        # Watchlist
        st.markdown("### 📊 Watchlist Management")
        popular_symbols = ["SPY","QQQ","IWM","DIA","AAPL","TSLA","NVDA","AMD","META","AMZN","GOOGL","MSFT","NFLX","BA","JPM","GS","XOM","CVX","PFE","JNJ"]

        # ✅ FIX: merge popular + saved watchlist to avoid crash
        all_options = sorted(list(set(popular_symbols + st.session_state.watchlist)))

        selected_symbols = st.multiselect(
            "Select Symbols to Monitor",
            options=all_options,
            default=st.session_state.watchlist
        )

        custom_symbols = st.text_input("Add Custom Symbols", placeholder="SYMBOL1, SYMBOL2")
        if custom_symbols:
            selected_symbols.extend([s.strip().upper() for s in custom_symbols.split(",") if s.strip()])
        st.session_state.watchlist = list(set(selected_symbols))

        st.markdown("### 👁️ Active Watchlist")
        st.write(", ".join(st.session_state.watchlist))

        st.divider()
        use_live = st.checkbox("Use Databricks SQL live feed", value=False)

        st.divider()
        st.markdown("### 💼 Portfolio")
        st.write(f"Cash: ${st.session_state.portfolio['cash']:,}")

    # Tabs
    tab1,tab2,tab3,tab4,tab5,tab6,tab7 = st.tabs(["🎯 Top Opportunities","📊 GEX Analysis","💎 Trade Setups","📈 Positions","⚠️ Alerts","📉 Performance","🔍 Strategy Guide"])

    with tab1:
        if use_live:
            try:
                df = load_live_opportunities(65)
                st.write("✅ Live results", df.head())
            except Exception as e:
                st.error(f"Live feed error: {e}")
        else:
            st.warning("📡 Local scan running...")
            all_setups=[]
            for sym in st.session_state.watchlist:
                gex = GEXCalculator(sym)
                if gex.fetch_options_data():
                    gex.calculate_gamma_exposure()
                    setups = TradeSetupDetector(gex).detect_all_setups()
                    all_setups.extend(setups)
            st.session_state.all_setups = all_setups
            st.write(all_setups)

    with tab2:
        st.write("Detailed analysis placeholder")

    with tab3:
        st.write(st.session_state.all_setups)

    with tab4:
        st.write("Positions placeholder")

    with tab5:
        st.write("Alerts placeholder")

    with tab6:
        st.write("Performance placeholder")

    with tab7:
        st.write("Strategy guide placeholder")

if __name__=="__main__":
    main()
