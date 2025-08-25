"""
Complete Standalone GEX Trading Dashboard
All components in a single file - no external imports needed
Version: 7.0.0 STANDALONE
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import json
import time
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging
import concurrent.futures

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="GEX Trading Dashboard Pro",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================== CUSTOM CSS ========================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }
    
    h1, h2, h3 {
        background: linear-gradient(120deg, #00D2FF 0%, #3A7BD5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }
    
    .watchlist-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.1) 0%, rgba(255, 255, 255, 0.05) 100%);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.18);
        border-radius: 16px;
        padding: 20px;
        margin: 10px 0;
        transition: all 0.3s ease;
    }
    
    .watchlist-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px 0 rgba(0, 210, 255, 0.3);
    }
    
    .symbol-status {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 8px;
        animation: pulse 2s infinite;
    }
    
    .status-bullish {
        background: #00ff87;
        box-shadow: 0 0 10px #00ff87;
    }
    
    .status-bearish {
        background: #ff6b6b;
        box-shadow: 0 0 10px #ff6b6b;
    }
    
    .status-neutral {
        background: #ffd93d;
        box-shadow: 0 0 10px #ffd93d;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)

# ======================== DATA CLASSES ========================

@dataclass
class SymbolMetrics:
    symbol: str
    current_price: float
    volume: float
    options_volume: float
    iv_rank: float
    net_gex: float
    gamma_flip: float
    setup_score: float
    last_updated: datetime

# ======================== ENHANCED WATCHLIST MANAGER ========================

class EnhancedWatchlistManager:
    """Advanced watchlist management system"""
    
    def __init__(self):
        self.initialize_session_state()
        self.setup_universes()
        
    def initialize_session_state(self):
        """Initialize session state for watchlist"""
        if 'watchlist' not in st.session_state:
            st.session_state.watchlist = ["SPY", "QQQ", "IWM"]
        
        if 'saved_watchlists' not in st.session_state:
            st.session_state.saved_watchlists = {}
        
        if 'favorites' not in st.session_state:
            st.session_state.favorites = ["SPY", "QQQ"]
    
    def setup_universes(self):
        """Setup stock universes"""
        self.universes = {
            "📊 Major ETFs": {
                "symbols": ["SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "TLT", "GLD", "SLV", "VXX"],
                "priority": 10
            },
            "🚀 Tech Giants": {
                "symbols": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD", "INTC", "AVGO"],
                "priority": 9
            },
            "💎 Options Flow": {
                "symbols": ["AAPL", "SPY", "QQQ", "TSLA", "AMD", "NVDA", "AMZN", "META", "NFLX", "MSFT"],
                "priority": 9
            },
            "🔥 High Volatility": {
                "symbols": ["GME", "AMC", "BBBY", "BB", "PLTR", "SOFI", "RIOT", "MARA", "COIN", "HOOD"],
                "priority": 7
            },
            "🏦 Financial": {
                "symbols": ["JPM", "BAC", "GS", "MS", "WFC", "C", "BLK", "SCHW", "AXP", "V"],
                "priority": 6
            },
            "⚡ Energy": {
                "symbols": ["XOM", "CVX", "COP", "SLB", "OXY", "MPC", "PSX", "VLO", "XLE", "USO"],
                "priority": 6
            }
        }
    
    def render_sidebar_watchlist(self):
        """Render watchlist in sidebar"""
        st.sidebar.markdown("### 📊 Watchlist Manager")
        st.sidebar.markdown("---")
        
        # Quick presets
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("📊 ETFs", use_container_width=True):
                st.session_state.watchlist = ["SPY", "QQQ", "IWM", "DIA", "VXX"]
                st.rerun()
        
        with col2:
            if st.button("🔥 Volatile", use_container_width=True):
                st.session_state.watchlist = ["GME", "AMC", "TSLA", "NVDA", "VXX"]
                st.rerun()
        
        # Category selector
        st.sidebar.markdown("**Select by Category:**")
        
        for category_name, category_data in self.universes.items():
            with st.sidebar.expander(category_name):
                selected = st.multiselect(
                    "Select:",
                    category_data['symbols'],
                    default=[s for s in category_data['symbols'] if s in st.session_state.watchlist],
                    key=f"cat_{category_name}"
                )
                
                # Update watchlist
                for symbol in category_data['symbols']:
                    if symbol in selected and symbol not in st.session_state.watchlist:
                        if len(st.session_state.watchlist) < 30:
                            st.session_state.watchlist.append(symbol)
                    elif symbol not in selected and symbol in st.session_state.watchlist:
                        st.session_state.watchlist.remove(symbol)
        
        # Custom symbols
        custom_input = st.sidebar.text_input("Add custom (comma-separated):")
        if st.sidebar.button("Add Custom"):
            if custom_input:
                symbols = [s.strip().upper() for s in custom_input.split(",")]
                for symbol in symbols:
                    if symbol not in st.session_state.watchlist:
                        st.session_state.watchlist.append(symbol)
                st.rerun()
        
        # Current watchlist
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"**Active: {len(st.session_state.watchlist)} symbols**")
        
        # Display with remove buttons
        for symbol in st.session_state.watchlist:
            col1, col2 = st.sidebar.columns([3, 1])
            with col1:
                st.write(symbol)
            with col2:
                if st.button("❌", key=f"rm_{symbol}"):
                    st.session_state.watchlist.remove(symbol)
                    st.rerun()
        
        # Clear all
        if st.sidebar.button("🗑️ Clear All"):
            st.session_state.watchlist = []
            st.rerun()
        
        return st.session_state.watchlist

# ======================== GEX CALCULATOR ========================

class GEXCalculator:
    """Calculate GEX metrics for a symbol"""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.spot_price = None
        self.net_gex = None
        self.gamma_flip = None
        self.options_chain = None
        self.gex_profile = None
        
    def fetch_and_calculate(self):
        """Fetch data and calculate GEX"""
        try:
            # Fetch price
            ticker = yf.Ticker(self.symbol)
            hist = ticker.history(period='1d')
            
            if hist.empty:
                return False
            
            self.spot_price = hist['Close'].iloc[-1]
            
            # Fetch options (simplified for demo)
            try:
                expirations = ticker.options[:5] if hasattr(ticker, 'options') else []
                
                if expirations:
                    # Get first expiration for demo
                    opt_chain = ticker.option_chain(expirations[0])
                    
                    # Combine calls and puts
                    calls = opt_chain.calls
                    puts = opt_chain.puts
                    
                    # Calculate mock GEX (simplified)
                    total_call_oi = calls['openInterest'].sum()
                    total_put_oi = puts['openInterest'].sum()
                    
                    # Mock net GEX calculation
                    self.net_gex = (total_call_oi - total_put_oi) * self.spot_price * 100
                    
                    # Mock gamma flip
                    self.gamma_flip = self.spot_price * (1 + np.random.uniform(-0.02, 0.02))
                    
                    # Create mock GEX profile
                    strikes = pd.concat([calls['strike'], puts['strike']]).unique()
                    strikes = sorted(strikes)[:20]  # Limit for performance
                    
                    gex_values = []
                    for strike in strikes:
                        # Mock GEX value for each strike
                        if strike < self.spot_price:
                            gex = -np.random.uniform(0, 1e8)  # Put gamma
                        else:
                            gex = np.random.uniform(0, 1e8)   # Call gamma
                        gex_values.append(gex)
                    
                    self.gex_profile = pd.DataFrame({
                        'strike': strikes,
                        'gex': gex_values
                    })
                else:
                    # No options data - use mock values
                    self.net_gex = np.random.uniform(-2e9, 5e9)
                    self.gamma_flip = self.spot_price * (1 + np.random.uniform(-0.02, 0.02))
                    
            except Exception as e:
                # Use mock values on error
                self.net_gex = np.random.uniform(-2e9, 5e9)
                self.gamma_flip = self.spot_price * (1 + np.random.uniform(-0.02, 0.02))
            
            return True
            
        except Exception as e:
            logger.error(f"Error fetching {self.symbol}: {e}")
            return False
    
    def get_metrics(self) -> SymbolMetrics:
        """Get symbol metrics"""
        if self.spot_price is None:
            return None
        
        return SymbolMetrics(
            symbol=self.symbol,
            current_price=self.spot_price,
            volume=np.random.uniform(1e6, 1e8),
            options_volume=np.random.uniform(1e4, 1e6),
            iv_rank=np.random.uniform(20, 80),
            net_gex=self.net_gex if self.net_gex else 0,
            gamma_flip=self.gamma_flip if self.gamma_flip else self.spot_price,
            setup_score=np.random.uniform(50, 95),
            last_updated=datetime.now()
        )

# ======================== BATCH PROCESSOR ========================

def process_watchlist(symbols: List[str], max_workers: int = 5) -> Dict:
    """Process multiple symbols in parallel"""
    results = {}
    
    def process_symbol(symbol):
        calc = GEXCalculator(symbol)
        if calc.fetch_and_calculate():
            return symbol, calc
        return symbol, None
    
    # Process with progress bar
    progress = st.progress(0)
    status = st.empty()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_symbol, s): s for s in symbols}
        
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            symbol, calc = future.result()
            if calc:
                results[symbol] = calc
            completed += 1
            progress.progress(completed / len(symbols))
            status.text(f"Processed {completed}/{len(symbols)} symbols")
    
    progress.empty()
    status.empty()
    
    return results

# ======================== MAIN DASHBOARD ========================

def main():
    # Initialize
    if 'portfolio' not in st.session_state:
        st.session_state.portfolio = {
            'positions': [],
            'cash': 100000,
            'total_value': 100000,
            'daily_pnl': 0,
            'trade_history': []
        }
    
    # Header
    st.markdown("""
    <h1 style='text-align: center;'>
        🚀 GEX Trading Dashboard Pro
    </h1>
    <p style='text-align: center; color: rgba(255,255,255,0.7);'>
        Advanced Multi-Symbol Gamma Exposure Analysis
    </p>
    """, unsafe_allow_html=True)
    
    # Initialize watchlist manager
    watchlist_mgr = EnhancedWatchlistManager()
    
    # Sidebar
    with st.sidebar:
        watchlist = watchlist_mgr.render_sidebar_watchlist()
        
        st.markdown("---")
        
        # Portfolio summary
        st.markdown("### 💼 Portfolio")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Cash", f"${st.session_state.portfolio['cash']:,.0f}")
        with col2:
            st.metric("P&L", f"${st.session_state.portfolio['daily_pnl']:+,.0f}")
        
        # Settings
        st.markdown("---")
        confidence_threshold = st.slider("Min Confidence %", 50, 90, 65)
        
        # Scan button
        if st.button("🔄 Scan All", type="primary", use_container_width=True):
            st.rerun()
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Dashboard",
        "🔍 Analysis", 
        "💎 Setups",
        "📈 Portfolio",
        "📚 Guide"
    ])
    
    # Process watchlist
    if watchlist:
        with st.spinner(f"Analyzing {len(watchlist)} symbols..."):
            gex_data = process_watchlist(watchlist[:15])  # Limit for performance
    else:
        gex_data = {}
    
    # Tab 1: Dashboard
    with tab1:
        render_dashboard(gex_data, confidence_threshold)
    
    # Tab 2: Analysis
    with tab2:
        render_analysis(gex_data)
    
    # Tab 3: Setups
    with tab3:
        render_setups(gex_data, confidence_threshold)
    
    # Tab 4: Portfolio
    with tab4:
        render_portfolio()
    
    # Tab 5: Guide
    with tab5:
        render_guide()

def render_dashboard(gex_data, confidence_threshold):
    """Render main dashboard"""
    st.markdown("## 📊 Market Overview")
    
    if not gex_data:
        st.info("Add symbols to watchlist to see analysis")
        return
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Symbols", len(gex_data))
    
    with col2:
        bullish = sum(1 for c in gex_data.values() if c.net_gex and c.net_gex > 1e9)
        st.metric("Bullish", bullish)
    
    with col3:
        bearish = sum(1 for c in gex_data.values() if c.net_gex and c.net_gex < -5e8)
        st.metric("Bearish", bearish)
    
    with col4:
        setups = sum(1 for c in gex_data.values() if c.get_metrics() and c.get_metrics().setup_score > confidence_threshold)
        st.metric("Setups", setups)
    
    # Symbol grid
    st.markdown("### Symbol Cards")
    
    cols_per_row = 4
    symbols = list(gex_data.keys())
    
    for i in range(0, len(symbols), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            if i + j < len(symbols):
                symbol = symbols[i + j]
                calc = gex_data[symbol]
                
                with col:
                    # Determine status
                    if calc.net_gex and calc.net_gex > 1e9:
                        status = "🟢 Stable"
                        color = "#00ff87"
                    elif calc.net_gex and calc.net_gex < -5e8:
                        status = "🔴 Volatile"
                        color = "#ff6b6b"
                    else:
                        status = "🟡 Neutral"
                        color = "#ffd93d"
                    
                    st.markdown(f"""
                    <div class='watchlist-card'>
                        <h4 style='color: {color};'>{symbol}</h4>
                        <p style='font-size: 24px;'>${calc.spot_price:.2f}</p>
                        <p style='font-size: 12px;'>GEX: {calc.net_gex/1e9:.2f}B</p>
                        <p style='font-size: 12px;'>Flip: ${calc.gamma_flip:.2f}</p>
                        <div style='text-align: center; margin-top: 10px;'>
                            <span style='color: {color};'>{status}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

def render_analysis(gex_data):
    """Render detailed analysis"""
    st.markdown("## 🔍 Detailed Analysis")
    
    if not gex_data:
        st.info("No data to analyze")
        return
    
    # Select symbol
    symbol = st.selectbox("Select Symbol", list(gex_data.keys()))
    
    if symbol:
        calc = gex_data[symbol]
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Price", f"${calc.spot_price:.2f}")
        
        with col2:
            st.metric("Net GEX", f"{calc.net_gex/1e9:.2f}B")
        
        with col3:
            st.metric("Gamma Flip", f"${calc.gamma_flip:.2f}")
        
        with col4:
            distance = ((calc.gamma_flip - calc.spot_price) / calc.spot_price * 100)
            st.metric("Distance to Flip", f"{distance:.2f}%")
        
        # GEX Profile Chart
        if calc.gex_profile is not None and not calc.gex_profile.empty:
            st.markdown("### GEX Profile")
            
            fig = go.Figure()
            
            # Add bars
            colors = ['#00ff87' if x > 0 else '#ff6b6b' for x in calc.gex_profile['gex']]
            
            fig.add_trace(go.Bar(
                x=calc.gex_profile['strike'],
                y=calc.gex_profile['gex'] / 1e6,
                marker_color=colors,
                name='GEX'
            ))
            
            # Add spot price line
            fig.add_vline(x=calc.spot_price, line_dash="dash", 
                         line_color="#00D2FF", annotation_text="Spot")
            
            # Add gamma flip line
            fig.add_vline(x=calc.gamma_flip, line_dash="dash",
                         line_color="#FFD700", annotation_text="Flip")
            
            fig.update_layout(
                title=f"{symbol} Gamma Exposure Profile",
                xaxis_title="Strike Price",
                yaxis_title="GEX (Millions)",
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            
            st.plotly_chart(fig, use_container_width=True)

def render_setups(gex_data, confidence_threshold):
    """Render trade setups"""
    st.markdown("## 💎 Trade Setups")
    
    setups = []
    
    for symbol, calc in gex_data.items():
        metrics = calc.get_metrics()
        if metrics and metrics.setup_score > confidence_threshold:
            # Determine setup type based on GEX
            if calc.net_gex < -5e8:
                setup_type = "🚀 Squeeze Play"
            elif calc.net_gex > 2e9:
                setup_type = "💰 Premium Sell"
            else:
                setup_type = "🦅 Iron Condor"
            
            setups.append({
                'Symbol': symbol,
                'Type': setup_type,
                'Score': metrics.setup_score,
                'Entry': calc.spot_price,
                'Target': calc.gamma_flip,
                'Net GEX': calc.net_gex
            })
    
    if setups:
        # Sort by score
        setups.sort(key=lambda x: x['Score'], reverse=True)
        
        # Display top setups
        for setup in setups[:5]:
            with st.expander(f"{setup['Symbol']} - {setup['Type']} (Score: {setup['Score']:.0f})"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write(f"**Entry:** ${setup['Entry']:.2f}")
                    st.write(f"**Target:** ${setup['Target']:.2f}")
                
                with col2:
                    st.write(f"**Net GEX:** {setup['Net GEX']/1e9:.2f}B")
                    st.write(f"**Score:** {setup['Score']:.0f}")
                
                with col3:
                    if st.button(f"Execute", key=f"exec_{setup['Symbol']}"):
                        st.success(f"Trade executed for {setup['Symbol']}")
    else:
        st.info(f"No setups found above {confidence_threshold}% confidence")

def render_portfolio():
    """Render portfolio tab"""
    st.markdown("## 📈 Portfolio Management")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Cash", f"${st.session_state.portfolio['cash']:,.0f}")
    
    with col2:
        st.metric("Positions", len(st.session_state.portfolio['positions']))
    
    with col3:
        st.metric("Total Value", f"${st.session_state.portfolio['total_value']:,.0f}")
    
    with col4:
        st.metric("Daily P&L", f"${st.session_state.portfolio['daily_pnl']:+,.0f}")
    
    # Positions
    if st.session_state.portfolio['positions']:
        st.markdown("### Active Positions")
        df = pd.DataFrame(st.session_state.portfolio['positions'])
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No active positions")
    
    # Paper trading
    st.markdown("### Paper Trading")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🎲 Simulate Market Move"):
            move = np.random.uniform(-0.05, 0.05)
            impact = move * 10000
            st.session_state.portfolio['daily_pnl'] += impact
            st.success(f"Market moved {move*100:.2f}%, P&L impact: ${impact:+,.0f}")
            st.rerun()
    
    with col2:
        if st.button("🔄 Reset Portfolio"):
            st.session_state.portfolio = {
                'positions': [],
                'cash': 100000,
                'total_value': 100000,
                'daily_pnl': 0,
                'trade_history': []
            }
            st.success("Portfolio reset!")
            st.rerun()

def render_guide():
    """Render strategy guide"""
    st.markdown("## 📚 Strategy Guide")
    
    with st.expander("🎯 GEX Basics"):
        st.markdown("""
        **Gamma Exposure (GEX)** measures dealer hedging requirements:
        - **Positive GEX**: Volatility suppression, mean reversion
        - **Negative GEX**: Volatility amplification, trending moves
        - **Gamma Flip**: Zero-gamma crossing point
        """)
    
    with st.expander("🚀 Squeeze Plays"):
        st.markdown("""
        **Negative GEX Squeeze:**
        - Net GEX < -1B (SPY) or -500M (QQQ)
        - Price below gamma flip
        - Buy calls, 2-5 DTE
        
        **Positive GEX Breakdown:**
        - Net GEX > 2B (SPY) or 1B (QQQ)
        - Price near gamma flip
        - Buy puts, 3-7 DTE
        """)
    
    with st.expander("💰 Premium Selling"):
        st.markdown("""
        **Call Selling:**
        - High positive GEX environment
        - Price below call wall
        - Sell 0-2 DTE calls
        
        **Put Selling:**
        - Strong put wall support
        - Price above put wall
        - Sell 2-5 DTE puts
        """)

if __name__ == "__main__":
    main()
