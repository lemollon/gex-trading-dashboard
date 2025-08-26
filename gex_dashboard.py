"""
GEX Trading Command Center - Ultimate Production Dashboard
Real data, beautiful UI, full universe scanning
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import requests
import json
from databricks import sql
import time

# Page configuration
st.set_page_config(
    page_title="🚀 GEX Trading Command Center",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional dark gradient theme with excellent readability
st.markdown("""
<style>
    /* Dark professional gradient background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        background-attachment: fixed;
    }
    
    /* Main container with better contrast */
    .main > div {
        background: rgba(15, 17, 26, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Remove default padding */
    .block-container {
        padding-top: 2rem;
        max-width: 100%;
    }
    
    /* Clean, professional trade cards */
    .trade-setup-card {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 10px 25px rgba(0,0,0,0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
    }
    
    .trade-setup-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.4);
        border-color: #4a9eff;
    }
    
    /* Clean metrics */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, rgba(30, 60, 114, 0.6) 0%, rgba(42, 82, 152, 0.6) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 1rem;
        backdrop-filter: blur(5px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.3);
        border-color: #4a9eff;
    }
    
    /* Clean tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: linear-gradient(90deg, #1e3c72, #2a5298);
        border-radius: 10px;
        padding: 0.5rem;
        gap: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #ffffff !important;
        font-weight: 600;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        transition: all 0.3s;
    }
    
    .stTabs [aria-selected="true"] {
        background: rgba(255, 255, 255, 0.15) !important;
        color: #4a9eff !important;
    }
    
    /* Professional buttons */
    .stButton > button {
        background: linear-gradient(135deg, #1e3c72, #2a5298);
        color: white;
        border: 1px solid #4a9eff;
        padding: 0.5rem 1.5rem;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
        text-transform: none;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(74, 158, 255, 0.3);
        background: linear-gradient(135deg, #2a5298, #1e3c72);
    }
    
    /* Headers with better readability */
    h1, h2, h3 {
        color: #ffffff !important;
        text-shadow: none !important;
    }
    
    /* Ensure text is readable */
    p, span, div, label {
        color: #e0e6f0 !important;
    }
    
    /* Better alerts */
    .alert-hot {
        background: linear-gradient(135deg, #c31432, #f64f59);
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
    }
    
    /* Improved dataframe */
    .dataframe {
        background: rgba(15, 17, 26, 0.9) !important;
        color: #e0e6f0 !important;
        border-radius: 10px;
        overflow: hidden;
    }
    
    .dataframe td, .dataframe th {
        color: #e0e6f0 !important;
        border-color: rgba(255, 255, 255, 0.1) !important;
        padding: 0.75rem !important;
    }
    
    .dataframe th {
        background: rgba(30, 60, 114, 0.5) !important;
        font-weight: 600;
    }
    
    .dataframe tbody tr:hover {
        background: rgba(74, 158, 255, 0.1) !important;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29 0%, #302b63 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Input fields */
    .stTextInput > div > div > input,
    .stSelectbox > div > div > select,
    .stMultiSelect > div > div > div {
        background: rgba(15, 17, 26, 0.7) !important;
        color: #e0e6f0 !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    
    /* Sliders */
    .stSlider > div > div > div {
        background: rgba(74, 158, 255, 0.2) !important;
    }
    
    .stSlider > div > div > div > div {
        background: #4a9eff !important;
    }
    
    /* Remove all unnecessary animations for better performance */
    @keyframes none {
        0% { opacity: 1; }
        100% { opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'selected_symbol' not in st.session_state:
    st.session_state.selected_symbol = None
if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = False

# ========== REAL DATABRICKS CONNECTION ==========
@st.cache_data(ttl=60)  # Refresh every minute
def load_real_databricks_data():
    """Load REAL data from your Databricks pipeline"""
    try:
        connection = sql.connect(
            server_hostname=st.secrets["databricks"]["server_hostname"],
            http_path=st.secrets["databricks"]["http_path"],
            access_token=st.secrets["databricks"]["access_token"]
        )
        
        cursor = connection.cursor()
        
        # Get ALL symbols from your pipeline - not just a few!
        query = """
        SELECT 
            symbol,
            condition_type,
            net_gex,
            distance_to_flip,
            confidence_score,
            position_size_percent,
            expected_move,
            setup_approved,
            action,
            raw_condition,
            run_timestamp,
            gamma_flip,
            call_wall,
            put_wall,
            spot_price
        FROM quant_projects.gex_trading.gex_pipeline_results
        WHERE pipeline_date >= current_date() - INTERVAL 1 DAYS
            AND symbol != 'NO_SETUPS'
        ORDER BY confidence_score DESC, net_gex DESC
        """
        
        cursor.execute(query)
        results = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        
        cursor.close()
        connection.close()
        
        if results:
            df = pd.DataFrame(results, columns=columns)
            return process_pipeline_data(df), "🟢 Live Databricks Connection"
        else:
            return create_demo_universe_data(), "🟡 Using Demo Data (Pipeline Empty)"
            
    except Exception as e:
        st.error(f"Connection Error: {e}")
        return create_demo_universe_data(), "🔴 Using Demo Mode"

def process_pipeline_data(df):
    """Process real pipeline data into trade setups"""
    setups = []
    
    for _, row in df.iterrows():
        # Map conditions to exciting trade types
        trade_type_map = {
            'NEGATIVE_GEX': '🔥 SQUEEZE EXPLOSION',
            'HIGH_POSITIVE_GEX': '💰 PREMIUM HARVEST',
            'NEAR_FLIP': '⚡ VOLATILITY FLIP',
            'COMPRESSION': '💎 BREAKOUT SETUP',
            'CALL_WALL': '🎯 RESISTANCE PLAY',
            'PUT_WALL': '🛡️ SUPPORT BOUNCE'
        }
        
        # Calculate potential profit
        expected_profit = row['expected_move'] * row['position_size_percent'] * 100
        
        setup = {
            'symbol': row['symbol'],
            'type': trade_type_map.get(row['condition_type'], '📊 OPPORTUNITY'),
            'confidence': row['confidence_score'],
            'net_gex': row['net_gex'],
            'entry': row.get('spot_price', 0),
            'target': row.get('spot_price', 0) * (1 + row['expected_move']/100),
            'stop': row.get('spot_price', 0) * 0.98,
            'expected_profit': expected_profit,
            'size': row['position_size_percent'],
            'gamma_flip': row.get('gamma_flip', 0),
            'call_wall': row.get('call_wall', 0),
            'put_wall': row.get('put_wall', 0),
            'action': row.get('action', 'MONITOR'),
            'timestamp': row['run_timestamp']
        }
        setups.append(setup)
    
    return {
        'setups': setups,
        'total_symbols': len(df['symbol'].unique()),
        'high_confidence': len(df[df['confidence_score'] >= 80]),
        'total_opportunities': len(setups)
    }

def create_demo_universe_data():
    """Create demo data with FULL universe of stocks"""
    # Simulate a large universe like your pipeline
    universe = ['SPY', 'QQQ', 'IWM', 'AAPL', 'MSFT', 'NVDA', 'TSLA', 'META', 'GOOGL', 'AMZN',
                'AMD', 'NFLX', 'BA', 'GS', 'JPM', 'BAC', 'XOM', 'CVX', 'PFE', 'JNJ',
                'UNH', 'V', 'MA', 'DIS', 'PYPL', 'SQ', 'SHOP', 'COIN', 'MARA', 'RIOT']
    
    setups = []
    for symbol in universe:
        if np.random.random() > 0.3:  # 70% chance of setup
            confidence = np.random.uniform(60, 95)
            net_gex = np.random.uniform(-2e9, 3e9)
            
            setup_types = ['🔥 SQUEEZE EXPLOSION', '💰 PREMIUM HARVEST', '⚡ VOLATILITY FLIP', 
                          '💎 BREAKOUT SETUP', '🎯 RESISTANCE PLAY', '🛡️ SUPPORT BOUNCE']
            
            setup = {
                'symbol': symbol,
                'type': np.random.choice(setup_types),
                'confidence': confidence,
                'net_gex': net_gex,
                'entry': np.random.uniform(50, 500),
                'target': np.random.uniform(51, 510),
                'stop': np.random.uniform(48, 495),
                'expected_profit': np.random.uniform(100, 5000),
                'size': np.random.uniform(1, 5),
                'gamma_flip': np.random.uniform(50, 500),
                'call_wall': np.random.uniform(52, 520),
                'put_wall': np.random.uniform(48, 490),
                'action': 'BUY' if confidence > 80 else 'MONITOR',
                'timestamp': datetime.now()
            }
            setups.append(setup)
    
    return {
        'setups': setups,
        'total_symbols': len(universe),
        'high_confidence': len([s for s in setups if s['confidence'] >= 80]),
        'total_opportunities': len(setups)
    }

# ========== MAIN DASHBOARD ==========
def main():
    # Professional header without diamonds
    st.markdown("""
    <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #1a1f3a, #2d3561); 
                border-radius: 20px; margin-bottom: 2rem; box-shadow: 0 10px 30px rgba(0,0,0,0.5);
                border: 1px solid rgba(255,255,255,0.1);'>
        <h1 style='font-size: 3rem; margin: 0; color: #ffffff; 
                   font-weight: 700; letter-spacing: 2px;'>
            GEX TRADING COMMAND CENTER
        </h1>
        <p style='font-size: 1.2rem; color: #a0a9c9; margin-top: 0.5rem; font-weight: 300;'>
            Full Universe Scanning | Real-Time Opportunities | Explosive Profits
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load REAL data
    data, connection_status = load_real_databricks_data()
    
    # Connection status bar
    status_color = "🟢" if "Live" in connection_status else "🟡" if "Demo" in connection_status else "🔴"
    st.markdown(f"""
    <div style='background: rgba(17, 25, 40, 0.9); padding: 1rem; border-radius: 10px; 
                margin-bottom: 2rem; border: 2px solid {"#00f2fe" if "Live" in connection_status else "#f5576c"};'>
        <h3 style='margin: 0; color: white;'>
            {status_color} {connection_status} | 
            📊 {data['total_symbols']} Symbols Analyzed | 
            🎯 {data['total_opportunities']} Opportunities Found | 
            🔥 {data['high_confidence']} High Confidence Setups
        </h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Main tabs with all features
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🔥 Trade Setups", 
        "📊 Market Analysis", 
        "💰 Top Movers", 
        "🤖 Auto Trader",
        "📈 Performance", 
        "⚙️ Settings"
    ])
    
    # ========== TAB 1: TRADE SETUPS (THE MAIN ATTRACTION) ==========
    with tab1:
        st.markdown("""
        <h2 style='text-align: center; color: white; text-shadow: 0 0 20px rgba(0,242,254,0.8);'>
            🚀 LIVE TRADING OPPORTUNITIES 🚀
        </h2>
        """, unsafe_allow_html=True)
        
        # Filters
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            min_confidence = st.slider("Min Confidence %", 50, 95, 70, 5)
        with col2:
            setup_type = st.selectbox("Setup Type", ["ALL"] + list(set([s['type'] for s in data['setups']])))
        with col3:
            sort_by = st.selectbox("Sort By", ["Confidence", "Expected Profit", "Net GEX", "Symbol"])
        with col4:
            show_count = st.slider("Show Top", 5, 50, 20, 5)
        
        # Filter setups
        filtered_setups = [s for s in data['setups'] if s['confidence'] >= min_confidence]
        if setup_type != "ALL":
            filtered_setups = [s for s in filtered_setups if s['type'] == setup_type]
        
        # Sort setups
        sort_key = {
            "Confidence": lambda x: x['confidence'],
            "Expected Profit": lambda x: x['expected_profit'],
            "Net GEX": lambda x: abs(x['net_gex']),
            "Symbol": lambda x: x['symbol']
        }
        filtered_setups.sort(key=sort_key[sort_by], reverse=True)
        
        # Display setups in clean, professional cards
        for i, setup in enumerate(filtered_setups[:show_count]):
            # Determine card styling based on confidence
            if setup['confidence'] >= 85:
                gradient = "linear-gradient(135deg, #c31432 0%, #f64f59 100%)"
                border_color = "#f64f59"
                badge = "🔥 HOT"
                badge_color = "#ff6b6b"
            elif setup['confidence'] >= 75:
                gradient = "linear-gradient(135deg, #1e3c72 0%, #4a9eff 100%)"
                border_color = "#4a9eff"
                badge = "⭐ STRONG"
                badge_color = "#4a9eff"
            else:
                gradient = "linear-gradient(135deg, #232526 0%, #414345 100%)"
                border_color = "#666"
                badge = "📊 WATCH"
                badge_color = "#888"
            
            st.markdown(f"""
            <div style='background: {gradient}; border: 1px solid {border_color};
                        border-radius: 12px; padding: 1.5rem; margin: 1rem 0;
                        box-shadow: 0 8px 20px rgba(0,0,0,0.3);'>
                <div style='display: flex; justify-content: space-between; align-items: center;'>
                    <div>
                        <h3 style='margin: 0; color: white; font-size: 1.8rem; font-weight: 700;'>
                            {setup['symbol']} - {setup['type']}
                        </h3>
                        <p style='margin: 0.5rem 0; font-size: 1.1rem; color: #e0e6f0;'>
                            Confidence: <span style='font-size: 1.3rem; font-weight: bold; color: white;'>
                                {setup['confidence']:.1f}%
                            </span>
                            | Net GEX: <span style='color: white;'>{setup['net_gex']/1e9:.2f}B</span>
                        </p>
                    </div>
                    <div style='text-align: right;'>
                        <div style='font-size: 1.2rem; font-weight: bold; padding: 0.4rem 1rem; 
                                    background: {badge_color}; border-radius: 8px; color: white;'>
                            {badge}
                        </div>
                        <div style='margin-top: 0.5rem; font-size: 1.2rem; color: #4ade80; font-weight: 600;'>
                            +${setup['expected_profit']:.0f}
                        </div>
                    </div>
                </div>
                
                <div style='margin-top: 1rem; padding-top: 1rem; border-top: 1px solid rgba(255,255,255,0.2);'>
                    <div style='display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem;'>
                        <div>
                            <p style='margin: 0; color: #a0a9c9; font-size: 0.85rem;'>Entry</p>
                            <p style='margin: 0; font-size: 1.1rem; font-weight: 600; color: white;'>
                                ${setup['entry']:.2f}
                            </p>
                        </div>
                        <div>
                            <p style='margin: 0; color: #a0a9c9; font-size: 0.85rem;'>Target</p>
                            <p style='margin: 0; font-size: 1.1rem; font-weight: 600; color: #4ade80;'>
                                ${setup['target']:.2f}
                            </p>
                        </div>
                        <div>
                            <p style='margin: 0; color: #a0a9c9; font-size: 0.85rem;'>Stop</p>
                            <p style='margin: 0; font-size: 1.1rem; font-weight: 600; color: #ff6b6b;'>
                                ${setup['stop']:.2f}
                            </p>
                        </div>
                        <div>
                            <p style='margin: 0; color: #a0a9c9; font-size: 0.85rem;'>Size</p>
                            <p style='margin: 0; font-size: 1.1rem; font-weight: 600; color: white;'>
                                {setup['size']:.1f}%
                            </p>
                        </div>
                    </div>
                </div>
                
                <div style='margin-top: 1rem; display: flex; gap: 0.5rem;'>
                    <div style='flex: 1; padding: 0.4rem; background: rgba(0,0,0,0.3); 
                                border-radius: 6px; text-align: center;'>
                        <p style='margin: 0; font-size: 0.85rem; color: #e0e6f0;'>
                            Flip: ${setup['gamma_flip']:.2f}
                        </p>
                    </div>
                    <div style='flex: 1; padding: 0.4rem; background: rgba(0,0,0,0.3); 
                                border-radius: 6px; text-align: center;'>
                        <p style='margin: 0; font-size: 0.85rem; color: #e0e6f0;'>
                            Call: ${setup['call_wall']:.2f}
                        </p>
                    </div>
                    <div style='flex: 1; padding: 0.4rem; background: rgba(0,0,0,0.3); 
                                border-radius: 6px; text-align: center;'>
                        <p style='margin: 0; font-size: 0.85rem; color: #e0e6f0;'>
                            Put: ${setup['put_wall']:.2f}
                        </p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Action buttons
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                if st.button(f"📊 Analyze", key=f"analyze_{i}"):
                    st.session_state.selected_symbol = setup['symbol']
            with col2:
                if st.button(f"💼 Trade", key=f"trade_{i}"):
                    st.success(f"Trade setup ready for {setup['symbol']}!")
            with col3:
                if st.button(f"📢 Alert", key=f"alert_{i}"):
                    st.success(f"Alert set for {setup['symbol']}!")
            with col4:
                if st.button(f"📈 Chart", key=f"chart_{i}"):
                    st.session_state.selected_symbol = setup['symbol']
    
    # ========== TAB 2: MARKET ANALYSIS ==========
    with tab2:
        st.markdown("<h2 style='color: white;'>📊 Market-Wide GEX Analysis</h2>", unsafe_allow_html=True)
        
        if data['setups']:
            # Create GEX distribution chart
            symbols = [s['symbol'] for s in data['setups'][:30]]
            net_gex_values = [s['net_gex']/1e9 for s in data['setups'][:30]]
            confidences = [s['confidence'] for s in data['setups'][:30]]
            
            fig = go.Figure()
            
            # Add bars with gradient colors
            colors = ['rgb(245, 87, 108)' if gex < 0 else 'rgb(0, 242, 254)' for gex in net_gex_values]
            
            fig.add_trace(go.Bar(
                x=symbols,
                y=net_gex_values,
                marker_color=colors,
                marker_line_color='white',
                marker_line_width=2,
                text=[f"{conf:.0f}%" for conf in confidences],
                textposition='outside',
                name='Net GEX (Billions)'
            ))
            
            fig.update_layout(
                title="Top 30 Symbols by GEX Exposure",
                template="plotly_dark",
                height=500,
                paper_bgcolor='rgba(17, 25, 40, 0.9)',
                plot_bgcolor='rgba(17, 25, 40, 0.7)',
                font=dict(color='white', size=14),
                xaxis=dict(title="Symbol", gridcolor='rgba(255,255,255,0.1)'),
                yaxis=dict(title="Net GEX (Billions)", gridcolor='rgba(255,255,255,0.1)'),
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Market statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_negative = len([s for s in data['setups'] if s['net_gex'] < 0])
                st.metric("Negative GEX Setups", total_negative, 
                         delta=f"{(total_negative/len(data['setups'])*100):.0f}% of total")
            
            with col2:
                avg_confidence = np.mean([s['confidence'] for s in data['setups']])
                st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
            
            with col3:
                total_expected = sum([s['expected_profit'] for s in data['setups']])
                st.metric("Total Expected Profit", f"${total_expected:,.0f}")
            
            with col4:
                regime = "VOLATILE" if sum([s['net_gex'] for s in data['setups']]) < 0 else "SUPPRESSED"
                st.metric("Market Regime", regime)
    
    # ========== TAB 3: TOP MOVERS ==========
    with tab3:
        st.markdown("<h2 style='color: white;'>💰 Top Profit Opportunities</h2>", unsafe_allow_html=True)
        
        # Sort by expected profit
        top_profits = sorted(data['setups'], key=lambda x: x['expected_profit'], reverse=True)[:10]
        
        # Create beautiful table
        df_display = pd.DataFrame(top_profits)
        df_display = df_display[['symbol', 'type', 'confidence', 'expected_profit', 'net_gex', 'action']]
        df_display.columns = ['Symbol', 'Setup Type', 'Confidence %', 'Expected Profit $', 'Net GEX', 'Action']
        
        # Style the dataframe
        st.dataframe(
            df_display.style.format({
                'Confidence %': '{:.1f}',
                'Expected Profit $': '${:.0f}',
                'Net GEX': lambda x: f'{x/1e9:.2f}B'
            }).background_gradient(subset=['Confidence %', 'Expected Profit $'], cmap='RdYlGn'),
            use_container_width=True,
            height=400
        )
    
    # ========== TAB 4: AUTO TRADER ==========
    with tab4:
        st.markdown("<h2 style='color: white;'>🤖 Automated Trading System</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div style='background: rgba(102, 126, 234, 0.2); padding: 1.5rem; border-radius: 15px; 
                        border: 2px solid #667eea;'>
                <h3 style='color: white;'>Auto Trade Settings</h3>
            </div>
            """, unsafe_allow_html=True)
            
            auto_enabled = st.toggle("Enable Auto Trading", value=False)
            min_auto_confidence = st.slider("Min Auto Trade Confidence", 75, 95, 85, 5)
            max_positions = st.number_input("Max Concurrent Positions", 1, 20, 5)
            position_size = st.slider("Position Size %", 1, 10, 3)
            
            if auto_enabled:
                st.success("✅ Auto Trader Active")
                
                # Find qualifying trades
                auto_trades = [s for s in data['setups'] if s['confidence'] >= min_auto_confidence]
                st.info(f"Found {len(auto_trades)} qualifying trades")
        
        with col2:
            st.markdown("""
            <div style='background: rgba(240, 147, 251, 0.2); padding: 1.5rem; border-radius: 15px; 
                        border: 2px solid #f093fb;'>
                <h3 style='color: white;'>Pending Auto Trades</h3>
            </div>
            """, unsafe_allow_html=True)
            
            if auto_enabled and auto_trades:
                for trade in auto_trades[:max_positions]:
                    st.markdown(f"""
                    <div style='background: rgba(0, 242, 254, 0.1); padding: 0.5rem; 
                                border-radius: 10px; margin: 0.5rem 0;'>
                        <p style='margin: 0; color: white;'>
                            {trade['symbol']} - {trade['type']} - {trade['confidence']:.1f}% - 
                            Est. +${trade['expected_profit']:.0f}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
    
    # ========== TAB 5: PERFORMANCE ==========
    with tab5:
        st.markdown("<h2 style='color: white;'>📈 Strategy Performance</h2>", unsafe_allow_html=True)
        
        # Performance by setup type
        setup_performance = {}
        for setup in data['setups']:
            setup_type = setup['type']
            if setup_type not in setup_performance:
                setup_performance[setup_type] = {
                    'count': 0,
                    'avg_confidence': 0,
                    'total_profit': 0
                }
            setup_performance[setup_type]['count'] += 1
            setup_performance[setup_type]['avg_confidence'] += setup['confidence']
            setup_performance[setup_type]['total_profit'] += setup['expected_profit']
        
        # Calculate averages
        for setup_type in setup_performance:
            count = setup_performance[setup_type]['count']
            setup_performance[setup_type]['avg_confidence'] /= count
        
        # Display performance metrics
        for setup_type, metrics in setup_performance.items():
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(f"{setup_type}", f"{metrics['count']} setups")
            with col2:
                st.metric("Avg Confidence", f"{metrics['avg_confidence']:.1f}%")
            with col3:
                st.metric("Total Expected", f"${metrics['total_profit']:.0f}")
    
    # ========== TAB 6: SETTINGS ==========
    with tab6:
        st.markdown("<h2 style='color: white;'>⚙️ System Settings</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Data Settings")
            auto_refresh = st.checkbox("Auto Refresh (1 min)", value=st.session_state.auto_refresh)
            if auto_refresh:
                st.session_state.auto_refresh = True
                time.sleep(60)
                st.rerun()
            
            if st.button("🔄 Manual Refresh"):
                st.cache_data.clear()
                st.rerun()
            
            st.markdown("### Alert Settings")
            webhook_url = st.text_input("Discord Webhook URL", type="password")
            alert_threshold = st.slider("Alert Confidence Threshold", 70, 95, 85, 5)
        
        with col2:
            st.markdown("### Display Settings")
            show_animations = st.checkbox("Show Animations", value=True)
            dark_mode = st.checkbox("Dark Mode", value=True)
            
            st.markdown("### Export Data")
            if st.button("📥 Export to CSV"):
                df_export = pd.DataFrame(data['setups'])
                csv = df_export.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"gex_setups_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

if __name__ == "__main__":
    main()
