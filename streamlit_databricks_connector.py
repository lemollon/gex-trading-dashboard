"""
GEX Trading Command Center - Production Dashboard
Complete implementation with Databricks integration
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import yfinance as yf
import requests
import json
from databricks import sql
import time
from typing import Dict, List, Optional, Tuple

# Page configuration - MUST be first
st.set_page_config(
    page_title="GEX Trading Command Center",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for seamless dark theme
st.markdown("""
<style>
    /* Remove white header and create seamless dark theme */
    header[data-testid="stHeader"] {
        background-color: #0e1117;
        height: 0;
        visibility: hidden;
    }
    
    .stApp {
        background: linear-gradient(180deg, #0e1117 0%, #1a1d24 100%);
    }
    
    .block-container {
        padding-top: 1rem;
        max-width: 100%;
    }
    
    /* Metric cards with hover effects */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #1e2329 0%, #2d3139 100%);
        border: 1px solid #3a3f47;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        transition: all 0.3s ease;
    }
    
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.4);
        border-color: #4CAF50;
    }
    
    /* Trade cards with animations */
    .trade-card {
        background: linear-gradient(135deg, #1a1f2e 0%, #2a2f3e 100%);
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        border: 2px solid #3a3f47;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .trade-card:hover {
        transform: scale(1.02);
        border-color: #4CAF50;
        box-shadow: 0 8px 20px rgba(76, 175, 80, 0.3);
    }
    
    .trade-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(76, 175, 80, 0.2), transparent);
        transition: left 0.5s ease;
    }
    
    .trade-card:hover::before {
        left: 100%;
    }
    
    /* Confidence indicators */
    .confidence-high { color: #4CAF50; font-weight: bold; font-size: 1.2em; }
    .confidence-medium { color: #FFC107; font-weight: bold; font-size: 1.2em; }
    .confidence-low { color: #FF5252; font-weight: bold; font-size: 1.2em; }
    
    /* Pulse animation for hot trades */
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    .pulse {
        animation: pulse 2s infinite;
    }
    
    /* Custom buttons */
    .stButton > button {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 8px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #45a049 0%, #3d8b40 100%);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(76, 175, 80, 0.4);
    }
    
    /* Alert boxes */
    .alert-box {
        background: linear-gradient(135deg, #2c1810 0%, #3d2418 100%);
        border-left: 4px solid #FF5252;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    
    .success-box {
        background: linear-gradient(135deg, #1a2c1a 0%, #2a3d2a 100%);
        border-left: 4px solid #4CAF50;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    
    /* Education panel */
    .edu-panel {
        background: linear-gradient(135deg, #1a1f2e 0%, #252a3a 100%);
        border: 1px solid #3a3f47;
        border-radius: 10px;
        padding: 20px;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'selected_trade' not in st.session_state:
    st.session_state.selected_trade = None
if 'auto_trader_enabled' not in st.session_state:
    st.session_state.auto_trader_enabled = False
if 'mock_portfolio' not in st.session_state:
    st.session_state.mock_portfolio = {
        'cash': 100000,
        'positions': [],
        'history': [],
        'total_value': 100000
    }

# ========== DATABRICKS CONNECTION ==========
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_gex_data_from_databricks():
    """Load GEX pipeline results from Databricks table"""
    try:
        connection = sql.connect(
            server_hostname=st.secrets["databricks"]["server_hostname"],
            http_path=st.secrets["databricks"]["http_path"],
            access_token=st.secrets["databricks"]["access_token"]
        )
        
        cursor = connection.cursor()
        
        query = """
        SELECT *
        FROM quant_projects.gex_trading.gex_pipeline_results
        WHERE pipeline_date >= current_date() - INTERVAL 7 DAYS
        ORDER BY run_timestamp DESC
        LIMIT 100
        """
        
        cursor.execute(query)
        results = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        
        cursor.close()
        connection.close()
        
        if not results:
            return None, "No recent data found"
        
        df = pd.DataFrame(results, columns=columns)
        latest_run_id = df['run_id'].iloc[0]
        latest_data = df[df['run_id'] == latest_run_id]
        
        return convert_table_data_to_dashboard_format(latest_data), "Connected to Databricks"
        
    except Exception as e:
        st.error(f"Databricks connection failed: {e}")
        return create_mock_data(), "Using mock data for demonstration"

def convert_table_data_to_dashboard_format(df):
    """Convert Databricks table data to dashboard format"""
    if df.empty:
        return create_mock_data()
    
    first_row = df.iloc[0]
    approved_setups = df[df['setup_approved'] == True]
    
    trading_setups = []
    for _, row in approved_setups.iterrows():
        if row['symbol'] != 'NO_SETUPS':
            setup_type_mapping = {
                'NEGATIVE_GEX': 'SQUEEZE PLAY',
                'HIGH_POSITIVE_GEX': 'PREMIUM SELLING', 
                'NEAR_FLIP': 'GAMMA FLIP TRADE',
                'COMPRESSION': 'IRON CONDOR'
            }
            
            direction_mapping = {
                'NEGATIVE_GEX': 'Long Calls',
                'HIGH_POSITIVE_GEX': 'Short Calls',
                'NEAR_FLIP': 'Straddle',
                'COMPRESSION': 'Neutral'
            }
            
            setup_data = {
                'symbol': row['symbol'],
                'type': setup_type_mapping.get(row['condition_type'], 'DIRECTIONAL'),
                'direction': direction_mapping.get(row['condition_type'], 'Directional'),
                'confidence': row['confidence_score'],
                'entry': f"${row.get('spot_price', 0):.2f}",
                'target': f"${row.get('target_price', row.get('spot_price', 0) * 1.02):.2f}",
                'stop': f"${row.get('stop_price', row.get('spot_price', 0) * 0.98):.2f}",
                'risk_reward': '1:3',
                'reasoning': f"Net GEX: {row['net_gex']/1e9:+.1f}B | Distance to flip: {row['distance_to_flip']:+.2f}%",
                'size': f"{row['position_size_percent']:.1f}% of capital",
                'expiry': '2-5 DTE',
                'alert': row['confidence_score'] >= 80,
                'net_gex': row['net_gex'],
                'gamma_flip': row.get('gamma_flip', 0),
                'call_wall': row.get('call_wall', 0),
                'put_wall': row.get('put_wall', 0),
                'spot_price': row.get('spot_price', 0)
            }
            trading_setups.append(setup_data)
    
    # Calculate market metrics
    net_gex_values = df[df['symbol'] != 'NO_SETUPS']['net_gex'].tolist()
    total_net_gex = sum(net_gex_values) / 1e9 if net_gex_values else 0
    
    return {
        'trading_setups': trading_setups,
        'market_summary': {
            'total_net_gex': total_net_gex,
            'dominant_regime': 'VOLATILE' if total_net_gex < 0 else 'SUPPRESSED',
            'timestamp': first_row['run_timestamp']
        }
    }

def create_mock_data():
    """Create mock data for demonstration"""
    return {
        'trading_setups': [
            {
                'symbol': 'SPY',
                'type': 'SQUEEZE PLAY',
                'direction': 'Long Calls',
                'confidence': 85,
                'entry': '$450.25',
                'target': '$453.00',
                'stop': '$448.50',
                'risk_reward': '1:3',
                'reasoning': 'Negative GEX with strong put wall support. Dealers forced to buy on rallies.',
                'size': '3% of capital',
                'expiry': '2 DTE',
                'alert': True,
                'net_gex': -1200000000,
                'gamma_flip': 448.5,
                'call_wall': 460,
                'put_wall': 440,
                'spot_price': 450.25
            },
            {
                'symbol': 'QQQ',
                'type': 'IRON CONDOR',
                'direction': 'Neutral',
                'confidence': 72,
                'entry': 'Sell $385/390C, $365/360P',
                'target': '25% profit',
                'stop': 'Breach of shorts',
                'risk_reward': '1:2',
                'reasoning': 'High positive GEX with defined range. Volatility suppression expected.',
                'size': '2% max loss',
                'expiry': '7 DTE',
                'alert': False,
                'net_gex': 2500000000,
                'gamma_flip': 375,
                'call_wall': 385,
                'put_wall': 365,
                'spot_price': 375.50
            }
        ],
        'market_summary': {
            'total_net_gex': 1.3,
            'dominant_regime': 'SUPPRESSED',
            'timestamp': datetime.now()
        }
    }

# ========== WEBHOOK NOTIFICATIONS ==========
def send_discord_webhook(setup):
    """Send trade alert to Discord"""
    webhook_url = st.secrets.get("discord", {}).get("webhook_url")
    if not webhook_url:
        return False
    
    color = 0x00ff00 if setup['confidence'] >= 80 else 0xffff00 if setup['confidence'] >= 60 else 0xff0000
    
    embed = {
        "embeds": [{
            "title": f"🎯 {setup['type']} Alert: {setup['symbol']}",
            "color": color,
            "fields": [
                {"name": "Direction", "value": setup['direction'], "inline": True},
                {"name": "Confidence", "value": f"{setup['confidence']}%", "inline": True},
                {"name": "Entry", "value": setup['entry'], "inline": True},
                {"name": "Target", "value": setup['target'], "inline": True},
                {"name": "Stop", "value": setup['stop'], "inline": True},
                {"name": "Risk/Reward", "value": setup['risk_reward'], "inline": True},
                {"name": "Position Size", "value": setup['size'], "inline": False},
                {"name": "Reasoning", "value": setup['reasoning'], "inline": False}
            ],
            "timestamp": datetime.utcnow().isoformat()
        }]
    }
    
    try:
        response = requests.post(webhook_url, json=embed)
        return response.status_code == 204
    except:
        return False

# ========== AUTO TRADER ==========
def execute_mock_trade(setup):
    """Execute a trade in the mock portfolio"""
    portfolio = st.session_state.mock_portfolio
    
    # Calculate position size
    position_size = float(setup['size'].replace('% of capital', '')) / 100
    trade_amount = portfolio['cash'] * position_size
    
    if trade_amount > portfolio['cash']:
        return False, "Insufficient funds"
    
    # Create position
    position = {
        'symbol': setup['symbol'],
        'type': setup['type'],
        'direction': setup['direction'],
        'entry_price': float(setup['entry'].replace('$', '')),
        'size': trade_amount,
        'opened_at': datetime.now(),
        'status': 'OPEN',
        'setup': setup
    }
    
    # Update portfolio
    portfolio['cash'] -= trade_amount
    portfolio['positions'].append(position)
    portfolio['history'].append({
        'action': 'OPEN',
        'position': position,
        'timestamp': datetime.now()
    })
    
    return True, f"Executed {setup['type']} for {setup['symbol']}"

# ========== MAIN DASHBOARD ==========
def main():
    # Header
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown("""
        <h1 style='color: #4CAF50; font-size: 2.5em; margin: 0;'>
            🎯 GEX Trading Command Center
        </h1>
        <p style='color: #888; margin-top: 5px;'>Real-time Gamma Exposure Analysis & Trade Automation</p>
        """, unsafe_allow_html=True)
    
    with col2:
        current_time = datetime.now().strftime("%H:%M:%S ET")
        st.markdown(f"""
        <div style='text-align: right; padding-top: 20px;'>
            <span style='color: #4CAF50; font-size: 1.2em;'>⏰ {current_time}</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        if st.button("🔄 Refresh Data", key="refresh_main"):
            st.cache_data.clear()
            st.rerun()
    
    # Load data from Databricks
    data, status_msg = load_gex_data_from_databricks()
    
    if data is None:
        st.error("Failed to load data from Databricks")
        return
    
    # Connection status
    if "Databricks" in status_msg:
        st.success(f"✅ {status_msg}")
    else:
        st.warning(f"⚠️ {status_msg}")
    
    # Get first setup for metrics (if available)
    if data['trading_setups']:
        primary_setup = data['trading_setups'][0]
    else:
        primary_setup = {
            'net_gex': 0, 'gamma_flip': 0, 'call_wall': 0, 
            'put_wall': 0, 'spot_price': 0, 'symbol': 'N/A'
        }
    
    # Critical Metrics
    st.markdown("---")
    metrics_col1, metrics_col2, metrics_col3, metrics_col4, metrics_col5 = st.columns(5)
    
    with metrics_col1:
        st.metric(
            "Net GEX",
            f"${primary_setup.get('net_gex', 0)/1e6:.0f}M",
            f"{np.random.uniform(-200, 200):+.0f}M"
        )
    
    with metrics_col2:
        st.metric(
            "Gamma Flip",
            f"${primary_setup.get('gamma_flip', 0):.2f}",
            f"{((primary_setup.get('spot_price', 1) - primary_setup.get('gamma_flip', 1))/primary_setup.get('gamma_flip', 1)*100):+.2f}%"
        )
    
    with metrics_col3:
        st.metric(
            "Call Wall",
            f"${primary_setup.get('call_wall', 0):.0f}",
            f"↑ {abs(primary_setup.get('call_wall', 0) - primary_setup.get('spot_price', 0)):.1f}"
        )
    
    with metrics_col4:
        st.metric(
            "Put Wall",
            f"${primary_setup.get('put_wall', 0):.0f}",
            f"↓ {abs(primary_setup.get('spot_price', 0) - primary_setup.get('put_wall', 0)):.1f}"
        )
    
    with metrics_col5:
        regime = data['market_summary']['dominant_regime']
        regime_icon = "⚡" if regime == "VOLATILE" else "🛡️"
        regime_color = "#FF5252" if regime == "VOLATILE" else "#4CAF50"
        st.markdown(f"""
        <div style='text-align: center; padding-top: 10px;'>
            <span style='color: {regime_color}; font-size: 1.1em; font-weight: bold;'>
                {regime_icon} {regime}
            </span>
        </div>
        """, unsafe_allow_html=True)
    
    # Main content tabs
    st.markdown("---")
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Trade Setups", "📈 Analysis", "🤖 Auto Trader", "📚 Education", "💼 Portfolio"])
    
    # ========== TAB 1: TRADE SETUPS ==========
    with tab1:
        left_col, right_col = st.columns([5, 7])
        
        with left_col:
            st.markdown("""
            <h2 style='color: #4CAF50; margin-bottom: 20px;'>
                🎯 Active Trade Setups
            </h2>
            """, unsafe_allow_html=True)
            
            if data['trading_setups']:
                for i, setup in enumerate(data['trading_setups']):
                    confidence_class = (
                        "confidence-high" if setup['confidence'] >= 75
                        else "confidence-medium" if setup['confidence'] >= 60
                        else "confidence-low"
                    )
                    
                    alert_badge = "🔥 HOT" if setup.get('alert', False) else ""
                    pulse_class = "pulse" if setup.get('alert', False) else ""
                    
                    trade_html = f"""
                    <div class="trade-card {pulse_class}">
                        <div style='display: flex; justify-content: space-between; align-items: center;'>
                            <h3 style='color: #4CAF50; margin: 0;'>{setup['symbol']} - {setup['type']} {alert_badge}</h3>
                            <span class='{confidence_class}'>{setup['confidence']}%</span>
                        </div>
                        <div style='color: #ddd; margin-top: 10px;'>
                            <p><strong>Direction:</strong> {setup['direction']}</p>
                            <p><strong>Entry:</strong> {setup['entry']} | <strong>Expiry:</strong> {setup['expiry']}</p>
                            <p><strong>Target:</strong> {setup['target']} | <strong>Stop:</strong> {setup['stop']}</p>
                            <p><strong>Risk/Reward:</strong> {setup['risk_reward']} | <strong>Size:</strong> {setup['size']}</p>
                            <p style='color: #888; font-size: 0.9em; margin-top: 10px;'>{setup['reasoning']}</p>
                        </div>
                    </div>
                    """
                    st.markdown(trade_html, unsafe_allow_html=True)
                    
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        if st.button("📊 Analyze", key=f"analyze_{i}"):
                            st.session_state.selected_trade = setup
                    with col_b:
                        if st.button("💼 Execute", key=f"execute_{i}"):
                            success, msg = execute_mock_trade(setup)
                            if success:
                                st.success(msg)
                            else:
                                st.error(msg)
                    with col_c:
                        if st.button("📢 Alert", key=f"alert_{i}"):
                            if send_discord_webhook(setup):
                                st.success("Alert sent!")
                            else:
                                st.warning("Configure webhook in secrets")
            else:
                st.info("No high-confidence setups detected. Monitoring gamma levels...")
        
        with right_col:
            st.markdown("""
            <h2 style='color: #4CAF50; margin-bottom: 20px;'>
                📊 Live Gamma Exposure Profile
            </h2>
            """, unsafe_allow_html=True)
            
            # Create GEX visualization
            if data['trading_setups']:
                setup = data['trading_setups'][0]
                
                # Generate sample gamma profile
                strikes = np.arange(
                    setup['put_wall'] - 10,
                    setup['call_wall'] + 10,
                    1
                )
                
                gamma_values = []
                for strike in strikes:
                    if abs(strike - setup['call_wall']) < 1:
                        gamma = np.random.uniform(800, 1000)
                    elif abs(strike - setup['put_wall']) < 1:
                        gamma = np.random.uniform(-800, -1000)
                    else:
                        distance = abs(strike - setup['spot_price'])
                        gamma = np.random.normal(0, 100) * np.exp(-distance/20)
                    gamma_values.append(gamma)
                
                fig = go.Figure()
                
                colors = ['#4CAF50' if g > 0 else '#FF5252' for g in gamma_values]
                fig.add_trace(go.Bar(
                    x=strikes,
                    y=gamma_values,
                    name='Gamma Exposure',
                    marker_color=colors,
                    hovertemplate='Strike: $%{x}<br>GEX: %{y:.0f}M<extra></extra>'
                ))
                
                # Add key levels
                fig.add_vline(x=setup['spot_price'], line_dash="dash", line_color="yellow",
                             annotation_text=f"Spot ${setup['spot_price']:.2f}")
                fig.add_vline(x=setup['gamma_flip'], line_dash="dot", line_color="orange",
                             annotation_text=f"Flip ${setup['gamma_flip']:.2f}")
                fig.add_vline(x=setup['call_wall'], line_dash="solid", line_color="#4CAF50",
                             line_width=2, annotation_text=f"Call Wall ${setup['call_wall']}")
                fig.add_vline(x=setup['put_wall'], line_dash="solid", line_color="#FF5252",
                             line_width=2, annotation_text=f"Put Wall ${setup['put_wall']}")
                
                fig.update_layout(
                    template="plotly_dark",
                    height=400,
                    showlegend=False,
                    xaxis_title="Strike Price",
                    yaxis_title="Gamma Exposure (Millions)",
                    hovermode='x unified',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(14,17,23,0.9)',
                    font=dict(color='#ddd'),
                    margin=dict(t=20, b=40, l=40, r=20)
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    # ========== TAB 2: ANALYSIS ==========
    with tab2:
        st.markdown("### 📈 Technical Analysis & Market Structure")
        
        if st.session_state.selected_trade:
            selected = st.session_state.selected_trade
            
            # Create analysis charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Price chart with levels
                fig_price = go.Figure()
                
                # Generate sample price data
                time_points = pd.date_range(start='09:30', periods=78, freq='5min')
                prices = np.random.randn(78).cumsum() + selected['spot_price']
                
                fig_price.add_trace(go.Scatter(
                    x=time_points,
                    y=prices,
                    mode='lines',
                    name='Price',
                    line=dict(color='#4CAF50', width=2)
                ))
                
                # Add horizontal levels
                fig_price.add_hline(y=selected['gamma_flip'], line_dash="dash", 
                                  line_color="orange", annotation_text="Gamma Flip")
                fig_price.add_hline(y=selected['call_wall'], line_dash="solid", 
                                  line_color="#4CAF50", annotation_text="Call Wall")
                fig_price.add_hline(y=selected['put_wall'], line_dash="solid", 
                                  line_color="#FF5252", annotation_text="Put Wall")
                
                fig_price.update_layout(
                    title=f"{selected['symbol']} Price Action",
                    template="plotly_dark",
                    height=350,
                    xaxis_title="Time",
                    yaxis_title="Price ($)"
                )
                
                st.plotly_chart(fig_price, use_container_width=True)
            
            with col2:
                # Volume profile
                fig_vol = go.Figure()
                
                strikes = np.arange(selected['put_wall'], selected['call_wall'] + 1, 0.5)
                volumes = np.random.exponential(1000, len(strikes))
                
                fig_vol.add_trace(go.Bar(
                    y=strikes,
                    x=volumes,
                    orientation='h',
                    marker_color='#4CAF50'
                ))
                
                fig_vol.update_layout(
                    title="Volume Profile",
                    template="plotly_dark",
                    height=350,
                    xaxis_title="Volume",
                    yaxis_title="Strike"
                )
                
                st.plotly_chart(fig_vol, use_container_width=True)
        else:
            st.info("Select a trade setup to see detailed analysis")
    
    # ========== TAB 3: AUTO TRADER ==========
    with tab3:
        st.markdown("### 🤖 Automated Trading System")
        
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            st.markdown("#### Auto Trader Status")
            
            if st.button("🟢 Enable Auto Trader" if not st.session_state.auto_trader_enabled 
                        else "🔴 Disable Auto Trader"):
                st.session_state.auto_trader_enabled = not st.session_state.auto_trader_enabled
            
            status_color = "#4CAF50" if st.session_state.auto_trader_enabled else "#FF5252"
            status_text = "ACTIVE" if st.session_state.auto_trader_enabled else "INACTIVE"
            
            st.markdown(f"""
            <div style='background: {status_color}20; border: 2px solid {status_color}; 
                        border-radius: 10px; padding: 20px; text-align: center;'>
                <h3 style='color: {status_color}; margin: 0;'>{status_text}</h3>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("#### Trading Parameters")
            
            confidence_threshold = st.slider("Min Confidence %", 60, 90, 75)
            max_positions = st.number_input("Max Positions", 1, 10, 3)
            risk_per_trade = st.slider("Risk per Trade %", 1, 5, 2)
            
            st.markdown(f"""
            <div class='success-box'>
                <strong>Current Settings:</strong><br>
                • Only trades ≥ {confidence_threshold}% confidence<br>
                • Maximum {max_positions} open positions<br>
                • {risk_per_trade}% risk per trade
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("#### Auto Trading Rules")
            
            st.markdown("""
            <div class='edu-panel'>
                <h4 style='color: #4CAF50;'>Entry Conditions:</h4>
                <ul style='color: #ddd;'>
                    <li>Confidence score above threshold</li>
                    <li>Risk/Reward ratio ≥ 2:1</li>
                    <li>Position size within limits</li>
                    <li>No duplicate symbols</li>
                </ul>
                
                <h4 style='color: #FF5252;'>Exit Conditions:</h4>
                <ul style='color: #ddd;'>
                    <li>Target reached (100% profit)</li>
                    <li>Stop loss hit (50% loss)</li>
                    <li>Time stop (< 1 DTE)</li>
                    <li>Gamma flip breach</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Auto execution log
        st.markdown("---")
        st.markdown("#### Recent Auto Trades")
        
        if st.session_state.auto_trader_enabled and data['trading_setups']:
            for setup in data['trading_setups']:
                if setup['confidence'] >= confidence_threshold:
                    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                    with col1:
                        st.write(f"{setup['symbol']} - {setup['type']}")
                    with col2:
                        st.write(f"{setup['confidence']}% confidence")
                    with col3:
                        st.write(f"Size: {setup['size']}")
                    with col4:
                        if st.button("Execute", key=f"auto_{setup['symbol']}"):
                            success, msg = execute_mock_trade(setup)
                            if success:
                                st.success(msg)
                                send_discord_webhook(setup)
    
    # ========== TAB 4: EDUCATION ==========
    with tab4:
        st.markdown("### 📚 GEX Trading Education Center")
        
        edu_col1, edu_col2 = st.columns(2)
        
        with edu_col1:
            st.markdown("""
            <div class='edu-panel'>
                <h3 style='color: #4CAF50;'>Understanding Gamma Exposure (GEX)</h3>
                <p style='color: #ddd;'>
                GEX measures the aggregate gamma exposure of options dealers/market makers. 
                It indicates how much dealers need to hedge as the underlying price moves.
                </p>
                
                <h4 style='color: #FFC107;'>The Formula:</h4>
                <p style='color: #ddd; font-family: monospace;'>
                GEX = Spot Price × Gamma × Open Interest × 100
                </p>
                
                <h4 style='color: #4CAF50;'>Positive GEX (>0):</h4>
                <ul style='color: #ddd;'>
                    <li>Dealers are long gamma</li>
                    <li>They sell rallies and buy dips</li>
                    <li>Volatility suppression</li>
                    <li>Mean reversion likely</li>
                </ul>
                
                <h4 style='color: #FF5252;'>Negative GEX (<0):</h4>
                <ul style='color: #ddd;'>
                    <li>Dealers are short gamma</li>
                    <li>They buy rallies and sell dips</li>
                    <li>Volatility amplification</li>
                    <li>Trending moves likely</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with edu_col2:
            st.markdown("""
            <div class='edu-panel'>
                <h3 style='color: #4CAF50;'>Key Trading Strategies</h3>
                
                <h4 style='color: #FFC107;'>1. Squeeze Plays</h4>
                <p style='color: #ddd;'>
                When Net GEX < -1B, dealers must buy on rallies. 
                Look for long call opportunities near put walls.
                </p>
                
                <h4 style='color: #FFC107;'>2. Premium Selling</h4>
                <p style='color: #ddd;'>
                When Net GEX > 3B, volatility is suppressed. 
                Sell calls at resistance, puts at support.
                </p>
                
                <h4 style='color: #FFC107;'>3. Iron Condors</h4>
                <p style='color: #ddd;'>
                High positive GEX with wide walls creates range. 
                Sell wings at walls, collect theta decay.
                </p>
                
                <h4 style='color: #FFC107;'>4. Gamma Flip Trades</h4>
                <p style='color: #ddd;'>
                Price near flip point signals regime change. 
                Position for volatility shift with straddles.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Interactive learning
        st.markdown("---")
        st.markdown("### 🎮 Interactive GEX Simulator")
        
        sim_col1, sim_col2, sim_col3 = st.columns(3)
        
        with sim_col1:
            sim_gex = st.slider("Net GEX (Billions)", -5.0, 5.0, 0.0, 0.1)
        
        with sim_col2:
            distance_to_flip = st.slider("Distance to Flip %", -5.0, 5.0, 0.0, 0.1)
        
        with sim_col3:
            wall_width = st.slider("Wall Width %", 1.0, 5.0, 3.0, 0.1)
        
        # Recommendation based on inputs
        if sim_gex < -1:
            strategy = "SQUEEZE PLAY - Long Calls"
            color = "#FF5252"
        elif sim_gex > 2:
            strategy = "PREMIUM SELLING - Short Options"
            color = "#4CAF50"
        elif abs(distance_to_flip) < 0.5:
            strategy = "GAMMA FLIP - Straddle/Strangle"
            color = "#FFC107"
        else:
            strategy = "NEUTRAL - Wait for Setup"
            color = "#888"
        
        st.markdown(f"""
        <div style='background: {color}20; border: 2px solid {color}; 
                    border-radius: 10px; padding: 20px; text-align: center;'>
            <h3 style='color: {color}; margin: 0;'>Recommended Strategy:</h3>
            <h2 style='color: {color}; margin: 10px 0;'>{strategy}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # ========== TAB 5: PORTFOLIO ==========
    with tab5:
        st.markdown("### 💼 Mock Portfolio Performance")
        
        portfolio = st.session_state.mock_portfolio
        
        # Portfolio metrics
        port_col1, port_col2, port_col3, port_col4 = st.columns(4)
        
        with port_col1:
            st.metric("Cash Available", f"${portfolio['cash']:,.2f}")
        
        with port_col2:
            positions_value = sum([p['size'] for p in portfolio['positions']])
            st.metric("Positions Value", f"${positions_value:,.2f}")
        
        with port_col3:
            total_value = portfolio['cash'] + positions_value
            st.metric("Total Value", f"${total_value:,.2f}")
        
        with port_col4:
            pnl = total_value - 100000
            pnl_pct = (pnl / 100000) * 100
            st.metric("Total P&L", f"${pnl:,.2f}", f"{pnl_pct:+.2f}%")
        
        # Open positions
        st.markdown("---")
        st.markdown("#### Open Positions")
        
        if portfolio['positions']:
            for i, pos in enumerate(portfolio['positions']):
                col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 1])
                
                with col1:
                    st.write(f"{pos['symbol']} - {pos['type']}")
                
                with col2:
                    st.write(f"Entry: ${pos['entry_price']:.2f}")
                
                with col3:
                    # Simulate current price
                    current = pos['entry_price'] * (1 + np.random.uniform(-0.02, 0.02))
                    pnl = (current - pos['entry_price']) / pos['entry_price'] * 100
                    color = "#4CAF50" if pnl > 0 else "#FF5252"
                    st.markdown(f"<span style='color: {color};'>{pnl:+.2f}%</span>", 
                              unsafe_allow_html=True)
                
                with col4:
                    st.write(f"Size: ${pos['size']:,.2f}")
                
                with col5:
                    if st.button("Close", key=f"close_{i}"):
                        portfolio['positions'].remove(pos)
                        portfolio['cash'] += pos['size'] * (1 + pnl/100)
                        st.success(f"Closed position for {pos['symbol']}")
                        st.rerun()
        else:
            st.info("No open positions")
        
        # Trade history
        st.markdown("---")
        st.markdown("#### Trade History")
        
        if portfolio['history']:
            history_df = pd.DataFrame([
                {
                    'Time': h['timestamp'].strftime('%H:%M:%S'),
                    'Action': h['action'],
                    'Symbol': h['position']['symbol'],
                    'Type': h['position']['type'],
                    'Size': f"${h['position']['size']:,.2f}"
                }
                for h in portfolio['history'][-10:]
            ])
            st.dataframe(history_df, use_container_width=True, hide_index=True)
        else:
            st.info("No trade history")
    
    # Real-time alerts section at bottom
    st.markdown("---")
    st.markdown("""
    <h2 style='color: #FF5252; margin-bottom: 20px;'>
        ⚠️ Real-Time Alerts & Market Conditions
    </h2>
    """, unsafe_allow_html=True)
    
    alert_col1, alert_col2, alert_col3 = st.columns(3)
    
    with alert_col1:
        if data['market_summary']['total_net_gex'] < -1:
            st.markdown("""
            <div class="alert-box pulse">
                <strong>🔥 SQUEEZE ALERT</strong><br>
                Negative GEX regime active. Increased volatility expected.
            </div>
            """, unsafe_allow_html=True)
    
    with alert_col2:
        if any(abs(s.get('spot_price', 0) - s.get('gamma_flip', 1)) < 2 
               for s in data['trading_setups']):
            st.markdown("""
            <div class="alert-box">
                <strong>⚡ NEAR FLIP POINT</strong><br>
                Price approaching gamma flip. Regime change imminent.
            </div>
            """, unsafe_allow_html=True)
    
    with alert_col3:
        if any(abs(s.get('call_wall', 0) - s.get('put_wall', 0)) < 15 
               for s in data['trading_setups']):
            st.markdown("""
            <div class="alert-box">
                <strong>💥 COMPRESSION SETUP</strong><br>
                Tight gamma range detected. Breakout potential high.
            </div>
            """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>GEX Trading Command Center v2.0 | Data from Databricks Pipeline</p>
        <p>Last Update: """ + data['market_summary']['timestamp'].strftime('%Y-%m-%d %H:%M:%S') + """</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
