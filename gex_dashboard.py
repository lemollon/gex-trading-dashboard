"""
🚀 GEX Trading Command Center - Complete Dashboard
Connected to Enhanced Databricks Pipeline with Market Regime Analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import json
import requests
from databricks import sql
import yfinance as yf

# Page config
st.set_page_config(
    page_title="GEX Trading Command Center",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS with futuristic styling
st.markdown("""
<style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Futuristic dark theme */
    .main {
        background: linear-gradient(-45deg, #0a0a0f, #1a1a2e, #16213e, #0f0f23);
        background-size: 400% 400%;
        animation: gradientShift 30s ease infinite;
        color: white;
        min-height: 100vh;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Cyberpunk neon accents */
    .neon-cyan { color: #00ffff; text-shadow: 0 0 10px #00ffff; }
    .neon-pink { color: #ff00ff; text-shadow: 0 0 10px #ff00ff; }
    .neon-green { color: #00ff00; text-shadow: 0 0 10px #00ff00; }
    .neon-orange { color: #ff6600; text-shadow: 0 0 10px #ff6600; }
    
    /* Hero section with enhanced glassmorphism */
    .hero-container {
        background: linear-gradient(135deg, rgba(0,255,255,0.1), rgba(255,0,255,0.1));
        border-radius: 20px;
        padding: 3rem;
        margin-bottom: 2rem;
        border: 1px solid rgba(0,255,255,0.4);
        backdrop-filter: blur(25px);
        box-shadow: 0 0 50px rgba(0,255,255,0.3), inset 0 0 20px rgba(255,0,255,0.1);
        position: relative;
        overflow: hidden;
    }
    
    .hero-container::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(0,255,255,0.1), transparent);
        animation: scan 8s linear infinite;
    }
    
    @keyframes scan {
        0% { transform: translateX(-100%) translateY(-100%) rotate(45deg); }
        100% { transform: translateX(100%) translateY(100%) rotate(45deg); }
    }
    
    /* Enhanced metric cards with hologram effect */
    .metric-card {
        background: linear-gradient(145deg, rgba(0,50,100,0.3), rgba(0,30,60,0.2));
        border-radius: 15px;
        padding: 2rem;
        margin: 0.8rem;
        border: 1px solid rgba(0,255,255,0.3);
        backdrop-filter: blur(20px);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5), 0 0 20px rgba(0,255,255,0.2);
        position: relative;
    }
    
    .metric-card::after {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
        transition: left 0.5s;
    }
    
    .metric-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 20px 40px rgba(0,255,255,0.4), 0 0 30px rgba(0,255,255,0.3);
        border-color: rgba(0,255,255,0.6);
    }
    
    .metric-card:hover::after {
        left: 100%;
    }
    
    /* Market regime indicator */
    .regime-high-suppression {
        background: linear-gradient(145deg, rgba(255,0,0,0.2), rgba(200,0,0,0.1));
        border-left: 4px solid #ff0000;
        box-shadow: 0 0 25px rgba(255,0,0,0.3);
    }
    
    .regime-negative-gex {
        background: linear-gradient(145deg, rgba(0,255,0,0.2), rgba(0,200,0,0.1));
        border-left: 4px solid #00ff00;
        box-shadow: 0 0 25px rgba(0,255,0,0.3);
    }
    
    .regime-neutral {
        background: linear-gradient(145deg, rgba(255,255,0,0.2), rgba(200,200,0,0.1));
        border-left: 4px solid #ffff00;
        box-shadow: 0 0 25px rgba(255,255,0,0.3);
    }
    
    /* Setup cards with priority styling */
    .setup-high-priority {
        background: linear-gradient(145deg, rgba(255,0,0,0.15), rgba(255,50,50,0.05));
        border-left: 5px solid #ff0000;
        backdrop-filter: blur(15px);
        animation: pulse-red 3s ease-in-out infinite alternate;
        margin: 1rem 0;
        padding: 1.5rem;
        border-radius: 12px;
    }
    
    .setup-moderate {
        background: linear-gradient(145deg, rgba(255,165,0,0.15), rgba(255,165,0,0.05));
        border-left: 4px solid #ffa500;
        backdrop-filter: blur(15px);
        animation: pulse-orange 4s ease-in-out infinite alternate;
        margin: 1rem 0;
        padding: 1.5rem;
        border-radius: 12px;
    }
    
    @keyframes pulse-red {
        0% { box-shadow: 0 0 20px rgba(255,0,0,0.3); }
        100% { box-shadow: 0 0 35px rgba(255,0,0,0.6); }
    }
    
    @keyframes pulse-orange {
        0% { box-shadow: 0 0 15px rgba(255,165,0,0.3); }
        100% { box-shadow: 0 0 25px rgba(255,165,0,0.5); }
    }
    
    /* Holographic text effect */
    .hologram-text {
        font-size: 3.5rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(45deg, #00ffff, #ff00ff, #00ff00, #ffff00);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: hologram 3s ease-in-out infinite alternate;
        text-shadow: 0 0 30px rgba(0,255,255,0.5);
    }
    
    @keyframes hologram {
        0% { filter: hue-rotate(0deg); }
        100% { filter: hue-rotate(360deg); }
    }
    
    .section-header {
        font-size: 2.2rem;
        font-weight: bold;
        color: #00ffff;
        margin: 2.5rem 0 1.5rem 0;
        text-align: center;
        text-shadow: 0 0 15px #00ffff;
    }
    
    /* Enhanced sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, rgba(0,0,80,0.9), rgba(0,0,40,0.9));
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(0,255,255,0.3);
    }
    
    /* Futuristic buttons */
    .stButton > button {
        background: linear-gradient(45deg, #00ffff, #ff00ff);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.8rem 2.5rem;
        font-weight: bold;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 5px 20px rgba(0,255,255,0.4);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.05);
        box-shadow: 0 10px 30px rgba(0,255,255,0.6);
        background: linear-gradient(45deg, #ff00ff, #00ffff);
    }
    
    /* Data table styling */
    .stDataFrame {
        background: rgba(0,0,50,0.3);
        backdrop-filter: blur(10px);
        border-radius: 10px;
    }
    
    /* Metrics styling */
    .metric-large {
        font-size: 2.8rem;
        font-weight: bold;
        text-shadow: 0 0 10px currentColor;
    }
    
    .metric-label {
        font-size: 1.1rem;
        opacity: 0.8;
    }
</style>
""", unsafe_allow_html=True)

# Databricks Connection
@st.cache_resource
def init_databricks_connection():
    """Initialize connection to Databricks"""
    try:
        connection = sql.connect(
            server_hostname=st.secrets["databricks"]["server_hostname"],
            http_path=st.secrets["databricks"]["http_path"],
            access_token=st.secrets["databricks"]["access_token"]
        )
        return connection
    except Exception as e:
        st.error(f"Failed to connect to Databricks: {str(e)}")
        return None

# Enhanced data loading functions
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_pipeline_results():
    """Load latest enhanced pipeline results from Databricks"""
    connection = init_databricks_connection()
    if not connection:
        # Demo data that matches your enhanced pipeline output
        return create_demo_data()
    
    try:
        cursor = connection.cursor()
        
        # Query your enhanced pipeline results
        query = """
        SELECT 
            symbol,
            strategy,
            confidence,
            spot_price,
            net_gex,
            gamma_flip,
            entry_criteria,
            target,
            position_size,
            expected_move,
            time_frame,
            setup_type,
            reason,
            notes,
            historical_win_rate,
            timestamp
        FROM gex_enhanced_results  -- Your enhanced results table
        WHERE DATE(timestamp) = CURRENT_DATE()
        ORDER BY confidence DESC, timestamp DESC
        LIMIT 100
        """
        
        cursor.execute(query)
        results = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        
        if results:
            return pd.DataFrame(results, columns=columns)
        else:
            return create_demo_data()
    
    except Exception as e:
        st.error(f"Error loading pipeline data: {str(e)}")
        return create_demo_data()
    finally:
        if connection:
            connection.close()

@st.cache_data(ttl=300)
def load_interesting_conditions():
    """Load interesting GEX conditions from Databricks"""
    connection = init_databricks_connection()
    if not connection:
        return create_demo_conditions()
    
    try:
        cursor = connection.cursor()
        
        query = """
        SELECT 
            symbol,
            regime,
            net_gex,
            distance_pct,
            action,
            gamma_flip,
            spot_price,
            timestamp
        FROM gex_interesting_conditions  -- Your conditions table
        WHERE DATE(timestamp) = CURRENT_DATE()
        ORDER BY ABS(net_gex) DESC
        LIMIT 50
        """
        
        cursor.execute(query)
        results = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        
        if results:
            df = pd.DataFrame(results, columns=columns)
            # Format for display
            df['gex_display'] = df['net_gex'].apply(lambda x: f"GEX: {x:.2f}B")
            df['distance_display'] = df['distance_pct'].apply(lambda x: f"Distance: {x:+.2f}%")
            return df
        else:
            return create_demo_conditions()
    
    except Exception as e:
        return create_demo_conditions()
    finally:
        if connection:
            connection.close()

@st.cache_data(ttl=300)
def load_market_regime_data():
    """Load market regime analysis from Databricks"""
    connection = init_databricks_connection()
    if not connection:
        return create_demo_regime_data()
    
    try:
        cursor = connection.cursor()
        
        query = """
        SELECT 
            market_regime,
            vix_level,
            total_market_gex,
            regime_strength,
            market_state,
            description,
            symbols_processed,
            setups_found,
            high_confidence_setups,
            timestamp
        FROM gex_market_regime  -- Your regime analysis table
        WHERE DATE(timestamp) = CURRENT_DATE()
        ORDER BY timestamp DESC
        LIMIT 1
        """
        
        cursor.execute(query)
        result = cursor.fetchone()
        
        if result:
            columns = [desc[0] for desc in cursor.description]
            return dict(zip(columns, result))
        else:
            return create_demo_regime_data()
    
    except Exception as e:
        return create_demo_regime_data()
    finally:
        if connection:
            connection.close()

def create_demo_data():
    """Create demo data matching your enhanced pipeline output"""
    return pd.DataFrame([
        {
            'symbol': 'AAPL', 'strategy': 'MODERATE recommendation', 'confidence': 89,
            'spot_price': 228.86, 'net_gex': 0.85, 'gamma_flip': 225.50,
            'entry_criteria': 'Buy puts below 228.86', 'target': 225.50,
            'position_size': '3.0%', 'expected_move': '1.5%', 'time_frame': '2-6 hours',
            'setup_type': 'breakdown_play', 'reason': 'High positive GEX breakdown setup',
            'notes': 'Strong resistance at current levels', 'historical_win_rate': '65%'
        },
        {
            'symbol': 'NVDA', 'strategy': 'HIGH PRIORITY', 'confidence': 92,
            'spot_price': 180.71, 'net_gex': -1.2, 'gamma_flip': 182.50,
            'entry_criteria': 'Buy calls above 182.50', 'target': 185.00,
            'position_size': '3.0%', 'expected_move': '2.4%', 'time_frame': '1-4 hours',
            'setup_type': 'squeeze_play', 'reason': 'Negative GEX squeeze opportunity',
            'notes': 'Dealers short gamma - explosive potential', 'historical_win_rate': '78%'
        },
        {
            'symbol': 'SPY', 'strategy': 'HIGH PRIORITY', 'confidence': 95,
            'spot_price': 455.20, 'net_gex': 1.24, 'gamma_flip': 449.28,
            'entry_criteria': 'Sell calls above 459.75', 'target': 455.20,
            'position_size': '5.0%', 'expected_move': '<1%', 'time_frame': '1-3 days',
            'setup_type': 'premium_selling', 'reason': 'Call wall resistance setup',
            'notes': 'Strong gamma wall provides resistance', 'historical_win_rate': '81%'
        }
    ])

def create_demo_conditions():
    """Create demo interesting conditions"""
    return pd.DataFrame([
        {'symbol': 'SPY', 'regime': 'HIGH_POSITIVE_GEX', 'net_gex': 1.24, 'distance_pct': -0.16, 
         'action': 'Watch for resistance', 'gex_display': 'GEX: +1.24B', 'distance_display': 'Distance: -0.16%'},
        {'symbol': 'QQQ', 'regime': 'HIGH_POSITIVE_GEX', 'net_gex': 0.61, 'distance_pct': -0.16,
         'action': 'Watch for resistance', 'gex_display': 'GEX: +0.61B', 'distance_display': 'Distance: -0.16%'},
        {'symbol': 'NVDA', 'regime': 'NEGATIVE_GEX', 'net_gex': -1.2, 'distance_pct': 1.05,
         'action': 'Watch for squeeze', 'gex_display': 'GEX: -1.20B', 'distance_display': 'Distance: +1.05%'},
        {'symbol': 'MRNA', 'regime': 'NEAR_FLIP', 'net_gex': -0.001, 'distance_pct': -0.43,
         'action': 'High volatility zone', 'gex_display': 'GEX: -0.00B', 'distance_display': 'Distance: -0.43%'}
    ])

def create_demo_regime_data():
    """Create demo market regime data"""
    return {
        'market_regime': 'HIGH_SUPPRESSION',
        'vix_level': 18.5,
        'total_market_gex': 12.45,
        'regime_strength': 0.8,
        'market_state': 'HIGH_SUPPRESSION',
        'description': 'Strong volatility suppression - premium selling favored',
        'symbols_processed': 98,
        'setups_found': 15,
        'high_confidence_setups': 8
    }

@st.cache_data(ttl=60)
def load_live_market_data():
    """Load live market data"""
    try:
        tickers = ['SPY', 'QQQ', 'IWM', '^VIX']
        data = yf.download(tickers, period='1d', interval='5m')
        
        if not data.empty:
            latest = data['Close'].iloc[-1]
            previous = data['Close'].iloc[-2] if len(data) > 1 else data['Close'].iloc[-1]
            
            market_data = {}
            for ticker in tickers:
                if ticker in latest:
                    current = latest[ticker]
                    prev = previous[ticker] if ticker in previous else current
                    change = ((current - prev) / prev * 100) if prev != 0 else 0
                    
                    market_data[ticker] = {
                        'price': current,
                        'change': change,
                        'direction': '📈' if change > 0 else '📉' if change < 0 else '➡️'
                    }
            
            return market_data
    except Exception as e:
        return {}

def send_discord_alert(setup_data):
    """Send alert to Discord"""
    try:
        if 'discord_webhook' not in st.secrets:
            return False
            
        webhook_url = st.secrets['discord_webhook']
        
        embed = {
            "title": f"🚀 Dashboard Alert: {setup_data['symbol']}",
            "color": 0x00FF00,
            "fields": [
                {
                    "name": "Strategy",
                    "value": setup_data['strategy'],
                    "inline": True
                },
                {
                    "name": "Confidence", 
                    "value": f"{setup_data['confidence']:.1f}%",
                    "inline": True
                },
                {
                    "name": "Entry",
                    "value": setup_data['entry_criteria'],
                    "inline": False
                }
            ],
            "timestamp": datetime.now().isoformat()
        }
        
        response = requests.post(webhook_url, json={"embeds": [embed]})
        return response.status_code == 204
        
    except Exception as e:
        return False

# Main Dashboard
def main():
    # Hero Section with enhanced holographic effect
    st.markdown("""
    <div class="hero-container">
        <div class="hologram-text">⚡ GEX COMMAND CENTER ⚡</div>
        <p style="text-align: center; font-size: 1.4rem; margin-top: 1.5rem; color: rgba(255,255,255,0.9);">
            Enhanced Market Regime Analysis • Live Gamma Exposure Intelligence • Automated Trading Signals
        </p>
        <p style="text-align: center; font-size: 1rem; margin-top: 0.5rem; color: #00ffff;">
            🔮 Powered by Advanced Databricks Pipeline • Real-time Discord Integration
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load all data
    pipeline_results = load_pipeline_results()
    interesting_conditions = load_interesting_conditions()
    regime_data = load_market_regime_data()
    market_data = load_live_market_data()
    
    # Enhanced Sidebar
    with st.sidebar:
        st.markdown("### 🎛️ COMMAND PANEL")
        
        # Auto-refresh with cyber styling
        auto_refresh = st.toggle("🔄 AUTO REFRESH (5min)", value=True)
        
        if st.button("🚀 REFRESH NOW"):
            st.cache_data.clear()
            st.rerun()
        
        st.markdown("### ⚙️ SETTINGS")
        confidence_filter = st.slider("MIN CONFIDENCE %", 0, 100, 60)
        show_regime_details = st.toggle("🌐 Regime Details", value=True)
        
        st.markdown("### 📊 LIVE MARKET")
        if market_data:
            for ticker, data in market_data.items():
                with st.container():
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.metric(
                            label=ticker.replace('^', ''),
                            value=f"${data['price']:.2f}" if ticker != '^VIX' else f"{data['price']:.2f}",
                            delta=f"{data['change']:.2f}%"
                        )
                    with col2:
                        st.markdown(f"<h2 style='margin:0;'>{data['direction']}</h2>", unsafe_allow_html=True)
        
        # Market Regime Indicator
        st.markdown("### 🌐 MARKET REGIME")
        regime_class = "regime-high-suppression" if "SUPPRESSION" in regime_data['market_state'] else "regime-negative-gex" if "AMPLIFICATION" in regime_data['market_state'] else "regime-neutral"
        
        st.markdown(f"""
        <div class="{regime_class}" style="padding: 1rem; border-radius: 10px; margin: 1rem 0;">
            <h4 style="margin: 0; color: white;">{regime_data['market_state']}</h4>
            <p style="margin: 0.5rem 0 0 0; color: rgba(255,255,255,0.8);">{regime_data['description']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Enhanced Metrics Row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    metrics_data = [
        ("🎯 SETUPS TODAY", len(pipeline_results), f"{regime_data['high_confidence_setups']} High Priority", "neon-cyan"),
        ("📊 AVG CONFIDENCE", f"{pipeline_results['confidence'].mean():.1f}%" if not pipeline_results.empty else "0%", "Quality Score", "neon-pink"),
        ("🌐 MARKET GEX", f"{regime_data['total_market_gex']:.1f}B", regime_data['market_state'], "neon-green"),
        ("📡 VIX LEVEL", f"{regime_data['vix_level']:.1f}", "Volatility Gauge", "neon-orange"),
        ("⚡ SUCCESS RATE", f"{(regime_data['setups_found']/regime_data['symbols_processed']*100):.1f}%" if regime_data['symbols_processed'] > 0 else "0%", f"{regime_data['symbols_processed']} Scanned", "neon-cyan")
    ]
    
    cols = [col1, col2, col3, col4, col5]
    
    for i, (title, value, subtitle, color_class) in enumerate(metrics_data):
        with cols[i]:
            st.markdown(f"""
            <div class="metric-card">
                <h3 class="{color_class}">{title}</h3>
                <div class="metric-large {color_class}">{value}</div>
                <div class="metric-label">{subtitle}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Live Trading Opportunities
    st.markdown('<div class="section-header">🚀 LIVE TRADING OPPORTUNITIES</div>', unsafe_allow_html=True)
    
    if not pipeline_results.empty:
        filtered_results = pipeline_results[pipeline_results['confidence'] >= confidence_filter]
        
        if not filtered_results.empty:
            for idx, setup in filtered_results.iterrows():
                card_class = "setup-high-priority" if setup.get('strategy', '').startswith('HIGH') else "setup-moderate"
                confidence_color = "neon-green" if setup['confidence'] >= 85 else "neon-orange" if setup['confidence'] >= 75 else "neon-cyan"
                
                st.markdown(f"""
                <div class="{card_class}">
                    <div style="display: flex; justify-content: space-between; align-items: start;">
                        <div style="flex: 1;">
                            <h3 class="neon-cyan" style="margin: 0 0 0.5rem 0;">{setup['symbol']} • {setup['strategy']}</h3>
                            <p style="margin: 0.5rem 0; color: white; font-size: 1.1rem;">
                                <span class="{confidence_color}"><strong>Confidence: {setup['confidence']:.0f}%</strong></span> | 
                                <span style="color: #ffffff;"><strong>Type:</strong> {setup.get('setup_type', 'N/A')}</span> | 
                                <span style="color: #ffffff;"><strong>Win Rate:</strong> {setup.get('historical_win_rate', 'N/A')}</span>
                            </p>
                            <p style="margin: 0.5rem 0; color: #90EE90; font-size: 1rem;">
                                <strong>💰 Entry:</strong> {setup['entry_criteria']}
                            </p>
                            <p style="margin: 0.5rem 0; color: #FFD700; font-size: 1rem;">
                                <strong>🎯 Target:</strong> ${setup['target']:.2f} | 
                                <strong>📊 Size:</strong> {setup['position_size']} | 
                                <strong>⏰ Time:</strong> {setup['time_frame']}
                            </p>
                            <p style="margin: 0.5rem 0 0 0; color: rgba(255,255,255,0.8); font-size: 0.9rem; font-style: italic;">
                                💡 {setup.get('reason', 'GEX-based opportunity')}
                            </p>
                        </div>
                        <div style="text-align: center; margin-left: 2rem;">
                            <div style="font-size: 3rem; opacity: 0.7;">
                                {'🚀' if setup.get('setup_type') == 'squeeze_play' else '📉' if setup.get('setup_type') == 'breakdown_play' else '💰'}
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Enhanced alert button
                if setup['confidence'] >= 75:
                    col_space1, col_alert, col_space2 = st.columns([6, 2, 2])
                    with col_alert:
                        if st.button(f"🚨 ALERT {setup['symbol']}", key=f"alert_{idx}"):
                            if send_discord_alert(setup):
                                st.success("🚀 Alert sent!")
                            else:
                                st.error("❌ Alert failed")
        else:
            st.info(f"🔍 No setups found with {confidence_filter}%+ confidence. Lower the threshold to see more opportunities.")
    else:
        st.info("📡 No pipeline results available. Run your enhanced Databricks pipeline to see live data.")
    
    # Interesting GEX Conditions
    if not interesting_conditions.empty:
        st.markdown('<div class="section-header">🔍 INTERESTING GEX CONDITIONS</div>', unsafe_allow_html=True)
        
        for idx, condition in interesting_conditions.iterrows():
            regime_color = {
                'HIGH_POSITIVE_GEX': 'neon-orange',
                'NEGATIVE_GEX': 'neon-green', 
                'NEAR_FLIP': 'neon-pink',
                'MODERATE_POSITIVE_GEX': 'neon-cyan'
            }.get(condition['regime'], 'neon-cyan')
            
            st.markdown(f"""
            <div style="background: rgba(0,50,100,0.2); padding: 1rem; border-radius: 8px; border-left: 3px solid; margin: 0.5rem 0;">
                <strong class="{regime_color}">{condition['symbol']}</strong> | 
                <strong class="{regime_color}">{condition['regime']}</strong> | 
                <span style="color: white;">{condition['gex_display']}</span> | 
                <span style="color: white;">{condition['distance_display']}</span>
                <br>
                <em style="color: #FFD700;">Action: {condition['action']}</em>
            </div>
            """, unsafe_allow_html=True)
    
    # Enhanced Market Regime Analysis
    if show_regime_details:
        st.markdown('<div class="section-header">🌐 MARKET REGIME ANALYSIS</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Regime strength gauge
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = regime_data['regime_strength'],
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Regime Strength"},
                gauge = {
                    'axis': {'range': [None, 1]},
                    'bar': {'color': "#00ffff"},
                    'steps': [
                        {'range': [0, 0.4], 'color': "#333"},
                        {'range': [0.4, 0.7], 'color': "#666"},
                        {'range': [0.7, 1], 'color': "#999"}
                    ],
                    'threshold': {
                        'line': {'color': "#ff00ff", 'width': 4},
                        'thickness': 0.75,
                        'value': 0.8
                    }
                }
            ))
            fig_gauge.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                height=300
            )
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        with col2:
            # Setup distribution
            if not pipeline_results.empty:
                setup_counts = pipeline_results.groupby('setup_type').size().reset_index(name='count')
                
                fig_pie = px.pie(
                    setup_counts, 
                    values='count', 
                    names='setup_type',
                    title='Setup Distribution',
                    color_discrete_sequence=['#00ffff', '#ff00ff', '#00ff00', '#ffff00']
                )
                fig_pie.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='white',
                    height=300
                )
                st.plotly_chart(fig_pie, use_container_width=True)
    
    # Educational Section
    with st.expander("📚 ENHANCED GEX STRATEGY GUIDE"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🎯 What is Enhanced GEX Analysis?
            
            Our **Enhanced Gamma Exposure System** combines:
            
            - **Market Regime Classification**: Identifies overall volatility environment
            - **Multi-Symbol GEX Aggregation**: Total market gamma exposure analysis  
            - **VIX Integration**: Volatility context for better decision making
            - **Advanced Setup Detection**: 8+ different trading opportunities
            - **Confidence Scoring**: ML-enhanced probability assessment
            """)
        
        with col2:
            st.markdown("""
            ### 🚀 Enhanced Trading Strategies
            
            1. **Squeeze Plays**: Exploit negative GEX with regime confirmation
            2. **Breakdown Plays**: Positive GEX breakdown with VIX context  
            3. **Premium Selling**: Wall-based resistance/support with regime filter
            4. **Regime Transitions**: Capitalize on GEX regime changes
            5. **Volatility Arbitrage**: VIX vs GEX regime mismatches
            """)
    
    # Auto-refresh functionality
    if auto_refresh:
        time.sleep(300)  # 5 minutes
        st.rerun()

if __name__ == "__main__":
    main()
