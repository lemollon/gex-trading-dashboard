"""
🚀 GEX Trading Dashboard - YOUR WORKING FOUNDATION (Enhanced)
Based on the code that made you say "databricks connection finally works"
Now with stunning visuals, interactive features, and professional design
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

# Try to import Databricks connector
try:
    from databricks import sql
    DATABRICKS_AVAILABLE = True
except ImportError:
    DATABRICKS_AVAILABLE = False

# Page config with enhanced settings
st.set_page_config(
    page_title="🚀 GEX Master Pro - Live Command Center",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Stunning CSS with advanced animations and professional design
st.markdown("""
<style>
    /* Import premium fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Advanced animated background */
    .stApp {
        background: linear-gradient(-45deg, #667eea, #764ba2, #f093fb, #f5576c);
        background-size: 400% 400%;
        animation: gradientShift 20s ease infinite;
        font-family: 'Inter', sans-serif;
        position: relative;
    }
    
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: 
            radial-gradient(circle at 20% 50%, rgba(102, 126, 234, 0.1) 0%, transparent 50%),
            radial-gradient(circle at 80% 20%, rgba(118, 75, 162, 0.1) 0%, transparent 50%),
            radial-gradient(circle at 40% 80%, rgba(79, 172, 254, 0.1) 0%, transparent 50%);
        pointer-events: none;
        z-index: -1;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Hero header with glassmorphism */
    .hero-header {
        background: linear-gradient(135deg, rgba(0,0,0,0.8), rgba(0,0,0,0.4));
        border-radius: 20px;
        padding: 2.5rem;
        margin-bottom: 2rem;
        border: 1px solid rgba(255,255,255,0.2);
        backdrop-filter: blur(20px);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    
    .hero-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(255,255,255,0.1), transparent);
        animation: shimmer 3s infinite;
    }
    
    @keyframes shimmer {
        0% { transform: translateX(-100%) translateY(-100%) rotate(30deg); }
        100% { transform: translateX(100%) translateY(100%) rotate(30deg); }
    }
    
    /* Premium metric cards */
    .metric-card {
        background: linear-gradient(145deg, rgba(255,255,255,0.15), rgba(255,255,255,0.05));
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0.5rem;
        border: 1px solid rgba(255,255,255,0.2);
        backdrop-filter: blur(15px);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        text-align: center;
        position: relative;
        overflow: hidden;
        cursor: pointer;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s;
    }
    
    .metric-card:hover::before {
        left: 100%;
    }
    
    .metric-card:hover {
        transform: translateY(-10px) scale(1.03);
        box-shadow: 0 20px 40px rgba(0,0,0,0.4);
        border-color: rgba(255,255,255,0.4);
    }
    
    /* Confidence-based setup cards */
    .setup-card {
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.2);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .setup-high {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        box-shadow: 0 8px 25px rgba(79, 172, 254, 0.3);
    }
    
    .setup-medium {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        box-shadow: 0 8px 25px rgba(250, 112, 154, 0.3);
    }
    
    .setup-low {
        background: linear-gradient(135deg, #ff6b6b 0%, #feca57 100%);
        box-shadow: 0 8px 25px rgba(255, 107, 107, 0.3);
    }
    
    .setup-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.4);
    }
    
    /* Animated status indicators */
    .status-live {
        width: 12px;
        height: 12px;
        background: #00f2fe;
        border-radius: 50%;
        display: inline-block;
        animation: pulse 2s infinite;
        margin-right: 8px;
    }
    
    @keyframes pulse {
        0% {
            transform: scale(0.95);
            box-shadow: 0 0 0 0 rgba(0, 242, 254, 0.7);
        }
        70% {
            transform: scale(1);
            box-shadow: 0 0 0 10px rgba(0, 242, 254, 0);
        }
        100% {
            transform: scale(0.95);
            box-shadow: 0 0 0 0 rgba(0, 242, 254, 0);
        }
    }
    
    /* Interactive buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        text-transform: uppercase;
        letter-spacing: 1px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.5);
    }
    
    /* Enhanced sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, rgba(0,0,0,0.9), rgba(0,0,0,0.7));
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255,255,255,0.2);
    }
    
    /* Professional tables */
    .dataframe {
        background: rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    /* Custom headers */
    h1, h2, h3, h4 {
        color: white;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    
    /* Chart containers */
    .plot-container {
        background: rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1rem;
        border: 1px solid rgba(255,255,255,0.2);
        margin: 1rem 0;
    }
    
    /* Loading animations */
    .loading-spinner {
        border: 4px solid rgba(255,255,255,0.1);
        border-radius: 50%;
        border-top: 4px solid #00f2fe;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
        margin: 20px auto;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
</style>
""", unsafe_allow_html=True)

# Initialize connection state
if 'connection_status' not in st.session_state:
    st.session_state.connection_status = 'disconnected'
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = None

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_databricks_data():
    """Load data from YOUR ACTUAL Databricks tables"""
    try:
        if not DATABRICKS_AVAILABLE or not all(key in st.secrets.get("databricks", {}) for key in ["server_hostname", "http_path", "access_token"]):
            # Fallback mode with realistic structure
            st.session_state.connection_status = 'fallback'
            return get_fallback_data()
        
        # Connect to your actual Databricks
        connection = sql.connect(
            server_hostname=st.secrets["databricks"]["server_hostname"],
            http_path=st.secrets["databricks"]["http_path"],
            access_token=st.secrets["databricks"]["access_token"]
        )
        
        cursor = connection.cursor()
        
        # Query your actual tables
        recommendations_query = """
        SELECT 
            recommendation_id,
            symbol,
            strategy,
            confidence_score,
            entry_price,
            target_price,
            stop_price,
            setup_type,
            market_regime,
            net_gex,
            gamma_flip,
            position_size,
            created_timestamp,
            sent_to_discord,
            discord_sent_timestamp
        FROM quant_projects.gex_trading.gex_recommendations
        WHERE created_timestamp >= current_timestamp() - INTERVAL 24 HOURS
        ORDER BY confidence_score DESC, created_timestamp DESC
        """
        
        cursor.execute(recommendations_query)
        recommendations = cursor.fetchall()
        recommendations_columns = [desc[0] for desc in cursor.description]
        recommendations_df = pd.DataFrame(recommendations, columns=recommendations_columns)
        
        # Query pipeline monitoring
        monitoring_query = """
        SELECT 
            run_timestamp,
            status,
            opportunities_processed,
            recommendations_generated,
            recommendations_stored,
            discord_alerts_sent
        FROM quant_projects.gex_trading.pipeline_monitoring
        ORDER BY run_timestamp DESC
        LIMIT 1
        """
        
        cursor.execute(monitoring_query)
        monitoring = cursor.fetchone()
        monitoring_columns = [desc[0] for desc in cursor.description]
        
        # Query pipeline results
        pipeline_query = """
        SELECT 
            run_id,
            symbol,
            confidence_score,
            setup_type,
            net_gex,
            distance_to_flip,
            setup_approved,
            raw_condition,
            created_at
        FROM quant_projects.gex_trading.gex_pipeline_results
        WHERE pipeline_date >= current_date() - INTERVAL 7 DAYS
        ORDER BY confidence_score DESC, created_at DESC
        LIMIT 50
        """
        
        cursor.execute(pipeline_query)
        pipeline_results = cursor.fetchall()
        pipeline_columns = [desc[0] for desc in cursor.description]
        pipeline_df = pd.DataFrame(pipeline_results, columns=pipeline_columns)
        
        cursor.close()
        connection.close()
        
        st.session_state.connection_status = 'connected'
        st.session_state.last_refresh = datetime.now()
        
        return {
            'recommendations': recommendations_df,
            'monitoring': dict(zip(monitoring_columns, monitoring)) if monitoring else {},
            'pipeline_results': pipeline_df,
            'status': 'success'
        }
        
    except Exception as e:
        st.session_state.connection_status = 'error'
        st.error(f"Databricks connection error: {str(e)}")
        return get_fallback_data()

def get_fallback_data():
    """Fallback data with your table structure for development"""
    
    # Recommendations data
    recommendations = pd.DataFrame([
        {
            'recommendation_id': 'REC_001',
            'symbol': 'AMC',
            'strategy': 'SQUEEZE_PLAY',
            'confidence_score': 92,
            'entry_price': 15.45,
            'target_price': 18.50,
            'stop_price': 13.80,
            'setup_type': 'SQUEEZE_PLAY',
            'market_regime': 'NEGATIVE_GEX',
            'net_gex': -1.2e9,
            'gamma_flip': 16.20,
            'position_size': '3%',
            'created_timestamp': datetime.now() - timedelta(hours=2),
            'sent_to_discord': True,
            'discord_sent_timestamp': datetime.now() - timedelta(hours=2)
        },
        {
            'recommendation_id': 'REC_002',
            'symbol': 'TSLA',
            'strategy': 'PREMIUM_SELLING',
            'confidence_score': 88,
            'entry_price': 245.30,
            'target_price': 240.00,
            'stop_price': 255.00,
            'setup_type': 'PREMIUM_SELLING',
            'market_regime': 'POSITIVE_GEX',
            'net_gex': 2.1e9,
            'gamma_flip': 242.80,
            'position_size': '2.5%',
            'created_timestamp': datetime.now() - timedelta(hours=1),
            'sent_to_discord': True,
            'discord_sent_timestamp': datetime.now() - timedelta(hours=1)
        },
        {
            'recommendation_id': 'REC_003',
            'symbol': 'PLTR',
            'strategy': 'GAMMA_FLIP_PLAY',
            'confidence_score': 75,
            'entry_price': 28.90,
            'target_price': 32.00,
            'stop_price': 26.50,
            'setup_type': 'GAMMA_FLIP_PLAY',
            'market_regime': 'NEUTRAL_GEX',
            'net_gex': -0.3e9,
            'gamma_flip': 29.10,
            'position_size': '2%',
            'created_timestamp': datetime.now() - timedelta(hours=3),
            'sent_to_discord': False,
            'discord_sent_timestamp': None
        }
    ])
    
    # Monitoring data
    monitoring = {
        'run_timestamp': datetime.now() - timedelta(minutes=15),
        'status': 'SUCCESS',
        'opportunities_processed': 127,
        'recommendations_generated': 8,
        'recommendations_stored': 8,
        'discord_alerts_sent': 3
    }
    
    # Pipeline results
    pipeline_results = pd.DataFrame([
        {
            'run_id': 'RUN_20250827_091500',
            'symbol': 'AMC',
            'confidence_score': 92,
            'setup_type': 'SQUEEZE_PLAY',
            'net_gex': -1.2e9,
            'distance_to_flip': -2.5,
            'setup_approved': True,
            'raw_condition': 'Strong negative GEX with price below flip point',
            'created_at': datetime.now() - timedelta(minutes=30)
        },
        {
            'run_id': 'RUN_20250827_091500',
            'symbol': 'TSLA',
            'confidence_score': 88,
            'setup_type': 'PREMIUM_SELLING',
            'net_gex': 2.1e9,
            'distance_to_flip': 1.8,
            'setup_approved': True,
            'raw_condition': 'High positive GEX creates volatility suppression',
            'created_at': datetime.now() - timedelta(minutes=30)
        }
    ])
    
    return {
        'recommendations': recommendations,
        'monitoring': monitoring,
        'pipeline_results': pipeline_results,
        'status': 'fallback'
    }

def create_confidence_gauge(confidence):
    """Create animated confidence gauge"""
    
    color = '#4facfe' if confidence >= 80 else '#feca57' if confidence >= 60 else '#ff6b6b'
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = confidence,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Confidence"},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 60], 'color': "rgba(255, 107, 107, 0.2)"},
                {'range': [60, 80], 'color': "rgba(254, 202, 87, 0.2)"},
                {'range': [80, 100], 'color': "rgba(79, 172, 254, 0.2)"}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': confidence
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'color': "white", 'family': "Inter"},
        height=250
    )
    
    return fig

def create_gex_distribution_chart(pipeline_df):
    """Create GEX distribution visualization"""
    
    if pipeline_df.empty:
        return go.Figure()
    
    # Calculate GEX regimes
    positive_gex = len(pipeline_df[pipeline_df['net_gex'] > 0])
    negative_gex = len(pipeline_df[pipeline_df['net_gex'] < 0])
    
    fig = go.Figure(data=[
        go.Pie(
            labels=['Positive GEX', 'Negative GEX'],
            values=[positive_gex, negative_gex],
            hole=0.4,
            marker_colors=['#4facfe', '#ff6b6b'],
            textfont={'size': 14, 'color': 'white'},
            hovertemplate="<b>%{label}</b><br>" +
                         "%{value} symbols<br>" +
                         "%{percent}<br>" +
                         "<extra></extra>"
        )
    ])
    
    fig.update_layout(
        title="Market Regime Distribution",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'color': 'white', 'family': 'Inter'},
        height=300,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_confidence_distribution(pipeline_df):
    """Create confidence distribution chart"""
    
    if pipeline_df.empty:
        return go.Figure()
    
    # Calculate confidence ranges
    high_conf = len(pipeline_df[pipeline_df['confidence_score'] >= 80])
    med_conf = len(pipeline_df[(pipeline_df['confidence_score'] >= 60) & (pipeline_df['confidence_score'] < 80)])
    low_conf = len(pipeline_df[pipeline_df['confidence_score'] < 60])
    
    fig = go.Figure(data=[
        go.Bar(
            x=['High (80%+)', 'Medium (60-79%)', 'Low (<60%)'],
            y=[high_conf, med_conf, low_conf],
            marker_color=['#4facfe', '#feca57', '#ff6b6b'],
            text=[high_conf, med_conf, low_conf],
            textposition='auto',
            hovertemplate="<b>%{x}</b><br>" +
                         "%{y} setups<br>" +
                         "<extra></extra>"
        )
    ])
    
    fig.update_layout(
        title="Setup Confidence Distribution",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'color': 'white', 'family': 'Inter'},
        height=300,
        xaxis_title="Confidence Level",
        yaxis_title="Number of Setups"
    )
    
    return fig

# Main dashboard layout
def main():
    
    # Hero header
    st.markdown("""
    <div class="hero-header">
        <h1 style="font-size: 3rem; margin-bottom: 0.5rem;">⚡ GEX Master Pro</h1>
        <h2 style="font-size: 1.5rem; color: #a0a9c0; margin-top: 0;">
            Live Databricks Pipeline Command Center
        </h2>
        <div style="margin-top: 1rem;">
            <span class="status-live"></span>
            <span style="font-size: 1.1rem;">Professional Gamma Exposure Analysis Platform</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar controls
    with st.sidebar:
        st.markdown("### 🎛️ Command Center")
        
        # Connection status
        if st.session_state.connection_status == 'connected':
            st.success("✅ Connected to Databricks")
            st.info(f"🔄 Last refresh: {st.session_state.last_refresh.strftime('%H:%M:%S') if st.session_state.last_refresh else 'Never'}")
        elif st.session_state.connection_status == 'fallback':
            st.warning("🔧 Development Mode")
            st.info("Add Databricks secrets for live data")
        else:
            st.error("❌ Connection Error")
        
        st.markdown("---")
        
        # Auto-refresh toggle
        auto_refresh = st.checkbox("🔄 Auto-refresh (30s)")
        if auto_refresh:
            time.sleep(30)
            st.experimental_rerun()
        
        # Manual refresh button
        if st.button("🚀 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.experimental_rerun()
        
        st.markdown("---")
        
        # Quick filters
        st.markdown("### 🎯 Filters")
        min_confidence = st.slider("Min Confidence", 0, 100, 70)
        
        setup_types = st.multiselect(
            "Setup Types",
            ['SQUEEZE_PLAY', 'PREMIUM_SELLING', 'GAMMA_FLIP_PLAY'],
            default=['SQUEEZE_PLAY', 'PREMIUM_SELLING']
        )
        
        st.markdown("---")
        st.markdown("### 📚 Quick Reference")
        st.markdown("""
        **🔴 Negative GEX**: Volatility amplification  
        **🔵 Positive GEX**: Volatility suppression  
        **🟡 Gamma Flip**: Zero-gamma crossing point  
        **📈 Call Wall**: Dealer resistance level  
        **📉 Put Wall**: Dealer support level  
        """)
    
    # Load data
    with st.spinner("Loading live data from Databricks..."):
        data = load_databricks_data()
    
    if data['status'] in ['success', 'fallback']:
        recommendations_df = data['recommendations']
        monitoring = data['monitoring']
        pipeline_df = data['pipeline_results']
        
        # Apply filters
        if not recommendations_df.empty:
            filtered_recs = recommendations_df[
                (recommendations_df['confidence_score'] >= min_confidence) &
                (recommendations_df['setup_type'].isin(setup_types) if setup_types else True)
            ]
        else:
            filtered_recs = recommendations_df
        
        # Key metrics row
        st.markdown("### 📊 Live Pipeline Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>🔄 Pipeline Status</h3>
                <h2 style="color: #4facfe;">ACTIVE</h2>
                <p>Last run: {}</p>
            </div>
            """.format(
                monitoring.get('run_timestamp', datetime.now()).strftime('%H:%M:%S') if monitoring.get('run_timestamp') else 'N/A'
            ), unsafe_allow_html=True)
        
        with col2:
            opportunities = monitoring.get('opportunities_processed', 0)
            st.markdown(f"""
            <div class="metric-card">
                <h3>📊 Opportunities</h3>
                <h2 style="color: #00f2fe;">{opportunities}</h2>
                <p>Symbols analyzed</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            high_conf_count = len(filtered_recs[filtered_recs['confidence_score'] >= 85]) if not filtered_recs.empty else 0
            st.markdown(f"""
            <div class="metric-card">
                <h3>🎯 High Confidence</h3>
                <h2 style="color: #4facfe;">{high_conf_count}</h2>
                <p>Premium setups</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            discord_alerts = monitoring.get('discord_alerts_sent', 0)
            st.markdown(f"""
            <div class="metric-card">
                <h3>🔔 Discord Alerts</h3>
                <h2 style="color: #feca57;">{discord_alerts}</h2>
                <p>Notifications sent</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Current high-confidence setups
        st.markdown("### 🎯 Live High-Confidence Trading Setups")
        
        if not filtered_recs.empty:
            high_conf_setups = filtered_recs[filtered_recs['confidence_score'] >= 75].head(3)
            
            if not high_conf_setups.empty:
                setup_cols = st.columns(min(len(high_conf_setups), 3))
                
                for i, (_, setup) in enumerate(high_conf_setups.iterrows()):
                    with setup_cols[i]:
                        confidence = setup['confidence_score']
                        card_class = 'setup-high' if confidence >= 85 else 'setup-medium' if confidence >= 70 else 'setup-low'
                        
                        st.markdown(f"""
                        <div class="{card_class} setup-card">
                            <h3 style="margin: 0; color: white;">{setup['symbol']}</h3>
                            <h2 style="margin: 0.5rem 0; color: white;">{confidence}% Confidence</h2>
                            <p style="margin: 0.5rem 0; color: white;"><strong>{setup['setup_type'].replace('_', ' ')}</strong></p>
                            <p style="margin: 0.5rem 0; color: rgba(255,255,255,0.9);">
                                Entry: ${setup['entry_price']:.2f}<br>
                                Target: ${setup['target_price']:.2f}<br>
                                GEX: {setup['net_gex']/1e9:.2f}B
                            </p>
                            <p style="font-size: 0.9rem; color: rgba(255,255,255,0.8);">
                                {setup['market_regime'].replace('_', ' ')} Regime
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Add confidence gauge
                        fig_gauge = create_confidence_gauge(confidence)
                        st.plotly_chart(fig_gauge, use_container_width=True, config={'displayModeBar': False})
            else:
                st.info("🎯 No high-confidence setups match your current filters")
        else:
            st.info("📊 No recommendations available - check your Databricks connection")
        
        # Analysis charts
        st.markdown("### 📈 Market Analysis")
        
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
            gex_chart = create_gex_distribution_chart(pipeline_df)
            st.plotly_chart(gex_chart, use_container_width=True, config={'displayModeBar': False})
            st.markdown('</div>', unsafe_allow_html=True)
        
        with chart_col2:
            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
            conf_chart = create_confidence_distribution(pipeline_df)
            st.plotly_chart(conf_chart, use_container_width=True, config={'displayModeBar': False})
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Detailed recommendations table
        st.markdown("### 💼 Complete Trading Recommendations")
        
        if not filtered_recs.empty:
            # Style the dataframe
            styled_df = filtered_recs.copy()
            
            # Format columns for display
            display_cols = [
                'symbol', 'setup_type', 'confidence_score', 'entry_price', 
                'target_price', 'position_size', 'market_regime', 'created_timestamp'
            ]
            
            display_df = styled_df[display_cols].copy()
            display_df['confidence_score'] = display_df['confidence_score'].apply(lambda x: f"{x}%")
            display_df['entry_price'] = display_df['entry_price'].apply(lambda x: f"${x:.2f}")
            display_df['target_price'] = display_df['target_price'].apply(lambda x: f"${x:.2f}")
            display_df['created_timestamp'] = display_df['created_timestamp'].dt.strftime('%H:%M:%S')
            
            # Rename columns for display
            display_df.columns = [
                'Symbol', 'Setup Type', 'Confidence', 'Entry', 
                'Target', 'Position Size', 'Market Regime', 'Created'
            ]
            
            st.dataframe(
                display_df,
                use_container_width=True,
                height=400
            )
            
            # Download button
            csv = filtered_recs.to_csv(index=False)
            st.download_button(
                label="📥 Download Recommendations CSV",
                data=csv,
                file_name=f"gex_recommendations_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
        else:
            st.info("📋 No recommendations match your current filters")
        
        # Raw pipeline data (expandable)
        with st.expander("🔍 Raw Pipeline Data", expanded=False):
            if not pipeline_df.empty:
                st.dataframe(pipeline_df, use_container_width=True)
            else:
                st.info("No pipeline data available")
    
    else:
        st.error("❌ Failed to load data from Databricks")
        st.info("Check your connection settings and try refreshing")

if __name__ == "__main__":
    main()
