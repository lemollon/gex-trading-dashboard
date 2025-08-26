"""
🚀 GEX Trading Dashboard - Live Databricks Integration
Beautiful, informative dashboard connected to your real pipeline data
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

# Page config
st.set_page_config(
    page_title="GEX Trading Command Center",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Stunning CSS with enhanced visuals
st.markdown("""
<style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main background with animated gradient */
    .main {
        background: linear-gradient(-45deg, #667eea, #764ba2, #f093fb, #f5576c);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
        color: white;
        min-height: 100vh;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Hero section with glassmorphism */
    .hero-container {
        background: linear-gradient(135deg, rgba(0,0,0,0.8), rgba(0,0,0,0.4));
        border-radius: 20px;
        padding: 2.5rem;
        margin-bottom: 2rem;
        border: 1px solid rgba(255,255,255,0.2);
        backdrop-filter: blur(20px);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    
    /* Premium metric cards */
    .metric-card {
        background: linear-gradient(145deg, rgba(255,255,255,0.15), rgba(255,255,255,0.05));
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem;
        border: 1px solid rgba(255,255,255,0.2);
        backdrop-filter: blur(15px);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card:before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
        transition: left 0.5s;
    }
    
    .metric-card:hover:before {
        left: 100%;
    }
    
    .metric-card:hover {
        transform: translateY(-10px) scale(1.02);
        box-shadow: 0 20px 40px rgba(0,0,0,0.4);
        border-color: rgba(255,255,255,0.4);
    }
    
    /* Enhanced setup cards */
    .setup-card {
        background: linear-gradient(145deg, rgba(255,255,255,0.15), rgba(255,255,255,0.05));
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 5px solid;
        backdrop-filter: blur(10px);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .setup-card:hover {
        transform: translateX(8px);
        background: linear-gradient(145deg, rgba(255,255,255,0.25), rgba(255,255,255,0.15));
        box-shadow: 0 15px 35px rgba(0,0,0,0.3);
    }
    
    /* Confidence-based styling */
    .confidence-high { 
        border-left-color: #00ff88;
        box-shadow: 0 0 20px rgba(0,255,136,0.3);
    }
    .confidence-medium { 
        border-left-color: #ffaa00;
        box-shadow: 0 0 20px rgba(255,170,0,0.3);
    }
    .confidence-low { 
        border-left-color: #ff4444;
        box-shadow: 0 0 20px rgba(255,68,68,0.3);
    }
    
    /* Animated status indicators */
    .status-indicator {
        display: inline-block;
        width: 14px;
        height: 14px;
        border-radius: 50%;
        margin-right: 10px;
        animation: pulse 2s infinite;
        box-shadow: 0 0 10px currentColor;
    }
    
    .status-live { 
        background: linear-gradient(45deg, #00ff88, #00cc70);
        animation: livePulse 2s infinite;
    }
    .status-error { 
        background: linear-gradient(45deg, #ff4444, #cc3333);
        animation: errorPulse 1s infinite;
    }
    
    @keyframes livePulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.7; transform: scale(1.1); }
    }
    
    @keyframes errorPulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.4; }
    }
    
    /* Typography enhancements */
    .big-number {
        font-size: 3rem;
        font-weight: 800;
        line-height: 1;
        text-shadow: 0 2px 10px rgba(0,0,0,0.3);
        background: linear-gradient(45deg, #ffffff, #f0f0f0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .setup-title {
        font-size: 1.2rem;
        font-weight: 700;
        margin-bottom: 0.8rem;
        text-shadow: 0 1px 3px rgba(0,0,0,0.3);
    }
    
    /* Error and loading containers */
    .error-container {
        background: linear-gradient(145deg, rgba(255,68,68,0.2), rgba(255,68,68,0.1));
        border-radius: 16px;
        padding: 2rem;
        margin: 2rem 0;
        border: 2px solid rgba(255,68,68,0.4);
        text-align: center;
        backdrop-filter: blur(10px);
    }
    
    .loading-container {
        background: linear-gradient(145deg, rgba(255,170,0,0.2), rgba(255,170,0,0.1));
        border-radius: 16px;
        padding: 2rem;
        margin: 2rem 0;
        border: 2px solid rgba(255,170,0,0.4);
        text-align: center;
        backdrop-filter: blur(10px);
    }
    
    .success-container {
        background: linear-gradient(145deg, rgba(0,255,136,0.2), rgba(0,255,136,0.1));
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 2px solid rgba(0,255,136,0.4);
        backdrop-filter: blur(10px);
    }
    
    /* Button enhancements */
    .stButton > button {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
    }
    
    /* Performance metrics styling */
    .perf-metric {
        background: linear-gradient(135deg, rgba(255,255,255,0.1), rgba(255,255,255,0.05));
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 3px solid #00aaff;
        backdrop-filter: blur(5px);
    }
    
    /* Chart container */
    .chart-container {
        background: linear-gradient(145deg, rgba(255,255,255,0.1), rgba(255,255,255,0.05));
        border-radius: 16px;
        padding: 1rem;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Databricks connection
@st.cache_resource
def init_databricks_connection():
    """Initialize Databricks connection"""
    try:
        from databricks import sql
        return True
    except ImportError:
        st.error("📦 Install databricks-sql-connector: pip install databricks-sql-connector")
        return False

@st.cache_data(ttl=180)  # Cache for 3 minutes - fresher data
def load_live_databricks_data():
    """Load live data from your Databricks pipeline"""
    
    try:
        from databricks import sql
        
        # Connect using your secrets
        connection = sql.connect(
            server_hostname=st.secrets["databricks"]["server_hostname"],
            http_path=st.secrets["databricks"]["http_path"], 
            access_token=st.secrets["databricks"]["access_token"]
        )
        
        cursor = connection.cursor()
        
        # Query your pipeline results - last 7 days
        query = """
        SELECT *
        FROM quant_projects.gex_trading.gex_pipeline_results
        WHERE pipeline_date >= current_date() - INTERVAL 7 DAYS
        ORDER BY run_timestamp DESC
        LIMIT 200
        """
        
        cursor.execute(query)
        results = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        
        cursor.close()
        connection.close()
        
        if not results:
            return create_empty_response(), "Connected but no recent data"
        
        # Convert to DataFrame
        df = pd.DataFrame(results, columns=columns)
        
        # Get latest run
        latest_run_id = df['run_id'].iloc[0]
        latest_data = df[df['run_id'] == latest_run_id]
        
        # Convert to dashboard format
        dashboard_data = convert_databricks_to_dashboard_format(latest_data, df)
        
        return dashboard_data, "Live from Databricks"
        
    except Exception as e:
        st.error(f"Databricks connection failed: {e}")
        return create_empty_response(), f"Connection failed: {str(e)[:100]}"

def convert_databricks_to_dashboard_format(latest_data, full_df):
    """Convert your Databricks data to dashboard format"""
    
    if latest_data.empty:
        return create_empty_response()
    
    # Get metadata
    first_row = latest_data.iloc[0]
    
    # Get approved setups from latest run
    approved_setups = latest_data[
        (latest_data['setup_approved'] == True) & 
        (latest_data['symbol'] != 'NO_SETUPS')
    ]
    
    # Create trading setups
    trading_setups = []
    for _, row in approved_setups.iterrows():
        
        # Map your conditions to setup types
        setup_type_mapping = {
            'NEGATIVE_GEX': 'SQUEEZE_PLAY',
            'HIGH_POSITIVE_GEX': 'PREMIUM_SELLING', 
            'NEAR_FLIP': 'GAMMA_FLIP'
        }
        
        direction_mapping = {
            'NEGATIVE_GEX': 'LONG_CALLS',
            'HIGH_POSITIVE_GEX': 'SHORT_CALLS',
            'NEAR_FLIP': 'VOLATILITY'
        }
        
        setup_data = {
            'setup': {
                'symbol': row['symbol'],
                'setup_type': setup_type_mapping.get(row['condition_type'], row['condition_type']),
                'direction': direction_mapping.get(row['condition_type'], 'DIRECTIONAL'),
                'confidence': float(row['confidence_score']),
                'reason': f"Pipeline Analysis: GEX {row['net_gex']/1e9:+.1f}B, {row['distance_to_flip']:+.2f}% from flip",
                'expected_move': float(row['expected_move']) * 100,  # Convert to percentage
                'hold_days': 3,
                'risk_level': 'MEDIUM' if row['confidence_score'] >= 80 else 'HIGH'
            },
            'position_size_percent': float(row['position_size_percent']),
            'dollar_amount': int(row['position_size_percent'] * 1000),
            'approved': bool(row['setup_approved']),
            'databricks_data': {
                'run_id': row['run_id'],
                'net_gex': int(row['net_gex']),
                'distance_to_flip': float(row['distance_to_flip'])
            }
        }
        
        trading_setups.append(setup_data)
    
    # Calculate market summary from latest data
    market_data = latest_data[latest_data['symbol'] != 'NO_SETUPS']
    
    if not market_data.empty:
        net_gex_values = market_data['net_gex'].tolist()
        total_net_gex = sum(net_gex_values) / 1e9
        
        # Count conditions
        condition_counts = market_data['condition_type'].value_counts().to_dict()
        negative_count = condition_counts.get('NEGATIVE_GEX', 0)
        positive_count = condition_counts.get('HIGH_POSITIVE_GEX', 0) 
        near_flip_count = condition_counts.get('NEAR_FLIP', 0)
        
        # Determine regime
        if negative_count > max(positive_count, near_flip_count):
            dominant_regime = 'NEGATIVE_GEX'
            stress_level = 'HIGH'
        elif positive_count > max(negative_count, near_flip_count):
            dominant_regime = 'HIGH_POSITIVE_GEX'
            stress_level = 'LOW'
        elif near_flip_count > 0:
            dominant_regime = 'NEAR_FLIP'
            stress_level = 'MEDIUM'
        else:
            dominant_regime = 'NEUTRAL'
            stress_level = 'LOW'
    else:
        total_net_gex = 0
        dominant_regime = 'NEUTRAL'
        stress_level = 'LOW'
        negative_count = positive_count = near_flip_count = 0
    
    # Historical performance metrics
    historical_metrics = calculate_historical_performance(full_df)
    
    return {
        'success': True,
        'analysis_time': pd.to_datetime(first_row['run_timestamp']),
        'pipeline_run_time': first_row['run_timestamp'],
        'symbols_analyzed': int(first_row['total_symbols_analyzed']),
        'trading_setups': trading_setups,
        'market_summary': {
            'total_net_gex_billions': total_net_gex,
            'dominant_regime': dominant_regime,
            'symbols_near_flip': near_flip_count,
            'market_stress_level': stress_level,
            'total_conditions_found': len(market_data),
            'regime_distribution': {
                'NEGATIVE_GEX': negative_count,
                'HIGH_POSITIVE_GEX': positive_count,
                'NEAR_FLIP': near_flip_count
            }
        },
        'risk_assessment': {
            'total_risk_percent': sum(s['position_size_percent'] for s in trading_setups),
            'num_positions': len(trading_setups),
            'risk_level': 'HIGH' if len(trading_setups) > 3 else 'MEDIUM' if len(trading_setups) > 1 else 'LOW'
        },
        'historical_metrics': historical_metrics,
        'raw_pipeline_data': {
            'latest_run_id': first_row['run_id'],
            'total_runs_analyzed': len(full_df['run_id'].unique()),
            'data_quality': 'EXCELLENT' if first_row['pipeline_success'] else 'DEGRADED'
        }
    }

def calculate_historical_performance(full_df):
    """Calculate historical performance metrics from your pipeline data"""
    
    if full_df.empty:
        return {}
    
    # Group by runs to get performance over time
    run_summary = full_df.groupby('run_id').agg({
        'setup_approved': 'sum',
        'total_setups_found': 'first', 
        'total_symbols_analyzed': 'first',
        'pipeline_date': 'first',
        'confidence_score': 'mean'
    }).reset_index()
    
    # Calculate metrics
    avg_setups_per_run = run_summary['setup_approved'].mean()
    success_rate = (run_summary['setup_approved'] > 0).mean() * 100
    avg_confidence = run_summary['confidence_score'].mean()
    
    # Recent vs historical comparison
    recent_runs = run_summary.tail(5)
    recent_avg = recent_runs['setup_approved'].mean() if not recent_runs.empty else 0
    
    return {
        'total_pipeline_runs': len(run_summary),
        'avg_setups_per_run': avg_setups_per_run,
        'pipeline_success_rate': success_rate,
        'avg_confidence_score': avg_confidence,
        'recent_performance': {
            'recent_avg_setups': recent_avg,
            'trend': 'IMPROVING' if recent_avg > avg_setups_per_run else 'DECLINING' if recent_avg < avg_setups_per_run else 'STABLE'
        }
    }

def create_empty_response():
    """Empty response when no data"""
    return {
        'success': True,
        'analysis_time': datetime.now(),
        'symbols_analyzed': 0,
        'trading_setups': [],
        'market_summary': {
            'total_net_gex_billions': 0,
            'dominant_regime': 'NEUTRAL',
            'symbols_near_flip': 0,
            'market_stress_level': 'LOW'
        },
        'risk_assessment': {
            'total_risk_percent': 0,
            'num_positions': 0,
            'risk_level': 'NONE'
        },
        'historical_metrics': {}
    }

def render_hero_section():
    """Beautiful hero section"""
    st.markdown("""
    <div class="hero-container">
        <h1 style="text-align: center; font-size: 3.5rem; margin-bottom: 0.5rem; text-shadow: 0 4px 20px rgba(0,0,0,0.5);">
            🚀 GEX Trading Command Center
        </h1>
        <p style="text-align: center; font-size: 1.3rem; opacity: 0.9; margin-bottom: 0; text-shadow: 0 2px 10px rgba(0,0,0,0.3);">
            Live Databricks Pipeline • Real Gamma Exposure Analysis • Institutional-Grade Setups
        </p>
        <div style="text-align: center; margin-top: 1rem; opacity: 0.7;">
            <span style="font-size: 0.9rem;">🔥 Connected to your actual GEX pipeline • Zero mock data</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_connection_status(source, last_update, data_quality="UNKNOWN"):
    """Enhanced connection status"""
    
    if "Live from Databricks" in source:
        status_class = "status-live"
        status_text = "🟢 LIVE CONNECTION"
        status_detail = f"Pipeline run: {last_update.strftime('%Y-%m-%d %H:%M:%S')}"
        quality_color = "#00ff88" if data_quality == "EXCELLENT" else "#ffaa00"
    else:
        status_class = "status-error"
        status_text = "🔴 CONNECTION ISSUE"
        status_detail = source
        quality_color = "#ff4444"
    
    st.markdown(f"""
    <div style="text-align: right; margin-bottom: 1.5rem;">
        <div style="display: inline-block; padding: 1rem; background: linear-gradient(145deg, rgba(255,255,255,0.1), rgba(255,255,255,0.05)); 
                    border-radius: 12px; backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.2);">
            <span class="status-indicator {status_class}"></span>
            <span style="font-weight: bold; font-size: 1.1rem;">{status_text}</span>
            <br>
            <span style="opacity: 0.8; font-size: 0.9rem;">{status_detail}</span>
            <br>
            <span style="color: {quality_color}; font-size: 0.8rem; font-weight: 600;">Data Quality: {data_quality}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_enhanced_metrics(data):
    """Enhanced metrics with your real Databricks data"""
    
    market_summary = data['market_summary']
    risk_assessment = data['risk_assessment']
    historical = data.get('historical_metrics', {})
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        net_gex = market_summary['total_net_gex_billions']
        gex_color = "#ff4444" if net_gex < 0 else "#00ff88"
        trend_arrow = "↓" if net_gex < 0 else "↑"
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="big-number" style="color: {gex_color};">
                {net_gex:+.2f}B
            </div>
            <div style="font-size: 1.1rem; font-weight: 600; margin: 0.5rem 0;">Net GEX</div>
            <div style="font-size: 0.9rem; opacity: 0.8;">
                From Your Pipeline {trend_arrow}
            </div>
            <div style="font-size: 0.8rem; opacity: 0.6; margin-top: 0.5rem;">
                {market_summary['dominant_regime'].replace('_', ' ').title()}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        approved_setups = len([s for s in data['trading_setups'] if s['approved']])
        avg_setups = historical.get('avg_setups_per_run', 0)
        comparison = "🔥" if approved_setups > avg_setups else "📊" if approved_setups == avg_setups else "🔍"
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="big-number" style="color: #00aaff;">
                {approved_setups}
            </div>
            <div style="font-size: 1.1rem; font-weight: 600; margin: 0.5rem 0;">Live Setups</div>
            <div style="font-size: 0.9rem; opacity: 0.8;">
                High Confidence {comparison}
            </div>
            <div style="font-size: 0.8rem; opacity: 0.6; margin-top: 0.5rem;">
                Avg: {avg_setups:.1f} per run
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        stress_level = market_summary['market_stress_level']
        stress_colors = {'LOW': '#00ff88', 'MEDIUM': '#ffaa00', 'HIGH': '#ff4444'}
        stress_color = stress_colors.get(stress_level, '#ffffff')
        stress_emoji = {'LOW': '😌', 'MEDIUM': '😐', 'HIGH': '😰'}.get(stress_level, '🤔')
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="big-number" style="color: {stress_color};">
                {stress_level}
            </div>
            <div style="font-size: 1.1rem; font-weight: 600; margin: 0.5rem 0;">Market Stress</div>
            <div style="font-size: 0.9rem; opacity: 0.8;">
                {market_summary['symbols_near_flip']} Near Flip {stress_emoji}
            </div>
            <div style="font-size: 0.8rem; opacity: 0.6; margin-top: 0.5rem;">
                Pipeline Analysis
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        total_risk = risk_assessment['total_risk_percent']
        risk_color = "#00ff88" if total_risk < 5 else "#ffaa00" if total_risk < 8 else "#ff4444"
        risk_emoji = "✅" if total_risk < 5 else "⚠️" if total_risk < 8 else "🚨"
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="big-number" style="color: {risk_color};">
                {total_risk:.1f}%
            </div>
            <div style="font-size: 1.1rem; font-weight: 600; margin: 0.5rem 0;">Portfolio Risk</div>
            <div style="font-size: 0.9rem; opacity: 0.8;">
                {risk_assessment['num_positions']} Positions {risk_emoji}
            </div>
            <div style="font-size: 0.8rem; opacity: 0.6; margin-top: 0.5rem;">
                Risk Level: {risk_assessment['risk_level']}
            </div>
        </div>
        """, unsafe_allow_html=True)

def render_pipeline_setups(data):
    """Render your actual pipeline setups with enhanced visuals"""
    
    st.markdown("## 🎯 Live Trading Setups from Your Databricks Pipeline")
    
    approved_setups = [s for s in data['trading_setups'] if s['approved']]
    
    if not approved_setups:
        st.markdown("""
        <div class="success-container">
            <h3>📊 No High-Confidence Setups in Latest Pipeline Run</h3>
            <p style="font-size: 1.1rem;">Your GEX analysis didn't find setups meeting the strict criteria.</p>
            <p style="opacity: 0.8;">✅ This is normal and shows quality control is working!</p>
            <div style="margin-top: 1rem; font-size: 0.9rem; opacity: 0.7;">
                <strong>Pipeline Status:</strong> Running successfully • Data being collected for backtesting
            </div>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Show summary first
    avg_confidence = sum(s['setup']['confidence'] for s in approved_setups) / len(approved_setups)
    total_allocation = sum(s['position_size_percent'] for s in approved_setups)
    
    st.markdown(f"""
    <div class="success-container">
        <div style="display: flex; justify-content: space-around; text-align: center;">
            <div>
                <div style="font-size: 1.5rem; font-weight: bold;">{len(approved_setups)}</div>
                <div style="opacity: 0.8;">Active Setups</div>
            </div>
            <div>
                <div style="font-size: 1.5rem; font-weight: bold;">{avg_confidence:.0f}%</div>
                <div style="opacity: 0.8;">Avg Confidence</div>
            </div>
            <div>
                <div style="font-size: 1.5rem; font-weight: bold;">{total_allocation:.1f}%</div>
                <div style="opacity: 0.8;">Total Allocation</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Render each setup
    for i, setup_info in enumerate(approved_setups):
        setup = setup_info['setup']
        databricks_data = setup_info.get('databricks_data', {})
        
        # Enhanced confidence styling
        confidence = setup['confidence']
        if confidence >= 85:
            confidence_class = "confidence-high"
            confidence_emoji = "🟢"
            confidence_label = "PREMIUM"
        elif confidence >= 75:
            confidence_class = "confidence-high" 
            confidence_emoji = "🟢"
            confidence_label = "HIGH"
        elif confidence >= 65:
            confidence_class = "confidence-medium"
            confidence_emoji = "🟡"
            confidence_label = "MEDIUM"
        else:
            confidence_class = "confidence-low"
            confidence_emoji = "🔴"
            confidence_label = "LOW"
        
        # Setup type styling
        type_emojis = {
            'SQUEEZE_PLAY': '🚀',
            'PREMIUM_SELLING': '💰',
            'GAMMA_FLIP': '⚡'
        }
        type_emoji = type_emojis.get(setup['setup_type'], '📊')
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"""
            <div class="setup-card {confidence_class}">
                <div class="setup-title">
                    {type_emoji} {setup['symbol']} - {setup['setup_type'].replace('_', ' ').title()}
                    <span style="float: right; font-size: 0.8rem; opacity: 0.7;">#{i+1}</span>
                </div>
                
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1rem;">
                    <div>
                        <strong>Direction:</strong> {setup['direction'].replace('_', ' ')}
                    </div>
                    <div style="text-align: right;">
                        {confidence_emoji} <strong>{confidence:.0f}% {confidence_label}</strong>
                    </div>
                </div>
                
                <div style="background: rgba(0,0,0,0.2); border-radius: 8px; padding: 0.8rem; margin: 1rem 0;">
                    <div style="font-size: 0.9rem;"><strong>Pipeline Analysis:</strong></div>
                    <div style="margin-top: 0.3rem;">{setup['reason']}</div>
                </div>
                
                <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1rem; font-size: 0.9rem;">
                    <div>
                        <div style="opacity: 0.7;">Position Size</div>
                        <div style="font-weight: bold;">{setup_info['position_size_percent']:.1f}%</div>
                        <div style="font-size: 0.8rem; opacity: 0.6;">${setup_info['dollar_amount']:,.0f}</div>
                    </div>
                    <div>
                        <div style="opacity: 0.7;">Expected Move</div>
                        <div style="font-weight: bold;">{setup['expected_move']:.1f}%</div>
                        <div style="font-size: 0.8rem; opacity: 0.6;">{setup['hold_days']} day hold</div>
                    </div>
                    <div>
                        <div style="opacity: 0.7;">Risk Level</div>
                        <div style="font-weight: bold;">{setup['risk_level']}</div>
                        <div style="font-size: 0.8rem; opacity: 0.6;">Pipeline Calc</div>
                    </div>
                </div>
                
                <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid rgba(255,255,255,0.1); font-size: 0.8rem; opacity: 0.7;">
                    🗄️ <strong>Databricks ID:</strong> {databricks_data.get('run_id', 'N/A')[-12:]} | 
                    <strong>Raw GEX:</strong> {databricks_data.get('net_gex', 0)/1e6:.0f}M
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Action buttons
            st.markdown(f"""
            <div style="display: flex; flex-direction: column; gap: 0.8rem; padding: 1rem;">
                <button style="background: linear-gradient(45deg, #00ff88, #00cc70); border: none; 
                              border-radius: 8px; padding: 0.7rem; color: white; font-weight: bold;
                              cursor: pointer; transition: all 0.3s ease;">
                    📊 Analyze Setup
                </button>
                <button style="background: linear-gradient(45deg, #0088ff, #0066cc); border: none; 
                              border-radius: 8px; padding: 0.7rem; color: white; font-weight: bold;
                              cursor: pointer; transition: all 0.3s ease;">
                    📈 View Chart
                </button>
                <button style="background: linear-gradient(45deg, #ff6b35, #f7931e); border: none; 
                              border-radius: 8px; padding: 0.7rem; color: white; font-weight: bold;
                              cursor: pointer; transition: all 0.3s ease;">
                    ⚡ Quick Trade
                </button>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button(f"📋 Raw Data", key=f"raw_{setup['symbol']}_{i}"):
                st.json(setup_info)

def render_market_overview_chart(data):
    """Enhanced market overview with your data"""
    
    st.markdown("## 📊 Live Market Intelligence")
    
    market_summary = data['market_summary']
    regime_dist = market_summary.get('regime_distribution', {})
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Create regime distribution chart
        if any(regime_dist.values()):
            regime_names = list(regime_dist.keys())
            regime_values = list(regime_dist.values())
            regime_colors = ['#ff4444', '#00ff88', '#ffaa00']
            
            fig = go.Figure(data=[
                go.Bar(
                    x=regime_names,
                    y=regime_values,
                    marker_color=regime_colors[:len(regime_names)],
                    text=regime_values,
                    textposition='auto',
                )
            ])
            
            fig.update_layout(
                title="GEX Regime Distribution from Your Pipeline",
                xaxis_title="Market Regime",
                yaxis_title="Number of Symbols",
                template="plotly_dark",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        else:
            st.info("📊 No regime data available from latest pipeline run")
    
    with col2:
        st.markdown("### 🎯 Pipeline Intelligence")
        
        # Market summary cards
        regime = market_summary['dominant_regime']
        regime_colors = {
            'NEGATIVE_GEX': '#ff4444',
            'HIGH_POSITIVE_GEX': '#00ff88',
            'NEAR_FLIP': '#ffaa00',
            'NEUTRAL': '#888888'
        }
        
        regime_color = regime_colors.get(regime, '#ffffff')
        
        st.markdown(f"""
        <div class="perf-metric">
            <div style="color: {regime_color}; font-weight: bold; margin-bottom: 0.5rem;">
                🎯 {regime.replace('_', ' ').title()}
            </div>
            <div style="font-size: 0.9rem; opacity: 0.8;">
                Dominant market regime from your analysis
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Additional metrics
        historical = data.get('historical_metrics', {})
        pipeline_runs = historical.get('total_pipeline_runs', 0)
        success_rate = historical.get('pipeline_success_rate', 0)
        
        metrics_data = [
            ("🔢 Pipeline Runs", f"{pipeline_runs}"),
            ("✅ Success Rate", f"{success_rate:.1f}%"),
            ("📊 Symbols Analyzed", f"{data['symbols_analyzed']}"),
            ("⚡ Conditions Found", f"{market_summary['total_conditions_found']}"),
            ("🎯 Near Flip Points", f"{market_summary['symbols_near_flip']}")
        ]
        
        for label, value in metrics_data:
            st.markdown(f"""
            <div style="display: flex; justify-content: space-between; padding: 0.5rem 0; 
                        border-bottom: 1px solid rgba(255,255,255,0.1);">
                <span style="opacity: 0.8;">{label}</span>
                <span style="font-weight: bold; color: #00aaff;">{value}</span>
            </div>
            """, unsafe_allow_html=True)

def render_performance_tracking(data):
    """Show pipeline performance over time"""
    
    historical = data.get('historical_metrics', {})
    
    if not historical:
        return
    
    st.markdown("## 📈 Pipeline Performance Tracking")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_setups = historical.get('avg_setups_per_run', 0)
        st.metric(
            label="Avg Setups per Run",
            value=f"{avg_setups:.1f}",
            delta=None
        )
    
    with col2:
        success_rate = historical.get('pipeline_success_rate', 0)
        st.metric(
            label="Pipeline Success Rate",
            value=f"{success_rate:.1f}%",
            delta=None
        )
    
    with col3:
        avg_confidence = historical.get('avg_confidence_score', 0)
        st.metric(
            label="Avg Confidence Score",
            value=f"{avg_confidence:.0f}%",
            delta=None
        )
    
    # Trend analysis
    recent_perf = historical.get('recent_performance', {})
    trend = recent_perf.get('trend', 'STABLE')
    
    trend_colors = {'IMPROVING': '#00ff88', 'STABLE': '#00aaff', 'DECLINING': '#ffaa00'}
    trend_color = trend_colors.get(trend, '#ffffff')
    
    st.markdown(f"""
    <div class="perf-metric">
        <div style="color: {trend_color}; font-weight: bold;">
            📊 Recent Trend: {trend}
        </div>
        <div style="font-size: 0.9rem; opacity: 0.8; margin-top: 0.3rem;">
            Based on last 5 pipeline runs vs historical average
        </div>
    </div>
    """, unsafe_allow_html=True)

def main():
    """Main dashboard application"""
    
    # Hero section
    render_hero_section()
    
    # Control panel
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col2:
        if st.button("🔄 Refresh Pipeline Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    with col3:
        auto_refresh = st.toggle("🔄 Auto Refresh", value=False)
        if auto_refresh:
            st.info("🔄 Auto-refreshing every 3 minutes")
    
    # Initialize connection
    if not init_databricks_connection():
        st.stop()
    
    # Load live data
    with st.spinner('🚀 Loading live data from your Databricks pipeline...'):
        data, source = load_live_databricks_data()
    
    if not data or not data.get('success'):
        st.error("❌ Failed to load pipeline data. Check your Databricks connection.")
        st.stop()
    
    # Show connection status
    data_quality = data.get('raw_pipeline_data', {}).get('data_quality', 'UNKNOWN')
    render_connection_status(source, data['analysis_time'], data_quality)
    
    # Main metrics
    render_enhanced_metrics(data)
    
    st.markdown("---")
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["🎯 Live Trading Setups", "📊 Market Overview", "📈 Performance"])
    
    with tab1:
        render_pipeline_setups(data)
    
    with tab2:
        render_market_overview_chart(data)
    
    with tab3:
        render_performance_tracking(data)
    
    # Footer with pipeline info
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        pipeline_time = data['analysis_time'].strftime('%Y-%m-%d %H:%M:%S')
        st.info(f"🗄️ **Pipeline Data**: {pipeline_time}")
    
    with col2:
        age_minutes = (datetime.now() - data['analysis_time']).total_seconds() / 60
        freshness = "🟢 Fresh" if age_minutes < 10 else "🟡 Recent" if age_minutes < 60 else "🔴 Stale"
        st.info(f"⏱️ **Data Age**: {age_minutes:.0f} minutes ({freshness})")
    
    # Auto refresh logic
    if auto_refresh:
        time.sleep(3)  # 3 minute refresh
        st.rerun()

if __name__ == "__main__":
    main()
