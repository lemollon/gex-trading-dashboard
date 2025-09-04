#!/usr/bin/env python3
"""
Production GEX Dashboard - Databricks Connected
Real-time Gamma Exposure analysis for options trading
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from databricks import sql
from datetime import datetime, date, timedelta
import os
import time
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="🎯 GEX Trading Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .metric-container {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    .signal-high {
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
        border: 1px solid #28a745;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(40, 167, 69, 0.2);
    }
    
    .signal-medium {
        background: linear-gradient(135deg, #fff3cd, #ffeaa7);
        border: 1px solid #ffc107;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(255, 193, 7, 0.2);
    }
    
    .signal-low {
        background: linear-gradient(135deg, #f8d7da, #fab1a0);
        border: 1px solid #dc3545;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(220, 53, 69, 0.2);
    }
    
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-online { background: #28a745; }
    .status-warning { background: #ffc107; }
    .status-offline { background: #dc3545; }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🎯 Gamma Exposure Trading Dashboard</h1>
    <p><strong>Production System</strong> | Real-time options analytics powered by Databricks</p>
    <p>Professional GEX analysis for institutional-grade trading decisions</p>
</div>
""", unsafe_allow_html=True)

# Databricks connection with enhanced error handling
@st.cache_resource
def init_databricks_connection():
    """Initialize production Databricks connection with comprehensive error handling"""
    try:
        # Connection parameters from environment or secrets
        server_hostname = (
            st.secrets.get("DATABRICKS_SERVER_HOSTNAME") or 
            os.getenv("DATABRICKS_SERVER_HOSTNAME")
        )
        http_path = (
            st.secrets.get("DATABRICKS_HTTP_PATH") or 
            os.getenv("DATABRICKS_HTTP_PATH")
        )
        access_token = (
            st.secrets.get("DATABRICKS_ACCESS_TOKEN") or 
            os.getenv("DATABRICKS_ACCESS_TOKEN")
        )
        
        if not all([server_hostname, http_path, access_token]):
            st.error("❌ Missing Databricks credentials")
            st.info("""
            **Setup Required**: Configure the following in `.streamlit/secrets.toml`:
            ```
            DATABRICKS_SERVER_HOSTNAME = "your-workspace.cloud.databricks.com"
            DATABRICKS_HTTP_PATH = "/sql/1.0/warehouses/your-warehouse-id"
            DATABRICKS_ACCESS_TOKEN = "your-access-token"
            ```
            """)
            return None
        
        connection = sql.connect(
            server_hostname=server_hostname,
            http_path=http_path,
            access_token=access_token
        )
        
        # Test connection
        cursor = connection.cursor()
        cursor.execute("SELECT 1")
        cursor.fetchall()
        cursor.close()
        
        return connection
        
    except Exception as e:
        st.error(f"❌ Databricks connection failed: {e}")
        st.info("💡 Verify your Databricks credentials and warehouse status")
        return None

# Enhanced data loading functions with caching and error handling
@st.cache_data(ttl=300, show_spinner=False)  # 5-minute cache
def load_latest_analytics(symbol, connection):
    """Load latest GEX analytics with comprehensive error handling"""
    if connection is None:
        return None
    
    try:
        cursor = connection.cursor()
        
        query = f"""
        SELECT 
            symbol,
            snapshot_date,
            snapshot_time,
            spot_price,
            net_gex,
            gamma_flip_point,
            call_wall_1,
            call_wall_2,
            call_wall_3,
            put_wall_1,
            put_wall_2,
            put_wall_3,
            max_call_gex,
            max_put_gex,
            gex_concentration_ratio,
            volatility_regime,
            distance_to_flip_pct,
            charm,
            vanna
        FROM gex_trading.options_data.gex_analytics
        WHERE symbol = '{symbol}'
        ORDER BY snapshot_time DESC
        LIMIT 1
        """
        
        cursor.execute(query)
        result = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        cursor.close()
        
        if result:
            df = pd.DataFrame(result, columns=columns)
            df['snapshot_time'] = pd.to_datetime(df['snapshot_time'])
            return df
        else:
            return None
            
    except Exception as e:
        st.error(f"❌ Error loading analytics for {symbol}: {e}")
        return None

@st.cache_data(ttl=300, show_spinner=False)
def load_gex_profile(symbol, connection, snapshot_date=None):
    """Load comprehensive GEX profile data"""
    if connection is None:
        return None
    
    if snapshot_date is None:
        snapshot_date = date.today()
    
    try:
        cursor = connection.cursor()
        
        query = f"""
        SELECT 
            symbol,
            strike,
            call_gex,
            put_gex,
            net_gex,
            call_gamma,
            put_gamma,
            net_gamma,
            call_oi,
            put_oi,
            snapshot_date,
            snapshot_time,
            spot_price
        FROM gex_trading.options_data.gex_profile
        WHERE symbol = '{symbol}'
        AND snapshot_date = '{snapshot_date}'
        ORDER BY strike
        """
        
        cursor.execute(query)
        result = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        cursor.close()
        
        if result:
            df = pd.DataFrame(result, columns=columns)
            # Calculate cumulative GEX for analysis
            df['cumulative_gex'] = df['net_gex'].cumsum()
            return df
        else:
            return None
            
    except Exception as e:
        st.error(f"❌ Error loading GEX profile for {symbol}: {e}")
        return None

@st.cache_data(ttl=180, show_spinner=False)  # 3-minute cache for signals
def load_trading_signals(symbol, connection):
    """Load active trading signals"""
    if connection is None:
        return None
    
    try:
        cursor = connection.cursor()
        
        query = f"""
        SELECT 
            signal_id,
            symbol,
            signal_type,
            confidence_score,
            entry_price,
            target_strike,
            expiration_date,
            strategy,
            rationale,
            risk_level,
            max_loss_pct,
            expected_return_pct,
            signal_time,
            signal_date,
            is_active
        FROM gex_trading.options_data.trading_signals
        WHERE symbol = '{symbol}'
        AND signal_date >= CURRENT_DATE() - 1
        AND is_active = true
        ORDER BY confidence_score DESC, signal_time DESC
        """
        
        cursor.execute(query)
        result = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        cursor.close()
        
        if result:
            df = pd.DataFrame(result, columns=columns)
            df['signal_time'] = pd.to_datetime(df['signal_time'])
            df['expiration_date'] = pd.to_datetime(df['expiration_date'])
            return df
        else:
            return None
            
    except Exception as e:
        st.error(f"❌ Error loading signals for {symbol}: {e}")
        return None

@st.cache_data(ttl=600, show_spinner=False)  # 10-minute cache for historical
def load_historical_data(symbol, days_back, connection):
    """Load historical GEX data for trend analysis"""
    if connection is None:
        return None
    
    try:
        cursor = connection.cursor()
        
        cutoff_date = (date.today() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        query = f"""
        SELECT 
            snapshot_date,
            snapshot_time,
            spot_price,
            net_gex,
            gamma_flip_point,
            distance_to_flip_pct,
            volatility_regime,
            max_call_gex,
            max_put_gex,
            gex_concentration_ratio
        FROM gex_trading.options_data.gex_analytics
        WHERE symbol = '{symbol}'
        AND snapshot_date >= '{cutoff_date}'
        ORDER BY snapshot_time ASC
        """
        
        cursor.execute(query)
        result = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        cursor.close()
        
        if result:
            df = pd.DataFrame(result, columns=columns)
            df['snapshot_time'] = pd.to_datetime(df['snapshot_time'])
            df['snapshot_date'] = pd.to_datetime(df['snapshot_date'])
            return df
        else:
            return None
            
    except Exception as e:
        st.error(f"❌ Error loading historical data for {symbol}: {e}")
        return None

def get_system_health(connection):
    """Check system health and data freshness"""
    if connection is None:
        return {"status": "offline", "details": "No Databricks connection"}
    
    try:
        cursor = connection.cursor()
        
        # Check latest data timestamp
        cursor.execute("""
        SELECT 
            symbol,
            MAX(analysis_timestamp) as latest_time,
            COUNT(*) as record_count
        FROM quant_projects.gex_trading.scheduled_pipeline_results
        WHERE pipeline_date >= CURRENT_DATE() - 1
        GROUP BY symbol
        ORDER BY latest_time DESC
        """)
        
        result = cursor.fetchall()
        cursor.close()
        
        if result:
            latest_time = result[0][1]
            hours_old = (datetime.now() - pd.to_datetime(latest_time)).total_seconds() / 3600
            
            if hours_old < 1:
                status = "online"
            elif hours_old < 6:
                status = "warning"
            else:
                status = "stale"
            
            return {
                "status": status,
                "latest_time": latest_time,
                "hours_old": hours_old,
                "symbols_count": len(result)
            }
        else:
            return {"status": "no_data", "details": "No recent data found"}
            
    except Exception as e:
        return {"status": "error", "details": str(e)}

# Sidebar configuration
st.sidebar.header("🎛️ Dashboard Configuration")

# Initialize connection
with st.spinner("🔌 Connecting to Databricks..."):
    connection = init_databricks_connection()

if connection is None:
    st.sidebar.error("❌ Databricks connection required")
    st.stop()

# System health check
system_health = get_system_health(connection)
status_color = {
    "online": "🟢", "warning": "🟡", 
    "stale": "🟠", "offline": "🔴", 
    "error": "🔴", "no_data": "🟡"
}
st.sidebar.markdown(f"**System Status:** {status_color.get(system_health['status'], '⚪')} {system_health['status'].title()}")

# Symbol selection with real-time data check
available_symbols = ["SPY", "QQQ", "IWM", "AAPL", "TSLA", "NVDA", "MSFT", "AMZN", "GOOGL", "META"]
symbol = st.sidebar.selectbox("📊 Select Symbol", available_symbols, index=0)

# Time controls
historical_days = st.sidebar.slider("📅 Historical Days", 7, 90, 30, 7)

# Display options
show_advanced = st.sidebar.checkbox("🧮 Advanced Analytics", value=True)
show_historical = st.sidebar.checkbox("📈 Historical Analysis", value=True)
auto_refresh = st.sidebar.checkbox("🔄 Auto-refresh (5min)", value=False)

# Alert settings
st.sidebar.subheader("🚨 Alert Settings")
flip_alert_threshold = st.sidebar.slider("Flip Distance Alert (%)", 0.1, 2.0, 0.5, 0.1)
gex_alert_threshold = st.sidebar.slider("GEX Change Alert (B)", 0.5, 5.0, 1.0, 0.5)

# Manual refresh
if st.sidebar.button("🔄 Refresh Data Now", type="primary"):
    st.cache_data.clear()
    st.rerun()

# Main data loading with progress indicator
with st.spinner(f"📡 Loading {symbol} data..."):
    analytics_data = load_latest_analytics(symbol, connection)
    gex_profile_data = load_gex_profile(symbol, connection)
    signals_data = load_trading_signals(symbol, connection)
    
    if show_historical:
        historical_data = load_historical_data(symbol, historical_days, connection)
    else:
        historical_data = None

# Validate data availability
if analytics_data is None or len(analytics_data) == 0:
    st.error(f"❌ No analytics data found for {symbol}")
    st.info("""
    **Possible solutions:**
    1. Check if the GEX engine has run today
    2. Verify symbol is supported in your system
    3. Ensure Databricks jobs are scheduled properly
    """)
    st.stop()

# Extract latest analytics
latest = analytics_data.iloc[0]

# Alert system
alerts = []
if abs(latest['distance_to_flip_pct']) < flip_alert_threshold:
    alerts.append(f"🔴 ALERT: Only {latest['distance_to_flip_pct']:.2f}% from gamma flip!")

if alerts:
    for alert in alerts:
        st.warning(alert)

# Main dashboard header with key metrics
st.markdown(f"## 📊 {symbol} Live Analysis")
st.markdown(f"*Last updated: {latest['snapshot_time'].strftime('%Y-%m-%d %H:%M:%S')} ET*")

# Key metrics in columns
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    price_change = "normal"  # Could calculate from historical data
    st.metric(
        "💰 Current Price",
        f"${latest['spot_price']:.2f}",
        delta=None
    )

with col2:
    net_gex_b = latest['net_gex'] / 1e9
    delta_color = "inverse" if net_gex_b < -1 else "normal"
    st.metric(
        "🌊 Net GEX",
        f"{net_gex_b:.2f}B",
        delta=None
    )

with col3:
    flip_delta = latest['spot_price'] - latest['gamma_flip_point']
    st.metric(
        "⚡ Gamma Flip",
        f"${latest['gamma_flip_point']:.2f}",
        delta=f"{flip_delta:+.2f}"
    )

with col4:
    distance_pct = latest['distance_to_flip_pct']
    delta_color = "inverse" if abs(distance_pct) < 0.5 else "normal"
    st.metric(
        "📏 Distance to Flip",
        f"{distance_pct:.2f}%",
        delta=None
    )

with col5:
    regime_map = {
        "POSITIVE_GEX_SUPPRESSION": ("🟢", "Suppression"),
        "NEGATIVE_GEX_AMPLIFICATION": ("🔴", "Amplification"), 
        "NEUTRAL_GEX": ("🟡", "Neutral")
    }
    regime_emoji, regime_text = regime_map.get(latest['volatility_regime'], ("⚪", "Unknown"))
    st.metric(
        "📈 Regime",
        f"{regime_emoji} {regime_text}",
        delta=None
    )

# Main visualization section
chart_tab1, chart_tab2, chart_tab3 = st.tabs(["📊 GEX Profile", "🎯 Trading Signals", "📈 Historical"])

with chart_tab1:
    if gex_profile_data is not None and len(gex_profile_data) > 0:
        # Enhanced GEX profile chart
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.7, 0.3],
            subplot_titles=("Gamma Exposure by Strike", "Cumulative GEX"),
            vertical_spacing=0.1
        )
        
        # Main GEX chart
        call_data = gex_profile_data[gex_profile_data['call_gex'] > 0]
        put_data = gex_profile_data[gex_profile_data['put_gex'] < 0]
        
        if len(call_data) > 0:
            fig.add_trace(
                go.Bar(
                    x=call_data['strike'],
                    y=call_data['call_gex'] / 1e6,
                    name='Call GEX',
                    marker_color='rgba(34, 139, 34, 0.8)',
                    hovertemplate='<b>Call Wall</b><br>Strike: $%{x}<br>GEX: %{y:.1f}M<br>OI: %{customdata}<extra></extra>',
                    customdata=call_data['call_oi']
                ), row=1, col=1
            )
        
        if len(put_data) > 0:
            fig.add_trace(
                go.Bar(
                    x=put_data['strike'],
                    y=put_data['put_gex'] / 1e6,
                    name='Put GEX',
                    marker_color='rgba(220, 20, 60, 0.8)',
                    hovertemplate='<b>Put Wall</b><br>Strike: $%{x}<br>GEX: %{y:.1f}M<br>OI: %{customdata}<extra></extra>',
                    customdata=put_data['put_oi']
                ), row=1, col=1
            )
        
        # Add key level lines
        fig.add_vline(x=latest['spot_price'], line_dash="solid", line_color="blue", 
                     annotation_text=f"Spot: ${latest['spot_price']:.2f}", row=1, col=1)
        fig.add_vline(x=latest['gamma_flip_point'], line_dash="dash", line_color="orange",
                     annotation_text=f"Flip: ${latest['gamma_flip_point']:.2f}", row=1, col=1)
        
        # Add major walls
        if pd.notna(latest['call_wall_1']):
            fig.add_vline(x=latest['call_wall_1'], line_dash="dot", line_color="green",
                         annotation_text=f"Call Wall: ${latest['call_wall_1']:.2f}", row=1, col=1)
        
        if pd.notna(latest['put_wall_1']):
            fig.add_vline(x=latest['put_wall_1'], line_dash="dot", line_color="red",
                         annotation_text=f"Put Wall: ${latest['put_wall_1']:.2f}", row=1, col=1)
        
        # Cumulative GEX
        fig.add_trace(
            go.Scatter(
                x=gex_profile_data['strike'],
                y=gex_profile_data['cumulative_gex'] / 1e9,
                mode='lines',
                name='Cumulative GEX',
                line=dict(color='purple', width=3),
                hovertemplate='Strike: $%{x}<br>Cumulative: %{y:.2f}B<extra></extra>'
            ), row=2, col=1
        )
        
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
        
        fig.update_layout(
            height=700,
            showlegend=True,
            title=f"{symbol} Comprehensive GEX Analysis",
            hovermode='x unified'
        )
        
        fig.update_xaxes(title_text="Strike Price ($)", row=2, col=1)
        fig.update_yaxes(title_text="GEX (Millions)", row=1, col=1)
        fig.update_yaxes(title_text="Cumulative GEX (Billions)", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Key levels summary
        levels_col1, levels_col2 = st.columns(2)
        
        with levels_col1:
            st.markdown("**🟢 Call Walls (Resistance)**")
            for i, wall in enumerate(['call_wall_1', 'call_wall_2', 'call_wall_3'], 1):
                if pd.notna(latest[wall]):
                    distance = ((latest[wall] - latest['spot_price']) / latest['spot_price']) * 100
                    st.markdown(f"**{i}.** ${latest[wall]:.2f} (+{distance:.1f}%)")
        
        with levels_col2:
            st.markdown("**🔴 Put Walls (Support)**")
            for i, wall in enumerate(['put_wall_1', 'put_wall_2', 'put_wall_3'], 1):
                if pd.notna(latest[wall]):
                    distance = ((latest[wall] - latest['spot_price']) / latest['spot_price']) * 100
                    st.markdown(f"**{i}.** ${latest[wall]:.2f} ({distance:.1f}%)")
    
    else:
        st.warning(f"⚠️ No GEX profile data available for {symbol}")

with chart_tab2:
    st.subheader("🎯 Live Trading Signals")
    
    if signals_data is not None and len(signals_data) > 0:
        for _, signal in signals_data.iterrows():
            confidence = signal['confidence_score']
            
            # Determine signal styling
            if confidence >= 75:
                signal_class = "signal-high"
                confidence_emoji = "🟢"
                confidence_text = "HIGH"
            elif confidence >= 50:
                signal_class = "signal-medium"
                confidence_emoji = "🟡"
                confidence_text = "MEDIUM"
            else:
                signal_class = "signal-low"
                confidence_emoji = "🔴"
                confidence_text = "LOW"
            
            # Calculate time since signal
            time_diff = datetime.now() - signal['signal_time']
            hours_old = time_diff.total_seconds() / 3600
            
            if hours_old < 1:
                age_text = f"{int(time_diff.total_seconds() / 60)} minutes ago"
            else:
                age_text = f"{int(hours_old)} hours ago"
            
            # Risk/reward ratio
            rr_ratio = signal['expected_return_pct'] / max(signal['max_loss_pct'], 1)
            
            st.markdown(f"""
            <div class="{signal_class}">
                <h4>{confidence_emoji} {signal['strategy']} - {confidence:.0f}% Confidence ({confidence_text})</h4>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1rem;">
                    <div>
                        <p><strong>🎯 Action:</strong> {signal['signal_type']}</p>
                        <p><strong>💰 Target Strike:</strong> ${signal['target_strike']:.0f}</p>
                        <p><strong>📅 Expiration:</strong> {signal['expiration_date'].strftime('%Y-%m-%d')}</p>
                        <p><strong>⚠️ Risk Level:</strong> {signal['risk_level']}</p>
                    </div>
                    <div>
                        <p><strong>📈 Expected Return:</strong> {signal['expected_return_pct']:.0f}%</p>
                        <p><strong>📉 Max Loss:</strong> {signal['max_loss_pct']:.0f}%</p>
                        <p><strong>⚖️ Risk/Reward:</strong> 1:{rr_ratio:.1f}</p>
                        <p><strong>🕒 Generated:</strong> {age_text}</p>
                    </div>
                </div>
                <p style="margin-top: 1rem;"><strong>💡 Rationale:</strong> {signal['rationale']}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("⚪ No active trading signals for today")
        st.markdown("""
        **Signals are generated when:**
        - Net GEX crosses key thresholds (±1B)
        - Price approaches gamma flip point (±0.5%)
        - Strong wall formations are detected
        - High concentration setups emerge
        """)

with chart_tab3:
    if show_historical and historical_data is not None and len(historical_data) > 0:
        st.subheader(f"📈 Historical Analysis ({historical_days} days)")
        
        # Historical trend charts
        hist_fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=("Net GEX Trend", "Distance to Flip", "Volatility Regime"),
            vertical_spacing=0.1,
            row_heights=[0.4, 0.4, 0.2]
        )
        
        # Net GEX trend
        hist_fig.add_trace(
            go.Scatter(
                x=historical_data['snapshot_date'],
                y=historical_data['net_gex'] / 1e9,
                mode='lines+markers',
                name='Net GEX (B)',
                line=dict(color='blue', width=2),
                hovertemplate='Date: %{x}<br>Net GEX: %{y:.2f}B<extra></extra>'
            ), row=1, col=1
        )
        
        hist_fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
        hist_fig.add_hline(y=1, line_dash="dot", line_color="green", row=1, col=1)
        hist_fig.add_hline(y=-1, line_dash="dot", line_color="red", row=1, col=1)
        
        # Distance to flip
        hist_fig.add_trace(
            go.Scatter(
                x=historical_data['snapshot_date'],
                y=historical_data['distance_to_flip_pct'],
                mode='lines+markers',
                name='Distance to Flip (%)',
                line=dict(color='orange', width=2),
                hovertemplate='Date: %{x}<br>Distance: %{y:.2f}%<extra></extra>'
            ), row=2, col=1
        )
        
        hist_fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
        
        # Regime visualization
        regime_numeric = historical_data['volatility_regime'].map({
            'POSITIVE_GEX_SUPPRESSION': 1,
            'NEGATIVE_GEX_AMPLIFICATION': -1,
            'NEUTRAL_GEX': 0
        })
        
        hist_fig.add_trace(
            go.Scatter(
                x=historical_data['snapshot_date'],
                y=regime_numeric,
                mode='markers',
                name='Regime',
                marker=dict(
                    color=regime_numeric,
                    colorscale=[[0, 'red'], [0.5, 'yellow'], [1, 'green']],
                    size=8
                ),
                hovertemplate='Date: %{x}<br>Regime: %{text}<extra></extra>',
                text=historical_data['volatility_regime']
            ), row=3, col=1
        )
        
        hist_fig.update_layout(
            height=800,
            showlegend=False,
            title=f"{symbol} Historical GEX Analysis"
        )
        
        hist_fig.update_yaxes(title_text="Net GEX (B)", row=1, col=1)
        hist_fig.update_yaxes(title_text="Distance (%)", row=2, col=1)
        hist_fig.update_yaxes(title_text="Regime", tickvals=[-1, 0, 1], 
                             ticktext=['Amplification', 'Neutral', 'Suppression'], row=3, col=1)
        
        st.plotly_chart(hist_fig, use_container_width=True)
        
        # Historical statistics
        hist_col1, hist_col2, hist_col3, hist_col4 = st.columns(4)
        
        with hist_col1:
            avg_gex = historical_data['net_gex'].mean() / 1e9
            st.metric("Avg Net GEX", f"{avg_gex:.2f}B")
        
        with hist_col2:
            avg_distance = historical_data['distance_to_flip_pct'].mean()
            st.metric("Avg Distance to Flip", f"{avg_distance:.2f}%")
        
        with hist_col3:
            volatility = historical_data['spot_price'].std() / historical_data['spot_price'].mean() * 100
            st.metric("Price Volatility", f"{volatility:.1f}%")
        
        with hist_col4:
            regime_counts = historical_data['volatility_regime'].value_counts()
            dominant = regime_counts.index[0].split('_')[0] if len(regime_counts) > 0 else "N/A"
            st.metric("Dominant Regime", dominant)

# Advanced analytics section
if show_advanced:
    st.subheader("🧮 Advanced Analytics")
    
    adv_col1, adv_col2, adv_col3 = st.columns(3)
    
    with adv_col1:
        st.markdown("**🎯 Setup Probabilities**")
        
        # Enhanced probability calculations
        net_gex_b = latest['net_gex'] / 1e9
        distance = latest['distance_to_flip_pct']
        concentration = latest['gex_concentration_ratio']
        
        # Negative GEX squeeze
        if net_gex_b < -1 and distance < -0.5:
            squeeze_prob = min(95, 70 + abs(distance) * 10 + abs(net_gex_b) * 5)
        else:
            squeeze_prob = max(5, 30 - abs(distance) * 5)
        
        # Positive GEX breakdown
        if net_gex_b > 2 and abs(distance) < 0.3:
            breakdown_prob = min(95, 75 + (0.3 / max(0.01, abs(distance))) * 5)
        else:
            breakdown_prob = max(5, 25 - max(0, net_gex_b - 1) * 10)
        
        # Iron condor
        if net_gex_b > 1 and concentration > 0.6:
            condor_prob = min(90, 50 + concentration * 40)
        else:
            condor_prob = max(10, 20 + concentration * 30)
        
        probabilities = {
            "🔴 Negative GEX Squeeze": squeeze_prob,
            "🟡 Positive GEX Breakdown": breakdown_prob,
            "🟢 Iron Condor": condor_prob
        }
        
        for setup, prob in probabilities.items():
            color = "🟢" if prob > 70 else "🟡" if prob > 40 else "🔴"
            st.markdown(f"{setup}: **{prob:.0f}%** {color}")
    
    with adv_col2:
        st.markdown("**⚡ Greeks Analysis**")
        
        if pd.notna(latest['charm']) and pd.notna(latest['vanna']):
            st.markdown(f"**Charm (γ decay):** {latest['charm']:.2e}")
            st.markdown(f"**Vanna (γ vol sens):** {latest['vanna']:.2e}")
        
        # Additional calculated metrics
        if latest['max_put_gex'] > 0:
            gex_skew = latest['max_call_gex'] / latest['max_put_gex']
            st.markdown(f"**GEX Skew:** {gex_skew:.2f}")
        
        st.markdown(f"**Concentration:** {latest['gex_concentration_ratio']:.1%}")
        
        # Market microstructure insights
        if abs(latest['distance_to_flip_pct']) < 0.5:
            st.markdown("🔥 **High gamma environment**")
        elif latest['net_gex'] > 3e9:
            st.markdown("🛡️ **Strong dealer hedging**")
        elif latest['net_gex'] < -1e9:
            st.markdown("⚡ **Volatility amplification zone**")
    
    with adv_col3:
        st.markdown("**🚨 Risk Monitoring**")
        
        risk_alerts = []
        
        # Risk assessment
        if abs(latest['distance_to_flip_pct']) < 0.25:
            risk_alerts.append("🔴 Critical: Very close to flip!")
        
        if latest['gex_concentration_ratio'] > 0.8:
            risk_alerts.append("⚠️ Warning: Extreme concentration")
        
        if abs(latest['net_gex']) > 3e9:
            risk_alerts.append("🟡 Notice: High GEX levels")
        
        # Wall proximity checks
        current_price = latest['spot_price']
        
        for wall_type, wall_col in [("Call", "call_wall_1"), ("Put", "put_wall_1")]:
            if pd.notna(latest[wall_col]):
                wall_distance = abs(current_price - latest[wall_col]) / current_price
                if wall_distance < 0.005:  # Within 0.5%
                    emoji = "🟢" if wall_type == "Call" else "🔴"
                    risk_alerts.append(f"{emoji} At {wall_type.lower()} wall!")
        
        if not risk_alerts:
            risk_alerts.append("✅ No active risk alerts")
        
        for alert in risk_alerts:
            st.markdown(alert)

# Footer with system information and tools
st.markdown("---")

footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.markdown("**🔌 System Status**")
    
    status_info = get_system_health(connection)
    if status_info["status"] == "online":
        st.markdown("🟢 **Databricks:** Connected")
        st.markdown(f"🟢 **Data Age:** {status_info.get('hours_old', 0):.1f} hours")
    else:
        st.markdown("🔴 **Databricks:** Issues detected")
    
    st.markdown(f"🟢 **Dashboard:** Online")

with footer_col2:
    st.markdown("**📊 Data Summary**")
    
    if gex_profile_data is not None:
        st.markdown(f"**GEX Strikes:** {len(gex_profile_data)}")
        total_oi = gex_profile_data['call_oi'].sum() + gex_profile_data['put_oi'].sum()
        st.markdown(f"**Total OI:** {total_oi:,}")
    
    if signals_data is not None:
        st.markdown(f"**Active Signals:** {len(signals_data)}")
    
    st.markdown(f"**Symbol:** {symbol}")

with footer_col3:
    st.markdown("**🛠️ Tools**")
    
    # Data export
    if st.button("📥 Export Data"):
        if gex_profile_data is not None:
            csv = gex_profile_data.to_csv(index=False)
            st.download_button(
                "Download GEX CSV",
                csv,
                f"{symbol}_gex_{date.today()}.csv",
                "text/csv"
            )
    
    # Quick actions
    if st.button("🔄 Force Refresh"):
        st.cache_data.clear()
        st.success("Cache cleared!")
    
    if st.button("📊 System Info"):
        st.info(f"""
        **Version:** 2.0 Production
        **Environment:** {os.getenv('ENVIRONMENT', 'Production')}
        **Last Deploy:** {os.getenv('DEPLOY_DATE', 'Unknown')}
        """)

# Auto-refresh functionality
if auto_refresh:
    # Display countdown
    placeholder = st.empty()
    for seconds in range(300, 0, -1):  # 5-minute countdown
        placeholder.markdown(f"🔄 Auto-refresh in {seconds//60}:{seconds%60:02d}")
        time.sleep(1)
    
    placeholder.empty()
    st.rerun()

# Risk disclaimer
st.markdown("""
---
**⚠️ IMPORTANT DISCLAIMER:** This dashboard is for educational and analytical purposes only. 
Options trading involves substantial risk of loss and is not suitable for all investors. 
The information provided should not be considered as investment advice. 
Always conduct your own research and consider consulting with a qualified financial advisor 
before making any trading decisions. Past performance does not guarantee future results.
""")
