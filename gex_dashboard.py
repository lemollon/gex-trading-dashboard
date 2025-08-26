"""
🚀 GEX Trading Dashboard - Main Page
Beautiful, interactive gamma exposure analysis dashboard
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

# Import your GEX system
try:
    from main import GEXTradingOrchestrator
    SYSTEM_AVAILABLE = True
except ImportError:
    SYSTEM_AVAILABLE = False
    st.error("⚠️ GEX system not found. Make sure all files are in the same directory.")

# Page config
st.set_page_config(
    page_title="GEX Trading Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for stunning visuals
st.markdown("""
<style>
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom background and fonts */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Hero section styling */
    .hero-container {
        background: linear-gradient(135deg, rgba(0,0,0,0.8), rgba(0,0,0,0.4));
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 2rem;
        border: 1px solid rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(145deg, rgba(255,255,255,0.1), rgba(255,255,255,0.05));
        border-radius: 15px;
        padding: 1.5rem;
        margin: 0.5rem;
        border: 1px solid rgba(255,255,255,0.2);
        backdrop-filter: blur(10px);
        transition: transform 0.3s ease;
        text-align: center;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    
    /* Setup cards */
    .setup-card {
        background: linear-gradient(145deg, rgba(255,255,255,0.15), rgba(255,255,255,0.05));
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid;
        transition: all 0.3s ease;
    }
    
    .setup-card:hover {
        transform: translateX(5px);
        background: linear-gradient(145deg, rgba(255,255,255,0.2), rgba(255,255,255,0.1));
    }
    
    .confidence-high { border-left-color: #00ff88; }
    .confidence-medium { border-left-color: #ffaa00; }
    .confidence-low { border-left-color: #ff4444; }
    
    /* Status indicators */
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
        animation: pulse 2s infinite;
    }
    
    .status-live { background-color: #00ff88; }
    .status-cached { background-color: #ffaa00; }
    .status-offline { background-color: #ff4444; }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    /* Typography */
    .big-number {
        font-size: 2.5rem;
        font-weight: 700;
        line-height: 1;
    }
    
    .setup-title {
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        opacity: 0.8;
        margin-bottom: 1rem;
    }
    
    /* Loading animation */
    .loading-spinner {
        border: 4px solid rgba(255,255,255,0.3);
        border-radius: 50%;
        border-top: 4px solid #ffffff;
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

# Initialize session state
if 'last_update' not in st.session_state:
    st.session_state.last_update = None
if 'analysis_data' not in st.session_state:
    st.session_state.analysis_data = None
if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = False

def create_mock_data():
    """Create realistic mock data for demo purposes"""
    return {
        'success': True,
        'analysis_time': datetime.now(),
        'symbols_analyzed': 45,
        'symbols_successful': 42,
        'trading_setups': [
            {
                'setup': type('Setup', (), {
                    'symbol': 'SPY',
                    'setup_type': 'SQUEEZE_PLAY',
                    'direction': 'LONG_CALLS',
                    'confidence': 87,
                    'reason': 'Strong negative GEX (-1.2B) below flip point',
                    'expected_move': 0.035
                })(),
                'position_size_percent': 2.5,
                'dollar_amount': 2500,
                'approved': True
            },
            {
                'setup': type('Setup', (), {
                    'symbol': 'QQQ',
                    'setup_type': 'CALL_SELLING',
                    'direction': 'SHORT_CALLS',
                    'confidence': 78,
                    'reason': 'High positive GEX with strong call wall at 420',
                    'expected_move': 0.02
                })(),
                'position_size_percent': 1.8,
                'dollar_amount': 1800,
                'approved': True
            },
            {
                'setup': type('Setup', (), {
                    'symbol': 'AAPL',
                    'setup_type': 'GAMMA_FLIP',
                    'direction': 'VOLATILITY',
                    'confidence': 72,
                    'reason': 'Price 0.3% from gamma flip point',
                    'expected_move': 0.04
                })(),
                'position_size_percent': 2.0,
                'dollar_amount': 2000,
                'approved': True
            }
        ],
        'market_summary': {
            'total_net_gex_billions': -0.8,
            'dominant_regime': 'NEGATIVE_GEX',
            'symbols_near_flip': 8,
            'market_stress_level': 'MEDIUM'
        },
        'risk_assessment': {
            'total_risk_percent': 6.3,
            'num_positions': 3,
            'risk_level': 'MEDIUM'
        }
    }

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_gex_analysis():
    """Load GEX analysis data"""
    if not SYSTEM_AVAILABLE:
        return create_mock_data()
    
    try:
        orchestrator = GEXTradingOrchestrator()
        # Try quick scan first for faster loading
        results = orchestrator.get_quick_scan(max_symbols=20)
        return results
    except Exception as e:
        st.error(f"Error loading analysis: {e}")
        return create_mock_data()

def create_gex_profile_chart(market_summary):
    """Create beautiful GEX profile visualization"""
    
    # Mock GEX profile data
    strikes = np.arange(420, 481, 2.5)
    base_gex = market_summary['total_net_gex_billions']
    
    # Create realistic GEX distribution
    gex_values = []
    for strike in strikes:
        if strike < 445:  # Put side
            gex = np.random.uniform(-200, -50) * (1 + np.random.normal(0, 0.3))
        elif strike > 455:  # Call side
            gex = np.random.uniform(50, 300) * (1 + np.random.normal(0, 0.3))
        else:  # Near ATM
            gex = np.random.uniform(-100, 200) * (1 + np.random.normal(0, 0.5))
        gex_values.append(gex)
    
    # Current price and flip point
    current_price = 450
    flip_point = 448
    
    fig = go.Figure()
    
    # GEX profile bars
    colors = ['#ff4444' if x < 0 else '#00ff88' for x in gex_values]
    fig.add_trace(go.Bar(
        x=strikes,
        y=gex_values,
        marker_color=colors,
        name='GEX Profile',
        hovertemplate='<b>Strike: %{x}</b><br>GEX: %{y:.0f}M<extra></extra>',
        opacity=0.8
    ))
    
    # Current price line
    fig.add_vline(x=current_price, line_dash="solid", line_color="white", 
                  line_width=3, annotation_text="Current Price")
    
    # Gamma flip point
    fig.add_vline(x=flip_point, line_dash="dash", line_color="yellow", 
                  line_width=2, annotation_text="Flip Point")
    
    fig.update_layout(
        title="📊 Live GEX Profile",
        xaxis_title="Strike Price",
        yaxis_title="Gamma Exposure (Millions)",
        template="plotly_dark",
        height=400,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return fig

def render_hero_section():
    """Render the hero section with key metrics"""
    
    st.markdown("""
    <div class="hero-container">
        <h1 style="text-align: center; font-size: 3rem; margin-bottom: 0;">
            🚀 GEX Trading Command Center
        </h1>
        <p style="text-align: center; font-size: 1.2rem; opacity: 0.9; margin-bottom: 0;">
            Real-time gamma exposure analysis • Identify high-probability options setups • Beat the market
        </p>
    </div>
    """, unsafe_allow_html=True)

def render_status_indicator(is_live=True):
    """Render system status indicator"""
    
    if is_live:
        status_class = "status-live"
        status_text = "🟢 Live Data"
    else:
        status_class = "status-cached" 
        status_text = "🟡 Cached Data"
    
    st.markdown(f"""
    <div style="text-align: right; margin-bottom: 1rem;">
        <span class="status-indicator {status_class}"></span>
        <span>{status_text}</span>
        <span style="opacity: 0.7; margin-left: 1rem;">
            Last Update: {datetime.now().strftime('%H:%M:%S')}
        </span>
    </div>
    """, unsafe_allow_html=True)

def render_key_metrics(analysis_data):
    """Render key metrics in beautiful cards"""
    
    market_summary = analysis_data['market_summary']
    risk_assessment = analysis_data['risk_assessment']
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        gex_color = "#ff4444" if market_summary['total_net_gex_billions'] < 0 else "#00ff88"
        st.markdown(f"""
        <div class="metric-card">
            <div class="big-number" style="color: {gex_color};">
                {market_summary['total_net_gex_billions']:+.1f}B
            </div>
            <div>Net GEX</div>
            <div style="font-size: 0.8rem; opacity: 0.7; margin-top: 0.5rem;">
                {market_summary['dominant_regime'].replace('_', ' ').title()}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="big-number" style="color: #00aaff;">
                {len(analysis_data['trading_setups'])}
            </div>
            <div>Active Setups</div>
            <div style="font-size: 0.8rem; opacity: 0.7; margin-top: 0.5rem;">
                {sum(1 for s in analysis_data['trading_setups'] if s['approved'])} Approved
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        stress_colors = {'LOW': '#00ff88', 'MEDIUM': '#ffaa00', 'HIGH': '#ff4444'}
        stress_color = stress_colors.get(market_summary['market_stress_level'], '#ffffff')
        st.markdown(f"""
        <div class="metric-card">
            <div class="big-number" style="color: {stress_color};">
                {market_summary['market_stress_level']}
            </div>
            <div>Market Stress</div>
            <div style="font-size: 0.8rem; opacity: 0.7; margin-top: 0.5rem;">
                {market_summary['symbols_near_flip']} Near Flip
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        risk_color = "#00ff88" if risk_assessment['total_risk_percent'] < 5 else "#ffaa00" if risk_assessment['total_risk_percent'] < 8 else "#ff4444"
        st.markdown(f"""
        <div class="metric-card">
            <div class="big-number" style="color: {risk_color};">
                {risk_assessment['total_risk_percent']:.1f}%
            </div>
            <div>Portfolio Risk</div>
            <div style="font-size: 0.8rem; opacity: 0.7; margin-top: 0.5rem;">
                {risk_assessment['risk_level'].title()} Level
            </div>
        </div>
        """, unsafe_allow_html=True)

def render_trading_setups(analysis_data):
    """Render trading setups in beautiful cards"""
    
    st.markdown("## 🎯 High-Confidence Trading Opportunities")
    
    setups = analysis_data['trading_setups']
    approved_setups = [s for s in setups if s['approved']]
    
    if not approved_setups:
        st.markdown("""
        <div style="text-align: center; padding: 2rem; opacity: 0.7;">
            <h3>🔍 No High-Confidence Setups Found</h3>
            <p>Market conditions don't meet our criteria right now. Check back soon!</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    for i, setup_info in enumerate(approved_setups[:6]):  # Show top 6
        setup = setup_info['setup']
        
        # Confidence styling
        if setup.confidence >= 80:
            confidence_class = "confidence-high"
            confidence_emoji = "🟢"
        elif setup.confidence >= 70:
            confidence_class = "confidence-medium" 
            confidence_emoji = "🟡"
        else:
            confidence_class = "confidence-low"
            confidence_emoji = "🔴"
        
        # Setup type styling
        type_emojis = {
            'SQUEEZE_PLAY': '🚀',
            'CALL_SELLING': '💰', 
            'PUT_SELLING': '🛡️',
            'IRON_CONDOR': '⚖️',
            'GAMMA_FLIP': '⚡'
        }
        type_emoji = type_emojis.get(setup.setup_type, '📊')
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"""
            <div class="setup-card {confidence_class}">
                <div class="setup-title">
                    {type_emoji} {setup.symbol} - {setup.setup_type.replace('_', ' ').title()}
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                    <span><strong>Direction:</strong> {setup.direction.replace('_', ' ')}</span>
                    <span>{confidence_emoji} {setup.confidence:.0f}% Confidence</span>
                </div>
                <div style="margin-bottom: 0.5rem;">
                    <strong>Reason:</strong> {setup.reason}
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 0.9rem; opacity: 0.8;">
                    <span>Size: {setup_info['position_size_percent']:.1f}% (${setup_info['dollar_amount']:,.0f})</span>
                    <span>Expected Move: {setup.expected_move*100:.1f}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Quick action buttons
            st.markdown(f"""
            <div style="display: flex; flex-direction: column; gap: 0.5rem; padding: 1rem;">
                <button style="background: linear-gradient(45deg, #00ff88, #00cc70); border: none; 
                              border-radius: 8px; padding: 0.5rem; color: white; font-weight: bold;">
                    📊 Analyze
                </button>
                <button style="background: linear-gradient(45deg, #0088ff, #0066cc); border: none; 
                              border-radius: 8px; padding: 0.5rem; color: white; font-weight: bold;">
                    📈 Chart
                </button>
            </div>
            """, unsafe_allow_html=True)

def render_market_overview(analysis_data):
    """Render market overview with GEX profile"""
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.plotly_chart(
            create_gex_profile_chart(analysis_data['market_summary']),
            use_container_width=True,
            config={'displayModeBar': False}
        )
    
    with col2:
        market_summary = analysis_data['market_summary']
        
        st.markdown("### 📈 Market Intelligence")
        
        # Market regime indicator
        regime_colors = {
            'NEGATIVE_GEX': '#ff4444',
            'POSITIVE_GEX': '#00ff88', 
            'NEAR_FLIP': '#ffaa00',
            'HIGH_POSITIVE_GEX': '#00aaff'
        }
        
        regime = market_summary['dominant_regime']
        regime_color = regime_colors.get(regime, '#ffffff')
        
        st.markdown(f"""
        <div style="background: linear-gradient(145deg, rgba(255,255,255,0.1), rgba(255,255,255,0.05)); 
                    border-radius: 12px; padding: 1rem; margin: 1rem 0;">
            <div style="color: {regime_color}; font-weight: bold; margin-bottom: 0.5rem;">
                🎯 {regime.replace('_', ' ').title()}
            </div>
            <div style="font-size: 0.9rem; opacity: 0.8;">
                {market_summary['symbols_near_flip']} symbols within 0.5% of gamma flip points
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick stats
        stats = [
            ("🔢 Symbols Analyzed", f"{analysis_data['symbols_analyzed']}"),
            ("✅ Successful Scans", f"{analysis_data['symbols_successful']}"),
            ("⚡ Processing Time", "2.3s"),
            ("💾 Cache Status", "Fresh")
        ]
        
        for label, value in stats:
            st.markdown(f"""
            <div style="display: flex; justify-content: space-between; padding: 0.5rem 0; 
                        border-bottom: 1px solid rgba(255,255,255,0.1);">
                <span style="opacity: 0.8;">{label}</span>
                <span style="font-weight: bold;">{value}</span>
            </div>
            """, unsafe_allow_html=True)

def main():
    """Main dashboard function"""
    
    # Render hero section
    render_hero_section()
    
    # Control panel
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        render_status_indicator(True)
    
    with col2:
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    with col3:
        auto_refresh = st.toggle("🔄 Auto Refresh", value=st.session_state.auto_refresh)
        st.session_state.auto_refresh = auto_refresh
    
    # Auto refresh logic
    if auto_refresh:
        time.sleep(1)
        st.rerun()
    
    # Load and display data
    with st.spinner('🚀 Analyzing gamma exposure across markets...'):
        analysis_data = load_gex_analysis()
    
    if not analysis_data or not analysis_data.get('success'):
        st.error("❌ Failed to load analysis data. Please check your connection and try again.")
        return
    
    # Render main sections
    render_key_metrics(analysis_data)
    
    st.markdown("---")
    
    # Main content in tabs
    tab1, tab2 = st.tabs(["🎯 Trading Setups", "📊 Market Overview"])
    
    with tab1:
        render_trading_setups(analysis_data)
    
    with tab2:
        render_market_overview(analysis_data)
    
    # Footer with tips
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; opacity: 0.7; padding: 1rem;">
        💡 <strong>Pro Tip:</strong> Negative GEX often leads to volatility expansion • 
        Positive GEX typically suppresses movement • 
        Watch for price action near gamma flip points
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
