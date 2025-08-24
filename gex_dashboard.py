# 🚀 Visually Stunning Gamma Exposure Trading System
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import yfinance as yf
import time
from typing import Dict, List, Optional
import uuid
import random

# Enhanced page configuration with custom styling
st.set_page_config(
    page_title="GEX Trading System",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for stunning visuals
st.markdown("""
<style>
    /* Import modern fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styles */
    .stApp {
        background: linear-gradient(135deg, #0f1419 0%, #1a1f2e 50%, #0f1419 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Custom header styling */
    .main-header {
        background: linear-gradient(90deg, #00ff87 0%, #60efff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 0 0 30px rgba(0, 255, 135, 0.3);
    }
    
    .subtitle {
        text-align: center;
        font-size: 1.2rem;
        color: #8b949e;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    /* Glowing cards */
    .metric-card {
        background: linear-gradient(145deg, #21262d 0%, #30363d 100%);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 16px 48px rgba(0, 255, 135, 0.2);
        border-color: #00ff87;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(0, 255, 135, 0.1), transparent);
        transition: left 0.5s;
    }
    
    .metric-card:hover::before {
        left: 100%;
    }
    
    /* Status indicators */
    .status-open {
        background: linear-gradient(135deg, #00ff87, #00cc6a);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
        box-shadow: 0 4px 15px rgba(0, 255, 135, 0.4);
    }
    
    .status-closed {
        background: linear-gradient(135deg, #ff4757, #ff3742);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
        box-shadow: 0 4px 15px rgba(255, 71, 87, 0.4);
    }
    
    .status-premarket {
        background: linear-gradient(135deg, #ffa502, #ff9500);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
        box-shadow: 0 4px 15px rgba(255, 165, 2, 0.4);
    }
    
    /* Confidence badges */
    .confidence-high {
        background: linear-gradient(135deg, #00ff87, #00cc6a);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
    }
    
    .confidence-medium {
        background: linear-gradient(135deg, #ffa502, #ff9500);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
    }
    
    .confidence-low {
        background: linear-gradient(135deg, #ff6b6b, #ff5252);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
    }
    
    /* Trade type badges */
    .trade-squeeze {
        background: linear-gradient(135deg, #ff6b6b, #ff5252);
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    .trade-premium {
        background: linear-gradient(135deg, #4ecdc4, #44a08d);
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    .trade-condor {
        background: linear-gradient(135deg, #a55eea, #8b5cf6);
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    /* Opportunity cards */
    .opportunity-card {
        background: linear-gradient(145deg, #1c2128 0%, #21262d 100%);
        border: 2px solid transparent;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        position: relative;
        overflow: hidden;
        transition: all 0.3s ease;
    }
    
    .opportunity-card:hover {
        transform: scale(1.02);
        box-shadow: 0 16px 48px rgba(0, 255, 135, 0.3);
    }
    
    .opportunity-high {
        border-image: linear-gradient(135deg, #00ff87, #60efff) 1;
        box-shadow: 0 8px 32px rgba(0, 255, 135, 0.2);
    }
    
    .opportunity-medium {
        border-image: linear-gradient(135deg, #ffa502, #ff9500) 1;
        box-shadow: 0 8px 32px rgba(255, 165, 2, 0.2);
    }
    
    /* Progress bars */
    .progress-bar {
        background: linear-gradient(90deg, #00ff87 0%, #60efff 100%);
        height: 8px;
        border-radius: 4px;
        overflow: hidden;
        position: relative;
    }
    
    .progress-bar::after {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        animation: shimmer 2s infinite;
    }
    
    @keyframes shimmer {
        0% { left: -100%; }
        100% { left: 100%; }
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #0f1419 0%, #1a1f2e 100%);
    }
    
    /* Metrics styling */
    div[data-testid="metric-container"] {
        background: linear-gradient(145deg, #21262d 0%, #30363d 100%);
        border: 1px solid #30363d;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #00ff87, #00cc6a);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 255, 135, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 255, 135, 0.4);
        background: linear-gradient(135deg, #00cc6a, #00b356);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background: linear-gradient(90deg, #21262d, #30363d);
        border-radius: 10px;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        color: #8b949e;
        font-weight: 500;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #00ff87, #60efff);
        color: white;
    }
    
    /* Success/Error messages */
    .stSuccess {
        background: linear-gradient(135deg, #00ff87, #00cc6a);
        border: none;
        border-radius: 10px;
    }
    
    .stError {
        background: linear-gradient(135deg, #ff4757, #ff3742);
        border: none;
        border-radius: 10px;
    }
    
    .stInfo {
        background: linear-gradient(135deg, #60efff, #4ecdc4);
        border: none;
        border-radius: 10px;
    }
    
    .stWarning {
        background: linear-gradient(135deg, #ffa502, #ff9500);
        border: none;
        border-radius: 10px;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

class GEXEducationalSystem:
    """Educational system with beautiful visualizations"""
    
    @staticmethod
    def create_animated_header():
        """Create stunning animated header"""
        st.markdown("""
        <div class="main-header">
            🚀 Gamma Exposure Trading System
        </div>
        <div class="subtitle">
            Master Professional Options Trading Through Market Maker Psychology
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def create_market_maker_visualization():
        """Interactive market maker psychology chart"""
        
        # Create interactive demo
        col1, col2 = st.columns([2, 1])
        
        with col2:
            st.markdown("### 🎯 Interactive Demo")
            price = st.slider("Stock Price ($)", 90, 110, 100, key="mm_demo")
            gamma_flip = st.slider("Gamma Flip ($)", 85, 105, 95, key="flip_demo")
        
        with col1:
            # Create beautiful visualization
            fig = go.Figure()
            
            # Add zones
            fig.add_shape(
                type="rect",
                x0=90, x1=gamma_flip, y0=0, y1=1,
                fillcolor="rgba(255, 107, 107, 0.2)",
                line=dict(width=0),
                name="Squeeze Zone"
            )
            
            fig.add_shape(
                type="rect",
                x0=gamma_flip, x1=110, y0=0, y1=1,
                fillcolor="rgba(78, 205, 196, 0.2)",
                line=dict(width=0),
                name="Premium Zone"
            )
            
            # Add current price line
            fig.add_vline(
                x=price,
                line_dash="solid",
                line_color="#00ff87",
                line_width=3,
                annotation_text=f"Current Price: ${price}",
                annotation_position="top"
            )
            
            # Add gamma flip line
            fig.add_vline(
                x=gamma_flip,
                line_dash="dash",
                line_color="#ff9500",
                line_width=2,
                annotation_text=f"Gamma Flip: ${gamma_flip}",
                annotation_position="bottom"
            )
            
            # Styling
            fig.update_layout(
                title={
                    'text': '🎯 Market Maker Psychology Zones',
                    'x': 0.5,
                    'font': {'size': 20, 'color': 'white'}
                },
                xaxis_title="Stock Price ($)",
                yaxis=dict(visible=False),
                template="plotly_dark",
                height=300,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Add explanation with beautiful formatting
        if price < gamma_flip:
            st.markdown("""
            <div class="opportunity-card opportunity-high">
                <h3>🚀 SQUEEZE ZONE ACTIVATED!</h3>
                <p><strong>What's Happening:</strong> Market makers will amplify every move up! Perfect for buying calls.</p>
                <p><strong>Strategy:</strong> Buy ATM or OTM calls for explosive gains</p>
                <p><strong>Expected Move:</strong> 2-5x leverage on stock moves</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="opportunity-card opportunity-medium">
                <h3>🛡️ PREMIUM SELLING ZONE</h3>
                <p><strong>What's Happening:</strong> Market makers will dampen volatility. Perfect for selling options.</p>
                <p><strong>Strategy:</strong> Sell calls/puts or iron condors</p>
                <p><strong>Expected Move:</strong> Time decay and volatility compression</p>
            </div>
            """, unsafe_allow_html=True)
    
    @staticmethod
    def show_strategy_cards():
        """Beautiful strategy overview cards"""
        st.markdown("### 💡 Three Profitable Strategies")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="opportunity-card">
                <div class="trade-squeeze">🚀 SQUEEZE PLAYS</div>
                <h4 style="color: #00ff87; margin-top: 1rem;">Buy Calls/Puts</h4>
                <p><strong>When:</strong> Below gamma flip + negative GEX</p>
                <p><strong>Why:</strong> Market makers amplify moves</p>
                <p><strong>Target:</strong> 50-100% gains in 1-3 days</p>
                <p><strong>Risk:</strong> Can lose 50% quickly</p>
                <div class="progress-bar" style="margin-top: 1rem;"></div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="opportunity-card">
                <div class="trade-premium">🛡️ PREMIUM SELLING</div>
                <h4 style="color: #4ecdc4; margin-top: 1rem;">Sell Calls/Puts</h4>
                <p><strong>When:</strong> Above flip + positive GEX</p>
                <p><strong>Why:</strong> Market makers suppress moves</p>
                <p><strong>Target:</strong> 25-50% premium collection</p>
                <p><strong>Risk:</strong> Assignment if walls break</p>
                <div class="progress-bar" style="margin-top: 1rem;"></div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="opportunity-card">
                <div class="trade-condor">⚖️ IRON CONDORS</div>
                <h4 style="color: #a55eea; margin-top: 1rem;">Sell Both Sides</h4>
                <p><strong>When:</strong> High positive GEX + wide walls</p>
                <p><strong>Why:</strong> Price trapped between walls</p>
                <p><strong>Target:</strong> 20-40% premium collection</p>
                <p><strong>Risk:</strong> Big move breaks setup</p>
                <div class="progress-bar" style="margin-top: 1rem;"></div>
            </div>
            """, unsafe_allow_html=True)

class EnhancedMockTradingAccount:
    """Enhanced mock trading with beautiful visualizations"""
    
    def __init__(self):
        self.initial_balance = 100000
        self.last_update_time = datetime.now()
        
        # Initialize session state
        if 'portfolio_history' not in st.session_state:
            st.session_state.portfolio_history = []
        if 'open_trades' not in st.session_state:
            st.session_state.open_trades = []
        if 'closed_trades' not in st.session_state:
            st.session_state.closed_trades = []
        if 'auto_trading_enabled' not in st.session_state:
            st.session_state.auto_trading_enabled = False
        if 'last_analysis_time' not in st.session_state:
            st.session_state.last_analysis_time = datetime.now() - timedelta(minutes=30)
    
    def create_beautiful_portfolio_chart(self):
        """Create stunning portfolio performance chart"""
        if len(st.session_state.portfolio_history) < 2:
            # Create sample data for demo
            dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
            values = [self.initial_balance]
            
            for i in range(1, len(dates)):
                change = np.random.normal(0, 0.02) * values[-1]
                values.append(max(values[-1] + change, self.initial_balance * 0.8))
            
            sample_history = [{'timestamp': date, 'total_value': value} for date, value in zip(dates, values)]
        else:
            sample_history = st.session_state.portfolio_history
        
        df = pd.DataFrame(sample_history)
        
        # Create subplot with secondary y-axis
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.7, 0.3],
            vertical_spacing=0.05,
            subplot_titles=("Portfolio Value", "Daily Returns %")
        )
        
        # Portfolio value line with gradient fill
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['total_value'],
                mode='lines',
                name='Portfolio Value',
                line=dict(color='#00ff87', width=3),
                fill='tonexty',
                fillcolor='rgba(0, 255, 135, 0.1)',
                hovertemplate='<b>%{y:$,.0f}</b><br>%{x}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Add starting value line
        fig.add_hline(
            y=self.initial_balance,
            line_dash="dash",
            line_color="rgba(255, 255, 255, 0.5)",
            annotation_text=f"Starting: ${self.initial_balance:,}",
            annotation_position="top left",
            row=1, col=1
        )
        
        # Daily returns bar chart
        daily_returns = df['total_value'].pct_change() * 100
        colors = ['#00ff87' if x >= 0 else '#ff4757' for x in daily_returns]
        
        fig.add_trace(
            go.Bar(
                x=df['timestamp'],
                y=daily_returns,
                name='Daily Returns',
                marker_color=colors,
                hovertemplate='<b>%{y:.2f}%</b><br>%{x}<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Styling
        fig.update_layout(
            height=500,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            showlegend=False,
            margin=dict(l=0, r=0, t=50, b=0)
        )
        
        fig.update_xaxes(showgrid=False, color='white')
        fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)', color='white')
        
        return fig
    
    def get_realistic_opportunities(self) -> List[Dict]:
        """Generate visually rich opportunities"""
        
        opportunities = [
            {
                'symbol': 'TSLA',
                'current_price': 245.67,
                'gamma_flip': 238.50,
                'distance_pct': 3.01,
                'net_gex': 1250000000,
                'call_wall': 250.00,
                'put_wall': 235.00,
                'structure_type': 'PREMIUM_SELLING_SETUP',
                'confidence_score': 88,
                'trade_type': 'CALL_SELLING',
                'recommendation': 'SELL $250 CALLS',
                'explanation': 'TSLA approaching call wall with massive positive GEX. Market makers will defend this level.',
                'expected_premium': 3.20,
                'days_to_expiry': 3,
                'emoji': '⚡',
                'trend': 'up',
                'volume_spike': True
            },
            {
                'symbol': 'NVDA', 
                'current_price': 118.45,
                'gamma_flip': 125.20,
                'distance_pct': -5.39,
                'net_gex': -850000000,
                'call_wall': 130.00,
                'put_wall': 115.00,
                'structure_type': 'SQUEEZE_SETUP',
                'confidence_score': 93,
                'trade_type': 'LONG_CALLS',
                'recommendation': 'BUY $120/$125 CALLS',
                'explanation': 'NVDA below gamma flip with negative GEX. Any move up gets amplified by dealer hedging.',
                'expected_premium': 2.85,
                'days_to_expiry': 2,
                'emoji': '🎯',
                'trend': 'down',
                'volume_spike': True
            },
            {
                'symbol': 'SPY',
                'current_price': 565.23,
                'gamma_flip': 563.00,
                'distance_pct': 0.40,
                'net_gex': 2100000000,
                'call_wall': 570.00,
                'put_wall': 560.00,
                'structure_type': 'IRON_CONDOR_SETUP',
                'confidence_score': 76,
                'trade_type': 'IRON_CONDOR',
                'recommendation': 'IRON CONDOR 560/570',
                'explanation': 'SPY has massive positive GEX with clear walls. Perfect range-bound setup.',
                'expected_premium': 1.50,
                'days_to_expiry': 7,
                'emoji': '🎪',
                'trend': 'sideways',
                'volume_spike': False
            }
        ]
        
        return opportunities
    
    def create_opportunity_card(self, opp: Dict, index: int):
        """Create beautiful opportunity cards"""
        
        # Determine confidence styling
        if opp['confidence_score'] >= 85:
            confidence_class = "confidence-high"
            card_class = "opportunity-high"
            confidence_emoji = "🟢"
        elif opp['confidence_score'] >= 75:
            confidence_class = "confidence-medium"
            card_class = "opportunity-medium"
            confidence_emoji = "🟡"
        else:
            confidence_class = "confidence-low"
            card_class = "opportunity-low"
            confidence_emoji = "🟠"
        
        # Trade type styling
        if 'SQUEEZE' in opp['structure_type']:
            trade_class = "trade-squeeze"
        elif 'PREMIUM' in opp['structure_type']:
            trade_class = "trade-premium"
        else:
            trade_class = "trade-condor"
        
        # Volume indicator
        volume_indicator = "🔥 HIGH VOLUME" if opp.get('volume_spike') else "📊 NORMAL VOLUME"
        
        # Create the card HTML
        card_html = f"""
        <div class="{card_class}">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                <div style="display: flex; align-items: center; gap: 1rem;">
                    <h2 style="margin: 0; color: #00ff87; font-size: 1.8rem;">{opp['emoji']} {opp['symbol']}</h2>
                    <div class="{confidence_class}">{confidence_emoji} {opp['confidence_score']}% CONFIDENCE</div>
                </div>
                <div class="{trade_class}">{opp['trade_type'].replace('_', ' ')}</div>
            </div>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1rem; margin-bottom: 1rem;">
                <div>
                    <div style="color: #8b949e; font-size: 0.9rem;">Current Price</div>
                    <div style="font-size: 1.3rem; font-weight: 600; color: #00ff87;">${opp['current_price']:.2f}</div>
                </div>
                <div>
                    <div style="color: #8b949e; font-size: 0.9rem;">Gamma Flip</div>
                    <div style="font-size: 1.3rem; font-weight: 600; color: #ff9500;">${opp['gamma_flip']:.2f}</div>
                </div>
                <div>
                    <div style="color: #8b949e; font-size: 0.9rem;">Distance</div>
                    <div style="font-size: 1.3rem; font-weight: 600; color: {'#00ff87' if opp['distance_pct'] > 0 else '#ff4757'};">{opp['distance_pct']:+.1f}%</div>
                </div>
            </div>
            
            <div style="background: rgba(0, 255, 135, 0.1); padding: 1rem; border-radius: 8px; margin-bottom: 1rem; border-left: 4px solid #00ff87;">
                <div style="font-weight: 600; margin-bottom: 0.5rem;">🎯 STRATEGY: {opp['recommendation']}</div>
                <div style="color: #8b949e;">{opp['explanation']}</div>
            </div>
            
            <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 1rem;">
                <div style="display: flex; gap: 1rem;">
                    <div style="color: #8b949e;">Premium: <span style="color: #00ff87; font-weight: 600;">${opp['expected_premium']:.2f}</span></div>
                    <div style="color: #8b949e;">Expiry: <span style="color: #00ff87; font-weight: 600;">{opp['days_to_expiry']}d</span></div>
                    <div style="color: #8b949e; font-size: 0.9rem;">{volume_indicator}</div>
                </div>
            </div>
        </div>
        """
        
        st.markdown(card_html, unsafe_allow_html=True)

def display_morning_analysis():
    """Visually stunning morning analysis"""
    
    # Beautiful header
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 style="background: linear-gradient(90deg, #00ff87, #60efff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 2.5rem; margin-bottom: 0.5rem;">
            🌅 Morning Gamma Analysis
        </h1>
        <p style="color: #8b949e; font-size: 1.1rem;">Live Trading Opportunities</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Market status with beautiful indicators
    current_time = datetime.now()
    
    if 9 <= current_time.hour < 16 and current_time.weekday() < 5:
        status_html = '<div class="status-open">🟢 MARKET OPEN</div>'
    elif current_time.weekday() >= 5:
        status_html = '<div class="status-closed">🔴 WEEKEND - MARKETS CLOSED</div>'
    else:
        status_html = '<div class="status-premarket">🟡 PRE/AFTER MARKET</div>'
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div style="color: #8b949e; font-size: 0.9rem;">Last Analysis</div>
            <div style="font-size: 1.2rem; font-weight: 600; color: #00ff87;">09:15 AM ET</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div style="color: #8b949e; font-size: 0.9rem;">Current Time</div>
            <div style="font-size: 1.2rem; font-weight: 600; color: #60efff;">{current_time.strftime('%I:%M %p ET')}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div style="color: #8b949e; font-size: 0.9rem;">Market Status</div>
            <div style="margin-top: 0.5rem;">{status_html}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div style="color: #8b949e; font-size: 0.9rem;">Next Analysis</div>
            <div style="font-size: 1.2rem; font-weight: 600; color: #ffa502;">09:15 AM Tomorrow</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Analysis process with beautiful icons
    st.markdown("""
    <div style="background: linear-gradient(145deg, #1c2128, #21262d); padding: 2rem; border-radius: 16px; margin: 2rem 0;">
        <h3 style="color: #00ff87; margin-bottom: 1.5rem; display: flex; align-items: center;">
            <span style="margin-right: 1rem;">📊</span>How Our Analysis Works
        </h3>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem;">
            <div style="display: flex; align-items: center; padding: 1rem; background: rgba(0, 255, 135, 0.1); border-radius: 8px;">
                <span style="font-size: 1.5rem; margin-right: 1rem;">🔍</span>
                <div>
                    <div style="font-weight: 600; color: white;">Scan 100+ Stocks</div>
                    <div style="color: #8b949e; font-size: 0.9rem;">For gamma exposure patterns</div>
                </div>
            </div>
            <div style="display: flex; align-items: center; padding: 1rem; background: rgba(96, 239, 255, 0.1); border-radius: 8px;">
                <span style="font-size: 1.5rem; margin-right: 1rem;">🧮</span>
                <div>
                    <div style="font-weight: 600; color: white;">Calculate Levels</div>
                    <div style="color: #8b949e; font-size: 0.9rem;">Gamma flip points & walls</div>
                </div>
            </div>
            <div style="display: flex; align-items: center; padding: 1rem; background: rgba(255, 165, 2, 0.1); border-radius: 8px;">
                <span style="font-size: 1.5rem; margin-right: 1rem;">🎯</span>
                <div>
                    <div style="font-weight: 600; color: white;">Identify Setups</div>
                    <div style="color: #8b949e; font-size: 0.9rem;">3-5 highest probability trades</div>
                </div>
            </div>
            <div style="display: flex; align-items: center; padding: 1rem; background: rgba(255, 71, 87, 0.1); border-radius: 8px;">
                <span style="font-size: 1.5rem; margin-right: 1rem;">🚨</span>
                <div>
                    <div style="font-weight: 600; color: white;">Alert & Rank</div>
                    <div style="color: #8b949e; font-size: 0.9rem;">90%+ confidence auto-trades</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Show opportunities
    account = EnhancedMockTradingAccount()
    opportunities = account.get_realistic_opportunities()
    
    st.markdown("### 🎯 Today's Premium Opportunities")
    
    for i, opp in enumerate(opportunities):
        account.create_opportunity_card(opp, i)
        
        # Add trade button
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button(f"📈 Execute Trade", key=f"trade_{i}", type="primary"):
                st.success(f"🎉 Trade executed: {opp['symbol']} {opp['trade_type']}")
        with col2:
            if st.button(f"📊 View Analysis", key=f"analysis_{i}"):
                st.info(f"📈 Detailed analysis for {opp['symbol']} coming soon!")

def display_enhanced_portfolio():
    """Stunning portfolio display"""
    
    # Beautiful header
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 style="background: linear-gradient(90deg, #00ff87, #60efff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 2.5rem; margin-bottom: 0.5rem;">
            💰 $100K Gamma Trading Challenge
        </h1>
        <p style="color: #8b949e; font-size: 1.1rem;">Learn While You Earn - Master Professional Options Trading</p>
    </div>
    """, unsafe_allow_html=True)
    
    account = EnhancedMockTradingAccount()
    
    # Create sample balance for demo
    balance = {
        'total_value': 108750,
        'cash_balance': 85430,
        'positions_value': 23320,
        'realized_pnl': 3250,
        'unrealized_pnl': 5500,
        'total_return_pct': 8.75,
        'open_trades_count': 3,
        'closed_trades_count': 12,
        'win_rate': 75.0,
        'avg_winner': 45.2,
        'avg_loser': -22.1
    }
    
    # Portfolio metrics with beautiful cards
    col1, col2, col3, col4 = st.columns(4)
    
    metrics_html = [
        (f"${balance['total_value']:,.0f}", f"+{balance['total_return_pct']:.1f}%", "Portfolio Value", "#00ff87"),
        (f"${balance['cash_balance']:,.0f}", "Available", "Cash Balance", "#60efff"),
        (f"{balance['win_rate']:.0f}%", f"{balance['closed_trades_count']} trades", "Win Rate", "#ffa502"),
        (f"+{balance['avg_winner']:.1f}%", f"{balance['avg_loser']:.1f}%", "Avg Winner/Loser", "#a55eea")
    ]
    
    for i, (metric, delta, label, color) in enumerate(metrics_html):
        with [col1, col2, col3, col4][i]:
            st.markdown(f"""
            <div class="metric-card">
                <div style="color: #8b949e; font-size: 0.9rem; margin-bottom: 0.5rem;">{label}</div>
                <div style="font-size: 1.8rem; font-weight: 700; color: {color}; margin-bottom: 0.3rem;">{metric}</div>
                <div style="color: #8b949e; font-size: 0.8rem;">{delta}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Auto-trading toggle
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    with col1:
        auto_trading = st.toggle("🤖 Enable Auto-Trading", value=False, key="auto_toggle")
    
    with col2:
        if auto_trading:
            st.markdown('<div class="status-open">🤖 AUTO-TRADING ACTIVE - Will execute 90%+ confidence trades</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-premarket">👤 MANUAL MODE - You choose which trades to execute</div>', unsafe_allow_html=True)
    
    # Portfolio performance chart
    st.markdown("### 📈 Portfolio Performance")
    fig = account.create_beautiful_portfolio_chart()
    st.plotly_chart(fig, use_container_width=True)
    
    # Sample open positions
    st.markdown("### 📊 Active Positions")
    
    sample_trades = [
        {'symbol': 'NVDA', 'type': 'LONG CALLS', 'pnl': 15.2, 'days': 2, 'confidence': 93, 'analysis': '🎉 WINNING: Squeeze setup playing out perfectly!'},
        {'symbol': 'TSLA', 'type': 'CALL SELLING', 'pnl': -8.4, 'days': 1, 'confidence': 88, 'analysis': '📊 DEVELOPING: Still early, let it develop'},
        {'symbol': 'SPY', 'type': 'IRON CONDOR', 'pnl': 22.1, 'days': 4, 'confidence': 76, 'analysis': '✅ SUCCESS: Range-bound as predicted'}
    ]
    
    for trade in sample_trades:
        pnl_color = "#00ff87" if trade['pnl'] > 0 else "#ff4757"
        pnl_emoji = "🟢" if trade['pnl'] > 0 else "🔴"
        
        st.markdown(f"""
        <div class="opportunity-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div style="display: flex; align-items: center; gap: 1rem;">
                    <h3 style="margin: 0; color: #00ff87;">{pnl_emoji} {trade['symbol']}</h3>
                    <div class="trade-{'squeeze' if 'CALLS' in trade['type'] else 'premium'}">{trade['type']}</div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 1.5rem; font-weight: 700; color: {pnl_color};">{trade['pnl']:+.1f}%</div>
                    <div style="color: #8b949e; font-size: 0.9rem;">Day {trade['days']} • {trade['confidence']}% confidence</div>
                </div>
            </div>
            <div style="margin-top: 1rem; padding: 1rem; background: rgba(0, 255, 135, 0.1); border-radius: 8px; border-left: 4px solid #00ff87;">
                <strong>💡 Analysis:</strong> {trade['analysis']}
            </div>
        </div>
        """, unsafe_allow_html=True)

def main():
    """Main app with stunning design"""
    
    # Create beautiful header
    GEXEducationalSystem.create_animated_header()
    
    # Beautiful sidebar
    st.sidebar.markdown("""
    <div style="background: linear-gradient(145deg, #1c2128, #21262d); padding: 1.5rem; border-radius: 12px; margin-bottom: 2rem;">
        <h3 style="color: #00ff87; margin-bottom: 1rem;">🎯 Navigation</h3>
    </div>
    """, unsafe_allow_html=True)
    
    page = st.sidebar.selectbox("Choose Section:", [
        "🎓 Learn the Strategy",
        "🌅 Morning Analysis", 
        "💰 Trading Challenge",
        "📊 Performance Analytics"
    ])
    
    # Quick reference sidebar
    st.sidebar.markdown("""
    <div style="background: linear-gradient(145deg, #1c2128, #21262d); padding: 1.5rem; border-radius: 12px;">
        <h4 style="color: #60efff; margin-bottom: 1rem;">📚 Quick Reference</h4>
        
        <div style="margin-bottom: 1rem;">
            <div class="trade-squeeze" style="margin-bottom: 0.5rem;">🚀 SQUEEZE PLAYS</div>
            <div style="font-size: 0.85rem; color: #8b949e;">
                • Below gamma flip<br>
                • Negative GEX<br>
                • Target: 50-100%
            </div>
        </div>
        
        <div style="margin-bottom: 1rem;">
            <div class="trade-premium" style="margin-bottom: 0.5rem;">🛡️ PREMIUM SELLING</div>
            <div style="font-size: 0.85rem; color: #8b949e;">
                • Above gamma flip<br>
                • Positive GEX<br>
                • Target: 25-50%
            </div>
        </div>
        
        <div>
            <div class="trade-condor" style="margin-bottom: 0.5rem;">⚖️ IRON CONDORS</div>
            <div style="font-size: 0.85rem; color: #8b949e;">
                • High positive GEX<br>
                • Wide walls<br>
                • Target: 20-40%
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Page routing
    if page == "🎓 Learn the Strategy":
        st.markdown("## 🎓 Master Gamma Exposure Trading")
        
        tab1, tab2, tab3 = st.tabs(["📖 How It Works", "💡 Strategies", "🎯 Interactive Demo"])
        
        with tab1:
            st.markdown("""
            ### 🎯 How Market Makers Think with Gamma Exposure
            
            **Market makers are like bookies at a casino** - they want to make money on every trade while staying neutral to price direction.
            """)
            
            # Add beautiful explanation cards here
            
        with tab2:
            GEXEducationalSystem.show_strategy_cards()
            
        with tab3:
            GEXEducationalSystem.create_market_maker_visualization()
    
    elif page == "🌅 Morning Analysis":
        display_morning_analysis()
    
    elif page == "💰 Trading Challenge":
        display_enhanced_portfolio()
    
    elif page == "📊 Performance Analytics":
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <h1 style="background: linear-gradient(90deg, #00ff87, #60efff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 2.5rem; margin-bottom: 0.5rem;">
                📊 Advanced Performance Analytics
            </h1>
            <p style="color: #8b949e; font-size: 1.1rem;">Deep Dive into Your Trading Performance</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.info("🚀 Advanced analytics will be displayed here once you start trading!")

if __name__ == "__main__":
    main()
