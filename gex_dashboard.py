# GEX Strategy Dashboard - Fixed Version
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
from typing import Dict, List, Optional

# Page configuration
st.set_page_config(
    page_title="GEX Trading Strategy Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Simplified and working
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 25%, #16213e 50%, #0f3460 75%, #533a71 100%);
        color: #ffffff;
    }
    
    .main-header {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00d4ff 0%, #090979 35%, #ff006e 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin: 2rem 0;
    }
    
    .subtitle {
        text-align: center;
        font-size: 1.2rem;
        color: #b8c5d1;
        margin-bottom: 2rem;
    }
    
    .strategy-box {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 2rem;
        border-radius: 20px;
        margin: 1.5rem 0;
        color: #ffffff;
    }
    
    .strategy-box h2, .strategy-box h3 {
        color: #ffffff !important;
    }
    
    .strategy-box p {
        color: #cbd5e0 !important;
    }
    
    .bullish-signal {
        background: linear-gradient(135deg, #00ff87 0%, #60efff 100%);
        color: #000000 !important;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
    
    .bullish-signal h3, .bullish-signal p, .bullish-signal strong {
        color: #000000 !important;
    }
    
    .bearish-signal {
        background: linear-gradient(135deg, #ff006e 0%, #fb5607 100%);
        color: #ffffff !important;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
    
    .bearish-signal h3, .bearish-signal p, .bearish-signal strong {
        color: #ffffff !important;
    }
    
    .neutral-signal {
        background: linear-gradient(135deg, #ffd60a 0%, #ff8500 100%);
        color: #000000 !important;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
    
    .neutral-signal h3, .neutral-signal p, .neutral-signal strong {
        color: #000000 !important;
    }
    
    .stSidebar {
        background: linear-gradient(180deg, #0f0f23 0%, #1a1a2e 100%);
    }
    
    .stSidebar .stMarkdown, .stSidebar label {
        color: #ffffff !important;
    }
    
    .stButton button {
        background: linear-gradient(135deg, #00d4ff 0%, #ff006e 100%);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
    }
    
    .stMetric {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .stMetric label, .stMetric [data-testid="metric-container"] {
        color: #ffffff !important;
    }
    
    h1, h2, h3, h4, h5, h6, p {
        color: #ffffff !important;
    }
    
    .stExpander {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
</style>
""", unsafe_allow_html=True)

class SimpleMockAccount:
    """Simplified mock trading account"""
    
    def __init__(self):
        self.initial_balance = 100000
        
    def get_current_balance(self) -> Dict:
        """Get current portfolio balance"""
        return {
            'total_value': 103800,
            'cash_balance': 95400,
            'positions_value': 8400,
            'realized_pnl': 3800,
            'total_return_pct': 3.8,
            'open_trades_count': 3,
            'win_rate': 68.5
        }
    
    def add_trade(self, symbol: str, trade_type: str, entry_price: float, 
                  quantity: int, confidence_score: int, setup_type: str, recommendation: str):
        """Add a new trade"""
        st.success(f"✅ Demo trade executed: {quantity} contracts of {symbol} {trade_type}")
        return True
    
    def get_open_trades(self) -> pd.DataFrame:
        """Get sample open trades"""
        trades = [
            {
                'symbol': 'MARA', 'trade_type': 'CALL_SELLING', 'entry_date': '2024-08-23',
                'entry_price': 1.45, 'quantity': 5, 'confidence_score': 88,
                'setup_type': 'CALL_SELLING_SETUP', 'days_held': 1
            },
            {
                'symbol': 'GME', 'trade_type': 'LONG_CALLS', 'entry_date': '2024-08-22',
                'entry_price': 2.35, 'quantity': 3, 'confidence_score': 92,
                'setup_type': 'SQUEEZE_SETUP', 'days_held': 2
            }
        ]
        return pd.DataFrame(trades)

class SimpleGEXDashboard:
    """Simplified GEX dashboard"""
    
    def __init__(self):
        self.mock_account = SimpleMockAccount()
    
    def get_morning_opportunities(self):
        """Get sample morning opportunities"""
        return [
            {
                'symbol': 'MARA',
                'current_price': 25.50,
                'gamma_flip': 23.20,
                'distance_pct': 9.02,
                'structure_type': 'CALL_SELLING_SETUP',
                'confidence_score': 95,
                'recommendation': 'SELL CALLS - Above gamma flip',
                'category': 'Crypto',
                'trade_type': 'CALL_SELLING'
            },
            {
                'symbol': 'GME',
                'current_price': 22.15,
                'gamma_flip': 24.80,
                'distance_pct': -11.97,
                'structure_type': 'SQUEEZE_SETUP',
                'confidence_score': 92,
                'recommendation': 'BUY CALLS - Squeeze potential',
                'category': 'Meme',
                'trade_type': 'LONG_CALLS'
            },
            {
                'symbol': 'COIN',
                'current_price': 245.30,
                'gamma_flip': 244.95,
                'distance_pct': 0.14,
                'structure_type': 'GAMMA_FLIP_CRITICAL',
                'confidence_score': 88,
                'recommendation': 'STRADDLE - At flip point',
                'category': 'Crypto',
                'trade_type': 'STRADDLE'
            }
        ]

def explain_gex_strategy():
    """Explain GEX strategy simply"""
    st.markdown("""
    <div class="strategy-box">
    <h2>🎯 What is Gamma Exposure (GEX) Trading?</h2>
    
    <p><strong>Gamma Exposure (GEX)</strong> is finding the "invisible hand" that moves stock prices. 
    Market makers have to buy and sell stocks to hedge their options positions, creating predictable patterns.</p>
    
    <h3>🔑 The "Gamma Flip Point"</h3>
    <p>Every stock has a special price level where market maker behavior changes:</p>
    <ul>
        <li><strong>Above the Flip:</strong> Market makers SELL when price rises → Creates resistance</li>
        <li><strong>Below the Flip:</strong> Market makers BUY when price rises → Creates squeeze potential</li>
        <li><strong>At the Flip:</strong> Maximum volatility and big moves in either direction</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

def explain_trading_signals():
    """Explain the trading signals"""
    st.markdown("## 🎯 Our 3 Main Trading Strategies:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="bullish-signal">
        <h3>🚀 SQUEEZE PLAYS</h3>
        <p><strong>When:</strong> Price BELOW gamma flip</p>
        <p><strong>Trade:</strong> Buy call options</p>
        <p><strong>Target:</strong> 100%+ gains</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="bearish-signal">
        <h3>💰 PREMIUM SELLING</h3>
        <p><strong>When:</strong> Price ABOVE gamma flip</p>
        <p><strong>Trade:</strong> Sell call options</p>
        <p><strong>Target:</strong> 50% premium collection</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="neutral-signal">
        <h3>⚖️ VOLATILITY PLAYS</h3>
        <p><strong>When:</strong> Price AT gamma flip</p>
        <p><strong>Trade:</strong> Straddles (calls + puts)</p>
        <p><strong>Target:</strong> Profit from big moves</p>
        </div>
        """, unsafe_allow_html=True)

def display_morning_analysis():
    """Display morning analysis"""
    st.header("🌅 This Morning's Analysis")
    
    dashboard = SimpleGEXDashboard()
    opportunities = dashboard.get_morning_opportunities()
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Opportunities", len(opportunities))
    with col2:
        high_conf = len([o for o in opportunities if o['confidence_score'] >= 90])
        st.metric("🔥 Exceptional (90%+)", high_conf)
    with col3:
        avg_conf = np.mean([o['confidence_score'] for o in opportunities])
        st.metric("Average Confidence", f"{avg_conf:.1f}%")
    
    # Opportunities
    st.subheader("🎯 Today's Top Picks")
    
    for i, opp in enumerate(opportunities):
        if opp['structure_type'] == 'SQUEEZE_SETUP':
            signal_class = "bullish-signal"
            emoji = "🚀"
        elif 'CALL_SELLING' in opp['structure_type']:
            signal_class = "bearish-signal"
            emoji = "💰"
        else:
            signal_class = "neutral-signal"
            emoji = "⚖️"
            
        confidence_badge = "🔥" if opp['confidence_score'] >= 90 else "🎯"
        
        st.markdown(f"""
        <div class="{signal_class}">
            <h3>{emoji} {opp['symbol']} - {confidence_badge} {opp['confidence_score']}% Confidence</h3>
            <p><strong>Setup:</strong> {opp['structure_type']}</p>
            <p><strong>Price:</strong> ${opp['current_price']:.2f} | <strong>Flip:</strong> ${opp['gamma_flip']:.2f}</p>
            <p><strong>🎯 Strategy:</strong> {opp['recommendation']}</p>
        </div>
        """, unsafe_allow_html=True)

def display_mock_portfolio():
    """Display mock portfolio"""
    st.header("💰 Mock Trading Account")
    
    dashboard = SimpleGEXDashboard()
    balance = dashboard.mock_account.get_current_balance()
    
    # Portfolio metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Portfolio Value", f"${balance['total_value']:,.2f}", f"{balance['total_return_pct']:+.1f}%")
    with col2:
        st.metric("Cash Balance", f"${balance['cash_balance']:,.2f}")
    with col3:
        st.metric("Open Positions", f"${balance['positions_value']:,.2f}")
    with col4:
        st.metric("Win Rate", f"{balance['win_rate']:.1f}%")
    
    # Trading opportunities
    st.subheader("📈 Execute Trades")
    opportunities = dashboard.get_morning_opportunities()
    
    for i, opp in enumerate(opportunities[:3]):
        with st.expander(f"🎯 {opp['symbol']} - {opp['confidence_score']}%", expanded=False):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write(f"**Strategy:** {opp['recommendation']}")
                st.write(f"**Setup:** {opp['structure_type']}")
                st.write(f"**Price:** ${opp['current_price']:.2f} | **Flip:** ${opp['gamma_flip']:.2f}")
            
            with col2:
                if st.button(f"Execute Trade", key=f"trade_{i}"):
                    dashboard.mock_account.add_trade(
                        symbol=opp['symbol'],
                        trade_type=opp['trade_type'],
                        entry_price=2.50,
                        quantity=5,
                        confidence_score=opp['confidence_score'],
                        setup_type=opp['structure_type'],
                        recommendation=opp['recommendation']
                    )
    
    # Open trades
    trades = dashboard.mock_account.get_open_trades()
    if not trades.empty:
        st.subheader("📊 Open Positions")
        st.dataframe(trades, use_container_width=True)

def main():
    """Main app"""
    
    # Header
    st.markdown('<div class="main-header">🎯 GEX TRADING DASHBOARD</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Turn Market Maker Psychology Into Profits</div>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.markdown("## 📋 Navigation")
    page = st.sidebar.radio("Choose Page:", [
        "🎓 Learn Strategy", 
        "🌅 Morning Analysis", 
        "💰 Trading Account"
    ])
    
    if page == "🎓 Learn Strategy":
        st.header("🎓 Learn the GEX Strategy")
        explain_gex_strategy()
        explain_trading_signals()
        
        st.markdown("""
        <div class="strategy-box">
        <h2>🎯 Why This Works</h2>
        <p><strong>Market Structure Based:</strong> Not just technical analysis</p>
        <p><strong>Predictable Patterns:</strong> Market makers must hedge</p>
        <p><strong>Quantified Approach:</strong> Remove emotion with confidence scores</p>
        </div>
        """, unsafe_allow_html=True)
        
    elif page == "🌅 Morning Analysis":
        display_morning_analysis()
        
    elif page == "💰 Trading Account":
        display_mock_portfolio()

if __name__ == "__main__":
    main()
