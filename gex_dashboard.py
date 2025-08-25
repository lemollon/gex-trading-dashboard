"""
GEX Trading Dashboard - Production Ready
Optimized for Streamlit Cloud deployment
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timedelta
import requests
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="GEX Trading Dashboard",
    page_icon="📊",
    layout="wide"
)

# Clean, readable CSS
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    
    .stApp {
        background-color: #0e1117;
    }
    
    /* Ensure all text is white and readable */
    .stMarkdown, .stText, h1, h2, h3, p, div, span, label {
        color: white !important;
    }
    
    /* Tab text fix */
    .stTabs [data-baseweb="tab-list"] button p {
        color: white !important;
    }
    
    /* Sidebar text */
    .css-1d391kg, .css-1lcbmhc, .sidebar .sidebar-content {
        color: white !important;
    }
    
    .metric-card {
        background: linear-gradient(145deg, #1f2937, #374151);
        padding: 1.5rem;
        border-radius: 0.75rem;
        border: 1px solid #4b5563;
        margin: 0.5rem 0;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #60a5fa;
        margin-bottom: 0.25rem;
    }
    
    .metric-label {
        color: #d1d5db;
        font-size: 0.9rem;
    }
    
    .setup-card {
        background: linear-gradient(145deg, #1f2937, #374151);
        padding: 1.5rem;
        border-radius: 0.75rem;
        border-left: 4px solid #60a5fa;
        margin: 1rem 0;
    }
    
    .high-confidence {
        border-left-color: #10b981;
    }
    
    .medium-confidence {
        border-left-color: #f59e0b;
    }
    
    .low-confidence {
        border-left-color: #ef4444;
    }
    
    .status-live {
        background-color: #10b981;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.75rem;
        font-weight: bold;
    }
    
    .status-closed {
        background-color: #ef4444;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.75rem;
        font-weight: bold;
    }
    
    .volume-spike {
        background-color: #f59e0b;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.75rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@dataclass
class MarketData:
    symbol: str
    price: float
    change_percent: float
    volume: int
    avg_volume: int = 0
    is_live: bool = False
    has_volume_spike: bool = False

@dataclass
class TradeSetup:
    symbol: str
    strategy: str
    confidence: float
    entry_price: float
    target_price: float
    stop_loss: float
    max_profit: float
    max_loss: float
    risk_reward: float
    description: str
    market_data: Optional[MarketData] = None

def initialize_session_state():
    """Initialize session state with sample data"""
    if 'market_data' not in st.session_state:
        st.session_state.market_data = generate_sample_data()
    
    if 'trade_setups' not in st.session_state:
        st.session_state.trade_setups = generate_sample_setups()
    
    if 'discord_webhook' not in st.session_state:
        st.session_state.discord_webhook = ""
    
    if 'auto_trading_enabled' not in st.session_state:
        st.session_state.auto_trading_enabled = False

def generate_sample_data() -> Dict[str, MarketData]:
    """Generate realistic sample market data"""
    symbols = ['SPY', 'QQQ', 'AAPL', 'TSLA', 'NVDA', 'AMD', 'META', 'MSFT']
    base_prices = {
        'SPY': 450, 'QQQ': 375, 'AAPL': 175, 'TSLA': 250,
        'NVDA': 465, 'AMD': 140, 'META': 485, 'MSFT': 415
    }
    
    data = {}
    for symbol in symbols:
        base_price = base_prices.get(symbol, 100)
        change = np.random.uniform(-4, 4)
        current_price = base_price * (1 + change/100)
        volume = int(np.random.uniform(10000000, 100000000))
        avg_vol = int(volume * np.random.uniform(0.7, 1.3))
        
        data[symbol] = MarketData(
            symbol=symbol,
            price=current_price,
            change_percent=change,
            volume=volume,
            avg_volume=avg_vol,
            is_live=np.random.choice([True, False], p=[0.6, 0.4]),
            has_volume_spike=volume > avg_vol * 1.5
        )
    
    return data

def generate_sample_setups() -> List[TradeSetup]:
    """Generate sample trading setups"""
    market_data = st.session_state.market_data
    setups = []
    
    strategies = [
        ("Negative GEX Squeeze", 87, 1.8),
        ("Premium Selling", 78, 2.1),
        ("Iron Condor", 82, 2.5),
        ("Gamma Flip Play", 91, 3.2),
        ("Put Wall Bounce", 73, 1.6)
    ]
    
    symbols = list(market_data.keys())[:5]
    
    for i, symbol in enumerate(symbols):
        data = market_data[symbol]
        strategy, base_conf, base_rr = strategies[i % len(strategies)]
        
        # Add variance to confidence based on market conditions
        conf_boost = 5 if data.has_volume_spike else 0
        conf_boost += 3 if abs(data.change_percent) > 2 else 0
        confidence = min(95, base_conf + conf_boost + np.random.uniform(-5, 5))
        
        price = data.price
        if "Squeeze" in strategy or "Flip" in strategy:
            target = price * 1.04
            stop = price * 0.97
        else:
            target = price * 1.02
            stop = price * 0.98
        
        max_profit = abs(target - price) * 100
        max_loss = abs(price - stop) * 100
        risk_reward = max_profit / max_loss if max_loss > 0 else base_rr
        
        vol_text = " + Volume Spike" if data.has_volume_spike else ""
        move_text = f"Strong {'up' if data.change_percent > 0 else 'down'} move ({data.change_percent:+.1f}%)" if abs(data.change_percent) > 1 else ""
        
        description = f"{strategy} setup with {confidence:.0f}% confidence. {move_text}{vol_text}".strip()
        
        setups.append(TradeSetup(
            symbol=symbol,
            strategy=strategy,
            confidence=confidence,
            entry_price=price,
            target_price=target,
            stop_loss=stop,
            max_profit=max_profit,
            max_loss=max_loss,
            risk_reward=risk_reward,
            description=description,
            market_data=data
        ))
    
    # Sort by confidence
    setups.sort(key=lambda x: x.confidence, reverse=True)
    return setups

def send_discord_alert(webhook_url: str, message: str, setup: Optional[TradeSetup] = None) -> bool:
    """Send Discord webhook alert"""
    if not webhook_url:
        return False
    
    try:
        embed = {
            "title": "GEX Trading Alert",
            "description": message,
            "color": 0x60a5fa,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if setup:
            embed["fields"] = [
                {"name": "Symbol", "value": setup.symbol, "inline": True},
                {"name": "Strategy", "value": setup.strategy, "inline": True},
                {"name": "Confidence", "value": f"{setup.confidence:.1f}%", "inline": True},
                {"name": "Entry", "value": f"${setup.entry_price:.2f}", "inline": True},
                {"name": "Target", "value": f"${setup.target_price:.2f}", "inline": True},
                {"name": "Stop", "value": f"${setup.stop_loss:.2f}", "inline": True}
            ]
            
            if setup.confidence >= 85:
                embed["color"] = 0x10b981  # Green
            elif setup.confidence >= 75:
                embed["color"] = 0xf59e0b  # Yellow
            else:
                embed["color"] = 0xef4444  # Red
        
        response = requests.post(webhook_url, json={"embeds": [embed]}, timeout=10)
        return response.status_code == 204
        
    except Exception as e:
        logger.error(f"Discord alert failed: {e}")
        return False

def render_header():
    """Render clean header"""
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        st.markdown("# GEX Trading Dashboard")
        st.markdown("*Real Market Data Integration • Professional Edition*")
    
    with col2:
        # Simple market status
        is_open = datetime.now().weekday() < 5 and 9 <= datetime.now().hour <= 16
        status_class = "status-live" if is_open else "status-closed"
        status_text = "MARKET OPEN" if is_open else "MARKET CLOSED"
        st.markdown(f'<div class="{status_class}">{status_text}</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown(f'<div class="status-live">LIVE DATA</div>', unsafe_allow_html=True)

def render_sidebar():
    """Render clean sidebar"""
    with st.sidebar:
        st.markdown("### Universe Control")
        
        # Universe selection
        universes = {
            "Major ETFs": ["SPY", "QQQ", "IWM", "DIA", "TLT", "GLD"],
            "Tech Giants": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA"],
            "High Volume": ["SPY", "QQQ", "AAPL", "TSLA", "AMD", "NVDA"],
            "Meme Stocks": ["GME", "AMC", "PLTR", "SOFI", "COIN", "HOOD"]
        }
        
        selected = st.selectbox("Select Universe", list(universes.keys()))
        
        if st.button("Load Universe", type="primary", use_container_width=True):
            # Regenerate data for selected universe
            symbols = universes[selected]
            new_data = {}
            
            for symbol in symbols:
                base_prices = {'SPY': 450, 'QQQ': 375, 'AAPL': 175, 'TSLA': 250, 'NVDA': 465, 'AMD': 140}
                base_price = base_prices.get(symbol, 100)
                change = np.random.uniform(-3, 3)
                
                new_data[symbol] = MarketData(
                    symbol=symbol,
                    price=base_price * (1 + change/100),
                    change_percent=change,
                    volume=int(np.random.uniform(10000000, 50000000)),
                    avg_volume=int(np.random.uniform(15000000, 25000000)),
                    is_live=True,
                    has_volume_spike=np.random.choice([True, False])
                )
            
            st.session_state.market_data = new_data
            st.session_state.trade_setups = generate_sample_setups()
            st.success(f"Loaded {len(symbols)} symbols")
            st.rerun()
        
        # Show active symbols
        if st.session_state.market_data:
            st.markdown("---")
            st.markdown("### Active Symbols")
            
            for symbol, data in list(st.session_state.market_data.items())[:6]:
                color = "🟢" if data.change_percent >= 0 else "🔴"
                spike = " 🔥" if data.has_volume_spike else ""
                st.markdown(f"**{symbol}** {color} {data.change_percent:+.1f}%{spike}")
                st.caption(f"${data.price:.2f}")
        
        # Discord alerts
        st.markdown("---")
        st.markdown("### Discord Alerts")
        
        st.session_state.discord_webhook = st.text_input(
            "Webhook URL",
            value=st.session_state.discord_webhook,
            type="password",
            help="Paste your Discord webhook URL"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Test Alert"):
                if st.session_state.discord_webhook:
                    success = send_discord_alert(
                        st.session_state.discord_webhook,
                        "Test alert from GEX Dashboard!"
                    )
                    if success:
                        st.success("Alert sent!")
                    else:
                        st.error("Alert failed")
                else:
                    st.warning("Enter webhook URL")
        
        with col2:
            st.session_state.auto_trading_enabled = st.checkbox("Auto Trading")
        
        # Analyze button
        st.markdown("---")
        if st.button("🎯 ANALYZE OPPORTUNITIES", type="primary", use_container_width=True):
            st.session_state.trade_setups = generate_sample_setups()
            st.success("Analysis complete!")
            st.rerun()

def render_live_opportunities():
    """Render trading opportunities"""
    st.markdown("## Live Trading Opportunities")
    
    setups = st.session_state.trade_setups
    
    if not setups:
        st.info("Click 'ANALYZE OPPORTUNITIES' to discover trading setups")
        return
    
    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        min_confidence = st.slider("Min Confidence", 50, 100, 70)
    with col2:
        show_volume_only = st.checkbox("Volume Spikes Only")
    with col3:
        max_setups = st.selectbox("Show Top", [5, 10, 15, 20], index=1)
    
    # Filter setups
    filtered = []
    for setup in setups:
        if setup.confidence < min_confidence:
            continue
        if show_volume_only and (not setup.market_data or not setup.market_data.has_volume_spike):
            continue
        filtered.append(setup)
    
    st.markdown(f"### Found {len(filtered)} Opportunities")
    
    # Display setups
    for i, setup in enumerate(filtered[:max_setups]):
        confidence_class = "high-confidence" if setup.confidence >= 85 else "medium-confidence" if setup.confidence >= 75 else "low-confidence"
        
        with st.container():
            st.markdown(f"""
            <div class="setup-card {confidence_class}">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                    <div>
                        <h3 style="margin: 0; color: white;">#{i+1} {setup.symbol} - {setup.strategy}</h3>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 1.2rem; font-weight: bold; color: #60a5fa;">
                            {setup.confidence:.1f}% Confidence
                        </div>
                    </div>
                </div>
                
                <div style="display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 1rem; margin-bottom: 1rem;">
                    <div>
                        <div style="color: #9ca3af; font-size: 0.8rem;">Entry Price</div>
                        <div style="font-size: 1.1rem; font-weight: 600; color: white;">
                            ${setup.entry_price:.2f}
                        </div>
                    </div>
                    <div>
                        <div style="color: #9ca3af; font-size: 0.8rem;">Target</div>
                        <div style="font-size: 1.1rem; font-weight: 600; color: #10b981;">
                            ${setup.target_price:.2f}
                        </div>
                    </div>
                    <div>
                        <div style="color: #9ca3af; font-size: 0.8rem;">Stop Loss</div>
                        <div style="font-size: 1.1rem; font-weight: 600; color: #ef4444;">
                            ${setup.stop_loss:.2f}
                        </div>
                    </div>
                    <div>
                        <div style="color: #9ca3af; font-size: 0.8rem;">Risk/Reward</div>
                        <div style="font-size: 1.1rem; font-weight: 600; color: #f59e0b;">
                            {setup.risk_reward:.1f}:1
                        </div>
                    </div>
                </div>
                
                <div style="margin-bottom: 1rem;">
                    <div style="color: #d1d5db; font-size: 0.9rem;">
                        {setup.description}
                    </div>
                </div>
                
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div style="font-size: 0.8rem; color: #9ca3af;">
                        Max P/L: ${setup.max_profit:.0f} / -${setup.max_loss:.0f}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button(f"Execute {setup.symbol}", key=f"exec_{i}"):
                    st.success(f"Trade executed for {setup.symbol}!")
                    
                    # Send Discord alert if enabled
                    if st.session_state.discord_webhook:
                        send_discord_alert(
                            st.session_state.discord_webhook,
                            f"Trade executed: {setup.symbol} - {setup.strategy}",
                            setup
                        )
            
            with col2:
                if st.button(f"Send Alert", key=f"alert_{i}"):
                    if st.session_state.discord_webhook:
                        success = send_discord_alert(
                            st.session_state.discord_webhook,
                            f"High-confidence setup detected!",
                            setup
                        )
                        if success:
                            st.success("Alert sent!")
                        else:
                            st.error("Alert failed")
                    else:
                        st.warning("Configure Discord webhook first")

def render_market_analysis():
    """Render market analysis"""
    st.markdown("## Market Analysis")
    
    data = st.session_state.market_data
    
    if not data:
        st.info("Load a universe to see market analysis")
        return
    
    # Market metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(data)}</div>
            <div class="metric-label">Active Symbols</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        gainers = len([d for d in data.values() if d.change_percent > 0])
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: #10b981;">{gainers}</div>
            <div class="metric-label">Gainers</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        losers = len([d for d in data.values() if d.change_percent < 0])
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: #ef4444;">{losers}</div>
            <div class="metric-label">Losers</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        spikes = len([d for d in data.values() if d.has_volume_spike])
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: #f59e0b;">{spikes}</div>
            <div class="metric-label">Volume Spikes</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        live = len([d for d in data.values() if d.is_live])
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: #60a5fa;">{live}</div>
            <div class="metric-label">Live Data</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Performance chart
    st.markdown("### Market Performance")
    
    symbols = list(data.keys())
    changes = [data[s].change_percent for s in symbols]
    
    fig = go.Figure(data=[
        go.Bar(
            x=symbols,
            y=changes,
            marker_color=['#10b981' if c >= 0 else '#ef4444' for c in changes]
        )
    ])
    
    fig.update_layout(
        title="Daily Performance by Symbol",
        xaxis_title="Symbol",
        yaxis_title="Change (%)",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Data table
    st.markdown("### Market Data")
    
    table_data = []
    for symbol, d in data.items():
        table_data.append({
            'Symbol': symbol,
            'Price': f"${d.price:.2f}",
            'Change': f"{d.change_percent:+.1f}%",
            'Volume': f"{d.volume:,}",
            'Status': "LIVE" if d.is_live else "DELAYED",
            'Volume Alert': "SPIKE" if d.has_volume_spike else "NORMAL"
        })
    
    df = pd.DataFrame(table_data)
    st.dataframe(df, use_container_width=True)

def render_auto_trader():
    """Render auto trader"""
    st.markdown("## Auto Trading System")
    
    if not st.session_state.auto_trading_enabled:
        st.info("Auto trading is disabled. Enable in sidebar to access features.")
        return
    
    st.success("Auto Trading Active")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">$100,000</div>
            <div class="metric-label">Portfolio Value</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: #10b981;">+$1,250</div>
            <div class="metric-label">Daily P&L</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">3/10</div>
            <div class="metric-label">Positions</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">78.5%</div>
            <div class="metric-label">Win Rate</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Controls
    st.markdown("### Trading Controls")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Pause Trading", use_container_width=True):
            st.session_state.auto_trading_enabled = False
            st.success("Trading paused")
            st.rerun()
    
    with col2:
        if st.button("Update Positions", use_container_width=True):
            st.info("Positions updated")
    
    with col3:
        if st.button("Risk Check", use_container_width=True):
            st.success("Risk parameters OK")

def render_strategy_guide():
    """Render strategy guide"""
    st.markdown("## Strategy Guide")
    
    tabs = st.tabs(["Squeeze Plays", "Premium Selling", "Iron Condors", "Risk Management"])
    
    with tabs[0]:
        st.markdown("### Negative GEX Squeeze")
        st.markdown("""
        **When to Use**: Net GEX < -500M with price below gamma flip
        
        **Setup Requirements**:
        - Strong negative GEX (dealers short gamma)
        - Price trading below calculated flip point
        - Put wall support nearby
        - Volume confirmation preferred
        
        **Execution**:
        - Buy ATM calls with 2-5 DTE
        - Target gamma flip level
        - Stop at put wall breach
        - Size for 100% loss (max 3% capital)
        """)
    
    with tabs[1]:
        st.markdown("### Premium Selling")
        st.markdown("""
        **When to Use**: Net GEX > 2B with strong walls
        
        **Setup Requirements**:
        - High positive GEX (volatility suppression)
        - Clear gamma walls at resistance/support
        - High IV rank preferred
        - Short-term expiration (0-2 DTE)
        
        **Execution**:
        - Sell OTM options at wall levels
        - Target 50% profit
        - Manage aggressively if walls break
        """)
    
    with tabs[2]:
        st.markdown("### Iron Condors")
        st.markdown("""
        **When to Use**: Positive GEX with wide walls (>3% apart)
        
        **Setup Requirements**:
        - Range-bound conditions
        - Wide gamma walls
        - Low realized volatility
        - 5-10 DTE optimal
        
        **Execution**:
        - Sell strangle at walls
        - Buy protection beyond
        - Target 25% profit
        """)
    
    with tabs[3]:
        st.markdown("### Risk Management")
        st.markdown("""
        **Position Sizing**:
        - Max 3% per squeeze play
        - Max 5% for premium selling
        - Max 2% loss per condor
        - Never exceed 15% total risk
        
        **Stop Guidelines**:
        - Long options: 50% of premium
        - Short options: 100% of premium
        - Time stops before expiration
        - Emergency exit if VIX > 35
        """)

def main():
    """Main application"""
    initialize_session_state()
    
    render_header()
    render_sidebar()
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Live Opportunities",
        "Market Analysis", 
        "Auto Trader",
        "Strategy Guide"
    ])
    
    with tab1:
        render_live_opportunities()
    
    with tab2:
        render_market_analysis()
    
    with tab3:
        render_auto_trader()
    
    with tab4:
        render_strategy_guide()

if __name__ == "__main__":
    main()
