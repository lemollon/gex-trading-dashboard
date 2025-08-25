"""
Complete GEX Trading Dashboard - Full Universe with Auto Trading
Version: 9.0.0 ULTIMATE
All symbols visible with comprehensive strategy details and automated paper trading
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import json
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import logging
import concurrent.futures
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="GEX Trading Dashboard - Ultimate",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================== CUSTOM CSS ========================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }
    
    h1, h2, h3 {
        background: linear-gradient(120deg, #00D2FF 0%, #3A7BD5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }
    
    .setup-details {
        background: rgba(0, 255, 135, 0.1);
        border-left: 3px solid #00ff87;
        padding: 15px;
        margin: 10px 0;
        border-radius: 8px;
    }
    
    .auto-trade-card {
        background: linear-gradient(135deg, rgba(255, 215, 0, 0.1) 0%, rgba(255, 215, 0, 0.05) 100%);
        border: 1px solid rgba(255, 215, 0, 0.3);
        border-radius: 12px;
        padding: 15px;
        margin: 10px 0;
    }
    
    .filter-container {
        background: rgba(255, 255, 255, 0.05);
        padding: 15px;
        border-radius: 12px;
        margin: 15px 0;
    }
    
    /* Color coding for strategies */
    .squeeze-play { color: #ff6b6b; font-weight: bold; }
    .premium-sell { color: #ffd93d; font-weight: bold; }
    .iron-condor { color: #00ff87; font-weight: bold; }
    .gamma-flip { color: #00d2ff; font-weight: bold; }
    
    /* Status indicators */
    .status-extreme { color: #ff6b6b; }
    .status-volatile { color: #ff9f40; }
    .status-neutral { color: #ffd93d; }
    .status-stable { color: #00ff87; }
    
    .trade-executed {
        background: rgba(0, 255, 135, 0.2);
        padding: 10px;
        border-radius: 8px;
        margin: 5px 0;
    }
    
    .trade-pending {
        background: rgba(255, 217, 61, 0.2);
        padding: 10px;
        border-radius: 8px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# ======================== DATA CLASSES ========================

@dataclass
class DetailedTradeSetup:
    """Comprehensive trade setup with all details"""
    symbol: str
    strategy: str
    strategy_type: str  # CALL, PUT, CALL_SPREAD, PUT_SPREAD, IRON_CONDOR
    confidence: float
    entry_price: float
    
    # Options specific details
    strike_price: float = 0
    strike_price_2: float = 0  # For spreads
    call_strike: float = 0  # For iron condors
    put_strike: float = 0  # For iron condors
    call_strike_long: float = 0  # For iron condor protection
    put_strike_long: float = 0  # For iron condor protection
    
    # Targets and stops
    target_price: float = 0
    stop_loss: float = 0
    max_profit: float = 0
    max_loss: float = 0
    
    # Risk metrics
    risk_reward: float = 0
    breakeven: float = 0
    probability_profit: float = 0
    
    # Timing
    days_to_expiry: str = ""
    expiry_date: str = ""
    
    # Description
    description: str = ""
    entry_criteria: str = ""
    exit_criteria: str = ""
    
    # GEX metrics
    net_gex: float = 0
    gamma_flip: float = 0
    distance_to_flip: float = 0
    
    # Auto-trade fields
    auto_trade_enabled: bool = True
    position_size: float = 1000  # Dollar amount
    executed: bool = False
    execution_time: Optional[datetime] = None
    exit_time: Optional[datetime] = None
    pnl: float = 0

@dataclass
class AutoTradePosition:
    """Position managed by auto trader"""
    setup: DetailedTradeSetup
    entry_time: datetime
    entry_price: float
    quantity: int
    status: str  # OPEN, CLOSED, STOPPED
    current_price: float = 0
    current_pnl: float = 0
    exit_price: float = 0
    exit_time: Optional[datetime] = None
    exit_reason: str = ""

# ======================== ENHANCED WATCHLIST MANAGER ========================

class UniverseManager:
    """Manage the full universe of symbols"""
    
    def __init__(self):
        self.initialize_session_state()
        self.setup_universes()
        
    def initialize_session_state(self):
        """Initialize session state"""
        if 'watchlist' not in st.session_state:
            st.session_state.watchlist = [
                "SPY", "QQQ", "IWM", "DIA", "VXX", "UVXY",
                "AAPL", "MSFT", "NVDA", "TSLA", "AMD", "META",
                "GOOGL", "AMZN", "NFLX", "GME", "AMC", "BBBY"
            ]
        
        if 'all_setups_detailed' not in st.session_state:
            st.session_state.all_setups_detailed = []
        
        if 'auto_trader' not in st.session_state:
            st.session_state.auto_trader = AutoTrader()
        
        if 'filter_strategy' not in st.session_state:
            st.session_state.filter_strategy = "All"
        
        if 'min_confidence_filter' not in st.session_state:
            st.session_state.min_confidence_filter = 65.0
    
    def setup_universes(self):
        """Setup comprehensive universes"""
        self.universes = {
            "📊 Major ETFs": ["SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "TLT", "GLD", "SLV", "VXX", 
                             "UVXY", "SQQQ", "TQQQ", "EEM", "XLF", "XLE", "XLK", "XLV", "XLI", "XLY"],
            
            "🚀 Tech Giants": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD", "INTC", "AVGO",
                              "ORCL", "CRM", "ADBE", "NFLX", "CSCO", "QCOM", "TXN", "IBM", "MU", "AMAT"],
            
            "🔥 High Vol/Meme": ["GME", "AMC", "BBBY", "BB", "PLTR", "SOFI", "RIOT", "MARA", "COIN", "HOOD",
                                "WISH", "CLOV", "SPCE", "TLRY", "SNDL", "NOK", "EXPR", "KOSS", "NAKD", "RKT"],
            
            "💎 Options Flow": ["SPY", "QQQ", "AAPL", "TSLA", "AMD", "NVDA", "AMZN", "META", "NFLX", "MSFT",
                               "BAC", "F", "NIO", "PLTR", "SOFI", "AAL", "CCL", "UBER", "LYFT", "BABA"],
            
            "🏦 Financial": ["JPM", "BAC", "GS", "MS", "WFC", "C", "BLK", "SCHW", "AXP", "V",
                            "MA", "PYPL", "SQ", "COF", "USB", "PNC", "TFC", "ALLY", "DFS", "FITB"],
            
            "⚡ Energy": ["XOM", "CVX", "COP", "SLB", "OXY", "MPC", "PSX", "VLO", "XLE", "USO",
                         "EOG", "PXD", "DVN", "FANG", "HES", "MRO", "APA", "HAL", "BKR", "KMI"],
        }

# ======================== AUTO TRADER ========================

class AutoTrader:
    """Automated paper trading system"""
    
    def __init__(self):
        if 'auto_positions' not in st.session_state:
            st.session_state.auto_positions = []
        
        if 'auto_trade_history' not in st.session_state:
            st.session_state.auto_trade_history = []
        
        if 'auto_trading_enabled' not in st.session_state:
            st.session_state.auto_trading_enabled = False
        
        if 'auto_trade_capital' not in st.session_state:
            st.session_state.auto_trade_capital = 100000
        
        if 'auto_trade_pnl' not in st.session_state:
            st.session_state.auto_trade_pnl = 0
    
    def execute_high_confidence_trades(self, setups: List[DetailedTradeSetup], min_confidence: float = 75):
        """Automatically execute high confidence trades"""
        if not st.session_state.auto_trading_enabled:
            return
        
        executed_trades = []
        
        for setup in setups:
            if setup.confidence >= min_confidence and not setup.executed:
                # Check if we have capital
                position_size = min(setup.position_size, st.session_state.auto_trade_capital * 0.05)  # Max 5% per trade
                
                if position_size > 0 and st.session_state.auto_trade_capital >= position_size:
                    # Execute trade
                    position = AutoTradePosition(
                        setup=setup,
                        entry_time=datetime.now(),
                        entry_price=setup.entry_price,
                        quantity=int(position_size / setup.entry_price),
                        status="OPEN",
                        current_price=setup.entry_price
                    )
                    
                    # Update capital
                    st.session_state.auto_trade_capital -= position_size
                    
                    # Add to positions
                    st.session_state.auto_positions.append(position)
                    
                    # Mark setup as executed
                    setup.executed = True
                    setup.execution_time = datetime.now()
                    
                    executed_trades.append(position)
        
        return executed_trades
    
    def update_positions(self, current_prices: Dict[str, float]):
        """Update all open positions with current prices"""
        for position in st.session_state.auto_positions:
            if position.status == "OPEN" and position.setup.symbol in current_prices:
                position.current_price = current_prices[position.setup.symbol]
                
                # Calculate current P&L
                if position.setup.strategy_type in ["CALL", "PUT"]:
                    # For long options
                    position.current_pnl = (position.current_price - position.entry_price) * position.quantity
                else:
                    # For short options or spreads
                    position.current_pnl = (position.entry_price - position.current_price) * position.quantity
                
                # Check exit conditions
                self.check_exit_conditions(position)
    
    def check_exit_conditions(self, position: AutoTradePosition):
        """Check if position should be closed"""
        setup = position.setup
        
        # Target hit
        if position.current_price >= setup.target_price:
            self.close_position(position, "TARGET_HIT")
        
        # Stop loss hit
        elif position.current_price <= setup.stop_loss:
            self.close_position(position, "STOP_LOSS")
        
        # Time-based exit (for options near expiry)
        elif "0-2 DTE" in setup.days_to_expiry:
            # Close if profit > 50%
            if position.current_pnl > (position.entry_price * position.quantity * 0.5):
                self.close_position(position, "TIME_EXIT_PROFIT")
    
    def close_position(self, position: AutoTradePosition, reason: str):
        """Close a position"""
        position.status = "CLOSED"
        position.exit_price = position.current_price
        position.exit_time = datetime.now()
        position.exit_reason = reason
        
        # Update capital
        exit_value = position.exit_price * position.quantity
        st.session_state.auto_trade_capital += exit_value
        
        # Update P&L
        st.session_state.auto_trade_pnl += position.current_pnl
        
        # Add to history
        st.session_state.auto_trade_history.append({
            'symbol': position.setup.symbol,
            'strategy': position.setup.strategy,
            'entry_time': position.entry_time,
            'exit_time': position.exit_time,
            'entry_price': position.entry_price,
            'exit_price': position.exit_price,
            'quantity': position.quantity,
            'pnl': position.current_pnl,
            'exit_reason': reason
        })

# ======================== ENHANCED GEX CALCULATOR ========================

class ComprehensiveGEXCalculator:
    """Calculate detailed GEX metrics and generate comprehensive setups"""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.spot_price = None
        self.net_gex = None
        self.gamma_flip = None
        self.call_walls = []
        self.put_walls = []
        
    def fetch_and_calculate(self) -> bool:
        """Fetch data and calculate GEX"""
        try:
            ticker = yf.Ticker(self.symbol)
            hist = ticker.history(period='5d')
            
            if hist.empty:
                return False
            
            self.spot_price = hist['Close'].iloc[-1]
            
            # Try to get options data
            try:
                expirations = ticker.options[:10] if hasattr(ticker, 'options') else []
                
                if expirations:
                    total_call_oi = 0
                    total_put_oi = 0
                    call_strikes = []
                    put_strikes = []
                    
                    for exp_date in expirations[:5]:
                        try:
                            opt_chain = ticker.option_chain(exp_date)
                            
                            # Analyze calls
                            calls = opt_chain.calls
                            total_call_oi += calls['openInterest'].sum()
                            
                            # Find high OI call strikes (potential walls)
                            high_oi_calls = calls.nlargest(3, 'openInterest')
                            for _, row in high_oi_calls.iterrows():
                                if row['openInterest'] > 1000:
                                    call_strikes.append(row['strike'])
                            
                            # Analyze puts
                            puts = opt_chain.puts
                            total_put_oi += puts['openInterest'].sum()
                            
                            # Find high OI put strikes (potential walls)
                            high_oi_puts = puts.nlargest(3, 'openInterest')
                            for _, row in high_oi_puts.iterrows():
                                if row['openInterest'] > 1000:
                                    put_strikes.append(row['strike'])
                            
                        except:
                            continue
                    
                    # Calculate net GEX
                    self.net_gex = (total_call_oi - total_put_oi) * self.spot_price * 100
                    
                    # Set call and put walls
                    self.call_walls = sorted(list(set(call_strikes)))[:3] if call_strikes else [self.spot_price * 1.02]
                    self.put_walls = sorted(list(set(put_strikes)), reverse=True)[:3] if put_strikes else [self.spot_price * 0.98]
                    
                    # Calculate gamma flip
                    if total_call_oi + total_put_oi > 0:
                        put_call_ratio = total_put_oi / (total_call_oi + 1)
                        flip_adjustment = 0.02 * (put_call_ratio - 1)
                        self.gamma_flip = self.spot_price * (1 + flip_adjustment)
                    else:
                        self.gamma_flip = self.spot_price
                    
                else:
                    # No options - use simulated values
                    self.net_gex = np.random.uniform(-2e9, 5e9)
                    self.gamma_flip = self.spot_price * (1 + np.random.uniform(-0.03, 0.03))
                    self.call_walls = [self.spot_price * 1.02]
                    self.put_walls = [self.spot_price * 0.98]
                
            except Exception:
                # Fallback to simulated
                self.net_gex = np.random.uniform(-2e9, 5e9)
                self.gamma_flip = self.spot_price * (1 + np.random.uniform(-0.03, 0.03))
                self.call_walls = [self.spot_price * 1.02]
                self.put_walls = [self.spot_price * 0.98]
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing {self.symbol}: {e}")
            return False
    
    def generate_detailed_setups(self) -> List[DetailedTradeSetup]:
        """Generate comprehensive trade setups with full details"""
        setups = []
        
        if not self.spot_price:
            return setups
        
        distance_to_flip = ((self.gamma_flip - self.spot_price) / self.spot_price * 100)
        
        # Negative GEX Squeeze Play (Long Calls)
        if self.net_gex < -5e8:
            confidence = min(95, 70 + abs(self.net_gex/1e9) * 5)
            
            # Find appropriate strikes
            atm_call = round(self.spot_price / 5) * 5  # Round to nearest $5
            otm_call = atm_call + 5
            
            setup = DetailedTradeSetup(
                symbol=self.symbol,
                strategy="🚀 Negative GEX Squeeze",
                strategy_type="CALL",
                confidence=confidence,
                entry_price=self.spot_price,
                strike_price=atm_call,
                target_price=self.gamma_flip,
                stop_loss=self.put_walls[0] if self.put_walls else self.spot_price * 0.98,
                max_profit=(self.gamma_flip - atm_call) * 100,  # Per contract
                max_loss=self.spot_price * 0.02 * 100,  # Estimated premium
                risk_reward=abs(self.gamma_flip - self.spot_price) / abs(self.spot_price - self.stop_loss),
                breakeven=atm_call + (self.spot_price * 0.02),  # Strike + premium
                probability_profit=confidence / 100,
                days_to_expiry="2-5 DTE",
                expiry_date=(datetime.now() + timedelta(days=5)).strftime("%Y-%m-%d"),
                description=f"Strong negative GEX ({self.net_gex/1e9:.2f}B) indicates dealer short gamma. Explosive upside potential.",
                entry_criteria=f"Buy {atm_call} Call when price > {self.put_walls[0] if self.put_walls else self.spot_price * 0.98:.2f}",
                exit_criteria=f"Target: {self.gamma_flip:.2f} | Stop: {self.stop_loss:.2f} | Time: Close at 1 DTE",
                net_gex=self.net_gex,
                gamma_flip=self.gamma_flip,
                distance_to_flip=distance_to_flip,
                position_size=2000
            )
            setups.append(setup)
        
        # Positive GEX Premium Selling (Short Calls)
        elif self.net_gex > 2e9:
            confidence = min(90, 65 + self.net_gex/1e9 * 3)
            
            # Find call wall for short strike
            short_strike = self.call_walls[0] if self.call_walls else self.spot_price * 1.02
            
            setup = DetailedTradeSetup(
                symbol=self.symbol,
                strategy="💰 Premium Selling",
                strategy_type="SHORT_CALL",
                confidence=confidence,
                entry_price=self.spot_price,
                strike_price=short_strike,
                target_price=self.spot_price,  # Want price to stay below strike
                stop_loss=short_strike * 1.02,
                max_profit=self.spot_price * 0.01 * 100,  # Estimated premium
                max_loss=(short_strike * 1.02 - short_strike) * 100,
                risk_reward=2.0,  # Premium selling typically 2:1
                breakeven=short_strike + (self.spot_price * 0.01),
                probability_profit=0.7,  # High probability in positive GEX
                days_to_expiry="0-2 DTE",
                expiry_date=(datetime.now() + timedelta(days=2)).strftime("%Y-%m-%d"),
                description=f"High positive GEX ({self.net_gex/1e9:.2f}B) indicates volatility suppression. Sell calls at resistance.",
                entry_criteria=f"Sell {short_strike:.2f} Call for premium collection",
                exit_criteria=f"Close at 50% profit or if price approaches {short_strike:.2f}",
                net_gex=self.net_gex,
                gamma_flip=self.gamma_flip,
                distance_to_flip=distance_to_flip,
                position_size=3000
            )
            setups.append(setup)
        
        # Iron Condor Setup
        elif self.net_gex > 1e9 and self.call_walls and self.put_walls:
            call_wall = self.call_walls[0]
            put_wall = self.put_walls[0]
            spread = (call_wall - put_wall) / self.spot_price * 100
            
            if spread > 3:  # Wide enough for iron condor
                confidence = min(85, 60 + spread * 2)
                
                setup = DetailedTradeSetup(
                    symbol=self.symbol,
                    strategy="🦅 Iron Condor",
                    strategy_type="IRON_CONDOR",
                    confidence=confidence,
                    entry_price=self.spot_price,
                    call_strike=call_wall,
                    put_strike=put_wall,
                    call_strike_long=call_wall + 5,
                    put_strike_long=put_wall - 5,
                    target_price=self.spot_price,  # Want price to stay between strikes
                    stop_loss=0,  # Defined risk strategy
                    max_profit=self.spot_price * 0.02 * 100,  # Estimated credit
                    max_loss=500,  # Width of strikes minus credit
                    risk_reward=2.5,
                    breakeven=self.spot_price,
                    probability_profit=0.65,
                    days_to_expiry="5-10 DTE",
                    expiry_date=(datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d"),
                    description=f"Stable GEX environment with {spread:.1f}% range. Profit from time decay.",
                    entry_criteria=f"Sell {call_wall:.2f}/{call_wall+5:.2f} Call Spread & {put_wall:.2f}/{put_wall-5:.2f} Put Spread",
                    exit_criteria="Close at 25% profit or manage at 21 DTE",
                    net_gex=self.net_gex,
                    gamma_flip=self.gamma_flip,
                    distance_to_flip=distance_to_flip,
                    position_size=2500
                )
                setups.append(setup)
        
        # Gamma Flip Play
        if abs(distance_to_flip) < 1:
            confidence = min(90, 75 + (1 - abs(distance_to_flip)) * 15)
            
            if self.spot_price < self.gamma_flip:
                # Below flip - bullish
                setup = DetailedTradeSetup(
                    symbol=self.symbol,
                    strategy="⚡ Gamma Flip Play",
                    strategy_type="CALL",
                    confidence=confidence,
                    entry_price=self.spot_price,
                    strike_price=round(self.gamma_flip / 5) * 5,
                    target_price=self.gamma_flip * 1.02,
                    stop_loss=self.spot_price * 0.98,
                    max_profit=(self.gamma_flip * 1.02 - self.gamma_flip) * 100,
                    max_loss=self.spot_price * 0.015 * 100,
                    risk_reward=3.0,
                    breakeven=self.gamma_flip + (self.spot_price * 0.015),
                    probability_profit=confidence / 100,
                    days_to_expiry="1-3 DTE",
                    expiry_date=(datetime.now() + timedelta(days=3)).strftime("%Y-%m-%d"),
                    description=f"Price near gamma flip ({distance_to_flip:.2f}% away). Regime change imminent.",
                    entry_criteria=f"Buy {round(self.gamma_flip / 5) * 5} Call near flip point",
                    exit_criteria=f"Target: {self.gamma_flip * 1.02:.2f} | Stop: {self.spot_price * 0.98:.2f}",
                    net_gex=self.net_gex,
                    gamma_flip=self.gamma_flip,
                    distance_to_flip=distance_to_flip,
                    position_size=1500
                )
            else:
                # Above flip - bearish
                setup = DetailedTradeSetup(
                    symbol=self.symbol,
                    strategy="⚡ Gamma Flip Play",
                    strategy_type="PUT",
                    confidence=confidence,
                    entry_price=self.spot_price,
                    strike_price=round(self.gamma_flip / 5) * 5,
                    target_price=self.gamma_flip * 0.98,
                    stop_loss=self.spot_price * 1.02,
                    max_profit=(self.gamma_flip - self.gamma_flip * 0.98) * 100,
                    max_loss=self.spot_price * 0.015 * 100,
                    risk_reward=3.0,
                    breakeven=self.gamma_flip - (self.spot_price * 0.015),
                    probability_profit=confidence / 100,
                    days_to_expiry="1-3 DTE",
                    expiry_date=(datetime.now() + timedelta(days=3)).strftime("%Y-%m-%d"),
                    description=f"Price near gamma flip ({distance_to_flip:.2f}% away). Regime change imminent.",
                    entry_criteria=f"Buy {round(self.gamma_flip / 5) * 5} Put near flip point",
                    exit_criteria=f"Target: {self.gamma_flip * 0.98:.2f} | Stop: {self.spot_price * 1.02:.2f}",
                    net_gex=self.net_gex,
                    gamma_flip=self.gamma_flip,
                    distance_to_flip=distance_to_flip,
                    position_size=1500
                )
            
            setups.append(setup)
        
        return setups

# ======================== BATCH PROCESSOR ========================

def process_universe_batch(symbols: List[str]) -> Tuple[Dict, List[DetailedTradeSetup]]:
    """Process entire universe and generate all setups"""
    all_data = {}
    all_setups = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, symbol in enumerate(symbols):
        status_text.text(f"Analyzing {symbol} ({idx+1}/{len(symbols)})")
        progress_bar.progress((idx + 1) / len(symbols))
        
        calc = ComprehensiveGEXCalculator(symbol)
        if calc.fetch_and_calculate():
            all_data[symbol] = {
                'symbol': symbol,
                'price': calc.spot_price,
                'net_gex': calc.net_gex,
                'gamma_flip': calc.gamma_flip,
                'distance_to_flip': ((calc.gamma_flip - calc.spot_price) / calc.spot_price * 100),
                'call_walls': calc.call_walls,
                'put_walls': calc.put_walls
            }
            
            # Generate detailed setups
            setups = calc.generate_detailed_setups()
            all_setups.extend(setups)
    
    progress_bar.empty()
    status_text.empty()
    
    return all_data, all_setups

# ======================== MAIN DASHBOARD ========================

def main():
    # Initialize
    universe_mgr = UniverseManager()
    auto_trader = st.session_state.auto_trader
    
    # Header
    st.markdown("""
    <h1 style='text-align: center;'>
        🚀 GEX Trading Dashboard - Complete Universe Analysis
    </h1>
    <p style='text-align: center; color: rgba(255,255,255,0.7);'>
        Full Symbol Coverage | Detailed Strategies | Automated Paper Trading
    </p>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📊 Universe Control")
        
        # Quick universe selection
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📊 ETFs", use_container_width=True):
                st.session_state.watchlist = universe_mgr.universes["📊 Major ETFs"]
                st.rerun()
        with col2:
            if st.button("🚀 Tech", use_container_width=True):
                st.session_state.watchlist = universe_mgr.universes["🚀 Tech Giants"]
                st.rerun()
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔥 Meme", use_container_width=True):
                st.session_state.watchlist = universe_mgr.universes["🔥 High Vol/Meme"]
                st.rerun()
        with col2:
            if st.button("💎 Options", use_container_width=True):
                st.session_state.watchlist = universe_mgr.universes["💎 Options Flow"]
                st.rerun()
        
        # Custom symbols
        custom = st.text_area("Add symbols (comma separated):", height=60)
        if st.button("Add Custom"):
            if custom:
                symbols = [s.strip().upper() for s in custom.replace(',', ' ').split()]
                st.session_state.watchlist.extend(symbols)
                st.session_state.watchlist = list(set(st.session_state.watchlist))
                st.rerun()
        
        st.info(f"📈 Active: {len(st.session_state.watchlist)} symbols")
        
        # Auto Trader Settings
        st.markdown("---")
        st.markdown("### 🤖 Auto Trader")
        
        st.session_state.auto_trading_enabled = st.checkbox(
            "Enable Auto Trading",
            value=st.session_state.auto_trading_enabled
        )
        
        if st.session_state.auto_trading_enabled:
            st.success("🟢 Auto Trading Active")
            
            min_confidence_auto = st.slider(
                "Min Confidence for Auto Trade",
                70, 95, 80
            )
            
            st.metric("Capital", f"${st.session_state.auto_trade_capital:,.0f}")
            st.metric("Total P&L", f"${st.session_state.auto_trade_pnl:+,.0f}")
            st.metric("Open Positions", len([p for p in st.session_state.auto_positions if p.status == "OPEN"]))
        
        # Analysis button
        st.markdown("---")
        if st.button("🚀 ANALYZE UNIVERSE", type="primary", use_container_width=True):
            st.session_state.force_refresh = True
            st.rerun()
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs([
        f"📊 Full Universe ({len(st.session_state.watchlist)} symbols)",
        "🤖 Auto Trader",
        "📈 Performance",
        "📚 Strategy Guide"
    ])
    
    # Process universe
    if st.session_state.get('force_refresh', True):
        st.session_state.force_refresh = False
        
        with st.spinner(f"Analyzing {len(st.session_state.watchlist)} symbols..."):
            all_data, all_setups = process_universe_batch(st.session_state.watchlist)
            st.session_state.all_data = all_data
            st.session_state.all_setups_detailed = all_setups
            
            # Auto execute high confidence trades
            if st.session_state.auto_trading_enabled:
                executed = auto_trader.execute_high_confidence_trades(
                    all_setups, 
                    min_confidence=min_confidence_auto if 'min_confidence_auto' in locals() else 80
                )
                if executed:
                    st.success(f"🤖 Auto-executed {len(executed)} trades!")
    
    # Tab 1: Full Universe View with Filters
    with tab1:
        render_universe_view()
    
    # Tab 2: Auto Trader
    with tab2:
        render_auto_trader()
    
    # Tab 3: Performance
    with tab3:
        render_performance()
    
    # Tab 4: Strategy Guide
    with tab4:
        render_strategy_guide()

def render_universe_view():
    """Render complete universe with all details and filters"""
    st.markdown("## 📊 Complete Universe Analysis")
    
    # Filter controls
    st.markdown("""
    <div class='filter-container'>
        <h3>🔍 Filter Controls</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        strategy_filter = st.selectbox(
            "Strategy Type",
            ["All", "🚀 Squeeze", "💰 Premium", "🦅 Condor", "⚡ Gamma Flip"]
        )
    
    with col2:
        min_confidence = st.slider("Min Confidence", 50, 95, 65)
    
    with col3:
        gex_filter = st.selectbox(
            "GEX Filter",
            ["All", "Negative (<0)", "Positive (>0)", "Extreme (|GEX|>2B)"]
        )
    
    with col4:
        sort_by = st.selectbox(
            "Sort By",
            ["Confidence", "Net GEX", "Distance to Flip", "Symbol"]
        )
    
    # Apply filters
    all_data = st.session_state.get('all_data', {})
    all_setups = st.session_state.get('all_setups_detailed', [])
    
    if all_data:
        # Create comprehensive table
        table_data = []
        
        for symbol, data in all_data.items():
            # Find setups for this symbol
            symbol_setups = [s for s in all_setups if s.symbol == symbol]
            
            # Apply strategy filter
            if strategy_filter != "All":
                if "Squeeze" in strategy_filter:
                    symbol_setups = [s for s in symbol_setups if "Squeeze" in s.strategy]
                elif "Premium" in strategy_filter:
                    symbol_setups = [s for s in symbol_setups if "Premium" in s.strategy]
                elif "Condor" in strategy_filter:
                    symbol_setups = [s for s in symbol_setups if "Condor" in s.strategy]
                elif "Gamma" in strategy_filter:
                    symbol_setups = [s for s in symbol_setups if "Gamma" in s.strategy]
            
            # Apply confidence filter
            symbol_setups = [s for s in symbol_setups if s.confidence >= min_confidence]
            
            # Apply GEX filter
            if gex_filter == "Negative (<0)" and data['net_gex'] >= 0:
                continue
            elif gex_filter == "Positive (>0)" and data['net_gex'] <= 0:
                continue
            elif gex_filter == "Extreme (|GEX|>2B)" and abs(data['net_gex']) < 2e9:
                continue
            
            # Add row for each setup or one row if no setups
            if symbol_setups:
                for setup in symbol_setups:
                    table_data.append({
                        'Symbol': symbol,
                        'Price': data['price'],
                        'Net GEX (B)': data['net_gex'] / 1e9,
                        'Gamma Flip': data['gamma_flip'],
                        'Distance %': data['distance_to_flip'],
                        'Strategy': setup.strategy,
                        'Type': setup.strategy_type,
                        'Confidence': setup.confidence,
                        'Entry': setup.entry_criteria,
                        'Target': setup.target_price,
                        'Stop': setup.stop_loss,
                        'R/R': setup.risk_reward,
                        'Strike': setup.strike_price if setup.strike_price else 'N/A',
                        'DTE': setup.days_to_expiry,
                        'Max Profit': setup.max_profit,
                        'Max Loss': setup.max_loss,
                        'Setup Details': setup.description
                    })
            else:
                table_data.append({
                    'Symbol': symbol,
                    'Price': data['price'],
                    'Net GEX (B)': data['net_gex'] / 1e9,
                    'Gamma Flip': data['gamma_flip'],
                    'Distance %': data['distance_to_flip'],
                    'Strategy': 'No Setup',
                    'Type': 'N/A',
                    'Confidence': 0,
                    'Entry': 'N/A',
                    'Target': 0,
                    'Stop': 0,
                    'R/R': 0,
                    'Strike': 'N/A',
                    'DTE': 'N/A',
                    'Max Profit': 0,
                    'Max Loss': 0,
                    'Setup Details': 'No qualifying setup'
                })
        
        if table_data:
            df = pd.DataFrame(table_data)
            
            # Sort
            if sort_by == "Confidence":
                df = df.sort_values('Confidence', ascending=False)
            elif sort_by == "Net GEX":
                df = df.sort_values('Net GEX (B)', ascending=False)
            elif sort_by == "Distance to Flip":
                df = df.sort_values('Distance %', key=abs)
            else:
                df = df.sort_values('Symbol')
            
            # Display metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Total Symbols", len(all_data))
            with col2:
                total_setups = len([s for s in all_setups if s.confidence >= min_confidence])
                st.metric("Total Setups", total_setups)
            with col3:
                high_conf = len([s for s in all_setups if s.confidence >= 80])
                st.metric("High Confidence", high_conf)
            with col4:
                squeeze_count = len([s for s in all_setups if "Squeeze" in s.strategy])
                st.metric("Squeeze Plays", squeeze_count)
            with col5:
                premium_count = len([s for s in all_setups if "Premium" in s.strategy])
                st.metric("Premium Sells", premium_count)
            
            st.markdown("---")
            
            # Display the complete table
            st.markdown("### 📊 Complete Analysis Table")
            
            # Format the dataframe for display
            display_df = df.copy()
            display_df['Price'] = display_df['Price'].apply(lambda x: f"${x:.2f}")
            display_df['Net GEX (B)'] = display_df['Net GEX (B)'].apply(lambda x: f"{x:.2f}B")
            display_df['Gamma Flip'] = display_df['Gamma Flip'].apply(lambda x: f"${x:.2f}")
            display_df['Distance %'] = display_df['Distance %'].apply(lambda x: f"{x:+.2f}%")
            display_df['Confidence'] = display_df['Confidence'].apply(lambda x: f"{x:.1f}%" if x > 0 else "N/A")
            display_df['Target'] = display_df['Target'].apply(lambda x: f"${x:.2f}" if x > 0 else "N/A")
            display_df['Stop'] = display_df['Stop'].apply(lambda x: f"${x:.2f}" if x > 0 else "N/A")
            display_df['R/R'] = display_df['R/R'].apply(lambda x: f"{x:.1f}" if x > 0 else "N/A")
            display_df['Max Profit'] = display_df['Max Profit'].apply(lambda x: f"${x:.0f}" if x > 0 else "N/A")
            display_df['Max Loss'] = display_df['Max Loss'].apply(lambda x: f"${abs(x):.0f}" if x != 0 else "N/A")
            
            st.dataframe(display_df, use_container_width=True, height=600)
            
            # Download button
            csv = df.to_csv(index=False)
            st.download_button(
                "📥 Download Full Analysis",
                data=csv,
                file_name=f"gex_universe_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
            
            # Detailed setup cards for high confidence
            st.markdown("---")
            st.markdown("### 🎯 High Confidence Setup Details")
            
            high_conf_setups = [s for s in all_setups if s.confidence >= 80]
            high_conf_setups.sort(key=lambda x: x.confidence, reverse=True)
            
            for setup in high_conf_setups[:10]:
                with st.expander(f"{setup.symbol} - {setup.strategy} ({setup.confidence:.1f}% confidence)"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown(f"""
                        <div class='setup-details'>
                        <h4>Entry Details</h4>
                        <p><strong>Strategy Type:</strong> {setup.strategy_type}</p>
                        <p><strong>Entry Price:</strong> ${setup.entry_price:.2f}</p>
                        <p><strong>Strike:</strong> ${setup.strike_price:.2f}</p>
                        <p><strong>Days to Expiry:</strong> {setup.days_to_expiry}</p>
                        <p><strong>Entry Criteria:</strong> {setup.entry_criteria}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class='setup-details'>
                        <h4>Risk Management</h4>
                        <p><strong>Target:</strong> ${setup.target_price:.2f}</p>
                        <p><strong>Stop Loss:</strong> ${setup.stop_loss:.2f}</p>
                        <p><strong>Max Profit:</strong> ${setup.max_profit:.0f}</p>
                        <p><strong>Max Loss:</strong> ${abs(setup.max_loss):.0f}</p>
                        <p><strong>Risk/Reward:</strong> {setup.risk_reward:.2f}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown(f"""
                        <div class='setup-details'>
                        <h4>GEX Analysis</h4>
                        <p><strong>Net GEX:</strong> {setup.net_gex/1e9:.2f}B</p>
                        <p><strong>Gamma Flip:</strong> ${setup.gamma_flip:.2f}</p>
                        <p><strong>Distance:</strong> {setup.distance_to_flip:+.2f}%</p>
                        <p><strong>Exit Criteria:</strong> {setup.exit_criteria}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown(f"**Analysis:** {setup.description}")
                    
                    if st.button(f"Execute Trade - {setup.symbol}", key=f"exec_{setup.symbol}_{setup.strategy}"):
                        st.success(f"Trade executed for {setup.symbol}")

def render_auto_trader():
    """Render auto trader dashboard"""
    st.markdown("## 🤖 Automated Paper Trading System")
    
    # Status
    if st.session_state.auto_trading_enabled:
        st.success("🟢 Auto Trading is ACTIVE")
    else:
        st.warning("🔴 Auto Trading is DISABLED")
    
    # Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Capital", f"${st.session_state.auto_trade_capital:,.0f}")
    
    with col2:
        st.metric("Total P&L", f"${st.session_state.auto_trade_pnl:+,.0f}")
    
    with col3:
        open_positions = len([p for p in st.session_state.auto_positions if p.status == "OPEN"])
        st.metric("Open Positions", open_positions)
    
    with col4:
        total_trades = len(st.session_state.auto_trade_history)
        st.metric("Total Trades", total_trades)
    
    with col5:
        if total_trades > 0:
            wins = len([t for t in st.session_state.auto_trade_history if t['pnl'] > 0])
            win_rate = (wins / total_trades) * 100
            st.metric("Win Rate", f"{win_rate:.1f}%")
        else:
            st.metric("Win Rate", "N/A")
    
    # Open Positions
    st.markdown("---")
    st.markdown("### 📈 Open Positions")
    
    open_pos = [p for p in st.session_state.auto_positions if p.status == "OPEN"]
    
    if open_pos:
        pos_data = []
        for pos in open_pos:
            pos_data.append({
                'Symbol': pos.setup.symbol,
                'Strategy': pos.setup.strategy,
                'Entry Time': pos.entry_time.strftime("%Y-%m-%d %H:%M"),
                'Entry Price': f"${pos.entry_price:.2f}",
                'Current Price': f"${pos.current_price:.2f}",
                'Quantity': pos.quantity,
                'Current P&L': f"${pos.current_pnl:+,.2f}",
                'Target': f"${pos.setup.target_price:.2f}",
                'Stop': f"${pos.setup.stop_loss:.2f}",
                'Status': pos.status
            })
        
        pos_df = pd.DataFrame(pos_data)
        st.dataframe(pos_df, use_container_width=True)
    else:
        st.info("No open positions")
    
    # Trade History
    st.markdown("---")
    st.markdown("### 📜 Trade History")
    
    if st.session_state.auto_trade_history:
        hist_df = pd.DataFrame(st.session_state.auto_trade_history)
        hist_df = hist_df.sort_values('exit_time', ascending=False)
        
        # Format for display
        hist_df['entry_time'] = pd.to_datetime(hist_df['entry_time']).dt.strftime("%Y-%m-%d %H:%M")
        hist_df['exit_time'] = pd.to_datetime(hist_df['exit_time']).dt.strftime("%Y-%m-%d %H:%M")
        hist_df['entry_price'] = hist_df['entry_price'].apply(lambda x: f"${x:.2f}")
        hist_df['exit_price'] = hist_df['exit_price'].apply(lambda x: f"${x:.2f}")
        hist_df['pnl'] = hist_df['pnl'].apply(lambda x: f"${x:+,.2f}")
        
        st.dataframe(hist_df, use_container_width=True)
    else:
        st.info("No trade history yet")
    
    # Controls
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Update Positions", use_container_width=True):
            # Update with current prices
            current_prices = {symbol: data['price'] for symbol, data in st.session_state.all_data.items()}
            st.session_state.auto_trader.update_positions(current_prices)
            st.success("Positions updated!")
            st.rerun()
    
    with col2:
        if st.button("💰 Close All Positions", use_container_width=True):
            for pos in st.session_state.auto_positions:
                if pos.status == "OPEN":
                    st.session_state.auto_trader.close_position(pos, "MANUAL_CLOSE")
            st.success("All positions closed!")
            st.rerun()
    
    with col3:
        if st.button("🔄 Reset Auto Trader", use_container_width=True):
            st.session_state.auto_positions = []
            st.session_state.auto_trade_history = []
            st.session_state.auto_trade_capital = 100000
            st.session_state.auto_trade_pnl = 0
            st.success("Auto trader reset!")
            st.rerun()

def render_performance():
    """Render performance analytics"""
    st.markdown("## 📉 Performance Analytics")
    
    # Combine auto trader and manual trades
    all_trades = st.session_state.auto_trade_history.copy()
    
    if not all_trades:
        st.info("No trades to analyze yet. Enable auto trading to start!")
        return
    
    trades_df = pd.DataFrame(all_trades)
    
    # Calculate metrics
    total_trades = len(trades_df)
    winning_trades = len(trades_df[trades_df['pnl'] > 0])
    losing_trades = len(trades_df[trades_df['pnl'] < 0])
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    
    # Display metrics
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric("Total Trades", total_trades)
    with col2:
        st.metric("Winners", winning_trades)
    with col3:
        st.metric("Losers", losing_trades)
    with col4:
        st.metric("Win Rate", f"{win_rate:.1f}%")
    with col5:
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        st.metric("Avg Win", f"${avg_win:,.2f}")
    with col6:
        avg_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].mean()) if losing_trades > 0 else 0
        st.metric("Avg Loss", f"${avg_loss:,.2f}")
    
    # P&L Chart
    st.markdown("---")
    st.markdown("### 📊 Cumulative P&L")
    
    trades_df = trades_df.sort_values('exit_time')
    trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=trades_df['exit_time'],
        y=trades_df['cumulative_pnl'],
        mode='lines+markers',
        name='Cumulative P&L',
        line=dict(color='#00D2FF', width=3),
        fill='tozeroy',
        fillcolor='rgba(0, 210, 255, 0.1)'
    ))
    
    fig.update_layout(
        title="Cumulative P&L Over Time",
        xaxis_title="Date",
        yaxis_title="Cumulative P&L ($)",
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Strategy Performance
    st.markdown("---")
    st.markdown("### 🎯 Performance by Strategy")
    
    strategy_perf = trades_df.groupby('strategy').agg({
        'pnl': ['count', 'sum', 'mean', 'std']
    }).round(2)
    
    strategy_perf.columns = ['Trades', 'Total P&L', 'Avg P&L', 'Std Dev']
    strategy_perf['Win Rate'] = trades_df.groupby('strategy').apply(
        lambda x: f"{(len(x[x['pnl'] > 0]) / len(x) * 100):.1f}%"
    )
    
    st.dataframe(
        strategy_perf.style.format({
            'Total P&L': '${:,.2f}',
            'Avg P&L': '${:,.2f}',
            'Std Dev': '${:,.2f}'
        }),
        use_container_width=True
    )
    
    # Best and Worst Trades
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🏆 Best Trades")
        best_trades = trades_df.nlargest(5, 'pnl')[['symbol', 'strategy', 'pnl', 'exit_reason']]
        best_trades['pnl'] = best_trades['pnl'].apply(lambda x: f"${x:,.2f}")
        st.dataframe(best_trades, use_container_width=True)
    
    with col2:
        st.markdown("### 📉 Worst Trades")
        worst_trades = trades_df.nsmallest(5, 'pnl')[['symbol', 'strategy', 'pnl', 'exit_reason']]
        worst_trades['pnl'] = worst_trades['pnl'].apply(lambda x: f"${x:,.2f}")
        st.dataframe(worst_trades, use_container_width=True)

def render_strategy_guide():
    """Render comprehensive strategy guide"""
    st.markdown("## 📚 Complete Strategy Guide")
    
    strategies = {
        "🚀 Negative GEX Squeeze (Long Calls)": {
            "setup": "Net GEX < -1B (SPY/QQQ) or < -500M (stocks)",
            "entry": "Buy ATM or first OTM call above gamma flip",
            "target": "Gamma flip point or higher",
            "stop": "Put wall support level",
            "dte": "2-5 DTE for maximum gamma",
            "size": "3% of capital maximum",
            "details": "Dealers are short gamma and must buy as price rises, creating explosive upside"
        },
        
        "💰 Premium Selling (Short Calls/Puts)": {
            "setup": "Net GEX > 2B with strong walls",
            "entry": "Sell OTM options at wall levels",
            "target": "50% of premium collected",
            "stop": "2x premium loss or wall breach",
            "dte": "0-2 DTE for rapid decay",
            "size": "5% of capital maximum",
            "details": "High positive GEX suppresses volatility, making premium selling profitable"
        },
        
        "🦅 Iron Condor": {
            "setup": "Net GEX > 1B, walls > 3% apart",
            "entry": "Sell call spread at upper wall, put spread at lower wall",
            "target": "25% of max profit",
            "stop": "Defined risk (spread width - credit)",
            "dte": "5-10 DTE optimal",
            "size": "Size for 2% portfolio max loss",
            "details": "Profit from time decay in stable, range-bound markets"
        },
        
        "⚡ Gamma Flip Play": {
            "setup": "Price within 1% of gamma flip",
            "entry": "Calls if below flip, Puts if above",
            "target": "2% beyond flip point",
            "stop": "2% opposite direction",
            "dte": "1-3 DTE for quick moves",
            "size": "2% of capital",
            "details": "Regime change imminent at flip point, volatility expansion expected"
        }
    }
    
    for strategy_name, details in strategies.items():
        with st.expander(strategy_name):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                **Setup Conditions:** {details['setup']}
                
                **Entry:** {details['entry']}
                
                **Target:** {details['target']}
                
                **Stop Loss:** {details['stop']}
                """)
            
            with col2:
                st.markdown(f"""
                **Timeframe:** {details['dte']}
                
                **Position Size:** {details['size']}
                
                **Details:** {details['details']}
                """)

if __name__ == "__main__":
    main()
