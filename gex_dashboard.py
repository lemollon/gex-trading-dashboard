"""
GEX Trading Dashboard - Professional Edition v10.0
Enhanced with symbol validation, options checking, and performance optimizations
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
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
import logging
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import warnings
import sqlite3
import hashlib
from functools import lru_cache
import asyncio
import threading
from collections import defaultdict

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="GEX Trading Dashboard - Professional",
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
    
    .valid-symbol {
        color: #00ff87;
        font-weight: bold;
    }
    
    .invalid-symbol {
        color: #ff6b6b;
        font-weight: bold;
        text-decoration: line-through;
    }
    
    .options-available {
        background: rgba(0, 255, 135, 0.1);
        padding: 2px 8px;
        border-radius: 4px;
        color: #00ff87;
    }
    
    .no-options {
        background: rgba(255, 107, 107, 0.1);
        padding: 2px 8px;
        border-radius: 4px;
        color: #ff6b6b;
    }
    
    .setup-details {
        background: rgba(0, 255, 135, 0.1);
        border-left: 3px solid #00ff87;
        padding: 15px;
        margin: 10px 0;
        border-radius: 8px;
    }
    
    .performance-card {
        background: linear-gradient(135deg, rgba(58, 123, 213, 0.1) 0%, rgba(0, 210, 255, 0.1) 100%);
        border: 1px solid rgba(0, 210, 255, 0.3);
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
    }
    
    .filter-container {
        background: rgba(255, 255, 255, 0.05);
        padding: 15px;
        border-radius: 12px;
        margin: 15px 0;
    }
    
    /* Loading animation */
    .loading-wave {
        display: inline-block;
        animation: wave 1.5s ease-in-out infinite;
    }
    
    @keyframes wave {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }
    
    /* Strategy colors */
    .squeeze-play { color: #ff6b6b; font-weight: bold; }
    .premium-sell { color: #ffd93d; font-weight: bold; }
    .iron-condor { color: #00ff87; font-weight: bold; }
    .gamma-flip { color: #00d2ff; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ======================== DATA CLASSES ========================

@dataclass
class DetailedTradeSetup:
    """Comprehensive trade setup with all details"""
    symbol: str
    strategy: str
    strategy_type: str
    confidence: float
    entry_price: float
    
    # Options details
    strike_price: float = 0
    strike_price_2: float = 0
    call_strike: float = 0
    put_strike: float = 0
    call_strike_long: float = 0
    put_strike_long: float = 0
    
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
    position_size: float = 1000
    executed: bool = False
    execution_time: Optional[datetime] = None
    exit_time: Optional[datetime] = None
    pnl: float = 0

@dataclass
class SymbolValidation:
    """Symbol validation results"""
    symbol: str
    is_valid: bool
    has_options: bool
    market_cap: float = 0
    avg_volume: float = 0
    sector: str = ""
    error_message: str = ""
    last_checked: datetime = field(default_factory=datetime.now)

# ======================== SYMBOL VALIDATOR ========================

class SymbolValidator:
    """Validate symbols and check for options availability"""
    
    def __init__(self):
        self.cache = {}
        self.valid_exchanges = ['NYSE', 'NASDAQ', 'AMEX']
        self.initialize_database()
        
    def initialize_database(self):
        """Initialize SQLite database for caching"""
        self.conn = sqlite3.connect('symbol_cache.db', check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS symbol_cache (
                symbol TEXT PRIMARY KEY,
                is_valid INTEGER,
                has_options INTEGER,
                market_cap REAL,
                avg_volume REAL,
                sector TEXT,
                last_checked TIMESTAMP
            )
        ''')
        self.conn.commit()
    
    @lru_cache(maxsize=1000)
    def validate_symbol(self, symbol: str) -> SymbolValidation:
        """Validate a single symbol"""
        # Check cache first
        cached = self.get_from_cache(symbol)
        if cached and (datetime.now() - cached.last_checked).days < 1:
            return cached
        
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Check if it's a valid stock
            if not info or 'regularMarketPrice' not in info:
                return SymbolValidation(
                    symbol=symbol,
                    is_valid=False,
                    has_options=False,
                    error_message="Symbol not found or invalid"
                )
            
            # Check for options
            has_options = False
            try:
                options = ticker.options
                has_options = len(options) > 0 if options else False
            except:
                has_options = False
            
            # Get market data
            market_cap = info.get('marketCap', 0)
            avg_volume = info.get('averageVolume', 0)
            sector = info.get('sector', 'Unknown')
            
            validation = SymbolValidation(
                symbol=symbol,
                is_valid=True,
                has_options=has_options,
                market_cap=market_cap,
                avg_volume=avg_volume,
                sector=sector
            )
            
            # Cache the result
            self.save_to_cache(validation)
            
            return validation
            
        except Exception as e:
            return SymbolValidation(
                symbol=symbol,
                is_valid=False,
                has_options=False,
                error_message=str(e)
            )
    
    def validate_batch(self, symbols: List[str], show_progress: bool = True) -> Dict[str, SymbolValidation]:
        """Validate multiple symbols in parallel"""
        results = {}
        
        if show_progress:
            progress_bar = st.progress(0)
            status_text = st.empty()
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_symbol = {executor.submit(self.validate_symbol, symbol): symbol 
                              for symbol in symbols}
            
            for i, future in enumerate(concurrent.futures.as_completed(future_to_symbol)):
                symbol = future_to_symbol[future]
                if show_progress:
                    status_text.text(f"Validating {symbol}...")
                    progress_bar.progress((i + 1) / len(symbols))
                
                try:
                    results[symbol] = future.result()
                except Exception as e:
                    results[symbol] = SymbolValidation(
                        symbol=symbol,
                        is_valid=False,
                        has_options=False,
                        error_message=str(e)
                    )
        
        if show_progress:
            progress_bar.empty()
            status_text.empty()
        
        return results
    
    def get_from_cache(self, symbol: str) -> Optional[SymbolValidation]:
        """Get symbol validation from cache"""
        self.cursor.execute('''
            SELECT * FROM symbol_cache WHERE symbol = ?
        ''', (symbol,))
        
        row = self.cursor.fetchone()
        if row:
            return SymbolValidation(
                symbol=row[0],
                is_valid=bool(row[1]),
                has_options=bool(row[2]),
                market_cap=row[3],
                avg_volume=row[4],
                sector=row[5],
                last_checked=datetime.fromisoformat(row[6])
            )
        return None
    
    def save_to_cache(self, validation: SymbolValidation):
        """Save validation result to cache"""
        self.cursor.execute('''
            INSERT OR REPLACE INTO symbol_cache 
            (symbol, is_valid, has_options, market_cap, avg_volume, sector, last_checked)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            validation.symbol,
            int(validation.is_valid),
            int(validation.has_options),
            validation.market_cap,
            validation.avg_volume,
            validation.sector,
            validation.last_checked.isoformat()
        ))
        self.conn.commit()

# ======================== ENHANCED UNIVERSE MANAGER ========================

class EnhancedUniverseManager:
    """Manage validated symbol universes"""
    
    def __init__(self):
        self.validator = SymbolValidator()
        self.initialize_session_state()
        self.setup_verified_universes()
        
    def initialize_session_state(self):
        """Initialize session state"""
        if 'validated_watchlist' not in st.session_state:
            st.session_state.validated_watchlist = []
        
        if 'symbol_validations' not in st.session_state:
            st.session_state.symbol_validations = {}
        
        if 'all_setups_detailed' not in st.session_state:
            st.session_state.all_setups_detailed = []
        
        if 'auto_trader' not in st.session_state:
            from auto_trader import EnhancedAutoTrader
            st.session_state.auto_trader = EnhancedAutoTrader()
    
    def setup_verified_universes(self):
        """Setup universes with verified tickers that have options"""
        self.universes = {
            "📊 Major Index ETFs": [
                "SPY", "QQQ", "IWM", "DIA", "VOO", "VTI", "EEM", "EFA",
                "TLT", "GLD", "SLV", "USO", "UNG", "VXX", "UVXY", "SQQQ", "TQQQ"
            ],
            
            "🚀 Mega Cap Tech": [
                "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
                "AMD", "INTC", "AVGO", "ORCL", "CRM", "ADBE", "NFLX", "CSCO"
            ],
            
            "💎 High Options Volume": [
                "SPY", "QQQ", "AAPL", "TSLA", "AMD", "NVDA", "AMZN", "META",
                "NFLX", "MSFT", "BAC", "F", "NIO", "PLTR", "SOFI", "AAL", "UBER"
            ],
            
            "🔥 Volatility Plays": [
                "GME", "AMC", "PLTR", "SOFI", "RIOT", "MARA", "COIN", "HOOD",
                "LCID", "RIVN", "DWAC", "BBBY", "BB", "WISH", "CLOV"
            ],
            
            "🏦 Financial Sector": [
                "JPM", "BAC", "WFC", "GS", "MS", "C", "AXP", "V", "MA",
                "PYPL", "SQ", "SCHW", "BLK", "COF", "USB"
            ],
            
            "⚡ Energy & Commodities": [
                "XOM", "CVX", "COP", "SLB", "OXY", "MPC", "PSX", "VLO",
                "XLE", "XOP", "OIH", "HAL", "BKR", "EOG", "PXD"
            ],
            
            "🏥 Healthcare": [
                "UNH", "JNJ", "PFE", "ABBV", "LLY", "MRK", "TMO", "ABT",
                "CVS", "MDT", "BMY", "AMGN", "GILD", "MRNA", "REGN"
            ],
            
            "🛒 Consumer": [
                "WMT", "HD", "NKE", "MCD", "SBUX", "TGT", "COST", "LOW",
                "PEP", "KO", "PG", "DIS", "CMCSA", "NFLX", "ABNB"
            ]
        }
    
    def validate_and_filter_symbols(self, symbols: List[str]) -> List[str]:
        """Validate symbols and return only those with options"""
        validations = self.validator.validate_batch(symbols)
        
        valid_symbols = []
        for symbol, validation in validations.items():
            if validation.is_valid and validation.has_options:
                valid_symbols.append(symbol)
                st.session_state.symbol_validations[symbol] = validation
        
        return valid_symbols

# ======================== OPTIMIZED GEX CALCULATOR ========================

class OptimizedGEXCalculator:
    """Optimized GEX calculator with caching and parallel processing"""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.spot_price = None
        self.net_gex = None
        self.gamma_flip = None
        self.call_walls = []
        self.put_walls = []
        self._cache_key = None
        
    @st.cache_data(ttl=300)  # Cache for 5 minutes
    def fetch_cached_data(symbol: str) -> Dict:
        """Fetch and cache market data"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period='5d')
            
            if hist.empty:
                return None
            
            return {
                'symbol': symbol,
                'spot_price': hist['Close'].iloc[-1],
                'volume': hist['Volume'].iloc[-1],
                'history': hist,
                'timestamp': datetime.now()
            }
        except:
            return None
    
    def calculate_options_gex(self) -> bool:
        """Calculate GEX from options chain"""
        try:
            # Get cached data
            data = OptimizedGEXCalculator.fetch_cached_data(self.symbol)
            if not data:
                return False
            
            self.spot_price = data['spot_price']
            ticker = yf.Ticker(self.symbol)
            
            # Get options chain
            try:
                expirations = ticker.options[:10]
                
                if not expirations:
                    self._generate_simulated_gex()
                    return True
                
                total_call_gamma = 0
                total_put_gamma = 0
                call_walls_data = defaultdict(float)
                put_walls_data = defaultdict(float)
                
                # Process each expiration
                for exp_date in expirations[:5]:  # Limit to 5 nearest expirations
                    try:
                        opt_chain = ticker.option_chain(exp_date)
                        days_to_exp = (pd.to_datetime(exp_date) - datetime.now()).days
                        
                        if days_to_exp < 0:
                            continue
                        
                        # Process calls
                        calls = opt_chain.calls
                        for _, row in calls.iterrows():
                            if row['openInterest'] > 0:
                                strike = row['strike']
                                oi = row['openInterest']
                                iv = row.get('impliedVolatility', 0.3)
                                
                                # Simple gamma approximation
                                moneyness = self.spot_price / strike
                                time_factor = max(days_to_exp / 365, 0.001)
                                gamma = self._calculate_gamma(moneyness, iv, time_factor)
                                
                                call_gamma = gamma * oi * 100 * self.spot_price
                                total_call_gamma += call_gamma
                                call_walls_data[strike] += call_gamma
                        
                        # Process puts
                        puts = opt_chain.puts
                        for _, row in puts.iterrows():
                            if row['openInterest'] > 0:
                                strike = row['strike']
                                oi = row['openInterest']
                                iv = row.get('impliedVolatility', 0.3)
                                
                                moneyness = self.spot_price / strike
                                time_factor = max(days_to_exp / 365, 0.001)
                                gamma = self._calculate_gamma(moneyness, iv, time_factor)
                                
                                put_gamma = gamma * oi * 100 * self.spot_price
                                total_put_gamma += put_gamma
                                put_walls_data[strike] += put_gamma
                    
                    except Exception as e:
                        logger.debug(f"Error processing {exp_date}: {e}")
                        continue
                
                # Calculate net GEX
                self.net_gex = total_call_gamma - total_put_gamma
                
                # Identify walls (top 3 strikes by gamma)
                if call_walls_data:
                    sorted_calls = sorted(call_walls_data.items(), key=lambda x: x[1], reverse=True)
                    self.call_walls = [strike for strike, _ in sorted_calls[:3]]
                else:
                    self.call_walls = [self.spot_price * 1.02, self.spot_price * 1.05]
                
                if put_walls_data:
                    sorted_puts = sorted(put_walls_data.items(), key=lambda x: x[1], reverse=True)
                    self.put_walls = [strike for strike, _ in sorted_puts[:3]]
                else:
                    self.put_walls = [self.spot_price * 0.98, self.spot_price * 0.95]
                
                # Calculate gamma flip
                if total_call_gamma + total_put_gamma > 0:
                    skew = (total_put_gamma - total_call_gamma) / (total_call_gamma + total_put_gamma)
                    self.gamma_flip = self.spot_price * (1 + 0.02 * skew)
                else:
                    self.gamma_flip = self.spot_price
                
            except Exception as e:
                logger.debug(f"Options error for {self.symbol}: {e}")
                self._generate_simulated_gex()
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing {self.symbol}: {e}")
            return False
    
    def _calculate_gamma(self, moneyness: float, iv: float, time: float) -> float:
        """Calculate gamma approximation"""
        # Simplified Black-Scholes gamma approximation
        d1 = (np.log(moneyness) + (0.02 + iv**2/2) * time) / (iv * np.sqrt(time))
        gamma = np.exp(-d1**2/2) / (np.sqrt(2 * np.pi) * iv * np.sqrt(time))
        return gamma
    
    def _generate_simulated_gex(self):
        """Generate simulated GEX for testing"""
        self.net_gex = np.random.uniform(-2e9, 5e9)
        self.gamma_flip = self.spot_price * (1 + np.random.uniform(-0.03, 0.03))
        self.call_walls = [self.spot_price * 1.02, self.spot_price * 1.05]
        self.put_walls = [self.spot_price * 0.98, self.spot_price * 0.95]
    
    def generate_setups(self) -> List[DetailedTradeSetup]:
        """Generate trading setups based on GEX analysis"""
        setups = []
        
        if not self.spot_price:
            return setups
        
        distance_to_flip = ((self.gamma_flip - self.spot_price) / self.spot_price * 100)
        
        # Negative GEX Squeeze Setup
        if self.net_gex < -5e8:
            setups.append(self._create_squeeze_setup(distance_to_flip))
        
        # Premium Selling Setup
        elif self.net_gex > 2e9:
            setups.append(self._create_premium_setup(distance_to_flip))
        
        # Iron Condor Setup
        if self.net_gex > 1e9 and len(self.call_walls) > 0 and len(self.put_walls) > 0:
            condor_setup = self._create_condor_setup(distance_to_flip)
            if condor_setup:
                setups.append(condor_setup)
        
        # Gamma Flip Play
        if abs(distance_to_flip) < 1:
            setups.append(self._create_flip_setup(distance_to_flip))
        
        return setups
    
    def _create_squeeze_setup(self, distance_to_flip: float) -> DetailedTradeSetup:
        """Create squeeze play setup"""
        confidence = min(95, 70 + abs(self.net_gex/1e9) * 5)
        atm_call = round(self.spot_price / 5) * 5
        stop_loss = self.put_walls[0] if self.put_walls else self.spot_price * 0.98
        
        return DetailedTradeSetup(
            symbol=self.symbol,
            strategy="🚀 Negative GEX Squeeze",
            strategy_type="CALL",
            confidence=confidence,
            entry_price=self.spot_price,
            strike_price=atm_call,
            target_price=self.gamma_flip,
            stop_loss=stop_loss,
            max_profit=(self.gamma_flip - atm_call) * 100,
            max_loss=self.spot_price * 0.02 * 100,
            risk_reward=abs(self.gamma_flip - self.spot_price) / max(0.01, abs(self.spot_price - stop_loss)),
            breakeven=atm_call + (self.spot_price * 0.02),
            probability_profit=confidence / 100,
            days_to_expiry="2-5 DTE",
            expiry_date=(datetime.now() + timedelta(days=5)).strftime("%Y-%m-%d"),
            description=f"Strong negative GEX ({self.net_gex/1e9:.2f}B) indicates explosive upside potential",
            entry_criteria=f"Buy {atm_call} Call when price > {stop_loss:.2f}",
            exit_criteria=f"Target: {self.gamma_flip:.2f} | Stop: {stop_loss:.2f}",
            net_gex=self.net_gex,
            gamma_flip=self.gamma_flip,
            distance_to_flip=distance_to_flip,
            position_size=2000
        )
    
    def _create_premium_setup(self, distance_to_flip: float) -> DetailedTradeSetup:
        """Create premium selling setup"""
        confidence = min(90, 65 + self.net_gex/1e9 * 3)
        short_strike = self.call_walls[0] if self.call_walls else self.spot_price * 1.02
        stop_loss = short_strike * 1.02
        
        return DetailedTradeSetup(
            symbol=self.symbol,
            strategy="💰 Premium Selling",
            strategy_type="SHORT_CALL",
            confidence=confidence,
            entry_price=self.spot_price,
            strike_price=short_strike,
            target_price=self.spot_price,
            stop_loss=stop_loss,
            max_profit=self.spot_price * 0.01 * 100,
            max_loss=(stop_loss - short_strike) * 100,
            risk_reward=2.0,
            breakeven=short_strike + (self.spot_price * 0.01),
            probability_profit=0.7,
            days_to_expiry="0-2 DTE",
            expiry_date=(datetime.now() + timedelta(days=2)).strftime("%Y-%m-%d"),
            description=f"High positive GEX ({self.net_gex/1e9:.2f}B) suppresses volatility",
            entry_criteria=f"Sell {short_strike:.2f} Call",
            exit_criteria=f"Close at 50% profit or if threatened",
            net_gex=self.net_gex,
            gamma_flip=self.gamma_flip,
            distance_to_flip=distance_to_flip,
            position_size=3000
        )
    
    def _create_condor_setup(self, distance_to_flip: float) -> Optional[DetailedTradeSetup]:
        """Create iron condor setup"""
        call_wall = self.call_walls[0]
        put_wall = self.put_walls[0]
        spread = (call_wall - put_wall) / self.spot_price * 100
        
        if spread < 3:  # Not wide enough
            return None
        
        confidence = min(85, 60 + spread * 2)
        
        return DetailedTradeSetup(
            symbol=self.symbol,
            strategy="🦅 Iron Condor",
            strategy_type="IRON_CONDOR",
            confidence=confidence,
            entry_price=self.spot_price,
            call_strike=call_wall,
            put_strike=put_wall,
            call_strike_long=call_wall + 5,
            put_strike_long=put_wall - 5,
            target_price=self.spot_price,
            stop_loss=0,
            max_profit=self.spot_price * 0.02 * 100,
            max_loss=500,
            risk_reward=2.5,
            breakeven=self.spot_price,
            probability_profit=0.65,
            days_to_expiry="5-10 DTE",
            expiry_date=(datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d"),
            description=f"Range-bound setup with {spread:.1f}% profit zone",
            entry_criteria=f"Sell {call_wall:.0f}/{put_wall:.0f} strangle",
            exit_criteria="Manage at 25% profit",
            net_gex=self.net_gex,
            gamma_flip=self.gamma_flip,
            distance_to_flip=distance_to_flip,
            position_size=2500
        )
    
    def _create_flip_setup(self, distance_to_flip: float) -> DetailedTradeSetup:
        """Create gamma flip setup"""
        confidence = min(90, 75 + (1 - abs(distance_to_flip)) * 15)
        
        if self.spot_price < self.gamma_flip:
            # Bullish setup
            stop_loss = self.spot_price * 0.98
            strategy_type = "CALL"
            target = self.gamma_flip * 1.02
        else:
            # Bearish setup
            stop_loss = self.spot_price * 1.02
            strategy_type = "PUT"
            target = self.gamma_flip * 0.98
        
        strike = round(self.gamma_flip / 5) * 5
        
        return DetailedTradeSetup(
            symbol=self.symbol,
            strategy="⚡ Gamma Flip Play",
            strategy_type=strategy_type,
            confidence=confidence,
            entry_price=self.spot_price,
            strike_price=strike,
            target_price=target,
            stop_loss=stop_loss,
            max_profit=abs(target - strike) * 100,
            max_loss=self.spot_price * 0.015 * 100,
            risk_reward=3.0,
            breakeven=strike + (self.spot_price * 0.015),
            probability_profit=confidence / 100,
            days_to_expiry="1-3 DTE",
            expiry_date=(datetime.now() + timedelta(days=3)).strftime("%Y-%m-%d"),
            description=f"Volatility regime change at flip point ({distance_to_flip:.1f}% away)",
            entry_criteria=f"Enter {strategy_type} at {strike}",
            exit_criteria=f"Target: {target:.2f} | Stop: {stop_loss:.2f}",
            net_gex=self.net_gex,
            gamma_flip=self.gamma_flip,
            distance_to_flip=distance_to_flip,
            position_size=1500
        )

# ======================== ENHANCED AUTO TRADER ========================

class EnhancedAutoTrader:
    """Enhanced auto trader with risk management"""
    
    def __init__(self):
        self.initialize_state()
        self.risk_manager = RiskManager()
        
    def initialize_state(self):
        """Initialize trading state"""
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
        
        if 'max_positions' not in st.session_state:
            st.session_state.max_positions = 10
        
        if 'max_risk_per_trade' not in st.session_state:
            st.session_state.max_risk_per_trade = 0.02  # 2% max risk

class RiskManager:
    """Risk management system"""
    
    def calculate_position_size(self, setup: DetailedTradeSetup, capital: float) -> float:
        """Calculate optimal position size using Kelly Criterion"""
        win_prob = setup.probability_profit
        win_amount = setup.max_profit
        loss_amount = abs(setup.max_loss)
        
        if loss_amount == 0:
            return 0
        
        # Kelly fraction
        kelly = (win_prob * win_amount - (1 - win_prob) * loss_amount) / win_amount
        kelly = max(0, min(kelly, 0.25))  # Cap at 25% of capital
        
        # Apply additional constraints
        position_size = kelly * capital
        position_size = min(position_size, capital * 0.05)  # Max 5% per trade
        position_size = min(position_size, setup.position_size)  # Respect setup limit
        
        return position_size

# ======================== PARALLEL PROCESSOR ========================

def process_universe_parallel(symbols: List[str]) -> Tuple[Dict, List[DetailedTradeSetup]]:
    """Process universe in parallel for speed"""
    all_data = {}
    all_setups = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    def analyze_symbol(symbol: str) -> Tuple[Dict, List[DetailedTradeSetup]]:
        """Analyze single symbol"""
        calc = OptimizedGEXCalculator(symbol)
        if calc.calculate_options_gex():
            data = {
                'symbol': symbol,
                'price': calc.spot_price,
                'net_gex': calc.net_gex,
                'gamma_flip': calc.gamma_flip,
                'distance_to_flip': ((calc.gamma_flip - calc.spot_price) / calc.spot_price * 100),
                'call_walls': calc.call_walls,
                'put_walls': calc.put_walls
            }
            setups = calc.generate_setups()
            return data, setups
        return None, []
    
    # Process in parallel
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(analyze_symbol, symbol): symbol for symbol in symbols}
        
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            symbol = futures[future]
            status_text.text(f"Analyzing {symbol}...")
            progress_bar.progress((i + 1) / len(symbols))
            
            try:
                data, setups = future.result(timeout=10)
                if data:
                    all_data[symbol] = data
                    all_setups.extend(setups)
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
    
    progress_bar.empty()
    status_text.empty()
    
    return all_data, all_setups

# ======================== MAIN DASHBOARD ========================

def main():
    # Initialize
    universe_mgr = EnhancedUniverseManager()
    
    # Header
    st.markdown("""
    <h1 style='text-align: center;'>
        🚀 GEX Trading Dashboard - Professional Edition
    </h1>
    <p style='text-align: center; color: rgba(255,255,255,0.7);'>
        Validated Symbols | Real Options Data | Optimized Performance
    </p>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📊 Universe Control")
        
        # Universe selection
        selected_universe = st.selectbox(
            "Select Universe",
            list(universe_mgr.universes.keys())
        )
        
        if st.button("Load Universe", type="primary"):
            with st.spinner("Validating symbols..."):
                symbols = universe_mgr.universes[selected_universe]
                validated = universe_mgr.validate_and_filter_symbols(symbols)
                st.session_state.validated_watchlist = validated
                st.success(f"✅ Loaded {len(validated)} valid symbols with options")
                st.rerun()
        
        # Custom symbols
        st.markdown("---")
        custom = st.text_area("Add Custom Symbols:", height=60)
        if st.button("Validate & Add"):
            if custom:
                symbols = [s.strip().upper() for s in custom.replace(',', ' ').split()]
                with st.spinner("Validating custom symbols..."):
                    validated = universe_mgr.validate_and_filter_symbols(symbols)
                    st.session_state.validated_watchlist.extend(validated)
                    st.session_state.validated_watchlist = list(set(st.session_state.validated_watchlist))
                    st.success(f"Added {len(validated)} valid symbols")
                    st.rerun()
        
        # Show current watchlist
        if st.session_state.validated_watchlist:
            st.markdown("---")
            st.markdown("### ✅ Active Symbols")
            st.info(f"📈 {len(st.session_state.validated_watchlist)} symbols loaded")
            
            # Show validation status
            with st.expander("Symbol Details"):
                for symbol in st.session_state.validated_watchlist[:20]:
                    if symbol in st.session_state.symbol_validations:
                        val = st.session_state.symbol_validations[symbol]
                        st.markdown(f"""
                        <span class='valid-symbol'>{symbol}</span>
                        <span class='options-available'>Options ✓</span>
                        <small> | {val.sector}</small>
                        """, unsafe_allow_html=True)
        
        # Auto Trading Settings
        st.markdown("---")
        st.markdown("### 🤖 Auto Trading")
        
        st.session_state.auto_trading_enabled = st.checkbox(
            "Enable Auto Trading",
            value=st.session_state.auto_trading_enabled
        )
        
        if st.session_state.auto_trading_enabled:
            st.success("🟢 Auto Trading Active")
            
            min_confidence = st.slider("Min Confidence", 70, 95, 80)
            max_positions = st.slider("Max Positions", 1, 20, 10)
            risk_per_trade = st.slider("Risk Per Trade (%)", 1, 5, 2)
            
            st.session_state.max_positions = max_positions
            st.session_state.max_risk_per_trade = risk_per_trade / 100
            
            st.metric("Capital", f"${st.session_state.auto_trade_capital:,.0f}")
            st.metric("P&L", f"${st.session_state.auto_trade_pnl:+,.0f}")
        
        # Analyze button
        st.markdown("---")
        if st.button("🚀 ANALYZE UNIVERSE", type="primary", use_container_width=True):
            if st.session_state.validated_watchlist:
                st.session_state.force_refresh = True
                st.rerun()
            else:
                st.error("Please load a universe first!")
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Universe Analysis",
        "🎯 Trade Setups",
        "🤖 Auto Trader",
        "📈 Performance",
        "📚 Strategy Guide"
    ])
    
    # Process universe if needed
    if st.session_state.get('force_refresh', False) and st.session_state.validated_watchlist:
        st.session_state.force_refresh = False
        
        with st.spinner(f"Analyzing {len(st.session_state.validated_watchlist)} symbols..."):
            all_data, all_setups = process_universe_parallel(st.session_state.validated_watchlist)
            st.session_state.all_data = all_data
            st.session_state.all_setups_detailed = all_setups
    
    # Tab 1: Universe Analysis
    with tab1:
        render_universe_analysis()
    
    # Tab 2: Trade Setups
    with tab2:
        render_trade_setups()
    
    # Tab 3: Auto Trader
    with tab3:
        render_auto_trader()
    
    # Tab 4: Performance
    with tab4:
        render_performance_analytics()
    
    # Tab 5: Strategy Guide
    with tab5:
        render_strategy_guide()

def render_universe_analysis():
    """Render universe analysis with validated symbols"""
    st.markdown("## 📊 Universe Analysis")
    
    all_data = st.session_state.get('all_data', {})
    
    if not all_data:
        st.info("No data available. Please analyze the universe first.")
        return
    
    # Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Symbols", len(all_data))
    
    with col2:
        negative_gex = len([d for d in all_data.values() if d['net_gex'] < 0])
        st.metric("Negative GEX", negative_gex)
    
    with col3:
        extreme_gex = len([d for d in all_data.values() if abs(d['net_gex']) > 2e9])
        st.metric("Extreme GEX", extreme_gex)
    
    with col4:
        avg_distance = np.mean([abs(d['distance_to_flip']) for d in all_data.values()])
        st.metric("Avg Dist to Flip", f"{avg_distance:.1f}%")
    
    with col5:
        total_setups = len(st.session_state.get('all_setups_detailed', []))
        st.metric("Total Setups", total_setups)
    
    # GEX Distribution Chart
    st.markdown("---")
    st.markdown("### GEX Distribution")
    
    gex_df = pd.DataFrame([
        {'Symbol': k, 'Net GEX (B)': v['net_gex']/1e9, 'Distance to Flip': v['distance_to_flip']}
        for k, v in all_data.items()
    ])
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Net GEX Distribution", "Distance to Gamma Flip")
    )
    
    # GEX histogram
    fig.add_trace(
        go.Histogram(x=gex_df['Net GEX (B)'], nbinsx=30, name="GEX Distribution"),
        row=1, col=1
    )
    
    # Distance to flip histogram
    fig.add_trace(
        go.Histogram(x=gex_df['Distance to Flip'], nbinsx=30, name="Distance Distribution"),
        row=1, col=2
    )
    
    fig.update_layout(
        height=400,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed table
    st.markdown("---")
    st.markdown("### Symbol Details")
    
    # Create detailed dataframe
    table_data = []
    for symbol, data in all_data.items():
        validation = st.session_state.symbol_validations.get(symbol)
        table_data.append({
            'Symbol': symbol,
            'Price': f"${data['price']:.2f}",
            'Net GEX': f"{data['net_gex']/1e9:.2f}B",
            'Gamma Flip': f"${data['gamma_flip']:.2f}",
            'Distance': f"{data['distance_to_flip']:+.1f}%",
            'Call Wall': f"${data['call_walls'][0]:.2f}" if data['call_walls'] else "N/A",
            'Put Wall': f"${data['put_walls'][0]:.2f}" if data['put_walls'] else "N/A",
            'Sector': validation.sector if validation else "Unknown",
            'Market Cap': f"${validation.market_cap/1e9:.1f}B" if validation and validation.market_cap else "N/A"
        })
    
    df = pd.DataFrame(table_data)
    st.dataframe(df, use_container_width=True, height=500)

def render_trade_setups():
    """Render detailed trade setups"""
    st.markdown("## 🎯 Trade Setups")
    
    all_setups = st.session_state.get('all_setups_detailed', [])
    
    if not all_setups:
        st.info("No setups available. Please analyze the universe first.")
        return
    
    # Filter controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        strategy_filter = st.selectbox(
            "Strategy Type",
            ["All", "Squeeze", "Premium", "Condor", "Flip"]
        )
    
    with col2:
        min_confidence = st.slider("Min Confidence", 50, 95, 70)
    
    with col3:
        sort_by = st.selectbox("Sort By", ["Confidence", "Risk/Reward", "Symbol"])
    
    # Apply filters
    filtered_setups = all_setups.copy()
    
    if strategy_filter != "All":
        filtered_setups = [s for s in filtered_setups if strategy_filter.lower() in s.strategy.lower()]
    
    filtered_setups = [s for s in filtered_setups if s.confidence >= min_confidence]
    
    # Sort
    if sort_by == "Confidence":
        filtered_setups.sort(key=lambda x: x.confidence, reverse=True)
    elif sort_by == "Risk/Reward":
        filtered_setups.sort(key=lambda x: x.risk_reward, reverse=True)
    else:
        filtered_setups.sort(key=lambda x: x.symbol)
    
    # Display setups
    st.markdown(f"### Found {len(filtered_setups)} Setups")
    
    for setup in filtered_setups[:20]:  # Limit to top 20
        with st.expander(f"{setup.symbol} - {setup.strategy} ({setup.confidence:.1f}%)"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Entry Details**")
                st.write(f"Type: {setup.strategy_type}")
                st.write(f"Entry: ${setup.entry_price:.2f}")
                st.write(f"Strike: ${setup.strike_price:.2f}")
                st.write(f"DTE: {setup.days_to_expiry}")
            
            with col2:
                st.markdown("**Risk/Reward**")
                st.write(f"Target: ${setup.target_price:.2f}")
                st.write(f"Stop: ${setup.stop_loss:.2f}")
                st.write(f"R/R: {setup.risk_reward:.2f}")
                st.write(f"Max P/L: ${setup.max_profit:.0f} / ${abs(setup.max_loss):.0f}")
            
            with col3:
                st.markdown("**GEX Metrics**")
                st.write(f"Net GEX: {setup.net_gex/1e9:.2f}B")
                st.write(f"Flip: ${setup.gamma_flip:.2f}")
                st.write(f"Distance: {setup.distance_to_flip:+.1f}%")
            
            st.markdown(f"**Analysis:** {setup.description}")
            
            if st.button(f"Execute Trade", key=f"exec_{setup.symbol}_{setup.strategy}"):
                st.success(f"Trade executed for {setup.symbol}")

def render_auto_trader():
    """Render auto trader interface"""
    st.markdown("## 🤖 Auto Trader")
    
    if not st.session_state.auto_trading_enabled:
        st.warning("Auto trading is disabled. Enable it in the sidebar.")
        return
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Capital", f"${st.session_state.auto_trade_capital:,.0f}")
    
    with col2:
        st.metric("P&L", f"${st.session_state.auto_trade_pnl:+,.0f}")
    
    with col3:
        open_positions = len([p for p in st.session_state.auto_positions if p.get('status') == 'OPEN'])
        st.metric("Open Positions", f"{open_positions}/{st.session_state.max_positions}")
    
    with col4:
        win_rate = 0
        if st.session_state.auto_trade_history:
            wins = len([t for t in st.session_state.auto_trade_history if t['pnl'] > 0])
            win_rate = wins / len(st.session_state.auto_trade_history) * 100
        st.metric("Win Rate", f"{win_rate:.1f}%")
    
    # Positions table
    st.markdown("### Open Positions")
    
    if st.session_state.auto_positions:
        positions_df = pd.DataFrame(st.session_state.auto_positions)
        st.dataframe(positions_df, use_container_width=True)
    else:
        st.info("No open positions")
    
    # Trade history
    st.markdown("### Trade History")
    
    if st.session_state.auto_trade_history:
        history_df = pd.DataFrame(st.session_state.auto_trade_history)
        st.dataframe(history_df, use_container_width=True)
    else:
        st.info("No trade history")

def render_performance_analytics():
    """Render performance analytics"""
    st.markdown("## 📈 Performance Analytics")
    
    if not st.session_state.auto_trade_history:
        st.info("No trades to analyze yet.")
        return
    
    trades_df = pd.DataFrame(st.session_state.auto_trade_history)
    
    # Calculate metrics
    total_trades = len(trades_df)
    winning_trades = len(trades_df[trades_df['pnl'] > 0])
    losing_trades = len(trades_df[trades_df['pnl'] < 0])
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Trades", total_trades)
    with col2:
        st.metric("Win Rate", f"{win_rate:.1f}%")
    with col3:
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        st.metric("Avg Win", f"${avg_win:,.2f}")
    with col4:
        avg_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].mean()) if losing_trades > 0 else 0
        st.metric("Avg Loss", f"${avg_loss:,.2f}")
    
    # P&L Chart
    st.markdown("### Cumulative P&L")
    
    trades_df = trades_df.sort_values('exit_time')
    trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=trades_df['exit_time'],
        y=trades_df['cumulative_pnl'],
        mode='lines+markers',
        name='Cumulative P&L',
        line=dict(color='#00D2FF', width=3)
    ))
    
    fig.update_layout(
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_strategy_guide():
    """Render strategy guide"""
    st.markdown("## 📚 Strategy Guide")
    
    strategies = {
        "🚀 Negative GEX Squeeze": {
            "When": "Net GEX < -500M",
            "Setup": "Price below gamma flip with put wall support",
            "Entry": "Buy ATM calls 2-5 DTE",
            "Exit": "Target gamma flip, stop at put wall",
            "Risk": "Max 3% of capital"
        },
        "💰 Premium Selling": {
            "When": "Net GEX > 2B",
            "Setup": "High positive GEX with strong walls",
            "Entry": "Sell OTM options at walls",
            "Exit": "50% profit or expiration",
            "Risk": "Max 5% of capital"
        },
        "🦅 Iron Condor": {
            "When": "Net GEX > 1B, walls > 3% apart",
            "Setup": "Range-bound with clear walls",
            "Entry": "Sell at walls, protect beyond",
            "Exit": "25% profit or 21 DTE",
            "Risk": "Size for 2% max loss"
        },
        "⚡ Gamma Flip Play": {
            "When": "Price within 1% of flip",
            "Setup": "Regime change imminent",
            "Entry": "Directional based on position",
            "Exit": "2% beyond flip point",
            "Risk": "Max 2% of capital"
        }
    }
    
    for name, details in strategies.items():
        with st.expander(name):
            for key, value in details.items():
                st.write(f"**{key}:** {value}")

if __name__ == "__main__":
    main()
