does the pipeline support my streamlit app?

"""
GEX Trading Dashboard - Multi-Symbol Gamma Exposure Analysis Platform
Author: GEX Trading System
Version: 3.0.0
Description: Comprehensive multi-symbol dashboard for gamma exposure analysis, 
             trade setup detection across entire watchlist, and position management 
             with beautiful modern UI design.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta, time
import pytz
import warnings
import json
import time as time_module
from typing import Dict, List, Tuple, Optional
import logging
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

# Page configuration with custom theme
st.set_page_config(
    page_title="GEX Trading Dashboard Pro",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS for beautiful modern design
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main container gradient background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }
    
    /* Headers with gradient text */
    h1, h2, h3 {
        background: linear-gradient(120deg, #00D2FF 0%, #3A7BD5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    
    /* Metrics with glassmorphism effect */
    div[data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 15px 20px;
        border-radius: 16px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        transition: all 0.3s ease;
    }
    
    div[data-testid="metric-container"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px 0 rgba(31, 38, 135, 0.5);
        background: rgba(255, 255, 255, 0.08);
    }
    
    /* Sidebar with glass effect */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: rgba(17, 25, 40, 0.75);
        backdrop-filter: blur(16px) saturate(180%);
        border-right: 1px solid rgba(255, 255, 255, 0.125);
    }
    
    /* Buttons with gradient */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 12px;
        font-weight: 600;
        letter-spacing: 0.02em;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px 0 rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px 0 rgba(102, 126, 234, 0.6);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* Success button variant */
    .success-button > button {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        box-shadow: 0 4px 15px 0 rgba(17, 153, 142, 0.4);
    }
    
    /* Danger button variant */
    .danger-button > button {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        box-shadow: 0 4px 15px 0 rgba(235, 51, 73, 0.4);
    }
    
    /* Trade setup cards */
    .trade-setup-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.1) 0%, rgba(255, 255, 255, 0.05) 100%);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.18);
        border-radius: 20px;
        padding: 25px;
        margin: 15px 0;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .trade-setup-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
        transition: left 0.5s;
    }
    
    .trade-setup-card:hover::before {
        left: 100%;
    }
    
    .trade-setup-card:hover {
        transform: translateY(-5px) scale(1.02);
        box-shadow: 0 12px 40px 0 rgba(0, 210, 255, 0.3);
        border: 1px solid rgba(0, 210, 255, 0.4);
    }
    
    /* Alert boxes with gradients */
    .alert-high {
        background: linear-gradient(135deg, rgba(235, 51, 73, 0.2) 0%, rgba(244, 92, 67, 0.2) 100%);
        border-left: 4px solid #eb3349;
        padding: 15px;
        border-radius: 12px;
        margin: 10px 0;
        backdrop-filter: blur(5px);
    }
    
    .alert-medium {
        background: linear-gradient(135deg, rgba(250, 177, 160, 0.2) 0%, rgba(255, 218, 185, 0.2) 100%);
        border-left: 4px solid #fab1a0;
        padding: 15px;
        border-radius: 12px;
        margin: 10px 0;
        backdrop-filter: blur(5px);
    }
    
    .alert-low {
        background: linear-gradient(135deg, rgba(17, 153, 142, 0.2) 0%, rgba(56, 239, 125, 0.2) 100%);
        border-left: 4px solid #11998e;
        padding: 15px;
        border-radius: 12px;
        margin: 10px 0;
        backdrop-filter: blur(5px);
    }
    
    /* Confidence badges */
    .confidence-badge {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 12px;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    
    .confidence-high {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
    }
    
    .confidence-medium {
        background: linear-gradient(135deg, #f2994a 0%, #f2c94c 100%);
        color: white;
    }
    
    .confidence-low {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        color: white;
    }
    
    /* Symbol cards */
    .symbol-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 12px;
        margin: 5px;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .symbol-card:hover {
        background: rgba(255, 255, 255, 0.08);
        transform: scale(1.05);
        box-shadow: 0 4px 15px rgba(0, 210, 255, 0.3);
    }
    
    /* Tabs with gradient underline */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 12px;
        padding: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: rgba(255, 255, 255, 0.7);
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 8px;
    }
    
    /* Info boxes */
    .info-box {
        background: rgba(0, 210, 255, 0.1);
        border: 1px solid rgba(0, 210, 255, 0.3);
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
    }
    
    /* Performance metrics grid */
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 15px;
        margin: 20px 0;
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* Animation for new alerts */
    @keyframes slideIn {
        from {
            transform: translateX(-100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    .new-alert {
        animation: slideIn 0.5s ease-out;
    }
    
    /* Pulse animation for live indicators */
    @keyframes pulse {
        0% {
            box-shadow: 0 0 0 0 rgba(0, 210, 255, 0.7);
        }
        70% {
            box-shadow: 0 0 0 10px rgba(0, 210, 255, 0);
        }
        100% {
            box-shadow: 0 0 0 0 rgba(0, 210, 255, 0);
        }
    }
    
    .live-indicator {
        display: inline-block;
        width: 8px;
        height: 8px;
        background: #00D2FF;
        border-radius: 50%;
        animation: pulse 2s infinite;
        margin-right: 8px;
    }
    
    /* Glow effect for important metrics */
    .glow {
        box-shadow: 0 0 20px rgba(0, 210, 255, 0.5);
    }
    
    /* Table styling */
    .dataframe {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        overflow: hidden;
    }
    
    .dataframe th {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        font-weight: 600;
        text-transform: uppercase;
        font-size: 12px;
        letter-spacing: 0.05em;
    }
    
    .dataframe td {
        background: rgba(255, 255, 255, 0.02);
        color: rgba(255, 255, 255, 0.9);
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .dataframe tr:hover td {
        background: rgba(255, 255, 255, 0.08);
    }
</style>
""", unsafe_allow_html=True)

# ======================== SESSION STATE INITIALIZATION ========================

if 'portfolio' not in st.session_state:
    st.session_state.portfolio = {
        'positions': [],
        'cash': 100000,
        'total_value': 100000,
        'daily_pnl': 0,
        'trade_history': []
    }

if 'all_gex_data' not in st.session_state:
    st.session_state.all_gex_data = {}

if 'all_setups' not in st.session_state:
    st.session_state.all_setups = []

if 'last_update' not in st.session_state:
    st.session_state.last_update = None

if 'alerts' not in st.session_state:
    st.session_state.alerts = []

if 'watchlist' not in st.session_state:
    st.session_state.watchlist = ["SPY", "QQQ", "IWM", "DIA"]

# ======================== CORE GEX CALCULATION FUNCTIONS ========================

class GEXCalculator:
    """Core GEX calculation engine with all gamma exposure metrics"""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.spot_price = None
        self.options_chain = None
        self.gex_profile = None
        self.gamma_flip = None
        self.net_gex = None
        
    def fetch_options_data(self) -> bool:
        """Fetch complete options chain for the symbol"""
        try:
            ticker = yf.Ticker(self.symbol)
            
            # Get current price
            hist = ticker.history(period='1d')
            if not hist.empty:
                self.spot_price = hist['Close'].iloc[-1]
            else:
                info = ticker.info
                self.spot_price = info.get('regularMarketPrice', info.get('currentPrice', 100))
            
            # Get all expiration dates
            expirations = ticker.options[:10] if hasattr(ticker, 'options') else []
            
            all_options = []
            for exp in expirations:
                try:
                    opt_chain = ticker.option_chain(exp)
                    
                    # Process calls
                    calls = opt_chain.calls.copy()
                    calls['type'] = 'call'
                    calls['expiration'] = exp
                    
                    # Process puts
                    puts = opt_chain.puts.copy()
                    puts['type'] = 'put'
                    puts['expiration'] = exp
                    
                    all_options.extend([calls, puts])
                except Exception as e:
                    logger.warning(f"Failed to fetch options for {exp}: {e}")
                    continue
            
            if all_options:
                self.options_chain = pd.concat(all_options, ignore_index=True)
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to fetch options data for {self.symbol}: {e}")
            return False
    
    def calculate_gamma_exposure(self) -> pd.DataFrame:
        """Calculate gamma exposure for all strikes"""
        if self.options_chain is None:
            return pd.DataFrame()
        
        try:
            # Calculate GEX for each option
            gex_data = []
            
            for _, option in self.options_chain.iterrows():
                strike = option['strike']
                gamma = option.get('gamma', 0)
                if gamma is None or gamma == 0:
                    # Estimate gamma if not provided
                    gamma = self._estimate_gamma(option)
                
                open_interest = option.get('openInterest', 0)
                if pd.isna(open_interest):
                    open_interest = 0
                
                # Calculate GEX: Spot × Gamma × OI × 100
                if option['type'] == 'call':
                    gex = self.spot_price * gamma * open_interest * 100
                else:  # put
                    gex = -1 * self.spot_price * gamma * open_interest * 100
                
                gex_data.append({
                    'strike': strike,
                    'type': option['type'],
                    'expiration': option['expiration'],
                    'gamma': gamma,
                    'openInterest': open_interest,
                    'gex': gex,
                    'volume': option.get('volume', 0),
                    'impliedVolatility': option.get('impliedVolatility', 0)
                })
            
            gex_df = pd.DataFrame(gex_data)
            
            # Aggregate GEX by strike
            self.gex_profile = gex_df.groupby('strike').agg({
                'gex': 'sum',
                'gamma': 'sum',
                'openInterest': 'sum',
                'volume': 'sum'
            }).reset_index()
            
            # Calculate cumulative GEX
            self.gex_profile = self.gex_profile.sort_values('strike')
            self.gex_profile['cumulative_gex'] = self.gex_profile['gex'].cumsum()
            
            # Calculate net GEX
            self.net_gex = self.gex_profile['gex'].sum()
            
            # Find gamma flip point (where cumulative GEX crosses zero)
            self._calculate_gamma_flip()
            
            # Identify walls
            self._identify_gamma_walls()
            
            return self.gex_profile
            
        except Exception as e:
            logger.error(f"Failed to calculate GEX for {self.symbol}: {e}")
            return pd.DataFrame()
    
    def _estimate_gamma(self, option: pd.Series) -> float:
        """Estimate gamma using simplified Black-Scholes approximation"""
        try:
            strike = option['strike']
            days_to_exp = (pd.to_datetime(option['expiration']) - pd.Timestamp.now()).days
            if days_to_exp <= 0:
                return 0
            
            time_to_exp = days_to_exp / 365.0
            iv = option.get('impliedVolatility', 0.2)
            
            # Simplified gamma estimation
            moneyness = self.spot_price / strike
            if 0.9 < moneyness < 1.1:  # Near the money
                gamma = np.exp(-0.5 * ((np.log(moneyness) / (iv * np.sqrt(time_to_exp))) ** 2))
                gamma = gamma / (self.spot_price * iv * np.sqrt(2 * np.pi * time_to_exp))
            else:
                gamma = 0.001  # Small gamma for far OTM/ITM
            
            return gamma
            
        except Exception:
            return 0.001
    
    def _calculate_gamma_flip(self):
        """Calculate the gamma flip point where net GEX crosses zero"""
        if self.gex_profile is None or len(self.gex_profile) == 0:
            return
        
        # Find where cumulative GEX is closest to zero
        abs_cumulative = np.abs(self.gex_profile['cumulative_gex'])
        flip_idx = abs_cumulative.idxmin()
        self.gamma_flip = self.gex_profile.loc[flip_idx, 'strike']
    
    def _identify_gamma_walls(self):
        """Identify call walls and put walls based on gamma concentration"""
        if self.gex_profile is None or len(self.gex_profile) == 0:
            return
        
        # Separate positive and negative GEX
        call_gex = self.gex_profile[self.gex_profile['gex'] > 0].copy()
        put_gex = self.gex_profile[self.gex_profile['gex'] < 0].copy()
        
        # Find top 3 call walls (highest positive GEX)
        if len(call_gex) > 0:
            call_walls = call_gex.nlargest(3, 'gex')['strike'].tolist()
            self.gex_profile['is_call_wall'] = self.gex_profile['strike'].isin(call_walls)
        else:
            self.gex_profile['is_call_wall'] = False
        
        # Find top 3 put walls (highest negative GEX magnitude)
        if len(put_gex) > 0:
            put_gex['abs_gex'] = np.abs(put_gex['gex'])
            put_walls = put_gex.nlargest(3, 'abs_gex')['strike'].tolist()
            self.gex_profile['is_put_wall'] = self.gex_profile['strike'].isin(put_walls)
        else:
            self.gex_profile['is_put_wall'] = False

# ======================== TRADE SETUP DETECTION ========================

class TradeSetupDetector:
    """Detect high-probability trade setups based on GEX profile"""
    
    def __init__(self, gex_calculator: GEXCalculator):
        self.gex = gex_calculator
        self.setups = []
    
    def detect_all_setups(self) -> List[Dict]:
        """Run all setup detection algorithms"""
        self.setups = []
        
        # Check for squeeze plays
        self._detect_squeeze_setups()
        
        # Check for premium selling opportunities
        self._detect_premium_selling()
        
        # Check for iron condor setups
        self._detect_iron_condors()
        
        # Sort by confidence score
        self.setups.sort(key=lambda x: x['confidence'], reverse=True)
        
        return self.setups
    
    def _detect_squeeze_setups(self):
        """Detect negative GEX squeeze and positive GEX breakdown setups"""
        if self.gex.net_gex is None or self.gex.gamma_flip is None:
            return
        
        spot = self.gex.spot_price
        flip = self.gex.gamma_flip
        net_gex = self.gex.net_gex
        symbol = self.gex.symbol
        
        # Adjust thresholds based on symbol
        if symbol in ['SPY', 'SPX']:
            neg_threshold = -1e9  # -1B for SPY
            pos_threshold = 2e9   # 2B for SPY
        elif symbol in ['QQQ', 'NDX']:
            neg_threshold = -5e8  # -500M for QQQ
            pos_threshold = 1e9   # 1B for QQQ
        else:  # Individual stocks
            neg_threshold = -1e8  # -100M for stocks
            pos_threshold = 2e8   # 200M for stocks
        
        if net_gex < neg_threshold:
            distance_to_flip = (flip - spot) / spot * 100
            
            if 0.5 <= distance_to_flip <= 1.5:
                # Find put wall support
                put_walls = self.gex.gex_profile[self.gex.gex_profile['is_put_wall'] == True]
                if len(put_walls) > 0:
                    nearest_put_wall = put_walls.iloc[0]['strike']
                    
                    setup = {
                        'symbol': symbol,
                        'type': 'SQUEEZE_LONG_CALL',
                        'strategy': '🚀 Negative GEX Squeeze',
                        'entry_price': spot,
                        'target_strike': flip,
                        'stop_loss': nearest_put_wall,
                        'confidence': min(85, 70 + abs(distance_to_flip) * 10),
                        'risk_reward': abs(flip - spot) / abs(spot - nearest_put_wall) if abs(spot - nearest_put_wall) > 0 else 0,
                        'description': f"Long Call: Strong negative GEX ({net_gex/1e9:.2f}B) with price {distance_to_flip:.1f}% below flip",
                        'entry_criteria': f"Buy ATM or first OTM call above {flip:.2f}",
                        'days_to_expiry': '2-5 DTE',
                        'position_size': '3% of capital max',
                        'notes': 'High volatility expected - dealers must buy on rally'
                    }
                    self.setups.append(setup)
        
        # Positive GEX Breakdown (Long Puts)
        elif net_gex > pos_threshold:
            distance_to_flip = (spot - flip) / spot * 100
            
            if 0 <= distance_to_flip <= 0.3:
                # Find call wall resistance
                call_walls = self.gex.gex_profile[self.gex.gex_profile['is_call_wall'] == True]
                if len(call_walls) > 0:
                    nearest_call_wall = call_walls.iloc[0]['strike']
                    
                    setup = {
                        'symbol': symbol,
                        'type': 'SQUEEZE_LONG_PUT',
                        'strategy': '📉 Positive GEX Breakdown',
                        'entry_price': spot,
                        'target_strike': flip,
                        'stop_loss': nearest_call_wall,
                        'confidence': min(80, 65 + distance_to_flip * 50),
                        'risk_reward': abs(spot - flip) / abs(nearest_call_wall - spot) if abs(nearest_call_wall - spot) > 0 else 0,
                        'description': f"Long Put: High positive GEX ({net_gex/1e9:.2f}B) hovering near flip",
                        'entry_criteria': f"Buy ATM or first OTM put below {flip:.2f}",
                        'days_to_expiry': '3-7 DTE',
                        'position_size': '3% of capital max',
                        'notes': 'Breakdown imminent - dealers must sell on decline'
                    }
                    self.setups.append(setup)
        
        # Gamma Wall Compression
        self._detect_wall_compression()
    
    def _detect_wall_compression(self):
        """Detect when call and put walls create compression setup"""
        call_walls = self.gex.gex_profile[self.gex.gex_profile['is_call_wall'] == True]
        put_walls = self.gex.gex_profile[self.gex.gex_profile['is_put_wall'] == True]
        
        if len(call_walls) > 0 and len(put_walls) > 0:
            highest_call = call_walls.iloc[0]['strike']
            highest_put = put_walls.iloc[0]['strike']
            
            wall_spread = (highest_call - highest_put) / self.gex.spot_price * 100
            
            if wall_spread < 2:  # Walls less than 2% apart
                spot = self.gex.spot_price
                symbol = self.gex.symbol
                
                # Determine direction based on position
                if abs(spot - highest_put) < abs(highest_call - spot):
                    # Closer to put wall - long calls
                    setup = {
                        'symbol': symbol,
                        'type': 'COMPRESSION_LONG_CALL',
                        'strategy': '💥 Gamma Wall Compression',
                        'entry_price': spot,
                        'target_strike': highest_call,
                        'stop_loss': highest_put,
                        'confidence': 75,
                        'risk_reward': abs(highest_call - spot) / abs(spot - highest_put) if abs(spot - highest_put) > 0 else 0,
                        'description': f"Compression Setup: Walls only {wall_spread:.1f}% apart",
                        'entry_criteria': f"Long calls - near put wall support at {highest_put:.2f}",
                        'days_to_expiry': '0-2 DTE for explosion',
                        'position_size': '2% of capital max',
                        'notes': 'Explosive move expected on wall break'
                    }
                else:
                    # Closer to call wall - long puts
                    setup = {
                        'symbol': symbol,
                        'type': 'COMPRESSION_LONG_PUT',
                        'strategy': '💥 Gamma Wall Compression',
                        'entry_price': spot,
                        'target_strike': highest_put,
                        'stop_loss': highest_call,
                        'confidence': 75,
                        'risk_reward': abs(spot - highest_put) / abs(highest_call - spot) if abs(highest_call - spot) > 0 else 0,
                        'description': f"Compression Setup: Walls only {wall_spread:.1f}% apart",
                        'entry_criteria': f"Long puts - near call wall resistance at {highest_call:.2f}",
                        'days_to_expiry': '0-2 DTE for explosion',
                        'position_size': '2% of capital max',
                        'notes': 'Explosive move expected on wall break'
                    }
                
                self.setups.append(setup)
    
    def _detect_premium_selling(self):
        """Detect premium selling opportunities at gamma walls"""
        if self.gex.net_gex is None:
            return
        
        spot = self.gex.spot_price
        net_gex = self.gex.net_gex
        symbol = self.gex.symbol
        
        # Adjust threshold for individual stocks
        gex_threshold = 3e9 if symbol in ['SPY', 'QQQ'] else 3e8
        
        # Call selling at resistance
        call_walls = self.gex.gex_profile[self.gex.gex_profile['is_call_wall'] == True]
        if len(call_walls) > 0 and net_gex > gex_threshold:  # Need high positive GEX
            nearest_call_wall = call_walls.iloc[0]
            wall_strength = abs(nearest_call_wall['gex'])
            
            strength_threshold = 5e8 if symbol in ['SPY', 'QQQ'] else 5e7
            
            if wall_strength > strength_threshold:
                distance = (nearest_call_wall['strike'] - spot) / spot * 100
                
                if 0.5 <= distance <= 2:
                    setup = {
                        'symbol': symbol,
                        'type': 'SELL_CALL',
                        'strategy': '💰 Call Premium Selling',
                        'entry_price': spot,
                        'strike': nearest_call_wall['strike'],
                        'confidence': min(85, 70 + (wall_strength / 1e9) * 5),
                        'description': f"Sell Call: Strong resistance at {nearest_call_wall['strike']:.2f}",
                        'entry_criteria': f"Sell calls at or above {nearest_call_wall['strike']:.2f}",
                        'days_to_expiry': '0-2 DTE',
                        'position_size': '5% of capital max',
                        'exit_criteria': 'Close at 50% profit or if approaching wall',
                        'notes': f'Wall strength: {wall_strength/1e9:.2f}B gamma'
                    }
                    self.setups.append(setup)
        
        # Put selling at support
        put_walls = self.gex.gex_profile[self.gex.gex_profile['is_put_wall'] == True]
        if len(put_walls) > 0:
            nearest_put_wall = put_walls.iloc[0]
            wall_strength = abs(nearest_put_wall['gex'])
            
            strength_threshold = 5e8 if symbol in ['SPY', 'QQQ'] else 5e7
            
            if wall_strength > strength_threshold:
                distance = (spot - nearest_put_wall['strike']) / spot * 100
                
                if distance >= 1:
                    setup = {
                        'symbol': symbol,
                        'type': 'SELL_PUT',
                        'strategy': '💸 Put Premium Selling',
                        'entry_price': spot,
                        'strike': nearest_put_wall['strike'],
                        'confidence': min(80, 65 + (wall_strength / 1e9) * 5),
                        'description': f"Sell Put: Strong support at {nearest_put_wall['strike']:.2f}",
                        'entry_criteria': f"Sell puts at or below {nearest_put_wall['strike']:.2f}",
                        'days_to_expiry': '2-5 DTE',
                        'position_size': '5% of capital max',
                        'exit_criteria': 'Close at 50% profit or define max loss',
                        'notes': f'Wall strength: {wall_strength/1e9:.2f}B gamma'
                    }
                    self.setups.append(setup)
    
    def _detect_iron_condors(self):
        """Detect iron condor opportunities based on gamma profile"""
        symbol = self.gex.symbol
        gex_threshold = 1e9 if symbol in ['SPY', 'QQQ'] else 1e8
        
        if self.gex.net_gex is None or self.gex.net_gex < gex_threshold:
            return  # Need positive gamma environment
        
        call_walls = self.gex.gex_profile[self.gex.gex_profile['is_call_wall'] == True]
        put_walls = self.gex.gex_profile[self.gex.gex_profile['is_put_wall'] == True]
        
        if len(call_walls) > 0 and len(put_walls) > 0:
            highest_call = call_walls.iloc[0]['strike']
            highest_put = put_walls.iloc[0]['strike']
            
            wall_spread = (highest_call - highest_put) / self.gex.spot_price * 100
            
            if wall_spread > 3:  # Walls more than 3% apart
                # Calculate directional bias
                total_call_gamma = call_walls['gex'].sum()
                total_put_gamma = abs(put_walls['gex'].sum())
                
                if total_put_gamma > total_call_gamma:
                    condor_type = "🦅 Broken Wing (Bullish)"
                    put_spread_mult = 1.5
                    call_spread_mult = 0.75
                elif total_call_gamma > total_put_gamma:
                    condor_type = "🦅 Broken Wing (Bearish)"
                    put_spread_mult = 0.75
                    call_spread_mult = 1.5
                else:
                    condor_type = "🦅 Standard"
                    put_spread_mult = 1.0
                    call_spread_mult = 1.0
                
                setup = {
                    'symbol': symbol,
                    'type': 'IRON_CONDOR',
                    'strategy': f'{condor_type} Iron Condor',
                    'entry_price': self.gex.spot_price,
                    'call_short': highest_call,
                    'put_short': highest_put,
                    'confidence': min(75, 60 + wall_spread * 2),
                    'description': f"Iron Condor: Walls {wall_spread:.1f}% apart",
                    'entry_criteria': f"Short strikes at walls: Call {highest_call:.2f}, Put {highest_put:.2f}",
                    'days_to_expiry': '5-10 DTE',
                    'position_size': 'Size for max 2% portfolio loss',
                    'adjustments': f"Put spread: {put_spread_mult}x, Call spread: {call_spread_mult}x",
                    'notes': f'Net GEX: {self.gex.net_gex/1e9:.2f}B (stable environment)'
                }
                self.setups.append(setup)

# ======================== MAIN DASHBOARD ========================

def main():
    # Animated Header with live indicator
    col1, col2, col3 = st.columns([1, 6, 1])
    with col2:
        st.markdown("""
        <h1 style='text-align: center; font-size: 48px; margin-bottom: 0;'>
            <span class='live-indicator'></span>
            GEX Trading Dashboard Pro
        </h1>
        <p style='text-align: center; color: rgba(255,255,255,0.7); font-size: 18px; margin-top: 10px;'>
            Real-time Multi-Symbol Gamma Exposure Analysis & Trade Detection
        </p>
        """, unsafe_allow_html=True)
    
    # Sidebar configuration with glassmorphism effect
    with st.sidebar:
        st.markdown("<h2 style='text-align: center;'>⚙️ Control Panel</h2>", unsafe_allow_html=True)
        
        # Watchlist Management
        st.markdown("### 📊 Watchlist Management")
        
        # Predefined symbols
        popular_symbols = ["SPY", "QQQ", "IWM", "DIA", "AAPL", "TSLA", "NVDA", 
                          "AMD", "META", "AMZN", "GOOGL", "MSFT", "NFLX", "BA",
                          "JPM", "GS", "XOM", "CVX", "PFE", "JNJ"]
        
        selected_symbols = st.multiselect(
            "Select Symbols to Monitor",
            options=popular_symbols,
            default=st.session_state.watchlist,
            help="Choose multiple symbols for GEX analysis"
        )
        
        # Custom symbols input
        custom_symbols = st.text_input(
            "Add Custom Symbols",
            placeholder="SYMBOL1, SYMBOL2, ...",
            help="Enter comma-separated symbols"
        )
        
        if custom_symbols:
            custom_list = [s.strip().upper() for s in custom_symbols.split(",") if s.strip()]
            selected_symbols.extend(custom_list)
        
        # Remove duplicates and update session state
        st.session_state.watchlist = list(set(selected_symbols))
        
        # Display current watchlist with cards
        st.markdown("### 👁️ Active Watchlist")
        cols = st.columns(3)
        for i, symbol in enumerate(st.session_state.watchlist):
            with cols[i % 3]:
                st.markdown(f"""
                <div class='symbol-card'>
                    <div style='font-weight: 600; color: #00D2FF;'>{symbol}</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.divider()
        
        # Auto-refresh settings
        st.markdown("### 🔄 Auto Refresh")
        auto_refresh = st.checkbox("Enable Auto Refresh", value=False)
        if auto_refresh:
            refresh_interval = st.slider("Interval (minutes)", 1, 30, 5)
        
        st.divider()
        
        # Portfolio overview with gradient cards
        st.markdown("### 💼 Portfolio Overview")
        
        st.markdown(f"""
        <div class='info-box'>
            <div style='display: flex; justify-content: space-between; margin-bottom: 10px;'>
                <span style='color: rgba(255,255,255,0.7);'>Cash Available</span>
                <span style='font-weight: 600; color: #00D2FF;'>${st.session_state.portfolio['cash']:,.0f}</span>
            </div>
            <div style='display: flex; justify-content: space-between; margin-bottom: 10px;'>
                <span style='color: rgba(255,255,255,0.7);'>Total Value</span>
                <span style='font-weight: 600; color: #00D2FF;'>${st.session_state.portfolio['total_value']:,.0f}</span>
            </div>
            <div style='display: flex; justify-content: space-between;'>
                <span style='color: rgba(255,255,255,0.7);'>Daily P&L</span>
                <span style='font-weight: 600; color: {"#38ef7d" if st.session_state.portfolio["daily_pnl"] >= 0 else "#f45c43"};'>
                    ${st.session_state.portfolio['daily_pnl']:+,.0f}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        # Risk Settings
        st.markdown("### 🎚️ Risk Management")
        max_position = st.slider("Max Position Size %", 1, 10, 5)
        max_loss = st.slider("Max Loss per Trade %", 1, 5, 3)
        confidence_threshold = st.slider("Min Confidence %", 50, 90, 65)
        
        st.divider()
        
        # Scan button with gradient
        if st.button("🚀 Scan All Symbols", type="primary", use_container_width=True):
            st.session_state.last_update = datetime.now()
            st.rerun()
    
    # Main content area with tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🎯 Top Opportunities", 
        "📊 GEX Analysis", 
        "💎 Trade Setups", 
        "📈 Positions", 
        "⚠️ Alerts",
        "📉 Performance",
        "🔍 Strategy Guide"
    ])
    
    # Scan all symbols and collect data
    with st.spinner("🔍 Scanning all symbols for opportunities..."):
        all_setups = []
        all_gex_data = {}
        scan_results = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, symbol in enumerate(st.session_state.watchlist):
            status_text.text(f"Analyzing {symbol}...")
            progress_bar.progress((idx + 1) / len(st.session_state.watchlist))
            
            gex_calc = GEXCalculator(symbol)
            
            if gex_calc.fetch_options_data():
                gex_calc.calculate_gamma_exposure()
                all_gex_data[symbol] = gex_calc
                
                # Detect setups for this symbol
                detector = TradeSetupDetector(gex_calc)
                setups = detector.detect_all_setups()
                
                # Add symbol to each setup
                for setup in setups:
                    setup['symbol'] = symbol
                
                all_setups.extend(setups)
                
                # Store scan results summary
                scan_results.append({
                    'symbol': symbol,
                    'spot': gex_calc.spot_price,
                    'net_gex': gex_calc.net_gex,
                    'gamma_flip': gex_calc.gamma_flip,
                    'setup_count': len(setups)
                })
        
        progress_bar.empty()
        status_text.empty()
        
        # Sort all setups by confidence
        all_setups.sort(key=lambda x: x['confidence'], reverse=True)
        
        st.session_state.all_setups = all_setups
        st.session_state.all_gex_data = all_gex_data
    
    # Tab 1: Top Opportunities Dashboard
    with tab1:
        st.markdown("## 🏆 Top Trading Opportunities Across All Symbols")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_setups = len(all_setups)
            st.metric("Total Setups Found", total_setups, 
                     delta=f"{len([s for s in all_setups if s['confidence'] > 75])} high confidence")
        
        with col2:
            symbols_with_setups = len(set([s['symbol'] for s in all_setups]))
            st.metric("Active Symbols", f"{symbols_with_setups}/{len(st.session_state.watchlist)}")
        
        with col3:
            if all_setups:
                avg_confidence = np.mean([s['confidence'] for s in all_setups])
                st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
            else:
                st.metric("Avg Confidence", "N/A")
        
        with col4:
            high_conf_setups = [s for s in all_setups if s['confidence'] > confidence_threshold]
            st.metric("Qualified Setups", len(high_conf_setups))
        
        st.divider()
        
        # Display top opportunities
        if all_setups:
            st.markdown("### 🎯 Best Opportunities Right Now")
            
            # Filter by minimum confidence
            filtered_setups = [s for s in all_setups if s['confidence'] >= confidence_threshold]
            
            if filtered_setups:
                # Display top 10 setups with beautiful cards
                for idx, setup in enumerate(filtered_setups[:10]):
                    # Determine confidence badge color
                    if setup['confidence'] > 80:
                        conf_class = "confidence-high"
                        conf_emoji = "🟢"
                    elif setup['confidence'] > 70:
                        conf_class = "confidence-medium"
                        conf_emoji = "🟡"
                    else:
                        conf_class = "confidence-low"
                        conf_emoji = "🔴"
                    
                    # Create expandable card for each setup
                    with st.expander(f"{setup['symbol']} - {setup['strategy']} {conf_emoji}", expanded=(idx < 3)):
                        col1, col2, col3 = st.columns([3, 1, 1])
                        
                        with col1:
                            st.markdown(f"""
                            <div class='trade-setup-card'>
                                <h3 style='margin: 0;'>{setup['symbol']} - {setup['strategy']}</h3>
                                <p style='color: rgba(255,255,255,0.9); margin: 10px 0;'>{setup['description']}</p>
                                <div style='margin: 15px 0;'>
                                    <strong>Entry:</strong> {setup['entry_criteria']}<br/>
                                    <strong>Timeframe:</strong> {setup['days_to_expiry']}<br/>
                                    <strong>Position Size:</strong> {setup['position_size']}<br/>
                                    <strong>Notes:</strong> {setup.get('notes', 'N/A')}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            st.metric("Confidence", f"{setup['confidence']:.0f}%")
                            if 'risk_reward' in setup and setup['risk_reward'] > 0:
                                st.metric("Risk/Reward", f"{setup['risk_reward']:.2f}")
                        
                        with col3:
                            if st.button(f"Execute Trade", key=f"execute_{setup['symbol']}_{setup['type']}_{idx}"):
                                # Execute trade logic
                                st.success(f"Trade executed for {setup['symbol']}")
                                st.balloons()
            else:
                st.info(f"No setups found with confidence >= {confidence_threshold}%. Try lowering the threshold.")
        else:
            st.warning("No trade setups detected across any symbols. Market conditions may not be favorable.")
        
        # Market Overview Table
        st.divider()
        st.markdown("### 📊 Market Overview")
        
        if scan_results:
            scan_df = pd.DataFrame(scan_results)
            scan_df['Net GEX (B)'] = scan_df['net_gex'] / 1e9
            scan_df['Regime'] = scan_df['net_gex'].apply(lambda x: 
                '🟢 Positive' if x > 0 else '🔴 Negative')
            
            display_df = scan_df[['symbol', 'spot', 'Net GEX (B)', 'gamma_flip', 'Regime', 'setup_count']]
            display_df.columns = ['Symbol', 'Spot Price', 'Net GEX (B)', 'Gamma Flip', 'Regime', 'Setups']
            
            st.dataframe(
                display_df.style.format({
                    'Spot Price': '${:.2f}',
                    'Net GEX (B)': '{:.2f}B',
                    'Gamma Flip': '${:.2f}'
                }),
                use_container_width=True,
                height=400
            )
    
    # Tab 2: GEX Analysis (Individual Symbol Deep Dive)
    with tab2:
        st.markdown("## 📊 Detailed GEX Analysis")
        
        # Symbol selector for detailed view
        if st.session_state.all_gex_data:
            selected_symbol = st.selectbox(
                "Select Symbol for Detailed Analysis",
                options=list(st.session_state.all_gex_data.keys())
            )
            
            if selected_symbol in st.session_state.all_gex_data:
                gex = st.session_state.all_gex_data[selected_symbol]
                
                # Key metrics with gradient cards
                st.markdown(f"### {selected_symbol} GEX Metrics")
                
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.metric("Spot Price", f"${gex.spot_price:.2f}")
                
                with col2:
                    net_gex_b = gex.net_gex / 1e9 if gex.net_gex else 0
                    gex_color = "🟢" if net_gex_b > 0 else "🔴"
                    st.metric("Net GEX", f"{gex_color} ${net_gex_b:.2f}B")
                
                with col3:
                    if gex.gamma_flip:
                        flip_distance = (gex.gamma_flip - gex.spot_price) / gex.spot_price * 100
                        st.metric("Gamma Flip", f"${gex.gamma_flip:.2f}", 
                                 delta=f"{flip_distance:+.2f}%")
                    else:
                        st.metric("Gamma Flip", "N/A")
                
                with col4:
                    call_walls = gex.gex_profile[gex.gex_profile['is_call_wall'] == True]
                    if len(call_walls) > 0:
                        st.metric("Call Wall", f"${call_walls.iloc[0]['strike']:.2f}")
                    else:
                        st.metric("Call Wall", "N/A")
                
                with col5:
                    put_walls = gex.gex_profile[gex.gex_profile['is_put_wall'] == True]
                    if len(put_walls) > 0:
                        st.metric("Put Wall", f"${put_walls.iloc[0]['strike']:.2f}")
                    else:
                        st.metric("Put Wall", "N/A")
                
                # GEX Profile Chart with dark theme
                st.markdown("### 📈 Gamma Exposure Profile")
                
                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=(f"{selected_symbol} Gamma Exposure by Strike", "Cumulative GEX"),
                    vertical_spacing=0.15,
                    row_heights=[0.7, 0.3]
                )
                
                # GEX bars with gradient colors
                colors = ['#38ef7d' if x > 0 else '#f45c43' for x in gex.gex_profile['gex']]
                
                fig.add_trace(
                    go.Bar(
                        x=gex.gex_profile['strike'],
                        y=gex.gex_profile['gex'] / 1e6,
                        name='GEX',
                        marker_color=colors,
                        marker_line_color='rgba(255,255,255,0.2)',
                        marker_line_width=1,
                        showlegend=False,
                        hovertemplate='Strike: $%{x}<br>GEX: %{y:.2f}M<extra></extra>'
                    ),
                    row=1, col=1
                )
                
                # Add spot price line
                fig.add_vline(x=gex.spot_price, line_dash="dash", line_color="#00D2FF", 
                             line_width=2, annotation_text=f"Spot: ${gex.spot_price:.2f}", 
                             annotation_position="top", row=1, col=1)
                
                # Add gamma flip line
                if gex.gamma_flip:
                    fig.add_vline(x=gex.gamma_flip, line_dash="dash", line_color="#FFD700",
                                line_width=2, annotation_text=f"Flip: ${gex.gamma_flip:.2f}",
                                annotation_position="top", row=1, col=1)
                
                # Cumulative GEX with gradient line
                fig.add_trace(
                    go.Scatter(
                        x=gex.gex_profile['strike'],
                        y=gex.gex_profile['cumulative_gex'] / 1e9,
                        mode='lines',
                        name='Cumulative GEX',
                        line=dict(color='#764ba2', width=3),
                        fill='tozeroy',
                        fillcolor='rgba(118, 75, 162, 0.2)',
                        hovertemplate='Strike: $%{x}<br>Cumulative: %{y:.2f}B<extra></extra>'
                    ),
                    row=2, col=1
                )
                
                fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)", 
                            line_width=1, row=2, col=1)
                
                # Update layout with dark theme
                fig.update_layout(
                    height=700,
                    showlegend=True,
                    hovermode='x unified',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='rgba(255,255,255,0.9)'),
                    xaxis=dict(
                        gridcolor='rgba(255,255,255,0.1)',
                        title="Strike Price",
                        title_font=dict(size=14)
                    ),
                    yaxis=dict(
                        gridcolor='rgba(255,255,255,0.1)',
                        title="GEX (Millions)",
                        title_font=dict(size=14)
                    ),
                    xaxis2=dict(
                        gridcolor='rgba(255,255,255,0.1)',
                        title="Strike Price",
                        title_font=dict(size=14)
                    ),
                    yaxis2=dict(
                        gridcolor='rgba(255,255,255,0.1)',
                        title="Cumulative GEX (Billions)",
                        title_font=dict(size=14)
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Market regime interpretation with gradient box
                st.markdown("### 🎭 Market Regime Analysis")
                
                if gex.net_gex > 2e9:
                    regime = "🟢 **HIGH POSITIVE GAMMA**"
                    interpretation = """
                    - Volatility suppression in effect
                    - Dealers sell rallies, buy dips
                    - Expect range-bound, mean-reverting action
                    - Good for: Premium selling, iron condors
                    """
                elif gex.net_gex > 0:
                    regime = "🟡 **MODERATE POSITIVE GAMMA**"
                    interpretation = """
                    - Mild volatility dampening
                    - Some dealer hedging flows
                    - Trending moves possible but limited
                    - Good for: Selective premium selling
                    """
                elif gex.net_gex > -1e9:
                    regime = "🟠 **MODERATE NEGATIVE GAMMA**"
                    interpretation = """
                    - Volatility amplification beginning
                    - Dealers chase moves (buy rallies, sell dips)
                    - Trending moves more likely
                    - Good for: Directional plays with stops
                    """
                else:
                    regime = "🔴 **HIGH NEGATIVE GAMMA**"
                    interpretation = """
                    - Maximum volatility regime
                    - Dealers heavily short gamma
                    - Explosive moves in both directions
                    - Good for: Squeeze plays, momentum trades
                    """
                
                st.markdown(f"""
                <div class='info-box'>
                    <h3>{regime}</h3>
                    <div style='color: rgba(255,255,255,0.8); line-height: 1.6;'>
                        {interpretation}
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    # Tab 3: All Trade Setups
    with tab3:
        st.markdown("## 💎 All Trade Setups")
        
        if st.session_state.all_setups:
            # Group setups by strategy type
            strategy_groups = {}
            for setup in st.session_state.all_setups:
                strategy = setup['strategy']
                if strategy not in strategy_groups:
                    strategy_groups[strategy] = []
                strategy_groups[strategy].append(setup)
            
            # Display setups by strategy
            for strategy, setups in strategy_groups.items():
                st.markdown(f"### {strategy}")
                
                for setup in setups[:5]:  # Show top 5 per strategy
                    if setup['confidence'] >= confidence_threshold:
                        st.markdown(f"""
                        <div class='trade-setup-card'>
                            <div style='display: flex; justify-content: space-between; align-items: center;'>
                                <h4 style='margin: 0;'>{setup['symbol']}</h4>
                                <span class='confidence-badge {"confidence-high" if setup["confidence"] > 80 else "confidence-medium" if setup["confidence"] > 70 else "confidence-low"}'>
                                    {setup['confidence']:.0f}% Confidence
                                </span>
                            </div>
                            <p style='margin: 10px 0;'>{setup['description']}</p>
                            <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 15px;'>
                                <div><strong>Entry:</strong> {setup['entry_criteria']}</div>
                                <div><strong>Time:</strong> {setup['days_to_expiry']}</div>
                                <div><strong>Size:</strong> {setup['position_size']}</div>
                                <div><strong>R/R:</strong> {setup.get('risk_reward', 0):.2f}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.info("No trade setups found. Try adjusting your watchlist or confidence threshold.")
    
    # Tab 4: Positions (Portfolio Management)
    with tab4:
        st.markdown("## 📈 Portfolio & Position Management")
        
        # Portfolio metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Open Positions", len(st.session_state.portfolio['positions']))
        
        with col2:
            total_exposure = sum([p['value'] for p in st.session_state.portfolio['positions']])
            st.metric("Total Exposure", f"${total_exposure:,.0f}")
        
        with col3:
            utilization = (total_exposure / st.session_state.portfolio['total_value'] * 100) if st.session_state.portfolio['total_value'] > 0 else 0
            st.metric("Capital Utilization", f"{utilization:.1f}%")
        
        with col4:
            st.metric("Daily P&L", 
                     f"${st.session_state.portfolio['daily_pnl']:+,.0f}",
                     delta=f"{st.session_state.portfolio['daily_pnl']/st.session_state.portfolio['total_value']*100:+.2f}%")
        
        if st.session_state.portfolio['positions']:
            st.markdown("### Active Positions")
            positions_df = pd.DataFrame(st.session_state.portfolio['positions'])
            st.dataframe(positions_df, use_container_width=True)
        else:
            st.info("No active positions. Start trading from the Top Opportunities tab!")
    
    # Tab 5: Alerts
    with tab5:
        st.markdown("## ⚠️ Trading Alerts")
        
        # Generate alerts based on current data
        alerts = []
        
        for symbol, gex in st.session_state.all_gex_data.items():
            # Check for critical GEX levels
            if gex.net_gex and gex.net_gex < -1e9:
                alerts.append({
                    'priority': 'HIGH',
                    'symbol': symbol,
                    'type': 'NEGATIVE_GEX',
                    'message': f'{symbol}: Net GEX below -1B threshold ({gex.net_gex/1e9:.2f}B)',
                    'action': 'Consider long volatility positions'
                })
            
            # Check distance to gamma flip
            if gex.gamma_flip and gex.spot_price:
                distance = abs(gex.spot_price - gex.gamma_flip) / gex.spot_price * 100
                if distance < 0.5:
                    alerts.append({
                        'priority': 'HIGH',
                        'symbol': symbol,
                        'type': 'NEAR_FLIP',
                        'message': f'{symbol}: Within {distance:.2f}% of gamma flip',
                        'action': 'Prepare for volatility regime change'
                    })
            
            # Check for extreme GEX
            if gex.net_gex and abs(gex.net_gex) > 5e9:
                alerts.append({
                    'priority': 'MEDIUM',
                    'symbol': symbol,
                    'type': 'EXTREME_GEX',
                    'message': f'{symbol}: Extreme GEX level ({gex.net_gex/1e9:.2f}B)',
                    'action': 'Market at extremes - prepare for reversal'
                })
        
        # Display alerts by priority
        high_alerts = [a for a in alerts if a['priority'] == 'HIGH']
        medium_alerts = [a for a in alerts if a['priority'] == 'MEDIUM']
        
        if high_alerts:
            st.markdown("### 🔴 High Priority Alerts")
            for alert in high_alerts:
                st.markdown(f"""
                <div class='alert-high new-alert'>
                    <div style='display: flex; justify-content: space-between;'>
                        <div>
                            <strong>{alert['symbol']} - {alert['type']}</strong><br/>
                            {alert['message']}<br/>
                            <em>Action: {alert['action']}</em>
                        </div>
                        <div style='font-size: 12px; color: rgba(255,255,255,0.6);'>
                            {datetime.now().strftime('%H:%M:%S')}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        if medium_alerts:
            st.markdown("### 🟡 Medium Priority Alerts")
            for alert in medium_alerts:
                st.markdown(f"""
                <div class='alert-medium new-alert'>
                    <div style='display: flex; justify-content: space-between;'>
                        <div>
                            <strong>{alert['symbol']} - {alert['type']}</strong><br/>
                            {alert['message']}<br/>
                            <em>Action: {alert['action']}</em>
                        </div>
                        <div style='font-size: 12px; color: rgba(255,255,255,0.6);'>
                            {datetime.now().strftime('%H:%M:%S')}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        if not alerts:
            st.success("✅ No active alerts. All systems normal.")
        
        # Alert configuration
        st.divider()
        st.markdown("### 🔔 Alert Settings")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.checkbox("GEX Threshold Alerts", value=True, key="alert_gex")
            st.checkbox("Wall Breach Alerts", value=True, key="alert_wall")
        
        with col2:
            st.checkbox("Gamma Flip Alerts", value=True, key="alert_flip")
            st.checkbox("Volume Spike Alerts", value=False, key="alert_volume")
        
        with col3:
            st.checkbox("Email Notifications", value=False, key="alert_email")
            st.checkbox("Sound Alerts", value=False, key="alert_sound")
    
    # Tab 6: Performance Analytics
    with tab6:
        st.markdown("## 📉 Performance Analytics")
        
        # Performance metrics
        if st.session_state.portfolio['trade_history']:
            trades_df = pd.DataFrame(st.session_state.portfolio['trade_history'])
            
            # Calculate metrics
            total_trades = len(trades_df)
            winning_trades = len(trades_df[trades_df['pnl'] > 0]) if 'pnl' in trades_df.columns else 0
            win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Trades", total_trades)
            
            with col2:
                st.metric("Win Rate", f"{win_rate:.1f}%",
                         delta="Above average" if win_rate > 50 else "Below average")
            
            with col3:
                if 'pnl' in trades_df.columns and len(trades_df[trades_df['pnl'] > 0]) > 0:
                    avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean()
                    st.metric("Avg Win", f"${avg_win:,.2f}")
                else:
                    st.metric("Avg Win", "$0")
            
            with col4:
                if 'pnl' in trades_df.columns and len(trades_df[trades_df['pnl'] < 0]) > 0:
                    avg_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].mean())
                    st.metric("Avg Loss", f"${avg_loss:,.2f}")
                else:
                    st.metric("Avg Loss", "$0")
            
            # P&L Chart
            st.markdown("### 📊 Cumulative P&L")
            
            if 'pnl' in trades_df.columns:
                trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
                
                fig = go.Figure()
                
                # Add gradient area chart
                fig.add_trace(go.Scatter(
                    x=trades_df.index,
                    y=trades_df['cumulative_pnl'],
                    mode='lines',
                    name='Cumulative P&L',
                    line=dict(
                        color='#00D2FF' if trades_df['cumulative_pnl'].iloc[-1] > 0 else '#f45c43',
                        width=3
                    ),
                    fill='tozeroy',
                    fillcolor='rgba(0, 210, 255, 0.1)' if trades_df['cumulative_pnl'].iloc[-1] > 0 else 'rgba(244, 92, 67, 0.1)',
                    hovertemplate='Trade #%{x}<br>Cumulative P&L: $%{y:,.2f}<extra></extra>'
                ))
                
                # Add markers for individual trades
                colors = ['#38ef7d' if x > 0 else '#f45c43' for x in trades_df['pnl']]
                fig.add_trace(go.Scatter(
                    x=trades_df.index,
                    y=trades_df['cumulative_pnl'],
                    mode='markers',
                    name='Trades',
                    marker=dict(
                        color=colors,
                        size=10,
                        line=dict(color='white', width=1)
                    ),
                    hovertemplate='Trade #%{x}<br>P&L: $%{customdata:,.2f}<extra></extra>',
                    customdata=trades_df['pnl']
                ))
                
                fig.update_layout(
                    height=400,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='rgba(255,255,255,0.9)'),
                    xaxis=dict(
                        gridcolor='rgba(255,255,255,0.1)',
                        title="Trade Number"
                    ),
                    yaxis=dict(
                        gridcolor='rgba(255,255,255,0.1)',
                        title="Cumulative P&L ($)"
                    ),
                    hovermode='x unified',
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Strategy Performance Breakdown
            if 'strategy' in trades_df.columns:
                st.markdown("### 🎯 Strategy Performance")
                
                strategy_stats = trades_df.groupby('strategy').agg({
                    'pnl': ['count', 'sum', 'mean'],
                    'symbol': 'first'
                }).round(2)
                
                strategy_stats.columns = ['Trades', 'Total P&L', 'Avg P&L', 'Sample Symbol']
                st.dataframe(
                    strategy_stats.style.format({
                        'Total P&L': '${:,.2f}',
                        'Avg P&L': '${:,.2f}'
                    }),
                    use_container_width=True
                )
        else:
            st.info("No trading history yet. Start trading to see performance analytics!")
            
            # Show demo chart
            st.markdown("### 📊 Sample Performance Chart")
            
            # Generate sample data
            sample_trades = 50
            sample_pnl = np.random.randn(sample_trades) * 500 + 100
            sample_cumulative = np.cumsum(sample_pnl)
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=list(range(sample_trades)),
                y=sample_cumulative,
                mode='lines',
                name='Sample P&L',
                line=dict(color='#00D2FF', width=3),
                fill='tozeroy',
                fillcolor='rgba(0, 210, 255, 0.1)'
            ))
            
            fig.update_layout(
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='rgba(255,255,255,0.9)'),
                xaxis=dict(
                    gridcolor='rgba(255,255,255,0.1)',
                    title="Trade Number"
                ),
                yaxis=dict(
                    gridcolor='rgba(255,255,255,0.1)',
                    title="Cumulative P&L ($)"
                ),
                title="Your performance chart will appear here"
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Tab 7: Strategy Guide
    with tab7:
        st.markdown("## 🔍 Strategy Guide & Education")
        
        strategy_option = st.selectbox(
            "Select Topic",
            ["Quick Start Guide", "Squeeze Plays", "Premium Selling", "Iron Condors", 
             "Risk Management", "GEX Fundamentals", "Market Regimes"]
        )
        
        if strategy_option == "Quick Start Guide":
            st.markdown("### 🚀 Quick Start Guide")
            
            st.markdown("#### 1. Set Up Your Watchlist")
            st.info("""
            • Add symbols you want to monitor in the sidebar
            • Include major indices (SPY, QQQ) and your favorite stocks  
            • The system will scan all symbols simultaneously
            """)
            
            st.markdown("#### 2. Configure Risk Settings")
            st.info("""
            • Set maximum position size (recommended: 3-5%)
            • Define minimum confidence threshold (recommended: 65-75%)
            • Adjust based on your risk tolerance
            """)
            
            st.markdown("#### 3. Monitor Top Opportunities")
            st.success("""
            • Check the "Top Opportunities" tab for best setups
            • Setups are ranked by confidence across ALL symbols
            • Green badges = high confidence (>80%)
            """)
            
            st.markdown("#### 4. Execute Trades")
            st.info("""
            • Review setup details carefully
            • Check entry criteria and position sizing
            • Click "Execute Trade" to add to portfolio
            """)
            
            st.markdown("#### 5. Manage Positions")
            st.info("""
            • Monitor active positions in the Positions tab
            • Set alerts for important levels
            • Follow exit criteria for each strategy
            """)
        
        elif strategy_option == "Squeeze Plays":
            st.markdown("### 🚀 Squeeze Play Strategies")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Negative GEX Squeeze (Long Calls)")
                
                st.markdown("**Setup Conditions:**")
                st.warning("""
                • Net GEX < -1B (SPY) or < -500M (QQQ) or < -100M (stocks)
                • Price 0.5-1.5% below gamma flip point
                • Strong put wall support within 1% below
                """)
                
                st.markdown("**Entry:**")
                st.info("""
                • Buy ATM or first OTM call above flip
                • Use 2-5 DTE options for maximum gamma
                • Size for potential 100% loss (3% max)
                """)
                
                st.markdown("**Exit:**")
                st.success("""
                • Target: Gamma flip point or above
                • Stop: Break below put wall support
                • Time stop: Close if <1 DTE remains
                """)
            
            with col2:
                st.markdown("#### Positive GEX Breakdown (Long Puts)")
                
                st.markdown("**Setup Conditions:**")
                st.warning("""
                • Net GEX > 2B (SPY) or > 1B (QQQ) or > 200M (stocks)
                • Price hovering near flip (within 0.3%)
                • Recent rejection from call wall
                """)
                
                st.markdown("**Entry:**")
                st.info("""
                • Buy ATM or first OTM put below flip
                • Use 3-7 DTE options
                • Size for 3% max portfolio risk
                """)
                
                st.markdown("**Exit:**")
                st.success("""
                • Target: First strike below flip
                • Stop: Break above call wall
                • Time stop: Close if <1 DTE remains
                """)
        
        elif strategy_option == "Premium Selling":
            st.markdown("### 💰 Premium Selling Strategies")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Call Selling at Resistance")
                
                st.warning("""
                **Setup Requirements:**
                • Net GEX > 3B with strong call wall
                • Wall strength > 500M gamma
                • Price 0.5-2% below wall
                """)
                
                st.info("""
                **Entry & Management:**
                • Sell calls at or above wall strike
                • Use 0-2 DTE for rapid decay
                • Size for 5% max capital risk
                • Close at 50% profit or approaching wall
                """)
            
            with col2:
                st.markdown("#### Put Selling at Support")
                
                st.warning("""
                **Setup Requirements:**
                • Strong put wall > 500M gamma
                • Price at least 1% above wall
                • Positive net GEX environment
                """)
                
                st.info("""
                **Entry & Management:**
                • Sell puts at or below wall strike
                • Use 2-5 DTE options
                • Size for 5% max capital risk
                • Close at 50% profit or define max loss
                """)
        
        elif strategy_option == "Iron Condors":
            st.markdown("### 🦅 Iron Condor Strategies")
            
            st.markdown("#### Standard Iron Condor")
            st.info("""
            **Setup Requirements:**
            • Net GEX > 1B (positive gamma environment)
            • Call and put walls > 3% apart
            • Low IV rank (<50th percentile)
            
            **Construction:**
            • Short strikes at gamma walls
            • Long strikes beyond major gamma concentrations
            • Use 5-10 DTE for optimal theta/gamma ratio
            • Size for 2% max portfolio loss
            """)
            
            st.markdown("#### Broken Wing Adjustments")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.success("""
                **Bullish Bias (Put gamma > Call gamma):**
                • Wider put spread (1.5x)
                • Narrower call spread (0.75x)
                • Collect more premium on put side
                """)
            
            with col2:
                st.warning("""
                **Bearish Bias (Call gamma > Put gamma):**
                • Wider call spread (1.5x)
                • Narrower put spread (0.75x)
                • Collect more premium on call side
                """)
        
        elif strategy_option == "Risk Management":
            st.markdown("### ⚖️ Risk Management Rules")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Position Sizing")
                st.error("""
                **Maximum Allocations:**
                • Squeeze Plays: 3% of capital
                • Premium Selling: 5% of capital
                • Iron Condors: Size for 2% max loss
                • Total Directional: 15% exposure
                • Portfolio Maximum: 50% invested
                """)
                
                st.markdown("#### Stop Losses")
                st.warning("""
                **Exit Rules:**
                • Long Options: 50% loss or wall breach
                • Short Options: 100% loss or defined risk
                • Iron Condors: Threatened strike
                • Time Stop: Close if <1 DTE
                """)
            
            with col2:
                st.markdown("#### Portfolio Limits")
                st.info("""
                **Risk Controls:**
                • Maximum 5-7 concurrent positions
                • Daily loss limit: 5% of portfolio
                • Correlation limit: Max 3 similar setups
                • Reduce size in high volatility
                """)
                
                st.markdown("#### Adjustment Rules")
                st.success("""
                **When to Adjust:**
                • Roll if breached with >3 DTE
                • Add hedges on regime change
                • Reduce on 2% daily loss
                • Close all if flip breached
                """)
        
        elif strategy_option == "GEX Fundamentals":
            st.markdown("### 📚 Understanding Gamma Exposure (GEX)")
            
            st.markdown("#### What is GEX?")
            st.info("""
            GEX measures the aggregate gamma exposure of options dealers/market makers. 
            It indicates how much dealers need to hedge as the underlying price moves.
            
            **The Formula:** GEX = Spot Price × Gamma × Open Interest × 100
            """)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Positive GEX Environment")
                st.success("""
                **Characteristics:**
                • Dealers are long gamma
                • They sell rallies and buy dips
                • Volatility suppression
                • Mean reversion likely
                • Range-bound trading
                
                **Best Strategies:**
                • Premium selling
                • Iron condors
                • Mean reversion trades
                """)
            
            with col2:
                st.markdown("#### Negative GEX Environment")
                st.error("""
                **Characteristics:**
                • Dealers are short gamma
                • They buy rallies and sell dips
                • Volatility amplification
                • Trending moves likely
                • Explosive price action
                
                **Best Strategies:**
                • Squeeze plays
                • Momentum trades
                • Directional options
                """)
            
            st.markdown("#### Key Levels to Watch")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.warning("""
                **Gamma Flip Point**
                • Where net GEX = 0
                • Regime change level
                • Critical S/R level
                • Volatility shifts here
                """)
            
            with col2:
                st.success("""
                **Call Walls**
                • Resistance levels
                • High positive gamma
                • Dealers sell here
                • Price often reverses
                """)
            
            with col3:
                st.error("""
                **Put Walls**
                • Support levels
                • High negative gamma
                • Dealers buy here
                • Bounces likely
                """)
        
        elif strategy_option == "Market Regimes":
            st.markdown("### 🎭 Market Regime Playbook")
            
            # Create a visual regime guide
            regimes = {
                "🔴 High Negative GEX (<-1B)": {
                    "characteristics": "Maximum volatility, explosive moves, dealers chase price",
                    "strategies": "Long volatility, squeeze plays, momentum trades",
                    "avoid": "Premium selling without protection",
                    "color": "error"
                },
                "🟠 Moderate Negative (-1B to 0)": {
                    "characteristics": "Elevated volatility, trending likely, some amplification",
                    "strategies": "Directional plays with stops, moderate squeezes",
                    "avoid": "Iron condors, naked short options",
                    "color": "warning"
                },
                "🟡 Low Positive (0 to 1B)": {
                    "characteristics": "Balanced market, mild suppression, selective opportunities",
                    "strategies": "Selective premium selling, tight condors",
                    "avoid": "Large squeeze plays",
                    "color": "info"
                },
                "🟢 High Positive (>1B)": {
                    "characteristics": "Volatility suppression, range-bound, mean reversion",
                    "strategies": "Premium selling, iron condors, fade extremes",
                    "avoid": "Breakout trades, long volatility",
                    "color": "success"
                }
            }
            
            for regime, details in regimes.items():
                with st.expander(regime, expanded=True):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("**Characteristics:**")
                        st.write(details["characteristics"])
                    
                    with col2:
                        st.markdown("**Best Strategies:**")
                        st.write(details["strategies"])
                    
                    with col3:
                        st.markdown("**Avoid:**")
                        st.write(details["avoid"])
            
            st.markdown("#### Quick Reference Table")
            
            regime_df = pd.DataFrame({
                "Net GEX": ["< -1B", "-1B to 0", "0 to 1B", "> 1B"],
                "Regime": ["High Negative", "Moderate Negative", "Low Positive", "High Positive"],
                "Volatility": ["Maximum", "Elevated", "Normal", "Suppressed"],
                "Best Play": ["Squeeze Long", "Directional", "Selective", "Premium Sell"],
                "Risk Level": ["🔴 High", "🟠 Medium-High", "🟡 Medium", "🟢 Low"]
            })
            
            st.dataframe(regime_df, use_container_width=True, hide_index=True)
    
    # Footer with gradient and animation
    st.divider()
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.session_state.last_update:
            update_time = st.session_state.last_update.strftime('%Y-%m-%d %H:%M:%S')
        else:
            update_time = "Never"
        
        st.markdown(f"""
        <div style='text-align: center; padding: 20px;'>
            <p style='color: rgba(255,255,255,0.6); margin: 0;'>
                Last Updated: {update_time}
            </p>
            <p style='color: rgba(255,255,255,0.4); font-size: 12px; margin-top: 10px;'>
                GEX Trading Dashboard Pro v3.0 | Scanning {len(st.session_state.watchlist)} Symbols
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Auto-refresh logic
    if auto_refresh:
        time_module.sleep(refresh_interval * 60)
        st.rerun()

if __name__ == "__main__":
    main()
