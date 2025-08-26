"""
GEX Trading Dashboard - Complete Gamma Exposure Analysis Platform
Author: GEX Trading System Expert
Version: 3.0.0
Description: Comprehensive dashboard for gamma exposure analysis, trade setup detection,
             and position management with TradingVolatility API integration
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
import requests
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass, field
import sqlite3
import hashlib
from functools import lru_cache
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

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
    
    .metric-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    }
    
    .trade-setup {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        margin: 15px 0;
        border-left: 5px solid #00ff87;
        color: white;
    }
    
    .success-alert {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    
    .warning-alert {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        color: #333;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    
    .danger-alert {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    
    .stSidebar {
        background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%);
    }
    
    .valid-symbol {
        color: #00ff87;
        font-weight: bold;
    }
    
    .invalid-symbol {
        color: #ff6b6b;
        font-weight: bold;
    }
    
    div[data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 15px;
        border-radius: 15px;
        margin: 10px 0;
    }
    
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 10px;
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

if 'gex_data' not in st.session_state:
    st.session_state.gex_data = {}

if 'last_update' not in st.session_state:
    st.session_state.last_update = None

if 'alerts' not in st.session_state:
    st.session_state.alerts = []

if 'trade_setups' not in st.session_state:
    st.session_state.trade_setups = []

if 'api_settings' not in st.session_state:
    st.session_state.api_settings = {
        'api_key': '',
        'username': 'I-RWFNBLR2S1DP',
        'base_url': 'https://stocks.tradingvolatility.net/api'
    }

# ======================== TRADING VOLATILITY API CLIENT ========================

class TradingVolatilityAPI:
    """Client for TradingVolatility.net API"""
    
    def __init__(self, api_key: str, username: str):
        self.api_key = api_key
        self.username = username
        self.base_url = "https://stocks.tradingvolatility.net/api"
        self.headers = {
            'Authorization': f'Bearer {api_key}',
            'User-Agent': f'GEX-Dashboard/{username}'
        }
        self.last_request_time = 0
        self.min_request_interval = 3  # 3 seconds between requests
    
    def _rate_limit(self):
        """Implement rate limiting"""
        now = time_module.time()
        time_since_last = now - self.last_request_time
        if time_since_last < self.min_request_interval:
            time_module.sleep(self.min_request_interval - time_since_last)
        self.last_request_time = time_module.time()
    
    def get_net_gex(self, symbol: str) -> Dict:
        """Get net GEX data for a symbol"""
        self._rate_limit()
        
        try:
            url = f"{self.base_url}/net-gex/{symbol}"
            response = requests.get(url, headers=self.headers, timeout=30)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"API error for {symbol}: {response.status_code}")
                return {"error": f"API error: {response.status_code}"}
                
        except Exception as e:
            logger.error(f"Error fetching GEX for {symbol}: {str(e)}")
            return {"error": str(e)}
    
    def get_gex_levels(self, symbol: str) -> Dict:
        """Get GEX levels CSV data"""
        self._rate_limit()
        
        try:
            url = f"{self.base_url}/gex-levels-csv/{symbol}"
            response = requests.get(url, headers=self.headers, timeout=30)
            
            if response.status_code == 200:
                return {"data": response.text, "symbol": symbol}
            else:
                logger.error(f"GEX levels error for {symbol}: {response.status_code}")
                return {"error": f"API error: {response.status_code}"}
                
        except Exception as e:
            logger.error(f"Error fetching GEX levels for {symbol}: {str(e)}")
            return {"error": str(e)}
    
    def get_options_flow(self, symbol: str) -> Dict:
        """Get options flow data"""
        self._rate_limit()
        
        try:
            url = f"{self.base_url}/options-flow/{symbol}"
            response = requests.get(url, headers=self.headers, timeout=30)
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"API error: {response.status_code}"}
                
        except Exception as e:
            return {"error": str(e)}

# ======================== GEX CALCULATION ENGINE ========================

@dataclass
class GEXProfile:
    """Complete GEX profile for a symbol"""
    symbol: str
    spot_price: float
    net_gex: float
    gamma_flip: float
    call_walls: List[Dict]
    put_walls: List[Dict]
    gex_by_strike: List[Dict]
    regime: str
    last_updated: datetime
    api_data: Dict = field(default_factory=dict)

class ComprehensiveGEXCalculator:
    """Advanced GEX calculator with TradingVolatility API integration"""
    
    def __init__(self, api_client: TradingVolatilityAPI):
        self.api_client = api_client
        self.profiles = {}
        
    def calculate_comprehensive_gex(self, symbol: str) -> Optional[GEXProfile]:
        """Calculate comprehensive GEX profile using API data"""
        try:
            # Get data from TradingVolatility API
            net_gex_data = self.api_client.get_net_gex(symbol)
            gex_levels_data = self.api_client.get_gex_levels(symbol)
            
            if "error" in net_gex_data:
                logger.error(f"Failed to get net GEX for {symbol}: {net_gex_data['error']}")
                return None
                
            if "error" in gex_levels_data:
                logger.error(f"Failed to get GEX levels for {symbol}: {gex_levels_data['error']}")
                return None
            
            # Parse the data
            profile = self._parse_api_data(symbol, net_gex_data, gex_levels_data)
            
            if profile:
                self.profiles[symbol] = profile
            
            return profile
            
        except Exception as e:
            logger.error(f"Error calculating GEX for {symbol}: {str(e)}")
            return None
    
    def _parse_api_data(self, symbol: str, net_gex_data: Dict, gex_levels_data: Dict) -> Optional[GEXProfile]:
        """Parse API data into GEX profile"""
        try:
            # Get current price (fallback to Yahoo if API doesn't provide)
            spot_price = self._get_current_price(symbol)
            
            # Parse GEX levels CSV
            levels_text = gex_levels_data.get('data', '')
            gex_levels = self._parse_gex_levels_csv(levels_text)
            
            if not gex_levels:
                logger.error(f"Failed to parse GEX levels for {symbol}")
                return None
            
            # Extract key levels
            gamma_flip = gex_levels.get('gamma_flip', spot_price)
            
            # Create call and put walls from the levels
            call_walls = []
            put_walls = []
            
            # Parse GEX levels for walls
            for i in range(1, 5):  # GEX_1 through GEX_4
                gex_key = f'gex_{i}'
                if gex_key in gex_levels:
                    strike = gex_levels[gex_key]
                    if strike > gamma_flip:
                        call_walls.append({
                            'strike': strike,
                            'gex': 1000000 * (5-i),  # Estimate strength
                            'strength': 'strong' if i <= 2 else 'moderate'
                        })
                    else:
                        put_walls.append({
                            'strike': strike,
                            'gex': -1000000 * (5-i),  # Estimate strength
                            'strength': 'strong' if i <= 2 else 'moderate'
                        })
            
            # Calculate net GEX estimate
            net_gex = net_gex_data.get('net_gex', 0)
            if isinstance(net_gex, str):
                try:
                    net_gex = float(net_gex.replace('B', '')) * 1e9
                except:
                    net_gex = 0
            
            # Create GEX by strike data
            gex_by_strike = []
            all_strikes = [gamma_flip] + [w['strike'] for w in call_walls + put_walls]
            
            for strike in sorted(set(all_strikes)):
                gex_value = 0
                
                # Find matching wall
                for wall in call_walls:
                    if wall['strike'] == strike:
                        gex_value = wall['gex']
                        break
                
                for wall in put_walls:
                    if wall['strike'] == strike:
                        gex_value = wall['gex']
                        break
                
                gex_by_strike.append({
                    'strike': strike,
                    'gex': gex_value,
                    'cumulative_gex': 0  # Will calculate this
                })
            
            # Calculate cumulative GEX
            cumulative = 0
            for entry in gex_by_strike:
                cumulative += entry['gex']
                entry['cumulative_gex'] = cumulative
            
            # Determine regime
            regime = 'positive' if net_gex > 0 else 'negative'
            
            return GEXProfile(
                symbol=symbol,
                spot_price=spot_price,
                net_gex=net_gex,
                gamma_flip=gamma_flip,
                call_walls=sorted(call_walls, key=lambda x: x['gex'], reverse=True),
                put_walls=sorted(put_walls, key=lambda x: x['gex']),
                gex_by_strike=gex_by_strike,
                regime=regime,
                last_updated=datetime.now(),
                api_data={'net_gex': net_gex_data, 'levels': gex_levels}
            )
            
        except Exception as e:
            logger.error(f"Error parsing API data for {symbol}: {str(e)}")
            return None
    
    def _parse_gex_levels_csv(self, csv_text: str) -> Dict:
        """Parse GEX levels CSV text"""
        try:
            levels = {}
            lines = csv_text.strip().split('\n')
            
            for line in lines:
                parts = line.split(',')
                if len(parts) >= 2:
                    key = parts[0].strip().lower().replace(' ', '_')
                    value_str = parts[1].strip()
                    
                    try:
                        value = float(value_str)
                        levels[key] = value
                    except ValueError:
                        continue
            
            return levels
            
        except Exception as e:
            logger.error(f"Error parsing GEX levels CSV: {str(e)}")
            return {}
    
    def _get_current_price(self, symbol: str) -> float:
        """Get current price using yfinance fallback"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1d", interval="1m")
            if not hist.empty:
                return float(hist['Close'].iloc[-1])
            else:
                # Fallback prices for major symbols
                fallback_prices = {
                    'SPY': 450.0, 'QQQ': 350.0, 'IWM': 200.0,
                    'AAPL': 180.0, 'MSFT': 400.0, 'GOOGL': 140.0,
                    'AMZN': 140.0, 'TSLA': 250.0, 'META': 500.0,
                    'NVDA': 900.0
                }
                return fallback_prices.get(symbol, 100.0)
                
        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {str(e)}")
            return 100.0

# ======================== SETUP DETECTION ENGINE ========================

class SetupDetector:
    """Detect trading setups from GEX profiles"""
    
    def __init__(self):
        self.setup_types = {
            'squeeze_long_calls': self._detect_squeeze_long_calls,
            'squeeze_long_puts': self._detect_squeeze_long_puts,
            'wall_compression': self._detect_wall_compression,
            'call_selling': self._detect_call_selling,
            'put_selling': self._detect_put_selling,
            'iron_condor': self._detect_iron_condor
        }
    
    def detect_all_setups(self, profile: GEXProfile) -> List[Dict]:
        """Detect all possible setups for a profile"""
        setups = []
        
        for setup_name, detector in self.setup_types.items():
            setup = detector(profile)
            if setup:
                setup['type'] = setup_name
                setup['symbol'] = profile.symbol
                setup['detected_at'] = datetime.now()
                setups.append(setup)
        
        return sorted(setups, key=lambda x: x.get('confidence', 0), reverse=True)
    
    def _detect_squeeze_long_calls(self, profile: GEXProfile) -> Optional[Dict]:
        """Detect negative GEX squeeze setup (long calls)"""
        if profile.net_gex >= -1e9:  # Need negative GEX < -1B
            return None
        
        # Price should be below gamma flip
        distance_to_flip = (profile.gamma_flip - profile.spot_price) / profile.spot_price
        if distance_to_flip < 0.005 or distance_to_flip > 0.02:  # 0.5% to 2%
            return None
        
        # Check for put wall support
        put_support = None
        if profile.put_walls:
            put_support = profile.put_walls[0]['strike']
            support_distance = (profile.spot_price - put_support) / profile.spot_price
            if support_distance > 0.015:  # Support too far
                return None
        
        confidence = 75
        if distance_to_flip < 0.01:
            confidence += 10
        if put_support and (profile.spot_price - put_support) / profile.spot_price < 0.01:
            confidence += 10
        if profile.net_gex < -2e9:
            confidence += 5
        
        return {
            'strategy': 'Long Calls (Negative GEX Squeeze)',
            'entry_strike': profile.gamma_flip,
            'target': profile.call_walls[0]['strike'] if profile.call_walls else profile.gamma_flip * 1.02,
            'stop_loss': put_support if put_support else profile.spot_price * 0.98,
            'confidence': min(confidence, 95),
            'reasoning': f'Negative GEX {profile.net_gex/1e9:.1f}B, price {distance_to_flip:.1%} below flip',
            'risk_reward': 2.5,
            'max_loss_pct': 50,
            'position_size_pct': 3
        }
    
    def _detect_squeeze_long_puts(self, profile: GEXProfile) -> Optional[Dict]:
        """Detect positive GEX breakdown setup (long puts)"""
        if profile.net_gex <= 2e9:  # Need positive GEX > 2B
            return None
        
        # Price should be near gamma flip (within 0.3%)
        distance_to_flip = abs(profile.gamma_flip - profile.spot_price) / profile.spot_price
        if distance_to_flip > 0.005:  # Must be very close
            return None
        
        confidence = 70
        if distance_to_flip < 0.002:
            confidence += 15
        if profile.net_gex > 3e9:
            confidence += 10
        
        return {
            'strategy': 'Long Puts (Positive GEX Breakdown)',
            'entry_strike': profile.gamma_flip,
            'target': profile.put_walls[0]['strike'] if profile.put_walls else profile.gamma_flip * 0.98,
            'stop_loss': profile.call_walls[0]['strike'] if profile.call_walls else profile.spot_price * 1.02,
            'confidence': min(confidence, 95),
            'reasoning': f'Positive GEX {profile.net_gex/1e9:.1f}B, price at flip point',
            'risk_reward': 2.0,
            'max_loss_pct': 50,
            'position_size_pct': 3
        }
    
    def _detect_wall_compression(self, profile: GEXProfile) -> Optional[Dict]:
        """Detect gamma wall compression setup"""
        if not profile.call_walls or not profile.put_walls:
            return None
        
        call_wall = profile.call_walls[0]['strike']
        put_wall = profile.put_walls[0]['strike']
        
        compression_ratio = (call_wall - put_wall) / profile.spot_price
        if compression_ratio > 0.025:  # Walls must be < 2.5% apart
            return None
        
        confidence = 80 if compression_ratio < 0.015 else 70
        
        # Determine direction based on current position
        if profile.spot_price < (call_wall + put_wall) / 2:
            direction = 'calls'
            entry = call_wall
        else:
            direction = 'puts'
            entry = put_wall
        
        return {
            'strategy': f'Wall Compression - Long {direction.capitalize()}',
            'entry_strike': entry,
            'target': call_wall if direction == 'calls' else put_wall,
            'stop_loss': put_wall if direction == 'calls' else call_wall,
            'confidence': confidence,
            'reasoning': f'Walls compressed to {compression_ratio:.1%} range',
            'risk_reward': 3.0,
            'max_loss_pct': 100,
            'position_size_pct': 2
        }
    
    def _detect_call_selling(self, profile: GEXProfile) -> Optional[Dict]:
        """Detect call selling opportunity at resistance"""
        if profile.net_gex <= 3e9 or not profile.call_walls:
            return None
        
        call_wall = profile.call_walls[0]
        distance_to_wall = (call_wall['strike'] - profile.spot_price) / profile.spot_price
        
        if distance_to_wall < 0.005 or distance_to_wall > 0.02:
            return None
        
        confidence = 75
        if call_wall['strength'] == 'strong':
            confidence += 10
        if distance_to_wall < 0.01:
            confidence += 10
        
        return {
            'strategy': 'Sell Calls at Wall',
            'entry_strike': call_wall['strike'],
            'target': profile.spot_price,
            'stop_loss': call_wall['strike'] * 1.01,
            'confidence': confidence,
            'reasoning': f'Strong call wall at {call_wall["strike"]}, positive GEX {profile.net_gex/1e9:.1f}B',
            'risk_reward': 1.5,
            'max_loss_pct': 100,
            'position_size_pct': 5
        }
    
    def _detect_put_selling(self, profile: GEXProfile) -> Optional[Dict]:
        """Detect put selling opportunity at support"""
        if not profile.put_walls:
            return None
        
        put_wall = profile.put_walls[0]
        distance_to_wall = (profile.spot_price - put_wall['strike']) / profile.spot_price
        
        if distance_to_wall < 0.01 or distance_to_wall > 0.03:
            return None
        
        confidence = 70
        if put_wall['strength'] == 'strong':
            confidence += 10
        if profile.net_gex > 1e9:
            confidence += 10
        
        return {
            'strategy': 'Sell Puts at Wall',
            'entry_strike': put_wall['strike'],
            'target': profile.spot_price,
            'stop_loss': put_wall['strike'] * 0.99,
            'confidence': confidence,
            'reasoning': f'Strong put wall at {put_wall["strike"]}',
            'risk_reward': 1.5,
            'max_loss_pct': 100,
            'position_size_pct': 5
        }
    
    def _detect_iron_condor(self, profile: GEXProfile) -> Optional[Dict]:
        """Detect iron condor setup"""
        if not profile.call_walls or not profile.put_walls or profile.net_gex <= 1e9:
            return None
        
        call_wall = profile.call_walls[0]['strike']
        put_wall = profile.put_walls[0]['strike']
        
        range_pct = (call_wall - put_wall) / profile.spot_price
        if range_pct < 0.03 or range_pct > 0.08:  # 3% to 8% range
            return None
        
        confidence = 65
        if 0.04 <= range_pct <= 0.06:  # Optimal range
            confidence += 15
        
        return {
            'strategy': 'Iron Condor',
            'entry_strike': f"{put_wall:.0f}/{call_wall:.0f}",
            'target': profile.spot_price,
            'stop_loss': 'Manage at 50% loss',
            'confidence': confidence,
            'reasoning': f'Stable {range_pct:.1%} range between walls',
            'risk_reward': 1.0,
            'max_loss_pct': 50,
            'position_size_pct': 10
        }

# ======================== MAIN DASHBOARD ========================

def create_gex_chart(profile: GEXProfile) -> go.Figure:
    """Create comprehensive GEX visualization"""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=['Gamma Exposure Profile', 'Cumulative GEX'],
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3]
    )
    
    # Prepare data
    strikes = [entry['strike'] for entry in profile.gex_by_strike]
    gex_values = [entry['gex'] for entry in profile.gex_by_strike]
    cumulative_gex = [entry['cumulative_gex'] for entry in profile.gex_by_strike]
    
    # GEX bar chart
    colors = ['green' if gex > 0 else 'red' for gex in gex_values]
    fig.add_trace(
        go.Bar(
            x=strikes,
            y=gex_values,
            name='GEX',
            marker_color=colors,
            opacity=0.7
        ),
        row=1, col=1
    )
    
    # Cumulative GEX line
    fig.add_trace(
        go.Scatter(
            x=strikes,
            y=cumulative_gex,
            name='Cumulative GEX',
            line=dict(color='blue', width=2)
        ),
        row=2, col=1
    )
    
    # Add key levels
    # Spot price line
    fig.add_vline(
        x=profile.spot_price,
        line_dash="solid",
        line_color="yellow",
        annotation_text=f"Spot: ${profile.spot_price:.2f}",
        row=1, col=1
    )
    
    # Gamma flip line
    fig.add_vline(
        x=profile.gamma_flip,
        line_dash="dash",
        line_color="orange",
        annotation_text=f"Flip: ${profile.gamma_flip:.2f}",
        row=1, col=1
    )
    
    # Call walls
    for wall in profile.call_walls[:3]:
        fig.add_vline(
            x=wall['strike'],
            line_dash="dot",
            line_color="green",
            annotation_text=f"Call Wall: ${wall['strike']:.0f}",
            row=1, col=1
        )
    
    # Put walls
    for wall in profile.put_walls[:3]:
        fig.add_vline(
            x=wall['strike'],
            line_dash="dot",
            line_color="red",
            annotation_text=f"Put Wall: ${wall['strike']:.0f}",
            row=1, col=1
        )
    
    fig.update_layout(
        title=f"{profile.symbol} Gamma Exposure Analysis",
        height=800,
        showlegend=True,
        template="plotly_dark"
    )
    
    fig.update_xaxes(title_text="Strike Price", row=1, col=1)
    fig.update_yaxes(title_text="Gamma Exposure", row=1, col=1)
    fig.update_xaxes(title_text="Strike Price", row=2, col=1)
    fig.update_yaxes(title_text="Cumulative GEX", row=2, col=1)
    
    return fig

def main():
    """Main dashboard application"""
    
    # Header
    st.title("🚀 Professional GEX Trading Dashboard")
    st.markdown("*Advanced Gamma Exposure Analysis for Strategic Options Trading*")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Settings
        st.subheader("TradingVolatility API")
        api_key = st.text_input(
            "API Key",
            value=st.session_state.api_settings['api_key'],
            type="password",
            help="Your TradingVolatility.net API key"
        )
        
        username = st.text_input(
            "Username",
            value=st.session_state.api_settings['username'],
            help="Your TradingVolatility.net username"
        )
        
        if api_key != st.session_state.api_settings['api_key']:
            st.session_state.api_settings['api_key'] = api_key
        
        if username != st.session_state.api_settings['username']:
            st.session_state.api_settings['username'] = username
        
        # Symbol selection
        st.subheader("📊 Symbol Selection")
        
        # Predefined symbol groups
        etf_symbols = ['SPY', 'QQQ', 'IWM', 'XLF', 'XLE', 'XLK', 'GLD', 'TLT']
        mega_cap_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'BRK-B']
        popular_symbols = ['AMD', 'NFLX', 'CRM', 'UBER', 'SQ', 'ROKU', 'ZM', 'PTON']
        
        symbol_group = st.selectbox(
            "Symbol Group",
            ["Custom", "ETFs", "Mega Cap", "Popular Stocks"]
        )
        
        if symbol_group == "ETFs":
            symbols = st.multiselect("Select ETFs", etf_symbols, default=['SPY', 'QQQ'])
        elif symbol_group == "Mega Cap":
            symbols = st.multiselect("Select Mega Cap", mega_cap_symbols, default=['AAPL', 'MSFT'])
        elif symbol_group == "Popular Stocks":
            symbols = st.multiselect("Select Popular", popular_symbols, default=['AMD', 'NFLX'])
        else:
            custom_symbols = st.text_input(
                "Custom Symbols (comma-separated)",
                value="SPY,QQQ,AAPL",
                help="Enter symbols separated by commas"
            )
            symbols = [s.strip().upper() for s in custom_symbols.split(',') if s.strip()]
        
        # Analysis options
        st.subheader("🔧 Analysis Options")
        auto_refresh = st.checkbox("Auto Refresh (5 min)", value=False)
        show_setups = st.checkbox("Show Trade Setups", value=True)
        show_detailed_charts = st.checkbox("Detailed Charts", value=True)
        confidence_threshold = st.slider("Setup Confidence Threshold", 50, 95, 70)
        
        # Analysis button
        analyze_button = st.button("🔄 Run Analysis", type="primary")
    
    # Main content area
    if not api_key:
        st.warning("⚠️ Please enter your TradingVolatility API key in the sidebar to begin analysis.")
        st.info("You can get your API key from https://stocks.tradingvolatility.net/api")
        return
    
    if not symbols:
        st.warning("⚠️ Please select symbols to analyze.")
        return
    
    # Initialize API client
    api_client = TradingVolatilityAPI(api_key, username)
    gex_calculator = ComprehensiveGEXCalculator(api_client)
    setup_detector = SetupDetector()
    
    # Run analysis
    if analyze_button or auto_refresh:
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        all_profiles = {}
        all_setups = []
        
        for i, symbol in enumerate(symbols):
            status_text.text(f"Analyzing {symbol}... ({i+1}/{len(symbols)})")
            
            # Calculate GEX profile
            profile = gex_calculator.calculate_comprehensive_gex(symbol)
            
            if profile:
                all_profiles[symbol] = profile
                
                # Detect setups
                if show_setups:
                    setups = setup_detector.detect_all_setups(profile)
                    for setup in setups:
                        if setup.get('confidence', 0) >= confidence_threshold:
                            all_setups.append(setup)
            
            progress_bar.progress((i + 1) / len(symbols))
        
        status_text.text("Analysis complete!")
        st.session_state.gex_data = all_profiles
        st.session_state.trade_setups = all_setups
        st.session_state.last_update = datetime.now()
        
        # Clear progress
        progress_bar.empty()
        status_text.empty()
    
    # Display results
    if st.session_state.gex_data:
        
        # Summary metrics
        st.subheader("📊 Market Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        profiles = list(st.session_state.gex_data.values())
        
        with col1:
            positive_gex = sum(1 for p in profiles if p.net_gex > 0)
            st.metric("Positive GEX Symbols", positive_gex, f"{positive_gex/len(profiles)*100:.0f}%")
        
        with col2:
            avg_distance_to_flip = np.mean([
                abs(p.gamma_flip - p.spot_price) / p.spot_price 
                for p in profiles
            ]) * 100
            st.metric("Avg Distance to Flip", f"{avg_distance_to_flip:.1f}%")
        
        with col3:
            total_setups = len(st.session_state.trade_setups)
            high_conf_setups = len([s for s in st.session_state.trade_setups if s.get('confidence', 0) >= 80])
            st.metric("Trade Setups", total_setups, f"{high_conf_setups} high confidence")
        
        with col4:
            if st.session_state.last_update:
                minutes_ago = int((datetime.now() - st.session_state.last_update).total_seconds() / 60)
                st.metric("Last Update", f"{minutes_ago} min ago")
        
        # Trade setups section
        if show_setups and st.session_state.trade_setups:
            st.subheader("🎯 High-Probability Trade Setups")
            
            for setup in st.session_state.trade_setups[:10]:  # Show top 10
                with st.expander(f"{setup['symbol']} - {setup['strategy']} (Confidence: {setup['confidence']}%)"):
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown(f"**Entry:** {setup['entry_strike']}")
                        st.markdown(f"**Target:** {setup['target']}")
                        st.markdown(f"**Stop Loss:** {setup['stop_loss']}")
                    
                    with col2:
                        st.markdown(f"**R/R Ratio:** {setup.get('risk_reward', 'N/A')}")
                        st.markdown(f"**Max Loss:** {setup.get('max_loss_pct', 'N/A')}%")
                        st.markdown(f"**Position Size:** {setup.get('position_size_pct', 'N/A')}%")
                    
                    with col3:
                        confidence_color = "🟢" if setup['confidence'] >= 80 else "🟡" if setup['confidence'] >= 70 else "🔴"
                        st.markdown(f"**Confidence:** {confidence_color} {setup['confidence']}%")
                        st.markdown(f"**Reasoning:** {setup['reasoning']}")
        
        # Individual symbol analysis
        st.subheader("🔍 Individual Symbol Analysis")
        
        # Symbol selector
        selected_symbol = st.selectbox(
            "Select Symbol for Detailed Analysis",
            list(st.session_state.gex_data.keys())
        )
        
        if selected_symbol and selected_symbol in st.session_state.gex_data:
            profile = st.session_state.gex_data[selected_symbol]
            
            # Symbol overview
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Spot Price", f"${profile.spot_price:.2f}")
            
            with col2:
                st.metric("Gamma Flip", f"${profile.gamma_flip:.2f}")
            
            with col3:
                regime_color = "🟢" if profile.regime == 'positive' else "🔴"
                st.metric("GEX Regime", f"{regime_color} {profile.regime.title()}")
            
            with col4:
                st.metric("Net GEX", f"{profile.net_gex/1e9:.1f}B")
            
            # Walls information
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📞 Call Walls (Resistance)")
                if profile.call_walls:
                    for i, wall in enumerate(profile.call_walls[:3]):
                        distance = (wall['strike'] - profile.spot_price) / profile.spot_price * 100
                        st.markdown(f"**{i+1}.** ${wall['strike']:.0f} (+{distance:.1f}%) - {wall['strength']}")
                else:
                    st.markdown("*No significant call walls*")
            
            with col2:
                st.subheader("📉 Put Walls (Support)")
                if profile.put_walls:
                    for i, wall in enumerate(profile.put_walls[:3]):
                        distance = (profile.spot_price - wall['strike']) / profile.spot_price * 100
                        st.markdown(f"**{i+1}.** ${wall['strike']:.0f} (-{distance:.1f}%) - {wall['strength']}")
                else:
                    st.markdown("*No significant put walls*")
            
            # Detailed chart
            if show_detailed_charts:
                st.subheader(f"📊 {selected_symbol} GEX Profile")
                fig = create_gex_chart(profile)
                st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("👆 Click 'Run Analysis' to begin GEX analysis of your selected symbols.")
        
        # Educational content
        with st.expander("📚 What is Gamma Exposure (GEX)?"):
            st.markdown("""
            **Gamma Exposure (GEX)** measures the aggregate gamma exposure of options dealers and market makers.
            
            ### How it Works:
            - **Positive GEX**: Dealers are long gamma → they sell rallies and buy dips → volatility suppression
            - **Negative GEX**: Dealers are short gamma → they buy rallies and sell dips → volatility amplification
            
            ### Key Levels:
            - **Gamma Flip Point**: Where net GEX crosses zero - critical support/resistance level
            - **Call Walls**: Strikes with high call gamma - act as resistance where dealers must sell
            - **Put Walls**: Strikes with high put gamma - act as support where dealers must buy
            
            ### Trading Applications:
            1. **Squeeze Plays**: Exploit negative GEX for trending moves
            2. **Mean Reversion**: Use positive GEX for range-bound strategies  
            3. **Wall Trading**: Sell premium at gamma walls, buy breakouts through walls
            4. **Regime Identification**: Adapt strategies based on GEX environment
            """)
    
    # Footer
    st.markdown("---")
    if st.session_state.last_update:
        st.caption(f"Last updated: {st.session_state.last_update.strftime('%Y-%m-%d %H:%M:%S')} | "
                  f"Symbols analyzed: {len(st.session_state.gex_data)} | "
                  f"Setups found: {len(st.session_state.trade_setups)}")
    else:
        st.caption("Professional GEX Trading Dashboard - Ready for analysis")
    
    # Auto-refresh logic
    if auto_refresh and st.session_state.last_update:
        if (datetime.now() - st.session_state.last_update).total_seconds() > 300:  # 5 minutes
            st.rerun()

if __name__ == "__main__":
    main()
