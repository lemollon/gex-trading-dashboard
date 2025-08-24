"""
GEX Trading Dashboard - Complete Gamma Exposure Analysis Platform
Author: GEX Trading System
Version: 2.0.0
Description: Comprehensive dashboard for gamma exposure analysis, trade setup detection,
             and position management with real-time monitoring capabilities.
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="GEX Trading Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        margin: 5px;
    }
    .success-box {
        padding: 10px;
        border-radius: 5px;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .warning-box {
        padding: 10px;
        border-radius: 5px;
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        color: #856404;
    }
    .danger-box {
        padding: 10px;
        border-radius: 5px;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
    .trade-setup {
        background-color: #e7f3ff;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #0066cc;
    }
    div[data-testid="metric-container"] {
        background-color: rgba(28, 131, 225, 0.1);
        border: 1px solid rgba(28, 131, 225, 0.2);
        padding: 10px 15px;
        border-radius: 10px;
        margin: 10px 0;
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
    st.session_state.gex_data = None

if 'last_update' not in st.session_state:
    st.session_state.last_update = None

if 'alerts' not in st.session_state:
    st.session_state.alerts = []

if 'trade_setups' not in st.session_state:
    st.session_state.trade_setups = []

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
            self.spot_price = ticker.info.get('regularMarketPrice', ticker.history(period='1d')['Close'].iloc[-1])
            
            # Get all expiration dates
            expirations = ticker.options[:10]  # Limit to first 10 expirations for performance
            
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
            logger.error(f"Failed to fetch options data: {e}")
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
            logger.error(f"Failed to calculate GEX: {e}")
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
        
        # Negative GEX Squeeze (Long Calls)
        if self.gex.symbol == 'SPY':
            neg_threshold = -1e9  # -1B for SPY
            pos_threshold = 2e9   # 2B for SPY
        else:  # QQQ
            neg_threshold = -5e8  # -500M for QQQ
            pos_threshold = 1e9   # 1B for QQQ
        
        if net_gex < neg_threshold:
            distance_to_flip = (flip - spot) / spot * 100
            
            if 0.5 <= distance_to_flip <= 1.5:
                # Find put wall support
                put_walls = self.gex.gex_profile[self.gex.gex_profile['is_put_wall'] == True]
                if len(put_walls) > 0:
                    nearest_put_wall = put_walls.iloc[0]['strike']
                    
                    setup = {
                        'type': 'SQUEEZE_LONG_CALL',
                        'strategy': 'Negative GEX Squeeze',
                        'entry_price': spot,
                        'target_strike': flip,
                        'stop_loss': nearest_put_wall,
                        'confidence': min(85, 70 + abs(distance_to_flip) * 10),
                        'risk_reward': abs(flip - spot) / abs(spot - nearest_put_wall),
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
                        'type': 'SQUEEZE_LONG_PUT',
                        'strategy': 'Positive GEX Breakdown',
                        'entry_price': spot,
                        'target_strike': flip,
                        'stop_loss': nearest_call_wall,
                        'confidence': min(80, 65 + distance_to_flip * 50),
                        'risk_reward': abs(spot - flip) / abs(nearest_call_wall - spot),
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
                
                # Determine direction based on position
                if abs(spot - highest_put) < abs(highest_call - spot):
                    # Closer to put wall - long calls
                    setup = {
                        'type': 'COMPRESSION_LONG_CALL',
                        'strategy': 'Gamma Wall Compression',
                        'entry_price': spot,
                        'target_strike': highest_call,
                        'stop_loss': highest_put,
                        'confidence': 75,
                        'risk_reward': abs(highest_call - spot) / abs(spot - highest_put),
                        'description': f"Compression Setup: Walls only {wall_spread:.1f}% apart",
                        'entry_criteria': f"Long calls - near put wall support at {highest_put:.2f}",
                        'days_to_expiry': '0-2 DTE for explosion',
                        'position_size': '2% of capital max',
                        'notes': 'Explosive move expected on wall break'
                    }
                else:
                    # Closer to call wall - long puts
                    setup = {
                        'type': 'COMPRESSION_LONG_PUT',
                        'strategy': 'Gamma Wall Compression',
                        'entry_price': spot,
                        'target_strike': highest_put,
                        'stop_loss': highest_call,
                        'confidence': 75,
                        'risk_reward': abs(spot - highest_put) / abs(highest_call - spot),
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
        
        # Call selling at resistance
        call_walls = self.gex.gex_profile[self.gex.gex_profile['is_call_wall'] == True]
        if len(call_walls) > 0 and net_gex > 3e9:  # Need high positive GEX
            nearest_call_wall = call_walls.iloc[0]
            wall_strength = abs(nearest_call_wall['gex'])
            
            if wall_strength > 5e8:  # 500M gamma concentration
                distance = (nearest_call_wall['strike'] - spot) / spot * 100
                
                if 0.5 <= distance <= 2:
                    setup = {
                        'type': 'SELL_CALL',
                        'strategy': 'Call Premium Selling',
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
            
            if wall_strength > 5e8:  # 500M gamma concentration
                distance = (spot - nearest_put_wall['strike']) / spot * 100
                
                if distance >= 1:
                    setup = {
                        'type': 'SELL_PUT',
                        'strategy': 'Put Premium Selling',
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
        if self.gex.net_gex is None or self.gex.net_gex < 1e9:
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
                    condor_type = "Broken Wing (Bullish)"
                    put_spread_mult = 1.5
                    call_spread_mult = 0.75
                elif total_call_gamma > total_put_gamma:
                    condor_type = "Broken Wing (Bearish)"
                    put_spread_mult = 0.75
                    call_spread_mult = 1.5
                else:
                    condor_type = "Standard"
                    put_spread_mult = 1.0
                    call_spread_mult = 1.0
                
                setup = {
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

# ======================== RISK MANAGEMENT ========================

class RiskManager:
    """Manage portfolio risk and position sizing"""
    
    def __init__(self, portfolio: Dict):
        self.portfolio = portfolio
        self.max_position_size = 0.05  # 5% max per position
        self.max_directional_exposure = 0.15  # 15% max directional
        self.max_loss_per_trade = 0.03  # 3% max loss per trade
    
    def calculate_position_size(self, setup: Dict) -> int:
        """Calculate appropriate position size based on setup and risk parameters"""
        capital = self.portfolio['cash']
        
        if 'SQUEEZE' in setup['type']:
            # 3% max for squeeze plays
            max_size = capital * 0.03
        elif 'SELL' in setup['type']:
            # 5% max for premium selling
            max_size = capital * 0.05
        elif 'IRON_CONDOR' in setup['type']:
            # Size for 2% max loss
            max_size = capital * 0.02
        else:
            max_size = capital * 0.03
        
        # Adjust for confidence
        confidence_mult = setup['confidence'] / 100
        position_size = max_size * confidence_mult
        
        # Round to nearest 100
        return int(position_size / 100) * 100
    
    def check_risk_limits(self, new_position: Dict) -> Tuple[bool, str]:
        """Check if new position violates risk limits"""
        current_exposure = sum([p['value'] for p in self.portfolio['positions']])
        
        if current_exposure + new_position['value'] > self.portfolio['total_value'] * 0.5:
            return False, "Would exceed 50% portfolio exposure limit"
        
        # Check directional exposure
        directional_exposure = sum([
            p['value'] for p in self.portfolio['positions'] 
            if p.get('type', '').startswith(new_position['type'].split('_')[0])
        ])
        
        if directional_exposure + new_position['value'] > self.portfolio['total_value'] * self.max_directional_exposure:
            return False, f"Would exceed {self.max_directional_exposure*100}% directional exposure limit"
        
        return True, "Risk check passed"
    
    def calculate_portfolio_metrics(self) -> Dict:
        """Calculate portfolio-wide risk metrics"""
        if not self.portfolio['positions']:
            return {
                'total_exposure': 0,
                'directional_bias': 'Neutral',
                'risk_score': 0,
                'var_95': 0,
                'max_loss': 0
            }
        
        total_exposure = sum([p['value'] for p in self.portfolio['positions']])
        
        # Calculate directional bias
        long_exposure = sum([p['value'] for p in self.portfolio['positions'] if 'LONG' in p.get('type', '')])
        short_exposure = sum([p['value'] for p in self.portfolio['positions'] if 'SELL' in p.get('type', '')])
        
        if long_exposure > short_exposure * 1.5:
            directional_bias = 'Bullish'
        elif short_exposure > long_exposure * 1.5:
            directional_bias = 'Bearish'
        else:
            directional_bias = 'Neutral'
        
        # Simple VaR calculation
        position_values = [p['value'] for p in self.portfolio['positions']]
        var_95 = np.percentile(position_values, 5) if position_values else 0
        
        # Max loss calculation
        max_loss = sum([p.get('max_loss', p['value'] * 0.5) for p in self.portfolio['positions']])
        
        return {
            'total_exposure': total_exposure,
            'directional_bias': directional_bias,
            'risk_score': min(100, (total_exposure / self.portfolio['total_value']) * 100),
            'var_95': var_95,
            'max_loss': max_loss
        }

# ======================== ALERT SYSTEM ========================

class AlertSystem:
    """Generate and manage trading alerts"""
    
    def __init__(self):
        self.alerts = []
        self.alert_history = []
    
    def check_gex_alerts(self, gex_calc: GEXCalculator) -> List[Dict]:
        """Check for GEX-based alert conditions"""
        self.alerts = []
        
        if gex_calc.net_gex is None:
            return self.alerts
        
        # High priority alerts
        if gex_calc.net_gex < -1e9:
            self.alerts.append({
                'priority': 'HIGH',
                'type': 'GEX_NEGATIVE',
                'message': f'Net GEX below -1B threshold: {gex_calc.net_gex/1e9:.2f}B',
                'action': 'Consider long volatility positions',
                'timestamp': datetime.now()
            })
        
        # Check distance to flip
        if gex_calc.gamma_flip:
            distance = abs(gex_calc.spot_price - gex_calc.gamma_flip) / gex_calc.spot_price * 100
            if distance < 0.25:
                self.alerts.append({
                    'priority': 'HIGH',
                    'type': 'NEAR_FLIP',
                    'message': f'Price within 0.25% of gamma flip at {gex_calc.gamma_flip:.2f}',
                    'action': 'Prepare for volatility regime change',
                    'timestamp': datetime.now()
                })
        
        # Check wall breaches
        call_walls = gex_calc.gex_profile[gex_calc.gex_profile['is_call_wall'] == True]
        if len(call_walls) > 0:
            nearest_call = call_walls.iloc[0]['strike']
            if gex_calc.spot_price > nearest_call:
                self.alerts.append({
                    'priority': 'HIGH',
                    'type': 'WALL_BREACH',
                    'message': f'Call wall breached at {nearest_call:.2f}',
                    'action': 'Expect continued momentum or reversal',
                    'timestamp': datetime.now()
                })
        
        # Medium priority alerts
        if abs(gex_calc.net_gex) > 5e9:
            self.alerts.append({
                'priority': 'MEDIUM',
                'type': 'EXTREME_GEX',
                'message': f'Extreme GEX level: {gex_calc.net_gex/1e9:.2f}B',
                'action': 'Market at extremes - prepare for reversal',
                'timestamp': datetime.now()
            })
        
        return self.alerts
    
    def check_position_alerts(self, positions: List[Dict], spot_price: float) -> List[Dict]:
        """Check for position-based alerts"""
        position_alerts = []
        
        for position in positions:
            # Check stop losses
            if position.get('stop_loss'):
                if position['type'].startswith('LONG') and spot_price <= position['stop_loss']:
                    position_alerts.append({
                        'priority': 'HIGH',
                        'type': 'STOP_LOSS',
                        'message': f"Stop loss triggered for {position['type']}",
                        'action': f"Close position at {position['stop_loss']:.2f}",
                        'timestamp': datetime.now()
                    })
            
            # Check profit targets
            if position.get('target'):
                if position['type'].startswith('LONG') and spot_price >= position['target']:
                    position_alerts.append({
                        'priority': 'MEDIUM',
                        'type': 'PROFIT_TARGET',
                        'message': f"Profit target reached for {position['type']}",
                        'action': f"Consider closing at {position['target']:.2f}",
                        'timestamp': datetime.now()
                    })
        
        self.alerts.extend(position_alerts)
        return self.alerts

# ======================== MAIN DASHBOARD ========================

def main():
    # Header
    st.title("🎯 GEX Trading Dashboard")
    st.markdown("**Real-time Gamma Exposure Analysis & Trade Setup Detection**")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        symbol = st.selectbox(
            "Select Symbol",
            ["SPY", "QQQ", "IWM", "DIA"],
            help="Choose the symbol for GEX analysis"
        )
        
        auto_refresh = st.checkbox("Auto Refresh (5 min)", value=False)
        
        st.divider()
        
        # Portfolio overview
        st.header("💼 Portfolio")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Cash", f"${st.session_state.portfolio['cash']:,.0f}")
        with col2:
            st.metric("Total Value", f"${st.session_state.portfolio['total_value']:,.0f}")
        
        st.metric("Daily P&L", 
                 f"${st.session_state.portfolio['daily_pnl']:+,.0f}",
                 delta=f"{st.session_state.portfolio['daily_pnl']/st.session_state.portfolio['total_value']*100:+.2f}%")
        
        st.divider()
        
        # Risk parameters
        st.header("🎚️ Risk Settings")
        max_position = st.slider("Max Position Size %", 1, 10, 5)
        max_loss = st.slider("Max Loss per Trade %", 1, 5, 3)
        
        st.divider()
        
        # Manual refresh button
        if st.button("🔄 Refresh Data", type="primary", use_container_width=True):
            st.session_state.last_update = datetime.now()
            st.rerun()
    
    # Main content area
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 GEX Analysis", 
        "🎯 Trade Setups", 
        "📈 Positions", 
        "⚠️ Alerts",
        "📉 Performance",
        "🔍 Strategy Guide"
    ])
    
    # Initialize calculators
    gex_calc = GEXCalculator(symbol)
    
    # Fetch and calculate GEX data
    with st.spinner("Fetching options data..."):
        if gex_calc.fetch_options_data():
            gex_calc.calculate_gamma_exposure()
            st.session_state.gex_data = gex_calc
            
            # Detect trade setups
            detector = TradeSetupDetector(gex_calc)
            setups = detector.detect_all_setups()
            st.session_state.trade_setups = setups
            
            # Generate alerts
            alert_system = AlertSystem()
            alerts = alert_system.check_gex_alerts(gex_calc)
            alerts.extend(alert_system.check_position_alerts(
                st.session_state.portfolio['positions'], 
                gex_calc.spot_price
            ))
            st.session_state.alerts = alerts
    
    # Tab 1: GEX Analysis
    with tab1:
        if st.session_state.gex_data:
            gex = st.session_state.gex_data
            
            # Key metrics
            st.subheader("📊 Key GEX Metrics")
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
            
            # GEX Profile Chart
            st.subheader("📈 Gamma Exposure Profile")
            
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=("Gamma Exposure by Strike", "Cumulative GEX"),
                vertical_spacing=0.15,
                row_heights=[0.7, 0.3]
            )
            
            # GEX bars
            colors = ['green' if x > 0 else 'red' for x in gex.gex_profile['gex']]
            fig.add_trace(
                go.Bar(
                    x=gex.gex_profile['strike'],
                    y=gex.gex_profile['gex'] / 1e6,  # Convert to millions
                    name='GEX',
                    marker_color=colors,
                    showlegend=False
                ),
                row=1, col=1
            )
            
            # Add spot price line
            fig.add_vline(x=gex.spot_price, line_dash="dash", line_color="blue", 
                         annotation_text=f"Spot: ${gex.spot_price:.2f}", row=1, col=1)
            
            # Add gamma flip line
            if gex.gamma_flip:
                fig.add_vline(x=gex.gamma_flip, line_dash="dash", line_color="orange",
                            annotation_text=f"Flip: ${gex.gamma_flip:.2f}", row=1, col=1)
            
            # Mark walls
            for _, wall in gex.gex_profile[gex.gex_profile['is_call_wall'] == True].iterrows():
                fig.add_vline(x=wall['strike'], line_dash="dot", line_color="green",
                            annotation_text="Call Wall", row=1, col=1)
            
            for _, wall in gex.gex_profile[gex.gex_profile['is_put_wall'] == True].iterrows():
                fig.add_vline(x=wall['strike'], line_dash="dot", line_color="red",
                            annotation_text="Put Wall", row=1, col=1)
            
            # Cumulative GEX
            fig.add_trace(
                go.Scatter(
                    x=gex.gex_profile['strike'],
                    y=gex.gex_profile['cumulative_gex'] / 1e9,  # Convert to billions
                    mode='lines',
                    name='Cumulative GEX',
                    line=dict(color='purple', width=2)
                ),
                row=2, col=1
            )
            
            fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
            
            fig.update_layout(
                height=700,
                showlegend=True,
                hovermode='x unified',
                xaxis_title="Strike Price",
                yaxis_title="GEX (Millions)",
                xaxis2_title="Strike Price",
                yaxis2_title="Cumulative GEX (Billions)"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Market regime interpretation
            st.subheader("🎭 Market Regime Analysis")
            
            if gex.net_gex > 2e9:
                regime = "🟢 **HIGH POSITIVE GAMMA**"
                interpretation = """
                - Volatility suppression in effect
                - Dealers sell rallies, buy dips
                - Expect range-bound, mean-reverting action
                - Good for: Premium selling, iron condors
                - Avoid: Momentum trades, breakout plays
                """
            elif gex.net_gex > 0:
                regime = "🟡 **MODERATE POSITIVE GAMMA**"
                interpretation = """
                - Mild volatility dampening
                - Some dealer hedging flows
                - Trending moves possible but limited
                - Good for: Selective premium selling
                - Watch for: Gamma flip proximity
                """
            elif gex.net_gex > -1e9:
                regime = "🟠 **MODERATE NEGATIVE GAMMA**"
                interpretation = """
                - Volatility amplification beginning
                - Dealers chase moves (buy rallies, sell dips)
                - Trending moves more likely
                - Good for: Directional plays with stops
                - Watch for: Accelerating moves
                """
            else:
                regime = "🔴 **HIGH NEGATIVE GAMMA**"
                interpretation = """
                - Maximum volatility regime
                - Dealers heavily short gamma
                - Explosive moves in both directions
                - Good for: Squeeze plays, momentum trades
                - Avoid: Premium selling without hedges
                """
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown(regime)
            with col2:
                st.markdown(interpretation)
    
    # Tab 2: Trade Setups
    with tab2:
        st.subheader("🎯 Active Trade Setups")
        
        if st.session_state.trade_setups:
            # Filter setups by confidence
            min_confidence = st.slider("Minimum Confidence %", 50, 90, 65)
            filtered_setups = [s for s in st.session_state.trade_setups 
                              if s['confidence'] >= min_confidence]
            
            if filtered_setups:
                for setup in filtered_setups[:5]:  # Show top 5 setups
                    with st.container():
                        col1, col2, col3 = st.columns([2, 1, 1])
                        
                        with col1:
                            confidence_color = "🟢" if setup['confidence'] > 75 else "🟡" if setup['confidence'] > 60 else "🔴"
                            st.markdown(f"### {setup['strategy']} {confidence_color}")
                            st.markdown(f"**{setup['description']}**")
                            st.markdown(f"Entry: {setup['entry_criteria']}")
                            st.markdown(f"Timeframe: {setup['days_to_expiry']}")
                            st.markdown(f"Position Size: {setup['position_size']}")
                            if 'notes' in setup:
                                st.markdown(f"📝 {setup['notes']}")
                        
                        with col2:
                            st.metric("Confidence", f"{setup['confidence']:.0f}%")
                            if 'risk_reward' in setup:
                                st.metric("Risk/Reward", f"{setup['risk_reward']:.2f}")
                        
                        with col3:
                            risk_mgr = RiskManager(st.session_state.portfolio)
                            size = risk_mgr.calculate_position_size(setup)
                            
                            if st.button(f"Execute Trade", key=f"trade_{setup['type']}_{setup.get('strike', 0)}"):
                                # Add position to portfolio
                                position = {
                                    'type': setup['type'],
                                    'strategy': setup['strategy'],
                                    'entry_price': setup['entry_price'],
                                    'strike': setup.get('strike', setup.get('target_strike')),
                                    'value': size,
                                    'stop_loss': setup.get('stop_loss'),
                                    'target': setup.get('target_strike'),
                                    'entry_time': datetime.now()
                                }
                                
                                # Check risk limits
                                can_trade, message = risk_mgr.check_risk_limits(position)
                                if can_trade:
                                    st.session_state.portfolio['positions'].append(position)
                                    st.session_state.portfolio['cash'] -= size
                                    st.success(f"Position opened: {setup['strategy']}")
                                else:
                                    st.error(message)
                        
                        st.divider()
            else:
                st.info(f"No setups found with confidence >= {min_confidence}%")
        else:
            st.warning("No trade setups detected. Waiting for favorable conditions...")
    
    # Tab 3: Positions
    with tab3:
        st.subheader("📈 Active Positions")
        
        if st.session_state.portfolio['positions']:
            positions_df = pd.DataFrame(st.session_state.portfolio['positions'])
            
            # Add current P&L calculation
            if st.session_state.gex_data:
                current_price = st.session_state.gex_data.spot_price
                positions_df['Current P&L'] = positions_df.apply(
                    lambda x: (current_price - x['entry_price']) * x['value'] / x['entry_price'], 
                    axis=1
                )
                positions_df['Current P&L %'] = positions_df['Current P&L'] / positions_df['value'] * 100
            
            # Display positions table
            st.dataframe(
                positions_df[['type', 'strategy', 'entry_price', 'strike', 'value', 'Current P&L %']],
                use_container_width=True
            )
            
            # Position management
            st.subheader("Position Management")
            position_to_close = st.selectbox(
                "Select position to close",
                range(len(positions_df)),
                format_func=lambda x: f"{positions_df.iloc[x]['strategy']} - {positions_df.iloc[x]['type']}"
            )
            
            if st.button("Close Position", type="secondary"):
                closed_position = st.session_state.portfolio['positions'].pop(position_to_close)
                pnl = positions_df.iloc[position_to_close]['Current P&L']
                st.session_state.portfolio['cash'] += closed_position['value'] + pnl
                st.session_state.portfolio['daily_pnl'] += pnl
                st.session_state.portfolio['trade_history'].append({
                    **closed_position,
                    'exit_time': datetime.now(),
                    'pnl': pnl
                })
                st.success(f"Position closed. P&L: ${pnl:+,.2f}")
                st.rerun()
        else:
            st.info("No active positions")
        
        # Risk metrics
        st.subheader("📊 Portfolio Risk Metrics")
        risk_mgr = RiskManager(st.session_state.portfolio)
        metrics = risk_mgr.calculate_portfolio_metrics()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Exposure", f"${metrics['total_exposure']:,.0f}")
        with col2:
            bias_color = "🟢" if metrics['directional_bias'] == 'Bullish' else "🔴" if metrics['directional_bias'] == 'Bearish' else "🟡"
            st.metric("Directional Bias", f"{bias_color} {metrics['directional_bias']}")
        with col3:
            risk_color = "🟢" if metrics['risk_score'] < 50 else "🟡" if metrics['risk_score'] < 75 else "🔴"
            st.metric("Risk Score", f"{risk_color} {metrics['risk_score']:.0f}/100")
        with col4:
            st.metric("Max Loss", f"${metrics['max_loss']:,.0f}")
    
    # Tab 4: Alerts
    with tab4:
        st.subheader("⚠️ Active Alerts")
        
        if st.session_state.alerts:
            # Group alerts by priority
            high_alerts = [a for a in st.session_state.alerts if a['priority'] == 'HIGH']
            medium_alerts = [a for a in st.session_state.alerts if a['priority'] == 'MEDIUM']
            
            if high_alerts:
                st.markdown("### 🔴 High Priority")
                for alert in high_alerts:
                    with st.container():
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.error(f"**{alert['type']}**: {alert['message']}")
                            st.markdown(f"Action: {alert['action']}")
                        with col2:
                            st.caption(f"{alert['timestamp'].strftime('%H:%M:%S')}")
                        st.divider()
            
            if medium_alerts:
                st.markdown("### 🟡 Medium Priority")
                for alert in medium_alerts:
                    with st.container():
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.warning(f"**{alert['type']}**: {alert['message']}")
                            st.markdown(f"Action: {alert['action']}")
                        with col2:
                            st.caption(f"{alert['timestamp'].strftime('%H:%M:%S')}")
                        st.divider()
        else:
            st.success("✅ No active alerts")
        
        # Alert settings
        st.subheader("🔔 Alert Configuration")
        col1, col2 = st.columns(2)
        
        with col1:
            st.checkbox("Enable GEX threshold alerts", value=True)
            st.checkbox("Enable wall breach alerts", value=True)
            st.checkbox("Enable position stop loss alerts", value=True)
        
        with col2:
            st.checkbox("Enable profit target alerts", value=True)
            st.checkbox("Enable regime change alerts", value=True)
            st.checkbox("Send email notifications", value=False)
    
    # Tab 5: Performance
    with tab5:
        st.subheader("📉 Trading Performance")
        
        if st.session_state.portfolio['trade_history']:
            trades_df = pd.DataFrame(st.session_state.portfolio['trade_history'])
            
            # Performance metrics
            col1, col2, col3, col4 = st.columns(4)
            
            total_trades = len(trades_df)
            winning_trades = len(trades_df[trades_df['pnl'] > 0])
            win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
            
            with col1:
                st.metric("Total Trades", total_trades)
            with col2:
                st.metric("Win Rate", f"{win_rate:.1f}%")
            with col3:
                st.metric("Avg Win", f"${trades_df[trades_df['pnl'] > 0]['pnl'].mean():,.2f}" if winning_trades > 0 else "$0")
            with col4:
                st.metric("Avg Loss", f"${trades_df[trades_df['pnl'] < 0]['pnl'].mean():,.2f}" if len(trades_df[trades_df['pnl'] < 0]) > 0 else "$0")
            
            # Strategy breakdown
            st.subheader("Strategy Performance")
            strategy_stats = trades_df.groupby('strategy').agg({
                'pnl': ['count', 'mean', 'sum'],
                'type': 'first'
            }).round(2)
            
            if not strategy_stats.empty:
                strategy_stats.columns = ['Count', 'Avg P&L', 'Total P&L', 'Type']
                st.dataframe(strategy_stats, use_container_width=True)
            
            # P&L chart
            st.subheader("Cumulative P&L")
            trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=trades_df['exit_time'],
                y=trades_df['cumulative_pnl'],
                mode='lines+markers',
                name='Cumulative P&L',
                line=dict(color='green' if trades_df['cumulative_pnl'].iloc[-1] > 0 else 'red', width=2)
            ))
            
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Cumulative P&L ($)",
                hovermode='x unified',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No completed trades yet")
    
    # Tab 6: Strategy Guide
    with tab6:
        st.subheader("🔍 GEX Trading Strategy Guide")
        
        strategy_option = st.selectbox(
            "Select Strategy",
            ["Squeeze Plays", "Premium Selling", "Iron Condors", "Risk Management", "GEX Basics"]
        )
        
        if strategy_option == "Squeeze Plays":
            st.markdown("""
            ### 🚀 Squeeze Play Strategies
            
            #### Negative GEX Squeeze (Long Calls)
            - **Setup**: Net GEX < -1B (SPY) or < -500M (QQQ)
            - **Entry**: Price 0.5-1.5% below gamma flip
            - **Target**: First strike above gamma flip
            - **Risk**: Maximum 3% of capital
            - **Timeframe**: 2-5 DTE for maximum gamma
            
            #### Positive GEX Breakdown (Long Puts)
            - **Setup**: Net GEX > 2B (SPY) or > 1B (QQQ)
            - **Entry**: Price hovering near gamma flip (within 0.3%)
            - **Target**: First strike below gamma flip
            - **Risk**: Maximum 3% of capital
            - **Timeframe**: 3-7 DTE
            
            #### Wall Compression
            - **Setup**: Call and put walls < 2% apart
            - **Entry**: Near wall support/resistance
            - **Target**: Opposite wall
            - **Risk**: Maximum 2% of capital
            - **Timeframe**: 0-2 DTE for explosion
            """)
        
        elif strategy_option == "Premium Selling":
            st.markdown("""
            ### 💰 Premium Selling Strategies
            
            #### Call Selling at Resistance
            - **Setup**: Net GEX > 3B with strong call wall
            - **Entry**: At or above call wall strike
            - **Exit**: 50% profit or approaching wall
            - **Risk**: Maximum 5% of capital
            - **Timeframe**: 0-2 DTE for rapid decay
            
            #### Put Selling at Support
            - **Setup**: Strong put wall > 500M gamma
            - **Entry**: At or below put wall strike
            - **Exit**: 50% profit or defined loss
            - **Risk**: Maximum 5% of capital
            - **Timeframe**: 2-5 DTE
            """)
        
        elif strategy_option == "Iron Condors":
            st.markdown("""
            ### 🦅 Iron Condor Strategies
            
            #### Standard Iron Condor
            - **Setup**: Net GEX > 1B, walls > 3% apart
            - **Short Strikes**: At gamma walls
            - **Long Strikes**: Beyond major gamma
            - **Risk**: Size for 2% max portfolio loss
            - **Timeframe**: 5-10 DTE optimal
            
            #### Broken Wing Adjustments
            - **Bullish Bias**: Put gamma > call gamma → wider put spread (1.5x)
            - **Bearish Bias**: Call gamma > put gamma → wider call spread (1.5x)
            - **Neutral**: Equal spreads on both sides
            """)
        
        elif strategy_option == "Risk Management":
            st.markdown("""
            ### ⚖️ Risk Management Rules
            
            #### Position Sizing
            - **Squeeze Plays**: Maximum 3% of capital
            - **Premium Selling**: Maximum 5% of capital
            - **Iron Condors**: Size for 2% max loss
            - **Total Directional**: Maximum 15% exposure
            
            #### Stop Losses
            - **Long Options**: 50% loss or wall breach
            - **Short Options**: 100% loss or defined risk
            - **Iron Condors**: Threatened strike or 25% profit
            
            #### Portfolio Limits
            - **Maximum Positions**: 5-7 concurrent
            - **Maximum Exposure**: 50% of capital
            - **Daily Loss Limit**: 5% of portfolio
            """)
        
        elif strategy_option == "GEX Basics":
            st.markdown("""
            ### 📚 Understanding Gamma Exposure (GEX)
            
            #### What is GEX?
            GEX measures the aggregate gamma exposure of options dealers. It indicates how much dealers need to hedge as the underlying moves.
            
            #### Calculation
            **GEX = Spot Price × Gamma × Open Interest × 100**
            - Calls contribute positive GEX
            - Puts contribute negative GEX
            
            #### Market Regimes
            
            **Positive GEX (> 1B)**
            - Dealers are long gamma
            - They sell rallies and buy dips
            - Volatility suppression
            - Mean reversion likely
            
            **Negative GEX (< -1B)**
            - Dealers are short gamma
            - They buy rallies and sell dips
            - Volatility amplification
            - Trending moves likely
            
            #### Key Levels
            
            **Gamma Flip Point**
            - Where net GEX crosses zero
            - Regime change level
            - Critical support/resistance
            
            **Call Walls**
            - Highest positive GEX strikes
            - Act as resistance
            - Dealers must sell here
            
            **Put Walls**
            - Highest negative GEX strikes
            - Act as support
            - Dealers must buy here
            """)
    
    # Footer with last update time
    st.divider()
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.session_state.last_update:
            st.caption(f"Last Updated: {st.session_state.last_update.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            st.caption("Last Updated: Never")
    
    # Auto-refresh logic
    if auto_refresh:
        time_module.sleep(300)  # 5 minutes
        st.rerun()

if __name__ == "__main__":
    main()
