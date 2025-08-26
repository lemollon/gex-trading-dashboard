"""
Professional GEX Trading Dashboard
Public Version - No API Keys Required
Built for Professional Traders and Institutions
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import warnings
import json
import time as time_module
from typing import Dict, List, Optional
import logging
from dataclasses import dataclass, field
import requests
from scipy.stats import norm
from scipy.optimize import minimize_scalar
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Professional GEX Analysis Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================== ENHANCED CSS FOR PROFESSIONAL LOOK ========================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: #ffffff;
    }
    
    .main-header {
        background: linear-gradient(90deg, #FF6B6B 0%, #4ECDC4 50%, #45B7D1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3.5rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .subtitle {
        text-align: center;
        font-size: 1.3rem;
        color: #E8E8E8;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    .metric-container {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 25px;
        margin: 15px 0;
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-container:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 45px rgba(31, 38, 135, 0.5);
    }
    
    .big-number {
        font-size: 2.5rem;
        font-weight: 700;
        color: #00FF87;
        text-shadow: 0 0 20px rgba(0, 255, 135, 0.5);
        margin: 0;
    }
    
    .metric-label {
        font-size: 1rem;
        color: #B8B8B8;
        font-weight: 500;
        margin-top: 5px;
    }
    
    .setup-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.1) 0%, rgba(255, 255, 255, 0.05) 100%);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 25px;
        margin: 20px 0;
        border-left: 5px solid;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    
    .setup-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
    }
    
    .setup-bullish {
        border-left-color: #00FF87;
        background: linear-gradient(135deg, rgba(0, 255, 135, 0.1) 0%, rgba(0, 255, 135, 0.05) 100%);
    }
    
    .setup-bearish {
        border-left-color: #FF6B6B;
        background: linear-gradient(135deg, rgba(255, 107, 107, 0.1) 0%, rgba(255, 107, 107, 0.05) 100%);
    }
    
    .setup-neutral {
        border-left-color: #FFD93D;
        background: linear-gradient(135deg, rgba(255, 217, 61, 0.1) 0%, rgba(255, 217, 61, 0.05) 100%);
    }
    
    .confidence-high {
        color: #00FF87;
        font-weight: 700;
        font-size: 1.1rem;
    }
    
    .confidence-medium {
        color: #FFD93D;
        font-weight: 600;
    }
    
    .confidence-low {
        color: #FF6B6B;
        font-weight: 500;
    }
    
    .stSidebar {
        background: linear-gradient(180deg, rgba(255, 255, 255, 0.1) 0%, rgba(255, 255, 255, 0.05) 100%);
        backdrop-filter: blur(20px);
    }
    
    .stSidebar .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.1);
        color: white;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    .stSidebar .stMultiSelect > div > div {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }
    
    .alert-success {
        background: linear-gradient(90deg, #00FF87 0%, #60EFFF 100%);
        color: #1a1a1a;
        padding: 20px;
        border-radius: 15px;
        font-weight: 600;
        margin: 15px 0;
        box-shadow: 0 5px 20px rgba(0, 255, 135, 0.3);
    }
    
    .alert-warning {
        background: linear-gradient(90deg, #FFD93D 0%, #FF6B6B 100%);
        color: #1a1a1a;
        padding: 20px;
        border-radius: 15px;
        font-weight: 600;
        margin: 15px 0;
        box-shadow: 0 5px 20px rgba(255, 217, 61, 0.3);
    }
    
    .section-header {
        font-size: 1.8rem;
        font-weight: 700;
        margin: 30px 0 20px 0;
        background: linear-gradient(90deg, #FF6B6B 0%, #4ECDC4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .wall-indicator {
        display: inline-block;
        padding: 8px 16px;
        border-radius: 25px;
        font-weight: 600;
        font-size: 0.9rem;
        margin: 5px;
        box-shadow: 0 3px 10px rgba(0, 0, 0, 0.3);
    }
    
    .call-wall {
        background: linear-gradient(135deg, #FF6B6B 0%, #FF8E53 100%);
        color: white;
    }
    
    .put-wall {
        background: linear-gradient(135deg, #4ECDC4 0%, #44A08D 100%);
        color: white;
    }
    
    .gamma-flip {
        background: linear-gradient(135deg, #FFD93D 0%, #FF9A3C 100%);
        color: #1a1a1a;
        font-weight: 700;
    }
    
    div[data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.1) !important;
        backdrop-filter: blur(20px) !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        border-radius: 15px !important;
        padding: 20px !important;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37) !important;
    }
    
    div[data-testid="metric-container"] > div {
        color: #00FF87 !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
    
    div[data-testid="metric-container"] > div:last-child {
        color: #B8B8B8 !important;
        font-size: 1rem !important;
        font-weight: 500 !important;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 25px !important;
        padding: 12px 30px !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4) !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.6) !important;
    }
    
    .data-table {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .highlight-positive {
        color: #00FF87;
        font-weight: 600;
    }
    
    .highlight-negative {
        color: #FF6B6B;
        font-weight: 600;
    }
    
    .highlight-neutral {
        color: #FFD93D;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ======================== SESSION STATE INITIALIZATION ========================

if 'gex_data' not in st.session_state:
    st.session_state.gex_data = {}

if 'last_update' not in st.session_state:
    st.session_state.last_update = None

if 'trade_setups' not in st.session_state:
    st.session_state.trade_setups = []

# ======================== BLACK-SCHOLES OPTIONS CALCULATIONS ========================

class AdvancedOptionsCalculator:
    """Advanced options calculations using Black-Scholes model"""
    
    @staticmethod
    def black_scholes_gamma(S, K, T, r, sigma, option_type='call'):
        """Calculate gamma using Black-Scholes formula"""
        if T <= 0:
            return 0.0
        
        try:
            d1 = (math.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
            gamma = math.exp(-0.5 * d1**2) / (S * sigma * math.sqrt(2 * math.pi * T))
            return gamma
        except:
            return 0.0
    
    @staticmethod
    def estimate_gamma_profile(current_price, strikes, dte, iv=0.25):
        """Estimate gamma profile for given strikes"""
        T = dte / 365.0
        r = 0.05  # Risk-free rate
        
        gammas = []
        for strike in strikes:
            gamma = AdvancedOptionsCalculator.black_scholes_gamma(
                current_price, strike, T, r, iv
            )
            gammas.append(gamma)
        
        return gammas

# ======================== GEX CALCULATION ENGINE ========================

@dataclass
class GEXProfile:
    symbol: str
    spot_price: float
    net_gex: float
    gamma_flip: float
    call_walls: List[Dict]
    put_walls: List[Dict]
    gex_by_strike: List[Dict]
    regime: str
    last_updated: datetime
    volatility_regime: str = "Normal"
    expected_move: float = 0.0
    wall_strength: str = "Moderate"

class ProfessionalGEXCalculator:
    """Professional-grade GEX calculator using advanced options pricing"""
    
    def __init__(self):
        self.profiles = {}
        self.calculator = AdvancedOptionsCalculator()
    
    def calculate_comprehensive_gex(self, symbol: str) -> Optional[GEXProfile]:
        """Calculate comprehensive GEX profile using market data"""
        try:
            # Get current market data
            ticker = yf.Ticker(symbol)
            
            # Get current price
            hist = ticker.history(period="2d", interval="1m")
            if hist.empty:
                return None
            
            current_price = float(hist['Close'].iloc[-1])
            
            # Get options chain
            options_data = self._get_options_chain(ticker, current_price)
            if not options_data:
                return None
            
            # Calculate GEX profile
            profile = self._calculate_gex_profile(symbol, current_price, options_data)
            
            if profile:
                self.profiles[symbol] = profile
            
            return profile
            
        except Exception as e:
            logger.error(f"Error calculating GEX for {symbol}: {str(e)}")
            return None
    
    def _get_options_chain(self, ticker, current_price):
        """Get options chain data from Yahoo Finance"""
        try:
            options = ticker.options
            if not options:
                return None
            
            # Get nearest expiration (weekly focus)
            nearest_exp = options[0]
            chain = ticker.option_chain(nearest_exp)
            
            # Combine calls and puts
            calls = chain.calls.copy()
            puts = chain.puts.copy()
            
            calls['option_type'] = 'call'
            puts['option_type'] = 'put'
            
            # Filter for liquid options (within reasonable range)
            price_range = 0.15  # 15% from current price
            min_strike = current_price * (1 - price_range)
            max_strike = current_price * (1 + price_range)
            
            calls = calls[(calls['strike'] >= min_strike) & (calls['strike'] <= max_strike)]
            puts = puts[(puts['strike'] >= min_strike) & (puts['strike'] <= max_strike)]
            
            # Combine
            all_options = pd.concat([calls, puts], ignore_index=True)
            
            # Calculate DTE
            exp_date = pd.to_datetime(nearest_exp)
            current_date = pd.Timestamp.now()
            dte = (exp_date - current_date).days
            
            return {
                'options': all_options,
                'expiration': nearest_exp,
                'dte': max(1, dte)
            }
            
        except Exception as e:
            logger.error(f"Error getting options chain: {str(e)}")
            return None
    
    def _calculate_gex_profile(self, symbol, current_price, options_data):
        """Calculate detailed GEX profile"""
        try:
            df = options_data['options']
            dte = options_data['dte']
            
            # Calculate gamma for each option
            df['calculated_gamma'] = df.apply(lambda row: 
                self.calculator.black_scholes_gamma(
                    current_price, row['strike'], dte/365.0, 0.05, 
                    row.get('impliedVolatility', 0.25)
                ), axis=1
            )
            
            # Calculate GEX for each option
            df['gex'] = current_price * df['calculated_gamma'] * df['openInterest'] * 100
            
            # Adjust sign for puts (negative GEX)
            df.loc[df['option_type'] == 'put', 'gex'] *= -1
            
            # Group by strike
            strike_gex = df.groupby('strike').agg({
                'gex': 'sum',
                'openInterest': 'sum',
                'volume': 'sum'
            }).reset_index()
            
            strike_gex = strike_gex.sort_values('strike')
            strike_gex['cumulative_gex'] = strike_gex['gex'].cumsum()
            
            # Find gamma flip point
            gamma_flip = self._find_gamma_flip(strike_gex, current_price)
            
            # Find walls
            call_walls = self._find_call_walls(strike_gex)
            put_walls = self._find_put_walls(strike_gex)
            
            # Calculate metrics
            net_gex = strike_gex['gex'].sum()
            regime = 'Positive GEX' if net_gex > 0 else 'Negative GEX'
            
            # Volatility regime
            vol_regime = self._determine_volatility_regime(net_gex, current_price, gamma_flip)
            
            # Expected move
            expected_move = self._calculate_expected_move(df, current_price, dte)
            
            # Wall strength
            wall_strength = self._assess_wall_strength(call_walls, put_walls, net_gex)
            
            return GEXProfile(
                symbol=symbol,
                spot_price=current_price,
                net_gex=net_gex,
                gamma_flip=gamma_flip,
                call_walls=call_walls,
                put_walls=put_walls,
                gex_by_strike=strike_gex.to_dict('records'),
                regime=regime,
                last_updated=datetime.now(),
                volatility_regime=vol_regime,
                expected_move=expected_move,
                wall_strength=wall_strength
            )
            
        except Exception as e:
            logger.error(f"Error calculating GEX profile: {str(e)}")
            return None
    
    def _find_gamma_flip(self, strike_gex, current_price):
        """Find gamma flip point with interpolation"""
        cumulative = strike_gex['cumulative_gex'].values
        strikes = strike_gex['strike'].values
        
        # Find zero crossing
        sign_changes = np.where(np.diff(np.sign(cumulative)))[0]
        
        if len(sign_changes) == 0:
            return current_price
        
        # Use first crossing
        idx = sign_changes[0]
        x1, x2 = strikes[idx], strikes[idx + 1]
        y1, y2 = cumulative[idx], cumulative[idx + 1]
        
        if y2 != y1:
            flip_point = x1 - y1 * (x2 - x1) / (y2 - y1)
        else:
            flip_point = x1
        
        return flip_point
    
    def _find_call_walls(self, strike_gex):
        """Find significant call walls"""
        positive_gex = strike_gex[strike_gex['gex'] > 0].copy()
        if positive_gex.empty:
            return []
        
        # Top 3 walls
        walls = positive_gex.nlargest(3, 'gex')
        threshold = positive_gex['gex'].quantile(0.7)
        
        return [
            {
                'strike': row['strike'],
                'gex': row['gex'],
                'strength': 'Strong' if row['gex'] > threshold else 'Moderate',
                'oi': row.get('openInterest', 0)
            }
            for _, row in walls.iterrows()
            if row['gex'] > threshold * 0.5  # Filter weak walls
        ]
    
    def _find_put_walls(self, strike_gex):
        """Find significant put walls"""
        negative_gex = strike_gex[strike_gex['gex'] < 0].copy()
        if negative_gex.empty:
            return []
        
        walls = negative_gex.nsmallest(3, 'gex')
        threshold = negative_gex['gex'].quantile(0.3)
        
        return [
            {
                'strike': row['strike'],
                'gex': row['gex'],
                'strength': 'Strong' if row['gex'] < threshold else 'Moderate',
                'oi': row.get('openInterest', 0)
            }
            for _, row in walls.iterrows()
            if row['gex'] < threshold * 0.5
        ]
    
    def _determine_volatility_regime(self, net_gex, spot_price, gamma_flip):
        """Determine current volatility regime"""
        flip_distance = abs(gamma_flip - spot_price) / spot_price
        
        if abs(net_gex) < 1e6:
            return "Low Gamma"
        elif net_gex > 2e6:
            return "Volatility Suppression"
        elif net_gex < -1e6:
            return "Volatility Expansion"
        else:
            return "Transitional"
    
    def _calculate_expected_move(self, options_df, current_price, dte):
        """Calculate expected move from options"""
        try:
            # Use ATM straddle for expected move
            atm_calls = options_df[
                (options_df['option_type'] == 'call') & 
                (abs(options_df['strike'] - current_price) < current_price * 0.02)
            ]
            
            atm_puts = options_df[
                (options_df['option_type'] == 'put') & 
                (abs(options_df['strike'] - current_price) < current_price * 0.02)
            ]
            
            if not atm_calls.empty and not atm_puts.empty:
                call_price = atm_calls['lastPrice'].mean()
                put_price = atm_puts['lastPrice'].mean()
                straddle_price = call_price + put_price
                return straddle_price * 0.85  # 85% probability move
            
            return current_price * 0.02 * math.sqrt(dte/7)  # Fallback
            
        except:
            return current_price * 0.02
    
    def _assess_wall_strength(self, call_walls, put_walls, net_gex):
        """Assess overall wall strength"""
        if not call_walls and not put_walls:
            return "Weak"
        
        total_wall_strength = 0
        if call_walls:
            total_wall_strength += sum(abs(w['gex']) for w in call_walls[:2])
        if put_walls:
            total_wall_strength += sum(abs(w['gex']) for w in put_walls[:2])
        
        if total_wall_strength > abs(net_gex) * 0.3:
            return "Strong"
        elif total_wall_strength > abs(net_gex) * 0.15:
            return "Moderate"
        else:
            return "Weak"

# ======================== SETUP DETECTION ENGINE ========================

class ProfessionalSetupDetector:
    """Detect professional trading setups"""
    
    def detect_all_setups(self, profile: GEXProfile) -> List[Dict]:
        """Detect all trading opportunities"""
        setups = []
        
        # Directional setups
        setups.extend(self._detect_directional_setups(profile))
        
        # Mean reversion setups
        setups.extend(self._detect_mean_reversion_setups(profile))
        
        # Volatility setups
        setups.extend(self._detect_volatility_setups(profile))
        
        # Range setups
        setups.extend(self._detect_range_setups(profile))
        
        return sorted(setups, key=lambda x: x.get('confidence', 0), reverse=True)
    
    def _detect_directional_setups(self, profile: GEXProfile) -> List[Dict]:
        """Detect directional trading opportunities"""
        setups = []
        
        flip_distance = (profile.gamma_flip - profile.spot_price) / profile.spot_price
        
        # Bullish squeeze setup
        if (profile.net_gex < -500000 and  # Negative GEX
            0.005 < flip_distance < 0.025 and  # Price below flip
            profile.put_walls):  # Has support
            
            confidence = 75
            if flip_distance < 0.015:
                confidence += 10
            if profile.net_gex < -1000000:
                confidence += 10
            
            setups.append({
                'type': 'Bullish Squeeze',
                'direction': 'Bullish',
                'strategy': 'Long Calls / Call Spreads',
                'entry_level': profile.spot_price,
                'target': profile.gamma_flip,
                'stop_loss': profile.put_walls[0]['strike'] if profile.put_walls else profile.spot_price * 0.97,
                'confidence': min(confidence, 95),
                'reasoning': f"Negative GEX ({profile.net_gex/1e6:.1f}M), price {flip_distance:.1%} below flip point",
                'timeframe': '1-3 days',
                'risk_level': 'Medium-High'
            })
        
        # Bearish breakdown setup
        if (profile.net_gex > 1000000 and  # Positive GEX
            abs(flip_distance) < 0.005 and  # Price near flip
            profile.call_walls):  # Has resistance
            
            confidence = 70
            if profile.net_gex > 2000000:
                confidence += 15
            
            setups.append({
                'type': 'Bearish Breakdown',
                'direction': 'Bearish',
                'strategy': 'Long Puts / Put Spreads',
                'entry_level': profile.spot_price,
                'target': profile.put_walls[0]['strike'] if profile.put_walls else profile.gamma_flip * 0.97,
                'stop_loss': profile.call_walls[0]['strike'] if profile.call_walls else profile.spot_price * 1.03,
                'confidence': min(confidence, 90),
                'reasoning': f"Positive GEX ({profile.net_gex/1e6:.1f}M), price at flip point",
                'timeframe': '1-2 days',
                'risk_level': 'Medium'
            })
        
        return setups
    
    def _detect_mean_reversion_setups(self, profile: GEXProfile) -> List[Dict]:
        """Detect mean reversion opportunities"""
        setups = []
        
        if profile.net_gex > 1000000 and profile.call_walls and profile.put_walls:
            
            # Call selling at resistance
            if profile.call_walls:
                call_wall = profile.call_walls[0]
                distance_to_wall = (call_wall['strike'] - profile.spot_price) / profile.spot_price
                
                if 0.01 < distance_to_wall < 0.03:  # 1-3% to wall
                    confidence = 65
                    if call_wall['strength'] == 'Strong':
                        confidence += 15
                    if profile.net_gex > 2000000:
                        confidence += 10
                    
                    setups.append({
                        'type': 'Call Wall Resistance',
                        'direction': 'Bearish',
                        'strategy': 'Sell Calls / Call Credit Spreads',
                        'entry_level': call_wall['strike'],
                        'target': profile.spot_price,
                        'stop_loss': call_wall['strike'] * 1.02,
                        'confidence': confidence,
                        'reasoning': f"{call_wall['strength']} call wall, positive GEX environment",
                        'timeframe': '3-7 days',
                        'risk_level': 'Medium'
                    })
            
            # Put selling at support
            if profile.put_walls:
                put_wall = profile.put_walls[0]
                distance_to_wall = (profile.spot_price - put_wall['strike']) / profile.spot_price
                
                if 0.01 < distance_to_wall < 0.04:  # 1-4% from wall
                    confidence = 65
                    if put_wall['strength'] == 'Strong':
                        confidence += 15
                    
                    setups.append({
                        'type': 'Put Wall Support',
                        'direction': 'Bullish',
                        'strategy': 'Sell Puts / Put Credit Spreads',
                        'entry_level': put_wall['strike'],
                        'target': profile.spot_price,
                        'stop_loss': put_wall['strike'] * 0.98,
                        'confidence': confidence,
                        'reasoning': f"{put_wall['strength']} put wall support",
                        'timeframe': '3-7 days',
                        'risk_level': 'Medium'
                    })
        
        return setups
    
    def _detect_volatility_setups(self, profile: GEXProfile) -> List[Dict]:
        """Detect volatility-based setups"""
        setups = []
        
        if profile.volatility_regime == "Low Gamma":
            setups.append({
                'type': 'Volatility Expansion',
                'direction': 'Neutral',
                'strategy': 'Long Straddles / Strangles',
                'entry_level': f"ATM Straddle ~{profile.spot_price:.0f}",
                'target': f"Move > {profile.expected_move:.0f}",
                'stop_loss': "50% premium loss",
                'confidence': 60,
                'reasoning': "Low gamma environment, potential for vol expansion",
                'timeframe': '1-5 days',
                'risk_level': 'High'
            })
        
        elif profile.volatility_regime == "Volatility Suppression":
            setups.append({
                'type': 'Volatility Contraction',
                'direction': 'Neutral',
                'strategy': 'Short Straddles / Iron Condors',
                'entry_level': f"Range: {profile.put_walls[0]['strike']:.0f}-{profile.call_walls[0]['strike']:.0f}" if profile.put_walls and profile.call_walls else "ATM",
                'target': "Premium decay",
                'stop_loss': "Tested boundaries",
                'confidence': 70,
                'reasoning': "High positive GEX, volatility suppression expected",
                'timeframe': '3-10 days',
                'risk_level': 'Medium'
            })
        
        return setups
    
    def _detect_range_setups(self, profile: GEXProfile) -> List[Dict]:
        """Detect range-bound setups"""
        setups = []
        
        if profile.call_walls and profile.put_walls:
            call_level = profile.call_walls[0]['strike']
            put_level = profile.put_walls[0]['strike']
            range_size = (call_level - put_level) / profile.spot_price
            
            if 0.03 < range_size < 0.08:  # 3-8% range
                confidence = 65
                if profile.wall_strength == "Strong":
                    confidence += 15
                if 0.04 < range_size < 0.06:  # Optimal range
                    confidence += 10
                
                setups.append({
                    'type': 'Range Trading',
                    'direction': 'Neutral',
                    'strategy': 'Iron Condor / Iron Butterfly',
                    'entry_level': f"Range: ${put_level:.0f} - ${call_level:.0f}",
                    'target': "Stay in range",
                    'stop_loss': "Range break + buffer",
                    'confidence': confidence,
                    'reasoning': f"{range_size:.1%} range with {profile.wall_strength.lower()} walls",
                    'timeframe': '5-14 days',
                    'risk_level': 'Low-Medium'
                })
        
        return setups

# ======================== VISUALIZATION FUNCTIONS ========================

def create_professional_gex_chart(profile: GEXProfile) -> go.Figure:
    """Create professional GEX visualization"""
    
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=[
            f'{profile.symbol} Gamma Exposure Profile',
            'Cumulative GEX Flow',
            'Open Interest Distribution'
        ],
        vertical_spacing=0.08,
        row_heights=[0.5, 0.3, 0.2]
    )
    
    # Prepare data
    strikes = [entry['strike'] for entry in profile.gex_by_strike]
    gex_values = [entry['gex'] for entry in profile.gex_by_strike]
    cumulative_gex = [entry['cumulative_gex'] for entry in profile.gex_by_strike]
    open_interest = [entry.get('openInterest', 0) for entry in profile.gex_by_strike]
    
    # Main GEX chart
    colors = ['#00FF87' if gex > 0 else '#FF6B6B' for gex in gex_values]
    fig.add_trace(
        go.Bar(
            x=strikes,
            y=[gex/1e6 for gex in gex_values],  # Convert to millions
            name='Gamma Exposure (M)',
            marker_color=colors,
            opacity=0.8,
            hovertemplate='<b>Strike:</b> $%{x:.0f}<br><b>GEX:</b> %{y:.1f}M<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Cumulative GEX
    fig.add_trace(
        go.Scatter(
            x=strikes,
            y=[cum/1e6 for cum in cumulative_gex],
            name='Cumulative GEX (M)',
            line=dict(color='#4ECDC4', width=3),
            hovertemplate='<b>Strike:</b> $%{x:.0f}<br><b>Cum GEX:</b> %{y:.1f}M<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Open Interest
    fig.add_trace(
        go.Bar(
            x=strikes,
            y=[oi/1000 for oi in open_interest],  # Convert to thousands
            name='Open Interest (K)',
            marker_color='#FFD93D',
            opacity=0.6,
            hovertemplate='<b>Strike:</b> $%{x:.0f}<br><b>OI:</b> %{y:.0f}K<extra></extra>'
        ),
        row=3, col=1
    )
    
    # Add key levels
    # Spot price
    for i in range(1, 4):
        fig.add_vline(
            x=profile.spot_price,
            line=dict(color='yellow', width=3, dash='solid'),
            annotation_text=f"Current: ${profile.spot_price:.2f}",
            annotation_position="top",
            row=i, col=1
        )
    
    # Gamma flip
    for i in range(1, 4):
        fig.add_vline(
            x=profile.gamma_flip,
            line=dict(color='orange', width=2, dash='dash'),
            annotation_text=f"Flip: ${profile.gamma_flip:.2f}",
            annotation_position="bottom",
            row=i, col=1
        )
    
    # Call walls
    for j, wall in enumerate(profile.call_walls[:2]):
        for i in range(1, 4):
            fig.add_vline(
                x=wall['strike'],
                line=dict(color='#FF6B6B', width=2, dash='dot'),
                annotation_text=f"Call Wall {j+1}",
                annotation_position="top" if j == 0 else "bottom",
                row=i, col=1
            )
    
    # Put walls
    for j, wall in enumerate(profile.put_walls[:2]):
        for i in range(1, 4):
            fig.add_vline(
                x=wall['strike'],
                line=dict(color='#4ECDC4', width=2, dash='dot'),
                annotation_text=f"Put Wall {j+1}",
                annotation_position="top" if j == 0 else "bottom",
                row=i, col=1
            )
    
    fig.update_layout(
        height=1000,
        title=dict(
            text=f"<b>{profile.symbol} Professional GEX Analysis</b>",
            font=dict(size=24, color='white'),
            x=0.5
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        showlegend=True,
        legend=dict(
            bgcolor='rgba(255,255,255,0.1)',
            bordercolor='rgba(255,255,255,0.3)',
            borderwidth=1
        )
    )
    
    # Update axes
    fig.update_xaxes(title_text="Strike Price ($)", row=3, col=1, color='white')
    fig.update_yaxes(title_text="GEX (Millions)", row=1, col=1, color='white')
    fig.update_yaxes(title_text="Cumulative (M)", row=2, col=1, color='white')
    fig.update_yaxes(title_text="OI (Thousands)", row=3, col=1, color='white')
    
    return fig

# ======================== MAIN APPLICATION ========================

def main():
    """Professional GEX Trading Dashboard"""
    
    # Header
    st.markdown('<h1 class="main-header">⚡ Professional GEX Analysis Platform</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Advanced Gamma Exposure Intelligence for Institutional Trading</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🎯 Analysis Configuration")
        
        # Symbol selection with professional groupings
        symbol_category = st.selectbox(
            "Asset Category",
            ["Major Indices", "FAANG+", "High Volume Equities", "Sector ETFs", "Custom Symbols"],
            help="Select from professionally curated symbol groups"
        )
        
        if symbol_category == "Major Indices":
            symbols = st.multiselect(
                "Select Indices",
                ["SPY", "QQQ", "IWM", "DIA", "VTI", "ARKK"],
                default=["SPY", "QQQ"],
                help="Major market indices with high options volume"
            )
        elif symbol_category == "FAANG+":
            symbols = st.multiselect(
                "Select FAANG+ Stocks",
                ["AAPL", "AMZN", "GOOGL", "META", "NFLX", "MSFT", "TSLA", "NVDA"],
                default=["AAPL", "MSFT"],
                help="High-cap technology leaders"
            )
        elif symbol_category == "High Volume Equities":
            symbols = st.multiselect(
                "Select High Volume Stocks",
                ["AMD", "UBER", "CRM", "BABA", "PLTR", "ROKU", "ZM", "SQ"],
                default=["AMD", "UBER"],
                help="Stocks with significant daily options volume"
            )
        elif symbol_category == "Sector ETFs":
            symbols = st.multiselect(
                "Select Sector ETFs",
                ["XLF", "XLE", "XLK", "XLV", "XLI", "XLP", "XLU", "XLRE"],
                default=["XLF", "XLK"],
                help="Sector-focused ETFs"
            )
        else:
            custom_input = st.text_input(
                "Custom Symbols",
                value="SPY,AAPL,MSFT",
                help="Enter comma-separated symbols"
            )
            symbols = [s.strip().upper() for s in custom_input.split(',') if s.strip()]
        
        st.markdown("### ⚙️ Analysis Parameters")
        
        confidence_threshold = st.slider(
            "Setup Confidence Threshold",
            min_value=50, max_value=95, value=70, step=5,
            help="Minimum confidence level for trade setups"
        )
        
        show_advanced_metrics = st.checkbox(
            "Advanced Metrics", value=True,
            help="Show detailed GEX analysis metrics"
        )
        
        auto_refresh = st.checkbox(
            "Auto Refresh (10min)", value=False,
            help="Automatically refresh data every 10 minutes"
        )
        
        # Analysis button
        run_analysis = st.button(
            "🚀 Run Professional Analysis",
            type="primary",
            help="Execute comprehensive GEX analysis"
        )
    
    # Main content
    if not symbols:
        st.markdown("""
        <div class="alert-warning">
            <strong>⚠️ No Symbols Selected</strong><br>
            Please select symbols from the sidebar to begin professional GEX analysis.
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Initialize calculator
    calculator = ProfessionalGEXCalculator()
    detector = ProfessionalSetupDetector()
    
    if run_analysis or (auto_refresh and st.session_state.last_update and 
                       (datetime.now() - st.session_state.last_update).seconds > 600):
        
        # Analysis progress
        progress_container = st.container()
        with progress_container:
            st.markdown("### 🔄 Running Professional Analysis...")
            progress_bar = st.progress(0)
            status_text = st.empty()
        
        profiles = {}
        all_setups = []
        
        for i, symbol in enumerate(symbols):
            status_text.markdown(f"**Analyzing {symbol}...** ({i+1}/{len(symbols)})")
            
            profile = calculator.calculate_comprehensive_gex(symbol)
            if profile:
                profiles[symbol] = profile
                
                # Detect setups
                setups = detector.detect_all_setups(profile)
                for setup in setups:
                    if setup.get('confidence', 0) >= confidence_threshold:
                        all_setups.append(setup)
            
            progress_bar.progress((i + 1) / len(symbols))
        
        # Store results
        st.session_state.gex_data = profiles
        st.session_state.trade_setups = all_setups
        st.session_state.last_update = datetime.now()
        
        # Clear progress
        progress_container.empty()
    
    # Display results
    if st.session_state.gex_data:
        
        # Market overview
        st.markdown('<h2 class="section-header">📊 Market Intelligence Overview</h2>', unsafe_allow_html=True)
        
        profiles = list(st.session_state.gex_data.values())
        
        # Key metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            positive_gex = sum(1 for p in profiles if p.net_gex > 0)
            total_symbols = len(profiles)
            st.metric(
                "Positive GEX Ratio",
                f"{positive_gex}/{total_symbols}",
                f"{positive_gex/total_symbols*100:.0f}%"
            )
        
        with col2:
            avg_net_gex = np.mean([p.net_gex for p in profiles]) / 1e6
            st.metric(
                "Avg Net GEX",
                f"{avg_net_gex:.1f}M",
                "Dealer positioning"
            )
        
        with col3:
            flip_distances = [abs(p.gamma_flip - p.spot_price) / p.spot_price for p in profiles]
            avg_flip_distance = np.mean(flip_distances) * 100
            st.metric(
                "Avg Flip Distance",
                f"{avg_flip_distance:.1f}%",
                "Regime proximity"
            )
        
        with col4:
            total_setups = len(st.session_state.trade_setups)
            high_conf = len([s for s in st.session_state.trade_setups if s.get('confidence', 0) >= 80])
            st.metric(
                "Trade Opportunities",
                f"{total_setups}",
                f"{high_conf} high confidence"
            )
        
        with col5:
            if st.session_state.last_update:
                minutes_ago = int((datetime.now() - st.session_state.last_update).total_seconds() / 60)
                st.metric(
                    "Data Freshness",
                    f"{minutes_ago}m ago",
                    "Last update"
                )
        
        # Trade setups
        if st.session_state.trade_setups:
            st.markdown('<h2 class="section-header">🎯 Professional Trading Opportunities</h2>', unsafe_allow_html=True)
            
            # Filter and sort setups
            high_confidence_setups = [s for s in st.session_state.trade_setups if s.get('confidence', 0) >= confidence_threshold]
            
            for setup in high_confidence_setups[:8]:  # Show top 8 setups
                
                # Determine setup styling
                if 'Bullish' in setup.get('direction', ''):
                    card_class = 'setup-card setup-bullish'
                elif 'Bearish' in setup.get('direction', ''):
                    card_class = 'setup-card setup-bearish'
                else:
                    card_class = 'setup-card setup-neutral'
                
                # Confidence styling
                confidence = setup.get('confidence', 0)
                if confidence >= 80:
                    conf_class = 'confidence-high'
                elif confidence >= 70:
                    conf_class = 'confidence-medium'
                else:
                    conf_class = 'confidence-low'
                
                st.markdown(f"""
                <div class="{card_class}">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                        <h3 style="margin: 0; color: white;">{setup.get('symbol', 'N/A')} - {setup['type']}</h3>
                        <span class="{conf_class}">{confidence}% Confidence</span>
                    </div>
                    
                    <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; margin-bottom: 15px;">
                        <div>
                            <strong>Strategy:</strong><br>
                            {setup['strategy']}
                        </div>
                        <div>
                            <strong>Entry Level:</strong><br>
                            {setup['entry_level']}
                        </div>
                        <div>
                            <strong>Target:</strong><br>
                            {setup['target']}
                        </div>
                    </div>
                    
                    <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; margin-bottom: 15px;">
                        <div>
                            <strong>Stop Loss:</strong><br>
                            {setup['stop_loss']}
                        </div>
                        <div>
                            <strong>Timeframe:</strong><br>
                            {setup.get('timeframe', 'N/A')}
                        </div>
                        <div>
                            <strong>Risk Level:</strong><br>
                            {setup.get('risk_level', 'Medium')}
                        </div>
                    </div>
                    
                    <div style="background: rgba(0,0,0,0.2); padding: 15px; border-radius: 10px;">
                        <strong>Analysis:</strong> {setup['reasoning']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Individual symbol analysis
        st.markdown('<h2 class="section-header">🔍 Detailed Symbol Analysis</h2>', unsafe_allow_html=True)
        
        selected_symbol = st.selectbox(
            "Select Symbol for Deep Analysis",
            list(st.session_state.gex_data.keys()),
            help="Choose a symbol for comprehensive GEX breakdown"
        )
        
        if selected_symbol and selected_symbol in st.session_state.gex_data:
            profile = st.session_state.gex_data[selected_symbol]
            
            # Symbol overview
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Current Price", f"${profile.spot_price:.2f}")
            
            with col2:
                st.metric("Gamma Flip", f"${profile.gamma_flip:.2f}")
            
            with col3:
                regime_color = "🟢" if "Positive" in profile.regime else "🔴"
                st.metric("GEX Regime", f"{regime_color} {profile.regime}")
            
            with col4:
                st.metric("Net GEX", f"{profile.net_gex/1e6:.1f}M")
            
            with col5:
                st.metric("Vol Regime", profile.volatility_regime)
            
            # Walls and levels
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📞 Call Walls (Resistance)")
                if profile.call_walls:
                    for i, wall in enumerate(profile.call_walls):
                        distance = (wall['strike'] - profile.spot_price) / profile.spot_price * 100
                        st.markdown(f"""
                        <div class="wall-indicator call-wall">
                            ${wall['strike']:.0f} (+{distance:.1f}%) - {wall['strength']}
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown("*No significant call walls detected*")
            
            with col2:
                st.markdown("#### 📉 Put Walls (Support)")
                if profile.put_walls:
                    for i, wall in enumerate(profile.put_walls):
                        distance = (profile.spot_price - wall['strike']) / profile.spot_price * 100
                        st.markdown(f"""
                        <div class="wall-indicator put-wall">
                            ${wall['strike']:.0f} (-{distance:.1f}%) - {wall['strength']}
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown("*No significant put walls detected*")
            
            # Gamma flip indicator
            flip_distance = (profile.gamma_flip - profile.spot_price) / profile.spot_price * 100
            st.markdown(f"""
            <div style="text-align: center; margin: 20px 0;">
                <div class="wall-indicator gamma-flip">
                    Gamma Flip: ${profile.gamma_flip:.2f} ({flip_distance:+.1f}% from spot)
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Professional chart
            st.markdown("#### 📈 Professional GEX Chart")
            fig = create_professional_gex_chart(profile)
            st.plotly_chart(fig, use_container_width=True)
            
            # Advanced metrics
            if show_advanced_metrics:
                st.markdown("#### 🎯 Advanced Metrics")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    **Expected Move:** ${profile.expected_move:.2f}  
                    **Wall Strength:** {profile.wall_strength}  
                    **Analysis Time:** {profile.last_updated.strftime('%H:%M:%S')}
                    """)
                
                with col2:
                    total_call_gex = sum(w['gex'] for w in profile.call_walls) if profile.call_walls else 0
                    total_put_gex = sum(abs(w['gex']) for w in profile.put_walls) if profile.put_walls else 0
                    st.markdown(f"""
                    **Call Wall GEX:** {total_call_gex/1e6:.1f}M  
                    **Put Wall GEX:** {total_put_gex/1e6:.1f}M  
                    **GEX Ratio:** {total_call_gex/max(total_put_gex, 1):.2f}
                    """)
                
                with col3:
                    flip_strength = abs(profile.gamma_flip - profile.spot_price) / profile.expected_move if profile.expected_move > 0 else 0
                    regime_stability = "High" if abs(profile.net_gex) > 1e6 else "Medium" if abs(profile.net_gex) > 0.5e6 else "Low"
                    st.markdown(f"""
                    **Flip Strength:** {flip_strength:.2f}  
                    **Regime Stability:** {regime_stability}  
                    **Data Quality:** Professional Grade
                    """)
    
    else:
        # Welcome message
        st.markdown("""
        <div class="alert-success">
            <strong>🚀 Welcome to Professional GEX Analysis Platform</strong><br>
            Select your symbols from the sidebar and click "Run Professional Analysis" to begin comprehensive gamma exposure intelligence gathering.
        </div>
        """, unsafe_allow_html=True)
        
        # Educational content
        with st.expander("📚 Professional GEX Intelligence Guide", expanded=True):
            st.markdown("""
            ### What is Gamma Exposure (GEX)?
            
            **Gamma Exposure** represents the aggregate gamma risk that options market makers and dealers must hedge. It's a critical indicator of market microstructure and price action dynamics.
            
            #### Key Concepts:
            
            **🟢 Positive GEX Environment:**
            - Market makers are long gamma
            - They provide liquidity by selling rallies and buying dips
            - Results in volatility suppression and mean reversion behavior
            - Ideal for income strategies and range-bound plays
            
            **🔴 Negative GEX Environment:**  
            - Market makers are short gamma
            - They must hedge by buying rallies and selling dips
            - Creates volatility amplification and trending behavior
            - Optimal for directional momentum strategies
            
            **🟡 Gamma Flip Point:**
            - Critical level where net GEX crosses zero
            - Represents a regime change between volatility suppression/amplification
            - Acts as dynamic support/resistance
            
            #### Professional Applications:
            
            1. **Institutional Flow Analysis** - Understand dealer positioning and likely hedging flows
            2. **Volatility Regime Identification** - Adapt strategies based on current gamma environment  
            3. **Options Strategy Selection** - Choose optimal strategies based on GEX structure
            4. **Risk Management** - Position size and hedge based on gamma exposure levels
            5. **Market Timing** - Enter positions when GEX supports desired price action
            
            #### Strategic Framework:
            - **High Confidence Setups (80%+):** Core position sizing, institutional focus
            - **Medium Confidence (70-79%):** Reduced sizing, tactical opportunities  
            - **Lower Confidence (60-69%):** Paper trading, strategy validation
            
            This platform provides institutional-grade analysis typically available only to professional trading desks.
            """)
    
    # Footer
    st.markdown("---")
    
    footer_col1, footer_col2, footer_col3 = st.columns(3)
    
    with footer_col1:
        if st.session_state.last_update:
            st.caption(f"📊 Last Analysis: {st.session_state.last_update.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            st.caption("📊 Professional GEX Analysis Platform - Ready")
    
    with footer_col2:
        st.caption(f"🎯 Symbols: {len(st.session_state.gex_data)} | Setups: {len(st.session_state.trade_setups)}")
    
    with footer_col3:
        st.caption("⚡ Professional Grade Market Intelligence")

if __name__ == "__main__":
    main()
