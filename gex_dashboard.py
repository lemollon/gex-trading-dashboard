#!/usr/bin/env python3
"""
Complete GEX Trading Dashboard with All Features
Fixed NaN handling and comprehensive strategy implementation
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, date, timedelta
import time
import warnings
from scipy.stats import norm
import json
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="GEX Trading Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }
    
    .strategy-card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    .squeeze-signal {
        border-left: 4px solid #FF5722 !important;
        background: linear-gradient(135deg, #FFF3E0, #FFE0B2) !important;
    }
    
    .premium-signal {
        border-left: 4px solid #4CAF50 !important;
        background: linear-gradient(135deg, #E8F5E9, #C8E6C9) !important;
    }
    
    .condor-signal {
        border-left: 4px solid #2196F3 !important;
        background: linear-gradient(135deg, #E3F2FD, #BBDEFB) !important;
    }
    
    .wall-card {
        background: #F5F5F5;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .metric-card {
        background: white;
        border-radius: 8px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    
    .alert-box {
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    
    .high-priority {
        background: #FFEBEE;
        border-left: 4px solid #F44336;
    }
    
    .medium-priority {
        background: #FFF3E0;
        border-left: 4px solid #FF9800;
    }
</style>
""", unsafe_allow_html=True)

class GEXAnalyzer:
    """Complete GEX analyzer with all strategies and features"""
    
    def __init__(self):
        self.risk_free_rate = 0.05
        self.strategies_config = self.load_strategies_config()
        
    def load_strategies_config(self):
        """Load strategy configurations"""
        return {
            'squeeze_plays': {
                'negative_gex_threshold_spy': -1e9,  # -1B for SPY
                'negative_gex_threshold_qqq': -500e6,  # -500M for QQQ
                'positive_gex_threshold_spy': 2e9,  # 2B for SPY
                'positive_gex_threshold_qqq': 1e9,  # 1B for QQQ
                'flip_distance_threshold': 1.5,  # 1.5% from flip
                'dte_range': [2, 7],
                'confidence_threshold': 65
            },
            'premium_selling': {
                'positive_gex_threshold': 3e9,  # 3B for calls
                'wall_strength_threshold': 500e6,  # 500M concentration
                'wall_distance_range': [1, 5],  # 1-5% for calls
                'put_distance_range': [1, 8],  # 1-8% for puts
                'dte_range_calls': [0, 2],
                'dte_range_puts': [2, 5]
            },
            'iron_condor': {
                'min_gex_threshold': 1e9,  # 1B minimum
                'min_wall_spread': 3,  # 3% minimum spread
                'dte_range': [5, 10],
                'iv_rank_threshold': 50
            },
            'risk_management': {
                'max_position_size_squeeze': 0.03,  # 3% of capital
                'max_position_size_premium': 0.05,  # 5% of capital
                'max_position_size_condor': 0.02,  # 2% max loss
                'stop_loss_percentage': 0.50,  # 50% stop loss
                'profit_target_long': 1.00,  # 100% profit target
                'profit_target_short': 0.50  # 50% profit target
            }
        }
    
    def black_scholes_gamma(self, S, K, T, r, sigma):
        """Calculate Black-Scholes gamma"""
        try:
            if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
                return 0
            
            d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
            gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
            return gamma if not np.isnan(gamma) else 0
        except:
            return 0
    
    def get_current_price(self, symbol):
        """Get current stock price with fallbacks"""
        try:
            ticker = yf.Ticker(symbol)
            
            # Try intraday data first
            hist = ticker.history(period="1d", interval="1m")
            if len(hist) > 0:
                return float(hist['Close'].iloc[-1])
            
            # Fallback to daily
            hist = ticker.history(period="5d")
            if len(hist) > 0:
                return float(hist['Close'].iloc[-1])
            
            return None
        except Exception as e:
            st.error(f"Error getting price for {symbol}: {str(e)}")
            return None
    
    def get_options_chain(self, symbol):
        """Get complete options chain with error handling"""
        try:
            ticker = yf.Ticker(symbol)
            
            # Get expiration dates
            exp_dates = ticker.options
            if not exp_dates:
                return None
            
            current_price = self.get_current_price(symbol)
            if current_price is None:
                return None
            
            all_chains = {}
            
            # Process first 10 expirations or 90 days
            for exp_date in exp_dates[:10]:
                try:
                    exp_dt = datetime.strptime(exp_date, '%Y-%m-%d')
                    dte = (exp_dt.date() - date.today()).days
                    
                    if dte <= 0 or dte > 90:
                        continue
                    
                    chain = ticker.option_chain(exp_date)
                    
                    # Process calls
                    calls = chain.calls.copy()
                    calls = calls[calls['openInterest'] > 0]
                    
                    # Process puts
                    puts = chain.puts.copy()
                    puts = puts[puts['openInterest'] > 0]
                    
                    if len(calls) == 0 and len(puts) == 0:
                        continue
                    
                    # Calculate gamma
                    T = dte / 365.0
                    
                    # Safe gamma calculation for calls
                    calls['gamma'] = calls.apply(
                        lambda row: self.black_scholes_gamma(
                            current_price, 
                            row['strike'], 
                            T, 
                            self.risk_free_rate,
                            max(row['impliedVolatility'], 0.15) if pd.notna(row['impliedVolatility']) else 0.30
                        ), axis=1
                    )
                    
                    # Safe gamma calculation for puts
                    puts['gamma'] = puts.apply(
                        lambda row: self.black_scholes_gamma(
                            current_price,
                            row['strike'],
                            T,
                            self.risk_free_rate,
                            max(row['impliedVolatility'], 0.15) if pd.notna(row['impliedVolatility']) else 0.30
                        ), axis=1
                    )
                    
                    # Calculate GEX (handle NaN values)
                    calls['gex'] = current_price * calls['gamma'] * calls['openInterest'] * 100
                    puts['gex'] = -current_price * puts['gamma'] * puts['openInterest'] * 100
                    
                    # Replace any NaN values with 0
                    calls['gex'] = calls['gex'].fillna(0)
                    puts['gex'] = puts['gex'].fillna(0)
                    
                    all_chains[exp_date] = {
                        'calls': calls,
                        'puts': puts,
                        'dte': dte,
                        'expiration': exp_dt
                    }
                    
                except Exception as e:
                    continue
            
            return {
                'chains': all_chains,
                'current_price': current_price,
                'symbol': symbol
            }
            
        except Exception as e:
            st.error(f"Error getting options chain: {str(e)}")
            return None
    
    def calculate_gex_profile(self, options_data):
        """Calculate complete GEX profile"""
        try:
            if not options_data or 'chains' not in options_data:
                return None
            
            current_price = options_data['current_price']
            chains = options_data['chains']
            
            if not chains:
                return None
            
            # Aggregate GEX by strike
            strike_data = {}
            
            for exp_date, chain_data in chains.items():
                calls = chain_data['calls']
                puts = chain_data['puts']
                
                # Process calls
                for _, call in calls.iterrows():
                    strike = float(call['strike'])
                    gex = float(call['gex']) if pd.notna(call['gex']) else 0
                    oi = int(call['openInterest']) if pd.notna(call['openInterest']) else 0
                    volume = int(call.get('volume', 0)) if pd.notna(call.get('volume', 0)) else 0
                    
                    if strike not in strike_data:
                        strike_data[strike] = {
                            'call_gex': 0, 'put_gex': 0,
                            'call_oi': 0, 'put_oi': 0,
                            'call_volume': 0, 'put_volume': 0
                        }
                    
                    strike_data[strike]['call_gex'] += gex
                    strike_data[strike]['call_oi'] += oi
                    strike_data[strike]['call_volume'] += volume
                
                # Process puts
                for _, put in puts.iterrows():
                    strike = float(put['strike'])
                    gex = float(put['gex']) if pd.notna(put['gex']) else 0
                    oi = int(put['openInterest']) if pd.notna(put['openInterest']) else 0
                    volume = int(put.get('volume', 0)) if pd.notna(put.get('volume', 0)) else 0
                    
                    if strike not in strike_data:
                        strike_data[strike] = {
                            'call_gex': 0, 'put_gex': 0,
                            'call_oi': 0, 'put_oi': 0,
                            'call_volume': 0, 'put_volume': 0
                        }
                    
                    strike_data[strike]['put_gex'] += gex
                    strike_data[strike]['put_oi'] += oi
                    strike_data[strike]['put_volume'] += volume
            
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(strike_data, orient='index')
            df.reset_index(inplace=True)
            df.rename(columns={'index': 'strike'}, inplace=True)
            df = df.sort_values('strike').reset_index(drop=True)
            
            # Calculate net GEX
            df['net_gex'] = df['call_gex'] + df['put_gex']
            df['cumulative_gex'] = df['net_gex'].cumsum()
            
            # Find gamma flip point
            gamma_flip = self.find_gamma_flip(df, current_price)
            
            # Identify walls
            call_walls = df[df['call_gex'] > 0].nlargest(5, 'call_gex')
            put_walls = df[df['put_gex'] < 0].nsmallest(5, 'put_gex')
            
            # Calculate totals (safe aggregation)
            total_call_gex = float(df['call_gex'].sum())
            total_put_gex = float(df['put_gex'].sum())
            net_gex = total_call_gex + total_put_gex
            total_volume = int(df['call_volume'].sum() + df['put_volume'].sum())
            total_oi = int(df['call_oi'].sum() + df['put_oi'].sum())
            
            distance_to_flip = ((current_price - gamma_flip) / current_price) * 100
            
            return {
                'strike_data': df,
                'current_price': current_price,
                'gamma_flip': gamma_flip,
                'net_gex': net_gex,
                'total_call_gex': total_call_gex,
                'total_put_gex': total_put_gex,
                'call_walls': call_walls,
                'put_walls': put_walls,
                'total_volume': total_volume,
                'total_oi': total_oi,
                'distance_to_flip': distance_to_flip
            }
            
        except Exception as e:
            st.error(f"Error in GEX calculation: {str(e)}")
            return None
    
    def find_gamma_flip(self, df, current_price):
        """Find the gamma flip point"""
        try:
            # Look for zero crossing in cumulative GEX
            for i in range(len(df) - 1):
                curr = df.iloc[i]['cumulative_gex']
                next_val = df.iloc[i + 1]['cumulative_gex']
                
                if (curr <= 0 <= next_val) or (curr >= 0 >= next_val):
                    # Linear interpolation
                    curr_strike = df.iloc[i]['strike']
                    next_strike = df.iloc[i + 1]['strike']
                    
                    if next_val != curr:
                        ratio = abs(curr) / abs(next_val - curr)
                        flip = curr_strike + ratio * (next_strike - curr_strike)
                        return flip
            
            # If no crossing, return strike with minimum absolute cumulative GEX
            min_idx = df['cumulative_gex'].abs().idxmin()
            return df.loc[min_idx, 'strike']
            
        except:
            return current_price
    
    def generate_squeeze_signals(self, gex_profile, symbol):
        """Generate squeeze play signals"""
        signals = []
        config = self.strategies_config['squeeze_plays']
        
        net_gex = gex_profile['net_gex']
        distance_to_flip = gex_profile['distance_to_flip']
        current_price = gex_profile['current_price']
        gamma_flip = gex_profile['gamma_flip']
        
        # Adjust thresholds by symbol
        neg_threshold = config['negative_gex_threshold_spy'] if symbol == 'SPY' else config['negative_gex_threshold_qqq']
        pos_threshold = config['positive_gex_threshold_spy'] if symbol == 'SPY' else config['positive_gex_threshold_qqq']
        
        # Negative GEX squeeze (Long Calls)
        if net_gex < neg_threshold and distance_to_flip < -0.5:
            confidence = min(85, 65 + abs(net_gex/neg_threshold) * 10 + abs(distance_to_flip) * 5)
            
            target_strike = gamma_flip if gamma_flip > current_price else current_price * 1.01
            
            signals.append({
                'type': 'SQUEEZE_PLAY',
                'direction': 'LONG_CALL',
                'confidence': confidence,
                'entry': f"Buy calls at/above ${gamma_flip:.2f}",
                'target': f"${target_strike * 1.02:.2f}",
                'stop': f"${current_price * 0.98:.2f}",
                'dte': f"{config['dte_range'][0]}-{config['dte_range'][1]} DTE",
                'size': f"{self.strategies_config['risk_management']['max_position_size_squeeze']*100:.0f}% max",
                'reasoning': f"Negative GEX: {net_gex/1e6:.0f}M, Price {abs(distance_to_flip):.1f}% below flip"
            })
        
        # Positive GEX breakdown (Long Puts)
        if net_gex > pos_threshold and abs(distance_to_flip) < 0.5:
            confidence = min(75, 60 + (net_gex/pos_threshold) * 10 + (0.5 - abs(distance_to_flip)) * 20)
            
            signals.append({
                'type': 'SQUEEZE_PLAY',
                'direction': 'LONG_PUT',
                'confidence': confidence,
                'entry': f"Buy puts at/below ${gamma_flip:.2f}",
                'target': f"${current_price * 0.97:.2f}",
                'stop': f"${current_price * 1.02:.2f}",
                'dte': f"3-7 DTE",
                'size': f"{self.strategies_config['risk_management']['max_position_size_squeeze']*100:.0f}% max",
                'reasoning': f"High positive GEX: {net_gex/1e6:.0f}M near flip point"
            })
        
        return signals
    
    def generate_premium_signals(self, gex_profile):
        """Generate premium selling signals"""
        signals = []
        config = self.strategies_config['premium_selling']
        
        net_gex = gex_profile['net_gex']
        current_price = gex_profile['current_price']
        call_walls = gex_profile['call_walls']
        put_walls = gex_profile['put_walls']
        
        # Call selling opportunities
        if net_gex > config['positive_gex_threshold'] and len(call_walls) > 0:
            strongest_call = call_walls.iloc[0]
            wall_distance = ((strongest_call['strike'] - current_price) / current_price) * 100
            
            if config['wall_distance_range'][0] < wall_distance < config['wall_distance_range'][1]:
                wall_strength = strongest_call['call_gex']
                confidence = min(80, 60 + (wall_strength/config['wall_strength_threshold']) * 10)
                
                signals.append({
                    'type': 'PREMIUM_SELLING',
                    'direction': 'SELL_CALL',
                    'confidence': confidence,
                    'entry': f"Sell calls at ${strongest_call['strike']:.2f}",
                    'target': "50% profit or expiration",
                    'stop': f"Price crosses ${strongest_call['strike']:.2f}",
                    'dte': f"{config['dte_range_calls'][0]}-{config['dte_range_calls'][1]} DTE",
                    'size': f"{self.strategies_config['risk_management']['max_position_size_premium']*100:.0f}% max",
                    'reasoning': f"Strong call wall ({wall_strength/1e6:.0f}M GEX) at {wall_distance:.1f}% above"
                })
        
        # Put selling opportunities
        if net_gex > 0 and len(put_walls) > 0:
            strongest_put = put_walls.iloc[0]
            wall_distance = ((current_price - strongest_put['strike']) / current_price) * 100
            
            if config['put_distance_range'][0] < wall_distance < config['put_distance_range'][1]:
                wall_strength = abs(strongest_put['put_gex'])
                confidence = min(75, 55 + (wall_strength/config['wall_strength_threshold']) * 10)
                
                signals.append({
                    'type': 'PREMIUM_SELLING',
                    'direction': 'SELL_PUT',
                    'confidence': confidence,
                    'entry': f"Sell puts at ${strongest_put['strike']:.2f}",
                    'target': "50% profit or expiration",
                    'stop': f"Price crosses ${strongest_put['strike']:.2f}",
                    'dte': f"{config['dte_range_puts'][0]}-{config['dte_range_puts'][1]} DTE",
                    'size': f"{self.strategies_config['risk_management']['max_position_size_premium']*100:.0f}% max",
                    'reasoning': f"Strong put wall ({wall_strength/1e6:.0f}M GEX) at {wall_distance:.1f}% below"
                })
        
        return signals
    
    def generate_condor_signals(self, gex_profile):
        """Generate iron condor signals"""
        signals = []
        config = self.strategies_config['iron_condor']
        
        net_gex = gex_profile['net_gex']
        call_walls = gex_profile['call_walls']
        put_walls = gex_profile['put_walls']
        current_price = gex_profile['current_price']
        
        if net_gex > config['min_gex_threshold'] and len(call_walls) > 0 and len(put_walls) > 0:
            call_strike = call_walls.iloc[0]['strike']
            put_strike = put_walls.iloc[0]['strike']
            
            range_width = ((call_strike - put_strike) / current_price) * 100
            
            if range_width > config['min_wall_spread']:
                # Calculate wing adjustments based on gamma bias
                call_gamma = gex_profile['total_call_gex']
                put_gamma = abs(gex_profile['total_put_gex'])
                
                if put_gamma > call_gamma:
                    wing_adjustment = "Wider put spread (bullish bias)"
                elif call_gamma > put_gamma:
                    wing_adjustment = "Wider call spread (bearish bias)"
                else:
                    wing_adjustment = "Balanced wings"
                
                confidence = min(85, 65 + (range_width - config['min_wall_spread']) * 2)
                
                signals.append({
                    'type': 'IRON_CONDOR',
                    'direction': 'NEUTRAL',
                    'confidence': confidence,
                    'entry': f"Short {put_strike:.0f}P/{call_strike:.0f}C",
                    'wings': wing_adjustment,
                    'target': "25% profit or 50% of max profit",
                    'stop': "Short strike threatened",
                    'dte': f"{config['dte_range'][0]}-{config['dte_range'][1]} DTE",
                    'size': f"{self.strategies_config['risk_management']['max_position_size_condor']*100:.0f}% max loss",
                    'reasoning': f"Clear {range_width:.1f}% range with {net_gex/1e6:.0f}M positive GEX"
                })
        
        return signals
    
    def generate_alerts(self, gex_profile, symbol):
        """Generate trading alerts"""
        alerts = []
        
        net_gex = gex_profile['net_gex']
        distance_to_flip = gex_profile['distance_to_flip']
        current_price = gex_profile['current_price']
        
        # High priority alerts
        if net_gex < -1e9:
            alerts.append({
                'priority': 'HIGH',
                'message': f'⚠️ Extreme negative GEX: {net_gex/1e6:.0f}M - High volatility expected',
                'action': 'Consider long gamma strategies'
            })
        
        if abs(distance_to_flip) < 0.25:
            alerts.append({
                'priority': 'HIGH',
                'message': f'⚡ Price within 0.25% of gamma flip - Regime change imminent',
                'action': 'Monitor for directional breakout'
            })
        
        # Medium priority alerts
        if len(gex_profile['call_walls']) > 0:
            nearest_call = gex_profile['call_walls'].iloc[0]
            call_distance = ((nearest_call['strike'] - current_price) / current_price) * 100
            
            if call_distance < 0.5:
                alerts.append({
                    'priority': 'MEDIUM',
                    'message': f'📊 Approaching call wall at ${nearest_call["strike"]:.2f}',
                    'action': 'Consider resistance plays'
                })
        
        return alerts

# Initialize analyzer
@st.cache_resource
def get_analyzer():
    return GEXAnalyzer()

analyzer = get_analyzer()

# Header
st.markdown("""
<div class="main-header">
    <h1>📊 Professional GEX Trading Dashboard</h1>
    <p>Complete gamma exposure analysis with squeeze, premium, and condor strategies</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("🎛️ Dashboard Controls")
    
    # Symbol input
    symbol = st.text_input(
        "Symbol to Analyze",
        value="SPY",
        help="Enter stock symbol (e.g., SPY, QQQ, AAPL)"
    ).upper().strip()
    
    # Quick select buttons
    st.markdown("### Quick Select")
    col1, col2, col3 = st.columns(3)
    
    symbols = ["SPY", "QQQ", "IWM", "AAPL", "TSLA", "NVDA"]
    for i, sym in enumerate(symbols):
        with [col1, col2, col3][i % 3]:
            if st.button(sym, key=f"btn_{sym}", use_container_width=True):
                symbol = sym
    
    # Strategy filters
    st.markdown("### Strategy Filters")
    show_squeeze = st.checkbox("Squeeze Plays", value=True)
    show_premium = st.checkbox("Premium Selling", value=True)
    show_condor = st.checkbox("Iron Condors", value=True)
    
    # Risk settings
    st.markdown("### Risk Management")
    capital = st.number_input("Trading Capital ($)", value=100000, step=1000)
    
    # Refresh controls
    st.markdown("### Refresh Settings")
    auto_refresh = st.selectbox(
        "Auto-refresh",
        ["Manual", "30 seconds", "1 minute", "5 minutes"],
        index=0
    )
    
    if st.button("🔄 Refresh Now", type="primary", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# Main content
if symbol:
    # Analysis tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 GEX Analysis", 
        "🎯 Trading Signals", 
        "📈 Options Flow",
        "⚠️ Alerts & Risk",
        "📋 Position Manager"
    ])
    
    # Get options data
    with st.spinner(f"Analyzing {symbol}..."):
        options_data = analyzer.get_options_chain(symbol)
    
    if options_data:
        gex_profile = analyzer.calculate_gex_profile(options_data)
        
        if gex_profile:
            # Tab 1: GEX Analysis
            with tab1:
                # Key metrics
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.markdown("""
                    <div class="metric-card">
                        <h4>Current Price</h4>
                        <h2>${:.2f}</h2>
                    </div>
                    """.format(gex_profile['current_price']), unsafe_allow_html=True)
                
                with col2:
                    net_gex = gex_profile['net_gex']
                    gex_color = "#4CAF50" if net_gex > 0 else "#F44336"
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>Net GEX</h4>
                        <h2 style="color: {gex_color}">{net_gex/1e6:.0f}M</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown("""
                    <div class="metric-card">
                        <h4>Gamma Flip</h4>
                        <h2>${:.2f}</h2>
                    </div>
                    """.format(gex_profile['gamma_flip']), unsafe_allow_html=True)
                
                with col4:
                    dist = gex_profile['distance_to_flip']
                    dist_color = "#FF9800" if abs(dist) < 1 else "#757575"
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>Flip Distance</h4>
                        <h2 style="color: {dist_color}">{dist:+.2f}%</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col5:
                    st.markdown("""
                    <div class="metric-card">
                        <h4>Total Volume</h4>
                        <h2>{:,}</h2>
                    </div>
                    """.format(gex_profile['total_volume']), unsafe_allow_html=True)
                
                # GEX visualization
                st.markdown("### Gamma Exposure Profile")
                
                fig = make_subplots(
                    rows=2, cols=1,
                    row_heights=[0.7, 0.3],
                    subplot_titles=("Strike-Level GEX", "Cumulative GEX"),
                    vertical_spacing=0.12
                )
                
                strike_data = gex_profile['strike_data']
                current_price = gex_profile['current_price']
                
                # Filter for display range
                display_range = strike_data[
                    (strike_data['strike'] >= current_price * 0.9) &
                    (strike_data['strike'] <= current_price * 1.1)
                ]
                
                # GEX bars
                fig.add_trace(
                    go.Bar(
                        x=display_range['strike'],
                        y=display_range['call_gex'] / 1e6,
                        name='Call GEX',
                        marker_color='green',
                        opacity=0.7
                    ), row=1, col=1
                )
                
                fig.add_trace(
                    go.Bar(
                        x=display_range['strike'],
                        y=display_range['put_gex'] / 1e6,
                        name='Put GEX',
                        marker_color='red',
                        opacity=0.7
                    ), row=1, col=1
                )
                
                # Cumulative GEX
                fig.add_trace(
                    go.Scatter(
                        x=display_range['strike'],
                        y=display_range['cumulative_gex'] / 1e6,
                        mode='lines',
                        name='Cumulative',
                        line=dict(color='purple', width=2)
                    ), row=2, col=1
                )
                
                # Add reference lines
                fig.add_vline(x=current_price, line_dash="solid", line_color="blue", row=1, col=1)
                fig.add_vline(x=gex_profile['gamma_flip'], line_dash="dash", line_color="orange", row=1, col=1)
                
                fig.update_layout(height=600, showlegend=True)
                fig.update_xaxes(title_text="Strike Price", row=2, col=1)
                fig.update_yaxes(title_text="GEX (Millions)", row=1, col=1)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Walls analysis
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 🟢 Call Walls (Resistance)")
                    call_walls = gex_profile['call_walls'].head(3)
                    for _, wall in call_walls.iterrows():
                        dist = ((wall['strike'] - current_price) / current_price) * 100
                        st.markdown(f"""
                        <div class="wall-card">
                            <strong>${wall['strike']:.2f}</strong> (+{dist:.1f}%)<br>
                            GEX: {wall['call_gex']/1e6:.1f}M | OI: {wall['call_oi']:,}
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("### 🔴 Put Walls (Support)")
                    put_walls = gex_profile['put_walls'].head(3)
                    for _, wall in put_walls.iterrows():
                        dist = ((wall['strike'] - current_price) / current_price) * 100
                        st.markdown(f"""
                        <div class="wall-card">
                            <strong>${wall['strike']:.2f}</strong> ({dist:.1f}%)<br>
                            GEX: {abs(wall['put_gex'])/1e6:.1f}M | OI: {wall['put_oi']:,}
                        </div>
                        """, unsafe_allow_html=True)
            
            # Tab 2: Trading Signals
            with tab2:
                st.markdown("### 🎯 Active Trading Signals")
                
                all_signals = []
                
                if show_squeeze:
                    squeeze_signals = analyzer.generate_squeeze_signals(gex_profile, symbol)
                    all_signals.extend(squeeze_signals)
                
                if show_premium:
                    premium_signals = analyzer.generate_premium_signals(gex_profile)
                    all_signals.extend(premium_signals)
                
                if show_condor:
                    condor_signals = analyzer.generate_condor_signals(gex_profile)
                    all_signals.extend(condor_signals)
                
                # Sort by confidence
                all_signals.sort(key=lambda x: x['confidence'], reverse=True)
                
                if all_signals:
                    for signal in all_signals:
                        # Determine card style
                        if signal['type'] == 'SQUEEZE_PLAY':
                            card_class = "strategy-card squeeze-signal"
                            icon = "⚡"
                        elif signal['type'] == 'PREMIUM_SELLING':
                            card_class = "strategy-card premium-signal"
                            icon = "💰"
                        else:
                            card_class = "strategy-card condor-signal"
                            icon = "🦅"
                        
                        # Calculate position size
                        if capital:
                            if 'SQUEEZE' in signal['type']:
                                position_size = capital * analyzer.strategies_config['risk_management']['max_position_size_squeeze']
                            elif 'PREMIUM' in signal['type']:
                                position_size = capital * analyzer.strategies_config['risk_management']['max_position_size_premium']
                            else:
                                position_size = capital * analyzer.strategies_config['risk_management']['max_position_size_condor']
                        
                        st.markdown(f"""
                        <div class="{card_class}">
                            <h3>{icon} {signal['direction']} - {signal['confidence']:.0f}% Confidence</h3>
                            <p><strong>Entry:</strong> {signal['entry']}</p>
                            <p><strong>Target:</strong> {signal['target']}</p>
                            <p><strong>Stop:</strong> {signal['stop']}</p>
                            <p><strong>Timeframe:</strong> {signal['dte']}</p>
                            <p><strong>Position Size:</strong> ${position_size:,.0f} ({signal['size']})</p>
                            <p><strong>Reasoning:</strong> {signal['reasoning']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No high-confidence signals detected for current market conditions")
            
            # Tab 3: Options Flow
            with tab3:
                st.markdown("### 📊 Options Chain Analysis")
                
                # Aggregate chain data
                chain_summary = []
                for exp_date, chain_data in options_data['chains'].items():
                    calls = chain_data['calls']
                    puts = chain_data['puts']
                    
                    summary = {
                        'Expiration': exp_date,
                        'DTE': chain_data['dte'],
                        'Call Volume': int(calls['volume'].sum()),
                        'Put Volume': int(puts['volume'].sum()),
                        'Call OI': int(calls['openInterest'].sum()),
                        'Put OI': int(puts['openInterest'].sum()),
                        'Call GEX': calls['gex'].sum() / 1e6,
                        'Put GEX': puts['gex'].sum() / 1e6
                    }
                    chain_summary.append(summary)
                
                df_summary = pd.DataFrame(chain_summary)
                
                st.dataframe(
                    df_summary.style.format({
                        'Call Volume': '{:,}',
                        'Put Volume': '{:,}',
                        'Call OI': '{:,}',
                        'Put OI': '{:,}',
                        'Call GEX': '{:.1f}M',
                        'Put GEX': '{:.1f}M'
                    }),
                    use_container_width=True
                )
                
                # Put/Call ratio chart
                fig_ratio = go.Figure()
                
                df_summary['P/C Volume'] = df_summary['Put Volume'] / df_summary['Call Volume'].replace(0, 1)
                df_summary['P/C OI'] = df_summary['Put OI'] / df_summary['Call OI'].replace(0, 1)
                
                fig_ratio.add_trace(go.Bar(
                    x=df_summary['Expiration'],
                    y=df_summary['P/C Volume'],
                    name='P/C Volume Ratio',
                    marker_color='orange'
                ))
                
                fig_ratio.add_trace(go.Bar(
                    x=df_summary['Expiration'],
                    y=df_summary['P/C OI'],
                    name='P/C OI Ratio',
                    marker_color='purple'
                ))
                
                fig_ratio.update_layout(
                    title="Put/Call Ratios by Expiration",
                    xaxis_title="Expiration",
                    yaxis_title="Ratio",
                    height=400
                )
                
                st.plotly_chart(fig_ratio, use_container_width=True)
            
            # Tab 4: Alerts & Risk
            with tab4:
                st.markdown("### ⚠️ Active Alerts")
                
                alerts = analyzer.generate_alerts(gex_profile, symbol)
                
                if alerts:
                    for alert in alerts:
                        if alert['priority'] == 'HIGH':
                            alert_class = "alert-box high-priority"
                        else:
                            alert_class = "alert-box medium-priority"
                        
                        st.markdown(f"""
                        <div class="{alert_class}">
                            <strong>{alert['priority']} PRIORITY</strong><br>
                            {alert['message']}<br>
                            <em>Action: {alert['action']}</em>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.success("✅ No critical alerts at this time")
                
                # Risk metrics
                st.markdown("### 📊 Risk Metrics")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    volatility_regime = "High" if gex_profile['net_gex'] < 0 else "Low"
                    st.metric("Volatility Regime", volatility_regime)
                
                with col2:
                    gamma_concentration = abs(gex_profile['net_gex']) / max(gex_profile['total_oi'], 1)
                    st.metric("Gamma Concentration", f"{gamma_concentration:.2f}")
                
                with col3:
                    wall_spread = 0
                    if len(gex_profile['call_walls']) > 0 and len(gex_profile['put_walls']) > 0:
                        wall_spread = ((gex_profile['call_walls'].iloc[0]['strike'] - 
                                      gex_profile['put_walls'].iloc[0]['strike']) / 
                                      gex_profile['current_price'] * 100)
                    st.metric("Wall Spread", f"{wall_spread:.1f}%")
            
            # Tab 5: Position Manager
            with tab5:
                st.markdown("### 📋 Position Management")
                
                # Position tracker (mock data for demonstration)
                st.info("Position tracking requires live trading integration")
                
                # Risk parameters display
                st.markdown("### ⚙️ Active Risk Parameters")
                
                risk_params = analyzer.strategies_config['risk_management']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"""
                    **Position Sizing:**
                    - Squeeze Plays: {risk_params['max_position_size_squeeze']*100:.0f}% of capital
                    - Premium Selling: {risk_params['max_position_size_premium']*100:.0f}% of capital
                    - Iron Condors: {risk_params['max_position_size_condor']*100:.0f}% max loss
                    """)
                
                with col2:
                    st.markdown(f"""
                    **Exit Rules:**
                    - Long Options: {risk_params['profit_target_long']*100:.0f}% profit / {risk_params['stop_loss_percentage']*100:.0f}% loss
                    - Short Options: {risk_params['profit_target_short']*100:.0f}% profit / 100% loss
                    - Iron Condors: 25% profit or threatened strike
                    """)
        
        else:
            st.error("Unable to calculate GEX profile")
    else:
        st.error(f"No options data available for {symbol}")

# Auto-refresh
if auto_refresh != "Manual":
    refresh_map = {
        "30 seconds": 30,
        "1 minute": 60,
        "5 minutes": 300
    }
    time.sleep(refresh_map[auto_refresh])
    st.rerun()

# Footer
st.markdown("---")
st.caption("GEX Trading Dashboard v2.0 | For educational purposes only")
