#!/usr/bin/env python3
"""
Complete GEX Dashboard with All Features Working
Fixed symbol analysis, gamma structures, and missing functionality
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
import concurrent.futures
from threading import Lock
import warnings
import math
from scipy.stats import norm
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
        background: linear-gradient(135deg, #2E7D32 0%, #388E3C 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }
    
    .analysis-card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 4px solid #2E7D32;
    }
    
    .signal-high {
        border-left: 4px solid #4CAF50 !important;
        background: linear-gradient(135deg, #E8F5E8, #F1F8E9) !important;
    }
    
    .signal-medium {
        border-left: 4px solid #FF9800 !important;
        background: linear-gradient(135deg, #FFF3E0, #FFF8E1) !important;
    }
    
    .signal-low {
        border-left: 4px solid #F44336 !important;
        background: linear-gradient(135deg, #FFEBEE, #FFECEE) !important;
    }
    
    .wall-info {
        background: #F5F5F5;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 3px solid #757575;
    }
    
    .call-wall {
        border-left: 3px solid #4CAF50 !important;
        background: #E8F5E8 !important;
    }
    
    .put-wall {
        border-left: 3px solid #F44336 !important;
        background: #FFEBEE !important;
    }
    
    .metric-box {
        background: white;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>📊 Professional GEX Trading Dashboard</h1>
    <p>Real-time gamma exposure analysis with complete options data</p>
</div>
""", unsafe_allow_html=True)

class CompletaGEXAnalyzer:
    """Complete GEX analyzer with all missing functionality restored"""
    
    def __init__(self):
        self.risk_free_rate = 0.05
        
    def black_scholes_gamma(self, S, K, T, r, sigma, option_type='call'):
        """Calculate theoretical gamma using Black-Scholes"""
        try:
            if T <= 0 or sigma <= 0:
                return 0
            
            d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
            gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
            return gamma
        except:
            return 0
    
    def get_current_price(self, symbol):
        """Get current stock price with error handling"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1d", interval="1m")
            if len(hist) > 0:
                return float(hist['Close'].iloc[-1])
            
            # Fallback to daily data
            hist = ticker.history(period="2d")
            if len(hist) > 0:
                return float(hist['Close'].iloc[-1])
            return None
        except Exception as e:
            st.error(f"Error getting price for {symbol}: {str(e)}")
            return None
    
    def get_options_chain_data(self, symbol):
        """Get complete options chain with all expirations"""
        try:
            ticker = yf.Ticker(symbol)
            
            # Get all expiration dates
            exp_dates = ticker.options
            if not exp_dates:
                return None
            
            all_chains = {}
            current_price = self.get_current_price(symbol)
            
            if current_price is None:
                return None
            
            for exp_date in exp_dates[:8]:  # First 8 expirations
                try:
                    exp_dt = datetime.strptime(exp_date, '%Y-%m-%d')
                    dte = (exp_dt.date() - date.today()).days
                    
                    if dte <= 0 or dte > 90:  # Skip expired or too far out
                        continue
                    
                    chain = ticker.option_chain(exp_date)
                    
                    # Clean the data
                    calls = chain.calls.copy()
                    puts = chain.puts.copy()
                    
                    # Filter out options with no volume or OI
                    calls = calls[(calls['openInterest'] > 0) | (calls['volume'] > 0)]
                    puts = puts[(puts['openInterest'] > 0) | (puts['volume'] > 0)]
                    
                    # Add calculated fields
                    T = dte / 365.0
                    
                    # Calculate implied gamma for calls
                    calls['calc_gamma'] = calls.apply(
                        lambda row: self.black_scholes_gamma(
                            current_price, row['strike'], T, self.risk_free_rate, 
                            row['impliedVolatility'] if row['impliedVolatility'] > 0 else 0.3
                        ), axis=1
                    )
                    
                    # Calculate implied gamma for puts  
                    puts['calc_gamma'] = puts.apply(
                        lambda row: self.black_scholes_gamma(
                            current_price, row['strike'], T, self.risk_free_rate,
                            row['impliedVolatility'] if row['impliedVolatility'] > 0 else 0.3
                        ), axis=1
                    )
                    
                    # Calculate GEX
                    calls['gex'] = current_price * calls['calc_gamma'] * calls['openInterest'] * 100
                    puts['gex'] = -current_price * puts['calc_gamma'] * puts['openInterest'] * 100  # Negative for puts
                    
                    all_chains[exp_date] = {
                        'calls': calls,
                        'puts': puts,
                        'dte': dte,
                        'expiration': exp_dt
                    }
                    
                except Exception as e:
                    st.warning(f"Error processing {exp_date}: {str(e)}")
                    continue
            
            return {
                'chains': all_chains,
                'current_price': current_price,
                'symbol': symbol
            }
            
        except Exception as e:
            st.error(f"Error getting options data for {symbol}: {str(e)}")
            return None
    
    def calculate_gex_profile(self, options_data):
        """Calculate complete GEX profile with walls and flip point"""
        try:
            if not options_data or 'chains' not in options_data:
                return None
            
            current_price = options_data['current_price']
            chains = options_data['chains']
            
            # Aggregate all strikes
            strike_gex = {}
            total_call_gex = 0
            total_put_gex = 0
            total_volume = 0
            total_oi = 0
            
            for exp_date, chain_data in chains.items():
                calls = chain_data['calls']
                puts = chain_data['puts']
                
                # Process calls
                for _, call in calls.iterrows():
                    strike = call['strike']
                    gex = call['gex']
                    
                    if strike not in strike_gex:
                        strike_gex[strike] = {
                            'call_gex': 0, 'put_gex': 0, 'net_gex': 0,
                            'call_oi': 0, 'put_oi': 0, 'call_volume': 0, 'put_volume': 0
                        }
                    
                    strike_gex[strike]['call_gex'] += gex
                    strike_gex[strike]['call_oi'] += call['openInterest']
                    strike_gex[strike]['call_volume'] += call.get('volume', 0)
                    total_call_gex += gex
                    total_volume += call.get('volume', 0)
                    total_oi += call['openInterest']
                
                # Process puts
                for _, put in puts.iterrows():
                    strike = put['strike']
                    gex = put['gex']  # Already negative
                    
                    if strike not in strike_gex:
                        strike_gex[strike] = {
                            'call_gex': 0, 'put_gex': 0, 'net_gex': 0,
                            'call_oi': 0, 'put_oi': 0, 'call_volume': 0, 'put_volume': 0
                        }
                    
                    strike_gex[strike]['put_gex'] += gex
                    strike_gex[strike]['put_oi'] += put['openInterest']
                    strike_gex[strike]['put_volume'] += put.get('volume', 0)
                    total_put_gex += gex
                    total_volume += put.get('volume', 0)
                    total_oi += put['openInterest']
            
            # Calculate net GEX for each strike
            for strike in strike_gex:
                strike_gex[strike]['net_gex'] = strike_gex[strike]['call_gex'] + strike_gex[strike]['put_gex']
            
            # Convert to DataFrame
            gex_df = pd.DataFrame.from_dict(strike_gex, orient='index')
            gex_df.reset_index(inplace=True)
            gex_df.rename(columns={'index': 'strike'}, inplace=True)
            gex_df = gex_df.sort_values('strike').reset_index(drop=True)
            
            # Calculate cumulative GEX
            gex_df['cumulative_gex'] = gex_df['net_gex'].cumsum()
            
            # Find gamma flip point (where cumulative GEX crosses zero)
            gamma_flip = current_price  # Default
            zero_crossings = []
            
            for i in range(len(gex_df) - 1):
                curr_cum = gex_df.iloc[i]['cumulative_gex']
                next_cum = gex_df.iloc[i + 1]['cumulative_gex']
                
                if (curr_cum <= 0 <= next_cum) or (curr_cum >= 0 >= next_cum):
                    # Linear interpolation to find exact crossing
                    curr_strike = gex_df.iloc[i]['strike']
                    next_strike = gex_df.iloc[i + 1]['strike']
                    
                    if next_cum != curr_cum:
                        ratio = abs(curr_cum) / abs(next_cum - curr_cum)
                        flip_strike = curr_strike + ratio * (next_strike - curr_strike)
                        zero_crossings.append(flip_strike)
            
            if zero_crossings:
                # Find crossing closest to current price
                gamma_flip = min(zero_crossings, key=lambda x: abs(x - current_price))
            
            # Identify walls (top gamma concentrations)
            call_walls = gex_df[gex_df['call_gex'] > 0].nlargest(5, 'call_gex')
            put_walls = gex_df[gex_df['put_gex'] < 0].nsmallest(5, 'put_gex')
            
            # Filter walls by proximity to current price (within ±20%)
            price_range_low = current_price * 0.8
            price_range_high = current_price * 1.2
            
            call_walls = call_walls[
                (call_walls['strike'] >= price_range_low) & 
                (call_walls['strike'] <= price_range_high)
            ]
            
            put_walls = put_walls[
                (put_walls['strike'] >= price_range_low) & 
                (put_walls['strike'] <= price_range_high)
            ]
            
            net_gex = total_call_gex + total_put_gex
            
            return {
                'strike_data': gex_df,
                'current_price': current_price,
                'gamma_flip': gamma_flip,
                'net_gex': net_gex,
                'total_call_gex': total_call_gex,
                'total_put_gex': total_put_gex,
                'call_walls': call_walls,
                'put_walls': put_walls,
                'total_volume': int(total_volume),
                'total_oi': int(total_oi),
                'distance_to_flip': ((current_price - gamma_flip) / current_price) * 100
            }
            
        except Exception as e:
            st.error(f"Error calculating GEX profile: {str(e)}")
            return None
    
    def analyze_symbol_complete(self, symbol):
        """Complete symbol analysis with all features"""
        try:
            symbol = symbol.upper().strip()
            
            # Get basic info
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period="5d")
            
            if len(hist) == 0:
                return {'error': f'No price data found for {symbol}'}
            
            current_price = hist['Close'].iloc[-1]
            prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
            price_change = ((current_price - prev_close) / prev_close) * 100
            
            # Get options data
            options_data = self.get_options_chain_data(symbol)
            if not options_data:
                return {
                    'error': f'No options data available for {symbol}',
                    'symbol': symbol,
                    'current_price': current_price,
                    'price_change': price_change
                }
            
            # Calculate GEX profile
            gex_profile = self.calculate_gex_profile(options_data)
            if not gex_profile:
                return {
                    'error': f'Unable to calculate GEX profile for {symbol}',
                    'symbol': symbol,
                    'current_price': current_price,
                    'price_change': price_change
                }
            
            # Generate signals
            signals = self.generate_trading_signals(gex_profile)
            
            return {
                'symbol': symbol,
                'company_name': info.get('longName', symbol),
                'sector': info.get('sector', 'Unknown'),
                'current_price': current_price,
                'price_change': price_change,
                'market_cap': info.get('marketCap', 0),
                'gex_profile': gex_profile,
                'signals': signals,
                'options_chains': options_data['chains'],
                'analysis_time': datetime.now()
            }
            
        except Exception as e:
            return {
                'error': f'Analysis failed for {symbol}: {str(e)}',
                'symbol': symbol
            }
    
    def generate_trading_signals(self, gex_profile):
        """Generate comprehensive trading signals"""
        try:
            net_gex = gex_profile['net_gex']
            distance_to_flip = gex_profile['distance_to_flip']
            current_price = gex_profile['current_price']
            call_walls = gex_profile['call_walls']
            put_walls = gex_profile['put_walls']
            
            signals = []
            
            # Gamma squeeze signals
            if net_gex < -50e6 and distance_to_flip < -0.5:
                signals.append({
                    'type': 'LONG_CALL_SQUEEZE',
                    'confidence': 85,
                    'description': f'Negative GEX ({net_gex/1e6:.0f}M) with price below flip - squeeze potential',
                    'strategy': 'Buy ATM/OTM calls, 2-7 DTE',
                    'target': call_walls.iloc[0]['strike'] if len(call_walls) > 0 else current_price * 1.05
                })
            
            # Breakdown signals
            if net_gex > 200e6 and abs(distance_to_flip) < 0.5:
                signals.append({
                    'type': 'LONG_PUT_BREAKDOWN',
                    'confidence': 75,
                    'description': f'High positive GEX ({net_gex/1e6:.0f}M) near flip - breakdown risk',
                    'strategy': 'Buy ATM/OTM puts, 3-7 DTE',
                    'target': put_walls.iloc[0]['strike'] if len(put_walls) > 0 else current_price * 0.95
                })
            
            # Premium selling signals
            if net_gex > 300e6 and len(call_walls) > 0:
                strongest_call_wall = call_walls.iloc[0]
                wall_distance = ((strongest_call_wall['strike'] - current_price) / current_price) * 100
                
                if 1 < wall_distance < 5:
                    signals.append({
                        'type': 'SELL_CALLS',
                        'confidence': 70,
                        'description': f'Strong call wall at ${strongest_call_wall["strike"]:.2f} with high positive GEX',
                        'strategy': f'Sell calls at ${strongest_call_wall["strike"]:.2f}, 7-21 DTE',
                        'target': 'Collect premium, expect resistance'
                    })
            
            if net_gex > 200e6 and len(put_walls) > 0:
                strongest_put_wall = put_walls.iloc[0]
                wall_distance = ((current_price - strongest_put_wall['strike']) / current_price) * 100
                
                if 1 < wall_distance < 8:
                    signals.append({
                        'type': 'SELL_PUTS',
                        'confidence': 65,
                        'description': f'Strong put wall at ${strongest_put_wall["strike"]:.2f} with dealer support',
                        'strategy': f'Sell puts at ${strongest_put_wall["strike"]:.2f}, 14-30 DTE',
                        'target': 'Collect premium, expect support'
                    })
            
            # Iron condor signals
            if (net_gex > 500e6 and len(call_walls) > 0 and len(put_walls) > 0):
                call_strike = call_walls.iloc[0]['strike']
                put_strike = put_walls.iloc[0]['strike']
                range_width = ((call_strike - put_strike) / current_price) * 100
                
                if range_width > 4:
                    signals.append({
                        'type': 'IRON_CONDOR',
                        'confidence': 80,
                        'description': f'Clear range ${put_strike:.2f} - ${call_strike:.2f} ({range_width:.1f}%)',
                        'strategy': f'Iron condor: Short {put_strike:.0f}P/{call_strike:.0f}C',
                        'target': 'Range-bound trading, collect premium'
                    })
            
            return signals
            
        except Exception as e:
            st.error(f"Error generating signals: {str(e)}")
            return []

# Initialize analyzer
@st.cache_resource
def get_analyzer():
    return CompletaGEXAnalyzer()

analyzer = get_analyzer()

# Sidebar controls
st.sidebar.header("Dashboard Controls")

# Main symbol input
main_symbol = st.sidebar.text_input(
    "Primary Symbol Analysis",
    value="SPY",
    placeholder="Enter symbol (e.g., AAPL, SPY)"
).upper()

# Quick symbol buttons
st.sidebar.markdown("**Quick Select:**")
quick_symbols = ["SPY", "QQQ", "AAPL", "TSLA", "NVDA", "MSFT"]
cols = st.sidebar.columns(2)

for i, sym in enumerate(quick_symbols):
    with cols[i % 2]:
        if st.button(sym, key=f"quick_{sym}"):
            main_symbol = sym

if st.sidebar.button("🔄 Refresh Analysis", type="primary"):
    st.cache_data.clear()
    st.rerun()

# Analysis interval
refresh_interval = st.sidebar.selectbox(
    "Auto-refresh",
    ["Manual", "30 seconds", "1 minute", "5 minutes"],
    index=0
)

# Main analysis
if main_symbol:
    st.markdown(f"## 📊 Complete Analysis: {main_symbol}")
    
    with st.spinner(f"Analyzing {main_symbol}..."):
        analysis = analyzer.analyze_symbol_complete(main_symbol)
    
    if 'error' in analysis:
        st.error(f"❌ {analysis['error']}")
        if 'current_price' in analysis:
            st.info(f"Current price available: ${analysis['current_price']:.2f}")
    else:
        # Success - show complete analysis
        gex_profile = analysis['gex_profile']
        
        # Header metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            price_emoji = "🟢" if analysis['price_change'] > 0 else "🔴" if analysis['price_change'] < 0 else "⚪"
            st.metric(
                f"{price_emoji} Price",
                f"${analysis['current_price']:.2f}",
                f"{analysis['price_change']:+.2f}%"
            )
        
        with col2:
            net_gex = gex_profile['net_gex']
            gex_emoji = "🟢" if net_gex > 0 else "🔴"
            st.metric(f"{gex_emoji} Net GEX", f"{net_gex/1e6:.0f}M")
        
        with col3:
            flip_distance = gex_profile['distance_to_flip']
            flip_emoji = "⚡" if abs(flip_distance) < 1 else "📊"
            st.metric(f"{flip_emoji} Flip Distance", f"{flip_distance:+.2f}%")
        
        with col4:
            st.metric("📈 Volume", f"{gex_profile['total_volume']:,}")
        
        with col5:
            st.metric("🏗️ Open Interest", f"{gex_profile['total_oi']:,}")
        
        # GEX Profile Chart
        st.markdown("### ⚡ Gamma Exposure Profile")
        
        strike_data = gex_profile['strike_data']
        current_price = gex_profile['current_price']
        gamma_flip = gex_profile['gamma_flip']
        
        # Create comprehensive GEX chart
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.7, 0.3],
            subplot_titles=(
                f"{main_symbol} Strike-by-Strike GEX Analysis",
                "Cumulative GEX Profile"
            ),
            vertical_spacing=0.1
        )
        
        # Filter data for reasonable range
        price_range = current_price * 0.2
        display_data = strike_data[
            (strike_data['strike'] >= current_price - price_range) &
            (strike_data['strike'] <= current_price + price_range)
        ].copy()
        
        if len(display_data) > 0:
            # Call GEX (positive bars)
            call_data = display_data[display_data['call_gex'] > 0]
            if len(call_data) > 0:
                fig.add_trace(
                    go.Bar(
                        x=call_data['strike'],
                        y=call_data['call_gex'] / 1e6,
                        name='Call GEX',
                        marker_color='rgba(76, 175, 80, 0.8)',
                        hovertemplate='<b>Call Wall</b><br>Strike: $%{x}<br>GEX: %{y:.1f}M<br>OI: %{customdata:,}<extra></extra>',
                        customdata=call_data['call_oi']
                    ), row=1, col=1
                )
            
            # Put GEX (negative bars)
            put_data = display_data[display_data['put_gex'] < 0]
            if len(put_data) > 0:
                fig.add_trace(
                    go.Bar(
                        x=put_data['strike'],
                        y=put_data['put_gex'] / 1e6,
                        name='Put GEX',
                        marker_color='rgba(244, 67, 54, 0.8)',
                        hovertemplate='<b>Put Wall</b><br>Strike: $%{x}<br>GEX: %{y:.1f}M<br>OI: %{customdata:,}<extra></extra>',
                        customdata=put_data['put_oi']
                    ), row=1, col=1
                )
            
            # Cumulative GEX
            fig.add_trace(
                go.Scatter(
                    x=display_data['strike'],
                    y=display_data['cumulative_gex'] / 1e6,
                    mode='lines',
                    name='Cumulative GEX',
                    line=dict(color='purple', width=3),
                    hovertemplate='Strike: $%{x}<br>Cumulative: %{y:.1f}M<extra></extra>'
                ), row=2, col=1
            )
            
            # Add reference lines
            fig.add_vline(x=current_price, line_dash="solid", line_color="blue",
                         annotation_text=f"Current: ${current_price:.2f}", row=1, col=1)
            fig.add_vline(x=gamma_flip, line_dash="dash", line_color="orange",
                         annotation_text=f"Flip: ${gamma_flip:.2f}", row=1, col=1)
            fig.add_hline(y=0, line_dash="dot", line_color="gray", row=2, col=1)
        
        fig.update_layout(
            height=700,
            showlegend=True,
            title=f"{main_symbol} Comprehensive GEX Analysis"
        )
        
        fig.update_xaxes(title_text="Strike Price ($)", row=2, col=1)
        fig.update_yaxes(title_text="GEX (Millions)", row=1, col=1)
        fig.update_yaxes(title_text="Cumulative GEX (M)", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Walls Analysis
        st.markdown("### 🏗️ Gamma Walls Analysis")
        
        wall_col1, wall_col2 = st.columns(2)
        
        with wall_col1:
            st.markdown("#### 🟢 Call Walls (Resistance)")
            call_walls = gex_profile['call_walls']
            
            if len(call_walls) > 0:
                for i, (_, wall) in enumerate(call_walls.head(5).iterrows(), 1):
                    distance = ((wall['strike'] - current_price) / current_price) * 100
                    strength = wall['call_gex'] / 1e6
                    
                    st.markdown(f"""
                    <div class="wall-info call-wall">
                        <strong>#{i}: ${wall['strike']:.2f}</strong> (+{distance:.1f}%)<br>
                        GEX: {strength:.1f}M | OI: {wall['call_oi']:,}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No significant call walls identified")
        
        with wall_col2:
            st.markdown("#### 🔴 Put Walls (Support)")
            put_walls = gex_profile['put_walls']
            
            if len(put_walls) > 0:
                for i, (_, wall) in enumerate(put_walls.head(5).iterrows(), 1):
                    distance = ((wall['strike'] - current_price) / current_price) * 100
                    strength = abs(wall['put_gex']) / 1e6
                    
                    st.markdown(f"""
                    <div class="wall-info put-wall">
                        <strong>#{i}: ${wall['strike']:.2f}</strong> ({distance:.1f}%)<br>
                        GEX: {strength:.1f}M | OI: {wall['put_oi']:,}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No significant put walls identified")
        
        # Trading Signals
        st.markdown("### 🎯 Trading Signals & Strategies")
        
        signals = analysis['signals']
        
        if signals:
            for signal in signals:
                confidence = signal['confidence']
                
                if confidence >= 75:
                    card_class = "analysis-card signal-high"
                elif confidence >= 60:
                    card_class = "analysis-card signal-medium"
                else:
                    card_class = "analysis-card signal-low"
                
                st.markdown(f"""
                <div class="{card_class}">
                    <h4>{signal['type']} - {confidence}% Confidence</h4>
                    <p><strong>Analysis:</strong> {signal['description']}</p>
                    <p><strong>Strategy:</strong> {signal['strategy']}</p>
                    <p><strong>Target:</strong> {signal['target']}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No high-confidence signals identified for current market conditions")
        
        # Options Chain Summary
        with st.expander("📋 Options Chain Summary", expanded=False):
            chains = analysis['options_chains']
            
            chain_summary = []
            for exp_date, chain_data in chains.items():
                calls = chain_data['calls']
                puts = chain_data['puts']
                dte = chain_data['dte']
                
                total_call_volume = calls['volume'].sum()
                total_put_volume = puts['volume'].sum()
                total_call_oi = calls['openInterest'].sum()
                total_put_oi = puts['openInterest'].sum()
                
                chain_summary.append({
                    'Expiration': exp_date,
                    'DTE': dte,
                    'Call Volume': int(total_call_volume),
                    'Put Volume': int(total_put_volume),
                    'Call OI': int(total_call_oi),
                    'Put OI': int(total_put_oi),
                    'P/C Volume': round(total_put_volume / max(total_call_volume, 1), 2),
                    'P/C OI': round(total_put_oi / max(total_call_oi, 1), 2)
                })
            
            if chain_summary:
                summary_df = pd.DataFrame(chain_summary)
                st.dataframe(summary_df, use_container_width=True)
        
        # Market Context
        st.markdown("### 📊 Market Context & Interpretation")
        
        net_gex_millions = net_gex / 1e6
        
        context_col1, context_col2 = st.columns(2)
        
        with context_col1:
            st.markdown("#### Current Market Regime")
            
            if net_gex > 500e6:
                st.success("🟢 **High Positive GEX** - Strong range-bound environment")
                st.markdown("- Dealers are long gamma")
                st.markdown("- Expect: Lower volatility, mean reversion")
                st.markdown("- Strategies: Iron condors, premium selling")
            elif net_gex > 0:
                st.info("🟡 **Positive GEX** - Mild range-bound bias")
                st.markdown("- Moderate dealer gamma long position")
                st.markdown("- Expect: Some support/resistance at walls")
                st.markdown("- Strategies: Range trading, selective premium selling")
            elif net_gex > -200e6:
                st.warning("🟠 **Low Negative GEX** - Neutral to slightly trending")
                st.markdown("- Dealers slightly short gamma")
                st.markdown("- Expect: Mixed signals, moderate volatility")
                st.markdown("- Strategies: Directional plays with tight stops")
            else:
                st.error("🔴 **High Negative GEX** - Strong trending environment")
                st.markdown("- Dealers are short gamma")
                st.markdown("- Expect: High volatility, momentum moves")
                st.markdown("- Strategies: Long options, gamma squeezes")
        
        with context_col2:
            st.markdown("#### Key Levels")
            
            st.markdown(f"**Current Price:** ${current_price:.2f}")
            st.markdown(f"**Gamma Flip:** ${gamma_flip:.2f}")
            st.markdown(f"**Distance to Flip:** {flip_distance:+.2f}%")
            
            if len(call_walls) > 0:
                nearest_call = call_walls.iloc[0]['strike']
                call_distance = ((nearest_call - current_price) / current_price) * 100
                st.markdown(f"**Next Call Wall:** ${nearest_call:.2f} (+{call_distance:.1f}%)")
            
            if len(put_walls) > 0:
                nearest_put = put_walls.iloc[0]['strike']
                put_distance = ((nearest_put - current_price) / current_price) * 100
                st.markdown(f"**Next Put Wall:** ${nearest_put:.2f} ({put_distance:.1f}%)")

# Auto-refresh logic
if refresh_interval != "Manual":
    time_map = {
        "30 seconds": 30,
        "1 minute": 60,
        "5 minutes": 300
    }
    
    time.sleep(time_map[refresh_interval])
    st.rerun()

# Footer
st.markdown("---")
st.markdown("**Disclaimer:** For educational purposes only. Options trading involves substantial risk.")
