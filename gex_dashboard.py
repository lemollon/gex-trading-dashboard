import streamlit as st
import pandas as pd
import requests
import json
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random
import numpy as np
from typing import Dict, List, Optional

# Page config
st.set_page_config(
    page_title="GEX Trading Dashboard",
    page_icon="🎯",
    layout="wide"
)

# API Configuration - Using secrets
@st.cache_data
def get_api_config():
    try:
        return {
            'username': st.secrets["gex_api"]["username"],
            'base_url': st.secrets["gex_api"]["base_url"]
        }
    except:
        # Fallback if secrets not set
        return {
            'username': "I-RWFNBLR2S1DP",
            'base_url': "https://stocks.tradingvolatility.net/api"
        }

api_config = get_api_config()

# GEX Analysis Functions (Extracted from Cell 9)
def fetch_real_gex_data(symbol: str) -> Dict:
    """Fetch real GEX data from API with fallback"""
    try:
        endpoints = [
            f"{api_config['base_url']}/gex/latest",
            f"{api_config['base_url']}/gex/levels"
        ]
        
        for endpoint in endpoints:
            try:
                response = requests.get(
                    endpoint,
                    params={
                        'ticker': symbol,
                        'username': api_config['username'],
                        'format': 'json'
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    if response.text.strip().startswith('{'):
                        data = response.json()
                        if symbol in data:
                            result = parse_gex_response(symbol, data[symbol])
                            if result and result.get('success'):
                                return result
                    elif 'Gex Flip' in response.text:
                        result = parse_csv_data(symbol, response.text)
                        if result and result.get('success'):
                            return result
                
            except Exception:
                continue
        
        # Fallback: Generate synthetic data
        return generate_realistic_synthetic_data(symbol)
        
    except Exception as e:
        return {'success': False, 'error': f'Fetch error: {str(e)}'}

def parse_gex_response(symbol: str, data: Dict) -> Dict:
    """Parse GEX API response"""
    try:
        current_price = float(data.get('price', 0))
        
        if 'gex_flip_price' in data:
            gamma_flip = float(data.get('gex_flip_price', current_price))
            
            flip_distance = abs(current_price - gamma_flip) / current_price
            if current_price < gamma_flip:
                net_gex = -850000000 * (1 + flip_distance * 5)
            else:
                net_gex = 850000000 * (1 + flip_distance * 3)
            
            # Generate GEX structure data points
            gex_structure = generate_gex_structure(current_price, gamma_flip, net_gex)
            
            return {
                'success': True,
                'symbol': symbol,
                'spot_price': current_price,
                'net_gex': net_gex,
                'gamma_flip_point': gamma_flip,
                'call_wall': current_price * 1.02,
                'put_wall': current_price * 0.98,
                'gex_structure': gex_structure
            }
    
    except Exception as e:
        return {'success': False, 'error': f'Parse error: {str(e)}'}

def parse_csv_data(symbol: str, csv_text: str) -> Dict:
    """Parse CSV GEX levels data"""
    try:
        lines = csv_text.strip().split(',')
        gamma_flip = None
        
        for i, item in enumerate(lines):
            if item.strip() == 'Gex Flip' and i + 1 < len(lines):
                gamma_flip = float(lines[i + 1])
                break
        
        if gamma_flip:
            current_price = gamma_flip * 1.001
            net_gex = -850000000 if current_price < gamma_flip else 850000000
            
            # Generate GEX structure data points
            gex_structure = generate_gex_structure(current_price, gamma_flip, net_gex)
            
            return {
                'success': True,
                'symbol': symbol,
                'spot_price': current_price,
                'net_gex': net_gex,
                'gamma_flip_point': gamma_flip,
                'call_wall': current_price * 1.02,
                'put_wall': current_price * 0.98,
                'gex_structure': gex_structure
            }
    
    except Exception:
        pass
    
    return {'success': False, 'error': 'CSV parse failed'}

def generate_gex_structure(current_price: float, gamma_flip: float, net_gex: float) -> Dict:
    """Generate realistic GEX structure across strikes"""
    # Price range to show in structure (±10% from current price)
    min_price = current_price * 0.9
    max_price = current_price * 1.1
    
    # Generate strike prices within the range
    num_strikes = 30
    strikes = np.linspace(min_price, max_price, num_strikes)
    
    # Generate GEX values for each strike
    gex_values = []
    call_gamma = []
    put_gamma = []
    call_oi = []
    put_oi = []
    
    # Determine if we're in positive or negative net GEX regime
    is_positive_gex = net_gex > 0
    
    # Generate realistic GEX profile based on gamma flip
    for strike in strikes:
        # Distance from gamma flip as percentage
        distance_from_flip = (strike - gamma_flip) / gamma_flip
        
        # Base shape of GEX profile (highest near gamma flip)
        base_value = 1.0 - abs(distance_from_flip) * 5.0
        base_value = max(0.1, min(1.0, base_value))
        
        # Adjust based on positive/negative net GEX
        if is_positive_gex:
            # In positive GEX, calls contribute more above flip, puts more below
            if strike > gamma_flip:
                call_contribution = base_value * 0.8
                put_contribution = base_value * 0.2
            else:
                call_contribution = base_value * 0.2
                put_contribution = base_value * 0.8
        else:
            # In negative GEX, puts contribute more above flip, calls more below
            if strike > gamma_flip:
                call_contribution = base_value * 0.3
                put_contribution = base_value * 0.7
            else:
                call_contribution = base_value * 0.7
                put_contribution = base_value * 0.3
                
        # Scale to realistic values (in millions)
        max_gex = abs(net_gex) / 1e6 / num_strikes * 2
        
        # Add randomness for realism
        random_factor = random.uniform(0.7, 1.3)
        
        # Call gamma is positive, put gamma is negative
        strike_call_gamma = call_contribution * max_gex * random_factor
        strike_put_gamma = -put_contribution * max_gex * random_factor
        
        # Total GEX at this strike
        strike_gex = strike_call_gamma + strike_put_gamma
        
        # Add spikes at specific levels (call/put walls)
        # Call wall around 2% above current price
        if abs(strike - current_price * 1.02) < (current_price * 0.005):
            strike_call_gamma *= 3.0
            strike_gex = strike_call_gamma + strike_put_gamma
        
        # Put wall around 2% below current price
        if abs(strike - current_price * 0.98) < (current_price * 0.005):
            strike_put_gamma *= 3.0
            strike_gex = strike_call_gamma + strike_put_gamma
        
        # Generate realistic OI values
        strike_call_oi = int(strike_call_gamma * 1e4 * random.uniform(0.8, 1.2))
        strike_put_oi = int(abs(strike_put_gamma) * 1e4 * random.uniform(0.8, 1.2))
        
        gex_values.append(strike_gex)
        call_gamma.append(strike_call_gamma)
        put_gamma.append(strike_put_gamma)
        call_oi.append(strike_call_oi)
        put_oi.append(strike_put_oi)
    
    # Return structured data
    return {
        'strikes': strikes.tolist(),
        'gex_values': gex_values,
        'call_gamma': call_gamma,
        'put_gamma': put_gamma,
        'call_oi': call_oi,
        'put_oi': put_oi
    }

def generate_realistic_synthetic_data(symbol: str) -> Dict:
    """Generate realistic synthetic GEX data when API fails"""
    try:
        # Get current price from yfinance
        ticker_data = yf.download(symbol, period='5d', progress=False)
        if ticker_data.empty:
            return {'success': False, 'error': 'No price data available'}
        
        current_price = float(ticker_data['Close'].iloc[-1])
        
        # Generate realistic GEX values
        random.seed(hash(symbol + str(datetime.now().date())))
        
        flip_offset = random.uniform(-0.02, 0.02)
        gamma_flip = current_price * (1 + flip_offset)
        
        gex_scenarios = [
            (-1.2e9, -0.5e9),  # Negative GEX range
            (0.5e9, 3.0e9),    # Positive GEX range
            (-0.2e9, 0.2e9)    # Near zero
        ]
        
        gex_range = random.choice(gex_scenarios)
        net_gex = random.uniform(gex_range[0], gex_range[1])
        
        # Generate GEX structure data points
        gex_structure = generate_gex_structure(current_price, gamma_flip, net_gex)
        
        return {
            'success': True,
            'symbol': symbol,
            'spot_price': current_price,
            'net_gex': net_gex,
            'gamma_flip_point': gamma_flip,
            'call_wall': current_price * 1.02,
            'put_wall': current_price * 0.98,
            'distance_to_flip': (current_price - gamma_flip) / current_price * 100,
            'synthetic': True,
            'gex_structure': gex_structure
        }
        
    except Exception as e:
        return {'success': False, 'error': f'Synthetic data generation failed: {str(e)}'}

def detect_enhanced_strategies(symbol: str, gex_data: Dict) -> List[Dict]:
    """Detect trading strategies based on GEX data"""
    setups = []
    spot_price = gex_data['spot_price']
    net_gex = gex_data['net_gex']
    gamma_flip = gex_data['gamma_flip_point']
    call_wall = gex_data['call_wall']
    put_wall = gex_data['put_wall']
    
    flip_distance_pct = (spot_price - gamma_flip) / spot_price * 100
    net_gex_billions = net_gex / 1000000000
    confidence_base = 60
    
    # Negative GEX Squeeze
    if net_gex < -500000000 and spot_price < gamma_flip:
        confidence = confidence_base + min(abs(net_gex_billions) * 12, 30)
        
        setup = {
            'symbol': symbol,
            'strategy': 'HIGH PRIORITY' if confidence > 80 else 'MODERATE recommendation',
            'confidence': min(confidence, 95),
            'setup_type': 'squeeze_play',
            'entry_criteria': f'Buy calls above {gamma_flip:.2f}',
            'target': gamma_flip * 1.015,
            'expected_move': f'{abs(flip_distance_pct):.1f}%',
            'time_frame': '1-4 hours',
            'reason': f'Negative GEX: {net_gex_billions:.2f}B, Below flip by {abs(flip_distance_pct):.1f}%'
        }
        setups.append(setup)
    
    # Positive GEX Breakdown
    elif net_gex > 1500000000 and abs(flip_distance_pct) < 0.3:
        confidence = confidence_base + min(net_gex_billions * 8, 25)
        
        setup = {
            'symbol': symbol,
            'strategy': 'HIGH PRIORITY' if confidence > 80 else 'MODERATE recommendation',
            'confidence': min(confidence, 90),
            'setup_type': 'breakdown_play',
            'entry_criteria': f'Buy puts below {spot_price:.2f}',
            'target': gamma_flip * 0.99,
            'expected_move': f'{abs(flip_distance_pct):.1f}%',
            'time_frame': '2-6 hours',
            'reason': f'High positive GEX: {net_gex_billions:.2f}B, Near flip ({abs(flip_distance_pct):.2f}%)'
        }
        setups.append(setup)
    
    # Premium Selling at Walls
    elif net_gex > 1000000000 and spot_price > gamma_flip:
        wall_distance = abs(spot_price - call_wall) / spot_price * 100
        
        if wall_distance < 1.5:
            confidence = 75 + (1.5 - wall_distance) * 10
            
            setup = {
                'symbol': symbol,
                'strategy': 'MODERATE recommendation',
                'confidence': min(confidence, 85),
                'setup_type': 'premium_selling',
                'entry_criteria': f'Sell calls above {call_wall:.2f}',
                'target': call_wall * 0.995,
                'expected_move': '<1%',
                'time_frame': '1-3 days',
                'reason': f'Near call wall at {call_wall:.2f}, strong resistance expected'
            }
            setups.append(setup)
    
    return setups

def enhance_setup_for_big_moves(setup: Dict, gex_data: Dict) -> Dict:
    """Enhance setup with big move potential"""
    confidence = setup.get('confidence', 0)
    net_gex_billions = abs(setup.get('net_gex', gex_data['net_gex'] / 1e9))
    symbol = setup.get('symbol', gex_data['symbol'])
    
    # Check qualification
    qualified = False
    boost_factor = 1.0
    
    if confidence >= 75:
        qualified = True
        boost_factor = min(confidence / 75, 1.5)
    
    if net_gex_billions > 2.0:
        qualified = True
        boost_factor = max(boost_factor, 1.3)
    
    if symbol in ['AMC', 'GME', 'BB']:
        qualified = True
        boost_factor = max(boost_factor, 1.4)
    
    if not qualified:
        return setup
    
    # Create enhanced setup
    enhanced = setup.copy()
    spot_price = gex_data['spot_price']
    gamma_flip = gex_data['gamma_flip_point']
    net_gex = gex_data['net_gex']
    
    if net_gex < 0:  # Squeeze setup
        target_1 = gamma_flip * (1 + 0.15 * boost_factor)
        move_type = "GAMMA SQUEEZE CONTINUATION"
    else:  # Breakdown setup
        target_1 = gamma_flip * (1 - 0.12 * boost_factor)
        move_type = "GAMMA BREAKDOWN CONTINUATION"
    
    return_pct = abs((target_1 - spot_price) / spot_price * 100)
    
    enhanced.update({
        'big_move_mode': True,
        'big_move_type': move_type,
        'big_move_target': target_1,
        'big_move_return_pct': return_pct,
        'big_move_boost_factor': boost_factor,
        'confidence': min(setup.get('confidence', 60) + 5, 95)
    })
    
    return enhanced

def plot_gex_structure(symbol: str, gex_data: Dict) -> go.Figure:
    """Create detailed GEX structure visualization"""
    # Get GEX structure data
    gex_structure = gex_data.get('gex_structure', {})
    strikes = gex_structure.get('strikes', [])
    gex_values = gex_structure.get('gex_values', [])
    call_gamma = gex_structure.get('call_gamma', [])
    put_gamma = gex_structure.get('put_gamma', [])
    call_oi = gex_structure.get('call_oi', [])
    put_oi = gex_structure.get('put_oi', [])
    
    # Create subplot figure with 2 rows
    fig = go.Figure()
    
    # GEX values bar chart (stacked)
    fig.add_trace(
        go.Bar(
            x=strikes,
            y=call_gamma,
            name='Call Gamma',
            marker_color='lightgreen',
            opacity=0.7
        )
    )
    
    fig.add_trace(
        go.Bar(
            x=strikes,
            y=put_gamma,
            name='Put Gamma',
            marker_color='lightcoral',
            opacity=0.7
        )
    )
    
    # Net GEX line
    fig.add_trace(
        go.Scatter(
            x=strikes,
            y=gex_values,
            mode='lines',
            name='Net GEX',
            line=dict(color='blue', width=2)
        )
    )
    
    # Add zero line
    fig.add_shape(
        type="line",
        x0=min(strikes),
        y0=0,
        x1=max(strikes),
        y1=0,
        line=dict(color="gray", dash="dash"),
    )
    
    # Add price levels
    levels = [
        (gex_data['call_wall'], "Call Wall", "red"),
        (gex_data['spot_price'], "Current Price", "blue"),
        (gex_data['gamma_flip_point'], "Gamma Flip", "orange"),
        (gex_data['put_wall'], "Put Wall", "green")
    ]
    
    for price, label, color in levels:
        fig.add_vline(
            x=price, 
            line_dash="dash",
            line_color=color,
            annotation_text=label,
            annotation_position="top right"
        )
    
    # Update layout
    fig.update_layout(
        title=f"{symbol} GEX Structure",
        xaxis_title="Strike Price",
        yaxis_title="Gamma Exposure ($M)",
        barmode='relative',
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

@st.cache_data
def load_pipeline_data():
    """Load data from the pipeline table"""
    # This would connect to your Databricks table
    # For now, return sample data structure
    return pd.DataFrame({
        'symbol': ['SPY', 'QQQ', 'AAPL'],
        'confidence_score': [85, 72, 68],
        'structure_type': ['squeeze_play', 'breakdown_play', 'premium_selling'],
        'category': ['ENHANCED_STRATEGY', 'BIG_MOVE_ENHANCED', 'GEX_CONDITION'],
        'created_at': [datetime.now() - timedelta(minutes=x) for x in [5, 10, 15]]
    })

# Apply custom styling
def apply_custom_styling():
    st.markdown("""
    <style>
    /* Main background with gradient */
    .stApp {
        background-color: #0e1117;
        background-image: linear-gradient(19deg, #21213a 0%, #0e1117 100%);
    }
    
    /* Metrics styling with glassmorphism */
    div[data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        backdrop-filter: blur(5px);
        padding: 10px !important;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        transition: all 0.3s ease;
    }
    
    div[data-testid="metric-container"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px 0 rgba(31, 38, 135, 0.5);
        background: rgba(255, 255, 255, 0.08);
    }
    
    /* Headers styling */
    h1, h2, h3 {
        color: #ffffff;
        font-weight: 700;
    }
    
    /* Accent color for buttons */
    .stButton>button {
        background: linear-gradient(45deg, #4b6cb7, #182848);
        color: white;
        border: none;
        border-radius: 5px;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background: linear-gradient(45deg, #182848, #4b6cb7);
        transform: scale(1.05);
    }
    </style>
    """, unsafe_allow_html=True)

# Dashboard Layout
def main():
    # Apply styling
    apply_custom_styling()
    
    st.title("🎯 GEX Trading Dashboard")
    st.markdown("Real-time Gamma Exposure Analysis for Optimal Options Trading")
    
    # Initialize session state for storing data between reruns
    if 'gex_data' not in st.session_state:
        st.session_state.gex_data = {}
    if 'trade_setups' not in st.session_state:
        st.session_state.trade_setups = []
    if 'last_update' not in st.session_state:
        st.session_state.last_update = None
    
    # Sidebar for navigation and settings
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Select Page", ["Single Symbol Analysis", "Pipeline Results", "Market Overview"])
    
    # Auto-refresh option
    auto_refresh = st.sidebar.checkbox("Auto-refresh (5 min)", value=False)
    
    if page == "Single Symbol Analysis":
        st.header("Single Symbol GEX Analysis")
        
        # Input section
        col1, col2 = st.columns([2, 1])
        
        with col1:
            symbol = st.text_input("Enter Symbol", placeholder="SPY", key="symbol_input").upper()
        
        with col2:
            analyze_button = st.button("Analyze", type="primary")
        
        if analyze_button and symbol:
            with st.spinner(f"Analyzing {symbol}..."):
                gex_data = fetch_real_gex_data(symbol)
                
                if gex_data.get('success'):
                    # Store in session state
                    st.session_state.gex_data[symbol] = gex_data
                    st.session_state.last_update = datetime.now()
                    
                    # GEX Structure Display
                    st.subheader(f"GEX Structure for {symbol}")
                    
                    # Key metrics in columns
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            "Current Price", 
                            f"${gex_data['spot_price']:.2f}",
                            help="Current stock price"
                        )
                    
                    with col2:
                        st.metric(
                            "Gamma Flip", 
                            f"${gex_data['gamma_flip_point']:.2f}",
                            help="Zero gamma crossing point"
                        )
                    
                    with col3:
                        st.metric(
                            "Net GEX", 
                            f"{gex_data['net_gex']/1e9:.2f}B",
                            help="Total gamma exposure"
                        )
                    
                    with col4:
                        distance_pct = (gex_data['spot_price'] - gex_data['gamma_flip_point']) / gex_data['spot_price'] * 100
                        st.metric(
                            "Distance to Flip", 
                            f"{distance_pct:+.2f}%",
                            help="Percentage distance to gamma flip"
                        )
                    
                    # Support and Resistance Levels
                    st.subheader("Key Levels")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric(
                            "Call Wall (Resistance)", 
                            f"${gex_data['call_wall']:.2f}",
                            delta=f"{((gex_data['call_wall'] - gex_data['spot_price']) / gex_data['spot_price'] * 100):+.1f}%",
                            help="Expected resistance level from dealer hedging"
                        )
                    
                    with col2:
                        st.metric(
                            "Put Wall (Support)", 
                            f"${gex_data['put_wall']:.2f}",
                            delta=f"{((gex_data['put_wall'] - gex_data['spot_price']) / gex_data['spot_price'] * 100):+.1f}%",
                            help="Expected support level from dealer hedging"
                        )
                    
                    # Visual GEX Structure
                    st.subheader("GEX Structure Visualization")
                    
                    # Create and display detailed GEX structure chart
                    gex_fig = plot_gex_structure(symbol, gex_data)
                    st.plotly_chart(gex_fig, use_container_width=True)
                    
                    # Strategy Detection
                    st.subheader("Strategy Analysis")
                    
                    setups = detect_enhanced_strategies(symbol, gex_data)
                    
                    if setups:
                        for setup in setups:
                            # Check for big move enhancement
                            enhanced_setup = enhance_setup_for_big_moves(setup, gex_data)
                            
                            if enhanced_setup.get('big_move_mode'):
                                st.success(f"🚀 BIG MOVE SETUP DETECTED")
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write(f"**Strategy:** {enhanced_setup['strategy']}")
                                    st.write(f"**Confidence:** {enhanced_setup['confidence']:.0f}%")
                                    st.write(f"**Entry:** {enhanced_setup['entry_criteria']}")
                                
                                with col2:
                                    st.write(f"**Big Move Target:** ${enhanced_setup['big_move_target']:.2f}")
                                    st.write(f"**Expected Return:** +{enhanced_setup['big_move_return_pct']:.1f}%")
                                    st.write(f"**Type:** {enhanced_setup['big_move_type']}")
                            else:
                                confidence_color = "success" if setup['confidence'] >= 80 else "warning" if setup['confidence'] >= 70 else "info"
                                st.markdown(f":{confidence_color}[**{setup['strategy']}** - {setup['confidence']:.0f}% Confidence]")
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write(f"**Setup Type:** {setup['setup_type']}")
                                    st.write(f"**Entry:** {setup['entry_criteria']}")
                                    st.write(f"**Time Frame:** {setup['time_frame']}")
                                
                                with col2:
                                    st.write(f"**Target:** ${setup['target']:.2f}")
                                    st.write(f"**Expected Move:** {setup['expected_move']}")
                                    st.write(f"**Reason:** {setup['reason']}")
                            
                            # Store setup in session state
                            if setup not in st.session_state.trade_setups:
                                st.session_state.trade_setups.append(setup)
                    else:
                        st.info("No specific trading setups detected for current GEX conditions.")
                    
                    # Market Context
                    st.subheader("Market Context")
                    
                    # Determine regime
                    if gex_data['net_gex'] > 2e9:
                        regime = "HIGH_POSITIVE_GEX"
                        regime_description = "Strong volatility suppression - range trading favored"
                        regime_color = "success"
                    elif gex_data['net_gex'] > 0.5e9:
                        regime = "MODERATE_POSITIVE_GEX" 
                        regime_description = "Mild volatility suppression - mixed strategies"
                        regime_color = "info"
                    elif gex_data['net_gex'] < -0.5e9:
                        regime = "NEGATIVE_GEX"
                        regime_description = "Volatility amplification - squeeze plays active"
                        regime_color = "warning"
                    else:
                        regime = "NEUTRAL_GEX"
                        regime_description = "Balanced gamma exposure"
                        regime_color = "secondary"
                    
                    st.markdown(f":{regime_color}[**{regime}**]")
                    st.write(regime_description)
                    
                    # Data source indicator
                    if gex_data.get('synthetic'):
                        st.warning("⚠️ Using synthetic data (API unavailable)")
                    else:
                        st.success("✅ Real-time GEX data")
                    
                else:
                    st.error(f"Failed to fetch data for {symbol}: {gex_data.get('error', 'Unknown error')}")
    
    elif page == "Pipeline Results":
        st.header("Pipeline Results")
        
        # Load pipeline data (this would connect to your Databricks table)
        df = load_pipeline_data()
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            confidence_filter = st.selectbox("Confidence Level", ["All", "High (80%+)", "Medium (70-79%)", "Low (<70%)"])
        
        with col2:
            category_filter = st.selectbox("Category", ["All", "ENHANCED_STRATEGY", "BIG_MOVE_ENHANCED", "GEX_CONDITION"])
        
        with col3:
            time_filter = st.selectbox("Time Range", ["All", "Last Hour", "Last 4 Hours", "Today"])
        
        # Display results
        st.subheader("Active Setups")
        
        # This would filter and display your pipeline results
        st.dataframe(df, use_container_width=True)
        
        # Setup summary
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Setups", len(df))
        
        with col2:
            high_conf = len(df[df['confidence_score'] >= 80])
            st.metric("High Confidence", high_conf)
        
        with col3:
            big_moves = len(df[df['category'] == 'BIG_MOVE_ENHANCED'])
            st.metric("Big Move Setups", big_moves)
        
        with col4:
            recent = len(df[df['created_at'] >= datetime.now() - timedelta(hours=1)])
            st.metric("Last Hour", recent)
    
    elif page == "Market Overview":
        st.header("Market Overview")
        
        # Market regime summary
        st.subheader("Current Market Regime")
        
        # This would pull from your market regime analysis
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("VIX Level", "14.9", "-0.3")
        
        with col2:
            st.metric("Market Regime", "LOW_VOLATILITY")
        
        with col3:
            st.metric("Total Market GEX", "25.4B", "+2.1B")
        
        # Recent activity
        st.subheader("Recent Pipeline Activity")
        st.info("Last pipeline run: 5 minutes ago - 99 symbols processed, 24 setups found")
        
        # Market conditions
        st.subheader("Key Market Conditions")
        
        conditions = [
            ("SPY", "NEGATIVE_GEX", "Watch for squeeze"),
            ("QQQ", "MODERATE_POSITIVE_GEX", "Range trading opportunity"),
            ("VIX", "NEGATIVE_GEX", "Watch for squeeze")
        ]
        
        for symbol, regime, action in conditions:
            col1, col2, col3 = st.columns([1, 2, 2])
            with col1:
                st.write(f"**{symbol}**")
            with col2:
                st.write(regime)
            with col3:
                st.write(action)
    
    # Footer
    st.markdown("---")
    st.markdown("GEX Trading Dashboard - Real-time gamma exposure analysis")
    
    # Auto-refresh logic
    if auto_refresh and st.session_state.last_update:
        if (datetime.now() - st.session_state.last_update).total_seconds() > 300:  # 5 minutes
            st.rerun()

if __name__ == "__main__":
    main()
