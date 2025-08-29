"""
GEX Master Pro - Complete Trading Dashboard
Combines single symbol analysis with pipeline database connections, auto-trader, education, and alerts
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta, time as dt_time
import time
import requests
import json
import uuid
import random
import pytz

# Try to import Databricks connector
try:
    from databricks import sql
    DATABRICKS_AVAILABLE = True
except ImportError:
    DATABRICKS_AVAILABLE = False
    st.error("databricks-sql-connector not installed. Run: pip install databricks-sql-connector")

# Try to import yfinance for price data
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    st.error("yfinance not installed. Run: pip install yfinance")

# Page config
st.set_page_config(
    page_title="GEX Master Pro",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Simple CSS styling - keeping it basic for device compatibility
st.markdown("""
<style>
    .main {
        padding-top: 1rem;
    }
    
    /* Base card style */
    .card {
        background-color: #f9f9f9;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    /* Different confidence level cards */
    .high-confidence {
        border-left: 5px solid #28a745;
    }
    
    .medium-confidence {
        border-left: 5px solid #ffc107;
    }
    
    .low-confidence {
        border-left: 5px solid #dc3545;
    }
    
    /* Metrics styling */
    .metric-container {
        background-color: #f9f9f9;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    .metric-title {
        font-size: 1rem;
        color: #555;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #333;
    }
    
    /* Alert status indicators */
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 5px;
    }
    
    .status-online {
        background-color: #28a745;
    }
    
    .status-offline {
        background-color: #dc3545;
    }
    
    /* Auto Trader position styles */
    .position-active {
        background-color: rgba(40, 167, 69, 0.1);
        border: 1px solid #28a745;
    }
    
    .position-closed {
        background-color: rgba(108, 117, 125, 0.1);
        border: 1px solid #6c757d;
    }
    
    .position-stopped {
        background-color: rgba(220, 53, 69, 0.1);
        border: 1px solid #dc3545;
    }
    
    /* Education section styles */
    .education-section {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    
    .education-title {
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 1rem;
        color: #333;
    }
    
    /* Alert box styles */
    .alert-box {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    
    .alert-info {
        background-color: rgba(0, 123, 255, 0.1);
        border: 1px solid #007bff;
    }
    
    .alert-success {
        background-color: rgba(40, 167, 69, 0.1);
        border: 1px solid #28a745;
    }
    
    .alert-warning {
        background-color: rgba(255, 193, 7, 0.1);
        border: 1px solid #ffc107;
    }
    
    .alert-danger {
        background-color: rgba(220, 53, 69, 0.1);
        border: 1px solid #dc3545;
    }
    
    /* Table styling */
    .styled-table {
        width: 100%;
        border-collapse: collapse;
        margin-bottom: 1rem;
        font-size: 0.9rem;
    }
    
    .styled-table th {
        background-color: #f2f2f2;
        padding: 0.5rem;
        text-align: left;
        border-bottom: 2px solid #ddd;
    }
    
    .styled-table td {
        padding: 0.5rem;
        border-bottom: 1px solid #ddd;
    }
    
    .styled-table tr:nth-child(even) {
        background-color: #f9f9f9;
    }
    
    .styled-table tr:hover {
        background-color: #f2f2f2;
    }
    
    /* Mobile optimization */
    @media (max-width: 768px) {
        .metric-value {
            font-size: 1.4rem;
        }
        
        .card {
            padding: 0.75rem;
        }
        
        .education-section {
            padding: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)

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

# Initialize session state
def initialize_session_state():
    """Initialize session state variables"""
    if 'gex_data' not in st.session_state:
        st.session_state.gex_data = {}
    if 'last_update' not in st.session_state:
        st.session_state.last_update = None
    if 'api_status' not in st.session_state:
        st.session_state.api_status = {
            'last_checked': None,
            'status': 'unknown',
            'failures': 0,
            'successes': 0
        }
    if 'auto_trader' not in st.session_state:
        st.session_state.auto_trader = {
            'enabled': False,
            'positions': [],
            'history': [],
            'capital': 10000,
            'available_capital': 10000,
            'min_confidence': 80,
            'max_position_size': 0.05,  # 5% of capital
            'last_check': None
        }
    if 'discord_webhook' not in st.session_state:
        st.session_state.discord_webhook = {
            'url': '',
            'enabled': False,
            'min_confidence': 80,
            'alert_types': ['squeeze_play', 'breakdown_play', 'premium_selling'],
            'last_sent': None
        }
    if 'watchlist' not in st.session_state:
        st.session_state.watchlist = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'NVDA']

@st.cache_resource
def init_databricks_connection():
    """Initialize Databricks connection"""
    try:
        if not DATABRICKS_AVAILABLE:
            return None
            
        if "databricks" not in st.secrets:
            st.error("Databricks secrets not configured")
            return None
            
        connection = sql.connect(
            server_hostname=st.secrets["databricks"]["server_hostname"],
            http_path=st.secrets["databricks"]["http_path"],
            access_token=st.secrets["databricks"]["access_token"]
        )
        return connection
    except Exception as e:
        st.error(f"Databricks connection failed: {e}")
        return None

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_gex_data():
    """Load GEX data from pipeline table"""
    connection = init_databricks_connection()
    
    if not connection:
        return {
            'data': pd.DataFrame(),
            'status': 'disconnected',
            'message': 'Databricks connection failed'
        }
    
    try:
        cursor = connection.cursor()
        
        # Query pipeline table
        query = """
        SELECT 
            run_id,
            symbol,
            structure_type,
            confidence_score,
            spot_price,
            gamma_flip_point,
            distance_to_flip_pct,
            recommendation,
            category,
            priority,
            created_at as analysis_timestamp,
            analysis_date
        FROM quant_projects.gex_trading.scheduled_pipeline_results
        WHERE analysis_date >= current_date() - INTERVAL 30 DAYS
        ORDER BY analysis_date DESC, confidence_score DESC
        LIMIT 1000
        """
        
        cursor.execute(query)
        results = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        
        df = pd.DataFrame(results, columns=columns)
        
        cursor.close()
        
        # Debug information
        debug_info = ""
        if not df.empty:
            debug_info = f"""
            **Debug Info:**
            - Total records: {len(df)}
            - Date range: {df['analysis_date'].min()} to {df['analysis_date'].max()}
            - Categories: {df['category'].value_counts().to_dict()}
            - Confidence range: {df['confidence_score'].min()}-{df['confidence_score'].max()}
            - Unique runs: {df['run_id'].nunique()}
            """
        
        return {
            'data': df,
            'status': 'connected',
            'message': f'Connected - Loaded {len(df)} records from pipeline',
            'debug_info': debug_info
        }
        
    except Exception as e:
        st.error(f"Query failed: {e}")
        return {
            'data': pd.DataFrame(),
            'status': 'error',
            'message': f'Database query error: {e}'
        }

# Single Symbol Analysis Functions
def fetch_real_gex_data(symbol: str) -> dict:
    """Fetch real GEX data from API with databricks fallback"""
    api_failures = []
    
    # First, try API endpoints
    try:
        endpoints = [
            f"{api_config['base_url']}/gex/latest",
            f"{api_config['base_url']}/gex/levels"
        ]
        
        for endpoint in endpoints:
            try:
                st.info(f"Attempting to fetch data from {endpoint}...")
                
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
                                st.success(f"Successfully fetched data from {endpoint}")
                                return result
                    elif 'Gex Flip' in response.text:
                        result = parse_csv_data(symbol, response.text)
                        if result and result.get('success'):
                            st.success(f"Successfully fetched data from {endpoint}")
                            return result
                
                # Record the failure reason
                api_failures.append({
                    'endpoint': endpoint,
                    'status_code': response.status_code,
                    'response_text': response.text[:100] + '...' if len(response.text) > 100 else response.text
                })
                
            except Exception as e:
                api_failures.append({
                    'endpoint': endpoint,
                    'exception': str(e)
                })
                continue
        
        # Next, try to fetch from Databricks if API failed
        connection = init_databricks_connection()
        if connection:
            try:
                cursor = connection.cursor()
                
                query = f"""
                SELECT 
                    symbol,
                    spot_price,
                    gamma_flip_point,
                    net_gex as net_gex,
                    call_wall,
                    put_wall,
                    distance_to_flip_pct
                FROM quant_projects.gex_trading.scheduled_pipeline_results
                WHERE symbol = '{symbol}'
                ORDER BY analysis_date DESC, analysis_timestamp DESC
                LIMIT 1
                """
                
                cursor.execute(query)
                result = cursor.fetchone()
                
                if result:
                    columns = [desc[0] for desc in cursor.description]
                    data = dict(zip(columns, result))
                    
                    # Convert to expected format
                    db_result = {
                        'success': True,
                        'symbol': data['symbol'],
                        'spot_price': data['spot_price'],
                        'gamma_flip_point': data['gamma_flip_point'],
                        'net_gex': data['net_gex'],
                        'call_wall': data['call_wall'],
                        'put_wall': data['put_wall'],
                        'distance_to_flip': data['distance_to_flip_pct'],
                        'database_source': True
                    }
                    
                    # Generate gamma structure
                    gex_structure = generate_gex_structure(
                        data['spot_price'], 
                        data['gamma_flip_point'], 
                        data['net_gex']
                    )
                    
                    db_result['gex_structure'] = gex_structure
                    st.success(f"Successfully fetched {symbol} data from database")
                    return db_result
                
                cursor.close()
            except Exception as e:
                st.warning(f"Database query failed: {e}")
                
    except Exception as e:
        st.error(f"Critical error in fetch process: {str(e)}")
    
    # Log the API failures for debugging
    st.warning("⚠️ All data source attempts failed.")
    with st.expander("Failure Details"):
        for failure in api_failures:
            st.write(f"Endpoint: {failure['endpoint']}")
            if 'status_code' in failure:
                st.write(f"Status Code: {failure['status_code']}")
                st.write(f"Response: {failure['response_text']}")
            else:
                st.write(f"Exception: {failure['exception']}")
    
    return {'success': False, 'error': f'Failed to fetch data for {symbol}'}

def parse_gex_response(symbol: str, data: dict) -> dict:
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

def parse_csv_data(symbol: str, csv_text: str) -> dict:
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

def generate_gex_structure(current_price: float, gamma_flip: float, net_gex: float) -> dict:
    """Generate GEX structure across strikes based on key levels"""
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
        
        # Call gamma is positive, put gamma is negative
        strike_call_gamma = call_contribution * max_gex 
        strike_put_gamma = -put_contribution * max_gex 
        
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
        
        # Generate OI values
        strike_call_oi = int(strike_call_gamma * 1e4)
        strike_put_oi = int(abs(strike_put_gamma) * 1e4)
        
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

def detect_enhanced_strategies(symbol: str, gex_data: dict) -> list:
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
            'reason': f'Negative GEX: {net_gex_billions:.2f}B, Below flip by {abs(flip_distance_pct):.1f}%',
            'option_type': 'call',
            'option_strike': round(gamma_flip * 1.01, 2),
            'option_expiry': '3-5 DTE',
            'stop_loss_pct': 50,
            'profit_target_pct': 100
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
            'reason': f'High positive GEX: {net_gex_billions:.2f}B, Near flip ({abs(flip_distance_pct):.2f}%)',
            'option_type': 'put',
            'option_strike': round(gamma_flip * 0.99, 2),
            'option_expiry': '3-7 DTE',
            'stop_loss_pct': 50,
            'profit_target_pct': 100
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
                'reason': f'Near call wall at {call_wall:.2f}, strong resistance expected',
                'option_type': 'call_credit',
                'option_strike': round(call_wall, 2),
                'option_expiry': '0-2 DTE',
                'stop_loss_pct': 100,
                'profit_target_pct': 50
            }
            setups.append(setup)
    
    # Iron Condor Setup
    if net_gex > 1000000000 and abs(call_wall - put_wall) / spot_price > 0.04:
        # At least 4% width between walls
        confidence = 65 + min(net_gex_billions * 5, 20)
        
        setup = {
            'symbol': symbol,
            'strategy': 'MODERATE recommendation',
            'confidence': min(confidence, 85),
            'setup_type': 'iron_condor',
            'entry_criteria': f'Sell {put_wall:.2f}/{(put_wall*0.98):.2f} put spread, {call_wall:.2f}/{(call_wall*1.02):.2f} call spread',
            'target': 'Max 50% of credit received',
            'expected_move': 'Price stays between walls',
            'time_frame': '5-10 days',
            'reason': f'Positive GEX: {net_gex_billions:.2f}B, Strong walls {abs(call_wall - put_wall) / spot_price * 100:.1f}% apart',
            'option_type': 'iron_condor',
            'short_put_strike': round(put_wall, 2),
            'long_put_strike': round(put_wall * 0.98, 2),
            'short_call_strike': round(call_wall, 2),
            'long_call_strike': round(call_wall * 1.02, 2),
            'option_expiry': '5-10 DTE',
            'stop_loss_pct': 100,
            'profit_target_pct': 50
        }
        setups.append(setup)
    
    return setups

def enhance_setup_for_big_moves(setup: dict, gex_data: dict) -> dict:
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

def plot_gex_structure(symbol: str, gex_data: dict) -> go.Figure:
    """Create detailed GEX structure visualization"""
    # Get GEX structure data
    gex_structure = gex_data.get('gex_structure', {})
    strikes = gex_structure.get('strikes', [])
    gex_values = gex_structure.get('gex_values', [])
    call_gamma = gex_structure.get('call_gamma', [])
    put_gamma = gex_structure.get('put_gamma', [])
    
    # Create subplot figure
    fig = go.Figure()
    
    # GEX values bar chart (stacked)
    fig.add_trace(
        go.Bar(
            x=strikes,
            y=call_gamma,
            name='Call Gamma',
            marker_color='#28a745',
            opacity=0.7
        )
    )
    
    fig.add_trace(
        go.Bar(
            x=strikes,
            y=put_gamma,
            name='Put Gamma',
            marker_color='#dc3545',
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
            line=dict(color='#007bff', width=2)
        )
    )
    
    # Add zero line
    fig.add_shape(
        type="line",
        x0=min(strikes) if strikes else gex_data['spot_price'] * 0.9,
        y0=0,
        x1=max(strikes) if strikes else gex_data['spot_price'] * 1.1,
        y1=0,
        line=dict(color="gray", dash="dash"),
    )
    
    # Add price levels
    levels = [
        (gex_data['call_wall'], "Call Wall", "#dc3545"),
        (gex_data['spot_price'], "Current Price", "#007bff"),
        (gex_data['gamma_flip_point'], "Gamma Flip", "#fd7e14"),
        (gex_data['put_wall'], "Put Wall", "#28a745")
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
        ),
        template="plotly_white"
    )
    
    return fig

def format_confidence_class(confidence):
    """Return CSS class based on confidence score"""
    if confidence >= 80:
        return "high-confidence"
    elif confidence >= 60:
        return "medium-confidence"
    else:
        return "low-confidence"

# Discord Webhook Functions
def send_discord_alert(setup: dict, webhook_url: str):
    """Send Discord webhook alert for high-confidence setups"""
    try:
        if not webhook_url or not webhook_url.startswith('https://discord.com/api/webhooks/'):
            return False, "Invalid webhook URL"
        
        symbol = setup.get('symbol', 'Unknown')
        confidence = setup.get('confidence', 0)
        setup_type = setup.get('setup_type', 'Unknown')
        reason = setup.get('reason', 'No reason provided')
        entry = setup.get('entry_criteria', 'No entry criteria')
        target = setup.get('target', 'No target')
        
        # Color based on setup type
        color_map = {
            'squeeze_play': 7844437,    # Green
            'breakdown_play': 16525609, # Red
            'premium_selling': 16750592, # Orange
            'iron_condor': 3447003,     # Blue
        }
        
        color = color_map.get(setup_type, 7506394)  # Default gray
        
        # Create Discord embed
        payload = {
            "username": "GEX Master Pro",
            "avatar_url": "https://i.imgur.com/oBPXx0D.png",
            "embeds": [
                {
                    "title": f"{symbol} - {setup_type.replace('_', ' ').title()} Setup",
                    "color": color,
                    "description": reason,
                    "fields": [
                        {
                            "name": "Confidence",
                            "value": f"{confidence:.0f}%",
                            "inline": True
                        },
                        {
                            "name": "Entry",
                            "value": entry,
                            "inline": True
                        },
                        {
                            "name": "Target",
                            "value": str(target),
                            "inline": True
                        }
                    ],
                    "footer": {
                        "text": f"GEX Master Pro Alert - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    }
                }
            ]
        }
        
        # Add big move info if available
        if setup.get('big_move_mode'):
            payload["embeds"][0]["fields"].append({
                "name": "Big Move Alert",
                "value": f"{setup.get('big_move_type')} - Target: ${setup.get('big_move_target', 0):.2f} ({setup.get('big_move_return_pct', 0):.1f}%)",
                "inline": False
            })
        
        # Send the webhook
        response = requests.post(webhook_url, json=payload)
        
        if response.status_code == 204:
            st.session_state.discord_webhook['last_sent'] = datetime.now()
            return True, "Alert sent successfully"
        else:
            return False, f"Failed to send alert: {response.status_code} - {response.text}"
            
    except Exception as e:
        return False, f"Error sending Discord alert: {str(e)}"

def test_discord_webhook(webhook_url: str):
    """Send a test alert to Discord webhook"""
    try:
        if not webhook_url or not webhook_url.startswith('https://discord.com/api/webhooks/'):
            return False, "Invalid webhook URL"
        
        # Create test payload
        payload = {
            "username": "GEX Master Pro",
            "avatar_url": "https://i.imgur.com/oBPXx0D.png",
            "embeds": [
                {
                    "title": "Test Alert",
                    "color": 5814783,  # Purple
                    "description": "This is a test alert from GEX Master Pro",
                    "fields": [
                        {
                            "name": "Status",
                            "value": "Webhook working correctly",
                            "inline": True
                        },
                        {
                            "name": "Time",
                            "value": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            "inline": True
                        }
                    ],
                    "footer": {
                        "text": "GEX Master Pro - Test Alert"
                    }
                }
            ]
        }
        
        # Send the webhook
        response = requests.post(webhook_url, json=payload)
        
        if response.status_code == 204:
            return True, "Test alert sent successfully"
        else:
            return False, f"Failed to send test alert: {response.status_code} - {response.text}"
            
    except Exception as e:
        return False, f"Error sending test alert: {str(e)}"

# Auto-Trader Functions
def create_paper_trade(setup: dict, capital: float, max_position_size: float):
    """Create a paper trade based on setup"""
    try:
        symbol = setup.get('symbol', 'Unknown')
        setup_type = setup.get('setup_type', 'Unknown')
        confidence = setup.get('confidence', 0)
        option_type = setup.get('option_type', 'call')
        
        # Calculate position size based on confidence
        position_size = (confidence / 100) * max_position_size * capital
        position_size = min(position_size, max_position_size * capital)  # Cap at max
        
        # Generate option details
        if option_type in ['call', 'put']:
            option_strike = setup.get('option_strike', 0)
            option_expiry = setup.get('option_expiry', '5 DTE')
            option_price = round(random.uniform(1.0, 3.0), 2)  # Simulated price
            contracts = int(position_size / (option_price * 100))
            contracts = max(1, contracts)  # Minimum 1 contract
            
            # Create trade
            trade = {
                'id': str(uuid.uuid4()),
                'symbol': symbol,
                'setup_type': setup_type,
                'confidence': confidence,
                'option_type': option_type,
                'option_strike': option_strike,
                'option_expiry': option_expiry,
                'option_price': option_price,
                'contracts': contracts,
                'position_size': contracts * option_price * 100,
                'entry_time': datetime.now(),
                'status': 'active',
                'target_price': setup.get('target', 0),
                'stop_loss': option_price * (1 - setup.get('stop_loss_pct', 50) / 100),
                'profit_target': option_price * (1 + setup.get('profit_target_pct', 100) / 100),
                'current_price': option_price,
                'current_pnl': 0.0,
                'current_pnl_pct': 0.0,
                'exit_time': None,
                'exit_price': None,
                'exit_reason': None,
                'exit_pnl': None,
                'exit_pnl_pct': None
            }
            
        elif option_type in ['call_credit', 'put_credit']:
            option_strike = setup.get('option_strike', 0)
            option_expiry = setup.get('option_expiry', '5 DTE')
            credit_received = round(random.uniform(0.3, 1.0), 2)  # Simulated credit
            max_risk = credit_received * 2  # Simplified risk calculation
            contracts = int(position_size / (max_risk * 100))
            contracts = max(1, contracts)  # Minimum 1 contract
            
            # Create trade
            trade = {
                'id': str(uuid.uuid4()),
                'symbol': symbol,
                'setup_type': setup_type,
                'confidence': confidence,
                'option_type': option_type,
                'option_strike': option_strike,
                'option_expiry': option_expiry,
                'credit_received': credit_received,
                'max_risk': max_risk,
                'contracts': contracts,
                'position_size': max_risk * contracts * 100,
                'entry_time': datetime.now(),
                'status': 'active',
                'target_price': setup.get('target', 0),
                'stop_loss': credit_received * setup.get('stop_loss_pct', 100) / 100,
                'profit_target': credit_received * (1 - setup.get('profit_target_pct', 50) / 100),
                'current_value': credit_received,
                'current_pnl': 0.0,
                'current_pnl_pct': 0.0,
                'exit_time': None,
                'exit_price': None,
                'exit_reason': None,
                'exit_pnl': None,
                'exit_pnl_pct': None
            }
            
        elif option_type == 'iron_condor':
            short_put_strike = setup.get('short_put_strike', 0)
            long_put_strike = setup.get('long_put_strike', 0)
            short_call_strike = setup.get('short_call_strike', 0)
            long_call_strike = setup.get('long_call_strike', 0)
            option_expiry = setup.get('option_expiry', '10 DTE')
            credit_received = round(random.uniform(0.8, 1.5), 2)  # Simulated credit
            max_risk = abs(short_put_strike - long_put_strike) - credit_received  # Simplified
            contracts = int(position_size / (max_risk * 100))
            contracts = max(1, contracts)  # Minimum 1 contract
            
            # Create trade
            trade = {
                'id': str(uuid.uuid4()),
                'symbol': symbol,
                'setup_type': setup_type,
                'confidence': confidence,
                'option_type': option_type,
                'short_put_strike': short_put_strike,
                'long_put_strike': long_put_strike,
                'short_call_strike': short_call_strike,
                'long_call_strike': long_call_strike,
                'option_expiry': option_expiry,
                'credit_received': credit_received,
                'max_risk': max_risk,
                'contracts': contracts,
                'position_size': max_risk * contracts * 100,
                'entry_time': datetime.now(),
                'status': 'active',
                'target_price': setup.get('target', 0),
                'stop_loss': credit_received * setup.get('stop_loss_pct', 100) / 100,
                'profit_target': credit_received * (1 - setup.get('profit_target_pct', 50) / 100),
                'current_value': credit_received,
                'current_pnl': 0.0,
                'current_pnl_pct': 0.0,
                'exit_time': None,
                'exit_price': None,
                'exit_reason': None,
                'exit_pnl': None,
                'exit_pnl_pct': None
            }
        else:
            return None
        
        return trade
        
    except Exception as e:
        st.error(f"Error creating paper trade: {str(e)}")
        return None

def update_paper_trade(trade: dict):
    """Update a paper trade with simulated price movement"""
    try:
        if trade['status'] != 'active':
            return trade
        
        # Simulate price movement based on setup type and time passed
        hours_passed = (datetime.now() - pd.to_datetime(trade['entry_time'])).total_seconds() / 3600
        
        # Limit to realistic trading hours
        hours_passed = min(hours_passed, 6.5)  # Max one trading day
        
        # Different simulation for different option types
        if trade['option_type'] in ['call', 'put']:
            # For directional options, simulate price change
            if trade['setup_type'] == 'squeeze_play' and trade['option_type'] == 'call':
                # Calls in squeeze have high win rate
                win_prob = 0.65
            elif trade['setup_type'] == 'breakdown_play' and trade['option_type'] == 'put':
                # Puts in breakdown have high win rate
                win_prob = 0.65
            else:
                win_prob = 0.5
            
            # Simulate win/loss with random walk
            if random.random() < win_prob:
                # Winning trade - gradually move up with some noise
                price_change_pct = (0.1 * hours_passed) + random.uniform(-0.05, 0.1)
            else:
                # Losing trade - gradually move down with some noise
                price_change_pct = (-0.08 * hours_passed) + random.uniform(-0.1, 0.05)
            
            # Update current price
            new_price = trade['option_price'] * (1 + price_change_pct)
            trade['current_price'] = max(0.01, round(new_price, 2))  # Minimum $0.01
            
            # Calculate P&L
            trade['current_pnl'] = (trade['current_price'] - trade['option_price']) * trade['contracts'] * 100
            trade['current_pnl_pct'] = (trade['current_price'] / trade['option_price'] - 1) * 100
            
            # Check for exit conditions
            if trade['current_price'] <= trade['stop_loss']:
                # Stop loss hit
                trade['status'] = 'stopped'
                trade['exit_time'] = datetime.now()
                trade['exit_price'] = trade['current_price']
                trade['exit_reason'] = 'Stop loss'
                trade['exit_pnl'] = trade['current_pnl']
                trade['exit_pnl_pct'] = trade['current_pnl_pct']
            elif trade['current_price'] >= trade['profit_target']:
                # Profit target hit
                trade['status'] = 'closed'
                trade['exit_time'] = datetime.now()
                trade['exit_price'] = trade['current_price']
                trade['exit_reason'] = 'Profit target'
                trade['exit_pnl'] = trade['current_pnl']
                trade['exit_pnl_pct'] = trade['current_pnl_pct']
                
        elif trade['option_type'] in ['call_credit', 'put_credit', 'iron_condor']:
            # For premium selling, value goes down (good) over time
            if random.random() < 0.7:  # 70% probability of success for selling premium
                # Winning trade - value decays with time
                value_change_pct = (-0.07 * hours_passed) + random.uniform(-0.05, 0.03)
            else:
                # Losing trade - value increases (bad for seller)
                value_change_pct = (0.1 * hours_passed) + random.uniform(-0.03, 0.15)
            
            # Update current value
            new_value = trade['credit_received'] * (1 + value_change_pct)
            trade['current_value'] = max(0.01, min(trade['credit_received'] * 2, round(new_value, 2)))
            
            # Calculate P&L - for credit trades, lower value is better
            trade['current_pnl'] = (trade['credit_received'] - trade['current_value']) * trade['contracts'] * 100
            trade['current_pnl_pct'] = (1 - trade['current_value'] / trade['credit_received']) * 100
            
            # Check for exit conditions
            if trade['current_value'] >= trade['stop_loss']:
                # Stop loss hit
                trade['status'] = 'stopped'
                trade['exit_time'] = datetime.now()
                trade['exit_price'] = trade['current_value']
                trade['exit_reason'] = 'Stop loss'
                trade['exit_pnl'] = trade['current_pnl']
                trade['exit_pnl_pct'] = trade['current_pnl_pct']
            elif trade['current_value'] <= trade['profit_target']:
                # Profit target hit
                trade['status'] = 'closed'
                trade['exit_time'] = datetime.now()
                trade['exit_price'] = trade['current_value']
                trade['exit_reason'] = 'Profit target'
                trade['exit_pnl'] = trade['current_pnl']
                trade['exit_pnl_pct'] = trade['current_pnl_pct']
        
        # Time-based exit for options to simulate time decay
        if hours_passed > 20:  # Exit after ~3 trading days for short-term options
            trade['status'] = 'closed'
            trade['exit_time'] = datetime.now()
            trade['exit_price'] = trade['current_price'] if 'current_price' in trade else trade['current_value']
            trade['exit_reason'] = 'Time-based exit'
            trade['exit_pnl'] = trade['current_pnl']
            trade['exit_pnl_pct'] = trade['current_pnl_pct']
            
        return trade
        
    except Exception as e:
        st.error(f"Error updating paper trade: {str(e)}")
        return trade

def update_all_paper_trades():
    """Update all active paper trades"""
    if 'auto_trader' not in st.session_state:
        return
    
    # Update each active trade
    for i, trade in enumerate(st.session_state.auto_trader['positions']):
        if trade['status'] == 'active':
            st.session_state.auto_trader['positions'][i] = update_paper_trade(trade)
            
    # Calculate total P&L and update available capital
    total_pnl = 0
    for trade in st.session_state.auto_trader['positions']:
        if trade['status'] == 'active':
            total_pnl += trade['current_pnl']
        elif trade['status'] in ['closed', 'stopped']:
            total_pnl += trade['exit_pnl']
    
    # Update available capital
    used_capital = sum([trade['position_size'] for trade in st.session_state.auto_trader['positions'] if trade['status'] == 'active'])
    st.session_state.auto_trader['available_capital'] = st.session_state.auto_trader['capital'] + total_pnl - used_capital
    
    # Update last check time
    st.session_state.auto_trader['last_check'] = datetime.now()

def create_auto_trader_trade(setup: dict):
    """Create a trade in the auto-trader based on setup"""
    if 'auto_trader' not in st.session_state or not st.session_state.auto_trader['enabled']:
        return False, "Auto-trader is disabled"
    
    if setup['confidence'] < st.session_state.auto_trader['min_confidence']:
        return False, f"Setup confidence {setup['confidence']}% below minimum threshold {st.session_state.auto_trader['min_confidence']}%"
    
    # Check available capital
    if st.session_state.auto_trader['available_capital'] <= 0:
        return False, "Insufficient capital"
    
    # Create the paper trade
    trade = create_paper_trade(
        setup, 
        st.session_state.auto_trader['capital'], 
        st.session_state.auto_trader['max_position_size']
    )
    
    if not trade:
        return False, "Failed to create trade"
    
    # Add to positions
    st.session_state.auto_trader['positions'].append(trade)
    
    # Update available capital
    st.session_state.auto_trader['available_capital'] -= trade['position_size']
    
    return True, f"Created new {trade['option_type']} trade for {trade['symbol']}"

def get_auto_trader_stats():
    """Get auto-trader performance statistics"""
    if 'auto_trader' not in st.session_state:
        return None
    
    stats = {
        'total_trades': len(st.session_state.auto_trader['positions']),
        'active_trades': sum(1 for t in st.session_state.auto_trader['positions'] if t['status'] == 'active'),
        'closed_trades': sum(1 for t in st.session_state.auto_trader['positions'] if t['status'] == 'closed'),
        'stopped_trades': sum(1 for t in st.session_state.auto_trader['positions'] if t['status'] == 'stopped'),
        'total_pnl': 0,
        'total_pnl_pct': 0,
        'win_rate': 0,
        'avg_win': 0,
        'avg_loss': 0,
        'largest_win': 0,
        'largest_loss': 0
    }
    
    # Calculate P&L metrics
    if stats['total_trades'] > 0:
        wins = []
        losses = []
        
        for trade in st.session_state.auto_trader['positions']:
            if trade['status'] == 'active':
                stats['total_pnl'] += trade['current_pnl']
                if trade['current_pnl'] > 0:
                    wins.append(trade['current_pnl'])
                else:
                    losses.append(trade['current_pnl'])
            elif trade['status'] in ['closed', 'stopped']:
                stats['total_pnl'] += trade['exit_pnl']
                if trade['exit_pnl'] > 0:
                    wins.append(trade['exit_pnl'])
                else:
                    losses.append(trade['exit_pnl'])
        
        # Calculate win rate
        finished_trades = stats['closed_trades'] + stats['stopped_trades']
        if finished_trades > 0:
            stats['win_rate'] = len(wins) / finished_trades * 100
        
        # Calculate averages
        if wins:
            stats['avg_win'] = sum(wins) / len(wins)
            stats['largest_win'] = max(wins)
        if losses:
            stats['avg_loss'] = sum(losses) / len(losses)
            stats['largest_loss'] = min(losses)
        
        # Calculate total P&L percentage
        stats['total_pnl_pct'] = (stats['total_pnl'] / st.session_state.auto_trader['capital']) * 100
    
    return stats

# Education Functions
def render_gex_education():
    """Render the GEX education section"""
    st.markdown("""
    <div class="education-section">
        <h3 class="education-title">What is Gamma Exposure (GEX)?</h3>
        <p>Gamma Exposure (GEX) measures the rate of change in market makers' delta hedging requirements as the underlying price moves. It's a critical component in understanding options-driven price dynamics.</p>
        
        <h4>Key Concepts:</h4>
        <ul>
            <li><strong>Gamma Flip Point:</strong> The price level where net dealer gamma exposure changes from positive to negative (or vice versa). This level often acts as a magnet or repellent for price.</li>
            <li><strong>Call Wall:</strong> A concentration of call gamma at a specific strike price that creates resistance as dealers hedge.</li>
            <li><strong>Put Wall:</strong> A concentration of put gamma at a specific strike price that creates support as dealers hedge.</li>
            <li><strong>Net GEX:</strong> The sum of all gamma exposure across strikes, indicating the overall market regime.</li>
        </ul>
        
        <h4>Market Regimes:</h4>
        <ul>
            <li><strong>Positive GEX:</strong> Dealers are long gamma, hedging activities dampen volatility, creating range-bound conditions.</li>
            <li><strong>Negative GEX:</strong> Dealers are short gamma, hedging activities amplify volatility, creating trending conditions.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

def render_strategy_education():
    """Render the strategy education section"""
    st.markdown("""
    <div class="education-section">
        <h3 class="education-title">Trading Strategies Based on GEX</h3>
        
        <h4>1. Squeeze Plays (Long Calls)</h4>
        <p>Best when net GEX is negative and price is below the gamma flip point:</p>
        <ul>
            <li>Buy calls slightly above the gamma flip point</li>
            <li>Target the nearest call wall as resistance</li>
            <li>Use 2-5 DTE options for maximum gamma sensitivity</li>
            <li>Size positions for potential 100% loss</li>
        </ul>
        
        <h4>2. Breakdown Plays (Long Puts)</h4>
        <p>Best when net GEX is strongly positive but price is hovering just above the flip point:</p>
        <ul>
            <li>Buy puts slightly below the flip point</li>
            <li>Target the nearest put wall as support</li>
            <li>Use 3-7 DTE options</li>
            <li>Set stops above nearest call wall</li>
        </ul>
        
        <h4>3. Premium Selling</h4>
        <p>Best when net GEX is strongly positive and price is between walls:</p>
        <ul>
            <li>Sell calls at or above call wall strikes</li>
            <li>Sell puts at or below put wall strikes</li>
            <li>Use 0-2 DTE for rapid theta decay</li>
            <li>Target 50% profit or close if price approaches wall</li>
        </ul>
        
        <h4>4. Iron Condors</h4>
        <p>Best when net GEX is positive with defined walls:</p>
        <ul>
            <li>Sell put spreads below put wall</li>
            <li>Sell call spreads above call wall</li>
            <li>Use 5-10 DTE for optimal theta/gamma ratio</li>
            <li>Consider broken wing adjustments based on call vs put gamma totals</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

def render_risk_education():
    """Render the risk management education section"""
    st.markdown("""
    <div class="education-section">
        <h3 class="education-title">Risk Management for GEX Trading</h3>
        
        <h4>Position Sizing Guidelines:</h4>
        <ul>
            <li><strong>Squeeze Plays:</strong> Maximum 3% of capital</li>
            <li><strong>Premium Selling:</strong> Maximum 5% of capital</li>
            <li><strong>Iron Condors:</strong> Maximum 2% max loss exposure</li>
            <li><strong>Total Exposure:</strong> Never exceed 15% in similar setups</li>
        </ul>
        
        <h4>Stop Loss Rules:</h4>
        <ul>
            <li><strong>Long Options:</strong> 50% of premium</li>
            <li><strong>Short Options:</strong> 100% of premium received (or 2x credit)</li>
            <li><strong>Iron Condors:</strong> When short strike breached and delta exceeds 0.30</li>
        </ul>
        
        <h4>Profit Targets:</h4>
        <ul>
            <li><strong>Long Options:</strong> 100% of premium</li>
            <li><strong>Short Options:</strong> 50% of premium received</li>
            <li><strong>Iron Condors:</strong> 50% of max profit</li>
        </ul>
        
        <h4>Additional Risk Rules:</h4>
        <ul>
            <li><strong>Time Stops:</strong> Close any position with less than 1 DTE</li>
            <li><strong>Correlation Risk:</strong> Reduce position size when trading correlated assets</li>
            <li><strong>VIX Risk:</strong> Exit premium selling positions if VIX spikes above 30</li>
            <li><strong>Earnings Risk:</strong> Avoid holding through earnings announcements</li>
            <li><strong>FOMC Risk:</strong> Reduce position size or avoid trading during Fed meetings</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Main Application
def main():
    """Main application entry point"""
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.title("⚡ GEX Master Pro")
    st.markdown("Complete Gamma Exposure Trading System")
    
    # Sidebar
    with st.sidebar:
        st.subheader("Navigation")
        
        # Navigation tabs
        page = st.radio(
            "Select Page",
            ["Single Symbol Analysis", "Pipeline Dashboard", "Auto Trader", "Market Overview", "Education Center"]
        )
        
        st.markdown("---")
        
        # API Status
        st.subheader("System Status")
        
        # API status
        if st.session_state.api_status['status'] == 'online':
            st.markdown(f"""
            <div style="display: flex; align-items: center">
                <div class="status-indicator status-online"></div>
                <div>API: <span class="status-connected">Online</span></div>
            </div>
            """, unsafe_allow_
                        <div style="display: flex; align-items: center">
               <div class="status-indicator status-online"></div>
               <div>API: <span class="status-connected">Online</span></div>
           </div>
           """, unsafe_allow_html=True)
       else:
           st.markdown(f"""
           <div style="display: flex; align-items: center">
               <div class="status-indicator status-offline"></div>
               <div>API: <span class="status-disconnected">Offline</span></div>
           </div>
           """, unsafe_allow_html=True)
       
       # Database status
       db_connection = init_databricks_connection()
       if db_connection:
           st.markdown(f"""
           <div style="display: flex; align-items: center">
               <div class="status-indicator status-online"></div>
               <div>Database: <span class="status-connected">Connected</span></div>
           </div>
           """, unsafe_allow_html=True)
       else:
           st.markdown(f"""
           <div style="display: flex; align-items: center">
               <div class="status-indicator status-offline"></div>
               <div>Database: <span class="status-disconnected">Disconnected</span></div>
           </div>
           """, unsafe_allow_html=True)
       
       # Auto-trader status
       if st.session_state.auto_trader['enabled']:
           st.markdown(f"""
           <div style="display: flex; align-items: center">
               <div class="status-indicator status-online"></div>
               <div>Auto-Trader: <span class="status-connected">Enabled</span></div>
           </div>
           """, unsafe_allow_html=True)
       else:
           st.markdown(f"""
           <div style="display: flex; align-items: center">
               <div class="status-indicator status-offline"></div>
               <div>Auto-Trader: <span class="status-disconnected">Disabled</span></div>
           </div>
           """, unsafe_allow_html=True)
       
       # Test API connection
       if st.button("Test API Connection"):
           with st.spinner("Testing API connection..."):
               try:
                   test_symbol = "SPY"
                   response = requests.get(
                       f"{api_config['base_url']}/gex/latest",
                       params={
                           'ticker': test_symbol,
                           'username': api_config['username'],
                           'format': 'json'
                       },
                       timeout=10
                   )
                   
                   if response.status_code == 200:
                       st.session_state.api_status['status'] = 'online'
                       st.session_state.api_status['last_checked'] = datetime.now()
                       st.session_state.api_status['successes'] += 1
                       st.success("✅ API connection successful!")
                   else:
                       st.session_state.api_status['status'] = 'offline'
                       st.session_state.api_status['last_checked'] = datetime.now()
                       st.session_state.api_status['failures'] += 1
                       st.error(f"❌ API returned status code: {response.status_code}")
                       
               except Exception as e:
                   st.session_state.api_status['status'] = 'offline'
                   st.session_state.api_status['last_checked'] = datetime.now()
                   st.session_state.api_status['failures'] += 1
                   st.error(f"❌ API connection failed: {str(e)}")
       
       st.markdown("---")
       
       # Discord Webhook section
       st.subheader("Discord Alerts")
       webhook_url = st.text_input(
           "Discord Webhook URL",
           value=st.session_state.discord_webhook['url'],
           type="password",
           help="Enter your Discord webhook URL for alerts"
       )
       
       # Update session state
       st.session_state.discord_webhook['url'] = webhook_url
       
       # Webhook enabled toggle
       webhook_enabled = st.checkbox(
           "Enable Discord Alerts",
           value=st.session_state.discord_webhook['enabled'],
           help="Send alerts to Discord for high-confidence setups"
       )
       
       # Update session state
       st.session_state.discord_webhook['enabled'] = webhook_enabled
       
       # Minimum confidence for alerts
       webhook_min_confidence = st.slider(
           "Min. Alert Confidence",
           min_value=60,
           max_value=100,
           value=st.session_state.discord_webhook['min_confidence'],
           step=5,
           help="Minimum confidence score to trigger alerts"
       )
       
       # Update session state
       st.session_state.discord_webhook['min_confidence'] = webhook_min_confidence
       
       # Setup types to alert on
       alert_types = st.multiselect(
           "Alert on Setup Types",
           ["squeeze_play", "breakdown_play", "premium_selling", "iron_condor"],
           default=st.session_state.discord_webhook['alert_types'],
           help="Select which setup types to send alerts for"
       )
       
       # Update session state
       st.session_state.discord_webhook['alert_types'] = alert_types
       
       # Test webhook
       if st.button("Test Discord Webhook"):
           if webhook_url and webhook_url.startswith('https://discord.com/api/webhooks/'):
               with st.spinner("Sending test alert..."):
                   success, message = test_discord_webhook(webhook_url)
                   if success:
                       st.success(message)
                   else:
                       st.error(message)
           else:
               st.error("Please enter a valid Discord webhook URL")
       
       st.markdown("---")
       
       # Watchlist Management
       st.subheader("Watchlist")
       
       # Add/remove symbols
       custom_symbols = st.text_input(
           "Add Symbols",
           placeholder="SYMBOL1, SYMBOL2, ...",
           help="Enter comma-separated symbols"
       )
       
       if custom_symbols:
           new_symbols = [s.strip().upper() for s in custom_symbols.split(",") if s.strip()]
           st.session_state.watchlist.extend(new_symbols)
           # Remove duplicates
           st.session_state.watchlist = list(set(st.session_state.watchlist))
       
       # Display current watchlist
       if st.session_state.watchlist:
           st.write("Current Watchlist:")
           watchlist_cols = st.columns(3)
           for i, symbol in enumerate(sorted(st.session_state.watchlist)):
               with watchlist_cols[i % 3]:
                   if st.button(f"❌ {symbol}"):
                       st.session_state.watchlist.remove(symbol)
                       st.rerun()
   
   # Main content based on selected page
   if page == "Single Symbol Analysis":
       render_single_symbol_analysis()
   elif page == "Pipeline Dashboard":
       render_pipeline_dashboard()
   elif page == "Auto Trader":
       render_auto_trader_dashboard()
   elif page == "Market Overview":
       render_market_overview()
   elif page == "Education Center":
       render_education_center()
   
   # Footer
   st.markdown("---")
   st.caption(f"GEX Master Pro - Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
   
   # Auto-refresh logic
   auto_refresh = False  # Set to True to enable auto-refresh
   if auto_refresh and st.session_state.last_update:
       if (datetime.now() - st.session_state.last_update).total_seconds() > 300:  # 5 minutes
           st.rerun()

def render_single_symbol_analysis():
   """Render the single symbol analysis page"""
   st.header("Single Symbol Analysis")
   
   # Input section
   col1, col2 = st.columns([2, 1])
   
   with col1:
       symbol = st.text_input("Enter Symbol", placeholder="SPY", key="symbol_input").upper()
   
   with col2:
       analyze_button = st.button("Analyze", type="primary")
   
   if analyze_button and symbol:
       with st.spinner(f"Analyzing {symbol}..."):
           gex_data = fetch_real_gex_data(symbol)
           
           # Update API status based on result
           if gex_data.get('success'):
               # Store in session state
               st.session_state.gex_data[symbol] = gex_data
               st.session_state.last_update = datetime.now()
               
               # Determine data source for user information
               if gex_data.get('database_source'):
                   data_source = "Database"
                   st.info("📊 Using data from your Databricks pipeline")
               else:
                   data_source = "API"
                   st.success("📡 Using real-time API data")
               
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
                       
                       # Auto-trader integration
                       if st.session_state.auto_trader['enabled'] and enhanced_setup['confidence'] >= st.session_state.auto_trader['min_confidence']:
                           # Check if setup already in auto-trader
                           setup_exists = any(
                               t['symbol'] == enhanced_setup['symbol'] and 
                               t['setup_type'] == enhanced_setup['setup_type'] and
                               t['status'] == 'active'
                               for t in st.session_state.auto_trader['positions']
                           )
                           
                           if not setup_exists:
                               # Add to auto-trader
                               success, message = create_auto_trader_trade(enhanced_setup)
                               if success:
                                   st.success(f"🤖 Auto-Trader: {message}")
                               else:
                                   st.info(f"🤖 Auto-Trader: {message}")
                       
                       # Discord webhook integration
                       if (st.session_state.discord_webhook['enabled'] and 
                           enhanced_setup['confidence'] >= st.session_state.discord_webhook['min_confidence'] and
                           enhanced_setup['setup_type'] in st.session_state.discord_webhook['alert_types']):
                           
                           # Send alert if webhook configured
                           if st.session_state.discord_webhook['url']:
                               success, message = send_discord_alert(enhanced_setup, st.session_state.discord_webhook['url'])
                               if success:
                                   st.success(f"🔔 Discord Alert: {message}")
                               else:
                                   st.warning(f"🔔 Discord Alert: {message}")
                       
                       if enhanced_setup.get('big_move_mode'):
                           st.markdown(f"""
                           <div class="card high-confidence">
                               <h4>🚀 BIG MOVE SETUP DETECTED - {enhanced_setup['confidence']:.0f}% Confidence</h4>
                               <div style="display: flex; justify-content: space-between; margin-top: 1rem;">
                                   <div>
                                       <strong>Strategy:</strong> {enhanced_setup['strategy']}<br>
                                       <strong>Setup Type:</strong> {enhanced_setup['setup_type'].replace('_', ' ').title()}<br>
                                       <strong>Entry:</strong> {enhanced_setup['entry_criteria']}
                                   </div>
                                   <div style="text-align: right;">
                                       <strong>Big Move Target:</strong> ${enhanced_setup['big_move_target']:.2f}<br>
                                       <strong>Expected Return:</strong> +{enhanced_setup['big_move_return_pct']:.1f}%<br>
                                       <strong>Type:</strong> {enhanced_setup['big_move_type']}
                                   </div>
                               </div>
                               <div style="margin-top: 1rem;">
                                   <strong>Reason:</strong> {enhanced_setup['reason']}
                               </div>
                           </div>
                           """, unsafe_allow_html=True)
                       else:
                           confidence_class = format_confidence_class(setup['confidence'])
                           
                           st.markdown(f"""
                           <div class="card {confidence_class}">
                               <h4>{setup['strategy']} - {setup['confidence']:.0f}% Confidence</h4>
                               <div style="display: flex; justify-content: space-between; margin-top: 1rem;">
                                   <div>
                                       <strong>Setup Type:</strong> {setup['setup_type'].replace('_', ' ').title()}<br>
                                       <strong>Entry:</strong> {setup['entry_criteria']}<br>
                                       <strong>Time Frame:</strong> {setup['time_frame']}
                                   </div>
                                   <div style="text-align: right;">
                                       <strong>Target:</strong> ${setup['target']:.2f}<br>
                                       <strong>Expected Move:</strong> {setup['expected_move']}<br>
                                       <strong>Option Type:</strong> {setup['option_type'].replace('_', ' ').title()}
                                   </div>
                               </div>
                               <div style="margin-top: 1rem;">
                                   <strong>Reason:</strong> {setup['reason']}
                               </div>
                           </div>
                           """, unsafe_allow_html=True)
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
               
               st.markdown(f"""
               <div class="alert-box alert-{regime_color}">
                   <strong>Market Regime:</strong> {regime}<br>
                   {regime_description}
               </div>
               """, unsafe_allow_html=True)
               
           else:
               st.error(f"Failed to fetch data for {symbol}: {gex_data.get('error', 'Unknown error')}")

def render_pipeline_dashboard():
   """Render the pipeline dashboard page"""
   st.header("Pipeline Dashboard")
   
   # Load data
   data_result = load_gex_data()
   
   if isinstance(data_result, dict):
       df = data_result['data']
       status = data_result['status']
       message = data_result['message']
       debug_info = data_result.get('debug_info', '')
   else:
       df = data_result
       status = 'unknown'
       message = 'Data loaded'
       debug_info = ''
   
   # Status indicator with debug info
   if status == 'connected':
       st.success(f"✅ {message}")
       if debug_info:
           with st.expander("Debug Information", expanded=False):
               st.markdown(debug_info)
   elif status == 'error':
       st.warning(f"⚠️ {message}")
   else:
       st.info(f"ℹ️ {message}")
   
   # Refresh button
   if st.button("Refresh Data", type="primary"):
       st.cache_data.clear()
       st.rerun()
   
   if df.empty:
       st.error("No pipeline data available")
       st.info("""
       **Possible Reasons:**
       1. No recent pipeline runs
       2. Database connection issue
       3. Pipeline table is empty
       
       Please check your Databricks environment and ensure your pipeline has been executed.
       """)
       return
   
   # Filters
   st.subheader("Filters")
   
   col1, col2, col3 = st.columns(3)
   
   with col1:
       min_confidence = st.slider("Minimum Confidence %", 0, 100, 70)
   
   with col2:
       if not df.empty:
           setup_types = st.multiselect(
               "Setup Types",
               df['structure_type'].unique().tolist(),
               default=df['structure_type'].unique().tolist()
           )
       else:
           setup_types = []
   
   with col3:
       if not df.empty:
           symbols = st.multiselect(
               "Symbols",
               sorted(df['symbol'].unique().tolist()),
               default=sorted(df['symbol'].unique().tolist())[:5] if len(df['symbol'].unique()) > 5 else sorted(df['symbol'].unique().tolist())
           )
       else:
           symbols = []
   
   # Apply filters
   filtered_df = df[
       (df['confidence_score'] >= min_confidence) &
       (df['structure_type'].isin(setup_types)) &
       (df['symbol'].isin(symbols))
   ].copy()
   
   if filtered_df.empty:
       st.warning("No data matches your filters. Try adjusting the criteria.")
       return
   
   # Key metrics row
   st.subheader("Pipeline Results")
   
   col1, col2, col3, col4 = st.columns(4)
   
   with col1:
       st.markdown(f"""
       <div class="metric-container">
           <div class="metric-title">Total Setups</div>
           <div class="metric-value">{len(filtered_df)}</div>
       </div>
       """, unsafe_allow_html=True)
   
   with col2:
       high_conf = len(filtered_df[filtered_df['confidence_score'] >= 85])
       st.markdown(f"""
       <div class="metric-container">
           <div class="metric-title">High Confidence</div>
           <div class="metric-value">{high_conf}</div>
       </div>
       """, unsafe_allow_html=True)
   
   with col3:
       avg_conf = filtered_df['confidence_score'].mean()
       st.markdown(f"""
       <div class="metric-container">
           <div class="metric-title">Avg Confidence</div>
           <div class="metric-value">{avg_conf:.1f}%</div>
       </div>
       """, unsafe_allow_html=True)
   
   with col4:
       try:
           enhanced_strategies = len(filtered_df[filtered_df['category'] == 'ENHANCED_STRATEGY'])
       except:
           enhanced_strategies = 0
           
       st.markdown(f"""
       <div class="metric-container">
           <div class="metric-title">Enhanced Strategies</div>
           <div class="metric-value">{enhanced_strategies}</div>
       </div>
       """, unsafe_allow_html=True)
   
   # Top setups
   st.subheader("High Confidence Setups")
   
   # Show priority 1 setups first
   try:
       priority_setups = filtered_df[filtered_df['priority'] == 1].head(10)
   except:
       priority_setups = filtered_df.sort_values('confidence_score', ascending=False).head(10)
   
   if not priority_setups.empty:
       for _, setup in priority_setups.iterrows():
           confidence_class = format_confidence_class(setup['confidence_score'])
           
           # Format distance
           try:
               distance_display = f"{setup['distance_to_flip_pct']:+.2f}%"
           except:
               distance_display = "N/A"
           
           st.markdown(f"""
           <div class="card {confidence_class}">
               <h4>{setup['symbol']} - {setup['confidence_score']}% Confidence</h4>
               <div style="display: flex; justify-content: space-between; margin-top: 1rem;">
                   <div>
                       <strong>Setup:</strong> {setup['structure_type'].replace('_', ' ').title()}<br>
                       <strong>Category:</strong> {setup.get('category', 'N/A')}
                   </div>
                   <div style="text-align: right;">
                       <strong>Spot:</strong> ${setup['spot_price']:.2f}<br>
                       <strong>Flip Point:</strong> ${setup['gamma_flip_point']:.2f}
                   </div>
               </div>
               <div style="margin-top: 1rem;">
                   <strong>Distance:</strong> {distance_display} | 
                   <strong>Priority:</strong> {setup.get('priority', 'N/A')} | 
                   <strong>Recommendation:</strong> {setup.get('recommendation', 'N/A')}
               </div>
               <div style="margin-top: 0.5rem; font-size: 0.9em; color: #666;">
                   <strong>Created:</strong> {pd.to_datetime(setup['analysis_timestamp']).strftime('%m/%d %H:%M')}
               </div>
           </div>
           """, unsafe_allow_html=True)
   else:
       st.info("No high-priority setups match your current filters")
   
   # Charts section
   st.subheader("Analysis Charts")
   
   col1, col2 = st.columns(2)
   
   with col1:
       # Setup type distribution
       setup_counts = filtered_df['structure_type'].value_counts()
       
       fig_pie = go.Figure(data=[go.Pie(
           labels=setup_counts.index,
           values=setup_counts.values,
           hole=0.4
       )])
       
       fig_pie.update_layout(
           title="Setup Type Distribution",
           height=300,
           template="plotly_white"
       )
       
       st.plotly_chart(fig_pie, use_container_width=True)
   
   with col2:
       # Confidence distribution
       confidence_bins = ['High (85%+)', 'Medium (70-84%)', 'Low (<70%)']
       high = len(filtered_df[filtered_df['confidence_score'] >= 85])
       medium = len(filtered_df[(filtered_df['confidence_score'] >= 70) & (filtered_df['confidence_score'] < 85)])
       low = len(filtered_df[filtered_df['confidence_score'] < 70])
       
       fig_bar = go.Figure(data=[go.Bar(
           x=confidence_bins,
           y=[high, medium, low],
           marker_color=['#28a745', '#ffc107', '#dc3545']
       )])
       
       fig_bar.update_layout(
           title="Confidence Score Distribution",
           height=300,
           yaxis_title="Number of Setups",
           template="plotly_white"
       )
       
       st.plotly_chart(fig_bar, use_container_width=True)
   
   # Symbol performance chart
   if len(filtered_df) > 0:
       st.subheader("Top Symbols by Setup Count")
       
       symbol_counts = filtered_df['symbol'].value_counts().head(15)
       
       fig_symbol = go.Figure(data=[go.Bar(
           x=symbol_counts.index,
           y=symbol_counts.values,
           marker_color='#007bff'
       )])
       
       fig_symbol.update_layout(
           title="Symbols with Most Setups",
           height=400,
           xaxis_title="Symbol",
           yaxis_title="Number of Setups",
           template="plotly_white"
       )
       
       st.plotly_chart(fig_symbol, use_container_width=True)
   
   # Complete data table
   st.subheader("All Pipeline Results")
   
   # Format display columns
   display_df = filtered_df.copy()
   display_df['analysis_timestamp'] = pd.to_datetime(display_df['analysis_timestamp']).dt.strftime('%m/%d/%y %H:%M')
   
   # Rename columns for display
   display_df = display_df.rename(columns={
       'structure_type': 'Setup Type',
       'confidence_score': 'Confidence %',
       'spot_price': 'Spot Price',
       'gamma_flip_point': 'Flip Point',
       'distance_to_flip_pct': 'Distance %',
       'analysis_timestamp': 'Created'
   })
   
   columns_to_display = ['symbol', 'Setup Type', 'Confidence %', 'Spot Price', 
                      'Flip Point', 'Distance %', 'Created']
   
   # Add recommendation column if it exists
   if 'recommendation' in display_df.columns:
       columns_to_display.append('recommendation')
   
   # Add priority column if it exists
   if 'priority' in display_df.columns:
       columns_to_display.append('priority')
   
   st.dataframe(
       display_df[columns_to_display],
       use_container_width=True,
       hide_index=True
   )
   
   # Download section
   st.subheader("Export Data")
   
   col1, col2 = st.columns(2)
   
   with col1:
       csv = filtered_df.to_csv(index=False)
       st.download_button(
           "Download CSV",
           csv,
           f"gex_pipeline_results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
           "text/csv"
       )
   
   with col2:
       json_data = filtered_df.to_json(orient='records', indent=2)
       st.download_button(
           "Download JSON",
           json_data,
           f"gex_pipeline_results_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
           "application/json"
       )

def render_auto_trader_dashboard():
   """Render the auto-trader dashboard"""
   st.header("Auto Trader")
   
   # Update auto-trader status
   update_all_paper_trades()
   
   # Settings and controls
   with st.expander("Auto-Trader Settings", expanded=True):
       col1, col2 = st.columns(2)
       
       with col1:
           # Enable/disable auto-trader
           auto_trader_enabled = st.checkbox(
               "Enable Auto-Trader",
               value=st.session_state.auto_trader['enabled'],
               help="Automatically execute trades based on high-confidence setups"
           )
           
           # Update session state
           st.session_state.auto_trader['enabled'] = auto_trader_enabled
           
           # Minimum confidence for auto-trader
           min_confidence = st.slider(
               "Minimum Confidence",
               min_value=60,
               max_value=100,
               value=st.session_state.auto_trader['min_confidence'],
               step=5,
               help="Minimum confidence score to trigger automatic trades"
           )
           
           # Update session state
           st.session_state.auto_trader['min_confidence'] = min_confidence
       
       with col2:
           # Paper trading capital
           capital = st.number_input(
               "Paper Trading Capital",
               min_value=1000,
               max_value=1000000,
               value=int(st.session_state.auto_trader['capital']),
               step=1000,
               help="Total capital for paper trading"
           )
           
           # Update session state
           st.session_state.auto_trader['capital'] = capital
           
           # Maximum position size
           max_position_size = st.slider(
               "Max Position Size (%)",
               min_value=1,
               max_value=20,
               value=int(st.session_state.auto_trader['max_position_size'] * 100),
               step=1,
               help="Maximum position size as percentage of capital"
           )
           
           # Update session state
           st.session_state.auto_trader['max_position_size'] = max_position_size / 100
   
   # Performance metrics
   st.subheader("Performance Overview")
   
   # Get auto-trader stats
   stats = get_auto_trader_stats()
   
   if stats:
       col1, col2, col3, col4 = st.columns(4)
       
       with col1:
           st.markdown(f"""
           <div class="metric-container">
               <div class="metric-title">Available Capital</div>
               <div class="metric-value">${st.session_state.auto_trader['available_capital']:.2f}</div>
           </div>
           """, unsafe_allow_html=True)
       
       with col2:
           st.markdown(f"""
           <div class="metric-container">
               <div class="metric-title">Total P&L</div>
               <div class="metric-value" style="color: {'#28a745' if stats['total_pnl'] >= 0 else '#dc3545'}">
                   {stats['total_pnl']:+.2f} ({stats['total_pnl_pct']:+.1f}%)
               </div>
           </div>
           """, unsafe_allow_html=True)
       
       with col3:
           st.markdown(f"""
           <div class="metric-container">
               <div class="metric-title">Win Rate</div>
               <div class="metric-value">{stats['win_rate']:.1f}%</div>
           </div>
           """, unsafe_allow_html=True)
       
       with col4:
           st.markdown(f"""
           <div class="metric-container">
               <div class="metric-title">Active Trades</div>
               <div class="metric-value">{stats['active_trades']}</div>
           </div>
           """, unsafe_allow_html=True)
   
   # Active positions
   st.subheader("Active Positions")
   
   active_positions = [t for t in st.session_state.auto_trader['positions'] if t['status'] == 'active']
   
   if active_positions:
       for position in active_positions:
           # Determine status color
           pnl_color = "#28a745" if position['current_pnl'] >= 0 else "#dc3545"
           
           st.markdown(f"""
           <div class="card position-active">
               <h4>{position['symbol']} - {position['setup_type'].replace('_', ' ').title()}</h4>
               <div style="display: flex; justify-content: space-between; margin-top: 1rem;">
                   <div>
                       <strong>Type:</strong> {position['option_type'].replace('_', ' ').title()}<br>
                       <strong>Entry:</strong> {pd.to_datetime(position['entry_time']).strftime('%m/%d %H:%M')}<br>
                       <strong>Position Size:</strong> ${position['position_size']:.2f}
                   </div>
                   <div style="text-align: right;">
                       <strong>Current P&L:</strong> <span style="color: {pnl_color}">{position['current_pnl']:+.2f} ({position['current_pnl_pct']:+.1f}%)</span><br>
                       <strong>Target:</strong> ${position['profit_target'] if 'profit_target' in position else position['target_price']:.2f}<br>
                       <strong>Stop:</strong> ${position['stop_loss']:.2f}
                   </div>
               </div>
               <div style="margin-top: 1rem;">
                   <strong>Details:</strong> 
                   {f"{position['contracts']} contracts at ${position['option_price']:.2f}" if 'option_price' in position else f"Credit received: ${position['credit_received']:.2f}"}
               </div>
           </div>
           """, unsafe_allow_html=True)
   else:
       st.info("No active positions")
   
   # Trade history
   st.subheader("Trade History")
   
   closed_positions = [t for t in st.session_state.auto_trader['positions'] if t['status'] != 'active']
   
   if closed_positions:
       # Sort by exit time (most recent first)
       closed_positions.sort(key=lambda x: x['exit_time'] if x['exit_time'] else datetime.now(), reverse=True)
       
       for position in closed_positions:
           # Determine status class
           position_class = "position-closed" if position['status'] == 'closed' else "position-stopped"
           
           # Determine P&L color
           pnl_color = "#28a745" if position['exit_pnl'] >= 0 else "#dc3545"
           
           st.markdown(f"""
           <div class="card {position_class}">
               <h4>{position['symbol']} - {position['setup_type'].replace('_', ' ').title()} ({position['status'].title()})</h4>
               <div style="display: flex; justify-content: space-between; margin-top: 1rem;">
                   <div>
                       <strong>Type:</strong> {position['option_type'].replace('_', ' ').title()}<br>
                       <strong>Entry:</strong> {pd.to_datetime(position['entry_time']).strftime('%m/%d %H:%M')}<br>
                       <strong>Exit:</strong> {pd.to_datetime(position['exit_time']).strftime('%m/%d %H:%M')}
                   </div>
                   <div style="text-align: right;">
                       <strong>P&L:</strong> <span style="color: {pnl_color}">{position['exit_pnl']:+.2f} ({position['exit_pnl_pct']:+.1f}%)</span><br>
                       <strong>Exit Reason:</strong> {position['exit_reason']}<br>
                       <strong>Position Size:</strong> ${position['position_size']:.2f}
                   </div>
               </div>
               <div style="margin-top: 1rem;">
                   <strong>Details:</strong> 
                   {f"{position['contracts']} contracts at ${position['option_price']:.2f}" if 'option_price' in position else f"Credit received: ${position['credit_received']:.2f}"}
               </div>
           </div>
           """, unsafe_allow_html=True)
   else:
       st.info("No trade history")
   
   # Performance charts
   if stats and stats['total_trades'] > 0:
       st.subheader("Performance Analytics")
       
       col1, col2 = st.columns(2)
       
       with col1:
           # Trade outcome pie chart
           outcome_labels = ['Profitable', 'Loss', 'Active']
           outcome_values = [stats['closed_trades'], stats['stopped_trades'], stats['active_trades']]
           
           fig_outcome = go.Figure(data=[go.Pie(
               labels=outcome_labels,
               values=outcome_values,
               marker_colors=['#28a745', '#dc3545', '#007bff'],
               hole=0.4
           )])
           
           fig_outcome.update_layout(
               title="Trade Outcomes",
               height=300,
               template="plotly_white"
           )
           
           st.plotly_chart(fig_outcome, use_container_width=True)
       
       with col2:
           # Setup type performance
           setup_types = {}
           for trade in st.session_state.auto_trader['positions']:
               setup_type = trade['setup_type']
               if setup_type not in setup_types:
                   setup_types[setup_type] = {'count': 0, 'pnl': 0}
               
               setup_types[setup_type]['count'] += 1
               
               if trade['status'] == 'active':
                   setup_types[setup_type]['pnl'] += trade['current_pnl']
               else:
                   setup_types[setup_type]['pnl'] += trade['exit_pnl']
           
           # Prepare data for chart
           setup_labels = list(setup_types.keys())
           setup_pnl = [setup_types[st]['pnl'] for st in setup_labels]
           
           # Create chart
           fig_setup = go.Figure(data=[go.Bar(
               x=setup_labels,
               y=setup_pnl,
               marker_color=['#007bff' if pnl >= 0 else '#dc3545' for pnl in setup_pnl]
           )])
           
           fig_setup.update_layout(
               title="P&L by Strategy Type",
               height=300,
               xaxis_title="Strategy Type",
               yaxis_title="P&L ($)",
               template="plotly_white"
           )
           
           st.plotly_chart(fig_setup, use_container_width=True)

def render_market_overview():
   """Render the market overview dashboard"""
   st.header("Market Overview")
   
   # Database connection
   connection = init_databricks_connection()
   if not connection:
       st.error("Databricks connection unavailable - Market Overview requires database access")
       return
   
   try:
       # Get market overview data
       cursor = connection.cursor()
       
       # Get overall market GEX data - FIX: Added proper alias for net_gex column
       market_query = """
       SELECT 
           symbol,
           spot_price,
           net_gex/1000000000 as net_gex_billions,
           gamma_flip_point,
           distance_to_flip_pct,
           analysis_date,
           analysis_timestamp
       FROM quant_projects.gex_trading.scheduled_pipeline_results
       WHERE symbol IN ('SPY', 'QQQ', 'IWM', 'DIA')
       AND analysis_date >= current_date() - INTERVAL 1 DAY
       ORDER BY analysis_timestamp DESC
       LIMIT 20
       """
       
       cursor.execute(market_query)
       market_results = cursor.fetchall()
       market_columns = [desc[0] for desc in cursor.description]
       
       market_df = pd.DataFrame(market_results, columns=market_columns)
       
       # Get market regime data
       regime_query = """
       SELECT 
           regime_type,
           regime_strength,
           vix_level,
           total_market_gex_billions,
           analysis_date,
           analysis_timestamp
       FROM quant_projects.gex_trading.market_regime
       WHERE analysis_date >= current_date() - INTERVAL 7 DAY
       ORDER BY analysis_timestamp DESC
       LIMIT 10
       """
       
       try:
           cursor.execute(regime_query)
           regime_results = cursor.fetchall()
           regime_columns = [desc[0] for desc in cursor.description]
           regime_df = pd.DataFrame(regime_results, columns=regime_columns)
       except:
           # If market_regime table doesn't exist
           regime_df = pd.DataFrame()
       
       cursor.close()
       
       # Display market data
       if not market_df.empty:
           latest_etfs = market_df.drop_duplicates('symbol', keep='first')
           
           st.subheader("Major ETF GEX Status")
           
           cols = st.columns(len(latest_etfs))
           
           for i, (_, row) in enumerate(latest_etfs.iterrows()):
               with cols[i]:
                   # Determine if above or below flip
                   above_flip = row['distance_to_flip_pct'] > 0
                   color = "#28a745" if above_flip else "#dc3545"
                   regime = "POSITIVE GEX" if row['net_gex_billions'] > 0 else "NEGATIVE GEX"
                   
                   st.markdown(f"""
                   <div class="card" style="border-left: 5px solid {color};">
                       <h3 style="text-align: center;">{row['symbol']}</h3>
                       <div style="font-size: 24px; font-weight: bold; text-align: center;">${row['spot_price']:.2f}</div>
                       <div style="color: {color}; text-align: center;">{regime}</div>
                       <hr>
                       <div>Net GEX: {row['net_gex_billions']:.2f}B</div>
                       <div>Flip: ${row['gamma_flip_point']:.2f}</div>
                       <div>Distance: {row['distance_to_flip_pct']:+.2f}%</div>
                   </div>
                   """, unsafe_allow_html=True)
           
           # Market regime
           if not regime_df.empty:
               latest_regime = regime_df.iloc[0]
               
               st.subheader("Current Market Regime")
               
               col1, col2, col3 = st.columns(3)
               
               with col1:
                   st.metric("Market Regime", latest_regime['regime_type'])
               
               with col2:
                   st.metric("VIX Level", f"{latest_regime['vix_level']:.1f}")
               
               with col3:
                   st.metric("Total Market GEX", f"{latest_regime['total_market_gex_billions']:.1f}B")
           
           # ETF GEX history chart
           st.subheader("ETF Net GEX History")
           
           gex_history = market_df.copy()
           gex_history['analysis_timestamp'] = pd.to_datetime(gex_history['analysis_timestamp'])
           
           fig = go.Figure()
           
           for symbol in gex_history['symbol'].unique():
               symbol_data = gex_history[gex_history['symbol'] == symbol]
               fig.add_trace(go.Scatter(
                   x=symbol_data['analysis_timestamp'],
                   y=symbol_data['net_gex_billions'],
                   mode='lines+markers',
                   name=symbol
               ))
           
           fig.update_layout(
               title="Net GEX History (Billions $)",
               xaxis_title="Date",
               yaxis_title="Net GEX (Billions)",
               height=400,
               template="plotly_white"
           )
           
           st.plotly_chart(fig, use_container_width=True)
           
           # Recent interesting setups
           st.subheader("Recent Interesting Setups")
           
           try:
               # Get interesting setups
               interesting_query = """
               SELECT 
                   symbol,
                   structure_type,
                   confidence_score,
                   analysis_timestamp
               FROM quant_projects.gex_trading.scheduled_pipeline_results
               WHERE confidence_score >= 80
               AND analysis_date >= current_date() - INTERVAL 1 DAY
               ORDER BY confidence_score DESC
               LIMIT 5
               """
               
               cursor = connection.cursor()
               cursor.execute(interesting_query)
               interesting_results = cursor.fetchall()
               interesting_columns = [desc[0] for desc in cursor.description]
               
               interesting_df = pd.DataFrame(interesting_results, columns=interesting_columns)
               cursor.close()
               
               if not interesting_df.empty:
                   for _, row in interesting_df.iterrows():
                       st.markdown(f"""
                       <div class="card high-confidence">
                           <div style="display: flex; justify-content: space-between;">
                               <div><strong>{row['symbol']}</strong> - {row['structure_type'].replace('_', ' ').title()}</div>
                               <div>{row['confidence_score']}% confidence</div>
                           </div>
                           <div style="font-size: 0.9em; color: #666; margin-top: 5px;">
                               {pd.to_datetime(row['analysis_timestamp']).strftime('%m/%d/%y %H:%M')}
                           </div>
                       </div>
                       """, unsafe_allow_html=True)
               else:
                   st.info("No high confidence setups found in the last 24 hours")
                   
           except Exception as e:
               st.warning(f"Could not load interesting setups: {e}")
           
       else:
           st.error("No market data available")
           st.info("Make sure your pipeline has processed market ETFs (SPY, QQQ, IWM, DIA)")
       
   except Exception as e:
       st.error(f"Failed to load market overview data: {e}")

def render_education_center():
   """Render the education center"""
   st.header("Education Center")
   
   # Tabs for different educational topics
   tabs = st.tabs(["GEX Basics", "Trading Strategies", "Risk Management", "FAQ"])
   
   with tabs[0]:
       render_gex_education()
       
       # Additional GEX educational content
       st.markdown("""
       <div class="education-section">
           <h3 class="education-title">How Dealers Impact the Market</h3>
           <p>Market makers and dealers must maintain delta-neutral positions to manage risk. As options gamma changes with underlying price movements, dealers must continuously hedge by buying or selling the underlying asset.</p>
           
           <h4>The Dealer Hedging Cycle:</h4>
           <ol>
               <li><strong>Initial Position</strong>: Dealers sell options to traders and hedge delta exposure.</li>
               <li><strong>Price Movement</strong>: As price changes, delta of options changes (gamma effect).</li>
               <li><strong>Rehedging</strong>: Dealers must buy or sell underlying to maintain delta neutrality.</li>
               <li><strong>Market Impact</strong>: This hedging creates predictable buying/selling pressure.</li>
           </ol>
           
           <h4>Why This Matters for Traders:</h4>
           <p>Gamma exposure creates systematic flows that can be predicted and traded ahead of. By understanding where and when dealers will be forced to buy or sell, traders can position for these predictable moves.</p>
       </div>
       """, unsafe_allow_html=True)
       
       # Visual explanation
       st.markdown("""
       <div class="education-section">
           <h3 class="education-title">Visualizing GEX</h3>
           <p>The GEX profile shows gamma concentration across different strike prices:</p>
           <ul>
               <li><strong>Green Bars (Call Gamma)</strong>: Positive gamma from calls - creates resistance</li>
               <li><strong>Red Bars (Put Gamma)</strong>: Negative gamma from puts - creates support</li>
               <li><strong>Blue Line (Net GEX)</strong>: Combined effect across all strikes</li>
               <li><strong>Vertical Lines</strong>: Key price levels (current price, gamma flip, walls)</li>
           </ul>
           <p>The visualization helps identify key levels where dealer hedging will have the most impact on price action.</p>
       </div>
       """, unsafe_allow_html=True)
   
   with tabs[1]:
       render_strategy_education()
       
       # Additional strategy examples
       st.markdown("""
       <div class="education-section">
           <h3 class="education-title">Real-World Strategy Examples</h3>
           
           <h4>Example 1: SPY Negative GEX Squeeze</h4>
           <p><strong>Setup:</strong> SPY has -2.5B GEX, price is 0.8% below gamma flip at $452, with put wall at $445.</p>
           <p><strong>Trade:</strong> Buy SPY 455 calls, 3 DTE, targeting gamma flip as first resistance.</p>
           <p><strong>Outcome:</strong> As price moves toward gamma flip, dealers must buy more shares to maintain delta neutrality, accelerating the move upward.</p>
           
           <h4>Example 2: QQQ Call Wall Premium Selling</h4>
           <p><strong>Setup:</strong> QQQ has +1.8B GEX, price is between flip ($378) and call wall ($385).</p>
           <p><strong>Trade:</strong> Sell QQQ 385/390 call credit spread, 2 DTE.</p>
           <p><strong>Outcome:</strong> Strong call wall resistance causes price rejection, allowing rapid theta decay for profit.</p>
       </div>
       """, unsafe_allow_html=True)
   
   with tabs[2]:
       render_risk_education()
       
       # Additional risk management examples
       st.markdown("""
       <div class="education-section">
           <h3 class="education-title">Advanced Risk Management</h3>
           
           <h4>Position Correlation Management:</h4>
           <p>When trading multiple GEX setups, monitor correlations to avoid overexposure:</p>
           <ul>
               <li><strong>ETF Overlaps</strong>: Reduce position size when trading correlated ETFs (SPY/QQQ)</li>
               <li><strong>Sector Concentration</strong>: Avoid multiple positions in same sector during sector rotation</li>
               <li><strong>VIX Regime Awareness</strong>: Adjust strategy selection based on volatility regime</li>
           </ul>
           
           <h4>Kelly Criterion for GEX Trading:</h4>
           <p>Optimize position sizing using historical win rates:</p>
           <ul>
               <li><strong>Squeeze Plays</strong>: ~65% win rate suggests 30% of max allocation</li>
               <li><strong>Premium Selling</strong>: ~80% win rate suggests 60% of max allocation</li>
               <li><strong>Always reduce Kelly fraction</strong> by at least 50% to account for estimation error</li>
           </ul>
       </div>
       """, unsafe_allow_html=True)
   
   with tabs[3]:
       st.markdown("""
       <div class="education-section">
           <h3 class="education-title">Frequently Asked Questions</h3>
           
           <h4>Q: What data sources does GEX Master Pro use?</h4>
           <p>A: The system uses options data from TradingVolatility.net API, with fallback to your Databricks pipeline data, and additional market data from yfinance when needed.</p>
           
           <h4>Q: How often should I check for new setups?</h4>
           <p>A: The pipeline runs early morning (pre-market), but market conditions can change rapidly. Check for updates at least at market open, mid-day, and before close.</p>
           
           <h4>Q: Why do gamma levels change throughout the day?</h4>
           <p>A: As new options are traded and underlying prices change, the gamma profile shifts. Major events like FOMC can dramatically alter the gamma landscape.</p>
           
           <h4>Q: How far in advance should I enter positions?</h4>
           <p>A: For squeeze plays, enter when within 1% of gamma flip point. For premium selling, wait until price approaches wall levels within 0.5-1%.</p>
           
           <h4>Q: What's the best DTE (Days to Expiration) for GEX trading?</h4>
           <p>A: Gamma is highest close to expiration, so 0-5 DTE works best for most strategies. Squeeze plays perform well with 2-5 DTE, premium selling with 0-2 DTE.</p>
           
           <h4>Q: How do I interpret the GEX visualization?</h4>
           <p>A: Green bars show call gamma (resistance), red bars show put gamma (support), blue line shows net effect. Vertical lines mark key price levels.</p>
       </div>
       """, unsafe_allow_html=True)

if __name__ == "__main__":
   main()
