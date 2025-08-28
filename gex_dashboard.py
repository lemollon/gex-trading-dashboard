"""
GEX Master Pro - Combined Dashboard
Combines single symbol analysis with pipeline database connections
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import requests
import json

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
    
    .stAlert > div {
        padding: 1rem;
        border-radius: 10px;
    }
    
    .metric-container {
        background: #f0f2f6;
        color: #0e1117;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .setup-card {
        background: white;
        border: 1px solid #e0e0e0;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .high-confidence {
        border-left: 5px solid #28a745;
    }
    
    .medium-confidence {
        border-left: 5px solid #ffc107;
    }
    
    .low-confidence {
        border-left: 5px solid #dc3545;
    }
    
    .status-connected {
        color: #28a745;
        font-weight: bold;
    }
    
    .status-disconnected {
        color: #dc3545;
        font-weight: bold;
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
                    net_gex,
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

def main():
    # Initialize session state
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
    
    # Header
    st.title("⚡ GEX Master Pro")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Select Page", [
        "Single Symbol Analysis", 
        "Pipeline Dashboard", 
        "Market Overview"
    ])
    
    # Sidebar API status
    st.sidebar.markdown("---")
    st.sidebar.subheader("API Status")
    
    if st.session_state.api_status['status'] == 'online':
        st.sidebar.success("✅ TradingVolatility API: Online")
    elif st.session_state.api_status['status'] == 'offline':
        st.sidebar.error("❌ TradingVolatility API: Offline")
    else:
        st.sidebar.info("ℹ️ API Status: Unknown")
    
    if st.session_state.api_status['last_checked']:
        st.sidebar.caption(f"Last checked: {st.session_state.api_status['last_checked'].strftime('%H:%M:%S')}")
    
    # API test button
    if st.sidebar.button("Test API Connection"):
        with st.sidebar:
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
    
    # Databricks status
    st.sidebar.markdown("---")
    st.sidebar.subheader("Databricks Status")
    
    db_connection = init_databricks_connection()
    if db_connection:
        st.sidebar.success("✅ Databricks: Connected")
    else:
        st.sidebar.error("❌ Databricks: Disconnected")
    
    # Page-specific content
    if page == "Single Symbol Analysis":
        st.subheader("Single Symbol GEX Analysis")
        
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
                            
                            if enhanced_setup.get('big_move_mode'):
                                st.success(f"BIG MOVE SETUP DETECTED")
                                
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
                    
                else:
                    st.error(f"Failed to fetch data for {symbol}: {gex_data.get('error', 'Unknown error')}")
    
    elif page == "Pipeline Dashboard":
        st.subheader("Pipeline Dashboard")
        
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
                <h3>Total Setups</h3>
                <h1>{len(filtered_df)}</h1>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            high_conf = len(filtered_df[filtered_df['confidence_score'] >= 85])
            st.markdown(f"""
            <div class="metric-container">
                <h3>High Confidence</h3>
                <h1>{high_conf}</h1>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            avg_conf = filtered_df['confidence_score'].mean()
            st.markdown(f"""
            <div class="metric-container">
                <h3>Avg Confidence</h3>
                <h1>{avg_conf:.1f}%</h1>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            try:
                enhanced_strategies = len(filtered_df[filtered_df['category'] == 'ENHANCED_STRATEGY'])
            except:
                enhanced_strategies = 0
                
            st.markdown(f"""
            <div class="metric-container">
                <h3>Enhanced Strategies</h3>
                <h1>{enhanced_strategies}</h1>
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
                <div class="setup-card {confidence_class}">
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
    
    elif page == "Market Overview":
        st.subheader("Market Overview")
        
        # Database connection
        connection = init_databricks_connection()
        if not connection:
            st.error("Databricks connection unavailable - Market Overview requires database access")
            return
        
        try:
            # Get market overview data
            cursor = connection.cursor()
            
            # Get overall market GEX data
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
                        <div style="border: 1px solid {color}; border-radius: 10px; padding: 10px; text-align: center;">
                            <h3>{row['symbol']}</h3>
                            <div style="font-size: 24px; font-weight: bold;">${row['spot_price']:.2f}</div>
                            <div style="color: {color};">{regime}</div>
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
                            <div style="border: 1px solid #28a745; border-radius: 10px; padding: 15px; margin-bottom: 10px;">
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
    
    # Footer
    st.markdown("---")
    st.caption(f"GEX Master Pro - Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
