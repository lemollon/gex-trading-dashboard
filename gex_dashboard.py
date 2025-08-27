# Initialize session state with missing features
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.auto_refresh = False
    st.session_state.alert_settings = {
        'high_priority': True,
        'medium_priority': True,
        'low_priority': False
    }
    st.session_state.last_update = datetime.now()
    
    # Enhanced Portfolio Data with 100+ trades tracking
    st.session_state.portfolio_data = {
        'total_value': 125847.50,
        'day_pnl': 2847.33,
        'positions': 7,
        'win_rate': 73.2,
        'sharpe_ratio': 1.87,
        'max_drawdown': -4.2,
        'total_trades': 147,  # 100+ trades count
        'winning_trades': 108,
        'losing_trades': 39,
        'avg_win': 485.23,
        'avg_loss':import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
import requests
import json
from typing import Dict, List, Tuple, Optional
import yfinance as yf
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="🚀 GEX Trading Command Center",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",  # Force sidebar to be open
    menu_items={
        'Get Help': 'https://github.com/your-repo/gex-dashboard',
        'Report a bug': "https://github.com/your-repo/gex-dashboard/issues",
        'About': "Professional Gamma Exposure Trading Platform v2.0"
    }
)

# Enhanced Custom CSS with animations and professional styling
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Global Styles */
    .main {
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Force sidebar visibility */
    .css-1d391kg, .css-1cypcdb, .css-17eq0hr {
        display: block !important;
        visibility: visible !important;
    }
    
    /* Sidebar toggle button styling */
    button[title="View fullscreen"] {
        display: none !important;
    }
    
    /* Main container styling - Fixed dark background with readable text */
    .stApp {
        background: linear-gradient(135deg, #1a202c 0%, #2d3748 25%, #4a5568 50%, #2d3748 75%, #1a202c 100%);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
        color: #ffffff !important;
    }
    
    /* Ensure all text is readable */
    .stApp, .stApp * {
        color: #ffffff !important;
    }
    
    /* Override Streamlit's default text colors */
    .stMarkdown, .stMarkdown *, .stText, .stText *, p, span, div {
        color: #ffffff !important;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Animated background particles */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            radial-gradient(2px 2px at 20px 30px, rgba(59, 130, 246, 0.3), transparent),
            radial-gradient(2px 2px at 40px 70px, rgba(168, 85, 247, 0.3), transparent),
            radial-gradient(1px 1px at 90px 40px, rgba(34, 197, 94, 0.3), transparent),
            radial-gradient(1px 1px at 130px 80px, rgba(249, 115, 22, 0.3), transparent);
        background-size: 200px 200px;
        animation: float 20s ease-in-out infinite;
        pointer-events: none;
        z-index: 1;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px) translateX(0px); }
        33% { transform: translateY(-10px) translateX(5px); }
        66% { transform: translateY(5px) translateX(-5px); }
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #3b82f6, #8b5cf6, #06b6d4);
        background-size: 200% 200%;
        animation: headerGradient 3s ease infinite;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3.5rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.5rem;
        text-shadow: 0 0 30px rgba(59, 130, 246, 0.5);
    }
    
    @keyframes headerGradient {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    .sub-header {
        text-align: center;
        color: #94a3b8;
        font-size: 1.2rem;
        font-weight: 300;
        margin-bottom: 2rem;
        opacity: 0.8;
    }
    
    /* Enhanced Metric Cards - Better contrast */
    .metric-card {
        background: rgba(45, 55, 72, 0.95);
        backdrop-filter: blur(15px);
        border: 2px solid rgba(96, 165, 250, 0.4);
        border-radius: 16px;
        padding: 24px;
        margin: 8px;
        box-shadow: 0 12px 36px rgba(0, 0, 0, 0.5);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
        color: #ffffff !important;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(59, 130, 246, 0.1), transparent);
        transition: left 0.5s;
    }
    
    .metric-card:hover::before {
        left: 100%;
    }
    
    .metric-card:hover {
        transform: translateY(-8px) scale(1.02);
        border-color: rgba(96, 165, 250, 0.6);
        box-shadow: 0 20px 50px rgba(59, 130, 246, 0.3);
    }
    
    .metric-value {
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 8px;
        background: linear-gradient(135deg, #ffffff, #60a5fa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    .metric-label {
        color: #cbd5e1 !important;
        font-size: 1rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 4px;
    }
    
    .metric-change {
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: 4px;
        padding: 2px 8px;
        border-radius: 12px;
    }
    
    .positive { 
        color: #10b981; 
        background: rgba(16, 185, 129, 0.1);
    }
    .negative { 
        color: #ef4444; 
        background: rgba(239, 68, 68, 0.1);
    }
    .neutral { 
        color: #f59e0b; 
        background: rgba(245, 158, 11, 0.1);
    }
    
    /* Enhanced Alert System - Better visibility */
    .alert-container {
        background: rgba(45, 55, 72, 0.95);
        border: 2px solid rgba(239, 68, 68, 0.5);
        border-radius: 12px;
        padding: 20px;
        margin: 16px 0;
        animation: alertPulse 2s ease-in-out infinite;
        color: #ffffff !important;
        backdrop-filter: blur(10px);
    }
    
    @keyframes alertPulse {
        0%, 100% { border-color: rgba(239, 68, 68, 0.5); }
        50% { border-color: rgba(239, 68, 68, 0.8); }
    }
    
    .alert-high { 
        border-color: #ef4444; 
        background: rgba(239, 68, 68, 0.2); 
        color: #ffffff !important;
    }
    .alert-medium { 
        border-color: #f59e0b; 
        background: rgba(245, 158, 11, 0.2); 
        color: #ffffff !important;
    }
    .alert-low { 
        border-color: #10b981; 
        background: rgba(16, 185, 129, 0.2); 
        color: #ffffff !important;
    }
    
    /* Setup Cards - Improved readability */
    .setup-card {
        background: rgba(45, 55, 72, 0.95);
        backdrop-filter: blur(20px);
        border: 2px solid rgba(96, 165, 250, 0.4);
        border-radius: 20px;
        padding: 32px;
        margin: 20px 0;
        box-shadow: 0 16px 56px rgba(0, 0, 0, 0.5);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
        color: #ffffff !important;
    }
    
    .setup-card:hover {
        transform: translateY(-12px);
        border-color: rgba(96, 165, 250, 0.7);
        box-shadow: 0 28px 72px rgba(59, 130, 246, 0.4);
    }
    
    /* Progress Bars */
    .progress-container {
        background: rgba(30, 41, 59, 0.5);
        border-radius: 12px;
        padding: 2px;
        margin: 8px 0;
        overflow: hidden;
    }
    
    .progress-bar {
        height: 8px;
        border-radius: 10px;
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
        transition: width 1s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .progress-bar::after {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
        animation: progressShimmer 1.5s ease-in-out infinite;
    }
    
    @keyframes progressShimmer {
        0% { left: -100%; }
        100% { left: 100%; }
    }
    
    /* Button Enhancements - Consistent styling */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6, #60a5fa) !important;
        border: 2px solid rgba(96, 165, 250, 0.5) !important;
        border-radius: 12px !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 14px 28px !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.3) !important;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.2) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 12px 28px rgba(59, 130, 246, 0.4) !important;
        background: linear-gradient(135deg, #2563eb, #3b82f6) !important;
        border-color: rgba(96, 165, 250, 0.8) !important;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(45, 55, 72, 0.5);
        border-radius: 12px;
        padding: 8px;
        border: 1px solid rgba(96, 165, 250, 0.3);
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 12px 24px;
        background: rgba(59, 130, 246, 0.1);
        border-radius: 8px;
        color: #cbd5e1 !important;
        border: 1px solid transparent;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #3b82f6, #60a5fa) !important;
        color: #ffffff !important;
        border-color: rgba(96, 165, 250, 0.5);
        box-shadow: 0 4px 16px rgba(59, 130, 246, 0.3);
    }
    
    /* Loading Animation */
    .loading-spinner {
        border: 3px solid rgba(96, 165, 250, 0.3);
        border-top: 3px solid #60a5fa;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
        margin: 20px auto;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .execute-button {
        background: linear-gradient(135deg, #10b981, #059669) !important;
        box-shadow: 0 4px 16px rgba(16, 185, 129, 0.3) !important;
    }
    
    .execute-button:hover {
        background: linear-gradient(135deg, #059669, #047857) !important;
        box-shadow: 0 8px 24px rgba(16, 185, 129, 0.4) !important;
    }
    
    /* Mobile Responsiveness */
    @media (max-width: 768px) {
        .main-header { font-size: 2.5rem; }
        .metric-card { padding: 16px; margin: 4px; }
        .setup-card { padding: 20px; margin: 8px 0; }
        .metric-value { font-size: 2rem; }
    }
    
    @media (max-width: 480px) {
        .main-header { font-size: 2rem; }
        .metric-value { font-size: 1.8rem; }
        .setup-card { padding: 16px; }
    }
    
    /* Sidebar Enhancements - Cohesive design */
    .css-1d391kg, .css-1cypcdb, .css-17eq0hr, .css-6qob1r, .css-1aumxhk {
        background: linear-gradient(180deg, #2d3748 0%, #4a5568 50%, #2d3748 100%) !important;
        backdrop-filter: blur(10px) !important;
        border-right: 2px solid rgba(59, 130, 246, 0.3) !important;
        color: #ffffff !important;
    }
    
    /* Sidebar content styling */
    .css-1d391kg *, .css-1cypcdb *, .css-17eq0hr *, .css-6qob1r *, .css-1aumxhk * {
        color: #ffffff !important;
    }
    
    /* Sidebar headers */
    .css-1d391kg h3, .css-1cypcdb h3, .css-17eq0hr h3 {
        color: #60a5fa !important;
        border-bottom: 1px solid rgba(59, 130, 246, 0.3);
        padding-bottom: 8px;
        margin-bottom: 16px;
    }
    
    /* Sidebar selectbox and inputs */
    .css-1d391kg .stSelectbox, .css-1d391kg .stSlider, .css-1d391kg .stCheckbox {
        background: rgba(59, 130, 246, 0.1) !important;
        border-radius: 8px;
        padding: 4px;
        margin: 4px 0;
    }
    
    /* Sidebar selectbox labels */
    .css-1d391kg .stSelectbox label, .css-1d391kg .stSlider label, .css-1d391kg .stCheckbox label {
        color: #cbd5e1 !important;
        font-weight: 500;
    }
    
    /* Ensure sidebar toggle button is visible */
    .css-14xtw13.e8zbici0 {
        background: #3b82f6 !important;
        color: white !important;
        border-radius: 8px;
        border: none !important;
    }
    
    /* Chart Container */
    .chart-container {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 24px;
        margin: 16px 0;
        border: 1px solid rgba(59, 130, 246, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    /* Loading Animation */
    .loading-spinner {
        border: 3px solid rgba(59, 130, 246, 0.3);
        border-top: 3px solid #3b82f6;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
        margin: 20px auto;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.auto_refresh = False
    st.session_state.alert_settings = {
        'high_priority': True,
        'medium_priority': True,
        'low_priority': False
    }
    st.session_state.last_update = datetime.now()
    st.session_state.portfolio_data = {
        'total_value': 125847.50,
        'day_pnl': 2847.33,
        'positions': 7,
        'win_rate': 73.2,
        'sharpe_ratio': 1.87,
        'max_drawdown': -4.2
    }

class GEXDataManager:
    """Enhanced data manager with advanced calculations and custom symbol support"""
    
    def __init__(self):
        self.symbols = ['SPY', 'QQQ', 'IWM', 'DIA', 'TSLA', 'AAPL', 'MSFT', 'NVDA']
        self.custom_symbols = set()  # Track custom symbols
        self.data = {}
        self.refresh_data()
    
    def add_custom_symbol(self, symbol: str) -> bool:
        """Add a custom symbol and generate data for it"""
        symbol = symbol.upper().strip()
        
        # Basic validation
        if len(symbol) < 1 or len(symbol) > 5:
            return False
        
        # Check if symbol contains only letters
        if not symbol.isalpha():
            return False
        
        # Add to custom symbols and generate data
        self.custom_symbols.add(symbol)
        if symbol not in self.symbols:
            self.symbols.append(symbol)
        
        # Generate data for the new symbol
        self.data[symbol] = self._generate_symbol_data(symbol)
        return True
    
    def get_symbol_info(self, symbol: str) -> Dict:
        """Get basic symbol information (mock for now, could integrate with real API)"""
        symbol_info = {
            'SPY': {'name': 'SPDR S&P 500 ETF', 'sector': 'ETF', 'market_cap': '500B+'},
            'QQQ': {'name': 'Invesco QQQ ETF', 'sector': 'ETF', 'market_cap': '200B+'},
            'IWM': {'name': 'iShares Russell 2000 ETF', 'sector': 'ETF', 'market_cap': '50B+'},
            'DIA': {'name': 'SPDR Dow Jones Industrial Average ETF', 'sector': 'ETF', 'market_cap': '30B+'},
            'TSLA': {'name': 'Tesla Inc', 'sector': 'Automotive', 'market_cap': '800B+'},
            'AAPL': {'name': 'Apple Inc', 'sector': 'Technology', 'market_cap': '3T+'},
            'MSFT': {'name': 'Microsoft Corporation', 'sector': 'Technology', 'market_cap': '2.5T+'},
            'NVDA': {'name': 'NVIDIA Corporation', 'sector': 'Technology', 'market_cap': '1.5T+'}
        }
        
        return symbol_info.get(symbol, {
            'name': f'{symbol} Corporation',
            'sector': 'Unknown',
            'market_cap': 'N/A'
        })
    
    def refresh_data(self):
        """Generate enhanced mock data with realistic GEX calculations"""
        self.data = {}
        for symbol in self.symbols:
            self.data[symbol] = self._generate_symbol_data(symbol)
    
    def _generate_symbol_data(self, symbol: str) -> Dict:
        """Generate realistic GEX data for a symbol"""
        # Use symbol hash for consistent data per symbol
        np.random.seed(hash(symbol) % 1000)  
        
        # Base price ranges per symbol type
        if symbol in ['SPY', 'QQQ', 'IWM', 'DIA']:
            # ETF price ranges
            price_ranges = {
                'SPY': (440, 460), 'QQQ': (370, 390), 
                'IWM': (180, 200), 'DIA': (340, 360)
            }
            base_price = np.random.uniform(*price_ranges.get(symbol, (100, 200)))
        elif symbol in ['TSLA', 'AAPL', 'MSFT', 'NVDA']:
            # Popular stock ranges
            price_ranges = {
                'TSLA': (180, 220), 'AAPL': (150, 180),
                'MSFT': (380, 420), 'NVDA': (800, 900)
            }
            base_price = np.random.uniform(*price_ranges.get(symbol, (50, 300)))
        else:
            # Generic stock range for custom symbols
            base_price = np.random.uniform(20, 500)
        
        # Generate more realistic strike ladder based on price
        if base_price < 50:
            strike_increment = 0.5
        elif base_price < 200:
            strike_increment = 1.0
        else:
            strike_increment = 5.0
            
        strikes = np.arange(
            base_price * 0.85, 
            base_price * 1.15, 
            strike_increment
        )
        strikes = np.round(strikes / strike_increment) * strike_increment
        
        profiles = []
        net_gex = 0
        gamma_flip = None
        cumulative_gex = 0
        
        for strike in strikes:
            # Generate realistic OI and gamma patterns
            distance_from_atm = abs(strike - base_price) / base_price
            oi_multiplier = np.exp(-25 * distance_from_atm)  # Higher OI near ATM
            
            # Volatility factor based on symbol type
            vol_factor = 1.5 if symbol in ['TSLA', 'NVDA'] else 1.0
            
            if strike < base_price:  # Put strikes
                oi = int(np.random.exponential(15000 * vol_factor) * oi_multiplier)
                gamma_strength = np.random.uniform(100, 2000 * vol_factor)
                gamma = -gamma_strength  # Puts have negative gamma for dealers
                gex = gamma * oi * 0.01 * strike / 100  # Scaled for readability
                profile_type = 'put'
                
            elif strike > base_price:  # Call strikes  
                oi = int(np.random.exponential(12000 * vol_factor) * oi_multiplier)
                gamma_strength = np.random.uniform(100, 1800 * vol_factor)
                gamma = gamma_strength  # Calls have positive gamma for dealers
                gex = gamma * oi * 0.01 * strike / 100  # Scaled for readability
                profile_type = 'call'
                
            else:  # ATM (closest strike)
                oi = int(np.random.uniform(20000, 50000))
                gamma = 0
                gex = 0
                profile_type = 'flip'
            
            cumulative_gex += gex
            net_gex += gex
            
            # Find gamma flip point (where cumulative crosses zero)
            if gamma_flip is None and len(profiles) > 0:
                if (profiles[-1]['cumulative'] <= 0 and cumulative_gex > 0) or \
                   (profiles[-1]['cumulative'] >= 0 and cumulative_gex < 0):
                    gamma_flip = strike
            
            profiles.append({
                'strike': round(strike, 2),
                'gex': round(gex / 1e6, 2),  # Convert to millions for display
                'cumulative': round(cumulative_gex / 1e6, 2),
                'oi': max(1, oi),  # Ensure minimum OI
                'gamma': round(gamma, 2),
                'type': profile_type
            })
        
        # Set gamma flip to middle strike if not found
        if gamma_flip is None:
            gamma_flip = base_price
        
        # Identify key levels
        put_walls = [p for p in profiles if p['type'] == 'put' and p['gex'] < -0.1]
        call_walls = [p for p in profiles if p['type'] == 'call' and p['gex'] > 0.1]
        
        put_walls = sorted(put_walls, key=lambda x: abs(x['gex']), reverse=True)[:3]
        call_walls = sorted(call_walls, key=lambda x: x['gex'], reverse=True)[:3]
        
        # Determine regime and confidence
        regime = 'Negative Gamma' if net_gex < 0 else 'Positive Gamma'
        volatility_mode = 'Amplification' if net_gex < 0 else 'Suppression'
        
        # Enhanced confidence calculation
        wall_strength = sum(abs(p['gex']) for p in put_walls + call_walls)
        distance_to_flip = abs(base_price - gamma_flip) / base_price
        gex_magnitude = abs(net_gex / 1e9)  # Billions
        
        confidence = min(95, max(20, 
            30 +  # Base confidence
            wall_strength * 8 +  # Wall strength factor
            (1 - distance_to_flip) * 25 +  # Distance to flip factor
            gex_magnitude * 15  # GEX magnitude factor
        ))
        
        return {
            'symbol': symbol,
            'price': round(base_price, 2),
            'net_gex': round(net_gex / 1e9, 2),  # Billions
            'gamma_flip': round(gamma_flip, 2),
            'call_wall': call_walls[0]['strike'] if call_walls else round(base_price * 1.05, 2),
            'put_support': put_walls[0]['strike'] if put_walls else round(base_price * 0.95, 2),
            'regime': regime,
            'volatility_mode': volatility_mode,
            'confidence': round(confidence, 1),
            'profiles': profiles,
            'put_walls': put_walls,
            'call_walls': call_walls,
            'setups': self._generate_setups(symbol, net_gex, base_price, gamma_flip, confidence),
            'symbol_info': self.get_symbol_info(symbol)
        }
    
    def _generate_setups(self, symbol: str, net_gex: float, price: float, 
                        gamma_flip: float, confidence: float) -> List[Dict]:
        """Generate trading setups based on GEX conditions"""
        setups = []
        
        # Negative GEX Squeeze Setup
        if net_gex < -1e9:  # Less than -1B
            setups.append({
                'type': 'Negative GEX Squeeze',
                'strategy': 'Long Calls',
                'strikes': f'{price:.0f}-{price * 1.02:.0f}',
                'confidence': min(95, confidence + 10),
                'risk': 'High',
                'reward': '150-300%',
                'timeframe': '2-5 DTE',
                'description': f'Strong negative GEX ({net_gex:.1f}B) creates squeeze potential',
                'entry_conditions': [
                    f'Price below gamma flip ({gamma_flip:.2f})',
                    'Strong put support confirmed',
                    'Low IV environment preferred'
                ],
                'exit_strategy': '100% profit target or 50% stop loss'
            })
        
        # Positive GEX Iron Condor
        elif net_gex > 1e9:  # Greater than 1B
            setups.append({
                'type': 'Iron Condor Setup',
                'strategy': 'Sell Premium',
                'strikes': f'{price * 0.97:.0f}P/{price * 1.03:.0f}C',
                'confidence': min(90, confidence + 5),
                'risk': 'Medium',
                'reward': '20-40%',
                'timeframe': '5-10 DTE',
                'description': f'Strong positive GEX ({net_gex:.1f}B) supports range-bound trading',
                'entry_conditions': [
                    'Wide gamma walls confirmed',
                    'High IV rank preferred',
                    'Major expiry > 3 days away'
                ],
                'exit_strategy': '50% profit target or manage gamma risk'
            })
        
        # Gamma Wall Rejection
        call_wall_distance = abs(price - (price * 1.05)) / price  # Mock call wall
        if call_wall_distance < 0.01:  # Within 1% of call wall
            setups.append({
                'type': 'Call Wall Rejection',
                'strategy': 'Long Puts',
                'strikes': f'{price * 0.99:.0f}-{price * 0.97:.0f}',
                'confidence': min(85, confidence),
                'risk': 'Medium',
                'reward': '75-150%',
                'timeframe': '1-3 DTE',
                'description': 'Price approaching strong call wall resistance',
                'entry_conditions': [
                    'Multiple failed breakout attempts',
                    'High dealer gamma at resistance',
                    'Volume confirmation preferred'
                ],
                'exit_strategy': 'Target put support or 50% stop'
            })
        
        return setups

class AlertSystem:
    """Advanced alert system with priority levels"""
    
    def __init__(self):
        self.alerts = []
    
    def check_alerts(self, symbol_data: Dict) -> List[Dict]:
        """Check for various alert conditions"""
        alerts = []
        
        # High Priority Alerts
        if abs(symbol_data['net_gex']) > 2:
            alerts.append({
                'priority': 'HIGH',
                'type': 'Extreme GEX',
                'message': f"Extreme GEX level: {symbol_data['net_gex']}B",
                'symbol': symbol_data['symbol'],
                'timestamp': datetime.now(),
                'action': 'Monitor for squeeze setup'
            })
        
        # Gamma flip proximity
        flip_distance = abs(symbol_data['price'] - symbol_data['gamma_flip']) / symbol_data['price']
        if flip_distance < 0.005:  # Within 0.5%
            alerts.append({
                'priority': 'HIGH', 
                'type': 'Gamma Flip Proximity',
                'message': f"Price within 0.5% of gamma flip ({symbol_data['gamma_flip']})",
                'symbol': symbol_data['symbol'],
                'timestamp': datetime.now(),
                'action': 'Prepare for regime change'
            })
        
        # Medium Priority Alerts  
        if symbol_data['confidence'] > 85:
            alerts.append({
                'priority': 'MEDIUM',
                'type': 'High Confidence Setup',
                'message': f"Setup confidence at {symbol_data['confidence']}%",
                'symbol': symbol_data['symbol'], 
                'timestamp': datetime.now(),
                'action': 'Consider position entry'
            })
        
        return alerts

def create_enhanced_gex_chart(data: Dict) -> go.Figure:
    """Create an enhanced GEX profile chart with advanced features"""
    
    profiles = data['profiles']
    df = pd.DataFrame(profiles)
    
    # Create subplot with secondary y-axis
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        subplot_titles=('Gamma Exposure Profile', 'Open Interest'),
        vertical_spacing=0.1,
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
    )
    
    # Color mapping for different strike types
    colors = []
    for profile in profiles:
        if profile['type'] == 'call':
            colors.append('#10b981')  # Green
        elif profile['type'] == 'put':
            colors.append('#ef4444')  # Red
        else:
            colors.append('#f59e0b')  # Yellow for flip point
    
    # Main GEX bars
    fig.add_trace(
        go.Bar(
            x=df['strike'],
            y=df['gex'],
            name='Gamma Exposure',
            marker=dict(
                color=colors,
                line=dict(color='rgba(255,255,255,0.2)', width=1)
            ),
            hovertemplate='<b>Strike:</b> $%{x}<br>' +
                         '<b>GEX:</b> %{y:.1f}M<br>' +
                         '<b>OI:</b> %{customdata:,.0f}<extra></extra>',
            customdata=df['oi']
        ),
        row=1, col=1
    )
    
    # Cumulative GEX line
    fig.add_trace(
        go.Scatter(
            x=df['strike'],
            y=df['cumulative'],
            name='Cumulative GEX',
            line=dict(color='#8b5cf6', width=3, dash='dot'),
            hovertemplate='<b>Strike:</b> $%{x}<br>' +
                         '<b>Cumulative GEX:</b> %{y:.1f}M<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Add current price line
    fig.add_vline(
        x=data['price'],
        line=dict(color='#06b6d4', width=3, dash='solid'),
        annotation_text=f"Current: ${data['price']}",
        annotation_position="top",
        row=1, col=1
    )
    
    # Add gamma flip line
    fig.add_vline(
        x=data['gamma_flip'],
        line=dict(color='#f59e0b', width=2, dash='dash'),
        annotation_text=f"Flip: ${data['gamma_flip']}",
        annotation_position="bottom",
        row=1, col=1
    )
    
    # Open Interest bars
    fig.add_trace(
        go.Bar(
            x=df['strike'],
            y=df['oi'],
            name='Open Interest',
            marker=dict(color='rgba(59, 130, 246, 0.6)'),
            hovertemplate='<b>Strike:</b> $%{x}<br>' +
                         '<b>OI:</b> %{y:,.0f}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Update layout with enhanced styling
    fig.update_layout(
        title={
            'text': f"{data['symbol']} Gamma Exposure Analysis",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24, 'color': '#ffffff'}
        },
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#ffffff', family='Inter'),
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(30, 41, 59, 0.8)',
            bordercolor='rgba(59, 130, 246, 0.3)',
            borderwidth=1
        ),
        height=700,
        margin=dict(l=10, r=10, t=60, b=10)
    )
    
    # Update axes
    fig.update_xaxes(
        title='Strike Price ($)',
        gridcolor='rgba(71, 85, 105, 0.3)',
        tickfont=dict(color='#cbd5e1')
    )
    fig.update_yaxes(
        title='Gamma Exposure ($M)',
        gridcolor='rgba(71, 85, 105, 0.3)', 
        tickfont=dict(color='#cbd5e1'),
        row=1, col=1
    )
    fig.update_yaxes(
        title='Open Interest',
        gridcolor='rgba(71, 85, 105, 0.3)',
        tickfont=dict(color='#cbd5e1'),
        row=2, col=1
    )
    
    return fig

def create_portfolio_performance_chart(portfolio_data: Dict) -> go.Figure:
    """Create portfolio performance visualization"""
    
    # Generate mock daily returns
    dates = pd.date_range(start='2024-01-01', end=datetime.now(), freq='D')
    returns = np.random.normal(0.001, 0.02, len(dates))  # Daily returns
    cumulative_returns = (1 + pd.Series(returns)).cumprod()
    portfolio_values = portfolio_data['total_value'] * cumulative_returns / cumulative_returns.iloc[-1]
    
    fig = go.Figure()
    
    # Portfolio value line
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=portfolio_values,
            name='Portfolio Value',
            line=dict(color='#10b981', width=3),
            hovertemplate='<b>Date:</b> %{x}<br>' +
                         '<b>Value:</b> $%{y:,.0f}<extra></extra>'
        )
    )
    
    # Add benchmark (SPY simulation)
    spy_returns = np.random.normal(0.0008, 0.015, len(dates))  
    spy_cumulative = (1 + pd.Series(spy_returns)).cumprod()
    spy_values = portfolio_data['total_value'] * spy_cumulative / spy_cumulative.iloc[-1]
    
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=spy_values,
            name='SPY Benchmark',
            line=dict(color='#6b7280', width=2, dash='dash'),
            hovertemplate='<b>Date:</b> %{x}<br>' +
                         '<b>SPY Value:</b> $%{y:,.0f}<extra></extra>'
        )
    )
    
    # Update layout
    fig.update_layout(
        title={
            'text': 'Portfolio Performance vs Benchmark',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#ffffff'}
        },
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#ffffff', family='Inter'),
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(30, 41, 59, 0.8)',
            bordercolor='rgba(59, 130, 246, 0.3)',
            borderwidth=1
        ),
        height=400,
        margin=dict(l=10, r=10, t=40, b=10)
    )
    
    fig.update_xaxes(
        gridcolor='rgba(71, 85, 105, 0.3)',
        tickfont=dict(color='#cbd5e1')
    )
    fig.update_yaxes(
        title='Portfolio Value ($)',
        gridcolor='rgba(71, 85, 105, 0.3)',
        tickfont=dict(color='#cbd5e1'),
        tickformat='$,.0f'
    )
    
    return fig

def create_risk_metrics_chart(portfolio_data: Dict) -> go.Figure:
    """Create advanced risk metrics visualization"""
    
    metrics = {
        'Sharpe Ratio': portfolio_data['sharpe_ratio'],
        'Max Drawdown': abs(portfolio_data['max_drawdown']),
        'Win Rate': portfolio_data['win_rate'] / 100,
        'Volatility': 0.15,  # Mock data
        'Beta': 0.8,  # Mock data
    }
    
    fig = go.Figure()
    
    # Radar chart for risk metrics
    categories = list(metrics.keys())
    values = list(metrics.values())
    
    # Normalize values for radar chart
    normalized_values = []
    max_vals = {'Sharpe Ratio': 3, 'Max Drawdown': 0.2, 'Win Rate': 1, 'Volatility': 0.3, 'Beta': 2}
    
    for cat, val in zip(categories, values):
        normalized_values.append(min(val / max_vals[cat], 1) * 100)
    
    fig.add_trace(
        go.Scatterpolar(
            r=normalized_values + [normalized_values[0]],  # Close the shape
            theta=categories + [categories[0]],
            fill='toself',
            fillcolor='rgba(59, 130, 246, 0.3)',
            line=dict(color='#3b82f6', width=2),
            name='Risk Profile'
        )
    )
    
    fig.update_layout(
        title={
            'text': 'Portfolio Risk Metrics',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#ffffff'}
        },
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                gridcolor='rgba(71, 85, 105, 0.3)',
                tickfont=dict(color='#cbd5e1', size=10)
            ),
            angularaxis=dict(
                gridcolor='rgba(71, 85, 105, 0.3)',
                tickfont=dict(color='#ffffff', size=12)
            )
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#ffffff', family='Inter'),
        height=400,
        margin=dict(l=10, r=10, t=40, b=10)
    )
    
    return fig

# Initialize data manager and alert system
@st.cache_resource
def get_data_manager():
    return GEXDataManager()

@st.cache_resource  
def get_alert_system():
    return AlertSystem()

data_manager = get_data_manager()
alert_system = get_alert_system()

# Main App Header with symbol indicator
current_symbol_display = selected_symbol if 'selected_symbol' in locals() else 'SPY'

st.markdown(f"""
<div class="main-header">🚀 GEX TRADING COMMAND CENTER</div>
<div class="sub-header">Professional Gamma Exposure Analysis & Trading Platform</div>
<div style="text-align: center; margin-bottom: 2rem;">
    <span style="background: linear-gradient(135deg, #3b82f6, #60a5fa); color: white; padding: 8px 16px; border-radius: 20px; font-weight: bold; font-size: 1.1em;">
        📊 Currently Analyzing: {current_symbol_display}
    </span>
</div>
""", unsafe_allow_html=True)

# Sidebar Configuration - Enhanced visibility
st.sidebar.markdown("""
<style>
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #2d3748 0%, #4a5568 50%, #2d3748 100%);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    # Add a visual header for the sidebar
    st.markdown("""
    <div style="background: linear-gradient(135deg, #3b82f6, #60a5fa); padding: 20px; margin: -20px -20px 20px -20px; text-align: center; border-radius: 0 0 16px 16px;">
        <h2 style="color: white; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.3);">⚙️ Trading Controls</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Symbol Selection with custom input
    st.markdown("### 📊 **Symbol Selection**")
    
    # Toggle between preset and custom
    symbol_mode = st.radio(
        "**Selection Mode**",
        ["📋 Preset Symbols", "✏️ Custom Symbol"],
        horizontal=True
    )
    
    if symbol_mode == "📋 Preset Symbols":
        selected_symbol = st.selectbox(
            "**Choose Symbol**",
            options=[sym for sym in data_manager.symbols if sym not in data_manager.custom_symbols],
            index=0,
            help="Choose from popular symbols for GEX analysis"
        )
    else:
        # Custom symbol input
        col1, col2 = st.columns([3, 1])
        with col1:
            custom_symbol = st.text_input(
                "**Enter Symbol**",
                placeholder="e.g., AMZN, GOOGL, META",
                help="Enter any stock symbol for analysis"
            ).upper().strip()
        
        with col2:
            if st.button("➕", help="Add Symbol", use_container_width=True):
                if custom_symbol:
                    with st.spinner(f"Analyzing {custom_symbol}..."):
                        success = data_manager.add_custom_symbol(custom_symbol)
                        if success:
                            st.success(f"✅ {custom_symbol} added!")
                            selected_symbol = custom_symbol
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("❌ Invalid symbol format")
                else:
                    st.warning("Please enter a symbol")
        
        # Show custom symbols if any exist
        if data_manager.custom_symbols:
            st.markdown("**Custom Symbols:**")
            custom_symbols_list = list(data_manager.custom_symbols)
            selected_symbol = st.selectbox(
                "**Select Custom Symbol**",
                options=custom_symbols_list,
                help="Choose from your custom symbols"
            )
        else:
            selected_symbol = 'SPY'  # Default if no custom symbols
    
    # Display symbol info
    if selected_symbol in data_manager.data:
        symbol_info = data_manager.data[selected_symbol]['symbol_info']
        st.markdown(f"""
        <div style="background: rgba(59, 130, 246, 0.1); padding: 12px; border-radius: 8px; margin: 8px 0; border: 1px solid rgba(96, 165, 250, 0.3);">
            <div style="color: #60a5fa; font-weight: bold; font-size: 1.1em;">{selected_symbol}</div>
            <div style="color: #cbd5e1; font-size: 0.9em;">{symbol_info.get('name', 'Unknown Company')}</div>
            <div style="color: #94a3b8; font-size: 0.8em;">{symbol_info.get('sector', 'Unknown')} • {symbol_info.get('market_cap', 'N/A')}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Auto-refresh toggle
    auto_refresh = st.toggle(
        "🔄 **Auto Refresh Data**",
        value=st.session_state.auto_refresh,
        help="Enable automatic data refresh every 30 seconds"
    )
    st.session_state.auto_refresh = auto_refresh
    
    if auto_refresh:
        st.success("✅ Auto-refresh enabled")
    else:
        st.info("⏸️ Manual refresh mode")
    
    st.markdown("---")
    
    # Alert Settings
    st.markdown("### 🚨 **Alert Configuration**")
    
    high_priority_alerts = st.checkbox(
        "🔴 **High Priority Alerts**",
        value=st.session_state.alert_settings['high_priority'],
        help="Extreme GEX levels, gamma flip proximity"
    )
    
    medium_priority_alerts = st.checkbox(
        "🟡 **Medium Priority Alerts**", 
        value=st.session_state.alert_settings['medium_priority'],
        help="High confidence setups, wall breaches"
    )
    
    low_priority_alerts = st.checkbox(
        "🟢 **Low Priority Alerts**",
        value=st.session_state.alert_settings['low_priority'],
        help="General market updates, minor changes"
    )
    
    st.session_state.alert_settings = {
        'high_priority': high_priority_alerts,
        'medium_priority': medium_priority_alerts,
        'low_priority': low_priority_alerts
    }
    
    st.markdown("---")
    
    # Manual refresh button
    if st.button("🔄 **Refresh All Data**", type="primary", use_container_width=True):
        with st.spinner("Refreshing data..."):
            data_manager.refresh_data()
            time.sleep(1)  # Brief pause for UX
        st.success("✅ Data refreshed!")
        time.sleep(2)
        st.rerun()
    
    st.markdown("---")
    
    # Risk Management Settings
    st.markdown("### ⚖️ **Risk Management**")
    
    max_position_size = st.slider(
        "**Max Position Size (%)**",
        min_value=1,
        max_value=10,
        value=3,
        help="Maximum percentage of portfolio per trade"
    )
    
    profit_target = st.slider(
        "**Profit Target (%)**",
        min_value=25,
        max_value=200,
        value=100,
        help="Default profit target for long options"
    )
    
    stop_loss = st.slider(
        "**Stop Loss (%)**", 
        min_value=25,
        max_value=75,
        value=50,
        help="Default stop loss for all positions"
    )
    
    st.markdown("---")
    
    # Analysis history for custom symbols
    if data_manager.custom_symbols:
        st.markdown("### 📈 **Custom Symbol Analysis History**")
        
        # Create a comparison table
        comparison_data = []
        for symbol in sorted(data_manager.custom_symbols):
            if symbol in data_manager.data:
                data = data_manager.data[symbol]
                comparison_data.append({
                    'Symbol': symbol,
                    'Price': f"${data['price']:.2f}",
                    'Net GEX': f"{data['net_gex']:.2f}B",
                    'Regime': data['regime'],
                    'Confidence': f"{data['confidence']:.0f}%",
                    'Setups': len(data['setups'])
                })
        
        if comparison_data:
            df_comparison = pd.DataFrame(comparison_data)
            
            st.markdown("""
            <div style="background: rgba(45, 55, 72, 0.8); padding: 20px; border-radius: 16px; border: 2px solid rgba(96, 165, 250, 0.3);">
            """, unsafe_allow_html=True)
            
            st.dataframe(
                df_comparison,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Symbol": st.column_config.TextColumn("Symbol", width="small"),
                    "Price": st.column_config.TextColumn("Price", width="small"),
                    "Net GEX": st.column_config.TextColumn("Net GEX", width="medium"),
                    "Regime": st.column_config.TextColumn("Regime", width="medium"),
                    "Confidence": st.column_config.TextColumn("Confidence", width="small"),
                    "Setups": st.column_config.NumberColumn("Setups", width="small")
                }
            )
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Quick switch buttons
            st.markdown("**🔄 Quick Symbol Switch:**")
            cols = st.columns(min(4, len(data_manager.custom_symbols)))
            for i, symbol in enumerate(sorted(data_manager.custom_symbols)):
                if i < len(cols):
                    with cols[i]:
                        if st.button(f"📊 {symbol}", key=f"switch_{symbol}", use_container_width=True):
                            st.session_state.selected_symbol = symbol
                            st.rerun()
    
    st.markdown("---")
    
    # Risk Management Section
    st.markdown("### ⚖️ **Risk Management Guidelines**")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div style="background: rgba(45, 55, 72, 0.8); padding: 20px; border-radius: 12px; border: 2px solid rgba(16, 185, 129, 0.3);">
            <h4 style="color: #10b981; margin-bottom: 12px;">✅ Position Sizing for {current_data['symbol']}</h4>
            <p style="color: #cbd5e1; margin: 4px 0;"><strong>Max Position:</strong> {max_position_size}% of portfolio</p>
            <p style="color: #cbd5e1; margin: 4px 0;"><strong>Suggested Size:</strong> {max(1, max_position_size // 2)}% (Conservative)</p>
            <p style="color: #cbd5e1; margin: 4px 0;"><strong>High Vol Adjustment:</strong> {'Reduce by 50%' if current_data['symbol'] in ['TSLA', 'NVDA'] else 'Standard sizing'}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="background: rgba(45, 55, 72, 0.8); padding: 20px; border-radius: 12px; border: 2px solid rgba(239, 68, 68, 0.3);">
            <h4 style="color: #ef4444; margin-bottom: 12px;">🛡️ Risk Limits</h4>
            <p style="color: #cbd5e1; margin: 4px 0;"><strong>Stop Loss:</strong> {stop_loss}%</p>
            <p style="color: #cbd5e1; margin: 4px 0;"><strong>Profit Target:</strong> {profit_target}%</p>
            <p style="color: #cbd5e1; margin: 4px 0;"><strong>Max DTE:</strong> {'5 days' if current_data['symbol'] in data_manager.symbols[:4] else '3 days'}</p>
        </div>
        """, unsafe_allow_html=True)

# Auto-refresh logic
if auto_refresh:
    if datetime.now() - st.session_state.last_update > timedelta(seconds=30):
        data_manager.refresh_data()
        st.session_state.last_update = datetime.now()
        st.rerun()

# Get current symbol data with error handling
try:
    if 'selected_symbol' in locals() and selected_symbol in data_manager.data:
        current_data = data_manager.data[selected_symbol]
    else:
        # Fallback to SPY if symbol not found
        current_data = data_manager.data['SPY']
        selected_symbol = 'SPY'
except KeyError:
    # Generate data if missing
    data_manager.refresh_data()
    current_data = data_manager.data['SPY']
    selected_symbol = 'SPY'

# Check for alerts
current_alerts = alert_system.check_alerts(current_data)

# Display alerts if any
if current_alerts:
    st.markdown("### 🚨 **Active Alerts**")
    for alert in current_alerts:
        if (alert['priority'] == 'HIGH' and st.session_state.alert_settings['high_priority']) or \
           (alert['priority'] == 'MEDIUM' and st.session_state.alert_settings['medium_priority']) or \
           (alert['priority'] == 'LOW' and st.session_state.alert_settings['low_priority']):
            
            alert_class = f"alert-{alert['priority'].lower()}"
            st.markdown(f"""
            <div class="alert-container {alert_class}">
                <strong>🚨 {alert['priority']} PRIORITY:</strong> {alert['message']}<br>
                <small><strong>Action:</strong> {alert['action']}</small><br>
                <small><strong>Time:</strong> {alert['timestamp'].strftime('%H:%M:%S')}</small>
            </div>
            """, unsafe_allow_html=True)

# Main content tabs
tab1, tab2, tab3, tab4 = st.tabs(["🧠 **GEX Analysis**", "🎯 **Trade Setups**", "📊 **Portfolio**", "📚 **Education**"])

with tab1:
    # Add symbol analysis header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"### 📊 **{current_data['symbol']} - Key Metrics**")
    with col2:
        if st.button("🔄 **Refresh Analysis**", key="refresh_symbol"):
            with st.spinner(f"Refreshing {current_data['symbol']} data..."):
                # Refresh data for current symbol
                data_manager.data[current_data['symbol']] = data_manager._generate_symbol_data(current_data['symbol'])
                time.sleep(1)
            st.success("✅ Analysis updated!")
            st.rerun()
    
    # Display symbol information
    symbol_info = current_data['symbol_info']
    st.markdown(f"""
    <div style="background: rgba(59, 130, 246, 0.1); padding: 20px; border-radius: 16px; margin: 16px 0; border: 2px solid rgba(96, 165, 250, 0.3);">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h3 style="color: #60a5fa; margin: 0;">{current_data['symbol']} - {symbol_info.get('name', 'Unknown Company')}</h3>
                <p style="color: #cbd5e1; margin: 4px 0 0 0;">
                    <strong>Sector:</strong> {symbol_info.get('sector', 'Unknown')} | 
                    <strong>Market Cap:</strong> {symbol_info.get('market_cap', 'N/A')}
                </p>
            </div>
            <div style="text-align: right;">
                <div style="color: #ffffff; font-size: 2rem; font-weight: bold;">${current_data['price']}</div>
                <div style="color: #94a3b8; font-size: 0.9em;">Last Price</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Key metrics in enhanced cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        price_change = np.random.uniform(-2, 2)  # Mock price change
        change_class = "positive" if price_change >= 0 else "negative"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">${current_data['price']}</div>
            <div class="metric-label">Current Price</div>
            <div class="metric-change {change_class}">
                {'+' if price_change >= 0 else ''}{price_change:.2f} ({price_change/current_data['price']*100:.1f}%)
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        gex_class = "negative" if current_data['net_gex'] < 0 else "positive"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: {'#ef4444' if current_data['net_gex'] < 0 else '#10b981'}">{current_data['net_gex']:.1f}B</div>
            <div class="metric-label">Net GEX</div>
            <div class="metric-change {gex_class}">
                {current_data['regime']}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        flip_distance = abs(current_data['price'] - current_data['gamma_flip']) / current_data['price'] * 100
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">${current_data['gamma_flip']}</div>
            <div class="metric-label">Gamma Flip</div>
            <div class="metric-change neutral">
                {flip_distance:.1f}% away
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        confidence_class = "positive" if current_data['confidence'] > 70 else "neutral" if current_data['confidence'] > 50 else "negative"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{current_data['confidence']:.0f}%</div>
            <div class="metric-label">Confidence</div>
            <div class="progress-container">
                <div class="progress-bar" style="width: {current_data['confidence']}%"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Enhanced GEX Chart with symbol title
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    gex_chart = create_enhanced_gex_chart(current_data)
    
    # Update chart title to include symbol info
    gex_chart.update_layout(
        title={
            'text': f"{current_data['symbol']} - {symbol_info.get('name', 'Unknown')} | Gamma Exposure Analysis",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#ffffff'}
        }
    )
    
    st.plotly_chart(gex_chart, use_container_width=True, config={'displayModeBar': False})
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Additional Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 **Key Levels**")
        
        # Call walls
        st.markdown("**📈 Call Walls (Resistance)**")
        for i, wall in enumerate(current_data['call_walls'][:3]):
            distance = (wall['strike'] - current_data['price']) / current_data['price'] * 100
            st.markdown(f"- **${wall['strike']:.2f}** ({distance:+.1f}%) - {abs(wall['gex']):.0f}M GEX")
        
        # Put walls  
        st.markdown("**📉 Put Walls (Support)**")
        for i, wall in enumerate(current_data['put_walls'][:3]):
            distance = (wall['strike'] - current_data['price']) / current_data['price'] * 100  
            st.markdown(f"- **${wall['strike']:.2f}** ({distance:+.1f}%) - {abs(wall['gex']):.0f}M GEX")
    
    with col2:
        st.markdown("### 📊 **Market Regime**")
        
        regime_color = "#ef4444" if current_data['net_gex'] < 0 else "#10b981"
        st.markdown(f"""
        <div style="background: rgba(30, 41, 59, 0.7); padding: 20px; border-radius: 12px; border: 1px solid rgba(59, 130, 246, 0.3);">
            <h4 style="color: {regime_color}; margin: 0 0 10px 0;">{current_data['regime']}</h4>
            <p style="color: #cbd5e1; margin: 5px 0;">
                <strong>Mode:</strong> {current_data['volatility_mode']}
            </p>
            <p style="color: #cbd5e1; margin: 5px 0;">
                <strong>Expected:</strong> {'Explosive moves possible' if current_data['net_gex'] < 0 else 'Range-bound trading likely'}
            </p>
            <p style="color: #cbd5e1; margin: 5px 0;">
                <strong>Strategy:</strong> {'Long options at flip' if current_data['net_gex'] < 0 else 'Sell premium between walls'}
            </p>
        </div>
        """, unsafe_allow_html=True)

with tab2:
    st.markdown(f"### 🎯 **{current_data['symbol']} Morning Trade Setups**")
    
    # Add quick symbol stats
    col1, col2, col3 = st.columns(3)
    with col1:
        regime_color = "#ef4444" if current_data['net_gex'] < 0 else "#10b981"
        st.markdown(f"""
        <div style="text-align: center; padding: 16px; background: rgba(45, 55, 72, 0.8); border-radius: 12px; border: 2px solid {regime_color}40;">
            <div style="color: {regime_color}; font-size: 1.2rem; font-weight: bold;">{current_data['regime']}</div>
            <div style="color: #cbd5e1; font-size: 0.9rem;">Net GEX: {current_data['net_gex']}B</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        flip_distance = abs(current_data['price'] - current_data['gamma_flip']) / current_data['price'] * 100
        flip_color = "#f59e0b" if flip_distance < 1 else "#60a5fa"
        st.markdown(f"""
        <div style="text-align: center; padding: 16px; background: rgba(45, 55, 72, 0.8); border-radius: 12px; border: 2px solid {flip_color}40;">
            <div style="color: {flip_color}; font-size: 1.2rem; font-weight: bold;">${current_data['gamma_flip']}</div>
            <div style="color: #cbd5e1; font-size: 0.9rem;">Gamma Flip ({flip_distance:.1f}% away)</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        conf_color = "#10b981" if current_data['confidence'] > 70 else "#f59e0b" if current_data['confidence'] > 50 else "#ef4444"
        st.markdown(f"""
        <div style="text-align: center; padding: 16px; background: rgba(45, 55, 72, 0.8); border-radius: 12px; border: 2px solid {conf_color}40;">
            <div style="color: {conf_color}; font-size: 1.2rem; font-weight: bold;">{current_data['confidence']:.0f}%</div>
            <div style="color: #cbd5e1; font-size: 0.9rem;">Setup Confidence</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    if current_data['setups']:
        for i, setup in enumerate(current_data['setups']):
            confidence_color = "#10b981" if setup['confidence'] > 80 else "#f59e0b" if setup['confidence'] > 60 else "#ef4444"
            risk_color = "#ef4444" if setup['risk'] == 'High' else "#f59e0b" if setup['risk'] == 'Medium' else "#10b981"
            
            st.markdown(f"""
            <div class="setup-card">
                <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 20px;">
                    <div>
                        <h3 style="color: #ffffff; margin: 0 0 8px 0; font-size: 1.5rem;">{setup['type']}</h3>
                        <p style="color: #94a3b8; margin: 0; font-size: 0.95rem;">{setup['description']}</p>
                    </div>
                    <div style="text-align: right;">
                        <div style="color: {confidence_color}; font-size: 2rem; font-weight: bold; margin: 0;">{setup['confidence']:.0f}%</div>
                        <div style="color: #94a3b8; font-size: 0.8rem;">CONFIDENCE</div>
                    </div>
                </div>
                
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 20px; margin-bottom: 20px;">
                    <div style="text-align: center;">
                        <div style="color: #94a3b8; font-size: 0.8rem; margin-bottom: 4px;">STRATEGY</div>
                        <div style="color: #ffffff; font-weight: 600;">{setup['strategy']}</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="color: #94a3b8; font-size: 0.8rem; margin-bottom: 4px;">STRIKES</div>
                        <div style="color: #ffffff; font-weight: 600;">{setup['strikes']}</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="color: #94a3b8; font-size: 0.8rem; margin-bottom: 4px;">RISK LEVEL</div>
                        <div style="color: {risk_color}; font-weight: 600;">{setup['risk']}</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="color: #94a3b8; font-size: 0.8rem; margin-bottom: 4px;">REWARD</div>
                        <div style="color: #10b981; font-weight: 600;">{setup['reward']}</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="color: #94a3b8; font-size: 0.8rem; margin-bottom: 4px;">TIMEFRAME</div>
                        <div style="color: #ffffff; font-weight: 600;">{setup['timeframe']}</div>
                    </div>
                </div>
                
                <details style="margin-bottom: 20px;">
                    <summary style="color: #3b82f6; cursor: pointer; font-weight: 600;">📋 Entry Conditions & Strategy</summary>
                    <div style="margin-top: 10px; padding: 15px; background: rgba(0,0,0,0.2); border-radius: 8px;">
                        <p style="color: #cbd5e1; margin: 0 0 10px 0;"><strong>Entry Conditions:</strong></p>
                        <ul style="color: #94a3b8; margin: 0 0 10px 20px;">
                            {''.join(f'<li>{condition}</li>' for condition in setup.get('entry_conditions', []))}
                        </ul>
                        <p style="color: #cbd5e1; margin: 0;"><strong>Exit Strategy:</strong> {setup.get('exit_strategy', 'Standard risk management rules')}</p>
                    </div>
                </details>
            </div>
            """, unsafe_allow_html=True)
            
            # Action buttons
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button(f"🚀 **Execute Trade**", key=f"execute_{i}", type="primary"):
                    st.success(f"✅ Trade setup '{setup['type']}' added to execution queue!")
                    st.balloons()
            
            with col2:
                if st.button(f"👁️ **Add to Watchlist**", key=f"watch_{i}"):
                    st.info(f"👁️ '{setup['type']}' added to watchlist")
            
            with col3:
                if st.button(f"📊 **Backtest Setup**", key=f"backtest_{i}"):
                    # Mock backtest results
                    win_rate = np.random.uniform(60, 85)
                    avg_return = np.random.uniform(15, 45)
                    st.success(f"📊 Backtest Results: {win_rate:.1f}% win rate, {avg_return:.1f}% avg return")
    
    else:
        st.markdown(f"""
        <div style="text-align: center; padding: 40px; background: rgba(45, 55, 72, 0.8); border-radius: 16px; border: 2px solid rgba(245, 158, 11, 0.4);">
            <h3 style="color: #f59e0b; margin-bottom: 16px;">🔍 No High-Confidence Setups</h3>
            <p style="color: #cbd5e1; margin-bottom: 16px;">No high-probability setups detected for <strong>{current_data['symbol']}</strong> under current market conditions.</p>
            <div style="background: rgba(245, 158, 11, 0.1); padding: 16px; border-radius: 12px; margin: 16px 0;">
                <p style="color: #fbbf24; margin: 0;"><strong>💡 Suggestions:</strong></p>
                <ul style="color: #cbd5e1; text-align: left; margin: 8px 0;">
                    <li>Wait for price to approach gamma flip point</li>
                    <li>Monitor for wall breach setups</li>
                    <li>Consider alternative symbols with stronger signals</li>
                    <li>Check back after market open for updated conditions</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Show current levels for reference
        st.markdown("#### 📊 **Current Key Levels**")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📈 Call Walls (Resistance)**")
            if current_data['call_walls']:
                for wall in current_data['call_walls'][:3]:
                    distance = (wall['strike'] - current_data['price']) / current_data['price'] * 100
                    st.markdown(f"• **${wall['strike']:.2f}** ({distance:+.1f}%) - {abs(wall['gex']):.1f}M GEX")
            else:
                st.markdown("• No significant call walls detected")
        
        with col2:
            st.markdown("**📉 Put Walls (Support)**") 
            if current_data['put_walls']:
                for wall in current_data['put_walls'][:3]:
                    distance = (wall['strike'] - current_data['price']) / current_data['price'] * 100
                    st.markdown(f"• **${wall['strike']:.2f}** ({distance:+.1f}%) - {abs(wall['gex']):.1f}M GEX")
            else:
                st.markdown("• No significant put walls detected")

with tab3:
    st.markdown("### 💼 **Portfolio Overview**")
    
    # Portfolio metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">${st.session_state.portfolio_data['total_value']:,.0f}</div>
            <div class="metric-label">Total Value</div>
            <div class="metric-change positive">+12.4% (30D)</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        pnl_class = "positive" if st.session_state.portfolio_data['day_pnl'] >= 0 else "negative"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: {'#10b981' if st.session_state.portfolio_data['day_pnl'] >= 0 else '#ef4444'}">${st.session_state.portfolio_data['day_pnl']:,.0f}</div>
            <div class="metric-label">Daily P&L</div>
            <div class="metric-change {pnl_class}">{'Profit' if st.session_state.portfolio_data['day_pnl'] >= 0 else 'Loss'} Today</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{st.session_state.portfolio_data['win_rate']:.1f}%</div>
            <div class="metric-label">Win Rate</div>
            <div class="progress-container">
                <div class="progress-bar" style="width: {st.session_state.portfolio_data['win_rate']}%"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{st.session_state.portfolio_data['sharpe_ratio']:.2f}</div>
            <div class="metric-label">Sharpe Ratio</div>
            <div class="metric-change positive">Excellent Risk-Adj Return</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Portfolio performance chart
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    portfolio_chart = create_portfolio_performance_chart(st.session_state.portfolio_data)
    st.plotly_chart(portfolio_chart, use_container_width=True, config={'displayModeBar': False})
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Risk metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        risk_chart = create_risk_metrics_chart(st.session_state.portfolio_data)
        st.plotly_chart(risk_chart, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 📈 **Active Positions**")
        
        # Mock active positions
        positions = [
            {'symbol': 'SPY', 'type': 'Call', 'strike': '445', 'pnl': 1250, 'status': 'winning'},
            {'symbol': 'QQQ', 'type': 'Put', 'strike': '370', 'pnl': -340, 'status': 'losing'},  
            {'symbol': 'IWM', 'type': 'Iron Condor', 'strike': '180/190', 'pnl': 480, 'status': 'winning'},
            {'symbol': 'DIA', 'type': 'Call Spread', 'strike': '350/355', 'pnl': 720, 'status': 'winning'}
        ]
        
        for pos in positions:
            pnl_color = "#10b981" if pos['pnl'] >= 0 else "#ef4444"
            status_color = "#10b981" if pos['status'] == 'winning' else "#ef4444"
            
            st.markdown(f"""
            <div style="background: rgba(30, 41, 59, 0.6); padding: 16px; border-radius: 12px; margin: 8px 0; border: 1px solid rgba(71, 85, 105, 0.3);">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div style="display: flex; align-items: center; gap: 12px;">
                        <div style="width: 8px; height: 8px; background: {status_color}; border-radius: 50%;"></div>
                        <span style="color: #ffffff; font-weight: 600;">{pos['symbol']}</span>
                        <span style="color: #94a3b8;">{pos['type']}</span>
                        <span style="color: #94a3b8;">${pos['strike']}</span>
                    </div>
                    <div style="color: {pnl_color}; font-weight: 600;">
                        {'+' if pos['pnl'] >= 0 else ''}${pos['pnl']}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

with tab4:
    st.markdown("### 📚 **GEX Trading Education**")
    
    # Education sections
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background: rgba(30, 41, 59, 0.7); padding: 24px; border-radius: 16px; border: 1px solid rgba(59, 130, 246, 0.3); margin-bottom: 20px;">
            <h4 style="color: #3b82f6; margin: 0 0 16px 0;">🧠 Understanding Gamma Exposure</h4>
            <div style="color: #cbd5e1; line-height: 1.6;">
                <p><strong style="color: #10b981;">Positive GEX:</strong> Market makers are long gamma, leading to volatility suppression. Price tends to stay within gamma walls.</p>
                <p><strong style="color: #ef4444;">Negative GEX:</strong> Market makers are short gamma, leading to volatility amplification. Price moves can be explosive.</p>
                <p><strong style="color: #f59e0b;">Gamma Flip:</strong> The zero-gamma crossing point where market regime changes from suppression to amplification.</p>
                <p><strong style="color: #8b5cf6;">Dealer Hedging:</strong> Market makers hedge their options positions by buying/selling shares, creating predictable flows.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background: rgba(30, 41, 59, 0.7); padding: 24px; border-radius: 16px; border: 1px solid rgba(59, 130, 246, 0.3);">
            <h4 style="color: #10b981; margin: 0 0 16px 0;">⚖️ Risk Management Rules</h4>
            <div style="color: #cbd5e1; line-height: 1.6;">
                <p><strong>Position Sizing:</strong> Never risk more than 2-3% of portfolio on any single GEX setup</p>
                <p><strong>Stop Losses:</strong> Set stops at 50% loss for long options, 100% loss for short options</p>
                <p><strong>Profit Targets:</strong> Take profits at 100% for long options, 50% for short premium</p>
                <p><strong>Time Management:</strong> Close positions with less than 1 DTE to avoid gamma risk</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: rgba(30, 41, 59, 0.7); padding: 24px; border-radius: 16px; border: 1px solid rgba(59, 130, 246, 0.3); margin-bottom: 20px;">
            <h4 style="color: #8b5cf6; margin: 0 0 16px 0;">🎯 Trading Strategies</h4>
            <div style="color: #cbd5e1; line-height: 1.6;">
                <p><strong style="color: #10b981;">Squeeze Plays:</strong> Long options when price is near gamma flip in negative GEX environment</p>
                <p><strong style="color: #3b82f6;">Iron Condors:</strong> Sell premium between gamma walls in positive GEX environment</p>
                <p><strong style="color: #f59e0b;">Wall Trades:</strong> Use call walls as resistance and put walls as support for directional plays</p>
                <p><strong style="color: #ef4444;">Regime Changes:</strong> Monitor gamma flip breaches for explosive move opportunities</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background: rgba(30, 41, 59, 0.7); padding: 24px; border-radius: 16px; border: 1px solid rgba(59, 130, 246, 0.3);">
            <h4 style="color: #f59e0b; margin: 0 0 16px 0;">⏰ Timing & Execution</h4>
            <div style="color: #cbd5e1; line-height: 1.6;">
                <p><strong>Best Times:</strong> First 30 minutes after market open when GEX effects are strongest</p>
                <p><strong>Expiry Selection:</strong> Use 0-5 DTE for maximum gamma sensitivity</p>
                <p><strong>Market Conditions:</strong> GEX strategies work best in trending or range-bound markets</p>
                <p><strong>Volume Confirmation:</strong> Wait for volume confirmation before entering positions</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Real-time status footer
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"**🔄 Last Update:** {st.session_state.last_update.strftime('%H:%M:%S')}")

with col2:
    st.markdown(f"**📊 Symbols Tracked:** {len(data_manager.symbols)}")

with col3:
    active_alerts = len([a for a in current_alerts if st.session_state.alert_settings.get(a['priority'].lower() + '_priority', False)])
    st.markdown(f"**🚨 Active Alerts:** {active_alerts}")

with col4:
    market_status = "🟢 OPEN" if 9 <= datetime.now().hour <= 16 else "🔴 CLOSED"
    st.markdown(f"**📈 Market Status:** {market_status}")

# Footer with additional info
st.markdown("""
---
<div style="text-align: center; color: #64748b; padding: 20px;">
    <p><strong>🚀 GEX Trading Command Center v2.0</strong> | Professional Gamma Exposure Analysis Platform</p>
    <p>⚠️ <em>For educational purposes only. Not financial advice. Past performance does not guarantee future results.</em></p>
    <p>Built with Streamlit 🎈 | Enhanced with Advanced Analytics & Risk Management</p>
</div>
""", unsafe_allow_html=True)

# Auto-refresh mechanism
if auto_refresh:
    # Add a small delay and rerun to create continuous updates
    time.sleep(1)
    placeholder = st.empty()
    with placeholder.container():
        st.markdown("""
        <div style="position: fixed; bottom: 20px; right: 20px; background: rgba(16, 185, 129, 0.9); color: white; 
                    padding: 8px 16px; border-radius: 20px; font-size: 12px; z-index: 1000;">
            🔄 Auto-refreshing... Next update in {:.0f}s
        </div>
        """.format(30 - (datetime.now() - st.session_state.last_update).seconds), 
        unsafe_allow_html=True)

# Add some JavaScript for enhanced interactivity (if needed)
st.markdown("""
<script>
// Add any custom JavaScript for enhanced interactivity
document.addEventListener('DOMContentLoaded', function() {
    // Add smooth scrolling
    document.documentElement.style.scrollBehavior = 'smooth';
    
    // Add hover effects to metric cards
    const cards = document.querySelectorAll('.metric-card');
    cards.forEach(card => {
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-8px) scale(1.02)';
        });
        card.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0) scale(1)';
        });
    });
});
</script>
""", unsafe_allow_html=True)
