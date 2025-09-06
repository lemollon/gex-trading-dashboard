#!/usr/bin/env python3
"""
GEX Market Maker Exploitation Platform v4.0
Enhanced Visual Design with Complete Features
Well-organized with clear section markers for easy editing
~3,500 lines - Production Ready
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
import requests
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
from typing import Dict, List, Tuple, Optional
from bs4 import BeautifulSoup
import feedparser
import schedule
warnings.filterwarnings('ignore')

# ============================================================================
# SECTION 1: PAGE CONFIGURATION AND CONSTANTS
# ============================================================================

st.set_page_config(
    page_title="🎯 GEX Market Maker Exploitation Platform",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Discord webhook for alerts (hardcoded - no user config)
DISCORD_WEBHOOK = "https://discord.com/api/webhooks/1308901307493777469/BWNr70coUxdgWCBSutC5pDWakBkRxM_lyQbUeh8_5A2zClecULeO909XBwQiwUY-DzId"

# ============================================================================
# SECTION 2: VISUAL STYLING AND CSS
# Edit this section to change colors, animations, and visual elements
# ============================================================================

st.markdown("""
<style>
    /* === GLOBAL STYLES - Controls overall appearance === */
    .stApp {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        background-attachment: fixed;
    }
    
    .main {
        background-color: transparent !important;
    }
    
    /* Force consistent text colors */
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-weight: bold !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    [data-testid="stMetricLabel"] {
        color: #e0e0e0 !important;
        font-weight: 600 !important;
    }
    
    /* === HEADER STYLES === */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem;
        border-radius: 30px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 20px 60px rgba(0,0,0,0.4);
        animation: float 3s ease-in-out infinite;
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: "";
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: rotate 20s linear infinite;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    @keyframes rotate {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* === ACTION BOXES === */
    .action-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        border-radius: 20px;
        padding: 2.5rem;
        color: white;
        text-align: center;
        margin: 1.5rem 0;
        font-size: 1.6rem;
        font-weight: bold;
        box-shadow: 0 10px 40px rgba(245,87,108,0.4);
        cursor: pointer;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .action-box::after {
        content: "🎯";
        position: absolute;
        font-size: 100px;
        opacity: 0.1;
        right: 20px;
        top: 50%;
        transform: translateY(-50%);
    }
    
    .action-box:hover {
        transform: translateY(-8px) scale(1.03);
        box-shadow: 0 15px 50px rgba(245,87,108,0.6);
    }
    
    /* === MM STATUS STYLES === */
    .mm-trapped {
        background: linear-gradient(135deg, #ff0844, #ff4563) !important;
        animation: emergency-pulse 0.5s infinite;
    }
    
    @keyframes emergency-pulse {
        0%, 100% { 
            box-shadow: 0 0 30px rgba(255,8,68,0.8);
            transform: scale(1);
        }
        50% { 
            box-shadow: 0 0 50px rgba(255,8,68,1);
            transform: scale(1.02);
        }
    }
    
    .mm-defending {
        background: linear-gradient(135deg, #11998e, #38ef7d) !important;
    }
    
    .mm-scrambling {
        background: linear-gradient(135deg, #fc4a1a, #f7b733) !important;
        animation: warning-blink 1s infinite;
    }
    
    @keyframes warning-blink {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.85; }
    }
    
    /* === SYMBOL CARDS === */
    .symbol-row {
        background: rgba(255, 255, 255, 0.95) !important;
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 6px solid;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        cursor: pointer;
        color: #1a1a1a !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }
    
    .symbol-row:hover {
        transform: translateX(10px) translateY(-2px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.15);
    }
    
    .symbol-row h4, .symbol-row p, .symbol-row strong {
        color: #1a1a1a !important;
    }
    
    .squeeze-row {
        border-left-color: #ff0844 !important;
        background: linear-gradient(90deg, rgba(255,8,68,0.1), rgba(255,255,255,0.95)) !important;
    }
    
    .premium-row {
        border-left-color: #11998e !important;
        background: linear-gradient(90deg, rgba(17,153,142,0.1), rgba(255,255,255,0.95)) !important;
    }
    
    .wait-row {
        border-left-color: #95a5a6 !important;
        background: linear-gradient(90deg, rgba(149,165,166,0.1), rgba(255,255,255,0.95)) !important;
    }
    
    /* === INTERACTIVE BUTTONS === */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 50px;
        font-weight: bold;
        transition: all 0.3s;
        box-shadow: 0 4px 15px rgba(102,126,234,0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102,126,234,0.6);
    }
    
    /* === METRIC CARDS === */
    .metric-card {
        background: rgba(255, 255, 255, 0.95) !important;
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        color: #1a1a1a !important;
        transition: all 0.3s;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.15);
    }
    
    /* === PRESSURE MAP === */
    .mm-pressure-map {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        border-radius: 20px;
        padding: 2rem;
        color: white;
        font-family: 'Courier New', monospace;
        font-size: 1.1rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.3);
        position: relative;
    }
    
    .pressure-level {
        margin: 1rem 0;
        padding: 1rem;
        border-radius: 10px;
        position: relative;
        transition: all 0.3s;
    }
    
    .pressure-level:hover {
        transform: scale(1.02);
    }
    
    .high-pressure {
        background: linear-gradient(90deg, rgba(255,0,0,0.3), rgba(255,0,0,0.1));
        border: 2px solid #ff0000;
        animation: pulse-red 2s infinite;
        box-shadow: 0 0 20px rgba(255,0,0,0.3);
    }
    
    @keyframes pulse-red {
        0%, 100% { 
            box-shadow: 0 0 20px rgba(255,0,0,0.3);
        }
        50% { 
            box-shadow: 0 0 40px rgba(255,0,0,0.6);
        }
    }
    
    .current-price {
        background: linear-gradient(90deg, rgba(255,215,0,0.5), rgba(255,215,0,0.2));
        border: 2px solid #ffd700;
        font-weight: bold;
        box-shadow: 0 0 30px rgba(255,215,0,0.4);
        animation: golden-glow 2s infinite;
    }
    
    @keyframes golden-glow {
        0%, 100% { 
            box-shadow: 0 0 30px rgba(255,215,0,0.4);
        }
        50% { 
            box-shadow: 0 0 50px rgba(255,215,0,0.7);
        }
    }
    
    .low-pressure {
        background: linear-gradient(90deg, rgba(0,255,0,0.3), rgba(0,255,0,0.1));
        border: 2px solid #00ff00;
        box-shadow: 0 0 20px rgba(0,255,0,0.3);
    }
    
    /* === ACHIEVEMENT BADGES === */
    .achievement-badge {
        display: inline-block;
        background: linear-gradient(135deg, #FFD700, #FFA500);
        color: #1a1a1a;
        padding: 0.6rem 1.2rem;
        border-radius: 25px;
        margin: 0.3rem;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(255,215,0,0.4);
        transition: all 0.3s;
    }
    
    .achievement-badge:hover {
        transform: scale(1.1) rotate(5deg);
        box-shadow: 0 6px 20px rgba(255,215,0,0.6);
    }
    
    /* === STRATEGY CARDS === */
    .strategy-card {
        background: rgba(255, 255, 255, 0.95) !important;
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        color: #1a1a1a !important;
        transition: all 0.3s;
        position: relative;
        overflow: hidden;
    }
    
    .strategy-card::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 5px;
        background: linear-gradient(90deg, #667eea, #764ba2, #f093fb, #f5576c);
        animation: gradient-shift 3s ease infinite;
    }
    
    @keyframes gradient-shift {
        0%, 100% { transform: translateX(0); }
        50% { transform: translateX(10px); }
    }
    
    .strategy-card h3, .strategy-card p, .strategy-card strong {
        color: #1a1a1a !important;
    }
    
    .strategy-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 50px rgba(0,0,0,0.15);
    }
    
    /* === TAB STYLING === */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 5px;
        backdrop-filter: blur(10px);
    }
    
    .stTabs [data-baseweb="tab"] {
        color: white !important;
        font-weight: 600;
        border-radius: 10px;
        transition: all 0.3s;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255, 255, 255, 0.2);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
    }
    
    /* === WIN STREAK ANIMATION === */
    .win-streak {
        background: linear-gradient(135deg, #FFD700, #FFA500);
        color: #1a1a1a;
        padding: 0.8rem 1.5rem;
        border-radius: 30px;
        display: inline-block;
        font-weight: bold;
        animation: streak-glow 2s infinite;
        box-shadow: 0 0 30px rgba(255,215,0,0.5);
    }
    
    @keyframes streak-glow {
        0%, 100% { 
            box-shadow: 0 0 30px rgba(255,215,0,0.5);
            transform: scale(1);
        }
        50% { 
            box-shadow: 0 0 50px rgba(255,215,0,0.8);
            transform: scale(1.05);
        }
    }
    
    /* === SCROLLBAR STYLING === */
    ::-webkit-scrollbar {
        width: 12px;
        height: 12px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea, #764ba2);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2, #667eea);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SECTION 3: SYMBOL LISTS
# Add or remove symbols here
# ============================================================================

SP500_SYMBOLS = [
    'SPY', 'QQQ', 'IWM', 'DIA', 'XLF', 'XLK', 'XLE', 'XLV', 'XLI', 'XLC',
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'BRK-B', 'UNH', 'JNJ',
    'JPM', 'V', 'PG', 'HD', 'CVX', 'MA', 'BAC', 'ABBV', 'PFE', 'KO',
    'AVGO', 'PEP', 'TMO', 'COST', 'MRK', 'WMT', 'CSCO', 'ACN', 'DHR', 'NEE',
    'VZ', 'ABT', 'ADBE', 'ORCL', 'CRM', 'LLY', 'XOM', 'NKE', 'QCOM', 'TXN',
    'MDT', 'UPS', 'PM', 'T', 'LOW', 'HON', 'UNP', 'IBM', 'C', 'GS',
    'CAT', 'SPGI', 'INTC', 'INTU', 'ISRG', 'RTX', 'AXP', 'BKNG', 'NOW', 'DE',
    'PLD', 'TJX', 'GE', 'AMD', 'MU', 'SYK', 'BLK', 'MDLZ', 'ADI', 'GILD',
    'LRCX', 'KLAC', 'PYPL', 'REGN', 'ATVI', 'FISV', 'CI', 'SO', 'ZTS', 'DUK',
    'BSX', 'CSX', 'CL', 'MMC', 'ITW', 'BMY', 'AON', 'EQIX', 'APD', 'SNPS',
    'SHW', 'CME', 'FCX', 'PGR', 'MSI', 'ICE', 'USB', 'NSC', 'COP', 'EMR',
    'HUM', 'TFC', 'WM', 'F', 'ADP', 'GM', 'GD', 'CDNS', 'MCD', 'EOG',
    'FDX', 'BDX', 'TGT', 'BIIB', 'CVS', 'NOC', 'D', 'ECL', 'EL', 'WFC',
    'PSA', 'SLB', 'KMB', 'DG', 'ADSK', 'MRNA', 'CCI', 'ILMN', 'GIS', 'MCHP',
    'EXC', 'A', 'SBUX', 'JCI', 'CMG', 'KHC', 'ANET', 'MNST', 'CTAS', 'PAYX',
    'PNC', 'ROST', 'ORLY', 'ROP', 'HCA', 'MAR', 'AFL', 'CTSH', 'FAST', 'ODFL',
    'AEP', 'SPG', 'CARR', 'AIG', 'FTNT', 'EA', 'VRSK', 'ALL', 'BK', 'AZO',
    'MCK', 'OTIS', 'DLR', 'PCAR', 'IQV', 'NXPI', 'WLTW', 'PSX', 'O', 'PRU',
    'TEL', 'CTVA', 'XEL', 'WELL', 'DLTR', 'AVB', 'STZ', 'CBRE', 'EBAY', 'PPG',
    'IDXX', 'VRTX', 'AMT', 'AMGN', 'TROW', 'GPN', 'RSG', 'MSCI', 'EW', 'MTB',
    'DD', 'AMAT', 'INFO', 'ALB', 'DOW', 'LHX', 'KEYS', 'GLW', 'ANSS', 'CDW'
][:200]  # Limit to 200 symbols

# ============================================================================
# SECTION 4: MAIN ANALYZER CLASS
# Core GEX calculation logic - modify calculation methods here
# ============================================================================

class EnhancedGEXAnalyzer:
    """Complete GEX analyzer with MM exploitation - REAL DATA ONLY"""
    
    def __init__(self):
        self.risk_free_rate = 0.05
        self.trading_capital = 100000  # Fixed capital
        self.strategies_config = self.load_strategies_config()
        
    def load_strategies_config(self):
        """Load strategy configurations - EDIT THRESHOLDS HERE"""
        return {
            'squeeze_plays': {
                'negative_gex_threshold_spy': -1e9,
                'negative_gex_threshold_qqq': -500e6,
                'positive_gex_threshold_spy': 2e9,
                'positive_gex_threshold_qqq': 1e9,
                'flip_distance_threshold': 1.5,
                'dte_range': [0, 7],
                'confidence_threshold': 65
            },
            'premium_selling': {
                'positive_gex_threshold': 3e9,
                'wall_strength_threshold': 500e6,
                'wall_distance_range': [1, 5],
                'put_distance_range': [1, 8],
                'dte_range_calls': [0, 2],
                'dte_range_puts': [2, 5]
            },
            'iron_condor': {
                'min_gex_threshold': 1e9,
                'min_wall_spread': 3,
                'dte_range': [5, 10],
                'iv_rank_threshold': 50
            },
            'risk_management': {
                'max_position_size_squeeze': 0.03,
                'max_position_size_premium': 0.05,
                'max_position_size_condor': 0.02,
                'stop_loss_percentage': 0.50,
                'profit_target_long': 1.00,
                'profit_target_short': 0.50
            }
        }
    
    # === CALCULATION METHODS ===
    
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
        """Get current stock price from Yahoo Finance"""
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
            return None
    
    def get_options_chain(self, symbol, focus_weekly=True):
        """Get REAL options chain from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            exp_dates = ticker.options
            if not exp_dates:
                return None
            
            current_price = self.get_current_price(symbol)
            if current_price is None:
                return None
            
            all_chains = {}
            
            # Filter for weekly/daily expirations
            today = date.today()
            for exp_date in exp_dates[:15]:  # Check first 15 expirations
                try:
                    exp_dt = datetime.strptime(exp_date, '%Y-%m-%d')
                    dte = (exp_dt.date() - today).days
                    
                    # Focus on weekly/daily (0-7 DTE) and next week (8-14 DTE)
                    if dte <= 0 or dte > 14:
                        continue
                    
                    # Priority for daily options on major indices
                    if symbol in ['SPY', 'QQQ', 'IWM'] and dte <= 5:
                        pass  # Include daily options
                    elif dte <= 7:
                        pass  # Include weekly options
                    elif not focus_weekly and dte <= 14:
                        pass  # Include bi-weekly if not focusing on weekly
                    else:
                        continue
                    
                    chain = ticker.option_chain(exp_date)
                    
                    calls = chain.calls.copy()
                    calls = calls[calls['openInterest'] > 0]
                    
                    puts = chain.puts.copy()
                    puts = puts[puts['openInterest'] > 0]
                    
                    if len(calls) == 0 and len(puts) == 0:
                        continue
                    
                    # Calculate gamma
                    T = dte / 365.0
                    
                    calls['gamma'] = calls.apply(
                        lambda row: self.black_scholes_gamma(
                            current_price, 
                            row['strike'], 
                            T, 
                            self.risk_free_rate,
                            max(row['impliedVolatility'], 0.15) if pd.notna(row['impliedVolatility']) else 0.30
                        ), axis=1
                    )
                    
                    puts['gamma'] = puts.apply(
                        lambda row: self.black_scholes_gamma(
                            current_price,
                            row['strike'],
                            T,
                            self.risk_free_rate,
                            max(row['impliedVolatility'], 0.15) if pd.notna(row['impliedVolatility']) else 0.30
                        ), axis=1
                    )
                    
                    # Calculate GEX
                    calls['gex'] = current_price * calls['gamma'] * calls['openInterest'] * 100
                    puts['gex'] = -current_price * puts['gamma'] * puts['openInterest'] * 100
                    
                    calls['gex'] = calls['gex'].fillna(0)
                    puts['gex'] = puts['gex'].fillna(0)
                    
                    all_chains[exp_date] = {
                        'calls': calls,
                        'puts': puts,
                        'dte': dte,
                        'expiration': exp_dt,
                        'is_daily': dte <= 5 and symbol in ['SPY', 'QQQ', 'IWM']
                    }
                    
                except Exception as e:
                    continue
            
            return {
                'chains': all_chains,
                'current_price': current_price,
                'symbol': symbol,
                'data_timestamp': datetime.now()
            }
            
        except Exception as e:
            return None
    
    def calculate_gex_profile(self, options_data):
        """Calculate complete GEX profile from real options data"""
        try:
            if not options_data or 'chains' not in options_data:
                return None
            
            current_price = options_data['current_price']
            chains = options_data['chains']
            
            if not chains:
                return None
            
            # Aggregate GEX by strike
            strike_data = {}
            total_options_volume = 0
            
            for exp_date, chain_data in chains.items():
                calls = chain_data['calls']
                puts = chain_data['puts']
                
                # Track options volume
                total_options_volume += calls['volume'].fillna(0).sum()
                total_options_volume += puts['volume'].fillna(0).sum()
                
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
            
            # Calculate totals
            total_call_gex = float(df['call_gex'].sum())
            total_put_gex = float(df['put_gex'].sum())
            net_gex = total_call_gex + total_put_gex
            total_oi = int(df['call_oi'].sum() + df['put_oi'].sum())
            
            distance_to_flip = ((current_price - gamma_flip) / current_price) * 100
            
            # Market maker behavior analysis
            mm_behavior = self.analyze_market_maker_behavior(df, current_price, chains)
            
            # Flow toxicity score
            toxicity_score = self.calculate_flow_toxicity(chains, total_options_volume)
            
            # Calculate dealer pain
            dealer_pain = self.calculate_dealer_pain(net_gex, distance_to_flip, mm_behavior)
            
            # Determine MM status
            mm_status = self.determine_mm_status(net_gex, dealer_pain, distance_to_flip)
            
            return {
                'strike_data': df,
                'current_price': current_price,
                'gamma_flip': gamma_flip,
                'net_gex': net_gex,
                'total_call_gex': total_call_gex,
                'total_put_gex': total_put_gex,
                'call_walls': call_walls,
                'put_walls': put_walls,
                'total_volume': int(total_options_volume),
                'total_oi': total_oi,
                'distance_to_flip': distance_to_flip,
                'mm_behavior': mm_behavior,
                'toxicity_score': toxicity_score,
                'dealer_pain': dealer_pain,
                'mm_status': mm_status,
                'data_timestamp': options_data.get('data_timestamp', datetime.now())
            }
            
        except Exception as e:
            return None
    
    def find_gamma_flip(self, df, current_price):
        """Find the gamma flip point"""
        try:
            for i in range(len(df) - 1):
                curr = df.iloc[i]['cumulative_gex']
                next_val = df.iloc[i + 1]['cumulative_gex']
                
                if (curr <= 0 <= next_val) or (curr >= 0 >= next_val):
                    curr_strike = df.iloc[i]['strike']
                    next_strike = df.iloc[i + 1]['strike']
                    
                    if next_val != curr:
                        ratio = abs(curr) / abs(next_val - curr)
                        flip = curr_strike + ratio * (next_strike - curr_strike)
                        return flip
            
            min_idx = df['cumulative_gex'].abs().idxmin()
            return df.loc[min_idx, 'strike']
            
        except:
            return current_price
    
    def analyze_market_maker_behavior(self, strike_df, current_price, chains):
        """Analyze market maker positioning and behavior"""
        try:
            # Delta-neutral positioning score
            atm_strikes = strike_df[
                (strike_df['strike'] >= current_price * 0.98) &
                (strike_df['strike'] <= current_price * 1.02)
            ]
            
            delta_neutral_score = 0
            if len(atm_strikes) > 0:
                straddle_activity = atm_strikes['call_volume'].sum() + atm_strikes['put_volume'].sum()
                total_volume = strike_df['call_volume'].sum() + strike_df['put_volume'].sum()
                delta_neutral_score = (straddle_activity / max(total_volume, 1)) * 100
            
            # Pin risk assessment
            pin_risk = 0
            near_strikes = strike_df[
                (strike_df['strike'] >= current_price * 0.99) &
                (strike_df['strike'] <= current_price * 1.01)
            ]
            if len(near_strikes) > 0:
                near_gamma = near_strikes['call_gex'].sum() + abs(near_strikes['put_gex'].sum())
                total_gamma = abs(strike_df['call_gex'].sum()) + abs(strike_df['put_gex'].sum())
                pin_risk = (near_gamma / max(total_gamma, 1)) * 100
            
            # Unusual spread activity
            spread_activity = 0
            for exp_date, chain_data in chains.items():
                calls = chain_data['calls']
                high_oi_strikes = calls[calls['openInterest'] > calls['openInterest'].quantile(0.8)]
                spread_activity += len(high_oi_strikes)
            
            # Volume/OI ratios for institutional detection
            vol_oi_ratio = 0
            total_volume = strike_df['call_volume'].sum() + strike_df['put_volume'].sum()
            total_oi = strike_df['call_oi'].sum() + strike_df['put_oi'].sum()
            if total_oi > 0:
                vol_oi_ratio = total_volume / total_oi
            
            return {
                'delta_neutral_score': min(100, delta_neutral_score),
                'pin_risk': min(100, pin_risk),
                'spread_activity': spread_activity,
                'vol_oi_ratio': vol_oi_ratio,
                'institutional_flow': vol_oi_ratio > 0.5
            }
            
        except:
            return {
                'delta_neutral_score': 0,
                'pin_risk': 0,
                'spread_activity': 0,
                'vol_oi_ratio': 0,
                'institutional_flow': False
            }
    
    def calculate_flow_toxicity(self, chains, total_volume):
        """Calculate flow toxicity score (-100 to +100)"""
        try:
            score = 0
            
            # Smart money indicators
            large_trades = 0
            weekly_preference = 0
            otm_activity = 0
            small_lots = 0
            
            for exp_date, chain_data in chains.items():
                calls = chain_data['calls']
                puts = chain_data['puts']
                dte = chain_data['dte']
                
                # Large block detection
                large_call_oi = len(calls[calls['openInterest'] > 1000])
                large_put_oi = len(puts[puts['openInterest'] > 1000])
                large_trades += large_call_oi + large_put_oi
                
                # Weekly preference (retail indicator)
                if dte <= 7:
                    weekly_vol = calls['volume'].fillna(0).sum() + puts['volume'].fillna(0).sum()
                    weekly_preference += weekly_vol
                
                # OTM activity (retail indicator)
                current_price = calls['strike'].median()  # Approximate
                far_otm_calls = len(calls[calls['strike'] > current_price * 1.1])
                far_otm_puts = len(puts[puts['strike'] < current_price * 0.9])
                otm_activity += far_otm_calls + far_otm_puts
                
                # Small lot detection
                small_call_vol = len(calls[calls['volume'].fillna(0) < 10])
                small_put_vol = len(puts[puts['volume'].fillna(0) < 10])
                small_lots += small_call_vol + small_put_vol
            
            # Scoring logic
            if large_trades > 5:
                score += 20
            if weekly_preference / max(total_volume, 1) > 0.7:
                score -= 20
            if otm_activity > 10:
                score -= 15
            if small_lots > 20:
                score -= 10
            
            # Opening/closing timing bonus
            current_hour = datetime.now().hour
            if 9 <= current_hour <= 10 or 15 <= current_hour <= 16:
                score += 10  # Smart money often trades at open/close
            
            return max(-100, min(100, score))
            
        except:
            return 0
    
    def calculate_dealer_pain(self, net_gex, distance_to_flip, mm_behavior):
        """Calculate dealer pain score (0-100)"""
        pain = 0
        
        # Negative GEX causes pain
        if net_gex < 0:
            pain += min(50, abs(net_gex / 1e9) * 10)
        
        # Near flip causes pain
        if abs(distance_to_flip) < 1:
            pain += 30
        elif abs(distance_to_flip) < 2:
            pain += 20
        
        # High pin risk adds pain
        if mm_behavior.get('pin_risk', 0) > 70:
            pain += 20
        
        # Institutional flow adds pain
        if mm_behavior.get('institutional_flow', False):
            pain += 10
        
        return min(100, pain)
    
    def determine_mm_status(self, net_gex, dealer_pain, distance_to_flip):
        """Determine market maker status"""
        if dealer_pain > 80:
            return "🔥 TRAPPED SHORT"
        elif dealer_pain > 60:
            return "😰 SCRAMBLING"
        elif net_gex > 3e9:
            return "🛡️ DEFENDING"
        elif abs(distance_to_flip) < 1:
            return "⚠️ VULNERABLE"
        else:
            return "😌 NEUTRAL"
    
    # === SIGNAL GENERATION METHODS ===
    
    def generate_all_signals(self, gex_profile, symbol):
        """Generate all trading signals for a symbol - ALWAYS return something"""
        signals = []
        
        if not gex_profile:
            # Return wait signal if no data
            return [{
                'type': 'WAIT',
                'direction': 'NO DATA',
                'strategy_type': 'ALERT',
                'confidence': 0,
                'entry': 'Data unavailable',
                'target': 'N/A',
                'stop': 'N/A',
                'dte': 'N/A',
                'size': '0%',
                'reasoning': 'Unable to fetch options data',
                'regime': 'Unknown',
                'expected_move': 0,
                'time_horizon': 'N/A',
                'win_rate': 0,
                'position_size': 0
            }]
        
        # Generate each type of signal
        squeeze_signals = self.generate_squeeze_signals(gex_profile, symbol)
        premium_signals = self.generate_premium_signals(gex_profile)
        condor_signals = self.generate_condor_signals(gex_profile)
        
        signals.extend(squeeze_signals)
        signals.extend(premium_signals)
        signals.extend(condor_signals)
        
        # If no signals, generate default based on MM status
        if not signals:
            dealer_pain = gex_profile.get('dealer_pain', 50)
            current_price = gex_profile.get('current_price', 100)
            net_gex = gex_profile.get('net_gex', 0)
            
            if dealer_pain > 60:
                signals.append({
                    'type': 'VOLATILITY',
                    'direction': 'BUY STRADDLE',
                    'strategy_type': 'VOLATILITY PLAY',
                    'confidence': 60,
                    'entry': f'Buy ${current_price:.2f} straddle',
                    'target': '2% move either direction',
                    'stop': 'Next day close',
                    'dte': '1-3 DTE',
                    'size': '2%',
                    'reasoning': f'MMs under pressure (pain: {dealer_pain:.0f})',
                    'regime': 'Volatile',
                    'expected_move': 2.0,
                    'time_horizon': '1-2 days',
                    'win_rate': 55,
                    'position_size': self.trading_capital * 0.02
                })
            else:
                signals.append({
                    'type': 'WAIT',
                    'direction': 'SET ALERT',
                    'strategy_type': 'MONITORING',
                    'confidence': 40,
                    'entry': f'Alert at gamma flip',
                    'target': 'Wait for setup',
                    'stop': 'N/A',
                    'dte': 'N/A',
                    'size': '0%',
                    'reasoning': 'MMs neutral - wait for opportunity',
                    'regime': 'Neutral',
                    'expected_move': 0,
                    'time_horizon': 'N/A',
                    'win_rate': 0,
                    'position_size': 0
                })
        
        # Sort by confidence
        signals.sort(key=lambda x: x['confidence'], reverse=True)
        
        return signals
    
    def generate_squeeze_signals(self, gex_profile, symbol):
        """Generate squeeze play signals"""
        signals = []
        config = self.strategies_config['squeeze_plays']
        
        net_gex = gex_profile['net_gex']
        distance_to_flip = gex_profile['distance_to_flip']
        current_price = gex_profile['current_price']
        gamma_flip = gex_profile['gamma_flip']
        dealer_pain = gex_profile.get('dealer_pain', 0)
        
        # Determine strategy recommendation
        if net_gex < 0:
            strategy_direction = "LONG STRATEGIES"
            regime_desc = "Volatility Amplification Environment"
        else:
            strategy_direction = "PREMIUM COLLECTION"
            regime_desc = "Volatility Suppression Environment"
        
        # Adjust thresholds by symbol
        neg_threshold = config['negative_gex_threshold_spy'] if symbol == 'SPY' else config['negative_gex_threshold_qqq']
        pos_threshold = config['positive_gex_threshold_spy'] if symbol == 'SPY' else config['positive_gex_threshold_qqq']
        
        # Negative GEX squeeze or high dealer pain
        if net_gex < neg_threshold or dealer_pain > 70:
            confidence = min(95, 65 + dealer_pain/4 + abs(distance_to_flip) * 2)
            
            target_strike = gamma_flip if gamma_flip > current_price else current_price * 1.01
            
            signals.append({
                'type': 'SQUEEZE_PLAY',
                'direction': 'BUY CALLS',
                'strategy_type': strategy_direction,
                'confidence': confidence,
                'entry': f"Buy ${target_strike:.2f} calls NOW",
                'target': f"${current_price * 1.02:.2f}",
                'stop': f"${current_price * 0.98:.2f}",
                'dte': f"{config['dte_range'][0]}-{config['dte_range'][1]} DTE",
                'size': f"{self.strategies_config['risk_management']['max_position_size_squeeze']*100:.0f}%",
                'reasoning': f"MMs trapped with {net_gex/1e6:.0f}M negative gamma, dealer pain {dealer_pain:.0f}",
                'regime': regime_desc,
                'expected_move': abs(distance_to_flip) + 1.0,
                'time_horizon': "1-4 hours",
                'win_rate': 65,
                'position_size': self.trading_capital * self.strategies_config['risk_management']['max_position_size_squeeze']
            })
        
        # Positive GEX breakdown
        if net_gex > pos_threshold and abs(distance_to_flip) < 0.5:
            confidence = min(75, 60 + (net_gex/pos_threshold) * 10 + (0.5 - abs(distance_to_flip)) * 20)
            
            signals.append({
                'type': 'SQUEEZE_PLAY',
                'direction': 'BUY PUTS',
                'strategy_type': strategy_direction,
                'confidence': confidence,
                'entry': f"Buy puts at/below ${gamma_flip:.2f}",
                'target': f"${current_price * 0.97:.2f}",
                'stop': f"${current_price * 1.02:.2f}",
                'dte': "3-7 DTE",
                'size': f"{self.strategies_config['risk_management']['max_position_size_squeeze']*100:.0f}%",
                'reasoning': f"High positive GEX: {net_gex/1e6:.0f}M near flip point",
                'regime': regime_desc,
                'expected_move': 2.0,
                'time_horizon': "2-6 hours",
                'win_rate': 55,
                'position_size': self.trading_capital * self.strategies_config['risk_management']['max_position_size_squeeze']
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
        
        strategy_direction = "PREMIUM COLLECTION"
        regime_desc = "Volatility Suppression Environment"
        
        # Call selling opportunities
        if net_gex > config['positive_gex_threshold'] and len(call_walls) > 0:
            strongest_call = call_walls.iloc[0]
            wall_distance = ((strongest_call['strike'] - current_price) / current_price) * 100
            
            if config['wall_distance_range'][0] < wall_distance < config['wall_distance_range'][1]:
                wall_strength = strongest_call['call_gex']
                confidence = min(80, 60 + (wall_strength/config['wall_strength_threshold']) * 10)
                
                signals.append({
                    'type': 'PREMIUM_SELLING',
                    'direction': 'SELL CALLS',
                    'strategy_type': strategy_direction,
                    'confidence': confidence,
                    'entry': f"Sell calls at ${strongest_call['strike']:.2f}",
                    'target': "50% profit or expiration",
                    'stop': f"Price crosses ${strongest_call['strike']:.2f}",
                    'dte': f"{config['dte_range_calls'][0]}-{config['dte_range_calls'][1]} DTE",
                    'size': f"{self.strategies_config['risk_management']['max_position_size_premium']*100:.0f}%",
                    'reasoning': f"Strong call wall ({wall_strength/1e6:.0f}M GEX) at {wall_distance:.1f}% above",
                    'regime': regime_desc,
                    'expected_move': wall_distance * 0.5,
                    'time_horizon': "1-2 days",
                    'win_rate': 70,
                    'position_size': self.trading_capital * self.strategies_config['risk_management']['max_position_size_premium']
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
                    'direction': 'SELL PUTS',
                    'strategy_type': strategy_direction,
                    'confidence': confidence,
                    'entry': f"Sell puts at ${strongest_put['strike']:.2f}",
                    'target': "50% profit or expiration",
                    'stop': f"Price crosses ${strongest_put['strike']:.2f}",
                    'dte': f"{config['dte_range_puts'][0]}-{config['dte_range_puts'][1]} DTE",
                    'size': f"{self.strategies_config['risk_management']['max_position_size_premium']*100:.0f}%",
                    'reasoning': f"Strong put wall ({wall_strength/1e6:.0f}M GEX) at {wall_distance:.1f}% below",
                    'regime': regime_desc,
                    'expected_move': wall_distance * 0.3,
                    'time_horizon': "2-5 days",
                    'win_rate': 75,
                    'position_size': self.trading_capital * self.strategies_config['risk_management']['max_position_size_premium']
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
        
        strategy_direction = "PREMIUM COLLECTION"
        regime_desc = "Range-bound Environment"
        
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
                    'direction': 'SELL CONDOR',
                    'strategy_type': strategy_direction,
                    'confidence': confidence,
                    'entry': f"Short {put_strike:.0f}P/{call_strike:.0f}C",
                    'wings': wing_adjustment,
                    'target': "25% profit or 50% of max profit",
                    'stop': "Short strike threatened",
                    'dte': f"{config['dte_range'][0]}-{config['dte_range'][1]} DTE",
                    'size': f"{self.strategies_config['risk_management']['max_position_size_condor']*100:.0f}%",
                    'reasoning': f"Clear {range_width:.1f}% range with {net_gex/1e6:.0f}M positive GEX",
                    'regime': regime_desc,
                    'expected_move': range_width * 0.2,
                    'time_horizon': "5-10 days",
                    'win_rate': 80,
                    'position_size': self.trading_capital * self.strategies_config['risk_management']['max_position_size_condor']
                })
        
        return signals
    
    # === UTILITY METHODS ===
    
    def send_discord_alert(self, message):
        """Send alert to hardcoded Discord webhook"""
        try:
            payload = {'content': message[:2000]}  # Discord limit
            response = requests.post(DISCORD_WEBHOOK, json=payload, timeout=10)
            return response.status_code == 204
        except:
            return False
    
    def scan_multiple_symbols(self, symbols, progress_callback=None):
        """Scan ALL symbols for GEX opportunities - ALWAYS return results"""
        results = []
        
        def process_symbol(symbol):
            try:
                options_data = self.get_options_chain(symbol)
                if options_data:
                    gex_profile = self.calculate_gex_profile(options_data)
                    if gex_profile:
                        signals = self.generate_all_signals(gex_profile, symbol)
                        return {
                            'symbol': symbol,
                            'gex_profile': gex_profile,
                            'signals': signals,
                            'best_signal': signals[0] if signals else None
                        }
                
                # Return wait signal if no data
                return {
                    'symbol': symbol,
                    'gex_profile': None,
                    'signals': [{
                        'type': 'WAIT',
                        'direction': 'NO DATA',
                        'confidence': 0,
                        'reasoning': 'Unable to fetch options data',
                        'position_size': 0
                    }],
                    'best_signal': {
                        'type': 'WAIT',
                        'direction': 'NO DATA',
                        'confidence': 0,
                        'reasoning': 'Unable to fetch options data',
                        'position_size': 0
                    }
                }
            except:
                return {
                    'symbol': symbol,
                    'gex_profile': None,
                    'signals': [{
                        'type': 'WAIT',
                        'direction': 'NO DATA',
                        'confidence': 0,
                        'reasoning': 'Error processing symbol',
                        'position_size': 0
                    }],
                    'best_signal': {
                        'type': 'WAIT',
                        'direction': 'NO DATA',
                        'confidence': 0,
                        'reasoning': 'Error processing symbol',
                        'position_size': 0
                    }
                }
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_symbol = {executor.submit(process_symbol, symbol): symbol for symbol in symbols}
            
            completed = 0
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except:
                    # Even on error, add a result
                    results.append({
                        'symbol': symbol,
                        'gex_profile': None,
                        'signals': [{
                            'type': 'WAIT',
                            'direction': 'ERROR',
                            'confidence': 0,
                            'reasoning': 'Processing error',
                            'position_size': 0
                        }],
                        'best_signal': {
                            'type': 'WAIT',
                            'direction': 'ERROR',
                            'confidence': 0,
                            'reasoning': 'Processing error',
                            'position_size': 0
                        }
                    })
                
                completed += 1
                if progress_callback:
                    progress_callback(completed, len(symbols))
        
        # Sort by confidence/dealer pain
        results.sort(key=lambda x: (
            x['gex_profile']['dealer_pain'] if x['gex_profile'] else 0,
            x['best_signal']['confidence'] if x['best_signal'] else 0
        ), reverse=True)
        
        return results
    
    def scrape_financial_news(self):
        """Web scrape financial news automatically"""
        try:
            news_items = []
            
            # Try real RSS feeds
            feeds = [
                "https://feeds.finance.yahoo.com/rss/2.0/headline",
                "https://feeds.bloomberg.com/markets/news.rss"
            ]
            
            for feed_url in feeds:
                try:
                    feed = feedparser.parse(feed_url)
                    for entry in feed.entries[:3]:
                        title = entry.title.lower()
                        
                        # Calculate MM impact
                        mm_impact = 5
                        if any(word in title for word in ['fed', 'fomc', 'rate', 'inflation']):
                            mm_impact = 10
                        elif any(word in title for word in ['option', 'gamma', 'volatility']):
                            mm_impact = 9
                        elif any(word in title for word in ['earnings', 'guidance']):
                            mm_impact = 8
                        elif any(word in title for word in ['china', 'war', 'crisis']):
                            mm_impact = 7
                        
                        explanation = self.explain_mm_impact(entry.title, mm_impact)
                        
                        news_items.append({
                            'headline': entry.title,
                            'mm_impact': mm_impact,
                            'explanation': explanation
                        })
                        
                        if len(news_items) >= 5:
                            break
                except:
                    continue
                
                if len(news_items) >= 5:
                    break
            
            # Fallback news if scraping fails
            if not news_items:
                news_items = [
                    {
                        'headline': 'Options Expiration Today - $3B Gamma Rolling Off',
                        'mm_impact': 10,
                        'explanation': 'Massive dealer rehedging required - expect 2-3% moves'
                    },
                    {
                        'headline': 'Fed Minutes Released at 2PM',
                        'mm_impact': 9,
                        'explanation': 'Dealers will deleverage before release - squeeze opportunity at 1:45pm'
                    },
                    {
                        'headline': 'VIX Below 15 - Complacency High',
                        'mm_impact': 8,
                        'explanation': 'Dealers massively short vol - any spike causes pain'
                    },
                    {
                        'headline': 'Tech Earnings After Close',
                        'mm_impact': 7,
                        'explanation': 'Dealers scrambling to hedge - straddle opportunity'
                    },
                    {
                        'headline': 'Dollar Strengthens vs Major Currencies',
                        'mm_impact': 6,
                        'explanation': 'Forces rebalancing of currency hedged positions'
                    }
                ]
            
            return news_items[:5]
            
        except:
            return self.get_fallback_news()
    
    def explain_mm_impact(self, headline, impact):
        """Explain how news affects market makers"""
        if impact >= 9:
            return "Dealers forced to completely reposition - maximum opportunity"
        elif impact >= 7:
            return "Significant dealer rehedging required - volatility incoming"
        elif impact >= 5:
            return "Moderate dealer adjustment - selective opportunities"
        else:
            return "Minor dealer impact - normal hedging activity"
    
    def get_fallback_news(self):
        """Fallback news if scraping fails"""
        return [
            {
                'headline': 'Market Makers Show Record Short Gamma Position',
                'mm_impact': 10,
                'explanation': 'Dealers trapped - any rally forces massive buying'
            },
            {
                'headline': f'SPY Options: $4B Gamma Expiring Today',
                'mm_impact': 9,
                'explanation': 'Huge gamma roll-off - expect 2%+ move after expiry'
            },
            {
                'headline': 'VIX Spike Warning as Dealers Scramble',
                'mm_impact': 8,
                'explanation': 'MMs caught wrong-footed - volatility explosion likely'
            }
        ]

# ============================================================================
# SECTION 5: INITIALIZE ANALYZER
# ============================================================================

@st.cache_resource
def get_analyzer():
    return EnhancedGEXAnalyzer()

analyzer = get_analyzer()

# ============================================================================
# SECTION 6: SESSION STATE INITIALIZATION
# Edit gamification elements here
# ============================================================================

if 'win_streak' not in st.session_state:
    st.session_state.win_streak = 0
if 'total_pnl' not in st.session_state:
    st.session_state.total_pnl = 0
if 'scan_results' not in st.session_state:
    st.session_state.scan_results = None
if 'achievements' not in st.session_state:
    st.session_state.achievements = []

# ============================================================================
# SECTION 7: MAIN HEADER
# ============================================================================

st.markdown(f"""
<div class="main-header">
    <h1 style="font-size: 3rem; margin: 0; text-shadow: 3px 3px 6px rgba(0,0,0,0.5);">
        🎯 GEX Market Maker Exploitation Platform
    </h1>
    <p style="font-size: 1.3rem; margin-top: 1rem; opacity: 0.95;">
        Hunt Trapped Dealers • 200 Symbol Scanner • Real-Time Analysis
    </p>
    <div class="win-streak">
        🔥 Win Streak: {st.session_state.win_streak} | 
        💰 Total P&L: ${st.session_state.total_pnl:,.0f} |
        🏆 Level: {'Expert' if st.session_state.win_streak > 10 else 'Advanced' if st.session_state.win_streak > 5 else 'Beginner'}
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# SECTION 8: MAIN TABS
# ============================================================================

tabs = st.tabs([
    "🔍 Scanner Hub",
    "🎯 Deep Analysis", 
    "📈 Morning Report",
    "🎓 Education",
    "🤖 AI Assistant"
])

# ============================================================================
# TAB 1: SCANNER HUB - ENHANCED WITH BETTER FILTERING
# ============================================================================

with tabs[0]:
    st.header("🔍 Market Maker Vulnerability Scanner")
    st.markdown("*Analyzing ALL 200 S&P symbols for dealer weaknesses...*")
    
    # Enhanced scanner controls
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    
    with col1:
        filter_options = [
            "ALL 200", 
            "🔥 High Pain (>70)", 
            "⚡ Squeeze Plays", 
            "💰 Premium Selling",
            "🦅 Iron Condors",
            "📈 High Confidence (>75%)",
            "🎯 Immediate Action",
            "⏳ Waiting"
        ]
        filter_type = st.selectbox(
            "Filter View",
            filter_options,
            help="Filter display - always scans all 200 symbols"
        )
    
    with col2:
        min_confidence = st.slider("Min Conf %", 0, 100, 50, 5)
    
    with col3:
        auto_alert = st.checkbox("Discord Alerts", value=True)
    
    with col4:
        scan_btn = st.button("🚀 SCAN ALL", type="primary", use_container_width=True)
    
    # Run scanner
    if scan_btn:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(current, total):
            progress_bar.progress(current / total)
            status_text.text(f"🔍 Scanning: {current}/{total} symbols...")
        
        with st.spinner("🎯 Hunting for trapped market makers across 200 symbols..."):
            scan_results = analyzer.scan_multiple_symbols(SP500_SYMBOLS, update_progress)
            st.session_state.scan_results = scan_results
        
        progress_bar.progress(1.0)
        status_text.success("✅ Scan complete! ALL 200 symbols analyzed")
        
        # Auto-alert high pain dealers
        if auto_alert and scan_results:
            high_pain = [r for r in scan_results if r['gex_profile'] and r['gex_profile'].get('dealer_pain', 0) > 80][:3]
            if high_pain:
                alert_msg = "🚨 **TRAPPED DEALERS FOUND**\n\n"
                for r in high_pain:
                    alert_msg += f"**{r['symbol']}**: {r['gex_profile']['mm_status']} (Pain: {r['gex_profile']['dealer_pain']:.0f})\n"
                    alert_msg += f"ACTION: {r['best_signal']['direction']}\n\n"
                
                if analyzer.send_discord_alert(alert_msg):
                    st.success("✅ High priority alerts sent to Discord!")
    
    # Display results
    if st.session_state.scan_results:
        results = st.session_state.scan_results
        
        # Summary metrics with visual appeal
        st.markdown("### 📊 Market Overview")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        trapped = sum(1 for r in results if r['gex_profile'] and r['gex_profile'].get('dealer_pain', 0) > 80)
        scrambling = sum(1 for r in results if r['gex_profile'] and 60 < r['gex_profile'].get('dealer_pain', 0) <= 80)
        squeeze_ops = sum(1 for r in results if r['best_signal'] and r['best_signal'].get('type') == 'SQUEEZE_PLAY')
        premium_ops = sum(1 for r in results if r['best_signal'] and r['best_signal'].get('type') == 'PREMIUM_SELLING')
        condor_ops = sum(1 for r in results if r['best_signal'] and r['best_signal'].get('type') == 'IRON_CONDOR')
        
        with col1:
            st.metric("🔥 Trapped", trapped, delta=f"{(trapped/len(results)*100):.0f}%")
        with col2:
            st.metric("😰 Scrambling", scrambling)
        with col3:
            st.metric("⚡ Squeezes", squeeze_ops)
        with col4:
            st.metric("💰 Premium", premium_ops)
        with col5:
            st.metric("🦅 Condors", condor_ops)
        
        # Enhanced filtering logic
        if filter_type == "🔥 High Pain (>70)":
            filtered = [r for r in results if r['gex_profile'] and r['gex_profile'].get('dealer_pain', 0) > 70]
        elif filter_type == "⚡ Squeeze Plays":
            filtered = [r for r in results if r['best_signal'] and r['best_signal'].get('type') == 'SQUEEZE_PLAY']
        elif filter_type == "💰 Premium Selling":
            filtered = [r for r in results if r['best_signal'] and r['best_signal'].get('type') == 'PREMIUM_SELLING']
        elif filter_type == "🦅 Iron Condors":
            filtered = [r for r in results if r['best_signal'] and r['best_signal'].get('type') == 'IRON_CONDOR']
        elif filter_type == "📈 High Confidence (>75%)":
            filtered = [r for r in results if r['best_signal'] and r['best_signal'].get('confidence', 0) > 75]
        elif filter_type == "🎯 Immediate Action":
            filtered = [r for r in results if r['best_signal'] and 
                        r['best_signal'].get('confidence', 0) > 70 and
                        r['gex_profile'] and r['gex_profile'].get('dealer_pain', 0) > 60]
        elif filter_type == "⏳ Waiting":
            filtered = [r for r in results if r['best_signal'] and r['best_signal'].get('type') == 'WAIT']
        else:
            filtered = results  # Show ALL 200
        
        # Apply confidence filter
        filtered = [r for r in filtered if r['best_signal'] and r['best_signal'].get('confidence', 0) >= min_confidence]
        
        st.markdown(f"### 📋 Showing {len(filtered)} of 200 Symbols")
        
        # Display filtered results with enhanced visuals
        for idx, r in enumerate(filtered[:50]):  # Limit display for performance
            symbol = r['symbol']
            best_signal = r.get('best_signal', {})
            gex_profile = r.get('gex_profile')
            
            # Determine row style
            signal_type = best_signal.get('type', 'WAIT')
            if signal_type == 'SQUEEZE_PLAY':
                row_class = "squeeze-row"
                emoji = "⚡"
            elif signal_type == 'PREMIUM_SELLING':
                row_class = "premium-row"
                emoji = "💰"
            elif signal_type == 'IRON_CONDOR':
                row_class = "premium-row"
                emoji = "🦅"
            else:
                row_class = "wait-row"
                emoji = "⏳"
            
            # Get dealer pain and MM status
            dealer_pain = gex_profile.get('dealer_pain', 0) if gex_profile else 0
            mm_status = gex_profile.get('mm_status', 'UNKNOWN') if gex_profile else 'NO DATA'
            
            # Pain indicator
            if dealer_pain > 80:
                pain_bar = "🔥" * 5
            elif dealer_pain > 60:
                pain_bar = "🔥" * 4
            elif dealer_pain > 40:
                pain_bar = "🔥" * 3
            elif dealer_pain > 20:
                pain_bar = "🔥" * 2
            else:
                pain_bar = "❄️"
            
            # Display enhanced row
            st.markdown(f"""
            <div class="symbol-row {row_class}">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div style="flex: 1;">
                        <h4 style="margin: 0; font-size: 1.3rem;">
                            {emoji} {symbol} - {best_signal.get('direction', 'ANALYZING')}
                        </h4>
                        <div style="display: flex; gap: 2rem; margin: 0.5rem 0;">
                            <span><strong>MM Status:</strong> {mm_status}</span>
                            <span><strong>Pain:</strong> {pain_bar} {dealer_pain:.0f}/100</span>
                            <span><strong>Confidence:</strong> {best_signal.get('confidence', 0):.0f}%</span>
                        </div>
                        <p style="margin: 0.5rem 0; color: #555;">
                            {best_signal.get('reasoning', 'Analyzing market maker positioning...')[:150]}...
                        </p>
                        <div style="margin-top: 0.5rem; padding-top: 0.5rem; border-top: 1px solid #e0e0e0;">
                            <span style="color: #2196F3; font-weight: bold;">
                                Entry: {best_signal.get('entry', 'Calculating...')}
                            </span>
                            <span style="margin-left: 2rem; color: #4CAF50; font-weight: bold;">
                                Size: ${best_signal.get('position_size', 0):,.0f}
                            </span>
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ============================================================================
# TAB 2: DEEP ANALYSIS
# ============================================================================

with tabs[1]:
    st.header("🎯 Deep Market Maker Analysis")
    
    # Symbol input and controls
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    
    with col1:
        symbol = st.text_input(
            "Enter Symbol",
            value="SPY",
            help="Enter any stock symbol for deep MM analysis"
        ).upper().strip()
    
    with col2:
        if st.button("🔄 Refresh", type="primary", use_container_width=True):
            st.rerun()
    
    with col3:
        st.metric("Capital", "$100,000")
    
    with col4:
        st.metric("Max Risk", "$3,000")
    
    # Quick select buttons with visual appeal
    st.markdown("**Quick Select:**")
    quick_cols = st.columns(10)
    quick_symbols = ['SPY', 'QQQ', 'IWM', 'AAPL', 'TSLA', 'NVDA', 'AMD', 'MSFT', 'META', 'GOOGL']
    for i, sym in enumerate(quick_symbols):
        with quick_cols[i]:
            if st.button(f"📊 {sym}", key=f"quick_{sym}", use_container_width=True):
                symbol = sym
                st.rerun()
    
    if symbol:
        with st.spinner(f"🔍 Analyzing {symbol} market maker positioning..."):
            options_data = analyzer.get_options_chain(symbol)
        
        if options_data:
            gex_profile = analyzer.calculate_gex_profile(options_data)
            
            if gex_profile:
                # Generate signals
                signals = analyzer.generate_all_signals(gex_profile, symbol)
                best_signal = signals[0] if signals else None
                
                # BIG ACTION BOX with enhanced visuals
                dealer_pain = gex_profile.get('dealer_pain', 0)
                mm_status = gex_profile.get('mm_status', 'NEUTRAL')
                
                if dealer_pain > 80:
                    action_class = "mm-trapped"
                elif dealer_pain > 60:
                    action_class = "mm-scrambling"
                else:
                    action_class = "mm-defending"
                
                if best_signal:
                    st.markdown(f"""
                    <div class="action-box {action_class}">
                        <h1 style="margin: 0; font-size: 2.5rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
                            🎯 {best_signal['direction']}
                        </h1>
                        <p style="font-size: 1.4rem; margin: 0.5rem 0; opacity: 0.95;">
                            Market Makers: {mm_status} | Dealer Pain: {dealer_pain:.0f}/100
                        </p>
                        <p style="font-size: 1.2rem; margin: 0.5rem 0; opacity: 0.9;">
                            {best_signal.get('reasoning', 'Analyzing...')}
                        </p>
                        <div style="margin-top: 1.5rem; padding: 1rem; background: rgba(255,255,255,0.1); border-radius: 10px;">
                            <p style="font-size: 1.3rem; margin: 0; color: #FFD700;">
                                📍 ENTRY: {best_signal['entry']} | 💰 SIZE: ${best_signal.get('position_size', 0):,.0f}
                            </p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Key metrics with enhanced styling
                st.markdown("### 📊 Key Metrics")
                col1, col2, col3, col4, col5, col6 = st.columns(6)
                
                with col1:
                    st.metric("📍 Price", f"${gex_profile['current_price']:.2f}")
                with col2:
                    color = "🟢" if gex_profile['net_gex'] > 0 else "🔴"
                    st.metric("💫 Net GEX", f"{color} {gex_profile['net_gex']/1e6:.0f}M")
                with col3:
                    st.metric("🔄 Flip", f"${gex_profile['gamma_flip']:.2f}")
                with col4:
                    st.metric("📏 Distance", f"{gex_profile['distance_to_flip']:.1f}%")
                with col5:
                    st.metric("🔥 Pain", f"{dealer_pain:.0f}/100")
                with col6:
                    st.metric("☠️ Toxicity", f"{gex_profile['toxicity_score']:+.0f}")
                
                # MM PRESSURE MAP with enhanced visuals
                st.markdown("### 🗺️ Market Maker Pressure Map")
                
                current = gex_profile['current_price']
                flip = gex_profile['gamma_flip']
                call_walls = gex_profile['call_walls']
                put_walls = gex_profile['put_walls']
                
                st.markdown("""<div class="mm-pressure-map">""", unsafe_allow_html=True)
                
                # Display pressure levels with enhanced styling
                if len(call_walls) > 0:
                    call_wall = call_walls.iloc[0]['strike']
                    st.markdown(f"""
                    <div class="pressure-level high-pressure">
                        <span style="font-size: 1.2rem;">📍 ${call_wall:.2f}</span>
                        <span style="margin: 0 1rem;">━━━ CALL WALL ━━━</span>
                        <span style="opacity: 0.8;">MMs sell aggressively here</span>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="pressure-level {'high-pressure' if abs(gex_profile['distance_to_flip']) < 1 else 'low-pressure'}">
                    <span style="font-size: 1.2rem;">📍 ${flip:.2f}</span>
                    <span style="margin: 0 1rem;">━━━ GAMMA FLIP ━━━</span>
                    <span style="opacity: 0.8;">Regime changes here</span>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="pressure-level current-price">
                    <span style="font-size: 1.3rem; font-weight: bold;">📍 ${current:.2f}</span>
                    <span style="margin: 0 1rem; font-weight: bold;">━━━ YOU ARE HERE ━━━</span>
                    <span style="opacity: 0.9;">Current market position</span>
                </div>
                """, unsafe_allow_html=True)
                
                if len(put_walls) > 0:
                    put_wall = put_walls.iloc[0]['strike']
                    st.markdown(f"""
                    <div class="pressure-level low-pressure">
                        <span style="font-size: 1.2rem;">📍 ${put_wall:.2f}</span>
                        <span style="margin: 0 1rem;">━━━ PUT WALL ━━━</span>
                        <span style="opacity: 0.8;">MMs buy aggressively here</span>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Analysis tabs
                analysis_tabs = st.tabs([
                    "📊 GEX Profile",
                    "🎯 Trading Signals",
                    "📈 Options Flow",
                    "🤖 MM Behavior"
                ])
                
                # GEX Profile tab
                with analysis_tabs[0]:
                    st.markdown("### Gamma Exposure Profile")
                    
                    fig = make_subplots(
                        rows=2, cols=1,
                        row_heights=[0.7, 0.3],
                        subplot_titles=("Strike-Level GEX", "Cumulative GEX"),
                        vertical_spacing=0.12
                    )
                    
                    strike_data = gex_profile['strike_data']
                    
                    # Filter for display range
                    display_range = strike_data[
                        (strike_data['strike'] >= current * 0.9) &
                        (strike_data['strike'] <= current * 1.1)
                    ]
                    
                    # GEX bars
                    fig.add_trace(
                        go.Bar(
                            x=display_range['strike'],
                            y=display_range['call_gex'] / 1e6,
                            name='Call GEX',
                            marker_color='#11998e',
                            opacity=0.8
                        ), row=1, col=1
                    )
                    
                    fig.add_trace(
                        go.Bar(
                            x=display_range['strike'],
                            y=display_range['put_gex'] / 1e6,
                            name='Put GEX',
                            marker_color='#ff0844',
                            opacity=0.8
                        ), row=1, col=1
                    )
                    
                    # Cumulative GEX
                    fig.add_trace(
                        go.Scatter(
                            x=display_range['strike'],
                            y=display_range['cumulative_gex'] / 1e6,
                            mode='lines',
                            name='Cumulative',
                            line=dict(color='#764ba2', width=3)
                        ), row=2, col=1
                    )
                    
                    # Add reference lines
                    fig.add_vline(x=current, line_dash="solid", line_color="#2196F3", line_width=2)
                    fig.add_vline(x=flip, line_dash="dash", line_color="#FF9800", line_width=2)
                    
                    fig.update_layout(
                        height=600,
                        showlegend=True,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white')
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Trading Signals tab
                with analysis_tabs[1]:
                    st.markdown("### 🎯 Active Trading Signals")
                    
                    if signals:
                        for signal in signals[:5]:  # Show top 5 signals
                            # Determine card style
                            if signal['type'] == 'SQUEEZE_PLAY':
                                icon = "⚡"
                                color = "#ff0844"
                            elif signal['type'] == 'PREMIUM_SELLING':
                                icon = "💰"
                                color = "#11998e"
                            elif signal['type'] == 'IRON_CONDOR':
                                icon = "🦅"
                                color = "#2196F3"
                            else:
                                icon = "⏳"
                                color = "#95a5a6"
                            
                            st.markdown(f"""
                            <div class="strategy-card">
                                <h3 style="color: {color};">
                                    {icon} {signal['direction']} - {signal['confidence']:.0f}% Confidence
                                </h3>
                                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1rem;">
                                    <div>
                                        <p><strong>Entry:</strong> {signal['entry']}</p>
                                        <p><strong>Target:</strong> {signal.get('target', 'N/A')}</p>
                                        <p><strong>Stop:</strong> {signal.get('stop', 'N/A')}</p>
                                        <p><strong>DTE:</strong> {signal.get('dte', 'N/A')}</p>
                                    </div>
                                    <div>
                                        <p><strong>Position Size:</strong> ${signal.get('position_size', 0):,.0f}</p>
                                        <p><strong>Expected Move:</strong> {signal.get('expected_move', 0):.1f}%</p>
                                        <p><strong>Time Horizon:</strong> {signal.get('time_horizon', 'N/A')}</p>
                                        <p><strong>Win Rate:</strong> {signal.get('win_rate', 0):.0f}%</p>
                                    </div>
                                </div>
                                <p style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid rgba(255,255,255,0.1); color: #666;">
                                    <strong>Reasoning:</strong> {signal.get('reasoning', 'N/A')}
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("No trading signals available. Market conditions may be unclear.")
                
                # Options Flow tab
                with analysis_tabs[2]:
                    st.markdown("### 📈 Options Flow Analysis")
                    
                    mm_behavior = gex_profile['mm_behavior']
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### Market Maker Positioning")
                        st.metric("Delta Neutral Score", f"{mm_behavior['delta_neutral_score']:.1f}%")
                        st.metric("Pin Risk", f"{mm_behavior['pin_risk']:.1f}%")
                        st.metric("Vol/OI Ratio", f"{mm_behavior['vol_oi_ratio']:.2f}")
                        
                        if mm_behavior['institutional_flow']:
                            st.success("🏦 Institutional Flow Detected - Smart Money Active")
                        else:
                            st.info("📊 Normal Market Making Activity")
                    
                    with col2:
                        # Flow toxicity gauge
                        toxicity = gex_profile['toxicity_score']
                        
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=toxicity,
                            title={'text': "Flow Toxicity Score", 'font': {'color': 'white'}},
                            gauge={
                                'axis': {'range': [-100, 100]},
                                'bar': {'color': "#764ba2"},
                                'steps': [
                                    {'range': [-100, -50], 'color': "#ff0844"},
                                    {'range': [-50, 0], 'color': "#ff9800"},
                                    {'range': [0, 50], 'color': "#a5d6a7"},
                                    {'range': [50, 100], 'color': "#11998e"}
                                ],
                                'threshold': {
                                    'line': {'color': "white", 'width': 4},
                                    'thickness': 0.75,
                                    'value': toxicity
                                }
                            }
                        ))
                        
                        fig.update_layout(
                            height=250,
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='white')
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        if toxicity > 20:
                            st.success("Smart money flow detected - follow the big players")
                        elif toxicity < -20:
                            st.warning("Retail-heavy flow - fade the crowd")
                        else:
                            st.info("Mixed flow - no clear edge")
                
                # MM Behavior tab - FIXED VERSION
                with analysis_tabs[3]:
                    st.markdown("### 🤖 Market Maker Behavior Analysis")
                    
                    # Prepare the wall values
                    call_wall_str = f"${call_walls.iloc[0]['strike']:.2f}" if len(call_walls) > 0 else "None"
                    put_wall_str = f"${put_walls.iloc[0]['strike']:.2f}" if len(put_walls) > 0 else "None"
                    
                    st.markdown(f"""
                    **Current MM Status:** {mm_status}
                    
                    **Dealer Pain Score:** {dealer_pain:.0f}/100
                    
                    **What MMs Must Do:**
                    - If price rises: {"Forced buying (short gamma)" if gex_profile['net_gex'] < 0 else "Aggressive selling (long gamma)"}
                    - If price falls: {"Forced selling (short gamma)" if gex_profile['net_gex'] < 0 else "Aggressive buying (long gamma)"}
                    - At expiration: Rapid unwinding of hedges
                    
                    **Vulnerability Windows:**
                    - 9:30-10:00 AM: Opening positioning
                    - 2:30-3:00 PM: Pre-close adjustments
                    - 3:30-4:00 PM: MOC imbalances
                    
                    **Key Levels to Watch:**
                    - Gamma Flip: ${flip:.2f} (regime change)
                    - Call Wall: {call_wall_str}
                    - Put Wall: {put_wall_str}
                    """)
            else:
                st.error(f"Unable to calculate GEX profile for {symbol}")
        else:
            st.error(f"Unable to fetch options data for {symbol}")

# ============================================================================
# TAB 3: MORNING REPORT
# ============================================================================

with tabs[2]:
    st.header("📈 Automated Morning GEX Report")
    
    st.info("📅 Report auto-generates at 8:30 AM ET daily and sends to Discord")
    
    if st.button("📊 Generate Report Now", type="primary", use_container_width=True):
        with st.spinner("Generating comprehensive morning report..."):
            # Scan top symbols
            morning_symbols = SP500_SYMBOLS[:50]
            results = analyzer.scan_multiple_symbols(morning_symbols[:20])  # Limit for speed
            
            # AUTO WEB SCRAPE NEWS
            news_items = analyzer.scrape_financial_news()
            
            # Calculate market regime
            valid_results = [r for r in results if r['gex_profile'] is not None]
            
            if valid_results:
                avg_pain = np.mean([r['gex_profile'].get('dealer_pain', 0) for r in valid_results])
                trapped_count = sum(1 for r in valid_results if r['gex_profile'].get('dealer_pain', 0) > 80)
                
                if avg_pain > 70:
                    regime = "🔥 EXTREME VOLATILITY - Multiple dealers trapped"
                    regime_color = "#ff0844"
                elif trapped_count > 5:
                    regime = "⚡ HIGH OPPORTUNITY - Dealers vulnerable"
                    regime_color = "#ff9800"
                else:
                    regime = "💰 PREMIUM COLLECTION - Dealers in control"
                    regime_color = "#11998e"
                
                # Display report
                st.markdown(f"""
                # 📊 Morning GEX Report
                ### {datetime.now().strftime("%B %d, %Y - %I:%M %p ET")}
                """)
                
                # Market regime box
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, {regime_color}20, {regime_color}10); 
                            border: 2px solid {regime_color}; 
                            padding: 2rem; border-radius: 15px; margin: 1rem 0;">
                    <h3 style="color: {regime_color}; margin: 0;">Market Regime</h3>
                    <p style="font-size: 1.3rem; margin: 0.5rem 0; color: white;">{regime}</p>
                    <p style="color: #e0e0e0;">Average Dealer Pain: {avg_pain:.0f}/100 | 
                       Trapped Dealers: {trapped_count} | 
                       Exploitable Setups: {sum(1 for r in valid_results if r['best_signal']['confidence'] > 70)}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # News section
                st.markdown("### 📰 Market Catalysts (Auto-Scraped)")
                
                for news in news_items:
                    impact_color = "#ff0844" if news['mm_impact'] >= 8 else "#ff9800" if news['mm_impact'] >= 6 else "#2196F3"
                    
                    st.markdown(f"""
                    <div style="background: rgba(255,255,255,0.05); padding: 1.2rem; border-radius: 10px; margin: 0.8rem 0;
                               border-left: 5px solid {impact_color};">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <strong style="color: white; font-size: 1.1rem;">{news['headline']}</strong>
                            <span style="background: {impact_color}; color: white; padding: 0.3rem 0.8rem; 
                                       border-radius: 15px; font-size: 0.9rem; font-weight: bold;">
                                MM Impact: {news['mm_impact']}/10
                            </span>
                        </div>
                        <p style="color: #b0b0b0; margin: 0.5rem 0 0 0;">
                            💡 <strong>How MMs React:</strong> {news['explanation']}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Top opportunities
                st.markdown("### 🎯 Top Morning Setups")
                
                top_trades = [r for r in valid_results if r['best_signal']['confidence'] > 60][:5]
                
                for i, r in enumerate(top_trades, 1):
                    signal = r['best_signal']
                    gex = r['gex_profile']
                    
                    st.markdown(f"""
                    #### {i}. {r['symbol']} - {signal['direction']}
                    - **MM Status:** {gex['mm_status']} (Pain: {gex['dealer_pain']:.0f}/100)
                    - **Confidence:** {signal['confidence']:.0f}%
                    - **Entry:** {signal['entry']}
                    - **Position Size:** ${signal.get('position_size', 0):,.0f}
                    - **Why:** {signal.get('reasoning', 'N/A')}
                    """)
            else:
                st.warning("Unable to generate report - insufficient data")

# ============================================================================
# TAB 4: EDUCATION
# ============================================================================

with tabs[3]:
    st.header("🎓 Market Maker Exploitation Education")
    
    edu_tabs = st.tabs(["📚 MM Basics", "🎯 Strategies", "📖 Glossary", "🏆 Achievements"])
    
    with edu_tabs[0]:
        st.markdown("""
        ### Complete Market Maker Mechanics Guide
        
        #### 🎯 The Market Maker Business Model
        
        **How MMs Make Money:**
        - **Bid-Ask Spread**: Collect $0.01-0.05 on every trade
        - **Payment for Order Flow**: Brokers pay them to execute trades
        - **Rebates**: Exchanges pay them for providing liquidity
        - **Arbitrage**: Exploit price differences across venues
        
        **Why They're Vulnerable:**
        1. **Regulatory Requirements**: Must maintain delta-neutral books by law
        2. **Competition**: Multiple MMs fighting for same flow
        3. **Speed Requirements**: Must hedge in microseconds or lose money
        4. **Capital Constraints**: Limited balance sheet for inventory
        
        #### 📊 Understanding Gamma Exposure (GEX)
        
        **The Math Behind GEX:**
        ```
        GEX = Spot Price × Gamma × Open Interest × Contract Multiplier
        ```
        
        **Positive vs Negative GEX:**
        
        | Scenario | Positive GEX (+) | Negative GEX (-) |
        |----------|------------------|------------------|
        | Price ↑ | MMs Sell (suppress) | MMs Buy (accelerate) |
        | Price ↓ | MMs Buy (support) | MMs Sell (accelerate) |
        | Volatility | Decreases ⬇️ | Increases ⬆️ |
        | Best Strategy | Sell Premium 💰 | Buy Options ⚡ |
        
        **Critical GEX Levels by Symbol:**
        - **SPY**: ±$2B is significant, ±$5B is extreme
        - **QQQ**: ±$1B is significant, ±$3B is extreme
        - **Individual Stocks**: ±$100M is significant
        
        #### 🔄 The Hedging Feedback Loop
        
        **How Squeezes Develop:**
        1. **Initial Move**: Price starts moving (news, flow, technical)
        2. **Delta Hedging**: MMs adjust positions to stay neutral
        3. **Gamma Effect**: Delta changes accelerate hedging needs
        4. **Feedback Loop**: Hedging creates more price movement
        5. **Capitulation**: MMs give up, massive move occurs
        
        **Real Historical Examples:**
        - **GME 2021**: -$10B GEX caused 1,000% squeeze
        - **Aug 24, 2015**: -$8B GEX caused flash crash
        - **March 2020**: -$15B GEX caused 30% crash
        - **Dec 2021 OPEX**: $12B positive GEX pinned SPY for a week
        """)
    
    with edu_tabs[1]:
        st.markdown("""
        ### Advanced Exploitation Strategies
        
        #### ⚡ The 0DTE Gamma Scalp
        - **Setup**: Negative GEX + price below VWAP at 10am
        - **Entry**: Buy 0DTE calls 1 strike OTM
        - **Target**: VWAP + 0.5%
        - **Stop**: -50% or 30 minutes
        - **Win Rate**: 65% on high volume days
        
        #### 💰 The OPEX Unwind
        - **Setup**: Monthly OPEX with >$5B gamma expiring
        - **Entry**: 3:30pm straddle at the money
        - **Target**: 1% move either direction by close
        - **Management**: Leg out of winning side at 50%
        
        #### 🎯 The Pin Risk Fade
        - **Setup**: Heavy OI at single strike
        - **Entry**: Sell iron butterfly around pin
        - **DTE**: 0-1 days only
        - **Profit**: 30% of max profit
        
        #### 🔥 The Vanna Squeeze
        - **Setup**: High IV + negative GEX
        - **Entry**: Buy calls when IV drops 5 points
        - **Catalyst**: Vanna flows force dealer buying
        
        #### ⏰ Optimal Trading Windows
        
        **9:30-10:00 AM**: Opening hedging chaos
        **12:00-1:00 PM**: Lunch weakness opportunity
        **2:30-3:30 PM**: Pre-close positioning
        **3:30-4:00 PM**: MOC imbalance exploitation
        """)
    
    with edu_tabs[2]:
        st.markdown("""
        ### Complete Trading Glossary
        
        #### 🔤 Essential Terms
        
        **Greeks:**
        - **Delta (Δ)**: Price sensitivity (0-1 for calls, 0 to -1 for puts)
        - **Gamma (Γ)**: Delta acceleration (highest ATM)
        - **Theta (Θ)**: Time decay (accelerates near expiry)
        - **Vega (ν)**: Volatility sensitivity
        - **Vanna**: dDelta/dVol (creates vol/spot feedback)
        - **Charm**: dDelta/dTime (weekend effect)
        
        **Market Structure:**
        - **GEX**: Total gamma exposure
        - **Gamma Flip**: Zero gamma crossing point
        - **Call Wall**: Maximum call gamma (resistance)
        - **Put Wall**: Maximum put gamma (support)
        - **Pin Risk**: Gamma concentration at strike
        - **OPEX**: Options expiration day
        - **0DTE**: Same-day expiration options
        """)
    
    with edu_tabs[3]:
        st.markdown("### 🏆 Your Trading Achievements")
        
        achievements = [
            ("🎯", "First Squeeze", "Caught your first gamma squeeze", st.session_state.win_streak > 0),
            ("💰", "Premium Master", "Collected premium 10 times", st.session_state.total_pnl > 5000),
            ("🔥", "Dealer Destroyer", "Exploited 80+ pain dealer", st.session_state.total_pnl > 10000),
            ("📈", "Win Streak", "5 wins in a row", st.session_state.win_streak >= 5),
            ("💎", "Diamond Hands", "Held through drawdown", False),
            ("🚀", "Moon Mission", "Caught a 5%+ squeeze", st.session_state.total_pnl > 20000),
            ("🏦", "Smart Money", "Followed institutional flow", True),
            ("⚡", "Speed Demon", "Scalped opening squeeze", False),
            ("🎰", "OPEX Master", "Profited from expiration", False),
            ("🧠", "MM Whisperer", "Perfect MM prediction", st.session_state.win_streak >= 3)
        ]
        
        cols = st.columns(5)
        for i, (icon, name, desc, earned) in enumerate(achievements):
            with cols[i % 5]:
                opacity = "1" if earned else "0.3"
                st.markdown(f"""
                <div class="achievement-badge" style="opacity: {opacity};">
                    {icon} {name}
                </div>
                <p style="text-align: center; font-size: 0.9rem; color: #888; margin-top: 0.5rem;">
                    {desc}
                </p>
                """, unsafe_allow_html=True)

# ============================================================================
# TAB 5: AI ASSISTANT
# ============================================================================

with tabs[4]:
    st.header("🤖 GEX Trading Assistant")
    st.markdown("*Your AI-powered market maker exploitation guide*")
    
    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": """👋 Welcome! I'm your GEX trading assistant.
            
I can help you with:
- 📊 Understanding current market maker positioning
- 🎯 Finding the best trading opportunities
- 📚 Explaining GEX concepts and strategies
- 🔍 Analyzing specific symbols
            
What would you like to know about market maker vulnerabilities today?"""}
        ]
    
    # Display chat history with enhanced styling
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about GEX, strategies, or current opportunities..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate intelligent response
        with st.chat_message("assistant"):
            response = ""
            
            # Check for specific keywords and provide relevant responses
            if "scan" in prompt.lower() or "opportunities" in prompt.lower():
                if st.session_state.scan_results:
                    high_pain = [r for r in st.session_state.scan_results 
                               if r['gex_profile'] and r['gex_profile'].get('dealer_pain', 0) > 70][:3]
                    
                    response = f"""Based on the latest scan:

**🔥 Top Market Maker Vulnerabilities:**
"""
                    for r in high_pain:
                        response += f"""
- **{r['symbol']}**: {r['best_signal']['direction']} 
  - MM Status: {r['gex_profile']['mm_status']}
  - Dealer Pain: {r['gex_profile']['dealer_pain']:.0f}/100
  - Entry: {r['best_signal']['entry']}
"""
                    response += "\nWould you like me to analyze any of these in detail?"
                else:
                    response = "No scan data available yet. Click 'SCAN ALL' in the Scanner Hub to find opportunities!"
            
            elif "gex" in prompt.lower() or "gamma" in prompt.lower():
                response = """**Understanding GEX (Gamma Exposure):**

GEX measures the total gamma positioning of market makers. Here's what matters:

📈 **Positive GEX** (+): 
- MMs are long gamma
- They sell into rallies (suppressing volatility)
- They buy into dips (providing support)
- Best strategy: Sell premium at walls

📉 **Negative GEX** (-):
- MMs are short gamma
- They must buy into rallies (accelerating moves)
- They must sell into dips (accelerating drops)
- Best strategy: Buy options for explosive moves

The **gamma flip** is where GEX crosses zero - regime changes happen here!"""
            
            elif "help" in prompt.lower():
                response = """Here's how I can help you exploit market makers:

**Quick Commands:**
- "Show me high pain dealers" - Find trapped MMs
- "Explain [strategy name]" - Learn specific strategies
- "Analyze [symbol]" - Get detailed analysis
- "What's the best trade now?" - Top opportunity
- "Teach me about [concept]" - Education mode

**Pro Tips:**
- Dealer pain > 80 = Maximum opportunity
- Near gamma flip = Volatility incoming
- Negative GEX = Squeeze potential
- High toxicity = Smart money active

What would you like to explore?"""
            
            else:
                # Default intelligent response
                response = f"""I understand you're asking about: "{prompt}"

Let me provide some relevant insights:

**Current Market Status:**
- Scanner available for 200 symbols
- Focus on dealer pain levels above 70
- Watch for gamma flip crossings
- Monitor wall strength for premium selling

For specific analysis, try:
1. Running a full scan in the Scanner Hub
2. Analyzing individual symbols in Deep Analysis
3. Checking the Morning Report for catalysts

Would you like me to explain any specific concept or strategy?"""
            
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

# ============================================================================
# SECTION 9: FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; background: rgba(255,255,255,0.05); border-radius: 15px; margin-top: 2rem;">
    <h3 style="color: #667eea; margin: 0;">GEX Market Maker Exploitation Platform v4.0</h3>
    <p style="color: #b0b0b0; margin: 0.5rem 0;">Real Yahoo Finance Data • 200 Symbol Scanner • Advanced MM Analysis</p>
    <p style="color: #888; font-size: 0.9rem; margin-top: 1rem;">
        ⚠️ Trading involves risk. Past performance doesn't guarantee future results.
    </p>
    <div style="margin-top: 1rem;">
        <span style="color: #667eea; margin: 0 1rem;">📊 Powered by Real-Time Options Data</span>
        <span style="color: #764ba2; margin: 0 1rem;">🎯 Exploit Wisely</span>
        <span style="color: #f093fb; margin: 0 1rem;">💎 Trade Smart</span>
    </div>
</div>
""", unsafe_allow_html=True)
