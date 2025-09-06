#!/usr/bin/env python3
"""
DealerEdge - Professional GEX Market Maker Exploitation Platform
Version 5.0 - Production Ready with Real-Time Alerts & Backtesting
Complete implementation with Discord, Telegram, Email notifications
~4,500 lines
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
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
warnings.filterwarnings('ignore')

# ============================================================================
# SECTION 1: PAGE CONFIGURATION AND CONSTANTS
# ============================================================================

st.set_page_config(
    page_title="🎯 DealerEdge - GEX Trading Platform",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Correct Discord webhook (using 1408... not 1308...)
DISCORD_WEBHOOK = "https://discord.com/api/webhooks/1408901307493777469/BWNr70coUxdgWCBSutC5pDWakBkRxM_lyQbUeh8_5A2zClecULeO909XBwQiwUY-DzId"

# ============================================================================
# SECTION 2: VISUAL STYLING - DEALEREDGE BRANDING
# ============================================================================

st.markdown("""
<style>
    /* === DEALEREDGE BRAND COLORS === */
    :root {
        --primary: #1e3c72;
        --secondary: #2a5298;
        --accent: #667eea;
        --success: #11998e;
        --danger: #ff0844;
        --warning: #ff9800;
        --dark: #0f0c29;
    }
    
    /* === GLOBAL STYLES === */
    .stApp {
        background: linear-gradient(135deg, var(--dark) 0%, var(--primary) 50%, var(--secondary) 100%);
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
    
    /* === DEALEREDGE HEADER === */
    .dealeredge-header {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        padding: 3rem;
        border-radius: 30px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 20px 60px rgba(0,0,0,0.5);
        position: relative;
        overflow: hidden;
        border: 2px solid var(--accent);
    }
    
    .dealeredge-header::before {
        content: "DEALEREDGE";
        position: absolute;
        font-size: 150px;
        opacity: 0.03;
        font-weight: bold;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%) rotate(-15deg);
        white-space: nowrap;
    }
    
    .dealeredge-header h1 {
        background: linear-gradient(90deg, #667eea, #764ba2, #f093fb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradient-shift 3s ease infinite;
    }
    
    @keyframes gradient-shift {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    /* === ACTION BOXES === */
    .action-box {
        background: linear-gradient(135deg, var(--accent) 0%, var(--warning) 100%);
        border-radius: 20px;
        padding: 2.5rem;
        color: white;
        text-align: center;
        margin: 1.5rem 0;
        font-size: 1.6rem;
        font-weight: bold;
        box-shadow: 0 10px 40px rgba(102,126,234,0.4);
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
        box-shadow: 0 15px 50px rgba(102,126,234,0.6);
    }
    
    /* === MM STATUS STYLES === */
    .mm-trapped {
        background: linear-gradient(135deg, var(--danger), #ff4563) !important;
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
        background: linear-gradient(135deg, var(--success), #38ef7d) !important;
    }
    
    .mm-scrambling {
        background: linear-gradient(135deg, var(--warning), #f7b733) !important;
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
        border-left-color: var(--danger) !important;
        background: linear-gradient(90deg, rgba(255,8,68,0.1), rgba(255,255,255,0.95)) !important;
    }
    
    .premium-row {
        border-left-color: var(--success) !important;
        background: linear-gradient(90deg, rgba(17,153,142,0.1), rgba(255,255,255,0.95)) !important;
    }
    
    .wait-row {
        border-left-color: #95a5a6 !important;
        background: linear-gradient(90deg, rgba(149,165,166,0.1), rgba(255,255,255,0.95)) !important;
    }
    
    /* === DEALEREDGE BUTTONS === */
    .stButton > button {
        background: linear-gradient(135deg, var(--accent) 0%, var(--secondary) 100%);
        color: white !important;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 50px;
        font-weight: bold;
        transition: all 0.3s;
        box-shadow: 0 4px 15px rgba(102,126,234,0.4);
        text-transform: uppercase;
        letter-spacing: 1px;
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
        border: 1px solid rgba(102,126,234,0.2);
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.15);
        border-color: var(--accent);
    }
    
    /* === PRESSURE MAP === */
    .mm-pressure-map {
        background: linear-gradient(135deg, var(--dark), #302b63, #24243e);
        border-radius: 20px;
        padding: 2rem;
        color: white;
        font-family: 'Courier New', monospace;
        font-size: 1.1rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.3);
        position: relative;
        border: 1px solid var(--accent);
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
        border: 2px solid var(--danger);
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
        border: 2px solid var(--success);
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
        background: linear-gradient(90deg, var(--accent), var(--secondary), var(--warning));
        animation: gradient-shift 3s ease infinite;
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
        background: linear-gradient(135deg, var(--accent), var(--secondary)) !important;
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
    
    /* === POSITION TRACKER === */
    .position-card {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 15px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid var(--accent);
    }
    
    .profit {
        color: var(--success);
        font-weight: bold;
    }
    
    .loss {
        color: var(--danger);
        font-weight: bold;
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
        background: linear-gradient(135deg, var(--accent), var(--secondary));
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, var(--secondary), var(--accent));
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SECTION 3: SYMBOL LISTS
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
    'DD', 'AMAT', 'INFO', 'ALB', 'DOW', 'LHX', 'KEYS', 'GLW', 'ANSS', 'CDW',
    'XRT'  # Added XRT specifically
][:200]

# ============================================================================
# SECTION 4: ENHANCED ANALYZER WITH POSITION TRACKING & BACKTESTING
# ============================================================================

class DealerEdgeAnalyzer:
    """DealerEdge GEX analyzer with position tracking and backtesting"""
    
    def __init__(self):
        self.risk_free_rate = 0.05
        self.trading_capital = 100000
        self.strategies_config = self.load_strategies_config()
        
    def load_strategies_config(self):
        """Load strategy configurations"""
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
            hist = ticker.history(period="1d", interval="1m")
            if len(hist) > 0:
                return float(hist['Close'].iloc[-1])
            
            hist = ticker.history(period="5d")
            if len(hist) > 0:
                return float(hist['Close'].iloc[-1])
            
            return None
        except:
            return None
    
    def get_historical_data(self, symbol, period="1mo"):
        """Get historical data for backtesting"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period, interval="1h")
            return hist
        except:
            return None
    
    def calculate_vix(self):
        """Get current VIX level"""
        try:
            vix = yf.Ticker("^VIX")
            hist = vix.history(period="1d")
            if len(hist) > 0:
                return float(hist['Close'].iloc[-1])
            return 15.0  # Default VIX
        except:
            return 15.0
    
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
            today = date.today()
            
            for exp_date in exp_dates[:15]:
                try:
                    exp_dt = datetime.strptime(exp_date, '%Y-%m-%d')
                    dte = (exp_dt.date() - today).days
                    
                    if dte <= 0 or dte > 14:
                        continue
                    
                    if symbol in ['SPY', 'QQQ', 'IWM'] and dte <= 5:
                        pass
                    elif dte <= 7:
                        pass
                    elif not focus_weekly and dte <= 14:
                        pass
                    else:
                        continue
                    
                    chain = ticker.option_chain(exp_date)
                    
                    calls = chain.calls.copy()
                    calls = calls[calls['openInterest'] > 0]
                    
                    puts = chain.puts.copy()
                    puts = puts[puts['openInterest'] > 0]
                    
                    if len(calls) == 0 and len(puts) == 0:
                        continue
                    
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
                    
                except:
                    continue
            
            return {
                'chains': all_chains,
                'current_price': current_price,
                'symbol': symbol,
                'data_timestamp': datetime.now()
            }
            
        except:
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
            
            strike_data = {}
            total_options_volume = 0
            
            for exp_date, chain_data in chains.items():
                calls = chain_data['calls']
                puts = chain_data['puts']
                
                total_options_volume += calls['volume'].fillna(0).sum()
                total_options_volume += puts['volume'].fillna(0).sum()
                
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
            
            df = pd.DataFrame.from_dict(strike_data, orient='index')
            df.reset_index(inplace=True)
            df.rename(columns={'index': 'strike'}, inplace=True)
            df = df.sort_values('strike').reset_index(drop=True)
            
            df['net_gex'] = df['call_gex'] + df['put_gex']
            df['cumulative_gex'] = df['net_gex'].cumsum()
            
            gamma_flip = self.find_gamma_flip(df, current_price)
            
            call_walls = df[df['call_gex'] > 0].nlargest(5, 'call_gex')
            put_walls = df[df['put_gex'] < 0].nsmallest(5, 'put_gex')
            
            total_call_gex = float(df['call_gex'].sum())
            total_put_gex = float(df['put_gex'].sum())
            net_gex = total_call_gex + total_put_gex
            total_oi = int(df['call_oi'].sum() + df['put_oi'].sum())
            
            distance_to_flip = ((current_price - gamma_flip) / current_price) * 100
            
            mm_behavior = self.analyze_market_maker_behavior(df, current_price, chains)
            toxicity_score = self.calculate_flow_toxicity(chains, total_options_volume)
            dealer_pain = self.calculate_dealer_pain(net_gex, distance_to_flip, mm_behavior)
            mm_status = self.determine_mm_status(net_gex, dealer_pain, distance_to_flip)
            
            # Get VIX for context
            vix = self.calculate_vix()
            
            # Determine regime
            if net_gex < -1e9:
                regime = "EXTREME_VOLATILITY"
            elif net_gex < 0:
                regime = "NORMAL_VOLATILITY"
            else:
                regime = "LOW_VOLATILITY"
            
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
                'vix': vix,
                'regime': regime,
                'data_timestamp': options_data.get('data_timestamp', datetime.now())
            }
            
        except:
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
            atm_strikes = strike_df[
                (strike_df['strike'] >= current_price * 0.98) &
                (strike_df['strike'] <= current_price * 1.02)
            ]
            
            delta_neutral_score = 0
            if len(atm_strikes) > 0:
                straddle_activity = atm_strikes['call_volume'].sum() + atm_strikes['put_volume'].sum()
                total_volume = strike_df['call_volume'].sum() + strike_df['put_volume'].sum()
                delta_neutral_score = (straddle_activity / max(total_volume, 1)) * 100
            
            pin_risk = 0
            near_strikes = strike_df[
                (strike_df['strike'] >= current_price * 0.99) &
                (strike_df['strike'] <= current_price * 1.01)
            ]
            if len(near_strikes) > 0:
                near_gamma = near_strikes['call_gex'].sum() + abs(near_strikes['put_gex'].sum())
                total_gamma = abs(strike_df['call_gex'].sum()) + abs(strike_df['put_gex'].sum())
                pin_risk = (near_gamma / max(total_gamma, 1)) * 100
            
            spread_activity = 0
            for exp_date, chain_data in chains.items():
                calls = chain_data['calls']
                high_oi_strikes = calls[calls['openInterest'] > calls['openInterest'].quantile(0.8)]
                spread_activity += len(high_oi_strikes)
            
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
            
            large_trades = 0
            weekly_preference = 0
            otm_activity = 0
            small_lots = 0
            
            for exp_date, chain_data in chains.items():
                calls = chain_data['calls']
                puts = chain_data['puts']
                dte = chain_data['dte']
                
                large_call_oi = len(calls[calls['openInterest'] > 1000])
                large_put_oi = len(puts[puts['openInterest'] > 1000])
                large_trades += large_call_oi + large_put_oi
                
                if dte <= 7:
                    weekly_vol = calls['volume'].fillna(0).sum() + puts['volume'].fillna(0).sum()
                    weekly_preference += weekly_vol
                
                current_price = calls['strike'].median()
                far_otm_calls = len(calls[calls['strike'] > current_price * 1.1])
                far_otm_puts = len(puts[puts['strike'] < current_price * 0.9])
                otm_activity += far_otm_calls + far_otm_puts
                
                small_call_vol = len(calls[calls['volume'].fillna(0) < 10])
                small_put_vol = len(puts[puts['volume'].fillna(0) < 10])
                small_lots += small_call_vol + small_put_vol
            
            if large_trades > 5:
                score += 20
            if weekly_preference / max(total_volume, 1) > 0.7:
                score -= 20
            if otm_activity > 10:
                score -= 15
            if small_lots > 20:
                score -= 10
            
            current_hour = datetime.now().hour
            if 9 <= current_hour <= 10 or 15 <= current_hour <= 16:
                score += 10
            
            return max(-100, min(100, score))
            
        except:
            return 0
    
    def calculate_dealer_pain(self, net_gex, distance_to_flip, mm_behavior):
        """Calculate dealer pain score (0-100)"""
        pain = 0
        
        if net_gex < 0:
            pain += min(50, abs(net_gex / 1e9) * 10)
        
        if abs(distance_to_flip) < 1:
            pain += 30
        elif abs(distance_to_flip) < 2:
            pain += 20
        
        if mm_behavior.get('pin_risk', 0) > 70:
            pain += 20
        
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
    
    def generate_all_signals(self, gex_profile, symbol):
        """Generate all trading signals for a symbol"""
        signals = []
        
        if not gex_profile:
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
        
        squeeze_signals = self.generate_squeeze_signals(gex_profile, symbol)
        premium_signals = self.generate_premium_signals(gex_profile)
        condor_signals = self.generate_condor_signals(gex_profile)
        
        signals.extend(squeeze_signals)
        signals.extend(premium_signals)
        signals.extend(condor_signals)
        
        if not signals:
            dealer_pain = gex_profile.get('dealer_pain', 50)
            current_price = gex_profile.get('current_price', 100)
            
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
        
        if net_gex < 0:
            strategy_direction = "LONG STRATEGIES"
            regime_desc = "Volatility Amplification Environment"
        else:
            strategy_direction = "PREMIUM COLLECTION"
            regime_desc = "Volatility Suppression Environment"
        
        neg_threshold = config['negative_gex_threshold_spy'] if symbol == 'SPY' else config['negative_gex_threshold_qqq']
        pos_threshold = config['positive_gex_threshold_spy'] if symbol == 'SPY' else config['positive_gex_threshold_qqq']
        
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
    
    def format_discord_alert(self, symbol, gex_profile, signal):
        """Format alert for Discord with real data"""
        if not gex_profile or not signal:
            return None
        
        # Determine recommendation level
        confidence = signal.get('confidence', 0)
        if confidence > 80:
            rec_level = "⚡ HIGH RECOMMENDATION"
        elif confidence > 65:
            rec_level = "⚡ MODERATE RECOMMENDATION"
        else:
            rec_level = "📊 LOW CONFIDENCE"
        
        # Signal type emoji
        type_emoji = {
            'SQUEEZE_PLAY': '⚡',
            'PREMIUM_SELLING': '💰',
            'IRON_CONDOR': '🦅',
            'VOLATILITY': '🌊',
            'WAIT': '⏳'
        }.get(signal.get('type', 'WAIT'), '📊')
        
        # Format the message
        message = f"""
{rec_level} - {symbol} {signal.get('type', 'Signal').replace('_', ' ').title()}
{signal.get('direction', 'Action')}

🎯 **Trade Setup**              📊 **Market Data**           🌍 **Market Context**
Strategy: {signal.get('strategy_type', 'N/A')}          Spot: ${gex_profile.get('current_price', 0):.2f}            Regime: {gex_profile.get('regime', 'NORMAL')}
Confidence: {confidence}%             Net GEX: {gex_profile.get('net_gex', 0)/1e9:.2f}B         VIX: {gex_profile.get('vix', 15):.1f}
Type: {signal.get('type', 'signal').lower()}          Gamma Flip: ${gex_profile.get('gamma_flip', 0):.2f}      Total GEX: {(gex_profile.get('total_call_gex', 0) + abs(gex_profile.get('total_put_gex', 0)))/1e9:.2f}B

💼 **Trade Details**
Entry: {signal.get('entry', 'N/A')}
Target: {signal.get('target', 'N/A')}
Position Size: {signal.get('size', 'N/A')}

📈 **Expected Performance**
Move: {signal.get('expected_move', 0):.1f}%
Time: {signal.get('time_horizon', 'N/A')}
Win Rate: {signal.get('win_rate', 0)}%

💡 **Analysis**
Reason: {signal.get('reasoning', 'Analysis unavailable')}
Notes: Dealers {gex_profile.get('mm_status', 'neutral')} - {self.get_action_notes(gex_profile, signal)}

⏰ Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S ET')}
"""
        
        return message[:2000]  # Discord limit
    
    def get_action_notes(self, gex_profile, signal):
        """Get actionable notes based on current conditions"""
        dealer_pain = gex_profile.get('dealer_pain', 0)
        net_gex = gex_profile.get('net_gex', 0)
        
        if dealer_pain > 80:
            return "explosive potential, dealers trapped and must capitulate"
        elif dealer_pain > 60:
            return "high volatility expected, dealers scrambling to hedge"
        elif net_gex < 0:
            return "negative gamma environment, moves will accelerate"
        elif net_gex > 3e9:
            return "extreme positive gamma, volatility suppressed"
        else:
            return "normal hedging activity, follow technical levels"
    
    def send_discord_alert(self, message):
        """Send alert to Discord webhook"""
        try:
            payload = {'content': message}
            response = requests.post(DISCORD_WEBHOOK, json=payload, timeout=10)
            return response.status_code == 204
        except Exception as e:
            st.error(f"Discord alert failed: {str(e)}")
            return False
    
    def send_telegram_alert(self, message, chat_id, bot_token):
        """Send alert to Telegram"""
        try:
            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            payload = {
                'chat_id': chat_id,
                'text': message,
                'parse_mode': 'Markdown'
            }
            response = requests.post(url, json=payload, timeout=10)
            return response.status_code == 200
        except:
            return False
    
    def send_email_alert(self, message, to_email, smtp_config):
        """Send email alert"""
        try:
            msg = MIMEMultipart()
            msg['From'] = smtp_config['from_email']
            msg['To'] = to_email
            msg['Subject'] = "DealerEdge Alert - Trading Opportunity"
            
            msg.attach(MIMEText(message, 'plain'))
            
            server = smtplib.SMTP(smtp_config['smtp_server'], smtp_config['smtp_port'])
            server.starttls()
            server.login(smtp_config['from_email'], smtp_config['password'])
            
            server.send_message(msg)
            server.quit()
            
            return True
        except:
            return False
    
    def backtest_strategy(self, symbol, strategy_type, period="1mo"):
        """Backtest a strategy on historical data"""
        try:
            hist_data = self.get_historical_data(symbol, period)
            if hist_data is None or len(hist_data) < 10:
                return None
            
            results = {
                'trades': [],
                'total_return': 0,
                'win_rate': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0
            }
            
            # Simple backtest logic (simplified for demonstration)
            trades = []
            for i in range(1, len(hist_data)):
                price_change = (hist_data['Close'].iloc[i] - hist_data['Close'].iloc[i-1]) / hist_data['Close'].iloc[i-1]
                
                # Simulate entry conditions based on strategy
                if strategy_type == 'squeeze' and price_change < -0.01:
                    # Simulate squeeze entry
                    entry_price = hist_data['Close'].iloc[i]
                    exit_price = hist_data['Close'].iloc[min(i+5, len(hist_data)-1)]
                    trade_return = (exit_price - entry_price) / entry_price
                    
                    trades.append({
                        'entry_date': hist_data.index[i],
                        'exit_date': hist_data.index[min(i+5, len(hist_data)-1)],
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'return': trade_return * 100
                    })
            
            if trades:
                results['trades'] = trades
                returns = [t['return'] for t in trades]
                results['total_return'] = sum(returns)
                results['win_rate'] = len([r for r in returns if r > 0]) / len(returns) * 100
                results['max_drawdown'] = min(returns) if returns else 0
                
                # Calculate Sharpe ratio (simplified)
                if len(returns) > 1:
                    avg_return = np.mean(returns)
                    std_return = np.std(returns)
                    results['sharpe_ratio'] = (avg_return / std_return) * np.sqrt(252) if std_return > 0 else 0
            
            return results
            
        except:
            return None
    
    def scan_multiple_symbols(self, symbols, progress_callback=None):
        """Scan ALL symbols for GEX opportunities"""
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
                }
        
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
        
        results.sort(key=lambda x: (
            x['gex_profile']['dealer_pain'] if x['gex_profile'] else 0,
            x['best_signal']['confidence'] if x['best_signal'] else 0
        ), reverse=True)
        
        return results

# ============================================================================
# SECTION 5: INITIALIZE ANALYZER & SESSION STATE
# ============================================================================

@st.cache_resource
def get_analyzer():
    return DealerEdgeAnalyzer()

analyzer = get_analyzer()

# Session state initialization
if 'win_streak' not in st.session_state:
    st.session_state.win_streak = 0
if 'total_pnl' not in st.session_state:
    st.session_state.total_pnl = 0
if 'scan_results' not in st.session_state:
    st.session_state.scan_results = None
if 'active_positions' not in st.session_state:
    st.session_state.active_positions = []
if 'position_history' not in st.session_state:
    st.session_state.position_history = []
if 'alerts_config' not in st.session_state:
    st.session_state.alerts_config = {
        'discord': True,
        'telegram': False,
        'email': False,
        'price_alerts': []
    }

# ============================================================================
# SECTION 6: POSITION TRACKING FUNCTIONS
# ============================================================================

def add_position(symbol, entry_price, size, strategy, signal_data):
    """Add a new position to tracking"""
    position = {
        'id': len(st.session_state.active_positions) + 1,
        'symbol': symbol,
        'entry_price': entry_price,
        'current_price': entry_price,
        'size': size,
        'strategy': strategy,
        'entry_time': datetime.now(),
        'status': 'OPEN',
        'pnl': 0,
        'pnl_percent': 0,
        'target': signal_data.get('target', 'N/A'),
        'stop': signal_data.get('stop', 'N/A'),
        'signal_confidence': signal_data.get('confidence', 0)
    }
    st.session_state.active_positions.append(position)
    return position

def update_positions():
    """Update all active positions with current prices"""
    for position in st.session_state.active_positions:
        if position['status'] == 'OPEN':
            current_price = analyzer.get_current_price(position['symbol'])
            if current_price:
                position['current_price'] = current_price
                position['pnl'] = (current_price - position['entry_price']) * position['size']
                position['pnl_percent'] = ((current_price - position['entry_price']) / position['entry_price']) * 100

def close_position(position_id, exit_price=None):
    """Close a position"""
    for position in st.session_state.active_positions:
        if position['id'] == position_id:
            if exit_price is None:
                exit_price = position['current_price']
            
            position['exit_price'] = exit_price
            position['exit_time'] = datetime.now()
            position['status'] = 'CLOSED'
            position['final_pnl'] = (exit_price - position['entry_price']) * position['size']
            position['final_pnl_percent'] = ((exit_price - position['entry_price']) / position['entry_price']) * 100
            
            # Update win streak and total P&L
            if position['final_pnl'] > 0:
                st.session_state.win_streak += 1
            else:
                st.session_state.win_streak = 0
            
            st.session_state.total_pnl += position['final_pnl']
            
            # Move to history
            st.session_state.position_history.append(position)
            st.session_state.active_positions.remove(position)
            
            return position
    return None

# ============================================================================
# SECTION 7: MAIN HEADER
# ============================================================================

st.markdown(f"""
<div class="dealeredge-header">
    <h1 style="font-size: 3.5rem; margin: 0; font-weight: 900;">
        DEALEREDGE
    </h1>
    <p style="font-size: 1.4rem; margin-top: 0.5rem; opacity: 0.9; letter-spacing: 2px;">
        PROFESSIONAL GEX TRADING PLATFORM
    </p>
    <div class="win-streak">
        🔥 Win Streak: {st.session_state.win_streak} | 
        💰 Total P&L: ${st.session_state.total_pnl:,.0f} |
        📊 Active Positions: {len([p for p in st.session_state.active_positions if p['status'] == 'OPEN'])}
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# SECTION 8: MAIN TABS
# ============================================================================

tabs = st.tabs([
    "🔍 Scanner",
    "🎯 Analysis", 
    "📊 Positions",
    "⚡ Alerts",
    "🧪 Backtest",
    "📈 Report",
    "🎓 Education"
])

# ============================================================================
# TAB 1: SCANNER HUB
# ============================================================================

with tabs[0]:
    st.header("🔍 Market Maker Vulnerability Scanner")
    
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
        filter_type = st.selectbox("Filter View", filter_options)
    
    with col2:
        min_confidence = st.slider("Min Conf %", 0, 100, 50, 5)
    
    with col3:
        auto_alert = st.checkbox("Auto Alert", value=True)
    
    with col4:
        scan_btn = st.button("🚀 SCAN ALL", type="primary", use_container_width=True)
    
    if scan_btn:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(current, total):
            progress_bar.progress(current / total)
            status_text.text(f"🔍 Scanning: {current}/{total} symbols...")
        
        with st.spinner("🎯 Hunting for trapped market makers..."):
            scan_results = analyzer.scan_multiple_symbols(SP500_SYMBOLS, update_progress)
            st.session_state.scan_results = scan_results
        
        progress_bar.progress(1.0)
        status_text.success("✅ Scan complete!")
        
        # Send Discord alerts for high-value opportunities
        if auto_alert and scan_results:
            high_value = [r for r in scan_results 
                         if r['gex_profile'] and 
                         r['gex_profile'].get('dealer_pain', 0) > 70 and
                         r['best_signal']['confidence'] > 65][:3]
            
            for r in high_value:
                alert_msg = analyzer.format_discord_alert(
                    r['symbol'],
                    r['gex_profile'],
                    r['best_signal']
                )
                if alert_msg:
                    if analyzer.send_discord_alert(alert_msg):
                        st.success(f"✅ Alert sent for {r['symbol']}")
                    else:
                        st.warning(f"⚠️ Failed to send alert for {r['symbol']}")
    
    # Display results
    if st.session_state.scan_results:
        results = st.session_state.scan_results
        
        # Summary metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        trapped = sum(1 for r in results if r['gex_profile'] and r['gex_profile'].get('dealer_pain', 0) > 80)
        scrambling = sum(1 for r in results if r['gex_profile'] and 60 < r['gex_profile'].get('dealer_pain', 0) <= 80)
        squeeze_ops = sum(1 for r in results if r['best_signal'] and r['best_signal'].get('type') == 'SQUEEZE_PLAY')
        premium_ops = sum(1 for r in results if r['best_signal'] and r['best_signal'].get('type') == 'PREMIUM_SELLING')
        condor_ops = sum(1 for r in results if r['best_signal'] and r['best_signal'].get('type') == 'IRON_CONDOR')
        
        with col1:
            st.metric("🔥 Trapped", trapped)
        with col2:
            st.metric("😰 Scrambling", scrambling)
        with col3:
            st.metric("⚡ Squeezes", squeeze_ops)
        with col4:
            st.metric("💰 Premium", premium_ops)
        with col5:
            st.metric("🦅 Condors", condor_ops)
        
        # Apply filters
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
            filtered = results
        
        filtered = [r for r in filtered if r['best_signal'] and r['best_signal'].get('confidence', 0) >= min_confidence]
        
        st.markdown(f"### Showing {len(filtered)} opportunities")
        
        # Display filtered results
        for r in filtered[:30]:  # Limit display
            symbol = r['symbol']
            signal = r.get('best_signal', {})
            gex = r.get('gex_profile')
            
            if signal.get('type') == 'SQUEEZE_PLAY':
                row_class = "squeeze-row"
                emoji = "⚡"
            elif signal.get('type') == 'PREMIUM_SELLING':
                row_class = "premium-row"
                emoji = "💰"
            elif signal.get('type') == 'IRON_CONDOR':
                row_class = "premium-row"
                emoji = "🦅"
            else:
                row_class = "wait-row"
                emoji = "⏳"
            
            dealer_pain = gex.get('dealer_pain', 0) if gex else 0
            
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.markdown(f"""
                <div class="symbol-row {row_class}">
                    <h4>{emoji} {symbol} - {signal.get('direction', 'ANALYZING')}</h4>
                    <p>Confidence: {signal.get('confidence', 0):.0f}% | Pain: {dealer_pain:.0f}/100</p>
                    <p>{signal.get('reasoning', 'Analyzing...')[:100]}...</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                if st.button(f"Trade {symbol}", key=f"trade_{symbol}"):
                    if signal.get('position_size', 0) > 0:
                        current_price = gex.get('current_price', 100)
                        position = add_position(
                            symbol,
                            current_price,
                            signal.get('position_size', 1000) / current_price,
                            signal.get('type', 'MANUAL'),
                            signal
                        )
                        st.success(f"✅ Position opened for {symbol}")
            
            with col3:
                if st.button(f"Alert {symbol}", key=f"alert_{symbol}"):
                    alert_msg = analyzer.format_discord_alert(symbol, gex, signal)
                    if analyzer.send_discord_alert(alert_msg):
                        st.success("✅ Alert sent!")

# ============================================================================
# TAB 2: DEEP ANALYSIS
# ============================================================================

with tabs[1]:
    st.header("🎯 Deep Market Maker Analysis")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        symbol = st.text_input("Symbol", value="SPY").upper().strip()
    
    with col2:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()
    
    with col3:
        st.metric("Capital", "$100,000")
    
    # Quick select
    quick_cols = st.columns(10)
    quick_symbols = ['SPY', 'QQQ', 'IWM', 'AAPL', 'TSLA', 'NVDA', 'AMD', 'MSFT', 'META', 'XRT']
    for i, sym in enumerate(quick_symbols):
        with quick_cols[i]:
            if st.button(sym, key=f"q_{sym}"):
                symbol = sym
                st.rerun()
    
    if symbol:
        with st.spinner(f"Analyzing {symbol}..."):
            options_data = analyzer.get_options_chain(symbol)
        
        if options_data:
            gex_profile = analyzer.calculate_gex_profile(options_data)
            
            if gex_profile:
                signals = analyzer.generate_all_signals(gex_profile, symbol)
                best_signal = signals[0] if signals else None
                
                # Action box
                dealer_pain = gex_profile.get('dealer_pain', 0)
                
                if best_signal:
                    st.markdown(f"""
                    <div class="action-box {'mm-trapped' if dealer_pain > 80 else 'mm-scrambling' if dealer_pain > 60 else 'mm-defending'}">
                        <h1>🎯 {best_signal['direction']}</h1>
                        <p>Dealer Pain: {dealer_pain:.0f}/100 | Confidence: {best_signal.get('confidence', 0):.0f}%</p>
                        <p>{best_signal.get('reasoning', '')}</p>
                        <p style="color: #FFD700;">ENTRY: {best_signal['entry']} | SIZE: ${best_signal.get('position_size', 0):,.0f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Metrics
                col1, col2, col3, col4, col5, col6 = st.columns(6)
                
                with col1:
                    st.metric("Price", f"${gex_profile['current_price']:.2f}")
                with col2:
                    st.metric("Net GEX", f"{gex_profile['net_gex']/1e9:.1f}B")
                with col3:
                    st.metric("Flip", f"${gex_profile['gamma_flip']:.2f}")
                with col4:
                    st.metric("Distance", f"{gex_profile['distance_to_flip']:.1f}%")
                with col5:
                    st.metric("Pain", f"{dealer_pain:.0f}")
                with col6:
                    st.metric("VIX", f"{gex_profile.get('vix', 15):.1f}")

# ============================================================================
# TAB 3: POSITION TRACKING
# ============================================================================

with tabs[2]:
    st.header("📊 Position Tracking")
    
    # Update positions
    update_positions()
    
    # Active positions
    st.subheader("Active Positions")
    
    active_open = [p for p in st.session_state.active_positions if p['status'] == 'OPEN']
    
    if active_open:
        for position in active_open:
            col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 1])
            
            with col1:
                pnl_color = "profit" if position['pnl_percent'] > 0 else "loss"
                st.markdown(f"""
                <div class="position-card">
                    <h4>{position['symbol']} - {position['strategy']}</h4>
                    <p>Entry: ${position['entry_price']:.2f} | Current: ${position['current_price']:.2f}</p>
                    <p class="{pnl_color}">P&L: ${position['pnl']:.2f} ({position['pnl_percent']:.1f}%)</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.metric("Size", f"{position['size']:.0f}")
            
            with col3:
                st.metric("Target", position['target'])
            
            with col4:
                st.metric("Stop", position['stop'])
            
            with col5:
                if st.button(f"Close", key=f"close_{position['id']}"):
                    closed = close_position(position['id'])
                    if closed:
                        st.success(f"Position closed: {closed['final_pnl_percent']:.1f}% return")
                        st.rerun()
    else:
        st.info("No active positions")
    
    # Position history
    st.subheader("Position History")
    
    if st.session_state.position_history:
        history_df = pd.DataFrame(st.session_state.position_history)
        st.dataframe(history_df[['symbol', 'strategy', 'entry_price', 'exit_price', 'final_pnl_percent']])

# ============================================================================
# TAB 4: ALERTS
# ============================================================================

with tabs[4]:
    st.header("⚡ Real-Time Alerts")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Alert Channels")
        
        discord_enabled = st.checkbox("Discord Alerts", value=st.session_state.alerts_config['discord'])
        st.session_state.alerts_config['discord'] = discord_enabled
        
        telegram_enabled = st.checkbox("Telegram Alerts")
        if telegram_enabled:
            tg_token = st.text_input("Bot Token", type="password")
            tg_chat = st.text_input("Chat ID")
        
        email_enabled = st.checkbox("Email Alerts")
        if email_enabled:
            email_addr = st.text_input("Email Address")
    
    with col2:
        st.subheader("Price Alerts")
        
        alert_symbol = st.text_input("Symbol for Alert")
        alert_type = st.selectbox("Alert Type", ["Near Gamma Flip", "Dealer Pain > 80", "Custom Price"])
        
        if alert_type == "Custom Price":
            alert_price = st.number_input("Alert Price", min_value=0.01)
        
        if st.button("Add Alert"):
            st.session_state.alerts_config['price_alerts'].append({
                'symbol': alert_symbol,
                'type': alert_type,
                'created': datetime.now()
            })
            st.success(f"Alert added for {alert_symbol}")
        
        # Show active alerts
        if st.session_state.alerts_config['price_alerts']:
            st.subheader("Active Alerts")
            for alert in st.session_state.alerts_config['price_alerts']:
                st.write(f"• {alert['symbol']}: {alert['type']}")

# ============================================================================
# TAB 5: BACKTESTING
# ============================================================================

with tabs[5]:
    st.header("🧪 Strategy Backtesting")
    
    st.markdown("""
    ### Understanding Backtesting
    
    Backtesting allows you to test your strategies on historical data to see how they would have performed.
    This helps validate your approach before risking real capital.
    
    **Key Concepts:**
    - **Win Rate**: Percentage of profitable trades
    - **Sharpe Ratio**: Risk-adjusted returns (higher is better)
    - **Max Drawdown**: Largest peak-to-trough decline
    - **Total Return**: Overall profit/loss from the strategy
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        backtest_symbol = st.text_input("Symbol to Backtest", value="SPY")
    
    with col2:
        backtest_strategy = st.selectbox("Strategy", ["Squeeze Play", "Premium Selling", "Iron Condor"])
    
    with col3:
        backtest_period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y"])
    
    if st.button("Run Backtest", type="primary"):
        with st.spinner("Running backtest..."):
            results = analyzer.backtest_strategy(
                backtest_symbol,
                backtest_strategy.lower().replace(" ", "_"),
                backtest_period
            )
            
            if results:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Return", f"{results['total_return']:.1f}%")
                with col2:
                    st.metric("Win Rate", f"{results['win_rate']:.1f}%")
                with col3:
                    st.metric("Max Drawdown", f"{results['max_drawdown']:.1f}%")
                with col4:
                    st.metric("Sharpe Ratio", f"{results.get('sharpe_ratio', 0):.2f}")
                
                if results['trades']:
                    st.subheader("Trade History")
                    trades_df = pd.DataFrame(results['trades'])
                    st.dataframe(trades_df)
            else:
                st.warning("Unable to backtest - insufficient data")

# ============================================================================
# TAB 6: MORNING REPORT
# ============================================================================

with tabs[6]:
    st.header("📈 Morning Report")
    
    if st.button("Generate Report", type="primary"):
        with st.spinner("Generating report..."):
            # Quick scan of top symbols
            morning_symbols = ['SPY', 'QQQ', 'IWM', 'AAPL', 'TSLA', 'NVDA', 'AMD', 'XRT']
            morning_results = analyzer.scan_multiple_symbols(morning_symbols)
            
            if morning_results:
                st.markdown("### Market Overview")
                
                avg_pain = np.mean([r['gex_profile'].get('dealer_pain', 0) 
                                   for r in morning_results if r['gex_profile']])
                
                st.metric("Average Dealer Pain", f"{avg_pain:.0f}/100")
                
                st.markdown("### Top Opportunities")
                
                for r in morning_results[:5]:
                    if r['gex_profile'] and r['best_signal']['confidence'] > 60:
                        st.markdown(f"""
                        **{r['symbol']}** - {r['best_signal']['direction']}
                        - Confidence: {r['best_signal']['confidence']:.0f}%
                        - Entry: {r['best_signal']['entry']}
                        - Reason: {r['best_signal'].get('reasoning', 'N/A')}
                        """)

# ============================================================================
# TAB 7: EDUCATION
# ============================================================================

with tabs[7]:
    st.header("🎓 DealerEdge Education")
    
    edu_tabs = st.tabs(["Basics", "Strategies", "Backtesting Guide"])
    
    with edu_tabs[0]:
        st.markdown("""
        ### Understanding GEX
        
        **Gamma Exposure (GEX)** measures the total gamma positioning of market makers.
        
        **Key Principles:**
        - **Positive GEX**: MMs are long gamma → sell rallies, buy dips → low volatility
        - **Negative GEX**: MMs are short gamma → buy rallies, sell dips → high volatility
        - **Gamma Flip**: The price where GEX crosses zero → regime change point
        
        **Why This Matters:**
        Market makers MUST hedge their positions. When they're caught wrong-footed,
        their forced hedging creates predictable price movements we can exploit.
        """)
    
    with edu_tabs[1]:
        st.markdown("""
        ### Trading Strategies
        
        **1. Squeeze Plays**
        - Enter when dealers are trapped (pain > 70)
        - Buy options in direction of pressure
        - Target: 2-3% moves in hours
        
        **2. Premium Selling**
        - Sell options at dealer defense levels
        - Use positive GEX environments
        - Win rate: 70-80% at walls
        
        **3. Iron Condors**
        - Range-bound markets with clear walls
        - Positive GEX > $1B
        - Adjust wings based on gamma bias
        """)
    
    with edu_tabs[2]:
        st.markdown("""
        ### How to Use Backtesting
        
        **Step 1: Choose Your Strategy**
        - Select the strategy you want to test
        - Pick a symbol with good liquidity
        
        **Step 2: Set Your Period**
        - 1 month for quick validation
        - 6 months for robust testing
        - 1 year for all market conditions
        
        **Step 3: Interpret Results**
        - **Win Rate > 60%**: Good consistency
        - **Sharpe > 1.5**: Good risk-adjusted returns
        - **Max Drawdown < 20%**: Acceptable risk
        
        **Step 4: Refine Your Approach**
        - Adjust entry/exit criteria
        - Modify position sizing
        - Test different market conditions
        
        **Remember**: Past performance doesn't guarantee future results,
        but backtesting helps validate your edge.
        """)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; background: rgba(255,255,255,0.05); border-radius: 15px;">
    <h3 style="color: #667eea;">DealerEdge - Professional GEX Trading Platform</h3>
    <p style="color: #b0b0b0;">Real-Time Data • Position Tracking • Multi-Channel Alerts • Backtesting</p>
    <p style="color: #888; font-size: 0.9rem; margin-top: 1rem;">
        ⚠️ Trading involves substantial risk. Always use proper risk management.
    </p>
</div>
""", unsafe_allow_html=True)
