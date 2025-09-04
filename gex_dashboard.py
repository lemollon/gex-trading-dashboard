#!/usr/bin/env python3
"""
Enhanced Dynamic GEX Scanner with Educational Content
Real-time analysis with improved strategies and fun educational features
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
import requests
from datetime import datetime, date, timedelta
import time
import concurrent.futures
from threading import Lock
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="🎯 Enhanced GEX Scanner",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced styling with more fun elements
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><circle cx="25" cy="25" r="2" fill="rgba(255,255,255,0.1)"/><circle cx="75" cy="75" r="1.5" fill="rgba(255,255,255,0.1)"/><circle cx="50" cy="10" r="1" fill="rgba(255,255,255,0.1)"/></svg>');
        animation: float 6s ease-in-out infinite;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    .scanner-card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        color: #333;
        transition: transform 0.2s ease;
    }
    
    .scanner-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(0,0,0,0.15);
    }
    
    .high-signal {
        border-left: 4px solid #28a745 !important;
        background: linear-gradient(135deg, #f8fff9, #e8f5e8) !important;
        color: #155724 !important;
    }
    
    .medium-signal {
        border-left: 4px solid #fd7e14 !important;
        background: linear-gradient(135deg, #fff8f0, #fef3e8) !important;
        color: #8a4a00 !important;
    }
    
    .low-signal {
        border-left: 4px solid #dc3545 !important;
        background: linear-gradient(135deg, #fff5f5, #ffeaea) !important;
        color: #721c24 !important;
    }
    
    .strategy-card {
        background: white;
        border-radius: 12px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border: 1px solid #e9ecef;
        transition: all 0.3s ease;
    }
    
    .strategy-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .strategy-high {
        border-left: 5px solid #28a745;
        background: linear-gradient(135deg, #f8fff9, #ffffff);
    }
    
    .strategy-medium {
        border-left: 5px solid #fd7e14;
        background: linear-gradient(135deg, #fff8f0, #ffffff);
    }
    
    .strategy-low {
        border-left: 5px solid #dc3545;
        background: linear-gradient(135deg, #fff5f5, #ffffff);
    }
    
    .educational-card {
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
        border-radius: 12px;
        padding: 2rem;
        margin: 1rem 0;
        border-left: 5px solid #6c757d;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }
    
    .fun-fact {
        background: linear-gradient(135deg, #fff3cd, #ffeaa7);
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        border-left: 4px solid #856404;
        color: #856404;
    }
    
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .confidence-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.9rem;
        margin-left: 1rem;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    
    .conf-high {
        background: #28a745;
        color: white;
    }
    
    .conf-medium {
        background: #fd7e14;
        color: white;
    }
    
    .conf-low {
        background: #dc3545;
        color: white;
    }
    
    .strategy-details {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid #dee2e6;
    }
    
    .risk-box {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        color: #8a4a00;
    }
    
    .quiz-option {
        background: #e9ecef;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    
    .quiz-option:hover {
        background: #dee2e6;
        transform: translateX(5px);
    }
    
    .quiz-correct {
        background: #d4edda !important;
        border: 2px solid #28a745;
        color: #155724;
    }
    
    .quiz-incorrect {
        background: #f8d7da !important;
        border: 2px solid #dc3545;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

# Header with animation
st.markdown("""
<div class="main-header">
    <h1>🎯 Enhanced GEX Scanner</h1>
    <p><strong>Real-Time Analysis</strong> | Live options data with advanced strategies & education</p>
    <p>Professional gamma exposure analysis with institutional-grade recommendations</p>
</div>
""", unsafe_allow_html=True)

class EnhancedOptionsScanner:
    """Enhanced scanner with improved strategy analysis and educational features"""
    
    def __init__(self):
        self.risk_free_rate = 0.05
        self.scan_lock = Lock()
        self.results_cache = {}
        
        # Base symbols with better categorization
        self.base_symbols = [
            # Major ETFs
            "SPY", "QQQ", "IWM", "EEM", "GLD", "VIX", "XLF", "XLE", "XLK", "XLP",
            "XLY", "XLI", "XLV", "XLU", "XLB", "XLRE", "XRT", "VXX", "UVXY", "SQQQ",
            
            # Mega Cap Tech
            "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "TSLA", "META", "NVDA", "NFLX", "CRM",
            "ORCL", "ADBE", "INTC", "AMD", "PYPL", "UBER", "LYFT", "SHOP", "SQ", "ZOOM",
            
            # Large Cap Growth
            "BRK-B", "JPM", "JNJ", "UNH", "V", "MA", "HD", "PG", "KO", "PFE",
            "DIS", "VZ", "T", "CSCO", "WMT", "BAC", "XOM", "CVX", "ABBV", "TMO",
            
            # High Beta/Meme Stocks
            "GME", "AMC", "BB", "NOK", "PLTR", "WISH", "CLOV", "SPCE", "NIO", "XPEV",
            "LI", "RIVN", "LCID", "F", "GM", "COIN", "HOOD", "SOFI", "ARKK", "ARKQ"
        ]
    
    # [Previous methods remain the same through get_options_volume_ranking, calculate_quick_gex, scan_for_signals, analyze_custom_symbol]
    
    @st.cache_data(ttl=1800)
    def get_dynamic_symbol_list(_self):
        """Get dynamic list of most active options symbols"""
        active_symbols = _self.base_symbols.copy()
        
        try:
            trending_symbols = [
                "PTON", "ROKU", "ZM", "DOCU", "WORK", "SNOW", "ABNB",
                "DKNG", "PENN", "MGM", "WYNN", "LVS", "CZR", "BYD", "VALE"
            ]
            active_symbols.extend(trending_symbols)
            active_symbols = list(set(active_symbols))
            
        except Exception as e:
            st.warning(f"Using base symbol list: {e}")
        
        return active_symbols[:150]
    
    @st.cache_data(ttl=3600)
    def get_options_volume_ranking(_self):
        """Get current options volume ranking for dynamic symbol list"""
        symbols_to_check = _self.get_dynamic_symbol_list()
        volume_data = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, symbol in enumerate(symbols_to_check):
            try:
                status_text.text(f"🔍 Checking options volume: {symbol} ({i+1}/{len(symbols_to_check)})")
                progress_bar.progress((i + 1) / len(symbols_to_check))
                
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="1d")
                if len(hist) == 0:
                    continue
                    
                current_price = hist['Close'].iloc[-1]
                
                try:
                    exp_dates = ticker.options
                    if not exp_dates:
                        continue
                        
                    chain = ticker.option_chain(exp_dates[0])
                    
                    call_volume = chain.calls['volume'].fillna(0).sum()
                    put_volume = chain.puts['volume'].fillna(0).sum()
                    total_volume = call_volume + put_volume
                    
                    call_oi = chain.calls['openInterest'].fillna(0).sum()
                    put_oi = chain.puts['openInterest'].fillna(0).sum()
                    total_oi = call_oi + put_oi
                    
                    if total_volume > 100 or total_oi > 1000:
                        volume_data.append({
                            'symbol': symbol,
                            'current_price': current_price,
                            'call_volume': int(call_volume),
                            'put_volume': int(put_volume),
                            'total_volume': int(total_volume),
                            'call_oi': int(call_oi),
                            'put_oi': int(put_oi),
                            'total_oi': int(total_oi),
                            'put_call_ratio': put_volume / max(call_volume, 1),
                            'last_updated': datetime.now()
                        })
                    
                except Exception:
                    continue
                    
            except Exception:
                continue
        
        progress_bar.empty()
        status_text.empty()
        
        if volume_data:
            df = pd.DataFrame(volume_data)
            df = df.sort_values('total_volume', ascending=False).reset_index(drop=True)
            return df
        
        return pd.DataFrame()
    
    def calculate_quick_gex(self, symbol, current_price):
        """Quick GEX calculation for scanning"""
        try:
            ticker = yf.Ticker(symbol)
            exp_dates = ticker.options[:3]
            
            total_gex = 0
            total_call_gex = 0
            total_put_gex = 0
            
            for exp_date in exp_dates:
                try:
                    exp_dt = datetime.strptime(exp_date, '%Y-%m-%d')
                    dte = (exp_dt.date() - date.today()).days
                    
                    if dte <= 0 or dte > 45:
                        continue
                    
                    chain = ticker.option_chain(exp_date)
                    
                    for _, call in chain.calls.iterrows():
                        if call['openInterest'] > 0:
                            gamma = 0.01 * np.exp(-abs(call['strike'] - current_price) / current_price * 5)
                            call_gex = current_price * gamma * call['openInterest'] * 100
                            total_call_gex += call_gex
                            total_gex += call_gex
                    
                    for _, put in chain.puts.iterrows():
                        if put['openInterest'] > 0:
                            gamma = 0.01 * np.exp(-abs(put['strike'] - current_price) / current_price * 5)
                            put_gex = current_price * gamma * put['openInterest'] * 100
                            total_put_gex += put_gex
                            total_gex -= put_gex
                
                except Exception:
                    continue
            
            return {
                'net_gex': total_gex,
                'call_gex': total_call_gex,
                'put_gex': total_put_gex
            }
            
        except Exception:
            return {'net_gex': 0, 'call_gex': 0, 'put_gex': 0}
    
    def scan_for_signals(self, top_symbols_df, max_symbols=50):
        """Scan top symbols for GEX signals"""
        
        def process_symbol(row):
            try:
                symbol = row['symbol']
                current_price = row['current_price']
                
                gex_data = self.calculate_quick_gex(symbol, current_price)
                net_gex = gex_data['net_gex']
                net_gex_millions = net_gex / 1e6
                
                signals = []
                confidence = 0
                
                if net_gex < -50e6:
                    signals.append("SQUEEZE_POTENTIAL")
                    confidence += 30
                
                if net_gex > 100e6:
                    signals.append("RANGE_BOUND")
                    confidence += 25
                
                if row['total_volume'] > 10000:
                    confidence += 20
                
                if row['total_oi'] > 50000:
                    confidence += 15
                
                pcr = row['put_call_ratio']
                if pcr > 1.5:
                    signals.append("BEARISH_FLOW")
                    confidence += 10
                elif pcr < 0.5:
                    signals.append("BULLISH_FLOW")
                    confidence += 10
                
                return {
                    'symbol': symbol,
                    'current_price': current_price,
                    'net_gex_millions': round(net_gex_millions, 1),
                    'call_gex_millions': round(gex_data['call_gex'] / 1e6, 1),
                    'put_gex_millions': round(gex_data['put_gex'] / 1e6, 1),
                    'total_volume': row['total_volume'],
                    'total_oi': row['total_oi'],
                    'put_call_ratio': round(pcr, 2),
                    'signals': signals,
                    'confidence': min(confidence, 100),
                    'scan_time': datetime.now()
                }
                
            except Exception:
                return None
        
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        symbols_to_scan = top_symbols_df.head(max_symbols)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_symbol = {
                executor.submit(process_symbol, row): row['symbol'] 
                for _, row in symbols_to_scan.iterrows()
            }
            
            completed = 0
            for future in concurrent.futures.as_completed(future_to_symbol):
                completed += 1
                symbol = future_to_symbol[future]
                
                status_text.text(f"🔄 Analyzing GEX: {symbol} ({completed}/{len(symbols_to_scan)})")
                progress_bar.progress(completed / len(symbols_to_scan))
                
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception:
                    continue
        
        progress_bar.empty()
        status_text.empty()
        
        if results:
            results_df = pd.DataFrame(results)
            results_df = results_df.sort_values('confidence', ascending=False).reset_index(drop=True)
            return results_df
        
        return pd.DataFrame()
    
    def analyze_custom_symbol(self, symbol):
        """Comprehensive analysis of a single custom symbol"""
        try:
            symbol = symbol.upper().strip()
            ticker = yf.Ticker(symbol)
            
            info = ticker.info
            hist = ticker.history(period="5d")
            
            if len(hist) == 0:
                return None
            
            current_price = hist['Close'].iloc[-1]
            prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
            price_change = ((current_price - prev_close) / prev_close) * 100
            
            try:
                exp_dates = ticker.options
                if not exp_dates:
                    return {
                        'symbol': symbol,
                        'error': 'No options available for this symbol',
                        'current_price': current_price,
                        'price_change': price_change
                    }
                
                all_analysis = []
                total_call_gex = 0
                total_put_gex = 0
                total_volume = 0
                total_oi = 0
                
                for exp_date in exp_dates[:5]:
                    try:
                        exp_dt = datetime.strptime(exp_date, '%Y-%m-%d')
                        dte = (exp_dt.date() - date.today()).days
                        
                        if dte <= 0 or dte > 60:
                            continue
                        
                        chain = ticker.option_chain(exp_date)
                        
                        exp_call_gex = 0
                        exp_put_gex = 0
                        exp_volume = 0
                        exp_oi = 0
                        
                        for _, call in chain.calls.iterrows():
                            if call['openInterest'] > 0:
                                moneyness = call['strike'] / current_price
                                time_factor = np.sqrt(dte / 365.0)
                                gamma = 0.02 * np.exp(-2 * abs(moneyness - 1)) * time_factor
                                
                                call_gex = current_price * gamma * call['openInterest'] * 100
                                exp_call_gex += call_gex
                                exp_volume += call.get('volume', 0)
                                exp_oi += call['openInterest']
                        
                        for _, put in chain.puts.iterrows():
                            if put['openInterest'] > 0:
                                moneyness = put['strike'] / current_price
                                time_factor = np.sqrt(dte / 365.0)
                                gamma = 0.02 * np.exp(-2 * abs(moneyness - 1)) * time_factor
                                
                                put_gex = current_price * gamma * put['openInterest'] * 100
                                exp_put_gex += put_gex
                                exp_volume += put.get('volume', 0)
                                exp_oi += put['openInterest']
                        
                        all_analysis.append({
                            'expiration': exp_date,
                            'dte': dte,
                            'call_gex': exp_call_gex,
                            'put_gex': exp_put_gex,
                            'net_gex': exp_call_gex - exp_put_gex,
                            'volume': exp_volume,
                            'open_interest': exp_oi
                        })
                        
                        total_call_gex += exp_call_gex
                        total_put_gex += exp_put_gex
                        total_volume += exp_volume
                        total_oi += exp_oi
                        
                    except Exception:
                        continue
                
                if not all_analysis:
                    return {
                        'symbol': symbol,
                        'error': 'Unable to analyze options data',
                        'current_price': current_price,
                        'price_change': price_change
                    }
                
                net_gex = total_call_gex - total_put_gex
                
                signals = []
                confidence = 0
                
                if net_gex < -50e6:
                    signals.append("NEGATIVE_GEX_SQUEEZE")
                    confidence += 35
                elif net_gex > 100e6:
                    signals.append("POSITIVE_GEX_RANGE")
                    confidence += 25
                
                if total_volume > 5000:
                    signals.append("HIGH_VOLUME")
                    confidence += 20
                
                if total_oi > 25000:
                    signals.append("HIGH_OPEN_INTEREST")
                    confidence += 15
                
                if abs(price_change) > 3:
                    signals.append("HIGH_MOMENTUM")
                    confidence += 10
                
                put_call_ratio = total_put_gex / max(total_call_gex, 1)
                if put_call_ratio > 1.5:
                    signals.append("BEARISH_POSITIONING")
                    confidence += 10
                elif put_call_ratio < 0.5:
                    signals.append("BULLISH_POSITIONING")
                    confidence += 10
                
                return {
                    'symbol': symbol,
                    'company_name': info.get('longName', symbol),
                    'sector': info.get('sector', 'Unknown'),
                    'current_price': current_price,
                    'price_change': price_change,
                    'market_cap': info.get('marketCap', 0),
                    'total_call_gex': total_call_gex,
                    'total_put_gex': total_put_gex,
                    'net_gex': net_gex,
                    'total_volume': int(total_volume),
                    'total_oi': int(total_oi),
                    'put_call_ratio': put_call_ratio,
                    'signals': signals,
                    'confidence': min(confidence, 100),
                    'expiration_analysis': all_analysis,
                    'analysis_time': datetime.now()
                }
                
            except Exception as e:
                return {
                    'symbol': symbol,
                    'error': f'Options analysis failed: {str(e)}',
                    'current_price': current_price,
                    'price_change': price_change
                }
                
        except Exception as e:
            return {
                'symbol': symbol,
                'error': f'Symbol analysis failed: {str(e)}'
            }
    
    def get_detailed_gex_profile(self, symbol):
        """Get detailed strike-by-strike GEX profile"""
        try:
            ticker = yf.Ticker(symbol)
            current_price = self.get_current_price(symbol)
            
            if current_price is None:
                return None
            
            exp_dates = ticker.options[:3]
            all_strikes = []
            
            for exp_date in exp_dates:
                try:
                    exp_dt = datetime.strptime(exp_date, '%Y-%m-%d')
                    dte = (exp_dt.date() - date.today()).days
                    
                    if dte <= 0 or dte > 45:
                        continue
                    
                    chain = ticker.option_chain(exp_date)
                    strikes_data = {}
                    
                    for _, call in chain.calls.iterrows():
                        strike = call['strike']
                        if call['openInterest'] > 0 and strike >= current_price * 0.85 and strike <= current_price * 1.15:
                            moneyness = strike / current_price
                            time_factor = np.sqrt(dte / 365.0)
                            gamma = 0.02 * np.exp(-2 * abs(moneyness - 1)) * time_factor
                            gex = current_price * gamma * call['openInterest'] * 100
                            
                            if strike not in strikes_data:
                                strikes_data[strike] = {'call_gex': 0, 'put_gex': 0, 'call_oi': 0, 'put_oi': 0}
                            
                            strikes_data[strike]['call_gex'] += gex
                            strikes_data[strike]['call_oi'] += call['openInterest']
                    
                    for _, put in chain.puts.iterrows():
                        strike = put['strike']
                        if put['openInterest'] > 0 and strike >= current_price * 0.85 and strike <= current_price * 1.15:
                            moneyness = strike / current_price
                            time_factor = np.sqrt(dte / 365.0)
                            gamma = 0.02 * np.exp(-2 * abs(moneyness - 1)) * time_factor
                            gex = current_price * gamma * put['openInterest'] * 100
                            
                            if strike not in strikes_data:
                                strikes_data[strike] = {'call_gex': 0, 'put_gex': 0, 'call_oi': 0, 'put_oi': 0}
                            
                            strikes_data[strike]['put_gex'] += gex
                            strikes_data[strike]['put_oi'] += put['openInterest']
                    
                    for strike, data in strikes_data.items():
                        all_strikes.append({
                            'strike': strike,
                            'call_gex': data['call_gex'],
                            'put_gex': -data['put_gex'],
                            'net_gex': data['call_gex'] - data['put_gex'],
                            'call_oi': data['call_oi'],
                            'put_oi': data['put_oi'],
                            'expiration': exp_date,
                            'dte': dte
                        })
                        
                except Exception:
                    continue
            
            if all_strikes:
                df = pd.DataFrame(all_strikes)
                
                strike_gex = df.groupby('strike').agg({
                    'call_gex': 'sum',
                    'put_gex': 'sum',
                    'net_gex': 'sum',
                    'call_oi': 'sum',
                    'put_oi': 'sum'
                }).reset_index()
                
                strike_gex = strike_gex.sort_values('strike').reset_index(drop=True)
                strike_gex['cumulative_gex'] = strike_gex['net_gex'].cumsum()
                
                flip_idx = strike_gex['cumulative_gex'].abs().idxmin()
                gamma_flip = strike_gex.iloc[flip_idx]['strike']
                
                return {
                    'current_price': current_price,
                    'gamma_flip': gamma_flip,
                    'strike_data': strike_gex,
                    'call_walls': strike_gex[strike_gex['call_gex'] > 0].nlargest(3, 'call_gex'),
                    'put_walls': strike_gex[strike_gex['put_gex'] < 0].nsmallest(3, 'put_gex')
                }
            
            return None
            
        except Exception:
            return None
    
    def analyze_options_strategies(self, symbol, gex_data=None):
        """Fixed and enhanced options strategy analysis"""
        try:
            if gex_data is None:
                gex_profile = self.get_detailed_gex_profile(symbol)
                if not gex_profile:
                    return None
                
                current_price = gex_profile['current_price']
                gamma_flip = gex_profile['gamma_flip']
                call_walls = gex_profile['call_walls']
                put_walls = gex_profile['put_walls']
                strike_data = gex_profile['strike_data']
            else:
                current_price = gex_data['current_price']
                gamma_flip = gex_data.get('gamma_flip', current_price)
                call_walls = gex_data.get('call_walls', pd.DataFrame())
                put_walls = gex_data.get('put_walls', pd.DataFrame())
                strike_data = gex_data.get('strike_data', pd.DataFrame())
            
            net_gex = strike_data['net_gex'].sum() if len(strike_data) > 0 else 0
            distance_to_flip = ((current_price - gamma_flip) / current_price) * 100
            
            strategies = []
            
            # Fixed strategy analyses
            long_call_analysis = self._analyze_long_calls(
                current_price, gamma_flip, net_gex, distance_to_flip, call_walls, strike_data
            )
            if long_call_analysis:
                strategies.append(long_call_analysis)
            
            long_put_analysis = self._analyze_long_puts(
                current_price, gamma_flip, net_gex, distance_to_flip, put_walls, strike_data
            )
            if long_put_analysis:
                strategies.append(long_put_analysis)
            
            sell_call_analysis = self._analyze_sell_calls(
                current_price, gamma_flip, net_gex, call_walls, strike_data
            )
            if sell_call_analysis:
                strategies.append(sell_call_analysis)
            
            sell_put_analysis = self._analyze_sell_puts(
                current_price, gamma_flip, net_gex, put_walls, strike_data
            )
            if sell_put_analysis:
                strategies.append(sell_put_analysis)
            
            iron_condor_analysis = self._analyze_iron_condors(
                current_price, gamma_flip, net_gex, call_walls, put_walls, strike_data
            )
            if iron_condor_analysis:
                strategies.append(iron_condor_analysis)
            
            return {
                'symbol': symbol,
                'current_price': current_price,
                'gamma_flip': gamma_flip,
                'net_gex': net_gex,
                'distance_to_flip': distance_to_flip,
                'strategies': sorted(strategies, key=lambda x: x.get('confidence', 0), reverse=True),
                'analysis_time': datetime.now()
            }
            
        except Exception as e:
            st.error(f"Strategy analysis error: {str(e)}")
            return None
    
    def _analyze_long_calls(self, current_price, gamma_flip, net_gex, distance_to_flip, call_walls, strike_data):
        """Fixed long call analysis"""
        confidence = 0
        strategy_details = {
            'strategy': 'LONG CALLS',
            'strategy_type': 'directional_bullish',
            'emoji': '🚀'
        }
        
        reasons = []
        
        if net_gex < -100e6:
            confidence += 35
            reasons.append(f"Negative GEX {net_gex/1e6:.0f}M creates volatility amplification")
        
        if distance_to_flip < -0.5:
            confidence += 25
            reasons.append(f"Price {abs(distance_to_flip):.1f}% below gamma flip point")
        
        if len(call_walls) > 0:
            nearest_call_wall = call_walls.iloc[0]['strike']
            wall_distance = ((nearest_call_wall - current_price) / current_price) * 100
            if wall_distance > 2:
                confidence += 20
                reasons.append(f"Call wall resistance at ${nearest_call_wall:.2f} (+{wall_distance:.1f}%)")
        
        if len(strike_data) > 0:
            atm_strikes = strike_data[
                (strike_data['strike'] >= current_price * 0.98) & 
                (strike_data['strike'] <= current_price * 1.02)
            ]
            if len(atm_strikes) > 0:
                avg_gamma = atm_strikes['net_gex'].mean()
                if avg_gamma < 50e6:
                    confidence += 15
                    reasons.append("Low gamma at-the-money reduces dealer hedging")
        
        if confidence >= 50:
            atm_strike = round(current_price / 5) * 5
            target_strikes = [atm_strike, atm_strike + 5]
            
            if len(call_walls) > 0:
                primary_target = call_walls.iloc[0]['strike']
                secondary_target = call_walls.iloc[1]['strike'] if len(call_walls) > 1 else primary_target * 1.05
            else:
                primary_target = current_price * 1.05
                secondary_target = current_price * 1.08
            
            expected_move = abs(distance_to_flip) * 0.6 if distance_to_flip < 0 else 3.0
            
            strategy_details.update({
                'confidence': min(confidence, 95),
                'recommended_strikes': target_strikes,
                'optimal_dte': '2-5 DTE' if confidence > 75 else '5-10 DTE',
                'primary_target': primary_target,
                'secondary_target': secondary_target,
                'expected_move': expected_move,
                'max_risk': '100% (premium paid)',
                'expected_return': f"{int(expected_move * 8)}-{int(expected_move * 15)}%",
                'stop_loss': current_price * 0.97,
                'reasons': reasons,
                'position_size': '2-3% of portfolio (high risk)',
                'time_horizon': '1-5 days',
                'volatility_bet': 'Long volatility + directional'
            })
            
            return strategy_details
        
        return None
    
    def _analyze_long_puts(self, current_price, gamma_flip, net_gex, distance_to_flip, put_walls, strike_data):
        """Fixed long put analysis"""
        confidence = 0
        strategy_details = {
            'strategy': 'LONG PUTS',
            'strategy_type': 'directional_bearish',
            'emoji': '📉'
        }
        
        reasons = []
        
        if net_gex > 200e6 and abs(distance_to_flip) < 0.5:
            confidence += 40
            reasons.append(f"High positive GEX {net_gex/1e6:.0f}M near flip point - breakdown setup")
        
        if len(put_walls) > 0:
            nearest_put_wall = put_walls.iloc[0]['strike']
            wall_distance = ((current_price - nearest_put_wall) / current_price) * 100
            if wall_distance > 1:
                confidence += 25
                reasons.append(f"Put wall support at ${nearest_put_wall:.2f} (-{wall_distance:.1f}%)")
        
        if len(strike_data) > 0:
            above_strikes = strike_data[strike_data['strike'] > current_price * 1.02]
            if len(above_strikes) > 0:
                call_gex_above = above_strikes['call_gex'].sum()
                if call_gex_above > 200e6:
                    confidence += 20
                    reasons.append(f"Strong call gamma above: {call_gex_above/1e6:.0f}M")
        
        if distance_to_flip > 0.3:
            confidence += 15
            reasons.append(f"Price {distance_to_flip:.1f}% above flip - vulnerable to breakdown")
        
        if confidence >= 50:
            atm_strike = round(current_price / 5) * 5
            target_strikes = [atm_strike, atm_strike - 5]
            
            if len(put_walls) > 0:
                primary_target = put_walls.iloc[0]['strike']
                secondary_target = put_walls.iloc[1]['strike'] if len(put_walls) > 1 else primary_target * 0.95
            else:
                primary_target = current_price * 0.95
                secondary_target = current_price * 0.92
            
            expected_move = max(abs(distance_to_flip) * 0.8, 2.5)
            
            strategy_details.update({
                'confidence': min(confidence, 95),
                'recommended_strikes': target_strikes,
                'optimal_dte': '3-7 DTE' if confidence > 75 else '7-14 DTE',
                'primary_target': primary_target,
                'secondary_target': secondary_target,
                'expected_move': expected_move,
                'max_risk': '100% (premium paid)',
                'expected_return': f"{int(expected_move * 6)}-{int(expected_move * 12)}%",
                'stop_loss': current_price * 1.03,
                'reasons': reasons,
                'position_size': '2-3% of portfolio',
                'time_horizon': '2-7 days',
                'volatility_bet': 'Long volatility + directional'
            })
            
            return strategy_details
        
        return None
    
    def _analyze_sell_calls(self, current_price, gamma_flip, net_gex, call_walls, strike_data):
        """Fixed call selling analysis"""
        confidence = 0
        strategy_details = {
            'strategy': 'SELL CALLS',
            'strategy_type': 'premium_collection',
            'emoji': '💰'
        }
        
        reasons = []
        
        if net_gex > 300e6:
            confidence += 30
            reasons.append(f"High positive GEX {net_gex/1e6:.0f}M suppresses volatility")
        
        if len(call_walls) > 0:
            strongest_wall = call_walls.iloc[0]
            wall_distance = ((strongest_wall['strike'] - current_price) / current_price) * 100
            wall_strength = strongest_wall['call_gex'] / 1e6
            
            if wall_distance < 3 and wall_strength > 100:
                confidence += 35
                reasons.append(f"Strong call wall at ${strongest_wall['strike']:.2f} ({wall_strength:.0f}M GEX)")
            elif wall_distance < 5:
                confidence += 20
                reasons.append(f"Call wall resistance at ${strongest_wall['strike']:.2f}")
        
        if len(call_walls) > 0:
            call_wall = call_walls.iloc[0]['strike']
            if gamma_flip < current_price < call_wall:
                confidence += 25
                reasons.append("Price in optimal zone between flip and call wall")
        
        if confidence >= 50:
            if len(call_walls) > 0:
                target_strike = call_walls.iloc[0]['strike']
                buffer_strike = target_strike + (target_strike - current_price) * 0.5
            else:
                target_strike = current_price * 1.03
                buffer_strike = current_price * 1.05
            
            strategy_details.update({
                'confidence': min(confidence, 90),
                'recommended_strikes': [target_strike, buffer_strike],
                'optimal_dte': '7-21 DTE',
                'primary_target': '50% profit',
                'secondary_target': '25% profit',
                'expected_move': 'Range-bound',
                'max_risk': f"${target_strike - current_price:.2f} per share",
                'expected_return': '15-35% on margin',
                'stop_loss': f"Close if price > ${target_strike * 1.02:.2f}",
                'reasons': reasons,
                'position_size': '5-10% of portfolio',
                'time_horizon': '1-3 weeks',
                'volatility_bet': 'Short volatility'
            })
            
            return strategy_details
        
        return None
    
    def _analyze_sell_puts(self, current_price, gamma_flip, net_gex, put_walls, strike_data):
        """Fixed put selling analysis"""
        confidence = 0
        strategy_details = {
            'strategy': 'SELL PUTS',
            'strategy_type': 'premium_collection',
            'emoji': '🛡️'
        }
        
        reasons = []
        
        if net_gex > 200e6:
            confidence += 25
            reasons.append(f"Positive GEX {net_gex/1e6:.0f}M provides downside support")
        
        if len(put_walls) > 0:
            strongest_wall = put_walls.iloc[0]
            wall_distance = ((current_price - strongest_wall['strike']) / current_price) * 100
            wall_strength = abs(strongest_wall['put_gex']) / 1e6
            
            if wall_distance > 1 and wall_strength > 75:
                confidence += 35
                reasons.append(f"Strong put wall at ${strongest_wall['strike']:.2f} ({wall_strength:.0f}M GEX)")
            elif wall_distance > 0.5:
                confidence += 20
                reasons.append(f"Put wall support at ${strongest_wall['strike']:.2f}")
        
        if current_price > gamma_flip:
            flip_distance = ((current_price - gamma_flip) / current_price) * 100
            confidence += min(flip_distance * 10, 25)
            reasons.append(f"Price {flip_distance:.1f}% above gamma flip")
        
        if confidence >= 50:
            if len(put_walls) > 0:
                target_strike = put_walls.iloc[0]['strike']
                safer_strike = target_strike * 0.98
            else:
                target_strike = current_price * 0.97
                safer_strike = current_price * 0.95
            
            strategy_details.update({
                'confidence': min(confidence, 88),
                'recommended_strikes': [target_strike, safer_strike],
                'optimal_dte': '14-30 DTE',
                'primary_target': '50% profit',
                'secondary_target': '25% profit',
                'expected_move': 'Sideways to up',
                'max_risk': f"Assignment at ${target_strike:.2f}",
                'expected_return': '20-40% annualized',
                'stop_loss': f"Close if price < ${target_strike * 0.95:.2f}",
                'reasons': reasons,
                'position_size': '5-15% of portfolio',
                'time_horizon': '2-4 weeks',
                'volatility_bet': 'Short volatility'
            })
            
            return strategy_details
        
        return None
    
    def _analyze_iron_condors(self, current_price, gamma_flip, net_gex, call_walls, put_walls, strike_data):
        """Fixed iron condor analysis"""
        confidence = 0
        strategy_details = {
            'strategy': 'IRON CONDOR',
            'strategy_type': 'range_trading',
            'emoji': '🦅'
        }
        
        reasons = []
        
        if net_gex > 500e6:
            confidence += 30
            reasons.append(f"Very high positive GEX {net_gex/1e6:.0f}M creates strong range")
        elif net_gex > 200e6:
            confidence += 20
            reasons.append(f"Positive GEX {net_gex/1e6:.0f}M supports range trading")
        
        if len(call_walls) > 0 and len(put_walls) > 0:
            call_wall = call_walls.iloc[0]['strike']
            put_wall = put_walls.iloc[0]['strike']
            range_width = ((call_wall - put_wall) / current_price) * 100
            
            if range_width > 4:
                confidence += 35
                reasons.append(f"Clear range: ${put_wall:.2f} - ${call_wall:.2f} ({range_width:.1f}%)")
                
                range_center = (call_wall + put_wall) / 2
                center_distance = abs(current_price - range_center) / current_price * 100
                if center_distance < 1:
                    confidence += 15
                    reasons.append("Price well-centered in range")
            elif range_width > 2:
                confidence += 20
                reasons.append(f"Moderate range: {range_width:.1f}% wide")
        
        if len(strike_data) > 0:
            total_gamma = strike_data['net_gex'].abs().sum()
            gamma_concentration = max(strike_data['net_gex'].abs()) / total_gamma if total_gamma > 0 else 0
            
            if gamma_concentration < 0.3:
                confidence += 15
                reasons.append("Well-distributed gamma supports range")
        
        if confidence >= 60:
            if len(call_walls) > 0 and len(put_walls) > 0:
                short_call = call_walls.iloc[0]['strike']
                short_put = put_walls.iloc[0]['strike']
                
                strike_spacing = 5 if current_price < 200 else 10
                long_call = short_call + strike_spacing
                long_put = short_put - strike_spacing
                
                expected_range = f"${short_put:.2f} - ${short_call:.2f}"
                range_width = ((short_call - short_put) / current_price) * 100
                
            else:
                short_call = current_price * 1.03
                short_put = current_price * 0.97
                long_call = current_price * 1.05
                long_put = current_price * 0.95
                expected_range = f"±3% range"
                range_width = 6
            
            strategy_details.update({
                'confidence': min(confidence, 92),
                'recommended_strikes': {
                    'long_put': long_put,
                    'short_put': short_put,
                    'short_call': short_call,
                    'long_call': long_call
                },
                'optimal_dte': '21-45 DTE',
                'primary_target': '25% profit',
                'secondary_target': '50% profit',
                'expected_move': expected_range,
                'max_risk': f"${min(short_call - long_call, short_put - long_put):.2f} per spread",
                'expected_return': f"{int(range_width * 2)}-{int(range_width * 4)}%",
                'stop_loss': 'Close if either short strike threatened',
                'reasons': reasons,
                'position_size': '10-20% of portfolio',
                'time_horizon': '3-6 weeks',
                'volatility_bet': 'Short volatility + range-bound'
            })
            
            return strategy_details
        
        return None
    
    def get_current_price(self, symbol):
        """Get current stock price"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1d")
            if len(hist) > 0:
                return float(hist['Close'].iloc[-1])
            return None
        except:
            return None

# Educational content functions
def render_educational_content():
    """Render the educational section with interactive content"""
    
    st.markdown("## 🎓 GEX Trading University")
    st.markdown("Master gamma exposure analysis with interactive lessons and quizzes")
    
    # Educational tabs
    edu_tab1, edu_tab2, edu_tab3, edu_tab4 = st.tabs([
        "📚 Fundamentals", 
        "⚡ GEX Mechanics", 
        "📊 Strategy Deep Dive", 
        "🧠 Quiz & Practice"
    ])
    
    with edu_tab1:
        render_fundamentals()
    
    with edu_tab2:
        render_gex_mechanics()
    
    with edu_tab3:
        render_strategy_deep_dive()
    
    with edu_tab4:
        render_quiz_section()

def render_fundamentals():
    """Render the fundamentals education section"""
    
    st.markdown("### 📚 Options & Gamma Fundamentals")
    
    # Interactive concept explanations
    concept = st.selectbox(
        "Choose a concept to explore:",
        [
            "What is Gamma?",
            "Options Greeks Overview", 
            "Dealer Hedging Basics",
            "Volatility vs Gamma",
            "Time Decay Effects"
        ]
    )
    
    if concept == "What is Gamma?":
        st.markdown("""
        <div class="educational-card">
            <h4>🔬 Understanding Gamma</h4>
            <p><strong>Gamma</strong> measures how much an option's delta changes when the stock price moves $1.</p>
            
            <h5>Key Points:</h5>
            <ul>
                <li><strong>Highest at-the-money:</strong> ATM options have maximum gamma</li>
                <li><strong>Decreases away from strike:</strong> OTM and ITM options have lower gamma</li>
                <li><strong>Time sensitive:</strong> Shorter time = higher gamma for ATM options</li>
                <li><strong>Always positive:</strong> Both calls and puts have positive gamma</li>
            </ul>
            
            <h5>Why Gamma Matters for Trading:</h5>
            <p>When dealers sell options to retail traders, they become <em>short gamma</em>. 
            To hedge this risk, they must:</p>
            <ul>
                <li><strong>Buy stock when price rises</strong> (chase momentum up)</li>
                <li><strong>Sell stock when price falls</strong> (chase momentum down)</li>
            </ul>
            <p>This creates the <strong>gamma effect</strong> that amplifies price movements!</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Interactive gamma visualization
        st.markdown("#### 📊 Interactive Gamma Visualization")
        
        col1, col2 = st.columns(2)
        with col1:
            stock_price = st.slider("Stock Price", 90, 110, 100, 1)
        with col2:
            dte_input = st.slider("Days to Expiration", 1, 30, 7, 1)
        
        # Calculate theoretical gamma curve
        strikes = np.arange(85, 115, 1)
        gammas = []
        
        for strike in strikes:
            # Simplified gamma calculation for visualization
            moneyness = strike / stock_price
            time_factor = np.sqrt(dte_input / 365.0)
            gamma = 0.02 * np.exp(-2 * abs(moneyness - 1)) * time_factor
            gammas.append(gamma)
        
        fig_gamma = go.Figure()
        fig_gamma.add_trace(go.Scatter(
            x=strikes,
            y=gammas,
            mode='lines',
            name='Gamma Profile',
            line=dict(color='blue', width=3)
        ))
        
        fig_gamma.add_vline(x=stock_price, line_dash="dash", line_color="red",
                           annotation_text=f"Current Price: ${stock_price}")
        
        fig_gamma.update_layout(
            title=f"Gamma Profile - {dte_input} DTE",
            xaxis_title="Strike Price",
            yaxis_title="Gamma",
            height=400
        )
        
        st.plotly_chart(fig_gamma, use_container_width=True)
    
    elif concept == "Options Greeks Overview":
        st.markdown("""
        <div class="educational-card">
            <h4>🧮 The Options Greeks Family</h4>
            
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem; margin: 1rem 0;">
                <div style="background: #e8f5e9; padding: 1rem; border-radius: 8px;">
                    <h5>🔵 Delta (Δ)</h5>
                    <p><strong>Price sensitivity</strong></p>
                    <p>How much option price changes per $1 stock move</p>
                    <p>Calls: 0 to 1<br>Puts: -1 to 0</p>
                </div>
                
                <div style="background: #e3f2fd; padding: 1rem; border-radius: 8px;">
                    <h5>⚡ Gamma (Γ)</h5>
                    <p><strong>Delta acceleration</strong></p>
                    <p>How much delta changes per $1 stock move</p>
                    <p>Always positive<br>Highest ATM</p>
                </div>
                
                <div style="background: #fce4ec; padding: 1rem; border-radius: 8px;">
                    <h5>⏰ Theta (Θ)</h5>
                    <p><strong>Time decay</strong></p>
                    <p>How much value lost per day</p>
                    <p>Always negative<br>Accelerates near expiry</p>
                </div>
                
                <div style="background: #fff3e0; padding: 1rem; border-radius: 8px;">
                    <h5>🌊 Vega (ν)</h5>
                    <p><strong>Volatility sensitivity</strong></p>
                    <p>How much price changes per 1% IV move</p>
                    <p>Always positive<br>Higher for longer DTE</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    elif concept == "Dealer Hedging Basics":
        st.markdown("""
        <div class="educational-card">
            <h4>🏦 How Market Makers Hedge Options</h4>
            
            <h5>The Hedging Process:</h5>
            <ol>
                <li><strong>Retail trader buys call</strong> → Market maker sells call</li>
                <li><strong>Market maker is now short gamma</strong> → Must hedge delta risk</li>
                <li><strong>If stock goes up</strong> → Call delta increases → MM must buy more stock</li>
                <li><strong>If stock goes down</strong> → Call delta decreases → MM must sell stock</li>
            </ol>
            
            <div class="fun-fact">
                <h5>💡 Fun Fact</h5>
                <p>Market makers are essentially <strong>momentum traders by necessity</strong>! 
                They're forced to buy high and sell low to maintain delta neutrality.</p>
            </div>
            
            <h5>🎯 GEX Impact on Markets:</h5>
            <ul>
                <li><strong>High positive GEX:</strong> Dealers long gamma → Sell rallies, buy dips → Range-bound market</li>
                <li><strong>Negative GEX:</strong> Dealers short gamma → Buy rallies, sell dips → Trending market</li>
                <li><strong>Gamma flip point:</strong> Where dealers switch from long to short gamma</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

def render_gex_mechanics():
    """Render GEX mechanics education"""
    
    st.markdown("### ⚡ GEX Mechanics Deep Dive")
    
    mechanism = st.selectbox(
        "Explore GEX mechanisms:",
        [
            "GEX Calculation Method",
            "Positive vs Negative GEX",
            "Gamma Walls & Support/Resistance", 
            "Flip Point Dynamics",
            "Real Market Examples"
        ]
    )
    
    if mechanism == "GEX Calculation Method":
        st.markdown("""
        <div class="educational-card">
            <h4>🧮 How to Calculate GEX</h4>
            
            <h5>The Formula:</h5>
            <code>
            GEX = Stock Price × Gamma × Open Interest × 100 (multiplier)
            </code>
            
            <h5>For Each Strike:</h5>
            <ul>
                <li><strong>Call GEX = Positive</strong> (dealers must buy stock when price rises)</li>
                <li><strong>Put GEX = Negative</strong> (dealers must sell stock when price rises)</li>
            </ul>
            
            <h5>Net GEX Calculation:</h5>
            <code>
            Net GEX = Sum of all Call GEX - Sum of all Put GEX
            </code>
        </div>
        """, unsafe_allow_html=True)
        
        # Interactive GEX calculator
        st.markdown("#### 🧮 Interactive GEX Calculator")
        
        calc_col1, calc_col2, calc_col3 = st.columns(3)
        
        with calc_col1:
            calc_price = st.number_input("Stock Price", value=100.0, step=1.0)
            calc_gamma = st.number_input("Gamma", value=0.05, step=0.01, format="%.3f")
        
        with calc_col2:
            calc_oi = st.number_input("Open Interest", value=1000, step=100)
            option_type = st.selectbox("Option Type", ["Call", "Put"])
        
        with calc_col3:
            calc_gex = calc_price * calc_gamma * calc_oi * 100
            if option_type == "Put":
                calc_gex = -calc_gex
            
            st.metric("Calculated GEX", f"{calc_gex/1e6:.1f}M")
            st.markdown(f"**Raw Value:** {calc_gex:,.0f}")
    
    elif mechanism == "Positive vs Negative GEX":
        st.markdown("""
        <div class="educational-card">
            <h4>⚡ Market Regimes: Positive vs Negative GEX</h4>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; margin: 1rem 0;">
                <div style="background: linear-gradient(135deg, #e8f5e9, #c8e6c9); padding: 1.5rem; border-radius: 10px;">
                    <h5>🟢 Positive GEX Market</h5>
                    <p><strong>Dealers are LONG gamma</strong></p>
                    
                    <h6>Dealer Actions:</h6>
                    <ul>
                        <li>✅ Sell into rallies</li>
                        <li>✅ Buy into dips</li>
                        <li>✅ Provide liquidity</li>
                    </ul>
                    
                    <h6>Market Behavior:</h6>
                    <ul>
                        <li>📊 Range-bound trading</li>
                        <li>🛡️ Strong support/resistance</li>
                        <li>📉 Lower volatility</li>
                        <li>🔄 Mean reversion</li>
                    </ul>
                    
                    <h6>Best Strategies:</h6>
                    <ul>
                        <li>Iron Condors</li>
                        <li>Sell Premium</li>
                        <li>Range Trading</li>
                    </ul>
                </div>
                
                <div style="background: linear-gradient(135deg, #ffebee, #ffcdd2); padding: 1.5rem; border-radius: 10px;">
                    <h5>🔴 Negative GEX Market</h5>
                    <p><strong>Dealers are SHORT gamma</strong></p>
                    
                    <h6>Dealer Actions:</h6>
                    <ul>
                        <li>❌ Buy into rallies</li>
                        <li>❌ Sell into dips</li>
                        <li>⚡ Amplify momentum</li>
                    </ul>
                    
                    <h6>Market Behavior:</h6>
                    <ul>
                        <li>📈 Trending markets</li>
                        <li>💥 Explosive moves</li>
                        <li>📊 Higher volatility</li>
                        <li>🚀 Momentum continuation</li>
                    </ul>
                    
                    <h6>Best Strategies:</h6>
                    <ul>
                        <li>Long Calls/Puts</li>
                        <li>Straddles/Strangles</li>
                        <li>Momentum Trading</li>
                    </ul>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    elif mechanism == "Gamma Walls & Support/Resistance":
        st.markdown("""
        <div class="educational-card">
            <h4>🏗️ Understanding Gamma Walls</h4>
            
            <h5>What Are Gamma Walls?</h5>
            <p>Strikes with heavy gamma concentration that act as <strong>magnetic price levels</strong></p>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin: 1rem 0;">
                <div style="background: #e8f5e9; padding: 1rem; border-radius: 8px;">
                    <h6>🟢 Call Walls (Resistance)</h6>
                    <ul>
                        <li>High call open interest</li>
                        <li>Dealers must SELL stock as price approaches</li>
                        <li>Creates resistance levels</li>
                        <li>Price tends to get "pinned" below</li>
                    </ul>
                </div>
                
                <div style="background: #ffebee; padding: 1rem; border-radius: 8px;">
                    <h6>🔴 Put Walls (Support)</h6>
                    <ul>
                        <li>High put open interest</li>
                        <li>Dealers must BUY stock as price approaches</li>
                        <li>Creates support levels</li>
                        <li>Price tends to get "defended" above</li>
                    </ul>
                </div>
            </div>
            
            <div class="fun-fact">
                <h5>💡 Wall Strength Indicator</h5>
                <p>A gamma wall with >100M GEX is considered <strong>very strong</strong> and likely to hold</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

def render_strategy_deep_dive():
    """Render strategy deep dive education"""
    
    st.markdown("### 📊 Strategy Deep Dive")
    
    strategy_focus = st.selectbox(
        "Choose strategy to master:",
        [
            "Long Calls (Gamma Squeeze)",
            "Long Puts (Breakdown Play)", 
            "Selling Calls (at Resistance)",
            "Selling Puts (at Support)",
            "Iron Condors (Range Trading)",
            "Risk Management Rules"
        ]
    )
    
    if strategy_focus == "Long Calls (Gamma Squeeze)":
        st.markdown("""
        <div class="educational-card">
            <h4>🚀 Long Calls: The Gamma Squeeze Setup</h4>
            
            <h5>🎯 Ideal Conditions:</h5>
            <ol>
                <li><strong>Negative Net GEX < -100M</strong> (dealers short gamma)</li>
                <li><strong>Price below gamma flip point</strong> by 0.5-1.5%</li>
                <li><strong>Strong put wall support</strong> within 1% below</li>
                <li><strong>Low gamma at current price</strong> (less dealer resistance)</li>
            </ol>
            
            <h5>📈 The Squeeze Mechanics:</h5>
            <div style="background: #f0f8ff; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
                <p><strong>Step 1:</strong> Retail buys calls → Dealers sell calls (now short gamma)</p>
                <p><strong>Step 2:</strong> Price starts moving up → Delta increases rapidly</p>
                <p><strong>Step 3:</strong> Dealers forced to buy stock to hedge → Creates buying pressure</p>
                <p><strong>Step 4:</strong> More buying → Price rises more → More hedging needed</p>
                <p><strong>Result:</strong> Self-reinforcing feedback loop! 🔄</p>
            </div>
            
            <h5>⚙️ Trade Setup:</h5>
            <ul>
                <li><strong>Strikes:</strong> ATM or first OTM above flip point</li>
                <li><strong>DTE:</strong> 2-5 days (maximum gamma sensitivity)</li>
                <li><strong>Size:</strong> 2-3% of portfolio (high risk/reward)</li>
                <li><strong>Target:</strong> First major call wall (100%+ gains possible)</li>
                <li><strong>Stop:</strong> 50% loss or momentum failure</li>
            </ul>
            
            <div class="risk-box">
                <h6>⚠️ High Risk Strategy</h6>
                <p>Gamma squeezes can fail spectacularly if momentum doesn't materialize. 
                Always size positions for 100% loss possibility.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    elif strategy_focus == "Iron Condors (Range Trading)":
        st.markdown("""
        <div class="educational-card">
            <h4>🦅 Iron Condors: Profiting from Range-Bound Markets</h4>
            
            <h5>🎯 Perfect Setup Conditions:</h5>
            <ul>
                <li><strong>High positive GEX > 500M</strong> (strong range environment)</li>
                <li><strong>Clear call and put walls > 4% apart</strong></li>
                <li><strong>Price centered between walls</strong></li>
                <li><strong>Well-distributed gamma</strong> (no single strike dominance)</li>
            </ul>
            
            <h5>🏗️ Construction Method:</h5>
            <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
                <p><strong>Short Strikes:</strong> Place at the gamma walls</p>
                <p><strong>Long Strikes:</strong> 1-2 strikes beyond walls for protection</p>
                <p><strong>Example Setup:</strong></p>
                <ul>
                    <li>Buy 95 Put</li>
                    <li>Sell 98 Put (put wall)</li>
                    <li>Sell 102 Call (call wall)</li>
                    <li>Buy 105 Call</li>
                </ul>
            </div>
            
            <h5>📊 Profit Mechanics:</h5>
            <ul>
                <li><strong>Time decay:</strong> Earn theta on both short strikes</li>
                <li><strong>Volatility crush:</strong> Benefit from decreasing IV</li>
                <li><strong>Range-bound price action:</strong> Price stays between short strikes</li>
            </ul>
            
            <h5>🎯 Management Rules:</h5>
            <ul>
                <li><strong>Profit target:</strong> 25-50% of maximum profit</li>
                <li><strong>Loss limit:</strong> Close if either short strike threatened</li>
                <li><strong>Time management:</strong> Close at 7-10 DTE if unprofitable</li>
                <li><strong>Adjustment:</strong> Roll threatened side if >14 DTE remaining</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Interactive condor P&L diagram
        st.markdown("#### 📊 Interactive Iron Condor P&L")
        
        pnl_col1, pnl_col2 = st.columns(2)
        with pnl_col1:
            current_price_pnl = st.slider("Current Stock Price", 90, 110, 100, 1)
            short_put = st.slider("Short Put Strike", 85, 95, 92, 1)
            short_call = st.slider("Short Call Strike", 105, 115, 108, 1)
        
        with pnl_col2:
            long_put = short_put - 3
            long_call = short_call + 3
            premium_collected = st.slider("Premium Collected", 0.5, 3.0, 1.5, 0.1)
        
        # Calculate P&L at different prices
        prices = np.arange(85, 115, 1)
        pnls = []
        
        for price in prices:
            if price <= long_put:
                pnl = -(short_put - long_put) + premium_collected
            elif price < short_put:
                pnl = -(short_put - price) + premium_collected
            elif price <= short_call:
                pnl = premium_collected
            elif price < long_call:
                pnl = -(price - short_call) + premium_collected
            else:
                pnl = -(long_call - short_call) + premium_collected
            
            pnls.append(pnl * 100)  # Convert to dollars
        
        fig_pnl = go.Figure()
        fig_pnl.add_trace(go.Scatter(
            x=prices,
            y=pnls,
            mode='lines',
            name='P&L',
            line=dict(color='blue', width=3)
        ))
        
        fig_pnl.add_vline(x=current_price_pnl, line_dash="dash", line_color="red",
                         annotation_text=f"Current: ${current_price_pnl}")
        fig_pnl.add_hline(y=0, line_dash="dot", line_color="gray")
        
        fig_pnl.update_layout(
            title="Iron Condor Profit/Loss Diagram",
            xaxis_title="Stock Price at Expiration",
            yaxis_title="Profit/Loss ($)",
            height=400
        )
        
        st.plotly_chart(fig_pnl, use_container_width=True)

def render_quiz_section():
    """Render interactive quiz section"""
    
    st.markdown("### 🧠 Test Your GEX Knowledge")
    
    quiz_type = st.selectbox(
        "Choose quiz difficulty:",
        ["Beginner", "Intermediate", "Advanced", "Expert"]
    )
    
    if quiz_type == "Beginner":
        render_beginner_quiz()
    elif quiz_type == "Intermediate":
        render_intermediate_quiz()
    elif quiz_type == "Advanced":
        render_advanced_quiz()
    else:
        render_expert_quiz()

def render_beginner_quiz():
    """Beginner level quiz"""
    
    st.markdown("#### 🌱 Beginner Quiz: GEX Basics")
    
    if 'quiz_score' not in st.session_state:
        st.session_state.quiz_score = 0
        st.session_state.quiz_answers = {}
    
    # Question 1
    st.markdown("**Question 1:** What does GEX stand for?")
    q1_options = [
        "Gamma Exchange",
        "Gamma Exposure", 
        "Greek Exchange",
        "Gamma Extension"
    ]
    
    q1_answer = st.radio("Choose your answer:", q1_options, key="q1")
    
    if st.button("Check Answer 1"):
        if q1_answer == "Gamma Exposure":
            st.success("✅ Correct! GEX stands for Gamma Exposure")
            st.session_state.quiz_answers['q1'] = True
        else:
            st.error("❌ Incorrect. GEX stands for Gamma Exposure")
            st.session_state.quiz_answers['q1'] = False
    
    # Question 2
    st.markdown("**Question 2:** When net GEX is positive, what type of market environment typically results?")
    q2_options = [
        "Highly volatile trending market",
        "Range-bound market with lower volatility",
        "Unpredictable random walk",
        "Always bearish market"
    ]
    
    q2_answer = st.radio("Choose your answer:", q2_options, key="q2")
    
    if st.button("Check Answer 2"):
        if q2_answer == "Range-bound market with lower volatility":
            st.success("✅ Correct! Positive GEX creates range-bound, lower volatility environments")
            st.session_state.quiz_answers['q2'] = True
        else:
            st.error("❌ Incorrect. Positive GEX typically creates range-bound markets with lower volatility")
            st.session_state.quiz_answers['q2'] = False
    
    # Question 3
    st.markdown("**Question 3:** What are 'gamma walls'?")
    q3_options = [
        "Physical barriers in trading floors",
        "Strikes with high gamma concentration that act as support/resistance",
        "A type of options strategy",
        "Software trading algorithms"
    ]
    
    q3_answer = st.radio("Choose your answer:", q3_options, key="q3")
    
    if st.button("Check Answer 3"):
        if q3_answer == "Strikes with high gamma concentration that act as support/resistance":
            st.success("✅ Correct! Gamma walls are strikes with heavy gamma that create support/resistance")
            st.session_state.quiz_answers['q3'] = True
        else:
            st.error("❌ Incorrect. Gamma walls are strikes with high gamma concentration that act as support/resistance levels")
            st.session_state.quiz_answers['q3'] = False
    
    # Show final score
    if len(st.session_state.quiz_answers) == 3:
        score = sum(st.session_state.quiz_answers.values())
        st.markdown(f"### 📊 Your Score: {score}/3")
        
        if score == 3:
            st.balloons()
            st.success("🎉 Perfect score! You're ready for intermediate level!")
        elif score == 2:
            st.info("📚 Good job! Review the concepts and try again")
        else:
            st.warning("📖 Keep studying! Review the educational content above")

def render_intermediate_quiz():
    """Intermediate level quiz"""
    
    st.markdown("#### 📈 Intermediate Quiz: Strategy Applications")
    
    # More complex scenario-based questions
    st.markdown("""
    **Scenario:** SPY is trading at $420, gamma flip is at $415, net GEX is -800M, 
    and there's a strong call wall at $425 with 300M GEX.
    """)
    
    st.markdown("**Question:** What's the best strategy setup?")
    scenario_options = [
        "Sell iron condors - range-bound market",
        "Buy calls targeting the gamma flip",
        "Buy calls targeting the call wall at $425", 
        "Sell puts at current support levels"
    ]
    
    scenario_answer = st.radio("Your strategy choice:", scenario_options)
    
    if st.button("Check Strategy Answer"):
        if scenario_answer == "Buy calls targeting the call wall at $425":
            st.success("""
            ✅ Excellent analysis! 
            
            **Reasoning:**
            - Negative GEX (-800M) = squeeze potential
            - Price above flip = momentum setup  
            - Call wall at $425 = clear target
            - 5 points upside = good risk/reward
            """)
        else:
            st.error("""
            ❌ Not optimal for this setup.
            
            **Key factors you might have missed:**
            - Large negative GEX suggests squeeze potential
            - Price above flip point favors bullish momentum
            - Call wall provides clear price target
            """)

# Initialize scanner
@st.cache_resource
def get_scanner():
    return EnhancedOptionsScanner()

scanner = get_scanner()

# Main interface with tabs including education
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍 Mass Scanner", 
    "🎯 Custom Analysis", 
    "📊 Detailed GEX", 
    "⚡ Strategies", 
    "🎓 Education"
])

with tab1:
    # Mass scanner (existing functionality)
    st.markdown("## 🔍 Real-Time Options Volume & GEX Scanner")
    
    st.sidebar.header("🎛️ Scanner Configuration")
    
    max_symbols = st.sidebar.slider("Max Symbols", 25, 150, 50, 5)
    min_confidence = st.sidebar.slider("Min Confidence %", 0, 100, 25, 5)
    volume_threshold = st.sidebar.number_input("Min Volume", 1000, 100000, 5000, 1000)
    
    signal_filters = st.sidebar.multiselect(
        "Signal Types",
        ["SQUEEZE_POTENTIAL", "RANGE_BOUND", "BEARISH_FLOW", "BULLISH_FLOW"],
        default=["SQUEEZE_POTENTIAL", "RANGE_BOUND"]
    )
    
    if st.sidebar.button("🚀 Start Scan", type="primary"):
        st.cache_data.clear()
        st.rerun()
    
    # Volume ranking
    with st.expander("📊 Dynamic Options Volume Ranking", expanded=True):
        st.markdown("🔄 Scanning for active options...")
        
        volume_data = scanner.get_options_volume_ranking()
        
        if len(volume_data) > 0:
            st.success(f"✅ Found {len(volume_data)} symbols")
            
            top_20 = volume_data.head(20)
            
            fig_volume = go.Figure()
            fig_volume.add_trace(go.Bar(
                x=top_20['symbol'],
                y=top_20['total_volume'],
                name='Volume',
                marker_color='rgba(54, 162, 235, 0.8)'
            ))
            
            fig_volume.update_layout(
                title="Top 20 by Options Volume",
                height=400
            )
            
            st.plotly_chart(fig_volume, use_container_width=True)
            
            display_cols = ['symbol', 'current_price', 'total_volume', 'total_oi', 'put_call_ratio']
            st.dataframe(top_20[display_cols], use_container_width=True)
        else:
            st.error("❌ No data found")
            st.stop()
    
    # GEX analysis
    with st.expander("⚡ GEX Signal Analysis", expanded=True):
        filtered_volume = volume_data[volume_data['total_volume'] >= volume_threshold]
        
        if len(filtered_volume) == 0:
            st.warning(f"⚠️ No symbols meet volume threshold")
            st.stop()
        
        scan_results = scanner.scan_for_signals(filtered_volume, max_symbols)
        
        if len(scan_results) > 0:
            filtered_results = scan_results[scan_results['confidence'] >= min_confidence]
            
            if len(signal_filters) > 0:
                filtered_results = filtered_results[
                    filtered_results['signals'].apply(
                        lambda x: any(signal in x for signal in signal_filters)
                    )
                ]
            
            st.success(f"✅ Found {len(filtered_results)} signals")
            
            if len(filtered_results) > 0:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    high_conf = len(filtered_results[filtered_results['confidence'] >= 75])
                    st.metric("High Confidence", high_conf)
                
                with col2:
                    avg_confidence = filtered_results['confidence'].mean()
                    st.metric("Avg Confidence", f"{avg_confidence:.0f}%")
                
                with col3:
                    total_gex = filtered_results['net_gex_millions'].sum()
                    st.metric("Total Net GEX", f"{total_gex:.0f}M")
                
                with col4:
                    bullish_signals = len(filtered_results[filtered_results['net_gex_millions'] > 0])
                    st.metric("Bullish Signals", bullish_signals)
                
                # Results display
                for _, result in filtered_results.head(20).iterrows():
                    confidence = result['confidence']
                    
                    if confidence >= 75:
                        card_class = "scanner-card high-signal"
                        conf_emoji = "🟢"
                    elif confidence >= 50:
                        card_class = "scanner-card medium-signal"  
                        conf_emoji = "🟡"
                    else:
                        card_class = "scanner-card low-signal"
                        conf_emoji = "🔴"
                    
                    signals_text = ", ".join(result['signals']) if result['signals'] else "No signals"
                    
                    st.markdown(f"""
                    <div class="{card_class}">
                        <h4>{conf_emoji} {result['symbol']} - {confidence:.0f}% Confidence</h4>
                        <div class="metric-grid">
                            <div><strong>Price:</strong> ${result['current_price']:.2f}</div>
                            <div><strong>Net GEX:</strong> {result['net_gex_millions']:.1f}M</div>
                            <div><strong>Volume:</strong> {result['total_volume']:,}</div>
                            <div><strong>P/C Ratio:</strong> {result['put_call_ratio']}</div>
                        </div>
                        <p><strong>Signals:</strong> {signals_text}</p>
                    </div>
                    """, unsafe_allow_html=True)

with tab2:
    # Custom symbol analysis (existing)
    st.markdown("## 🎯 Custom Symbol Analysis")
    
    custom_symbol = st.text_input("Enter Symbol", placeholder="e.g., AAPL")
    
    if custom_symbol:
        with st.spinner(f"Analyzing {custom_symbol}..."):
            analysis = scanner.analyze_custom_symbol(custom_symbol)
        
        if analysis and 'error' not in analysis:
            st.success(f"✅ Analysis complete for {analysis['symbol']}")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Price", f"${analysis['current_price']:.2f}", 
                         f"{analysis['price_change']:+.2f}%")
            
            with col2:
                net_gex_m = analysis['net_gex'] / 1e6
                st.metric("Net GEX", f"{net_gex_m:.0f}M")
            
            with col3:
                st.metric("Volume", f"{analysis['total_volume']:,}")
            
            with col4:
                st.metric("Confidence", f"{analysis['confidence']:.0f}%")

with tab3:
    # Detailed GEX profile (existing)
    st.markdown("## 📊 Detailed GEX Profile")
    
    detail_symbol = st.text_input("Symbol for Detail", placeholder="e.g., SPY")
    
    if detail_symbol:
        gex_profile = scanner.get_detailed_gex_profile(detail_symbol)
        
        if gex_profile:
            st.success(f"GEX profile for {detail_symbol}")
            # Add detailed charts and analysis here

with tab4:
    # Fixed strategies section
    st.markdown("## ⚡ Options Strategies")
    
    strategy_symbol = st.text_input("Strategy Symbol", placeholder="e.g., AAPL")
    
    if strategy_symbol:
        with st.spinner(f"Analyzing strategies for {strategy_symbol}..."):
            strategy_analysis = scanner.analyze_options_strategies(strategy_symbol)
        
        if strategy_analysis and strategy_analysis.get('strategies'):
            st.success(f"Strategy analysis for {strategy_analysis['symbol']}")
            
            strategies = strategy_analysis['strategies']
            
            for strategy in strategies:
                confidence = strategy.get('confidence', 0)
                
                if confidence >= 75:
                    strategy_class = "strategy-card strategy-high"
                    conf_class = "conf-high"
                    conf_text = "HIGH"
                elif confidence >= 60:
                    strategy_class = "strategy-card strategy-medium"
                    conf_class = "conf-medium"
                    conf_text = "MEDIUM"
                else:
                    strategy_class = "strategy-card strategy-low" 
                    conf_class = "conf-low"
                    conf_text = "LOWER"
                
                st.markdown(f"""
                <div class="{strategy_class}">
                    <h2>{strategy.get('emoji', '⚡')} {strategy.get('strategy', 'Strategy')} 
                    <span class="confidence-badge {conf_class}">{confidence:.0f}% {conf_text}</span>
                    </h2>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**Strategy Details**")
                    st.markdown(f"• Type: {strategy.get('strategy_type', 'N/A')}")
                    st.markdown(f"• DTE: {strategy.get('optimal_dte', 'N/A')}")
                    st.markdown(f"• Size: {strategy.get('position_size', 'N/A')}")
                
                with col2:
                    st.markdown("**Targets & Risk**")
                    st.markdown(f"• Target: {strategy.get('primary_target', 'N/A')}")
                    st.markdown(f"• Return: {strategy.get('expected_return', 'N/A')}")
                    st.markdown(f"• Risk: {strategy.get('max_risk', 'N/A')}")
                
                with col3:
                    st.markdown("**Rationale**")
                    for reason in strategy.get('reasons', [])[:3]:
                        st.markdown(f"• {reason}")
        else:
            st.warning("No strategies identified")

with tab5:
    # Educational content
    render_educational_content()

# Footer
st.markdown("---")
st.markdown("""
**⚠️ EDUCATIONAL PURPOSE ONLY:** This scanner uses real market data for educational analysis. 
Options trading involves substantial risk. Always conduct thorough research and consult professionals.
""")
