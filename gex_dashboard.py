#!/usr/bin/env python3
"""
Complete Dynamic GEX Scanner with Options Strategies
Real-time analysis with improved styling and readability
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
    page_title="🎯 Dynamic GEX Scanner",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Fixed professional styling with better readability
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
    }
    
    .scanner-card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        color: #333;
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
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🎯 Dynamic GEX Scanner</h1>
    <p><strong>Real-Time Analysis</strong> | Live options data with advanced strategies</p>
    <p>Professional gamma exposure analysis with institutional-grade recommendations</p>
</div>
""", unsafe_allow_html=True)

class DynamicOptionsScanner:
    """Dynamic scanner with real-time list updates and custom symbol analysis"""
    
    def __init__(self):
        self.risk_free_rate = 0.05
        self.scan_lock = Lock()
        self.results_cache = {}
        
        # Base symbols (will be dynamically updated)
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
            "LI", "RIVN", "LCID", "F", "GM", "COIN", "HOOD", "SOFI", "ARKK", "ARKQ",
            
            # Finance & Energy
            "GS", "MS", "C", "WFC", "USB", "PNC", "COF", "AXP", "BLK", "SCHW",
            "SLB", "HAL", "OXY", "COP", "EOG", "DVN", "MRO", "APA", "FANG", "PXD"
        ]
    
    @st.cache_data(ttl=1800)  # 30 minutes cache
    def get_dynamic_symbol_list(_self):
        """Get dynamic list of most active options symbols"""
        
        # Method 1: Use predefined high-activity symbols
        active_symbols = _self.base_symbols.copy()
        
        # Method 2: Add popular symbols from various sources
        try:
            # Get trending symbols (simplified approach)
            trending_symbols = [
                "PTON", "PELOTON", "ROKU", "ZM", "DOCU", "WORK", "SNOW", "ABNB",
                "DKNG", "PENN", "MGM", "WYNN", "LVS", "CZR", "BYD", "VALE",
                "X", "CLF", "MT", "SCCO", "FCX", "AA", "CENX", "STLD"
            ]
            active_symbols.extend(trending_symbols)
            
            # Remove duplicates and invalid symbols
            active_symbols = list(set(active_symbols))
            
        except Exception as e:
            st.warning(f"Using base symbol list: {e}")
        
        return active_symbols[:150]  # Limit to 150 symbols
    
    @st.cache_data(ttl=3600)
    def get_options_volume_ranking(_self):
        """Get current options volume ranking for dynamic symbol list"""
        
        # Get dynamic symbol list
        symbols_to_check = _self.get_dynamic_symbol_list()
        
        volume_data = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, symbol in enumerate(symbols_to_check):
            try:
                status_text.text(f"🔍 Checking options volume: {symbol} ({i+1}/{len(symbols_to_check)})")
                progress_bar.progress((i + 1) / len(symbols_to_check))
                
                ticker = yf.Ticker(symbol)
                
                # Get current price
                hist = ticker.history(period="1d")
                if len(hist) == 0:
                    continue
                    
                current_price = hist['Close'].iloc[-1]
                
                # Get first expiration to check options availability
                try:
                    exp_dates = ticker.options
                    if not exp_dates:
                        continue
                        
                    # Get options chain for first expiration
                    chain = ticker.option_chain(exp_dates[0])
                    
                    # Calculate total volume and open interest
                    call_volume = chain.calls['volume'].fillna(0).sum()
                    put_volume = chain.puts['volume'].fillna(0).sum()
                    total_volume = call_volume + put_volume
                    
                    call_oi = chain.calls['openInterest'].fillna(0).sum()
                    put_oi = chain.puts['openInterest'].fillna(0).sum()
                    total_oi = call_oi + put_oi
                    
                    # Only include if there's significant activity
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
            # Sort by total volume (most active first)
            df = df.sort_values('total_volume', ascending=False).reset_index(drop=True)
            return df
        
        return pd.DataFrame()
    
    def calculate_quick_gex(self, symbol, current_price):
        """Quick GEX calculation for scanning"""
        try:
            ticker = yf.Ticker(symbol)
            exp_dates = ticker.options[:3]  # Only first 3 expirations for speed
            
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
                    T = dte / 365.0
                    
                    # Quick gamma approximation
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
                            total_gex -= put_gex  # Puts are negative
                
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
                
                # Quick GEX calculation
                gex_data = self.calculate_quick_gex(symbol, current_price)
                
                net_gex = gex_data['net_gex']
                net_gex_millions = net_gex / 1e6
                
                # Simple signal detection
                signals = []
                confidence = 0
                
                # Negative GEX squeeze potential
                if net_gex < -50e6:  # Less than -50M
                    signals.append("SQUEEZE_POTENTIAL")
                    confidence += 30
                
                # High positive GEX (range bound)
                if net_gex > 100e6:  # More than 100M
                    signals.append("RANGE_BOUND")
                    confidence += 25
                
                # High options volume
                if row['total_volume'] > 10000:
                    confidence += 20
                
                # High open interest
                if row['total_oi'] > 50000:
                    confidence += 15
                
                # Put/call ratio analysis
                pcr = row['put_call_ratio']
                if pcr > 1.5:  # Heavy put activity
                    signals.append("BEARISH_FLOW")
                    confidence += 10
                elif pcr < 0.5:  # Heavy call activity
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
                
            except Exception as e:
                return None
        
        # Process symbols in parallel for speed
        results = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Take top symbols by volume
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
                except Exception as e:
                    continue
        
        progress_bar.empty()
        status_text.empty()
        
        if results:
            results_df = pd.DataFrame(results)
            # Sort by confidence score
            results_df = results_df.sort_values('confidence', ascending=False).reset_index(drop=True)
            return results_df
        
        return pd.DataFrame()
    
    def analyze_custom_symbol(self, symbol):
        """Comprehensive analysis of a single custom symbol"""
        try:
            symbol = symbol.upper().strip()
            
            # Validate symbol
            ticker = yf.Ticker(symbol)
            
            # Get basic info
            info = ticker.info
            hist = ticker.history(period="5d")
            
            if len(hist) == 0:
                return None
            
            current_price = hist['Close'].iloc[-1]
            prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
            price_change = ((current_price - prev_close) / prev_close) * 100
            
            # Get options data
            try:
                exp_dates = ticker.options
                if not exp_dates:
                    return {
                        'symbol': symbol,
                        'error': 'No options available for this symbol',
                        'current_price': current_price,
                        'price_change': price_change
                    }
                
                # Analyze multiple expirations
                all_analysis = []
                total_call_gex = 0
                total_put_gex = 0
                total_volume = 0
                total_oi = 0
                
                for exp_date in exp_dates[:5]:  # First 5 expirations
                    try:
                        exp_dt = datetime.strptime(exp_date, '%Y-%m-%d')
                        dte = (exp_dt.date() - date.today()).days
                        
                        if dte <= 0 or dte > 60:
                            continue
                        
                        chain = ticker.option_chain(exp_date)
                        
                        # Calculate GEX for this expiration
                        exp_call_gex = 0
                        exp_put_gex = 0
                        exp_volume = 0
                        exp_oi = 0
                        
                        # Process calls
                        for _, call in chain.calls.iterrows():
                            if call['openInterest'] > 0:
                                # Improved gamma calculation
                                moneyness = call['strike'] / current_price
                                time_factor = np.sqrt(dte / 365.0)
                                gamma = 0.02 * np.exp(-2 * abs(moneyness - 1)) * time_factor
                                
                                call_gex = current_price * gamma * call['openInterest'] * 100
                                exp_call_gex += call_gex
                                exp_volume += call.get('volume', 0)
                                exp_oi += call['openInterest']
                        
                        # Process puts  
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
                        
                    except Exception as e:
                        continue
                
                if not all_analysis:
                    return {
                        'symbol': symbol,
                        'error': 'Unable to analyze options data',
                        'current_price': current_price,
                        'price_change': price_change
                    }
                
                # Calculate overall metrics
                net_gex = total_call_gex - total_put_gex
                
                # Determine signals and confidence
                signals = []
                confidence = 0
                
                # GEX-based signals
                if net_gex < -50e6:
                    signals.append("NEGATIVE_GEX_SQUEEZE")
                    confidence += 35
                elif net_gex > 100e6:
                    signals.append("POSITIVE_GEX_RANGE")
                    confidence += 25
                
                # Volume-based signals
                if total_volume > 5000:
                    signals.append("HIGH_VOLUME")
                    confidence += 20
                
                # Open interest signals
                if total_oi > 25000:
                    signals.append("HIGH_OPEN_INTEREST")
                    confidence += 15
                
                # Price movement signals
                if abs(price_change) > 3:
                    signals.append("HIGH_MOMENTUM")
                    confidence += 10
                
                # Put/call ratio
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
            
            exp_dates = ticker.options[:3]  # First 3 expirations
            all_strikes = []
            
            for exp_date in exp_dates:
                try:
                    exp_dt = datetime.strptime(exp_date, '%Y-%m-%d')
                    dte = (exp_dt.date() - date.today()).days
                    
                    if dte <= 0 or dte > 45:
                        continue
                    
                    chain = ticker.option_chain(exp_date)
                    
                    # Process all strikes
                    strikes_data = {}
                    
                    # Calls
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
                    
                    # Puts
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
                            'put_gex': -data['put_gex'],  # Negative for puts
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
                
                # Aggregate by strike
                strike_gex = df.groupby('strike').agg({
                    'call_gex': 'sum',
                    'put_gex': 'sum',
                    'net_gex': 'sum',
                    'call_oi': 'sum',
                    'put_oi': 'sum'
                }).reset_index()
                
                strike_gex = strike_gex.sort_values('strike').reset_index(drop=True)
                strike_gex['cumulative_gex'] = strike_gex['net_gex'].cumsum()
                
                # Find gamma flip
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
            
        except Exception as e:
            return None
    
    def analyze_options_strategies(self, symbol, gex_data=None):
        """Advanced options strategy analysis with price targets and confidence"""
        try:
            if gex_data is None:
                # Get comprehensive GEX analysis
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
            
            # Calculate key metrics
            net_gex = strike_data['net_gex'].sum() if len(strike_data) > 0 else 0
            distance_to_flip = ((current_price - gamma_flip) / current_price) * 100
            
            strategies = []
            
            # 1. LONG CALL ANALYSIS
            long_call_analysis = self._analyze_long_calls(
                current_price, gamma_flip, net_gex, distance_to_flip, call_walls, strike_data
            )
            if long_call_analysis:
                strategies.append(long_call_analysis)
            
            # 2. LONG PUT ANALYSIS
            long_put_analysis = self._analyze_long_puts(
                current_price, gamma_flip, net_gex, distance_to_flip, put_walls, strike_data
            )
            if long_put_analysis:
                strategies.append(long_put_analysis)
            
            # 3. SELL CALL ANALYSIS
            sell_call_analysis = self._analyze_sell_calls(
                current_price, gamma_flip, net_gex, call_walls, strike_data
            )
            if sell_call_analysis:
                strategies.append(sell_call_analysis)
            
            # 4. SELL PUT ANALYSIS
            sell_put_analysis = self._analyze_sell_puts(
                current_price, gamma_flip, net_gex, put_walls, strike_data
            )
            if sell_put_analysis:
                strategies.append(sell_put_analysis)
            
            # 5. IRON CONDOR ANALYSIS
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
                'strategies': sorted(strategies, key=lambda x: x['confidence'], reverse=True),
                'analysis_time': datetime.now()
            }
            
        except Exception as e:
            return None
    
    def _analyze_long_calls(self, current_price, gamma_flip, net_gex, distance_to_flip, call_walls, strike_data):
        """Analyze long call opportunities"""
        confidence = 0
        strategy_details = {
            'strategy': 'LONG CALLS',
            'strategy_type': 'directional_bullish',
            'emoji': '🚀'
        }
        
        # Conditions for long calls
        reasons = []
        
        # 1. Negative GEX environment (volatility amplification)
        if net_gex < -100e6:
            confidence += 35
            reasons.append(f"Negative GEX {net_gex/1e6:.0f}M creates volatility amplification")
        
        # 2. Price below gamma flip
        if distance_to_flip < -0.5:
            confidence += 25
            reasons.append(f"Price {abs(distance_to_flip):.1f}% below gamma flip point")
        
        # 3. Strong put wall support
        if len(call_walls) > 0:
            nearest_call_wall = call_walls.iloc[0]['strike']
            wall_distance = ((nearest_call_wall - current_price) / current_price) * 100
            if wall_distance > 2:  # At least 2% upside to wall
                confidence += 20
                reasons.append(f"Call wall resistance at ${nearest_call_wall:.2f} (+{wall_distance:.1f}%)")
        
        # 4. Low gamma environment around current price
        if len(strike_data) > 0:
            atm_strikes = strike_data[
                (strike_data['strike'] >= current_price * 0.98) & 
                (strike_data['strike'] <= current_price * 1.02)
            ]
            if len(atm_strikes) > 0:
                avg_gamma = atm_strikes['net_gex'].mean()
                if avg_gamma < 50e6:  # Low gamma at the money
                    confidence += 15
                    reasons.append("Low gamma at-the-money reduces dealer hedging")
        
        if confidence >= 50:
            # Determine optimal strikes and price targets
            atm_strike = round(current_price / 5) * 5  # Round to nearest $5
            target_strikes = [atm_strike, atm_strike + 5]
            
            # Price targets
            if len(call_walls) > 0:
                primary_target = call_walls.iloc[0]['strike']
                secondary_target = call_walls.iloc[1]['strike'] if len(call_walls) > 1 else primary_target * 1.05
            else:
                primary_target = current_price * 1.05
                secondary_target = current_price * 1.08
            
            # Expected move calculation
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
        """Analyze long put opportunities"""
        confidence = 0
        strategy_details = {
            'strategy': 'LONG PUTS',
            'strategy_type': 'directional_bearish',
            'emoji': '📉'
        }
        
        reasons = []
        
        # 1. High positive GEX near flip point (breakdown potential)
        if net_gex > 200e6 and abs(distance_to_flip) < 0.5:
            confidence += 40
            reasons.append(f"High positive GEX {net_gex/1e6:.0f}M near flip point - breakdown setup")
        
        # 2. Price rejection from call wall
        if len(put_walls) > 0:
            nearest_put_wall = put_walls.iloc[0]['strike']
            wall_distance = ((current_price - nearest_put_wall) / current_price) * 100
            if wall_distance > 1:  # At least 1% downside to support
                confidence += 25
                reasons.append(f"Put wall support at ${nearest_put_wall:.2f} (-{wall_distance:.1f}%)")
        
        # 3. High gamma concentration above (dealer resistance)
        if len(strike_data) > 0:
            above_strikes = strike_data[strike_data['strike'] > current_price * 1.02]
            if len(above_strikes) > 0:
                call_gex_above = above_strikes['call_gex'].sum()
                if call_gex_above > 200e6:
                    confidence += 20
                    reasons.append(f"Strong call gamma above: {call_gex_above/1e6:.0f}M")
        
        # 4. Distance from gamma flip suggests vulnerability
        if distance_to_flip > 0.3:
            confidence += 15
            reasons.append(f"Price {distance_to_flip:.1f}% above flip - vulnerable to breakdown")
        
        if confidence >= 50:
            # Determine optimal strikes and targets
            atm_strike = round(current_price / 5) * 5
            target_strikes = [atm_strike, atm_strike - 5]
            
            # Price targets
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
        """Analyze call selling opportunities"""
        confidence = 0
        strategy_details = {
            'strategy': 'SELL CALLS',
            'strategy_type': 'premium_collection',
            'emoji': '💰'
        }
        
        reasons = []
        
        # 1. High positive GEX (volatility suppression)
        if net_gex > 300e6:
            confidence += 30
            reasons.append(f"High positive GEX {net_gex/1e6:.0f}M suppresses volatility")
        
        # 2. Strong call wall resistance
        if len(call_walls) > 0:
            strongest_wall = call_walls.iloc[0]
            wall_distance = ((strongest_wall['strike'] - current_price) / current_price) * 100
            wall_strength = strongest_wall['call_gex'] / 1e6
            
            if wall_distance < 3 and wall_strength > 100:  # Close to strong wall
                confidence += 35
                reasons.append(f"Strong call wall at ${strongest_wall['strike']:.2f} ({wall_strength:.0f}M GEX)")
            elif wall_distance < 5:
                confidence += 20
                reasons.append(f"Call wall resistance at ${strongest_wall['strike']:.2f}")
        
        # 3. Price between flip and call wall (optimal zone)
        if len(call_walls) > 0:
            call_wall = call_walls.iloc[0]['strike']
            if gamma_flip < current_price < call_wall:
                confidence += 25
                reasons.append("Price in optimal zone between flip and call wall")
        
        if confidence >= 50:
            # Determine optimal strikes
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
        """Analyze put selling opportunities"""
        confidence = 0
        strategy_details = {
            'strategy': 'SELL PUTS',
            'strategy_type': 'premium_collection',
            'emoji': '🛡️'
        }
        
        reasons = []
        
        # 1. Positive GEX environment
        if net_gex > 200e6:
            confidence += 25
            reasons.append(f"Positive GEX {net_gex/1e6:.0f}M provides downside support")
        
        # 2. Strong put wall support
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
        
        # 3. Price above gamma flip
        if current_price > gamma_flip:
            flip_distance = ((current_price - gamma_flip) / current_price) * 100
            confidence += min(flip_distance * 10, 25)
            reasons.append(f"Price {flip_distance:.1f}% above gamma flip")
        
        if confidence >= 50:
            # Determine optimal strikes
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
        """Analyze iron condor opportunities"""
        confidence = 0
        strategy_details = {
            'strategy': 'IRON CONDOR',
            'strategy_type': 'range_trading',
            'emoji': '🦅'
        }
        
        reasons = []
        
        # 1. High positive GEX (range-bound environment)
        if net_gex > 500e6:
            confidence += 30
            reasons.append(f"Very high positive GEX {net_gex/1e6:.0f}M creates strong range")
        elif net_gex > 200e6:
            confidence += 20
            reasons.append(f"Positive GEX {net_gex/1e6:.0f}M supports range trading")
        
        # 2. Clear call and put walls
        if len(call_walls) > 0 and len(put_walls) > 0:
            call_wall = call_walls.iloc[0]['strike']
            put_wall = put_walls.iloc[0]['strike']
            range_width = ((call_wall - put_wall) / current_price) * 100
            
            if range_width > 4:  # At least 4% range
                confidence += 35
                reasons.append(f"Clear range: ${put_wall:.2f} - ${call_wall:.2f} ({range_width:.1f}%)")
                
                # Check if price is centered
                range_center = (call_wall + put_wall) / 2
                center_distance = abs(current_price - range_center) / current_price * 100
                if center_distance < 1:  # Within 1% of center
                    confidence += 15
                    reasons.append("Price well-centered in range")
            elif range_width > 2:
                confidence += 20
                reasons.append(f"Moderate range: {range_width:.1f}% wide")
        
        # 3. Low recent volatility
        if len(strike_data) > 0:
            total_gamma = strike_data['net_gex'].abs().sum()
            gamma_concentration = max(strike_data['net_gex'].abs()) / total_gamma if total_gamma > 0 else 0
            
            if gamma_concentration < 0.3:  # Well distributed gamma
                confidence += 15
                reasons.append("Well-distributed gamma supports range")
        
        if confidence >= 60:
            # Determine condor strikes
            if len(call_walls) > 0 and len(put_walls) > 0:
                # Short strikes at the walls
                short_call = call_walls.iloc[0]['strike']
                short_put = put_walls.iloc[0]['strike']
                
                # Long strikes outside the walls
                strike_spacing = 5 if current_price < 200 else 10  # Adjust spacing based on price
                long_call = short_call + strike_spacing
                long_put = short_put - strike_spacing
                
                # Expected range
                expected_range = f"${short_put:.2f} - ${short_call:.2f}"
                range_width = ((short_call - short_put) / current_price) * 100
                
            else:
                # Default strikes based on current price
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

# Initialize scanner
@st.cache_resource
def get_scanner():
    return DynamicOptionsScanner()

scanner = get_scanner()

# Main interface with tabs
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Mass Scanner", "🎯 Custom Symbol Analysis", "📊 Detailed GEX Profile", "⚡ Options Strategies"])

with tab1:
    # Original mass scanner functionality
    st.markdown("## 🔍 Real-Time Options Volume & GEX Scanner")
    
    # Sidebar controls for mass scanner
    st.sidebar.header("🎛️ Mass Scanner Configuration")
    
    # Scanning parameters
    max_symbols = st.sidebar.slider("Max Symbols to Analyze", 25, 150, 50, 5)
    min_confidence = st.sidebar.slider("Minimum Confidence %", 0, 100, 25, 5)
    volume_threshold = st.sidebar.number_input("Min Options Volume", 1000, 100000, 5000, 1000)
    
    # Signal filters
    signal_filters = st.sidebar.multiselect(
        "Signal Types to Show",
        ["SQUEEZE_POTENTIAL", "RANGE_BOUND", "BEARISH_FLOW", "BULLISH_FLOW"],
        default=["SQUEEZE_POTENTIAL", "RANGE_BOUND"]
    )
    
    # Refresh controls
    auto_scan = st.sidebar.checkbox("🔄 Auto-scan (30min)", value=False)
    
    if st.sidebar.button("🚀 Start New Scan", type="primary"):
        st.cache_data.clear()
        st.rerun()
    
    # Market overview
    st.sidebar.markdown("**📊 Scan Parameters:**")
    st.sidebar.markdown(f"• Symbols: Top {max_symbols} by volume")
    st.sidebar.markdown(f"• Min Confidence: {min_confidence}%")
    st.sidebar.markdown(f"• Min Volume: {volume_threshold:,}")
    
    # Step 1: Get options volume ranking
    with st.expander("📊 Step 1: Dynamic Options Volume Ranking", expanded=True):
        st.markdown("🔄 Scanning dynamic symbol list for current options activity...")
        
        volume_data = scanner.get_options_volume_ranking()
        
        if len(volume_data) > 0:
            st.success(f"✅ Found {len(volume_data)} symbols with options data")
            
            # Show top 20 by volume
            top_20 = volume_data.head(20)
            
            # Volume ranking chart
            fig_volume = go.Figure()
            
            fig_volume.add_trace(go.Bar(
                x=top_20['symbol'],
                y=top_20['total_volume'],
                name='Total Volume',
                marker_color='rgba(54, 162, 235, 0.8)',
                hovertemplate='<b>%{x}</b><br>Volume: %{y:,}<br>Price: $%{customdata:.2f}<extra></extra>',
                customdata=top_20['current_price']
            ))
            
            fig_volume.update_layout(
                title="Top 20 Symbols by Options Volume (Live Update)",
                xaxis_title="Symbol",
                yaxis_title="Total Options Volume",
                height=400
            )
            
            st.plotly_chart(fig_volume, use_container_width=True)
            
            # Volume summary table
            st.markdown("**📋 Top 20 by Options Volume:**")
            display_cols = ['symbol', 'current_price', 'total_volume', 'total_oi', 'put_call_ratio']
            formatted_df = top_20[display_cols].copy()
            formatted_df['current_price'] = formatted_df['current_price'].round(2)
            formatted_df['total_volume'] = formatted_df['total_volume'].apply(lambda x: f"{x:,}")
            formatted_df['total_oi'] = formatted_df['total_oi'].apply(lambda x: f"{x:,}")
            
            st.dataframe(formatted_df, use_container_width=True)
            
        else:
            st.error("❌ No options data found. Please try again.")
            st.stop()
    
    # Step 2: GEX Analysis
    with st.expander("⚡ Step 2: GEX Signal Analysis", expanded=True):
        st.markdown(f"Analyzing top {max_symbols} symbols for gamma exposure signals...")
        
        # Filter by volume threshold
        filtered_volume = volume_data[volume_data['total_volume'] >= volume_threshold]
        
        if len(filtered_volume) == 0:
            st.warning(f"⚠️ No symbols meet volume threshold of {volume_threshold:,}")
            st.stop()
        
        # Run GEX scan
        scan_results = scanner.scan_for_signals(filtered_volume, max_symbols)
        
        if len(scan_results) > 0:
            # Filter by confidence and signals
            filtered_results = scan_results[scan_results['confidence'] >= min_confidence]
            
            if len(signal_filters) > 0:
                filtered_results = filtered_results[
                    filtered_results['signals'].apply(
                        lambda x: any(signal in x for signal in signal_filters)
                    )
                ]
            
            st.success(f"✅ Found {len(filtered_results)} signals meeting criteria")
            
            # Results overview
            if len(filtered_results) > 0:
                # Summary metrics
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
                
                # Signal distribution chart
                fig_signals = px.scatter(
                    filtered_results,
                    x='total_volume',
                    y='net_gex_millions',
                    size='confidence',
                    color='confidence',
                    hover_name='symbol',
                    title="GEX vs Options Volume",
                    color_continuous_scale='RdYlGn'
                )
                
                fig_signals.update_layout(height=500)
                st.plotly_chart(fig_signals, use_container_width=True)
                
                # Detailed results
                st.markdown("### 🎯 Detailed Signal Analysis")
                
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
                    
                    signals_text = ", ".join(result['signals']) if result['signals'] else "No specific signals"
                    
                    st.markdown(f"""
                    <div class="{card_class}">
                        <h4>{conf_emoji} {result['symbol']} - {confidence:.0f}% Confidence</h4>
                        <div class="metric-grid">
                            <div><strong>Price:</strong> ${result['current_price']:.2f}</div>
                            <div><strong>Net GEX:</strong> {result['net_gex_millions']:.1f}M</div>
                            <div><strong>Options Volume:</strong> {result['total_volume']:,}</div>
                            <div><strong>P/C Ratio:</strong> {result['put_call_ratio']}</div>
                        </div>
                        <p><strong>🎯 Signals:</strong> {signals_text}</p>
                        <p><strong>📊 Analysis:</strong> Call GEX: {result['call_gex_millions']:.1f}M | Put GEX: {result['put_gex_millions']:.1f}M | OI: {result['total_oi']:,}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            else:
                st.warning("⚠️ No signals meet current filter criteria")
                st.info("Try lowering the confidence threshold or adjusting signal filters")
        
        else:
            st.error("❌ GEX analysis failed. Please try again with different parameters.")

with tab2:
    # Custom symbol analysis
    st.markdown("## 🎯 Custom Symbol Analysis")
    st.markdown("Enter any symbol to get comprehensive GEX and options analysis")
    
    # Symbol input
    col1, col2 = st.columns([3, 1])
    
    with col1:
        custom_symbol = st.text_input(
            "🔍 Enter Symbol (e.g., AAPL, TSLA, NVDA)",
            placeholder="Type any stock symbol...",
            key="custom_symbol_input"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        analyze_button = st.button("📊 Analyze Symbol", type="primary")
    
    # Popular symbols quick access
    st.markdown("**Quick Access:**")
    quick_symbols = ["AAPL", "TSLA", "NVDA", "MSFT", "GOOGL", "AMZN", "META", "SPY", "QQQ", "IWM"]
    
    quick_cols = st.columns(5)
    for i, sym in enumerate(quick_symbols):
        with quick_cols[i % 5]:
            if st.button(sym, key=f"quick_{sym}"):
                custom_symbol = sym
                analyze_button = True
    
    # Analysis section
    if (analyze_button or custom_symbol) and custom_symbol:
        with st.spinner(f"🔄 Analyzing {custom_symbol.upper()}..."):
            analysis = scanner.analyze_custom_symbol(custom_symbol)
        
        if analysis and 'error' not in analysis:
            # Success - show comprehensive analysis
            st.success(f"✅ Analysis complete for {analysis['symbol']}")
            
            # Company header
            st.markdown(f"""
            ### 📈 {analysis['symbol']} - {analysis.get('company_name', 'Unknown')}
            **Sector:** {analysis.get('sector', 'Unknown')} | **Market Cap:** ${analysis.get('market_cap', 0)/1e9:.1f}B
            """)
            
            # Key metrics
            metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)
            
            with metric_col1:
                price_emoji = "🟢" if analysis['price_change'] > 0 else "🔴" if analysis['price_change'] < 0 else "⚪"
                st.metric(
                    f"{price_emoji} Current Price",
                    f"${analysis['current_price']:.2f}",
                    f"{analysis['price_change']:+.2f}%"
                )
            
            with metric_col2:
                net_gex_m = analysis['net_gex'] / 1e6
                gex_emoji = "🟢" if net_gex_m > 0 else "🔴" if net_gex_m < -50 else "🟡"
                st.metric(f"{gex_emoji} Net GEX", f"{net_gex_m:.0f}M")
            
            with metric_col3:
                st.metric("📊 Options Volume", f"{analysis['total_volume']:,}")
            
            with metric_col4:
                st.metric("🏗️ Open Interest", f"{analysis['total_oi']:,}")
            
            with metric_col5:
                confidence = analysis['confidence']
                conf_emoji = "🟢" if confidence >= 75 else "🟡" if confidence >= 50 else "🔴"
                st.metric(f"{conf_emoji} Confidence", f"{confidence:.0f}%")
            
            # Signals analysis
            if analysis['signals']:
                st.markdown("### 🎯 Trading Signals")
                
                signal_descriptions = {
                    'NEGATIVE_GEX_SQUEEZE': '🔴 Negative GEX Squeeze - Long calls potential',
                    'POSITIVE_GEX_RANGE': '🟢 Positive GEX Range - Iron condor setup',
                    'HIGH_VOLUME': '📊 High Options Volume - Increased liquidity',
                    'HIGH_OPEN_INTEREST': '🏗️ High Open Interest - Strong positioning',
                    'HIGH_MOMENTUM': '⚡ High Price Momentum - Trend continuation',
                    'BEARISH_POSITIONING': '🐻 Bearish Positioning - Put heavy',
                    'BULLISH_POSITIONING': '🐂 Bullish Positioning - Call heavy'
                }
                
                for signal in analysis['signals']:
                    st.markdown(f"• {signal_descriptions.get(signal, signal)}")
            
            # GEX breakdown
            st.markdown("### ⚡ Gamma Exposure Breakdown")
            
            gex_col1, gex_col2 = st.columns(2)
            
            with gex_col1:
                # GEX metrics
                call_gex_m = analysis['total_call_gex'] / 1e6
                put_gex_m = analysis['total_put_gex'] / 1e6
                
                st.markdown(f"""
                **Call GEX:** {call_gex_m:.1f}M  
                **Put GEX:** {put_gex_m:.1f}M  
                **Net GEX:** {net_gex_m:.1f}M  
                **Put/Call Ratio:** {analysis['put_call_ratio']:.2f}
                """)
            
            with gex_col2:
                # Simple GEX visualization
                fig_pie = go.Figure(data=[go.Pie(
                    labels=['Call GEX', 'Put GEX'],
                    values=[abs(call_gex_m), abs(put_gex_m)],
                    hole=.3,
                    marker_colors=['green', 'red']
                )])
                
                fig_pie.update_layout(
                    title="GEX Distribution",
                    height=300,
                    showlegend=True
                )
                
                st.plotly_chart(fig_pie, use_container_width=True)
        
        elif analysis and 'error' in analysis:
            st.error(f"❌ {analysis['error']}")
            if 'current_price' in analysis:
                st.info(f"Current price: ${analysis['current_price']:.2f}")
        
        else:
            st.error(f"❌ Unable to analyze {custom_symbol.upper()}")

with tab3:
    # Detailed GEX profile
    st.markdown("## 📊 Detailed Strike-by-Strike GEX Profile")
    st.markdown("Get comprehensive gamma wall analysis with strike-level detail")
    
    # Symbol input for detailed analysis
    detail_col1, detail_col2 = st.columns([3, 1])
    
    with detail_col1:
        detail_symbol = st.text_input(
            "🔍 Enter Symbol for Detailed Analysis",
            placeholder="e.g., SPY, QQQ, AAPL...",
            key="detail_symbol_input"
        )
    
    with detail_col2:
        st.markdown("<br>", unsafe_allow_html=True)
        detail_button = st.button("📊 Get Detailed Profile", type="primary")
    
    if (detail_button or detail_symbol) and detail_symbol:
        with st.spinner(f"🔄 Building detailed GEX profile for {detail_symbol.upper()}..."):
            gex_profile = scanner.get_detailed_gex_profile(detail_symbol)
        
        if gex_profile:
            st.success(f"✅ Detailed GEX profile ready for {detail_symbol.upper()}")
            
            current_price = gex_profile['current_price']
            gamma_flip = gex_profile['gamma_flip']
            strike_data = gex_profile['strike_data']
            
            # Key levels summary
            level_col1, level_col2, level_col3 = st.columns(3)
            
            with level_col1:
                st.metric("💰 Current Price", f"${current_price:.2f}")
            
            with level_col2:
                st.metric("⚡ Gamma Flip", f"${gamma_flip:.2f}")
            
            with level_col3:
                distance_to_flip = ((current_price - gamma_flip) / current_price) * 100
                st.metric("📏 Distance to Flip", f"{distance_to_flip:.2f}%")
            
            # Detailed GEX chart
            fig_detail = make_subplots(
                rows=2, cols=1,
                row_heights=[0.7, 0.3],
                subplot_titles=(f"{detail_symbol.upper()} Strike-by-Strike GEX", "Cumulative GEX"),
                vertical_spacing=0.1
            )
            
            # Main GEX bars
            call_strikes = strike_data[strike_data['call_gex'] > 0]
            put_strikes = strike_data[strike_data['put_gex'] < 0]
            
            if len(call_strikes) > 0:
                fig_detail.add_trace(
                    go.Bar(
                        x=call_strikes['strike'],
                        y=call_strikes['call_gex'] / 1e6,
                        name='Call GEX',
                        marker_color='rgba(34, 139, 34, 0.8)',
                        hovertemplate='<b>Call Wall</b><br>Strike: $%{x}<br>GEX: %{y:.1f}M<br>OI: %{customdata}<extra></extra>',
                        customdata=call_strikes['call_oi']
                    ), row=1, col=1
                )
            
            if len(put_strikes) > 0:
                fig_detail.add_trace(
                    go.Bar(
                        x=put_strikes['strike'],
                        y=put_strikes['put_gex'] / 1e6,
                        name='Put GEX',
                        marker_color='rgba(220, 20, 60, 0.8)',
                        hovertemplate='<b>Put Wall</b><br>Strike: $%{x}<br>GEX: %{y:.1f}M<br>OI: %{customdata}<extra></extra>',
                        customdata=put_strikes['put_oi']
                    ), row=1, col=1
                )
            
            # Add key levels
            fig_detail.add_vline(x=current_price, line_dash="solid", line_color="blue",
                               annotation_text=f"Current: ${current_price:.2f}", row=1, col=1)
            fig_detail.add_vline(x=gamma_flip, line_dash="dash", line_color="orange",
                               annotation_text=f"Flip: ${gamma_flip:.2f}", row=1, col=1)
            
            # Cumulative GEX
            fig_detail.add_trace(
                go.Scatter(
                    x=strike_data['strike'],
                    y=strike_data['cumulative_gex'] / 1e6,
                    mode='lines',
                    name='Cumulative GEX',
                    line=dict(color='purple', width=3),
                    hovertemplate='Strike: $%{x}<br>Cumulative: %{y:.1f}M<extra></extra>'
                ), row=2, col=1
            )
            
            fig_detail.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
            
            fig_detail.update_layout(
                height=700,
                showlegend=True,
                title=f"{detail_symbol.upper()} Detailed GEX Analysis",
                hovermode='x unified'
            )
            
            fig_detail.update_xaxes(title_text="Strike Price ($)", row=2, col=1)
            fig_detail.update_yaxes(title_text="GEX (Millions)", row=1, col=1)
            fig_detail.update_yaxes(title_text="Cumulative GEX (Millions)", row=2, col=1)
            
            st.plotly_chart(fig_detail, use_container_width=True)
            
            # Walls analysis
            wall_col1, wall_col2 = st.columns(2)
            
            with wall_col1:
                st.markdown("**🟢 Call Walls (Resistance)**")
                for i, (_, wall) in enumerate(gex_profile['call_walls'].head(5).iterrows(), 1):
                    distance = ((wall['strike'] - current_price) / current_price) * 100
                    st.markdown(f"**{i}.** ${wall['strike']:.2f} (+{distance:.1f}%) - {wall['call_gex']/1e6:.1f}M GEX")
            
            with wall_col2:
                st.markdown("**🔴 Put Walls (Support)**")
                for i, (_, wall) in enumerate(gex_profile['put_walls'].head(5).iterrows(), 1):
                    distance = ((wall['strike'] - current_price) / current_price) * 100
                    st.markdown(f"**{i}.** ${wall['strike']:.2f} ({distance:.1f}%) - {abs(wall['put_gex'])/1e6:.1f}M GEX")
        
        else:
            st.error(f"❌ Unable to build detailed profile for {detail_symbol.upper()}")

with tab4:
    # Options Strategies Analysis
    st.markdown("## ⚡ Advanced Options Strategies Analysis")
    st.markdown("Get specific strategy recommendations with price targets and confidence levels")
    
    # Symbol input for strategy analysis
    strat_col1, strat_col2 = st.columns([3, 1])
    
    with strat_col1:
        strategy_symbol = st.text_input(
            "🔍 Enter Symbol for Strategy Analysis",
            placeholder="e.g., SPY, AAPL, TSLA...",
            key="strategy_symbol_input"
        )
    
    with strat_col2:
        st.markdown("<br>", unsafe_allow_html=True)
        strategy_button = st.button("⚡ Analyze Strategies", type="primary")
    
    # Quick strategy symbols
    st.markdown("**Popular Strategy Symbols:**")
    strategy_quick = ["SPY", "QQQ", "AAPL", "TSLA", "NVDA", "AMZN", "MSFT", "GOOGL"]
    
    strat_quick_cols = st.columns(4)
    for i, sym in enumerate(strategy_quick):
        with strat_quick_cols[i % 4]:
            if st.button(sym, key=f"strat_quick_{sym}"):
                strategy_symbol = sym
                strategy_button = True
    
    if (strategy_button or strategy_symbol) and strategy_symbol:
        with st.spinner(f"🔄 Analyzing options strategies for {strategy_symbol.upper()}..."):
            strategy_analysis = scanner.analyze_options_strategies(strategy_symbol)
        
        if strategy_analysis and strategy_analysis.get('strategies'):
            st.success(f"✅ Strategy analysis complete for {strategy_analysis['symbol']}")
            
            # Market context header
            st.markdown(f"""
            ### 📊 Market Context for {strategy_analysis['symbol']}
            **Current Price:** ${strategy_analysis['current_price']:.2f} | 
            **Gamma Flip:** ${strategy_analysis['gamma_flip']:.2f} | 
            **Net GEX:** {strategy_analysis['net_gex']/1e6:.0f}M | 
            **Distance to Flip:** {strategy_analysis['distance_to_flip']:.2f}%
            """)
            
            # Strategy cards with improved styling
            strategies = strategy_analysis['strategies']
            
            for i, strategy in enumerate(strategies):
                confidence = strategy['confidence']
                
                # Determine card styling based on confidence
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
                    <h2>{strategy['emoji']} {strategy['strategy']} 
                    <span class="confidence-badge {conf_class}">{confidence:.0f}% {conf_text} CONFIDENCE</span>
                    </h2>
                </div>
                """, unsafe_allow_html=True)
                
                # Strategy details in organized layout
                detail_col1, detail_col2, detail_col3 = st.columns(3)
                
                with detail_col1:
                    st.markdown("""
                    <div class="strategy-details">
                        <h4>📋 Strategy Details</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"**• Type:** {strategy.get('strategy_type', 'N/A')}")
                    st.markdown(f"**• Optimal DTE:** {strategy.get('optimal_dte', 'N/A')}")
                    st.markdown(f"**• Position Size:** {strategy.get('position_size', 'N/A')}")
                    st.markdown(f"**• Time Horizon:** {strategy.get('time_horizon', 'N/A')}")
                    st.markdown(f"**• Volatility Bet:** {strategy.get('volatility_bet', 'N/A')}")
                
                with detail_col2:
                    st.markdown("""
                    <div class="strategy-details">
                        <h4>🎯 Targets & Risk</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if isinstance(strategy.get('recommended_strikes'), list):
                        strikes_text = ", ".join([f"${s:.2f}" for s in strategy['recommended_strikes']])
                    elif isinstance(strategy.get('recommended_strikes'), dict):
                        strikes_dict = strategy['recommended_strikes']
                        strikes_text = f"Put: ${strikes_dict.get('long_put', 0):.2f}/{strikes_dict.get('short_put', 0):.2f}, Call: ${strikes_dict.get('short_call', 0):.2f}/{strikes_dict.get('long_call', 0):.2f}"
                    else:
                        strikes_text = "See analysis"
                    
                    st.markdown(f"**• Strikes:** {strikes_text}")
                    st.markdown(f"**• Primary Target:** {strategy.get('primary_target', 'N/A')}")
                    st.markdown(f"**• Expected Return:** {strategy.get('expected_return', 'N/A')}")
                    st.markdown(f"**• Max Risk:** {strategy.get('max_risk', 'N/A')}")
                    st.markdown(f"**• Stop Loss:** {strategy.get('stop_loss', 'N/A')}")
                
                with detail_col3:
                    st.markdown("""
                    <div class="strategy-details">
                        <h4>💡 Analysis Rationale</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    for reason in strategy.get('reasons', []):
                        st.markdown(f"• {reason}")
                
                # Risk management box
                st.markdown(f"""
                <div class="risk-box">
                    <h4>⚠️ Risk Management Guidelines</h4>
                    <p>• Monitor daily for changes in GEX profile and market conditions</p>
                    <p>• Set alerts at key gamma levels and price targets</p>
                    <p>• Consider volatility changes and time decay effects</p>
                    <p>• Follow position sizing guidelines strictly</p>
                    <p>• Have exit plan ready before entering position</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Advanced details in expander
                with st.expander(f"📈 Advanced Details - {strategy['strategy']}"):
                    
                    adv_col1, adv_col2 = st.columns(2)
                    
                    with adv_col1:
                        st.markdown("**🔢 Key Metrics:**")
                        st.markdown(f"• **Expected Move:** {strategy.get('expected_move', 'N/A')}")
                        st.markdown(f"• **Secondary Target:** {strategy.get('secondary_target', 'N/A')}")
                        
                        if strategy['strategy'] == 'IRON CONDOR':
                            strikes = strategy.get('recommended_strikes', {})
                            st.markdown("**Iron Condor Structure:**")
                            st.markdown(f"• Long Put: ${strikes.get('long_put', 0):.2f}")
                            st.markdown(f"• Short Put: ${strikes.get('short_put', 0):.2f}")
                            st.markdown(f"• Short Call: ${strikes.get('short_call', 0):.2f}")
                            st.markdown(f"• Long Call: ${strikes.get('long_call', 0):.2f}")
                    
                    with adv_col2:
                        st.markdown("**⚠️ Specific Risk Factors:**")
                        
                        if strategy.get('strategy_type') == 'directional_bullish':
                            st.markdown("• Watch for gamma flip breach to downside")
                            st.markdown("• Monitor put wall integrity for support")
                            st.markdown("• Consider early exit if momentum stalls")
                        elif strategy.get('strategy_type') == 'directional_bearish':
                            st.markdown("• Monitor call wall breakthrough")
                            st.markdown("• Watch for put wall defense breakdown")
                            st.markdown("• Consider profit taking at support levels")
                        elif strategy.get('strategy_type') == 'premium_collection':
                            st.markdown("• Close early if price approaches strike")
                            st.markdown("• Monitor implied volatility changes")
                            st.markdown("• Have assignment plan if applicable")
                        elif strategy.get('strategy_type') == 'range_trading':
                            st.markdown("• Exit immediately if range breaks")
                            st.markdown("• Monitor both sides equally")
                            st.markdown("• Consider adjustments if threatened")
                
                st.markdown("---")
            
            # Strategy summary
            st.markdown("### 📊 Strategy Summary")
            
            summary_col1, summary_col2, summary_col3 = st.columns(3)
            
            with summary_col1:
                high_conf_strategies = len([s for s in strategies if s['confidence'] >= 75])
                st.metric("High Confidence Strategies", high_conf_strategies)
            
            with summary_col2:
                avg_confidence = sum(s['confidence'] for s in strategies) / len(strategies)
                st.metric("Average Confidence", f"{avg_confidence:.0f}%")
            
            with summary_col3:
                strategy_types = list(set(s['strategy_type'] for s in strategies))
                st.metric("Strategy Types", len(strategy_types))
            
            # Best strategy recommendation
            if strategies:
                best_strategy = strategies[0]  # Already sorted by confidence
                
                st.markdown("### 🏆 Top Recommendation")
                st.markdown(f"""
                <div class="strategy-card strategy-high">
                    <h3>{best_strategy['emoji']} {best_strategy['strategy']} - {best_strategy['confidence']:.0f}% Confidence</h3>
                    <p><strong>Strategy Type:</strong> {best_strategy.get('strategy_type', 'N/A')}</p>
                    <p><strong>Optimal DTE:</strong> {best_strategy.get('optimal_dte', 'N/A')}</p>
                    <p><strong>Expected Return:</strong> {best_strategy.get('expected_return', 'N/A')}</p>
                    <p><strong>Position Size:</strong> {best_strategy.get('position_size', 'N/A')}</p>
                    <p><strong>Top Reason:</strong> {best_strategy.get('reasons', ['See full analysis'])[0]}</p>
                </div>
                """, unsafe_allow_html=True)
        
        elif strategy_analysis:
            st.warning("⚠️ No high-confidence strategies identified for current market conditions")
            st.info(f"""
            **Market Context for {strategy_analysis['symbol']}:**
            - Current Price: ${strategy_analysis['current_price']:.2f}
            - Gamma Flip: ${strategy_analysis['gamma_flip']:.2f}
            - Net GEX: {strategy_analysis['net_gex']/1e6:.0f}M
            
            **Suggestions:**
            - Wait for clearer GEX setup
            - Check back after market movement
            - Consider different time horizons
            """)
        
        else:
            st.error(f"❌ Unable to analyze strategies for {strategy_symbol.upper()}")
            st.info("This could be due to limited options data or unusual market conditions")

# Footer
st.markdown("---")
st.markdown("""
**⚠️ DISCLAIMER:** This scanner uses real market data for educational purposes only. 
Options trading involves substantial risk and requires proper education and risk management. 
Always conduct your own research and consult with qualified professionals before making trading decisions.
""")
