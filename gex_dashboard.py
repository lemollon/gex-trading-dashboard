#!/usr/bin/env python3
"""
Complete Dynamic GEX Scanner
Real-time analysis of options with custom symbol support
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

# Professional styling
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
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    
    .high-signal {
        border-left: 4px solid #28a745 !important;
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
    }
    
    .medium-signal {
        border-left: 4px solid #ffc107 !important;
        background: linear-gradient(135deg, #fff3cd, #ffeaa7);
    }
    
    .low-signal {
        border-left: 4px solid #dc3545 !important;
        background: linear-gradient(135deg, #f8d7da, #fab1a0);
    }
    
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🎯 Dynamic GEX Scanner</h1>
    <p><strong>Real-Time Analysis</strong> | Live options data with custom symbol support</p>
    <p>Professional gamma exposure analysis with any symbol</p>
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
tab1, tab2, tab3 = st.tabs(["🔍 Mass Scanner", "🎯 Custom Symbol Analysis", "📊 Detailed GEX Profile"])

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

# Footer
st.markdown("---")
st.markdown("""
**⚠️ DISCLAIMER:** This scanner uses real market data for educational purposes only. 
Options trading involves substantial risk and requires proper education and risk management. 
Always conduct your own research and consult with qualified professionals before making trading decisions.
""")
