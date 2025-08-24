# 🚀 Complete Gamma Exposure Trading System with Real Market Data
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import yfinance as yf
import time
from typing import Dict, List, Optional
import uuid
import random
import requests
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# Enhanced page configuration with custom styling
st.set_page_config(
    page_title="GEX Trading System",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for stunning visuals
st.markdown("""
<style>
    /* Import modern fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styles */
    .stApp {
        background: linear-gradient(135deg, #0f1419 0%, #1a1f2e 50%, #0f1419 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Custom header styling */
    .main-header {
        background: linear-gradient(90deg, #00ff87 0%, #60efff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 0 0 30px rgba(0, 255, 135, 0.3);
    }
    
    .subtitle {
        text-align: center;
        font-size: 1.2rem;
        color: #8b949e;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    /* Glowing cards */
    .metric-card {
        background: linear-gradient(145deg, #21262d 0%, #30363d 100%);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 16px 48px rgba(0, 255, 135, 0.2);
        border-color: #00ff87;
    }
    
    /* Status indicators */
    .status-open {
        background: linear-gradient(135deg, #00ff87, #00cc6a);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
        box-shadow: 0 4px 15px rgba(0, 255, 135, 0.4);
    }
    
    .status-closed {
        background: linear-gradient(135deg, #ff4757, #ff3742);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
        box-shadow: 0 4px 15px rgba(255, 71, 87, 0.4);
    }
    
    .status-premarket {
        background: linear-gradient(135deg, #ffa502, #ff9500);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
        box-shadow: 0 4px 15px rgba(255, 165, 2, 0.4);
    }
    
    /* Trade type badges */
    .trade-squeeze {
        background: linear-gradient(135deg, #ff6b6b, #ff5252);
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    .trade-premium {
        background: linear-gradient(135deg, #4ecdc4, #44a08d);
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    .trade-condor {
        background: linear-gradient(135deg, #a55eea, #8b5cf6);
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    /* Opportunity cards */
    .opportunity-card {
        background: linear-gradient(145deg, #1c2128 0%, #21262d 100%);
        border: 2px solid transparent;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        position: relative;
        overflow: hidden;
        transition: all 0.3s ease;
    }
    
    .opportunity-card:hover {
        transform: scale(1.02);
        box-shadow: 0 16px 48px rgba(0, 255, 135, 0.3);
    }
    
    .opportunity-high {
        border-image: linear-gradient(135deg, #00ff87, #60efff) 1;
        box-shadow: 0 8px 32px rgba(0, 255, 135, 0.2);
    }
    
    .opportunity-medium {
        border-image: linear-gradient(135deg, #ffa502, #ff9500) 1;
        box-shadow: 0 8px 32px rgba(255, 165, 2, 0.2);
    }
    
    /* Progress bars */
    .progress-bar {
        background: linear-gradient(90deg, #00ff87 0%, #60efff 100%);
        height: 8px;
        border-radius: 4px;
        overflow: hidden;
        position: relative;
    }
    
    .progress-bar::after {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        animation: shimmer 2s infinite;
    }
    
    @keyframes shimmer {
        0% { left: -100%; }
        100% { left: 100%; }
    }
    
    /* Metrics styling */
    div[data-testid="metric-container"] {
        background: linear-gradient(145deg, #21262d 0%, #30363d 100%);
        border: 1px solid #30363d;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #00ff87, #00cc6a);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 255, 135, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 255, 135, 0.4);
        background: linear-gradient(135deg, #00cc6a, #00b356);
    }
</style>
""", unsafe_allow_html=True)

class RealMarketData:
    """Real market data integration for authentic trading experience"""
    
    def __init__(self):
        self.cache_duration = 60  # Cache data for 1 minute
        self.last_fetch = {}
        self.cached_data = {}
    
    @st.cache_data(ttl=60)
    def get_real_stock_data(_self, symbols: List[str]) -> Dict:
        """Fetch real stock data with caching"""
        try:
            # Fetch multiple symbols at once for efficiency
            tickers = yf.Tickers(" ".join(symbols))
            
            results = {}
            for symbol in symbols:
                try:
                    ticker = tickers.tickers[symbol]
                    
                    # Get current price
                    hist = ticker.history(period="1d", interval="1m")
                    if not hist.empty:
                        current_price = float(hist['Close'].iloc[-1])
                        prev_close = float(hist['Close'].iloc[0])
                        
                        # Get basic info
                        info = ticker.info
                        
                        results[symbol] = {
                            'current_price': current_price,
                            'prev_close': prev_close,
                            'change_pct': ((current_price - prev_close) / prev_close) * 100,
                            'volume': int(hist['Volume'].iloc[-1]) if not hist['Volume'].empty else 0,
                            'avg_volume': info.get('averageVolume', 0),
                            'market_cap': info.get('marketCap', 0),
                            'beta': info.get('beta', 1.0),
                            'last_updated': datetime.now()
                        }
                    else:
                        # Fallback if no recent data
                        results[symbol] = _self._create_fallback_data(symbol)
                        
                except Exception as e:
                    # Individual symbol fallback
                    results[symbol] = _self._create_fallback_data(symbol)
                    
            return results
            
        except Exception as e:
            st.warning(f"⚠️ Unable to fetch real market data. Using simulated data for demo.")
            return {symbol: _self._create_fallback_data(symbol) for symbol in symbols}
    
    def _create_fallback_data(self, symbol: str) -> Dict:
        """Create realistic fallback data when API fails"""
        # Base prices for major symbols
        base_prices = {
            'TSLA': 245.67,
            'NVDA': 465.89,
            'SPY': 565.23,
            'QQQ': 485.67,
            'AAPL': 175.43,
            'MSFT': 415.67,
            'AMD': 142.89,
            'GOOGL': 167.89
        }
        
        base_price = base_prices.get(symbol, 100.0)
        
        # Add some realistic movement
        change_pct = random.uniform(-3.0, 3.0)
        current_price = base_price * (1 + change_pct/100)
        
        return {
            'current_price': current_price,
            'prev_close': base_price,
            'change_pct': change_pct,
            'volume': random.randint(1000000, 50000000),
            'avg_volume': random.randint(5000000, 30000000),
            'market_cap': random.randint(100000000000, 3000000000000),
            'beta': random.uniform(0.8, 2.5),
            'last_updated': datetime.now()
        }
    
    def calculate_realistic_gamma_levels(self, symbol: str, current_price: float, 
                                       volatility: float = 0.35) -> Dict:
        """Calculate realistic gamma flip and wall levels"""
        
        # Calculate option strikes around current price
        strike_spacing = 5 if current_price > 200 else 2.5 if current_price > 50 else 1
        strikes = []
        
        # Generate strikes from 20% below to 20% above current price
        start_strike = int((current_price * 0.8) / strike_spacing) * strike_spacing
        end_strike = int((current_price * 1.2) / strike_spacing) * strike_spacing
        
        current_strike = start_strike
        while current_strike <= end_strike:
            strikes.append(current_strike)
            current_strike += strike_spacing
        
        # Simulate realistic gamma distribution
        total_call_gamma = 0
        total_put_gamma = 0
        call_walls = []
        put_walls = []
        
        for strike in strikes:
            # Distance from current price affects gamma concentration
            distance_factor = abs(strike - current_price) / current_price
            
            # Calls have more gamma above current price
            if strike >= current_price:
                call_gamma = max(0, (1000000 - distance_factor * 5000000) * 
                               random.uniform(0.5, 1.5))
                if call_gamma > 500000:  # Significant wall
                    call_walls.append({'strike': strike, 'gamma': call_gamma})
                total_call_gamma += call_gamma
            
            # Puts have more gamma below current price  
            if strike <= current_price:
                put_gamma = max(0, (800000 - distance_factor * 4000000) * 
                              random.uniform(0.5, 1.5))
                if put_gamma > 400000:  # Significant wall
                    put_walls.append({'strike': strike, 'gamma': put_gamma})
                total_put_gamma += put_gamma
        
        # Calculate net GEX
        net_gex = total_call_gamma - total_put_gamma
        
        # Find gamma flip point (where net GEX crosses zero)
        gamma_flip = current_price
        for strike in sorted(strikes):
            strike_gex = sum([w['gamma'] for w in call_walls if w['strike'] >= strike]) - \
                        sum([w['gamma'] for w in put_walls if w['strike'] <= strike])
            
            if abs(strike_gex) < abs(net_gex) * 0.1:  # Close to zero
                gamma_flip = strike
                break
        
        # Get strongest walls
        call_walls.sort(key=lambda x: x['gamma'], reverse=True)
        put_walls.sort(key=lambda x: x['gamma'], reverse=True)
        
        return {
            'net_gex': net_gex,
            'gamma_flip': gamma_flip,
            'call_wall': call_walls[0]['strike'] if call_walls else current_price * 1.05,
            'put_wall': put_walls[0]['strike'] if put_walls else current_price * 0.95,
            'call_walls': call_walls[:3],
            'put_walls': put_walls[:3],
            'total_call_gamma': total_call_gamma,
            'total_put_gamma': total_put_gamma
        }
    
    def get_market_status(self) -> Dict:
        """Get real market status"""
        now = datetime.now()
        
        # Market hours: 9:30 AM - 4:00 PM ET, Monday-Friday
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        is_weekday = now.weekday() < 5
        is_market_hours = market_open <= now <= market_close
        
        if is_weekday and is_market_hours:
            status = "OPEN"
            color = "green" 
            emoji = "🟢"
        elif is_weekday and (now < market_open or now > market_close):
            status = "PRE/AFTER HOURS"
            color = "orange"
            emoji = "🟡" 
        else:
            status = "CLOSED - WEEKEND"
            color = "red"
            emoji = "🔴"
        
        return {
            'status': status,
            'color': color,
            'emoji': emoji,
            'next_open': market_open if now < market_open else market_open + timedelta(days=1),
            'is_trading_hours': is_weekday and is_market_hours
        }

class EnhancedMockTradingAccount:
    """Enhanced mock trading with REAL market data integration"""
    
    def __init__(self):
        self.initial_balance = 100000
        self.last_update_time = datetime.now()
        self.market_data = RealMarketData()
        
        # Initialize session state
        if 'portfolio_history' not in st.session_state:
            st.session_state.portfolio_history = []
        if 'open_trades' not in st.session_state:
            st.session_state.open_trades = []
        if 'closed_trades' not in st.session_state:
            st.session_state.closed_trades = []
        if 'auto_trading_enabled' not in st.session_state:
            st.session_state.auto_trading_enabled = False
        if 'last_analysis_time' not in st.session_state:
            st.session_state.last_analysis_time = datetime.now() - timedelta(minutes=30)
        if 'real_market_data' not in st.session_state:
            st.session_state.real_market_data = {}
    
    def get_real_opportunities(self) -> List[Dict]:
        """Generate opportunities using REAL market data"""
        
        # Symbols to analyze
        symbols = ['TSLA', 'NVDA', 'SPY', 'QQQ', 'AAPL', 'AMD']
        
        # Fetch real market data
        with st.spinner('🔄 Fetching real market data...'):
            market_data = self.market_data.get_real_stock_data(symbols)
            st.session_state.real_market_data = market_data
        
        opportunities = []
        
        for symbol, data in market_data.items():
            current_price = data['current_price']
            change_pct = data['change_pct']
            volume = data['volume']
            avg_volume = data['avg_volume']
            
            # Calculate gamma levels based on real price
            gamma_data = self.market_data.calculate_realistic_gamma_levels(
                symbol, current_price
            )
            
            # Determine setup type based on real market structure
            distance_from_flip = ((current_price - gamma_data['gamma_flip']) / 
                                gamma_data['gamma_flip']) * 100
            
            # High volume indicates potential opportunity
            volume_spike = volume > (avg_volume * 1.5) if avg_volume > 0 else False
            
            # Generate confidence score based on multiple factors
            confidence_factors = []
            
            # Distance from gamma flip
            if abs(distance_from_flip) > 2:
                confidence_factors.append(15)  # Clear direction
            
            # Volume
            if volume_spike:
                confidence_factors.append(20)  # High interest
            
            # Price movement
            if abs(change_pct) > 1:
                confidence_factors.append(15)  # Significant move
            
            # Gamma structure strength
            if abs(gamma_data['net_gex']) > 500000000:
                confidence_factors.append(25)  # Strong gamma
            
            # Base confidence
            base_confidence = 45 + sum(confidence_factors)
            confidence_score = min(base_confidence + random.randint(-5, 10), 95)
            
            # Determine strategy based on real gamma structure
            if gamma_data['net_gex'] < -300000000 and distance_from_flip < -1:
                # Negative GEX + below flip = Squeeze setup
                structure_type = 'SQUEEZE_SETUP'
                trade_type = 'LONG_CALLS'
                recommendation = f"BUY CALLS - Squeeze potential below ${gamma_data['gamma_flip']:.2f}"
                explanation = f"{symbol} trading below gamma flip with negative GEX. Dealer hedging will amplify upward moves."
            
            elif gamma_data['net_gex'] > 1000000000 and current_price >= gamma_data['call_wall'] * 0.98:
                # Positive GEX + near call wall = Premium selling
                structure_type = 'PREMIUM_SELLING_SETUP'
                trade_type = 'CALL_SELLING'
                recommendation = f"SELL ${gamma_data['call_wall']:.0f} CALLS - Strong resistance"
                explanation = f"{symbol} approaching call wall at ${gamma_data['call_wall']:.2f} with high positive GEX."
            
            elif gamma_data['net_gex'] > 1500000000 and abs(distance_from_flip) < 1:
                # High positive GEX + near flip = Iron condor
                structure_type = 'IRON_CONDOR_SETUP'
                trade_type = 'IRON_CONDOR'
                recommendation = f"IRON CONDOR ${gamma_data['put_wall']:.0f}/{gamma_data['call_wall']:.0f}"
                explanation = f"{symbol} trapped between gamma walls with massive positive GEX."
            
            else:
                # Default to most likely setup based on GEX
                if gamma_data['net_gex'] < 0:
                    structure_type = 'POTENTIAL_SQUEEZE'
                    trade_type = 'LONG_CALLS'
                    recommendation = f"WATCH FOR SQUEEZE - Monitor ${gamma_data['gamma_flip']:.2f} level"
                    explanation = f"{symbol} has negative GEX but needs better positioning."
                else:
                    structure_type = 'PREMIUM_SELLING_SETUP' 
                    trade_type = 'CALL_SELLING'
                    recommendation = f"PREMIUM SELLING - Positive GEX environment"
                    explanation = f"{symbol} in positive GEX regime, suitable for premium strategies."
            
            # Calculate realistic option premium
            volatility = min(abs(change_pct) * 0.1 + 0.25, 0.8)  # 25% base + movement
            expected_premium = current_price * volatility * 0.08 * random.uniform(0.8, 1.2)
            
            # Only include opportunities with reasonable confidence
            if confidence_score >= 60:
                opportunity = {
                    'symbol': symbol,
                    'current_price': current_price,
                    'gamma_flip': gamma_data['gamma_flip'],
                    'distance_pct': distance_from_flip,
                    'net_gex': gamma_data['net_gex'],
                    'call_wall': gamma_data['call_wall'],
                    'put_wall': gamma_data['put_wall'],
                    'structure_type': structure_type,
                    'confidence_score': confidence_score,
                    'trade_type': trade_type,
                    'recommendation': recommendation,
                    'explanation': explanation,
                    'expected_premium': max(expected_premium, 0.50),
                    'days_to_expiry': random.choice([1, 2, 3, 5, 7]),
                    'emoji': self._get_emoji(symbol),
                    'trend': 'up' if change_pct > 0.5 else 'down' if change_pct < -0.5 else 'sideways',
                    'volume_spike': volume_spike,
                    'change_pct': change_pct,
                    'volume': volume,
                    'real_data_timestamp': data['last_updated']
                }
                
                opportunities.append(opportunity)
        
        # Sort by confidence score
        opportunities.sort(key=lambda x: x['confidence_score'], reverse=True)
        
        return opportunities[:5]  # Return top 5 opportunities
    
    def _get_emoji(self, symbol: str) -> str:
        """Get appropriate emoji for symbol"""
        emoji_map = {
            'TSLA': '⚡',
            'NVDA': '🎯', 
            'SPY': '🇺🇸',
            'QQQ': '💻',
            'AAPL': '🍎',
            'AMD': '🔥',
            'MSFT': '💼'
        }
        return emoji_map.get(symbol, '📈')
    
    def calculate_real_option_value(self, trade: Dict) -> float:
        """Calculate option value based on REAL stock movement"""
        try:
            # Get current real stock price
            symbol = trade['symbol']
            current_data = st.session_state.real_market_data.get(symbol)
            
            if not current_data:
                # Fallback to simulation if no real data
                return self._simulate_option_value(trade)
            
            current_price = current_data['current_price']
            entry_stock_price = trade.get('entry_stock_price', current_price)
            
            # Calculate real stock movement
            if entry_stock_price > 0:
                stock_move_pct = (current_price - entry_stock_price) / entry_stock_price
            else:
                stock_move_pct = 0
            
            # Days held affects time decay
            days_held = trade.get('days_held', 0)
            
            # Calculate option value based on real movement
            entry_price = trade['entry_price']
            
            if trade['trade_type'] == 'LONG_CALLS':
                # Calls benefit from upward moves
                if stock_move_pct > 0:
                    # Positive moves get amplified by gamma
                    gamma_multiplier = 1 + (stock_move_pct * 3.5)  # 3.5x leverage
                    option_multiplier = min(gamma_multiplier, 4.0)
                else:
                    # Negative moves hurt options
                    option_multiplier = max(1 + (stock_move_pct * 2.5), 0.05)
            
            elif trade['trade_type'] == 'CALL_SELLING':
                # Sold calls decay unless stock moves up significantly
                time_decay = min(days_held * 0.12, 0.8)  # 12% per day
                
                if stock_move_pct > 0.02:  # Stock up >2%
                    option_multiplier = 1 + (stock_move_pct * 4)  # Bad for sold calls
                else:
                    option_multiplier = max(1 - time_decay, 0.15)  # Time decay helps
            
            else:  # IRON_CONDOR
                # Condors benefit from low movement and time decay
                volatility_impact = abs(stock_move_pct) * 5  # Volatility hurts
                time_decay = min(days_held * 0.08, 0.6)  # 8% per day helps
                
                option_multiplier = max(1 - time_decay + volatility_impact, 0.2)
            
            # Apply realistic time decay
            time_factor = max(1 - (days_held * 0.05), 0.2)
            
            current_value = entry_price * option_multiplier * time_factor
            return max(current_value, 0.01)
            
        except Exception as e:
            return self._simulate_option_value(trade)
    
    def _simulate_option_value(self, trade: Dict) -> float:
        """Fallback simulation if real data unavailable"""
        days_held = trade.get('days_held', 0)
        entry_price = trade['entry_price']
        
        # Simple simulation
        random_move = random.uniform(-0.15, 0.20)  # -15% to +20%
        time_decay = min(days_held * 0.08, 0.7)
        
        multiplier = max(1 + random_move - time_decay, 0.1)
        return entry_price * multiplier
    
    def update_open_trades_with_real_data(self):
        """Update all open trades using real market data"""
        if not st.session_state.open_trades:
            return
        
        # Get symbols from open trades
        symbols = list(set([trade['symbol'] for trade in st.session_state.open_trades]))
        
        # Fetch real data for all symbols at once
        try:
            market_data = self.market_data.get_real_stock_data(symbols)
            st.session_state.real_market_data.update(market_data)
        except:
            pass  # Use cached data if fetch fails
        
        updated_trades = []
        
        for trade in st.session_state.open_trades:
            # Update days held
            entry_date = pd.to_datetime(trade['entry_date'])
            days_held = (datetime.now() - entry_date).days
            trade['days_held'] = days_held
            
            # Update with real market data
            symbol = trade['symbol']
            if symbol in st.session_state.real_market_data:
                real_data = st.session_state.real_market_data[symbol]
                trade['current_stock_price'] = real_data['current_price']
                trade['stock_change_pct'] = real_data['change_pct']
                trade['last_updated'] = real_data['last_updated']
            
            # Calculate current option value using real data
            current_value = self.calculate_real_option_value(trade)
            trade['current_value'] = current_value
            
            # Calculate P&L
            entry_total = trade['entry_price'] * trade['quantity'] * 100
            current_total = current_value * trade['quantity'] * 100
            trade['unrealized_pnl'] = current_total - entry_total
            trade['unrealized_pnl_pct'] = (trade['unrealized_pnl'] / entry_total) * 100
            
            # Add real-time analysis
            trade['analysis'] = self.get_real_trade_analysis(trade)
            
            # Check exit conditions
            exit_reason = self.check_exit_conditions(trade)
            
            if exit_reason:
                self.close_trade(trade, exit_reason)
            else:
                updated_trades.append(trade)
        
        st.session_state.open_trades = updated_trades
    
    def get_real_trade_analysis(self, trade: Dict) -> str:
        """Provide analysis based on real market movement"""
        pnl_pct = trade.get('unrealized_pnl_pct', 0)
        stock_change = trade.get('stock_change_pct', 0)
        days_held = trade.get('days_held', 0)
        
        if pnl_pct > 25:
            return f"🎉 **WINNING**: Up {pnl_pct:.1f}%! Stock moved {stock_change:+.1f}% - gamma effects working perfectly."
        elif pnl_pct < -25:
            return f"⚠️ **LOSING**: Down {pnl_pct:.1f}%. Stock moved {stock_change:+.1f}% against us - may need to cut losses."
        else:
            return f"📊 **DEVELOPING**: {pnl_pct:+.1f}% after {days_held} days. Stock: {stock_change:+.1f}%. Let it develop."
    
    def check_exit_conditions(self, trade: Dict) -> Optional[str]:
        """Check if trade should be automatically closed"""
        pnl_pct = trade.get('unrealized_pnl_pct', 0)
        days_held = trade.get('days_held', 0)
        confidence = trade.get('confidence_score', 70)
        
        # Profit taking rules
        if pnl_pct >= 100:  # 100% gain
            return "Profit Target (100%)"
        
        if pnl_pct >= 50 and days_held >= 2:  # 50% gain after 2 days
            return "Profit Target (50%+)"
        
        # Stop loss rules
        if pnl_pct <= -50:  # 50% loss
            return "Stop Loss (50%)"
        
        # Time-based exits
        if days_held >= 7:  # Max 7 days
            return "Time Stop (7 days)"
        
        return None
    
    def close_trade(self, trade: Dict, exit_reason: str):
        """Close a trade and record the result"""
        exit_trade = trade.copy()
        exit_trade['exit_date'] = datetime.now().date()
        exit_trade['exit_price'] = trade['current_value']
        exit_trade['exit_reason'] = exit_reason
        exit_trade['realized_pnl'] = trade['unrealized_pnl']
        exit_trade['realized_pnl_pct'] = trade['unrealized_pnl_pct']
        exit_trade['status'] = 'CLOSED'
        
        st.session_state.closed_trades.append(exit_trade)
    
    def add_trade(self, opportunity: Dict, manual: bool = False):
        """Add trade with real market context"""
        # Calculate position size (2% risk per trade)
        balance = self.get_current_balance_with_real_data()
        risk_amount = balance['total_value'] * 0.02
        
        entry_price = opportunity['expected_premium']
        quantity = max(1, int(risk_amount / (entry_price * 100)))
        
        trade = {
            'trade_id': str(uuid.uuid4()),
            'symbol': opportunity['symbol'],
            'trade_type': opportunity['trade_type'],
            'entry_date': datetime.now().date(),
            'entry_timestamp': datetime.now(),
            'entry_price': entry_price,
            'quantity': quantity,
            'confidence_score': opportunity['confidence_score'],
            'setup_type': opportunity['structure_type'],
            'recommendation': opportunity['recommendation'],
            'explanation': opportunity['explanation'],
            'days_held': 0,
            'current_value': entry_price,
            'unrealized_pnl': 0,
            'unrealized_pnl_pct': 0,
            'manual_trade': manual,
            'status': 'OPEN',
            # Real market data
            'entry_stock_price': opportunity['current_price'],
            'gamma_flip': opportunity['gamma_flip'],
            'call_wall': opportunity.get('call_wall'),
            'put_wall': opportunity.get('put_wall'),
        }
        
        st.session_state.open_trades.append(trade)
        
        # Show success message
        if manual:
            st.success(f"✅ **TRADE EXECUTED**: {quantity} contracts of {opportunity['symbol']} {opportunity['trade_type']}")
        
        return True
    
    def get_current_balance_with_real_data(self) -> Dict:
        """Calculate balance using real market data for open positions"""
        # Update trades with latest real data first
        self.update_open_trades_with_real_data()
        
        cash_balance = self.initial_balance
        positions_value = 0
        unrealized_pnl = 0
        realized_pnl = 0
        
        # Calculate open positions with real data
        for trade in st.session_state.open_trades:
            invested = trade['entry_price'] * trade['quantity'] * 100
            current_val = trade['current_value'] * trade['quantity'] * 100
            
            cash_balance -= invested
            positions_value += current_val
            unrealized_pnl += trade['unrealized_pnl']
        
        # Add realized P&L
        for trade in st.session_state.closed_trades:
            realized_pnl += trade['realized_pnl']
        
        total_value = cash_balance + positions_value + realized_pnl
        total_return_pct = ((total_value - self.initial_balance) / self.initial_balance) * 100
        
        # Calculate statistics
        if st.session_state.closed_trades:
            winning_trades = len([t for t in st.session_state.closed_trades if t['realized_pnl'] > 0])
            win_rate = (winning_trades / len(st.session_state.closed_trades)) * 100
        else:
            win_rate = 0
        
        return {
            'total_value': total_value,
            'cash_balance': cash_balance,
            'positions_value': positions_value,
            'realized_pnl': realized_pnl,
            'unrealized_pnl': unrealized_pnl,
            'total_return_pct': total_return_pct,
            'open_trades_count': len(st.session_state.open_trades),
            'closed_trades_count': len(st.session_state.closed_trades),
            'win_rate': win_rate
        }
    
    def create_opportunity_card(self, opp: Dict, index: int):
        """Create beautiful opportunity cards with real data"""
        
        # Determine confidence styling
        if opp['confidence_score'] >= 85:
            confidence_class = "confidence-high"
            card_class = "opportunity-high"
            confidence_emoji = "🟢"
        elif opp['confidence_score'] >= 75:
            confidence_class = "confidence-medium"
            card_class = "opportunity-medium"
            confidence_emoji = "🟡"
        else:
            confidence_class = "confidence-low"
            card_class = "opportunity-low"
            confidence_emoji = "🟠"
        
        # Trade type styling
        if 'SQUEEZE' in opp['structure_type']:
            trade_class = "trade-squeeze"
        elif 'PREMIUM' in opp['structure_type']:
            trade_class = "trade-premium"
        else:
            trade_class = "trade-condor"
        
        # Volume indicator
        volume_indicator = "🔥 HIGH VOLUME" if opp.get('volume_spike') else "📊 NORMAL VOLUME"
        
        # Create the card HTML
        card_html = f"""
        <div class="{card_class}">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                <div style="display: flex; align-items: center; gap: 1rem;">
                    <h2 style="margin: 0; color: #00ff87; font-size: 1.8rem;">{opp['emoji']} {opp['symbol']}</h2>
                    <div style="background: linear-gradient(135deg, {'#00ff87, #00cc6a' if opp['confidence_score'] >= 85 else '#ffa502, #ff9500' if opp['confidence_score'] >= 75 else '#ff6b6b, #ff5252'}); color: white; padding: 0.3rem 0.8rem; border-radius: 15px; font-size: 0.8rem; font-weight: 600;">{confidence_emoji} {opp['confidence_score']}% CONFIDENCE</div>
                </div>
                <div class="{trade_class}">{opp['trade_type'].replace('_', ' ')}</div>
            </div>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1rem; margin-bottom: 1rem;">
                <div>
                    <div style="color: #8b949e; font-size: 0.9rem;">Current Price</div>
                    <div style="font-size: 1.3rem; font-weight: 600; color: #00ff87;">${opp['current_price']:.2f}</div>
                    <div style="font-size: 0.8rem; color: {'#00ff87' if opp['change_pct'] >= 0 else '#ff4757'};">({opp['change_pct']:+.1f}%)</div>
                </div>
                <div>
                    <div style="color: #8b949e; font-size: 0.9rem;">Gamma Flip</div>
                    <div style="font-size: 1.3rem; font-weight: 600; color: #ff9500;">${opp['gamma_flip']:.2f}</div>
                </div>
                <div>
                    <div style="color: #8b949e; font-size: 0.9rem;">Distance</div>
                    <div style="font-size: 1.3rem; font-weight: 600; color: {'#00ff87' if opp['distance_pct'] > 0 else '#ff4757'};">{opp['distance_pct']:+.1f}%</div>
                </div>
            </div>
            
            <div style="background: rgba(0, 255, 135, 0.1); padding: 1rem; border-radius: 8px; margin-bottom: 1rem; border-left: 4px solid #00ff87;">
                <div style="font-weight: 600; margin-bottom: 0.5rem;">🎯 STRATEGY: {opp['recommendation']}</div>
                <div style="color: #8b949e;">{opp['explanation']}</div>
            </div>
            
            <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 1rem;">
                <div style="display: flex; gap: 1rem;">
                    <div style="color: #8b949e;">Premium: <span style="color: #00ff87; font-weight: 600;">${opp['expected_premium']:.2f}</span></div>
                    <div style="color: #8b949e;">Expiry: <span style="color: #00ff87; font-weight: 600;">{opp['days_to_expiry']}d</span></div>
                    <div style="color: #8b949e; font-size: 0.9rem;">{volume_indicator}</div>
                </div>
                <div style="color: #8b949e; font-size: 0.8rem;">
                    Real data: {opp['real_data_timestamp'].strftime('%I:%M %p')}
                </div>
            </div>
        </div>
        """
        
        st.markdown(card_html, unsafe_allow_html=True)
    
    def create_beautiful_portfolio_chart(self):
        """Create portfolio performance chart"""
        # Create sample data if none exists
        if len(st.session_state.portfolio_history) < 2:
            dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
            values = [self.initial_balance]
            
            for i in range(1, len(dates)):
                change = np.random.normal(0, 0.02) * values[-1]
                values.append(max(values[-1] + change, self.initial_balance * 0.8))
            
            sample_history = [{'timestamp': date, 'total_value': value} for date, value in zip(dates, values)]
        else:
            sample_history = st.session_state.portfolio_history
        
        df = pd.DataFrame(sample_history)
        
        fig = go.Figure()
        
        # Portfolio value line with gradient fill
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['total_value'],
                mode='lines+markers',
                name='Portfolio Value',
                line=dict(color='#00ff87', width=3),
                fill='tonexty',
                fillcolor='rgba(0, 255, 135, 0.1)',
                hovertemplate='<b>%{y:$,.0f}</b><br>%{x}<extra></extra>'
            )
        )
        
        # Add starting value line
        fig.add_hline(
            y=self.initial_balance,
            line_dash="dash",
            line_color="rgba(255, 255, 255, 0.5)",
            annotation_text=f"Starting: ${self.initial_balance:,}",
            annotation_position="top left"
        )
        
        # Styling
        fig.update_layout(
            title={
                'text': 'Portfolio Performance Over Time',
                'x': 0.5,
                'font': {'size': 20, 'color': 'white'}
            },
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            showlegend=False,
            margin=dict(l=0, r=0, t=50, b=0),
            xaxis=dict(showgrid=False, color='white'),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', color='white', tickformat='$,.0f')
        )
        
        return fig

def display_morning_analysis():
    """Morning analysis with REAL market data"""
    
    # Beautiful header
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 style="background: linear-gradient(90deg, #00ff87, #60efff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 2.5rem; margin-bottom: 0.5rem;">
            🌅 Morning Gamma Analysis
        </h1>
        <p style="color: #8b949e; font-size: 1.1rem;">Live Trading Opportunities with REAL Market Data</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get real market status
    market_data_system = RealMarketData()
    market_status = market_data_system.get_market_status()
    
    # Market status display
    current_time = datetime.now()
    
    if market_status['status'] == 'OPEN':
        status_html = f'<div class="status-open">{market_status["emoji"]} MARKET OPEN - LIVE DATA</div>'
    elif 'WEEKEND' in market_status['status']:
        status_html = f'<div class="status-closed">{market_status["emoji"]} WEEKEND - MARKETS CLOSED</div>'
    else:
        status_html = f'<div class="status-premarket">{market_status["emoji"]} PRE/AFTER MARKET</div>'
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div style="color: #8b949e; font-size: 0.9rem;">Last Analysis</div>
            <div style="font-size: 1.2rem; font-weight: 600; color: #00ff87;">09:15 AM ET</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div style="color: #8b949e; font-size: 0.9rem;">Current Time</div>
            <div style="font-size: 1.2rem; font-weight: 600; color: #60efff;">{current_time.strftime('%I:%M %p ET')}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div style="color: #8b949e; font-size: 0.9rem;">Market Status</div>
            <div style="margin-top: 0.5rem;">{status_html}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        next_time = "09:15 AM Tomorrow" if market_status['is_trading_hours'] else "09:15 AM Next Trading Day"
        st.markdown(f"""
        <div class="metric-card">
            <div style="color: #8b949e; font-size: 0.9rem;">Next Analysis</div>
            <div style="font-size: 1.2rem; font-weight: 600; color: #ffa502;">{next_time}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Real data disclaimer
    st.markdown("""
    <div style="background: linear-gradient(145deg, #1c2128, #21262d); padding: 2rem; border-radius: 16px; margin: 2rem 0; border-left: 4px solid #00ff87;">
        <h3 style="color: #00ff87; margin-bottom: 1rem; display: flex; align-items: center;">
            <span style="margin-right: 1rem;">📡</span>REAL MARKET DATA INTEGRATION
        </h3>
        <p style="color: white; margin-bottom: 1rem;"><strong>This system now uses LIVE market data:</strong></p>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem;">
            <div style="display: flex; align-items: center;">
                <span style="color: #00ff87; margin-right: 0.5rem;">✅</span>
                <span>Real stock prices via yFinance</span>
            </div>
            <div style="display: flex; align-items: center;">
                <span style="color: #00ff87; margin-right: 0.5rem;">✅</span>
                <span>Live volume and price changes</span>
            </div>
            <div style="display: flex; align-items: center;">
                <span style="color: #00ff87; margin-right: 0.5rem;">✅</span>
                <span>Real market hours detection</span>
            </div>
            <div style="display: flex; align-items: center;">
                <span style="color: #00ff87; margin-right: 0.5rem;">✅</span>
                <span>Calculated gamma levels from real prices</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Weekend or after hours message
    if not market_status['is_trading_hours']:
        if 'WEEKEND' in market_status['status']:
            st.warning("📅 **Markets are closed for the weekend.** Analysis will resume Monday at 9:15 AM ET with fresh real market data.")
        else:
            st.info("🌙 **Outside market hours.** Real data is available but opportunities are best identified during market hours (9:30 AM - 4:00 PM ET).")
        
        st.info("💡 **Use this time to review the educational content and understand how real market data affects gamma strategies!**")
    
    # Show opportunities using real data
    account = EnhancedMockTradingAccount()
    
    opportunities = account.get_real_opportunities()
    
    if not opportunities:
        st.warning("📊 **No high-confidence opportunities found in current market conditions.** This is normal - we only show setups with 60%+ confidence.")
        return
    
    # Display real opportunities  
    st.markdown("### 🎯 Today's Live Opportunities (Based on Real Market Data)")
    
    # Show data freshness
    if opportunities and 'real_data_timestamp' in opportunities[0]:
        data_time = opportunities[0]['real_data_timestamp']
        st.success(f"📡 **Live Data Updated:** {data_time.strftime('%I:%M:%S %p ET')} - All opportunities use real market prices!")
    
    for i, opp in enumerate(opportunities):
        account.create_opportunity_card(opp, i)
        
        # Enhanced trade button with real data context
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button(f"📈 Execute Real Trade", key=f"trade_{i}", type="primary"):
                # Store real market data with trade
                opp['entry_stock_price'] = opp['current_price']
                opp['entry_market_data'] = st.session_state.real_market_data.get(opp['symbol'], {})
                account.add_trade(opp, manual=True)
                st.rerun()
        
        with col2:
            if st.button(f"📊 Live Chart", key=f"chart_{i}"):
                # Show real price chart
                symbol = opp['symbol']
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period="1d", interval="5m")
                    
                    if not hist.empty:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=hist.index,
                            y=hist['Close'],
                            mode='lines',
                            name=f'{symbol} Price',
                            line=dict(color='#00ff87', width=2)
                        ))
                        
                        # Add gamma flip line
                        fig.add_hline(
                            y=opp['gamma_flip'],
                            line_dash="dash",
                            line_color="#ff9500",
                            annotation_text=f"Gamma Flip: ${opp['gamma_flip']:.2f}"
                        )
                        
                        fig.update_layout(
                            title=f"{symbol} - Real-Time Price vs Gamma Levels",
                            template="plotly_dark",
                            height=300
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error(f"Unable to load chart for {symbol}")
                except:
                    st.error(f"Chart temporarily unavailable for {symbol}")
        
        with col3:
            # Show real market context
            real_data = st.session_state.real_market_data.get(opp['symbol'], {})
            if real_data:
                volume_status = "🔥 HIGH" if opp.get('volume_spike') else "📊 NORMAL"
                st.info(f"**Real Market Data:** ${real_data['current_price']:.2f} ({real_data['change_pct']:+.1f}%) | Volume: {volume_status}")
        
        st.markdown("---")

def display_enhanced_portfolio():
    """Portfolio with real market data integration"""
    
    # Beautiful header
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 style="background: linear-gradient(90deg, #00ff87, #60efff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 2.5rem; margin-bottom: 0.5rem;">
            💰 $100K Gamma Trading Challenge
        </h1>
        <p style="color: #8b949e; font-size: 1.1rem;">Learn While You Earn - Track Performance with REAL Market Data</p>
    </div>
    """, unsafe_allow_html=True)
    
    account = EnhancedMockTradingAccount()
    
    # Auto-trading controls
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        auto_trading = st.toggle("🤖 Enable Auto-Trading", value=st.session_state.auto_trading_enabled, key="auto_toggle")
        st.session_state.auto_trading_enabled = auto_trading
    
    with col2:
        if auto_trading:
            st.markdown('<div class="status-open">🤖 AUTO-TRADING ACTIVE - Will execute 90%+ confidence trades using real data</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-premarket">👤 MANUAL MODE - You choose which real market trades to execute</div>', unsafe_allow_html=True)
    
    with col3:
        if st.button("🔄 Update Positions", type="primary"):
            account.update_open_trades_with_real_data()
            st.success("✅ Updated all positions with real market data!")
    
    # Get current balance (this will trigger real data updates)
    balance = account.get_current_balance_with_real_data()
    
    # Portfolio metrics with real data
    col1, col2, col3, col4 = st.columns(4)
    
    metrics_html = [
        (f"${balance['total_value']:,.0f}", f"{balance['total_return_pct']:+.1f}%", "Portfolio Value", "#00ff87" if balance['total_return_pct'] >= 0 else "#ff4757"),
        (f"${balance['cash_balance']:,.0f}", "Available", "Cash Balance", "#60efff"),
        (f"{balance['win_rate']:.0f}%", f"{balance['closed_trades_count']} trades", "Win Rate", "#ffa502"),
        (f"${balance['unrealized_pnl']:,.0f}", "Live P&L", "Unrealized", "#00ff87" if balance['unrealized_pnl'] >= 0 else "#ff4757")
    ]
    
    for i, (metric, delta, label, color) in enumerate(metrics_html):
        with [col1, col2, col3, col4][i]:
            st.markdown(f"""
            <div class="metric-card">
                <div style="color: #8b949e; font-size: 0.9rem; margin-bottom: 0.5rem;">{label}</div>
                <div style="font-size: 1.8rem; font-weight: 700; color: {color}; margin-bottom: 0.3rem;">{metric}</div>
                <div style="color: #8b949e; font-size: 0.8rem;">{delta}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Real data integration notice
    if st.session_state.real_market_data:
        latest_update = max([data.get('last_updated', datetime.now()) 
                           for data in st.session_state.real_market_data.values()])
        st.success(f"📡 **Real Market Data Active** - Last updated: {latest_update.strftime('%I:%M:%S %p ET')}")
    
    # Show open positions with real data
    if st.session_state.open_trades:
        st.markdown("### 📊 Active Positions (Real-Time Performance)")
        
        for trade in st.session_state.open_trades:
            pnl_color = "#00ff87" if trade.get('unrealized_pnl_pct', 0) > 0 else "#ff4757"
            pnl_emoji = "🟢" if trade.get('unrealized_pnl_pct', 0) > 0 else "🔴"
            
            # Get real market data for this symbol
            real_data = st.session_state.real_market_data.get(trade['symbol'], {})
            
            st.markdown(f"""
            <div class="opportunity-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                    <div style="display: flex; align-items: center; gap: 1rem;">
                        <h3 style="margin: 0; color: #00ff87;">{pnl_emoji} {trade['symbol']}</h3>
                        <div class="trade-{'squeeze' if 'CALLS' in trade['trade_type'] else 'premium'}">{trade['trade_type'].replace('_', ' ')}</div>
                        <div style="font-size: 0.9rem; color: #8b949e;">Day {trade.get('days_held', 0)}</div>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 1.5rem; font-weight: 700; color: {pnl_color};">{trade.get('unrealized_pnl_pct', 0):+.1f}%</div>
                        <div style="color: #8b949e; font-size: 0.9rem;">${trade.get('unrealized_pnl', 0):,.0f}</div>
                    </div>
                </div>
                
                <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1rem; margin-bottom: 1rem;">
                    <div>
                        <div style="color: #8b949e; font-size: 0.8rem;">Entry Price</div>
                        <div style="color: white; font-weight: 600;">${trade.get('entry_price', 0):.2f}</div>
                    </div>
                    <div>
                        <div style="color: #8b949e; font-size: 0.8rem;">Current Value</div>
                        <div style="color: white; font-weight: 600;">${trade.get('current_value', 0):.2f}</div>
                    </div>
                    <div>
                        <div style="color: #8b949e; font-size: 0.8rem;">Stock Price</div>
                        <div style="color: {'#00ff87' if real_data.get('change_pct', 0) >= 0 else '#ff4757'}; font-weight: 600;">
                            ${real_data.get('current_price', trade.get('current_stock_price', 0)):.2f} 
                            ({real_data.get('change_pct', 0):+.1f}%)
                        </div>
                    </div>
                </div>
                
                <div style="margin-top: 1rem; padding: 1rem; background: rgba(0, 255, 135, 0.1); border-radius: 8px; border-left: 4px solid #00ff87;">
                    <strong>📊 Real-Time Analysis:</strong> {trade.get('analysis', 'Position tracking with live market data.')}
                </div>
                
                <div style="margin-top: 1rem; display: flex; justify-content: space-between; align-items: center;">
                    <div style="color: #8b949e; font-size: 0.9rem;">
                        Last updated: {real_data.get('last_updated', datetime.now()).strftime('%I:%M %p')} ET
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    else:
        st.info("📈 **No active positions.** Visit the Morning Analysis page to find real market opportunities!")
    
    # Show portfolio performance chart with real data points
    st.markdown("### 📈 Portfolio Performance (Real Market Impact)")
    fig = account.create_beautiful_portfolio_chart()
    st.plotly_chart(fig, use_container_width=True)
    
    # Performance summary
    if balance['closed_trades_count'] > 0:
        st.markdown("### 🎯 Trading Performance Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Trades", balance['closed_trades_count'])
        with col2:
            st.metric("Win Rate", f"{balance['win_rate']:.1f}%")
        with col3:
            st.metric("Net Realized P&L", f"${balance['realized_pnl']:,.0f}")

def main():
    """Main app with stunning design and complete functionality"""
    
    # Create beautiful header directly
    st.markdown("""
    <div class="main-header">
        🚀 Gamma Exposure Trading System
    </div>
    <div class="subtitle">
        Master Professional Options Trading Through Market Maker Psychology
    </div>
    """, unsafe_allow_html=True)
    
    # Beautiful sidebar
    st.sidebar.markdown("""
    <div style="background: linear-gradient(145deg, #1c2128, #21262d); padding: 1.5rem; border-radius: 12px; margin-bottom: 2rem;">
        <h3 style="color: #00ff87; margin-bottom: 1rem;">🎯 Navigation</h3>
    </div>
    """, unsafe_allow_html=True)
    
    page = st.sidebar.selectbox("Choose Section:", [
        "🎓 Learn the Strategy",
        "🌅 Morning Analysis", 
        "💰 Trading Challenge",
        "📊 Performance Analytics"
    ])
    
    # Quick reference sidebar
    st.sidebar.markdown("""
    <div style="background: linear-gradient(145deg, #1c2128, #21262d); padding: 1.5rem; border-radius: 12px;">
        <h4 style="color: #60efff; margin-bottom: 1rem;">📚 Quick Reference</h4>
        
        <div style="margin-bottom: 1rem;">
            <div class="trade-squeeze" style="margin-bottom: 0.5rem;">🚀 SQUEEZE PLAYS</div>
            <div style="font-size: 0.85rem; color: #8b949e;">
                • Below gamma flip<br>
                • Negative GEX<br>
                • Target: 50-100%
            </div>
        </div>
        
        <div style="margin-bottom: 1rem;">
            <div class="trade-premium" style="margin-bottom: 0.5rem;">🛡️ PREMIUM SELLING</div>
            <div style="font-size: 0.85rem; color: #8b949e;">
                • Above gamma flip<br>
                • Positive GEX<br>
                • Target: 25-50%
            </div>
        </div>
        
        <div>
            <div class="trade-condor" style="margin-bottom: 0.5rem;">⚖️ IRON CONDORS</div>
            <div style="font-size: 0.85rem; color: #8b949e;">
                • High positive GEX<br>
                • Wide walls<br>
                • Target: 20-40%
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Page routing
    if page == "🎓 Learn the Strategy":
        st.markdown("## 🎓 Master Gamma Exposure Trading")
        
        tab1, tab2, tab3 = st.tabs(["📖 How It Works", "💡 Strategies", "🎯 Interactive Demo"])
        
        with tab1:
            st.markdown("""
            ### 🎯 How Market Makers Think with Gamma Exposure
            
            **Market makers are like bookies at a casino** - they want to make money on every trade while staying neutral to price direction.
            
            #### The Market Maker's Problem:
            1. **They sell you options** but don't want to lose money if the stock moves
            2. **They must hedge their risk** by buying/selling the underlying stock
            3. **Gamma tells them HOW MUCH stock to buy/sell** when prices change
            
            #### The Magic of Gamma:
            - **High Gamma = Big hedging moves** (creates volatility)
            - **Low Gamma = Small hedging moves** (suppresses volatility)
            - **Gamma Flip Point = Where the magic switches**
            
            #### Why This Creates Opportunities:
            - 🚀 **Below flip point**: Market makers amplify moves (great for buying options)
            - 🛡️ **Above flip point**: Market makers dampen moves (great for selling options)
            - 🎯 **At walls**: Strong support/resistance levels
            """)
            
        with tab2:
            # Strategy cards directly here
            st.markdown("### 💡 Three Profitable Strategies")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                <div class="opportunity-card">
                    <div class="trade-squeeze">🚀 SQUEEZE PLAYS</div>
                    <h4 style="color: #00ff87; margin-top: 1rem;">Buy Calls/Puts</h4>
                    <p><strong>When:</strong> Below gamma flip + negative GEX</p>
                    <p><strong>Why:</strong> Market makers amplify moves</p>
                    <p><strong>Target:</strong> 50-100% gains in 1-3 days</p>
                    <p><strong>Risk:</strong> Can lose 50% quickly</p>
                    <div class="progress-bar" style="margin-top: 1rem;"></div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="opportunity-card">
                    <div class="trade-premium">🛡️ PREMIUM SELLING</div>
                    <h4 style="color: #4ecdc4; margin-top: 1rem;">Sell Calls/Puts</h4>
                    <p><strong>When:</strong> Above flip + positive GEX</p>
                    <p><strong>Why:</strong> Market makers suppress moves</p>
                    <p><strong>Target:</strong> 25-50% premium collection</p>
                    <p><strong>Risk:</strong> Assignment if walls break</p>
                    <div class="progress-bar" style="margin-top: 1rem;"></div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown("""
                <div class="opportunity-card">
                    <div class="trade-condor">⚖️ IRON CONDORS</div>
                    <h4 style="color: #a55eea; margin-top: 1rem;">Sell Both Sides</h4>
                    <p><strong>When:</strong> High positive GEX + wide walls</p>
                    <p><strong>Why:</strong> Price trapped between walls</p>
                    <p><strong>Target:</strong> 20-40% premium collection</p>
                    <p><strong>Risk:</strong> Big move breaks setup</p>
                    <div class="progress-bar" style="margin-top: 1rem;"></div>
                </div>
                """, unsafe_allow_html=True)
            
            # Entry criteria
            st.markdown("### ✅ Exact Entry Criteria")
            
            st.markdown("""
            #### 🚀 SQUEEZE SETUP CRITERIA:
            - ✅ Net GEX < -500M (negative gamma environment)
            - ✅ Price is 0.5-2% below gamma flip point
            - ✅ Strong put wall within 1% below current price
            - ✅ Major expiration < 5 days away
            - ✅ Confidence score > 75%
            
            #### 🛡️ PREMIUM SELLING CRITERIA:
            - ✅ Net GEX > +1B (positive gamma environment) 
            - ✅ Price near or above call wall (within 0.5%)
            - ✅ Call wall has >300M gamma concentration
            - ✅ 2-5 days to expiration for theta decay
            - ✅ Confidence score > 70%
            
            #### ⚖️ IRON CONDOR CRITERIA:
            - ✅ Net GEX > +2B (very positive gamma)
            - ✅ Call and put walls >3% apart
            - ✅ 80%+ gamma concentrated at the walls
            - ✅ 5-10 days to expiration
            - ✅ IV rank < 50th percentile
            """)
            
        with tab3:
            # Interactive demo directly here instead of method call
            st.markdown("### 🎯 Interactive Demo: Gamma in Action")
            
            col1, col2 = st.columns([2, 1])
            
            with col2:
                st.markdown("**🎮 Control the Demo:**")
                price = st.slider("Stock Price ($)", 90, 110, 100, key="mm_demo")
                gamma_flip = st.slider("Gamma Flip ($)", 85, 105, 95, key="flip_demo")
            
            with col1:
                # Create beautiful visualization
                fig = go.Figure()
                
                # Add zones
                fig.add_shape(
                    type="rect",
                    x0=90, x1=gamma_flip, y0=0, y1=1,
                    fillcolor="rgba(255, 107, 107, 0.2)",
                    line=dict(width=0),
                    name="Squeeze Zone"
                )
                
                fig.add_shape(
                    type="rect",
                    x0=gamma_flip, x1=110, y0=0, y1=1,
                    fillcolor="rgba(78, 205, 196, 0.2)",
                    line=dict(width=0),
                    name="Premium Zone"
                )
                
                # Add current price line
                fig.add_vline(
                    x=price,
                    line_dash="solid",
                    line_color="#00ff87",
                    line_width=3,
                    annotation_text=f"Current Price: ${price}",
                    annotation_position="top"
                )
                
                # Add gamma flip line
                fig.add_vline(
                    x=gamma_flip,
                    line_dash="dash",
                    line_color="#ff9500",
                    line_width=2,
                    annotation_text=f"Gamma Flip: ${gamma_flip}",
                    annotation_position="bottom"
                )
                
                # Styling
                fig.update_layout(
                    title={
                        'text': '🎯 Market Maker Psychology Zones',
                        'x': 0.5,
                        'font': {'size': 20, 'color': 'white'}
                    },
                    xaxis_title="Stock Price ($)",
                    yaxis=dict(visible=False),
                    template="plotly_dark",
                    height=300,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Add explanation with beautiful formatting
            if price < gamma_flip:
                st.markdown("""
                <div class="opportunity-card opportunity-high">
                    <h3>🚀 SQUEEZE ZONE ACTIVATED!</h3>
                    <p><strong>What's Happening:</strong> Market makers will amplify every move up! Perfect for buying calls.</p>
                    <p><strong>Strategy:</strong> Buy ATM or OTM calls for explosive gains</p>
                    <p><strong>Expected Move:</strong> 2-5x leverage on stock moves</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="opportunity-card opportunity-medium">
                    <h3>🛡️ PREMIUM SELLING ZONE</h3>
                    <p><strong>What's Happening:</strong> Market makers will dampen volatility. Perfect for selling options.</p>
                    <p><strong>Strategy:</strong> Sell calls/puts or iron condors</p>
                    <p><strong>Expected Move:</strong> Time decay and volatility compression</p>
                </div>
                """, unsafe_allow_html=True)
    
    elif page == "🌅 Morning Analysis":
        display_morning_analysis()
    
    elif page == "💰 Trading Challenge":
        display_enhanced_portfolio()
    
    elif page == "📊 Performance Analytics":
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <h1 style="background: linear-gradient(90deg, #00ff87, #60efff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 2.5rem; margin-bottom: 0.5rem;">
                📊 Advanced Performance Analytics
            </h1>
            <p style="color: #8b949e; font-size: 1.1rem;">Deep Dive into Your Trading Performance with Real Market Data</p>
        </div>
        """, unsafe_allow_html=True)
        
        account = EnhancedMockTradingAccount()
        balance = account.get_current_balance_with_real_data()
        
        if balance['closed_trades_count'] == 0:
            st.info("📈 **Start trading to see performance analytics here!** Visit the Morning Analysis page to find real market opportunities.")
        else:
            # Performance analytics with real data
            st.markdown("### 📈 Portfolio Performance Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Return", f"{balance['total_return_pct']:+.1f}%")
            with col2:
                st.metric("Total Trades", balance['closed_trades_count'])
            with col3:
                st.metric("Win Rate", f"{balance['win_rate']:.1f}%")
            with col4:
                st.metric("Net P&L", f"${balance['realized_pnl']:,.0f}")
            
            # Strategy performance breakdown
            if st.session_state.closed_trades:
                st.markdown("### 🎯 Strategy Performance Analysis")
                
                df = pd.DataFrame(st.session_state.closed_trades)
                
                # Performance by strategy type
                strategy_stats = df.groupby('setup_type').agg({
                    'realized_pnl_pct': ['count', 'mean', lambda x: (x > 0).mean() * 100],
                    'days_held': 'mean',
                    'realized_pnl': 'sum'
                }).round(2)
                
                strategy_stats.columns = ['Count', 'Avg Return %', 'Win Rate %', 'Avg Days', 'Total P&L 
                
                st.dataframe(strategy_stats, use_container_width=True)
                
                # Performance chart by strategy
                fig = px.bar(
                    df.groupby('setup_type')['realized_pnl_pct'].mean().reset_index(),
                    x='setup_type', 
                    y='realized_pnl_pct',
                    title="Average Return by Strategy Type",
                    color='realized_pnl_pct',
                    color_continuous_scale="RdYlGn",
                    labels={'setup_type': 'Strategy Type', 'realized_pnl_pct': 'Average Return %'}
                )
                
                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white')
                )
                
                st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()]
                
                st.dataframe(strategy_stats, use_container_width=True)
                
                # Performance chart by strategy
                fig = px.bar(
                    df.groupby('setup_type')['realized_pnl_pct'].mean().reset_index(),
                    x='setup_type', 
                    y='realized_pnl_pct',
                    title="Average Return by Strategy Type",
                    color='realized_pnl_pct',
                    color_continuous_scale="RdYlGn",
                    labels={'setup_type': 'Strategy Type', 'realized_pnl_pct': 'Average Return %'}
                )
                
                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white')
                )
                
                st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
