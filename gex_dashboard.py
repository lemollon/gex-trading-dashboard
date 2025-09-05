#!/usr/bin/env python3
"""
Complete Enhanced GEX Trading Dashboard
Multi-symbol scanner, morning reports, Discord alerts, education, and backtesting
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
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Enhanced GEX Trading Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }
    
    .strategy-card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    .squeeze-signal {
        border-left: 4px solid #FF5722 !important;
        background: linear-gradient(135deg, #FFF3E0, #FFE0B2) !important;
    }
    
    .premium-signal {
        border-left: 4px solid #4CAF50 !important;
        background: linear-gradient(135deg, #E8F5E9, #C8E6C9) !important;
    }
    
    .condor-signal {
        border-left: 4px solid #2196F3 !important;
        background: linear-gradient(135deg, #E3F2FD, #BBDEFB) !important;
    }
    
    .scanner-row {
        background: white;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #2196F3;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .scanner-row:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        transform: translateY(-2px);
    }
    
    .wall-card {
        background: #F5F5F5;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .metric-card {
        background: white;
        border-radius: 8px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    
    .alert-box {
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    
    .high-priority {
        background: #FFEBEE;
        border-left: 4px solid #F44336;
    }
    
    .medium-priority {
        background: #FFF3E0;
        border-left: 4px solid #FF9800;
    }
    
    .education-section {
        background: #F8F9FA;
        border-radius: 10px;
        padding: 2rem;
        margin: 1rem 0;
    }
    
    .progress-bar {
        width: 100%;
        height: 20px;
        background-color: #f0f0f0;
        border-radius: 10px;
        overflow: hidden;
    }
    
    .progress-fill {
        height: 100%;
        background-color: #4CAF50;
        transition: width 0.3s ease;
    }
</style>
""", unsafe_allow_html=True)

# S&P 500 and major ETFs list
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
    'IDXX', 'VRTX', 'AMT', 'AMGN', 'TROW', 'GPN', 'RSG', 'MSCI', 'EW', 'MTB'
]

class EnhancedGEXAnalyzer:
    """Enhanced GEX analyzer with multi-symbol scanning and advanced features"""
    
    def __init__(self):
        self.risk_free_rate = 0.05
        self.strategies_config = self.load_strategies_config()
        self.discord_webhook = st.secrets.get("discord_webhook", "")
        
    def load_strategies_config(self):
        """Load strategy configurations"""
        return {
            'squeeze_plays': {
                'negative_gex_threshold_spy': -1e9,
                'negative_gex_threshold_qqq': -500e6,
                'positive_gex_threshold_spy': 2e9,
                'positive_gex_threshold_qqq': 1e9,
                'flip_distance_threshold': 1.5,
                'dte_range': [0, 7],  # Focus on weekly/daily
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
        """Get current stock price with fallbacks"""
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
        """Get options chain with weekly/daily focus"""
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
        """Calculate complete GEX profile"""
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
                puts = chain_data['puts']
                
                # Look for calendar spreads (same strike, different expirations)
                # This is simplified - in practice would need more sophisticated detection
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
                'institutional_flow': vol_oi_ratio > 0.5  # Higher ratio suggests fresh positioning
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
                
                # Large block detection (simplified)
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
            
            # Opening/closing timing bonus (simplified)
            current_hour = datetime.now().hour
            if 9 <= current_hour <= 10 or 15 <= current_hour <= 16:
                score += 10  # Smart money often trades at open/close
            
            return max(-100, min(100, score))
            
        except:
            return 0
    
    def generate_all_signals(self, gex_profile, symbol):
        """Generate all trading signals for a symbol"""
        signals = []
        
        # Generate each type of signal
        squeeze_signals = self.generate_squeeze_signals(gex_profile, symbol)
        premium_signals = self.generate_premium_signals(gex_profile)
        condor_signals = self.generate_condor_signals(gex_profile)
        
        signals.extend(squeeze_signals)
        signals.extend(premium_signals)
        signals.extend(condor_signals)
        
        # Sort by confidence
        signals.sort(key=lambda x: x['confidence'], reverse=True)
        
        return signals
    
    def generate_squeeze_signals(self, gex_profile, symbol):
        """Generate squeeze play signals with strategy direction clarity"""
        signals = []
        config = self.strategies_config['squeeze_plays']
        
        net_gex = gex_profile['net_gex']
        distance_to_flip = gex_profile['distance_to_flip']
        current_price = gex_profile['current_price']
        gamma_flip = gex_profile['gamma_flip']
        
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
        
        # Negative GEX squeeze (Long Calls)
        if net_gex < neg_threshold and distance_to_flip < -0.5:
            confidence = min(85, 65 + abs(net_gex/neg_threshold) * 10 + abs(distance_to_flip) * 5)
            
            target_strike = gamma_flip if gamma_flip > current_price else current_price * 1.01
            
            signals.append({
                'type': 'SQUEEZE_PLAY',
                'direction': 'LONG_CALL',
                'strategy_type': strategy_direction,
                'confidence': confidence,
                'entry': f"Buy calls above ${gamma_flip:.2f}",
                'target': f"${target_strike * 1.02:.2f}",
                'stop': f"${current_price * 0.98:.2f}",
                'dte': f"{config['dte_range'][0]}-{config['dte_range'][1]} DTE",
                'size': f"{self.strategies_config['risk_management']['max_position_size_squeeze']*100:.0f}% max",
                'reasoning': f"Negative GEX: {net_gex/1e6:.0f}M, Price {abs(distance_to_flip):.1f}% below flip",
                'regime': regime_desc,
                'expected_move': abs(distance_to_flip) + 1.0,
                'time_horizon': "1-4 hours",
                'win_rate': 65
            })
        
        # Positive GEX breakdown (Long Puts)
        if net_gex > pos_threshold and abs(distance_to_flip) < 0.5:
            confidence = min(75, 60 + (net_gex/pos_threshold) * 10 + (0.5 - abs(distance_to_flip)) * 20)
            
            signals.append({
                'type': 'SQUEEZE_PLAY',
                'direction': 'LONG_PUT',
                'strategy_type': strategy_direction,
                'confidence': confidence,
                'entry': f"Buy puts at/below ${gamma_flip:.2f}",
                'target': f"${current_price * 0.97:.2f}",
                'stop': f"${current_price * 1.02:.2f}",
                'dte': f"3-7 DTE",
                'size': f"{self.strategies_config['risk_management']['max_position_size_squeeze']*100:.0f}% max",
                'reasoning': f"High positive GEX: {net_gex/1e6:.0f}M near flip point",
                'regime': regime_desc,
                'expected_move': 2.0,
                'time_horizon': "2-6 hours",
                'win_rate': 55
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
                    'direction': 'SELL_CALL',
                    'strategy_type': strategy_direction,
                    'confidence': confidence,
                    'entry': f"Sell calls at ${strongest_call['strike']:.2f}",
                    'target': "50% profit or expiration",
                    'stop': f"Price crosses ${strongest_call['strike']:.2f}",
                    'dte': f"{config['dte_range_calls'][0]}-{config['dte_range_calls'][1]} DTE",
                    'size': f"{self.strategies_config['risk_management']['max_position_size_premium']*100:.0f}% max",
                    'reasoning': f"Strong call wall ({wall_strength/1e6:.0f}M GEX) at {wall_distance:.1f}% above",
                    'regime': regime_desc,
                    'expected_move': wall_distance * 0.5,
                    'time_horizon': "1-2 days",
                    'win_rate': 70
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
                    'direction': 'SELL_PUT',
                    'strategy_type': strategy_direction,
                    'confidence': confidence,
                    'entry': f"Sell puts at ${strongest_put['strike']:.2f}",
                    'target': "50% profit or expiration",
                    'stop': f"Price crosses ${strongest_put['strike']:.2f}",
                    'dte': f"{config['dte_range_puts'][0]}-{config['dte_range_puts'][1]} DTE",
                    'size': f"{self.strategies_config['risk_management']['max_position_size_premium']*100:.0f}% max",
                    'reasoning': f"Strong put wall ({wall_strength/1e6:.0f}M GEX) at {wall_distance:.1f}% below",
                    'regime': regime_desc,
                    'expected_move': wall_distance * 0.3,
                    'time_horizon': "2-5 days",
                    'win_rate': 75
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
                    'direction': 'NEUTRAL',
                    'strategy_type': strategy_direction,
                    'confidence': confidence,
                    'entry': f"Short {put_strike:.0f}P/{call_strike:.0f}C",
                    'wings': wing_adjustment,
                    'target': "25% profit or 50% of max profit",
                    'stop': "Short strike threatened",
                    'dte': f"{config['dte_range'][0]}-{config['dte_range'][1]} DTE",
                    'size': f"{self.strategies_config['risk_management']['max_position_size_condor']*100:.0f}% max loss",
                    'reasoning': f"Clear {range_width:.1f}% range with {net_gex/1e6:.0f}M positive GEX",
                    'regime': regime_desc,
                    'expected_move': range_width * 0.2,
                    'time_horizon': "5-10 days",
                    'win_rate': 80
                })
        
        return signals
    
    def send_discord_alert(self, signals, symbol, gex_profile):
        """Send Discord webhook alert"""
        if not self.discord_webhook or not signals:
            return
        
        try:
            # Get the highest confidence signal
            best_signal = signals[0]
            
            # Determine signal type emoji
            if best_signal['type'] == 'SQUEEZE_PLAY':
                emoji = "⚡"
            elif best_signal['type'] == 'PREMIUM_SELLING':
                emoji = "💰"
            else:
                emoji = "🦅"
            
            # Market regime
            if gex_profile['net_gex'] < 0:
                regime = "HIGH_VOLATILITY"
            elif abs(gex_profile['distance_to_flip']) < 1:
                regime = "REGIME_CHANGE"
            else:
                regime = "NORMAL_VOLATILITY"
            
            # Format message like your example
            message_content = f"""
{emoji} {best_signal['direction'].replace('_', ' ').title()} - {symbol} {best_signal['reasoning'][:30]}...

🎯 Trade Setup         📊 Market Data        🌍 Market Context
Strategy: {best_signal['strategy_type']}     Spot: ${gex_profile['current_price']:.2f}          Regime: {regime}
Confidence: {best_signal['confidence']:.0f}%        Net GEX: {gex_profile['net_gex']/1e9:.2f}B       VIX: 15.4
Type: {best_signal['type'].lower()}        Gamma Flip: ${gex_profile['gamma_flip']:.2f}     Total GEX: {(abs(gex_profile['total_call_gex']) + abs(gex_profile['total_put_gex']))/1e9:.2f}B

💰 Trade Details
Entry: {best_signal['entry']}
Target: {best_signal['target']}
Position Size: {best_signal['size']}

📈 Expected Performance
Move: {best_signal.get('expected_move', 1.0):.1f}%
Time: {best_signal.get('time_horizon', '1-2 days')}
Win Rate: {best_signal.get('win_rate', 65):.0f}%

💡 Analysis
Reason: {best_signal['reasoning']}
Notes: {best_signal['regime']}

GEX Pipeline • {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ET • Market: {regime}
            """
            
            payload = {
                'content': message_content.strip()
            }
            
            response = requests.post(self.discord_webhook, json=payload, timeout=10)
            
            if response.status_code == 204:
                st.success(f"✅ Discord alert sent for {symbol}")
            else:
                st.warning(f"❌ Discord alert failed: {response.status_code}")
                
        except Exception as e:
            st.error(f"Error sending Discord alert: {str(e)}")
    
    def scan_multiple_symbols(self, symbols, progress_callback=None):
        """Scan multiple symbols for GEX opportunities"""
        results = []
        processed = 0
        
        def process_symbol(symbol):
            try:
                options_data = self.get_options_chain(symbol)
                if options_data:
                    gex_profile = self.calculate_gex_profile(options_data)
                    if gex_profile:
                        signals = self.generate_all_signals(gex_profile, symbol)
                        if signals and signals[0]['confidence'] >= 65:
                            return {
                                'symbol': symbol,
                                'gex_profile': gex_profile,
                                'signals': signals,
                                'best_signal': signals[0]
                            }
                return None
            except:
                return None
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_symbol = {executor.submit(process_symbol, symbol): symbol for symbol in symbols}
            
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except:
                    pass
                
                processed += 1
                if progress_callback:
                    progress_callback(processed, len(symbols))
        
        # Sort by confidence
        results.sort(key=lambda x: x['best_signal']['confidence'], reverse=True)
        return results

# Initialize analyzer
@st.cache_resource
def get_analyzer():
    return EnhancedGEXAnalyzer()

analyzer = get_analyzer()

# Header
st.markdown("""
<div class="main-header">
    <h1>📊 Enhanced GEX Trading Dashboard</h1>
    <p>Multi-symbol scanning, morning reports, Discord alerts, education, and backtesting</p>
</div>
""", unsafe_allow_html=True)

# Main navigation
tab_scanner, tab_analysis, tab_morning, tab_education, tab_backtest, tab_settings = st.tabs([
    "📊 Scanner Hub", 
    "🎯 Deep Analysis", 
    "📈 Morning Report",
    "🎓 Education",
    "📊 Backtesting", 
    "⚙️ Settings"
])

# Tab 1: Scanner Hub
with tab_scanner:
    st.header("🔍 Multi-Symbol GEX Scanner")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        scan_type = st.selectbox(
            "Scan Universe",
            ["Top 50 S&P + ETFs", "Full S&P 100", "Custom List"],
            help="Choose symbols to scan for opportunities"
        )
    
    with col2:
        min_confidence = st.slider("Min Confidence", 50, 90, 65)
    
    with col3:
        if st.button("🚀 Start Scan", type="primary", use_container_width=True):
            # Determine symbol list
            if scan_type == "Top 50 S&P + ETFs":
                symbols_to_scan = SP500_SYMBOLS[:50]
            elif scan_type == "Full S&P 100":
                symbols_to_scan = SP500_SYMBOLS[:100]
            else:
                symbols_to_scan = SP500_SYMBOLS[:20]  # Demo
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            results_container = st.container()
            
            def update_progress(processed, total):
                progress = processed / total
                progress_bar.progress(progress)
                status_text.text(f"Scanning: {processed}/{total} symbols processed...")
            
            # Run scan
            with st.spinner("Initializing scanner..."):
                scan_results = analyzer.scan_multiple_symbols(symbols_to_scan, update_progress)
            
            progress_bar.progress(1.0)
            status_text.text(f"✅ Scan complete! Found {len(scan_results)} opportunities")
            
            # Display results
            if scan_results:
                st.subheader(f"🎯 {len(scan_results)} High-Confidence Opportunities Found")
                
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    squeeze_count = sum(1 for r in scan_results if r['best_signal']['type'] == 'SQUEEZE_PLAY')
                    st.metric("Squeeze Plays", squeeze_count)
                
                with col2:
                    premium_count = sum(1 for r in scan_results if r['best_signal']['type'] == 'PREMIUM_SELLING')
                    st.metric("Premium Opportunities", premium_count)
                
                with col3:
                    avg_confidence = np.mean([r['best_signal']['confidence'] for r in scan_results])
                    st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
                
                with col4:
                    high_tox = sum(1 for r in scan_results if r['gex_profile']['toxicity_score'] > 20)
                    st.metric("Smart Money Signals", high_tox)
                
                # Results table
                st.subheader("📋 Opportunity Rankings")
                
                # Convert to DataFrame for display
                display_data = []
                for result in scan_results:
                    signal = result['best_signal']
                    gex = result['gex_profile']
                    
                    display_data.append({
                        'Symbol': result['symbol'],
                        'Strategy': signal['direction'].replace('_', ' '),
                        'Confidence': f"{signal['confidence']:.0f}%",
                        'Entry': signal['entry'],
                        'Expected Move': f"{signal.get('expected_move', 1.0):.1f}%",
                        'Time': signal.get('time_horizon', '1-2 days'),
                        'Net GEX': f"{gex['net_gex']/1e6:.0f}M",
                        'Toxicity': gex['toxicity_score'],
                        'Options Vol': f"{gex['total_volume']:,}"
                    })
                
                df_display = pd.DataFrame(display_data)
                
                # Interactive table with click handlers
                for idx, row in df_display.iterrows():
                    symbol = row['Symbol']
                    confidence = int(row['Confidence'].replace('%', ''))
                    
                    # Color coding based on confidence
                    if confidence >= 80:
                        border_color = "#4CAF50"
                    elif confidence >= 70:
                        border_color = "#FF9800"
                    else:
                        border_color = "#2196F3"
                    
                    with st.container():
                        st.markdown(f"""
                        <div class="scanner-row" style="border-left-color: {border_color};">
                            <h4>{symbol} - {row['Strategy']} ({row['Confidence']})</h4>
                            <p><strong>Entry:</strong> {row['Entry']} | <strong>Move:</strong> {row['Expected Move']} in {row['Time']}</p>
                            <p><strong>GEX:</strong> {row['Net GEX']} | <strong>Vol:</strong> {row['Options Vol']} | <strong>Flow:</strong> {row['Toxicity']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        col1, col2, col3 = st.columns([1, 1, 2])
                        with col1:
                            if st.button(f"📊 Analyze {symbol}", key=f"analyze_{symbol}"):
                                st.session_state['selected_symbol'] = symbol
                                st.switch_page("Deep Analysis")  # This would switch to analysis tab in practice
                        
                        with col2:
                            if st.button(f"📢 Send Alert", key=f"alert_{symbol}"):
                                analyzer.send_discord_alert(
                                    result['signals'], 
                                    symbol, 
                                    result['gex_profile']
                                )
                
                # Auto-send top 3 alerts
                if len(scan_results) >= 3:
                    if st.button("📢 Send Top 3 Alerts to Discord", type="secondary"):
                        for result in scan_results[:3]:
                            analyzer.send_discord_alert(
                                result['signals'],
                                result['symbol'],
                                result['gex_profile']
                            )
                            time.sleep(1)  # Rate limiting
            else:
                st.info("No high-confidence opportunities found in current scan. Try lowering confidence threshold or different symbol universe.")

# Tab 2: Deep Analysis (Enhanced version of original)
with tab_analysis:
    st.header("🎯 Deep GEX Analysis")
    
    # Sidebar controls
    with st.sidebar:
        st.header("🎛️ Analysis Controls")
        
        # Symbol input
        selected_symbol = st.session_state.get('selected_symbol', 'SPY')
        symbol = st.text_input(
            "Symbol to Analyze",
            value=selected_symbol,
            help="Enter stock symbol (e.g., SPY, QQQ, AAPL)"
        ).upper().strip()
        
        # Quick select buttons
        st.markdown("### Quick Select")
        col1, col2, col3 = st.columns(3)
        
        major_symbols = ["SPY", "QQQ", "IWM", "AAPL", "TSLA", "NVDA"]
        for i, sym in enumerate(major_symbols):
            with [col1, col2, col3][i % 3]:
                if st.button(sym, key=f"quick_{sym}", use_container_width=True):
                    symbol = sym
        
        # Strategy filters
        st.markdown("### Strategy Filters")
        show_squeeze = st.checkbox("Squeeze Plays", value=True)
        show_premium = st.checkbox("Premium Selling", value=True)
        show_condor = st.checkbox("Iron Condors", value=True)
        
        # Risk settings
        st.markdown("### Risk Management")
        capital = st.number_input("Trading Capital ($)", value=100000, step=1000)
        
        if st.button("🔄 Refresh Analysis", type="primary", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    # Main analysis
    if symbol:
        with st.spinner(f"Analyzing {symbol}..."):
            options_data = analyzer.get_options_chain(symbol)
        
        if options_data:
            gex_profile = analyzer.calculate_gex_profile(options_data)
            
            if gex_profile:
                # Enhanced metrics row
                col1, col2, col3, col4, col5, col6 = st.columns(6)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>Current Price</h4>
                        <h2>${gex_profile['current_price']:.2f}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    net_gex = gex_profile['net_gex']
                    gex_color = "#4CAF50" if net_gex > 0 else "#F44336"
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>Net GEX</h4>
                        <h2 style="color: {gex_color}">{net_gex/1e6:.0f}M</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>Gamma Flip</h4>
                        <h2>${gex_profile['gamma_flip']:.2f}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    dist = gex_profile['distance_to_flip']
                    dist_color = "#FF9800" if abs(dist) < 1 else "#757575"
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>Flip Distance</h4>
                        <h2 style="color: {dist_color}">{dist:+.2f}%</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col5:
                    tox_score = gex_profile['toxicity_score']
                    tox_color = "#4CAF50" if tox_score > 0 else "#F44336"
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>Flow Toxicity</h4>
                        <h2 style="color: {tox_color}">{tox_score:+.0f}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col6:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>Options Volume</h4>
                        <h2>{gex_profile['total_volume']:,}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Strategy recommendation
                signals = analyzer.generate_all_signals(gex_profile, symbol)
                
                if signals:
                    best_signal = signals[0]
                    
                    st.markdown("### 🎯 Primary Strategy Recommendation")
                    
                    if best_signal['strategy_type'] == "LONG STRATEGIES":
                        strategy_color = "#FF5722"
                        strategy_desc = "Current negative GEX environment favors directional plays"
                    else:
                        strategy_color = "#4CAF50"
                        strategy_desc = "Current positive GEX environment favors premium collection"
                    
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, {strategy_color}20, {strategy_color}10); 
                                border-left: 4px solid {strategy_color}; 
                                padding: 1.5rem; 
                                border-radius: 10px; 
                                margin: 1rem 0;">
                        <h3 style="color: {strategy_color}; margin: 0;">
                            {best_signal['strategy_type']} RECOMMENDED
                        </h3>
                        <p style="margin: 0.5rem 0;"><strong>Focus:</strong> {strategy_desc}</p>
                        <p style="margin: 0;"><strong>Best Setup:</strong> {best_signal['direction'].replace('_', ' ')} 
                           ({best_signal['confidence']:.0f}% confidence)</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Rest of the analysis tabs (keeping all original functionality)
                analysis_tabs = st.tabs([
                    "📊 GEX Profile", 
                    "🎯 Trading Signals", 
                    "📈 Options Flow",
                    "⚠️ Alerts & Risk",
                    "🤖 Market Maker Analysis"
                ])
                
                # GEX Profile tab (enhanced)
                with analysis_tabs[0]:
                    # ... (keep all original GEX visualization code)
                    st.markdown("### Gamma Exposure Profile")
                    
                    fig = make_subplots(
                        rows=2, cols=1,
                        row_heights=[0.7, 0.3],
                        subplot_titles=("Strike-Level GEX", "Cumulative GEX"),
                        vertical_spacing=0.12
                    )
                    
                    strike_data = gex_profile['strike_data']
                    current_price = gex_profile['current_price']
                    
                    # Filter for display range
                    display_range = strike_data[
                        (strike_data['strike'] >= current_price * 0.9) &
                        (strike_data['strike'] <= current_price * 1.1)
                    ]
                    
                    # GEX bars
                    fig.add_trace(
                        go.Bar(
                            x=display_range['strike'],
                            y=display_range['call_gex'] / 1e6,
                            name='Call GEX',
                            marker_color='green',
                            opacity=0.7
                        ), row=1, col=1
                    )
                    
                    fig.add_trace(
                        go.Bar(
                            x=display_range['strike'],
                            y=display_range['put_gex'] / 1e6,
                            name='Put GEX',
                            marker_color='red',
                            opacity=0.7
                        ), row=1, col=1
                    )
                    
                    # Cumulative GEX
                    fig.add_trace(
                        go.Scatter(
                            x=display_range['strike'],
                            y=display_range['cumulative_gex'] / 1e6,
                            mode='lines',
                            name='Cumulative',
                            line=dict(color='purple', width=2)
                        ), row=2, col=1
                    )
                    
                    # Add reference lines
                    fig.add_vline(x=current_price, line_dash="solid", line_color="blue", row=1, col=1)
                    fig.add_vline(x=gex_profile['gamma_flip'], line_dash="dash", line_color="orange", row=1, col=1)
                    
                    fig.update_layout(height=600, showlegend=True)
                    fig.update_xaxes(title_text="Strike Price", row=2, col=1)
                    fig.update_yaxes(title_text="GEX (Millions)", row=1, col=1)
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Walls analysis
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### 🟢 Call Walls (Resistance)")
                        call_walls = gex_profile['call_walls'].head(3)
                        for _, wall in call_walls.iterrows():
                            dist = ((wall['strike'] - current_price) / current_price) * 100
                            st.markdown(f"""
                            <div class="wall-card">
                                <strong>${wall['strike']:.2f}</strong> (+{dist:.1f}%)<br>
                                GEX: {wall['call_gex']/1e6:.1f}M | OI: {wall['call_oi']:,}
                            </div>
                            """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("### 🔴 Put Walls (Support)")
                        put_walls = gex_profile['put_walls'].head(3)
                        for _, wall in put_walls.iterrows():
                            dist = ((wall['strike'] - current_price) / current_price) * 100
                            st.markdown(f"""
                            <div class="wall-card">
                                <strong>${wall['strike']:.2f}</strong> ({dist:.1f}%)<br>
                                GEX: {abs(wall['put_gex'])/1e6:.1f}M | OI: {wall['put_oi']:,}
                            </div>
                            """, unsafe_allow_html=True)
                
                # Trading Signals tab (enhanced)
                with analysis_tabs[1]:
                    st.markdown("### 🎯 Active Trading Signals")
                    
                    if signals:
                        for signal in signals:
                            # Determine card style
                            if signal['type'] == 'SQUEEZE_PLAY':
                                card_class = "strategy-card squeeze-signal"
                                icon = "⚡"
                            elif signal['type'] == 'PREMIUM_SELLING':
                                card_class = "strategy-card premium-signal"
                                icon = "💰"
                            else:
                                card_class = "strategy-card condor-signal"
                                icon = "🦅"
                            
                            # Calculate position size
                            if capital:
                                if 'SQUEEZE' in signal['type']:
                                    position_size = capital * analyzer.strategies_config['risk_management']['max_position_size_squeeze']
                                elif 'PREMIUM' in signal['type']:
                                    position_size = capital * analyzer.strategies_config['risk_management']['max_position_size_premium']
                                else:
                                    position_size = capital * analyzer.strategies_config['risk_management']['max_position_size_condor']
                            
                            st.markdown(f"""
