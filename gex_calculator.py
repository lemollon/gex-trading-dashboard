"""
GEX Calculator - Core Gamma Exposure Calculation Engine
Handles options data processing and GEX calculations for trading analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import requests
import json
import time

class GEXCalculator:
    """
    Core Gamma Exposure calculation engine
    """
    
    def __init__(self):
        self.multiplier = 100  # Options multiplier
        self.risk_free_rate = 0.05  # 5% risk-free rate assumption
        
    def calculate_gamma_exposure(self, 
                               options_df: pd.DataFrame, 
                               spot_price: float,
                               symbol: str) -> Dict:
        """
        Calculate comprehensive GEX metrics for a symbol
        
        Args:
            options_df: DataFrame with options chain data
            spot_price: Current underlying price
            symbol: Stock/ETF symbol
            
        Returns:
            Dictionary with GEX calculations and key levels
        """
        
        try:
            if options_df.empty:
                return self._empty_result(symbol, "Empty options data")
            
            # Clean and validate data
            df = self._clean_options_data(options_df)
            if df.empty:
                return self._empty_result(symbol, "No valid options after cleaning")
            
            # Calculate gamma if not present
            if 'gamma' not in df.columns:
                df['gamma'] = self._estimate_gamma(df, spot_price)
            
            # Calculate GEX for each option
            df['gex'] = self._calculate_option_gex(df, spot_price)
            
            # Aggregate by strike
            gex_by_strike = df.groupby('strike').agg({
                'gex': 'sum',
                'open_interest': 'sum',
                'gamma': 'sum'
            }).reset_index()
            
            # Calculate key metrics
            net_gex = gex_by_strike['gex'].sum()
            
            # Find gamma flip point
            cumulative_gex = gex_by_strike.sort_values('strike')['gex'].cumsum()
            flip_point = self._find_flip_point(gex_by_strike, spot_price)
            
            # Identify walls
            call_walls = self._find_call_walls(gex_by_strike)
            put_walls = self._find_put_walls(gex_by_strike)
            
            # Calculate distances and levels
            distance_to_flip = ((spot_price - flip_point) / spot_price) * 100
            
            # Determine market regime
            regime = self._determine_regime(net_gex, distance_to_flip)
            
            # Calculate expected move
            expected_move = self._calculate_expected_move(gex_by_strike, spot_price)
            
            return {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'spot_price': spot_price,
                'net_gex': net_gex,
                'gamma_flip_point': flip_point,
                'distance_to_flip': distance_to_flip,
                'regime': regime,
                'call_walls': call_walls,
                'put_walls': put_walls,
                'expected_move': expected_move,
                'gex_by_strike': gex_by_strike.to_dict('records'),
                'total_call_gex': gex_by_strike[gex_by_strike['gex'] > 0]['gex'].sum(),
                'total_put_gex': gex_by_strike[gex_by_strike['gex'] < 0]['gex'].sum(),
                'success': True
            }
            
        except Exception as e:
            return self._empty_result(symbol, f"Calculation error: {str(e)}")
    
    def _clean_options_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate options data"""
        
        # Required columns
        required_cols = ['strike', 'option_type', 'open_interest']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Convert to numeric
        df = df.copy()
        df['strike'] = pd.to_numeric(df['strike'], errors='coerce')
        df['open_interest'] = pd.to_numeric(df['open_interest'], errors='coerce')
        
        # Clean option type
        df['option_type'] = df['option_type'].str.upper()
        df = df[df['option_type'].isin(['C', 'CALL', 'P', 'PUT'])]
        
        # Remove invalid data
        df = df.dropna(subset=['strike', 'open_interest'])
        df = df[df['open_interest'] > 0]  # Only options with OI
        
        return df
    
    def _estimate_gamma(self, df: pd.DataFrame, spot_price: float) -> pd.Series:
        """Estimate gamma for options using simplified BSM approximation"""
        
        # Simplified gamma estimation
        # Gamma is highest ATM and decreases with distance
        strikes = df['strike']
        
        # Distance from ATM (normalized)
        distance = np.abs(strikes - spot_price) / spot_price
        
        # Simplified gamma curve (highest ATM, decreases exponentially)
        gamma_est = np.exp(-10 * distance**2) * 0.02  # Peak gamma of ~0.02
        
        return gamma_est
    
    def _calculate_option_gex(self, df: pd.DataFrame, spot_price: float) -> pd.Series:
        """Calculate GEX for each option"""
        
        def gex_for_option(row):
            # GEX = Spot × Gamma × Open Interest × Multiplier
            base_gex = spot_price * row['gamma'] * row['open_interest'] * self.multiplier
            
            # Calls contribute positive GEX, puts negative
            if row['option_type'] in ['C', 'CALL']:
                return base_gex
            else:
                return -base_gex
        
        return df.apply(gex_for_option, axis=1)
    
    def _find_flip_point(self, gex_by_strike: pd.DataFrame, spot_price: float) -> float:
        """Find gamma flip point where cumulative GEX crosses zero"""
        
        sorted_strikes = gex_by_strike.sort_values('strike')
        cumulative_gex = sorted_strikes['gex'].cumsum()
        
        # Find where cumulative GEX goes from negative to positive
        for i, (_, row) in enumerate(sorted_strikes.iterrows()):
            if cumulative_gex.iloc[i] >= 0:
                return row['strike']
        
        # If no flip point found, return spot price
        return spot_price
    
    def _find_call_walls(self, gex_by_strike: pd.DataFrame, top_n: int = 3) -> List[Dict]:
        """Find top call walls (resistance levels)"""
        
        call_gex = gex_by_strike[gex_by_strike['gex'] > 0].copy()
        call_gex = call_gex.sort_values('gex', ascending=False)
        
        walls = []
        for _, row in call_gex.head(top_n).iterrows():
            walls.append({
                'strike': row['strike'],
                'gex': row['gex'],
                'open_interest': row['open_interest'],
                'type': 'resistance'
            })
        
        return walls
    
    def _find_put_walls(self, gex_by_strike: pd.DataFrame, top_n: int = 3) -> List[Dict]:
        """Find top put walls (support levels)"""
        
        put_gex = gex_by_strike[gex_by_strike['gex'] < 0].copy()
        put_gex = put_gex.sort_values('gex', ascending=True)  # Most negative first
        
        walls = []
        for _, row in put_gex.head(top_n).iterrows():
            walls.append({
                'strike': row['strike'],
                'gex': row['gex'],
                'open_interest': row['open_interest'],
                'type': 'support'
            })
        
        return walls
    
    def _determine_regime(self, net_gex: float, distance_to_flip: float) -> str:
        """Determine market regime based on GEX"""
        
        if net_gex > 1e9:  # > 1B
            return "High Positive GEX - Volatility Suppression"
        elif net_gex > 0:
            return "Positive GEX - Dealer Long Gamma"
        elif net_gex > -5e8:  # > -500M
            return "Low Negative GEX - Neutral"
        else:
            return "High Negative GEX - Volatility Amplification"
    
    def _calculate_expected_move(self, gex_by_strike: pd.DataFrame, spot_price: float) -> Dict:
        """Calculate expected move based on GEX concentration"""
        
        # Find strikes with significant GEX concentration
        significant_gex = gex_by_strike[
            np.abs(gex_by_strike['gex']) > np.abs(gex_by_strike['gex']).quantile(0.8)
        ]
        
        if significant_gex.empty:
            return {'up': 0, 'down': 0, 'range_pct': 0}
        
        max_strike = significant_gex['strike'].max()
        min_strike = significant_gex['strike'].min()
        
        up_move = ((max_strike - spot_price) / spot_price) * 100
        down_move = ((spot_price - min_strike) / spot_price) * 100
        range_pct = ((max_strike - min_strike) / spot_price) * 100
        
        return {
            'up': max(0, up_move),
            'down': max(0, down_move),
            'range_pct': range_pct
        }
    
    def _empty_result(self, symbol: str, error: str) -> Dict:
        """Return empty result structure"""
        return {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'error': error,
            'success': False,
            'spot_price': 0,
            'net_gex': 0,
            'gamma_flip_point': 0,
            'distance_to_flip': 0,
            'regime': 'Unknown',
            'call_walls': [],
            'put_walls': [],
            'expected_move': {'up': 0, 'down': 0, 'range_pct': 0},
            'gex_by_strike': [],
            'total_call_gex': 0,
            'total_put_gex': 0
        }

class TradingVolatilityAPI:
    """
    API client for TradingVolatility.net options data
    """
    
    def __init__(self, username: str = None):
        self.base_url = "https://stocks.tradingvolatility.net/api"
        self.username = username
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
        })
        self.api_calls_made = 0
        self.last_call_time = 0
    
    def fetch_options_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch options data with rate limiting and error handling"""
        
        # Rate limiting (2 calls per minute during market hours)
        current_time = time.time()
        if current_time - self.last_call_time < 30:  # 30 seconds between calls
            time.sleep(30 - (current_time - self.last_call_time))
        
        url = f"{self.base_url}/options/{symbol.upper()}"
        
        try:
            response = self.session.get(url, timeout=15)
            self.last_call_time = time.time()
            self.api_calls_made += 1
            
            if response.status_code != 200:
                return None
            
            # Check for empty or HTML response
            if not response.text.strip() or response.text.startswith('<'):
                return None
            
            data = response.json()
            
            if not isinstance(data, dict) or 'data' not in data:
                return None
            
            options_data = data.get('data', [])
            if not options_data:
                return None
            
            df = pd.DataFrame(options_data)
            df['symbol'] = symbol.upper()
            df['fetch_timestamp'] = datetime.now()
            
            return df
            
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            return None

def get_stock_price(symbol: str) -> float:
    """
    Get current stock price (placeholder - implement with your preferred data source)
    """
    
    # This is a placeholder - replace with actual price feed
    # You could use yfinance, Alpha Vantage, IEX, etc.
    
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        data = ticker.history(period="1d", interval="1m")
        if not data.empty:
            return float(data['Close'].iloc[-1])
    except:
        pass
    
    # Fallback: estimate from options strikes
    return 0.0

if __name__ == "__main__":
    # Example usage
    calculator = GEXCalculator()
    api = TradingVolatilityAPI()
    
    # Test with SPY
    print("Testing GEX Calculator with SPY...")
    options_data = api.fetch_options_data("SPY")
    
    if options_data is not None:
        spot_price = get_stock_price("SPY") or options_data['strike'].median()
        result = calculator.calculate_gamma_exposure(options_data, spot_price, "SPY")
        
        print(f"Net GEX: {result['net_gex']/1e9:.2f}B")
        print(f"Flip Point: {result['gamma_flip_point']:.2f}")
        print(f"Distance to Flip: {result['distance_to_flip']:.2f}%")
        print(f"Regime: {result['regime']}")
    else:
        print("Failed to fetch options data")
