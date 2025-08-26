"""
Data Sources - Handle various options data APIs and sources
"""

import requests
import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import json
import time
import logging

logger = logging.getLogger(__name__)

class DataSourceManager:
    """
    Manage multiple data sources for options and market data
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json, text/plain, */*'
        })
        self.api_calls_made = 0
        self.cache = {}
        
    def get_options_data(self, symbol: str, source: str = 'auto') -> Optional[pd.DataFrame]:
        """
        Get options data from specified source
        
        Args:
            symbol: Stock symbol
            source: 'tradingvolatility', 'yahoo', 'polygon', 'auto'
        """
        
        if source == 'auto':
            # Try sources in order of preference
            for src in ['tradingvolatility', 'yahoo']:
                try:
                    data = self._fetch_from_source(symbol, src)
                    if data is not None and not data.empty:
                        return data
                except Exception as e:
                    logger.warning(f"Failed to get {symbol} from {src}: {e}")
                    continue
            return None
        else:
            return self._fetch_from_source(symbol, source)
    
    def _fetch_from_source(self, symbol: str, source: str) -> Optional[pd.DataFrame]:
        """Fetch from specific source"""
        
        if source == 'tradingvolatility':
            return self._fetch_tradingvolatility(symbol)
        elif source == 'yahoo':
            return self._fetch_yahoo_options(symbol)
        elif source == 'polygon':
            return self._fetch_polygon(symbol)
        else:
            raise ValueError(f"Unknown source: {source}")
    
    def _fetch_tradingvolatility(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch from TradingVolatility API"""
        
        url = f"https://stocks.tradingvolatility.net/api/options/{symbol.upper()}"
        
        try:
            response = self.session.get(url, timeout=15)
            self.api_calls_made += 1
            
            if response.status_code != 200:
                return None
            
            if not response.text.strip():
                return None
                
            if response.text.strip().startswith(('<!DOCTYPE', '<html')):
                return None
            
            data = response.json()
            
            if not isinstance(data, dict) or 'data' not in data:
                return None
            
            options_data = data.get('data', [])
            if not options_data:
                return None
            
            df = pd.DataFrame(options_data)
            df['symbol'] = symbol.upper()
            df['source'] = 'tradingvolatility'
            df['fetch_timestamp'] = datetime.now()
            
            return self._standardize_columns(df)
            
        except Exception as e:
            logger.error(f"TradingVolatility API error for {symbol}: {e}")
            return None
    
    def _fetch_yahoo_options(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch from Yahoo Finance (yfinance)"""
        
        try:
            ticker = yf.Ticker(symbol)
            
            # Get available expiration dates
            expirations = ticker.options
            if not expirations:
                return None
            
            all_options = []
            
            # Get options for next 3 expirations (to limit data)
            for exp_date in expirations[:3]:
                try:
                    option_chain = ticker.option_chain(exp_date)
                    
                    # Calls
                    calls = option_chain.calls.copy()
                    calls['option_type'] = 'C'
                    calls['expiration'] = exp_date
                    
                    # Puts
                    puts = option_chain.puts.copy()
                    puts['option_type'] = 'P' 
                    puts['expiration'] = exp_date
                    
                    all_options.extend([calls, puts])
                    
                except Exception as e:
                    logger.warning(f"Failed to get {symbol} options for {exp_date}: {e}")
                    continue
            
            if not all_options:
                return None
            
            df = pd.concat(all_options, ignore_index=True)
            df['symbol'] = symbol.upper()
            df['source'] = 'yahoo'
            df['fetch_timestamp'] = datetime.now()
            
            return self._standardize_columns(df)
            
        except Exception as e:
            logger.error(f"Yahoo Finance error for {symbol}: {e}")
            return None
    
    def _fetch_polygon(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch from Polygon.io API (requires API key)"""
        
        api_key = self.config.get('polygon_api_key')
        if not api_key:
            logger.warning("Polygon API key not configured")
            return None
        
        # This would be implemented with actual Polygon API calls
        # For now, return None as placeholder
        logger.info("Polygon integration not yet implemented")
        return None
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names across different sources"""
        
        # Column mapping from different sources to standard names
        column_mapping = {
            # TradingVolatility format
            'Strike': 'strike',
            'Type': 'option_type', 
            'OpenInterest': 'open_interest',
            'Volume': 'volume',
            'Bid': 'bid',
            'Ask': 'ask',
            'Last': 'last',
            'IV': 'implied_volatility',
            'Delta': 'delta',
            'Gamma': 'gamma',
            'Theta': 'theta',
            'Vega': 'vega',
            
            # Yahoo Finance format
            'contractSymbol': 'contract_symbol',
            'lastTradeDate': 'last_trade_date',
            'openInterest': 'open_interest',
            'impliedVolatility': 'implied_volatility',
            'contractSize': 'contract_size',
            
            # Standard variations
            'open_int': 'open_interest',
            'iv': 'implied_volatility',
            'oi': 'open_interest'
        }
        
        # Rename columns
        df = df.rename(columns=column_mapping)
        
        # Ensure required columns exist
        required_columns = ['strike', 'option_type', 'open_interest']
        
        for col in required_columns:
            if col not in df.columns:
                if col == 'open_interest':
                    df[col] = 0  # Default to 0 if missing
                else:
                    logger.warning(f"Missing required column: {col}")
        
        # Standardize option_type values
        if 'option_type' in df.columns:
            df['option_type'] = df['option_type'].str.upper().str[0]  # 'C' or 'P'
        
        # Convert data types
        numeric_columns = ['strike', 'open_interest', 'volume', 'bid', 'ask', 'last',
                          'implied_volatility', 'delta', 'gamma', 'theta', 'vega']
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def get_spot_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Get current spot prices for symbols"""
        
        spot_prices = {}
        
        # Try to get from Yahoo Finance
        try:
            tickers = yf.download(symbols, period='1d', progress=False)['Close']
            
            if len(symbols) == 1:
                spot_prices[symbols[0]] = float(tickers.iloc[-1])
            else:
                for symbol in symbols:
                    if symbol in tickers.columns:
                        spot_prices[symbol] = float(tickers[symbol].iloc[-1])
                        
        except Exception as e:
            logger.error(f"Failed to get spot prices: {e}")
        
        return spot_prices
    
    def get_market_data(self, symbol: str) -> Dict:
        """Get comprehensive market data for a symbol"""
        
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period='5d')
            
            if hist.empty:
                return {}
            
            current_price = hist['Close'].iloc[-1]
            volume = hist['Volume'].iloc[-1]
            avg_volume = hist['Volume'].mean()
            
            return {
                'symbol': symbol,
                'price': float(current_price),
                'volume': int(volume),
                'avg_volume': int(avg_volume),
                'volume_ratio': float(volume / avg_volume) if avg_volume > 0 else 1.0,
                'market_cap': info.get('marketCap'),
                'sector': info.get('sector'),
                'industry': info.get('industry'),
                'beta': info.get('beta'),
                'pe_ratio': info.get('trailingPE'),
                'update_time': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Failed to get market data for {symbol}: {e}")
            return {'symbol': symbol, 'error': str(e)}


class MockDataSource:
    """
    Mock data source for testing and development
    """
    
    @staticmethod
    def generate_mock_options_chain(
        symbol: str,
        spot_price: float,
        num_strikes: int = 20
    ) -> pd.DataFrame:
        """Generate realistic mock options data"""
        
        # Generate strikes around current price
        strike_range = np.linspace(
            spot_price * 0.85, 
            spot_price * 1.15, 
            num_strikes
        )
        
        options_data = []
        
        for strike in strike_range:
            # Create calls and puts
            for option_type in ['C', 'P']:
                
                # Mock realistic values
                moneyness = abs(strike - spot_price) / spot_price
                
                # Higher OI for ATM options
                base_oi = max(100, int(5000 * np.exp(-10 * moneyness)))
                open_interest = np.random.poisson(base_oi)
                
                # Simple Black-Scholes approximation for Greeks
                gamma = 0.02 * np.exp(-5 * moneyness)
                
                options_data.append({
                    'symbol': symbol,
                    'strike': round(strike, 2),
                    'option_type': option_type,
                    'open_interest': open_interest,
                    'volume': max(0, np.random.poisson(open_interest * 0.1)),
                    'bid': max(0.01, np.random.uniform(0.5, 5.0)),
                    'ask': max(0.02, np.random.uniform(0.6, 6.0)),
                    'last': max(0.01, np.random.uniform(0.55, 5.5)),
                    'implied_volatility': np.random.uniform(0.15, 0.45),
                    'delta': np.random.uniform(-1, 1),
                    'gamma': gamma,
                    'theta': np.random.uniform(-0.1, 0),
                    'vega': np.random.uniform(0, 0.3),
                    'expiration': '2024-12-20',  # Example date
                    'source': 'mock',
                    'fetch_timestamp': datetime.now()
                })
        
        return pd.DataFrame(options_data)
