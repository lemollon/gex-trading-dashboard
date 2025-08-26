"""
GEX Calculator - Core gamma exposure calculations
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class GEXCalculator:
    """
    Core GEX calculation engine for options market analysis
    """
    
    def __init__(self):
        self.multiplier = 100  # Standard options multiplier
        
    def calculate_gamma_exposure(
        self,
        options_df: pd.DataFrame,
        spot_price: float,
        symbol: str
    ) -> Dict:
        """
        Calculate comprehensive gamma exposure metrics
        
        Args:
            options_df: DataFrame with options data
            spot_price: Current underlying price
            symbol: Symbol being analyzed
            
        Returns:
            Dictionary with GEX analysis results
        """
        
        try:
            if options_df.empty:
                return self._empty_result(symbol, "Empty options data")
            
            # Clean and validate data
            df = self._clean_options_data(options_df.copy())
            
            if df.empty:
                return self._empty_result(symbol, "No valid data after cleaning")
            
            # Calculate individual option GEX
            df['gex'] = df.apply(
                lambda row: self._calculate_option_gex(row, spot_price), 
                axis=1
            )
            
            # Aggregate by strike
            gex_by_strike = df.groupby('strike')['gex'].sum().sort_index()
            
            # Calculate key metrics
            results = {
                'success': True,
                'symbol': symbol,
                'spot_price': spot_price,
                'calculation_time': datetime.now(),
                
                # Core metrics
                'net_gex': gex_by_strike.sum(),
                'total_call_gex': df[df['option_type'].str.upper().str.startswith('C')]['gex'].sum(),
                'total_put_gex': df[df['option_type'].str.upper().str.startswith('P')]['gex'].sum(),
                
                # Gamma flip point
                'gamma_flip_point': self._find_gamma_flip_point(gex_by_strike),
                
                # Wall analysis
                'call_walls': self._find_call_walls(gex_by_strike),
                'put_walls': self._find_put_walls(gex_by_strike),
                
                # Strike-level data
                'gex_by_strike': gex_by_strike.to_dict(),
                'strike_count': len(gex_by_strike),
                'price_range': {
                    'min_strike': float(gex_by_strike.index.min()),
                    'max_strike': float(gex_by_strike.index.max())
                }
            }
            
            # Calculate distance to flip
            results['distance_to_flip'] = self._calculate_distance_to_flip(
                spot_price, results['gamma_flip_point']
            )
            
            # Market regime classification
            results['market_regime'] = self._classify_market_regime(
                results['net_gex'], results['distance_to_flip']
            )
            
            # Trading setup identification
            results['setups'] = self._identify_setups(results, df)
            
            return results
            
        except Exception as e:
            logger.error(f"GEX calculation failed for {symbol}: {e}")
            return self._empty_result(symbol, str(e))
    
    def _clean_options_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate options data"""
        
        # Required columns
        required_cols = ['strike', 'option_type', 'open_interest']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Convert to numeric
        df['strike'] = pd.to_numeric(df['strike'], errors='coerce')
        df['open_interest'] = pd.to_numeric(df['open_interest'], errors='coerce')
        
        # Handle gamma - use if available, otherwise approximate
        if 'gamma' in df.columns:
            df['gamma'] = pd.to_numeric(df['gamma'], errors='coerce').fillna(0)
        else:
            # Simple gamma approximation for options
            df['gamma'] = 0.01  # Will be adjusted based on moneyness
        
        # Remove invalid rows
        df = df.dropna(subset=['strike', 'open_interest'])
        df = df[df['open_interest'] > 0]  # Only include options with OI
        
        return df
    
    def _calculate_option_gex(self, row: pd.Series, spot_price: float) -> float:
        """Calculate GEX for individual option"""
        
        # Adjust gamma based on moneyness if using approximation
        if row['gamma'] == 0.01:  # Using approximation
            # Higher gamma for ATM options, lower for OTM
            moneyness = abs(row['strike'] - spot_price) / spot_price
            gamma = 0.01 * np.exp(-2 * moneyness)  # Simplified model
        else:
            gamma = row['gamma']
        
        # Base GEX calculation
        base_gex = spot_price * gamma * row['open_interest'] * self.multiplier
        
        # Calls are positive, puts are negative
        if row['option_type'].upper().startswith('C'):
            return base_gex
        else:
            return -base_gex
    
    def _find_gamma_flip_point(self, gex_by_strike: pd.Series) -> float:
        """Find the gamma flip point where cumulative GEX crosses zero"""
        
        cumulative_gex = gex_by_strike.cumsum()
        
        # Find first point where cumulative GEX >= 0
        positive_points = cumulative_gex[cumulative_gex >= 0]
        
        if not positive_points.empty:
            return float(positive_points.index[0])
        
        # If no positive cumulative GEX, return highest strike
        return float(gex_by_strike.index.max())
    
    def _find_call_walls(self, gex_by_strike: pd.Series, top_n: int = 3) -> List[Dict]:
        """Find strongest call walls (positive GEX concentrations)"""
        
        call_gex = gex_by_strike[gex_by_strike > 0].sort_values(ascending=False)
        
        walls = []
        for strike, gex in call_gex.head(top_n).items():
            walls.append({
                'strike': float(strike),
                'gex': float(gex),
                'strength': 'Strong' if gex > call_gex.quantile(0.8) else 'Moderate'
            })
        
        return walls
    
    def _find_put_walls(self, gex_by_strike: pd.Series, top_n: int = 3) -> List[Dict]:
        """Find strongest put walls (negative GEX concentrations)"""
        
        put_gex = gex_by_strike[gex_by_strike < 0].sort_values()
        
        walls = []
        for strike, gex in put_gex.head(top_n).items():
            walls.append({
                'strike': float(strike),
                'gex': float(gex),
                'strength': 'Strong' if abs(gex) > put_gex.abs().quantile(0.8) else 'Moderate'
            })
        
        return walls
    
    def _calculate_distance_to_flip(self, spot_price: float, flip_point: float) -> float:
        """Calculate percentage distance to gamma flip point"""
        
        if flip_point == 0:
            return 0.0
            
        return ((spot_price - flip_point) / flip_point) * 100
    
    def _classify_market_regime(self, net_gex: float, distance_to_flip: float) -> str:
        """Classify market regime based on GEX"""
        
        # Convert to billions for thresholds
        net_gex_b = net_gex / 1e9
        
        if net_gex_b < -1.0:
            return "NEGATIVE_GEX"
        elif net_gex_b > 2.0:
            return "HIGH_POSITIVE_GEX" 
        elif abs(distance_to_flip) < 0.5:
            return "NEAR_FLIP"
        elif net_gex_b > 0:
            return "POSITIVE_GEX"
        else:
            return "NEUTRAL"
    
    def _identify_setups(self, gex_results: Dict, options_df: pd.DataFrame) -> List[Dict]:
        """Identify potential trading setups based on GEX"""
        
        setups = []
        regime = gex_results['market_regime']
        net_gex = gex_results['net_gex']
        distance = gex_results['distance_to_flip']
        
        # Negative GEX squeeze setups
        if regime == "NEGATIVE_GEX":
            setups.append({
                'type': 'SQUEEZE_PLAY',
                'direction': 'LONG_CALLS',
                'confidence': min(95, 60 + abs(net_gex / 1e8)),
                'reason': f'Strong negative GEX ({net_gex/1e9:.1f}B) suggests volatility expansion',
                'target': 'ATM calls 2-5 DTE',
                'risk': 'High - can lose 100% if no move'
            })
        
        # High positive GEX resistance
        elif regime == "HIGH_POSITIVE_GEX":
            if gex_results['call_walls']:
                strongest_wall = max(gex_results['call_walls'], key=lambda x: x['gex'])
                setups.append({
                    'type': 'RESISTANCE_PLAY',
                    'direction': 'SHORT_CALLS',
                    'confidence': min(90, 50 + (net_gex / 1e8) / 10),
                    'reason': f'High positive GEX with strong call wall at {strongest_wall["strike"]}',
                    'target': f'Sell calls at/above {strongest_wall["strike"]}',
                    'risk': 'Moderate - defined by call wall'
                })
        
        # Near flip volatility
        elif regime == "NEAR_FLIP":
            setups.append({
                'type': 'VOLATILITY_PLAY',
                'direction': 'STRADDLE',
                'confidence': 70 + (1 - abs(distance)) * 20,
                'reason': f'Price very close to gamma flip ({distance:.2f}%)',
                'target': 'ATM straddle or iron butterfly',
                'risk': 'Moderate - volatility dependent'
            })
        
        return setups
    
    def _empty_result(self, symbol: str, error: str) -> Dict:
        """Return empty result structure for failed calculations"""
        
        return {
            'success': False,
            'symbol': symbol,
            'error': error,
            'calculation_time': datetime.now(),
            'net_gex': 0,
            'gamma_flip_point': 0,
            'distance_to_flip': 0,
            'market_regime': 'UNKNOWN',
            'call_walls': [],
            'put_walls': [],
            'setups': []
        }

    def calculate_portfolio_gex(self, symbols_data: Dict[str, pd.DataFrame], spot_prices: Dict[str, float]) -> Dict:
        """Calculate GEX for multiple symbols"""
        
        results = {}
        summary = {
            'total_symbols': len(symbols_data),
            'successful': 0,
            'failed': 0,
            'net_portfolio_gex': 0,
            'regime_distribution': {}
        }
        
        for symbol, options_df in symbols_data.items():
            spot_price = spot_prices.get(symbol, options_df['strike'].median())
            result = self.calculate_gamma_exposure(options_df, spot_price, symbol)
            
            results[symbol] = result
            
            if result['success']:
                summary['successful'] += 1
                summary['net_portfolio_gex'] += result['net_gex']
                
                regime = result['market_regime']
                summary['regime_distribution'][regime] = summary['regime_distribution'].get(regime, 0) + 1
            else:
                summary['failed'] += 1
        
        return {
            'results': results,
            'summary': summary,
            'calculation_time': datetime.now()
        }
