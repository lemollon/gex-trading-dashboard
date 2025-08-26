"""
GEX Trading Dashboard - Main Module
Fixed version with proper imports
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Set  # <-- THIS IS THE FIX
import logging
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataValidator:
    """Validate and clean data"""
    
    @staticmethod
    def validate_options_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate options data structure and content"""
        errors = []
        
        if df.empty:
            errors.append("DataFrame is empty")
            return False, errors
        
        # Required columns
        required_cols = ['symbol', 'strike', 'option_type', 'open_interest']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
        
        # Data type validation
        if 'strike' in df.columns:
            non_numeric_strikes = df[pd.to_numeric(df['strike'], errors='coerce').isna()]
            if not non_numeric_strikes.empty:
                errors.append(f"Non-numeric strikes found: {len(non_numeric_strikes)} rows")
        
        if 'open_interest' in df.columns:
            non_numeric_oi = df[pd.to_numeric(df['open_interest'], errors='coerce').isna()]
            if not non_numeric_oi.empty:
                errors.append(f"Non-numeric open interest found: {len(non_numeric_oi)} rows")
        
        # Option type validation
        if 'option_type' in df.columns:
            valid_types = df['option_type'].str.upper().str[0].isin(['C', 'P'])
            if not valid_types.all():
                invalid_count = (~valid_types).sum()
                errors.append(f"Invalid option types found: {invalid_count} rows")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def clean_options_data(df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize options data"""
        if df.empty:
            return df
        
        df_clean = df.copy()
        
        # Convert numeric columns
        numeric_cols = ['strike', 'open_interest', 'volume', 'bid', 'ask', 'last',
                       'implied_volatility', 'delta', 'gamma', 'theta', 'vega']
        
        for col in numeric_cols:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        # Standardize option types
        if 'option_type' in df_clean.columns:
            df_clean['option_type'] = df_clean['option_type'].str.upper().str[0]
            df_clean = df_clean[df_clean['option_type'].isin(['C', 'P'])]
        
        # Remove invalid strikes
        if 'strike' in df_clean.columns:
            df_clean = df_clean.dropna(subset=['strike'])
            df_clean = df_clean[df_clean['strike'] > 0]
        
        # Remove invalid open interest
        if 'open_interest' in df_clean.columns:
            df_clean = df_clean.dropna(subset=['open_interest'])
            df_clean = df_clean[df_clean['open_interest'] >= 0]
        
        return df_clean


class GEXCalculator:
    """Core GEX calculation engine"""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.spot_price = None
        self.options_data = None
        self.gex_profile = None
        
    def calculate_gex(self, options_df: pd.DataFrame, spot_price: float) -> Dict:
        """Calculate complete GEX profile"""
        self.spot_price = spot_price
        self.options_data = options_df
        
        # Validate data
        is_valid, errors = DataValidator.validate_options_data(options_df)
        if not is_valid:
            logger.error(f"Data validation failed for {self.symbol}: {errors}")
            return {"error": "Data validation failed", "details": errors}
        
        # Clean data
        clean_df = DataValidator.clean_options_data(options_df)
        
        if clean_df.empty:
            return {"error": "No valid data after cleaning"}
        
        # Calculate GEX for each option
        clean_df['gex'] = self._calculate_option_gex(clean_df, spot_price)
        
        # Aggregate by strike
        gex_by_strike = clean_df.groupby('strike')['gex'].sum().reset_index()
        gex_by_strike = gex_by_strike.sort_values('strike')
        
        # Calculate cumulative GEX
        gex_by_strike['cumulative_gex'] = gex_by_strike['gex'].cumsum()
        
        # Find gamma flip point
        gamma_flip = self._find_gamma_flip(gex_by_strike)
        
        # Find walls
        call_walls = self._find_call_walls(gex_by_strike)
        put_walls = self._find_put_walls(gex_by_strike)
        
        # Calculate net GEX
        net_gex = gex_by_strike['gex'].sum()
        
        profile = {
            'symbol': self.symbol,
            'spot_price': spot_price,
            'net_gex': net_gex,
            'gamma_flip': gamma_flip,
            'call_walls': call_walls,
            'put_walls': put_walls,
            'gex_by_strike': gex_by_strike.to_dict('records'),
            'regime': 'positive' if net_gex > 0 else 'negative',
            'calculation_time': datetime.now().isoformat()
        }
        
        self.gex_profile = profile
        return profile
    
    def _calculate_option_gex(self, df: pd.DataFrame, spot_price: float) -> pd.Series:
        """Calculate GEX for each option"""
        # Estimate gamma if not provided
        if 'gamma' not in df.columns:
            df['gamma'] = self._estimate_gamma(df, spot_price)
        
        # Calculate GEX
        gex = spot_price * df['gamma'] * df['open_interest'] * 100
        
        # Calls contribute positive GEX, puts contribute negative GEX
        gex = np.where(df['option_type'] == 'C', gex, -gex)
        
        return gex
    
    def _estimate_gamma(self, df: pd.DataFrame, spot_price: float) -> pd.Series:
        """Estimate gamma using Black-Scholes approximation"""
        # Simple gamma estimation for demonstration
        # In production, use proper BS formula
        
        strikes = df['strike']
        time_to_expiry = 0.1  # Assume ~36 days average
        volatility = 0.25     # Assume 25% IV
        
        # Distance from spot
        moneyness = strikes / spot_price
        
        # Simple gamma approximation (peaks at ATM)
        gamma = np.exp(-0.5 * ((moneyness - 1) / 0.1) ** 2) * 0.01
        
        return gamma
    
    def _find_gamma_flip(self, gex_df: pd.DataFrame) -> float:
        """Find the gamma flip point where cumulative GEX crosses zero"""
        cumulative = gex_df['cumulative_gex'].values
        strikes = gex_df['strike'].values
        
        # Find zero crossing
        sign_changes = np.where(np.diff(np.sign(cumulative)))[0]
        
        if len(sign_changes) == 0:
            # No crossing, return spot price as best estimate
            return self.spot_price if self.spot_price else strikes[len(strikes)//2]
        
        # Use first sign change (from negative to positive typically)
        idx = sign_changes[0]
        
        # Linear interpolation for more precise flip point
        x1, x2 = strikes[idx], strikes[idx + 1]
        y1, y2 = cumulative[idx], cumulative[idx + 1]
        
        if y2 != y1:
            flip_point = x1 - y1 * (x2 - x1) / (y2 - y1)
        else:
            flip_point = x1
        
        return flip_point
    
    def _find_call_walls(self, gex_df: pd.DataFrame, n_walls: int = 3) -> List[Dict]:
        """Find top call walls (positive GEX)"""
        positive_gex = gex_df[gex_df['gex'] > 0].copy()
        if positive_gex.empty:
            return []
        
        walls = positive_gex.nlargest(n_walls, 'gex')
        
        return [
            {
                'strike': row['strike'],
                'gex': row['gex'],
                'strength': 'strong' if row['gex'] > positive_gex['gex'].quantile(0.8) else 'moderate'
            }
            for _, row in walls.iterrows()
        ]
    
    def _find_put_walls(self, gex_df: pd.DataFrame, n_walls: int = 3) -> List[Dict]:
        """Find top put walls (negative GEX)"""
        negative_gex = gex_df[gex_df['gex'] < 0].copy()
        if negative_gex.empty:
            return []
        
        # For puts, we want the most negative (strongest support)
        walls = negative_gex.nsmallest(n_walls, 'gex')
        
        return [
            {
                'strike': row['strike'],
                'gex': row['gex'],
                'strength': 'strong' if row['gex'] < negative_gex['gex'].quantile(0.2) else 'moderate'
            }
            for _, row in walls.iterrows()
        ]


# Example usage and testing
if __name__ == "__main__":
    # Test data
    test_data = pd.DataFrame({
        'symbol': ['SPY'] * 10,
        'strike': [440, 445, 450, 455, 460, 465, 470, 475, 480, 485],
        'option_type': ['P', 'P', 'P', 'P', 'P', 'C', 'C', 'C', 'C', 'C'],
        'open_interest': [1000, 1500, 2000, 1800, 1200, 1500, 2200, 1800, 1000, 500]
    })
    
    # Test validation
    is_valid, errors = DataValidator.validate_options_data(test_data)
    print(f"Validation result: {is_valid}, Errors: {errors}")
    
    # Test GEX calculation
    calculator = GEXCalculator('SPY')
    profile = calculator.calculate_gex(test_data, 460.0)
    
    print(f"Net GEX: {profile.get('net_gex', 'N/A')}")
    print(f"Gamma Flip: {profile.get('gamma_flip', 'N/A')}")
    print(f"Call Walls: {len(profile.get('call_walls', []))}")
    print(f"Put Walls: {len(profile.get('put_walls', []))}")
