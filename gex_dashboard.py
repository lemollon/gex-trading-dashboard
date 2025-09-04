import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class GammaExposureEngine:
    """
    Core engine for calculating and analyzing Gamma Exposure (GEX) for options trading
    """
    
    def __init__(self, symbol='SPY'):
        self.symbol = symbol
        self.options_data = None
        self.gex_profile = None
        self.current_price = None
        self.gamma_flip_point = None
        
    def fetch_options_data(self, max_dte=45):
        """
        Fetch options chain data using yfinance
        """
        try:
            ticker = yf.Ticker(self.symbol)
            self.current_price = ticker.history(period="1d")['Close'].iloc[-1]
            
            # Get available expiration dates
            exp_dates = ticker.options[:8]  # Limit to first 8 expirations
            
            all_options = []
            
            for exp_date in exp_dates:
                try:
                    # Calculate days to expiration
                    exp_dt = datetime.strptime(exp_date, '%Y-%m-%d')
                    dte = (exp_dt - datetime.now()).days
                    
                    if dte > max_dte:
                        continue
                        
                    # Get calls and puts
                    calls = ticker.option_chain(exp_date).calls
                    puts = ticker.option_chain(exp_date).puts
                    
                    # Add metadata
                    calls['option_type'] = 'call'
                    calls['expiration'] = exp_date
                    calls['dte'] = dte
                    
                    puts['option_type'] = 'put'
                    puts['expiration'] = exp_date
                    puts['dte'] = dte
                    
                    all_options.append(calls)
                    all_options.append(puts)
                    
                except Exception as e:
                    print(f"Error processing expiration {exp_date}: {e}")
                    continue
            
            if all_options:
                self.options_data = pd.concat(all_options, ignore_index=True)
                self._clean_options_data()
                print(f"Successfully loaded {len(self.options_data)} options contracts")
            else:
                raise Exception("No options data retrieved")
                
        except Exception as e:
            print(f"Error fetching options data: {e}")
            raise
    
    def _clean_options_data(self):
        """
        Clean and prepare options data for GEX calculations
        """
        # Remove options with zero open interest or volume
        self.options_data = self.options_data[
            (self.options_data['openInterest'] > 0) | 
            (self.options_data['volume'] > 0)
        ].copy()
        
        # Calculate mid price
        self.options_data['mid_price'] = (
            self.options_data['bid'] + self.options_data['ask']
        ) / 2
        
        # Filter out options too far OTM (reduce noise)
        price_filter = (
            (self.options_data['strike'] >= self.current_price * 0.7) &
            (self.options_data['strike'] <= self.current_price * 1.3)
        )
        self.options_data = self.options_data[price_filter].copy()
    
    def calculate_black_scholes_gamma(self, S, K, T, r=0.05, sigma=0.2):
        """
        Calculate Black-Scholes gamma for options
        """
        if T <= 0:
            return 0
            
        from scipy.stats import norm
        import math
        
        d1 = (math.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
        gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))
        
        return gamma
    
    def calculate_gex_profile(self):
        """
        Calculate Gamma Exposure (GEX) profile across all strikes
        """
        if self.options_data is None:
            raise Exception("No options data available. Run fetch_options_data() first.")
        
        gex_data = []
        
        for _, row in self.options_data.iterrows():
            try:
                # Calculate gamma using Black-Scholes approximation
                # Using implied volatility if available, otherwise default
                iv = row.get('impliedVolatility', 0.2)
                if pd.isna(iv) or iv <= 0:
                    iv = 0.2  # Default volatility
                
                gamma = self.calculate_black_scholes_gamma(
                    S=self.current_price,
                    K=row['strike'],
                    T=row['dte'] / 365.0,  # Convert days to years
                    sigma=iv
                )
                
                # Calculate GEX = Spot * Gamma * Open Interest * 100 (multiplier)
                spot_gamma_exposure = (
                    self.current_price * gamma * row['openInterest'] * 100
                )
                
                # Calls contribute positive GEX, puts contribute negative GEX
                if row['option_type'] == 'call':
                    gex = spot_gamma_exposure
                else:  # put
                    gex = -spot_gamma_exposure
                
                gex_data.append({
                    'strike': row['strike'],
                    'gex': gex,
                    'gamma': gamma,
                    'open_interest': row['openInterest'],
                    'option_type': row['option_type'],
                    'expiration': row['expiration'],
                    'dte': row['dte']
                })
                
            except Exception as e:
                print(f"Error calculating GEX for strike {row['strike']}: {e}")
                continue
        
        # Convert to DataFrame and aggregate by strike
        gex_df = pd.DataFrame(gex_data)
        
        if len(gex_df) == 0:
            raise Exception("No valid GEX calculations completed")
        
        # Aggregate GEX by strike (sum across all expirations)
        self.gex_profile = gex_df.groupby('strike').agg({
            'gex': 'sum',
            'gamma': 'sum',
            'open_interest': 'sum'
        }).reset_index()
        
        # Sort by strike
        self.gex_profile = self.gex_profile.sort_values('strike').reset_index(drop=True)
        
        # Calculate cumulative GEX to find gamma flip point
        self.gex_profile['cumulative_gex'] = self.gex_profile['gex'].cumsum()
        
        self._find_gamma_flip_point()
        self._identify_walls()
        
        print(f"GEX profile calculated for {len(self.gex_profile)} strikes")
    
    def _find_gamma_flip_point(self):
        """
        Find the gamma flip point where cumulative GEX crosses zero
        """
        # Find where cumulative GEX crosses zero
        cumulative = self.gex_profile['cumulative_gex'].values
        strikes = self.gex_profile['strike'].values
        
        # Find the strike where cumulative GEX is closest to zero
        zero_cross_idx = np.argmin(np.abs(cumulative))
        self.gamma_flip_point = strikes[zero_cross_idx]
        
        print(f"Gamma Flip Point: ${self.gamma_flip_point:.2f}")
        print(f"Current Price: ${self.current_price:.2f}")
        print(f"Distance to Flip: {((self.current_price - self.gamma_flip_point) / self.current_price) * 100:.2f}%")
    
    def _identify_walls(self):
        """
        Identify call walls (resistance) and put walls (support)
        """
        # Call walls - highest positive GEX strikes
        call_gex = self.gex_profile[self.gex_profile['gex'] > 0].copy()
        call_gex = call_gex.nlargest(3, 'gex')
        
        # Put walls - highest absolute negative GEX strikes  
        put_gex = self.gex_profile[self.gex_profile['gex'] < 0].copy()
        put_gex = put_gex.nsmallest(3, 'gex')
        
        print("\n=== CALL WALLS (Resistance) ===")
        for _, wall in call_gex.iterrows():
            print(f"${wall['strike']:.2f}: {wall['gex']/1e6:.1f}M GEX")
        
        print("\n=== PUT WALLS (Support) ===")
        for _, wall in put_gex.iterrows():
            print(f"${wall['strike']:.2f}: {wall['gex']/1e6:.1f}M GEX")
    
    def get_net_gex(self):
        """
        Calculate total net GEX across all strikes
        """
        if self.gex_profile is None:
            return None
        
        net_gex = self.gex_profile['gex'].sum()
        return net_gex
    
    def analyze_setup_conditions(self):
        """
        Analyze current market conditions for trading setups
        """
        if self.gex_profile is None:
            print("No GEX profile available")
            return
        
        net_gex = self.get_net_gex()
        distance_to_flip = ((self.current_price - self.gamma_flip_point) / self.current_price) * 100
        
        print(f"\n=== MARKET ANALYSIS ===")
        print(f"Net GEX: {net_gex/1e9:.2f}B")
        print(f"Current Price: ${self.current_price:.2f}")
        print(f"Gamma Flip Point: ${self.gamma_flip_point:.2f}")
        print(f"Distance to Flip: {distance_to_flip:.2f}%")
        
        # Determine market regime
        if net_gex > 1e9:  # > 1B
            regime = "POSITIVE GEX (Volatility Suppression)"
        elif net_gex < -1e9:  # < -1B
            regime = "NEGATIVE GEX (Volatility Amplification)"
        else:
            regime = "NEUTRAL GEX"
        
        print(f"Market Regime: {regime}")
        
        # Setup recommendations
        self._generate_setup_recommendations(net_gex, distance_to_flip)
    
    def _generate_setup_recommendations(self, net_gex, distance_to_flip):
        """
        Generate trading setup recommendations based on GEX analysis
        """
        recommendations = []
        
        # Negative GEX Squeeze (Long Calls)
        if net_gex < -1e9 and distance_to_flip < -0.5:
            recommendations.append({
                'strategy': 'NEGATIVE GEX SQUEEZE - Long Calls',
                'confidence': 'HIGH',
                'rationale': 'Net GEX < -1B, price below flip point',
                'target': f'ATM calls at ${self.current_price:.0f} strike',
                'risk': 'High - size for 100% loss potential'
            })
        
        # Positive GEX Breakdown (Long Puts)
        elif net_gex > 2e9 and abs(distance_to_flip) < 0.3:
            recommendations.append({
                'strategy': 'POSITIVE GEX BREAKDOWN - Long Puts',
                'confidence': 'HIGH',
                'rationale': 'Net GEX > 2B, price near flip point',
                'target': f'ATM puts at ${self.current_price:.0f} strike',
                'risk': 'Medium - set stops above call walls'
            })
        
        # Iron Condor Setup
        elif net_gex > 1e9:
            recommendations.append({
                'strategy': 'IRON CONDOR',
                'confidence': 'MEDIUM',
                'rationale': 'Positive GEX environment supports range trading',
                'target': 'Short strangles at identified walls',
                'risk': 'Monitor wall integrity daily'
            })
        
        print(f"\n=== SETUP RECOMMENDATIONS ===")
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                print(f"\n{i}. {rec['strategy']}")
                print(f"   Confidence: {rec['confidence']}")
                print(f"   Rationale: {rec['rationale']}")
                print(f"   Target: {rec['target']}")
                print(f"   Risk: {rec['risk']}")
        else:
            print("No high-confidence setups identified")
    
    def get_summary_stats(self):
        """
        Get summary statistics for the current GEX profile
        """
        if self.gex_profile is None:
            return None
        
        net_gex = self.get_net_gex()
        max_call_gex = self.gex_profile['gex'].max()
        min_put_gex = self.gex_profile['gex'].min()
        
        stats = {
            'symbol': self.symbol,
            'current_price': self.current_price,
            'gamma_flip_point': self.gamma_flip_point,
            'net_gex': net_gex,
            'net_gex_billions': net_gex / 1e9,
            'distance_to_flip_pct': ((self.current_price - self.gamma_flip_point) / self.current_price) * 100,
            'max_call_wall_gex': max_call_gex,
            'max_put_wall_gex': abs(min_put_gex),
            'total_strikes': len(self.gex_profile)
        }
        
        return stats


# Example usage and testing
if __name__ == "__main__":
    # Initialize GEX engine
    gex_engine = GammaExposureEngine('SPY')
    
    try:
        print("Fetching options data...")
        gex_engine.fetch_options_data()
        
        print("Calculating GEX profile...")
        gex_engine.calculate_gex_profile()
        
        print("Analyzing market conditions...")
        gex_engine.analyze_setup_conditions()
        
        # Get summary stats
        stats = gex_engine.get_summary_stats()
        print(f"\n=== SUMMARY STATS ===")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"{key}: {value:.2f}")
            else:
                print(f"{key}: {value}")
                
    except Exception as e:
        print(f"Error in main execution: {e}")
        print("This might be due to:")
        print("1. Network connectivity issues")
        print("2. Yahoo Finance API limitations") 
        print("3. Missing dependencies (yfinance, scipy)")
        print("\nTo install dependencies: pip install yfinance scipy pandas numpy")
