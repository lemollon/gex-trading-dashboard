"""
Streamlit Databricks Connector
Add this to your Streamlit app to read from your Databricks table
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import json

# Add this to your Streamlit secrets.toml:
# [databricks]
# hostname = "your-workspace.cloud.databricks.com"
# http_path = "/sql/1.0/warehouses/your-warehouse-id"
# token = "your-access-token"

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_gex_data_from_databricks():
    """
    Load GEX pipeline results from Databricks table
    """
    
    try:
        # METHOD 1: Using databricks-sql-connector (install: pip install databricks-sql-connector)
        from databricks import sql
        
        connection = sql.connect(
            server_hostname=st.secrets["databricks"]["server_hostname"],  # Changed from hostname
            http_path=st.secrets["databricks"]["http_path"], 
            access_token=st.secrets["databricks"]["access_token"]          # Changed from token
        )
        
        cursor = connection.cursor()
        
        # Get latest pipeline run
        query = """
        SELECT *
        FROM quant_projects.gex_trading.gex_pipeline_results
        WHERE pipeline_date >= current_date() - INTERVAL 7 DAYS
        ORDER BY run_timestamp DESC
        LIMIT 100
        """
        
        cursor.execute(query)
        results = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        
        cursor.close()
        connection.close()
        
        if not results:
            return None, "No recent data found"
        
        # Convert to DataFrame
        df = pd.DataFrame(results, columns=columns)
        
        # Get latest run data
        latest_run_id = df['run_id'].iloc[0]
        latest_data = df[df['run_id'] == latest_run_id]
        
        # Convert to dashboard format
        dashboard_data = convert_table_data_to_dashboard_format(latest_data)
        
        return dashboard_data, "Connected to Databricks"
        
    except ImportError:
        st.error("Install databricks-sql-connector: pip install databricks-sql-connector")
        return None, "Missing connector"
        
    except Exception as e:
        st.error(f"Databricks connection failed: {e}")
        return create_fallback_data(), "Using fallback data"

def convert_table_data_to_dashboard_format(df):
    """Convert Databricks table data to dashboard format"""
    
    if df.empty:
        return create_empty_dashboard_data()
    
    # Get metadata from first row
    first_row = df.iloc[0]
    
    # Filter approved setups
    approved_setups = df[df['setup_approved'] == True]
    
    # Create trading setups
    trading_setups = []
    for _, row in approved_setups.iterrows():
        if row['symbol'] != 'NO_SETUPS':
            
            # Map condition types
            setup_type_mapping = {
                'NEGATIVE_GEX': 'SQUEEZE_PLAY',
                'HIGH_POSITIVE_GEX': 'PREMIUM_SELLING', 
                'NEAR_FLIP': 'GAMMA_FLIP'
            }
            
            direction_mapping = {
                'NEGATIVE_GEX': 'LONG_CALLS',
                'HIGH_POSITIVE_GEX': 'SHORT_CALLS',
                'NEAR_FLIP': 'VOLATILITY'
            }
            
            setup_data = {
                'setup': {
                    'symbol': row['symbol'],
                    'setup_type': setup_type_mapping.get(row['condition_type'], row['condition_type']),
                    'direction': direction_mapping.get(row['condition_type'], 'DIRECTIONAL'),
                    'confidence': row['confidence_score'],
                    'reason': f"GEX: {row['net_gex']/1e9:+.1f}B, Distance: {row['distance_to_flip']:+.2f}% from flip",
                    'expected_move': row['expected_move'],
                    'hold_days': 3,
                    'risk_level': 'MEDIUM' if row['confidence_score'] >= 80 else 'HIGH'
                },
                'position_size_percent': row['position_size_percent'],
                'dollar_amount': int(row['position_size_percent'] * 1000),
                'approved': row['setup_approved']
            }
            
            trading_setups.append(setup_data)
    
    # Calculate market summary
    net_gex_values = df[df['symbol'] != 'NO_SETUPS']['net_gex'].tolist()
    total_net_gex = sum(net_gex_values) / 1e9 if net_gex_values else 0
    
    # Count conditions
    condition_counts = df['condition_type'].value_counts()
    negative_count = condition_counts.get('NEGATIVE_GEX', 0)
    positive_count = condition_counts.get('HIGH_POSITIVE_GEX', 0)
    near_flip_count = condition_counts.get('NEAR_FLIP', 0)
    
    # Determine dominant regime
    if negative_count > positive_count and negative_count > 0:
        dominant_regime = 'NEGATIVE_GEX'
        stress_level = 'HIGH'
    elif positive_count > 0:
        dominant_regime = 'HIGH_POSITIVE_GEX'
        stress_level = 'LOW'
    elif near_flip_count > 0:
        dominant_regime = 'NEAR_FLIP'
        stress_level = 'MEDIUM'
    else:
        dominant_regime = 'NEUTRAL'
        stress_level = 'LOW'
    
    return {
        'success': True,
        'analysis_time': pd.to_datetime(first_row['run_timestamp']),
        'pipeline_run_time': first_row['run_timestamp'],
        'symbols_analyzed': first_row['total_symbols_analyzed'],
        'symbols_successful': first_row['total_symbols_analyzed'],
        'trading_setups': trading_setups,
        'market_summary': {
            'total_net_gex_billions': total_net_gex,
            'dominant_regime': dominant_regime,
            'symbols_near_flip': near_flip_count,
            'market_stress_level': stress_level,
            'total_conditions_found': len(df[df['symbol'] != 'NO_SETUPS'])
        },
        'risk_assessment': {
            'total_risk_percent': sum(s['position_size_percent'] for s in trading_setups),
            'num_positions': len(trading_setups),
            'risk_level': 'MEDIUM' if len(trading_setups) > 2 else 'LOW'
        }
    }

def create_empty_dashboard_data():
    """Create empty dashboard data structure"""
    return {
        'success': True,
        'analysis_time': datetime.now(),
        'symbols_analyzed': 0,
        'trading_setups': [],
        'market_summary': {
            'total_net_gex_billions': 0,
            'dominant_regime': 'NEUTRAL',
            'symbols_near_flip': 0,
            'market_stress_level': 'LOW'
        },
        'risk_assessment': {
            'total_risk_percent': 0,
            'num_positions': 0,
            'risk_level': 'NONE'
        }
    }

def create_fallback_data():
    """Create fallback data if connection fails"""
    return {
        'success': False,
        'error': 'Connection failed',
        'analysis_time': datetime.now(),
        'symbols_analyzed': 0,
        'trading_setups': [],
        'market_summary': {
            'total_net_gex_billions': 0,
            'dominant_regime': 'UNKNOWN',
            'symbols_near_flip': 0,
            'market_stress_level': 'UNKNOWN'
        },
        'risk_assessment': {
            'total_risk_percent': 0,
            'num_positions': 0,
            'risk_level': 'UNKNOWN'
        }
    }

# Add this to your main dashboard file
def get_live_databricks_data():
    """Main function to call from your dashboard"""
    return load_gex_data_from_databricks()
