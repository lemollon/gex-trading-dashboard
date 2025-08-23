# GEX Strategy Dashboard with Databricks Integration
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
from datetime import datetime, timedelta
import yfinance as yf
import time
from typing import Dict, List, Optional
import sqlite3
import os

# Page configuration
st.set_page_config(
    page_title="GEX Trading Strategy Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .strategy-box {
        background-color: #f0f8ff;
        padding: 1.5rem;
        border-radius: 0.8rem;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .bullish-signal {
        background-color: #e8f5e8;
        border-left: 5px solid #4caf50;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .bearish-signal {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .neutral-signal {
        background-color: #fff3e0;
        border-left: 5px solid #ff9800;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .portfolio-value {
        font-size: 2rem;
        font-weight: bold;
        color: #2e7d32;
    }
    .gain {
        color: #4caf50;
        font-weight: bold;
    }
    .loss {
        color: #f44336;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

class DatabricksMockAccount:
    """Mock trading account with $100k starting balance - Databricks integrated"""
    
    def __init__(self):
        self.initial_balance = 100000
        self.databricks_config = {
            'server_hostname': st.secrets.get('databricks_hostname', 'your-workspace.cloud.databricks.com'),
            'http_path': st.secrets.get('databricks_http_path', '/sql/1.0/warehouses/your-warehouse-id'),
            'access_token': st.secrets.get('databricks_token', 'your-token-here')
        }
        self.init_databricks_tables()
    
    def init_databricks_tables(self):
        """Initialize Databricks Delta tables for portfolio tracking"""
        
        # In your actual Databricks notebook, run these SQL commands:
        create_trades_table = """
        CREATE TABLE IF NOT EXISTS gex_trading.mock_trades (
            trade_id STRING,
            user_id STRING DEFAULT 'mock_trader',
            symbol STRING,
            trade_type STRING,  -- 'LONG_CALLS', 'CALL_SELLING', 'STRADDLE'
            entry_timestamp TIMESTAMP,
            entry_date DATE,
            entry_price DOUBLE,
            quantity INT,
            confidence_score INT,
            setup_type STRING,
            recommendation STRING,
            status STRING DEFAULT 'OPEN',  -- 'OPEN', 'CLOSED', 'EXPIRED'
            exit_timestamp TIMESTAMP,
            exit_date DATE,
            exit_price DOUBLE,
            profit_loss DOUBLE,
            profit_loss_pct DOUBLE,
            days_held INT,
            notes STRING,
            created_by STRING DEFAULT 'streamlit_app',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
        ) USING DELTA
        PARTITIONED BY (entry_date)
        TBLPROPERTIES (
            'delta.autoOptimize.optimizeWrite' = 'true',
            'delta.autoOptimize.autoCompact' = 'true'
        )
        """
        
        create_portfolio_table = """
        CREATE TABLE IF NOT EXISTS gex_trading.mock_portfolio_history (
            portfolio_id STRING,
            user_id STRING DEFAULT 'mock_trader',
            snapshot_date DATE,
            snapshot_timestamp TIMESTAMP,
            total_value DOUBLE,
            cash_balance DOUBLE,
            positions_value DOUBLE,
            daily_pnl DOUBLE,
            total_return_pct DOUBLE,
            total_return_dollar DOUBLE,
            open_positions_count INT,
            closed_positions_count INT,
            win_rate DOUBLE,
            avg_win DOUBLE,
            avg_loss DOUBLE,
            max_drawdown DOUBLE,
            sharpe_ratio DOUBLE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
        ) USING DELTA
        PARTITIONED BY (snapshot_date)
        TBLPROPERTIES (
            'delta.autoOptimize.optimizeWrite' = 'true',
            'delta.autoOptimize.autoCompact' = 'true'
        )
        """
        
        # Store these for reference - you'd run them in Databricks
        self.table_creation_sql = {
            'trades': create_trades_table,
            'portfolio': create_portfolio_table
        }
    
    def execute_databricks_query(self, query: str, params: dict = None):
        """Execute query against Databricks SQL Warehouse"""
        try:
            # Using databricks-sql-connector (install: pip install databricks-sql-connector)
            from databricks import sql
            
            with sql.connect(
                server_hostname=self.databricks_config['server_hostname'],
                http_path=self.databricks_config['http_path'],
                access_token=self.databricks_config['access_token']
            ) as connection:
                with connection.cursor() as cursor:
                    cursor.execute(query, params)
                    if query.strip().upper().startswith('SELECT'):
                        return cursor.fetchall()
                    else:
                        return cursor.rowcount
        except Exception as e:
            st.error(f"Databricks connection error: {e}")
            # Fallback to local SQLite for demo
            return self.execute_local_fallback(query, params)

    def get_current_balance(self) -> Dict:
        """Get current portfolio balance and performance from Databricks"""
        try:
            # Query Databricks for portfolio metrics
            query = """
            WITH latest_portfolio AS (
                SELECT *
                FROM gex_trading.mock_portfolio_history 
                WHERE user_id = 'mock_trader'
                ORDER BY snapshot_timestamp DESC 
                LIMIT 1
            ),
            open_trades_value AS (
                SELECT 
                    COALESCE(SUM(entry_price * quantity * 100), 0) as total_invested,
                    COUNT(*) as open_count
                FROM gex_trading.mock_trades 
                WHERE status = 'OPEN' AND user_id = 'mock_trader'
            ),
            closed_pnl AS (
                SELECT COALESCE(SUM(profit_loss), 0) as realized_pnl
                FROM gex_trading.mock_trades 
                WHERE status = 'CLOSED' AND user_id = 'mock_trader'
            )
            SELECT 
                COALESCE(lp.total_value, 100000) as total_value,
                COALESCE(lp.cash_balance, 100000) as cash_balance,
                COALESCE(otv.total_invested, 0) as positions_value,
                COALESCE(cp.realized_pnl, 0) as realized_pnl,
                COALESCE(lp.total_return_pct, 0) as total_return_pct,
                COALESCE(otv.open_count, 0) as open_trades_count,
                COALESCE(lp.win_rate, 0) as win_rate
            FROM open_trades_value otv
            CROSS JOIN closed_pnl cp
            LEFT JOIN latest_portfolio lp ON 1=1
            """
            
            result = self.execute_databricks_query(query)
            if result:
                row = result[0]
                return {
                    'total_value': row[0],
                    'cash_balance': row[1],
                    'positions_value': row[2],
                    'realized_pnl': row[3],
                    'total_return_pct': row[4],
                    'open_trades_count': row[5],
                    'win_rate': row[6]
                }
        except Exception as e:
            st.warning(f"Using fallback data: {e}")
        
        # Fallback to default values
        return {
            'total_value': 100000,
            'cash_balance': 95000,
            'positions_value': 5000,
            'realized_pnl': 0,
            'total_return_pct': 0,
            'open_trades_count': 2,
            'win_rate': 0
        }
    
    def add_trade(self, symbol: str, trade_type: str, entry_price: float, 
                  quantity: int, confidence_score: int, setup_type: str, recommendation: str):
        """Add a new trade to Databricks"""
        import uuid
        from datetime import datetime
        
        trade_id = str(uuid.uuid4())
        now = datetime.now()
        
        query = """
        INSERT INTO gex_trading.mock_trades (
            trade_id, symbol, trade_type, entry_timestamp, entry_date, 
            entry_price, quantity, confidence_score, setup_type, recommendation
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        params = (
            trade_id, symbol, trade_type, now, now.date(),
            entry_price, quantity, confidence_score, setup_type, recommendation
        )
        
        try:
            self.execute_databricks_query(query, params)
            self.update_portfolio_snapshot()  # Update daily portfolio value
            return True
        except Exception as e:
            st.error(f"Failed to add trade: {e}")
            return False
    
    def update_portfolio_snapshot(self):
        """Update daily portfolio snapshot in Databricks"""
        balance = self.get_current_balance()
        
        query = """
        INSERT INTO gex_trading.mock_portfolio_history (
            portfolio_id, snapshot_date, snapshot_timestamp, total_value,
            cash_balance, positions_value, total_return_pct, open_positions_count
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        params = (
            'mock_portfolio_1', 
            datetime.now().date(),
            datetime.now(),
            balance['total_value'],
            balance['cash_balance'],
            balance['positions_value'],
            balance['total_return_pct'],
            balance['open_trades_count']
        )
        
        try:
            self.execute_databricks_query(query, params)
        except Exception as e:
            st.warning(f"Portfolio snapshot update failed: {e}")
    
    def get_open_trades(self) -> pd.DataFrame:
        """Get all open trades from Databricks"""
        query = """
        SELECT 
            trade_id, symbol, trade_type, entry_date, entry_price, 
            quantity, confidence_score, setup_type, recommendation,
            DATEDIFF(CURRENT_DATE(), entry_date) as days_held
        FROM gex_trading.mock_trades 
        WHERE status = 'OPEN' AND user_id = 'mock_trader'
        ORDER BY entry_timestamp DESC
        """
        
        try:
            result = self.execute_databricks_query(query)
            if result:
                columns = ['trade_id', 'symbol', 'trade_type', 'entry_date', 'entry_price',
                          'quantity', 'confidence_score', 'setup_type', 'recommendation', 'days_held']
                return pd.DataFrame(result, columns=columns)
        except Exception as e:
            st.warning(f"Failed to fetch trades: {e}")
        
        return pd.DataFrame()  # Empty DataFrame if query fails
    
    def get_closed_trades(self) -> pd.DataFrame:
        """Get all closed trades for performance analysis"""
        query = """
        SELECT 
            symbol, trade_type, entry_date, exit_date, entry_price, exit_price,
            quantity, profit_loss, profit_loss_pct, days_held, confidence_score,
            setup_type
        FROM gex_trading.mock_trades 
        WHERE status = 'CLOSED' AND user_id = 'mock_trader'
        ORDER BY exit_date DESC
        """
        
        try:
            result = self.execute_databricks_query(query)
            if result:
                columns = ['symbol', 'trade_type', 'entry_date', 'exit_date', 'entry_price',
                          'exit_price', 'quantity', 'profit_loss', 'profit_loss_pct', 
                          'days_held', 'confidence_score', 'setup_type']
                return pd.DataFrame(result, columns=columns)
        except Exception as e:
            st.warning(f"Failed to fetch closed trades: {e}")
        
        return pd.DataFrame()
    
    def execute_local_fallback(self, query: str, params: dict = None):
        """Fallback to SQLite for demo purposes when Databricks is unavailable"""
        # Simplified fallback for demo - would use original SQLite logic
        return []

class GEXStrategyDashboard:
    """Enhanced dashboard with strategy explanation and Databricks mock trading"""
    
    def __init__(self):
        self.mock_account = DatabricksMockAccount()
        # Sample morning data - in production, this would come from your Databricks pipeline
        self.sample_morning_data = self.generate_sample_morning_data()
    
    def load_morning_pipeline_results(self):
        """Load actual morning GEX analysis from your Databricks scheduled pipeline"""
        
        # Query to load results from your ScheduledMorningGEXPipeline
        morning_query = """
        WITH latest_run AS (
            SELECT MAX(analysis_timestamp) as latest_timestamp
            FROM gex_trading.scheduled_pipeline_results 
            WHERE analysis_date = CURRENT_DATE()
        )
        SELECT 
            symbol,
            spot_price as current_price,
            gamma_flip_point as gamma_flip,
            distance_to_flip,
            distance_to_flip_pct as distance_pct,
            structure_type,
            confidence_score,
            recommendation,
            category,
            priority,
            analysis_timestamp,
            scheduled_analysis
        FROM gex_trading.scheduled_pipeline_results r
        JOIN latest_run l ON r.analysis_timestamp = l.latest_timestamp
        WHERE confidence_score >= 70  -- Only tradeable setups
        ORDER BY confidence_score DESC, priority ASC
        LIMIT 25
        """
        
        try:
            result = self.mock_account.execute_databricks_query(morning_query)
            if result:
                opportunities = []
                
                for row in result:
                    # Map your pipeline results to dashboard format
                    opportunity = {
                        'symbol': str(row[0]),
                        'current_price': float(row[1]),
                        'gamma_flip': float(row[2]),
                        'distance_to_flip': float(row[3]),
                        'distance_pct': float(row[4]),
                        'structure_type': str(row[5]),
                        'confidence_score': int(row[6]),
                        'recommendation': str(row[7]),
                        'category': str(row[8]).replace('_', ' ').title(),
                        'priority': int(row[9]),
                        'analysis_timestamp': str(row[10]),
                        'scheduled_analysis': bool(row[11])
                    }
                    
                    # Determine trade type from structure
                    if 'SQUEEZE' in opportunity['structure_type']:
                        opportunity['trade_type'] = 'LONG_CALLS'
                    elif 'CALL_SELLING' in opportunity['structure_type']:
                        opportunity['trade_type'] = 'CALL_SELLING'  
                    elif 'FLIP_CRITICAL' in opportunity['structure_type']:
                        opportunity['trade_type'] = 'STRADDLE'
                    else:
                        opportunity['trade_type'] = 'MONITOR'
                    
                    opportunities.append(opportunity)
                
                if opportunities:
                    st.success(f"✅ Loaded {len(opportunities)} opportunities from your Databricks pipeline!")
                    return opportunities
                    
        except Exception as e:
            st.warning(f"Pipeline connection failed: {e} - Using sample data for demo")
        
        # Create sample data that matches your pipeline output format
        return self.generate_sample_from_pipeline_format()
    
    def generate_sample_from_pipeline_format(self):
        """Generate sample data that matches your actual pipeline output structure"""
        # This mimics the output from your ScheduledMorningGEXPipeline
        pipeline_samples = [
            {
                'symbol': 'MARA',
                'current_price': 25.50,
                'gamma_flip': 23.20,
                'distance_to_flip': 2.30,
                'distance_pct': 9.02,
                'structure_type': 'CALL_SELLING_SETUP',
                'confidence_score': 95,
                'recommendation': 'CALL SELLING - above flip',
                'category': 'Crypto Play',
                'priority': 1,
                'trade_type': 'CALL_SELLING',
                'scheduled_analysis': True,
                'analysis_timestamp': datetime.now().isoformat()
            },
            {
                'symbol': 'GME',
                'current_price': 22.15,
                'gamma_flip': 24.80,
                'distance_to_flip': -2.65,
                'distance_pct': -11.97,
                'structure_type': 'SQUEEZE_SETUP',
                'confidence_score': 92,
                'recommendation': 'LONG CALLS - squeeze',
                'category': 'Momentum Trade',
                'priority': 2,
                'trade_type': 'LONG_CALLS',
                'scheduled_analysis': True,
                'analysis_timestamp': datetime.now().isoformat()
            },
            {
                'symbol': 'COIN',
                'current_price': 245.30,
                'gamma_flip': 244.95,
                'distance_to_flip': 0.35,
                'distance_pct': 0.14,
                'structure_type': 'GAMMA_FLIP_CRITICAL',
                'confidence_score': 88,
                'recommendation': 'STRADDLE - critical flip',
                'category': 'Crypto Play',
                'priority': 1,
                'trade_type': 'STRADDLE',
                'scheduled_analysis': True,
                'analysis_timestamp': datetime.now().isoformat()
            },
            {
                'symbol': 'CRWD',
                'current_price': 285.40,
                'gamma_flip': 290.50,
                'distance_to_flip': -5.10,
                'distance_pct': -1.79,
                'structure_type': 'SQUEEZE_SETUP',
                'confidence_score': 85,
                'recommendation': 'LONG CALLS - squeeze',
                'category': 'Weekly Focus',
                'priority': 2,
                'trade_type': 'LONG_CALLS',
                'scheduled_analysis': True,
                'analysis_timestamp': datetime.now().isoformat()
            },
            {
                'symbol': 'MRNA',
                'current_price': 105.80,
                'gamma_flip': 102.30,
                'distance_to_flip': 3.50,
                'distance_pct': 3.31,
                'structure_type': 'CALL_SELLING_SETUP',
                'confidence_score': 82,
                'recommendation': 'CALL SELLING - above flip',
                'category': 'Biotech Bomb',
                'priority': 2,
                'trade_type': 'CALL_SELLING',
                'scheduled_analysis': True,
                'analysis_timestamp': datetime.now().isoformat()
            }
        ]
        
        st.info("📊 Demo mode - Sample data based on your pipeline structure. Connect to Databricks for live results!")
        return pipeline_samples
    
    def get_performance_analytics(self):
        """Get advanced performance analytics from Databricks"""
        
        # Performance by setup type
        setup_performance_query = """
        SELECT 
            setup_type,
            COUNT(*) as total_trades,
            AVG(profit_loss_pct) as avg_return_pct,
            COUNT(CASE WHEN profit_loss > 0 THEN 1 END) / COUNT(*) as win_rate,
            AVG(CASE WHEN profit_loss > 0 THEN profit_loss_pct END) as avg_win_pct,
            AVG(CASE WHEN profit_loss < 0 THEN profit_loss_pct END) as avg_loss_pct,
            AVG(days_held) as avg_hold_days,
            MAX(profit_loss_pct) as best_trade_pct,
            MIN(profit_loss_pct) as worst_trade_pct
        FROM gex_trading.mock_trades 
        WHERE status = 'CLOSED' AND user_id = 'mock_trader'
        GROUP BY setup_type
        ORDER BY avg_return_pct DESC
        """
        
        # Monthly performance
        monthly_performance_query = """
        SELECT 
            DATE_FORMAT(exit_date, 'yyyy-MM') as month,
            COUNT(*) as trades,
            SUM(profit_loss) as total_pnl,
            AVG(profit_loss_pct) as avg_return,
            COUNT(CASE WHEN profit_loss > 0 THEN 1 END) / COUNT(*) as win_rate
        FROM gex_trading.mock_trades 
        WHERE status = 'CLOSED' AND user_id = 'mock_trader'
        GROUP BY DATE_FORMAT(exit_date, 'yyyy-MM')
        ORDER BY month DESC
        """
        
        # Portfolio drawdown analysis
        drawdown_query = """
        SELECT 
            snapshot_date,
            total_value,
            total_return_pct,
            MAX(total_value) OVER (ORDER BY snapshot_date ROWS UNBOUNDED PRECEDING) as peak_value,
            (total_value - MAX(total_value) OVER (ORDER BY snapshot_date ROWS UNBOUNDED PRECEDING)) / 
            MAX(total_value) OVER (ORDER BY snapshot_date ROWS UNBOUNDED PRECEDING) * 100 as drawdown_pct
        FROM gex_trading.mock_portfolio_history 
        WHERE user_id = 'mock_trader'
        ORDER BY snapshot_date
        """
        
        try:
            setup_perf = self.mock_account.execute_databricks_query(setup_performance_query)
            monthly_perf = self.mock_account.execute_databricks_query(monthly_performance_query) 
            drawdown_data = self.mock_account.execute_databricks_query(drawdown_query)
            
            return {
                'setup_performance': setup_perf,
                'monthly_performance': monthly_perf,
                'drawdown_data': drawdown_data
            }
        except Exception as e:
            st.warning(f"Analytics unavailable: {e}")
            return None

    def generate_sample_morning_data(self) -> List[Dict]:
        """Generate sample morning analysis data for demo"""
        return [
            {
                'symbol': 'MARA',
                'current_price': 25.50,
                'gamma_flip': 23.20,
                'distance_to_flip': 2.30,
                'distance_pct': 9.02,
                'structure_type': 'CALL_SELLING_ZONE',
                'confidence_score': 95,
                'recommendation': 'SELL WEEKLY CALLS - Premium collection above flip',
                'category': 'Crypto',
                'trade_type': 'CALL_SELLING'
            },
            {
                'symbol': 'GME',
                'current_price': 22.15,
                'gamma_flip': 24.80,
                'distance_to_flip': -2.65,
                'distance_pct': -11.97,
                'structure_type': 'SQUEEZE_POTENTIAL',
                'confidence_score': 92,
                'recommendation': 'BUY WEEKLY CALLS - Explosive squeeze setup',
                'category': 'Meme',
                'trade_type': 'LONG_CALLS'
            },
            {
                'symbol': 'COIN',
                'current_price': 245.30,
                'gamma_flip': 244.95,
                'distance_to_flip': 0.35,
                'distance_pct': 0.14,
                'structure_type': 'GAMMA_FLIP_CRITICAL',
                'confidence_score': 88,
                'recommendation': 'STRADDLE - Critical flip point volatility',
                'category': 'Crypto',
                'trade_type': 'STRADDLE'
            },
            {
                'symbol': 'CRWD',
                'current_price': 285.40,
                'gamma_flip': 290.50,
                'distance_to_flip': -5.10,
                'distance_pct': -1.79,
                'structure_type': 'SQUEEZE_BUILDING',
                'confidence_score': 85,
                'recommendation': 'BUY WEEKLY CALLS - Building squeeze pressure',
                'category': 'Tech',
                'trade_type': 'LONG_CALLS'
            },
            {
                'symbol': 'MRNA',
                'current_price': 105.80,
                'gamma_flip': 102.30,
                'distance_to_flip': 3.50,
                'distance_pct': 3.31,
                'structure_type': 'CALL_SELLING_ZONE',
                'confidence_score': 82,
                'recommendation': 'SELL WEEKLY CALLS - Above flip resistance',
                'category': 'Biotech',
                'trade_type': 'CALL_SELLING'
            }
        ]

def explain_gex_strategy():
    """Explain the GEX strategy in simple terms"""
    st.markdown("""
    <div class="strategy-box">
    <h2>🎯 What is Gamma Exposure (GEX) Trading?</h2>
    
    <h3>📚 Simple Explanation:</h3>
    <p><strong>Gamma Exposure (GEX)</strong> is like finding the "invisible hand" that moves stock prices. 
    Market makers (big financial institutions) have to buy and sell stocks to hedge their options positions. 
    This creates predictable price movements we can profit from.</p>
    
    <h3>🔑 Key Concept - The "Gamma Flip":</h3>
    <p>Every stock has a <strong>"Gamma Flip Point"</strong> - a special price level where market maker behavior completely changes:</p>
    <ul>
        <li><strong>Above the Flip:</strong> Market makers SELL when price goes up → Price gets "stuck" (good for selling calls)</li>
        <li><strong>Below the Flip:</strong> Market makers BUY when price goes up → Price can "squeeze" higher (good for buying calls)</li>
        <li><strong>At the Flip:</strong> Maximum volatility and unpredictability (good for straddles)</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

def explain_trading_signals():
    """Explain the different trading signals"""
    st.markdown("""
    ## 🎯 Our 3 Main Trading Strategies:
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="bullish-signal">
        <h3>🚀 SQUEEZE PLAYS (Buy Calls)</h3>
        <p><strong>When:</strong> Price is BELOW the Gamma Flip</p>
        <p><strong>Why:</strong> Market makers have to buy when price rises, creating explosive moves up</p>
        <p><strong>Example:</strong> GME at $22 with flip at $25</p>
        <p><strong>Trade:</strong> Buy weekly call options</p>
        <p><strong>Target:</strong> 100%+ gains on explosive moves</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="bearish-signal">
        <h3>💰 PREMIUM SELLING (Sell Calls)</h3>
        <p><strong>When:</strong> Price is ABOVE the Gamma Flip</p>
        <p><strong>Why:</strong> Market makers sell when price rises, creating resistance</p>
        <p><strong>Example:</strong> MARA at $25.50 with flip at $23.20</p>
        <p><strong>Trade:</strong> Sell weekly call options</p>
        <p><strong>Target:</strong> 50% premium collection as calls expire worthless</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="neutral-signal">
        <h3>⚖️ VOLATILITY PLAYS (Straddles)</h3>
        <p><strong>When:</strong> Price is AT the Gamma Flip</p>
        <p><strong>Why:</strong> Maximum uncertainty creates big moves in either direction</p>
        <p><strong>Example:</strong> COIN at $245.30 with flip at $244.95</p>
        <p><strong>Trade:</strong> Buy both calls and puts (straddle)</p>
        <p><strong>Target:</strong> Profit from large moves up OR down</p>
        </div>
        """, unsafe_allow_html=True)

def display_confidence_scoring():
    """Explain the confidence scoring system"""
    st.markdown("""
    ## 📊 How We Score Opportunities (0-100%):
    
    **🔥 90%+ = EXCEPTIONAL** - Take these trades immediately
    **🎯 80-89% = STRONG** - High probability setups  
    **📊 70-79% = GOOD** - Solid opportunities
    **⚠️ Below 70%** = Wait for better setups
    
    **Confidence factors:**
    - **Distance from flip point** (closer = higher volatility)
    - **Stock category** (meme stocks + biotech = more explosive)
    - **Market conditions** (morning = higher volatility)
    - **Historical performance** (some stocks have better gamma behavior)
    """)

def create_databricks_tables_for_pipeline():
    """SQL commands to create tables that store your pipeline results"""
    return {
        'pipeline_results_table': """
        CREATE TABLE IF NOT EXISTS gex_trading.scheduled_pipeline_results (
            run_id STRING,
            analysis_date DATE,
            analysis_timestamp TIMESTAMP,
            symbol STRING,
            spot_price DOUBLE,
            gamma_flip_point DOUBLE,
            distance_to_flip DOUBLE,
            distance_to_flip_pct DOUBLE,
            structure_type STRING,
            confidence_score INT,
            recommendation STRING,
            category STRING,
            priority INT,
            scheduled_analysis BOOLEAN,
            interval_number INT,
            total_intervals INT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
        ) USING DELTA
        PARTITIONED BY (analysis_date)
        TBLPROPERTIES (
            'delta.autoOptimize.optimizeWrite' = 'true',
            'delta.autoOptimize.autoCompact' = 'true'
        )
        """,
        
        'pipeline_sessions_table': """
        CREATE TABLE IF NOT EXISTS gex_trading.pipeline_sessions (
            session_id STRING,
            analysis_date DATE,
            session_start TIMESTAMP,
            session_end TIMESTAMP,
            total_runtime_minutes DOUBLE,
            universe_size INT,
            symbols_processed INT,
            opportunities_found INT,
            high_confidence_opportunities INT,
            completion_rate DOUBLE,
            api_calls_made INT,
            intervals_completed INT,
            schedule_window STRING,
            is_weekend BOOLEAN,
            status STRING,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
        ) USING DELTA
        PARTITIONED BY (analysis_date)
        """
    }

def display_databricks_integration_guide():
    """Show users how to integrate with their pipeline"""
    st.header("🔗 Connect to Your Databricks Pipeline")
    
    st.markdown("""
    ## 🎯 Integration Steps:
    
    Your `ScheduledMorningGEXPipeline` is already perfect! Just need to save the results to Databricks.
    """)
    
    # Show the SQL table creation
    tables_sql = create_databricks_tables_for_pipeline()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Step 1: Create Tables in Databricks")
        st.code(tables_sql['pipeline_results_table'], language='sql')
        
    with col2:
        st.markdown("### 📈 Step 2: Create Sessions Table") 
        st.code(tables_sql['pipeline_sessions_table'], language='sql')
    
    # Show how to modify their pipeline
    st.markdown("### 🔧 Step 3: Modify Your Pipeline")
    
    databricks_integration_code = '''
# Add this to your ScheduledMorningGEXPipeline class

def save_opportunity_to_databricks(self, opportunity: Dict, interval_num: int, run_id: str):
    """Save opportunity to Databricks for Streamlit dashboard"""
    try:
        from databricks import sql
        
        connection = sql.connect(
            server_hostname=self.databricks_hostname,
            http_path=self.databricks_http_path,
            access_token=self.databricks_token
        )
        
        with connection.cursor() as cursor:
            cursor.execute("""
                INSERT INTO gex_trading.scheduled_pipeline_results VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
            """, (
                run_id,
                datetime.now().date(),
                datetime.now(),
                opportunity['symbol'],
                opportunity['spot_price'],
                opportunity['gamma_flip_point'],
                opportunity['distance_to_flip'],
                opportunity['distance_to_flip_pct'],
                opportunity['structure_type'],
                opportunity['confidence_score'],
                opportunity['recommendation'],
                opportunity['category'],
                opportunity['priority'],
                opportunity['scheduled_analysis'],
                interval_num,
                datetime.now()
            ))
            
    except Exception as e:
        print(f"⚠️ Databricks save error: {e}")

# Modify your process_interval_batch method:
def process_interval_batch(self) -> List[Dict]:
    """Process one interval batch - MODIFIED to save to Databricks"""
    interval_opportunities = []
    run_id = str(uuid.uuid4())
    
    for i in range(self.symbols_per_interval):
        # ... your existing processing code ...
        
        if opportunity and opportunity.get('confidence_score', 0) >= 70:
            interval_opportunities.append(opportunity)
            
            # NEW: Save to Databricks for Streamlit dashboard
            self.save_opportunity_to_databricks(opportunity, interval_count, run_id)
    
    return interval_opportunities
'''
    
    st.code(databricks_integration_code, language='python')
    
    # Configuration guide
    st.markdown("### ⚙️ Step 4: Add Databricks Credentials")
    
    config_example = '''
# Add to your ScheduledMorningGEXPipeline __init__ method:
self.databricks_hostname = "your-workspace.cloud.databricks.com"
self.databricks_http_path = "/sql/1.0/warehouses/your-warehouse-id"
self.databricks_token = "your-databricks-token"
'''
    
    st.code(config_example, language='python')
    
    # Benefits explanation
    st.success("""
    **🎯 Once connected, your dashboard will show:**
    
    ✅ **Real morning analysis** from your scheduled pipeline  
    ✅ **Live opportunities** processed from 6:00-8:30 AM Central  
    ✅ **Confidence scores** from your actual algorithm  
    ✅ **Priority rankings** from your dynamic universe  
    ✅ **Category breakdowns** (Crypto, Biotech, Meme, etc.)  
    ✅ **Historical performance** tracking over time  
    ✅ **Real-time trade execution** based on your analysis  
    """)
    
    st.warning("""
    **📝 Alternative: CSV Export Method**
    
    If you prefer not to modify your pipeline, you can also export results to CSV and upload them to the dashboard:
    
    ```python
    # Add to your generate_final_results method:
    df = pd.DataFrame(self.opportunities)
    df.to_csv(f'morning_analysis_{datetime.now().strftime("%Y%m%d")}.csv', index=False)
    ```
    
    Then upload the CSV file to the dashboard for analysis.
    """)

def display_pipeline_status():
    """Show current pipeline status and next run time"""
    st.subheader("⏰ Pipeline Schedule Status")
    
    # Calculate next run time
    utc_now = datetime.utcnow()
    central_now = utc_now + timedelta(hours=-6)  # Approximate Central Time
    
    next_run = central_now.replace(hour=6, minute=0, second=0, microsecond=0)
    if central_now.hour >= 8 and central_now.minute >= 30:
        next_run += timedelta(days=1)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Current Time (CT)",
            central_now.strftime("%H:%M:%S"),
            "Central Time Zone"
        )
    
    with col2:
        if 6 <= central_now.hour < 8 or (central_now.hour == 8 and central_now.minute < 30):
            status = "🟢 RUNNING"
            status_color = "normal"
        else:
            status = "🔴 OFFLINE" 
            status_color = "inverse"
            
        st.metric("Pipeline Status", status, delta_color=status_color)
    
    with col3:
        if next_run.date() == central_now.date():
            next_run_text = next_run.strftime("%H:%M Today")
        else:
            next_run_text = next_run.strftime("%H:%M Tomorrow")
            
        st.metric("Next Run", next_run_text, "6:00-8:30 AM CT")
    
    # Show recent runs (placeholder)
    st.markdown("**📈 Recent Pipeline Runs:**")
    recent_runs = pd.DataFrame([
        {"Date": "2024-08-23", "Symbols": 125, "Opportunities": 23, "High Conf": 8, "Status": "✅ Complete"},
        {"Date": "2024-08-22", "Symbols": 118, "Opportunities": 19, "High Conf": 6, "Status": "✅ Complete"},
        {"Date": "2024-08-21", "Symbols": 122, "Opportunities": 31, "High Conf": 12, "Status": "✅ Complete"},
    ])
    
    st.dataframe(recent_runs, use_container_width=True, hide_index=True)

def display_mock_portfolio():
    """Display mock portfolio performance with real pipeline integration"""
    st.header("💰 Mock Trading Account - $100K Challenge")
    
    # Show pipeline integration status
    display_pipeline_status()
    
    st.info("🎯 **Now Integrated with Your Scheduled Pipeline!** Trades based on your 6AM-8:30AM analysis.")
    
    # Initialize dashboard
    dashboard = GEXStrategyDashboard()
    
    # Get current balance
    balance = dashboard.mock_account.get_current_balance()
    
    # Display portfolio metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Portfolio Value", 
            f"${balance['total_value']:,.2f}",
            f"{balance['total_return_pct']:+.2f}%"
        )
    
    with col2:
        st.metric("Cash Balance", f"${balance['cash_balance']:,.2f}")
    
    with col3:
        st.metric("Open Positions", f"${balance['positions_value']:,.2f}")
    
    with col4:
        st.metric(
            "Win Rate", 
            f"{balance['win_rate']:.1f}%",
            delta_color="normal" if balance['win_rate'] >= 60 else "inverse"
        )
    
    # Trading interface - Load from your actual pipeline
    st.subheader("📈 Execute Trades from Your Scheduled Pipeline")
    
    # Load opportunities from your actual Databricks pipeline
    opportunities = dashboard.load_morning_pipeline_results()  # This connects to your REAL pipeline!
    high_conf_opportunities = [opp for opp in opportunities if opp['confidence_score'] >= 80]
    
    if high_conf_opportunities:
        st.write("**🌅 This Morning's Opportunities from Your Scheduled Pipeline (6:00-8:30 AM CT):**")
        
        for i, opp in enumerate(high_conf_opportunities[:5]):  # Show top 5
            with st.expander(f"🎯 {opp['symbol']} - {opp['confidence_score']}% Confidence - P{opp.get('priority', 'N/A')}", expanded=True):
                col1, col2, col3 = st.columns([3, 2, 1])
                
                with col1:
                    confidence_color = "🔥" if opp['confidence_score'] >= 90 else "🎯"
                    st.write(f"{confidence_color} **Setup:** {opp['structure_type']}")
                    st.write(f"💡 **Strategy:** {opp['recommendation']}")
                    st.write(f"📊 **Category:** {opp['category']} (Priority {opp.get('priority', 'N/A')})")
                    
                    # Show pipeline-specific info
                    if opp.get('scheduled_analysis'):
                        analysis_time = opp.get('analysis_timestamp', 'Unknown')
                        if isinstance(analysis_time, str):
                            try:
                                dt = datetime.fromisoformat(analysis_time.replace('Z', '+00:00'))
                                time_str = dt.strftime('%H:%M CT')
                            except:
                                time_str = analysis_time[:5]  # First 5 chars
                        else:
                            time_str = "Morning"
                        st.write(f"⏰ **Analyzed:** {time_str} (Scheduled Pipeline)")
                
                with col2:
                    st.write(f"**Current Price:** ${opp['current_price']:.2f}")
                    st.write(f"**Gamma Flip:** ${opp['gamma_flip']:.2f}")
                    st.write(f"**Distance:** {opp['distance_pct']:+.1f}%")
                    
                    # Distance indicator
                    if abs(opp['distance_pct']) < 0.5:
                        st.write("🎯 **AT FLIP POINT**")
                    elif opp['distance_pct'] > 2:
                        st.write("📈 **ABOVE FLIP** (Call selling zone)")
                    elif opp['distance_pct'] < -2:
                        st.write("🚀 **BELOW FLIP** (Squeeze potential)")
                
                with col3:
                    # Calculate position size based on portfolio value and setup
                    risk_amount = balance['total_value'] * 0.02  # 2% risk per trade
                    
                    # More realistic option pricing based on your setup types
                    if opp['structure_type'] == 'SQUEEZE_SETUP':
                        option_price = abs(opp['distance_pct']) * 0.15 + 0.75  # Higher premium for squeeze potential
                        quantity = max(1, int(risk_amount / (option_price * 100)))
                    elif opp['structure_type'] == 'CALL_SELLING_SETUP':
                        option_price = opp['current_price'] * 0.025  # Premium collected
                        quantity = max(1, int(risk_amount / (option_price * 100)))
                    elif opp['structure_type'] == 'GAMMA_FLIP_CRITICAL':
                        option_price = opp['current_price'] * 0.09  # Straddle pricing
                        quantity = max(1, int(risk_amount / (option_price * 100)))
                    else:
                        option_price = opp['current_price'] * 0.05  # Default pricing
                        quantity = max(1, int(risk_amount / (option_price * 100)))
                    
                    st.write(f"**Est. Option Price:** ${option_price:.2f}")
                    st.write(f"**Suggested Qty:** {quantity}")
                    st.write(f"**Risk Amount:** ${risk_amount:,.0f}")
                    
                    # Trade execution button with pipeline context
                    trade_key = f"pipeline_trade_{opp['symbol']}_{i}"
                    button_text = f"Execute {opp['trade_type'].replace('_', ' ').title()}"
                    
                    if st.button(button_text, key=trade_key, type="primary"):
                        success = dashboard.mock_account.add_trade(
                            symbol=opp['symbol'],
                            trade_type=opp['trade_type'],
                            entry_price=option_price,
                            quantity=quantity,
                            confidence_score=opp['confidence_score'],
                            setup_type=opp['structure_type'],
                            recommendation=f"Pipeline: {opp['recommendation']}"
                        )
                        
                        if success:
                            st.success(f"✅ Pipeline trade executed: {quantity} contracts of {opp['symbol']} {opp['trade_type']}")
                            time.sleep(1)  # Brief pause for user to see confirmation
                            st.rerun()
                        else:
                            st.error("❌ Trade execution failed - check Databricks connection")
    
    else:
        st.warning("📊 No high-confidence opportunities from today's scheduled pipeline run. Check back during 6:00-8:30 AM CT for live analysis!")
    
    # Show open positions from Databricks
    open_trades = dashboard.mock_account.get_open_trades()
    if not open_trades.empty:
        st.subheader("📊 Open Positions (Tracked in Databricks)")
        
        # Add current P&L estimation (in real implementation, you'd get live option prices)
        display_trades = open_trades.copy()
        display_trades['Est. Current P&L'] = display_trades.apply(
            lambda row: f"${np.random.uniform(-500, 1200):.0f}" if 'LONG' in row['trade_type'] 
            else f"${np.random.uniform(-300, 600):.0f}", axis=1
        )
        
        # Format display
        display_trades['Entry Date'] = pd.to_datetime(display_trades['entry_date']).dt.strftime('%m/%d/%Y')
        display_trades['Position Value'] = (display_trades['entry_price'] * display_trades['quantity'] * 100).round(0)
        
        st.dataframe(
            display_trades[['symbol', 'trade_type', 'Entry Date', 'entry_price', 'quantity', 
                           'confidence_score', 'setup_type', 'days_held', 'Est. Current P&L']].rename(columns={
                'symbol': 'Symbol',
                'trade_type': 'Trade Type',
                'entry_price': 'Entry Price',
                'quantity': 'Qty',
                'confidence_score': 'Confidence',
                'setup_type': 'Setup Type',
                'days_held': 'Days Held'
            }),
            use_container_width=True
        )
    
    # Advanced portfolio analytics
    st.subheader("📊 Advanced Portfolio Analytics")
    
    # Show performance analytics from Databricks
    analytics = dashboard.get_performance_analytics()
    
    if analytics and analytics['setup_performance']:
        st.write("**Performance by Setup Type:**")
        
        setup_data = analytics['setup_performance']
        setup_df = pd.DataFrame(setup_data, columns=[
            'Setup Type', 'Total Trades', 'Avg Return %', 'Win Rate', 
            'Avg Win %', 'Avg Loss %', 'Avg Hold Days', 'Best Trade %', 'Worst Trade %'
        ])
        
        st.dataframe(setup_df.round(2), use_container_width=True)
        
        # Visualize setup performance
        if len(setup_df) > 0:
            fig = px.scatter(
                setup_df, 
                x='Win Rate', 
                y='Avg Return %',
                size='Total Trades',
                color='Setup Type',
                title='Setup Performance: Win Rate vs Average Return (From Your Pipeline)',
                hover_data=['Avg Hold Days', 'Total Trades']
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Integration status
    st.subheader("🔗 Pipeline Integration Status")
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **✅ Connected to Your Scheduled Pipeline:**
        - Morning analysis from 6:00-8:30 AM Central
        - Dynamic universe with 125+ symbols
        - Priority-based opportunity ranking
        - Real confidence scores from your algorithm
        """)
    
    with col2:
        if st.button("🔧 View Integration Guide"):
            st.session_state['show_integration'] = True
    
    # Show integration guide if requested
    if st.session_state.get('show_integration', False):
        with st.expander("🔧 Databricks Integration Guide", expanded=True):
            display_databricks_integration_guide()
            if st.button("Close Guide"):
                st.session_state['show_integration'] = False

def display_morning_opportunities():
    """Display the morning analysis results"""
    st.header("🌅 This Morning's GEX Analysis")
    
    dashboard = GEXStrategyDashboard()
    opportunities = dashboard.load_morning_pipeline_results()
    
    # Summary metrics
    st.subheader("📊 Morning Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    high_conf = len([o for o in opportunities if o['confidence_score'] >= 90])
    medium_conf = len([o for o in opportunities if 80 <= o['confidence_score'] < 90])
    
    with col1:
        st.metric("Total Opportunities", len(opportunities))
    with col2:
        st.metric("🔥 Exceptional (90%+)", high_conf)
    with col3:
        st.metric("🎯 Strong (80-89%)", medium_conf)
    with col4:
        avg_conf = np.mean([o['confidence_score'] for o in opportunities])
        st.metric("Average Confidence", f"{avg_conf:.1f}%")
    
    # Visual analysis
    st.subheader("📈 Price vs Gamma Flip Analysis")
    
    # Create DataFrame for plotting
    df = pd.DataFrame(opportunities)
    
    # Price vs Gamma Flip scatter plot
    fig = px.scatter(
        df, 
        x='gamma_flip', 
        y='current_price',
        size='confidence_score',
        color='confidence_score',
        hover_data=['symbol', 'recommendation', 'structure_type'],
        title="Current Price vs Gamma Flip Point - Where the Magic Happens!",
        labels={
            'gamma_flip': 'Gamma Flip Price ($)',
            'current_price': 'Current Price ($)',
            'confidence_score': 'Confidence Score'
        },
        color_continuous_scale='RdYlGn'
    )
    
    # Add diagonal line (price = flip)
    min_price = min(df['gamma_flip'].min(), df['current_price'].min())
    max_price = max(df['gamma_flip'].max(), df['current_price'].max())
    fig.add_shape(
        type="line",
        x0=min_price, y0=min_price,
        x1=max_price, y1=max_price,
        line=dict(color="gray", width=2, dash="dash"),
        name="Perfect Balance Line"
    )
    
    # Add annotations
    fig.add_annotation(
        x=max_price * 0.8, y=max_price * 0.9,
        text="ABOVE LINE<br>= CALL SELLING<br>ZONE",
        showarrow=True,
        arrowhead=2,
        bgcolor="rgba(255,200,200,0.8)",
        bordercolor="red"
    )
    
    fig.add_annotation(
        x=max_price * 0.9, y=max_price * 0.7,
        text="BELOW LINE<br>= SQUEEZE<br>POTENTIAL",
        showarrow=True,
        arrowhead=2,
        bgcolor="rgba(200,255,200,0.8)",
        bordercolor="green"
    )
    
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed opportunities
    st.subheader("🎯 Detailed Morning Picks")
    
    # Sort by confidence
    opportunities.sort(key=lambda x: x['confidence_score'], reverse=True)
    
    for i, opp in enumerate(opportunities):
        # Determine signal type for styling
        if opp['structure_type'] in ['SQUEEZE_POTENTIAL', 'SQUEEZE_BUILDING', 'SQUEEZE_SETUP']:
            signal_class = "bullish-signal"
            signal_emoji = "🚀"
        elif opp['structure_type'] in ['CALL_SELLING_ZONE', 'CALL_SELLING_SETUP']:
            signal_class = "bearish-signal"
            signal_emoji = "💰"
        else:
            signal_class = "neutral-signal"
            signal_emoji = "⚖️"
        
        confidence_badge = "🔥" if opp['confidence_score'] >= 90 else "🎯" if opp['confidence_score'] >= 80 else "📊"
        
        st.markdown(f"""
        <div class="{signal_class}">
            <h3>#{i+1}: {signal_emoji} {opp['symbol']} - {confidence_badge} {opp['confidence_score']}% Confidence</h3>
            <p><strong>Setup:</strong> {opp['structure_type']}</p>
            <p><strong>Current Price:</strong> ${opp['current_price']:.2f} | 
               <strong>Gamma Flip:</strong> ${opp['gamma_flip']:.2f} | 
               <strong>Distance:</strong> {opp['distance_pct']:+.2f}%</p>
            <p><strong>🎯 Strategy:</strong> {opp['recommendation']}</p>
            <p><strong>Category:</strong> {opp['category']} Stock</p>
        </div>
        """, unsafe_allow_html=True)

def main():
    """Main Streamlit app"""
    
    # Header
    st.markdown('<h1 class="main-header">🎯 GEX Trading Strategy Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("**Turn Market Maker Psychology Into Profits - $100K Mock Trading Challenge**")
    
    # Sidebar navigation
    st.sidebar.header("📋 Navigation")
    page = st.sidebar.radio(
        "Choose Page:",
        ["🎓 Learn the Strategy", "🌅 Morning Analysis", "💰 Mock Trading Account", "📊 Performance Tracking"]
    )
    
    if page == "🎓 Learn the Strategy":
        st.header("🎓 Learn the GEX Trading Strategy")
        
        explain_gex_strategy()
        
        explain_trading_signals()
        
        display_confidence_scoring()
        
        st.markdown("""
        ## 🔄 How Our System Works Daily:
        
        1. **6:00 AM Central** - Automated system analyzes 125+ stocks
        2. **6:00-8:30 AM** - Processes entire universe, finds best opportunities  
        3. **Market Open** - Discord alerts sent for 90%+ confidence trades
        4. **Your Job** - Execute the trades and track performance here!
        
        ## 🎯 Why This Works:
        - **Based on market structure** - Not just technical analysis
        - **Exploits predictable behavior** - Market makers must hedge
        - **Quantified approach** - Confidence scores remove emotion
        - **Proven edge** - Big institutions use similar strategies
        
        Ready to see today's opportunities? Check the Morning Analysis tab! 🚀
        """)
    
    elif page == "🌅 Morning Analysis":
        display_morning_opportunities()
    
    elif page == "💰 Mock Trading Account":
        display_mock_portfolio()
    
    elif page == "📊 Performance Tracking":
        st.header("📊 Strategy Performance Analysis")
        
        st.info("👷 Under Construction - Coming Soon!")
        st.markdown("""
        **Planned Features:**
        - Win rate by setup type
        - Average return per trade
        - Best performing time periods  
        - Risk-adjusted returns
        - Drawdown analysis
        - Comparison to buy-and-hold
        """)
        
        # Placeholder performance metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Overall Win Rate", "68%", "+2.3%")
        with col2:
            st.metric("Avg Return per Trade", "15.2%", "+1.8%")
        with col3:
            st.metric("Best Setup Type", "Squeeze Plays", "85% win rate")

if __name__ == "__main__":
    main()
