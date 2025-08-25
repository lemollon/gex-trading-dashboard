"""
Production-Ready Streamlit Dashboard Configuration
Connects to Databricks Delta tables instead of using simulated data
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import requests
import json
from typing import Dict, List, Tuple, Optional
import logging

# ======================== PRODUCTION DATA CONNECTION ========================

class ProductionDataConnector:
    """Connects Streamlit to production Databricks Delta tables"""
    
    def __init__(self):
        # Databricks connection settings - store in st.secrets
        self.databricks_host = st.secrets["databricks"]["host"]
        self.databricks_token = st.secrets["databricks"]["token"]
        self.catalog = "quant_projects"
        self.schema = "gex_trading"
        
        # API endpoints for data access
        self.sql_endpoint_id = st.secrets["databricks"]["sql_endpoint_id"]
        
    def execute_sql(self, query: str) -> pd.DataFrame:
        """Execute SQL query against Databricks via REST API"""
        try:
            url = f"https://{self.databricks_host}/api/2.0/sql/statements/"
            
            headers = {
                "Authorization": f"Bearer {self.databricks_token}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "statement": query,
                "warehouse_id": self.sql_endpoint_id,
                "wait_timeout": "30s"
            }
            
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            
            result = response.json()
            
            if result["status"]["state"] == "SUCCEEDED":
                # Convert result to DataFrame
                columns = [col["name"] for col in result["manifest"]["schema"]["columns"]]
                rows = result["result"]["data_array"]
                return pd.DataFrame(rows, columns=columns)
            else:
                st.error(f"Query failed: {result['status']}")
                return pd.DataFrame()
                
        except Exception as e:
            st.error(f"Database connection error: {e}")
            return pd.DataFrame()
    
    def get_latest_gex_data(self, symbols: List[str] = None) -> Dict:
        """Get latest GEX data for all symbols"""
        symbol_filter = ""
        if symbols:
            symbol_list = "', '".join(symbols)
            symbol_filter = f"AND symbol IN ('{symbol_list}')"
        
        query = f"""
        SELECT 
            symbol,
            spot_price,
            net_gex,
            gamma_flip,
            call_walls,
            put_walls,
            gex_profile,
            data_timestamp,
            market_regime
        FROM {self.catalog}.{self.schema}.gex_analysis
        WHERE DATE(data_timestamp) = CURRENT_DATE()
        {symbol_filter}
        ORDER BY data_timestamp DESC
        """
        
        df = self.execute_sql(query)
        
        # Convert to dictionary format expected by dashboard
        gex_data = {}
        for _, row in df.iterrows():
            symbol = row['symbol']
            gex_data[symbol] = {
                'spot_price': float(row['spot_price']),
                'net_gex': float(row['net_gex']),
                'gamma_flip': float(row['gamma_flip']) if row['gamma_flip'] else None,
                'call_walls': json.loads(row['call_walls']) if row['call_walls'] else [],
                'put_walls': json.loads(row['put_walls']) if row['put_walls'] else [],
                'gex_profile': json.loads(row['gex_profile']) if row['gex_profile'] else {},
                'timestamp': row['data_timestamp'],
                'market_regime': row['market_regime']
            }
        
        return gex_data
    
    def get_active_recommendations(self) -> List[Dict]:
        """Get active trade recommendations from production pipeline"""
        query = f"""
        SELECT 
            symbol,
            recommendation_type,
            strategy_name,
            entry_price,
            target_strike,
            confidence_score,
            risk_reward_ratio,
            position_size_pct,
            entry_criteria,
            exit_criteria,
            notes,
            created_timestamp,
            expires_timestamp,
            is_active
        FROM {self.catalog}.{self.schema}.trade_recommendations
        WHERE is_active = true
        AND expires_timestamp > CURRENT_TIMESTAMP()
        ORDER BY confidence_score DESC, created_timestamp DESC
        """
        
        df = self.execute_sql(query)
        
        recommendations = []
        for _, row in df.iterrows():
            recommendations.append({
                'symbol': row['symbol'],
                'type': row['recommendation_type'],
                'strategy': row['strategy_name'],
                'entry_price': float(row['entry_price']),
                'target_strike': float(row['target_strike']) if row['target_strike'] else None,
                'confidence': float(row['confidence_score']),
                'risk_reward': float(row['risk_reward_ratio']) if row['risk_reward_ratio'] else 0,
                'position_size': row['position_size_pct'],
                'entry_criteria': row['entry_criteria'],
                'exit_criteria': row['exit_criteria'],
                'notes': row['notes'],
                'created': row['created_timestamp'],
                'expires': row['expires_timestamp']
            })
        
        return recommendations
    
    def get_performance_metrics(self) -> Dict:
        """Get performance metrics from executed trades"""
        query = f"""
        SELECT 
            COUNT(*) as total_trades,
            COUNT(CASE WHEN realized_pnl > 0 THEN 1 END) as winning_trades,
            AVG(realized_pnl) as avg_pnl,
            SUM(realized_pnl) as total_pnl,
            AVG(CASE WHEN realized_pnl > 0 THEN realized_pnl END) as avg_win,
            AVG(CASE WHEN realized_pnl < 0 THEN ABS(realized_pnl) END) as avg_loss,
            MAX(realized_pnl) as max_win,
            MIN(realized_pnl) as max_loss
        FROM {self.catalog}.{self.schema}.executed_trades
        WHERE DATE(executed_timestamp) >= CURRENT_DATE() - INTERVAL 30 DAYS
        """
        
        df = self.execute_sql(query)
        
        if len(df) > 0:
            row = df.iloc[0]
            return {
                'total_trades': int(row['total_trades']) if row['total_trades'] else 0,
                'winning_trades': int(row['winning_trades']) if row['winning_trades'] else 0,
                'avg_pnl': float(row['avg_pnl']) if row['avg_pnl'] else 0,
                'total_pnl': float(row['total_pnl']) if row['total_pnl'] else 0,
                'avg_win': float(row['avg_win']) if row['avg_win'] else 0,
                'avg_loss': float(row['avg_loss']) if row['avg_loss'] else 0,
                'max_win': float(row['max_win']) if row['max_win'] else 0,
                'max_loss': float(row['max_loss']) if row['max_loss'] else 0
            }
        else:
            return {
                'total_trades': 0, 'winning_trades': 0, 'avg_pnl': 0,
                'total_pnl': 0, 'avg_win': 0, 'avg_loss': 0,
                'max_win': 0, 'max_loss': 0
            }
    
    def get_alerts(self) -> List[Dict]:
        """Get active alerts from production system"""
        query = f"""
        SELECT 
            symbol,
            alert_type,
            alert_message,
            priority_level,
            created_timestamp,
            is_active
        FROM {self.catalog}.{self.schema}.trading_alerts
        WHERE is_active = true
        AND created_timestamp > CURRENT_TIMESTAMP() - INTERVAL 4 HOURS
        ORDER BY priority_level DESC, created_timestamp DESC
        """
        
        df = self.execute_sql(query)
        
        alerts = []
        for _, row in df.iterrows():
            alerts.append({
                'symbol': row['symbol'],
                'type': row['alert_type'],
                'message': row['alert_message'],
                'priority': row['priority_level'],
                'timestamp': row['created_timestamp'],
                'is_active': bool(row['is_active'])
            })
        
        return alerts

# ======================== UPDATED MAIN APPLICATION ========================

def main():
    # Initialize production data connector
    if 'data_connector' not in st.session_state:
        st.session_state.data_connector = ProductionDataConnector()
    
    # Page configuration (keep your existing styling)
    st.set_page_config(
        page_title="GEX Trading Dashboard Pro",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Keep your existing CSS styling
    # ... (all your existing CSS code) ...
    
    # Header
    st.markdown("""
    <h1 style='text-align: center; font-size: 48px; margin-bottom: 0;'>
        <span class='live-indicator'></span>
        GEX Trading Dashboard Pro - PRODUCTION
    </h1>
    <p style='text-align: center; color: rgba(255,255,255,0.7); font-size: 18px; margin-top: 10px;'>
        Real-time Production Data from Databricks Pipeline
    </p>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown("<h2 style='text-align: center;'>⚙️ Control Panel</h2>", unsafe_allow_html=True)
        
        # Watchlist Management
        st.markdown("### 📊 Watchlist Management")
        
        # Get available symbols from production data
        try:
            available_symbols_query = f"""
            SELECT DISTINCT symbol 
            FROM {st.session_state.data_connector.catalog}.{st.session_state.data_connector.schema}.gex_analysis
            WHERE DATE(data_timestamp) = CURRENT_DATE()
            ORDER BY symbol
            """
            available_df = st.session_state.data_connector.execute_sql(available_symbols_query)
            available_symbols = available_df['symbol'].tolist() if len(available_df) > 0 else ["SPY", "QQQ", "IWM"]
        except:
            available_symbols = ["SPY", "QQQ", "IWM"]  # Fallback
        
        selected_symbols = st.multiselect(
            "Select Symbols to Monitor",
            options=available_symbols,
            default=available_symbols[:4],
            help="Choose symbols with active GEX data"
        )
        
        st.session_state.watchlist = selected_symbols
        
        # Refresh button
        if st.button("🔄 Refresh Data", type="primary", use_container_width=True):
            st.session_state.last_update = datetime.now()
            st.rerun()
        
        # Connection status
        st.divider()
        st.markdown("### 📡 Connection Status")
        
        try:
            # Test connection
            test_query = "SELECT 1 as test"
            test_result = st.session_state.data_connector.execute_sql(test_query)
            if len(test_result) > 0:
                st.success("✅ Connected to Databricks")
            else:
                st.error("❌ Connection Failed")
        except Exception as e:
            st.error(f"❌ Connection Error: {str(e)[:50]}...")
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Live Recommendations", 
        "📊 GEX Analysis", 
        "📈 Performance", 
        "⚠️ Alerts",
        "🔧 System Status"
    ])
    
    # Tab 1: Live Recommendations from Production
    with tab1:
        st.markdown("## 🏆 Live Production Recommendations")
        
        # Get active recommendations
        with st.spinner("Loading live recommendations from production pipeline..."):
            recommendations = st.session_state.data_connector.get_active_recommendations()
        
        if recommendations:
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Active Recommendations", len(recommendations))
            
            with col2:
                high_conf = len([r for r in recommendations if r['confidence'] > 80])
                st.metric("High Confidence", high_conf, delta=f"{high_conf/len(recommendations)*100:.0f}%")
            
            with col3:
                avg_confidence = np.mean([r['confidence'] for r in recommendations])
                st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
            
            with col4:
                symbols = len(set([r['symbol'] for r in recommendations]))
                st.metric("Active Symbols", symbols)
            
            st.divider()
            
            # Display recommendations
            for idx, rec in enumerate(recommendations[:10]):
                confidence_color = "🟢" if rec['confidence'] > 80 else "🟡" if rec['confidence'] > 70 else "🔴"
                
                with st.expander(f"{rec['symbol']} - {rec['strategy']} {confidence_color}", expanded=(idx < 3)):
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        st.markdown(f"""
                        **Strategy:** {rec['strategy']}  
                        **Entry:** {rec['entry_criteria']}  
                        **Exit:** {rec['exit_criteria']}  
                        **Position Size:** {rec['position_size']}  
                        **Notes:** {rec['notes']}
                        """)
                    
                    with col2:
                        st.metric("Confidence", f"{rec['confidence']:.0f}%")
                        if rec['risk_reward'] > 0:
                            st.metric("R/R Ratio", f"{rec['risk_reward']:.2f}")
                    
                    with col3:
                        st.metric("Entry Price", f"${rec['entry_price']:.2f}")
                        if rec['target_strike']:
                            st.metric("Target", f"${rec['target_strike']:.2f}")
                        
                        # Time remaining
                        if rec['expires']:
                            expires = pd.to_datetime(rec['expires'])
                            now = pd.Timestamp.now()
                            hours_left = (expires - now).total_seconds() / 3600
                            st.metric("Hours Left", f"{hours_left:.1f}h")
        else:
            st.info("No active recommendations from production pipeline")
    
    # Tab 2: GEX Analysis from Production Data
    with tab2:
        st.markdown("## 📊 Production GEX Analysis")
        
        # Get production GEX data
        with st.spinner("Loading GEX data from Databricks..."):
            gex_data = st.session_state.data_connector.get_latest_gex_data(st.session_state.watchlist)
        
        if gex_data:
            # Market overview table
            overview_data = []
            for symbol, data in gex_data.items():
                overview_data.append({
                    'Symbol': symbol,
                    'Spot Price': f"${data['spot_price']:.2f}",
                    'Net GEX': f"{data['net_gex']/1e9:.2f}B",
                    'Gamma Flip': f"${data['gamma_flip']:.2f}" if data['gamma_flip'] else "N/A",
                    'Regime': data['market_regime'],
                    'Last Update': pd.to_datetime(data['timestamp']).strftime('%H:%M:%S')
                })
            
            overview_df = pd.DataFrame(overview_data)
            st.dataframe(overview_df, use_container_width=True)
            
            # Individual symbol analysis
            if st.session_state.watchlist:
                selected_symbol = st.selectbox(
                    "Select Symbol for Detailed Analysis",
                    options=list(gex_data.keys())
                )
                
                if selected_symbol in gex_data:
                    symbol_data = gex_data[selected_symbol]
                    
                    # Key metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Spot Price", f"${symbol_data['spot_price']:.2f}")
                    
                    with col2:
                        net_gex_b = symbol_data['net_gex'] / 1e9
                        regime_color = "🟢" if net_gex_b > 0 else "🔴"
                        st.metric("Net GEX", f"{regime_color} {net_gex_b:.2f}B")
                    
                    with col3:
                        if symbol_data['gamma_flip']:
                            flip_distance = (symbol_data['gamma_flip'] - symbol_data['spot_price']) / symbol_data['spot_price'] * 100
                            st.metric("Gamma Flip", f"${symbol_data['gamma_flip']:.2f}", 
                                     delta=f"{flip_distance:+.2f}%")
                        else:
                            st.metric("Gamma Flip", "N/A")
                    
                    with col4:
                        st.metric("Market Regime", symbol_data['market_regime'])
                    
                    # GEX Profile Visualization (if available)
                    if symbol_data['gex_profile']:
                        st.markdown("### 📈 GEX Profile")
                        
                        # Convert JSON profile to DataFrame for plotting
                        profile_data = symbol_data['gex_profile']
                        if isinstance(profile_data, dict) and 'strikes' in profile_data:
                            strikes = profile_data['strikes']
                            gex_values = profile_data['gex_values']
                            
                            fig = go.Figure()
                            
                            # GEX bars
                            colors = ['#38ef7d' if x > 0 else '#f45c43' for x in gex_values]
                            
                            fig.add_trace(go.Bar(
                                x=strikes,
                                y=[g/1e6 for g in gex_values],  # Convert to millions
                                marker_color=colors,
                                name='GEX (M)',
                                hovertemplate='Strike: $%{x}<br>GEX: %{y:.2f}M<extra></extra>'
                            ))
                            
                            # Add spot price line
                            fig.add_vline(x=symbol_data['spot_price'], line_dash="dash", 
                                         line_color="#00D2FF", line_width=2,
                                         annotation_text=f"Spot: ${symbol_data['spot_price']:.2f}")
                            
                            # Add gamma flip line
                            if symbol_data['gamma_flip']:
                                fig.add_vline(x=symbol_data['gamma_flip'], line_dash="dash",
                                             line_color="#FFD700", line_width=2,
                                             annotation_text=f"Flip: ${symbol_data['gamma_flip']:.2f}")
                            
                            fig.update_layout(
                                height=500,
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)',
                                font=dict(color='rgba(255,255,255,0.9)'),
                                xaxis=dict(title="Strike Price", gridcolor='rgba(255,255,255,0.1)'),
                                yaxis=dict(title="GEX (Millions)", gridcolor='rgba(255,255,255,0.1)')
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No GEX data available from production pipeline")
    
    # Tab 3: Performance Metrics from Production
    with tab3:
        st.markdown("## 📈 Production Performance Metrics")
        
        with st.spinner("Loading performance data..."):
            performance = st.session_state.data_connector.get_performance_metrics()
        
        if performance['total_trades'] > 0:
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Trades (30d)", performance['total_trades'])
            
            with col2:
                win_rate = (performance['winning_trades'] / performance['total_trades']) * 100
                st.metric("Win Rate", f"{win_rate:.1f}%")
            
            with col3:
                st.metric("Total P&L", f"${performance['total_pnl']:+,.0f}")
            
            with col4:
                st.metric("Avg P&L", f"${performance['avg_pnl']:+,.2f}")
            
            # Additional metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Avg Win", f"${performance['avg_win']:,.2f}")
            
            with col2:
                st.metric("Avg Loss", f"${performance['avg_loss']:,.2f}")
            
            with col3:
                st.metric("Max Win", f"${performance['max_win']:,.2f}")
            
            with col4:
                st.metric("Max Loss", f"${performance['max_loss']:,.2f}")
            
            # Risk-adjusted metrics
            if performance['avg_loss'] > 0:
                profit_factor = performance['avg_win'] / performance['avg_loss']
                st.metric("Profit Factor", f"{profit_factor:.2f}")
        else:
            st.info("No trade performance data available yet")
    
    # Tab 4: Production Alerts
    with tab4:
        st.markdown("## ⚠️ Production Trading Alerts")
        
        with st.spinner("Loading alerts..."):
            alerts = st.session_state.data_connector.get_alerts()
        
        if alerts:
            # Group alerts by priority
            high_alerts = [a for a in alerts if a['priority'] == 'HIGH']
            medium_alerts = [a for a in alerts if a['priority'] == 'MEDIUM']
            low_alerts = [a for a in alerts if a['priority'] == 'LOW']
            
            if high_alerts:
                st.markdown("### 🔴 High Priority Alerts")
                for alert in high_alerts:
                    st.markdown(f"""
                    <div class='alert-high'>
                        <strong>{alert['symbol']} - {alert['type']}</strong><br/>
                        {alert['message']}<br/>
                        <small>{pd.to_datetime(alert['timestamp']).strftime('%H:%M:%S')}</small>
                    </div>
                    """, unsafe_allow_html=True)
            
            if medium_alerts:
                st.markdown("### 🟡 Medium Priority Alerts")
                for alert in medium_alerts:
                    st.markdown(f"""
                    <div class='alert-medium'>
                        <strong>{alert['symbol']} - {alert['type']}</strong><br/>
                        {alert['message']}<br/>
                        <small>{pd.to_datetime(alert['timestamp']).strftime('%H:%M:%S')}</small>
                    </div>
                    """, unsafe_allow_html=True)
            
            if low_alerts:
                st.markdown("### 🟢 Low Priority Alerts")
                for alert in low_alerts:
                    st.markdown(f"""
                    <div class='alert-low'>
                        <strong>{alert['symbol']} - {alert['type']}</strong><br/>
                        {alert['message']}<br/>
                        <small>{pd.to_datetime(alert['timestamp']).strftime('%H:%M:%S')}</small>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.success("✅ No active alerts")
    
    # Tab 5: System Status
    with tab5:
        st.markdown("## 🔧 Production System Status")
        
        # Pipeline status
        try:
            pipeline_status_query = f"""
            SELECT 
                run_timestamp,
                status,
                opportunities_processed,
                recommendations_generated,
                discord_alerts_sent
            FROM {st.session_state.data_connector.catalog}.{st.session_state.data_connector.schema}.pipeline_monitoring
            ORDER BY run_timestamp DESC
            LIMIT 10
            """
            
            status_df = st.session_state.data_connector.execute_sql(pipeline_status_query)
            
            if len(status_df) > 0:
                st.markdown("### 🚀 Recent Pipeline Runs")
                st.dataframe(status_df, use_container_width=True)
                
                # Latest run status
                latest = status_df.iloc[0]
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    status_color = "🟢" if latest['status'] == 'SUCCESS' else "🔴"
                    st.metric("Latest Status", f"{status_color} {latest['status']}")
                
                with col2:
                    st.metric("Opportunities", latest['opportunities_processed'])
                
                with col3:
                    st.metric("Recommendations", latest['recommendations_generated'])
                
                with col4:
                    st.metric("Alerts Sent", latest['discord_alerts_sent'])
            else:
                st.warning("No pipeline monitoring data found")
                
        except Exception as e:
            st.error(f"Unable to load system status: {e}")
        
        # Data freshness
        st.markdown("### 📊 Data Freshness")
        try:
            freshness_query = f"""
            SELECT 
                symbol,
                MAX(data_timestamp) as latest_update,
                COUNT(*) as records_today
            FROM {st.session_state.data_connector.catalog}.{st.session_state.data_connector.schema}.gex_analysis
            WHERE DATE(data_timestamp) = CURRENT_DATE()
            GROUP BY symbol
            ORDER BY latest_update DESC
            """
            
            freshness_df = st.session_state.data_connector.execute_sql(freshness_query)
            
            if len(freshness_df) > 0:
                st.dataframe(freshness_df, use_container_width=True)
            else:
                st.warning("No data freshness information available")
                
        except Exception as e:
            st.error(f"Unable to load data freshness: {e}")

if __name__ == "__main__":
    main()
