"""
GEX Trading Dashboard - Clean Minimal Version
No colors, pure functionality
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# Try to import Databricks connector
try:
    from databricks import sql
    DATABRICKS_AVAILABLE = True
except ImportError:
    DATABRICKS_AVAILABLE = False

# Page config
st.set_page_config(
    page_title="GEX Master Pro",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Minimal CSS - no colors
st.markdown("""
<style>
    .main {
        padding-top: 1rem;
    }
    
    .metric-container {
        border: 1px solid #ccc;
        padding: 1rem;
        border-radius: 5px;
        text-align: center;
        margin: 0.5rem 0;
        background: white;
    }
    
    .setup-card {
        border: 1px solid #ccc;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        background: white;
    }
</style>
""", unsafe_allow_html=True)

def load_databricks_data():
    """Load data from YOUR ACTUAL working Databricks table structure"""
    
    try:
        if DATABRICKS_AVAILABLE and "databricks" in st.secrets:
            connection = sql.connect(
                server_hostname=st.secrets["databricks"]["server_hostname"],
                http_path=st.secrets["databricks"]["http_path"],
                access_token=st.secrets["databricks"]["access_token"]
            )
            
            cursor = connection.cursor()
            
            # Query using the ACTUAL column names from your table
            cursor.execute("""
                SELECT 
                    run_id,
                    symbol,
                    condition_type as setup_type,
                    confidence_score,
                    spot_price as entry_price,
                    gamma_flip_point as target_price,
                    distance_to_flip as net_gex,
                    recommendation as market_regime,
                    '2%' as position_size,
                    created_at as created_timestamp
                FROM quant_projects.gex_trading.gex_pipeline_results
                ORDER BY confidence_score DESC, created_at DESC
            """)
            
            recommendations = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            recommendations_df = pd.DataFrame(recommendations, columns=columns)
            
            cursor.close()
            connection.close()
            
            return {
                'recommendations': recommendations_df,
                'status': 'connected',
                'message': f'Connected - Found {len(recommendations_df)} recommendations from your pipeline'
            }
            
    except Exception as e:
        st.error(f"Databricks error: {str(e)}")
        return {
            'recommendations': pd.DataFrame(),
            'status': 'error', 
            'message': f'Connection failed: {str(e)}'
        }

def main():
    # Header
    st.title("GEX Master Pro")
    st.subheader("Live Databricks Pipeline Dashboard")
    
    # Load data
    data = load_databricks_data()
    recommendations_df = data['recommendations']
    
    # Connection status
    if data['status'] == 'connected':
        st.success("Connected to Databricks")
    else:
        st.info("Development Mode - " + data['message'])
    
    # Sidebar
    with st.sidebar:
        st.header("Controls")
        
        # Refresh button
        if st.button("Refresh Data"):
            st.rerun()
        
        st.markdown("---")
        
        # Filters
        min_confidence = st.slider("Min Confidence %", 0, 100, 70)
        
        setup_types = st.multiselect(
            "Setup Types",
            ['squeeze_play', 'premium_selling', 'gamma_flip_play'],
            default=['squeeze_play', 'premium_selling', 'gamma_flip_play']
        )
        
        st.markdown("---")
        st.markdown("### Quick Reference")
        st.markdown("""
        - **Negative GEX**: Volatility amplification
        - **Positive GEX**: Volatility suppression  
        - **Gamma Flip**: Zero-gamma crossing
        """)
    
    # Main content
    if not recommendations_df.empty:
        # Apply filters
        filtered_df = recommendations_df[
            (recommendations_df['confidence_score'] >= min_confidence) &
            (recommendations_df['setup_type'].isin(setup_types))
        ]
        
        # Key metrics
        st.subheader("Key Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-container">
                <h3>Total Setups</h3>
                <h2>{len(filtered_df)}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            high_conf = len(filtered_df[filtered_df['confidence_score'] >= 85])
            st.markdown(f"""
            <div class="metric-container">
                <h3>High Confidence</h3>
                <h2>{high_conf}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            avg_conf = filtered_df['confidence_score'].mean() if not filtered_df.empty else 0
            st.markdown(f"""
            <div class="metric-container">
                <h3>Avg Confidence</h3>
                <h2>{avg_conf:.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            positive_gex = len(filtered_df[filtered_df['net_gex'] > 0])
            st.markdown(f"""
            <div class="metric-container">
                <h3>Positive GEX</h3>
                <h2>{positive_gex}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # High confidence setups
        st.subheader("High Confidence Setups")
        
        high_conf_setups = filtered_df[filtered_df['confidence_score'] >= 80].head(5)
        
        if not high_conf_setups.empty:
            for _, setup in high_conf_setups.iterrows():
                confidence = setup['confidence_score']
                
                st.markdown(f"""
                <div class="setup-card">
                    <h4>{setup['symbol']} - {confidence}% Confidence</h4>
                    <p><strong>Setup:</strong> {setup['setup_type'].replace('_', ' ')}</p>
                    <p><strong>Entry:</strong> ${setup['entry_price']:.2f} | <strong>Target:</strong> ${setup['target_price']:.2f}</p>
                    <p><strong>GEX:</strong> {setup['net_gex']:.2f} | <strong>Regime:</strong> {setup['market_regime']}</p>
                    <p><strong>Position Size:</strong> {setup['position_size']}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No high-confidence setups match your filters")
        
        # Charts
        st.subheader("Analysis")
        
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            # GEX distribution
            positive_count = len(filtered_df[filtered_df['net_gex'] > 0])
            negative_count = len(filtered_df[filtered_df['net_gex'] < 0])
            
            fig_pie = go.Figure(data=[go.Pie(
                labels=['Positive GEX', 'Negative GEX'],
                values=[positive_count, negative_count],
                marker_colors=['#666666', '#999999']
            )])
            
            fig_pie.update_layout(
                title="GEX Distribution",
                height=300,
                paper_bgcolor='white',
                plot_bgcolor='white'
            )
            
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with chart_col2:
            # Confidence distribution
            high = len(filtered_df[filtered_df['confidence_score'] >= 80])
            medium = len(filtered_df[(filtered_df['confidence_score'] >= 60) & (filtered_df['confidence_score'] < 80)])
            low = len(filtered_df[filtered_df['confidence_score'] < 60])
            
            fig_bar = go.Figure(data=[go.Bar(
                x=['High (80%+)', 'Medium (60-79%)', 'Low (<60%)'],
                y=[high, medium, low],
                marker_color=['#333333', '#666666', '#999999']
            )])
            
            fig_bar.update_layout(
                title="Confidence Distribution",
                height=300,
                paper_bgcolor='white',
                plot_bgcolor='white'
            )
            
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # Complete data table
        st.subheader("All Recommendations")
        
        # Format for display
        display_df = filtered_df.copy()
        if 'confidence_score' in display_df.columns:
            display_df['confidence_score'] = display_df['confidence_score'].apply(lambda x: f"{x}%")
        if 'entry_price' in display_df.columns:
            display_df['entry_price'] = display_df['entry_price'].apply(lambda x: f"${x:.2f}")
        if 'target_price' in display_df.columns:
            display_df['target_price'] = display_df['target_price'].apply(lambda x: f"${x:.2f}")
        if 'created_timestamp' in display_df.columns:
            display_df['created_timestamp'] = pd.to_datetime(display_df['created_timestamp']).dt.strftime('%H:%M:%S')
        
        st.dataframe(display_df, use_container_width=True)
        
        # Download button
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            "Download CSV",
            csv,
            f"gex_recommendations_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            "text/csv"
        )
        
    else:
        st.error("No data available")

if __name__ == "__main__":
    main()
