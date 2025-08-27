# Query recommendations - REMOVE DATE FILTER (this was the issue!)
            st.write("📡 Querying recommendations table...")
            cursor.execute("""
                SELECT * FROM quant_projects.gex_trading.gex_recommendations
                ORDER BY created_timestamp DESC
                LIMIT 50
            """)
            
            recommendations = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            recommendations_df = pd.DataFrame(recommendations, columns=columns)
            
            st.write(f"📊 Found {len(recommendations_df)} recommendations from Databricks")
                    """
🚀 GEX Trading Dashboard - Simplified Working Version
Based on your working foundation, simplified for reliability
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

# Simple, reliable CSS
st.markdown("""
<style>
    .main {
        padding-top: 1rem;
    }
    
    .metric-container {
        background: linear-gradient(90deg, #667eea, #764ba2);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .setup-card {
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid;
    }
    
    .setup-high {
        background: #e8f5e8;
        border-left-color: #28a745;
    }
    
    .setup-medium {
        background: #fff3cd;
        border-left-color: #ffc107;
    }
    
    .setup-low {
        background: #f8d7da;
        border-left-color: #dc3545;
    }
</style>
""", unsafe_allow_html=True)

def load_databricks_data():
    """Load data from Databricks tables - ALWAYS RETURN DATA"""
    
    # Always show debug info
    st.write("🔍 **Debug Info:**")
    st.write(f"- DATABRICKS_AVAILABLE: {DATABRICKS_AVAILABLE}")
    st.write(f"- Secrets available: {'databricks' in st.secrets}")
    
    try:
        if DATABRICKS_AVAILABLE and "databricks" in st.secrets:
            st.write("📡 Attempting Databricks connection...")
            
            connection = sql.connect(
                server_hostname=st.secrets["databricks"]["server_hostname"],
                http_path=st.secrets["databricks"]["http_path"],
                access_token=st.secrets["databricks"]["access_token"]
            )
            
            cursor = connection.cursor()
            
            # Query recommendations
            cursor.execute("""
                SELECT * FROM quant_projects.gex_trading.gex_recommendations
                WHERE created_timestamp >= current_timestamp() - INTERVAL 24 HOURS
                ORDER BY confidence_score DESC
                LIMIT 20
            """)
            
            recommendations = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            recommendations_df = pd.DataFrame(recommendations, columns=columns)
            
            cursor.close()
            connection.close()
            
            st.success("✅ Connected to Databricks successfully!")
            st.write(f"📊 Found {len(recommendations_df)} recommendations")
            
            return {
                'recommendations': recommendations_df,
                'status': 'connected',
                'message': 'Connected to Databricks'
            }
            
    except Exception as e:
        st.warning(f"⚠️ Databricks connection failed: {str(e)}")
        st.write("🔄 Loading fallback data...")
    
    # ALWAYS create fallback data - this was the issue!
    st.info("📋 Loading sample data for demonstration")
    
    recommendations = pd.DataFrame([
        {
            'symbol': 'AMC',
            'setup_type': 'SQUEEZE_PLAY',
            'confidence_score': 92,
            'entry_price': 15.45,
            'target_price': 18.50,
            'net_gex': -1.2e9,
            'market_regime': 'NEGATIVE_GEX',
            'position_size': '3%',
            'created_timestamp': datetime.now() - timedelta(hours=2)
        },
        {
            'symbol': 'TSLA', 
            'setup_type': 'PREMIUM_SELLING',
            'confidence_score': 88,
            'entry_price': 245.30,
            'target_price': 240.00,
            'net_gex': 2.1e9,
            'market_regime': 'POSITIVE_GEX',
            'position_size': '2.5%',
            'created_timestamp': datetime.now() - timedelta(hours=1)
        },
        {
            'symbol': 'PLTR',
            'setup_type': 'GAMMA_FLIP_PLAY', 
            'confidence_score': 75,
            'entry_price': 28.90,
            'target_price': 32.00,
            'net_gex': -0.3e9,
            'market_regime': 'NEUTRAL_GEX',
            'position_size': '2%',
            'created_timestamp': datetime.now() - timedelta(hours=3)
        },
        {
            'symbol': 'GME',
            'setup_type': 'SQUEEZE_PLAY',
            'confidence_score': 89,
            'entry_price': 25.80,
            'target_price': 30.00,
            'net_gex': -0.8e9,
            'market_regime': 'NEGATIVE_GEX',
            'position_size': '2.5%',
            'created_timestamp': datetime.now() - timedelta(hours=4)
        },
        {
            'symbol': 'NVDA',
            'setup_type': 'PREMIUM_SELLING',
            'confidence_score': 82,
            'entry_price': 890.50,
            'target_price': 880.00,
            'net_gex': 1.5e9,
            'market_regime': 'POSITIVE_GEX',
            'position_size': '1.5%',
            'created_timestamp': datetime.now() - timedelta(hours=5)
        }
    ])
    
    st.write(f"✅ Fallback data loaded: {len(recommendations)} recommendations")
    
    return {
        'recommendations': recommendations,
        'status': 'fallback',
        'message': 'Using sample data - add Databricks secrets for live connection'
    }

def main():
    # Header
    st.title("⚡ GEX Master Pro")
    st.subheader("Live Databricks Pipeline Dashboard")
    
    # Load data
    data = load_databricks_data()
    recommendations_df = data['recommendations']
    
    # Connection status
    if data['status'] == 'connected':
        st.success("✅ Connected to Databricks")
    else:
        st.info("🔧 Development Mode - " + data['message'])
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Controls")
        
        # Refresh button
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()
        
        st.markdown("---")
        
        # Filters
        min_confidence = st.slider("Min Confidence %", 0, 100, 70)
        
        setup_types = st.multiselect(
            "Setup Types",
            ['SQUEEZE_PLAY', 'PREMIUM_SELLING', 'GAMMA_FLIP_PLAY'],
            default=['SQUEEZE_PLAY', 'PREMIUM_SELLING', 'GAMMA_FLIP_PLAY']
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
        st.subheader("📊 Key Metrics")
        
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
        st.subheader("🎯 High Confidence Setups")
        
        high_conf_setups = filtered_df[filtered_df['confidence_score'] >= 80].head(3)
        
        if not high_conf_setups.empty:
            for _, setup in high_conf_setups.iterrows():
                confidence = setup['confidence_score']
                
                if confidence >= 85:
                    card_class = 'setup-high'
                elif confidence >= 75:
                    card_class = 'setup-medium'  
                else:
                    card_class = 'setup-low'
                
                st.markdown(f"""
                <div class="{card_class} setup-card">
                    <h4>{setup['symbol']} - {confidence}% Confidence</h4>
                    <p><strong>Setup:</strong> {setup['setup_type'].replace('_', ' ')}</p>
                    <p><strong>Entry:</strong> ${setup['entry_price']:.2f} | <strong>Target:</strong> ${setup['target_price']:.2f}</p>
                    <p><strong>GEX:</strong> {setup['net_gex']/1e9:.2f}B | <strong>Regime:</strong> {setup['market_regime'].replace('_', ' ')}</p>
                    <p><strong>Position Size:</strong> {setup['position_size']}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No high-confidence setups match your filters")
        
        # Charts
        st.subheader("📈 Analysis")
        
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            # GEX distribution
            positive_count = len(filtered_df[filtered_df['net_gex'] > 0])
            negative_count = len(filtered_df[filtered_df['net_gex'] < 0])
            
            fig_pie = go.Figure(data=[go.Pie(
                labels=['Positive GEX', 'Negative GEX'],
                values=[positive_count, negative_count],
                marker_colors=['#28a745', '#dc3545']
            )])
            
            fig_pie.update_layout(
                title="GEX Distribution",
                height=300
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
                marker_color=['#28a745', '#ffc107', '#dc3545']
            )])
            
            fig_bar.update_layout(
                title="Confidence Distribution",
                height=300
            )
            
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # Complete data table
        st.subheader("💼 All Recommendations")
        
        # Format for display
        display_df = filtered_df.copy()
        display_df['confidence_score'] = display_df['confidence_score'].apply(lambda x: f"{x}%")
        display_df['entry_price'] = display_df['entry_price'].apply(lambda x: f"${x:.2f}")
        display_df['target_price'] = display_df['target_price'].apply(lambda x: f"${x:.2f}")
        display_df['net_gex'] = display_df['net_gex'].apply(lambda x: f"{x/1e9:.2f}B")
        display_df['created_timestamp'] = display_df['created_timestamp'].dt.strftime('%H:%M:%S')
        
        st.dataframe(
            display_df[[
                'symbol', 'setup_type', 'confidence_score', 'entry_price', 
                'target_price', 'net_gex', 'market_regime', 'created_timestamp'
            ]],
            use_container_width=True
        )
        
        # Download button
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            "📥 Download CSV",
            csv,
            f"gex_recommendations_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            "text/csv"
        )
        
    else:
        st.error("No data available")

if __name__ == "__main__":
    main()
