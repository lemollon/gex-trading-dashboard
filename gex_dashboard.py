# gex_dashboard.py
"""
Simple GEX Trading Dashboard
Tests Databricks connection and displays basic GEX data
"""

import streamlit as st
import pandas as pd
from databricks import sql
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json

# Page config
st.set_page_config(
    page_title="GEX Trading Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Connection function
@st.cache_resource
def get_databricks_connection():
    """Get Databricks connection with error handling"""
    try:
        connection = sql.connect(
            server_hostname=st.secrets["databricks"]["server_hostname"],
            http_path=st.secrets["databricks"]["http_path"],
            access_token=st.secrets["databricks"]["access_token"]
        )
        return connection
    except Exception as e:
        st.error(f"❌ Databricks connection failed: {str(e)}")
        return None

# Data fetching functions
@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_scan_results():
    """Fetch latest GEX scan results"""
    connection = get_databricks_connection()
    if not connection:
        return pd.DataFrame()
    
    try:
        cursor = connection.cursor()
        query = """
        SELECT 
            symbol,
            current_price,
            gamma_flip_point,
            net_gex,
            setup_type,
            confidence_score,
            days_to_expiration,
            distance_to_flip_pct,
            scan_timestamp
        FROM gex_trading.scan_results 
        WHERE scan_timestamp >= current_timestamp() - interval 4 hours
        ORDER BY confidence_score DESC
        LIMIT 20
        """
        
        cursor.execute(query)
        columns = [desc[0] for desc in cursor.description]
        results = cursor.fetchall()
        
        if results:
            df = pd.DataFrame(results, columns=columns)
            return df
        else:
            return pd.DataFrame()
            
    except Exception as e:
        st.error(f"Data fetch error: {str(e)}")
        return pd.DataFrame()
    finally:
        if cursor:
            cursor.close()

@st.cache_data(ttl=300)
def fetch_portfolio_stats():
    """Fetch basic portfolio statistics"""
    connection = get_databricks_connection()
    if not connection:
        return {}
    
    try:
        cursor = connection.cursor()
        
        # Try to get basic stats from scan results
        query = """
        SELECT 
            COUNT(*) as total_symbols,
            COUNT(CASE WHEN confidence_score >= 70 THEN 1 END) as high_confidence,
            AVG(confidence_score) as avg_confidence,
            MAX(analysis_timestamp) as last_scan
        FROM quant_projects.gex_trading.scheduled_pipeline_results 
        WHERE analysis_date >= current_date() - interval 1 day
        """
        
        cursor.execute(query)
        result = cursor.fetchone()
        
        if result:
            return {
                'total_symbols': result[0] or 0,
                'high_confidence': result[1] or 0,
                'avg_confidence': float(result[2] or 0),
                'last_scan': result[3]
            }
        else:
            return {}
            
    except Exception as e:
        st.warning(f"Stats fetch error: {str(e)}")
        return {}
    finally:
        if cursor:
            cursor.close()

def create_gex_scatter_plot(df):
    """Create scatter plot of confidence vs distance to flip"""
    if df.empty:
        return go.Figure().add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
    
    fig = px.scatter(
        df, 
        x='distance_to_flip_pct',
        y='confidence_score',
        color='setup_type',
        size='days_to_expiration',
        hover_name='symbol',
        hover_data=['current_price', 'gamma_flip_point'],
        title="GEX Setup Quality: Confidence vs Distance to Flip"
    )
    
    fig.update_layout(
        height=400,
        xaxis_title="Distance to Gamma Flip (%)",
        yaxis_title="Confidence Score (%)",
        plot_bgcolor='white'
    )
    
    return fig

def create_setup_distribution(df):
    """Create bar chart of setup types"""
    if df.empty:
        return go.Figure().add_annotation(
            text="No setup data available",
            xref="paper", yref="paper", 
            x=0.5, y=0.5, showarrow=False
        )
    
    setup_counts = df['setup_type'].value_counts()
    
    fig = go.Figure(data=[
        go.Bar(
            x=setup_counts.index,
            y=setup_counts.values,
            marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(setup_counts)],
            text=setup_counts.values,
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title="Trading Setup Distribution",
        xaxis_title="Setup Type",
        yaxis_title="Count",
        height=300,
        showlegend=False,
        plot_bgcolor='white'
    )
    
    return fig

# Main dashboard
def main():
    # Header
    st.title("📊 GEX Trading Dashboard")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Dashboard Controls")
        
        # Connection status
        connection = get_databricks_connection()
        if connection:
            st.markdown('<div class="success-box">✅ Connected to Databricks</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="warning-box">❌ Databricks connection failed</div>', unsafe_allow_html=True)
            st.stop()
        
        # Refresh controls
        st.subheader("Data Refresh")
        if st.button("🔄 Refresh Data", type="primary"):
            st.cache_data.clear()
            st.experimental_rerun()
        
        # Filters
        st.subheader("Filters")
        min_confidence = st.slider("Minimum Confidence", 0, 100, 65)
        max_dte = st.slider("Max Days to Expiration", 1, 30, 7)
    
    # Main content
    col1, col2, col3, col4 = st.columns(4)
    
    # Fetch data
    with st.spinner("Loading GEX data..."):
        scan_df = fetch_scan_results()
        portfolio_stats = fetch_portfolio_stats()
    
    # Display metrics
    with col1:
        st.metric(
            "Total Symbols Scanned",
            portfolio_stats.get('total_symbols', 0)
        )
    
    with col2:
        st.metric(
            "High Confidence Setups",
            portfolio_stats.get('high_confidence', 0)
        )
    
    with col3:
        st.metric(
            "Avg Confidence Score",
            f"{portfolio_stats.get('avg_confidence', 0):.1f}%"
        )
    
    with col4:
        last_scan = portfolio_stats.get('last_scan')
        if last_scan:
            st.metric("Last Scan", str(last_scan)[:16])
        else:
            st.metric("Last Scan", "N/A")
    
    st.markdown("---")
    
    # Filter data
    if not scan_df.empty:
        filtered_df = scan_df[
            (scan_df['confidence_score'] >= min_confidence) & 
            (scan_df['days_to_expiration'] <= max_dte)
        ]
    else:
        filtered_df = pd.DataFrame()
    
    # Charts section
    if not filtered_df.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            fig_scatter = create_gex_scatter_plot(filtered_df)
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        with col2:
            fig_bar = create_setup_distribution(filtered_df)
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # Data table
        st.subheader("📋 Current Trading Setups")
        
        # Display formatted data
        display_df = filtered_df.copy()
        if 'current_price' in display_df.columns:
            display_df['current_price'] = display_df['current_price'].apply(lambda x: f"${x:.2f}")
        if 'gamma_flip_point' in display_df.columns:
            display_df['gamma_flip_point'] = display_df['gamma_flip_point'].apply(lambda x: f"${x:.2f}")
        if 'confidence_score' in display_df.columns:
            display_df['confidence_score'] = display_df['confidence_score'].apply(lambda x: f"{x:.1f}%")
        if 'distance_to_flip_pct' in display_df.columns:
            display_df['distance_to_flip_pct'] = display_df['distance_to_flip_pct'].apply(lambda x: f"{x:.2f}%")
        if 'net_gex' in display_df.columns:
            display_df['net_gex'] = display_df['net_gex'].apply(lambda x: f"{x/1e9:.2f}B" if abs(x) > 1e6 else f"{x/1e6:.1f}M")
        
        st.dataframe(
            display_df[['symbol', 'setup_type', 'confidence_score', 'current_price', 
                       'gamma_flip_point', 'distance_to_flip_pct', 'category', 'priority']],
            use_container_width=True
        )
        
        # Export data
        if st.button("📥 Download Current Data"):
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"gex_setups_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
    
    else:
        st.warning("⚠️ No trading setups found matching your criteria")
        
        # Show debug info
        with st.expander("🔍 Debug Information"):
            st.write("Raw scan results shape:", scan_df.shape if not scan_df.empty else "No data")
            if not scan_df.empty:
                st.write("Sample data:")
                st.dataframe(scan_df.head(3))
            
            st.write("Portfolio stats:", portfolio_stats)
    
    # Footer
    st.markdown("---")
    st.markdown(
        f"🕒 Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
        f"📡 Data cached for 5 minutes"
    )

if __name__ == "__main__":
    main()
