"""
GEX Trading Dashboard - CORRECTED VERSION
Connects to your actual pipeline table with real data
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time

# Try to import Databricks connector
try:
    from databricks import sql
    DATABRICKS_AVAILABLE = True
except ImportError:
    DATABRICKS_AVAILABLE = False
    st.error("databricks-sql-connector not installed. Run: pip install databricks-sql-connector")

# Page config
st.set_page_config(
    page_title="GEX Master Pro",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS styling
st.markdown("""
<style>
    .main {
        padding-top: 1rem;
    }
    
    .stAlert > div {
        padding: 1rem;
        border-radius: 10px;
    }
    
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .setup-card {
        background: white;
        border: 1px solid #e0e0e0;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .high-confidence {
        border-left: 5px solid #28a745;
    }
    
    .medium-confidence {
        border-left: 5px solid #ffc107;
    }
    
    .low-confidence {
        border-left: 5px solid #dc3545;
    }
    
    .status-connected {
        color: #28a745;
        font-weight: bold;
    }
    
    .status-disconnected {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def init_databricks_connection():
    """Initialize Databricks connection"""
    try:
        if not DATABRICKS_AVAILABLE:
            return None
            
        if "databricks" not in st.secrets:
            st.error("Databricks secrets not configured")
            return None
            
        connection = sql.connect(
            server_hostname=st.secrets["databricks"]["server_hostname"],
            http_path=st.secrets["databricks"]["http_path"],
            access_token=st.secrets["databricks"]["access_token"]
        )
        return connection
    except Exception as e:
        st.error(f"Databricks connection failed: {e}")
        return None

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_gex_data():
    """Load GEX data from your ACTUAL pipeline table - REAL DATA ONLY"""
    connection = init_databricks_connection()
    
    if not connection:
        return {
            'data': pd.DataFrame(),
            'status': 'disconnected',
            'message': 'Databricks connection failed - no mock data'
        }
    
    try:
        cursor = connection.cursor()
        
        # Query YOUR ACTUAL table - get ALL records from recent runs
        query = """
        SELECT 
            run_id,
            symbol,
            structure_type,
            confidence_score,
            spot_price,
            gamma_flip_point,
            distance_to_flip_pct,
            recommendation,
            category,
            priority,
            created_at as analysis_timestamp,
            analysis_date
        FROM quant_projects.gex_trading.scheduled_pipeline_results
        WHERE analysis_date >= current_date() - INTERVAL 30 DAYS
        ORDER BY analysis_date DESC, confidence_score DESC
        LIMIT 1000
        """
        
        cursor.execute(query)
        results = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        
        df = pd.DataFrame(results, columns=columns)
        
        cursor.close()
        
        # Debug information - show what we actually got
        debug_info = ""
        if not df.empty:
            debug_info = f"""
            **Debug Info:**
            - Total records: {len(df)}
            - Date range: {df['analysis_date'].min()} to {df['analysis_date'].max()}
            - Categories: {df['category'].value_counts().to_dict()}
            - Confidence range: {df['confidence_score'].min()}-{df['confidence_score'].max()}
            - Unique runs: {df['run_id'].nunique()}
            """
        
        return {
            'data': df,
            'status': 'connected',
            'message': f'Connected - Loaded {len(df)} real records from your pipeline',
            'debug_info': debug_info
        }
        
    except Exception as e:
        st.error(f"Query failed: {e}")
        return {
            'data': pd.DataFrame(),
            'status': 'error',
            'message': f'Database query error: {e}'
        }

# Mock data function removed - using REAL DATA ONLY

def format_confidence_class(confidence):
    """Return CSS class based on confidence score"""
    if confidence >= 80:
        return "high-confidence"
    elif confidence >= 60:
        return "medium-confidence"
    else:
        return "low-confidence"

def main():
    # Header
    st.title("🚀 GEX Master Pro")
    st.subheader("Live Pipeline Dashboard - Real Databricks Data")
    
    # Load data
    data_result = load_gex_data()
    
    if isinstance(data_result, dict):
        df = data_result['data']
        status = data_result['status']
        message = data_result['message']
        debug_info = data_result.get('debug_info', '')
    else:
        df = data_result
        status = 'unknown'
        message = 'Data loaded'
        debug_info = ''
    
    # Status indicator with debug info
    if status == 'connected':
        st.success(f"✅ {message}")
        if debug_info:
            with st.expander("🔍 Debug Information", expanded=False):
                st.markdown(debug_info)
    elif status == 'error':
        st.warning(f"⚠️ {message}")
    else:
        st.info(f"ℹ️ {message}")
    
    # Sidebar controls
    with st.sidebar:
        st.header("🎛️ Controls")
        
        if st.button("🔄 Refresh Data", type="primary"):
            st.cache_data.clear()
            st.rerun()
        
        st.markdown("---")
        
        # Filters with real-time feedback
        min_confidence = st.slider("Minimum Confidence %", 0, 100, 70)
        
        # Show live filtering results
        if not df.empty:
            total_before_filter = len(df)
            after_confidence_filter = len(df[df['confidence_score'] >= min_confidence])
            st.caption(f"📊 {after_confidence_filter} of {total_before_filter} setups meet confidence threshold")
        
        if not df.empty:
            # Apply confidence filter first for other filters
            confidence_filtered_df = df[df['confidence_score'] >= min_confidence]
            
            available_types = confidence_filtered_df['structure_type'].unique().tolist() if not confidence_filtered_df.empty else []
            setup_types = st.multiselect(
                "Setup Types",
                available_types,
                default=available_types
            )
            
            symbols = st.multiselect(
                "Symbols",
                sorted(confidence_filtered_df['symbol'].unique().tolist()) if not confidence_filtered_df.empty else [],
                default=sorted(confidence_filtered_df['symbol'].unique().tolist())[:10] if not confidence_filtered_df.empty else []
            )
        else:
            setup_types = []
            symbols = []
        
        st.markdown("---")
        
        # Show pipeline stats
        if not df.empty:
            total_runs = df['run_id'].nunique()
            total_symbols = df['symbol'].nunique()
            latest_run = df['analysis_timestamp'].max()
            
            st.markdown("### 📊 Pipeline Stats")
            st.markdown(f"**Runs:** {total_runs}")
            st.markdown(f"**Symbols:** {total_symbols}")
            st.markdown(f"**Latest:** {pd.to_datetime(latest_run).strftime('%m/%d %H:%M')}")
        
        st.markdown("---")
        st.markdown("### 📖 Quick Reference")
        st.markdown("""
        **Your Pipeline Data:**
        - 🎯 Enhanced Strategies
        - 🔍 GEX Conditions  
        - ⚡ Live from Databricks
        
        **Setup Types:**
        - 🚀 Squeeze Play: Negative GEX breakout
        - 💰 Premium Selling: Range bound
        - 📈 Gamma Flip: Regime change
        """)
    
    # Main dashboard
    if df.empty:
        st.error("❌ No real data available from your pipeline")
        st.info("🔧 **Next Steps:**")
        st.markdown("""
        1. **Check Databricks Connection** - Verify secrets are configured
        2. **Run Your Pipeline** - Execute your GEX pipeline in Databricks to generate data  
        3. **Verify Table** - Confirm data exists in `quant_projects.gex_trading.scheduled_pipeline_results`
        4. **Check Date Range** - Pipeline looks for data from last 7 days
        """)
        st.stop()
    
    # Apply all filters
    filtered_df = df[
        (df['confidence_score'] >= min_confidence) &
        (df['structure_type'].isin(setup_types)) &
        (df['symbol'].isin(symbols))
    ].copy()
    
    # Show filtering results
    st.sidebar.success(f"✅ Showing {len(filtered_df)} filtered setups")
    
    if filtered_df.empty:
        st.warning("No data matches your filters. Try adjusting the criteria.")
        return
    
    # Key metrics row
    st.subheader("📊 Pipeline Results")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <h3>Total Setups</h3>
            <h1>{len(filtered_df)}</h1>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        high_conf = len(filtered_df[filtered_df['confidence_score'] >= 85])
        st.markdown(f"""
        <div class="metric-container">
            <h3>High Confidence</h3>
            <h1>{high_conf}</h1>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_conf = filtered_df['confidence_score'].mean()
        st.markdown(f"""
        <div class="metric-container">
            <h3>Avg Confidence</h3>
            <h1>{avg_conf:.1f}%</h1>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        enhanced_strategies = len(filtered_df[filtered_df['category'] == 'ENHANCED_STRATEGY'])
        st.markdown(f"""
        <div class="metric-container">
            <h3>Enhanced Strategies</h3>
            <h1>{enhanced_strategies}</h1>
        </div>
        """, unsafe_allow_html=True)
    
    # Top setups from your actual pipeline
    st.subheader("🎯 High Confidence Setups (From Your Pipeline)")
    
    # Show priority 1 setups first (like your pipeline results)
    priority_setups = filtered_df[filtered_df['priority'] == 1].head(10)
    
    if not priority_setups.empty:
        for _, setup in priority_setups.iterrows():
            confidence_class = format_confidence_class(setup['confidence_score'])
            
            # Calculate distance display like your pipeline
            distance_display = f"{setup['distance_to_flip_pct']:+.2f}%"
            
            st.markdown(f"""
            <div class="setup-card {confidence_class}">
                <h4>{setup['symbol']} - {setup['confidence_score']}% Confidence</h4>
                <div style="display: flex; justify-content: space-between; margin-top: 1rem;">
                    <div>
                        <strong>Setup:</strong> {setup['structure_type'].replace('_', ' ').title()}<br>
                        <strong>Category:</strong> {setup['category']}
                    </div>
                    <div style="text-align: right;">
                        <strong>Spot:</strong> ${setup['spot_price']:.2f}<br>
                        <strong>Flip Point:</strong> ${setup['gamma_flip_point']:.2f}
                    </div>
                </div>
                <div style="margin-top: 1rem;">
                    <strong>Distance:</strong> {distance_display} | 
                    <strong>Priority:</strong> {setup['priority']} | 
                    <strong>Recommendation:</strong> {setup['recommendation']}
                </div>
                <div style="margin-top: 0.5rem; font-size: 0.9em; color: #666;">
                    <strong>Run ID:</strong> {setup['run_id']} | 
                    <strong>Created:</strong> {pd.to_datetime(setup['analysis_timestamp']).strftime('%m/%d %H:%M')}
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No high-priority setups match your current filters")
    
    # Charts section
    st.subheader("📈 Analysis Charts")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Setup type distribution
        setup_counts = filtered_df['structure_type'].value_counts()
        
        fig_pie = go.Figure(data=[go.Pie(
            labels=setup_counts.index,
            values=setup_counts.values,
            marker_colors=['#667eea', '#764ba2', '#f093fb'],
            hole=0.4
        )])
        
        fig_pie.update_layout(
            title="Setup Type Distribution",
            height=300
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Confidence distribution
        confidence_bins = ['High (85%+)', 'Medium (70-84%)', 'Low (<70%)']
        high = len(filtered_df[filtered_df['confidence_score'] >= 85])
        medium = len(filtered_df[(filtered_df['confidence_score'] >= 70) & (filtered_df['confidence_score'] < 85)])
        low = len(filtered_df[filtered_df['confidence_score'] < 70])
        
        fig_bar = go.Figure(data=[go.Bar(
            x=confidence_bins,
            y=[high, medium, low],
            marker_color=['#28a745', '#ffc107', '#dc3545']
        )])
        
        fig_bar.update_layout(
            title="Confidence Score Distribution",
            height=300,
            yaxis_title="Number of Setups"
        )
        
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Symbol performance chart
    if len(filtered_df) > 0:
        st.subheader("📊 Top Symbols by Setup Count")
        
        symbol_counts = filtered_df['symbol'].value_counts().head(15)
        
        fig_symbol = go.Figure(data=[go.Bar(
            x=symbol_counts.index,
            y=symbol_counts.values,
            marker_color='#667eea'
        )])
        
        fig_symbol.update_layout(
            title="Symbols with Most Setups",
            height=400,
            xaxis_title="Symbol",
            yaxis_title="Number of Setups"
        )
        
        st.plotly_chart(fig_symbol, use_container_width=True)
    
    # Pipeline run analysis
    st.subheader("📋 Pipeline Run Analysis")
    
    if not filtered_df.empty:
        run_summary = filtered_df.groupby('run_id').agg({
            'symbol': 'count',
            'confidence_score': 'mean',
            'analysis_timestamp': 'first'
        }).rename(columns={
            'symbol': 'setup_count',
            'confidence_score': 'avg_confidence'
        }).round(1).sort_values('analysis_timestamp', ascending=False)
        
        run_summary['analysis_timestamp'] = pd.to_datetime(run_summary['analysis_timestamp']).dt.strftime('%m/%d %H:%M')
        
        st.dataframe(
            run_summary,
            column_config={
                "setup_count": "Setups Found",
                "avg_confidence": "Avg Confidence %", 
                "analysis_timestamp": "Run Time"
            },
            use_container_width=True
        )
    
    # Complete data table
    st.subheader("📋 All Pipeline Results")
    
    # Format display columns
    display_df = filtered_df.copy()
    display_df['analysis_timestamp'] = pd.to_datetime(display_df['analysis_timestamp']).dt.strftime('%m/%d/%y %H:%M')
    
    # Rename columns for display
    display_df = display_df.rename(columns={
        'structure_type': 'Setup Type',
        'confidence_score': 'Confidence %',
        'spot_price': 'Spot Price',
        'gamma_flip_point': 'Flip Point',
        'distance_to_flip_pct': 'Distance %',
        'analysis_timestamp': 'Created'
    })
    
    st.dataframe(
        display_df[[
            'symbol', 'Setup Type', 'Confidence %', 'Spot Price', 
            'Flip Point', 'Distance %', 'recommendation', 'priority', 'Created'
        ]],
        use_container_width=True,
        hide_index=True
    )
    
    # Download section
    st.subheader("💾 Export Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            "📄 Download CSV",
            csv,
            f"gex_pipeline_results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            "text/csv",
            type="secondary"
        )
    
    with col2:
        json_data = filtered_df.to_json(orient='records', indent=2)
        st.download_button(
            "🔧 Download JSON",
            json_data,
            f"gex_pipeline_results_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
            "application/json",
            type="secondary"
        )
    
    # Footer
    st.markdown("---")
    st.markdown(f"**GEX Master Pro** - Connected to: `quant_projects.gex_trading.scheduled_pipeline_results` | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
