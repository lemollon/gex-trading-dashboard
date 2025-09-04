import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import yfinance as yf

# Page configuration
st.set_page_config(
    page_title="GEX Trading Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and header
st.title("🎯 Gamma Exposure (GEX) Trading Dashboard")
st.markdown("**Real-time analysis of options gamma exposure for strategic positioning**")

# Sidebar controls
st.sidebar.header("📋 Configuration")
symbol = st.sidebar.selectbox("Select Symbol", ["SPY", "QQQ", "IWM", "AAPL", "TSLA"], index=0)
max_dte = st.sidebar.slider("Maximum Days to Expiration", 7, 60, 30)

# Test data generation function (fallback if API fails)
def generate_test_gex_data():
    """Generate sample GEX data for testing purposes"""
    current_price = 450.0
    strikes = np.arange(400, 500, 2.5)
    
    # Simulate GEX profile with realistic patterns
    gex_values = []
    for strike in strikes:
        # Calls create positive GEX, puts create negative GEX
        distance = abs(strike - current_price)
        
        if strike > current_price:  # Call wall
            gex = np.exp(-distance/10) * np.random.normal(50e6, 20e6)
        else:  # Put wall  
            gex = -np.exp(-distance/10) * np.random.normal(50e6, 20e6)
        
        gex_values.append(gex)
    
    df = pd.DataFrame({
        'strike': strikes,
        'gex': gex_values
    })
    
    # Calculate cumulative GEX
    df['cumulative_gex'] = df['gex'].cumsum()
    
    return df, current_price

# Simple GEX calculation function
def calculate_simple_gex(symbol, max_dte=30):
    """
    Simplified GEX calculation with better error handling
    """
    try:
        st.info(f"📡 Fetching {symbol} options data...")
        
        ticker = yf.Ticker(symbol)
        current_price = ticker.history(period="1d")['Close'].iloc[-1]
        
        # Get first few expiration dates
        exp_dates = ticker.options[:3]  # Limit to 3 expirations for speed
        
        all_gex_data = []
        
        for exp_date in exp_dates:
            try:
                # Calculate days to expiration
                exp_dt = datetime.strptime(exp_date, '%Y-%m-%d')
                dte = (exp_dt - datetime.now()).days
                
                if dte > max_dte:
                    continue
                
                # Get options chain
                chain = ticker.option_chain(exp_date)
                calls = chain.calls
                puts = chain.puts
                
                # Process calls (positive GEX)
                for _, call in calls.iterrows():
                    if call['openInterest'] > 0:
                        # Simplified gamma calculation
                        gamma = 0.01 * np.exp(-abs(call['strike'] - current_price) / 20)
                        gex = current_price * gamma * call['openInterest'] * 100
                        
                        all_gex_data.append({
                            'strike': call['strike'],
                            'gex': gex,
                            'type': 'call'
                        })
                
                # Process puts (negative GEX)
                for _, put in puts.iterrows():
                    if put['openInterest'] > 0:
                        # Simplified gamma calculation
                        gamma = 0.01 * np.exp(-abs(put['strike'] - current_price) / 20)
                        gex = -current_price * gamma * put['openInterest'] * 100
                        
                        all_gex_data.append({
                            'strike': put['strike'],
                            'gex': gex,
                            'type': 'put'
                        })
                        
            except Exception as e:
                st.warning(f"⚠️ Error processing {exp_date}: {str(e)}")
                continue
        
        if not all_gex_data:
            raise Exception("No valid options data found")
        
        # Convert to DataFrame and aggregate by strike
        gex_df = pd.DataFrame(all_gex_data)
        gex_profile = gex_df.groupby('strike')['gex'].sum().reset_index()
        gex_profile = gex_profile.sort_values('strike')
        gex_profile['cumulative_gex'] = gex_profile['gex'].cumsum()
        
        return gex_profile, current_price
        
    except Exception as e:
        st.error(f"❌ Error fetching real data: {str(e)}")
        st.info("🔄 Using test data instead...")
        return generate_test_gex_data()

# Main execution
if st.button("🚀 Calculate GEX Profile") or 'gex_data' not in st.session_state:
    with st.spinner("Calculating gamma exposure..."):
        gex_data, current_price = calculate_simple_gex(symbol, max_dte)
        st.session_state.gex_data = gex_data
        st.session_state.current_price = current_price

# Display results if data exists
if 'gex_data' in st.session_state:
    gex_data = st.session_state.gex_data
    current_price = st.session_state.current_price
    
    # Calculate key metrics
    net_gex = gex_data['gex'].sum()
    gamma_flip = gex_data.loc[gex_data['cumulative_gex'].abs().idxmin(), 'strike']
    
    # Key metrics display
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Current Price", f"${current_price:.2f}")
    
    with col2:
        st.metric("Net GEX", f"{net_gex/1e9:.2f}B")
    
    with col3:
        st.metric("Gamma Flip", f"${gamma_flip:.2f}")
    
    with col4:
        distance_to_flip = ((current_price - gamma_flip) / current_price) * 100
        st.metric("Distance to Flip", f"{distance_to_flip:.2f}%")
    
    # GEX Profile Chart
    st.subheader("📊 Gamma Exposure Profile")
    
    fig = go.Figure()
    
    # GEX bars
    colors = ['red' if x < 0 else 'green' for x in gex_data['gex']]
    fig.add_trace(go.Bar(
        x=gex_data['strike'],
        y=gex_data['gex']/1e6,  # Convert to millions
        name='GEX',
        marker_color=colors,
        opacity=0.7
    ))
    
    # Current price line
    fig.add_vline(
        x=current_price,
        line_dash="dash",
        line_color="blue",
        annotation_text=f"Current: ${current_price:.2f}"
    )
    
    # Gamma flip line
    fig.add_vline(
        x=gamma_flip,
        line_dash="dot",
        line_color="orange",
        annotation_text=f"Flip: ${gamma_flip:.2f}"
    )
    
    fig.update_layout(
        title=f"{symbol} Gamma Exposure by Strike",
        xaxis_title="Strike Price",
        yaxis_title="GEX (Millions)",
        height=500,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Walls identification
    st.subheader("🏗️ Support & Resistance Levels")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Call Walls (Resistance)**")
        call_walls = gex_data[gex_data['gex'] > 0].nlargest(3, 'gex')
        for _, wall in call_walls.iterrows():
            st.write(f"${wall['strike']:.2f}: {wall['gex']/1e6:.1f}M GEX")
    
    with col2:
        st.write("**Put Walls (Support)**")
        put_walls = gex_data[gex_data['gex'] < 0].nsmallest(3, 'gex')
        for _, wall in put_walls.iterrows():
            st.write(f"${wall['strike']:.2f}: {wall['gex']/1e6:.1f}M GEX")
    
    # Trading setup recommendations
    st.subheader("🎯 Setup Recommendations")
    
    recommendations = []
    
    # Negative GEX squeeze setup
    if net_gex < -1e9 and distance_to_flip < -0.5:
        recommendations.append({
            'Strategy': 'NEGATIVE GEX SQUEEZE',
            'Action': 'Long Calls',
            'Confidence': 'HIGH',
            'Rationale': f'Net GEX {net_gex/1e9:.2f}B, price {distance_to_flip:.2f}% below flip'
        })
    
    # Positive GEX breakdown setup
    elif net_gex > 2e9 and abs(distance_to_flip) < 0.3:
        recommendations.append({
            'Strategy': 'POSITIVE GEX BREAKDOWN',
            'Action': 'Long Puts',
            'Confidence': 'HIGH',
            'Rationale': f'Net GEX {net_gex/1e9:.2f}B, price near flip point'
        })
    
    # Iron condor setup
    elif net_gex > 1e9:
        recommendations.append({
            'Strategy': 'IRON CONDOR',
            'Action': 'Short Strangles at Walls',
            'Confidence': 'MEDIUM',
            'Rationale': f'Positive GEX environment, Net GEX {net_gex/1e9:.2f}B'
        })
    else:
        recommendations.append({
            'Strategy': 'NO CLEAR SETUP',
            'Action': 'Wait for Better Conditions',
            'Confidence': 'LOW',
            'Rationale': 'Mixed signals, unclear gamma regime'
        })
    
    if recommendations:
        rec_df = pd.DataFrame(recommendations)
        st.dataframe(rec_df, use_container_width=True)
    
    # Raw data table
    if st.checkbox("Show Raw GEX Data"):
        st.subheader("📋 Raw Data")
        display_df = gex_data.copy()
        display_df['gex_millions'] = display_df['gex'] / 1e6
        st.dataframe(display_df[['strike', 'gex_millions', 'cumulative_gex']], use_container_width=True)

else:
    st.info("👆 Click 'Calculate GEX Profile' to begin analysis")

# Footer
st.markdown("---")
st.markdown("**Disclaimer**: This is for educational purposes only. Not financial advice.")
