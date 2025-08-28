<li><strong>Correlation Risk:</strong> Reduce position size when trading correlated assets</li>
            <li><strong>VIX Risk:</strong> Exit premium selling positions if VIX spikes above 30</li>
            <li><strong>Earnings Risk:</strong> Avoid holding through earnings announcements</li>
            <li><strong>FOMC Risk:</strong> Reduce position size or avoid trading during Fed meetings</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Main Application
def main():
    """Main application entry point"""
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.title("⚡ GEX Master Pro")
    st.markdown("Complete Gamma Exposure Trading System")
    
    # Sidebar
    with st.sidebar:
        st.subheader("Navigation")
        
        # Navigation tabs
        page = st.radio(
            "Select Page",
            ["Single Symbol Analysis", "Pipeline Dashboard", "Auto Trader", "Market Overview", "Education Center"]
        )
        
        st.markdown("---")
        
        # API Status
        st.subheader("System Status")
        
        # API status
        if st.session_state.api_status['status'] == 'online':
            st.markdown(f"""
            <div style="display: flex; align-items: center">
                <div class="status-indicator status-online"></div>
                <div>API: <span class="status-connected">Online</span></div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="display: flex; align-items: center">
                <div class="status-indicator status-offline"></div>
                <div>API: <span class="status-disconnected">Offline</span></div>
            </div>
            """, unsafe_allow_html=True)
        
        # Database status
        db_connection = init_databricks_connection()
        if db_connection:
            st.markdown(f"""
            <div style="display: flex; align-items: center">
                <div class="status-indicator status-online"></div>
                <div>Database: <span class="status-connected">Connected</span></div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="display: flex; align-items: center">
                <div class="status-indicator status-offline"></div>
                <div>Database: <span class="status-disconnected">Disconnected</span></div>
            </div>
            """, unsafe_allow_html=True)
        
        # Auto-trader status
        if st.session_state.auto_trader['enabled']:
            st.markdown(f"""
            <div style="display: flex; align-items: center">
                <div class="status-indicator status-online"></div>
                <div>Auto-Trader: <span class="status-connected">Enabled</span></div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="display: flex; align-items: center">
                <div class="status-indicator status-offline"></div>
                <div>Auto-Trader: <span class="status-disconnected">Disabled</span></div>
            </div>
            """, unsafe_allow_html=True)
        
        # Test API connection
        if st.button("Test API Connection"):
            with st.spinner("Testing API connection..."):
                try:
                    test_symbol = "SPY"
                    response = requests.get(
                        f"{api_config['base_url']}/gex/latest",
                        params={
                            'ticker': test_symbol,
                            'username': api_config['username'],
                            'format': 'json'
                        },
                        timeout=10
                    )
                    
                    if response.status_code == 200:
                        st.session_state.api_status['status'] = 'online'
                        st.session_state.api_status['last_checked'] = datetime.now()
                        st.session_state.api_status['successes'] += 1
                        st.success("✅ API connection successful!")
                    else:
                        st.session_state.api_status['status'] = 'offline'
                        st.session_state.api_status['last_checked'] = datetime.now()
                        st.session_state.api_status['failures'] += 1
                        st.error(f"❌ API returned status code: {response.status_code}")
                        
                except Exception as e:
                    st.session_state.api_status['status'] = 'offline'
                    st.session_state.api_status['last_checked'] = datetime.now()
                    st.session_state.api_status['failures'] += 1
                    st.error(f"❌ API connection failed: {str(e)}")
        
        st.markdown("---")
        
        # Discord Webhook section
        st.subheader("Discord Alerts")
        webhook_url = st.text_input(
            "Discord Webhook URL",
            value=st.session_state.discord_webhook['url'],
            type="password",
            help="Enter your Discord webhook URL for alerts"
        )
        
        # Update session state
        st.session_state.discord_webhook['url'] = webhook_url
        
        # Webhook enabled toggle
        webhook_enabled = st.checkbox(
            "Enable Discord Alerts",
            value=st.session_state.discord_webhook['enabled'],
            help="Send alerts to Discord for high-confidence setups"
        )
        
        # Update session state
        st.session_state.discord_webhook['enabled'] = webhook_enabled
        
        # Minimum confidence for alerts
        webhook_min_confidence = st.slider(
            "Min. Alert Confidence",
            min_value=60,
            max_value=100,
            value=st.session_state.discord_webhook['min_confidence'],
            step=5,
            help="Minimum confidence score to trigger alerts"
        )
        
        # Update session state
        st.session_state.discord_webhook['min_confidence'] = webhook_min_confidence
        
        # Setup types to alert on
        alert_types = st.multiselect(
            "Alert on Setup Types",
            ["squeeze_play", "breakdown_play", "premium_selling", "iron_condor"],
            default=st.session_state.discord_webhook['alert_types'],
            help="Select which setup types to send alerts for"
        )
        
        # Update session state
        st.session_state.discord_webhook['alert_types'] = alert_types
        
        # Test webhook
        if st.button("Test Discord Webhook"):
            if webhook_url and webhook_url.startswith('https://discord.com/api/webhooks/'):
                with st.spinner("Sending test alert..."):
                    success, message = test_discord_webhook(webhook_url)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
            else:
                st.error("Please enter a valid Discord webhook URL")
        
        st.markdown("---")
        
        # Watchlist Management
        st.subheader("Watchlist")
        
        # Add/remove symbols
        custom_symbols = st.text_input(
            "Add Symbols",
            placeholder="SYMBOL1, SYMBOL2, ...",
            help="Enter comma-separated symbols"
        )
        
        if custom_symbols:
            new_symbols = [s.strip().upper() for s in custom_symbols.split(",") if s.strip()]
            st.session_state.watchlist.extend(new_symbols)
            # Remove duplicates
            st.session_state.watchlist = list(set(st.session_state.watchlist))
        
        # Display current watchlist
        if st.session_state.watchlist:
            st.write("Current Watchlist:")
            watchlist_cols = st.columns(3)
            for i, symbol in enumerate(sorted(st.session_state.watchlist)):
                with watchlist_cols[i % 3]:
                    if st.button(f"❌ {symbol}"):
                        st.session_state.watchlist.remove(symbol)
                        st.rerun()
    
    # Main content based on selected page
    if page == "Single Symbol Analysis":
        render_single_symbol_analysis()
    elif page == "Pipeline Dashboard":
        render_pipeline_dashboard()
    elif page == "Auto Trader":
        render_auto_trader_dashboard()
    elif page == "Market Overview":
        render_market_overview()
    elif page == "Education Center":
        render_education_center()
    
    # Footer
    st.markdown("---")
    st.caption(f"GEX Master Pro - Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Auto-refresh logic
    auto_refresh = False  # Set to True to enable auto-refresh
    if auto_refresh and st.session_state.last_update:
        if (datetime.now() - st.session_state.last_update).total_seconds() > 300:  # 5 minutes
            st.rerun()

def render_single_symbol_analysis():
    """Render the single symbol analysis page"""
    st.header("Single Symbol Analysis")
    
    # Input section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        symbol = st.text_input("Enter Symbol", placeholder="SPY", key="symbol_input").upper()
    
    with col2:
        analyze_button = st.button("Analyze", type="primary")
    
    if analyze_button and symbol:
        with st.spinner(f"Analyzing {symbol}..."):
            gex_data = fetch_real_gex_data(symbol)
            
            # Update API status based on result
            if gex_data.get('success'):
                # Store in session state
                st.session_state.gex_data[symbol] = gex_data
                st.session_state.last_update = datetime.now()
                
                # Determine data source for user information
                if gex_data.get('database_source'):
                    data_source = "Database"
                    st.info("📊 Using data from your Databricks pipeline")
                else:
                    data_source = "API"
                    st.success("📡 Using real-time API data")
                
                # GEX Structure Display
                st.subheader(f"GEX Structure for {symbol}")
                
                # Key metrics in columns
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Current Price", 
                        f"${gex_data['spot_price']:.2f}",
                        help="Current stock price"
                    )
                
                with col2:
                    st.metric(
                        "Gamma Flip", 
                        f"${gex_data['gamma_flip_point']:.2f}",
                        help="Zero gamma crossing point"
                    )
                
                with col3:
                    st.metric(
                        "Net GEX", 
                        f"{gex_data['net_gex']/1e9:.2f}B",
                        help="Total gamma exposure"
                    )
                
                with col4:
                    distance_pct = (gex_data['spot_price'] - gex_data['gamma_flip_point']) / gex_data['spot_price'] * 100
                    st.metric(
                        "Distance to Flip", 
                        f"{distance_pct:+.2f}%",
                        help="Percentage distance to gamma flip"
                    )
                
                # Support and Resistance Levels
                st.subheader("Key Levels")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        "Call Wall (Resistance)", 
                        f"${gex_data['call_wall']:.2f}",
                        delta=f"{((gex_data['call_wall'] - gex_data['spot_price']) / gex_data['spot_price'] * 100):+.1f}%",
                        help="Expected resistance level from dealer hedging"
                    )
                
                with col2:
                    st.metric(
                        "Put Wall (Support)", 
                        f"${gex_data['put_wall']:.2f}",
                        delta=f"{((gex_data['put_wall'] - gex_data['spot_price']) / gex_data['spot_price'] * 100):+.1f}%",
                        help="Expected support level from dealer hedging"
                    )
                
                # Visual GEX Structure
                st.subheader("GEX Structure Visualization")
                
                # Create and display detailed GEX structure chart
                gex_fig = plot_gex_structure(symbol, gex_data)
                st.plotly_chart(gex_fig, use_container_width=True)
                
                # Strategy Detection
                st.subheader("Strategy Analysis")
                
                setups = detect_enhanced_strategies(symbol, gex_data)
                
                if setups:
                    for setup in setups:
                        # Check for big move enhancement
                        enhanced_setup = enhance_setup_for_big_moves(setup, gex_data)
                        
                        # Auto-trader integration
                        if st.session_state.auto_trader['enabled'] and enhanced_setup['confidence'] >= st.session_state.auto_trader['min_confidence']:
                            # Check if setup already in auto-trader
                            setup_exists = any(
                                t['symbol'] == enhanced_setup['symbol'] and 
                                t['setup_type'] == enhanced_setup['setup_type'] and
                                t['status'] == 'active'
                                for t in st.session_state.auto_trader['positions']
                            )
                            
                            if not setup_exists:
                                # Add to auto-trader
                                success, message = create_auto_trader_trade(enhanced_setup)
                                if success:
                                    st.success(f"🤖 Auto-Trader: {message}")
                                else:
                                    st.info(f"🤖 Auto-Trader: {message}")
                        
                        # Discord webhook integration
                        if (st.session_state.discord_webhook['enabled'] and 
                            enhanced_setup['confidence'] >= st.session_state.discord_webhook['min_confidence'] and
                            enhanced_setup['setup_type'] in st.session_state.discord_webhook['alert_types']):
                            
                            # Send alert if webhook configured
                            if st.session_state.discord_webhook['url']:
                                success, message = send_discord_alert(enhanced_setup, st.session_state.discord_webhook['url'])
                                if success:
                                    st.success(f"🔔 Discord Alert: {message}")
                                else:
                                    st.warning(f"🔔 Discord Alert: {message}")
                        
                        if enhanced_setup.get('big_move_mode'):
                            st.markdown(f"""
                            <div class="card high-confidence">
                                <h4>🚀 BIG MOVE SETUP DETECTED - {enhanced_setup['confidence']:.0f}% Confidence</h4>
                                <div style="display: flex; justify-content: space-between; margin-top: 1rem;">
                                    <div>
                                        <strong>Strategy:</strong> {enhanced_setup['strategy']}<br>
                                        <strong>Setup Type:</strong> {enhanced_setup['setup_type'].replace('_', ' ').title()}<br>
                                        <strong>Entry:</strong> {enhanced_setup['entry_criteria']}
                                    </div>
                                    <div style="text-align: right;">
                                        <strong>Big Move Target:</strong> ${enhanced_setup['big_move_target']:.2f}<br>
                                        <strong>Expected Return:</strong> +{enhanced_setup['big_move_return_pct']:.1f}%<br>
                                        <strong>Type:</strong> {enhanced_setup['big_move_type']}
                                    </div>
                                </div>
                                <div style="margin-top: 1rem;">
                                    <strong>Reason:</strong> {enhanced_setup['reason']}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            confidence_class = format_confidence_class(setup['confidence'])
                            
                            st.markdown(f"""
                            <div class="card {confidence_class}">
                                <h4>{setup['strategy']} - {setup['confidence']:.0f}% Confidence</h4>
                                <div style="display: flex; justify-content: space-between; margin-top: 1rem;">
                                    <div>
                                        <strong>Setup Type:</strong> {setup['setup_type'].replace('_', ' ').title()}<br>
                                        <strong>Entry:</strong> {setup['entry_criteria']}<br>
                                        <strong>Time Frame:</strong> {setup['time_frame']}
                                    </div>
                                    <div style="text-align: right;">
                                        <strong>Target:</strong> ${setup['target']:.2f}<br>
                                        <strong>Expected Move:</strong> {setup['expected_move']}<br>
                                        <strong>Option Type:</strong> {setup['option_type'].replace('_', ' ').title()}
                                    </div>
                                </div>
                                <div style="margin-top: 1rem;">
                                    <strong>Reason:</strong> {setup['reason']}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.info("No specific trading setups detected for current GEX conditions.")
                
                # Market Context
                st.subheader("Market Context")
                
                # Determine regime
                if gex_data['net_gex'] > 2e9:
                    regime = "HIGH_POSITIVE_GEX"
                    regime_description = "Strong volatility suppression - range trading favored"
                    regime_color = "success"
                elif gex_data['net_gex'] > 0.5e9:
                    regime = "MODERATE_POSITIVE_GEX" 
                    regime_description = "Mild volatility suppression - mixed strategies"
                    regime_color = "info"
                elif gex_data['net_gex'] < -0.5e9:
                    regime = "NEGATIVE_GEX"
                    regime_description = "Volatility amplification - squeeze plays active"
                    regime_color = "warning"
                else:
                    regime = "NEUTRAL_GEX"
                    regime_description = "Balanced gamma exposure"
                    regime_color = "secondary"
                
                st.markdown(f"""
                <div class="alert-box alert-{regime_color}">
                    <strong>Market Regime:</strong> {regime}<br>
                    {regime_description}
                </div>
                """, unsafe_allow_html=True)
                
            else:
                st.error(f"Failed to fetch data for {symbol}: {gex_data.get('error', 'Unknown error')}")

def render_pipeline_dashboard():
    """Render the pipeline dashboard page"""
    st.header("Pipeline Dashboard")
    
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
            with st.expander("Debug Information", expanded=False):
                st.markdown(debug_info)
    elif status == 'error':
        st.warning(f"⚠️ {message}")
    else:
        st.info(f"ℹ️ {message}")
    
    # Refresh button
    if st.button("Refresh Data", type="primary"):
        st.cache_data.clear()
        st.rerun()
    
    if df.empty:
        st.error("No pipeline data available")
        st.info("""
        **Possible Reasons:**
        1. No recent pipeline runs
        2. Database connection issue
        3. Pipeline table is empty
        
        Please check your Databricks environment and ensure your pipeline has been executed.
        """)
        return
    
    # Filters
    st.subheader("Filters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        min_confidence = st.slider("Minimum Confidence %", 0, 100, 70)
    
    with col2:
        if not df.empty:
            setup_types = st.multiselect(
                "Setup Types",
                df['structure_type'].unique().tolist(),
                default=df['structure_type'].unique().tolist()
            )
        else:
            setup_types = []
    
    with col3:
        if not df.empty:
            symbols = st.multiselect(
                "Symbols",
                sorted(df['symbol'].unique().tolist()),
                default=sorted(df['symbol'].unique().tolist())[:5] if len(df['symbol'].unique()) > 5 else sorted(df['symbol'].unique().tolist())
            )
        else:
            symbols = []
    
    # Apply filters
    filtered_df = df[
        (df['confidence_score'] >= min_confidence) &
        (df['structure_type'].isin(setup_types)) &
        (df['symbol'].isin(symbols))
    ].copy()
    
    if filtered_df.empty:
        st.warning("No data matches your filters. Try adjusting the criteria.")
        return
    
    # Key metrics row
    st.subheader("Pipeline Results")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-title">Total Setups</div>
            <div class="metric-value">{len(filtered_df)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        high_conf = len(filtered_df[filtered_df['confidence_score'] >= 85])
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-title">High Confidence</div>
            <div class="metric-value">{high_conf}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_conf = filtered_df['confidence_score'].mean()
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-title">Avg Confidence</div>
            <div class="metric-value">{avg_conf:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        try:
            enhanced_strategies = len(filtered_df[filtered_df['category'] == 'ENHANCED_STRATEGY'])
        except:
            enhanced_strategies = 0
            
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-title">Enhanced Strategies</div>
            <div class="metric-value">{enhanced_strategies}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Top setups
    st.subheader("High Confidence Setups")
    
    # Show priority 1 setups first
    try:
        priority_setups = filtered_df[filtered_df['priority'] == 1].head(10)
    except:
        priority_setups = filtered_df.sort_values('confidence_score', ascending=False).head(10)
    
    if not priority_setups.empty:
        for _, setup in priority_setups.iterrows():
            confidence_class = format_confidence_class(setup['confidence_score'])
            
            # Format distance
            try:
                distance_display = f"{setup['distance_to_flip_pct']:+.2f}%"
            except:
                distance_display = "N/A"
            
            st.markdown(f"""
            <div class="card {confidence_class}">
                <h4>{setup['symbol']} - {setup['confidence_score']}% Confidence</h4>
                <div style="display: flex; justify-content: space-between; margin-top: 1rem;">
                    <div>
                        <strong>Setup:</strong> {setup['structure_type'].replace('_', ' ').title()}<br>
                        <strong>Category:</strong> {setup.get('category', 'N/A')}
                    </div>
                    <div style="text-align: right;">
                        <strong>Spot:</strong> ${setup['spot_price']:.2f}<br>
                        <strong>Flip Point:</strong> ${setup['gamma_flip_point']:.2f}
                    </div>
                </div>
                <div style="margin-top: 1rem;">
                    <strong>Distance:</strong> {distance_display} | 
                    <strong>Priority:</strong> {setup.get('priority', 'N/A')} | 
                    <strong>Recommendation:</strong> {setup.get('recommendation', 'N/A')}
                </div>
                <div style="margin-top: 0.5rem; font-size: 0.9em; color: #666;">
                    <strong>Created:</strong> {pd.to_datetime(setup['analysis_timestamp']).strftime('%m/%d %H:%M')}
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No high-priority setups match your current filters")
    
    # Charts section
    st.subheader("Analysis Charts")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Setup type distribution
        setup_counts = filtered_df['structure_type'].value_counts()
        
        fig_pie = go.Figure(data=[go.Pie(
            labels=setup_counts.index,
            values=setup_counts.values,
            hole=0.4
        )])
        
        fig_pie.update_layout(
            title="Setup Type Distribution",
            height=300,
            template="plotly_white"
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
            yaxis_title="Number of Setups",
            template="plotly_white"
        )
        
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Symbol performance chart
    if len(filtered_df) > 0:
        st.subheader("Top Symbols by Setup Count")
        
        symbol_counts = filtered_df['symbol'].value_counts().head(15)
        
        fig_symbol = go.Figure(data=[go.Bar(
            x=symbol_counts.index,
            y=symbol_counts.values,
            marker_color='#007bff'
        )])
        
        fig_symbol.update_layout(
            title="Symbols with Most Setups",
            height=400,
            xaxis_title="Symbol",
            yaxis_title="Number of Setups",
            template="plotly_white"
        )
        
        st.plotly_chart(fig_symbol, use_container_width=True)
    
    # Complete data table
    st.subheader("All Pipeline Results")
    
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
    
    columns_to_display = ['symbol', 'Setup Type', 'Confidence %', 'Spot Price', 
                       'Flip Point', 'Distance %', 'Created']
    
    # Add recommendation column if it exists
    if 'recommendation' in display_df.columns:
        columns_to_display.append('recommendation')
    
    # Add priority column if it exists
    if 'priority' in display_df.columns:
        columns_to_display.append('priority')
    
    st.dataframe(
        display_df[columns_to_display],
        use_container_width=True,
        hide_index=True
    )
    
    # Download section
    st.subheader("Export Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            "Download CSV",
            csv,
            f"gex_pipeline_results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            "text/csv"
        )
    
    with col2:
        json_data = filtered_df.to_json(orient='records', indent=2)
        st.download_button(
            "Download JSON",
            json_data,
            f"gex_pipeline_results_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
            "application/json"
        )

def render_auto_trader_dashboard():
    """Render the auto-trader dashboard"""
    st.header("Auto Trader")
    
    # Update auto-trader status
    update_all_paper_trades()
    
    # Settings and controls
    with st.expander("Auto-Trader Settings", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            # Enable/disable auto-trader
            auto_trader_enabled = st.checkbox(
                "Enable Auto-Trader",
                value=st.session_state.auto_trader['enabled'],
                help="Automatically execute trades based on high-confidence setups"
            )
            
            # Update session state
            st.session_state.auto_trader['enabled'] = auto_trader_enabled
            
            # Minimum confidence for auto-trader
            min_confidence = st.slider(
                "Minimum Confidence",
                min_value=60,
                max_value=100,
                value=st.session_state.auto_trader['min_confidence'],
                step=5,
                help="Minimum confidence score to trigger automatic trades"
            )
            
            # Update session state
            st.session_state.auto_trader['min_confidence'] = min_confidence
        
        with col2:
            # Paper trading capital
            capital = st.number_input(
                "Paper Trading Capital",
                min_value=1000,
                max_value=1000000,
                value=int(st.session_state.auto_trader['capital']),
                step=1000,
                help="Total capital for paper trading"
            )
            
            # Update session state
            st.session_state.auto_trader['capital'] = capital
            
            # Maximum position size
            max_position_size = st.slider(
                "Max Position Size (%)",
                min_value=1,
                max_value=20,
                value=int(st.session_state.auto_trader['max_position_size'] * 100),
                step=1,
                help="Maximum position size as percentage of capital"
            )
            
            # Update session state
            st.session_state.auto_trader['max_position_size'] = max_position_size / 100
    
    # Performance metrics
    st.subheader("Performance Overview")
    
    # Get auto-trader stats
    stats = get_auto_trader_stats()
    
    if stats:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-title">Available Capital</div>
                <div class="metric-value">${st.session_state.auto_trader['available_capital']:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-title">Total P&L</div>
                <div class="metric-value" style="color: {'#28a745' if stats['total_pnl'] >= 0 else '#dc3545'}">
                    {stats['total_pnl']:+.2f} ({stats['total_pnl_pct']:+.1f}%)
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-title">Win Rate</div>
                <div class="metric-value">{stats['win_rate']:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-title">Active Trades</div>
                <div class="metric-value">{stats['active_trades']}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Active positions
    st.subheader("Active Positions")
    
    active_positions = [t for t in st.session_state.auto_trader['positions'] if t['status'] == 'active']
    
    if active_positions:
        for position in active_positions:
            # Determine status color
            pnl_color = "#28a745" if position['current_pnl'] >= 0 else "#dc3545"
            
            st.markdown(f"""
            <div class="card position-active">
                <h4>{position['symbol']} - {position['setup_type'].replace('_', ' ').title()}</h4>
                <div style="display: flex; justify-content: space-between; margin-top: 1rem;">
                    <div>
                        <strong>Type:</strong> {position['option_type'].replace('_', ' ').title()}<br>
                        <strong>Entry:</strong> {pd.to_datetime(position['entry_time']).strftime('%m/%d %H:%M')}<br>
                        <strong>Position Size:</strong> ${position['position_size']:.2f}
                    </div>
                    <div style="text-align: right;">
                        <strong>Current P&L:</strong> <span style="color: {pnl_color}">{position['current_pnl']:+.2f} ({position['current_pnl_pct']:+.1f}%)</span><br>
                        <strong>Target:</strong> ${position['profit_target'] if 'profit_target' in position else position['target_price']:.2f}<br>
                        <strong>Stop:</strong> ${position['stop_loss']:.2f}
                    </div>
                </div>
                <div style="margin-top: 1rem;">
                    <strong>Details:</strong> 
                    {f"{position['contracts']} contracts at ${position['option_price']:.2f}" if 'option_price' in position else f"Credit received: ${position['credit_received']:.2f}"}
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No active positions")
    
    # Trade history
    st.subheader("Trade History")
    
    closed_positions = [t for t in st.session_state.auto_trader['positions'] if t['status'] != 'active']
    
    if closed_positions:
        # Sort by exit time (most recent first)
        closed_positions.sort(key=lambda x: x['exit_time'] if x['exit_time'] else datetime.now(), reverse=True)
        
        for position in closed_positions:
            # Determine status class
            position_class = "position-closed" if position['status'] == 'closed' else "position-stopped"
            
            # Determine P&L color
            pnl_color = "#28a745" if position['exit_pnl'] >= 0 else "#dc3545"
            
            st.markdown(f"""
            <div class="card {position_class}">
                <h4>{position['symbol']} - {position['setup_type'].replace('_', ' ').title()} ({position['status'].title()})</h4>
                <div style="display: flex; justify-content: space-between; margin-top: 1rem;">
                    <div>
                        <strong>Type:</strong> {position['option_type'].replace('_', ' ').title()}<br>
                        <strong>Entry:</strong> {pd.to_datetime(position['entry_time']).strftime('%m/%d %H:%M')}<br>
                        <strong>Exit:</strong> {pd.to_datetime(position['exit_time']).strftime('%m/%d %H:%M')}
                    </div>
                    <div style="text-align: right;">
                        <strong>P&L:</strong> <span style="color: {pnl_color}">{position['exit_pnl']:+.2f} ({position['exit_pnl_pct']:+.1f}%)</span><br>
                        <strong>Exit Reason:</strong> {position['exit_reason']}<br>
                        <strong>Position Size:</strong> ${position['position_size']:.2f}
                    </div>
                </div>
                <div style="margin-top: 1rem;">
                    <strong>Details:</strong> 
                    {f"{position['contracts']} contracts at ${position['option_price']:.2f}" if 'option_price' in position else f"Credit received: ${position['credit_received']:.2f}"}
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No trade history")
    
    # Performance charts
    if stats and stats['total_trades'] > 0:
        st.subheader("Performance Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Trade outcome pie chart
            outcome_labels = ['Profitable', 'Loss', 'Active']
            outcome_values = [stats['closed_trades'], stats['stopped_trades'], stats['active_trades']]
            
            fig_outcome = go.Figure(data=[go.Pie(
                labels=outcome_labels,
                values=outcome_values,
                marker_colors=['#28a745', '#dc3545', '#007bff'],
                hole=0.4
            )])
            
            fig_outcome.update_layout(
                title="Trade Outcomes",
                height=300,
                template="plotly_white"
            )
            
            st.plotly_chart(fig_outcome, use_container_width=True)
        
        with col2:
            # Setup type performance
            setup_types = {}
            for trade in st.session_state.auto_trader['positions']:
                setup_type = trade['setup_type']
                if setup_type not in setup_types:
                    setup_types[setup_type] = {'count': 0, 'pnl': 0}
                
                setup_types[setup_type]['count'] += 1
                
                if trade['status'] == 'active':
                    setup_types[setup_type]['pnl'] += trade['current_pnl']
                else:
                    setup_types[setup_type]['pnl'] += trade['exit_pnl']
            
            # Prepare data for chart
            setup_labels = list(setup_types.keys())
            setup_pnl = [setup_types[st]['pnl'] for st in setup_labels]
            
            # Create chart
            fig_setup = go.Figure(data=[go.Bar(
                x=setup_labels,
                y=setup_pnl,
                marker_color=['#007bff' if pnl >= 0 else '#dc3545' for pnl in setup_pnl]
            )])
            
            fig_setup.update_layout(
                title="P&L by Strategy Type",
                height=300,
                xaxis_title="Strategy Type",
                yaxis_title="P&L ($)",
                template="plotly_white"
            )
            
            st.plotly_chart(fig_setup, use_container_width=True)

def render_market_overview():
    """Render the market overview dashboard"""
    st.header("Market Overview")
    
    # Database connection
    connection = init_databricks_connection()
    if not connection:
        st.error("Databricks connection unavailable - Market Overview requires database access")
        return
    
    try:
        # Get market overview data
        cursor = connection.cursor()
        
        # Get overall market GEX data
        market_query = """
        SELECT 
            symbol,
            spot_price,
            net_gex/1000000000 as net_gex_billions,
            gamma_flip_point,
            distance_to_flip_pct,
            analysis_date,
            analysis_timestamp
        FROM quant_projects.gex_trading.scheduled_pipeline_results
        WHERE symbol IN ('SPY', 'QQQ', 'IWM', 'DIA')
        AND analysis_date >= current_date() - INTERVAL 1 DAY
        ORDER BY analysis_timestamp DESC
        LIMIT 20
        """
        
        cursor.execute(market_query)
        market_results = cursor.fetchall()
        market_columns = [desc[0] for desc in cursor.description]
        
        market_df = pd.DataFrame(market_results, columns=market_columns)
        
        # Get market regime data
        regime_query = """
        SELECT 
            regime_type,
            regime_strength,
            vix_level,
            total_market_gex_billions,
            analysis_date,
            analysis_timestamp
        FROM quant_projects.gex_trading.market_regime
        WHERE analysis_date >= current_date() - INTERVAL 7 DAY
        ORDER BY analysis_timestamp DESC
        LIMIT 10
        """
        
        try:
            cursor.execute(regime_query)
            regime_results = cursor.fetchall()
            regime_columns = [desc[0] for desc in cursor.description]
            regime_df = pd.DataFrame(regime_results, columns=regime_columns)
        except:
            # If market_regime table doesn't exist
            regime_df = pd.DataFrame()
        
        cursor.close()
        
        # Display market data
        if not market_df.empty:
            latest_etfs = market_df.drop_duplicates('symbol', keep='first')
            
            st.subheader("Major ETF GEX Status")
            
            cols = st.columns(len(latest_etfs))
            
            for i, (_, row) in enumerate(latest_etfs.iterrows()):
                with cols[i]:
                    # Determine if above or below flip
                    above_flip = row['distance_to_flip_pct'] > 0
                    color = "#28a745" if above_flip else "#dc3545"
                    regime = "POSITIVE GEX" if row['net_gex_billions'] > 0 else "NEGATIVE GEX"
                    
                    st.markdown(f"""
                    <div class="card" style="border-left: 5px solid {color};">
                        <h3 style="text-align: center;">{row['symbol']}</h3>
                        <div style="font-size: 24px; font-weight: bold; text-align: center;">${row['spot_price']:.2f}</div>
                        <div style="color: {color}; text-align: center;">{regime}</div>
                        <hr>
                        <div>Net GEX: {row['net_gex_billions']:.2f}B</div>
                        <div>Flip: ${row['gamma_flip_point']:.2f}</div>
                        <div>Distance: {row['distance_to_flip_pct']:+.2f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Market regime
            if not regime_df.empty:
                latest_regime = regime_df.iloc[0]
                
                st.subheader("Current Market Regime")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Market Regime", latest_regime['regime_type'])
                
                with col2:
                    st.metric("VIX Level", f"{latest_regime['vix_level']:.1f}")
                
                with col3:
                    st.metric("Total Market GEX", f"{latest_regime['total_market_gex_billions']:.1f}B")
            
            # ETF GEX history chart
            st.subheader("ETF Net GEX History")
            
            gex_history = market_df.copy()
            gex_history['analysis_timestamp'] = pd.to_datetime(gex_history['analysis_timestamp'])
            
            fig = go.Figure()
            
            for symbol in gex_history['symbol'].unique():
                symbol_data = gex_history[gex_history['symbol'] == symbol]
                fig.add_trace(go.Scatter(
                    x=symbol_data['analysis_timestamp'],
                    y=symbol_data['net_gex_billions'],
                    mode='lines+markers',
                    name=symbol
                ))
            
            fig.update_layout(
                title="Net GEX History (Billions $)",
                xaxis_title="Date",
                yaxis_title="Net GEX (Billions)",
                height=400,
                template="plotly_white"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Recent interesting setups
            st.subheader("Recent Interesting Setups")
            
            try:
                # Get interesting setups
                interesting_query = """
                SELECT 
                    symbol,
                    structure_type,
                    confidence_score,
                    analysis_timestamp
                FROM quant_projects.gex_trading.scheduled_pipeline_results
                WHERE confidence_score >= 80
                AND analysis_date >= current_date() - INTERVAL 1 DAY
                ORDER BY confidence_score DESC
                LIMIT 5
                """
                
                cursor = connection.cursor()
                cursor.execute(interesting_query)
                interesting_results = cursor.fetchall()
                interesting_columns = [desc[0] for desc in cursor.description]
                
                interesting_df = pd.DataFrame(interesting_results, columns=interesting_columns)
                cursor.close()
                
                if not interesting_df.empty:
                    for _, row in interesting_df.iterrows():
                        st.markdown(f"""
                        <div class="card high-confidence">
                            <div style="display: flex; justify-content: space-between;">
                                <div><strong>{row['symbol']}</strong> - {row['structure_type'].replace('_', ' ').title()}</div>
                                <div>{row['confidence_score']}% confidence</div>
                            </div>
                            <div style="font-size: 0.9em; color: #666; margin-top: 5px;">
                                {pd.to_datetime(row['analysis_timestamp']).strftime('%m/%d/%y %H:%M')}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No high confidence setups found in the last 24 hours")
                    
            except Exception as e:
                st.warning(f"Could not load interesting setups: {e}")
            
        else:
            st.error("No market data available")
            st.info("Make sure your pipeline has processed market ETFs (SPY, QQQ, IWM, DIA)")
        
    except Exception as e:
        st.error(f"Failed to load market overview data: {e}")

def render_education_center():
    """Render the education center"""
    st.header("Education Center")
    
    # Tabs for different educational topics
    tabs = st.tabs(["GEX Basics", "Trading Strategies", "Risk Management", "FAQ"])
    
    with tabs[0]:
        render_gex_education()
        
        # Additional GEX educational content
        st.markdown("""
        <div class="education-section">
            <h3 class="education-title">How Dealers Impact the Market</h3>
            <p>Market makers and dealers must maintain delta-neutral positions to manage risk. As options gamma changes with underlying price movements, dealers must continuously hedge by buying or selling the underlying asset.</p>
            
            <h4>The Dealer Hedging Cycle:</h4>
            <ol>
                <li><strong>Initial Position</strong>: Dealers sell options to traders and hedge delta exposure.</li>
                <li><strong>Price Movement</strong>: As price changes, delta of options changes (gamma effect).</li>
                <li><strong>Rehedging</strong>: Dealers must buy or sell underlying to maintain delta neutrality.</li>
                <li><strong>Market Impact</strong>: This hedging creates predictable buying/selling pressure.</li>
            </ol>
            
            <h4>Why This Matters for Traders:</h4>
            <p>Gamma exposure creates systematic flows that can be predicted and traded ahead of. By understanding where and when dealers will be forced to buy or sell, traders can position for these predictable moves.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Visual explanation
        st.markdown("""
        <div class="education-section">
            <h3 class="education-title">Visualizing GEX</h3>
            <p>The GEX profile shows gamma concentration across different strike prices:</p>
            <ul>
                <li><strong>Green Bars (Call Gamma)</strong>: Positive gamma from calls - creates resistance</li>
                <li><strong>Red Bars (Put Gamma)</strong>: Negative gamma from puts - creates support</li>
                <li><strong>Blue Line (Net GEX)</strong>: Combined effect across all strikes</li>
                <li><strong>Vertical Lines</strong>: Key price levels (current price, gamma flip, walls)</li>
            </ul>
            <p>The visualization helps identify key levels where dealer hedging will have the most impact on price action.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with tabs[1]:
        render_strategy_education()
        
        # Additional strategy examples
        st.markdown("""
        <div class="education-section">
            <h3 class="education-title">Real-World Strategy Examples</h3>
            
            <h4>Example 1: SPY Negative GEX Squeeze</h4>
            <p><strong>Setup:</strong> SPY has -2.5B GEX, price is 0.8% below gamma flip at $452, with put wall at $445.</p>
            <p><strong>Trade:</strong> Buy SPY 455 calls, 3 DTE, targeting gamma flip as first resistance.</p>
            <p><strong>Outcome:</strong> As price moves toward gamma flip, dealers must buy more shares to maintain delta neutrality, accelerating the move upward.</p>
            
            <h4>Example 2: QQQ Call Wall Premium Selling</h4>
            <p><strong>Setup:</strong> QQQ has +1.8B GEX, price is between flip ($378) and call wall ($385).</p>
            <p><strong>Trade:</strong> Sell QQQ 385/390 call credit spread, 2 DTE.</p>
            <p><strong>Outcome:</strong> Strong call wall resistance causes price rejection, allowing rapid theta decay for profit.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with tabs[2]:
        render_risk_education()
        
        # Additional risk management examples
        st.markdown("""
        <div class="education-section">
            <h3 class="education-title">Advanced Risk Management</h3>
            
            <h4>Position Correlation Management:</h4>
            <p>When trading multiple GEX setups, monitor correlations to avoid overexposure:</p>
            <ul>
                <li><strong>ETF Overlaps</strong>: Reduce position size when trading correlated ETFs (SPY/QQQ)</li>
                <li><strong>Sector Concentration</strong>: Avoid multiple positions in same sector during sector rotation</li>
                <li><strong>VIX Regime Awareness</strong>: Adjust strategy selection based on volatility regime</li>
            </ul>
            
            <h4>Kelly Criterion for GEX Trading:</h4>
            <p>Optimize position sizing using historical win rates:</p>
            <ul>
                <li><strong>Squeeze Plays</strong>: ~65% win rate suggests 30% of max allocation</li>
                <li><strong>Premium Selling</strong>: ~80% win rate suggests 60% of max allocation</li>
                <li><strong>Always reduce Kelly fraction</strong> by at least 50% to account for estimation error</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with tabs[3]:
        st.markdown("""
        <div class="education-section">
            <h3 class="education-title">Frequently Asked Questions</h3>
            
            <h4>Q: What data sources does GEX Master Pro use?</h4>
            <p>A: The system uses options data from TradingVolatility.net API, with fallback to your Databricks pipeline data, and additional market data from yfinance when needed.</p>
            
            <h4>Q: How often should I check for new setups?</h4>
            <p>A: The pipeline runs early morning (pre-market), but market conditions can change rapidly. Check for updates at least at market open, mid-day, and before close.</p>
            
            <h4>Q: Why do gamma levels change throughout the day?</h4>
            <p>A: As new options are traded and underlying prices change, the gamma profile shifts. Major events like FOMC can dramatically alter the gamma landscape.</p>
            
            <h4>Q: How far in advance should I enter positions?</h4>
            <p>A: For squeeze plays, enter when within 1% of gamma flip point. For premium selling, wait until price approaches wall levels within 0.5-1%.</p>
            
            <h4>Q: What's the best DTE (Days to Expiration) for GEX trading?</h4>
            <p>A: Gamma is highest close to expiration, so 0-5 DTE works best for most strategies. Squeeze plays perform well with 2-5 DTE, premium selling with 0-2 DTE.</p>
            
            <h4>Q: How do I interpret the GEX visualization?</h4>
            <p>A: Green bars show call gamma (resistance), red bars show put gamma (support), blue line shows net effect. Vertical lines mark key price levels.</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
