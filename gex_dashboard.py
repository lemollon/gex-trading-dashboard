# Enhanced Mock Trading System with Auto-Trading and Real Performance
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yfinance as yf
import time
from typing import Dict, List, Optional
import uuid

class EnhancedMockTradingAccount:
    """Enhanced mock trading with auto-trading and real performance tracking"""
    
    def __init__(self):
        self.initial_balance = 100000
        # Initialize session state for persistence
        if 'portfolio_history' not in st.session_state:
            st.session_state.portfolio_history = []
        if 'open_trades' not in st.session_state:
            st.session_state.open_trades = []
        if 'closed_trades' not in st.session_state:
            st.session_state.closed_trades = []
        if 'auto_trading_enabled' not in st.session_state:
            st.session_state.auto_trading_enabled = False
        
    def get_real_stock_price(self, symbol: str) -> Optional[float]:
        """Get real-time stock price using yfinance"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1d", interval="1m")
            if not data.empty:
                return float(data['Close'].iloc[-1])
        except:
            pass
        return None
    
    def calculate_option_value(self, trade: Dict) -> float:
        """Calculate current option value based on real stock movement"""
        try:
            current_stock_price = self.get_real_stock_price(trade['symbol'])
            if not current_stock_price:
                # Simulate realistic price movement if no real data
                time_decay = min(trade['days_held'] * 0.1, 0.8)  # Up to 80% time decay
                price_movement = np.random.normal(0, 0.15)  # ±15% volatility
                return trade['entry_price'] * (1 + price_movement - time_decay)
            
            entry_stock_price = trade.get('entry_stock_price', current_stock_price)
            price_change_pct = (current_stock_price - entry_stock_price) / entry_stock_price
            
            # Option value calculation based on trade type and stock movement
            if trade['trade_type'] == 'LONG_CALLS':
                # Calls gain value when stock goes up
                if price_change_pct > 0:
                    option_multiplier = min(1 + (price_change_pct * 5), 3.0)  # Max 3x gain
                else:
                    option_multiplier = max(1 + (price_change_pct * 3), 0.1)  # Max 90% loss
            
            elif trade['trade_type'] == 'CALL_SELLING':
                # Sold calls lose value when stock goes down (good for us)
                if price_change_pct < 0:
                    option_multiplier = max(1 + (price_change_pct * 2), 0.2)  # Premium decay
                else:
                    option_multiplier = min(1 + (price_change_pct * 4), 2.5)  # Assignment risk
            
            else:  # STRADDLE
                # Straddles gain from volatility (big moves either direction)
                volatility_gain = abs(price_change_pct) * 3
                option_multiplier = 1 + volatility_gain
            
            # Apply time decay
            days_held = trade.get('days_held', 1)
            time_decay = min(days_held * 0.05, 0.6)  # 5% per day, max 60%
            
            current_value = trade['entry_price'] * option_multiplier * (1 - time_decay)
            return max(current_value, 0.01)  # Options can't be negative
            
        except Exception as e:
            # Fallback to simple simulation
            return trade['entry_price'] * np.random.uniform(0.3, 2.5)
    
    def update_open_trades(self):
        """Update all open trades with current values and auto-exit logic"""
        updated_trades = []
        
        for trade in st.session_state.open_trades:
            # Update days held
            entry_date = pd.to_datetime(trade['entry_date'])
            days_held = (datetime.now() - entry_date).days
            trade['days_held'] = days_held
            
            # Calculate current value
            current_value = self.calculate_option_value(trade)
            trade['current_value'] = current_value
            
            # Calculate P&L
            entry_total = trade['entry_price'] * trade['quantity'] * 100
            current_total = current_value * trade['quantity'] * 100
            trade['unrealized_pnl'] = current_total - entry_total
            trade['unrealized_pnl_pct'] = (trade['unrealized_pnl'] / entry_total) * 100
            
            # Auto-exit logic
            should_exit = self.check_exit_conditions(trade)
            
            if should_exit:
                # Close the trade
                exit_reason = should_exit
                self.close_trade(trade, exit_reason)
            else:
                updated_trades.append(trade)
        
        st.session_state.open_trades = updated_trades
    
    def check_exit_conditions(self, trade: Dict) -> Optional[str]:
        """Check if trade should be automatically closed"""
        pnl_pct = trade.get('unrealized_pnl_pct', 0)
        days_held = trade.get('days_held', 0)
        confidence = trade.get('confidence_score', 70)
        
        # Profit taking rules
        if pnl_pct >= 100:  # 100% gain
            return "Profit Target (100%)"
        
        if pnl_pct >= 50 and days_held >= 2:  # 50% gain after 2 days
            return "Profit Target (50%+)"
        
        # Stop loss rules
        if pnl_pct <= -50:  # 50% loss
            return "Stop Loss (50%)"
        
        # Time-based exits
        if days_held >= 7:  # Max 7 days
            return "Time Stop (7 days)"
        
        if days_held >= 3 and pnl_pct <= -25:  # Cut losses after 3 days
            return "Time Stop + Loss"
        
        # High confidence trades get more room
        if confidence >= 90 and pnl_pct <= -30:  # High confidence stop
            return "High Confidence Stop"
        
        return None
    
    def close_trade(self, trade: Dict, exit_reason: str):
        """Close a trade and record the result"""
        exit_trade = trade.copy()
        exit_trade['exit_date'] = datetime.now().date()
        exit_trade['exit_price'] = trade['current_value']
        exit_trade['exit_reason'] = exit_reason
        exit_trade['realized_pnl'] = trade['unrealized_pnl']
        exit_trade['realized_pnl_pct'] = trade['unrealized_pnl_pct']
        exit_trade['status'] = 'CLOSED'
        
        st.session_state.closed_trades.append(exit_trade)
        
        # Log the closure
        if exit_trade['realized_pnl'] > 0:
            st.success(f"🎉 WINNING TRADE: {exit_trade['symbol']} closed with {exit_trade['realized_pnl_pct']:.1f}% gain! Reason: {exit_reason}")
        else:
            st.warning(f"❌ Trade closed: {exit_trade['symbol']} with {exit_trade['realized_pnl_pct']:.1f}% loss. Reason: {exit_reason}")
    
    def auto_trade_opportunities(self, opportunities: List[Dict]):
        """Automatically trade high-confidence opportunities"""
        if not st.session_state.auto_trading_enabled:
            return
        
        balance = self.get_current_balance()
        available_cash = balance['cash_balance']
        
        # Only trade 90%+ confidence opportunities
        high_conf_opps = [opp for opp in opportunities if opp['confidence_score'] >= 90]
        
        for opp in high_conf_opps[:2]:  # Max 2 auto-trades per session
            # Check if we already have this symbol
            existing_symbols = [trade['symbol'] for trade in st.session_state.open_trades]
            if opp['symbol'] in existing_symbols:
                continue
            
            # Position sizing (2% risk per trade)
            risk_amount = balance['total_value'] * 0.02
            
            # Calculate option price
            if opp['trade_type'] == 'LONG_CALLS':
                option_price = abs(opp['distance_pct']) * 0.12 + 0.8
            elif opp['trade_type'] == 'CALL_SELLING':
                option_price = opp['current_price'] * 0.025
            else:  # STRADDLE
                option_price = opp['current_price'] * 0.08
            
            quantity = max(1, int(risk_amount / (option_price * 100)))
            trade_cost = option_price * quantity * 100
            
            if trade_cost <= available_cash:
                self.add_trade(
                    symbol=opp['symbol'],
                    trade_type=opp['trade_type'],
                    entry_price=option_price,
                    quantity=quantity,
                    confidence_score=opp['confidence_score'],
                    setup_type=opp['structure_type'],
                    recommendation=f"AUTO: {opp['recommendation']}",
                    auto_trade=True
                )
                available_cash -= trade_cost
    
    def add_trade(self, symbol: str, trade_type: str, entry_price: float, 
                  quantity: int, confidence_score: int, setup_type: str, 
                  recommendation: str, auto_trade: bool = False):
        """Add a new trade to the portfolio"""
        
        # Get current stock price for reference
        current_stock_price = self.get_real_stock_price(symbol)
        
        trade = {
            'trade_id': str(uuid.uuid4()),
            'symbol': symbol,
            'trade_type': trade_type,
            'entry_date': datetime.now().date(),
            'entry_timestamp': datetime.now(),
            'entry_price': entry_price,
            'entry_stock_price': current_stock_price,
            'quantity': quantity,
            'confidence_score': confidence_score,
            'setup_type': setup_type,
            'recommendation': recommendation,
            'days_held': 0,
            'current_value': entry_price,
            'unrealized_pnl': 0,
            'unrealized_pnl_pct': 0,
            'auto_trade': auto_trade,
            'status': 'OPEN'
        }
        
        st.session_state.open_trades.append(trade)
        
        if auto_trade:
            st.info(f"🤖 **AUTO-TRADE EXECUTED:** {quantity} contracts of {symbol} {trade_type}")
        else:
            st.success(f"✅ **TRADE EXECUTED:** {quantity} contracts of {symbol} {trade_type}")
        
        return True
    
    def get_current_balance(self) -> Dict:
        """Calculate current portfolio balance"""
        self.update_open_trades()  # Update all trades first
        
        # Calculate values
        cash_balance = self.initial_balance
        positions_value = 0
        unrealized_pnl = 0
        realized_pnl = 0
        
        # Subtract invested amounts and add current position values
        for trade in st.session_state.open_trades:
            invested = trade['entry_price'] * trade['quantity'] * 100
            current_val = trade['current_value'] * trade['quantity'] * 100
            
            cash_balance -= invested  # Remove initial investment
            positions_value += current_val  # Add current value
            unrealized_pnl += trade['unrealized_pnl']
        
        # Add realized P&L from closed trades
        for trade in st.session_state.closed_trades:
            realized_pnl += trade['realized_pnl']
        
        total_value = cash_balance + positions_value + realized_pnl
        total_return_pct = ((total_value - self.initial_balance) / self.initial_balance) * 100
        
        # Calculate win rate
        if st.session_state.closed_trades:
            winning_trades = len([t for t in st.session_state.closed_trades if t['realized_pnl'] > 0])
            win_rate = (winning_trades / len(st.session_state.closed_trades)) * 100
        else:
            win_rate = 0
        
        # Update portfolio history for charting
        portfolio_snapshot = {
            'timestamp': datetime.now(),
            'total_value': total_value,
            'total_return_pct': total_return_pct,
            'positions_value': positions_value,
            'cash_balance': cash_balance,
            'realized_pnl': realized_pnl,
            'unrealized_pnl': unrealized_pnl
        }
        
        st.session_state.portfolio_history.append(portfolio_snapshot)
        
        # Keep only last 100 snapshots
        if len(st.session_state.portfolio_history) > 100:
            st.session_state.portfolio_history = st.session_state.portfolio_history[-100:]
        
        return {
            'total_value': total_value,
            'cash_balance': cash_balance,
            'positions_value': positions_value,
            'realized_pnl': realized_pnl,
            'unrealized_pnl': unrealized_pnl,
            'total_return_pct': total_return_pct,
            'open_trades_count': len(st.session_state.open_trades),
            'closed_trades_count': len(st.session_state.closed_trades),
            'win_rate': win_rate
        }
    
    def get_portfolio_performance_chart(self):
        """Generate portfolio performance chart"""
        if len(st.session_state.portfolio_history) < 2:
            return None
        
        df = pd.DataFrame(st.session_state.portfolio_history)
        
        fig = go.Figure()
        
        # Add portfolio value line
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['total_value'],
            mode='lines+markers',
            name='Portfolio Value',
            line=dict(color='#00ff87', width=3),
            marker=dict(size=6)
        ))
        
        # Add starting value line
        fig.add_hline(
            y=self.initial_balance,
            line_dash="dash",
            line_color="white",
            annotation_text=f"Starting Value: ${self.initial_balance:,}",
            annotation_position="top left"
        )
        
        # Styling
        fig.update_layout(
            title={
                'text': 'Portfolio Performance Over Time',
                'x': 0.5,
                'font': {'size': 20, 'color': 'white'}
            },
            xaxis_title="Time",
            yaxis_title="Portfolio Value ($)",
            template="plotly_dark",
            height=400,
            showlegend=True,
            xaxis=dict(color='white'),
            yaxis=dict(color='white', tickformat='$,.0f'),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig
    
    def get_trade_performance_summary(self) -> pd.DataFrame:
        """Get summary of trade performance"""
        if not st.session_state.closed_trades:
            return pd.DataFrame()
        
        df = pd.DataFrame(st.session_state.closed_trades)
        
        summary = df.groupby(['trade_type', 'setup_type']).agg({
            'realized_pnl': ['count', 'mean', 'sum'],
            'realized_pnl_pct': ['mean', lambda x: (x > 0).mean() * 100],  # win rate
            'days_held': 'mean'
        }).round(2)
        
        summary.columns = ['Trades', 'Avg P&L ($)', 'Total P&L ($)', 'Avg Return (%)', 'Win Rate (%)', 'Avg Days']
        
        return summary.reset_index()

def display_enhanced_mock_portfolio():
    """Display the enhanced mock portfolio with auto-trading"""
    st.header("💰 Enhanced Mock Trading Account - $100K Challenge")
    
    # Initialize enhanced account
    account = EnhancedMockTradingAccount()
    
    # Auto-trading controls
    st.subheader("🤖 Auto-Trading Settings")
    col1, col2 = st.columns(2)
    
    with col1:
        auto_trading = st.checkbox(
            "Enable Auto-Trading", 
            value=st.session_state.auto_trading_enabled,
            help="Automatically trade 90%+ confidence opportunities"
        )
        st.session_state.auto_trading_enabled = auto_trading
    
    with col2:
        if st.button("🔄 Update All Trades"):
            account.update_open_trades()
            st.success("All trades updated with current market data!")
    
    # Get current balance
    balance = account.get_current_balance()
    
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
        st.metric("Positions Value", f"${balance['positions_value']:,.2f}")
    
    with col4:
        st.metric(
            "Win Rate",
            f"{balance['win_rate']:.1f}%",
            f"{balance['closed_trades_count']} closed trades"
        )
    
    # Portfolio performance chart
    fig = account.get_portfolio_performance_chart()
    if fig:
        st.subheader("📈 Portfolio Performance Chart")
        st.plotly_chart(fig, use_container_width=True)
    
    # Display P&L breakdown
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Realized P&L", f"${balance['realized_pnl']:,.2f}")
    with col2:
        st.metric("Unrealized P&L", f"${balance['unrealized_pnl']:,.2f}")
    
    # Sample opportunities for demo
    sample_opportunities = [
        {
            'symbol': 'MARA', 'current_price': 25.50, 'gamma_flip': 23.20,
            'distance_pct': 9.02, 'structure_type': 'CALL_SELLING_SETUP',
            'confidence_score': 95, 'recommendation': 'SELL CALLS - above flip',
            'trade_type': 'CALL_SELLING'
        },
        {
            'symbol': 'GME', 'current_price': 22.15, 'gamma_flip': 24.80,
            'distance_pct': -11.97, 'structure_type': 'SQUEEZE_SETUP',
            'confidence_score': 92, 'recommendation': 'BUY CALLS - squeeze',
            'trade_type': 'LONG_CALLS'
        }
    ]
    
    # Auto-trade opportunities if enabled
    if st.session_state.auto_trading_enabled:
        account.auto_trade_opportunities(sample_opportunities)
    
    # Manual trading interface
    st.subheader("📈 Manual Trading")
    for i, opp in enumerate(sample_opportunities):
        with st.expander(f"🎯 {opp['symbol']} - {opp['confidence_score']}%"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write(f"**Strategy:** {opp['recommendation']}")
                st.write(f"**Price:** ${opp['current_price']:.2f} | **Flip:** ${opp['gamma_flip']:.2f}")
            
            with col2:
                if st.button(f"Manual Trade", key=f"manual_{i}"):
                    account.add_trade(
                        symbol=opp['symbol'],
                        trade_type=opp['trade_type'],
                        entry_price=2.50,
                        quantity=5,
                        confidence_score=opp['confidence_score'],
                        setup_type=opp['structure_type'],
                        recommendation=opp['recommendation']
                    )
    
    # Show open positions
    if st.session_state.open_trades:
        st.subheader("📊 Open Positions")
        trades_df = pd.DataFrame(st.session_state.open_trades)
        trades_df = trades_df[['symbol', 'trade_type', 'entry_date', 'entry_price', 
                              'current_value', 'quantity', 'days_held', 'unrealized_pnl_pct', 'confidence_score']]
        st.dataframe(trades_df, use_container_width=True)
    
    # Show closed trades
    if st.session_state.closed_trades:
        st.subheader("📈 Closed Trades")
        closed_df = pd.DataFrame(st.session_state.closed_trades)
        closed_df = closed_df[['symbol', 'trade_type', 'exit_date', 'realized_pnl', 
                              'realized_pnl_pct', 'days_held', 'exit_reason']]
        st.dataframe(closed_df, use_container_width=True)
        
        # Performance summary
        summary = account.get_trade_performance_summary()
        if not summary.empty:
            st.subheader("📊 Performance Summary")
            st.dataframe(summary, use_container_width=True)

# Example usage
if __name__ == "__main__":
    display_enhanced_mock_portfolio()
