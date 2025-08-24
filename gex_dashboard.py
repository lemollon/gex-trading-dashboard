# Enhanced Educational Gamma Exposure Trading System
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
import random

# Page configuration
st.set_page_config(
    page_title="GEX Trading System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

class GEXEducationalSystem:
    """Educational system to explain gamma exposure concepts"""
    
    @staticmethod
    def explain_market_makers():
        st.markdown("""
        ### 🎯 How Market Makers Think with Gamma Exposure
        
        **Market makers are like bookies at a casino** - they want to make money on every trade while staying neutral to price direction.
        
        #### The Market Maker's Problem:
        1. **They sell you options** but don't want to lose money if the stock moves
        2. **They must hedge their risk** by buying/selling the underlying stock
        3. **Gamma tells them HOW MUCH stock to buy/sell** when prices change
        
        #### The Magic of Gamma:
        - **High Gamma = Big hedging moves** (creates volatility)
        - **Low Gamma = Small hedging moves** (suppresses volatility)
        - **Gamma Flip Point = Where the magic switches**
        
        #### Why This Creates Opportunities:
        - 🚀 **Below flip point**: Market makers amplify moves (great for buying options)
        - 🛡️ **Above flip point**: Market makers dampen moves (great for selling options)
        - 🎯 **At walls**: Strong support/resistance levels
        """)
    
    @staticmethod
    def show_strategy_overview():
        st.markdown("""
        ### 💡 The Three Money-Making Strategies
        
        #### 1. 🚀 SQUEEZE PLAYS (Buy Calls/Puts)
        **When**: Price below gamma flip + negative GEX
        **Why**: Market makers amplify every move up
        **Target**: 50-100% gains in 1-3 days
        **Risk**: Can lose 50% quickly
        
        #### 2. 🛡️ PREMIUM SELLING (Sell Calls/Puts)  
        **When**: Price above flip + positive GEX near walls
        **Why**: Market makers suppress moves, options decay
        **Target**: 25-50% premium collection
        **Risk**: Assignment if walls break
        
        #### 3. ⚖️ IRON CONDORS (Sell both sides)
        **When**: High positive GEX + wide walls
        **Why**: Price stays trapped between walls
        **Target**: 20-40% premium collection
        **Risk**: Big move breaks setup
        """)
    
    @staticmethod
    def show_entry_criteria():
        st.markdown("""
        ### ✅ Exact Entry Criteria (What We Look For)
        
        #### 🚀 SQUEEZE SETUP CRITERIA:
        - ✅ Net GEX < -500M (negative gamma environment)
        - ✅ Price is 0.5-2% below gamma flip point
        - ✅ Strong put wall within 1% below current price
        - ✅ Major expiration < 5 days away
        - ✅ Confidence score > 75%
        
        #### 🛡️ PREMIUM SELLING CRITERIA:
        - ✅ Net GEX > +1B (positive gamma environment) 
        - ✅ Price near or above call wall (within 0.5%)
        - ✅ Call wall has >300M gamma concentration
        - ✅ 2-5 days to expiration for theta decay
        - ✅ Confidence score > 70%
        
        #### ⚖️ IRON CONDOR CRITERIA:
        - ✅ Net GEX > +2B (very positive gamma)
        - ✅ Call and put walls >3% apart
        - ✅ 80%+ gamma concentrated at the walls
        - ✅ 5-10 days to expiration
        - ✅ IV rank < 50th percentile
        """)

class EnhancedMockTradingAccount:
    """Enhanced educational mock trading with detailed explanations"""
    
    def __init__(self):
        self.initial_balance = 100000
        self.last_update_time = datetime.now()
        
        # Initialize session state
        if 'portfolio_history' not in st.session_state:
            st.session_state.portfolio_history = []
        if 'open_trades' not in st.session_state:
            st.session_state.open_trades = []
        if 'closed_trades' not in st.session_state:
            st.session_state.closed_trades = []
        if 'auto_trading_enabled' not in st.session_state:
            st.session_state.auto_trading_enabled = False
        if 'last_analysis_time' not in st.session_state:
            st.session_state.last_analysis_time = datetime.now() - timedelta(minutes=30)
    
    def get_realistic_opportunities(self) -> List[Dict]:
        """Generate realistic trading opportunities with educational context"""
        
        # Simulate morning analysis time
        st.session_state.last_analysis_time = datetime.now().replace(hour=9, minute=15, second=0, microsecond=0)
        
        opportunities = [
            {
                'symbol': 'TSLA',
                'current_price': 245.67,
                'gamma_flip': 238.50,
                'distance_pct': 3.01,
                'net_gex': 1250000000,  # 1.25B positive
                'call_wall': 250.00,
                'put_wall': 235.00,
                'structure_type': 'PREMIUM_SELLING_SETUP',
                'confidence_score': 88,
                'trade_type': 'CALL_SELLING',
                'recommendation': 'SELL $250 CALLS - Price approaching call wall with high positive GEX',
                'explanation': 'TSLA is trading near the $250 call wall with massive positive GEX. Market makers will defend this level by selling stock as price approaches, creating resistance.',
                'entry_logic': '✅ Above gamma flip (+3.01%) ✅ Near call wall ($250) ✅ High positive GEX (1.25B)',
                'profit_target': '30-50% in 2-3 days',
                'risk_warning': 'Risk: If TSLA breaks $250 decisively, calls could be assigned',
                'expected_premium': 3.20,
                'days_to_expiry': 3
            },
            {
                'symbol': 'NVDA',
                'current_price': 118.45,
                'gamma_flip': 125.20,
                'distance_pct': -5.39,
                'net_gex': -850000000,  # -850M negative
                'call_wall': 130.00,
                'put_wall': 115.00,
                'structure_type': 'SQUEEZE_SETUP',
                'confidence_score': 93,
                'trade_type': 'LONG_CALLS',
                'recommendation': 'BUY $120/$125 CALLS - Classic squeeze setup below flip',
                'explanation': 'NVDA is trading 5.39% below gamma flip with negative GEX. Any upward move will force market makers to buy more stock, amplifying the move.',
                'entry_logic': '✅ Below gamma flip (-5.39%) ✅ Negative GEX (-850M) ✅ Put wall support at $115',
                'profit_target': '75-150% if it breaks above $125 flip',
                'risk_warning': 'Risk: Can lose 50%+ if it drops to put wall at $115',
                'expected_premium': 2.85,
                'days_to_expiry': 2
            },
            {
                'symbol': 'SPY',
                'current_price': 565.23,
                'gamma_flip': 563.00,
                'distance_pct': 0.40,
                'net_gex': 2100000000,  # 2.1B positive
                'call_wall': 570.00,
                'put_wall': 560.00,
                'structure_type': 'IRON_CONDOR_SETUP',
                'confidence_score': 76,
                'trade_type': 'IRON_CONDOR',
                'recommendation': 'IRON CONDOR 560/570 - Perfect range-bound setup',
                'explanation': 'SPY has massive positive GEX with clear walls at $560 and $570. Price should stay trapped between these levels with high probability.',
                'entry_logic': '✅ High positive GEX (2.1B) ✅ Wide walls ($560-$570) ✅ Price in middle of range',
                'profit_target': '25-40% premium collection',
                'risk_warning': 'Risk: Major news could break the range',
                'expected_premium': 1.50,
                'days_to_expiry': 7
            },
            {
                'symbol': 'AMD',
                'current_price': 142.89,
                'gamma_flip': 145.75,
                'distance_pct': -1.96,
                'net_gex': -320000000,  # -320M negative
                'call_wall': 150.00,
                'put_wall': 140.00,
                'structure_type': 'POTENTIAL_SQUEEZE',
                'confidence_score': 67,
                'trade_type': 'LONG_CALLS',
                'recommendation': 'BUY $145 CALLS - Near flip point, moderate setup',
                'explanation': 'AMD is close to gamma flip with moderate negative GEX. Not as strong as NVDA setup but still has squeeze potential.',
                'entry_logic': '✅ Below gamma flip (-1.96%) ⚠️ Moderate negative GEX (-320M) ✅ Put support nearby',
                'profit_target': '40-80% if it breaks flip',
                'risk_warning': 'Risk: Lower confidence, could fail at flip point',
                'expected_premium': 2.10,
                'days_to_expiry': 4
            }
        ]
        
        return opportunities
    
    def calculate_option_value(self, trade: Dict) -> float:
        """Calculate realistic option value with educational breakdown"""
        
        days_held = trade.get('days_held', 0)
        entry_price = trade['entry_price']
        
        # Simulate realistic price movement based on setup type
        if trade['trade_type'] == 'LONG_CALLS':
            # Simulate stock movement and gamma effect
            base_move = random.uniform(-0.08, 0.12)  # -8% to +12%
            
            if base_move > 0:
                # Positive moves amplified by gamma
                gamma_multiplier = 1 + (abs(base_move) * 2.5)  # 2.5x leverage
                option_multiplier = min(gamma_multiplier, 3.0)  # Cap at 3x
            else:
                # Negative moves hurt options
                option_multiplier = max(1 + (base_move * 2), 0.1)  # Max 90% loss
        
        elif trade['trade_type'] == 'CALL_SELLING':
            # Sold calls decay over time unless stock moves up aggressively
            time_decay = min(days_held * 0.15, 0.7)  # 15% per day decay
            stock_move = random.uniform(-0.05, 0.08)  # -5% to +8%
            
            if stock_move > 0.03:  # If stock up >3%
                option_multiplier = 1 + (stock_move * 3)  # Options gain value
            else:
                option_multiplier = max(1 - time_decay, 0.2)  # Time decay wins
        
        else:  # IRON_CONDOR
            # Iron condors profit from time decay and low volatility
            time_decay = min(days_held * 0.08, 0.5)  # 8% per day
            volatility = abs(random.uniform(-0.04, 0.04))  # Small moves
            
            if volatility < 0.02:  # Low volatility = good
                option_multiplier = max(1 - time_decay, 0.4)
            else:  # High volatility = bad
                option_multiplier = 1 + (volatility * 4)
        
        # Apply time decay
        time_decay_factor = max(1 - (days_held * 0.05), 0.3)
        
        current_value = entry_price * option_multiplier * time_decay_factor
        return max(current_value, 0.05)
    
    def get_educational_trade_analysis(self, trade: Dict) -> str:
        """Provide educational analysis of why trade is winning/losing"""
        
        pnl_pct = trade.get('unrealized_pnl_pct', 0)
        days_held = trade.get('days_held', 0)
        
        if pnl_pct > 25:
            return f"🎉 **WINNING TRADE**: This {trade['trade_type']} is up {pnl_pct:.1f}% because the gamma structure is playing out as expected. Market makers are hedging exactly as predicted!"
        
        elif pnl_pct < -25:
            return f"⚠️ **LOSING TRADE**: This {trade['trade_type']} is down {pnl_pct:.1f}%. Either the gamma structure changed or price moved against our thesis. Time to reassess."
        
        else:
            return f"📊 **DEVELOPING TRADE**: This {trade['trade_type']} is {pnl_pct:+.1f}% after {days_held} day(s). Still within expected range - let it play out."
    
    def check_exit_conditions(self, trade: Dict) -> Optional[str]:
        """Enhanced exit logic with educational explanations"""
        
        pnl_pct = trade.get('unrealized_pnl_pct', 0)
        days_held = trade.get('days_held', 0)
        confidence = trade.get('confidence_score', 70)
        
        # Profit targets based on strategy
        if trade['trade_type'] == 'LONG_CALLS':
            if pnl_pct >= 100:
                return "🎯 Profit Target: 100% gain on squeeze play"
            if pnl_pct >= 75 and days_held >= 2:
                return "🎯 Profit Target: 75% gain - take profits on calls"
        
        elif trade['trade_type'] == 'CALL_SELLING':
            if pnl_pct >= 50:
                return "🎯 Profit Target: 50% premium collected"
            if pnl_pct >= 30 and days_held >= 2:
                return "🎯 Profit Target: 30% premium - close early"
        
        elif trade['trade_type'] == 'IRON_CONDOR':
            if pnl_pct >= 40:
                return "🎯 Profit Target: 40% of max profit on condor"
            if pnl_pct >= 25 and days_held >= 3:
                return "🎯 Profit Target: 25% profit on condor"
        
        # Stop losses
        if pnl_pct <= -50:
            return "🛑 Stop Loss: 50% maximum loss rule"
        
        # Time stops
        if days_held >= 7:
            return "⏰ Time Stop: Maximum 7-day hold period"
        
        # Confidence-based stops
        if confidence >= 85 and pnl_pct <= -35:
            return "🎯 High Confidence Stop: -35% on strong setup"
        
        if confidence < 70 and pnl_pct <= -25:
            return "⚠️ Low Confidence Stop: -25% on weak setup"
        
        return None
    
    def add_trade(self, opportunity: Dict, manual: bool = False):
        """Add trade with full educational context"""
        
        # Calculate position size (2% risk per trade)
        balance = self.get_current_balance()
        risk_amount = balance['total_value'] * 0.02
        
        entry_price = opportunity['expected_premium']
        quantity = max(1, int(risk_amount / (entry_price * 100)))
        
        trade = {
            'trade_id': str(uuid.uuid4()),
            'symbol': opportunity['symbol'],
            'trade_type': opportunity['trade_type'],
            'entry_date': datetime.now().date(),
            'entry_timestamp': datetime.now(),
            'entry_price': entry_price,
            'quantity': quantity,
            'confidence_score': opportunity['confidence_score'],
            'setup_type': opportunity['structure_type'],
            'recommendation': opportunity['recommendation'],
            'explanation': opportunity['explanation'],
            'entry_logic': opportunity['entry_logic'],
            'profit_target': opportunity['profit_target'],
            'risk_warning': opportunity['risk_warning'],
            'days_held': 0,
            'current_value': entry_price,
            'unrealized_pnl': 0,
            'unrealized_pnl_pct': 0,
            'manual_trade': manual,
            'status': 'OPEN',
            # GEX data for analysis
            'entry_gex': opportunity['net_gex'],
            'gamma_flip': opportunity['gamma_flip'],
            'call_wall': opportunity.get('call_wall'),
            'put_wall': opportunity.get('put_wall'),
        }
        
        st.session_state.open_trades.append(trade)
        
        # Show educational popup
        if manual:
            st.success(f"✅ **MANUAL TRADE EXECUTED**")
        else:
            st.info(f"🤖 **AUTO-TRADE EXECUTED**")
        
        st.info(f"""
        **Trade Details:**
        - {quantity} contracts of {opportunity['symbol']} {opportunity['trade_type']}
        - Entry: ${entry_price:.2f} per contract
        - Total Cost: ${entry_price * quantity * 100:,.2f}
        - Strategy: {opportunity['structure_type']}
        """)
        
        return True
    
    def update_open_trades(self):
        """Update trades with educational analysis"""
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
            
            # Add educational analysis
            trade['analysis'] = self.get_educational_trade_analysis(trade)
            
            # Check exit conditions
            exit_reason = self.check_exit_conditions(trade)
            
            if exit_reason:
                self.close_trade(trade, exit_reason)
            else:
                updated_trades.append(trade)
        
        st.session_state.open_trades = updated_trades
    
    def close_trade(self, trade: Dict, exit_reason: str):
        """Close trade with educational summary"""
        exit_trade = trade.copy()
        exit_trade['exit_date'] = datetime.now().date()
        exit_trade['exit_price'] = trade['current_value']
        exit_trade['exit_reason'] = exit_reason
        exit_trade['realized_pnl'] = trade['unrealized_pnl']
        exit_trade['realized_pnl_pct'] = trade['unrealized_pnl_pct']
        exit_trade['status'] = 'CLOSED'
        
        # Add educational summary
        if exit_trade['realized_pnl'] > 0:
            exit_trade['lesson_learned'] = f"✅ SUCCESS: The {exit_trade['setup_type']} played out perfectly. Market makers behaved as predicted."
        else:
            exit_trade['lesson_learned'] = f"📚 LESSON: The {exit_trade['setup_type']} didn't work. Gamma structure likely changed or external factors interfered."
        
        st.session_state.closed_trades.append(exit_trade)
        
        # Educational notification
        if exit_trade['realized_pnl'] > 0:
            st.success(f"""
            🎉 **WINNING TRADE CLOSED**
            {exit_trade['symbol']} {exit_trade['trade_type']}: +{exit_trade['realized_pnl_pct']:.1f}% 
            Reason: {exit_reason}
            💡 {exit_trade['lesson_learned']}
            """)
        else:
            st.warning(f"""
            📚 **LEARNING TRADE CLOSED**
            {exit_trade['symbol']} {exit_trade['trade_type']}: {exit_trade['realized_pnl_pct']:.1f}%
            Reason: {exit_reason}
            💡 {exit_trade['lesson_learned']}
            """)
    
    def get_current_balance(self) -> Dict:
        """Calculate current balance with detailed breakdown"""
        self.update_open_trades()
        
        cash_balance = self.initial_balance
        positions_value = 0
        unrealized_pnl = 0
        realized_pnl = 0
        
        # Calculate open positions
        for trade in st.session_state.open_trades:
            invested = trade['entry_price'] * trade['quantity'] * 100
            current_val = trade['current_value'] * trade['quantity'] * 100
            
            cash_balance -= invested
            positions_value += current_val
            unrealized_pnl += trade['unrealized_pnl']
        
        # Add realized P&L
        for trade in st.session_state.closed_trades:
            realized_pnl += trade['realized_pnl']
        
        total_value = cash_balance + positions_value + realized_pnl
        total_return_pct = ((total_value - self.initial_balance) / self.initial_balance) * 100
        
        # Win rate and other stats
        if st.session_state.closed_trades:
            winning_trades = len([t for t in st.session_state.closed_trades if t['realized_pnl'] > 0])
            win_rate = (winning_trades / len(st.session_state.closed_trades)) * 100
            avg_winner = np.mean([t['realized_pnl_pct'] for t in st.session_state.closed_trades if t['realized_pnl'] > 0]) if winning_trades > 0 else 0
            avg_loser = np.mean([t['realized_pnl_pct'] for t in st.session_state.closed_trades if t['realized_pnl'] <= 0]) if len(st.session_state.closed_trades) - winning_trades > 0 else 0
        else:
            win_rate = 0
            avg_winner = 0
            avg_loser = 0
        
        return {
            'total_value': total_value,
            'cash_balance': cash_balance,
            'positions_value': positions_value,
            'realized_pnl': realized_pnl,
            'unrealized_pnl': unrealized_pnl,
            'total_return_pct': total_return_pct,
            'open_trades_count': len(st.session_state.open_trades),
            'closed_trades_count': len(st.session_state.closed_trades),
            'win_rate': win_rate,
            'avg_winner': avg_winner,
            'avg_loser': avg_loser
        }

def display_morning_analysis():
    """Enhanced morning analysis with clear timing and explanations"""
    st.header("🌅 Morning Gamma Analysis - Live Trading Opportunities")
    
    # Analysis timing info
    current_time = datetime.now()
    last_update = st.session_state.get('last_analysis_time', current_time - timedelta(minutes=30))
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"**Last Analysis:** {last_update.strftime('%I:%M %p ET')}")
    with col2:
        st.info(f"**Current Time:** {current_time.strftime('%I:%M %p ET')}")
    with col3:
        st.info(f"**Next Update:** {(current_time + timedelta(minutes=15)).strftime('%I:%M %p ET')}")
    
    st.markdown("""
    ### 📊 How Our Analysis Works
    
    **Every Morning at 9:15 AM ET, we:**
    1. 🔍 Scan 100+ stocks for gamma exposure patterns
    2. 🧮 Calculate exact gamma flip points and wall levels  
    3. 🎯 Identify the 3-5 highest probability setups
    4. 📈 Rank by confidence score (65%+ makes the list)
    5. 🚨 Alert on 90%+ confidence opportunities for auto-trading
    
    **Why 9:15 AM?** Options market makers adjust their hedging positions based on overnight news and pre-market moves. By 9:15, the gamma structure is clear and opportunities are identified.
    """)
    
    # Get opportunities
    account = EnhancedMockTradingAccount()
    opportunities = account.get_realistic_opportunities()
    
    # Display opportunities with full educational context
    st.subheader("🎯 Today's Top Opportunities")
    
    for i, opp in enumerate(opportunities):
        confidence_color = "🟢" if opp['confidence_score'] >= 85 else "🟡" if opp['confidence_score'] >= 75 else "🟠"
        
        with st.expander(f"{confidence_color} {opp['symbol']} - {opp['confidence_score']}% Confidence - {opp['structure_type']}"):
            
            # Key metrics
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"""
                **🎯 Strategy:** {opp['recommendation']}
                
                **📈 Current Setup:**
                - Price: ${opp['current_price']:.2f}
                - Gamma Flip: ${opp['gamma_flip']:.2f} ({opp['distance_pct']:+.1f}%)
                - Net GEX: {opp['net_gex']/1000000:.0f}M
                """)
                
                if 'call_wall' in opp:
                    st.markdown(f"- Call Wall: ${opp['call_wall']:.2f}")
                if 'put_wall' in opp:
                    st.markdown(f"- Put Wall: ${opp['put_wall']:.2f}")
            
            with col2:
                # Visual setup diagram
                fig = go.Figure()
                
                current = opp['current_price']
                flip = opp['gamma_flip']
                
                # Add price levels
                fig.add_hline(y=current, line_dash="solid", line_color="yellow", 
                             annotation_text=f"Current: ${current:.2f}")
                fig.add_hline(y=flip, line_dash="dash", line_color="orange",
                             annotation_text=f"Gamma Flip: ${flip:.2f}")
                
                if 'call_wall' in opp:
                    fig.add_hline(y=opp['call_wall'], line_dash="dot", line_color="red",
                                 annotation_text=f"Call Wall: ${opp['call_wall']:.2f}")
                
                if 'put_wall' in opp:
                    fig.add_hline(y=opp['put_wall'], line_dash="dot", line_color="green",
                                 annotation_text=f"Put Wall: ${opp['put_wall']:.2f}")
                
                fig.update_layout(
                    title=f"{opp['symbol']} Levels",
                    height=300,
                    yaxis_title="Price ($)",
                    showlegend=False,
                    template="plotly_dark"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Educational explanation
            st.markdown(f"""
            ### 💡 Why This Works:
            {opp['explanation']}
            
            ### ✅ Entry Logic:
            {opp['entry_logic']}
            
            ### 🎯 Profit Target:
            {opp['profit_target']}
            
            ### ⚠️ Risk Warning:
            {opp['risk_warning']}
            """)
            
            # Trading buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button(f"📈 Execute Trade", key=f"trade_{i}"):
                    account.add_trade(opp, manual=True)
            
            with col2:
                st.info(f"Estimated Premium: ${opp['expected_premium']:.2f}")

def display_enhanced_portfolio():
    """Enhanced portfolio with educational features"""
    st.header("💰 $100K Gamma Trading Challenge - Learn While You Earn")
    
    account = EnhancedMockTradingAccount()
    
    # Educational intro
    st.markdown("""
    ### 🎓 Welcome to Your Educational Trading Account
    
    This is your **$100,000 virtual trading account** where you can practice gamma exposure strategies without risk.
    Every trade comes with detailed explanations of why it works (or doesn't work).
    
    **Key Features:**
    - 🎯 Start with $100K virtual money
    - 📚 Learn from every win AND loss
    - 🤖 Auto-trading for 90%+ confidence setups
    - 📊 Track performance vs. gamma predictions
    """)
    
    # Auto-trading toggle with explanation
    st.subheader("🤖 Auto-Trading Education")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        auto_trading = st.checkbox(
            "Enable Auto-Trading",
            value=st.session_state.auto_trading_enabled,
            help="Automatically execute trades with 90%+ confidence"
        )
        st.session_state.auto_trading_enabled = auto_trading
    
    with col2:
        if auto_trading:
            st.info("🤖 **Auto-trading ON**: Will execute 90%+ confidence trades automatically")
        else:
            st.warning("👤 **Manual mode**: You choose which trades to execute")
    
    # Portfolio metrics
    balance = account.get_current_balance()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        color = "normal" if balance['total_return_pct'] >= 0 else "inverse"
        st.metric(
            "Portfolio Value",
            f"${balance['total_value']:,.0f}",
            f"{balance['total_return_pct']:+.1f}%",
            delta_color=color
        )
    
    with col2:
        st.metric("Cash Available", f"${balance['cash_balance']:,.0f}")
    
    with col3:
        st.metric("Win Rate", f"{balance['win_rate']:.0f}%", 
                 f"{balance['closed_trades_count']} completed")
    
    with col4:
        if balance['avg_winner'] > 0:
            st.metric("Avg Winner", f"+{balance['avg_winner']:.1f}%",
                     f"Avg Loser: {balance['avg_loser']:.1f}%")
        else:
            st.metric("Open Trades", f"{balance['open_trades_count']}")
    
    # Current positions with educational analysis
    if st.session_state.open_trades:
        st.subheader("📊 Your Active Positions")
        
        for trade in st.session_state.open_trades:
            pnl_color = "🟢" if trade['unrealized_pnl_pct'] > 0 else "🔴" if trade['unrealized_pnl_pct'] < -10 else "🟡"
            
            with st.expander(f"{pnl_color} {trade['symbol']} {trade['trade_type']} - {trade['unrealized_pnl_pct']:+.1f}%"):
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"""
                    **Original Strategy:** {trade['setup_type']}
                    **Entry Logic:** {trade['entry_logic']}
                    **Days Held:** {trade['days_held']}
                    **Position Size:** {trade['quantity']} contracts
                    
                    {trade.get('analysis', '')}
                    """)
                
                with col2:
                    st.metric("Current P&L", 
                             f"${trade['unrealized_pnl']:,.0f}",
                             f"{trade['unrealized_pnl_pct']:+.1f}%")
                    
                    if st.button(f"Close Position", key=f"close_{trade['trade_id']}"):
                        account.close_trade(trade, "Manual Close")
                        st.rerun()
    
    # Completed trades with lessons learned
    if st.session_state.closed_trades:
        st.subheader("📈 Completed Trades - Learning History")
        
        recent_trades = st.session_state.closed_trades[-5:]  # Show last 5
        
        for trade in reversed(recent_trades):
            result_color = "🎉" if trade['realized_pnl'] > 0 else "📚"
            
            with st.expander(f"{result_color} {trade['symbol']} - {trade['realized_pnl_pct']:+.1f}% - {trade['exit_reason']}"):
                st.markdown(f"""
                **Strategy Used:** {trade['setup_type']}
                **Held for:** {trade['days_held']} days
                **Final P&L:** ${trade['realized_pnl']:,.0f} ({trade['realized_pnl_pct']:+.1f}%)
                
                **💡 Lesson Learned:**
                {trade.get('lesson_learned', 'Trade completed successfully.')}
                """)
    
    # Performance analytics
    if balance['closed_trades_count'] > 0:
        st.subheader("📊 Performance Analytics")
        
        # Create performance breakdown by strategy type
        df = pd.DataFrame(st.session_state.closed_trades)
        
        if not df.empty:
            strategy_performance = df.groupby('setup_type').agg({
                'realized_pnl_pct': ['count', 'mean', lambda x: (x > 0).mean() * 100],
                'days_held': 'mean'
            }).round(2)
            
            strategy_performance.columns = ['Count', 'Avg Return %', 'Win Rate %', 'Avg Days']
            
            st.dataframe(strategy_performance, use_container_width=True)

def main():
    """Main application with enhanced navigation and education"""
    
    st.title("🚀 Gamma Exposure Trading System")
    st.markdown("**Learn Professional Options Trading Through Market Maker Psychology**")
    
    # Enhanced sidebar navigation
    st.sidebar.title("🎯 Navigation")
    
    page = st.sidebar.selectbox("Choose Section:", [
        "🎓 Learn the Strategy",
        "🌅 Morning Analysis", 
        "💰 Trading Challenge",
        "📊 Portfolio Performance"
    ])
    
    # Educational sidebar content
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### 📚 Quick Reference
    
    **🚀 Squeeze Plays:**
    - Below gamma flip
    - Negative GEX
    - Target: 50-100%
    
    **🛡️ Premium Selling:**
    - Above gamma flip  
    - Positive GEX
    - Target: 25-50%
    
    **⚖️ Iron Condors:**
    - High positive GEX
    - Wide walls
    - Target: 20-40%
    """)
    
    # Page routing with educational enhancements
    if page == "🎓 Learn the Strategy":
        st.header("🎓 Master Gamma Exposure Trading")
        
        # Educational tabs
        tab1, tab2, tab3 = st.tabs(["📖 How It Works", "💡 Strategies", "✅ Entry Rules"])
        
        with tab1:
            GEXEducationalSystem.explain_market_makers()
            
            # Interactive demo
            st.subheader("🎯 Interactive Demo: Gamma in Action")
            
            demo_price = st.slider("Stock Price", 95, 105, 100)
            gamma_flip = 98
            
            if demo_price < gamma_flip:
                st.error(f"📉 Below Flip: Market makers will AMPLIFY moves up! Perfect for buying calls.")
            else:
                st.success(f"📈 Above Flip: Market makers will DAMPEN moves down! Perfect for selling options.")
        
        with tab2:
            GEXEducationalSystem.show_strategy_overview()
        
        with tab3:
            GEXEducationalSystem.show_entry_criteria()
    
    elif page == "🌅 Morning Analysis":
        display_morning_analysis()
    
    elif page == "💰 Trading Challenge":
        display_enhanced_portfolio()
    
    elif page == "📊 Portfolio Performance":
        st.header("📊 Advanced Performance Analytics")
        
        account = EnhancedMockTradingAccount()
        balance = account.get_current_balance()
        
        if balance['closed_trades_count'] == 0:
            st.info("📈 Start trading to see performance analytics here!")
            return
        
        # Detailed analytics would go here
        st.metric("Total Return", f"{balance['total_return_pct']:+.1f}%")
        
        # Performance by strategy
        if st.session_state.closed_trades:
            df = pd.DataFrame(st.session_state.closed_trades)
            
            fig = px.bar(
                df.groupby('setup_type')['realized_pnl_pct'].mean().reset_index(),
                x='setup_type', 
                y='realized_pnl_pct',
                title="Average Return by Strategy Type",
                color='realized_pnl_pct',
                color_continuous_scale="RdYlGn"
            )
            
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
