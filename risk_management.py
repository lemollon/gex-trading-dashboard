"""
Risk Management - Portfolio risk controls and position sizing
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class RiskMetrics:
    """Risk metrics for a position or portfolio"""
    max_loss: float
    expected_return: float
    sharpe_ratio: float
    max_drawdown: float
    var_95: float  # Value at Risk 95%
    kelly_fraction: float
    risk_score: int  # 1-10 scale

@dataclass 
class PositionRisk:
    """Risk assessment for individual position"""
    symbol: str
    setup_type: str
    position_size: float
    max_loss_dollars: float
    max_loss_percent: float
    profit_probability: float
    risk_reward_ratio: float
    holding_period: int
    risk_level: str

class RiskManager:
    """
    Comprehensive risk management system for GEX trading
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        
    def _default_config(self) -> Dict:
        """Default risk management configuration"""
        return {
            # Portfolio limits
            'max_portfolio_risk': 10.0,  # % of portfolio at risk
            'max_single_position': 3.0,  # % per position
            'max_sector_concentration': 15.0,  # % per sector
            'max_correlation_exposure': 20.0,  # % in correlated positions
            
            # Position sizing
            'kelly_multiplier': 0.25,  # Conservative Kelly sizing
            'min_position_size': 0.5,   # Minimum position size
            'max_leverage': 1.0,        # No leverage by default
            
            # Risk limits
            'max_daily_loss': 2.0,      # % daily loss limit
            'max_weekly_loss': 5.0,     # % weekly loss limit
            'max_monthly_loss': 10.0,   # % monthly loss limit
            
            # Options-specific
            'max_options_allocation': 25.0,  # % in options
            'min_time_to_expiry': 2,         # Days minimum
            'max_iv_percentile': 80,         # Don't buy high IV
            
            # GEX-specific
            'min_confidence_threshold': 65,   # Minimum setup confidence
            'max_negative_gex_exposure': 15,  # % in negative GEX plays
            'max_premium_selling': 20,       # % in premium selling
        }
    
    def calculate_position_size(
        self,
        setup: 'TradingSetup',
        portfolio_value: float,
        current_positions: Dict = None
    ) -> Dict:
        """
        Calculate optimal position size for a setup
        
        Args:
            setup: TradingSetup object
            portfolio_value: Total portfolio value
            current_positions: Current position allocations
            
        Returns:
            Dictionary with position sizing recommendations
        """
        
        current_positions = current_positions or {}
        
        # Base position size from setup
        base_size = setup.size_percentage
        
        # Apply Kelly criterion adjustment
        kelly_size = self._calculate_kelly_size(setup)
        
        # Risk-adjusted size
        risk_adjusted_size = self._risk_adjust_size(setup, base_size)
        
        # Portfolio constraint adjustments
        portfolio_adjusted_size = self._apply_portfolio_constraints(
            setup, risk_adjusted_size, current_positions
        )
        
        # Final size calculation
        recommended_size = min(
            base_size,
            kelly_size,
            risk_adjusted_size, 
            portfolio_adjusted_size,
            self.config['max_single_position']
        )
        
        # Convert to dollar amount
        dollar_amount = (recommended_size / 100) * portfolio_value
        
        # Calculate risk metrics
        risk_metrics = self._calculate_position_risk(setup, recommended_size, portfolio_value)
        
        return {
            'recommended_size_percent': recommended_size,
            'dollar_amount': dollar_amount,
            'base_size': base_size,
            'kelly_size': kelly_size,
            'risk_adjusted_size': risk_adjusted_size,
            'final_adjustment_reason': self._get_adjustment_reason(
                base_size, recommended_size
            ),
            'risk_metrics': risk_metrics,
            'max_loss': dollar_amount * (setup.stop_price / setup.entry_price - 1) if setup.stop_price else dollar_amount * 0.5
        }
    
    def _calculate_kelly_size(self, setup: 'TradingSetup') -> float:
        """Calculate Kelly criterion position size"""
        
        # Estimate win probability from confidence
        win_prob = setup.confidence / 100
        
        # Estimate win/loss ratio from setup
        if setup.target_price and setup.stop_price:
            win_amount = abs(setup.target_price - setup.entry_price)
            loss_amount = abs(setup.entry_price - setup.stop_price)
            win_loss_ratio = win_amount / loss_amount if loss_amount > 0 else 2.0
        else:
            # Default estimates by setup type
            win_loss_ratios = {
                'SQUEEZE_PLAY': 2.0,
                'PREMIUM_SELLING': 0.5,
                'IRON_CONDOR': 0.3,
                'GAMMA_FLIP': 1.5
            }
            win_loss_ratio = win_loss_ratios.get(setup.setup_type, 1.5)
        
        # Kelly formula: f = (bp - q) / b
        # where b = win/loss ratio, p = win prob, q = loss prob
        kelly_fraction = (win_loss_ratio * win_prob - (1 - win_prob)) / win_loss_ratio
        
        # Apply conservative multiplier and bounds
        kelly_size = max(0, kelly_fraction * self.config['kelly_multiplier'])
        kelly_size = min(kelly_size, self.config['max_single_position'])
        
        return kelly_size
    
    def _risk_adjust_size(self, setup: 'TradingSetup', base_size: float) -> float:
        """Adjust size based on risk characteristics"""
        
        risk_multipliers = {
            'LOW': 1.2,
            'MEDIUM': 1.0,
            'HIGH': 0.7
        }
        
        risk_multiplier = risk_multipliers.get(setup.risk_level, 1.0)
        
        # Adjust for setup type risk
        setup_multipliers = {
            'SQUEEZE_PLAY': 0.8,      # High volatility
            'PREMIUM_SELLING': 1.1,   # More predictable
            'IRON_CONDOR': 1.0,       # Neutral
            'GAMMA_FLIP': 0.9         # Uncertain direction
        }
        
        setup_multiplier = setup_multipliers.get(setup.setup_type, 1.0)
        
        # Adjust for confidence
        confidence_multiplier = 0.5 + (setup.confidence / 100) * 0.8
        
        adjusted_size = base_size * risk_multiplier * setup_multiplier * confidence_multiplier
        
        return max(self.config['min_position_size'], adjusted_size)
    
    def _apply_portfolio_constraints(
        self,
        setup: 'TradingSetup',
        proposed_size: float,
        current_positions: Dict
    ) -> float:
        """Apply portfolio-level constraints"""
        
        # Calculate current risk exposure
        current_risk = sum(pos.get('risk_percent', 0) for pos in current_positions.values())
        
        # Check portfolio risk limit
        if current_risk + proposed_size > self.config['max_portfolio_risk']:
            max_additional = self.config['max_portfolio_risk'] - current_risk
            proposed_size = max(0, min(proposed_size, max_additional))
        
        # Check setup type concentration
        setup_exposure = sum(
            pos.get('size_percent', 0) 
            for pos in current_positions.values() 
            if pos.get('setup_type') == setup.setup_type
        )
        
        type_limits = {
            'SQUEEZE_PLAY': self.config['max_negative_gex_exposure'],
            'PREMIUM_SELLING': self.config['max_premium_selling'],
            'IRON_CONDOR': 10.0,
            'GAMMA_FLIP': 8.0
        }
        
        type_limit = type_limits.get(setup.setup_type, 5.0)
        
        if setup_exposure + proposed_size > type_limit:
            max_additional = type_limit - setup_exposure
            proposed_size = max(0, min(proposed_size, max_additional))
        
        return proposed_size
    
    def _calculate_position_risk(
        self,
        setup: 'TradingSetup',
        position_size: float,
        portfolio_value: float
    ) -> PositionRisk:
        """Calculate comprehensive position risk metrics"""
        
        dollar_size = (position_size / 100) * portfolio_value
        
        # Estimate maximum loss
        if setup.stop_price:
            max_loss_percent = abs(setup.entry_price - setup.stop_price) / setup.entry_price
        else:
            # Default max loss by setup type
            default_losses = {
                'SQUEEZE_PLAY': 0.5,      # Options can lose 100%
                'PREMIUM_SELLING': 0.3,   # Limited by strikes
                'IRON_CONDOR': 0.2,       # Limited risk
                'GAMMA_FLIP': 0.4         # Directional uncertainty
            }
            max_loss_percent = default_losses.get(setup.setup_type, 0.5)
        
        max_loss_dollars = dollar_size * max_loss_percent
        max_loss_portfolio_percent = (max_loss_dollars / portfolio_value) * 100
        
        # Estimate profit probability from confidence
        profit_prob = setup.confidence / 100
        
        # Calculate risk-reward ratio
        if setup.target_price:
            potential_profit = abs(setup.target_price - setup.entry_price) / setup.entry_price
            risk_reward = potential_profit / max_loss_percent if max_loss_percent > 0 else 0
        else:
            risk_reward = 1.5  # Default estimate
        
        return PositionRisk(
            symbol=setup.symbol,
            setup_type=setup.setup_type,
            position_size=position_size,
            max_loss_dollars=max_loss_dollars,
            max_loss_percent=max_loss_portfolio_percent,
            profit_probability=profit_prob,
            risk_reward_ratio=risk_reward,
            holding_period=setup.hold_days,
            risk_level=setup.risk_level
        )
    
    def _get_adjustment_reason(self, original_size: float, final_size: float) -> str:
        """Get human-readable reason for size adjustment"""
        
        if abs(original_size - final_size) < 0.1:
            return "No adjustment needed"
        elif final_size < original_size:
            reduction = ((original_size - final_size) / original_size) * 100
            return f"Reduced {reduction:.0f}% due to risk constraints"
        else:
            increase = ((final_size - original_size) / original_size) * 100
            return f"Increased {increase:.0f}% due to favorable risk profile"
    
    def assess_portfolio_risk(
        self,
        positions: Dict,
        portfolio_value: float
    ) -> RiskMetrics:
        """Assess overall portfolio risk"""
        
        if not positions:
            return RiskMetrics(0, 0, 0, 0, 0, 0, 1)
        
        # Calculate portfolio-level metrics
        total_risk = sum(pos.get('max_loss', 0) for pos in positions.values())
        total_expected_return = sum(pos.get('expected_return', 0) for pos in positions.values())
        
        # Risk percentages
        risk_percent = (total_risk / portfolio_value) * 100
        expected_return_percent = (total_expected_return / portfolio_value) * 100
        
        # Estimate Sharpe ratio (simplified)
        volatility = np.sqrt(len(positions)) * 0.15  # Rough estimate
        sharpe = expected_return_percent / volatility if volatility > 0 else 0
        
        # Calculate diversification metrics
        setup_types = [pos.get('setup_type', 'UNKNOWN') for pos in positions.values()]
        type_diversity = len(set(setup_types)) / len(setup_types) if setup_types else 0
        
        # Risk score (1-10 scale)
        risk_score = min(10, max(1, int(risk_percent / 2)))
        
        # Apply diversification bonus
        if type_diversity > 0.6:
            risk_score = max(1, risk_score - 1)
        
        return RiskMetrics(
            max_loss=total_risk,
            expected_return=total_expected_return,
            sharpe_ratio=sharpe,
            max_drawdown=risk_percent,
            var_95=total_risk * 1.65,  # Rough VaR estimate
            kelly_fraction=0.25,  # Conservative
            risk_score=risk_score
        )
    
    def check_risk_limits(
        self,
        positions: Dict,
        portfolio_value: float,
        daily_pnl: float = 0
    ) -> Dict:
        """Check if portfolio exceeds risk limits"""
        
        violations = []
        warnings = []
        
        # Daily loss check
        daily_loss_percent = (daily_pnl / portfolio_value) * 100
        if daily_loss_percent < -self.config['max_daily_loss']:
            violations.append(f"Daily loss limit exceeded: {daily_loss_percent:.1f}%")
        elif daily_loss_percent < -self.config['max_daily_loss'] * 0.8:
            warnings.append(f"Approaching daily loss limit: {daily_loss_percent:.1f}%")
        
        # Portfolio risk check
        total_risk = sum(pos.get('max_loss', 0) for pos in positions.values())
        risk_percent = (total_risk / portfolio_value) * 100
        
        if risk_percent > self.config['max_portfolio_risk']:
            violations.append(f"Portfolio risk limit exceeded: {risk_percent:.1f}%")
        elif risk_percent > self.config['max_portfolio_risk'] * 0.9:
            warnings.append(f"Approaching portfolio risk limit: {risk_percent:.1f}%")
        
        # Concentration checks
        setup_types = {}
        for pos in positions.values():
            setup_type = pos.get('setup_type', 'UNKNOWN')
            setup_types[setup_type] = setup_types.get(setup_type, 0) + pos.get('size_percent', 0)
        
        for setup_type, concentration in setup_types.items():
            if setup_type == 'SQUEEZE_PLAY' and concentration > self.config['max_negative_gex_exposure']:
                violations.append(f"Squeeze play concentration too high: {concentration:.1f}%")
            elif setup_type == 'PREMIUM_SELLING' and concentration > self.config['max_premium_selling']:
                violations.append(f"Premium selling concentration too high: {concentration:.1f}%")
        
        return {
            'violations': violations,
            'warnings': warnings,
            'risk_score': min(10, len(violations) * 3 + len(warnings)),
            'action_required': len(violations) > 0
        }
