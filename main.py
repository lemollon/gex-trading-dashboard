"""
Utilities - Helper functions and common utilities
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta
import pytz
import logging
import yaml
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class ConfigManager:
    """Manage configuration loading and validation"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self._config = None
        self.load_config()
    
    def load_config(self) -> Dict:
        """Load configuration from YAML file"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    self._config = yaml.safe_load(f)
                logger.info(f"Configuration loaded from {self.config_path}")
            else:
                self._config = self._default_config()
                logger.warning(f"Config file not found, using defaults")
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            self._config = self._default_config()
        
        return self._config
    
    def get(self, key_path: str, default=None):
        """Get nested configuration value using dot notation"""
        keys = key_path.split('.')
        value = self._config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def _default_config(self) -> Dict:
        """Default configuration if file not found"""
        return {
            'app': {'name': 'GEX Trading Dashboard', 'debug': False},
            'data_sources': {'primary': 'tradingvolatility', 'cache_duration_hours': 2},
            'risk_management': {'max_portfolio_risk': 10.0, 'max_single_position': 3.0}
        }

class MarketTimeManager:
    """Handle market hours and timezone operations"""
    
    def __init__(self, timezone: str = 'US/Eastern'):
        self.timezone = pytz.timezone(timezone)
        self.market_open = datetime.strptime('09:30', '%H:%M').time()
        self.market_close = datetime.strptime('16:00', '%H:%M').time()
    
    def is_market_open(self, dt: datetime = None) -> bool:
        """Check if market is currently open"""
        if dt is None:
            dt = datetime.now(self.timezone)
        elif dt.tzinfo is None:
            dt = self.timezone.localize(dt)
        
        # Check if weekday
        if dt.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
        
        # Check if within market hours
        current_time = dt.time()
        return self.market_open <= current_time <= self.market_close
    
    def get_market_status(self) -> Dict:
        """Get detailed market status"""
        now = datetime.now(self.timezone)
        is_open = self.is_market_open(now)
        
        if is_open:
            status = "OPEN"
            next_close = now.replace(
                hour=self.market_close.hour,
                minute=self.market_close.minute,
                second=0,
                microsecond=0
            )
            time_until_event = next_close - now
        else:
            status = "CLOSED"
            # Calculate next open
            next_open = now.replace(
                hour=self.market_open.hour,
                minute=self.market_open.minute,
                second=0,
                microsecond=0
            )
            
            # If after market close, move to next day
            if now.time() >= self.market_close:
                next_open += timedelta(days=1)
            
            # Skip weekends
            while next_open.weekday() >= 5:
                next_open += timedelta(days=1)
            
            time_until_event = next_open - now
        
        return {
            'status': status,
            'is_open': is_open,
            'current_time': now,
            'time_until_next_event': time_until_event,
            'next_event': 'close' if is_open else 'open'
        }

class DataValidator:
    """Validate and clean data"""
    
    @staticmethod
    def validate_options_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate options data structure and content"""
        errors = []
        
        if df.empty:
            errors.append("DataFrame is empty")
            return False, errors
        
        # Required columns
        required_cols = ['symbol', 'strike', 'option_type', 'open_interest']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
        
        # Data type validation
        if 'strike' in df.columns:
            non_numeric_strikes = df[pd.to_numeric(df['strike'], errors='coerce').isna()]
            if not non_numeric_strikes.empty:
                errors.append(f"Non-numeric strikes found: {len(non_numeric_strikes)} rows")
        
        if 'open_interest' in df.columns:
            non_numeric_oi = df[pd.to_numeric(df['open_interest'], errors='coerce').isna()]
            if not non_numeric_oi.empty:
                errors.append(f"Non-numeric open interest found: {len(non_numeric_oi)} rows")
        
        # Option type validation
        if 'option_type' in df.columns:
            valid_types = df['option_type'].str.upper().str[0].isin(['C', 'P'])
            if not valid_types.all():
                invalid_count = (~valid_types).sum()
                errors.append(f"Invalid option types found: {invalid_count} rows")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def clean_options_data(df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize options data"""
        if df.empty:
            return df
        
        df_clean = df.copy()
        
        # Convert numeric columns
        numeric_cols = ['strike', 'open_interest', 'volume', 'bid', 'ask', 'last',
                       'implied_volatility', 'delta', 'gamma', 'theta', 'vega']
        
        for col in numeric_cols:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        # Standardize option types
        if 'option_type' in df_clean.columns:
            df_clean['option_type'] = df_clean['option_type'].str.upper().str[0]
            df_clean = df_clean[df_clean['option_type'].isin(['C', 'P'])]
        
        # Remove rows with invalid data
        df_clean = df_clean.dropna(subset=['strike', 'option_type'])
        
        # Remove zero/negative strikes
        if 'strike' in df_clean.columns:
            df_clean = df_clean[df_clean['strike'] > 0]
        
        # Remove negative open interest
        if 'open_interest' in df_clean.columns:
            df_clean = df_clean[df_clean['open_interest'] >= 0]
        
        return df_clean

class PerformanceTracker:
    """Track and analyze performance metrics"""
    
    def __init__(self):
        self.trades = []
        self.daily_pnl = []
    
    def add_trade(self, trade_data: Dict):
        """Add a completed trade"""
        required_fields = ['symbol', 'entry_date', 'exit_date', 'pnl', 'setup_type']
        
        if all(field in trade_data for field in required_fields):
            trade_data['trade_id'] = len(self.trades) + 1
            trade_data['add_time'] = datetime.now()
            self.trades.append(trade_data)
        else:
            logger.warning(f"Incomplete trade data: missing {set(required_fields) - set(trade_data.keys())}")
    
    def calculate_metrics(self, period_days: int = 30) -> Dict:
        """Calculate performance metrics"""
        
        if not self.trades:
            return {'error': 'No trades available'}
        
        # Filter to period
        cutoff_date = datetime.now() - timedelta(days=period_days)
        period_trades = [t for t in self.trades if t['exit_date'] >= cutoff_date]
        
        if not period_trades:
            return {'error': f'No trades in last {period_days} days'}
        
        # Basic metrics
        total_trades = len(period_trades)
        winning_trades = [t for t in period_trades if t['pnl'] > 0]
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        total_pnl = sum(t['pnl'] for t in period_trades)
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        losing_trades = [t for t in period_trades if t['pnl'] < 0]
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
        
        # Risk metrics
        pnls = [t['pnl'] for t in period_trades]
        max_win = max(pnls) if pnls else 0
        max_loss = min(pnls) if pnls else 0
        
        # Calculate max drawdown
        cumulative_pnl = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdown = running_max - cumulative_pnl
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
        
        # Sharpe ratio (simplified)
        if len(pnls) > 1:
            returns_std = np.std(pnls)
            avg_return = np.mean(pnls)
            sharpe_ratio = avg_return / returns_std if returns_std > 0 else 0
        else:
            sharpe_ratio = 0
        
        return {
            'period_days': period_days,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'max_win': max_win,
            'max_loss': max_loss,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else 0
        }
    
    def get_setup_performance(self) -> Dict:
        """Analyze performance by setup type"""
        
        if not self.trades:
            return {}
        
        setup_stats = {}
        
        for trade in self.trades:
            setup_type = trade.get('setup_type', 'UNKNOWN')
            
            if setup_type not in setup_stats:
                setup_stats[setup_type] = {
                    'trades': [],
                    'total_pnl': 0,
                    'win_count': 0,
                    'loss_count': 0
                }
            
            setup_stats[setup_type]['trades'].append(trade)
            setup_stats[setup_type]['total_pnl'] += trade['pnl']
            
            if trade['pnl'] > 0:
                setup_stats[setup_type]['win_count'] += 1
            else:
                setup_stats[setup_type]['loss_count'] += 1
        
        # Calculate metrics for each setup type
        for setup_type, stats in setup_stats.items():
            total_trades = len(stats['trades'])
            stats['win_rate'] = stats['win_count'] / total_trades if total_trades > 0 else 0
            stats['avg_pnl'] = stats['total_pnl'] / total_trades if total_trades > 0 else 0
            stats['total_trades'] = total_trades
        
        return setup_stats

class AlertManager:
    """Manage alerts and notifications"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.discord_enabled = self.config.get('discord', {}).get('enabled', False)
        self.webhook_url = os.getenv('DISCORD_WEBHOOK_URL')
    
    def send_alert(self, message: str, alert_type: str = 'INFO', include_timestamp: bool = True):
        """Send alert through configured channels"""
        
        if include_timestamp:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            formatted_message = f"[{timestamp}] {message}"
        else:
            formatted_message = message
        
        # Console logging
        if alert_type == 'ERROR':
            logger.error(formatted_message)
        elif alert_type == 'WARNING':
            logger.warning(formatted_message)
        else:
            logger.info(formatted_message)
        
        # Discord webhook
        if self.discord_enabled and self.webhook_url:
            try:
                self._send_discord_message(formatted_message, alert_type)
            except Exception as e:
                logger.error(f"Failed to send Discord alert: {e}")
    
    def _send_discord_message(self, message: str, alert_type: str):
        """Send message to Discord webhook"""
        import requests
        
        # Color coding by alert type
        colors = {
            'INFO': 3447003,      # Blue
            'WARNING': 16776960,  # Yellow  
            'ERROR': 15158332,    # Red
            'SUCCESS': 3066993    # Green
        }
        
        embed = {
            "title": f"GEX Trading Alert - {alert_type}",
            "description": message,
            "color": colors.get(alert_type, 3447003),
            "timestamp": datetime.now().isoformat()
        }
        
        payload = {"embeds": [embed]}
        
        response = requests.post(self.webhook_url, json=payload, timeout=10)
        response.raise_for_status()

def format_number(value: Union[int, float], format_type: str = 'auto') -> str:
    """Format numbers for display"""
    
    if pd.isna(value) or value is None:
        return "N/A"
    
    if format_type == 'currency':
        return f"${value:,.2f}"
    elif format_type == 'percentage':
        return f"{value:.2f}%"
    elif format_type == 'billions':
        return f"{value/1e9:.2f}B"
    elif format_type == 'millions':
        return f"{value/1e6:.1f}M"
    elif format_type == 'auto':
        if abs(value) >= 1e9:
            return f"{value/1e9:.2f}B"
        elif abs(value) >= 1e6:
            return f"{value/1e6:.1f}M"
        elif abs(value) >= 1e3:
            return f"{value/1e3:.1f}K"
        else:
            return f"{value:.2f}"
    else:
        return str(value)

def calculate_business_days(start_date: datetime, end_date: datetime) -> int:
    """Calculate business days between two dates"""
    
    business_days = 0
    current_date = start_date
    
    while current_date <= end_date:
        if current_date.weekday() < 5:  # Monday = 0, Friday = 4
            business_days += 1
        current_date += timedelta(days=1)
    
    return business_days

def get_project_root() -> Path:
    """Get the project root directory"""
    return Path(__file__).parent.parent

def setup_logging(config: Dict = None) -> None:
    """Setup logging configuration"""
    
    config = config or {}
    log_level = config.get('level', 'INFO')
    log_format = config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File logging if enabled
    if config.get('file_logging', {}).get('enabled', False):
        filename = config['file_logging'].get('filename', 'gex_trading.log')
        
        file_handler = logging.FileHandler(filename)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(logging.Formatter(log_format))
        
        logging.getLogger().addHandler(file_handler)

# Global instances
config_manager = ConfigManager()
market_time_manager = MarketTimeManager()
performance_tracker = PerformanceTracker()
data_validator = DataValidator()
