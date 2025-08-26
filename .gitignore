# 🚀 GEX Trading System

A comprehensive **Gamma Exposure (GEX) trading system** that identifies high-probability options trading setups based on market microstructure analysis. The system analyzes dealer gamma positioning to predict market volatility regimes and identify optimal entry points.

## 🎯 What This System Does

**Gamma Exposure Analysis**: Calculates how much gamma dealers must hedge across all strikes, identifying:
- **Negative GEX Zones**: Where dealers are short gamma → volatility expansion expected
- **Positive GEX Zones**: Where dealers are long gamma → volatility suppression expected  
- **Gamma Flip Points**: Critical levels where market regime changes
- **Call/Put Walls**: Major support/resistance levels from option positioning

**Trading Setup Identification**: Automatically identifies high-confidence setups:
- 🎯 **Squeeze Plays**: Long calls/puts in negative GEX environments
- 💰 **Premium Selling**: Short options at major resistance/support walls
- ⚖️ **Iron Condors**: Range-bound plays between strong walls
- ⚡ **Gamma Flip Trades**: Volatility plays near regime transition points

## 📊 System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Sources  │    │  GEX Calculator  │    │ Setup Identifier│
│                 │    │                  │    │                 │
│ • TradingVol    │───▶│ • Gamma Exposure │───▶│ • Squeeze Plays │
│ • Yahoo Finance │    │ • Flip Points    │    │ • Premium Sell  │
│ • Polygon.io    │    │ • Call/Put Walls │    │ • Iron Condors  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Risk Management │    │   Orchestrator   │    │   Streamlit     │
│                 │    │                  │    │   Dashboard     │
│ • Position Size │◀───│ • Coordinates    │───▶│                 │
│ • Portfolio Risk│    │ • Caches Results │    │ • Live Analysis │
│ • Kelly Sizing  │    │ • Sends Alerts   │    │ • Setup Display │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### 1. Clone and Setup
```bash
git clone <your-repo-url>
cd gex-trading-system
pip install -r requirements.txt
```

### 2. Configuration
Copy and customize the configuration:
```bash
cp config.yaml config.local.yaml
# Edit config.local.yaml with your settings
```

### 3. Run Analysis
```python
from main import GEXTradingOrchestrator

# Initialize system
orchestrator = GEXTradingOrchestrator()

# Run full analysis
results = orchestrator.run_full_analysis()

# Or quick scan of priority symbols
results = orchestrator.get_quick_scan()
```

### 4. Launch Dashboard
```bash
streamlit run gex_dashboard.py
```

## 📁 File Structure

```
gex-trading-system/
├── main.py                 # Main orchestrator
├── gex_calculator.py       # Core GEX calculations  
├── data_sources.py         # API integrations
├── setup_identification.py # Trading setup detection
├── risk_management.py      # Position sizing & risk
├── utils.py               # Helper functions
├── config.yaml            # Configuration
├── requirements.txt       # Dependencies
├── gex_dashboard.py       # Streamlit dashboard (you build this)
└── README.md             # This file
```

## 🧮 Core GEX Calculations

### Gamma Exposure Formula
```
GEX = Spot Price × Gamma × Open Interest × 100
```

**For each strike:**
- **Calls**: Positive GEX (dealers must sell on rallies)
- **Puts**: Negative GEX (dealers must buy on dips)

### Key Metrics
- **Net GEX**: Sum of all strikes (market regime indicator)
- **Gamma Flip Point**: Where cumulative GEX crosses zero
- **Call Walls**: Highest positive GEX strikes (resistance)
- **Put Walls**: Highest negative GEX strikes (support)

## 🎯 Trading Strategies

### 1. Squeeze Plays (Negative GEX)
**When**: Net GEX < -500M, price below flip point
```python
# Example: SPY negative GEX setup
{
    'type': 'SQUEEZE_PLAY',
    'direction': 'LONG_CALLS', 
    'confidence': 85,
    'reason': 'Strong negative GEX (-1.2B) suggests dealer short gamma',
    'target': 'ATM calls 2-5 DTE'
}
```

### 2. Premium Selling (High Positive GEX) 
**When**: Net GEX > 2B, strong walls present
```python
# Example: Call selling at resistance
{
    'type': 'CALL_SELLING',
    'direction': 'SHORT_CALLS',
    'confidence': 78,
    'reason': 'Strong call wall at 450 with 800M GEX',
    'target': 'Sell calls at/above wall strikes'
}
```

### 3. Iron Condors (Range-Bound)
**When**: Strong walls 3-15% apart, positive GEX environment
```python
# Example: Range trade
{
    'type': 'IRON_CONDOR', 
    'direction': 'NEUTRAL',
    'confidence': 72,
    'reason': 'Strong walls at 440 and 460, 4.5% range'
}
```

## ⚙️ Configuration

Key configuration sections in `config.yaml`:

### Data Sources
```yaml
data_sources:
  primary: "tradingvolatility"
  fallback: "yahoo" 
  cache_duration_hours: 2
  max_api_calls_per_hour: 100
```

### Risk Management
```yaml
risk_management:
  max_portfolio_risk: 10.0      # % of portfolio at risk
  max_single_position: 3.0      # % per position
  kelly_multiplier: 0.25        # Conservative Kelly sizing
  max_daily_loss: 2.0          # Daily loss limit %
```

### Universe
```yaml
universe:
  priority: [SPY, QQQ, IWM, AAPL, MSFT, NVDA]
  etfs: [SPY, QQQ, IWM, DIA, GLD, SLV]
  sectors: [XLF, XLE, XLK, XLV, XLI, XLP]
```

## 🔧 Integration Options

### Databricks Integration
The system is designed to work with your existing Databricks pipeline:
```python
# In Databricks notebook
from main import GEXTradingOrchestrator

orchestrator = GEXTradingOrchestrator()
results = orchestrator.run_full_analysis()

# Use results with your existing cells
interesting_conditions = results['trading_setups']
```

### Discord Alerts
Set environment variable and enable in config:
```bash
export DISCORD_WEBHOOK_URL="your_webhook_url"
```

```yaml
alerts:
  discord:
    enabled: true
  triggers:
    high_confidence_setup: 85
```

## 📈 Performance Tracking

Built-in performance tracking with metrics:
- **Win Rate**: Percentage of profitable trades
- **Sharpe Ratio**: Risk-adjusted returns  
- **Max Drawdown**: Worst losing streak
- **Setup Performance**: Win rates by setup type

```python
# Access performance metrics
metrics = orchestrator.performance_tracker.calculate_metrics(period_days=30)
print(f"Win Rate: {metrics['win_rate']:.1%}")
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
```

## 🛡️ Risk Management

### Position Sizing
- **Kelly Criterion**: Optimal position sizes based on win probability
- **Risk-Based Sizing**: Adjusted for setup risk level
- **Portfolio Constraints**: Maximum allocations by strategy type

### Risk Limits
- **Daily Loss Limits**: Automatic position reduction
- **Concentration Limits**: Max exposure to single setup type  
- **Portfolio Risk**: Total capital at risk monitoring

## 🐛 Debugging & Monitoring

### Health Checks
```python
health = orchestrator.health_check()
print(f"System Status: {health['overall_status']}")
print(f"Market Open: {health['market_open']}")
```

### Logging
Comprehensive logging with configurable levels:
```yaml
logging:
  level: "INFO"
  file_logging:
    enabled: true
    filename: "gex_trading.log"
```

## 🔮 Usage Examples

### Morning Scan
```python
# Quick morning analysis
results = orchestrator.get_quick_scan(max_symbols=10)

for setup_info in results['trading_setups']:
    setup = setup_info['setup']
    if setup.confidence > 80:
        print(f"{setup.symbol}: {setup.setup_type} - {setup.confidence}%")
```

### Risk Assessment
```python
# Check portfolio risk
risk_check = orchestrator.risk_manager.check_risk_limits(
    current_positions, 
    portfolio_value=100000
)

if risk_check['violations']:
    print(f"Risk violations: {risk_check['violations']}")
```

### Alert Integration
```python
# Send custom alert
orchestrator.alert_manager.send_alert(
    "🔥 SPY showing strong negative GEX - squeeze setup active",
    alert_type='WARNING'
)
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open Pull Request

## ⚡ Next Steps

Now you're ready to:
1. **Build your Streamlit dashboard** (`gex_dashboard.py`) using these components
2. **Integrate with your Databricks pipeline** using the orchestrator
3. **Customize the configuration** for your specific needs
4. **Add additional data sources** or trading strategies

The system is designed to be modular - you can use individual components or the full orchestrator based on your needs.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This software is for educational and research purposes only. Options trading involves substantial risk and is not suitable for all investors. Past performance is not indicative of future results. Always consult with a qualified financial advisor before making trading decisions.
