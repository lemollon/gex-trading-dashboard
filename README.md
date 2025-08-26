# 🚀 GEX Trading System

A comprehensive Gamma Exposure (GEX) analysis platform for identifying high-probability options trading setups based on dealer hedging flows and market microstructure.

## 🎯 Overview

This system analyzes options gamma exposure to identify:
- **Gamma Flip Points**: Where market volatility regime changes
- **Call Walls**: Resistance levels where dealers must sell
- **Put Support**: Support levels where dealers must buy  
- **Squeeze Setups**: High-probability directional moves
- **Premium Selling Opportunities**: Mean reversion trades

## 📊 Key Features

### Core Analytics
- **Real-time GEX Calculations**: Spot × Gamma × OI × 100
- **Market Regime Detection**: Positive/Negative gamma environments
- **Wall Identification**: Key support/resistance from dealer flows
- **Expected Move Analysis**: Range-bound vs breakout setups

### Trading Strategies
1. **Negative GEX Squeezes**: Long calls/puts in volatility amplification zones
2. **Positive GEX Mean Reversion**: Premium selling at walls
3. **Iron Condors**: Range-bound trades between strong walls
4. **Gamma Flip Plays**: High volatility around transition points

### Technical Stack
- **Databricks**: Distributed options data processing
- **Streamlit**: Interactive trading dashboard
- **TradingVolatility API**: Real-time options chain data
- **GitHub**: Version control and CI/CD

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- TradingVolatility.net API access
- (Optional) Databricks workspace for production scaling

### Quick Start
```bash
# Clone repository
git clone https://github.com/yourusername/gex-trading-system.git
cd gex-trading-system

# Install dependencies
pip install -r requirements.txt

# Run Streamlit dashboard
streamlit run gex_dashboard.py
```

### Configuration
1. **API Setup**: Add your TradingVolatility username to config
2. **Discord Alerts** (Optional): Add webhook URL for notifications
3. **Databricks** (Optional): Configure for production data processing

## 📁 File Structure

```
gex-trading-system/
│
├── gex_calculator.py      # Core GEX calculation engine
├── gex_dashboard.py       # Streamlit trading dashboard  
├── requirements.txt       # Python dependencies
├── .gitignore            # Git ignore patterns
├── README.md             # This file
│
└── docs/                 # Documentation (optional)
    ├── trading_guide.md  # Trading strategy documentation
    └── api_reference.md  # Technical API reference
```

## 🔧 Usage

### Basic GEX Analysis
```python
from gex_calculator import GEXCalculator, TradingVolatilityAPI

# Initialize
calculator = GEXCalculator()
api = TradingVolatilityAPI(username="your-api-username")

# Analyze a symbol
options_data = api.fetch_options_data("SPY")
result = calculator.calculate_gamma_exposure(options_data, spot_price=450, symbol="SPY")

print(f"Net GEX: {result['net_gex']/1e9:.2f}B")
print(f"Flip Point: {result['gamma_flip_point']:.2f}")
print(f"Regime: {result['regime']}")
```

### Streamlit Dashboard
```bash
streamlit run gex_dashboard.py
```

Features:
- Real-time GEX profiles for 70+ symbols
- Interactive setup detection
- Risk management tools
- Position tracking
- Discord alert integration

## 📈 Trading Strategies

### 1. Negative GEX Squeeze (Long Calls)
**Setup Criteria:**
- Net GEX < -1B (SPY) or < -500M (QQQ)
- Price below gamma flip point by 0.5-1.5%
- Strong put wall support within 1% below

**Execution:**
- Buy ATM or 1st OTM calls above flip point
- Use 2-5 DTE for maximum gamma sensitivity
- Target: 100% profit | Stop: 50% loss

### 2. Premium Selling at Walls
**Setup Criteria:**
- Net GEX > 3B (high positive gamma)
- Strong call wall with >500M gamma concentration
- Price between flip and call wall

**Execution:**
- Sell calls at or above wall strikes
- Use 0-2 DTE for rapid theta decay
- Close at 50% profit or if wall breached

### 3. Iron Condors
**Setup Criteria:**
- Net GEX > 1B (positive gamma environment)
- Call and put walls >3% apart
- Low volatility environment (IV rank <50%)

**Execution:**
- Short strikes at walls, long strikes beyond
- Use 5-10 DTE for optimal theta/gamma ratio
- Manage at 25% profit or threatened strike

## ⚠️ Risk Management

### Position Sizing
- **Maximum 3% of capital** per directional trade
- **Maximum 5% of capital** in sold options
- **Maximum 2% portfolio loss** on iron condors

### Stop Losses
- **Directional plays**: 50% loss or flip point breach
- **Premium selling**: 100% loss or wall breach
- **Iron condors**: Threatened strike or 25% profit

### Time Management
- **Close positions** with <1 DTE remaining
- **No new positions** in last 30 minutes
- **Roll threatened strikes** if >3 DTE remains

## 🔔 Alerts & Monitoring

### High Priority Alerts
- Net GEX crosses -1B threshold
- Price within 0.25% of gamma flip
- Major wall breach
- 80% of gamma expiring within 1 day

### Discord Integration
```python
# Configure webhook in dashboard
DISCORD_WEBHOOK = "https://discord.com/api/webhooks/your-webhook"
```

## 📊 Performance Tracking

Monitor these key metrics:
- **Win rate by setup type**
- **Average return per trade** 
- **Maximum drawdown per strategy**
- **Sharpe ratio by strategy**
- **Success rate at different GEX levels**

## 🔄 API Rate Limits

**TradingVolatility.net Limits:**
- **Weekdays**: 20 calls/minute (non-realtime), 2 calls/minute (realtime)
- **Weekends**: 2 calls/minute (all endpoints)

The system automatically handles rate limiting with proper delays between calls.

## 🚀 Production Deployment

### Databricks Setup
1. Create Databricks workspace
2. Upload notebook pipeline
3. Schedule morning scans (7 AM Central)
4. Configure Delta Lake storage

### Streamlit Cloud
1. Connect GitHub repository
2. Add secrets for API keys
3. Deploy dashboard
4. Configure custom domain (optional)

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This software is for educational and research purposes only. Options trading involves substantial risk and is not suitable for all investors. Past performance is not indicative of future results. Always consult with a qualified financial advisor before making investment decisions.

## 🆘 Support

- **Issues**: Open a GitHub issue for bugs or feature requests
- **Discussions**: Use GitHub Discussions for questions
- **Documentation**: Check the `docs/` folder for detailed guides

## 🎯 Roadmap

### Version 2.0
- [ ] Machine learning setup detection
- [ ] Automated paper trading
- [ ] Advanced backtesting engine
- [ ] Multi-timeframe analysis

### Version 3.0
- [ ] Real-time order execution
- [ ] Portfolio optimization
- [ ] Risk-adjusted position sizing
- [ ] Advanced Greeks analysis

---

**Built with ❤️ for options traders who understand market microstructure**
