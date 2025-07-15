# GCT Market Sentiment Analysis System

## Overview
A comprehensive real-time market monitoring system based on Grounded Coherence Theory (GCT) that analyzes market sentiment through three lenses:

1. **Coherence Analysis** - Measures market clarity and alignment
2. **Truth Cost Calculation** - Identifies unsustainable market patterns
3. **Emotional Superposition** - Tracks quantum-like emotional states

## Components

### 1. Core Analysis Tools

#### Coherence Analysis (`analyze_any_stock.py`)
- Calculates GCT coherence score (0-1) for any stock
- Components: ψ (clarity), ρ (wisdom), q (emotion), f (social)
- Usage: `./venv/bin/python analyze_any_stock.py TSLA AAPL NVDA`

#### Truth Cost Calculator (`truth_cost_calculator.py`)
- Measures energy required to maintain market patterns
- Identifies unsustainable behaviors that will revert
- Generates detailed visualizations
- Usage: `./venv/bin/python truth_cost_calculator.py SPY QQQ`

#### Emotional Superposition (`emotional_superposition.py`)
- Analyzes quantum-like emotional states
- Tracks fear/greed, hope/despair, euphoria/panic
- Calculates emotional entropy and coherence
- Usage: `./venv/bin/python emotional_superposition.py NVDA TSLA`

### 2. Dashboard Options

#### Static HTML Dashboard (`local_dashboard.py`)
- Self-contained HTML with charts
- No server required
- Most stable option

#### Terminal Dashboard (`terminal_dashboard.py`)
- Runs in console with Rich UI
- Auto-refreshes every 10 seconds

#### PDF Reports (`pdf_report.py`)
- Comprehensive PDF generation
- Perfect for archiving

#### Real-time Monitor Dashboard (`realtime_dashboard.html`)
- Auto-updates every 2 minutes
- Shows all monitored stocks
- Displays alerts and market health

### 3. Real-time Monitoring System

#### Monitor Service (`realtime_monitor.py`)
- Runs 24/7 as background process
- Monitors:
  - Magnificent 7 (AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA)
  - Market indices (SPY, QQQ, IWM, DIA)
  - 30 most volatile stocks (auto-detected)
- Updates every 2-5 minutes
- Generates alerts for critical conditions

#### Alert System (`alert_notifier.py`)
- Sends push notifications via:
  - Email
  - Telegram
  - Pushover
  - Webhooks
- Configurable through `alert_config.json`

### 4. Integration Tools

#### Integrated Analysis (`integrated_analysis.py`)
- Runs all three analyses together
- Generates unified health scores
- Provides actionable insights
- Usage: `./venv/bin/python integrated_analysis.py SPY QQQ NVDA`

#### Stock Management (`add_stocks.py`)
- Add stocks to monitoring list
- Fetches historical data
- Usage: `./venv/bin/python add_stocks.py AMD NFLX DIS`

## Starting the System

### Quick Start
```bash
# Run one-time analysis
./venv/bin/python integrated_analysis.py SPY QQQ NVDA TSLA

# Generate dashboard
./venv/bin/python local_dashboard.py
```

### Start 24/7 Monitoring
```bash
# Start the monitor
./start_monitor.sh

# View real-time dashboard
open realtime_dashboard.html

# Stop the monitor
./stop_monitor.sh
```

### Configure Alerts
1. Edit `alert_config.json` with your credentials
2. Test alerts: `./venv/bin/python alert_notifier.py`
3. Monitor will automatically send alerts when running

## Key Metrics Explained

### Coherence (0-1)
- **>0.7**: Strong unified movement
- **0.3-0.7**: Moderate coherence
- **<0.3**: Chaotic/uncertain

### Truth Cost (0-1)
- **<0.2**: Natural, sustainable
- **0.2-0.4**: Moderate strain
- **>0.4**: Unsustainable pattern

### Fear/Greed Index (0-1)
- **>0.8**: Extreme greed
- **0.6-0.8**: Greed
- **0.4-0.6**: Neutral
- **0.2-0.4**: Fear
- **<0.2**: Extreme fear

## Alert Conditions

1. **Extreme Greed** (Fear/Greed > 0.9)
2. **Panic Selling** (Panic level > 0.7)
3. **High Truth Cost** (>0.5)
4. **Coherence Breakdown** (<0.2)
5. **Building Coherence** (>0.7 with positive trend)

## Data Storage

- **Real-time data**: `realtime_data.json`
- **Historical data**: `data/tickers/[TICKER]/`
- **Coherence calculations**: `data/raw/coherence_*.csv`
- **Logs**: `logs/monitor.log`, `realtime_monitor.log`
- **Critical alerts**: `critical_alerts.log`

## API Keys Required

- **Tiingo**: Set in environment or use default in code
  - Current: `ef1915ca2231c0c953e4fb7b72dec74bc767d9d1`
- **yfinance**: No API key needed (free)

## Troubleshooting

1. **Dashboard crashes**: Use local_dashboard.py instead of Streamlit
2. **No data**: Check internet connection and API limits
3. **Missing dependencies**: Run `pip install -r requirements.txt`
4. **Monitor won't start**: Check `logs/monitor.log` for errors

## Theory Background

Based on Grounded Coherence Theory (GCT):
- Markets exhibit quantum-like properties
- Coherence emerges from aligned variables
- Truth costs reveal unsustainable patterns
- Emotional superposition drives volatility

## Next Steps

1. Backtest coherence signals
2. Add machine learning predictions
3. Integrate with trading APIs
4. Build mobile app dashboard
5. Add voice alerts for critical events