# GCT (Grounded Coherence Theory) Project Overview

## Project Structure

```
GCT/
├── LICENSE
├── README.md
├── docs/
│   ├── SPEC-1-Market-Sentiment-Engine.md
│   └── WHITEPAPER.md
├── gct-market-sentiment/
│   ├── Market analysis and sentiment tools
│   ├── Real-time monitoring scripts
│   ├── Dashboard applications
│   └── Data visualization tools
└── soulmath-moderation-system/
    └── Node.js moderation system
```

## Key Components

### 1. Market Sentiment Analysis (`gct-market-sentiment/`)
- **Purpose**: Analyze market sentiment using Grounded Coherence Theory principles
- **Main Scripts**:
  - `app.py` - Main application
  - `realtime_monitor.py` - Real-time market monitoring
  - `coherence_dashboard.py` - Coherence metrics visualization
  - `truth_cost_calculator.py` - Calculate truth cost metrics
  - `analyze_any_stock.py` - Analyze individual stocks

### 2. Dashboard Tools
- `dash_dashboard.py` - Plotly Dash-based dashboard
- `streamlit.log` - Streamlit dashboard logs
- `local_dashboard.py` - Local dashboard interface
- `terminal_dashboard.py` - Terminal-based dashboard

### 3. Data Collection & Processing
- `fetch_news.py` - News data collection
- `fetch_prices_generate_news.py` - Price data and news generation
- `calculate_coherence_offline.py` - Offline coherence calculations

### 4. Monitoring & Alerts
- `coherence_alerts.py` - Alert system for coherence thresholds
- `alert_notifier.py` - Notification system
- `simple_coherence_monitor.py` - Basic monitoring tool

## Quick Start

```bash
# Navigate to the market sentiment directory
cd gct-market-sentiment/

# Install dependencies
pip install -r requirements.txt

# Run the main application
python app.py

# Or start the monitoring system
./start_monitor.sh
```

## Documentation

- **Whitepaper**: See `docs/WHITEPAPER.md` for theoretical foundation
- **Market Sentiment Engine Spec**: See `docs/SPEC-1-Market-Sentiment-Engine.md`
- **Usage Guide**: See `gct-market-sentiment/README_USAGE.md`
- **System Overview**: See `gct-market-sentiment/SYSTEM_OVERVIEW.md`

## Key Features

1. **Real-time Market Analysis**: Monitor market sentiment in real-time
2. **Coherence Metrics**: Calculate and visualize coherence scores
3. **Truth Cost Analysis**: Analyze the "cost" of market truths
4. **Multi-format Dashboards**: Terminal, web, and GUI dashboards
5. **Alert System**: Automated alerts for significant coherence changes
6. **Stock Analysis**: Analyze individual stocks or custom portfolios

## Recent Activity

Based on file timestamps, recent work includes:
- Custom analysis reports (CSV format)
- Dashboard screenshots and HTML exports
- Truth cost calculations for NVDA, SPY, and TSLA
- Real-time monitoring logs