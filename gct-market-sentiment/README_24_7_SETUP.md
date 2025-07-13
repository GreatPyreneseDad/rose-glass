# GCT Market Sentiment 24/7 Setup Guide

## Overview

This system provides continuous market sentiment analysis with:
- 24/7 automated data collection
- SQLite database for historical storage
- Real-time HTML dashboard updates
- Historical analysis viewer
- Options trading signals
- Alert monitoring

## Components

1. **Database Manager** (`database_manager.py`)
   - SQLite database with tables for market data, alerts, and signals
   - Stores daily snapshots and historical trends

2. **Continuous Collector** (`continuous_collector.py`)
   - Runs 24/7 collecting market data
   - Updates database every 30 minutes during market hours
   - Generates real-time dashboard

3. **Database Dashboard** (`database_dashboard.py`)
   - Streamlit app for viewing historical data
   - Interactive charts and analysis

4. **Service Setup** (`setup_service.sh`)
   - Automated setup for macOS (launchd) or Linux (systemd)

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up the Service

```bash
# Make setup script executable
chmod +x setup_service.sh

# Run setup (works on both macOS and Linux)
./setup_service.sh
```

### 3. Configure Watchlist (Optional)

Create a `watchlist.json` file to add custom stocks:

```json
[
    "AAPL", "MSFT", "GOOGL", "TSLA", "NVDA",
    "SPY", "QQQ", "DIA", "IWM"
]
```

## Usage

### Starting the Service

```bash
# Using the management script
./manage_collector.sh start

# Or directly:
# macOS
launchctl start com.gct.market-collector

# Linux
sudo systemctl start gct-collector
```

### Viewing Status

```bash
./manage_collector.sh status
```

### Viewing Logs

```bash
./manage_collector.sh logs

# Or directly
tail -f logs/collector.log
```

### Stopping the Service

```bash
./manage_collector.sh stop
```

## Accessing Dashboards

### 1. Real-time HTML Dashboard
- Location: `realtime_dashboard.html`
- Auto-updates every 2 minutes
- Open in browser: `open realtime_dashboard.html`

### 2. Historical Analysis Dashboard

```bash
# Run the Streamlit dashboard
streamlit run database_dashboard.py
```

Features:
- Market trends over time
- Individual stock analysis
- Top movers
- Options signals
- Recent alerts

## Database Schema

### Tables:
1. **market_summary** - Daily market metrics
2. **stock_data** - Individual stock records
3. **alerts** - Trading alerts and warnings
4. **options_signals** - Call/Put signals
5. **sector_performance** - Sector analysis

### Accessing Database Directly

```bash
# Open SQLite database
sqlite3 market_analysis.db

# Example queries
.tables
SELECT * FROM market_summary ORDER BY date DESC LIMIT 10;
SELECT * FROM stock_data WHERE symbol='AAPL' ORDER BY date DESC;
```

## Scheduling

The collector runs:
- **Market Hours**: Every 30 minutes (9:30 AM - 4:00 PM EST)
- **Specific Times**: 9:30 AM, 12:00 PM, 3:30 PM, 4:00 PM
- **Dashboard Updates**: Every 2 minutes

## Maintenance

### Database Cleanup

The system automatically removes data older than 365 days. To manually clean:

```python
from database_manager import MarketDatabase
db = MarketDatabase()
db.cleanup_old_data(days_to_keep=180)  # Keep 6 months
```

### Backup Database

```bash
cp market_analysis.db "backups/market_analysis_$(date +%Y%m%d).db"
```

## Troubleshooting

### Service Won't Start
- Check logs: `cat logs/collector_error.log`
- Verify Python path: `which python3`
- Check permissions: `ls -la continuous_collector.py`

### Database Locked
- Stop the service and restart
- Check for zombie processes: `ps aux | grep continuous_collector`

### Missing Data
- Check if service is running: `./manage_collector.sh status`
- Verify market hours (weekdays 9:30 AM - 4:00 PM EST)
- Check API credentials if using real data sources

## Integration with Existing Scripts

To integrate with your existing analysis scripts:

```python
from database_manager import MarketDatabase

# Get latest data
db = MarketDatabase()
latest_stocks = db.get_top_movers(limit=50)
historical = db.get_historical_data('AAPL', days=30)
```

## Future Enhancements

1. Add real-time data feeds (yfinance, Alpha Vantage)
2. Implement backtesting framework
3. Add email/SMS alerts
4. Create mobile-friendly dashboard
5. Add machine learning predictions