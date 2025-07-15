# GCT Market Sentiment - Usage Guide

## What We've Built

1. **Coherence Pattern Detection System**
   - Analyzes price patterns to detect building/falling coherence
   - Uses your Tiingo API key: `ef1915ca2231c0c953e4fb7b72dec74bc767d9d1`
   - Generates alerts when significant patterns are detected

2. **Data Pipeline**
   - Fetches and saves market data locally (CSV files)
   - Calculates GCT coherence metrics offline
   - No dependency on unstable web servers

## Quick Start

### 1. Fetch Market Data
```bash
./venv/bin/python fetch_and_save_data.py
```
This downloads price data for major stocks and saves to `data/tickers/`

### 2. Calculate Coherence
```bash
./venv/bin/python calculate_coherence_offline.py
```
This analyzes the saved data and generates coherence metrics.

### 3. View Results
```bash
./venv/bin/python view_coherence_data.py
```
Shows the latest coherence scores and alerts.

### 4. Monitor Specific Tickers
```bash
./venv/bin/python simple_coherence_monitor.py
```
Monitors AAPL, MSFT, NVDA, SPY, QQQ for coherence patterns.

### 5. Dashboard Options (Choose One)

#### Option A: Static HTML Dashboard (Recommended)
```bash
./venv/bin/python local_dashboard.py
```
- Generates a self-contained HTML file with charts
- No server required - opens directly in browser
- Most stable option

#### Option B: Terminal Dashboard
```bash
./venv/bin/python terminal_dashboard.py
```
- Runs entirely in your terminal
- Auto-refreshes every 10 seconds
- No web dependencies

#### Option C: PDF Report
```bash
./venv/bin/python pdf_report.py
```
- Creates a comprehensive PDF report
- Opens automatically when generated
- Perfect for archiving or sharing

#### Option D: Plotly Dash (Web)
```bash
./venv/bin/python dash_dashboard.py
```
- More stable than Streamlit
- Runs on http://localhost:8050
- Interactive web interface

#### Option E: Streamlit (Original - May Crash)
```bash
./run_dashboard.sh
```
- Launches at http://localhost:8505
- Known to have connection errors after 3 seconds

## Data Files Created

- `data/tickers/[TICKER]/` - Raw price data for each ticker
- `data/raw/coherence_calculated_*.csv` - Coherence metrics
- `data/raw/coherence_alerts_*.csv` - Pattern alerts
- `data/summaries/market_summary_*.csv` - Daily market summaries

## Understanding Coherence

**High Coherence (>0.7)**: Strong unified market movement
- Clear trend direction (high ψ)
- Low volatility (high ρ)
- Consistent volume (moderate f)

**Low Coherence (<0.3)**: Chaotic/uncertain market
- No clear trend
- High volatility
- Erratic volume

**Building Coherence**: Potential bullish signal
- Increasing trend strength
- Decreasing volatility
- Rising volume

**Falling Coherence**: Potential bearish signal
- Trend breakdown
- Increasing volatility
- Volume spikes

## Troubleshooting

1. **Connection Errors**: The dashboard may disconnect. Use the offline scripts instead.
2. **API Rate Limits**: Tiingo limits requests. Wait a few minutes between runs.
3. **No Alerts**: Normal during stable markets. Alerts only trigger on significant patterns.

## Next Steps

1. Schedule `fetch_and_save_data.py` to run daily
2. Build historical coherence database
3. Backtest coherence signals against actual price movements
4. Add more sophisticated pattern detection