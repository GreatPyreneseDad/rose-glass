#!/bin/bash

# Start the GCT Real-time Market Monitor

echo "Starting GCT Real-time Market Monitor..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install required packages if needed
echo "Checking dependencies..."
pip install -q yfinance pandas numpy scipy matplotlib schedule requests

# Create log directory
mkdir -p logs

# Kill any existing monitor process
pkill -f "realtime_monitor.py" 2>/dev/null

# Start monitor in background
echo "Starting monitor process..."
nohup python realtime_monitor.py > logs/monitor.log 2>&1 &
MONITOR_PID=$!

echo "Monitor started with PID: $MONITOR_PID"
echo "PID saved to monitor.pid"
echo $MONITOR_PID > monitor.pid

# Give it a moment to start
sleep 5

# Check if it's running
if ps -p $MONITOR_PID > /dev/null; then
    echo "✓ Monitor is running successfully"
    echo "Dashboard will be available at: file://$(pwd)/realtime_dashboard.html"
    echo "Logs available at: logs/monitor.log"
    echo ""
    echo "To stop the monitor, run: ./stop_monitor.sh"
else
    echo "✗ Monitor failed to start. Check logs/monitor.log for details"
    exit 1
fi