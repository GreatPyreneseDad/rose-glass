#!/bin/bash

# Stop the GCT Real-time Market Monitor

echo "Stopping GCT Real-time Market Monitor..."

if [ -f monitor.pid ]; then
    PID=$(cat monitor.pid)
    if ps -p $PID > /dev/null; then
        kill $PID
        echo "✓ Monitor stopped (PID: $PID)"
        rm monitor.pid
    else
        echo "Monitor not running (PID: $PID not found)"
        rm monitor.pid
    fi
else
    # Try to find and kill by process name
    pkill -f "realtime_monitor.py"
    echo "✓ Stopped any running monitor processes"
fi