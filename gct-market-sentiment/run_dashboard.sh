#!/bin/bash

# Kill any existing Streamlit processes
echo "Stopping any existing Streamlit processes..."
pkill -f streamlit

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Set environment variables for stability
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
export STREAMLIT_SERVER_FILE_WATCHER_TYPE=none

# Run Streamlit with auto-restart
echo "Starting Coherence Dashboard..."
while true; do
    streamlit run coherence_dashboard.py \
        --server.port 8505 \
        --server.address localhost \
        --server.fileWatcherType none \
        --browser.gatherUsageStats false
    
    echo "Streamlit crashed. Restarting in 5 seconds..."
    sleep 5
done