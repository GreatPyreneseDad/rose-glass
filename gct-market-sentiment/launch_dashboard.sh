#!/bin/bash
# Launch the GCT Database Dashboard cleanly

echo "🚀 Launching GCT Market Analysis Dashboard..."

# Kill any existing streamlit processes
echo "Cleaning up any existing processes..."
pkill -f streamlit 2>/dev/null || true
sleep 1

# Activate virtual environment
echo "Activating virtual environment..."
source /Users/chris/gct_env/bin/activate

# Check database exists
if [ ! -f "market_analysis.db" ]; then
    echo "⚠️  Database not found. Creating and populating with historical data..."
    python populate_historical_data.py
fi

# Clear Streamlit cache to avoid duplication issues
echo "Clearing Streamlit cache..."
rm -rf ~/.streamlit/cache 2>/dev/null || true

# Launch dashboard
echo "Starting dashboard on http://localhost:8501"
echo "Press Ctrl+C to stop"
echo ""

# Run streamlit with specific settings to avoid duplication
streamlit run database_dashboard.py \
    --server.headless true \
    --server.port 8501 \
    --browser.gatherUsageStats false \
    --theme.base "dark" \
    --server.runOnSave true