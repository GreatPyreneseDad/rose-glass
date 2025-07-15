#!/bin/bash

# GCT Creative Flow - Quick Start Script

echo "🎨 Starting GCT Creative Flow Tracker..."
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -q streamlit plotly numpy pandas scipy networkx

# Launch the app
echo ""
echo "✨ Launching Creative Flow Tracker..."
echo "📱 Opening in your browser..."
echo ""

streamlit run app_simple.py