#!/bin/bash

# GCT Dream Analysis Engine - Quick Start Script

echo "🌙 Starting GCT Dream Analysis Engine..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -q -r requirements.txt

# Create data directory if it doesn't exist
mkdir -p data

# Launch the application
echo "Launching Dream Analysis Engine..."
streamlit run app.py --theme.base="dark" --theme.primaryColor="#60a5fa" --theme.backgroundColor="#0f172a" --theme.secondaryBackgroundColor="#1e293b"