#!/bin/bash

echo "🌹 Starting Rose Glass Dream Analysis Server..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Add parent directory to Python path
export PYTHONPATH="${PYTHONPATH}:$(dirname $(pwd))"

# Check for .env file
if [ ! -f ".env" ]; then
    echo "No .env file found. Creating from template..."
    cp .env.example .env
    echo "Please edit .env to add your API keys (optional)"
fi

# Start the server
echo "Starting server on http://localhost:8000"
python dream_app.py