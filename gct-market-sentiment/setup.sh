#!/bin/bash

echo "🧠 Setting up GCT Market Sentiment Analysis..."

# Create data directory
mkdir -p data

# Create virtual environment
echo "Creating Python virtual environment..."
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt

# Download spaCy model
echo "Downloading spaCy language model..."
python -m spacy download en_core_web_sm

# Download NLTK data
echo "Downloading NLTK data..."
python -c "import nltk; nltk.download('vader_lexicon')"

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp .env.template .env
    echo "⚠️  Please edit .env and add your Tiingo API token"
fi

echo "✅ Setup complete!"
echo ""
echo "To run the dashboard:"
echo "  source venv/bin/activate"
echo "  streamlit run app.py"
echo ""
echo "Dashboard will open at http://localhost:8501"