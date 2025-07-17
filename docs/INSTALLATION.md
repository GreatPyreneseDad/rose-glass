# GCT Installation Guide

This guide will help you install and configure the Grounded Coherence Theory (GCT) implementation suite.

## Prerequisites

- **Python 3.8 or higher** (3.10+ recommended)
- **Git** for cloning the repository
- **pip** for package management
- **Virtual environment** support (venv)

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/GreatPyreneseDad/GCT.git
cd GCT
```

### 2. Choose Your Module

The GCT suite includes several modules:

- `gct-market-sentiment/` - Financial market analysis
- `soulmath-moderation-system/` - Content moderation with coherence
- `gct-creative-flow/` - Creative process analysis
- `gct-dream-analysis/` - Dream pattern coherence
- `soulmath-fear-elevation/` - Fear transformation system

### 3. Install Market Sentiment Module

```bash
cd gct-market-sentiment

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLP models
python -m spacy download en_core_web_sm
```

### 4. Configure Environment

```bash
# Copy template
cp .env.template .env

# Edit .env with your API tokens
nano .env  # or use your preferred editor
```

### 5. Get Required API Keys

#### Tiingo API (Required)
1. Visit [https://tiingo.com/](https://tiingo.com/)
2. Sign up for a free account
3. Go to your account settings
4. Copy your API token
5. Add to `.env`: `TIINGO_API_TOKEN=your_token_here`

#### GitHub Token (Optional)
1. Go to GitHub Settings > Developer settings > Personal access tokens
2. Generate new token (classic)
3. Select scopes: `repo`, `read:org`
4. Add to `.env`: `GITHUB_TOKEN=your_token_here`

### 6. Run the Application

```bash
# Basic dashboard
streamlit run app.py

# With specific port
streamlit run app.py --server.port 8501

# Minimal version (fewer features)
streamlit run app_minimal.py

# Stable version
streamlit run app_stable.py
```

## Troubleshooting

### Python Version Issues

If you encounter issues with dependencies (especially on Python 3.13):

```bash
# Use the minimal requirements
pip install -r requirements-minimal.txt
```

### Port Already in Use

```bash
# Kill existing Streamlit processes
pkill -f streamlit

# Or use a different port
streamlit run app.py --server.port 8503
```

### Missing Dependencies

```bash
# For NLP features
pip install spacy
python -m spacy download en_core_web_sm

# For enhanced NLP (requires compatible Python version)
pip install transformers torch
```

### Database Issues

```bash
# Reset database
rm data/gct_market.db
python -c "from src.database import GCTDatabase; GCTDatabase().init_db()"
```

## Docker Installation (Alternative)

```bash
# Build Docker image
docker build -t gct-market-sentiment .

# Run container
docker run -p 8501:8501 -e TIINGO_API_TOKEN=your_token gct-market-sentiment
```

## Verifying Installation

```bash
# Run tests
pytest

# Check system
python test_system.py

# Validate API connection
python test_api.py
```

## Next Steps

1. **Explore the Dashboard**: Open http://localhost:8501 in your browser
2. **Read the Documentation**: Check `/docs` folder for detailed guides
3. **Configure Alerts**: Set up email or webhook notifications
4. **Customize Tickers**: Edit the ticker list in the dashboard
5. **Review Theory**: Read `WHITEPAPER.md` for GCT mathematical details

## Support

- **Issues**: [GitHub Issues](https://github.com/GreatPyreneseDad/GCT/issues)
- **Discussions**: [GitHub Discussions](https://github.com/GreatPyreneseDad/GCT/discussions)
- **Email**: See repository contact information

## License

This project is licensed under the MIT License - see the LICENSE file for details.