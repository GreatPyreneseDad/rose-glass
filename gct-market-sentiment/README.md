# GCT Market Sentiment Analysis

A financial market sentiment analysis system powered by **Grounded Coherence Theory (GCT)**. This system analyzes financial news to detect coherence shifts in market narratives and predict potential price movements.

## 🧠 Overview

Unlike traditional sentiment analysis that relies on keyword matching, this system uses GCT to model the psychological dynamics of market narratives through:

- **ψ (Psi)**: Clarity and precision of financial narratives
- **ρ (Rho)**: Reflective depth and analytical nuance
- **q (Q)**: Emotional charge (optimized through wisdom modulation)
- **f (F)**: Social belonging and market consensus signals

By tracking coherence (C) and its derivatives over time, the system identifies:
- **Bullish signals**: dC/dt > 0.05 (rising narrative coherence)
- **Bearish signals**: dC/dt < -0.05 (falling narrative coherence)
- **Spike alerts**: |d²C/dt²| > 0.1 (rapid coherence acceleration)

## 🚀 Quick Start

### 1. Clone and Setup

```bash
cd gct-market-sentiment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure API Access

```bash
cp .env.template .env
# Edit .env and add your Tiingo API token
```

To get a free Tiingo API token:
1. Sign up at https://tiingo.com/
2. Go to your account settings
3. Copy your API token

### 3. Run the Dashboard

```bash
streamlit run app.py
```

The dashboard will open at http://localhost:8501

## 📊 Features

### Real-time Analysis
- Ingest financial news via Tiingo REST and WebSocket APIs
- Extract GCT variables using advanced NLP
- Compute coherence scores and derivatives
- Generate bullish/bearish/neutral signals

### Dashboard Components
- **Market Overview**: Key metrics and sentiment distribution
- **Top Movers**: Tickers with strongest bullish/bearish signals
- **Coherence Timeline**: Interactive charts showing C and dC/dt over time
- **Article Analysis**: Recent news with GCT scores
- **Spike Alerts**: Detect rapid coherence changes

### Sector-Aware Tuning
Different market sectors have customized GCT parameters:
- **Tech**: More sensitive to innovation narratives
- **Finance**: Conservative coherence thresholds
- **Energy**: Higher spike detection thresholds
- **Healthcare**: Balanced coupling strength

## 🔧 Architecture

```
├── src/
│   ├── gct_engine.py         # Core GCT coherence computations
│   ├── nlp_extractor.py      # Extract ψ, ρ, q, f from text
│   ├── database.py           # SQLite storage layer
│   ├── tiingo_client.py      # News data ingestion
│   └── analysis_pipeline.py  # Main processing pipeline
├── app.py                    # Streamlit dashboard
├── requirements.txt          # Python dependencies
└── data/                     # Database storage
```

## 📈 Using the System

### Without API Token (Mock Mode)
The system includes mock data for testing:
```bash
streamlit run app.py
```

### With Real Data
1. Add your Tiingo API token to `.env`
2. Click "Backfill Historical Data" to load past news
3. Monitor real-time signals as they emerge

### API Docs
When the dashboard is running you can access the underlying API specification at
`/docs` for an interactive view or `/openapi.json` for machine consumption.

### Interpreting Signals

1. **Coherence Score (C)**: Overall narrative strength (0-1)
   - High coherence = strong, unified market narrative
   - Low coherence = confused, conflicting narratives

2. **First Derivative (dC/dt)**: Momentum
   - Positive = strengthening narrative (bullish)
   - Negative = weakening narrative (bearish)

3. **Second Derivative (d²C/dt²)**: Acceleration
   - Large absolute values = rapid narrative shifts
   - Potential inflection points in market sentiment

## 🛠️ Development

### Running Tests
```bash
pytest tests/
```

### Adding New Sectors
Edit `SECTOR_PARAMS` in `src/gct_engine.py`:
```python
'your_sector': GCTParameters(km=0.3, ki=0.1, bullish_threshold=0.06)
```

### Customizing NLP
Modify extraction methods in `src/nlp_extractor.py` to tune variable extraction for your domain.

## 📚 Theory

This implementation is based on the enhanced Grounded Coherence Theory (eGCT) model:

```
C = ψ + ρ·ψ + q_opt + f·ψ + α·ρ·q_opt

where q_opt = q_raw / (Km + q_raw + q_raw²/Ki)
```

The model captures how market narratives evolve through the interplay of clarity, wisdom, emotion, and social dynamics.

## ⚠️ Disclaimer

This system is for research and analysis purposes only. It does not provide investment advice or automated trading. Always conduct your own research before making investment decisions.

## 📄 License

This project is part of the GCT research initiative. See LICENSE for details.
