# SPEC-1: Market Sentiment Forecasting with GCT

This specification outlines a sentiment engine that applies **Grounded Coherence Theory (GCT)** to financial news. The goal is to detect early shifts in narrative coherence in order to anticipate bullish or bearish behavior in markets.

## Overview
GCT describes how coherence in beliefs, emotions, and social belonging evolves. By measuring coherence and its derivatives from financial narratives, we can generate actionable sentiment signals before they appear in trading data.

### Key Requirements
- Ingest financial news via the provided API token.
- Compute coherence `(C)` and derivatives `dC/dt` using GCT variables `(\Psi, \rho, q, f, K_m, K_i)`.
- Identify tickers mentioned in each article and store them with coherence scores.
- Label sentiment for each ticker based on `dC/dt` thresholds:
  - **Bullish**: `dC/dt > 0.05`
  - **Bearish**: `dC/dt < -0.05`
  - **Neutral**: otherwise
- Provide a dashboard showing signals and coherence trajectories.

### Optional Enhancements
- Spike alerts based on `d^2C/dt^2`.
- Sector tuning of parameters and integration with historical stock data.

## Architecture Diagram
```
User --> Dashboard --> Coherence_DB
Tiingo_REST --> News_Processor --> Entity_Mapper --> GCT_Engine --> Coherence_DB
Tiingo_WS  --> News_Processor
GCT_Engine --> Sentiment_Labeler --> Coherence_DB
```

## Core Coherence Function
```python
def compute_gct_coherence(psi, rho, q_raw, f, km, ki):
    q_opt = (q_raw) / (km + q_raw + (q_raw ** 2) / ki)
    base = psi
    wisdom_amp = rho * psi
    social_amp = f * psi
    coupling = 0.15 * rho * q_opt
    coherence = base + wisdom_amp + q_opt + social_amp + coupling
    return coherence, q_opt
```

## Database Schema
```sql
CREATE TABLE NewsArticles (
  id UUID PRIMARY KEY,
  timestamp TIMESTAMPTZ,
  source TEXT,
  title TEXT,
  body TEXT,
  tickers TEXT[]
);

CREATE TABLE GCTScores (
  id UUID PRIMARY KEY,
  article_id UUID REFERENCES NewsArticles(id),
  psi FLOAT,
  rho FLOAT,
  q_raw FLOAT,
  f FLOAT,
  q_opt FLOAT,
  coherence FLOAT,
  dc_dt FLOAT,
  dc2_dt2 FLOAT,
  label TEXT
);

CREATE TABLE TickerTimeline (
  id UUID PRIMARY KEY,
  ticker TEXT,
  timestamp TIMESTAMPTZ,
  coherence FLOAT,
  dc_dt FLOAT,
  label TEXT
);
```

## Milestones
1. **Week 1** – News ingestion and database setup.
2. **Week 2** – NLP extraction and GCT engine implementation.
3. **Week 3** – Sentiment scoring pipeline.
4. **Week 4** – Dashboard MVP with coherence plots.
5. **Week 5** – Historical validation and testing.
6. **Week 6** – Deployment and documentation.
```
