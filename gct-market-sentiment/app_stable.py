"""
Stable GCT Market Sentiment Analysis with LLM Integration
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import requests
from typing import Dict, List, Optional

# Initialize session state
if 'articles' not in st.session_state:
    st.session_state.articles = []
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = []

# Page config
st.set_page_config(
    page_title="GCT Market Sentiment",
    page_icon="📈",
    layout="wide"
)

# Simple GCT implementation
class SimpleGCT:
    """Simplified GCT implementation for stability"""
    
    @staticmethod
    def compute_coherence(psi: float, rho: float, q_raw: float, f: float) -> dict:
        """Compute GCT coherence"""
        km, ki = 0.3, 0.1
        q_opt = q_raw / (km + q_raw + (q_raw ** 2) / ki)
        
        base = psi
        wisdom_amp = rho * psi
        social_amp = f * psi
        coupling = 0.15 * rho * q_opt
        
        coherence = base + wisdom_amp + q_opt + social_amp + coupling
        
        return {
            'coherence': round(coherence, 3),
            'q_opt': round(q_opt, 3),
            'components': {
                'base': round(base, 3),
                'wisdom': round(wisdom_amp, 3),
                'emotional': round(q_opt, 3),
                'social': round(social_amp, 3),
                'coupling': round(coupling, 3)
            }
        }

# LLM Integration
class LLMAnalyzer:
    """Simple LLM integration for analysis"""
    
    @staticmethod
    def analyze_with_ollama(text: str, model: str = "llama2") -> dict:
        """Analyze text using Ollama (if available)"""
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model,
                    "prompt": f"""Analyze this financial news for market sentiment. Extract:
1. Clarity (0-1): How clear and specific is the information?
2. Reflection (0-1): How much analytical depth is shown?
3. Emotion (0-1): What's the emotional intensity?
4. Social (0-1): How much collective/consensus language?
5. Tickers: What stock symbols are mentioned?
6. Sentiment: bullish/bearish/neutral

News: {text}

Response in JSON format:""",
                    "stream": False
                },
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                # Parse LLM response
                try:
                    analysis = json.loads(result.get('response', '{}'))
                    return {
                        'psi': float(analysis.get('clarity', 0.5)),
                        'rho': float(analysis.get('reflection', 0.5)),
                        'q_raw': float(analysis.get('emotion', 0.5)),
                        'f': float(analysis.get('social', 0.5)),
                        'tickers': analysis.get('tickers', []),
                        'llm_sentiment': analysis.get('sentiment', 'neutral')
                    }
                except:
                    pass
        except:
            pass
            
        # Fallback to simple analysis
        return SimpleAnalyzer.analyze_text(text)

class SimpleAnalyzer:
    """Fallback analyzer when LLM is not available"""
    
    @staticmethod
    def analyze_text(text: str) -> dict:
        """Simple text analysis without LLM"""
        text_lower = text.lower()
        
        # Simple keyword-based analysis
        bullish_words = ['surge', 'rally', 'gain', 'rise', 'growth', 'profit', 'breakthrough']
        bearish_words = ['crash', 'fall', 'drop', 'loss', 'decline', 'concern', 'risk']
        
        bullish_score = sum(1 for word in bullish_words if word in text_lower)
        bearish_score = sum(1 for word in bearish_words if word in text_lower)
        
        # Extract simple metrics
        words = text.split()
        sentences = text.split('.')
        
        psi = min(1.0, len([w for w in words if len(w) > 6]) / max(len(words), 1))  # Specificity
        rho = min(1.0, len(sentences) / 10)  # Depth
        q_raw = min(1.0, (bullish_score + bearish_score) / 5)  # Emotion
        f = 0.3  # Default social signal
        
        # Extract tickers (simple pattern matching)
        import re
        ticker_pattern = r'\b[A-Z]{2,5}\b'
        potential_tickers = re.findall(ticker_pattern, text)
        tickers = [t for t in potential_tickers if len(t) >= 2 and t not in ['THE', 'AND', 'FOR', 'CEO']]
        
        return {
            'psi': round(psi, 3),
            'rho': round(rho, 3),
            'q_raw': round(q_raw, 3),
            'f': round(f, 3),
            'tickers': tickers[:5],  # Limit to 5 tickers
            'llm_sentiment': 'bullish' if bullish_score > bearish_score else 'bearish' if bearish_score > bullish_score else 'neutral'
        }

# Tiingo Integration
def fetch_tiingo_news(api_key: str, tickers: List[str] = None) -> List[dict]:
    """Fetch news from Tiingo API"""
    if not api_key:
        return []
        
    try:
        url = "https://api.tiingo.com/tiingo/news"
        params = {
            'token': api_key,
            'limit': 20,
            'sortBy': 'publishedDate'
        }
        
        if tickers:
            params['tickers'] = ','.join(tickers)
            
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            articles = response.json()
            return [
                {
                    'id': article.get('id', ''),
                    'timestamp': article.get('publishedDate', ''),
                    'title': article.get('title', ''),
                    'body': article.get('description', ''),
                    'source': article.get('source', ''),
                    'tickers': article.get('tickers', [])
                }
                for article in articles
            ]
    except Exception as e:
        st.error(f"Error fetching news: {e}")
        
    return []

# Main UI
st.title("🧠 GCT Market Sentiment Analysis")
st.markdown("**Stable Version** - Analyzes financial news using Grounded Coherence Theory")

# Sidebar
with st.sidebar:
    st.header("Configuration")
    
    # API Key
    api_key = st.text_input(
        "Tiingo API Key",
        value=os.getenv('TIINGO_API_TOKEN', 'ef1915ca2231c0c953e4fb7b72dec74bc767d9d1'),
        type="password"
    )
    
    # LLM Settings
    st.subheader("LLM Settings")
    use_llm = st.checkbox("Use LLM Analysis (Ollama)", value=False)
    llm_model = st.selectbox("LLM Model", ["llama2", "mistral", "llama3.2"], index=0)
    
    # Analysis Mode
    st.subheader("Analysis Mode")
    mode = st.radio(
        "Select Mode",
        ["Manual Input", "Tiingo News", "Sample Data"]
    )

# Main content
if mode == "Manual Input":
    st.header("📝 Manual News Input")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        title = st.text_input("Article Title")
        body = st.text_area("Article Body", height=150)
        
        if st.button("Analyze", type="primary"):
            if title or body:
                full_text = f"{title} {body}"
                
                with st.spinner("Analyzing..."):
                    # Get analysis
                    if use_llm:
                        analysis = LLMAnalyzer.analyze_with_ollama(full_text, llm_model)
                    else:
                        analysis = SimpleAnalyzer.analyze_text(full_text)
                    
                    # Compute coherence
                    gct_result = SimpleGCT.compute_coherence(
                        analysis['psi'],
                        analysis['rho'],
                        analysis['q_raw'],
                        analysis['f']
                    )
                    
                    # Store result
                    result = {
                        'timestamp': datetime.now().isoformat(),
                        'title': title,
                        'analysis': analysis,
                        'gct': gct_result,
                        'sentiment': 'bullish' if gct_result['coherence'] > 0.6 else 'bearish' if gct_result['coherence'] < 0.4 else 'neutral'
                    }
                    
                    st.session_state.analysis_results.append(result)
                    st.success("Analysis complete!")
                    
    with col2:
        if st.session_state.analysis_results:
            latest = st.session_state.analysis_results[-1]
            
            st.subheader("Latest Analysis")
            
            # Metrics
            st.metric("Coherence", f"{latest['gct']['coherence']:.3f}")
            st.metric("Sentiment", latest['sentiment'])
            
            # Variables
            st.write("**GCT Variables:**")
            st.write(f"- ψ (Clarity): {latest['analysis']['psi']:.3f}")
            st.write(f"- ρ (Reflection): {latest['analysis']['rho']:.3f}")
            st.write(f"- q (Emotion): {latest['analysis']['q_raw']:.3f}")
            st.write(f"- f (Social): {latest['analysis']['f']:.3f}")
            
            if latest['analysis'].get('tickers'):
                st.write(f"**Tickers:** {', '.join(latest['analysis']['tickers'])}")
                
elif mode == "Tiingo News":
    st.header("📰 Real-Time News Analysis")
    
    # Fetch news
    if st.button("Fetch Latest News"):
        with st.spinner("Fetching news..."):
            articles = fetch_tiingo_news(api_key)
            
            if articles:
                st.success(f"Fetched {len(articles)} articles")
                st.session_state.articles = articles
            else:
                st.warning("No articles fetched. Check your API key.")
                
    # Display articles
    if st.session_state.articles:
        for article in st.session_state.articles[:5]:  # Show only 5
            with st.expander(f"{article['title'][:80]}..."):
                st.write(f"**Source:** {article['source']}")
                st.write(f"**Time:** {article['timestamp']}")
                st.write(f"**Tickers:** {', '.join(article.get('tickers', []))}")
                st.write(f"**Description:** {article['body']}")
                
                if st.button(f"Analyze", key=f"analyze_{article['id']}"):
                    with st.spinner("Analyzing..."):
                        full_text = f"{article['title']} {article['body']}"
                        
                        if use_llm:
                            analysis = LLMAnalyzer.analyze_with_ollama(full_text, llm_model)
                        else:
                            analysis = SimpleAnalyzer.analyze_text(full_text)
                            
                        gct_result = SimpleGCT.compute_coherence(
                            analysis['psi'],
                            analysis['rho'],
                            analysis['q_raw'],
                            analysis['f']
                        )
                        
                        st.json({
                            'analysis': analysis,
                            'gct': gct_result,
                            'coherence': gct_result['coherence']
                        })
                        
else:  # Sample Data
    st.header("🧪 Sample Data Analysis")
    
    sample_articles = [
        {
            'title': 'Tech Giants Rally on AI Breakthrough',
            'body': 'Major technology companies including NVDA and MSFT saw significant gains today following breakthrough AI announcements. Investors are extremely optimistic about growth potential.',
            'timestamp': datetime.now().isoformat()
        },
        {
            'title': 'Fed Signals Concern Over Economic Data',
            'body': 'The Federal Reserve expressed concerns about recent economic indicators. Market volatility increased as investors worry about potential policy changes affecting SPY and bond markets.',
            'timestamp': (datetime.now() - timedelta(hours=2)).isoformat()
        }
    ]
    
    for i, article in enumerate(sample_articles):
        with st.container():
            st.subheader(article['title'])
            st.write(article['body'])
            
            if st.button(f"Analyze Sample {i+1}"):
                with st.spinner("Analyzing..."):
                    full_text = f"{article['title']} {article['body']}"
                    
                    if use_llm:
                        analysis = LLMAnalyzer.analyze_with_ollama(full_text, llm_model)
                    else:
                        analysis = SimpleAnalyzer.analyze_text(full_text)
                        
                    gct_result = SimpleGCT.compute_coherence(
                        analysis['psi'],
                        analysis['rho'],
                        analysis['q_raw'],
                        analysis['f']
                    )
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Coherence", f"{gct_result['coherence']:.3f}")
                    with col2:
                        sentiment = 'bullish' if gct_result['coherence'] > 0.6 else 'bearish' if gct_result['coherence'] < 0.4 else 'neutral'
                        st.metric("Sentiment", sentiment)
                    with col3:
                        st.metric("Confidence", f"{min(gct_result['coherence'] * 100, 95):.0f}%")
                        
                    st.json(gct_result)

# History
if st.session_state.analysis_results:
    st.header("📊 Analysis History")
    
    df = pd.DataFrame([
        {
            'Time': r['timestamp'],
            'Title': r['title'][:50] + '...' if len(r['title']) > 50 else r['title'],
            'Coherence': r['gct']['coherence'],
            'Sentiment': r['sentiment']
        }
        for r in st.session_state.analysis_results
    ])
    
    st.dataframe(df, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("GCT Market Sentiment - Stable Version with LLM Support")