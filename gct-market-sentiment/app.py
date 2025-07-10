"""
Streamlit Dashboard for GCT Market Sentiment Analysis
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.analysis_pipeline import GCTAnalysisPipeline
from src.database import GCTDatabase


# Page config
st.set_page_config(
    page_title="GCT Market Sentiment",
    page_icon="📈",
    layout="wide"
)

# Initialize components
@st.cache_resource
def init_pipeline():
    """Initialize the analysis pipeline"""
    # Check for API token
    api_token = os.getenv('TIINGO_API_TOKEN')
    use_mock = not api_token
    
    if use_mock:
        st.warning("No Tiingo API token found. Using mock data. Set TIINGO_API_TOKEN to use real data.")
        
    return GCTAnalysisPipeline(api_token=api_token, use_mock=use_mock)

@st.cache_resource
def init_database():
    """Initialize database connection"""
    return GCTDatabase()

pipeline = init_pipeline()
db = init_database()

# Title and description
st.title("🧠 GCT Market Sentiment Analysis")
st.markdown("""
**Grounded Coherence Theory** applied to financial markets - detecting narrative coherence shifts to predict market movements.

- **Bullish Signal**: dC/dt > 0.05 (rising coherence)
- **Bearish Signal**: dC/dt < -0.05 (falling coherence)
- **Spike Alert**: |d²C/dt²| > 0.1 (rapid acceleration)
""")

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    
    # Refresh data
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()
        
    # Backfill option
    st.subheader("Data Ingestion")
    backfill_days = st.number_input("Backfill Days", 1, 30, 7)
    if st.button("📥 Backfill Historical Data"):
        with st.spinner("Backfilling data..."):
            pipeline.backfill_historical(days=backfill_days)
            st.success("Backfill complete!")
            st.rerun()
            
    # View options
    st.subheader("View Options")
    show_raw_data = st.checkbox("Show Raw Data Tables")
    
    # Time range
    time_range = st.selectbox(
        "Time Range",
        ["Last 24 Hours", "Last 7 Days", "Last 30 Days"],
        index=0
    )

# Main dashboard layout
col1, col2, col3 = st.columns([1, 1, 1])

# Get market summary
summary = pipeline.get_market_summary()
stats = summary['overall_stats'] or {}

# Display key metrics
with col1:
    st.metric(
        "Total Articles Analyzed",
        stats.get('total_articles', 0),
        delta=None
    )
    
with col2:
    avg_coherence = stats.get('avg_coherence', 0) or 0
    st.metric(
        "Average Coherence",
        f"{avg_coherence:.3f}",
        delta=None
    )
    
with col3:
    bullish_pct = 0
    total = stats.get('total_articles', 0)
    if total > 0:
        bullish = stats.get('bullish_count', 0)
        bullish_pct = (bullish / total) * 100
    st.metric(
        "Bullish Sentiment %",
        f"{bullish_pct:.1f}%",
        delta=None
    )

# Top movers section
st.header("📊 Market Signals")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🚀 Top Bullish Signals")
    top_bullish = summary.get('top_bullish', [])
    
    if top_bullish:
        bullish_df = pd.DataFrame(top_bullish)
        bullish_df['avg_dc_dt'] = bullish_df['avg_dc_dt'].round(3)
        bullish_df['avg_coherence'] = bullish_df['avg_coherence'].round(3)
        
        st.dataframe(
            bullish_df[['ticker', 'avg_dc_dt', 'avg_coherence', 'mention_count']],
            hide_index=True
        )
    else:
        st.info("No bullish signals in the selected timeframe")
        
with col2:
    st.subheader("📉 Top Bearish Signals")
    top_bearish = summary.get('top_bearish', [])
    
    if top_bearish:
        bearish_df = pd.DataFrame(top_bearish)
        bearish_df['avg_dc_dt'] = bearish_df['avg_dc_dt'].round(3)
        bearish_df['avg_coherence'] = bearish_df['avg_coherence'].round(3)
        
        st.dataframe(
            bearish_df[['ticker', 'avg_dc_dt', 'avg_coherence', 'mention_count']],
            hide_index=True
        )
    else:
        st.info("No bearish signals in the selected timeframe")

# Coherence timeline
st.header("📈 Coherence Analysis")

# Ticker selection
all_tickers = set()
if top_bullish:
    all_tickers.update([item['ticker'] for item in top_bullish])
if top_bearish:
    all_tickers.update([item['ticker'] for item in top_bearish])
    
if all_tickers:
    selected_ticker = st.selectbox(
        "Select Ticker for Detailed Analysis",
        sorted(list(all_tickers))
    )
    
    # Get timeline data
    hours = {"Last 24 Hours": 24, "Last 7 Days": 168, "Last 30 Days": 720}[time_range]
    timeline_df = db.get_ticker_timeline(selected_ticker, hours=hours)
    
    if not timeline_df.empty:
        # Coherence over time
        fig1 = go.Figure()
        
        # Add coherence line
        fig1.add_trace(go.Scatter(
            x=timeline_df['timestamp'],
            y=timeline_df['coherence'],
            mode='lines+markers',
            name='Coherence',
            line=dict(color='blue', width=2)
        ))
        
        # Add sentiment coloring
        colors = {'bullish': 'green', 'bearish': 'red', 'neutral': 'gray'}
        for sentiment in ['bullish', 'bearish', 'neutral']:
            sent_df = timeline_df[timeline_df['sentiment'] == sentiment]
            if not sent_df.empty:
                fig1.add_trace(go.Scatter(
                    x=sent_df['timestamp'],
                    y=sent_df['coherence'],
                    mode='markers',
                    name=sentiment.capitalize(),
                    marker=dict(color=colors[sentiment], size=10)
                ))
                
        fig1.update_layout(
            title=f"{selected_ticker} - Coherence Timeline",
            xaxis_title="Time",
            yaxis_title="Coherence Score",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig1, use_container_width=True)
        
        # Derivative plot
        fig2 = go.Figure()
        
        fig2.add_trace(go.Scatter(
            x=timeline_df['timestamp'],
            y=timeline_df['dc_dt'],
            mode='lines+markers',
            name='dC/dt',
            line=dict(color='purple', width=2)
        ))
        
        # Add threshold lines
        fig2.add_hline(y=0.05, line_dash="dash", line_color="green", 
                       annotation_text="Bullish Threshold")
        fig2.add_hline(y=-0.05, line_dash="dash", line_color="red", 
                       annotation_text="Bearish Threshold")
        fig2.add_hline(y=0, line_color="gray")
        
        fig2.update_layout(
            title=f"{selected_ticker} - Coherence Derivative (Momentum)",
            xaxis_title="Time",
            yaxis_title="dC/dt",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig2, use_container_width=True)
        
    else:
        st.info(f"No data available for {selected_ticker} in the selected timeframe")
        
# Recent articles analysis
st.header("📰 Recent Article Analysis")

recent_articles = db.get_recent_articles(limit=10)

if recent_articles:
    # Create a simple view of recent articles
    articles_data = []
    for article in recent_articles:
        # Get GCT scores for this article
        article_with_scores = db.get_article_with_scores(article['id'])
        
        if article_with_scores and article_with_scores.get('coherence') is not None:
            articles_data.append({
                'Time': pd.to_datetime(article['timestamp']).strftime('%Y-%m-%d %H:%M'),
                'Title': article['title'][:80] + '...' if len(article['title']) > 80 else article['title'],
                'Tickers': ', '.join(article['tickers'][:3]),
                'Coherence': round(article_with_scores['coherence'], 3),
                'Sentiment': article_with_scores.get('sentiment', 'neutral'),
                'dC/dt': round(article_with_scores.get('dc_dt', 0), 3)
            })
            
    if articles_data:
        articles_df = pd.DataFrame(articles_data)
        
        # Color code by sentiment
        def color_sentiment(val):
            if val == 'bullish':
                return 'background-color: #90EE90'
            elif val == 'bearish':
                return 'background-color: #FFB6C1'
            return ''
            
        styled_df = articles_df.style.applymap(
            color_sentiment, 
            subset=['Sentiment']
        )
        
        st.dataframe(styled_df, hide_index=True)
        
# Spike detection
st.header("⚡ Coherence Spike Alerts")

spikes = pipeline.detect_coherence_spikes()
if spikes:
    spike_df = pd.DataFrame(spikes)
    spike_df['timestamp'] = pd.to_datetime(spike_df['timestamp'])
    spike_df = spike_df.sort_values('timestamp', ascending=False)
    
    st.dataframe(
        spike_df[['timestamp', 'sector', 'coherence', 'd2c_dt2', 'type']],
        hide_index=True
    )
else:
    st.info("No significant coherence spikes detected")
    
# Raw data tables (optional)
if show_raw_data:
    st.header("🗃️ Raw Data Tables")
    
    with st.expander("Recent GCT Scores"):
        query = """
            SELECT * FROM GCTScores 
            ORDER BY created_at DESC 
            LIMIT 100
        """
        scores_df = pd.read_sql_query(query, db.get_connection())
        st.dataframe(scores_df)
        
    with st.expander("Ticker Timeline"):
        query = """
            SELECT * FROM TickerTimeline 
            ORDER BY created_at DESC 
            LIMIT 100
        """
        timeline_df = pd.read_sql_query(query, db.get_connection())
        st.dataframe(timeline_df)
        
# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>GCT Market Sentiment Analysis | Powered by Grounded Coherence Theory</p>
    <p style='font-size: 0.8em'>ψ: Clarity | ρ: Wisdom | q: Emotion | f: Social Belonging</p>
</div>
""", unsafe_allow_html=True)