"""
Simple Coherence Dashboard using Streamlit
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os

# Import our monitor
from simple_coherence_monitor import fetch_ticker_data, calculate_simple_coherence

st.set_page_config(page_title="GCT Coherence Monitor", page_icon="📊", layout="wide")

st.title("📊 Market Coherence Monitor")
st.markdown("Track building and falling coherence patterns in real-time")

# Sidebar
with st.sidebar:
    st.header("Settings")
    
    # Ticker input
    ticker = st.text_input("Ticker Symbol", value="AAPL").upper()
    
    # Time range
    days = st.slider("Days of History", 10, 90, 30)
    
    # Refresh button
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"{ticker} Coherence Analysis")
    
    # Fetch data
    with st.spinner("Fetching data..."):
        df = fetch_ticker_data(ticker, days)
        
    if df is not None:
        # Calculate coherence over time
        coherence_data = []
        
        for i in range(20, len(df)):
            window = df.iloc[:i]
            metrics = calculate_simple_coherence(window)
            if metrics:
                coherence_data.append({
                    'date': df.index[i],
                    'coherence': metrics['coherence'],
                    'trend': metrics['trend_strength'],
                    'price': window['close'].iloc[-1]
                })
        
        if coherence_data:
            coh_df = pd.DataFrame(coherence_data)
            
            # Create figure with secondary y-axis
            fig = go.Figure()
            
            # Add price line
            fig.add_trace(go.Scatter(
                x=coh_df['date'],
                y=coh_df['price'],
                name='Price',
                line=dict(color='gray', width=1),
                yaxis='y2'
            ))
            
            # Add coherence line
            fig.add_trace(go.Scatter(
                x=coh_df['date'],
                y=coh_df['coherence'],
                name='Coherence',
                line=dict(color='blue', width=2)
            ))
            
            # Add coherence zones
            fig.add_hline(y=0.7, line_dash="dash", line_color="green", 
                         annotation_text="High Coherence", annotation_position="left")
            fig.add_hline(y=0.3, line_dash="dash", line_color="red", 
                         annotation_text="Low Coherence", annotation_position="left")
            
            # Update layout
            fig.update_layout(
                title=f"{ticker} Price and Coherence",
                xaxis_title="Date",
                yaxis=dict(title="Coherence", side="left", range=[0, 1]),
                yaxis2=dict(title="Price ($)", side="right", overlaying="y"),
                hovermode='x unified',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Trend strength chart
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=coh_df['date'],
                y=coh_df['trend'],
                name='Trend Strength',
                line=dict(color='purple', width=2),
                fill='tozeroy'
            ))
            
            fig2.add_hline(y=0, line_color="black", line_width=1)
            
            fig2.update_layout(
                title="Trend Strength (5-day vs 20-day)",
                xaxis_title="Date",
                yaxis_title="Trend Strength",
                height=300
            )
            
            st.plotly_chart(fig2, use_container_width=True)
            
    else:
        st.error(f"Failed to fetch data for {ticker}")

with col2:
    st.subheader("Current Metrics")
    
    if df is not None:
        current = calculate_simple_coherence(df)
        
        if current:
            # Display metrics
            st.metric("Coherence Score", f"{current['coherence']:.3f}")
            
            # Coherence interpretation
            if current['coherence'] > 0.7:
                st.success("🟢 HIGH COHERENCE - Strong unified movement")
            elif current['coherence'] < 0.3:
                st.error("🔴 LOW COHERENCE - Chaotic/uncertain")
            else:
                st.warning("🟡 MODERATE COHERENCE - Normal market")
            
            st.metric("Trend Strength", f"{current['trend_strength']:+.2%}")
            st.metric("Current Price", f"${current['last_price']:.2f}")
            st.metric("5-Day Change", f"{current['price_change_5d']:+.2f}%")
            
            # Additional metrics
            st.divider()
            st.write("**Volatility Ratio**:", f"{current['volatility_ratio']:.2f}")
            st.write("**Volume Ratio**:", f"{current['volume_ratio']:.2f}")
            
            # Pattern detection
            st.divider()
            st.subheader("Pattern Detection")
            
            if current['trend_strength'] > 0.03 and current['coherence'] > 0.6:
                st.info("📈 Building coherence pattern detected")
            elif current['trend_strength'] < -0.03 and current['coherence'] < 0.4:
                st.warning("📉 Falling coherence pattern detected")
            else:
                st.write("No significant patterns")

# Bottom section - Quick analysis of multiple tickers
st.divider()
st.subheader("Multi-Ticker Coherence Scan")

scan_tickers = st.text_input("Enter tickers (comma-separated)", 
                            value="AAPL,MSFT,NVDA,SPY,QQQ").upper()

if st.button("Run Coherence Scan"):
    tickers = [t.strip() for t in scan_tickers.split(',')]
    
    scan_results = []
    
    progress = st.progress(0)
    for i, t in enumerate(tickers):
        progress.progress((i + 1) / len(tickers))
        
        df = fetch_ticker_data(t, 30)
        if df is not None:
            metrics = calculate_simple_coherence(df)
            if metrics:
                scan_results.append({
                    'Ticker': t,
                    'Coherence': metrics['coherence'],
                    'Trend': f"{metrics['trend_strength']:+.2%}",
                    'Price': f"${metrics['last_price']:.2f}",
                    '5D Change': f"{metrics['price_change_5d']:+.1f}%",
                    'Pattern': '🟢 Building' if metrics['coherence'] > 0.7 and metrics['trend_strength'] > 0.02 
                              else '🔴 Falling' if metrics['coherence'] < 0.3 
                              else '⚪ Neutral'
                })
    
    progress.empty()
    
    if scan_results:
        scan_df = pd.DataFrame(scan_results)
        st.dataframe(scan_df, hide_index=True, use_container_width=True)