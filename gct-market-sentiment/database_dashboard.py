#!/usr/bin/env python3
"""
GCT Database Dashboard - View historical market analysis data
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, date, timedelta
from database_manager import MarketDatabase

# Page config
st.set_page_config(
    page_title="GCT Market Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

# Initialize database
@st.cache_resource
def get_database():
    return MarketDatabase()

db = get_database()

# Title
st.title("🌐 GCT Market Analysis Dashboard")
st.markdown("Historical market sentiment and coherence analysis")

# Date range selector
col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    start_date = st.date_input(
        "Start Date",
        value=date.today() - timedelta(days=30),
        max_value=date.today()
    )
with col2:
    end_date = st.date_input(
        "End Date",
        value=date.today(),
        max_value=date.today()
    )

# Market Summary Section
st.header("📈 Market Overview")

# Get market summary history
market_history = db.get_market_summary_history(days=(date.today() - start_date).days)
market_df = pd.DataFrame(market_history)

# Remove duplicates if any
if not market_df.empty:
    market_df = market_df.drop_duplicates(subset=['date'], keep='last')
    market_df = market_df.sort_values('date', ascending=False)

if not market_df.empty:
    # Latest metrics
    latest = market_df.iloc[0]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Average Coherence",
            f"{latest['avg_coherence']:.3f}",
            delta=f"{latest['avg_coherence'] - market_df.iloc[1]['avg_coherence']:.3f}" if len(market_df) > 1 else None
        )
    
    with col2:
        st.metric(
            "Average Truth Cost",
            f"{latest['avg_truth_cost']:.3f}",
            delta=f"{latest['avg_truth_cost'] - market_df.iloc[1]['avg_truth_cost']:.3f}" if len(market_df) > 1 else None,
            delta_color="inverse"
        )
    
    with col3:
        st.metric(
            "Market Health",
            latest['market_health'],
            delta=None
        )
    
    with col4:
        st.metric(
            "Fear & Greed Index",
            f"{latest['fear_greed_index']:.2f}",
            delta=f"{latest['fear_greed_index'] - market_df.iloc[1]['fear_greed_index']:.2f}" if len(market_df) > 1 else None
        )
    
    # Market trends chart
    st.subheader("Market Trends")
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=pd.to_datetime(market_df['date']),
        y=market_df['avg_coherence'],
        mode='lines+markers',
        name='Avg Coherence',
        line=dict(color='blue', width=2),
        yaxis='y'
    ))
    
    fig.add_trace(go.Scatter(
        x=pd.to_datetime(market_df['date']),
        y=market_df['avg_truth_cost'],
        mode='lines+markers',
        name='Avg Truth Cost',
        line=dict(color='red', width=2),
        yaxis='y2'
    ))
    
    fig.update_layout(
        title="Market Coherence vs Truth Cost",
        xaxis_title="Date",
        yaxis=dict(
            title="Coherence",
            side='left',
            range=[0, 1]
        ),
        yaxis2=dict(
            title="Truth Cost",
            side='right',
            overlaying='y',
            range=[0, 0.5]
        ),
        hovermode='x unified',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Stock Analysis Section
st.header("📊 Individual Stock Analysis")

# Stock selector
stocks = db.conn.execute(
    "SELECT DISTINCT symbol FROM stock_data ORDER BY symbol"
).fetchall()
stock_symbols = [row[0] for row in stocks]

selected_stocks = st.multiselect(
    "Select stocks to analyze",
    stock_symbols,
    default=stock_symbols[:5] if len(stock_symbols) > 5 else stock_symbols
)

if selected_stocks:
    # Get coherence trends
    coherence_trends = db.get_coherence_trends(
        selected_stocks, 
        days=(date.today() - start_date).days
    )
    
    # Coherence comparison chart
    st.subheader("Coherence Comparison")
    
    fig = go.Figure()
    
    for symbol, data in coherence_trends.items():
        df = pd.DataFrame(data)
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(df['date']),
            y=df['coherence'],
            mode='lines+markers',
            name=symbol,
            line=dict(width=2)
        ))
    
    fig.update_layout(
        title="Stock Coherence Trends",
        xaxis_title="Date",
        yaxis_title="Coherence",
        yaxis=dict(range=[0, 1]),
        hovermode='x unified',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Truth Cost comparison
    st.subheader("Truth Cost Analysis")
    
    fig2 = go.Figure()
    
    for symbol, data in coherence_trends.items():
        df = pd.DataFrame(data)
        fig2.add_trace(go.Scatter(
            x=pd.to_datetime(df['date']),
            y=df['truth_cost'],
            mode='lines+markers',
            name=symbol,
            line=dict(width=2)
        ))
    
    fig2.update_layout(
        title="Stock Truth Cost Trends",
        xaxis_title="Date",
        yaxis_title="Truth Cost",
        yaxis=dict(range=[0, 0.6]),
        hovermode='x unified',
        height=400
    )
    
    st.plotly_chart(fig2, use_container_width=True)

# Top Movers Section
st.header("🎯 Today's Top Movers")

today_movers = db.get_top_movers(date.today(), limit=10)

col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Top Gainers")
    if today_movers['gainers']:
        gainers_df = pd.DataFrame(today_movers['gainers'])
        gainers_df = gainers_df[['symbol', 'price', 'day_change_pct', 'coherence', 'truth_cost']]
        st.dataframe(
            gainers_df.style.format({
                'price': '${:.2f}',
                'day_change_pct': '{:+.2f}%',
                'coherence': '{:.3f}',
                'truth_cost': '{:.3f}'
            }),
            use_container_width=True
        )

with col2:
    st.subheader("📉 Top Losers")
    if today_movers['losers']:
        losers_df = pd.DataFrame(today_movers['losers'])
        losers_df = losers_df[['symbol', 'price', 'day_change_pct', 'coherence', 'truth_cost']]
        st.dataframe(
            losers_df.style.format({
                'price': '${:.2f}',
                'day_change_pct': '{:+.2f}%',
                'coherence': '{:.3f}',
                'truth_cost': '{:.3f}'
            }),
            use_container_width=True
        )

# Options Signals Section
st.header("🎯 Options Signals")

# Get recent options signals
options_query = """
SELECT DISTINCT date, symbol, signal_type, signal_strength, price, coherence, day_change_pct, month_change_pct
FROM options_signals 
WHERE date >= date('now', '-7 days')
ORDER BY date DESC, signal_strength DESC
"""

options_signals = pd.read_sql_query(options_query, db.conn)

if not options_signals.empty:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🟢 Call Signals")
        calls = options_signals[options_signals['signal_type'] == 'CALL']
        if not calls.empty:
            st.dataframe(
                calls[['symbol', 'signal_strength', 'price', 'coherence', 'day_change_pct']].style.format({
                    'price': '${:.2f}',
                    'coherence': '{:.3f}',
                    'day_change_pct': '{:+.2f}%'
                }),
                use_container_width=True
            )
    
    with col2:
        st.subheader("🔴 Put Signals")
        puts = options_signals[options_signals['signal_type'] == 'PUT']
        if not puts.empty:
            st.dataframe(
                puts[['symbol', 'signal_strength', 'price', 'coherence', 'day_change_pct']].style.format({
                    'price': '${:.2f}',
                    'coherence': '{:.3f}',
                    'day_change_pct': '{:+.2f}%'
                }),
                use_container_width=True
            )

# Recent Alerts
st.header("🚨 Recent Alerts")

alerts_query = """
SELECT * FROM alerts 
WHERE date >= date('now', '-1 days')
ORDER BY timestamp DESC
LIMIT 20
"""

recent_alerts = pd.read_sql_query(alerts_query, db.conn)

if not recent_alerts.empty:
    for _, alert in recent_alerts.iterrows():
        severity_color = {
            'CRITICAL': '🔴',
            'WARNING': '🟡',
            'INFO': '🔵'
        }.get(alert['severity'], '⚪')
        
        st.markdown(f"{severity_color} **{alert['symbol']}** - {alert['message']} ({alert['timestamp']})")

# Database Stats
st.sidebar.header("📊 Database Statistics")

stats = {
    'Total Records': db.conn.execute("SELECT COUNT(*) FROM stock_data").fetchone()[0],
    'Unique Stocks': db.conn.execute("SELECT COUNT(DISTINCT symbol) FROM stock_data").fetchone()[0],
    'Days of Data': db.conn.execute("SELECT COUNT(DISTINCT date) FROM market_summary").fetchone()[0],
    'Total Alerts': db.conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0],
}

for key, value in stats.items():
    st.sidebar.metric(key, f"{value:,}")

# Auto-refresh
st.sidebar.markdown("---")
if st.sidebar.button("🔄 Refresh Data"):
    st.rerun()

st.sidebar.markdown("Dashboard auto-refreshes every 2 minutes")

# Footer
st.markdown("---")
st.markdown("🔧 GCT Market Sentiment Analysis System | Data updates continuously during market hours")