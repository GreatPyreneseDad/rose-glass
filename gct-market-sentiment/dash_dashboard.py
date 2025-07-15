#!/usr/bin/env python3
"""
Alternative dashboard using Plotly Dash - more stable than Streamlit
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dash import Dash, html, dcc, Input, Output
import dash_bootstrap_components as dbc
from datetime import datetime
import os
import glob

# Initialize Dash app with Bootstrap theme
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

def load_latest_coherence_data():
    """Load most recent coherence calculation"""
    pattern = "data/raw/coherence_calculated_*.csv"
    files = glob.glob(pattern)
    if not files:
        return pd.DataFrame()
    latest = sorted(files)[-1]
    return pd.read_csv(latest)

def load_ticker_history(ticker):
    """Load historical data for a ticker"""
    ticker_dir = f"data/tickers/{ticker}"
    if not os.path.exists(ticker_dir):
        return pd.DataFrame()
    
    files = [f for f in os.listdir(ticker_dir) if f.endswith('.csv')]
    if not files:
        return pd.DataFrame()
    
    latest_file = sorted(files)[-1]
    df = pd.read_csv(os.path.join(ticker_dir, latest_file))
    df['date'] = pd.to_datetime(df['date'])
    return df

def create_coherence_gauge(value, title):
    """Create gauge chart for coherence"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title},
        delta={'reference': 0.5},
        gauge={
            'axis': {'range': [None, 1]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 0.3], 'color': "lightgray"},
                {'range': [0.3, 0.7], 'color': "gray"}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 0.9}}))
    fig.update_layout(height=300)
    return fig

# Define layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("GCT Market Coherence Dashboard", className="text-center mb-4"),
            html.P(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                   className="text-center text-muted")
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='coherence-bar-chart'),
        ], width=6),
        dbc.Col([
            dcc.Graph(id='coherence-scatter'),
        ], width=6),
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dcc.Dropdown(
                id='ticker-dropdown',
                options=[],
                value='NVDA',
                className="mb-3"
            ),
            dcc.Graph(id='price-chart'),
        ], width=8),
        dbc.Col([
            dcc.Graph(id='coherence-gauge'),
        ], width=4),
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            html.H3("Coherence Data Table"),
            html.Div(id='data-table')
        ])
    ]),
    
    # Auto-refresh every 30 seconds
    dcc.Interval(
        id='interval-component',
        interval=30*1000,  # in milliseconds
        n_intervals=0
    )
], fluid=True)

@app.callback(
    [Output('coherence-bar-chart', 'figure'),
     Output('coherence-scatter', 'figure'),
     Output('ticker-dropdown', 'options'),
     Output('data-table', 'children')],
    [Input('interval-component', 'n_intervals')]
)
def update_overview(n):
    """Update overview charts"""
    df = load_latest_coherence_data()
    
    if df.empty:
        return {}, {}, [], "No data available"
    
    # Bar chart of top coherence
    top_10 = df.nlargest(10, 'coherence')
    bar_fig = px.bar(top_10, x='coherence', y='ticker', orientation='h',
                     title='Top 10 Coherent Stocks',
                     color='coherence', color_continuous_scale='Viridis')
    bar_fig.update_layout(height=400)
    
    # Scatter plot of components
    scatter_fig = px.scatter(df, x='psi', y='rho', size='coherence', 
                           color='coherence', hover_data=['ticker'],
                           title='Coherence Components (ψ vs ρ)',
                           labels={'psi': 'ψ (Clarity)', 'rho': 'ρ (Wisdom)'},
                           color_continuous_scale='Viridis')
    scatter_fig.update_layout(height=400)
    
    # Dropdown options
    options = [{'label': ticker, 'value': ticker} for ticker in df['ticker']]
    
    # Data table
    table = dbc.Table.from_dataframe(
        df.round(3), 
        striped=True, 
        bordered=True, 
        hover=True,
        responsive=True,
        className="mt-3"
    )
    
    return bar_fig, scatter_fig, options, table

@app.callback(
    [Output('price-chart', 'figure'),
     Output('coherence-gauge', 'figure')],
    [Input('ticker-dropdown', 'value'),
     Input('interval-component', 'n_intervals')]
)
def update_ticker_charts(ticker, n):
    """Update individual ticker charts"""
    if not ticker:
        return {}, {}
    
    # Price chart
    df = load_ticker_history(ticker)
    if df.empty:
        price_fig = go.Figure()
        price_fig.add_annotation(text="No data available", 
                               xref="paper", yref="paper",
                               x=0.5, y=0.5, showarrow=False)
    else:
        price_fig = go.Figure()
        price_fig.add_trace(go.Candlestick(
            x=df['date'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price'
        ))
        price_fig.update_layout(
            title=f'{ticker} Price Chart',
            yaxis_title='Price ($)',
            xaxis_title='Date',
            height=400
        )
    
    # Coherence gauge
    coherence_df = load_latest_coherence_data()
    ticker_data = coherence_df[coherence_df['ticker'] == ticker]
    
    if ticker_data.empty:
        gauge_fig = create_coherence_gauge(0, f"{ticker} Coherence")
    else:
        coherence_value = ticker_data.iloc[0]['coherence']
        gauge_fig = create_coherence_gauge(coherence_value, f"{ticker} Coherence")
    
    return price_fig, gauge_fig

if __name__ == '__main__':
    print("Starting Dash dashboard on http://localhost:8050")
    print("This is more stable than Streamlit - refresh page if needed")
    app.run_server(debug=False, host='127.0.0.1', port=8050)