#!/usr/bin/env python3
"""
Simple coherence monitoring system for market data
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

API_KEY = "ef1915ca2231c0c953e4fb7b72dec74bc767d9d1"

# Key tickers to monitor
WATCH_LIST = ['AAPL', 'MSFT', 'NVDA', 'SPY', 'QQQ']

def fetch_ticker_data(ticker, days=30):
    """Fetch price data for a ticker"""
    url = f"https://api.tiingo.com/tiingo/daily/{ticker}/prices"
    params = {
        'token': API_KEY,
        'startDate': (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d'),
        'endDate': datetime.now().strftime('%Y-%m-%d')
    }
    
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            df = pd.DataFrame(response.json())
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            return df
    except:
        pass
    return None

def calculate_simple_coherence(df):
    """Calculate simplified coherence metrics"""
    if df is None or len(df) < 10:
        return None
    
    # Price momentum (5-day vs 20-day trend)
    sma5 = df['close'].rolling(5).mean()
    sma20 = df['close'].rolling(20).mean()
    trend_strength = ((sma5 - sma20) / sma20).iloc[-1]
    
    # Volatility (normalized)
    returns = df['close'].pct_change()
    recent_vol = returns.tail(5).std()
    normal_vol = returns.std()
    vol_ratio = recent_vol / (normal_vol + 0.0001)
    
    # Volume activity
    recent_volume = df['volume'].tail(5).mean()
    normal_volume = df['volume'].mean()
    volume_ratio = recent_volume / (normal_volume + 1)
    
    # Simple coherence score
    # High coherence = strong trend + normal volatility + normal volume
    coherence = 0.5 + (trend_strength * 2) - (abs(vol_ratio - 1) * 0.3) - (abs(volume_ratio - 1) * 0.2)
    coherence = max(0, min(1, coherence))  # Bound to [0,1]
    
    return {
        'coherence': float(round(coherence, 3)),
        'trend_strength': float(round(trend_strength, 4)),
        'volatility_ratio': float(round(vol_ratio, 2)),
        'volume_ratio': float(round(volume_ratio, 2)),
        'last_price': float(round(df['close'].iloc[-1], 2)),
        'price_change_5d': float(round((df['close'].iloc[-1] / df['close'].iloc[-6] - 1) * 100, 2))
    }

def detect_coherence_alerts(ticker, current, history_df):
    """Detect coherence patterns"""
    alerts = []
    
    if len(history_df) < 5:
        return alerts
    
    # Calculate coherence trend
    recent_coherence = history_df['coherence'].tail(5).mean()
    current_coherence = current['coherence']
    coherence_change = current_coherence - recent_coherence
    
    # Building coherence
    if coherence_change > 0.1 and current['trend_strength'] > 0.02:
        alerts.append({
            'type': 'BUILDING',
            'ticker': ticker,
            'message': f"{ticker} showing building coherence pattern (+{coherence_change:.3f})",
            'strength': 'HIGH' if coherence_change > 0.2 else 'MODERATE',
            'data': current
        })
    
    # Falling coherence
    elif coherence_change < -0.1 and current['volatility_ratio'] > 1.5:
        alerts.append({
            'type': 'FALLING',
            'ticker': ticker,
            'message': f"{ticker} coherence breaking down ({coherence_change:.3f})",
            'strength': 'HIGH' if coherence_change < -0.2 else 'MODERATE',
            'data': current
        })
    
    # Volume spike with coherence
    if current['volume_ratio'] > 2.0 and current['coherence'] > 0.6:
        alerts.append({
            'type': 'VOLUME_COHERENCE',
            'ticker': ticker,
            'message': f"{ticker} high volume with strong coherence",
            'strength': 'SIGNAL',
            'data': current
        })
    
    return alerts

def main():
    print("=== Simple Coherence Monitor ===")
    print(f"Monitoring: {', '.join(WATCH_LIST)}\n")
    
    all_data = []
    all_alerts = []
    
    for ticker in WATCH_LIST:
        print(f"Analyzing {ticker}...")
        
        # Fetch data
        df = fetch_ticker_data(ticker)
        if df is None:
            print(f"  Failed to fetch data")
            continue
            
        # Calculate coherence over time
        coherence_history = []
        for i in range(20, len(df)):
            window = df.iloc[:i]
            metrics = calculate_simple_coherence(window)
            if metrics:
                coherence_history.append({
                    'date': df.index[i],
                    'coherence': metrics['coherence'],
                    'trend': metrics['trend_strength']
                })
        
        if coherence_history:
            history_df = pd.DataFrame(coherence_history)
            
            # Current metrics
            current = calculate_simple_coherence(df)
            if current:
                all_data.append({
                    'ticker': ticker,
                    **current
                })
                
                # Detect alerts
                alerts = detect_coherence_alerts(ticker, current, history_df)
                all_alerts.extend(alerts)
                
                # Display current status
                print(f"  Coherence: {current['coherence']:.3f}")
                print(f"  Trend: {'+' if current['trend_strength'] > 0 else ''}{current['trend_strength']:.3%}")
                print(f"  Price: ${current['last_price']} ({'+' if current['price_change_5d'] > 0 else ''}{current['price_change_5d']}% 5d)")
    
    # Save data
    os.makedirs("data/raw", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if all_data:
        data_df = pd.DataFrame(all_data)
        data_file = f"data/raw/coherence_monitor_{timestamp}.csv"
        data_df.to_csv(data_file, index=False)
        print(f"\nSaved data to {data_file}")
    
    # Display alerts
    if all_alerts:
        print("\n=== COHERENCE ALERTS ===")
        for alert in all_alerts:
            icon = "🟢" if alert['type'] == 'BUILDING' else "🔴" if alert['type'] == 'FALLING' else "⚡"
            print(f"{icon} [{alert['strength']}] {alert['message']}")
            print(f"   Coherence: {alert['data']['coherence']:.3f}, Trend: {alert['data']['trend_strength']:.3%}")
        
        # Save alerts
        alerts_df = pd.DataFrame(all_alerts)
        alerts_file = f"data/raw/coherence_alerts_{timestamp}.csv"
        alerts_df.to_csv(alerts_file, index=False)
        print(f"\nSaved {len(alerts_df)} alerts to {alerts_file}")
    else:
        print("\n✓ No significant coherence patterns detected")

if __name__ == "__main__":
    main()