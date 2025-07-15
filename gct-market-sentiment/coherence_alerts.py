#!/usr/bin/env python3
"""
Generate coherence alerts based on price patterns and market dynamics
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

API_KEY = "ef1915ca2231c0c953e4fb7b72dec74bc767d9d1"

# Market sectors and their tickers
MARKET_SECTORS = {
    'tech': ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META'],
    'finance': ['JPM', 'BAC', 'GS', 'WFC', 'C'],
    'energy': ['XOM', 'CVX', 'COP', 'SLB', 'OXY'],
    'healthcare': ['JNJ', 'PFE', 'UNH', 'CVS', 'ABBV'],
    'indices': ['SPY', 'QQQ', 'DIA', 'IWM', 'VTI']
}

def fetch_price_data(ticker, days=30):
    """Fetch historical price data for a ticker"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    url = f"https://api.tiingo.com/tiingo/daily/{ticker}/prices"
    params = {
        'token': API_KEY,
        'startDate': start_date.strftime('%Y-%m-%d'),
        'endDate': end_date.strftime('%Y-%m-%d')
    }
    
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            return pd.DataFrame(response.json())
        return None
    except:
        return None

def calculate_coherence_metrics(df):
    """Calculate GCT coherence metrics from price data"""
    if df is None or len(df) < 5:
        return None
    
    # Calculate returns and volatility
    df['returns'] = df['close'].pct_change()
    df['volume_norm'] = df['volume'] / df['volume'].rolling(20).mean()
    
    # ψ (Psi) - Price clarity/trend strength
    # Strong trends have clear direction
    sma_5 = df['close'].rolling(5).mean()
    sma_20 = df['close'].rolling(20).mean()
    df['trend_clarity'] = (sma_5 - sma_20) / sma_20
    psi = abs(df['trend_clarity'].iloc[-1]) * 10  # Scale to 0-1
    psi = min(1.0, psi)
    
    # ρ (Rho) - Market reflection/mean reversion
    # How much the price reflects around its mean
    price_std = df['close'].rolling(20).std()
    price_mean = df['close'].rolling(20).mean()
    df['z_score'] = (df['close'] - price_mean) / price_std
    rho = 1 / (1 + abs(df['z_score'].iloc[-1]))  # Higher when close to mean
    
    # q (Emotion) - Volatility and volume spikes
    # High emotion = high volatility + high volume
    vol_percentile = df['returns'].rolling(20).std().rank(pct=True).iloc[-1]
    volume_spike = df['volume_norm'].iloc[-1]
    q_raw = (vol_percentile + min(2, volume_spike)) / 3
    
    # f (Social) - Correlation with market/sector
    # Implemented later with sector data
    f = 0.5  # Default
    
    # Calculate coherence
    coherence = psi * 0.3 + rho * 0.2 + q_raw * 0.3 + f * 0.2
    
    # Calculate derivatives (momentum)
    if len(df) >= 10:
        recent_returns = df['returns'].tail(5).mean()
        older_returns = df['returns'].tail(10).head(5).mean()
        momentum = recent_returns - older_returns
    else:
        momentum = 0
    
    return {
        'psi': round(psi, 3),
        'rho': round(rho, 3),
        'q_raw': round(q_raw, 3),
        'f': round(f, 3),
        'coherence': round(coherence, 3),
        'momentum': round(momentum, 4),
        'last_price': df['close'].iloc[-1],
        'volume_spike': round(volume_spike, 2),
        'volatility': round(df['returns'].std() * np.sqrt(252) * 100, 2)  # Annualized
    }

def detect_coherence_patterns(ticker, metrics, df):
    """Detect building or falling coherence patterns"""
    alerts = []
    
    # Calculate historical coherence
    coherence_history = []
    for i in range(5, len(df)):
        window_df = df.iloc[:i]
        hist_metrics = calculate_coherence_metrics(window_df)
        if hist_metrics:
            coherence_history.append(hist_metrics['coherence'])
    
    if len(coherence_history) < 3:
        return alerts
    
    # Detect patterns
    current_coherence = metrics['coherence']
    recent_coherence = np.mean(coherence_history[-3:])
    older_coherence = np.mean(coherence_history[-6:-3]) if len(coherence_history) >= 6 else recent_coherence
    
    coherence_change = current_coherence - recent_coherence
    coherence_acceleration = (current_coherence - recent_coherence) - (recent_coherence - older_coherence)
    
    # Building Coherence Patterns
    if coherence_change > 0.05:
        if metrics['psi'] > 0.7 and metrics['momentum'] > 0:
            alerts.append({
                'type': 'BUILDING_COHERENCE',
                'strength': 'STRONG',
                'ticker': ticker,
                'message': f"Strong upward coherence building - Clear trend emerging with positive momentum",
                'metrics': metrics,
                'coherence_change': round(coherence_change, 3)
            })
        elif metrics['volume_spike'] > 1.5:
            alerts.append({
                'type': 'BUILDING_COHERENCE',
                'strength': 'MODERATE',
                'ticker': ticker,
                'message': f"Volume-driven coherence increase - Market attention focusing",
                'metrics': metrics,
                'coherence_change': round(coherence_change, 3)
            })
    
    # Falling Coherence Patterns
    if coherence_change < -0.05:
        if metrics['q_raw'] > 0.7 and metrics['volatility'] > 20:
            alerts.append({
                'type': 'FALLING_COHERENCE',
                'strength': 'STRONG',
                'ticker': ticker,
                'message': f"Rapid coherence breakdown - High emotion and volatility",
                'metrics': metrics,
                'coherence_change': round(coherence_change, 3)
            })
        elif metrics['psi'] < 0.3:
            alerts.append({
                'type': 'FALLING_COHERENCE',
                'strength': 'MODERATE',
                'ticker': ticker,
                'message': f"Trend clarity lost - Direction becoming uncertain",
                'metrics': metrics,
                'coherence_change': round(coherence_change, 3)
            })
    
    # Acceleration Patterns
    if abs(coherence_acceleration) > 0.02:
        if coherence_acceleration > 0:
            alerts.append({
                'type': 'COHERENCE_ACCELERATION',
                'strength': 'SIGNAL',
                'ticker': ticker,
                'message': f"Coherence building accelerating - Potential breakout",
                'metrics': metrics,
                'acceleration': round(coherence_acceleration, 4)
            })
        else:
            alerts.append({
                'type': 'COHERENCE_DECELERATION',
                'strength': 'WARNING',
                'ticker': ticker,
                'message': f"Coherence decline accelerating - Breakdown risk",
                'metrics': metrics,
                'acceleration': round(coherence_acceleration, 4)
            })
    
    return alerts

def analyze_sector_coherence(sector_name, tickers):
    """Analyze coherence across a sector"""
    sector_data = []
    sector_alerts = []
    
    for ticker in tickers:
        print(f"  Analyzing {ticker}...")
        df = fetch_price_data(ticker)
        
        if df is not None and len(df) > 5:
            metrics = calculate_coherence_metrics(df)
            if metrics:
                sector_data.append({
                    'ticker': ticker,
                    'metrics': metrics
                })
                
                # Detect patterns
                alerts = detect_coherence_patterns(ticker, metrics, df)
                sector_alerts.extend(alerts)
    
    # Calculate sector-wide coherence
    if sector_data:
        avg_coherence = np.mean([d['metrics']['coherence'] for d in sector_data])
        coherence_dispersion = np.std([d['metrics']['coherence'] for d in sector_data])
        
        # Sector-wide alerts
        if coherence_dispersion < 0.1 and avg_coherence > 0.6:
            sector_alerts.append({
                'type': 'SECTOR_COHERENCE',
                'strength': 'STRONG',
                'sector': sector_name,
                'message': f"Strong sector-wide coherence in {sector_name} - Unified movement",
                'avg_coherence': round(avg_coherence, 3),
                'dispersion': round(coherence_dispersion, 3)
            })
        elif coherence_dispersion > 0.3:
            sector_alerts.append({
                'type': 'SECTOR_DIVERGENCE',
                'strength': 'WARNING',
                'sector': sector_name,
                'message': f"High divergence in {sector_name} - Sector breaking apart",
                'avg_coherence': round(avg_coherence, 3),
                'dispersion': round(coherence_dispersion, 3)
            })
    
    return sector_data, sector_alerts

def main():
    print("=== GCT Coherence Pattern Detection ===")
    print(f"Analyzing market coherence patterns...\n")
    
    all_alerts = []
    all_metrics = []
    
    # Analyze each sector
    for sector_name, tickers in MARKET_SECTORS.items():
        print(f"\nAnalyzing {sector_name.upper()} sector:")
        sector_data, sector_alerts = analyze_sector_coherence(sector_name, tickers)
        
        all_alerts.extend(sector_alerts)
        all_metrics.extend(sector_data)
    
    # Save results
    os.makedirs("data/raw", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save alerts
    if all_alerts:
        alerts_df = pd.DataFrame(all_alerts)
        alerts_file = f"data/raw/coherence_alerts_{timestamp}.csv"
        alerts_df.to_csv(alerts_file, index=False)
        print(f"\nSaved {len(alerts_df)} alerts to {alerts_file}")
        
        # Display critical alerts
        print("\n=== CRITICAL COHERENCE ALERTS ===")
        
        # Building coherence
        building = alerts_df[alerts_df['type'].str.contains('BUILDING')]
        if not building.empty:
            print("\n🟢 BUILDING COHERENCE:")
            for _, alert in building.iterrows():
                print(f"  [{alert['strength']}] {alert.get('ticker', alert.get('sector', 'MARKET'))}: {alert['message']}")
        
        # Falling coherence
        falling = alerts_df[alerts_df['type'].str.contains('FALLING')]
        if not falling.empty:
            print("\n🔴 FALLING COHERENCE:")
            for _, alert in falling.iterrows():
                print(f"  [{alert['strength']}] {alert.get('ticker', alert.get('sector', 'MARKET'))}: {alert['message']}")
        
        # Acceleration alerts
        accel = alerts_df[alerts_df['type'].str.contains('ACCELERATION')]
        if not accel.empty:
            print("\n⚡ ACCELERATION ALERTS:")
            for _, alert in accel.iterrows():
                print(f"  [{alert['strength']}] {alert.get('ticker', alert.get('sector', 'MARKET'))}: {alert['message']}")
    
    # Save metrics summary
    if all_metrics:
        metrics_data = []
        for item in all_metrics:
            row = {'ticker': item['ticker']}
            row.update(item['metrics'])
            metrics_data.append(row)
        
        metrics_df = pd.DataFrame(metrics_data)
        metrics_file = f"data/raw/coherence_metrics_{timestamp}.csv"
        metrics_df.to_csv(metrics_file, index=False)
        print(f"\nSaved metrics to {metrics_file}")
        
        # Show top coherent stocks
        print("\n=== TOP COHERENT STOCKS ===")
        top_coherent = metrics_df.nlargest(5, 'coherence')
        for _, row in top_coherent.iterrows():
            print(f"  {row['ticker']}: {row['coherence']:.3f} (ψ={row['psi']:.2f}, ρ={row['rho']:.2f}, q={row['q_raw']:.2f})")

if __name__ == "__main__":
    main()