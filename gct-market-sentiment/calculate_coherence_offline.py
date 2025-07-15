#!/usr/bin/env python3
"""
Calculate coherence metrics from saved ticker data
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

def calculate_coherence_from_file(ticker_dir):
    """Calculate coherence from saved ticker CSV files"""
    if not os.path.exists(ticker_dir):
        return None
        
    # Get latest file
    files = [f for f in os.listdir(ticker_dir) if f.endswith('.csv')]
    if not files:
        return None
        
    latest_file = sorted(files)[-1]
    df = pd.read_csv(os.path.join(ticker_dir, latest_file))
    
    if len(df) < 10:
        return None
        
    # Calculate metrics
    df['returns'] = df['close'].pct_change()
    
    # Trend strength (5-day vs 20-day)
    sma5 = df['close'].rolling(5).mean()
    sma20 = df['close'].rolling(20).mean() if len(df) >= 20 else df['close'].mean()
    trend_strength = ((sma5.iloc[-1] - sma20.iloc[-1]) / sma20.iloc[-1]) if sma20.iloc[-1] > 0 else 0
    
    # Volatility
    volatility = df['returns'].std() * np.sqrt(252) * 100  # Annualized
    
    # Volume activity
    recent_volume = df['volume'].tail(5).mean()
    avg_volume = df['volume'].mean()
    volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1
    
    # Price momentum
    momentum = df['returns'].tail(5).mean()
    
    # Calculate coherence components
    psi = min(1.0, abs(trend_strength) * 10)  # Clarity
    rho = 1 / (1 + volatility / 20)  # Reflection (inverse of volatility)
    q_raw = min(1.0, abs(momentum) * 50)  # Emotion
    f = min(1.0, volume_ratio * 0.5)  # Social signal
    
    # Simple coherence formula
    coherence = (psi * 0.3 + rho * 0.3 + q_raw * 0.2 + f * 0.2)
    
    return {
        'ticker': os.path.basename(ticker_dir),
        'coherence': round(coherence, 3),
        'psi': round(psi, 3),
        'rho': round(rho, 3),
        'q_raw': round(q_raw, 3),
        'f': round(f, 3),
        'trend_strength': round(trend_strength, 4),
        'volatility': round(volatility, 2),
        'momentum': round(momentum, 4),
        'last_price': round(df['close'].iloc[-1], 2),
        'volume_ratio': round(volume_ratio, 2)
    }

def main():
    print("=== Offline Coherence Calculator ===\n")
    
    ticker_dir = "data/tickers"
    if not os.path.exists(ticker_dir):
        print("No ticker data found. Run fetch_and_save_data.py first.")
        return
        
    results = []
    
    # Calculate coherence for each ticker
    for ticker in os.listdir(ticker_dir):
        ticker_path = os.path.join(ticker_dir, ticker)
        if os.path.isdir(ticker_path):
            metrics = calculate_coherence_from_file(ticker_path)
            if metrics:
                results.append(metrics)
                
    if results:
        # Sort by coherence
        results.sort(key=lambda x: x['coherence'], reverse=True)
        
        # Save results
        df = pd.DataFrame(results)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"data/raw/coherence_calculated_{timestamp}.csv"
        df.to_csv(output_file, index=False)
        print(f"Saved coherence metrics to {output_file}\n")
        
        # Display top coherent stocks
        print("Top Coherent Stocks:")
        print("-" * 80)
        print(f"{'Ticker':<8} {'Coherence':<10} {'ψ (Clarity)':<12} {'ρ (Wisdom)':<12} {'Trend':<8} {'Price':<8}")
        print("-" * 80)
        
        for r in results[:10]:
            print(f"{r['ticker']:<8} {r['coherence']:<10.3f} {r['psi']:<12.3f} "
                  f"{r['rho']:<12.3f} {r['trend_strength']:>7.2%} ${r['last_price']:>7.2f}")
            
        # Detect patterns
        print("\n\nCoherence Patterns Detected:")
        print("-" * 60)
        
        for r in results:
            if r['coherence'] > 0.7 and r['trend_strength'] > 0.02:
                print(f"🟢 BUILDING: {r['ticker']} - High coherence ({r['coherence']:.3f}) with positive trend")
            elif r['coherence'] < 0.3 and r['volatility'] > 30:
                print(f"🔴 FALLING: {r['ticker']} - Low coherence ({r['coherence']:.3f}) with high volatility")
            elif r['momentum'] > 0.01 and r['volume_ratio'] > 1.5:
                print(f"⚡ ALERT: {r['ticker']} - Strong momentum with high volume")

if __name__ == "__main__":
    main()