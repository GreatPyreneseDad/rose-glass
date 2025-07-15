#!/usr/bin/env python3
"""
Analyze coherence for any stock ticker
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import os

# Use environment variable for API key
TIINGO_API_KEY = os.environ.get('TIINGO_API_KEY', 'ef1915ca2231c0c953e4fb7b72dec74bc767d9d1')

def fetch_stock_data(ticker, days=30):
    """Fetch stock data using yfinance (free, no API key needed)"""
    print(f"Fetching data for {ticker}...")
    
    try:
        # Download data
        stock = yf.Ticker(ticker)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        df = stock.history(start=start_date, end=end_date)
        
        if df.empty:
            print(f"No data found for {ticker}")
            return None
            
        # Rename columns to match our format
        df = df.reset_index()
        df.columns = [col.lower() for col in df.columns]
        
        return df
        
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return None

def calculate_coherence(df, ticker):
    """Calculate GCT coherence metrics"""
    if len(df) < 10:
        return None
        
    # Calculate returns
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
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    current_rsi = rsi.iloc[-1]
    
    # Calculate coherence components
    psi = min(1.0, abs(trend_strength) * 10)  # Clarity
    rho = 1 / (1 + volatility / 20)  # Reflection (inverse of volatility)
    q_raw = min(1.0, abs(momentum) * 50)  # Emotion
    f = min(1.0, volume_ratio * 0.5)  # Social signal
    
    # Enhanced coherence formula
    coherence = (psi * 0.3 + rho * 0.3 + q_raw * 0.2 + f * 0.2)
    
    # Adjust for extreme RSI
    if current_rsi > 70 or current_rsi < 30:
        coherence *= 0.9  # Reduce coherence at extremes
    
    return {
        'ticker': ticker.upper(),
        'coherence': round(coherence, 3),
        'psi': round(psi, 3),
        'rho': round(rho, 3),
        'q_raw': round(q_raw, 3),
        'f': round(f, 3),
        'trend_strength': round(trend_strength, 4),
        'volatility': round(volatility, 2),
        'momentum': round(momentum, 4),
        'rsi': round(current_rsi, 1),
        'last_price': round(df['close'].iloc[-1], 2),
        'volume_ratio': round(volume_ratio, 2),
        'price_change_pct': round((df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0] * 100, 2)
    }

def analyze_pattern(metrics):
    """Analyze coherence pattern and provide insights"""
    c = metrics['coherence']
    trend = metrics['trend_strength']
    vol = metrics['volatility']
    rsi = metrics['rsi']
    
    insights = []
    
    # Overall coherence assessment
    if c > 0.7:
        insights.append("🟢 VERY HIGH COHERENCE: Strong unified market movement")
    elif c > 0.5:
        insights.append("🟢 HIGH COHERENCE: Clear directional movement")
    elif c > 0.3:
        insights.append("🟡 MODERATE COHERENCE: Mixed signals")
    else:
        insights.append("🔴 LOW COHERENCE: Chaotic/uncertain movement")
    
    # Trend analysis
    if trend > 0.05:
        insights.append("📈 Strong upward trend")
    elif trend > 0.02:
        insights.append("📈 Moderate upward trend")
    elif trend < -0.05:
        insights.append("📉 Strong downward trend")
    elif trend < -0.02:
        insights.append("📉 Moderate downward trend")
    else:
        insights.append("➡️ Sideways movement")
    
    # Volatility assessment
    if vol > 50:
        insights.append("⚡ Very high volatility - extreme price swings")
    elif vol > 30:
        insights.append("⚡ High volatility - significant price movement")
    elif vol < 15:
        insights.append("😴 Low volatility - stable price action")
    
    # RSI signals
    if rsi > 70:
        insights.append("🔥 Overbought (RSI > 70) - potential reversal")
    elif rsi < 30:
        insights.append("❄️ Oversold (RSI < 30) - potential bounce")
    
    # Pattern detection
    if c > 0.5 and trend > 0.02 and vol < 30:
        insights.append("✅ BULLISH COHERENCE PATTERN: Strong buy signal")
    elif c < 0.3 and vol > 40:
        insights.append("⚠️ FALLING COHERENCE: High risk, potential breakdown")
    elif metrics['volume_ratio'] > 2 and abs(trend) > 0.03:
        insights.append("🔊 High volume surge - significant move underway")
    
    return insights

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_any_stock.py TICKER [TICKER2 TICKER3 ...]")
        print("Example: python analyze_any_stock.py TSLA GME AMC")
        return
    
    # Check for yfinance
    try:
        import yfinance
    except ImportError:
        print("Installing yfinance for free stock data access...")
        os.system(f"{sys.executable} -m pip install yfinance")
        import yfinance
    
    tickers = sys.argv[1:]
    results = []
    
    print("\n=== GCT Coherence Analysis ===\n")
    
    for ticker in tickers:
        df = fetch_stock_data(ticker.upper())
        if df is not None:
            metrics = calculate_coherence(df, ticker)
            if metrics:
                results.append(metrics)
                
                # Display results
                print(f"\n{'='*60}")
                print(f"TICKER: {metrics['ticker']}")
                print(f"{'='*60}")
                print(f"Price: ${metrics['last_price']} ({metrics['price_change_pct']:+.2f}% period change)")
                print(f"Coherence Score: {metrics['coherence']} {'█' * int(metrics['coherence'] * 20)}")
                print(f"\nComponents:")
                print(f"  ψ (Clarity):     {metrics['psi']:.3f}")
                print(f"  ρ (Wisdom):      {metrics['rho']:.3f}")
                print(f"  q (Emotion):     {metrics['q_raw']:.3f}")
                print(f"  f (Social):      {metrics['f']:.3f}")
                print(f"\nMetrics:")
                print(f"  Trend Strength:  {metrics['trend_strength']:+.2%}")
                print(f"  Volatility:      {metrics['volatility']:.1f}%")
                print(f"  RSI:             {metrics['rsi']:.1f}")
                print(f"  Volume Ratio:    {metrics['volume_ratio']:.2f}x")
                
                # Pattern insights
                insights = analyze_pattern(metrics)
                print(f"\nInsights:")
                for insight in insights:
                    print(f"  {insight}")
    
    # Save results
    if results:
        df_results = pd.DataFrame(results)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"custom_analysis_{timestamp}.csv"
        df_results.to_csv(output_file, index=False)
        print(f"\n\nResults saved to: {output_file}")
        
        # Comparison summary if multiple tickers
        if len(results) > 1:
            print("\n\n=== COMPARISON SUMMARY ===")
            df_results = df_results.sort_values('coherence', ascending=False)
            print("\nRanked by Coherence:")
            for idx, row in df_results.iterrows():
                bar = '█' * int(row['coherence'] * 20)
                print(f"{row['ticker']:6} {row['coherence']:.3f} {bar}")

if __name__ == "__main__":
    main()