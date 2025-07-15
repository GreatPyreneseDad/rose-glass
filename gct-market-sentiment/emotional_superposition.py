#!/usr/bin/env python3
"""
Emotional Superposition Analyzer
Measures the quantum-like emotional states of stocks based on GCT principles
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import sys
from scipy import signal
from scipy.stats import entropy
import json

def fetch_stock_data(ticker, days=30):
    """Fetch stock data for analysis"""
    try:
        stock = yf.Ticker(ticker)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        df = stock.history(start=start_date, end=end_date, interval='1h')
        if df.empty:
            return None
            
        df = df.reset_index()
        df.columns = [col.lower() for col in df.columns]
        return df
    except:
        return None

def calculate_emotional_states(df, ticker):
    """
    Calculate emotional superposition states:
    - Fear/Greed spectrum
    - Hope/Despair dynamics
    - Euphoria/Panic detection
    - Emotional coherence/decoherence
    """
    
    # Basic calculations
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    
    # 1. Fear/Greed Index (0-1 scale)
    # Based on momentum, volatility, and volume
    df['momentum'] = df['returns'].rolling(12).mean()
    df['volatility'] = df['returns'].rolling(24).std()
    df['volume_surge'] = df['volume'] / df['volume'].rolling(24).mean()
    
    # Greed increases with positive momentum and volume
    df['greed_component'] = (df['momentum'] + 0.1) / 0.2  # Normalize
    df['fear_component'] = df['volatility'] * 10  # High volatility = fear
    
    # RSI for overbought/oversold
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Combine into fear/greed
    df['fear_greed'] = (
        df['greed_component'].clip(0, 1) * 0.3 +
        (df['rsi'] / 100) * 0.3 +
        (1 - df['fear_component'].clip(0, 1)) * 0.4
    ).clip(0, 1)
    
    # 2. Hope/Despair Dynamics
    # Hope: Positive momentum after decline
    # Despair: Negative momentum after rise
    df['price_ma'] = df['close'].rolling(24).mean()
    df['distance_from_ma'] = (df['close'] - df['price_ma']) / df['price_ma']
    
    df['hope'] = np.where(
        (df['distance_from_ma'] < -0.05) & (df['momentum'] > 0),
        df['momentum'] * 5,
        0
    ).clip(0, 1)
    
    df['despair'] = np.where(
        (df['distance_from_ma'] > 0.05) & (df['momentum'] < 0),
        abs(df['momentum']) * 5,
        0
    ).clip(0, 1)
    
    # 3. Euphoria/Panic Detection
    # Euphoria: Extreme greed + low volatility
    # Panic: Extreme fear + high volatility
    df['euphoria'] = np.where(
        (df['fear_greed'] > 0.8) & (df['volatility'] < df['volatility'].quantile(0.3)),
        df['fear_greed'],
        0
    )
    
    df['panic'] = np.where(
        (df['fear_greed'] < 0.2) & (df['volatility'] > df['volatility'].quantile(0.7)),
        1 - df['fear_greed'],
        0
    )
    
    # 4. Emotional Coherence/Decoherence
    # Measure how aligned emotions are with price action
    df['emotion_price_correlation'] = df['fear_greed'].rolling(24).corr(df['returns'])
    df['emotional_coherence'] = (df['emotion_price_correlation'] + 1) / 2  # Normalize to 0-1
    
    # 5. Quantum Emotional Superposition
    # Multiple emotional states existing simultaneously
    states = ['fear_greed', 'hope', 'despair', 'euphoria', 'panic']
    
    # Calculate probability amplitudes for each state
    df['total_emotion'] = df[states].sum(axis=1).clip(0.1, 5)  # Avoid division by zero
    
    for state in states:
        df[f'{state}_amplitude'] = df[state] / df['total_emotion']
    
    # Emotional entropy (uncertainty in emotional state)
    df['emotional_entropy'] = df[[f'{s}_amplitude' for s in states]].apply(
        lambda x: entropy(x.values + 0.001), axis=1  # Add small value to avoid log(0)
    )
    
    # 6. Emotional Wave Function Collapse
    # When one emotion dominates (>70% amplitude)
    df['dominant_emotion'] = df[[f'{s}_amplitude' for s in states]].idxmax(axis=1)
    df['max_amplitude'] = df[[f'{s}_amplitude' for s in states]].max(axis=1)
    df['collapsed_state'] = df['max_amplitude'] > 0.7
    
    # 7. Emotional Momentum (rate of change)
    df['emotional_velocity'] = df['fear_greed'].diff()
    df['emotional_acceleration'] = df['emotional_velocity'].diff()
    
    return df

def analyze_superposition(df, ticker):
    """Analyze the emotional superposition state"""
    if len(df) < 24:
        return None
    
    recent = df.tail(24).copy()  # Last 24 hours
    current = df.iloc[-1]
    
    analysis = {
        'ticker': ticker,
        'timestamp': datetime.now().isoformat(),
        
        # Current emotional readings
        'fear_greed_index': float(current['fear_greed']),
        'hope_level': float(current['hope']),
        'despair_level': float(current['despair']),
        'euphoria_level': float(current['euphoria']),
        'panic_level': float(current['panic']),
        
        # Superposition state
        'emotional_entropy': float(current['emotional_entropy']),
        'emotional_coherence': float(current['emotional_coherence']) if not pd.isna(current['emotional_coherence']) else 0.5,
        'collapsed_state': bool(current['collapsed_state']),
        'dominant_emotion': current['dominant_emotion'].replace('_amplitude', '') if current['collapsed_state'] else 'superposition',
        
        # Dynamics
        'emotional_velocity': float(current['emotional_velocity']) if not pd.isna(current['emotional_velocity']) else 0,
        'emotional_acceleration': float(current['emotional_acceleration']) if not pd.isna(current['emotional_acceleration']) else 0,
        
        # Statistical measures
        'avg_fear_greed_24h': float(recent['fear_greed'].mean()),
        'volatility_24h': float(recent['volatility'].mean()),
        'price_change_24h': float((current['close'] - recent['close'].iloc[0]) / recent['close'].iloc[0]),
        
        # Amplitudes for quantum state
        'amplitudes': {
            'fear_greed': float(current['fear_greed_amplitude']),
            'hope': float(current['hope_amplitude']),
            'despair': float(current['despair_amplitude']),
            'euphoria': float(current['euphoria_amplitude']),
            'panic': float(current['panic_amplitude'])
        }
    }
    
    # Pattern detection
    patterns = []
    
    # Extreme states
    if analysis['fear_greed_index'] > 0.85:
        patterns.append("🔥 EXTREME GREED: Market euphoria detected")
    elif analysis['fear_greed_index'] < 0.15:
        patterns.append("🧊 EXTREME FEAR: Market panic detected")
    
    # Emotional transitions
    if analysis['emotional_velocity'] > 0.1:
        patterns.append("📈 RISING SENTIMENT: Shifting toward greed")
    elif analysis['emotional_velocity'] < -0.1:
        patterns.append("📉 FALLING SENTIMENT: Shifting toward fear")
    
    # Superposition states
    if analysis['emotional_entropy'] > 1.5:
        patterns.append("🌀 HIGH UNCERTAINTY: Multiple emotional states active")
    elif analysis['collapsed_state']:
        patterns.append(f"🎯 COLLAPSED STATE: {analysis['dominant_emotion'].upper()} dominates")
    
    # Coherence warnings
    if analysis['emotional_coherence'] < 0.3:
        patterns.append("⚠️ DECOHERENCE: Emotions disconnected from price")
    elif analysis['emotional_coherence'] > 0.7:
        patterns.append("✅ HIGH COHERENCE: Emotions aligned with price")
    
    # Hope/Despair dynamics
    if analysis['hope_level'] > 0.5:
        patterns.append("🌅 HOPE RISING: Recovery sentiment building")
    elif analysis['despair_level'] > 0.5:
        patterns.append("🌑 DESPAIR SETTING: Capitulation risk")
    
    analysis['patterns'] = patterns
    
    return analysis

def create_superposition_report(analyses):
    """Create comprehensive emotional superposition report"""
    report = {
        'timestamp': datetime.now().isoformat(),
        'market_summary': {},
        'individual_analyses': analyses,
        'alerts': []
    }
    
    if analyses:
        # Market-wide metrics
        avg_fear_greed = np.mean([a['fear_greed_index'] for a in analyses])
        avg_entropy = np.mean([a['emotional_entropy'] for a in analyses])
        panic_count = len([a for a in analyses if a['panic_level'] > 0.5])
        euphoria_count = len([a for a in analyses if a['euphoria_level'] > 0.5])
        
        report['market_summary'] = {
            'avg_fear_greed': avg_fear_greed,
            'avg_entropy': avg_entropy,
            'panic_stocks': panic_count,
            'euphoric_stocks': euphoria_count,
            'market_emotion': 'GREED' if avg_fear_greed > 0.6 else 'FEAR' if avg_fear_greed < 0.4 else 'NEUTRAL'
        }
        
        # Generate alerts
        for analysis in analyses:
            if analysis['fear_greed_index'] > 0.9:
                report['alerts'].append({
                    'type': 'EXTREME_GREED',
                    'ticker': analysis['ticker'],
                    'message': f"{analysis['ticker']}: Extreme greed detected ({analysis['fear_greed_index']:.2f})",
                    'severity': 'high'
                })
            
            if analysis['panic_level'] > 0.7:
                report['alerts'].append({
                    'type': 'PANIC',
                    'ticker': analysis['ticker'],
                    'message': f"{analysis['ticker']}: Panic selling detected",
                    'severity': 'critical'
                })
            
            if analysis['emotional_entropy'] > 1.8:
                report['alerts'].append({
                    'type': 'UNCERTAINTY',
                    'ticker': analysis['ticker'],
                    'message': f"{analysis['ticker']}: Extreme emotional uncertainty",
                    'severity': 'medium'
                })
    
    return report

def main():
    if len(sys.argv) < 2:
        print("Usage: python emotional_superposition.py TICKER [TICKER2 ...]")
        return
    
    tickers = [t.upper() for t in sys.argv[1:]]
    analyses = []
    
    print("\n=== EMOTIONAL SUPERPOSITION ANALYSIS ===\n")
    
    for ticker in tickers:
        print(f"Analyzing {ticker}...")
        
        df = fetch_stock_data(ticker, days=7)  # 7 days of hourly data
        if df is None:
            print(f"  Failed to fetch data for {ticker}")
            continue
        
        df = calculate_emotional_states(df, ticker)
        analysis = analyze_superposition(df, ticker)
        
        if analysis:
            analyses.append(analysis)
            
            # Display results
            print(f"\n{ticker} Emotional State:")
            fg_index = analysis['fear_greed_index']
            if pd.isna(fg_index):
                print(f"  Fear/Greed Index: N/A (insufficient data)")
            else:
                print(f"  Fear/Greed Index: {fg_index:.2f} {'🔥' if fg_index > 0.7 else '🧊' if fg_index < 0.3 else '😐'}")
            
            entropy = analysis['emotional_entropy']
            print(f"  Emotional Entropy: {entropy:.2f if not pd.isna(entropy) else 'N/A'}")
            print(f"  Dominant State: {analysis['dominant_emotion']}")
            print(f"  Coherence: {analysis['emotional_coherence']:.2f}")
            
            print(f"\n  Quantum Amplitudes:")
            for state, amp in analysis['amplitudes'].items():
                amp_value = amp if not pd.isna(amp) else 0
                bar = '█' * int(amp_value * 10)
                print(f"    {state:12} {amp_value:.2f} {bar}")
            
            if analysis['patterns']:
                print(f"\n  Patterns:")
                for pattern in analysis['patterns']:
                    print(f"    {pattern}")
    
    # Generate report
    report = create_superposition_report(analyses)
    
    # Save report
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = f"emotional_superposition_{timestamp}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n\nReport saved: {report_file}")

if __name__ == "__main__":
    main()