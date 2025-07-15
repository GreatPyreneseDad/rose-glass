#!/usr/bin/env python3
"""
Integrated Market Analysis
Combines coherence, truth cost, and emotional superposition
"""

import subprocess
import json
import pandas as pd
from datetime import datetime
import sys

def run_analysis(tickers):
    """Run all three analyses on given tickers"""
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'tickers': {},
        'market_summary': {}
    }
    
    # Join tickers into string
    ticker_str = ' '.join(tickers)
    
    print("="*60)
    print("INTEGRATED GCT MARKET ANALYSIS")
    print("="*60)
    print(f"Analyzing: {ticker_str}")
    print("="*60)
    
    # 1. Run coherence analysis
    print("\n1. COHERENCE ANALYSIS")
    print("-"*40)
    result = subprocess.run(
        [sys.executable, 'analyze_any_stock.py'] + tickers,
        capture_output=True, text=True
    )
    
    # Parse coherence results from CSV
    try:
        files = [f for f in os.listdir('.') if f.startswith('custom_analysis_') and f.endswith('.csv')]
        if files:
            latest_file = sorted(files)[-1]
            coherence_df = pd.read_csv(latest_file)
            for _, row in coherence_df.iterrows():
                ticker = row['ticker']
                if ticker not in results['tickers']:
                    results['tickers'][ticker] = {}
                results['tickers'][ticker]['coherence'] = {
                    'score': row['coherence'],
                    'psi': row['psi'],
                    'rho': row['rho'],
                    'trend': row['trend_strength'],
                    'volatility': row['volatility']
                }
    except Exception as e:
        print(f"Error parsing coherence: {e}")
    
    # 2. Run truth cost analysis
    print("\n2. TRUTH COST ANALYSIS")
    print("-"*40)
    result = subprocess.run(
        [sys.executable, 'truth_cost_calculator.py'] + tickers,
        capture_output=True, text=True
    )
    
    # Parse truth cost from output
    if result.stdout:
        lines = result.stdout.split('\n')
        current_ticker = None
        for line in lines:
            if 'TICKER:' in line:
                current_ticker = line.split('TICKER:')[1].strip()
            elif 'Current Truth Cost:' in line and current_ticker:
                cost = float(line.split(':')[1].strip())
                if current_ticker not in results['tickers']:
                    results['tickers'][current_ticker] = {}
                results['tickers'][current_ticker]['truth_cost'] = cost
            elif 'Wisdom/Emotion Ratio:' in line and current_ticker:
                ratio = float(line.split(':')[1].strip())
                results['tickers'][current_ticker]['wisdom_emotion_ratio'] = ratio
    
    # 3. Run emotional superposition
    print("\n3. EMOTIONAL SUPERPOSITION")
    print("-"*40)
    result = subprocess.run(
        [sys.executable, 'emotional_superposition.py'] + tickers,
        capture_output=True, text=True
    )
    
    # Parse emotional data
    try:
        files = [f for f in os.listdir('.') if f.startswith('emotional_superposition_') and f.endswith('.json')]
        if files:
            latest_file = sorted(files)[-1]
            with open(latest_file, 'r') as f:
                emotional_data = json.load(f)
                for analysis in emotional_data.get('individual_analyses', []):
                    ticker = analysis['ticker']
                    if ticker not in results['tickers']:
                        results['tickers'][ticker] = {}
                    results['tickers'][ticker]['emotions'] = {
                        'fear_greed': analysis['fear_greed_index'],
                        'entropy': analysis['emotional_entropy'],
                        'dominant': analysis['dominant_emotion'],
                        'coherence': analysis['emotional_coherence']
                    }
    except Exception as e:
        print(f"Error parsing emotions: {e}")
    
    # Generate integrated insights
    print("\n" + "="*60)
    print("INTEGRATED INSIGHTS")
    print("="*60)
    
    for ticker, data in results['tickers'].items():
        print(f"\n{ticker}:")
        print("-"*40)
        
        coherence = data.get('coherence', {}).get('score', 0)
        truth_cost = data.get('truth_cost', 0)
        fear_greed = data.get('emotions', {}).get('fear_greed', 0.5)
        
        # Overall health score (0-100)
        health_score = (
            (coherence * 40) +  # 40% weight on coherence
            ((1 - truth_cost) * 30) +  # 30% weight on low truth cost
            (abs(fear_greed - 0.5) * 30)  # 30% weight on emotional balance
        ) * 100
        
        print(f"  Health Score: {health_score:.1f}/100")
        print(f"  Coherence: {coherence:.3f}")
        print(f"  Truth Cost: {truth_cost:.3f}")
        print(f"  Fear/Greed: {fear_greed:.2f}")
        
        # Generate actionable insights
        insights = []
        
        if coherence > 0.6 and truth_cost < 0.2:
            insights.append("✅ STRONG BUY: High coherence, sustainable pattern")
        elif coherence < 0.3 and truth_cost > 0.5:
            insights.append("🚫 AVOID: Low coherence, unsustainable costs")
        
        if fear_greed > 0.8 and truth_cost > 0.4:
            insights.append("⚠️ REVERSAL RISK: Extreme greed with high costs")
        elif fear_greed < 0.2 and coherence > 0.5:
            insights.append("🎯 OVERSOLD OPPORTUNITY: Fear with building coherence")
        
        if data.get('wisdom_emotion_ratio', 1) < 0.5:
            insights.append("🎭 EMOTIONAL TRADING: Low wisdom/emotion ratio")
        
        if insights:
            print("\n  Insights:")
            for insight in insights:
                print(f"    {insight}")
    
    # Market summary
    avg_coherence = sum(d.get('coherence', {}).get('score', 0) for d in results['tickers'].values()) / len(results['tickers'])
    avg_truth_cost = sum(d.get('truth_cost', 0) for d in results['tickers'].values()) / len(results['tickers'])
    avg_fear_greed = sum(d.get('emotions', {}).get('fear_greed', 0.5) for d in results['tickers'].values()) / len(results['tickers'])
    
    print("\n" + "="*60)
    print("MARKET SUMMARY")
    print("="*60)
    print(f"Average Coherence: {avg_coherence:.3f}")
    print(f"Average Truth Cost: {avg_truth_cost:.3f}")
    print(f"Market Sentiment: {avg_fear_greed:.2f} ({'GREED' if avg_fear_greed > 0.6 else 'FEAR' if avg_fear_greed < 0.4 else 'NEUTRAL'})")
    
    if avg_coherence > 0.5 and avg_truth_cost < 0.3:
        print("\n🟢 MARKET CONDITION: Healthy and sustainable")
    elif avg_truth_cost > 0.5:
        print("\n🔴 MARKET CONDITION: Unsustainable - expect corrections")
    else:
        print("\n🟡 MARKET CONDITION: Mixed signals - trade carefully")
    
    # Save integrated report
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = f'integrated_analysis_{timestamp}.json'
    with open(report_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n\nFull report saved: {report_file}")
    
    return results

if __name__ == "__main__":
    import os
    
    if len(sys.argv) < 2:
        print("Usage: python integrated_analysis.py TICKER [TICKER2 ...]")
        print("Example: python integrated_analysis.py SPY QQQ NVDA TSLA")
        sys.exit(1)
    
    tickers = [t.upper() for t in sys.argv[1:]]
    run_analysis(tickers)