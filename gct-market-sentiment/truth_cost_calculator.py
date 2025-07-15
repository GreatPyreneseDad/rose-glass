#!/usr/bin/env python3
"""
Truth Cost Calculator based on GCT principles
Measures the energetic cost of deviations from coherent market behavior
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import sys
import matplotlib.pyplot as plt
from scipy import stats

def fetch_stock_data(ticker, days=90):
    """Fetch stock data for analysis"""
    stock = yf.Ticker(ticker)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    df = stock.history(start=start_date, end=end_date)
    if df.empty:
        return None
        
    df = df.reset_index()
    df.columns = [col.lower() for col in df.columns]
    return df

def calculate_gct_variables(df):
    """Calculate GCT variables over time"""
    # Returns
    df['returns'] = df['close'].pct_change()
    
    # ψ (Clarity) - Trend strength
    df['sma5'] = df['close'].rolling(5).mean()
    df['sma20'] = df['close'].rolling(20).mean()
    df['psi'] = (df['sma5'] - df['sma20']) / df['sma20'] * 10
    df['psi'] = df['psi'].clip(-1, 1)
    
    # ρ (Wisdom/Reflection) - Inverse volatility
    df['volatility'] = df['returns'].rolling(20).std() * np.sqrt(252) * 100
    df['rho'] = 1 / (1 + df['volatility'] / 20)
    
    # q (Emotion) - Momentum
    df['momentum'] = df['returns'].rolling(5).mean()
    df['q_raw'] = df['momentum'].abs() * 50
    df['q_raw'] = df['q_raw'].clip(0, 1)
    
    # f (Social) - Volume ratio
    df['volume_ma'] = df['volume'].rolling(20).mean()
    df['f'] = (df['volume'] / df['volume_ma']).clip(0, 2) * 0.5
    
    # Calculate coherence
    df['coherence'] = (
        df['psi'].abs() * 0.3 + 
        df['rho'] * 0.3 + 
        df['q_raw'] * 0.2 + 
        df['f'] * 0.2
    )
    
    return df

def calculate_truth_cost(df):
    """
    Calculate truth cost based on GCT principles:
    - Cost increases when market deviates from coherent patterns
    - Cost = Energy required to maintain false/unsustainable patterns
    """
    
    # 1. Coherence Deviation Cost
    # Optimal coherence assumed at 0.5-0.7 range
    df['optimal_coherence'] = 0.6
    df['coherence_deviation'] = np.abs(df['coherence'] - df['optimal_coherence'])
    
    # 2. Wisdom-Emotion Imbalance Cost
    # Cost when high emotion (q) with low wisdom (ρ)
    df['wisdom_emotion_ratio'] = df['rho'] / (df['q_raw'] + 0.1)  # Add small value to avoid division by zero
    df['we_imbalance_cost'] = np.where(
        df['wisdom_emotion_ratio'] < 1,  # More emotion than wisdom
        (1 - df['wisdom_emotion_ratio']) * df['q_raw'],  # Cost proportional to emotion level
        0
    )
    
    # 3. Volatility Spike Cost
    # Sudden volatility increases require energy to sustain
    df['volatility_change'] = df['volatility'].diff().abs()
    df['volatility_cost'] = df['volatility_change'] / 100
    
    # 4. Volume Manipulation Cost
    # Abnormal volume patterns indicate forced movements
    df['volume_zscore'] = np.abs(stats.zscore(df['volume'].dropna()))
    df['volume_cost'] = np.where(df['volume_zscore'] > 2, df['volume_zscore'] / 10, 0)
    
    # 5. Trend Reversal Cost
    # Energy required to reverse established trends
    df['trend_change'] = df['psi'].diff().abs()
    df['reversal_cost'] = df['trend_change'] * 0.5
    
    # Total Truth Cost
    df['truth_cost'] = (
        df['coherence_deviation'] * 0.25 +
        df['we_imbalance_cost'] * 0.25 +
        df['volatility_cost'] * 0.20 +
        df['volume_cost'] * 0.15 +
        df['reversal_cost'] * 0.15
    )
    
    # Cumulative truth cost (total energy expended)
    df['cumulative_truth_cost'] = df['truth_cost'].cumsum()
    
    # Truth cost acceleration (increasing cost = unsustainable)
    df['truth_cost_acceleration'] = df['truth_cost'].rolling(5).mean().diff()
    
    return df

def analyze_truth_cost_patterns(df, ticker):
    """Analyze patterns in truth cost"""
    recent_df = df.tail(20)
    
    analysis = {
        'ticker': ticker,
        'current_coherence': df['coherence'].iloc[-1],
        'current_truth_cost': df['truth_cost'].iloc[-1],
        'avg_truth_cost_30d': df['truth_cost'].tail(30).mean(),
        'cumulative_cost': df['cumulative_truth_cost'].iloc[-1],
        'cost_acceleration': df['truth_cost_acceleration'].iloc[-1] if not pd.isna(df['truth_cost_acceleration'].iloc[-1]) else 0,
        'wisdom_emotion_ratio': df['wisdom_emotion_ratio'].iloc[-1],
        'volatility': df['volatility'].iloc[-1]
    }
    
    # Pattern detection
    patterns = []
    
    # High truth cost warning
    if analysis['current_truth_cost'] > 0.5:
        patterns.append("⚠️ HIGH TRUTH COST: Unsustainable market behavior")
    
    # Accelerating cost
    if analysis['cost_acceleration'] > 0.05:
        patterns.append("📈 ACCELERATING COST: Increasing energy to maintain pattern")
    elif analysis['cost_acceleration'] < -0.05:
        patterns.append("📉 DECELERATING COST: Moving toward natural equilibrium")
    
    # Wisdom-emotion imbalance
    if analysis['wisdom_emotion_ratio'] < 0.5:
        patterns.append("🎭 EMOTION DOMINANCE: High emotion, low wisdom - unstable")
    elif analysis['wisdom_emotion_ratio'] > 2:
        patterns.append("🧘 WISDOM DOMINANCE: Rational market, sustainable patterns")
    
    # Coherence-cost relationship
    if analysis['current_coherence'] > 0.7 and analysis['current_truth_cost'] < 0.2:
        patterns.append("✅ NATURAL COHERENCE: Low-cost, sustainable movement")
    elif analysis['current_coherence'] < 0.3 and analysis['current_truth_cost'] > 0.4:
        patterns.append("🔴 FORCED CHAOS: High cost to maintain disorder")
    
    analysis['patterns'] = patterns
    
    return analysis

def create_truth_cost_visualization(df, ticker, save_path='truth_cost_analysis.png'):
    """Create comprehensive truth cost visualization"""
    fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)
    fig.suptitle(f'{ticker} Truth Cost Analysis', fontsize=16, fontweight='bold')
    
    # 1. Price and Coherence
    ax1 = axes[0]
    ax1_twin = ax1.twinx()
    
    ax1.plot(df['date'], df['close'], 'b-', label='Price', linewidth=2)
    ax1_twin.plot(df['date'], df['coherence'], 'g-', label='Coherence', alpha=0.7, linewidth=2)
    ax1_twin.axhline(y=0.6, color='g', linestyle='--', alpha=0.3, label='Optimal Coherence')
    
    ax1.set_ylabel('Price ($)', color='b')
    ax1_twin.set_ylabel('Coherence', color='g')
    ax1.set_title('Price Movement and Market Coherence')
    ax1.legend(loc='upper left')
    ax1_twin.legend(loc='upper right')
    
    # 2. Truth Cost Components
    ax2 = axes[1]
    ax2.fill_between(df['date'], 0, df['coherence_deviation'], alpha=0.3, label='Coherence Deviation')
    ax2.fill_between(df['date'], 0, df['we_imbalance_cost'], alpha=0.3, label='Wisdom-Emotion Imbalance')
    ax2.fill_between(df['date'], 0, df['volatility_cost'], alpha=0.3, label='Volatility Cost')
    ax2.plot(df['date'], df['truth_cost'], 'r-', linewidth=2, label='Total Truth Cost')
    
    ax2.set_ylabel('Truth Cost')
    ax2.set_title('Truth Cost Components')
    ax2.legend()
    ax2.set_ylim(bottom=0)
    
    # 3. Cumulative Truth Cost
    ax3 = axes[2]
    ax3.plot(df['date'], df['cumulative_truth_cost'], 'purple', linewidth=2)
    ax3.fill_between(df['date'], 0, df['cumulative_truth_cost'], alpha=0.3, color='purple')
    ax3.set_ylabel('Cumulative Truth Cost')
    ax3.set_title('Total Energy Expended Over Time')
    
    # 4. Wisdom-Emotion Balance
    ax4 = axes[3]
    ax4.plot(df['date'], df['wisdom_emotion_ratio'], 'orange', linewidth=2)
    ax4.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='Balance Point')
    ax4.fill_between(df['date'], 1, df['wisdom_emotion_ratio'], 
                     where=df['wisdom_emotion_ratio']>1, alpha=0.3, color='blue', label='Wisdom Dominance')
    ax4.fill_between(df['date'], 1, df['wisdom_emotion_ratio'], 
                     where=df['wisdom_emotion_ratio']<1, alpha=0.3, color='red', label='Emotion Dominance')
    
    ax4.set_ylabel('Wisdom/Emotion Ratio')
    ax4.set_xlabel('Date')
    ax4.set_title('Market Psychology Balance')
    ax4.legend()
    ax4.set_ylim(0, 3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return save_path

def generate_truth_cost_report(analyses):
    """Generate comprehensive truth cost report"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    report = f"""
TRUTH COST ANALYSIS REPORT
Generated: {timestamp}
{'='*60}

THEORETICAL FOUNDATION:
Truth Cost measures the energy required to maintain patterns that
deviate from natural market coherence. High truth costs indicate
unsustainable behaviors that will eventually revert.

KEY METRICS:
"""
    
    # Sort by current truth cost
    sorted_analyses = sorted(analyses, key=lambda x: x['current_truth_cost'], reverse=True)
    
    for analysis in sorted_analyses:
        report += f"""
{'='*60}
TICKER: {analysis['ticker']}
{'='*60}
Current Coherence:      {analysis['current_coherence']:.3f}
Current Truth Cost:     {analysis['current_truth_cost']:.3f}
30-Day Avg Cost:        {analysis['avg_truth_cost_30d']:.3f}
Cumulative Energy:      {analysis['cumulative_cost']:.2f}
Cost Acceleration:      {analysis['cost_acceleration']:+.3f}
Wisdom/Emotion Ratio:   {analysis['wisdom_emotion_ratio']:.2f}
Current Volatility:     {analysis['volatility']:.1f}%

PATTERNS DETECTED:
"""
        for pattern in analysis['patterns']:
            report += f"  • {pattern}\n"
    
    # Market-wide summary
    avg_truth_cost = np.mean([a['current_truth_cost'] for a in analyses])
    high_cost_count = len([a for a in analyses if a['current_truth_cost'] > 0.5])
    
    report += f"""

{'='*60}
MARKET-WIDE SUMMARY:
{'='*60}
Average Truth Cost:     {avg_truth_cost:.3f}
High Cost Warnings:     {high_cost_count} tickers
Market Sustainability:  {'LOW' if avg_truth_cost > 0.4 else 'MODERATE' if avg_truth_cost > 0.2 else 'HIGH'}

INTERPRETATION GUIDE:
• Truth Cost < 0.2:  Natural, sustainable market behavior
• Truth Cost 0.2-0.4: Moderate strain, watch for reversals  
• Truth Cost > 0.4:  High strain, unsustainable patterns
• Truth Cost > 0.6:  Extreme - expect significant corrections

INVESTMENT IMPLICATIONS:
- High truth cost stocks likely to revert to mean
- Low truth cost with high coherence = sustainable trends
- Rising cumulative costs signal exhaustion points
- Wisdom/Emotion < 1 indicates emotional extremes
"""
    
    return report

def main():
    if len(sys.argv) < 2:
        print("Usage: python truth_cost_calculator.py TICKER [TICKER2 ...]")
        print("Example: python truth_cost_calculator.py SPY QQQ TSLA NVDA")
        return
    
    tickers = [t.upper() for t in sys.argv[1:]]
    analyses = []
    
    print("\n=== TRUTH COST CALCULATION ===\n")
    
    for ticker in tickers:
        print(f"Analyzing {ticker}...")
        
        # Fetch data
        df = fetch_stock_data(ticker)
        if df is None:
            print(f"Failed to fetch data for {ticker}")
            continue
        
        # Calculate GCT variables
        df = calculate_gct_variables(df)
        
        # Calculate truth costs
        df = calculate_truth_cost(df)
        
        # Analyze patterns
        analysis = analyze_truth_cost_patterns(df, ticker)
        analyses.append(analysis)
        
        # Create visualization
        viz_path = f"truth_cost_{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        create_truth_cost_visualization(df, ticker, viz_path)
        print(f"  Saved visualization: {viz_path}")
        
        # Save detailed data
        csv_path = f"truth_cost_data_{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(csv_path, index=False)
        print(f"  Saved data: {csv_path}")
    
    # Generate report
    report = generate_truth_cost_report(analyses)
    
    # Save report
    report_path = f"truth_cost_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"\nReport saved: {report_path}")
    print("\n" + report)

if __name__ == "__main__":
    main()