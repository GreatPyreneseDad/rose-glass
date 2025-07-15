#!/usr/bin/env python3
"""
Simple local viewer for coherence data
"""

import pandas as pd
import os
from datetime import datetime

def view_latest_data():
    """View the latest coherence data saved locally"""
    
    print("=== GCT Coherence Data Viewer ===\n")
    
    # Check for data files
    if not os.path.exists("data/raw"):
        print("No data directory found. Run fetch_and_save_data.py first.")
        return
        
    # Find latest files
    files = os.listdir("data/raw")
    
    # Look for coherence metrics
    metric_files = [f for f in files if f.startswith("coherence_metrics_")]
    if metric_files:
        latest_metrics = sorted(metric_files)[-1]
        print(f"Latest Coherence Metrics: {latest_metrics}")
        
        df = pd.read_csv(f"data/raw/{latest_metrics}")
        print(f"\nTop 10 Most Coherent Stocks:")
        print("-" * 60)
        
        # Filter out rows with NaN coherence
        df_clean = df.dropna(subset=['coherence'])
        if not df_clean.empty:
            top = df_clean.nlargest(10, 'coherence')
            for _, row in top.iterrows():
                print(f"{row['ticker']:6} | Coherence: {row['coherence']:.3f} | "
                      f"Trend: {row['momentum']:+.4f} | Price: ${row['last_price']:.2f}")
        else:
            print("No valid coherence data found")
    
    # Look for alerts
    alert_files = [f for f in files if f.startswith("coherence_alerts_")]
    if alert_files:
        latest_alerts = sorted(alert_files)[-1]
        print(f"\n\nLatest Alerts: {latest_alerts}")
        print("-" * 60)
        
        df = pd.read_csv(f"data/raw/{latest_alerts}")
        for _, alert in df.iterrows():
            print(f"[{alert['type']}] {alert.get('ticker', 'MARKET')}: {alert['message']}")
    else:
        print("\n\nNo alerts found")
    
    # Look for summaries
    if os.path.exists("data/summaries"):
        summary_files = [f for f in os.listdir("data/summaries") if f.startswith("market_summary_")]
        if summary_files:
            latest_summary = sorted(summary_files)[-1]
            print(f"\n\nMarket Summary: {latest_summary}")
            print("-" * 60)
            
            df = pd.read_csv(f"data/summaries/{latest_summary}")
            
            # Show category breakdown
            categories = df.groupby('category')['daily_change_%'].agg(['mean', 'count'])
            print("\nSector Performance (Average Daily Change):")
            for cat, row in categories.iterrows():
                print(f"  {cat:15} | {row['mean']:+.2f}% ({int(row['count'])} stocks)")

if __name__ == "__main__":
    view_latest_data()