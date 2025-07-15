#!/usr/bin/env python3
"""
Fetch and save market data for offline analysis
"""

import requests
import pandas as pd
import json
from datetime import datetime, timedelta
import os
import time

API_KEY = "ef1915ca2231c0c953e4fb7b72dec74bc767d9d1"

# Comprehensive ticker list
TICKERS = {
    'mega_tech': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA'],
    'indices': ['SPY', 'QQQ', 'DIA', 'IWM', 'VTI'],
    'finance': ['JPM', 'BAC', 'GS', 'WFC', 'MS'],
    'popular': ['GME', 'AMC', 'BB', 'PLTR', 'SOFI'],
    'crypto_related': ['COIN', 'MSTR', 'RIOT', 'MARA'],
}

def fetch_and_save_ticker(ticker, days=30):
    """Fetch ticker data and save to CSV"""
    try:
        url = f"https://api.tiingo.com/tiingo/daily/{ticker}/prices"
        params = {
            'token': API_KEY,
            'startDate': (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d'),
            'endDate': datetime.now().strftime('%Y-%m-%d')
        }
        
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data)
            
            # Save individual ticker data
            ticker_dir = f"data/tickers/{ticker}"
            os.makedirs(ticker_dir, exist_ok=True)
            
            filename = f"{ticker_dir}/{ticker}_{datetime.now().strftime('%Y%m%d')}.csv"
            df.to_csv(filename, index=False)
            
            print(f"✓ {ticker}: Saved {len(df)} days of data")
            return df
        else:
            print(f"✗ {ticker}: Failed to fetch (status {response.status_code})")
            
    except Exception as e:
        print(f"✗ {ticker}: Error - {str(e)}")
        
    return None

def create_summary_report(all_data):
    """Create a summary report of all tickers"""
    summary = []
    
    for category, tickers in all_data.items():
        for ticker, df in tickers.items():
            if df is not None and len(df) > 0:
                # Calculate metrics
                last_price = df['close'].iloc[-1]
                prev_price = df['close'].iloc[-2] if len(df) > 1 else last_price
                week_ago = df['close'].iloc[-5] if len(df) >= 5 else df['close'].iloc[0]
                
                daily_change = (last_price - prev_price) / prev_price * 100
                weekly_change = (last_price - week_ago) / week_ago * 100
                volatility = df['close'].pct_change().std() * 100
                avg_volume = df['volume'].mean()
                
                summary.append({
                    'category': category,
                    'ticker': ticker,
                    'last_price': round(last_price, 2),
                    'daily_change_%': round(daily_change, 2),
                    'weekly_change_%': round(weekly_change, 2),
                    'volatility_%': round(volatility, 2),
                    'avg_volume': int(avg_volume),
                    'last_date': df['date'].iloc[-1]
                })
    
    return pd.DataFrame(summary)

def main():
    print("=== Market Data Fetcher ===")
    print(f"Fetching data for {sum(len(t) for t in TICKERS.values())} tickers...\n")
    
    # Create directories
    os.makedirs("data/tickers", exist_ok=True)
    os.makedirs("data/summaries", exist_ok=True)
    
    all_data = {}
    
    # Fetch data for each category
    for category, ticker_list in TICKERS.items():
        print(f"\n{category.upper()}:")
        all_data[category] = {}
        
        for ticker in ticker_list:
            df = fetch_and_save_ticker(ticker)
            all_data[category][ticker] = df
            time.sleep(0.1)  # Be nice to the API
    
    # Create summary report
    print("\nCreating summary report...")
    summary_df = create_summary_report(all_data)
    
    if not summary_df.empty:
        # Save summary
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        summary_file = f"data/summaries/market_summary_{timestamp}.csv"
        summary_df.to_csv(summary_file, index=False)
        print(f"\nSaved summary to {summary_file}")
        
        # Display top movers
        print("\n=== TOP MOVERS ===")
        print("\nBiggest Gainers (Daily):")
        top_gainers = summary_df.nlargest(5, 'daily_change_%')
        for _, row in top_gainers.iterrows():
            print(f"  {row['ticker']}: +{row['daily_change_%']:.2f}% @ ${row['last_price']}")
            
        print("\nBiggest Losers (Daily):")
        top_losers = summary_df.nsmallest(5, 'daily_change_%')
        for _, row in top_losers.iterrows():
            print(f"  {row['ticker']}: {row['daily_change_%']:.2f}% @ ${row['last_price']}")
            
        print("\nMost Volatile:")
        most_volatile = summary_df.nlargest(5, 'volatility_%')
        for _, row in most_volatile.iterrows():
            print(f"  {row['ticker']}: {row['volatility_%']:.2f}% volatility")
    
    # Save metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'tickers_fetched': sum(len(t) for t in TICKERS.values()),
        'api_key': API_KEY[:10] + '...',
        'categories': list(TICKERS.keys())
    }
    
    with open('data/fetch_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n✓ Data fetch complete!")

if __name__ == "__main__":
    main()