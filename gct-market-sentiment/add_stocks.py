#!/usr/bin/env python3
"""
Add stocks to your monitoring list and fetch their data
"""

import os
import sys
import pandas as pd
from datetime import datetime, timedelta
import requests
import time

TIINGO_API_KEY = 'ef1915ca2231c0c953e4fb7b72dec74bc767d9d1'

def fetch_and_save_ticker(ticker):
    """Fetch data for a ticker and save to CSV"""
    print(f"Fetching {ticker}...")
    
    # Create directory
    ticker_dir = f"data/tickers/{ticker}"
    os.makedirs(ticker_dir, exist_ok=True)
    
    # Tiingo API endpoint
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    
    url = f"https://api.tiingo.com/tiingo/daily/{ticker}/prices"
    params = {
        'startDate': start_date.strftime('%Y-%m-%d'),
        'endDate': end_date.strftime('%Y-%m-%d'),
        'token': TIINGO_API_KEY
    }
    
    try:
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            if data:
                df = pd.DataFrame(data)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_file = f"{ticker_dir}/{ticker}_{timestamp}.csv"
                df.to_csv(output_file, index=False)
                print(f"✓ Saved {ticker} data to {output_file}")
                return True
            else:
                print(f"✗ No data returned for {ticker}")
        elif response.status_code == 404:
            print(f"✗ Ticker {ticker} not found")
        elif response.status_code == 429:
            print(f"✗ Rate limit hit. Wait a moment.")
        else:
            print(f"✗ Error {response.status_code} for {ticker}")
            
    except Exception as e:
        print(f"✗ Error fetching {ticker}: {e}")
    
    return False

def update_ticker_list(new_tickers):
    """Update the list of monitored tickers"""
    ticker_file = "data/monitored_tickers.txt"
    
    # Load existing tickers
    existing = set()
    if os.path.exists(ticker_file):
        with open(ticker_file, 'r') as f:
            existing = set(line.strip() for line in f if line.strip())
    
    # Add new tickers
    all_tickers = existing.union(set(new_tickers))
    
    # Save updated list
    with open(ticker_file, 'w') as f:
        for ticker in sorted(all_tickers):
            f.write(f"{ticker}\n")
    
    print(f"\nMonitoring {len(all_tickers)} tickers total")
    return all_tickers

def main():
    if len(sys.argv) < 2:
        print("Usage: python add_stocks.py TICKER [TICKER2 TICKER3 ...]")
        print("Example: python add_stocks.py AMD NFLX DIS")
        print("\nCurrently monitored tickers:")
        
        # Show current list
        if os.path.exists("data/monitored_tickers.txt"):
            with open("data/monitored_tickers.txt", 'r') as f:
                tickers = [line.strip() for line in f if line.strip()]
                print(", ".join(tickers))
        else:
            print("None")
        return
    
    new_tickers = [t.upper() for t in sys.argv[1:]]
    print(f"\nAdding tickers: {', '.join(new_tickers)}")
    
    # Fetch data for each new ticker
    success_count = 0
    for ticker in new_tickers:
        if fetch_and_save_ticker(ticker):
            success_count += 1
        time.sleep(0.5)  # Rate limiting
    
    print(f"\nSuccessfully fetched {success_count}/{len(new_tickers)} tickers")
    
    # Update monitored list
    if success_count > 0:
        all_tickers = update_ticker_list([t for t in new_tickers])
        
        print("\nNext steps:")
        print("1. Run: ./venv/bin/python calculate_coherence_offline.py")
        print("2. Run: ./venv/bin/python local_dashboard.py")

if __name__ == "__main__":
    main()