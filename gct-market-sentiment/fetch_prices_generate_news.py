#!/usr/bin/env python3
"""
Fetch stock prices from Tiingo and generate synthetic news based on price movements
"""

import requests
import pandas as pd
import json
from datetime import datetime, timedelta
import os

API_KEY = "ef1915ca2231c0c953e4fb7b72dec74bc767d9d1"

# Popular tickers to track
TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA',
    'JPM', 'BAC', 'GS', 'WFC',
    'XOM', 'CVX', 'COP',
    'JNJ', 'PFE', 'UNH',
    'SPY', 'QQQ', 'DIA', 'IWM'
]

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
            return response.json()
        else:
            print(f"Error fetching {ticker}: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return None

def calculate_price_changes(price_data):
    """Calculate price changes and volatility"""
    if not price_data or len(price_data) < 2:
        return None
        
    df = pd.DataFrame(price_data)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # Calculate returns
    df['daily_return'] = df['close'].pct_change()
    df['volume_change'] = df['volume'].pct_change()
    
    # Recent metrics
    last_price = df['close'].iloc[-1]
    prev_price = df['close'].iloc[-2]
    week_ago_price = df['close'].iloc[-5] if len(df) >= 5 else df['close'].iloc[0]
    
    return {
        'last_price': last_price,
        'daily_change': (last_price - prev_price) / prev_price,
        'weekly_change': (last_price - week_ago_price) / week_ago_price,
        'volatility': df['daily_return'].std(),
        'avg_volume': df['volume'].mean(),
        'last_volume': df['volume'].iloc[-1],
        'price_data': df
    }

def generate_news_from_prices(ticker, metrics, company_names):
    """Generate synthetic news based on price movements"""
    
    news_items = []
    timestamp = datetime.now()
    
    daily_pct = metrics['daily_change'] * 100
    weekly_pct = metrics['weekly_change'] * 100
    company = company_names.get(ticker, ticker)
    
    # Generate news based on price movement
    if abs(daily_pct) > 3:  # Significant daily move
        if daily_pct > 0:
            title = f"{company} ({ticker}) Surges {daily_pct:.1f}% in Heavy Trading"
            body = (f"{company} shares jumped {daily_pct:.1f}% to ${metrics['last_price']:.2f} "
                   f"on volume of {metrics['last_volume']:,.0f} shares. "
                   f"The stock has gained {weekly_pct:.1f}% over the past week.")
            sentiment_hint = "bullish"
        else:
            title = f"{company} ({ticker}) Plunges {abs(daily_pct):.1f}% Amid Market Concerns"
            body = (f"{company} shares fell {abs(daily_pct):.1f}% to ${metrics['last_price']:.2f} "
                   f"as investors showed concern. Trading volume reached {metrics['last_volume']:,.0f} shares. "
                   f"The stock is {'down' if weekly_pct < 0 else 'up'} {abs(weekly_pct):.1f}% for the week.")
            sentiment_hint = "bearish"
            
        news_items.append({
            'id': f'price_{ticker}_{timestamp.strftime("%Y%m%d_%H%M%S")}',
            'timestamp': timestamp.isoformat(),
            'title': title,
            'body': body,
            'source': 'Price Analysis',
            'tickers': [ticker],
            'sentiment_hint': sentiment_hint,
            'metrics': {
                'daily_change': daily_pct,
                'weekly_change': weekly_pct,
                'volatility': metrics['volatility']
            }
        })
    
    # Volatility news
    if metrics['volatility'] > 0.03:  # High volatility
        title = f"High Volatility in {company} ({ticker}) Signals Uncertainty"
        body = (f"{company} has shown increased volatility with daily movements averaging "
               f"{metrics['volatility']*100:.1f}%. Current price stands at ${metrics['last_price']:.2f}.")
        
        news_items.append({
            'id': f'vol_{ticker}_{timestamp.strftime("%Y%m%d_%H%M%S")}',
            'timestamp': (timestamp - timedelta(hours=1)).isoformat(),
            'title': title,
            'body': body,
            'source': 'Volatility Alert',
            'tickers': [ticker],
            'sentiment_hint': 'neutral',
            'metrics': {
                'volatility': metrics['volatility']
            }
        })
    
    return news_items

def main():
    print("=== Fetching Price Data and Generating News ===")
    
    # Company names for better news generation
    company_names = {
        'AAPL': 'Apple', 'MSFT': 'Microsoft', 'GOOGL': 'Google',
        'AMZN': 'Amazon', 'NVDA': 'NVIDIA', 'META': 'Meta',
        'TSLA': 'Tesla', 'JPM': 'JPMorgan', 'BAC': 'Bank of America',
        'GS': 'Goldman Sachs', 'WFC': 'Wells Fargo', 'XOM': 'Exxon Mobil',
        'CVX': 'Chevron', 'COP': 'ConocoPhillips', 'JNJ': 'Johnson & Johnson',
        'PFE': 'Pfizer', 'UNH': 'UnitedHealth', 'SPY': 'S&P 500 ETF',
        'QQQ': 'Nasdaq ETF', 'DIA': 'Dow Jones ETF', 'IWM': 'Russell 2000 ETF'
    }
    
    all_news = []
    price_summaries = []
    
    # Fetch data for each ticker
    for ticker in TICKERS:
        print(f"Processing {ticker}...")
        
        # Fetch price data
        price_data = fetch_price_data(ticker, days=30)
        
        if price_data:
            # Calculate metrics
            metrics = calculate_price_changes(price_data)
            
            if metrics:
                # Generate news
                news_items = generate_news_from_prices(ticker, metrics, company_names)
                all_news.extend(news_items)
                
                # Store summary
                price_summaries.append({
                    'ticker': ticker,
                    'last_price': metrics['last_price'],
                    'daily_change': metrics['daily_change'],
                    'weekly_change': metrics['weekly_change'],
                    'volatility': metrics['volatility']
                })
    
    # Save generated news to CSV
    if all_news:
        os.makedirs("data/raw", exist_ok=True)
        
        # Convert to DataFrame
        news_df = pd.DataFrame(all_news)
        
        # Save main file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        news_file = f"data/raw/generated_news_{timestamp}.csv"
        news_df.to_csv(news_file, index=False)
        print(f"\nSaved {len(news_df)} news items to {news_file}")
        
        # Save price summary
        price_df = pd.DataFrame(price_summaries)
        price_file = f"data/raw/price_summary_{timestamp}.csv"
        price_df.to_csv(price_file, index=False)
        print(f"Saved price summary to {price_file}")
        
        # Print summary
        print("\n=== Summary ===")
        print(f"Total news items generated: {len(all_news)}")
        print(f"Tickers processed: {len(price_summaries)}")
        
        # Show biggest movers
        price_df['daily_change_pct'] = price_df['daily_change'] * 100
        top_gainers = price_df.nlargest(5, 'daily_change_pct')
        top_losers = price_df.nsmallest(5, 'daily_change_pct')
        
        print("\nTop Gainers:")
        for _, row in top_gainers.iterrows():
            print(f"  {row['ticker']}: +{row['daily_change_pct']:.2f}%")
            
        print("\nTop Losers:")
        for _, row in top_losers.iterrows():
            print(f"  {row['ticker']}: {row['daily_change_pct']:.2f}%")
    else:
        print("No news generated")

if __name__ == "__main__":
    main()