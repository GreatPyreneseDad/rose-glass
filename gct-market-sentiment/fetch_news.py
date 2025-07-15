#!/usr/bin/env python3
"""
Simple script to fetch news from Tiingo API and save to CSV
"""

import requests
import pandas as pd
import json
from datetime import datetime, timedelta
import os

# Your API key
API_KEY = "ef1915ca2231c0c953e4fb7b72dec74bc767d9d1"

def fetch_tiingo_news(days_back=7, limit=100):
    """Fetch news from Tiingo API"""
    
    print("Connecting to Tiingo API...")
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    # API endpoint
    url = "https://api.tiingo.com/tiingo/news"
    
    # Parameters
    params = {
        'token': API_KEY,
        'startDate': start_date.strftime('%Y-%m-%d'),
        'endDate': end_date.strftime('%Y-%m-%d'),
        'limit': limit,
        'sortBy': 'publishedDate'
    }
    
    print(f"Fetching news from {start_date.date()} to {end_date.date()}...")
    
    try:
        # Make request
        response = requests.get(url, params=params)
        
        print(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            articles = response.json()
            print(f"Successfully fetched {len(articles)} articles")
            return articles
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            return None
            
    except Exception as e:
        print(f"Connection error: {e}")
        return None

def save_to_csv(articles, filename="tiingo_news.csv"):
    """Save articles to CSV file"""
    
    if not articles:
        print("No articles to save")
        return
        
    # Create data directory if it doesn't exist
    os.makedirs("data/raw", exist_ok=True)
    
    # Convert to DataFrame
    data = []
    for article in articles:
        data.append({
            'id': article.get('id', ''),
            'timestamp': article.get('publishedDate', ''),
            'title': article.get('title', ''),
            'description': article.get('description', ''),
            'source': article.get('source', ''),
            'url': article.get('url', ''),
            'tickers': json.dumps(article.get('tickers', [])),  # Store as JSON string
            'tags': json.dumps(article.get('tags', []))
        })
    
    df = pd.DataFrame(data)
    
    # Save to CSV
    filepath = f"data/raw/{filename}"
    df.to_csv(filepath, index=False)
    print(f"Saved {len(df)} articles to {filepath}")
    
    # Also save a sample for inspection
    sample_file = "data/raw/sample_5_articles.csv"
    df.head(5).to_csv(sample_file, index=False)
    print(f"Saved sample (first 5) to {sample_file}")
    
    return filepath

def main():
    """Main function"""
    print("=== Tiingo News Fetcher ===")
    print(f"API Key: {API_KEY[:10]}...{API_KEY[-5:]}")
    
    # Fetch news
    articles = fetch_tiingo_news(days_back=7, limit=100)
    
    if articles:
        # Save to CSV with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"tiingo_news_{timestamp}.csv"
        
        filepath = save_to_csv(articles, filename)
        
        # Print summary
        print("\n=== Summary ===")
        print(f"Total articles: {len(articles)}")
        
        # Count by source
        sources = {}
        for article in articles:
            source = article.get('source', 'Unknown')
            sources[source] = sources.get(source, 0) + 1
            
        print("\nArticles by source:")
        for source, count in sorted(sources.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {source}: {count}")
            
        # Extract unique tickers
        all_tickers = set()
        for article in articles:
            tickers = article.get('tickers', [])
            all_tickers.update(tickers)
            
        print(f"\nUnique tickers mentioned: {len(all_tickers)}")
        if all_tickers:
            print(f"Top tickers: {', '.join(list(all_tickers)[:10])}")
    else:
        print("Failed to fetch articles")

if __name__ == "__main__":
    main()