#!/usr/bin/env python3
"""
Test what Tiingo API endpoints are available with this key
"""

import requests

API_KEY = "ef1915ca2231c0c953e4fb7b72dec74bc767d9d1"

def test_endpoint(name, url, params=None):
    """Test a single endpoint"""
    print(f"\n{'='*50}")
    print(f"Testing: {name}")
    print(f"URL: {url}")
    
    if params is None:
        params = {}
    params['token'] = API_KEY
    
    try:
        response = requests.get(url, params=params, timeout=10)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ SUCCESS")
            data = response.json()
            if isinstance(data, list):
                print(f"Response: List with {len(data)} items")
                if data:
                    print(f"First item: {str(data[0])[:200]}...")
            else:
                print(f"Response: {str(data)[:200]}...")
        else:
            print("❌ FAILED")
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Connection Error: {e}")

def main():
    print("=== Testing Tiingo API Access ===")
    print(f"API Key: {API_KEY[:10]}...{API_KEY[-5:]}")
    
    # Test various endpoints
    test_endpoint(
        "News API",
        "https://api.tiingo.com/tiingo/news"
    )
    
    test_endpoint(
        "Daily Prices (AAPL)",
        "https://api.tiingo.com/tiingo/daily/AAPL/prices"
    )
    
    test_endpoint(
        "Metadata (AAPL)",
        "https://api.tiingo.com/tiingo/daily/AAPL"
    )
    
    test_endpoint(
        "Crypto Top",
        "https://api.tiingo.com/tiingo/crypto/top"
    )
    
    test_endpoint(
        "IEX Real-time",
        "https://api.tiingo.com/iex/?tickers=AAPL"
    )
    
    print("\n" + "="*50)
    print("Testing complete!")

if __name__ == "__main__":
    main()