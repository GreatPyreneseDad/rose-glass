"""
Tiingo API client for financial news ingestion
"""

import os
import json
import asyncio
import aiohttp
import websockets
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
import requests
from dotenv import load_dotenv

load_dotenv()


class TiingoClient:
    """Client for Tiingo REST and WebSocket APIs"""
    
    def __init__(self, api_token: Optional[str] = None):
        self.api_token = api_token or os.getenv('TIINGO_API_TOKEN')
        if not self.api_token:
            raise ValueError("Tiingo API token required. Set TIINGO_API_TOKEN environment variable.")
            
        self.rest_base_url = "https://api.tiingo.com"
        self.ws_url = f"wss://api.tiingo.com/iex?token={self.api_token}"
        
    def get_historical_news(self, tickers: List[str] = None, 
                           start_date: datetime = None,
                           end_date: datetime = None,
                           limit: int = 1000) -> List[Dict]:
        """
        Fetch historical news from Tiingo REST API
        
        Args:
            tickers: List of stock tickers to filter news
            start_date: Start date for news
            end_date: End date for news
            limit: Maximum number of articles
            
        Returns:
            List of news articles
        """
        if not start_date:
            start_date = datetime.now() - timedelta(days=7)
        if not end_date:
            end_date = datetime.now()
            
        url = f"{self.rest_base_url}/tiingo/news"
        
        params = {
            'token': self.api_token,
            'startDate': start_date.strftime('%Y-%m-%d'),
            'endDate': end_date.strftime('%Y-%m-%d'),
            'limit': limit,
            'sortBy': 'publishedDate'
        }
        
        if tickers:
            params['tickers'] = ','.join(tickers)
            
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            articles = response.json()
            
            # Convert to our format
            processed_articles = []
            for article in articles:
                processed_articles.append({
                    'id': article.get('id', ''),
                    'timestamp': article.get('publishedDate', ''),
                    'source': article.get('source', ''),
                    'title': article.get('title', ''),
                    'body': article.get('description', ''),
                    'tickers': article.get('tickers', []),
                    'url': article.get('url', ''),
                    'raw_data': article
                })
                
            return processed_articles
            
        except requests.RequestException as e:
            print(f"Error fetching news: {e}")
            return []
            
    async def stream_realtime_news(self, callback: Callable[[Dict], None],
                                  tickers: List[str] = None):
        """
        Stream real-time news via WebSocket
        
        Args:
            callback: Function to call with each news article
            tickers: List of tickers to subscribe to
        """
        async with websockets.connect(self.ws_url) as websocket:
            # Subscribe to news feed
            subscribe_msg = {
                'eventName': 'subscribe',
                'eventData': {
                    'thresholdLevel': 5  # Get all news
                }
            }
            
            if tickers:
                subscribe_msg['eventData']['tickers'] = tickers
                
            await websocket.send(json.dumps(subscribe_msg))
            
            # Listen for messages
            async for message in websocket:
                try:
                    data = json.loads(message)
                    
                    if data.get('messageType') == 'news':
                        article = self._process_ws_article(data)
                        await callback(article)
                        
                except json.JSONDecodeError:
                    print(f"Failed to parse message: {message}")
                except Exception as e:
                    print(f"Error processing message: {e}")
                    
    def _process_ws_article(self, ws_data: Dict) -> Dict:
        """Process WebSocket news data to our format"""
        article_data = ws_data.get('data', {})
        
        return {
            'id': article_data.get('id', ''),
            'timestamp': article_data.get('publishedDate', datetime.now().isoformat()),
            'source': article_data.get('source', ''),
            'title': article_data.get('title', ''),
            'body': article_data.get('description', ''),
            'tickers': article_data.get('tickers', []),
            'url': article_data.get('url', ''),
            'raw_data': article_data
        }
        
    def get_ticker_metadata(self, ticker: str) -> Dict:
        """Get metadata for a specific ticker"""
        url = f"{self.rest_base_url}/tiingo/daily/{ticker}"
        
        params = {'token': self.api_token}
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Error fetching ticker metadata: {e}")
            return {}
            
    async def get_ticker_prices(self, tickers: List[str]) -> Dict[str, Dict]:
        """Get current prices for multiple tickers"""
        async with aiohttp.ClientSession() as session:
            tasks = []
            for ticker in tickers:
                url = f"{self.rest_base_url}/tiingo/daily/{ticker}/prices"
                params = {'token': self.api_token}
                tasks.append(self._fetch_price(session, ticker, url, params))
                
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            prices = {}
            for ticker, result in zip(tickers, results):
                if not isinstance(result, Exception) and result:
                    prices[ticker] = result[0]  # Latest price
                    
            return prices
            
    async def _fetch_price(self, session: aiohttp.ClientSession, 
                          ticker: str, url: str, params: Dict) -> Optional[List]:
        """Fetch price for a single ticker"""
        try:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
        except Exception as e:
            print(f"Error fetching price for {ticker}: {e}")
        return None


class MockTiingoClient(TiingoClient):
    """Mock client for testing without API token"""
    
    def __init__(self):
        self.api_token = "mock_token"
        
    def get_historical_news(self, **kwargs) -> List[Dict]:
        """Return mock news data"""
        mock_articles = [
            {
                'id': 'mock_1',
                'timestamp': datetime.now().isoformat(),
                'source': 'MockNews',
                'title': 'Tech Stocks Rally on AI Breakthrough Announcement',
                'body': 'Major technology companies saw significant gains today following breakthrough announcements in artificial intelligence. Investors are optimistic about the growth potential in the sector.',
                'tickers': ['NVDA', 'MSFT', 'GOOGL'],
                'url': 'https://example.com/1'
            },
            {
                'id': 'mock_2',
                'timestamp': (datetime.now() - timedelta(hours=2)).isoformat(),
                'source': 'MockFinance',
                'title': 'Federal Reserve Signals Potential Rate Cuts Amid Economic Concerns',
                'body': 'The Federal Reserve indicated today that it may consider rate cuts if economic conditions worsen. Market analysts are divided on the implications for equity markets.',
                'tickers': ['SPY', 'QQQ', 'DIA'],
                'url': 'https://example.com/2'
            },
            {
                'id': 'mock_3',
                'timestamp': (datetime.now() - timedelta(hours=4)).isoformat(),
                'source': 'MockBusiness',
                'title': 'Energy Sector Faces Volatility as Oil Prices Fluctuate',
                'body': 'Oil prices showed significant volatility today, causing uncertainty in energy stocks. Some analysts worry about the impact on broader market stability.',
                'tickers': ['XOM', 'CVX', 'COP'],
                'url': 'https://example.com/3'
            }
        ]
        
        return mock_articles[:kwargs.get('limit', 1000)]
        
    async def stream_realtime_news(self, callback, tickers=None):
        """Simulate streaming news"""
        while True:
            await asyncio.sleep(30)  # New article every 30 seconds
            
            mock_article = {
                'id': f'mock_stream_{datetime.now().timestamp()}',
                'timestamp': datetime.now().isoformat(),
                'source': 'MockStream',
                'title': f'Breaking: Market Update at {datetime.now().strftime("%H:%M")}',
                'body': 'This is a simulated real-time news update for testing purposes.',
                'tickers': ['AAPL', 'TSLA'],
                'url': 'https://example.com/stream'
            }
            
            await callback(mock_article)