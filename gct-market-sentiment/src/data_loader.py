"""
Data loader module for importing news from CSV/JSON files
"""

import os
import json
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime
import glob


class DataLoader:
    """Load financial news from local CSV/JSON files"""
    
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
    def load_csv_news(self, filename: str) -> List[Dict]:
        """Load news from CSV file"""
        filepath = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
            
        df = pd.read_csv(filepath)
        
        # Expected columns: id, timestamp, source, title, body, tickers
        required_cols = ['id', 'timestamp', 'source', 'title', 'body']
        
        # Check for required columns
        missing = set(required_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
            
        # Convert to list of dicts
        articles = []
        for _, row in df.iterrows():
            article = {
                'id': str(row['id']),
                'timestamp': str(row['timestamp']),
                'source': row['source'],
                'title': row['title'],
                'body': row['body'],
                'tickers': []
            }
            
            # Parse tickers if present
            if 'tickers' in row and pd.notna(row['tickers']):
                if isinstance(row['tickers'], str):
                    # Try to parse as JSON array or comma-separated
                    try:
                        article['tickers'] = json.loads(row['tickers'])
                    except:
                        article['tickers'] = [t.strip() for t in row['tickers'].split(',')]
                else:
                    article['tickers'] = [str(row['tickers'])]
                    
            articles.append(article)
            
        return articles
        
    def load_json_news(self, filename: str) -> List[Dict]:
        """Load news from JSON file"""
        filepath = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
            
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        # Handle both single article and array of articles
        if isinstance(data, dict):
            articles = [data]
        else:
            articles = data
            
        # Validate and normalize
        normalized = []
        for article in articles:
            normalized_article = {
                'id': str(article.get('id', f"local_{len(normalized)}")),
                'timestamp': article.get('timestamp', datetime.now().isoformat()),
                'source': article.get('source', 'Local File'),
                'title': article.get('title', ''),
                'body': article.get('body', article.get('content', '')),
                'tickers': article.get('tickers', [])
            }
            normalized.append(normalized_article)
            
        return normalized
        
    def list_available_files(self) -> Dict[str, List[str]]:
        """List all available data files"""
        csv_files = glob.glob(os.path.join(self.data_dir, "*.csv"))
        json_files = glob.glob(os.path.join(self.data_dir, "*.json"))
        
        return {
            'csv': [os.path.basename(f) for f in csv_files],
            'json': [os.path.basename(f) for f in json_files]
        }
        
    def create_sample_csv(self):
        """Create a sample CSV file for reference"""
        sample_data = pd.DataFrame([
            {
                'id': 'sample_1',
                'timestamp': '2024-01-07T10:00:00Z',
                'source': 'Financial Times',
                'title': 'Tech Stocks Rally as AI Innovation Accelerates',
                'body': 'Major technology companies saw significant gains today as investors showed renewed confidence in artificial intelligence developments. NVIDIA and Microsoft led the charge.',
                'tickers': '["NVDA", "MSFT"]'
            },
            {
                'id': 'sample_2',
                'timestamp': '2024-01-07T11:30:00Z',
                'source': 'Bloomberg',
                'title': 'Federal Reserve Signals Cautious Approach to Rate Changes',
                'body': 'The Federal Reserve indicated a data-dependent approach to future rate decisions, causing mixed reactions in equity markets.',
                'tickers': '["SPY", "TLT"]'
            }
        ])
        
        sample_path = os.path.join(self.data_dir, 'sample_news.csv')
        sample_data.to_csv(sample_path, index=False)
        return sample_path
        
    def create_sample_json(self):
        """Create a sample JSON file for reference"""
        sample_data = [
            {
                'id': 'json_1',
                'timestamp': '2024-01-07T12:00:00Z',
                'source': 'Reuters',
                'title': 'Oil Prices Surge on Supply Concerns',
                'body': 'Crude oil prices jumped 3% today as geopolitical tensions raised concerns about supply disruptions.',
                'tickers': ['XOM', 'CVX', 'USO']
            },
            {
                'id': 'json_2',
                'timestamp': '2024-01-07T13:00:00Z',
                'source': 'WSJ',
                'title': 'Banking Sector Shows Resilience Despite Economic Headwinds',
                'body': 'Major banks reported better-than-expected earnings, demonstrating resilience in a challenging economic environment.',
                'tickers': ['JPM', 'BAC', 'WFC']
            }
        ]
        
        sample_path = os.path.join(self.data_dir, 'sample_news.json')
        with open(sample_path, 'w') as f:
            json.dump(sample_data, f, indent=2)
        return sample_path