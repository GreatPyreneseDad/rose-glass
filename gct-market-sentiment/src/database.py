"""
Database module for GCT Market Sentiment system
"""

import sqlite3
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from contextlib import contextmanager
import pandas as pd


class GCTDatabase:
    """SQLite database for storing news and GCT analysis"""
    
    def __init__(self, db_path: str = "data/gct_market.db"):
        self.db_path = db_path
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.init_database()
        
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
            
    def init_database(self):
        """Initialize database schema"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # News Articles table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS NewsArticles (
                    id TEXT PRIMARY KEY,
                    timestamp TIMESTAMP,
                    source TEXT,
                    title TEXT,
                    body TEXT,
                    tickers TEXT,
                    raw_data TEXT
                )
            """)
            
            # GCT Scores table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS GCTScores (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    article_id TEXT REFERENCES NewsArticles(id),
                    timestamp TIMESTAMP,
                    psi REAL,
                    rho REAL,
                    q_raw REAL,
                    f REAL,
                    q_opt REAL,
                    coherence REAL,
                    dc_dt REAL,
                    d2c_dt2 REAL,
                    sentiment TEXT,
                    components TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Ticker Timeline table for time series view
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS TickerTimeline (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT,
                    timestamp TIMESTAMP,
                    coherence REAL,
                    dc_dt REAL,
                    sentiment TEXT,
                    article_count INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_articles_timestamp ON NewsArticles(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_gct_article ON GCTScores(article_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_ticker_time ON TickerTimeline(ticker, timestamp)")
            
            conn.commit()
            
    def insert_article(self, article: Dict) -> None:
        """Insert a news article"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            tickers_json = json.dumps(article.get('tickers', []))
            
            cursor.execute("""
                INSERT OR REPLACE INTO NewsArticles 
                (id, timestamp, source, title, body, tickers, raw_data)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                article['id'],
                article['timestamp'],
                article['source'],
                article['title'],
                article['body'],
                tickers_json,
                json.dumps(article)
            ))
            
            conn.commit()
            
    def insert_gct_score(self, score_data: Dict) -> None:
        """Insert GCT analysis results"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            components_json = json.dumps(score_data.get('components', {}))
            
            cursor.execute("""
                INSERT INTO GCTScores 
                (article_id, timestamp, psi, rho, q_raw, f, q_opt, 
                 coherence, dc_dt, d2c_dt2, sentiment, components)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                score_data['article_id'],
                score_data['timestamp'],
                score_data['psi'],
                score_data['rho'],
                score_data['q_raw'],
                score_data['f'],
                score_data['q_opt'],
                score_data['coherence'],
                score_data['dc_dt'],
                score_data['d2c_dt2'],
                score_data['sentiment'],
                components_json
            ))
            
            conn.commit()
            
    def update_ticker_timeline(self, ticker: str, timestamp: datetime, 
                              coherence: float, dc_dt: float, sentiment: str) -> None:
        """Update ticker timeline with aggregated data"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Check if entry exists for this ticker and hour
            hour_timestamp = timestamp.replace(minute=0, second=0, microsecond=0)
            
            cursor.execute("""
                SELECT id, coherence, article_count 
                FROM TickerTimeline 
                WHERE ticker = ? AND timestamp = ?
            """, (ticker, hour_timestamp))
            
            existing = cursor.fetchone()
            
            if existing:
                # Update with weighted average
                new_count = existing['article_count'] + 1
                new_coherence = (existing['coherence'] * existing['article_count'] + coherence) / new_count
                
                cursor.execute("""
                    UPDATE TickerTimeline 
                    SET coherence = ?, dc_dt = ?, sentiment = ?, article_count = ?
                    WHERE id = ?
                """, (new_coherence, dc_dt, sentiment, new_count, existing['id']))
            else:
                # Insert new entry
                cursor.execute("""
                    INSERT INTO TickerTimeline 
                    (ticker, timestamp, coherence, dc_dt, sentiment, article_count)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (ticker, hour_timestamp, coherence, dc_dt, sentiment, 1))
                
            conn.commit()
            
    def get_recent_articles(self, limit: int = 100) -> List[Dict]:
        """Get recent articles"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM NewsArticles 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (limit,))
            
            articles = []
            for row in cursor.fetchall():
                article = dict(row)
                article['tickers'] = json.loads(article['tickers'])
                articles.append(article)
                
            return articles
            
    def get_ticker_timeline(self, ticker: str, hours: int = 24) -> pd.DataFrame:
        """Get timeline data for a specific ticker"""
        with self.get_connection() as conn:
            query = """
                SELECT * FROM TickerTimeline 
                WHERE ticker = ? 
                AND timestamp > datetime('now', '-{} hours')
                ORDER BY timestamp
            """.format(hours)
            
            df = pd.read_sql_query(query, conn, params=(ticker,))
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
            return df
            
    def get_top_movers(self, sentiment: str = 'bullish', limit: int = 10) -> List[Dict]:
        """Get top bullish or bearish tickers"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT ticker, 
                       AVG(coherence) as avg_coherence,
                       AVG(dc_dt) as avg_dc_dt,
                       COUNT(*) as mention_count
                FROM TickerTimeline 
                WHERE sentiment = ? 
                AND timestamp > datetime('now', '-24 hours')
                GROUP BY ticker
                ORDER BY avg_dc_dt DESC
                LIMIT ?
            """, (sentiment, limit))
            
            return [dict(row) for row in cursor.fetchall()]
            
    def get_article_with_scores(self, article_id: str) -> Optional[Dict]:
        """Get article with its GCT scores"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT a.*, g.*
                FROM NewsArticles a
                LEFT JOIN GCTScores g ON a.id = g.article_id
                WHERE a.id = ?
            """, (article_id,))
            
            row = cursor.fetchone()
            if row:
                result = dict(row)
                result['tickers'] = json.loads(result['tickers'])
                if result.get('components'):
                    result['components'] = json.loads(result['components'])
                return result
                
            return None
            
    def get_coherence_stats(self) -> Dict:
        """Get overall coherence statistics"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_articles,
                    AVG(coherence) as avg_coherence,
                    MIN(coherence) as min_coherence,
                    MAX(coherence) as max_coherence,
                    SUM(CASE WHEN sentiment = 'bullish' THEN 1 ELSE 0 END) as bullish_count,
                    SUM(CASE WHEN sentiment = 'bearish' THEN 1 ELSE 0 END) as bearish_count,
                    SUM(CASE WHEN sentiment = 'neutral' THEN 1 ELSE 0 END) as neutral_count
                FROM GCTScores
                WHERE timestamp > datetime('now', '-24 hours')
            """)
            
            return dict(cursor.fetchone())
