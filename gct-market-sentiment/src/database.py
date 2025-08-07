"""
Database module for GCT Market Sentiment system
"""

import sqlite3
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from contextlib import contextmanager
import pandas as pd
import time
import logging

logger = logging.getLogger(__name__)


class GCTDatabase:
    """SQLite database for storing news and GCT analysis"""

    def __init__(self, db_path: str = "data/gct_market.db"):
        self.db_path = db_path
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.init_database()

    @contextmanager
    def get_connection(self, max_retries: int = 3, retry_delay: float = 0.5):
        """Context manager for database connections with retry logic"""
        conn = None
        for attempt in range(max_retries):
            try:
                conn = sqlite3.connect(self.db_path, timeout=30.0)
                conn.row_factory = sqlite3.Row
                # Enable WAL mode for better concurrency
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA busy_timeout=5000")
                yield conn
                conn.commit()
                break
            except sqlite3.OperationalError as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Database connection attempt {attempt + 1} failed: {e}")
                    time.sleep(retry_delay * (attempt + 1))
                else:
                    logger.error(f"Failed to connect to database after {max_retries} attempts: {e}")
                    raise
            except Exception as e:
                logger.error(f"Database error: {e}")
                if conn:
                    conn.rollback()
                raise
            finally:
                if conn:
                    conn.close()

    def init_database(self):
        """Initialize database schema"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # News Articles table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS NewsArticles (
                    id TEXT PRIMARY KEY,
                    timestamp TIMESTAMP,
                    source TEXT,
                    title TEXT,
                    body TEXT,
                    tickers TEXT,
                    raw_data TEXT
                )
            """
            )

            # GCT Scores table
            cursor.execute(
                """
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
            """
            )

            # Ticker Timeline table for time series view
            cursor.execute(
                """
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
            """
            )

            # Create comprehensive indexes for better performance
            
            # News Articles indexes
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_articles_timestamp ON NewsArticles(timestamp DESC)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_articles_source_timestamp ON NewsArticles(source, timestamp DESC)"
            )
            
            # GCT Scores indexes
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_gct_article ON GCTScores(article_id)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_gct_timestamp ON GCTScores(timestamp DESC)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_gct_sentiment ON GCTScores(sentiment, timestamp DESC)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_gct_coherence ON GCTScores(coherence DESC, timestamp DESC)"
            )
            
            # Ticker Timeline indexes
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_ticker_time ON TickerTimeline(ticker, timestamp DESC)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_ticker_sentiment_time ON TickerTimeline(ticker, sentiment, timestamp DESC)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_ticker_coherence ON TickerTimeline(ticker, coherence DESC)"
            )
            
            # Create summary statistics table for fast aggregations
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS TickerSummaryCache (
                    ticker TEXT,
                    period TEXT,  -- '1h', '24h', '7d'
                    avg_coherence REAL,
                    max_coherence REAL,
                    min_coherence REAL,
                    std_coherence REAL,
                    avg_dc_dt REAL,
                    bullish_count INTEGER,
                    bearish_count INTEGER,
                    neutral_count INTEGER,
                    total_articles INTEGER,
                    last_updated TIMESTAMP,
                    PRIMARY KEY (ticker, period)
                )
                """
            )
            
            # Create query performance tracking table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS QueryPerformance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_type TEXT,
                    execution_time_ms REAL,
                    rows_returned INTEGER,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            
            # Enable query optimizer
            cursor.execute("PRAGMA optimize")
            cursor.execute("PRAGMA analysis_limit=1000")
            cursor.execute("ANALYZE")

            conn.commit()

    def insert_article(self, article: Dict) -> None:
        """Insert a news article with validation and error handling"""
        # Validate required fields
        required_fields = ['id', 'timestamp', 'source', 'title', 'body']
        for field in required_fields:
            if field not in article:
                raise ValueError(f"Missing required field: {field}")
        
        # Validate data types
        if not isinstance(article.get('timestamp'), (str, datetime)):
            raise TypeError("timestamp must be string or datetime")
            
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                tickers_json = json.dumps(article.get("tickers", []))

                cursor.execute(
                    """
                    INSERT OR REPLACE INTO NewsArticles 
                    (id, timestamp, source, title, body, tickers, raw_data)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        article["id"],
                        article["timestamp"],
                        article["source"],
                        article["title"],
                        article["body"],
                        tickers_json,
                        json.dumps(article),
                    ),
                )
                logger.info(f"Successfully inserted article: {article['id']}")
                
        except sqlite3.IntegrityError as e:
            logger.warning(f"Article {article['id']} already exists or constraint violation: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to insert article {article.get('id', 'unknown')}: {e}")
            raise

    def insert_gct_score(self, score_data: Dict) -> None:
        """Insert GCT analysis results"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            components_json = json.dumps(score_data.get("components", {}))

            cursor.execute(
                """
                INSERT INTO GCTScores 
                (article_id, timestamp, psi, rho, q_raw, f, q_opt, 
                 coherence, dc_dt, d2c_dt2, sentiment, components)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    score_data["article_id"],
                    score_data["timestamp"],
                    score_data["psi"],
                    score_data["rho"],
                    score_data["q_raw"],
                    score_data["f"],
                    score_data["q_opt"],
                    score_data["coherence"],
                    score_data["dc_dt"],
                    score_data["d2c_dt2"],
                    score_data["sentiment"],
                    components_json,
                ),
            )

            conn.commit()

    def update_ticker_timeline(
        self,
        ticker: str,
        timestamp: datetime,
        coherence: float,
        dc_dt: float,
        sentiment: str,
    ) -> None:
        """Update ticker timeline with aggregated data"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Check if entry exists for this ticker and hour
            hour_timestamp = timestamp.replace(minute=0, second=0, microsecond=0)

            cursor.execute(
                """
                SELECT id, coherence, article_count 
                FROM TickerTimeline 
                WHERE ticker = ? AND timestamp = ?
            """,
                (ticker, hour_timestamp),
            )

            existing = cursor.fetchone()

            if existing:
                # Update with weighted average
                new_count = existing["article_count"] + 1
                new_coherence = (
                    existing["coherence"] * existing["article_count"] + coherence
                ) / new_count

                cursor.execute(
                    """
                    UPDATE TickerTimeline 
                    SET coherence = ?, dc_dt = ?, sentiment = ?, article_count = ?
                    WHERE id = ?
                """,
                    (new_coherence, dc_dt, sentiment, new_count, existing["id"]),
                )
            else:
                # Insert new entry
                cursor.execute(
                    """
                    INSERT INTO TickerTimeline 
                    (ticker, timestamp, coherence, dc_dt, sentiment, article_count)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (ticker, hour_timestamp, coherence, dc_dt, sentiment, 1),
                )

            conn.commit()

    def get_recent_articles(self, limit: int = 100) -> List[Dict]:
        """Get recent articles"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT * FROM NewsArticles 
                ORDER BY timestamp DESC 
                LIMIT ?
            """,
                (limit,),
            )

            articles = []
            for row in cursor.fetchall():
                article = dict(row)
                article["tickers"] = json.loads(article["tickers"])
                articles.append(article)

            return articles

    def get_ticker_timeline(self, ticker: str, hours: int = 24) -> pd.DataFrame:
        """Get timeline data for a specific ticker with caching"""
        import time
        start_time = time.time()
        
        # Check cache first
        cache_key = f"{ticker}_{hours}h"
        cache_result = self._check_summary_cache(ticker, f"{hours}h")
        
        if cache_result and self._is_cache_fresh(cache_result, minutes=5):
            logger.info(f"Cache hit for {cache_key}")
            return cache_result
        
        with self.get_connection() as conn:
            query = """
                SELECT * FROM TickerTimeline 
                WHERE ticker = ? 
                AND timestamp > datetime('now', '-' || ? || ' hours')
                ORDER BY timestamp DESC
            """

            df = pd.read_sql_query(query, conn, params=(ticker, str(hours)))
            if not df.empty:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                
                # Update cache
                self._update_summary_cache(ticker, f"{hours}h", df)
            
            # Log query performance
            execution_time = (time.time() - start_time) * 1000
            self._log_query_performance("get_ticker_timeline", execution_time, len(df))

            return df
    
    def get_ticker_timeline_optimized(self, ticker: str, period: str = "24h") -> pd.DataFrame:
        """Optimized ticker timeline query using summary cache"""
        with self.get_connection() as conn:
            # First try to get from cache
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM TickerSummaryCache
                WHERE ticker = ? AND period = ?
                AND last_updated > datetime('now', '-5 minutes')
                """,
                (ticker, period)
            )
            
            cache_result = cursor.fetchone()
            if cache_result:
                # Return cached summary
                return pd.DataFrame([dict(cache_result)])
            
            # If not in cache, compute and store
            hours = self._parse_period(period)
            
            query = """
                WITH TickerStats AS (
                    SELECT 
                        ticker,
                        AVG(coherence) as avg_coherence,
                        MAX(coherence) as max_coherence,
                        MIN(coherence) as min_coherence,
                        AVG(dc_dt) as avg_dc_dt,
                        COUNT(CASE WHEN sentiment = 'bullish' THEN 1 END) as bullish_count,
                        COUNT(CASE WHEN sentiment = 'bearish' THEN 1 END) as bearish_count,
                        COUNT(CASE WHEN sentiment = 'neutral' THEN 1 END) as neutral_count,
                        COUNT(*) as total_articles,
                        MAX(timestamp) as last_update
                    FROM TickerTimeline
                    WHERE ticker = ?
                    AND timestamp > datetime('now', '-' || ? || ' hours')
                )
                SELECT * FROM TickerStats
            """
            
            df = pd.read_sql_query(query, conn, params=(ticker, str(hours)))
            
            if not df.empty:
                # Update cache
                self._update_cache_table(ticker, period, df.iloc[0].to_dict())
            
            return df
    
    def _parse_period(self, period: str) -> int:
        """Parse period string to hours"""
        if period.endswith('h'):
            return int(period[:-1])
        elif period.endswith('d'):
            return int(period[:-1]) * 24
        else:
            return 24  # default to 24 hours
    
    def _log_query_performance(self, query_type: str, execution_time_ms: float, rows: int):
        """Log query performance metrics"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO QueryPerformance (query_type, execution_time_ms, rows_returned)
                    VALUES (?, ?, ?)
                    """,
                    (query_type, execution_time_ms, rows)
                )
        except Exception as e:
            logger.warning(f"Failed to log query performance: {e}")

    def get_top_movers(self, sentiment: str = "bullish", limit: int = 10) -> List[Dict]:
        """Get top bullish or bearish tickers"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
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
            """,
                (sentiment, limit),
            )

            return [dict(row) for row in cursor.fetchall()]

    def get_article_with_scores(self, article_id: str) -> Optional[Dict]:
        """Get article with its GCT scores"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT a.*, g.*
                FROM NewsArticles a
                LEFT JOIN GCTScores g ON a.id = g.article_id
                WHERE a.id = ?
            """,
                (article_id,),
            )

            row = cursor.fetchone()
            if row:
                result = dict(row)
                result["tickers"] = json.loads(result["tickers"])
                if result.get("components"):
                    result["components"] = json.loads(result["components"])
                return result

            return None

    def get_coherence_stats(self) -> Dict:
        """Get overall coherence statistics"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
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
            """
            )

            return dict(cursor.fetchone())
    
    def _check_summary_cache(self, ticker: str, period: str) -> Optional[pd.DataFrame]:
        """Check if summary cache exists and is fresh"""
        try:
            with self.get_connection() as conn:
                query = """
                    SELECT * FROM TickerSummaryCache
                    WHERE ticker = ? AND period = ?
                    AND last_updated > datetime('now', '-5 minutes')
                """
                df = pd.read_sql_query(query, conn, params=(ticker, period))
                return df if not df.empty else None
        except:
            return None
    
    def _is_cache_fresh(self, cache_data: Any, minutes: int = 5) -> bool:
        """Check if cache data is fresh"""
        if isinstance(cache_data, pd.DataFrame) and not cache_data.empty:
            if 'last_updated' in cache_data.columns:
                last_updated = pd.to_datetime(cache_data['last_updated'].iloc[0])
                return (datetime.now() - last_updated).total_seconds() < minutes * 60
        return False
    
    def _update_summary_cache(self, ticker: str, period: str, df: pd.DataFrame):
        """Update summary cache with computed statistics"""
        if df.empty:
            return
            
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Compute statistics
                stats = {
                    'ticker': ticker,
                    'period': period,
                    'avg_coherence': df['coherence'].mean() if 'coherence' in df else 0,
                    'max_coherence': df['coherence'].max() if 'coherence' in df else 0,
                    'min_coherence': df['coherence'].min() if 'coherence' in df else 0,
                    'std_coherence': df['coherence'].std() if 'coherence' in df else 0,
                    'avg_dc_dt': df['dc_dt'].mean() if 'dc_dt' in df else 0,
                    'bullish_count': len(df[df['sentiment'] == 'bullish']) if 'sentiment' in df else 0,
                    'bearish_count': len(df[df['sentiment'] == 'bearish']) if 'sentiment' in df else 0,
                    'neutral_count': len(df[df['sentiment'] == 'neutral']) if 'sentiment' in df else 0,
                    'total_articles': len(df),
                    'last_updated': datetime.now()
                }
                
                # Insert or update cache
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO TickerSummaryCache
                    (ticker, period, avg_coherence, max_coherence, min_coherence, 
                     std_coherence, avg_dc_dt, bullish_count, bearish_count, 
                     neutral_count, total_articles, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    tuple(stats.values())
                )
        except Exception as e:
            logger.warning(f"Failed to update summary cache: {e}")
    
    def _update_cache_table(self, ticker: str, period: str, stats: dict):
        """Update cache table with pre-computed statistics"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO TickerSummaryCache
                    (ticker, period, avg_coherence, max_coherence, min_coherence,
                     std_coherence, avg_dc_dt, bullish_count, bearish_count,
                     neutral_count, total_articles, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                    """,
                    (
                        ticker, period,
                        stats.get('avg_coherence', 0),
                        stats.get('max_coherence', 0),
                        stats.get('min_coherence', 0),
                        stats.get('std_coherence', 0),
                        stats.get('avg_dc_dt', 0),
                        stats.get('bullish_count', 0),
                        stats.get('bearish_count', 0),
                        stats.get('neutral_count', 0),
                        stats.get('total_articles', 0)
                    )
                )
        except Exception as e:
            logger.warning(f"Failed to update cache table: {e}")
