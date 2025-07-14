"""
Database for Dream Analysis using SQLite
"""

import sqlite3
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from contextlib import contextmanager
import pandas as pd


class DreamDatabase:
    """SQLite database for storing dreams and analysis"""
    
    def __init__(self, db_path: str = "data/dreams.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
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
            
            # Dreams table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS Dreams (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    dream_date DATE NOT NULL,
                    title TEXT,
                    narrative TEXT NOT NULL,
                    lucid BOOLEAN DEFAULT FALSE,
                    sleep_quality REAL,
                    sleep_hours REAL,
                    tags TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Dream Analysis table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS DreamAnalysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dream_id INTEGER REFERENCES Dreams(id),
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    coherence REAL NOT NULL,
                    psi REAL NOT NULL,
                    rho REAL NOT NULL,
                    q_raw REAL NOT NULL,
                    q_opt REAL NOT NULL,
                    f REAL NOT NULL,
                    lucidity REAL NOT NULL,
                    recurring REAL NOT NULL,
                    shadow REAL NOT NULL,
                    dc_dt REAL,
                    d2c_dt2 REAL,
                    dream_state TEXT NOT NULL,
                    components TEXT NOT NULL,
                    insights TEXT NOT NULL
                )
            """)
            
            # Symbol Tracking table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS DreamSymbols (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dream_id INTEGER REFERENCES Dreams(id),
                    symbol TEXT NOT NULL,
                    context TEXT,
                    emotional_charge REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Sleep Patterns table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS SleepPatterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    date DATE NOT NULL,
                    bedtime TIME,
                    wake_time TIME,
                    sleep_quality REAL,
                    dream_recall_count INTEGER DEFAULT 0,
                    notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, date)
                )
            """)
            
            # User Insights table (AI-generated insights over time)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS UserInsights (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    date DATE NOT NULL,
                    insight_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    confidence REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_dreams_user_date ON Dreams(user_id, dream_date)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_analysis_dream ON DreamAnalysis(dream_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_symbols_dream ON DreamSymbols(dream_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sleep_user_date ON SleepPatterns(user_id, date)")
            
            conn.commit()
    
    def save_dream(self, user_id: str, narrative: str, dream_date: datetime,
                   title: Optional[str] = None, lucid: bool = False,
                   sleep_quality: Optional[float] = None, 
                   sleep_hours: Optional[float] = None,
                   tags: Optional[List[str]] = None) -> int:
        """Save a new dream entry"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            tags_json = json.dumps(tags) if tags else None
            
            cursor.execute("""
                INSERT INTO Dreams 
                (user_id, dream_date, title, narrative, lucid, sleep_quality, sleep_hours, tags)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (user_id, dream_date.date(), title, narrative, lucid, 
                  sleep_quality, sleep_hours, tags_json))
            
            conn.commit()
            return cursor.lastrowid
    
    def save_analysis(self, dream_id: int, analysis_result: Dict) -> int:
        """Save dream analysis results"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO DreamAnalysis
                (dream_id, coherence, psi, rho, q_raw, q_opt, f, lucidity, 
                 recurring, shadow, dc_dt, d2c_dt2, dream_state, components, insights)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                dream_id,
                analysis_result['coherence'],
                analysis_result['variables']['psi'],
                analysis_result['variables']['rho'],
                analysis_result['variables']['q_raw'],
                analysis_result['q_opt'],
                analysis_result['variables']['f'],
                analysis_result['variables']['lucidity'],
                analysis_result['variables']['recurring'],
                analysis_result['variables']['shadow'],
                analysis_result['dc_dt'],
                analysis_result['d2c_dt2'],
                analysis_result['dream_state'],
                json.dumps(analysis_result['components']),
                json.dumps(analysis_result['insights'])
            ))
            
            conn.commit()
            return cursor.lastrowid
    
    def save_symbols(self, dream_id: int, symbols: List[Tuple[str, str, float]]):
        """Save dream symbols"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            for symbol, context, charge in symbols:
                cursor.execute("""
                    INSERT INTO DreamSymbols (dream_id, symbol, context, emotional_charge)
                    VALUES (?, ?, ?, ?)
                """, (dream_id, symbol, context, charge))
            
            conn.commit()
    
    def get_user_dreams(self, user_id: str, days: int = 30) -> pd.DataFrame:
        """Get recent dreams for a user"""
        with self.get_connection() as conn:
            query = """
                SELECT d.*, a.coherence, a.dream_state, a.insights
                FROM Dreams d
                LEFT JOIN DreamAnalysis a ON d.id = a.dream_id
                WHERE d.user_id = ?
                AND d.dream_date >= date('now', '-{} days')
                ORDER BY d.dream_date DESC
            """.format(days)
            
            df = pd.read_sql_query(query, conn, params=(user_id,))
            
            # Parse JSON fields
            if not df.empty:
                df['tags'] = df['tags'].apply(lambda x: json.loads(x) if x else [])
                df['insights'] = df['insights'].apply(lambda x: json.loads(x) if x else [])
            
            return df
    
    def get_coherence_timeline(self, user_id: str, days: int = 30) -> pd.DataFrame:
        """Get coherence timeline for visualization"""
        with self.get_connection() as conn:
            query = """
                SELECT 
                    d.dream_date,
                    a.coherence,
                    a.psi,
                    a.rho,
                    a.q_opt,
                    a.f,
                    a.lucidity,
                    a.shadow,
                    a.dream_state
                FROM Dreams d
                JOIN DreamAnalysis a ON d.id = a.dream_id
                WHERE d.user_id = ?
                AND d.dream_date >= date('now', '-{} days')
                ORDER BY d.dream_date
            """.format(days)
            
            return pd.read_sql_query(query, conn, params=(user_id,))
    
    def get_symbol_frequency(self, user_id: str, days: int = 30) -> pd.DataFrame:
        """Get symbol frequency analysis"""
        with self.get_connection() as conn:
            query = """
                SELECT 
                    s.symbol,
                    COUNT(*) as frequency,
                    AVG(s.emotional_charge) as avg_charge,
                    MAX(d.dream_date) as last_seen
                FROM DreamSymbols s
                JOIN Dreams d ON s.dream_id = d.id
                WHERE d.user_id = ?
                AND d.dream_date >= date('now', '-{} days')
                GROUP BY s.symbol
                ORDER BY frequency DESC
            """.format(days)
            
            return pd.read_sql_query(query, conn, params=(user_id,))
    
    def get_dream_states_distribution(self, user_id: str) -> Dict[str, int]:
        """Get distribution of dream states"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT a.dream_state, COUNT(*) as count
                FROM DreamAnalysis a
                JOIN Dreams d ON a.dream_id = d.id
                WHERE d.user_id = ?
                GROUP BY a.dream_state
            """, (user_id,))
            
            return dict(cursor.fetchall())
    
    def save_sleep_pattern(self, user_id: str, date: datetime,
                          bedtime: Optional[datetime] = None,
                          wake_time: Optional[datetime] = None,
                          quality: Optional[float] = None,
                          dream_count: int = 0,
                          notes: Optional[str] = None):
        """Save or update sleep pattern data"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO SleepPatterns
                (user_id, date, bedtime, wake_time, sleep_quality, dream_recall_count, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (user_id, date.date(), 
                  bedtime.time() if bedtime else None,
                  wake_time.time() if wake_time else None,
                  quality, dream_count, notes))
            
            conn.commit()
    
    def get_sleep_coherence_correlation(self, user_id: str) -> pd.DataFrame:
        """Analyze correlation between sleep patterns and dream coherence"""
        with self.get_connection() as conn:
            query = """
                SELECT 
                    sp.date,
                    sp.sleep_quality,
                    sp.dream_recall_count,
                    AVG(a.coherence) as avg_coherence,
                    COUNT(d.id) as dream_count
                FROM SleepPatterns sp
                LEFT JOIN Dreams d ON sp.user_id = d.user_id AND sp.date = d.dream_date
                LEFT JOIN DreamAnalysis a ON d.id = a.dream_id
                WHERE sp.user_id = ?
                GROUP BY sp.date
                ORDER BY sp.date DESC
                LIMIT 30
            """
            
            return pd.read_sql_query(query, conn, params=(user_id,))
    
    def save_user_insight(self, user_id: str, insight_type: str, 
                         content: str, confidence: float = 0.8):
        """Save AI-generated user insight"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO UserInsights (user_id, date, insight_type, content, confidence)
                VALUES (?, date('now'), ?, ?, ?)
            """, (user_id, insight_type, content, confidence))
            
            conn.commit()
    
    def get_recent_insights(self, user_id: str, limit: int = 10) -> List[Dict]:
        """Get recent insights for user"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM UserInsights
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (user_id, limit))
            
            return [dict(row) for row in cursor.fetchall()]