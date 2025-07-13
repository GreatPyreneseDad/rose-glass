#!/usr/bin/env python3
"""
GCT Market Sentiment Database Manager
Stores daily market analysis data for historical tracking
"""

import sqlite3
import json
from datetime import datetime, date
from pathlib import Path
import logging
from typing import Dict, List, Any, Optional

class MarketDatabase:
    def __init__(self, db_path: str = "market_analysis.db"):
        self.db_path = Path(db_path)
        self.conn = None
        self.init_database()
        
    def init_database(self):
        """Initialize database with required tables"""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        
        # Create tables
        self.conn.executescript("""
        -- Daily market summary
        CREATE TABLE IF NOT EXISTS market_summary (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date DATE UNIQUE NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            avg_coherence REAL,
            avg_truth_cost REAL,
            market_health TEXT,
            fear_greed_index REAL,
            high_coherence_count INTEGER,
            high_truth_cost_warnings INTEGER,
            total_stocks_analyzed INTEGER
        );
        
        -- Individual stock data
        CREATE TABLE IF NOT EXISTS stock_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date DATE NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            symbol TEXT NOT NULL,
            price REAL,
            day_change_pct REAL,
            month_change_pct REAL,
            coherence REAL,
            truth_cost REAL,
            emotion TEXT,
            volatility_rank INTEGER,
            sector TEXT,
            category TEXT,
            UNIQUE(date, symbol)
        );
        
        -- Alerts and signals
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            date DATE NOT NULL,
            symbol TEXT NOT NULL,
            alert_type TEXT NOT NULL,
            message TEXT,
            severity TEXT,
            coherence REAL,
            truth_cost REAL
        );
        
        -- Options signals
        CREATE TABLE IF NOT EXISTS options_signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date DATE NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            symbol TEXT NOT NULL,
            signal_type TEXT NOT NULL, -- 'CALL' or 'PUT'
            signal_strength TEXT NOT NULL, -- 'STRONG BUY', 'BUY', 'WEAK', etc.
            coherence REAL,
            price REAL,
            day_change_pct REAL,
            month_change_pct REAL,
            UNIQUE(date, symbol, signal_type)
        );
        
        -- Sector performance
        CREATE TABLE IF NOT EXISTS sector_performance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date DATE NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            sector TEXT NOT NULL,
            avg_coherence REAL,
            avg_truth_cost REAL,
            avg_day_change REAL,
            avg_month_change REAL,
            top_performer_symbol TEXT,
            top_performer_gain REAL,
            UNIQUE(date, sector)
        );
        
        -- Create indexes for performance
        CREATE INDEX IF NOT EXISTS idx_stock_date ON stock_data(date);
        CREATE INDEX IF NOT EXISTS idx_stock_symbol ON stock_data(symbol);
        CREATE INDEX IF NOT EXISTS idx_alerts_date ON alerts(date);
        CREATE INDEX IF NOT EXISTS idx_options_date ON options_signals(date);
        CREATE INDEX IF NOT EXISTS idx_sector_date ON sector_performance(date);
        """)
        
        self.conn.commit()
    
    def store_market_summary(self, data: Dict[str, Any]) -> int:
        """Store daily market summary"""
        query = """
        INSERT OR REPLACE INTO market_summary 
        (date, avg_coherence, avg_truth_cost, market_health, fear_greed_index,
         high_coherence_count, high_truth_cost_warnings, total_stocks_analyzed)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        cursor = self.conn.cursor()
        cursor.execute(query, (
            data.get('date', date.today()),
            data.get('avg_coherence'),
            data.get('avg_truth_cost'),
            data.get('market_health'),
            data.get('fear_greed_index'),
            data.get('high_coherence_count'),
            data.get('high_truth_cost_warnings'),
            data.get('total_stocks_analyzed')
        ))
        self.conn.commit()
        return cursor.lastrowid
    
    def store_stock_data(self, stocks: List[Dict[str, Any]], analysis_date: Optional[date] = None) -> int:
        """Store multiple stock records"""
        if analysis_date is None:
            analysis_date = date.today()
            
        query = """
        INSERT OR REPLACE INTO stock_data 
        (date, symbol, price, day_change_pct, month_change_pct, 
         coherence, truth_cost, emotion, volatility_rank, sector, category)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        cursor = self.conn.cursor()
        inserted = 0
        
        for stock in stocks:
            try:
                cursor.execute(query, (
                    analysis_date,
                    stock.get('symbol'),
                    stock.get('price'),
                    stock.get('day_change_pct'),
                    stock.get('month_change_pct'),
                    stock.get('coherence'),
                    stock.get('truth_cost'),
                    stock.get('emotion'),
                    stock.get('volatility_rank'),
                    stock.get('sector'),
                    stock.get('category')
                ))
                inserted += 1
            except Exception as e:
                logging.error(f"Error storing {stock.get('symbol')}: {e}")
                
        self.conn.commit()
        return inserted
    
    def store_alert(self, alert: Dict[str, Any]) -> int:
        """Store an alert"""
        query = """
        INSERT INTO alerts 
        (date, symbol, alert_type, message, severity, coherence, truth_cost)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        
        cursor = self.conn.cursor()
        cursor.execute(query, (
            alert.get('date', date.today()),
            alert.get('symbol'),
            alert.get('alert_type'),
            alert.get('message'),
            alert.get('severity'),
            alert.get('coherence'),
            alert.get('truth_cost')
        ))
        self.conn.commit()
        return cursor.lastrowid
    
    def store_options_signal(self, signal: Dict[str, Any]) -> int:
        """Store options trading signal"""
        query = """
        INSERT OR REPLACE INTO options_signals
        (date, symbol, signal_type, signal_strength, coherence, 
         price, day_change_pct, month_change_pct)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        cursor = self.conn.cursor()
        cursor.execute(query, (
            signal.get('date', date.today()),
            signal.get('symbol'),
            signal.get('signal_type'),
            signal.get('signal_strength'),
            signal.get('coherence'),
            signal.get('price'),
            signal.get('day_change_pct'),
            signal.get('month_change_pct')
        ))
        self.conn.commit()
        return cursor.lastrowid
    
    def get_historical_data(self, symbol: str, days: int = 30) -> List[Dict]:
        """Get historical data for a symbol"""
        query = """
        SELECT * FROM stock_data 
        WHERE symbol = ? AND date >= date('now', '-' || ? || ' days')
        ORDER BY date DESC
        """
        
        cursor = self.conn.cursor()
        cursor.execute(query, (symbol, days))
        
        return [dict(row) for row in cursor.fetchall()]
    
    def get_market_summary_history(self, days: int = 30) -> List[Dict]:
        """Get historical market summaries"""
        query = """
        SELECT * FROM market_summary 
        WHERE date >= date('now', '-' || ? || ' days')
        ORDER BY date DESC
        """
        
        cursor = self.conn.cursor()
        cursor.execute(query, (days,))
        
        return [dict(row) for row in cursor.fetchall()]
    
    def get_top_movers(self, analysis_date: Optional[date] = None, limit: int = 10) -> Dict[str, List[Dict]]:
        """Get top gainers and losers for a date"""
        if analysis_date is None:
            analysis_date = date.today()
            
        gainers_query = """
        SELECT * FROM stock_data 
        WHERE date = ? AND day_change_pct IS NOT NULL
        ORDER BY day_change_pct DESC
        LIMIT ?
        """
        
        losers_query = """
        SELECT * FROM stock_data 
        WHERE date = ? AND day_change_pct IS NOT NULL
        ORDER BY day_change_pct ASC
        LIMIT ?
        """
        
        cursor = self.conn.cursor()
        
        cursor.execute(gainers_query, (analysis_date, limit))
        gainers = [dict(row) for row in cursor.fetchall()]
        
        cursor.execute(losers_query, (analysis_date, limit))
        losers = [dict(row) for row in cursor.fetchall()]
        
        return {'gainers': gainers, 'losers': losers}
    
    def get_coherence_trends(self, symbols: List[str], days: int = 30) -> Dict[str, List[Dict]]:
        """Get coherence trends for multiple symbols"""
        query = """
        SELECT date, symbol, coherence, truth_cost 
        FROM stock_data 
        WHERE symbol IN ({}) AND date >= date('now', '-' || ? || ' days')
        ORDER BY symbol, date
        """.format(','.join(['?'] * len(symbols)))
        
        cursor = self.conn.cursor()
        cursor.execute(query, (*symbols, days))
        
        results = {}
        for row in cursor.fetchall():
            symbol = row['symbol']
            if symbol not in results:
                results[symbol] = []
            results[symbol].append({
                'date': row['date'],
                'coherence': row['coherence'],
                'truth_cost': row['truth_cost']
            })
            
        return results
    
    def cleanup_old_data(self, days_to_keep: int = 365):
        """Remove data older than specified days"""
        queries = [
            "DELETE FROM market_summary WHERE date < date('now', '-' || ? || ' days')",
            "DELETE FROM stock_data WHERE date < date('now', '-' || ? || ' days')",
            "DELETE FROM alerts WHERE date < date('now', '-' || ? || ' days')",
            "DELETE FROM options_signals WHERE date < date('now', '-' || ? || ' days')",
            "DELETE FROM sector_performance WHERE date < date('now', '-' || ? || ' days')"
        ]
        
        cursor = self.conn.cursor()
        for query in queries:
            cursor.execute(query, (days_to_keep,))
        
        self.conn.commit()
        self.conn.execute("VACUUM")
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()


if __name__ == "__main__":
    # Test the database
    db = MarketDatabase()
    
    # Test data
    test_summary = {
        'avg_coherence': 0.405,
        'avg_truth_cost': 0.175,
        'market_health': 'HEALTHY',
        'fear_greed_index': 0.33,
        'high_coherence_count': 3,
        'high_truth_cost_warnings': 0,
        'total_stocks_analyzed': 50
    }
    
    test_stock = {
        'symbol': 'AAPL',
        'price': 195.89,
        'day_change_pct': 1.25,
        'month_change_pct': 5.67,
        'coherence': 0.456,
        'truth_cost': 0.123,
        'emotion': 'NEUTRAL',
        'sector': 'Technology',
        'category': 'Magnificent 7'
    }
    
    # Store test data
    summary_id = db.store_market_summary(test_summary)
    print(f"Stored market summary with ID: {summary_id}")
    
    stock_count = db.store_stock_data([test_stock])
    print(f"Stored {stock_count} stock records")
    
    # Retrieve data
    history = db.get_historical_data('AAPL', 7)
    print(f"Retrieved {len(history)} historical records for AAPL")
    
    db.close()