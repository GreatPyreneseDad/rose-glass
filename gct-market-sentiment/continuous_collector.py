#!/usr/bin/env python3
"""
GCT Continuous Market Data Collector
Runs 24/7 to collect and store market analysis data
"""

import time
import schedule
import logging
from datetime import datetime, date, timedelta
import json
import os
import sys
from pathlib import Path
import signal
import threading
from typing import Dict, List, Any

# Import local modules
sys.path.append(str(Path(__file__).parent))
from database_manager import MarketDatabase
from fetch_and_save_data import fetch_and_analyze_stocks
from coherence_calculator import calculate_coherence
from truth_cost_calculator import calculate_truth_cost
from emotional_superposition import detect_emotion
from realtime_dashboard_generator import generate_dashboard

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('continuous_collector.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ContinuousCollector:
    def __init__(self):
        self.db = MarketDatabase()
        self.running = True
        self.watchlist = self.load_watchlist()
        self.collection_times = ['09:30', '12:00', '15:30', '16:00']  # Market hours EST
        self.dashboard_update_interval = 120  # seconds
        
    def load_watchlist(self) -> List[str]:
        """Load stock watchlist"""
        # Default watchlist
        default_stocks = [
            # Market Indices
            'SPY', 'QQQ', 'DIA', 'IWM',
            # Magnificent 7
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA',
            # High volatility
            'AMD', 'AVGO', 'CRM', 'NFLX', 'ADBE', 'PYPL', 'SQ', 'ROKU',
            # Financial
            'JPM', 'BAC', 'GS', 'MS', 'BX', 'KKR', 'APO',
            # Energy
            'XOM', 'CVX', 'COP', 'SLB', 'OXY',
            # Healthcare
            'JNJ', 'UNH', 'PFE', 'ABBV', 'MRK', 'LLY',
            # Consumer
            'AMZN', 'WMT', 'HD', 'NKE', 'SBUX', 'MCD',
            # Additional volatile stocks
            'COIN', 'PLTR', 'SOFI', 'HOOD', 'DKNG', 'LCID', 'RIVN'
        ]
        
        # Try to load custom watchlist
        watchlist_file = Path('watchlist.json')
        if watchlist_file.exists():
            try:
                with open(watchlist_file, 'r') as f:
                    custom_stocks = json.load(f)
                    return list(set(default_stocks + custom_stocks))
            except Exception as e:
                logger.error(f"Error loading watchlist: {e}")
                
        return default_stocks
    
    def collect_market_data(self):
        """Collect and analyze market data"""
        logger.info("Starting market data collection...")
        
        try:
            # Fetch stock data
            stocks_data = []
            alerts = []
            options_signals = []
            
            for symbol in self.watchlist:
                try:
                    # Fetch price and analysis data
                    stock_info = self.fetch_stock_data(symbol)
                    if stock_info:
                        stocks_data.append(stock_info)
                        
                        # Check for alerts
                        alert = self.check_alerts(stock_info)
                        if alert:
                            alerts.append(alert)
                            
                        # Check for options signals
                        signal = self.check_options_signals(stock_info)
                        if signal:
                            options_signals.append(signal)
                            
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")
                    
            # Calculate market summary
            market_summary = self.calculate_market_summary(stocks_data)
            
            # Store in database
            self.db.store_market_summary(market_summary)
            self.db.store_stock_data(stocks_data)
            
            for alert in alerts:
                self.db.store_alert(alert)
                
            for signal in options_signals:
                self.db.store_options_signal(signal)
                
            logger.info(f"Stored {len(stocks_data)} stocks, {len(alerts)} alerts, {len(options_signals)} signals")
            
            # Generate updated dashboard
            self.update_dashboard(stocks_data, market_summary, alerts)
            
        except Exception as e:
            logger.error(f"Error in data collection: {e}")
    
    def fetch_stock_data(self, symbol: str) -> Dict[str, Any]:
        """Fetch and analyze single stock"""
        try:
            # This is a placeholder - integrate with your actual data fetching
            # In production, use yfinance or your preferred data source
            
            # Mock data for now
            import random
            
            price = random.uniform(50, 500)
            day_change = random.uniform(-5, 5)
            month_change = random.uniform(-20, 20)
            
            # Calculate metrics
            coherence = calculate_coherence(symbol)  # Your coherence calculation
            truth_cost = calculate_truth_cost(symbol)  # Your truth cost calculation
            emotion = detect_emotion(coherence, truth_cost, day_change)
            
            return {
                'symbol': symbol,
                'price': round(price, 2),
                'day_change_pct': round(day_change, 2),
                'month_change_pct': round(month_change, 2),
                'coherence': round(coherence, 3),
                'truth_cost': round(truth_cost, 3),
                'emotion': emotion,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            return None
    
    def calculate_market_summary(self, stocks_data: List[Dict]) -> Dict[str, Any]:
        """Calculate overall market metrics"""
        if not stocks_data:
            return {}
            
        coherences = [s['coherence'] for s in stocks_data if s.get('coherence')]
        truth_costs = [s['truth_cost'] for s in stocks_data if s.get('truth_cost')]
        
        avg_coherence = sum(coherences) / len(coherences) if coherences else 0
        avg_truth_cost = sum(truth_costs) / len(truth_costs) if truth_costs else 0
        
        # Calculate fear & greed index (0-1 scale)
        fear_greed_index = avg_coherence  # Simplified version
        
        # Determine market health
        if avg_coherence > 0.5 and avg_truth_cost < 0.2:
            market_health = 'HEALTHY'
        elif avg_coherence < 0.3 or avg_truth_cost > 0.4:
            market_health = 'STRESSED'
        else:
            market_health = 'NEUTRAL'
            
        high_coherence_count = sum(1 for c in coherences if c > 0.6)
        high_truth_cost_warnings = sum(1 for t in truth_costs if t > 0.4)
        
        return {
            'date': date.today(),
            'avg_coherence': round(avg_coherence, 3),
            'avg_truth_cost': round(avg_truth_cost, 3),
            'market_health': market_health,
            'fear_greed_index': round(fear_greed_index, 3),
            'high_coherence_count': high_coherence_count,
            'high_truth_cost_warnings': high_truth_cost_warnings,
            'total_stocks_analyzed': len(stocks_data)
        }
    
    def check_alerts(self, stock_info: Dict) -> Dict[str, Any]:
        """Check for alert conditions"""
        alerts = []
        
        # High coherence alert
        if stock_info['coherence'] > 0.7:
            return {
                'symbol': stock_info['symbol'],
                'alert_type': 'HIGH_COHERENCE',
                'message': f"{stock_info['symbol']}: High coherence detected ({stock_info['coherence']})",
                'severity': 'INFO',
                'coherence': stock_info['coherence'],
                'truth_cost': stock_info['truth_cost']
            }
            
        # High truth cost alert
        if stock_info['truth_cost'] > 0.4:
            return {
                'symbol': stock_info['symbol'],
                'alert_type': 'HIGH_TRUTH_COST',
                'message': f"{stock_info['symbol']}: High truth cost warning ({stock_info['truth_cost']})",
                'severity': 'WARNING',
                'coherence': stock_info['coherence'],
                'truth_cost': stock_info['truth_cost']
            }
            
        # Extreme emotion alert
        if stock_info['emotion'] in ['EXTREME_FEAR', 'EXTREME_GREED']:
            return {
                'symbol': stock_info['symbol'],
                'alert_type': 'EXTREME_EMOTION',
                'message': f"{stock_info['symbol']}: {stock_info['emotion']} detected",
                'severity': 'CRITICAL',
                'coherence': stock_info['coherence'],
                'truth_cost': stock_info['truth_cost']
            }
            
        return None
    
    def check_options_signals(self, stock_info: Dict) -> Dict[str, Any]:
        """Generate options trading signals"""
        # Call signals
        if (stock_info['coherence'] > 0.6 and 
            stock_info['truth_cost'] < 0.2 and 
            stock_info['day_change_pct'] > 0):
            return {
                'symbol': stock_info['symbol'],
                'signal_type': 'CALL',
                'signal_strength': 'STRONG BUY' if stock_info['coherence'] > 0.7 else 'BUY',
                'coherence': stock_info['coherence'],
                'price': stock_info['price'],
                'day_change_pct': stock_info['day_change_pct'],
                'month_change_pct': stock_info['month_change_pct']
            }
            
        # Put signals
        if (stock_info['coherence'] < 0.3 and 
            stock_info['day_change_pct'] < 0):
            return {
                'symbol': stock_info['symbol'],
                'signal_type': 'PUT',
                'signal_strength': 'STRONG SELL' if stock_info['coherence'] < 0.25 else 'SELL',
                'coherence': stock_info['coherence'],
                'price': stock_info['price'],
                'day_change_pct': stock_info['day_change_pct'],
                'month_change_pct': stock_info['month_change_pct']
            }
            
        return None
    
    def update_dashboard(self, stocks_data: List[Dict], market_summary: Dict, alerts: List[Dict]):
        """Update the HTML dashboard"""
        try:
            # Call your dashboard generator
            generate_dashboard(stocks_data, market_summary, alerts)
            logger.info("Dashboard updated successfully")
        except Exception as e:
            logger.error(f"Error updating dashboard: {e}")
    
    def run_scheduled_collection(self):
        """Run scheduled collections"""
        # Schedule collections at specific times
        for collection_time in self.collection_times:
            schedule.every().day.at(collection_time).do(self.collect_market_data)
            
        # Also collect every 30 minutes during market hours
        schedule.every(30).minutes.do(self.collect_during_market_hours)
        
        # Dashboard updates every 2 minutes
        schedule.every(self.dashboard_update_interval).seconds.do(self.update_dashboard_only)
        
        logger.info(f"Scheduled collections at: {', '.join(self.collection_times)}")
        
    def collect_during_market_hours(self):
        """Only collect during market hours"""
        now = datetime.now()
        # Check if it's a weekday and during market hours (9:30 AM - 4:00 PM EST)
        if now.weekday() < 5:  # Monday = 0, Friday = 4
            hour = now.hour
            if 9 <= hour <= 16:  # Rough market hours check
                self.collect_market_data()
    
    def update_dashboard_only(self):
        """Update dashboard with latest data from database"""
        try:
            # Get latest data from database
            latest_summary = self.db.get_market_summary_history(1)
            if latest_summary:
                # Get today's stock data
                today_stocks = self.db.get_top_movers(date.today(), 50)
                # Update dashboard
                self.update_dashboard(today_stocks.get('gainers', []), latest_summary[0], [])
        except Exception as e:
            logger.error(f"Error updating dashboard: {e}")
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info("Received shutdown signal, cleaning up...")
        self.running = False
        self.db.close()
        sys.exit(0)
    
    def run(self):
        """Main run loop"""
        # Set up signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        # Run initial collection
        self.collect_market_data()
        
        # Set up schedules
        self.run_scheduled_collection()
        
        logger.info("Continuous collector started. Press Ctrl+C to stop.")
        
        # Keep running
        while self.running:
            schedule.run_pending()
            time.sleep(1)


if __name__ == "__main__":
    collector = ContinuousCollector()
    collector.run()