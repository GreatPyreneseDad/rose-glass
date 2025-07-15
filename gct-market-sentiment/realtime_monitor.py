#!/usr/bin/env python3
"""
Real-time Market Monitoring System
Integrates coherence, truth cost, and emotional superposition
Updates dashboard every 2-5 minutes
"""

import os
import sys
import time
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import threading
import schedule
import subprocess
from collections import deque
import requests
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('realtime_monitor.log'),
        logging.StreamHandler()
    ]
)

class MarketMonitor:
    def __init__(self):
        # Core tickers to monitor
        self.mag7 = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA']
        self.indices = ['SPY', 'QQQ', 'IWM', 'DIA']  # S&P 500, Nasdaq, Russell 2000, Dow
        
        # Will be populated with most volatile stocks
        self.volatile_stocks = []
        
        # Data storage
        self.market_data = {}
        self.coherence_history = deque(maxlen=288)  # 24 hours at 5-min intervals
        self.alerts = deque(maxlen=100)
        
        # Analysis results
        self.latest_coherence = {}
        self.latest_truth_cost = {}
        self.latest_emotions = {}
        
        # Dashboard update flag
        self.last_update = datetime.now()
        
    def get_most_volatile_stocks(self, n=30):
        """Get the most volatile stocks from market"""
        logging.info("Fetching most volatile stocks...")
        
        try:
            # Use S&P 500 components as universe
            sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
            tickers = sp500['Symbol'].str.replace('.', '-').tolist()[:100]  # Top 100 for speed
            
            volatilities = []
            
            for ticker in tickers:
                try:
                    stock = yf.Ticker(ticker)
                    hist = stock.history(period='1mo')
                    if len(hist) > 10:
                        returns = hist['Close'].pct_change().dropna()
                        vol = returns.std() * np.sqrt(252) * 100
                        volatilities.append((ticker, vol))
                except:
                    continue
            
            # Sort by volatility and get top N
            volatilities.sort(key=lambda x: x[1], reverse=True)
            self.volatile_stocks = [t[0] for t in volatilities[:n]]
            
            logging.info(f"Found {len(self.volatile_stocks)} volatile stocks")
            return self.volatile_stocks
            
        except Exception as e:
            logging.error(f"Error getting volatile stocks: {e}")
            # Fallback to predefined volatile stocks
            self.volatile_stocks = ['GME', 'AMC', 'BBBY', 'BB', 'PLTR', 'RIOT', 'MARA', 'COIN']
            return self.volatile_stocks
    
    def fetch_market_data(self, ticker):
        """Fetch latest market data for a ticker"""
        try:
            stock = yf.Ticker(ticker)
            
            # Get intraday data
            hist = stock.history(period='1d', interval='5m')
            if hist.empty:
                return None
            
            # Get daily data for longer-term metrics
            daily = stock.history(period='30d')
            
            # Calculate price changes
            current_price = hist['Close'].iloc[-1]
            
            # Daily change
            if len(hist) > 1:
                day_open = daily['Open'].iloc[-1] if len(daily) > 0 else hist['Open'].iloc[0]
                daily_change = ((current_price - day_open) / day_open) * 100
            else:
                daily_change = 0
            
            # Monthly change
            if len(daily) >= 20:
                month_ago_price = daily['Close'].iloc[-20]
                monthly_change = ((current_price - month_ago_price) / month_ago_price) * 100
            else:
                monthly_change = 0
            
            return {
                'ticker': ticker,
                'current_price': current_price,
                'daily_change': daily_change,
                'monthly_change': monthly_change,
                'intraday_data': hist,
                'daily_data': daily,
                'volume': hist['Volume'].iloc[-1],
                'timestamp': datetime.now()
            }
        except Exception as e:
            logging.error(f"Error fetching data for {ticker}: {e}")
            return None
    
    def calculate_realtime_coherence(self, data):
        """Calculate coherence from market data"""
        try:
            df = data['daily_data'].copy()
            
            # Calculate returns
            df['returns'] = df['Close'].pct_change()
            
            # Trend strength
            sma5 = df['Close'].rolling(5).mean()
            sma20 = df['Close'].rolling(20).mean()
            trend_strength = ((sma5.iloc[-1] - sma20.iloc[-1]) / sma20.iloc[-1]) if sma20.iloc[-1] > 0 else 0
            
            # Volatility
            volatility = df['returns'].std() * np.sqrt(252) * 100
            
            # Volume ratio
            recent_volume = df['Volume'].tail(5).mean()
            avg_volume = df['Volume'].mean()
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1
            
            # Momentum
            momentum = df['returns'].tail(5).mean()
            
            # Calculate coherence
            psi = min(1.0, abs(trend_strength) * 10)
            rho = 1 / (1 + volatility / 20)
            q_raw = min(1.0, abs(momentum) * 50)
            f = min(1.0, volume_ratio * 0.5)
            
            coherence = (psi * 0.3 + rho * 0.3 + q_raw * 0.2 + f * 0.2)
            
            return {
                'coherence': coherence,
                'psi': psi,
                'rho': rho,
                'q_raw': q_raw,
                'f': f,
                'trend_strength': trend_strength,
                'volatility': volatility
            }
        except Exception as e:
            logging.error(f"Error calculating coherence: {e}")
            return None
    
    def calculate_realtime_truth_cost(self, data, coherence_data):
        """Calculate truth cost from market data"""
        try:
            df = data['daily_data'].copy()
            
            # Coherence deviation from optimal (0.6)
            coherence_deviation = abs(coherence_data['coherence'] - 0.6)
            
            # Wisdom-emotion imbalance
            we_ratio = coherence_data['rho'] / (coherence_data['q_raw'] + 0.1)
            we_imbalance_cost = (1 - we_ratio) * coherence_data['q_raw'] if we_ratio < 1 else 0
            
            # Volatility cost
            volatility_cost = coherence_data['volatility'] / 200
            
            # Truth cost
            truth_cost = (
                coherence_deviation * 0.4 +
                we_imbalance_cost * 0.3 +
                volatility_cost * 0.3
            )
            
            return {
                'truth_cost': truth_cost,
                'coherence_deviation': coherence_deviation,
                'we_imbalance_cost': we_imbalance_cost,
                'volatility_cost': volatility_cost,
                'wisdom_emotion_ratio': we_ratio
            }
        except Exception as e:
            logging.error(f"Error calculating truth cost: {e}")
            return None
    
    def calculate_realtime_emotions(self, data):
        """Calculate emotional state from market data"""
        try:
            df = data['intraday_data'].copy()
            
            # Calculate returns
            df['returns'] = df['Close'].pct_change()
            
            # Momentum and volatility
            momentum = df['returns'].rolling(12).mean().iloc[-1]
            volatility = df['returns'].rolling(12).std().iloc[-1]
            
            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1]
            
            # Fear/Greed calculation
            greed_component = (momentum + 0.001) / 0.002
            fear_component = volatility * 1000
            
            fear_greed = (
                greed_component * 0.3 +
                (current_rsi / 100) * 0.3 +
                (1 - min(1, fear_component)) * 0.4
            ).clip(0, 1)
            
            # Emotional state
            if fear_greed > 0.8:
                emotional_state = "EXTREME_GREED"
            elif fear_greed > 0.6:
                emotional_state = "GREED"
            elif fear_greed < 0.2:
                emotional_state = "EXTREME_FEAR"
            elif fear_greed < 0.4:
                emotional_state = "FEAR"
            else:
                emotional_state = "NEUTRAL"
            
            return {
                'fear_greed_index': fear_greed,
                'emotional_state': emotional_state,
                'rsi': current_rsi,
                'momentum': momentum,
                'intraday_volatility': volatility
            }
        except Exception as e:
            logging.error(f"Error calculating emotions: {e}")
            return None
    
    def generate_alerts(self, ticker, coherence, truth_cost, emotions):
        """Generate alerts based on analysis"""
        alerts = []
        
        # Coherence alerts
        if coherence and coherence['coherence'] > 0.7:
            alerts.append({
                'type': 'HIGH_COHERENCE',
                'ticker': ticker,
                'message': f"{ticker}: High coherence detected ({coherence['coherence']:.3f})",
                'severity': 'info',
                'timestamp': datetime.now()
            })
        
        # Truth cost alerts
        if truth_cost and truth_cost['truth_cost'] > 0.5:
            alerts.append({
                'type': 'HIGH_TRUTH_COST',
                'ticker': ticker,
                'message': f"{ticker}: Unsustainable pattern detected (cost: {truth_cost['truth_cost']:.3f})",
                'severity': 'warning',
                'timestamp': datetime.now()
            })
        
        # Emotional alerts
        if emotions:
            if emotions['emotional_state'] == 'EXTREME_GREED':
                alerts.append({
                    'type': 'EXTREME_GREED',
                    'ticker': ticker,
                    'message': f"{ticker}: Extreme greed detected - potential reversal",
                    'severity': 'critical',
                    'timestamp': datetime.now()
                })
            elif emotions['emotional_state'] == 'EXTREME_FEAR':
                alerts.append({
                    'type': 'EXTREME_FEAR',
                    'ticker': ticker,
                    'message': f"{ticker}: Extreme fear detected - potential bounce",
                    'severity': 'critical',
                    'timestamp': datetime.now()
                })
        
        return alerts
    
    def save_historical_snapshot(self, dashboard_data):
        """Save a historical snapshot of the data"""
        try:
            # Create historical directory if it doesn't exist
            os.makedirs('historical_snapshots', exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now()
            filename = timestamp.strftime('historical_snapshots/snapshot_%Y%m%d_%H%M%S.json')
            
            # Save the snapshot
            with open(filename, 'w') as f:
                json.dump(dashboard_data, f, indent=2, default=str)
            
            # Also append to a master historical file for easy analysis
            historical_entry = {
                'timestamp': timestamp.isoformat(),
                'market_health': dashboard_data['market_summary']['market_health'],
                'avg_coherence': dashboard_data['market_summary']['avg_coherence'],
                'avg_truth_cost': dashboard_data['market_summary']['avg_truth_cost'],
                'avg_fear_greed': dashboard_data['market_summary']['avg_fear_greed'],
                'extreme_emotions_count': dashboard_data['market_summary']['extreme_emotions'],
                'alert_count': len([a for a in dashboard_data['alerts'] if a.get('severity') == 'critical'])
            }
            
            # Append to master historical file
            master_file = 'historical_snapshots/master_history.jsonl'
            with open(master_file, 'a') as f:
                f.write(json.dumps(historical_entry) + '\n')
                
            logging.info(f"Saved historical snapshot: {filename}")
            
        except Exception as e:
            logging.error(f"Error saving historical snapshot: {e}")
    
    def update_dashboard(self):
        """Update the HTML dashboard with latest data"""
        try:
            # Prepare data for dashboard
            dashboard_data = {
                'timestamp': datetime.now().isoformat(),
                'market_summary': self.calculate_market_summary(),
                'coherence_data': self.latest_coherence,
                'truth_cost_data': self.latest_truth_cost,
                'emotional_data': self.latest_emotions,
                'alerts': list(self.alerts)[-20:],  # Last 20 alerts
                'indices': {},
                'mag7': {},
                'volatile': {}
            }
            
            # Organize by category
            for ticker, data in self.latest_coherence.items():
                if ticker in self.indices:
                    dashboard_data['indices'][ticker] = {
                        'coherence': data,
                        'truth_cost': self.latest_truth_cost.get(ticker, {}),
                        'emotions': self.latest_emotions.get(ticker, {})
                    }
                elif ticker in self.mag7:
                    dashboard_data['mag7'][ticker] = {
                        'coherence': data,
                        'truth_cost': self.latest_truth_cost.get(ticker, {}),
                        'emotions': self.latest_emotions.get(ticker, {})
                    }
                else:
                    dashboard_data['volatile'][ticker] = {
                        'coherence': data,
                        'truth_cost': self.latest_truth_cost.get(ticker, {}),
                        'emotions': self.latest_emotions.get(ticker, {})
                    }
            
            # Save data
            with open('realtime_data.json', 'w') as f:
                json.dump(dashboard_data, f, indent=2, default=str)
            
            # Save historical snapshot
            self.save_historical_snapshot(dashboard_data)
            
            # Run the dashboard generator
            subprocess.run([sys.executable, 'realtime_dashboard_generator.py'], 
                         capture_output=True, text=True)
            
            logging.info("Dashboard updated successfully")
            
        except Exception as e:
            logging.error(f"Error updating dashboard: {e}")
    
    def calculate_market_summary(self):
        """Calculate overall market metrics"""
        if not self.latest_coherence:
            return {}
        
        coherence_values = [v['coherence'] for v in self.latest_coherence.values()]
        truth_costs = [v['truth_cost'] for v in self.latest_truth_cost.values() if v]
        fear_greed_values = [v['fear_greed_index'] for v in self.latest_emotions.values() if v]
        
        return {
            'avg_coherence': np.mean(coherence_values) if coherence_values else 0,
            'avg_truth_cost': np.mean(truth_costs) if truth_costs else 0,
            'avg_fear_greed': np.mean(fear_greed_values) if fear_greed_values else 0.5,
            'high_coherence_count': len([c for c in coherence_values if c > 0.6]),
            'high_truth_cost_count': len([t for t in truth_costs if t > 0.5]),
            'extreme_emotions': len([f for f in fear_greed_values if f > 0.8 or f < 0.2]),
            'market_health': 'HEALTHY' if np.mean(coherence_values) > 0.4 and np.mean(truth_costs) < 0.3 else 'STRESSED'
        }
    
    def run_analysis_cycle(self):
        """Run one complete analysis cycle"""
        logging.info("Starting analysis cycle...")
        
        all_tickers = list(set(self.mag7 + self.indices + self.volatile_stocks))
        
        for ticker in all_tickers:
            try:
                # Fetch data
                data = self.fetch_market_data(ticker)
                if not data:
                    continue
                
                # Run analyses
                coherence = self.calculate_realtime_coherence(data)
                truth_cost = self.calculate_realtime_truth_cost(data, coherence) if coherence else None
                emotions = self.calculate_realtime_emotions(data)
                
                # Store results with price data
                if coherence:
                    coherence['price'] = data['current_price']
                    coherence['daily_change'] = data['daily_change']
                    coherence['monthly_change'] = data['monthly_change']
                    self.latest_coherence[ticker] = coherence
                if truth_cost:
                    self.latest_truth_cost[ticker] = truth_cost
                if emotions:
                    self.latest_emotions[ticker] = emotions
                
                # Generate alerts
                new_alerts = self.generate_alerts(ticker, coherence, truth_cost, emotions)
                for alert in new_alerts:
                    self.alerts.append(alert)
                    if alert['severity'] == 'critical':
                        self.send_push_notification(alert)
                
            except Exception as e:
                logging.error(f"Error analyzing {ticker}: {e}")
        
        # Update dashboard
        self.update_dashboard()
        
        logging.info(f"Analysis cycle completed. Processed {len(all_tickers)} tickers")
    
    def send_push_notification(self, alert):
        """Send push notification for critical alerts"""
        # This would integrate with a service like Pushover, Telegram, or email
        logging.info(f"CRITICAL ALERT: {alert['message']}")
        
        # Example: Write to alert file that could be monitored by external service
        with open('critical_alerts.log', 'a') as f:
            f.write(f"{alert['timestamp']}: {alert['message']}\n")
    
    def start_monitoring(self):
        """Start the monitoring loop"""
        logging.info("Market Monitor starting...")
        
        # Get initial volatile stocks
        self.get_most_volatile_stocks()
        
        # Schedule tasks
        schedule.every(5).minutes.do(self.run_analysis_cycle)
        schedule.every(1).hours.do(self.get_most_volatile_stocks)
        
        # Run initial analysis
        self.run_analysis_cycle()
        
        # Main loop
        while True:
            try:
                schedule.run_pending()
                time.sleep(30)  # Check every 30 seconds
            except KeyboardInterrupt:
                logging.info("Monitoring stopped by user")
                break
            except Exception as e:
                logging.error(f"Error in main loop: {e}")
                time.sleep(60)  # Wait a minute before retrying

def main():
    monitor = MarketMonitor()
    monitor.start_monitoring()

if __name__ == "__main__":
    main()