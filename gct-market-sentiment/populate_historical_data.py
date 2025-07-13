#!/usr/bin/env python3
"""
Populate database with historical data from the past week
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, date, timedelta
import numpy as np
from database_manager import MarketDatabase
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HistoricalDataPopulator:
    def __init__(self):
        self.db = MarketDatabase()
        self.watchlist = [
            # Market Indices
            'SPY', 'QQQ', 'DIA', 'IWM',
            # Magnificent 7
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA',
            # High volatility from your dashboard
            'CNC', 'ANET', 'ACN', 'CZR', 'AXON', 'BBY', 'AVGO', 'AMD', 
            'TECH', 'BA', 'CCL', 'APO', 'BLDR', 'CHTR', 'AES', 'APTV',
            'BX', 'ALGN', 'AMAT', 'ALB', 'AMGN', 'BG', 'ADSK', 'KMX',
            'CF', 'ADBE', 'BAX', 'APA', 'BF.B', 'ADM',
            # Additional tech stocks
            'CRM', 'NFLX', 'PYPL', 'SQ', 'ROKU', 'COIN', 'PLTR', 'SOFI',
            # Financial
            'JPM', 'BAC', 'GS', 'MS', 'KKR',
            # Energy
            'XOM', 'CVX', 'COP', 'SLB', 'OXY'
        ]
        
    def calculate_coherence(self, price_data, volume_data):
        """Calculate coherence based on price stability and volume patterns"""
        if len(price_data) < 2:
            return 0.5
            
        # Price volatility (inverse gives coherence)
        returns = np.diff(price_data) / price_data[:-1]
        volatility = np.std(returns) if len(returns) > 0 else 0
        
        # Volume consistency
        if len(volume_data) > 1:
            volume_cv = np.std(volume_data) / np.mean(volume_data) if np.mean(volume_data) > 0 else 1
        else:
            volume_cv = 1
            
        # Coherence increases with lower volatility and consistent volume
        coherence = 1 / (1 + volatility * 10) * (1 / (1 + volume_cv))
        return min(max(coherence, 0.1), 0.9)
    
    def calculate_truth_cost(self, coherence, price_change, volume_ratio):
        """Calculate truth cost based on coherence and market stress"""
        # Higher price changes with low coherence = higher truth cost
        stress = abs(price_change) / 100  # Convert percentage to ratio
        
        # Volume spike indicates effort to maintain trend
        volume_stress = max(0, volume_ratio - 1) * 0.1
        
        # Truth cost increases with stress and decreases with coherence
        truth_cost = (stress + volume_stress) * (1 - coherence)
        return min(max(truth_cost, 0.05), 0.8)
    
    def detect_emotion(self, coherence, truth_cost, price_change):
        """Detect market emotion"""
        if truth_cost > 0.4:
            return 'EXTREME_FEAR' if price_change < 0 else 'EXTREME_GREED'
        elif coherence < 0.3:
            return 'FEAR'
        elif coherence > 0.6:
            return 'NEUTRAL'
        elif price_change > 2:
            return 'GREED'
        elif price_change < -2:
            return 'FEAR'
        else:
            return 'NEUTRAL'
    
    def populate_week_data(self):
        """Populate database with past week's data"""
        end_date = date.today()
        start_date = end_date - timedelta(days=7)
        
        logger.info(f"Fetching data from {start_date} to {end_date}")
        
        # Process each day
        current_date = start_date
        while current_date <= end_date:
            # Skip weekends
            if current_date.weekday() >= 5:
                current_date += timedelta(days=1)
                continue
                
            logger.info(f"Processing {current_date}")
            
            daily_stocks = []
            successful_fetches = 0
            
            for symbol in self.watchlist:
                try:
                    # Fetch data using yfinance
                    ticker = yf.Ticker(symbol)
                    
                    # Get historical data
                    hist = ticker.history(
                        start=current_date - timedelta(days=30),
                        end=current_date + timedelta(days=1)
                    )
                    
                    if hist.empty:
                        logger.warning(f"No data for {symbol}")
                        continue
                    
                    # Get specific day's data
                    if current_date.strftime('%Y-%m-%d') in hist.index.strftime('%Y-%m-%d'):
                        day_idx = hist.index.strftime('%Y-%m-%d') == current_date.strftime('%Y-%m-%d')
                        day_data = hist[day_idx].iloc[0]
                        
                        # Calculate metrics
                        recent_prices = hist['Close'].values[-20:]
                        recent_volumes = hist['Volume'].values[-20:]
                        
                        # Price changes
                        price = float(day_data['Close'])
                        if len(hist) >= 2:
                            prev_close = float(hist['Close'].iloc[-2])
                            day_change_pct = ((price - prev_close) / prev_close) * 100
                        else:
                            day_change_pct = 0
                            
                        # Month change
                        if len(hist) >= 20:
                            month_ago_price = float(hist['Close'].iloc[-20])
                            month_change_pct = ((price - month_ago_price) / month_ago_price) * 100
                        else:
                            month_change_pct = 0
                        
                        # Volume ratio
                        avg_volume = np.mean(recent_volumes) if len(recent_volumes) > 0 else day_data['Volume']
                        volume_ratio = day_data['Volume'] / avg_volume if avg_volume > 0 else 1
                        
                        # Calculate metrics
                        coherence = self.calculate_coherence(recent_prices, recent_volumes)
                        truth_cost = self.calculate_truth_cost(coherence, day_change_pct, volume_ratio)
                        emotion = self.detect_emotion(coherence, truth_cost, day_change_pct)
                        
                        # Determine sector (simplified)
                        info = ticker.info
                        sector = info.get('sector', 'Unknown')
                        
                        stock_data = {
                            'date': current_date,
                            'symbol': symbol,
                            'price': round(price, 2),
                            'day_change_pct': round(day_change_pct, 2),
                            'month_change_pct': round(month_change_pct, 2),
                            'coherence': round(coherence, 3),
                            'truth_cost': round(truth_cost, 3),
                            'emotion': emotion,
                            'sector': sector,
                            'category': self.categorize_stock(symbol)
                        }
                        
                        daily_stocks.append(stock_data)
                        successful_fetches += 1
                        
                        # Check for alerts
                        if coherence > 0.7 or truth_cost > 0.4 or emotion in ['EXTREME_FEAR', 'EXTREME_GREED']:
                            alert = {
                                'date': current_date,
                                'symbol': symbol,
                                'alert_type': 'HIGH_COHERENCE' if coherence > 0.7 else 'EXTREME_EMOTION',
                                'message': f"{symbol}: {emotion} detected - coherence: {coherence:.3f}",
                                'severity': 'CRITICAL' if emotion.startswith('EXTREME') else 'WARNING',
                                'coherence': coherence,
                                'truth_cost': truth_cost
                            }
                            self.db.store_alert(alert)
                        
                        # Check for options signals
                        if coherence > 0.6 and truth_cost < 0.2 and day_change_pct > 0:
                            signal = {
                                'date': current_date,
                                'symbol': symbol,
                                'signal_type': 'CALL',
                                'signal_strength': 'STRONG BUY' if coherence > 0.7 else 'BUY',
                                'coherence': coherence,
                                'price': price,
                                'day_change_pct': day_change_pct,
                                'month_change_pct': month_change_pct
                            }
                            self.db.store_options_signal(signal)
                        elif coherence < 0.3 and day_change_pct < 0:
                            signal = {
                                'date': current_date,
                                'symbol': symbol,
                                'signal_type': 'PUT',
                                'signal_strength': 'STRONG SELL' if coherence < 0.25 else 'SELL',
                                'coherence': coherence,
                                'price': price,
                                'day_change_pct': day_change_pct,
                                'month_change_pct': month_change_pct
                            }
                            self.db.store_options_signal(signal)
                            
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")
                    continue
            
            # Store daily data
            if daily_stocks:
                self.db.store_stock_data(daily_stocks, current_date)
                
                # Calculate and store market summary
                market_summary = self.calculate_market_summary(daily_stocks, current_date)
                self.db.store_market_summary(market_summary)
                
                logger.info(f"Stored {successful_fetches} stocks for {current_date}")
            
            current_date += timedelta(days=1)
            
        logger.info("Historical data population complete!")
    
    def categorize_stock(self, symbol):
        """Categorize stock"""
        mag7 = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA']
        indices = ['SPY', 'QQQ', 'DIA', 'IWM']
        
        if symbol in mag7:
            return 'Magnificent 7'
        elif symbol in indices:
            return 'Market Index'
        else:
            return 'Individual Stock'
    
    def calculate_market_summary(self, stocks_data, analysis_date):
        """Calculate market summary from daily stocks"""
        coherences = [s['coherence'] for s in stocks_data]
        truth_costs = [s['truth_cost'] for s in stocks_data]
        
        avg_coherence = np.mean(coherences)
        avg_truth_cost = np.mean(truth_costs)
        
        # Fear & Greed Index (simplified)
        fear_greed_index = avg_coherence
        
        # Market health
        if avg_coherence > 0.5 and avg_truth_cost < 0.2:
            market_health = 'HEALTHY'
        elif avg_coherence < 0.3 or avg_truth_cost > 0.4:
            market_health = 'STRESSED'
        else:
            market_health = 'NEUTRAL'
        
        return {
            'date': analysis_date,
            'avg_coherence': round(avg_coherence, 3),
            'avg_truth_cost': round(avg_truth_cost, 3),
            'market_health': market_health,
            'fear_greed_index': round(fear_greed_index, 3),
            'high_coherence_count': sum(1 for c in coherences if c > 0.6),
            'high_truth_cost_warnings': sum(1 for t in truth_costs if t > 0.4),
            'total_stocks_analyzed': len(stocks_data)
        }


if __name__ == "__main__":
    populator = HistoricalDataPopulator()
    populator.populate_week_data()
    
    # Show summary
    db = MarketDatabase()
    summaries = db.get_market_summary_history(7)
    print(f"\nStored {len(summaries)} days of market summaries")
    
    for summary in summaries:
        print(f"{summary['date']}: {summary['market_health']} - "
              f"Coherence: {summary['avg_coherence']:.3f}, "
              f"Truth Cost: {summary['avg_truth_cost']:.3f}")
    
    db.close()