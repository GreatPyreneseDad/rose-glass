"""
Optimized Analysis Pipeline combining all performance improvements
"""

import asyncio
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import numpy as np

from .database import GCTDatabase
from .gct_engine import GCTEngine, GCTVariables, GCTResult
from .optimized_gct_engine import OptimizedGCTEngine, BatchProcessor
from .memory_efficient_pipeline import (
    MemoryEfficientPipeline, 
    CompressedCache,
    MemoryEfficientDataFrame,
    monitor_memory_usage
)

logger = logging.getLogger(__name__)


class OptimizedAnalysisPipeline:
    """High-performance analysis pipeline with all optimizations"""
    
    def __init__(
        self,
        db_path: str = "data/gct_market.db",
        max_workers: int = 4,
        batch_size: int = 100,
        max_memory_mb: int = 1024
    ):
        self.db = GCTDatabase(db_path)
        self.gct_engine = GCTEngine(use_optimized=True)
        self.batch_processor = self.gct_engine.get_batch_processor()
        self.memory_pipeline = MemoryEfficientPipeline(max_memory_mb)
        
        # Caches
        self.analysis_cache = CompressedCache(maxsize=5000)
        self.ticker_cache = {}
        
        # Thread pool for I/O operations
        self.io_executor = ThreadPoolExecutor(max_workers=max_workers)
        # Process pool for CPU-intensive operations
        self.cpu_executor = ProcessPoolExecutor(max_workers=max_workers)
        
        self.batch_size = batch_size
    
    @monitor_memory_usage
    async def analyze_articles_async(self, articles: List[Dict]) -> List[Dict]:
        """Analyze articles asynchronously with all optimizations"""
        # Split into batches
        batches = [
            articles[i:i+self.batch_size] 
            for i in range(0, len(articles), self.batch_size)
        ]
        
        # Process batches concurrently
        tasks = []
        for batch in batches:
            task = asyncio.create_task(self._process_batch_async(batch))
            tasks.append(task)
        
        # Wait for all batches
        results = await asyncio.gather(*tasks)
        
        # Flatten results
        all_results = []
        for batch_results in results:
            all_results.extend(batch_results)
        
        return all_results
    
    async def _process_batch_async(self, articles: List[Dict]) -> List[Dict]:
        """Process a batch of articles asynchronously"""
        # Extract features in parallel
        loop = asyncio.get_event_loop()
        
        # Extract variables from articles
        variables_list = await loop.run_in_executor(
            self.io_executor,
            self._extract_variables_batch,
            articles
        )
        
        # Run GCT analysis in process pool for CPU efficiency
        gct_results = await loop.run_in_executor(
            self.cpu_executor,
            self._analyze_gct_batch,
            variables_list
        )
        
        # Store results in database asynchronously
        await loop.run_in_executor(
            self.io_executor,
            self._store_results_batch,
            articles,
            gct_results
        )
        
        # Combine results
        results = []
        for article, gct_result in zip(articles, gct_results):
            result = {
                **article,
                'gct_analysis': {
                    'coherence': gct_result.coherence,
                    'q_opt': gct_result.q_opt,
                    'sentiment': gct_result.sentiment,
                    'dc_dt': gct_result.dc_dt,
                    'd2c_dt2': gct_result.d2c_dt2,
                    'components': gct_result.components
                }
            }
            results.append(result)
        
        return results
    
    def _extract_variables_batch(self, articles: List[Dict]) -> List[GCTVariables]:
        """Extract GCT variables from articles"""
        variables_list = []
        
        for article in articles:
            # Check cache first
            cache_key = f"vars_{article.get('id', '')}"
            cached = self.analysis_cache.get(cache_key)
            
            if cached:
                variables_list.append(cached)
            else:
                variables = self._extract_variables(article)
                variables_list.append(variables)
                self.analysis_cache.put(cache_key, variables)
        
        return variables_list
    
    def _extract_variables(self, article: Dict) -> GCTVariables:
        """Extract GCT variables from single article"""
        # This would contain actual extraction logic
        # For now, using placeholder values
        
        # Information value based on content length and keywords
        content = article.get('body', '') + ' ' + article.get('title', '')
        psi = min(1.0, len(content) / 1000)  # Normalize by expected length
        
        # Wisdom based on source credibility
        source = article.get('source', '').lower()
        credible_sources = ['reuters', 'bloomberg', 'wsj', 'ft', 'economist']
        rho = 0.8 if any(s in source for s in credible_sources) else 0.4
        
        # Emotional charge from sentiment keywords
        emotional_words = ['amazing', 'terrible', 'crash', 'surge', 'explode']
        q_raw = sum(1 for word in emotional_words if word in content.lower()) * 0.2
        
        # Social belonging (placeholder)
        f = 0.5
        
        return GCTVariables(
            psi=psi,
            rho=rho,
            q_raw=q_raw,
            f=f,
            timestamp=article.get('timestamp', datetime.now()),
            ticker=article.get('ticker')
        )
    
    def _analyze_gct_batch(self, variables_list: List[GCTVariables]) -> List[GCTResult]:
        """Analyze GCT variables in batch"""
        # Use optimized batch processing
        return self.gct_engine.analyze_batch(variables_list)
    
    def _store_results_batch(self, articles: List[Dict], gct_results: List[GCTResult]):
        """Store results in database efficiently"""
        # Prepare batch data
        gct_scores = []
        ticker_updates = {}
        
        for article, result in zip(articles, gct_results):
            # Prepare GCT score entry
            score_data = {
                'article_id': article['id'],
                'timestamp': article['timestamp'],
                'psi': article.get('psi', 0),
                'rho': article.get('rho', 0),
                'q_raw': article.get('q_raw', 0),
                'f': article.get('f', 0),
                'q_opt': result.q_opt,
                'coherence': result.coherence,
                'dc_dt': result.dc_dt,
                'd2c_dt2': result.d2c_dt2,
                'sentiment': result.sentiment,
                'components': result.components
            }
            gct_scores.append(score_data)
            
            # Aggregate by ticker
            tickers = article.get('tickers', [])
            for ticker in tickers:
                if ticker not in ticker_updates:
                    ticker_updates[ticker] = {
                        'ticker': ticker,
                        'coherence_sum': 0,
                        'count': 0,
                        'sentiment_counts': {'bullish': 0, 'bearish': 0, 'neutral': 0}
                    }
                
                ticker_updates[ticker]['coherence_sum'] += result.coherence
                ticker_updates[ticker]['count'] += 1
                ticker_updates[ticker]['sentiment_counts'][result.sentiment] += 1
        
        # Batch insert
        try:
            # Insert GCT scores
            for score in gct_scores:
                self.db.insert_gct_score(score)
            
            # Update ticker timeline
            for ticker, data in ticker_updates.items():
                avg_coherence = data['coherence_sum'] / data['count']
                
                # Determine dominant sentiment
                sentiment_counts = data['sentiment_counts']
                dominant_sentiment = max(sentiment_counts, key=sentiment_counts.get)
                
                ticker_data = {
                    'ticker': ticker,
                    'timestamp': datetime.now(),
                    'coherence': avg_coherence,
                    'dc_dt': np.mean([r.dc_dt for r in gct_results]),
                    'sentiment': dominant_sentiment,
                    'article_count': data['count']
                }
                
                self.db.insert_ticker_data(ticker_data)
                
        except Exception as e:
            logger.error(f"Failed to store batch results: {e}")
    
    def get_optimized_ticker_analysis(self, ticker: str, period: str = "24h") -> Dict:
        """Get optimized ticker analysis using all caching layers"""
        # Check memory cache first
        cache_key = f"{ticker}_{period}"
        if cache_key in self.ticker_cache:
            cached_time, cached_data = self.ticker_cache[cache_key]
            if (datetime.now() - cached_time).seconds < 300:  # 5 minute cache
                return cached_data
        
        # Get from database with optimization
        df = self.db.get_ticker_timeline_optimized(ticker, period)
        
        if df.empty:
            return {'error': 'No data found'}
        
        # Optimize DataFrame memory
        df = MemoryEfficientDataFrame.optimize_dtypes(df)
        
        # Calculate additional metrics
        result = {
            'ticker': ticker,
            'period': period,
            'summary': df.to_dict('records')[0] if len(df) > 0 else {},
            'performance': self._calculate_performance_metrics(df),
            'last_updated': datetime.now().isoformat()
        }
        
        # Update cache
        self.ticker_cache[cache_key] = (datetime.now(), result)
        
        return result
    
    def _calculate_performance_metrics(self, df) -> Dict:
        """Calculate performance metrics from DataFrame"""
        if df.empty:
            return {}
        
        return {
            'volatility': df['avg_coherence'].std() if 'avg_coherence' in df else 0,
            'trend': 'bullish' if df['bullish_count'].sum() > df['bearish_count'].sum() else 'bearish',
            'signal_strength': df['avg_coherence'].mean() if 'avg_coherence' in df else 0
        }
    
    def cleanup(self):
        """Cleanup resources"""
        self.io_executor.shutdown(wait=True)
        self.cpu_executor.shutdown(wait=True)


# Convenience function for synchronous usage
def create_optimized_pipeline(**kwargs) -> OptimizedAnalysisPipeline:
    """Create an optimized analysis pipeline"""
    return OptimizedAnalysisPipeline(**kwargs)