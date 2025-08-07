"""
Memory-efficient data structures and processing pipeline
"""

import gc
import psutil
from collections import OrderedDict, deque
from functools import lru_cache
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
import json
import zlib
import pickle

logger = logging.getLogger(__name__)


class LRUCache:
    """Space-efficient LRU cache implementation"""
    def __init__(self, maxsize: int, ttl_seconds: int = 300):
        self.cache = OrderedDict()
        self.maxsize = maxsize
        self.ttl_seconds = ttl_seconds
        self.access_times = {}
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if exists and not expired"""
        if key in self.cache:
            # Check if expired
            if datetime.now() - self.access_times[key] > timedelta(seconds=self.ttl_seconds):
                del self.cache[key]
                del self.access_times[key]
                return None
            
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return self.cache[key]
        return None
    
    def put(self, key: str, value: Any):
        """Put value in cache with TTL"""
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        self.access_times[key] = datetime.now()
        
        if len(self.cache) > self.maxsize:
            # Remove least recently used
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            del self.access_times[oldest_key]
    
    def clear_expired(self):
        """Remove all expired entries"""
        now = datetime.now()
        expired_keys = [
            k for k, v in self.access_times.items()
            if now - v > timedelta(seconds=self.ttl_seconds)
        ]
        for key in expired_keys:
            del self.cache[key]
            del self.access_times[key]
    
    def __len__(self):
        return len(self.cache)


class CompressedCache:
    """Cache with compression for large data"""
    def __init__(self, maxsize: int, compression_threshold: int = 1024):
        self.cache = LRUCache(maxsize)
        self.compression_threshold = compression_threshold
    
    def get(self, key: str) -> Optional[Any]:
        """Get and decompress if needed"""
        data = self.cache.get(key)
        if data is None:
            return None
        
        if isinstance(data, bytes) and data.startswith(b'COMPRESSED:'):
            # Decompress
            compressed_data = data[11:]  # Remove prefix
            return pickle.loads(zlib.decompress(compressed_data))
        return data
    
    def put(self, key: str, value: Any):
        """Compress and store if large"""
        pickled = pickle.dumps(value)
        
        if len(pickled) > self.compression_threshold:
            # Compress large data
            compressed = b'COMPRESSED:' + zlib.compress(pickled)
            if len(compressed) < len(pickled):  # Only use if actually smaller
                self.cache.put(key, compressed)
                return
        
        self.cache.put(key, value)


class MemoryEfficientDataFrame:
    """Memory-efficient DataFrame operations"""
    
    @staticmethod
    def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage by converting to efficient dtypes"""
        for col in df.columns:
            col_type = df[col].dtype
            
            if col_type != 'object':
                c_min = df[col].min()
                c_max = df[col].max()
                
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
            else:
                # Convert object to category if low cardinality
                n_unique = df[col].nunique()
                n_total = len(df[col])
                if n_unique / n_total < 0.5:
                    df[col] = df[col].astype('category')
        
        return df
    
    @staticmethod
    def chunked_processing(df: pd.DataFrame, func, chunk_size: int = 10000) -> pd.DataFrame:
        """Process DataFrame in chunks to avoid memory issues"""
        chunks = []
        for start in range(0, len(df), chunk_size):
            chunk = df.iloc[start:start + chunk_size]
            processed_chunk = func(chunk)
            chunks.append(processed_chunk)
            
            # Force garbage collection periodically
            if start % (chunk_size * 10) == 0:
                gc.collect()
        
        return pd.concat(chunks, ignore_index=True)


class MemoryEfficientPipeline:
    """Memory-efficient processing pipeline"""
    
    def __init__(self, max_memory_mb: int = 1024):
        self.max_memory_mb = max_memory_mb
        self.processing_queue = deque(maxlen=10000)
        self.result_cache = CompressedCache(maxsize=1000)
        self.article_cache = LRUCache(maxsize=5000)
        
    def process_with_memory_limit(self, articles: List[Dict]) -> List[Dict]:
        """Process articles with memory constraints"""
        results = []
        batch_size = self._calculate_optimal_batch_size()
        
        for i in range(0, len(articles), batch_size):
            batch = articles[i:i+batch_size]
            
            # Check memory usage
            if self._memory_usage_mb() > self.max_memory_mb * 0.8:
                logger.warning("Memory usage high, triggering cleanup")
                gc.collect()
                self._flush_caches()
            
            # Process batch
            batch_results = self._process_batch(batch)
            results.extend(batch_results)
            
            # Store results efficiently
            self._store_results_compressed(batch_results)
        
        return results
    
    def _calculate_optimal_batch_size(self) -> int:
        """Calculate batch size based on available memory"""
        available_memory = psutil.virtual_memory().available
        article_size_estimate = 5 * 1024  # 5KB per article
        
        # Use 20% of available memory for processing
        max_articles = int((available_memory * 0.2) / article_size_estimate)
        
        return min(max_articles, 1000)  # Cap at 1000
    
    def _memory_usage_mb(self) -> float:
        """Get current process memory usage in MB"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def _flush_caches(self):
        """Clear caches to free memory"""
        self.result_cache.cache.clear_expired()
        self.article_cache.clear_expired()
        
        # Clear old items from processing queue
        if len(self.processing_queue) > 5000:
            for _ in range(1000):
                self.processing_queue.popleft()
    
    def _process_batch(self, batch: List[Dict]) -> List[Dict]:
        """Process a batch of articles"""
        results = []
        
        for article in batch:
            # Check cache first
            cache_key = self._get_cache_key(article)
            cached_result = self.result_cache.get(cache_key)
            
            if cached_result:
                results.append(cached_result)
            else:
                # Process article
                result = self._process_article(article)
                results.append(result)
                
                # Cache result
                self.result_cache.put(cache_key, result)
        
        return results
    
    def _process_article(self, article: Dict) -> Dict:
        """Process single article (placeholder)"""
        # This would contain actual processing logic
        return {
            'id': article.get('id'),
            'processed': True,
            'timestamp': datetime.now()
        }
    
    def _get_cache_key(self, article: Dict) -> str:
        """Generate cache key for article"""
        return f"{article.get('id', '')}_{article.get('timestamp', '')}"
    
    def _store_results_compressed(self, results: List[Dict]):
        """Store results with compression"""
        if not results:
            return
        
        # Group by timestamp for efficient storage
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        key = f"results_{timestamp}"
        
        self.result_cache.put(key, results)
    
    @lru_cache(maxsize=1000)
    def compute_coherence_cached(self, key: str, *args) -> float:
        """Cache coherence calculations for repeated data"""
        # This would call the actual GCT engine
        return 0.5  # Placeholder


class StreamingProcessor:
    """Process data in streaming fashion to minimize memory usage"""
    
    def __init__(self, buffer_size: int = 1000):
        self.buffer = deque(maxlen=buffer_size)
        self.processed_count = 0
    
    def add_item(self, item: Dict) -> Optional[Dict]:
        """Add item to buffer, process if full"""
        self.buffer.append(item)
        
        if len(self.buffer) >= self.buffer.maxlen:
            return self._process_buffer()
        return None
    
    def _process_buffer(self) -> Dict:
        """Process items in buffer"""
        # Convert to numpy for efficient computation
        values = np.array([item.get('value', 0) for item in self.buffer])
        
        result = {
            'mean': np.mean(values),
            'std': np.std(values),
            'max': np.max(values),
            'min': np.min(values),
            'count': len(values),
            'timestamp': datetime.now()
        }
        
        self.processed_count += len(self.buffer)
        self.buffer.clear()
        
        return result
    
    def flush(self) -> Optional[Dict]:
        """Process remaining items in buffer"""
        if self.buffer:
            return self._process_buffer()
        return None


def monitor_memory_usage(func):
    """Decorator to monitor memory usage of functions"""
    def wrapper(*args, **kwargs):
        # Memory before
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024
        
        # Execute function
        result = func(*args, **kwargs)
        
        # Memory after
        mem_after = process.memory_info().rss / 1024 / 1024
        mem_used = mem_after - mem_before
        
        if mem_used > 100:  # Log if more than 100MB used
            logger.warning(
                f"{func.__name__} used {mem_used:.2f}MB memory "
                f"(before: {mem_before:.2f}MB, after: {mem_after:.2f}MB)"
            )
        
        return result
    
    return wrapper