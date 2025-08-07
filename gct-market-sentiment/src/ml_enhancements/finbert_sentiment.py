"""
FinBERT-based Financial Sentiment Analysis for TraderAI
Leverages pre-trained FinBERT model for accurate financial text sentiment
"""

import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import logging
from dataclasses import dataclass
from functools import lru_cache
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


@dataclass
class SentimentResult:
    """Container for sentiment analysis results"""
    text: str
    sentiment: str  # positive, negative, neutral
    confidence: float
    scores: Dict[str, float]
    
    @property
    def sentiment_score(self) -> float:
        """Convert to numerical score: positive=1, neutral=0, negative=-1"""
        mapping = {'positive': 1.0, 'neutral': 0.0, 'negative': -1.0}
        return mapping.get(self.sentiment, 0.0)


class FinBERTSentimentAnalyzer:
    """
    Advanced financial sentiment analyzer using FinBERT
    with caching, batching, and async support
    """
    
    def __init__(self, 
                 model_name: str = "ProsusAI/finbert",
                 device: Optional[str] = None,
                 max_length: int = 512,
                 batch_size: int = 16,
                 cache_size: int = 1000):
        """
        Initialize FinBERT sentiment analyzer
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to run on ('cuda', 'cpu', or None for auto)
            max_length: Maximum sequence length
            batch_size: Batch size for processing
            cache_size: Size of LRU cache for results
        """
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        
        # Set device
        if device is None:
            self.device = 0 if torch.cuda.is_available() else -1
        else:
            self.device = 0 if device == 'cuda' else -1
            
        logger.info(f"Loading FinBERT model on device: {self.device}")
        
        # Load model and tokenizer
        self._load_model()
        
        # Initialize cache
        self._cache_size = cache_size
        if cache_size > 0:
            self.analyze_text = lru_cache(maxsize=cache_size)(self._analyze_text_impl)
        else:
            self.analyze_text = self._analyze_text_impl
            
        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    def _load_model(self):
        """Load FinBERT model and tokenizer"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            
            # Create pipeline for easier inference
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device,
                max_length=self.max_length,
                truncation=True
            )
            
            # Get label mappings
            self.label2id = self.model.config.label2id
            self.id2label = self.model.config.id2label
            
            logger.info("FinBERT model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading FinBERT model: {e}")
            raise
            
    def _analyze_text_impl(self, text: str) -> SentimentResult:
        """Analyze sentiment of a single text (implementation)"""
        if not text or not text.strip():
            return SentimentResult(
                text=text,
                sentiment='neutral',
                confidence=1.0,
                scores={'positive': 0.0, 'neutral': 1.0, 'negative': 0.0}
            )
            
        try:
            # Get prediction
            result = self.pipeline(text)[0]
            
            # Extract scores for all labels
            scores = {}
            for item in self.pipeline(text, top_k=None)[0]:
                label = item['label'].lower()
                scores[label] = item['score']
                
            return SentimentResult(
                text=text,
                sentiment=result['label'].lower(),
                confidence=result['score'],
                scores=scores
            )
            
        except Exception as e:
            logger.error(f"Error analyzing text: {e}")
            return SentimentResult(
                text=text,
                sentiment='neutral',
                confidence=0.0,
                scores={'positive': 0.0, 'neutral': 0.0, 'negative': 0.0}
            )
            
    def analyze_batch(self, texts: List[str]) -> List[SentimentResult]:
        """
        Analyze sentiment of multiple texts efficiently
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of SentimentResult objects
        """
        results = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            try:
                # Get batch predictions
                batch_results = self.pipeline(batch, top_k=None)
                
                for text, result_list in zip(batch, batch_results):
                    # Extract top prediction
                    top_result = max(result_list, key=lambda x: x['score'])
                    
                    # Extract all scores
                    scores = {
                        item['label'].lower(): item['score'] 
                        for item in result_list
                    }
                    
                    results.append(SentimentResult(
                        text=text,
                        sentiment=top_result['label'].lower(),
                        confidence=top_result['score'],
                        scores=scores
                    ))
                    
            except Exception as e:
                logger.error(f"Error in batch analysis: {e}")
                # Add neutral results for failed batch
                for text in batch:
                    results.append(SentimentResult(
                        text=text,
                        sentiment='neutral',
                        confidence=0.0,
                        scores={'positive': 0.0, 'neutral': 0.0, 'negative': 0.0}
                    ))
                    
        return results
        
    async def analyze_async(self, text: str) -> SentimentResult:
        """Async wrapper for single text analysis"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.analyze_text, text)
        
    async def analyze_batch_async(self, texts: List[str]) -> List[SentimentResult]:
        """Async wrapper for batch analysis"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.analyze_batch, texts)
        
    def analyze_dataframe(self, 
                         df: pd.DataFrame, 
                         text_column: str,
                         output_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Analyze sentiment for texts in a DataFrame
        
        Args:
            df: Input DataFrame
            text_column: Name of column containing text
            output_columns: Names for output columns 
                           [sentiment, confidence, score, pos, neu, neg]
                           
        Returns:
            DataFrame with sentiment columns added
        """
        if output_columns is None:
            output_columns = [
                'sentiment', 'sentiment_confidence', 'sentiment_score',
                'positive_score', 'neutral_score', 'negative_score'
            ]
            
        # Analyze all texts
        texts = df[text_column].fillna('').tolist()
        results = self.analyze_batch(texts)
        
        # Add results to DataFrame
        df[output_columns[0]] = [r.sentiment for r in results]
        df[output_columns[1]] = [r.confidence for r in results]
        df[output_columns[2]] = [r.sentiment_score for r in results]
        df[output_columns[3]] = [r.scores.get('positive', 0) for r in results]
        df[output_columns[4]] = [r.scores.get('neutral', 0) for r in results]
        df[output_columns[5]] = [r.scores.get('negative', 0) for r in results]
        
        return df
        
    def get_market_sentiment(self, news_items: List[Dict[str, str]]) -> Dict[str, float]:
        """
        Calculate aggregated market sentiment from news items
        
        Args:
            news_items: List of dicts with 'text' and optional 'weight' keys
            
        Returns:
            Dictionary with aggregated sentiment metrics
        """
        if not news_items:
            return {
                'sentiment_score': 0.0,
                'positive_ratio': 0.0,
                'negative_ratio': 0.0,
                'neutral_ratio': 0.0,
                'confidence_mean': 0.0,
                'num_items': 0
            }
            
        # Extract texts and weights
        texts = [item['text'] for item in news_items]
        weights = [item.get('weight', 1.0) for item in news_items]
        
        # Analyze sentiments
        results = self.analyze_batch(texts)
        
        # Calculate weighted metrics
        total_weight = sum(weights)
        weighted_scores = []
        sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
        confidences = []
        
        for result, weight in zip(results, weights):
            weighted_scores.append(result.sentiment_score * weight)
            sentiment_counts[result.sentiment] += weight
            confidences.append(result.confidence)
            
        return {
            'sentiment_score': sum(weighted_scores) / total_weight,
            'positive_ratio': sentiment_counts['positive'] / total_weight,
            'negative_ratio': sentiment_counts['negative'] / total_weight,
            'neutral_ratio': sentiment_counts['neutral'] / total_weight,
            'confidence_mean': np.mean(confidences),
            'num_items': len(news_items)
        }
        
    def create_sentiment_features(self, 
                                 df: pd.DataFrame,
                                 text_column: str,
                                 window_sizes: List[int] = [5, 10, 20]) -> pd.DataFrame:
        """
        Create time-series sentiment features with rolling windows
        
        Args:
            df: DataFrame with time-series data
            text_column: Column containing text
            window_sizes: List of rolling window sizes
            
        Returns:
            DataFrame with sentiment features added
        """
        # First, add basic sentiment scores
        df = self.analyze_dataframe(df, text_column)
        
        # Add rolling sentiment features
        for window in window_sizes:
            # Rolling mean of sentiment score
            df[f'sentiment_ma_{window}'] = (
                df['sentiment_score'].rolling(window).mean()
            )
            
            # Rolling sentiment volatility
            df[f'sentiment_vol_{window}'] = (
                df['sentiment_score'].rolling(window).std()
            )
            
            # Sentiment momentum
            df[f'sentiment_momentum_{window}'] = (
                df['sentiment_score'] - df[f'sentiment_ma_{window}']
            )
            
            # Positive/negative ratios
            df[f'positive_ratio_{window}'] = (
                df['positive_score'].rolling(window).mean()
            )
            df[f'negative_ratio_{window}'] = (
                df['negative_score'].rolling(window).mean()
            )
            
        # Sentiment change indicators
        df['sentiment_change'] = df['sentiment_score'].diff()
        df['sentiment_acceleration'] = df['sentiment_change'].diff()
        
        # Sentiment regime
        df['sentiment_regime'] = pd.cut(
            df['sentiment_score'],
            bins=[-1, -0.5, 0.5, 1],
            labels=['bearish', 'neutral', 'bullish']
        )
        
        return df
        
    def clear_cache(self):
        """Clear the results cache"""
        if hasattr(self.analyze_text, 'cache_clear'):
            self.analyze_text.cache_clear()
            logger.info("Sentiment analysis cache cleared")
            
    def get_cache_info(self) -> Optional[Dict]:
        """Get cache statistics"""
        if hasattr(self.analyze_text, 'cache_info'):
            info = self.analyze_text.cache_info()
            return {
                'hits': info.hits,
                'misses': info.misses,
                'maxsize': info.maxsize,
                'currsize': info.currsize,
                'hit_rate': info.hits / (info.hits + info.misses) if info.hits + info.misses > 0 else 0
            }
        return None