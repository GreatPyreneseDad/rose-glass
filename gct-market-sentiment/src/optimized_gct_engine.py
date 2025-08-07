"""
Optimized GCT Engine with batch processing and vectorized operations
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import logging
from collections import deque

logger = logging.getLogger(__name__)

@dataclass
class GCTParameters:
    """Parameters for GCT calculations"""
    km: float = 0.3
    ki: float = 0.1
    coupling_strength: float = 0.15
    bullish_threshold: float = 0.1
    bearish_threshold: float = -0.1
    spike_threshold: float = 0.5


@dataclass
class GCTVariables:
    """Input variables for GCT calculation"""
    psi: float  # Information value
    rho: float  # Wisdom/credibility
    q_raw: float  # Raw emotional charge
    f: float  # Social belonging
    timestamp: datetime
    ticker: Optional[str] = None


@dataclass
class GCTResult:
    """Result of GCT analysis"""
    coherence: float
    q_opt: float
    dc_dt: float
    d2c_dt2: float
    sentiment: str
    components: Dict[str, float]


class CircularBuffer:
    """Efficient circular buffer for time series data"""
    def __init__(self, maxsize: int):
        self.maxsize = maxsize
        self.data = deque(maxlen=maxsize)
        
    def append(self, item: Tuple[datetime, float]):
        self.data.append(item)
    
    def get_recent(self, n: int) -> List[Tuple[datetime, float]]:
        """Get n most recent items"""
        return list(self.data)[-n:] if n < len(self.data) else list(self.data)
    
    def __len__(self):
        return len(self.data)


class OptimizedGCTEngine:
    """Optimized GCT computation engine with batch processing"""
    
    def __init__(self, params: GCTParameters = None):
        self.params = params or GCTParameters()
        self.history_buffer = CircularBuffer(maxsize=1000)
        self.batch_size = 100
        self._preallocate_arrays()
    
    def _preallocate_arrays(self):
        """Pre-allocate numpy arrays for batch processing"""
        self.batch_psi = np.zeros(self.batch_size, dtype=np.float32)
        self.batch_rho = np.zeros(self.batch_size, dtype=np.float32)
        self.batch_q = np.zeros(self.batch_size, dtype=np.float32)
        self.batch_f = np.zeros(self.batch_size, dtype=np.float32)
        self.batch_coherence = np.zeros(self.batch_size, dtype=np.float32)
    
    def compute_coherence_vectorized(
        self, psi: np.ndarray, rho: np.ndarray, 
        q_raw: np.ndarray, f: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Vectorized coherence computation"""
        # Validate inputs
        if self.params.ki == 0:
            raise ValueError("ki parameter cannot be zero")
        
        # Ensure non-negative q_raw
        q_raw = np.maximum(q_raw, 0)
        
        # Optimized q calculation
        denominator = self.params.km + q_raw + (q_raw**2) / self.params.ki
        
        # Avoid division by zero
        denominator = np.maximum(denominator, 1e-10)
        q_opt = q_raw / denominator
        
        # Vectorized coherence calculation
        coherence = (
            psi +                                    # base
            rho * psi +                              # wisdom amplification
            q_opt +                                  # emotional component
            f * psi +                                # social amplification
            self.params.coupling_strength * rho * q_opt  # coupling term
        )
        
        return coherence, q_opt
    
    def process_batch(self, variables_list: List[GCTVariables]) -> List[GCTResult]:
        """Process multiple articles in batch for efficiency"""
        batch_size = len(variables_list)
        
        if batch_size == 0:
            return []
        
        # Extract data into numpy arrays
        psi_arr = np.array([v.psi for v in variables_list], dtype=np.float32)
        rho_arr = np.array([v.rho for v in variables_list], dtype=np.float32)
        q_arr = np.array([v.q_raw for v in variables_list], dtype=np.float32)
        f_arr = np.array([v.f for v in variables_list], dtype=np.float32)
        
        # Batch computation
        coherence_arr, q_opt_arr = self.compute_coherence_vectorized(
            psi_arr, rho_arr, q_arr, f_arr
        )
        
        # Calculate derivatives efficiently
        results = []
        for i, (variables, coherence, q_opt) in enumerate(
            zip(variables_list, coherence_arr, q_opt_arr)
        ):
            # Add to history
            self.history_buffer.append((variables.timestamp, coherence))
            
            # Compute derivatives using sliding window
            dc_dt, d2c_dt2 = self._compute_derivatives_optimized()
            
            results.append(GCTResult(
                coherence=float(coherence),
                q_opt=float(q_opt),
                dc_dt=dc_dt,
                d2c_dt2=d2c_dt2,
                sentiment=self.classify_sentiment(dc_dt),
                components=self._calculate_components(variables, q_opt)
            ))
        
        return results
    
    def process_single(self, variables: GCTVariables) -> GCTResult:
        """Process a single data point (fallback for non-batch operations)"""
        return self.process_batch([variables])[0]
    
    def _compute_derivatives_optimized(self) -> Tuple[float, float]:
        """Optimized derivative calculation using numpy"""
        if len(self.history_buffer) < 3:
            return 0.0, 0.0
        
        # Get recent history
        recent_data = self.history_buffer.get_recent(20)
        if len(recent_data) < 3:
            return 0.0, 0.0
            
        times = np.array([
            (t - recent_data[0][0]).total_seconds() 
            for t, _ in recent_data
        ])
        values = np.array([c for _, c in recent_data])
        
        # Avoid division by zero in time differences
        if np.all(times == 0):
            return 0.0, 0.0
        
        # Use numpy's optimized gradient calculation
        if len(times) >= 2:
            dc_dt = np.gradient(values, times)[-1]
        else:
            dc_dt = 0.0
        
        if len(times) >= 3:
            # Second derivative using central differences
            h = times[-1] - times[-2]
            if h > 0:
                d2c_dt2 = (values[-1] - 2*values[-2] + values[-3]) / (h**2)
            else:
                d2c_dt2 = 0.0
        else:
            d2c_dt2 = 0.0
        
        return float(dc_dt), float(d2c_dt2)
    
    def classify_sentiment(self, dc_dt: float) -> str:
        """Classify sentiment based on coherence derivative"""
        if dc_dt > self.params.bullish_threshold:
            return "bullish"
        elif dc_dt < self.params.bearish_threshold:
            return "bearish"
        else:
            return "neutral"
    
    def _calculate_components(self, variables: GCTVariables, q_opt: float) -> Dict[str, float]:
        """Calculate individual component contributions"""
        base = variables.psi
        wisdom_amp = variables.rho * variables.psi
        emotional = q_opt
        social_amp = variables.f * variables.psi
        coupling = self.params.coupling_strength * variables.rho * q_opt
        
        total = base + wisdom_amp + emotional + social_amp + coupling
        
        # Calculate percentage contributions
        if total > 0:
            return {
                "base_pct": (base / total) * 100,
                "wisdom_pct": (wisdom_amp / total) * 100,
                "emotional_pct": (emotional / total) * 100,
                "social_pct": (social_amp / total) * 100,
                "coupling_pct": (coupling / total) * 100
            }
        else:
            return {
                "base_pct": 0.0,
                "wisdom_pct": 0.0,
                "emotional_pct": 0.0,
                "social_pct": 0.0,
                "coupling_pct": 0.0
            }
    
    def analyze(self, variables: GCTVariables) -> GCTResult:
        """Analyze single data point (backwards compatibility)"""
        return self.process_single(variables)
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics"""
        if len(self.history_buffer) == 0:
            return {
                "avg_coherence": 0.0,
                "std_coherence": 0.0,
                "trend": 0.0
            }
        
        recent_data = self.history_buffer.get_recent(100)
        values = np.array([c for _, c in recent_data])
        
        return {
            "avg_coherence": float(np.mean(values)),
            "std_coherence": float(np.std(values)),
            "trend": float(np.polyfit(range(len(values)), values, 1)[0]) if len(values) > 1 else 0.0
        }


class BatchProcessor:
    """Batch processor for handling large volumes of articles"""
    
    def __init__(self, gct_engine: OptimizedGCTEngine, batch_size: int = 100):
        self.gct_engine = gct_engine
        self.batch_size = batch_size
        self.pending_batch = []
        
    def add_article(self, article: Dict) -> Optional[GCTResult]:
        """Add article to batch, process if batch is full"""
        # Convert article to GCTVariables
        variables = self._article_to_variables(article)
        self.pending_batch.append(variables)
        
        if len(self.pending_batch) >= self.batch_size:
            return self.process_pending_batch()
        return None
    
    def process_pending_batch(self) -> List[GCTResult]:
        """Process all pending articles"""
        if not self.pending_batch:
            return []
        
        results = self.gct_engine.process_batch(self.pending_batch)
        self.pending_batch = []
        return results
    
    def _article_to_variables(self, article: Dict) -> GCTVariables:
        """Convert article dict to GCTVariables"""
        # This would contain the actual extraction logic
        # For now, returning placeholder values
        return GCTVariables(
            psi=article.get('psi', 0.5),
            rho=article.get('rho', 0.5),
            q_raw=article.get('q_raw', 0.5),
            f=article.get('f', 0.5),
            timestamp=article.get('timestamp', datetime.now()),
            ticker=article.get('ticker')
        )