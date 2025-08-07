"""
GCT (Grounded Coherence Theory) Engine for Market Sentiment Analysis
"""

import numpy as np
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
from datetime import datetime

# Import optimized engine for batch processing
try:
    from .optimized_gct_engine import OptimizedGCTEngine, BatchProcessor
    OPTIMIZED_AVAILABLE = True
except ImportError:
    OPTIMIZED_AVAILABLE = False


@dataclass
class GCTParameters:
    """Parameters for GCT model"""

    km: float = 0.3  # Saturation constant for wisdom
    ki: float = 0.1  # Inhibition constant for wisdom
    coupling_strength: float = 0.15  # Coupling between components

    # Sentiment thresholds
    bullish_threshold: float = 0.05
    bearish_threshold: float = -0.05
    spike_threshold: float = 0.1


@dataclass
class GCTVariables:
    """Variables extracted from text for GCT analysis"""

    psi: float  # Clarity/precision of narrative
    rho: float  # Reflective depth/nuance
    q_raw: float  # Emotional charge (raw)
    f: float  # Social belonging signal
    timestamp: datetime


@dataclass
class GCTResult:
    """Results from GCT coherence computation"""

    coherence: float
    q_opt: float  # Optimized emotional charge
    dc_dt: float  # First derivative
    d2c_dt2: float  # Second derivative
    sentiment: str  # bullish/bearish/neutral
    components: Dict[str, float]


class GCTEngine:
    """Core GCT computation engine"""

    def __init__(self, params: GCTParameters = None, use_optimized: bool = True):
        self.params = params or GCTParameters()
        self.history: List[Tuple[datetime, float]] = []
        
        # Use optimized engine if available and requested
        self.use_optimized = use_optimized and OPTIMIZED_AVAILABLE
        if self.use_optimized:
            self.optimized_engine = OptimizedGCTEngine(params)
            self.batch_processor = BatchProcessor(self.optimized_engine)

    def compute_coherence(self, variables: GCTVariables) -> Tuple[float, float]:
        """
        Compute GCT coherence score

        Returns:
            coherence: Overall coherence score
            q_opt: Optimized emotional charge
        """
        # Validate inputs to prevent division by zero
        if self.params.ki == 0:
            raise ValueError("ki parameter cannot be zero")
        
        # Ensure q_raw is within valid range
        if variables.q_raw < 0:
            raise ValueError("q_raw must be non-negative")
            
        # Optimize emotional charge with wisdom modulation
        denominator = self.params.km + variables.q_raw + (variables.q_raw**2) / self.params.ki
        
        # Additional safety check
        if denominator == 0:
            raise ValueError("Denominator in q_opt calculation cannot be zero")
            
        q_opt = variables.q_raw / denominator

        # Component contributions
        base = variables.psi
        wisdom_amp = variables.rho * variables.psi
        social_amp = variables.f * variables.psi
        coupling = self.params.coupling_strength * variables.rho * q_opt

        # Total coherence
        coherence = base + wisdom_amp + q_opt + social_amp + coupling

        return coherence, q_opt

    def compute_derivatives(
        self, coherence: float, timestamp: datetime
    ) -> Tuple[float, float]:
        """
        Compute time derivatives of coherence

        Returns:
            dc_dt: First derivative
            d2c_dt2: Second derivative
        """
        # Add to history
        self.history.append((timestamp, coherence))

        # Keep only recent history (e.g., last 100 points)
        if len(self.history) > 100:
            self.history.pop(0)

        # Need at least 3 points for derivatives
        if len(self.history) < 3:
            return 0.0, 0.0

        # Extract recent points
        times = np.array(
            [(t - self.history[0][0]).total_seconds() for t, _ in self.history]
        )
        values = np.array([c for _, c in self.history])

        # Compute derivatives using finite differences
        if len(times) >= 2:
            dc_dt = np.gradient(values, times)[-1]
        else:
            dc_dt = 0.0

        if len(times) >= 3:
            d2c_dt2 = np.gradient(np.gradient(values, times), times)[-1]
        else:
            d2c_dt2 = 0.0

        return dc_dt, d2c_dt2

    def classify_sentiment(self, dc_dt: float) -> str:
        """
        Classify sentiment based on coherence derivative

        Returns:
            sentiment: 'bullish', 'bearish', or 'neutral'
        """
        if dc_dt > self.params.bullish_threshold:
            return "bullish"
        elif dc_dt < self.params.bearish_threshold:
            return "bearish"
        else:
            return "neutral"

    def detect_spike(self, d2c_dt2: float) -> bool:
        """
        Detect if there's a spike in coherence acceleration
        """
        return abs(d2c_dt2) > self.params.spike_threshold

    def analyze(self, variables: GCTVariables) -> GCTResult:
        """
        Complete GCT analysis pipeline
        """
        # Compute coherence
        coherence, q_opt = self.compute_coherence(variables)

        # Compute derivatives
        dc_dt, d2c_dt2 = self.compute_derivatives(coherence, variables.timestamp)

        # Classify sentiment
        sentiment = self.classify_sentiment(dc_dt)

        # Component breakdown
        components = {
            "base": variables.psi,
            "wisdom_amp": variables.rho * variables.psi,
            "emotional": q_opt,
            "social_amp": variables.f * variables.psi,
            "coupling": self.params.coupling_strength * variables.rho * q_opt,
        }

        return GCTResult(
            coherence=coherence,
            q_opt=q_opt,
            dc_dt=dc_dt,
            d2c_dt2=d2c_dt2,
            sentiment=sentiment,
            components=components,
        )

    def reset_history(self):
        """Reset the coherence history"""
        self.history = []
    
    def analyze_batch(self, variables_list: List[GCTVariables]) -> List[GCTResult]:
        """
        Analyze multiple data points in batch for better performance
        """
        if self.use_optimized:
            return self.optimized_engine.process_batch(variables_list)
        else:
            # Fallback to sequential processing
            return [self.analyze(v) for v in variables_list]
    
    def get_batch_processor(self) -> Optional[BatchProcessor]:
        """Get batch processor for streaming operations"""
        if self.use_optimized:
            return self.batch_processor
        return None


class SectorGCTEngine(GCTEngine):
    """GCT Engine with sector-specific parameter tuning"""

    SECTOR_PARAMS = {
        "tech": GCTParameters(km=0.25, ki=0.08, bullish_threshold=0.06),
        "finance": GCTParameters(km=0.35, ki=0.12, bearish_threshold=-0.04),
        "energy": GCTParameters(km=0.4, ki=0.15, spike_threshold=0.15),
        "healthcare": GCTParameters(km=0.3, ki=0.1, coupling_strength=0.12),
        "default": GCTParameters(),
    }

    def __init__(self, sector: str = "default"):
        params = self.SECTOR_PARAMS.get(sector, self.SECTOR_PARAMS["default"])
        super().__init__(params)
        self.sector = sector
