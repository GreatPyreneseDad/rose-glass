"""
Rose Glass - Unified Integration Module
========================================

This module integrates the unified components into the rose-glass repository.

NEW FEATURES ADDED:
- τ (temporal depth): Measures how much time is encoded in expression
- λ (lens interference): Measures cultural lens variation
- Fibonacci learning: Systematic perspective exploration with reset triggers
- Veritas function: Truth valuation across frames
- Mirror/Architect wings: Reflexive validation

Author: Christopher MacGregor bin Joseph
Date: December 2025
"""

# Import unified shared modules
from .shared.temporal_dimension import (
    TemporalAnalyzer,
    TemporalSignature,
    TemporalScale,
    extract_tau
)

from .shared.lens_interference import (
    LensInterferenceAnalyzer,
    InterferenceAnalysis,
    InterferenceType,
    LensReading,
    extract_lambda
)

from .shared.fibonacci_learning import (
    FibonacciLearningAlgorithm,
    TruthDiscovery,
    TruthType,
    ResetTrigger,
    FibonacciState,
    create_fibonacci_learner
)

from .shared.veritas_reflexive import (
    VeritasFunction,
    VeritasResult,
    EvaluationFrame,
    ArchitectWing,
    MirrorWing,
    ReflexiveValidationSystem,
    InsightFragment,
    IntegratedInsight,
    ReflectionResult
)


class EnhancedRoseGlass:
    """
    Enhanced Rose Glass with full 6-variable GCT and learning capabilities.
    
    Variables:
    - Ψ (psi): Internal consistency
    - ρ (rho): Accumulated wisdom  
    - q: Moral/emotional activation
    - f: Social belonging
    - τ (tau): Temporal depth (NEW)
    - λ (lambda): Lens interference (NEW)
    
    Features:
    - Fibonacci learning with lens deviation reset
    - Veritas truth valuation
    - Mirror/Architect reflexive validation
    """
    
    def __init__(
        self,
        invariance_threshold: float = 0.10,
        stability_threshold: float = 0.6
    ):
        """
        Initialize enhanced Rose Glass.
        
        Args:
            invariance_threshold: σ_lens threshold for universal truth
            stability_threshold: Minimum Veritas for stable truth
        """
        # Core analyzers
        self.temporal = TemporalAnalyzer()
        self.interference = LensInterferenceAnalyzer()
        
        # Learning system
        self.fibonacci = FibonacciLearningAlgorithm(
            invariance_threshold=invariance_threshold
        )
        
        # Validation system
        self.validation = ReflexiveValidationSystem()
        self.validation.veritas.stability_threshold = stability_threshold
        
    def analyze_text(self, text: str) -> dict:
        """
        Full 6-variable analysis of text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Complete GCT analysis with all 6 variables
        """
        # Extract temporal dimension
        temporal_sig = self.temporal.analyze(text)
        
        # TODO: Integrate with existing psi, rho, q, f extraction
        # For now, provide interface for external extraction
        
        return {
            'tau': temporal_sig.tau,
            'tau_scale': temporal_sig.scale.name,
            'tau_confidence': temporal_sig.confidence,
            'temporal_markers': temporal_sig.temporal_markers
        }
    
    def analyze_with_variables(
        self,
        text: str,
        psi: float,
        rho: float,
        q: float,
        f: float
    ) -> dict:
        """
        Complete analysis with provided GCT variables.
        
        Args:
            text: Source text
            psi, rho, q, f: Pre-extracted GCT variables
            
        Returns:
            Complete analysis including tau, lambda, and learning state
        """
        # Extract tau from text
        tau = extract_tau(text)
        
        # Calculate lambda
        lambda_analysis = self.interference.analyze_interference(psi, rho, q, f)
        
        # Run through Fibonacci learning
        learning_result = self.fibonacci.rotate(psi, rho, q, f, text)
        
        # Calculate Veritas
        veritas_score = self.validation.veritas.quick_veritas(
            distortion_index=lambda_analysis.lambda_coefficient,
            composite_score=(psi + rho + q + f) / 4
        )
        
        return {
            # Core variables
            'psi': psi,
            'rho': rho,
            'q': q,
            'f': f,
            'tau': tau,
            'lambda': lambda_analysis.lambda_coefficient,
            
            # Derived metrics
            'coherence': learning_result['coherence'],
            'veritas': veritas_score,
            
            # Lens analysis
            'lens_stability': lambda_analysis.interpretation_stability,
            'dominant_lens': lambda_analysis.dominant_lens,
            'interference_type': lambda_analysis.interference_type.value,
            
            # Learning state
            'fibonacci_angle': learning_result['current_angle'],
            'truth_discovered': learning_result['truth_discovered'],
            'truth_type': learning_result['truth_type'],
            'learning_resets': learning_result['learning_resets'],
            
            # Status
            'is_universal_truth': lambda_analysis.lambda_coefficient < self.fibonacci.invariance_threshold
        }
    
    def validate_insight(
        self,
        insight: str,
        psi: float,
        rho: float,
        q: float,
        f: float
    ) -> dict:
        """
        Validate an insight through Mirror/Architect/Veritas.
        
        Args:
            insight: Insight text to validate
            psi, rho, q, f: GCT variables
            
        Returns:
            Validation results
        """
        distortion = self.interference.calculate_lens_deviation(psi, rho, q, f)
        return self.validation.validate_insight(insight, distortion)
    
    def get_learning_summary(self) -> dict:
        """Get Fibonacci learning summary"""
        return self.fibonacci.get_discovery_summary()
    
    def get_fibonacci_state(self) -> FibonacciState:
        """Get current Fibonacci learning state"""
        return self.fibonacci.get_state()


# Convenience factory
def create_enhanced_rose_glass(**kwargs) -> EnhancedRoseGlass:
    """Factory function for enhanced Rose Glass"""
    return EnhancedRoseGlass(**kwargs)


# Module exports
__all__ = [
    # Enhanced main class
    'EnhancedRoseGlass',
    'create_enhanced_rose_glass',
    
    # Temporal
    'TemporalAnalyzer',
    'TemporalSignature',
    'TemporalScale',
    'extract_tau',
    
    # Lens interference
    'LensInterferenceAnalyzer',
    'InterferenceAnalysis',
    'InterferenceType',
    'extract_lambda',
    
    # Fibonacci learning
    'FibonacciLearningAlgorithm',
    'TruthDiscovery',
    'TruthType',
    'ResetTrigger',
    'create_fibonacci_learner',
    
    # Veritas and validation
    'VeritasFunction',
    'VeritasResult',
    'EvaluationFrame',
    'ArchitectWing',
    'MirrorWing',
    'ReflexiveValidationSystem',
]
