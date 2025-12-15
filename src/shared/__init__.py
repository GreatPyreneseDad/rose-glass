"""
Rose Unified Shared Modules
===========================

Shared components across all Rose repositories:
- rose-glass
- rose-looking-glass
- RoseGlassLE

These modules provide consistent implementation of:
- τ (temporal depth): temporal_dimension.py
- λ (lens interference): lens_interference.py
- Fibonacci learning: fibonacci_learning.py
- Veritas + Mirror/Architect: veritas_reflexive.py

Author: Christopher MacGregor bin Joseph
Date: December 2025
"""

from .temporal_dimension import (
    TemporalAnalyzer,
    TemporalSignature,
    TemporalScale,
    extract_tau
)

from .lens_interference import (
    LensInterferenceAnalyzer,
    InterferenceAnalysis,
    InterferenceType,
    LensReading,
    extract_lambda
)

from .fibonacci_learning import (
    FibonacciLearningAlgorithm,
    TruthDiscovery,
    TruthType,
    ResetTrigger,
    FibonacciState,
    create_fibonacci_learner
)

from .veritas_reflexive import (
    VeritasFunction,
    VeritasResult,
    EvaluationFrame,
    FrameEvaluation,
    ArchitectWing,
    MirrorWing,
    ReflexiveValidationSystem,
    InsightFragment,
    IntegratedInsight,
    ReflectionResult
)

__version__ = '1.0.0'
__author__ = 'Christopher MacGregor bin Joseph'

__all__ = [
    # Version
    '__version__',
    '__author__',
    
    # Temporal
    'TemporalAnalyzer',
    'TemporalSignature',
    'TemporalScale',
    'extract_tau',
    
    # Lens interference
    'LensInterferenceAnalyzer',
    'InterferenceAnalysis',
    'InterferenceType',
    'LensReading',
    'extract_lambda',
    
    # Fibonacci learning
    'FibonacciLearningAlgorithm',
    'TruthDiscovery',
    'TruthType',
    'ResetTrigger',
    'FibonacciState',
    'create_fibonacci_learner',
    
    # Veritas and validation
    'VeritasFunction',
    'VeritasResult',
    'EvaluationFrame',
    'FrameEvaluation',
    'ArchitectWing',
    'MirrorWing',
    'ReflexiveValidationSystem',
    'InsightFragment',
    'IntegratedInsight',
    'ReflectionResult',
]
