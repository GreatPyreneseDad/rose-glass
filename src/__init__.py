"""
The Rose Glass: Mathematical Lens for Synthetic-Organic Intelligence Translation
==============================================================================

A framework for enabling synthetic minds to perceive and understand organic
intelligence patterns without judgment or measurement.
"""

__version__ = "2.0.0"
__author__ = "Christopher MacGregor bin Joseph"

# Core components
from .core.rose_glass_v2 import (
    RoseGlassV2,
    CulturalCalibration,
    PatternInterpretation,
    TranslationConfidence,
    LensState
)

from .core.rose_glass_lens import (
    RoseGlass,
    OrganicSyntheticTranslator,
    MultiLensViewer,
    PatternVisibility,
    LensCalibration,
    CulturalContext,
    TemporalPeriod
)

# Ethics components
from .ethics.communication_style_adapter import (
    CommunicationStyleAdapter,
    CommunicationPattern
)

from .ethics.interaction_quality_monitor import (
    InteractionQualityMonitor,
    InteractionMetrics,
    ConversationQuality
)

from .ethics.ethical_rose_glass_pipeline import (
    EthicalRoseGlassPipeline,
    EthicalTranslationEvent
)

__all__ = [
    # Version
    "__version__",
    "__author__",
    
    # Core v2
    "RoseGlassV2",
    "CulturalCalibration", 
    "PatternInterpretation",
    "TranslationConfidence",
    "LensState",
    
    # Core v1 (for compatibility)
    "RoseGlass",
    "OrganicSyntheticTranslator",
    "MultiLensViewer",
    "PatternVisibility",
    "LensCalibration",
    "CulturalContext",
    "TemporalPeriod",
    
    # Ethics
    "CommunicationStyleAdapter",
    "CommunicationPattern",
    "InteractionQualityMonitor",
    "InteractionMetrics",
    "ConversationQuality",
    "EthicalRoseGlassPipeline",
    "EthicalTranslationEvent"
]