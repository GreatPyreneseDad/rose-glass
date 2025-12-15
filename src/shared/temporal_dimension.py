"""
Unified Temporal Dimension Module
=================================

Shared across: rose-glass, rose-looking-glass, RoseGlassLE

Implements τ (tau) - the temporal depth dimension that measures
how much time is encoded in expression.

"Ancient wisdom speaks differently than breaking news."

Author: Christopher MacGregor bin Joseph
Date: December 2025
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum
import re


class TemporalScale(Enum):
    """Temporal scales from immediate to eternal"""
    IMMEDIATE = 0.0      # Breaking news, live reaction
    RECENT = 0.2         # Days to weeks
    CONTEMPORARY = 0.4   # Months to years
    HISTORICAL = 0.6     # Decades to centuries
    ANCIENT = 0.8        # Millennia
    ETERNAL = 1.0        # Timeless wisdom


@dataclass
class TemporalSignature:
    """Complete temporal analysis of text"""
    tau: float                          # Primary temporal depth (0-1)
    scale: TemporalScale               # Categorical classification
    compression_ratio: float           # How much time compressed into expression
    decay_resistance: float            # How well meaning survives time
    temporal_markers: List[str]        # Detected time-related phrases
    confidence: float                  # Confidence in assessment


class TemporalAnalyzer:
    """
    Analyzes temporal depth in text.
    
    τ measures how much time is encoded in expression:
    - Low τ (0.0-0.3): Immediate, reactive, ephemeral
    - Mid τ (0.3-0.6): Contemporary, contextual
    - High τ (0.6-1.0): Enduring, wisdom-encoded, eternal
    """
    
    # Temporal markers by scale
    IMMEDIATE_MARKERS = [
        r'\bjust\b', r'\bright now\b', r'\bbreaking\b', r'\blive\b',
        r'\btoday\b', r'\bthis moment\b', r'\bcurrently\b', r'\bongoing\b',
        r'\bOMG\b', r'\bwtf\b', r'\blol\b', r'\bsmh\b'
    ]
    
    RECENT_MARKERS = [
        r'\byesterday\b', r'\blast week\b', r'\brecently\b', r'\bthis month\b',
        r'\bthis year\b', r'\bnew\b', r'\blatest\b', r'\bupdated\b'
    ]
    
    HISTORICAL_MARKERS = [
        r'\bdecades\b', r'\bcentury\b', r'\bhistorically\b', r'\btradition\b',
        r'\bgeneration\b', r'\blegacy\b', r'\bfounders\b', r'\bprecedent\b'
    ]
    
    ANCIENT_MARKERS = [
        r'\bancient\b', r'\beternal\b', r'\btimeless\b', r'\bwisdom\b',
        r'\bsages\b', r'\bprophets\b', r'\bmillennia\b', r'\bforever\b',
        r'\balways\b', r'\bnever\b', r'\buniversal\b', r'\btruth\b'
    ]
    
    # Wisdom patterns that suggest temporal depth
    WISDOM_PATTERNS = [
        r'every\s+\w+\s+(that|who|which)',  # Universal statements
        r'those who',                        # Archetypal reference
        r'it is said',                       # Transmitted wisdom
        r'as .+ taught',                     # Referenced teaching
        r'the nature of',                    # Essential qualities
        r'in all things',                    # Universal scope
    ]
    
    def __init__(self, calibration: Optional[Dict] = None):
        """
        Initialize temporal analyzer.
        
        Args:
            calibration: Optional calibration overrides
        """
        self.calibration = calibration or {}
        
    def analyze(self, text: str) -> TemporalSignature:
        """
        Analyze temporal depth of text.
        
        Args:
            text: Text to analyze
            
        Returns:
            TemporalSignature with complete analysis
        """
        text_lower = text.lower()
        
        # Detect temporal markers
        immediate_count = self._count_patterns(text_lower, self.IMMEDIATE_MARKERS)
        recent_count = self._count_patterns(text_lower, self.RECENT_MARKERS)
        historical_count = self._count_patterns(text_lower, self.HISTORICAL_MARKERS)
        ancient_count = self._count_patterns(text_lower, self.ANCIENT_MARKERS)
        wisdom_count = self._count_patterns(text_lower, self.WISDOM_PATTERNS)
        
        # Collect all detected markers
        all_markers = []
        for pattern_list in [self.IMMEDIATE_MARKERS, self.RECENT_MARKERS,
                            self.HISTORICAL_MARKERS, self.ANCIENT_MARKERS]:
            for pattern in pattern_list:
                matches = re.findall(pattern, text_lower)
                all_markers.extend(matches)
        
        # Calculate weighted temporal score
        total_markers = immediate_count + recent_count + historical_count + ancient_count + 1
        
        # Weights: immediate pulls down, ancient pulls up
        weighted_sum = (
            immediate_count * 0.0 +
            recent_count * 0.25 +
            historical_count * 0.65 +
            ancient_count * 0.9 +
            wisdom_count * 0.75  # Wisdom patterns suggest depth
        )
        
        raw_tau = weighted_sum / total_markers
        
        # Structural analysis
        compression = self._calculate_compression(text)
        decay_resistance = self._calculate_decay_resistance(text)
        
        # Combine factors
        tau = np.clip(
            raw_tau * 0.5 + compression * 0.25 + decay_resistance * 0.25,
            0.0, 1.0
        )
        
        # Determine scale
        scale = self._tau_to_scale(tau)
        
        # Calculate confidence based on marker density
        marker_density = len(all_markers) / max(len(text.split()), 1)
        confidence = min(0.5 + marker_density * 5, 0.95)
        
        return TemporalSignature(
            tau=tau,
            scale=scale,
            compression_ratio=compression,
            decay_resistance=decay_resistance,
            temporal_markers=all_markers[:10],  # Top 10
            confidence=confidence
        )
    
    def _count_patterns(self, text: str, patterns: List[str]) -> int:
        """Count matches for pattern list"""
        count = 0
        for pattern in patterns:
            count += len(re.findall(pattern, text))
        return count
    
    def _calculate_compression(self, text: str) -> float:
        """
        Calculate temporal compression ratio.
        
        High compression = much time/experience packed into few words
        """
        words = text.split()
        word_count = len(words)
        
        if word_count == 0:
            return 0.0
        
        # Aphoristic density (short but meaningful)
        if word_count < 20:
            # Short texts can be highly compressed wisdom
            unique_concepts = len(set(words))
            concept_density = unique_concepts / word_count
            if concept_density > 0.7:  # High variety = compressed meaning
                return 0.8
        
        # Look for compression indicators
        compression_markers = [
            r'\btherefore\b', r'\bthus\b', r'\bhence\b',
            r'\bin essence\b', r'\bfundamentally\b',
            r'\bultimately\b', r'\bthe key is\b'
        ]
        
        compression_count = self._count_patterns(text.lower(), compression_markers)
        
        return min(compression_count * 0.2, 0.9)
    
    def _calculate_decay_resistance(self, text: str) -> float:
        """
        Calculate how well meaning survives temporal translation.
        
        High decay resistance = meaning persists across time/context
        """
        # Universal statements resist decay
        universal_patterns = [
            r'\ball\b', r'\bevery\b', r'\bnone\b', r'\balways\b',
            r'\bnever\b', r'\bmust\b', r'\bshall\b'
        ]
        
        universal_count = self._count_patterns(text.lower(), universal_patterns)
        
        # Concrete details decay; abstract principles persist
        concrete_patterns = [
            r'\$\d+', r'\d{4}', r'@\w+',  # Money, years, handles
            r'\b(Monday|Tuesday|Wednesday|Thursday|Friday)\b'
        ]
        
        concrete_count = self._count_patterns(text, concrete_patterns)
        
        # Balance
        resistance = (universal_count * 0.15) - (concrete_count * 0.1)
        return np.clip(resistance + 0.5, 0.0, 1.0)
    
    def _tau_to_scale(self, tau: float) -> TemporalScale:
        """Convert tau value to categorical scale"""
        if tau < 0.1:
            return TemporalScale.IMMEDIATE
        elif tau < 0.3:
            return TemporalScale.RECENT
        elif tau < 0.5:
            return TemporalScale.CONTEMPORARY
        elif tau < 0.7:
            return TemporalScale.HISTORICAL
        elif tau < 0.9:
            return TemporalScale.ANCIENT
        else:
            return TemporalScale.ETERNAL


def extract_tau(text: str) -> float:
    """
    Convenience function for GCT integration.
    
    Args:
        text: Text to analyze
        
    Returns:
        τ value (0-1)
    """
    analyzer = TemporalAnalyzer()
    signature = analyzer.analyze(text)
    return signature.tau


# === Tests ===
if __name__ == "__main__":
    print("=" * 70)
    print("TEMPORAL DIMENSION ANALYSIS (τ)")
    print("=" * 70)
    
    test_cases = [
        ("OMG just saw this breaking news live rn!!!", "Twitter reaction"),
        ("Yesterday's meeting addressed last week's concerns.", "Recent business"),
        ("The Constitution established precedents that shaped generations.", "Historical"),
        ("Every stone worn smooth by water teaches patience to those who listen.", "Ancient wisdom"),
        ("Truth persists. Lies decay. Time reveals all.", "Eternal aphorism"),
    ]
    
    analyzer = TemporalAnalyzer()
    
    for text, label in test_cases:
        sig = analyzer.analyze(text)
        print(f"\n[{label}]")
        print(f"  Text: {text[:60]}...")
        print(f"  τ = {sig.tau:.3f} ({sig.scale.name})")
        print(f"  Compression: {sig.compression_ratio:.2f}")
        print(f"  Decay Resistance: {sig.decay_resistance:.2f}")
        print(f"  Confidence: {sig.confidence:.2%}")
