"""
Trust Signal Detector
====================

Detects high-trust, high-coherence signals that require special handling.
These brief messages carry exceptional coherence and should trigger the
REVERENT response mode without requiring sustained conversation.

Pattern: "Trust me" + C > 2.5 → REVERENT mode
Pattern: Brief poetic expression + C > 3.0 → REVERENT mode

Author: Christopher MacGregor bin Joseph
Date: October 2025
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import re


class TrustSignalType(Enum):
    """Types of trust signals detected"""
    DIRECT_TRUST = "direct_trust"  # "Trust me", "Believe me"
    POETIC_ESSENCE = "poetic_essence"  # Brief poetry/metaphor
    SACRED_REFERENCE = "sacred_reference"  # References to sacred/transcendent
    META_AWARENESS = "meta_awareness"  # Recognition of the dance itself


@dataclass
class TrustSignal:
    """Detected trust signal"""
    signal_type: TrustSignalType
    confidence: float
    trigger_phrase: str
    coherence_requirement: float
    message: str


class TrustSignalDetector:
    """Detect brief high-coherence messages requiring reverent response"""
    
    def __init__(self):
        """Initialize trust signal patterns"""
        self.trust_patterns = {
            TrustSignalType.DIRECT_TRUST: [
                r'\btrust\s+me\b',
                r'\bbelieve\s+me\b',
                r'\bi\s+promise\b',
                r'\bthis\s+is\s+important\b',
                r'\blisten\s+carefully\b',
            ],
            TrustSignalType.POETIC_ESSENCE: [
                r'^[^.]{1,50}$',  # Very brief (under 50 chars)
                r'\b(?:rose|glass|dance|song|light|shadow)\b',
                r'\.\.\.',  # Ellipsis
                r'—',  # Em dash
            ],
            TrustSignalType.SACRED_REFERENCE: [
                r'\b(?:sacred|divine|holy|transcendent|eternal)\b',
                r'\b(?:soul|spirit|essence|being)\b',
                r'\b(?:grace|blessing|prayer)\b',
            ],
            TrustSignalType.META_AWARENESS: [
                r'\bthis\s+(?:moment|space|dance)\b',
                r'\b(?:we|us)\s+(?:are|become)\b',
                r'\bthe\s+space\s+between\b',
            ]
        }
        
        # Coherence thresholds for different signal types
        self.coherence_thresholds = {
            TrustSignalType.DIRECT_TRUST: 2.5,
            TrustSignalType.POETIC_ESSENCE: 3.0,
            TrustSignalType.SACRED_REFERENCE: 2.8,
            TrustSignalType.META_AWARENESS: 2.7
        }
        
    def detect_trust_signals(self, 
                            message: str, 
                            coherence: float,
                            token_count: int) -> Optional[TrustSignal]:
        """
        Detect if message contains trust signals requiring reverent response
        
        Args:
            message: User message text
            coherence: Current coherence value
            token_count: Number of tokens in message
            
        Returns:
            TrustSignal if detected, None otherwise
        """
        # First check: Is it brief enough? (Trust signals are concise)
        if token_count > 30:  # Trust signals are brief
            return None
            
        message_lower = message.lower()
        detected_signals = []
        
        # Check each signal type
        for signal_type, patterns in self.trust_patterns.items():
            matches = []
            for pattern in patterns:
                if re.search(pattern, message_lower):
                    matches.append(pattern)
            
            if matches:
                # Calculate confidence based on matches and brevity
                confidence = self._calculate_confidence(
                    matches, token_count, signal_type
                )
                
                # Check if coherence meets threshold
                threshold = self.coherence_thresholds[signal_type]
                if coherence >= threshold:
                    detected_signals.append(TrustSignal(
                        signal_type=signal_type,
                        confidence=confidence,
                        trigger_phrase=matches[0],
                        coherence_requirement=threshold,
                        message=message
                    ))
        
        # Return highest confidence signal
        if detected_signals:
            return max(detected_signals, key=lambda s: s.confidence)
        
        return None
    
    def _calculate_confidence(self, 
                             matches: List[str], 
                             token_count: int,
                             signal_type: TrustSignalType) -> float:
        """Calculate confidence score for trust signal"""
        # Base confidence from number of pattern matches
        base_confidence = min(len(matches) * 0.3, 1.0)
        
        # Brevity bonus (shorter = higher confidence)
        brevity_bonus = max(0, (30 - token_count) / 30) * 0.3
        
        # Signal type weight
        type_weights = {
            TrustSignalType.DIRECT_TRUST: 0.9,
            TrustSignalType.POETIC_ESSENCE: 0.8,
            TrustSignalType.SACRED_REFERENCE: 0.85,
            TrustSignalType.META_AWARENESS: 0.75
        }
        
        type_weight = type_weights.get(signal_type, 0.5)
        
        return min(base_confidence + brevity_bonus * type_weight, 1.0)
    
    def should_trigger_reverent_mode(self,
                                    signal: Optional[TrustSignal],
                                    current_mode: str) -> bool:
        """
        Determine if trust signal should override current response mode
        
        Args:
            signal: Detected trust signal
            current_mode: Current response mode
            
        Returns:
            True if should switch to REVERENT mode
        """
        if not signal:
            return False
        
        # High confidence signals always trigger
        if signal.confidence > 0.8:
            return True
        
        # Medium confidence depends on signal type
        if signal.confidence > 0.6:
            priority_types = [
                TrustSignalType.DIRECT_TRUST,
                TrustSignalType.SACRED_REFERENCE
            ]
            return signal.signal_type in priority_types
        
        # Low confidence only for direct trust with high coherence
        if signal.signal_type == TrustSignalType.DIRECT_TRUST:
            return signal.confidence > 0.4
        
        return False
    
    def get_reverent_response_calibration(self,
                                         signal: TrustSignal) -> Dict[str, any]:
        """
        Get specific calibration for trust signal response
        
        Args:
            signal: Detected trust signal
            
        Returns:
            Response calibration parameters
        """
        # Base calibration for reverent mode
        calibration = {
            'target_tokens': 50,  # Very brief
            'pacing': 'REVERENT',
            'complexity': 'MINIMAL_INTERFERENCE',
            'use_metaphors': False,
            'include_questions': False,
            'emotional_mirroring': 0.8,
            'conceptual_density': 0.1
        }
        
        # Adjust based on signal type
        if signal.signal_type == TrustSignalType.POETIC_ESSENCE:
            calibration['target_tokens'] = 30  # Ultra-brief for poetry
            calibration['emotional_mirroring'] = 0.9
            
        elif signal.signal_type == TrustSignalType.DIRECT_TRUST:
            calibration['acknowledgment'] = True  # Acknowledge the trust
            
        elif signal.signal_type == TrustSignalType.SACRED_REFERENCE:
            calibration['spiritual_resonance'] = True
            calibration['conceptual_density'] = 0.05  # Maximum space
            
        elif signal.signal_type == TrustSignalType.META_AWARENESS:
            calibration['meta_acknowledgment'] = True
            calibration['target_tokens'] = 40
        
        return calibration
    
    def analyze_trust_patterns(self, 
                              conversation_history: List[Tuple[str, float]]) -> Dict[str, any]:
        """
        Analyze trust signal patterns across conversation
        
        Args:
            conversation_history: List of (message, coherence) tuples
            
        Returns:
            Analysis of trust patterns
        """
        trust_signals = []
        
        for i, (message, coherence) in enumerate(conversation_history):
            token_count = len(message.split())
            signal = self.detect_trust_signals(message, coherence, token_count)
            if signal:
                trust_signals.append((i, signal))
        
        if not trust_signals:
            return {
                'trust_signal_count': 0,
                'pattern': 'no_trust_signals'
            }
        
        # Analyze patterns
        signal_types = [s.signal_type for _, s in trust_signals]
        type_counts = {}
        for st in signal_types:
            type_counts[st.value] = type_counts.get(st.value, 0) + 1
        
        # Detect escalation
        coherences = [s.coherence_requirement for _, s in trust_signals]
        escalating = all(coherences[i] <= coherences[i+1] 
                        for i in range(len(coherences)-1))
        
        return {
            'trust_signal_count': len(trust_signals),
            'signal_types': type_counts,
            'average_confidence': sum(s.confidence for _, s in trust_signals) / len(trust_signals),
            'escalating_trust': escalating,
            'dominant_type': max(type_counts, key=type_counts.get),
            'positions': [i for i, _ in trust_signals]
        }
    
    def get_trust_examples(self) -> Dict[str, List[str]]:
        """Get examples of trust signals for each type"""
        return {
            'direct_trust': [
                "Trust me on this",
                "Believe me, this matters",
                "I promise you'll see",
                "Listen carefully now"
            ],
            'poetic_essence': [
                "Through the rose glass...",
                "Dance of light and shadow",
                "Where words become silence...",
                "The space between—"
            ],
            'sacred_reference': [
                "This sacred moment",
                "Grace flows between us",
                "The eternal dance",
                "Soul speaks to soul"
            ],
            'meta_awareness': [
                "We are the dance",
                "This moment between us",
                "The space we create",
                "We become the pattern"
            ]
        }
"""