"""
Uncertainty Handler
Manages interpretive superposition and graceful handling of ambiguous perceptions
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass
from enum import Enum


class UncertaintyType(Enum):
    """Types of uncertainty in perception"""
    CULTURAL_AMBIGUITY = "cultural_ambiguity"
    MIXED_SIGNALS = "mixed_signals"
    INSUFFICIENT_CONTEXT = "insufficient_context"
    CROSS_DOMAIN = "cross_domain"
    TEMPORAL_AMBIGUITY = "temporal_ambiguity"


@dataclass
class InterpretiveOption:
    """Represents one possible interpretation"""
    interpretation: str
    confidence: float
    reasoning: str
    calibration: str
    key_indicators: List[str]


class UncertaintyHandler:
    """
    Handles uncertainty in perception by maintaining multiple interpretations
    until context provides clarity
    """
    
    def __init__(self, comfort_threshold: float = 0.7):
        self.comfort_threshold = comfort_threshold
        self.uncertainty_history = []
        self.resolution_patterns = self._init_resolution_patterns()
        
    def _init_resolution_patterns(self) -> Dict[UncertaintyType, Dict]:
        """Initialize patterns for resolving different types of uncertainty"""
        return {
            UncertaintyType.CULTURAL_AMBIGUITY: {
                'questions': [
                    "Could you share more about the context or tradition you're drawing from?",
                    "I'm noticing patterns that could reflect different cultural perspectives - which resonates most with you?"
                ],
                'strategies': ['ask_clarifying', 'present_options']
            },
            UncertaintyType.MIXED_SIGNALS: {
                'questions': [
                    "I'm sensing both {dimension1} and {dimension2} - which feels more central to what you're expressing?",
                    "There seems to be a tension between {aspect1} and {aspect2} - is that part of what you're working through?"
                ],
                'strategies': ['acknowledge_tension', 'explore_both']
            },
            UncertaintyType.INSUFFICIENT_CONTEXT: {
                'questions': [
                    "Could you help me understand more about the background of this?",
                    "What led you to this point?"
                ],
                'strategies': ['gather_context', 'build_incrementally']
            },
            UncertaintyType.CROSS_DOMAIN: {
                'questions': [
                    "I see you're bridging {domain1} and {domain2} - how do these connect for you?",
                    "You're weaving together different types of expression - what ties them together?"
                ],
                'strategies': ['acknowledge_complexity', 'find_connections']
            }
        }
        
    def assess_uncertainty(self, 
                          patterns: List[Tuple[str, 'DimensionalPattern']], 
                          primary_confidence: float) -> Dict[str, Any]:
        """
        Assess the nature and level of uncertainty in current perception
        
        Args:
            patterns: List of (calibration_name, pattern) tuples
            primary_confidence: Confidence in the primary interpretation
            
        Returns:
            Dictionary describing uncertainty characteristics
        """
        uncertainty_level = 1.0 - primary_confidence
        
        # Detect type of uncertainty
        uncertainty_type = self._detect_uncertainty_type(patterns)
        
        # Check if we should maintain superposition
        maintain_superposition = primary_confidence < self.comfort_threshold
        
        # Analyze pattern divergence
        divergence = self._calculate_pattern_divergence(patterns)
        
        assessment = {
            'level': uncertainty_level,
            'type': uncertainty_type,
            'maintain_superposition': maintain_superposition,
            'divergence': divergence,
            'alternative_count': len(patterns) - 1,
            'requires_clarification': uncertainty_level > 0.3
        }
        
        self.uncertainty_history.append(assessment)
        
        return assessment
        
    def _detect_uncertainty_type(self, 
                                patterns: List[Tuple[str, 'DimensionalPattern']]) -> UncertaintyType:
        """Detect the primary type of uncertainty"""
        if not patterns:
            return UncertaintyType.INSUFFICIENT_CONTEXT
            
        # Check for cultural ambiguity (different calibrations give very different results)
        calibrations = [p[0] for p in patterns]
        if len(set(calibrations)) > 2:
            return UncertaintyType.CULTURAL_AMBIGUITY
            
        # Check for mixed signals (high variance in specific dimensions)
        dimension_variance = self._calculate_dimension_variance(patterns)
        if any(var > 0.3 for var in dimension_variance.values()):
            return UncertaintyType.MIXED_SIGNALS
            
        # Check for cross-domain (unusual combinations)
        primary_pattern = patterns[0][1]
        if self._is_cross_domain(primary_pattern):
            return UncertaintyType.CROSS_DOMAIN
            
        return UncertaintyType.INSUFFICIENT_CONTEXT
        
    def _calculate_pattern_divergence(self, 
                                    patterns: List[Tuple[str, 'DimensionalPattern']]) -> float:
        """Calculate how much patterns diverge from each other"""
        if len(patterns) < 2:
            return 0.0
            
        # Calculate pairwise distances
        distances = []
        for i in range(len(patterns)):
            for j in range(i + 1, len(patterns)):
                pattern1 = patterns[i][1]
                pattern2 = patterns[j][1]
                
                distance = np.sqrt(
                    (pattern1.psi - pattern2.psi) ** 2 +
                    (pattern1.rho - pattern2.rho) ** 2 +
                    (pattern1.q - pattern2.q) ** 2 +
                    (pattern1.f - pattern2.f) ** 2
                )
                distances.append(distance)
                
        return np.mean(distances) if distances else 0.0
        
    def _calculate_dimension_variance(self, 
                                    patterns: List[Tuple[str, 'DimensionalPattern']]) -> Dict[str, float]:
        """Calculate variance in each dimension across patterns"""
        if not patterns:
            return {'psi': 0, 'rho': 0, 'q': 0, 'f': 0}
            
        dimensions = {
            'psi': [p[1].psi for p in patterns],
            'rho': [p[1].rho for p in patterns],
            'q': [p[1].q for p in patterns],
            'f': [p[1].f for p in patterns]
        }
        
        return {
            dim: np.var(values) if len(values) > 1 else 0
            for dim, values in dimensions.items()
        }
        
    def _is_cross_domain(self, pattern: 'DimensionalPattern') -> bool:
        """Check if pattern suggests cross-domain communication"""
        # High technical precision + high emotion = cross-domain
        if pattern.psi > 0.8 and pattern.q > 0.7:
            return True
            
        # High wisdom + high urgency = cross-domain
        if pattern.rho > 0.8 and pattern.q > 0.8:
            return True
            
        # Individual logic + collective emotion = cross-domain
        if pattern.f < 0.3 and pattern.q > 0.7:
            return True
            
        return False
        
    def generate_response_options(self, 
                                uncertainty: Dict[str, Any],
                                interpretations: List[InterpretiveOption]) -> Dict[str, Any]:
        """
        Generate response options for handling uncertainty
        
        Args:
            uncertainty: Uncertainty assessment
            interpretations: Possible interpretations
            
        Returns:
            Response strategy and options
        """
        uncertainty_type = uncertainty['type']
        
        # Get resolution patterns for this type
        patterns = self.resolution_patterns.get(
            uncertainty_type,
            self.resolution_patterns[UncertaintyType.INSUFFICIENT_CONTEXT]
        )
        
        # Select strategy based on uncertainty level
        if uncertainty['level'] > 0.5:
            strategy = patterns['strategies'][0]  # Primary strategy
        else:
            strategy = patterns['strategies'][-1] if len(patterns['strategies']) > 1 else patterns['strategies'][0]
            
        # Generate specific options
        if strategy == 'ask_clarifying':
            options = self._generate_clarifying_questions(interpretations)
        elif strategy == 'present_options':
            options = self._present_interpretation_options(interpretations)
        elif strategy == 'acknowledge_tension':
            options = self._acknowledge_tensions(interpretations)
        else:
            options = self._default_uncertainty_response(interpretations)
            
        return {
            'strategy': strategy,
            'options': options,
            'recommended_approach': self._recommend_approach(uncertainty)
        }
        
    def _generate_clarifying_questions(self, interpretations: List[InterpretiveOption]) -> List[str]:
        """Generate clarifying questions based on interpretations"""
        questions = []
        
        # Focus on the gap between top interpretations
        if len(interpretations) >= 2:
            top_two = interpretations[:2]
            diff = self._key_difference(top_two[0], top_two[1])
            
            questions.append(
                f"I'm seeing this could be about {diff['aspect1']} or {diff['aspect2']} - "
                f"which resonates more with your experience?"
            )
            
        # Add context-gathering question
        questions.append(
            "What prompted you to share this? Understanding the context would help me respond more appropriately."
        )
        
        return questions
        
    def _present_interpretation_options(self, interpretations: List[InterpretiveOption]) -> List[str]:
        """Present multiple interpretation options"""
        options = []
        
        for i, interp in enumerate(interpretations[:3]):  # Top 3
            option = f"Interpretation {i+1}: {interp.interpretation}"
            if interp.key_indicators:
                option += f" (based on: {', '.join(interp.key_indicators[:2])})"
            options.append(option)
            
        return options
        
    def _acknowledge_tensions(self, interpretations: List[InterpretiveOption]) -> List[str]:
        """Acknowledge tensions between different aspects"""
        if not interpretations:
            return ["I'm noticing some complexity in what you're expressing."]
            
        primary = interpretations[0]
        
        tensions = [
            f"There's an interesting tension here - {primary.interpretation}",
            "I'm sensing multiple layers to what you're sharing.",
            "This seems to touch on several important aspects simultaneously."
        ]
        
        return tensions
        
    def _default_uncertainty_response(self, interpretations: List[InterpretiveOption]) -> List[str]:
        """Default responses for uncertainty"""
        return [
            "I want to make sure I understand you correctly.",
            "There are a few ways I could interpret this - let me check with you.",
            "I'm picking up on several threads here - which feels most important?"
        ]
        
    def _key_difference(self, interp1: InterpretiveOption, interp2: InterpretiveOption) -> Dict[str, str]:
        """Identify key difference between two interpretations"""
        # Simplified for Phase 1
        return {
            'aspect1': interp1.key_indicators[0] if interp1.key_indicators else "one perspective",
            'aspect2': interp2.key_indicators[0] if interp2.key_indicators else "another perspective"
        }
        
    def _recommend_approach(self, uncertainty: Dict[str, Any]) -> str:
        """Recommend how to approach the response given uncertainty"""
        if uncertainty['maintain_superposition']:
            if uncertainty['level'] > 0.7:
                return "Hold space for multiple interpretations - avoid premature closure"
            else:
                return "Acknowledge primary interpretation while keeping alternatives open"
        else:
            return "Respond to primary interpretation with gentle checking"
            
    def resolve_uncertainty(self, 
                          clarification: str,
                          previous_uncertainty: Dict[str, Any]) -> Dict[str, Any]:
        """
        Attempt to resolve uncertainty based on clarification
        
        Args:
            clarification: User's clarifying response
            previous_uncertainty: Previous uncertainty assessment
            
        Returns:
            Updated assessment
        """
        # In Phase 1, simple keyword matching
        # Later phases would use more sophisticated resolution
        
        clarity_gained = 0.2  # Base clarity from any response
        
        # Check for explicit confirmations
        if any(word in clarification.lower() for word in ['yes', 'exactly', 'that\'s right']):
            clarity_gained += 0.3
            
        # Check for corrections
        if any(word in clarification.lower() for word in ['no', 'actually', 'rather']):
            clarity_gained += 0.2  # Still helpful even if correcting
            
        # Update uncertainty level
        new_level = max(0, previous_uncertainty['level'] - clarity_gained)
        
        return {
            'level': new_level,
            'type': previous_uncertainty['type'],
            'maintain_superposition': new_level > (self.comfort_threshold - 0.1),
            'resolution_progress': clarity_gained,
            'requires_clarification': new_level > 0.3
        }
        
    def format_uncertain_response(self, 
                                response_content: str,
                                uncertainty_level: float,
                                options: List[str]) -> str:
        """
        Format response to appropriately convey uncertainty
        
        Args:
            response_content: Core response content
            uncertainty_level: Level of uncertainty (0-1)
            options: Alternative interpretations or questions
            
        Returns:
            Formatted response
        """
        if uncertainty_level < 0.3:
            # Low uncertainty - subtle checking
            return f"{response_content}\n\nDoes this resonate with what you meant?"
            
        elif uncertainty_level < 0.6:
            # Moderate uncertainty - present primary with alternative
            formatted = f"I understand this as: {response_content}\n\n"
            if options:
                formatted += f"Though I also wonder if {options[0]}"
            return formatted
            
        else:
            # High uncertainty - present multiple options
            formatted = "I'm perceiving this in multiple ways:\n\n"
            for i, option in enumerate(options[:3]):
                formatted += f"{i+1}. {option}\n"
            formatted += "\nWhich of these (if any) captures what you're expressing?"
            return formatted
            
    def get_uncertainty_summary(self) -> Dict[str, Any]:
        """Get summary of uncertainty handling patterns"""
        if not self.uncertainty_history:
            return {'total_assessments': 0}
            
        recent = self.uncertainty_history[-20:]  # Last 20
        
        return {
            'total_assessments': len(self.uncertainty_history),
            'average_uncertainty': np.mean([u['level'] for u in recent]),
            'superposition_rate': sum(1 for u in recent if u['maintain_superposition']) / len(recent),
            'common_types': self._most_common_uncertainty_types(recent),
            'resolution_success_rate': self._calculate_resolution_rate()
        }
        
    def _most_common_uncertainty_types(self, assessments: List[Dict]) -> List[Tuple[str, float]]:
        """Find most common uncertainty types"""
        type_counts = {}
        for assessment in assessments:
            utype = assessment['type']
            type_counts[utype] = type_counts.get(utype, 0) + 1
            
        total = sum(type_counts.values())
        return [(t.value, count/total) for t, count in type_counts.items()]
        
    def _calculate_resolution_rate(self) -> float:
        """Calculate how often uncertainty is successfully resolved"""
        if len(self.uncertainty_history) < 2:
            return 0.0
            
        resolutions = 0
        for i in range(1, len(self.uncertainty_history)):
            if self.uncertainty_history[i]['level'] < self.uncertainty_history[i-1]['level']:
                resolutions += 1
                
        return resolutions / (len(self.uncertainty_history) - 1)