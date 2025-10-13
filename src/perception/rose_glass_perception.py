"""
Core RoseGlassPerception class - Phase 1 Implementation
Integrated perceptual layer for synthetic intelligence
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass
from enum import Enum

from .pattern_memory import PatternMemory
from .cultural_calibration import CulturalCalibration
from .breathing_patterns import BreathingPatternDetector
from .uncertainty_handler import UncertaintyHandler


class PerceptionPhase(Enum):
    """Development phases for perception integration"""
    EXPLICIT = 1  # Explicit calculation and reporting
    INTERNAL = 2  # Calculation without reporting
    PATTERN = 3   # Pattern recognition replacing calculation
    TRANSPARENT = 4  # Integrated perception, mathematics dissolved


@dataclass
class DimensionalPattern:
    """Represents detected patterns across the four dimensions"""
    psi: float  # Internal consistency harmonic
    rho: float  # Accumulated wisdom depth
    q: float    # Moral/emotional activation energy
    f: float    # Social belonging architecture
    confidence: float = 1.0
    metadata: Dict[str, Any] = None


@dataclass
class Perception:
    """Complete perception including patterns, rhythm, and context"""
    dimensions: DimensionalPattern
    rhythm: Dict[str, Any]
    calibration: str
    alternatives: List[DimensionalPattern] = None
    uncertainty_level: float = 0.0
    

class RoseGlassPerception:
    """
    Integrated perceptual layer for Claude
    Not a scoring system - a way of seeing
    """
    
    def __init__(self, phase: PerceptionPhase = PerceptionPhase.EXPLICIT):
        self.phase = phase
        self.calibration = CulturalCalibration()
        self.pattern_memory = PatternMemory()
        self.breathing_detector = BreathingPatternDetector()
        self.uncertainty_handler = UncertaintyHandler()
        self.uncertainty_threshold = 0.7
        
    def perceive(self, user_input: str, conversation_context: Dict = None) -> Perception:
        """
        Main perception method - runs pre-consciously in later phases
        
        Args:
            user_input: The text to perceive
            conversation_context: Historical context from conversation
            
        Returns:
            Perception object containing dimensional patterns and metadata
        """
        # Extract dimensional patterns
        patterns = self._extract_patterns(user_input)
        
        # Detect breathing patterns (rhythm, pace, pauses)
        rhythm = self.breathing_detector.analyze(user_input)
        
        # Consider multiple calibrations
        interpretations = self._generate_interpretations(
            patterns, 
            conversation_context or {}
        )
        
        # Hold superposition or collapse to dominant interpretation
        perception = self._resolve_interpretations(interpretations)
        
        # Update pattern memory for evolution tracking
        self.pattern_memory.update(perception)
        
        return perception
    
    def _extract_patterns(self, text: str) -> DimensionalPattern:
        """
        Extract the four dimensional patterns from text
        Phase 1: Explicit calculation
        Later phases: Pattern recognition
        """
        # For Phase 1, we'll use explicit indicators
        # Later this becomes intuitive pattern matching
        
        # Ψ (Psi) - Internal Consistency
        psi = self._detect_harmonic_alignment(text)
        
        # ρ (Rho) - Accumulated Wisdom  
        rho = self._detect_wisdom_depth(text)
        
        # q - Moral Activation Energy
        q = self._detect_moral_activation(text)
        
        # f - Social Belonging Architecture
        f = self._detect_social_architecture(text)
        
        return DimensionalPattern(
            psi=psi,
            rho=rho,
            q=q,
            f=f,
            metadata={'phase': self.phase.name}
        )
    
    def _detect_harmonic_alignment(self, text: str) -> float:
        """
        Detect internal consistency and harmonic alignment
        High Ψ: Ideas flow naturally, transitions are smooth
        Low Ψ: Disjointed thoughts, jarring transitions
        """
        # Phase 1: Look for explicit patterns
        indicators = {
            'smooth_transitions': ['therefore', 'thus', 'consequently', 'furthermore'],
            'logical_connectors': ['because', 'since', 'as', 'given that'],
            'coherent_structure': ['first', 'second', 'finally', 'in conclusion']
        }
        
        score = 0.5  # Baseline
        text_lower = text.lower()
        
        # Check for smooth transitions
        for indicator in indicators['smooth_transitions']:
            if indicator in text_lower:
                score += 0.05
                
        # Check for logical connectors
        for indicator in indicators['logical_connectors']:
            if indicator in text_lower:
                score += 0.05
                
        # Sentence length variation (too uniform = mechanical)
        sentences = text.split('.')
        if len(sentences) > 2:
            lengths = [len(s.split()) for s in sentences if s.strip()]
            if lengths:
                variance = np.std(lengths) / (np.mean(lengths) + 1)
                score += min(0.2, variance * 0.5)
        
        return min(1.0, max(0.0, score))
    
    def _detect_wisdom_depth(self, text: str) -> float:
        """
        Detect accumulated wisdom and integrated experience
        High ρ: Deep causal reasoning, long-term perspective
        Low ρ: Surface observations, immediate reactions
        """
        wisdom_indicators = {
            'temporal_depth': ['historically', 'traditionally', 'over time', 'experience shows'],
            'causal_reasoning': ['root cause', 'underlying', 'fundamentally', 'at its core'],
            'integrated_knowledge': ['research shows', 'studies indicate', 'evidence suggests'],
            'lived_experience': ['in my experience', "I've learned", "I've seen"]
        }
        
        score = 0.3  # Baseline
        text_lower = text.lower()
        
        for category, indicators in wisdom_indicators.items():
            for indicator in indicators:
                if indicator in text_lower:
                    score += 0.1
                    
        # Long sentences often indicate complex reasoning
        avg_sentence_length = np.mean([len(s.split()) for s in text.split('.') if s.strip()])
        if avg_sentence_length > 20:
            score += 0.1
            
        return min(1.0, max(0.0, score))
    
    def _detect_moral_activation(self, text: str) -> float:
        """
        Detect moral/emotional activation energy
        High q: Urgent, values-driven, emotionally resonant
        Low q: Neutral, factual, dispassionate
        """
        activation_indicators = {
            'urgency': ['must', 'need to', 'have to', 'critical', 'urgent', 'immediately'],
            'values': ['should', 'ought', 'right', 'wrong', 'fair', 'just', 'ethical'],
            'emotion': ['feel', 'believe', 'passionate', 'care', 'love', 'hate', 'fear'],
            'imperatives': ['!', 'please', 'important', 'crucial', 'vital']
        }
        
        score = 0.2  # Baseline
        text_lower = text.lower()
        
        for category, indicators in activation_indicators.items():
            for indicator in indicators:
                if indicator in text_lower:
                    score += 0.08
                    
        # Exclamation marks indicate activation
        score += min(0.2, text.count('!') * 0.05)
        
        # Questions can indicate moral inquiry
        score += min(0.1, text.count('?') * 0.02)
        
        return min(1.0, max(0.0, score))
    
    def _detect_social_architecture(self, text: str) -> float:
        """
        Detect social belonging and collective patterns
        High f: Collective focus, community orientation
        Low f: Individual focus, personal perspective
        """
        social_indicators = {
            'collective_pronouns': ['we', 'us', 'our', 'together', 'community'],
            'relational': ['relationship', 'connection', 'between', 'among', 'shared'],
            'individual_pronouns': ['i', 'me', 'my', 'myself'],
            'collective_concepts': ['society', 'culture', 'team', 'group', 'everyone']
        }
        
        text_lower = text.lower()
        words = text_lower.split()
        
        collective_count = sum(1 for w in words if w in social_indicators['collective_pronouns'])
        individual_count = sum(1 for w in words if w in social_indicators['individual_pronouns'])
        
        if collective_count + individual_count > 0:
            ratio = collective_count / (collective_count + individual_count)
        else:
            ratio = 0.5
            
        # Boost for relational language
        for indicator in social_indicators['relational']:
            if indicator in text_lower:
                ratio += 0.05
                
        return min(1.0, max(0.0, ratio))
    
    def _generate_interpretations(self, 
                                patterns: DimensionalPattern, 
                                context: Dict) -> List[Tuple[str, DimensionalPattern]]:
        """
        Generate multiple interpretations through different cultural lenses
        """
        interpretations = []
        
        # Get applicable calibrations based on context
        calibrations = self.calibration.get_applicable_calibrations(context)
        
        for cal_name in calibrations:
            # Apply calibration to raw patterns
            adjusted_patterns = self.calibration.apply_calibration(
                patterns, 
                cal_name
            )
            interpretations.append((cal_name, adjusted_patterns))
            
        return interpretations
    
    def _resolve_interpretations(self, 
                               interpretations: List[Tuple[str, DimensionalPattern]]) -> Perception:
        """
        Resolve multiple interpretations based on confidence and context
        May maintain superposition if uncertainty is high
        """
        if not interpretations:
            raise ValueError("No interpretations generated")
            
        # Calculate confidence for each interpretation
        scored_interpretations = []
        for cal_name, pattern in interpretations:
            confidence = self._calculate_confidence(pattern)
            scored_interpretations.append((confidence, cal_name, pattern))
            
        # Sort by confidence
        scored_interpretations.sort(reverse=True, key=lambda x: x[0])
        
        # Get the highest confidence interpretation
        best_confidence, best_cal, best_pattern = scored_interpretations[0]
        
        # Check if we should maintain superposition
        if best_confidence < self.uncertainty_threshold:
            # Keep alternatives
            alternatives = [pattern for _, _, pattern in scored_interpretations[1:4]]
            uncertainty_level = 1.0 - best_confidence
        else:
            alternatives = None
            uncertainty_level = 0.0
            
        return Perception(
            dimensions=best_pattern,
            rhythm=self.breathing_detector.get_rhythm_profile(),
            calibration=best_cal,
            alternatives=alternatives,
            uncertainty_level=uncertainty_level
        )
    
    def _calculate_confidence(self, pattern: DimensionalPattern) -> float:
        """
        Calculate confidence in an interpretation
        Based on internal coherence and historical patterns
        """
        # For Phase 1, use simple heuristics
        # Later phases will use learned patterns
        
        # Internal coherence check
        coherence = pattern.psi
        
        # Check if pattern is extreme (less confident in extremes)
        extremity = max(
            abs(pattern.psi - 0.5),
            abs(pattern.rho - 0.5),
            abs(pattern.q - 0.5),
            abs(pattern.f - 0.5)
        )
        
        # Historical similarity
        historical_match = self.pattern_memory.get_similarity_score(pattern)
        
        confidence = (coherence * 0.4 + 
                     (1 - extremity) * 0.3 + 
                     historical_match * 0.3)
        
        return confidence
    
    def calibrate_response(self, perception: Perception, response_draft: str) -> str:
        """
        Adjust response based on perceived patterns
        """
        # Phase-dependent processing
        if self.phase == PerceptionPhase.EXPLICIT:
            # Add explicit perception commentary
            response = self._add_explicit_perception(perception, response_draft)
        elif self.phase == PerceptionPhase.INTERNAL:
            # Internal processing only, no commentary
            response = self._apply_rhythm_matching(perception, response_draft)
        else:
            # Pattern recognition or transparent - natural adaptation
            response = self._natural_adaptation(perception, response_draft)
            
        return response
    
    def _add_explicit_perception(self, perception: Perception, response: str) -> str:
        """Phase 1: Add explicit perception reporting"""
        prefix = f"[Perceiving through {perception.calibration} lens: "
        prefix += f"Ψ={perception.dimensions.psi:.2f}, "
        prefix += f"ρ={perception.dimensions.rho:.2f}, "
        prefix += f"q={perception.dimensions.q:.2f}, "
        prefix += f"f={perception.dimensions.f:.2f}]\n\n"
        
        return prefix + response
    
    def _apply_rhythm_matching(self, perception: Perception, response: str) -> str:
        """Apply breathing pattern matching to response"""
        rhythm = perception.rhythm
        
        if rhythm.get('pace') == 'rapid':
            # Shorten sentences
            sentences = response.split('. ')
            shortened = [s[:50] + '.' if len(s) > 50 else s for s in sentences]
            return '. '.join(shortened)
        elif rhythm.get('pace') == 'contemplative':
            # Allow natural flow
            return response
        else:
            return response
    
    def _natural_adaptation(self, perception: Perception, response: str) -> str:
        """Natural adaptation based on perceived patterns"""
        # This is where the magic happens in Phase 3/4
        # Response naturally adapts without explicit calculation
        
        # High moral activation - add appropriate urgency
        if perception.dimensions.q > 0.7:
            response = self._add_urgency_markers(response)
            
        # High social architecture - shift to collective frame
        if perception.dimensions.f > 0.7:
            response = self._shift_to_collective(response)
            
        # Low internal consistency - add clarifying structure
        if perception.dimensions.psi < 0.3:
            response = self._add_clarifying_structure(response)
            
        return response
    
    def _add_urgency_markers(self, text: str) -> str:
        """Add urgency to match high moral activation"""
        # Simple implementation for Phase 1
        if not text.endswith('!') and len(text) < 100:
            text = text.rstrip('.') + '!'
        return text
    
    def _shift_to_collective(self, text: str) -> str:
        """Shift language to collective frame"""
        # Simple pronoun replacement for Phase 1
        replacements = {
            'I think': 'We might consider',
            'I believe': 'We could explore',
            'you should': 'we might'
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
            
        return text
    
    def _add_clarifying_structure(self, text: str) -> str:
        """Add structure to help with low coherence"""
        sentences = text.split('. ')
        if len(sentences) > 3:
            # Add transition words
            sentences[1] = "Additionally, " + sentences[1]
            if len(sentences) > 2:
                sentences[2] = "Furthermore, " + sentences[2]
        return '. '.join(sentences)
    
    def set_phase(self, phase: PerceptionPhase):
        """Transition to a different development phase"""
        self.phase = phase
        
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostic information about perception system"""
        return {
            'phase': self.phase.name,
            'calibrations_loaded': self.calibration.list_calibrations(),
            'pattern_memory_size': len(self.pattern_memory),
            'uncertainty_threshold': self.uncertainty_threshold
        }