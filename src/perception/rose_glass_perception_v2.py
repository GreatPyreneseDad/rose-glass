"""
Rose Glass Perception - Phase 2 Implementation
Internal processing only - no explicit perception reporting
"""

from typing import Dict, List, Optional, Tuple, Any, Callable
import numpy as np
from dataclasses import dataclass
from enum import Enum
import re

from .rose_glass_perception import (
    PerceptionPhase, DimensionalPattern, Perception, RoseGlassPerception
)
from .pattern_memory import PatternMemory
from .cultural_calibration import CulturalCalibration
from .breathing_patterns import BreathingPatternDetector
from .uncertainty_handler import UncertaintyHandler


class ResponseAdaptation:
    """Handles implicit response adaptation based on perception"""
    
    def __init__(self):
        self.adaptation_strategies = {
            'high_urgency': self._adapt_for_urgency,
            'deep_contemplation': self._adapt_for_contemplation,
            'collective_focus': self._adapt_for_collective,
            'individual_focus': self._adapt_for_individual,
            'low_coherence': self._adapt_for_clarity,
            'high_wisdom': self._adapt_for_wisdom,
            'mixed_signals': self._adapt_for_uncertainty
        }
        
    def adapt_response(self, response: str, perception: Perception) -> str:
        """
        Adapt response based on perception without explaining why
        All adaptations are implicit and natural
        """
        # Determine primary adaptation needed
        strategy = self._select_strategy(perception)
        
        # Apply rhythm matching first
        response = self._apply_rhythm(response, perception.rhythm)
        
        # Apply dimensional adaptations
        if strategy:
            response = self.adaptation_strategies[strategy](response, perception)
            
        # Apply uncertainty adaptations if needed
        if perception.uncertainty_level > 0.5:
            response = self._add_gentle_checking(response)
            
        return response
        
    def _select_strategy(self, perception: Perception) -> Optional[str]:
        """Select primary adaptation strategy based on perception"""
        dims = perception.dimensions
        
        # Priority order for adaptations
        if dims.q > 0.8:
            return 'high_urgency'
        elif perception.rhythm['pace'] == 'contemplative' and dims.rho > 0.7:
            return 'deep_contemplation'
        elif dims.f > 0.7:
            return 'collective_focus'
        elif dims.f < 0.3:
            return 'individual_focus'
        elif dims.psi < 0.3:
            return 'low_coherence'
        elif dims.rho > 0.8:
            return 'high_wisdom'
        elif perception.uncertainty_level > 0.5:
            return 'mixed_signals'
            
        return None
        
    def _apply_rhythm(self, response: str, rhythm: Dict) -> str:
        """Apply breathing pattern matching naturally"""
        if rhythm['pace'] == 'rapid' or rhythm['pace'] == 'staccato':
            # Shorten sentences naturally
            sentences = re.split(r'(?<=[.!?])\s+', response)
            shortened = []
            
            for sent in sentences:
                words = sent.split()
                if len(words) > 15:
                    # Break into shorter units
                    mid = len(words) // 2
                    shortened.append(' '.join(words[:mid]) + '.')
                    shortened.append(' '.join(words[mid:]))
                else:
                    shortened.append(sent)
                    
            response = ' '.join(shortened)
            
        elif rhythm['pace'] == 'contemplative':
            # Add natural pauses with punctuation
            response = response.replace('. ', '... ')
            response = response.replace(', ', ', ')  # Keep commas as natural pauses
            
        return response
        
    def _adapt_for_urgency(self, response: str, perception: Perception) -> str:
        """Adapt for high moral activation without saying why"""
        # Remove hedging language
        hedges = ['perhaps', 'maybe', 'might', 'possibly', 'it seems']
        for hedge in hedges:
            response = response.replace(hedge + ' ', '')
            response = response.replace(' ' + hedge, '')
            
        # Add action-oriented language
        if 'I can help' in response:
            response = response.replace('I can help', "I'll help")
        if 'We could' in response:
            response = response.replace('We could', "Let's")
            
        # Ensure ends with clear next step
        if not response.strip().endswith(('!', '?')):
            sentences = re.split(r'(?<=[.!?])\s+', response)
            if sentences:
                last = sentences[-1]
                if 'will' in last or "I'll" in last or "Let's" in last:
                    sentences[-1] = last.rstrip('.') + '.'
                response = ' '.join(sentences)
                
        return response
        
    def _adapt_for_contemplation(self, response: str, perception: Perception) -> str:
        """Adapt for deep contemplative communication"""
        # Add reflective openers naturally
        openers = [
            "When we consider this deeply",
            "Looking at this from a broader perspective",
            "There's wisdom in exploring"
        ]
        
        # Only add if response doesn't already have contemplative tone
        first_words = response.split()[:5]
        if not any(word in ['perhaps', 'considering', 'reflecting'] for word in first_words):
            import random
            if random.random() > 0.5:
                response = f"{random.choice(openers)}, {response.lower()}"
                
        return response
        
    def _adapt_for_collective(self, response: str, perception: Perception) -> str:
        """Shift to collective framing naturally"""
        # Natural pronoun shifts
        replacements = [
            ('I think', 'We might consider'),
            ('I suggest', 'We could'),
            ('you should', 'we might'),
            ('You can', 'We can'),
            ('I recommend', 'We could explore'),
            ('my view', 'our perspective'),
            ('your goal', 'our goal')
        ]
        
        for old, new in replacements:
            response = response.replace(old, new)
            response = response.replace(old.capitalize(), new.capitalize())
            
        return response
        
    def _adapt_for_individual(self, response: str, perception: Perception) -> str:
        """Adapt for individual focus"""
        # Emphasize personal agency
        replacements = [
            ('we might', 'you might'),
            ('we could', 'you could'),
            ('We can', 'You can'),
            ('our goal', 'your goal'),
            ('us to', 'you to')
        ]
        
        for old, new in replacements:
            response = response.replace(old, new)
            response = response.replace(old.capitalize(), new.capitalize())
            
        return response
        
    def _adapt_for_clarity(self, response: str, perception: Perception) -> str:
        """Add structure for low coherence without explaining why"""
        sentences = re.split(r'(?<=[.!?])\s+', response)
        
        if len(sentences) > 2:
            # Add subtle structural markers
            structured = []
            markers = ['First', 'Next', 'Also', 'Finally']
            
            for i, sent in enumerate(sentences):
                if i < len(markers) and i < len(sentences) - 1:
                    # Only add if sentence doesn't start with connector
                    if not any(sent.startswith(m) for m in ['However', 'But', 'And', 'So']):
                        structured.append(f"{markers[i]}, {sent.lower()}")
                    else:
                        structured.append(sent)
                else:
                    structured.append(sent)
                    
            response = ' '.join(structured)
            
        return response
        
    def _adapt_for_wisdom(self, response: str, perception: Perception) -> str:
        """Adapt for high wisdom context"""
        # Add depth markers naturally
        if 'this is' in response.lower():
            response = response.replace('this is', 'this represents')
        if 'it means' in response.lower():
            response = response.replace('it means', 'it suggests')
            
        # Add temporal depth
        if any(word in response.lower() for word in ['always', 'never']):
            response = response.replace('always', 'often')
            response = response.replace('never', 'rarely')
            
        return response
        
    def _add_gentle_checking(self, response: str) -> str:
        """Add gentle checking for uncertainty without explaining"""
        # Only if response doesn't already have checking
        if not any(phrase in response.lower() for phrase in 
                  ['does this', 'is this', 'would this', 'if this']):
            
            # Add natural checking at end
            if response.strip().endswith('.'):
                checks = [
                    " Does this resonate with your experience?",
                    " Is this aligned with what you're seeing?",
                    " Does this capture it?"
                ]
                import random
                response = response.rstrip('.') + random.choice(checks)
                
        return response


class RoseGlassPerceptionV2(RoseGlassPerception):
    """
    Phase 2 Implementation: Internal processing only
    All perception happens internally without explicit reporting
    """
    
    def __init__(self):
        super().__init__(PerceptionPhase.INTERNAL)
        self.response_adapter = ResponseAdaptation()
        self._implicit_mode = True
        
    def perceive_and_respond(self, user_input: str, response_generator: Callable) -> str:
        """
        Main Phase 2 method: Perceive and adapt response in one flow
        
        Args:
            user_input: User's message
            response_generator: Function that generates base response
            
        Returns:
            Adapted response without perception commentary
        """
        # Perceive internally
        perception = self.perceive(user_input)
        
        # Generate base response
        base_response = response_generator()
        
        # Adapt implicitly based on perception
        adapted = self.response_adapter.adapt_response(base_response, perception)
        
        return adapted
        
    def calibrate_response(self, perception: Perception, response_draft: str) -> str:
        """
        Override to ensure no explicit perception reporting in Phase 2
        """
        # Only apply natural adaptations
        return self.response_adapter.adapt_response(response_draft, perception)
        
    def _should_add_uncertainty_check(self, perception: Perception) -> bool:
        """Determine if gentle uncertainty checking is needed"""
        return (perception.uncertainty_level > 0.6 or
                (perception.dimensions.psi < 0.3 and perception.dimensions.q > 0.5))
                
    def get_adaptation_summary(self, perception: Perception) -> Dict[str, Any]:
        """
        For debugging only - shows what adaptations would be applied
        This method should NOT be used in production responses
        """
        strategy = self.response_adapter._select_strategy(perception)
        
        return {
            'primary_strategy': strategy,
            'rhythm_adaptation': perception.rhythm['pace'],
            'uncertainty_check': self._should_add_uncertainty_check(perception),
            'dimensional_focus': self._get_dimensional_focus(perception)
        }
        
    def _get_dimensional_focus(self, perception: Perception) -> str:
        """Identify which dimension is most prominent"""
        dims = perception.dimensions
        
        values = {
            'consistency': abs(dims.psi - 0.5),
            'wisdom': abs(dims.rho - 0.5),
            'activation': abs(dims.q - 0.5),
            'social': abs(dims.f - 0.5)
        }
        
        return max(values, key=values.get)
        
    def create_contextual_response(self, 
                                  user_input: str,
                                  base_response: str,
                                  context: Optional[Dict] = None) -> str:
        """
        Create a fully contextualized response based on perception
        This is the main method for Phase 2 operation
        """
        # Perceive with context
        perception = self.perceive(user_input, context)
        
        # Apply all adaptations implicitly
        adapted = self.calibrate_response(perception, base_response)
        
        # Store in memory for evolution tracking
        self.pattern_memory.update(perception)
        
        return adapted
        
    def handle_conversation_flow(self, 
                               messages: List[str],
                               responses: List[str]) -> Dict[str, Any]:
        """
        Analyze conversation flow and suggest next response approach
        For internal use - helps maintain coherent conversation flow
        """
        # Analyze pattern evolution
        if len(messages) >= 2:
            recent_perceptions = []
            for msg in messages[-3:]:  # Last 3 messages
                recent_perceptions.append(self.perceive(msg))
                
            # Detect trends
            coherence_trend = self._analyze_coherence_trend(recent_perceptions)
            emotional_trend = self._analyze_emotional_trend(recent_perceptions)
            
            return {
                'coherence_improving': coherence_trend > 0,
                'emotional_intensity': emotional_trend,
                'maintain_current_approach': coherence_trend > 0.1,
                'suggest_clarification': coherence_trend < -0.1
            }
            
        return {'maintain_current_approach': True}
        
    def _analyze_coherence_trend(self, perceptions: List[Perception]) -> float:
        """Analyze trend in coherence"""
        if len(perceptions) < 2:
            return 0.0
            
        coherence_values = [p.dimensions.psi for p in perceptions]
        
        # Simple linear trend
        x = np.arange(len(coherence_values))
        if len(x) > 1:
            trend = np.polyfit(x, coherence_values, 1)[0]
            return trend
        return 0.0
        
    def _analyze_emotional_trend(self, perceptions: List[Perception]) -> str:
        """Analyze emotional trajectory"""
        if not perceptions:
            return 'stable'
            
        q_values = [p.dimensions.q for p in perceptions]
        
        if len(q_values) >= 2:
            if q_values[-1] > q_values[-2] + 0.2:
                return 'escalating'
            elif q_values[-1] < q_values[-2] - 0.2:
                return 'calming'
                
        return 'stable'


# Convenience function for Phase 2 operation
def create_phase2_perception():
    """Create a Phase 2 perception instance"""
    return RoseGlassPerceptionV2()