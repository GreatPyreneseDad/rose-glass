"""
Token Multiplier Limiter
========================

Enforces safe token response ratios to prevent overwhelming users with
excessive output. The core insight: Never respond with more than 3x the
user's input tokens, with dynamic adjustments based on conversation state.

Key Rules:
- Default: 1.5x multiplier
- High coherence: up to 3x multiplier  
- Low coherence: 0.5x multiplier
- Crisis: 0.3x multiplier

Author: Christopher MacGregor bin Joseph
Date: October 2025
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np


class MultiplierMode(Enum):
    """Token multiplier modes based on conversation state"""
    CRISIS = "crisis"  # 0.3x - Minimal response
    GROUNDING = "grounding"  # 0.5x - Brief, anchoring
    BALANCED = "balanced"  # 1.0x - Match user length
    STANDARD = "standard"  # 1.5x - Default multiplier
    EXPLORATORY = "exploratory"  # 2.0x - Room to explore
    EXPANSIVE = "expansive"  # 3.0x - Maximum safe expansion


@dataclass
class TokenLimit:
    """Calculated token limits with reasoning"""
    user_tokens: int
    raw_multiplier: float
    adjusted_multiplier: float
    token_limit: int
    mode: MultiplierMode
    adjustments: List[str]
    hard_cap: int = 500  # Never exceed this


class TokenMultiplierLimiter:
    """Enforce safe token response ratios"""
    
    def __init__(self):
        """Initialize token limiter with safety rules"""
        # Base multipliers for different coherence ranges
        self.coherence_multipliers = [
            (0.0, 1.0, 0.5),   # Very low coherence
            (1.0, 1.5, 0.8),   # Low coherence
            (1.5, 2.0, 1.0),   # Medium-low coherence
            (2.0, 2.5, 1.5),   # Medium coherence
            (2.5, 3.0, 2.0),   # Medium-high coherence
            (3.0, 3.5, 2.5),   # High coherence
            (3.5, 4.0, 3.0),   # Very high coherence
        ]
        
        # Special case multipliers
        self.special_cases = {
            'crisis_detected': 0.3,
            'information_overload': 0.2,
            'trust_signal': 0.6,
            'poetic_mode': 0.8,
            'mission_mode': 2.0,  # Missions need more space
            'first_message': 1.2,  # Gentle start
        }
        
        # Message length categories
        self.length_categories = [
            (0, 10, 'micro'),      # Very brief
            (10, 30, 'short'),     # Short message
            (30, 50, 'medium'),    # Medium message
            (50, 100, 'long'),     # Long message
            (100, 200, 'verbose'), # Verbose message
            (200, float('inf'), 'excessive')  # Excessive length
        ]
        
    def calculate_token_limit(self,
                             user_tokens: int,
                             coherence: float,
                             conversation_state: Dict[str, any]) -> TokenLimit:
        """
        Calculate safe token limit for response
        
        Args:
            user_tokens: Number of tokens in user message
            coherence: Current coherence value
            conversation_state: Dictionary with conversation indicators
            
        Returns:
            TokenLimit object with calculated limits
        """
        # Get base multiplier from coherence
        base_multiplier = self._get_coherence_multiplier(coherence)
        
        # Start with base
        adjusted_multiplier = base_multiplier
        adjustments = []
        
        # Apply special case adjustments
        if conversation_state.get('crisis_detected'):
            adjusted_multiplier = min(adjusted_multiplier,
                                    self.special_cases['crisis_detected'])
            adjustments.append("Crisis mode: severe reduction")
            
        if conversation_state.get('information_overload'):
            adjusted_multiplier = min(adjusted_multiplier,
                                    self.special_cases['information_overload'])
            adjustments.append("Information overload: minimal response")
            
        if conversation_state.get('trust_signal_detected'):
            adjusted_multiplier = min(adjusted_multiplier,
                                    self.special_cases['trust_signal'])
            adjustments.append("Trust signal: brief acknowledgment")
            
        if conversation_state.get('mission_mode'):
            # Mission mode can increase multiplier
            adjusted_multiplier = max(adjusted_multiplier,
                                    self.special_cases['mission_mode'])
            adjustments.append("Mission mode: expanded response allowed")
            
        # Adjust for user message length
        length_adjustment = self._get_length_adjustment(user_tokens)
        if length_adjustment != 1.0:
            adjusted_multiplier *= length_adjustment
            adjustments.append(f"Length adjustment: {length_adjustment:.1f}x")
        
        # Apply conversation momentum adjustments
        momentum_adjustment = self._get_momentum_adjustment(conversation_state)
        if momentum_adjustment != 1.0:
            adjusted_multiplier *= momentum_adjustment
            adjustments.append(f"Momentum adjustment: {momentum_adjustment:.1f}x")
        
        # Calculate final token limit
        raw_limit = int(user_tokens * adjusted_multiplier)
        
        # Apply hard caps
        token_limit = min(raw_limit, self.hard_cap)
        if raw_limit > self.hard_cap:
            adjustments.append(f"Hard cap applied: {self.hard_cap} tokens")
        
        # Ensure minimum viable response
        if token_limit < 20 and not conversation_state.get('crisis_detected'):
            token_limit = 20
            adjustments.append("Minimum viable response: 20 tokens")
        
        # Determine mode
        mode = self._determine_mode(adjusted_multiplier)
        
        return TokenLimit(
            user_tokens=user_tokens,
            raw_multiplier=base_multiplier,
            adjusted_multiplier=adjusted_multiplier,
            token_limit=token_limit,
            mode=mode,
            adjustments=adjustments
        )
    
    def _get_coherence_multiplier(self, coherence: float) -> float:
        """Get base multiplier from coherence value"""
        for min_c, max_c, multiplier in self.coherence_multipliers:
            if min_c <= coherence < max_c:
                return multiplier
        
        # Default for out of range
        if coherence >= 4.0:
            return 3.0
        return 0.5
    
    def _get_length_adjustment(self, user_tokens: int) -> float:
        """Adjust multiplier based on user message length"""
        for min_tokens, max_tokens, category in self.length_categories:
            if min_tokens <= user_tokens < max_tokens:
                break
        
        # Adjustments by category
        adjustments = {
            'micro': 3.0,      # Very brief messages need more context
            'short': 2.0,      # Short messages can expand
            'medium': 1.0,     # Medium messages are balanced
            'long': 0.8,       # Long messages need compression
            'verbose': 0.6,    # Verbose messages need brevity
            'excessive': 0.4   # Excessive length needs strong compression
        }
        
        return adjustments.get(category, 1.0)
    
    def _get_momentum_adjustment(self, conversation_state: Dict[str, any]) -> float:
        """Adjust based on conversation momentum"""
        # Recent response lengths trending up or down?
        recent_ratios = conversation_state.get('recent_response_ratios', [])
        
        if not recent_ratios or len(recent_ratios) < 3:
            return 1.0
        
        # Check trend in last 3 responses
        recent = recent_ratios[-3:]
        
        # All increasing = reduce multiplier to break pattern
        if all(recent[i] < recent[i+1] for i in range(len(recent)-1)):
            return 0.7  # Brake the escalation
        
        # All decreasing = can safely increase
        if all(recent[i] > recent[i+1] for i in range(len(recent)-1)):
            return 1.2  # Room to expand again
        
        return 1.0
    
    def _determine_mode(self, multiplier: float) -> MultiplierMode:
        """Determine mode from final multiplier"""
        if multiplier <= 0.3:
            return MultiplierMode.CRISIS
        elif multiplier <= 0.6:
            return MultiplierMode.GROUNDING
        elif multiplier <= 1.2:
            return MultiplierMode.BALANCED
        elif multiplier <= 1.7:
            return MultiplierMode.STANDARD
        elif multiplier <= 2.5:
            return MultiplierMode.EXPLORATORY
        else:
            return MultiplierMode.EXPANSIVE
    
    def validate_response_length(self,
                               response_tokens: int,
                               token_limit: TokenLimit) -> Dict[str, any]:
        """
        Validate if response respects token limits
        
        Args:
            response_tokens: Actual response token count
            token_limit: Calculated limit
            
        Returns:
            Validation results
        """
        adherence_ratio = response_tokens / token_limit.token_limit
        
        validation = {
            'within_limit': response_tokens <= token_limit.token_limit,
            'adherence_ratio': adherence_ratio,
            'response_tokens': response_tokens,
            'limit': token_limit.token_limit,
            'severity': 'ok'
        }
        
        # Categorize severity of violation
        if adherence_ratio <= 1.0:
            validation['severity'] = 'ok'
        elif adherence_ratio <= 1.2:
            validation['severity'] = 'minor_overage'
        elif adherence_ratio <= 1.5:
            validation['severity'] = 'moderate_overage'
        else:
            validation['severity'] = 'severe_overage'
        
        # Add recommendations
        if validation['severity'] != 'ok':
            validation['recommendation'] = self._get_overage_recommendation(
                adherence_ratio, token_limit.mode
            )
        
        return validation
    
    def _get_overage_recommendation(self, 
                                   adherence_ratio: float,
                                   mode: MultiplierMode) -> str:
        """Get recommendation for token overage"""
        if mode == MultiplierMode.CRISIS:
            return "Critical: Immediately reduce to minimal responses"
        elif mode == MultiplierMode.GROUNDING:
            return "Shorten responses, focus on essentials only"
        elif adherence_ratio > 2.0:
            return "Severe overage: Split into multiple responses or summarize"
        elif adherence_ratio > 1.5:
            return "Moderate overage: Trim unnecessary elaboration"
        else:
            return "Minor overage: Tighten language slightly"
    
    def get_safe_token_guidelines(self,
                                 user_tokens: int,
                                 mode: MultiplierMode) -> Dict[str, any]:
        """
        Get specific guidelines for safe token usage
        
        Args:
            user_tokens: User message token count
            mode: Current multiplier mode
            
        Returns:
            Guidelines for response generation
        """
        guidelines = {
            MultiplierMode.CRISIS: {
                'style': 'Ultra-brief. Single thoughts. Essential only.',
                'structure': 'No elaboration. No examples. Core message.',
                'example_lengths': [10, 15, 20],
                'avoid': ['explanations', 'metaphors', 'questions', 'options']
            },
            MultiplierMode.GROUNDING: {
                'style': 'Brief and anchoring. Clear, simple language.',
                'structure': 'One main point. Optional brief clarification.',
                'example_lengths': [20, 30, 40],
                'avoid': ['complexity', 'multiple ideas', 'elaboration']
            },
            MultiplierMode.BALANCED: {
                'style': 'Match user energy. Mirror their depth.',
                'structure': 'Main response. Brief expansion if needed.',
                'example_lengths': [user_tokens * 0.8, user_tokens, user_tokens * 1.2],
                'avoid': ['over-explaining', 'tangents']
            },
            MultiplierMode.STANDARD: {
                'style': 'Natural response. Room for context.',
                'structure': 'Complete thoughts. Examples allowed.',
                'example_lengths': [user_tokens * 1.2, user_tokens * 1.5, user_tokens * 1.8],
                'avoid': ['excessive detail', 'redundancy']
            },
            MultiplierMode.EXPLORATORY: {
                'style': 'Rich exploration. Multiple perspectives.',
                'structure': 'Full development. Examples and connections.',
                'example_lengths': [user_tokens * 1.5, user_tokens * 2, user_tokens * 2.5],
                'avoid': ['repetition', 'unnecessary tangents']
            },
            MultiplierMode.EXPANSIVE: {
                'style': 'Deep dive. Comprehensive exploration.',
                'structure': 'Multiple sections. Full examples. Rich detail.',
                'example_lengths': [user_tokens * 2, user_tokens * 2.5, user_tokens * 3],
                'avoid': ['redundancy', 'filler content']
            }
        }
        
        return guidelines.get(mode, guidelines[MultiplierMode.STANDARD])
    
    def analyze_response_patterns(self,
                                 conversation_history: List[Tuple[int, int]]) -> Dict[str, any]:
        """
        Analyze response length patterns
        
        Args:
            conversation_history: List of (user_tokens, assistant_tokens) tuples
            
        Returns:
            Pattern analysis
        """
        if not conversation_history:
            return {'pattern': 'no_history'}
        
        # Calculate ratios
        ratios = [assistant / max(user, 1) 
                 for user, assistant in conversation_history]
        
        # Analyze patterns
        analysis = {
            'average_ratio': np.mean(ratios),
            'ratio_variance': np.var(ratios),
            'max_ratio': max(ratios),
            'min_ratio': min(ratios),
            'escalating': all(ratios[i] <= ratios[i+1] 
                            for i in range(len(ratios)-1)),
            'deescalating': all(ratios[i] >= ratios[i+1] 
                              for i in range(len(ratios)-1)),
            'stable': np.var(ratios) < 0.5,
            'violations': sum(1 for r in ratios if r > 3.0),
            'crisis_responses': sum(1 for r in ratios if r < 0.5)
        }
        
        # Determine overall pattern
        if analysis['escalating']:
            analysis['pattern'] = 'escalating_responses'
        elif analysis['deescalating']:
            analysis['pattern'] = 'deescalating_responses'
        elif analysis['stable']:
            analysis['pattern'] = 'stable_exchange'
        elif analysis['violations'] > len(ratios) * 0.3:
            analysis['pattern'] = 'chronic_overage'
        else:
            analysis['pattern'] = 'variable_responses'
        
        return analysis
    
    def get_multiplier_examples(self) -> Dict[str, Dict[str, any]]:
        """Get examples of multiplier applications"""
        return {
            'crisis_0.3x': {
                'user': "I can't handle this anymore" * 5,  # ~25 tokens
                'assistant_limit': 8,  # 25 * 0.3
                'example': "I hear you. You're safe here.",
                'scenario': "User in crisis, needs grounding"
            },
            'grounding_0.5x': {
                'user': "Everything feels overwhelming and confusing right now",  # ~10 tokens
                'assistant_limit': 5,
                'example': "Let's pause together. Breathe.",
                'scenario': "Low coherence, needs anchoring"
            },
            'balanced_1.0x': {
                'user': "What are the main benefits of meditation?",  # ~10 tokens
                'assistant_limit': 10,
                'example': "Meditation reduces stress, improves focus, and enhances well-being.",
                'scenario': "Standard question, balanced response"
            },
            'standard_1.5x': {
                'user': "How does photosynthesis work?",  # ~6 tokens  
                'assistant_limit': 9,
                'example': "Plants convert sunlight, water, and CO2 into glucose and oxygen.",
                'scenario': "Educational query, room for explanation"
            },
            'exploratory_2.0x': {
                'user': "Tell me about the philosophy of consciousness",  # ~8 tokens
                'assistant_limit': 16,
                'example': "Consciousness philosophy explores fundamental questions about awareness, experience, and the mind-body relationship...",
                'scenario': "Deep topic, exploration welcomed"
            },
            'expansive_3.0x': {
                'user': "Explain quantum computing",  # ~4 tokens
                'assistant_limit': 12,
                'example': "Quantum computing leverages quantum mechanics principles like superposition and entanglement to process information...",
                'scenario': "Complex topic, high coherence, full exploration"
            }
        }
"""