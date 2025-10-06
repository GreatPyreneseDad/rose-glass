"""
Coherence Temporal Dynamics with d/tokens Enhancement
====================================================

Implements the critical insight from SPT conversations:
Time = Token Flow Rate

This module extends the Rose Glass framework with information-theoretic
derivatives that measure coherence velocity in terms of information 
exchange density rather than clock time.

Author: Christopher MacGregor bin Joseph
Date: October 2025
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import time


@dataclass
class TemporalReading:
    """Single temporal measurement point"""
    coherence: float
    token_count: int
    timestamp: float
    message_length: int
    speaker: str  # 'user' or 'assistant'
    
    
class CoherenceTemporalDynamics:
    """Enhanced coherence tracking with token-based derivatives"""
    
    def __init__(self, window_size: int = 20):
        """
        Initialize temporal dynamics tracker
        
        Args:
            window_size: Number of recent readings to keep for derivative calculation
        """
        self.coherence_history: List[float] = []
        self.token_history: List[int] = []
        self.timestamp_history: List[float] = []
        self.reading_history: List[TemporalReading] = []
        self.window_size = window_size
        
    def add_reading(self, 
                   coherence: float,
                   message: str,
                   speaker: str,
                   timestamp: Optional[float] = None):
        """
        Add a new coherence reading with token information
        
        Args:
            coherence: Current coherence value
            message: The message text
            speaker: 'user' or 'assistant'
            timestamp: Unix timestamp (auto-generated if not provided)
        """
        if timestamp is None:
            timestamp = time.time()
            
        # Calculate tokens (simple approximation - could use tiktoken for accuracy)
        token_count = len(message.split())
        
        # Create reading
        reading = TemporalReading(
            coherence=coherence,
            token_count=token_count,
            timestamp=timestamp,
            message_length=len(message),
            speaker=speaker
        )
        
        # Add to histories
        self.reading_history.append(reading)
        self.coherence_history.append(coherence)
        
        # Calculate cumulative token count
        if self.token_history:
            cumulative_tokens = self.token_history[-1] + token_count
        else:
            cumulative_tokens = token_count
        self.token_history.append(cumulative_tokens)
        self.timestamp_history.append(timestamp)
        
        # Maintain window size
        if len(self.reading_history) > self.window_size:
            self.reading_history.pop(0)
            self.coherence_history.pop(0)
            self.token_history.pop(0)
            self.timestamp_history.pop(0)
    
    def calculate_token_derivative(self) -> float:
        """
        Calculate dC/d(tokens) - coherence change per information unit
        
        Returns:
            Token-based coherence derivative
        """
        if len(self.coherence_history) < 2:
            return 0.0
            
        # Calculate finite difference over token delta
        dC = self.coherence_history[-1] - self.coherence_history[-2]
        d_tokens = self.token_history[-1] - self.token_history[-2]
        
        if d_tokens == 0:
            return 0.0
            
        return dC / d_tokens
    
    def calculate_time_derivative(self) -> float:
        """
        Calculate traditional dC/dt for comparison
        
        Returns:
            Time-based coherence derivative
        """
        if len(self.timestamp_history) < 2:
            return 0.0
            
        dt = self.timestamp_history[-1] - self.timestamp_history[-2]
        dC = self.coherence_history[-1] - self.coherence_history[-2]
        
        if dt == 0:
            return 0.0
            
        return dC / dt
    
    def calculate_flow_rate(self) -> float:
        """
        Calculate tokens per second - the 'tempo' of dialogue
        
        Returns:
            Current token flow rate (tokens/second)
        """
        if len(self.timestamp_history) < 2:
            return 0.0
            
        # Look at recent window
        window_start = max(0, len(self.timestamp_history) - 5)
        
        dt = self.timestamp_history[-1] - self.timestamp_history[window_start]
        d_tokens = self.token_history[-1] - self.token_history[window_start]
        
        if dt == 0:
            return float('inf')  # Instantaneous response
            
        return d_tokens / dt
    
    def calculate_dual_derivatives(self) -> Dict[str, float]:
        """
        Calculate both temporal and information-theoretic derivatives
        
        Returns:
            {
                'dC_dt': Traditional time derivative,
                'dC_dtokens': Information-theoretic derivative,
                'flow_rate': Current token flow (tokens/second),
                'interpretation': Semantic meaning
            }
        """
        # Traditional time derivative
        dC_dt = self.calculate_time_derivative()
        
        # Information-theoretic derivative
        dC_dtokens = self.calculate_token_derivative()
        
        # Flow rate
        flow_rate = self.calculate_flow_rate()
        
        # Interpret the dynamics
        interpretation = self.interpret_derivatives(dC_dt, dC_dtokens, flow_rate)
        
        return {
            'dC_dt': dC_dt,
            'dC_dtokens': dC_dtokens,
            'flow_rate': flow_rate,
            'interpretation': interpretation
        }
    
    def interpret_derivatives(self, 
                             dC_dt: float, 
                             dC_dtokens: float,
                             flow_rate: float) -> str:
        """
        Semantic interpretation of derivative patterns
        
        High flow rate = rapid token exchange
        Low flow rate = contemplative pace
        """
        # High flow + positive dC/dtokens = productive rapid exchange
        if flow_rate > 100 and dC_dtokens > 0:
            return "high_energy_convergence"
        
        # High flow + negative dC/dtokens = crisis spiral
        elif flow_rate > 100 and dC_dtokens < 0:
            return "crisis_spiral"
        
        # Low flow + positive dC/dtokens = contemplative integration
        elif flow_rate < 30 and dC_dtokens > 0:
            return "contemplative_growth"
        
        # Low flow + negative dC/dtokens = disengagement
        elif flow_rate < 30 and dC_dtokens < 0:
            return "disengagement_pattern"
        
        # Moderate flow = standard dialogue
        else:
            return "standard_dialogue"
    
    def get_conversation_rhythm(self) -> Dict[str, any]:
        """
        Analyze the rhythm and pacing of the conversation
        
        Returns:
            Dictionary with rhythm metrics
        """
        if len(self.reading_history) < 3:
            return {
                'average_flow_rate': 0.0,
                'flow_variance': 0.0,
                'dominant_speaker': None,
                'turn_taking_ratio': 0.0,
                'average_message_tokens': 0.0
            }
        
        # Calculate flow rates between messages
        flow_rates = []
        for i in range(1, len(self.reading_history)):
            dt = self.reading_history[i].timestamp - self.reading_history[i-1].timestamp
            tokens = self.reading_history[i].token_count
            if dt > 0:
                flow_rates.append(tokens / dt)
        
        # Speaker analysis
        user_messages = sum(1 for r in self.reading_history if r.speaker == 'user')
        total_messages = len(self.reading_history)
        
        # Token distribution
        avg_tokens = np.mean([r.token_count for r in self.reading_history])
        
        return {
            'average_flow_rate': np.mean(flow_rates) if flow_rates else 0.0,
            'flow_variance': np.var(flow_rates) if flow_rates else 0.0,
            'dominant_speaker': 'user' if user_messages > total_messages / 2 else 'assistant',
            'turn_taking_ratio': user_messages / total_messages if total_messages > 0 else 0.0,
            'average_message_tokens': avg_tokens
        }
    
    def detect_crisis_patterns(self) -> Dict[str, bool]:
        """
        Detect crisis patterns in conversation dynamics
        
        Returns:
            Dictionary of crisis indicators
        """
        derivatives = self.calculate_dual_derivatives()
        rhythm = self.get_conversation_rhythm()
        
        # Crisis indicators
        indicators = {
            'rapid_degradation': derivatives['dC_dtokens'] < -0.001 and derivatives['flow_rate'] > 80,
            'coherence_collapse': len(self.coherence_history) > 2 and self.coherence_history[-1] < 1.0,
            'frantic_pace': derivatives['flow_rate'] > 120,
            'message_fragmentation': rhythm['average_message_tokens'] < 10,
            'interpretation': derivatives['interpretation']
        }
        
        # Overall crisis assessment
        crisis_count = sum(1 for k, v in indicators.items() 
                          if k != 'interpretation' and v is True)
        indicators['crisis_detected'] = crisis_count >= 2
        
        return indicators
    
    def recommend_pacing_adjustment(self, current_coherence: float) -> Dict[str, any]:
        """
        Recommend pacing adjustments based on current dynamics
        
        Args:
            current_coherence: Current coherence value
            
        Returns:
            Pacing recommendations
        """
        derivatives = self.calculate_dual_derivatives()
        crisis = self.detect_crisis_patterns()
        
        recommendations = {
            'adjust_pace': False,
            'target_flow_rate': derivatives['flow_rate'],
            'message_length': 'maintain',
            'pause_recommendation': False,
            'reasoning': []
        }
        
        # Crisis response
        if crisis['crisis_detected']:
            recommendations['adjust_pace'] = True
            recommendations['target_flow_rate'] = max(30, derivatives['flow_rate'] * 0.5)
            recommendations['message_length'] = 'shorten'
            recommendations['pause_recommendation'] = True
            recommendations['reasoning'].append("Crisis pattern detected - slow down pace")
        
        # Low coherence response
        elif current_coherence < 1.0:
            recommendations['adjust_pace'] = True
            recommendations['target_flow_rate'] = min(60, derivatives['flow_rate'])
            recommendations['message_length'] = 'shorten'
            recommendations['reasoning'].append("Low coherence - simplify and slow down")
        
        # High coherence opportunity
        elif current_coherence > 2.5 and derivatives['dC_dtokens'] > 0:
            recommendations['adjust_pace'] = True
            recommendations['target_flow_rate'] = min(100, derivatives['flow_rate'] * 1.2)
            recommendations['message_length'] = 'expand'
            recommendations['reasoning'].append("High coherence - can explore deeper")
        
        return recommendations
    
    def get_summary_stats(self) -> Dict[str, float]:
        """
        Get summary statistics for the conversation dynamics
        
        Returns:
            Dictionary of summary metrics
        """
        if not self.coherence_history:
            return {}
            
        derivatives = self.calculate_dual_derivatives()
        rhythm = self.get_conversation_rhythm()
        
        return {
            'current_coherence': self.coherence_history[-1],
            'average_coherence': np.mean(self.coherence_history),
            'coherence_variance': np.var(self.coherence_history),
            'total_tokens': self.token_history[-1] if self.token_history else 0,
            'dC_dt': derivatives['dC_dt'],
            'dC_dtokens': derivatives['dC_dtokens'],
            'current_flow_rate': derivatives['flow_rate'],
            'average_flow_rate': rhythm['average_flow_rate'],
            'interpretation': derivatives['interpretation']
        }
"""