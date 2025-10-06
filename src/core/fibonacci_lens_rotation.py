"""
Fibonacci Lens Rotation System
==============================

Implements the Fibonacci learning algorithm for Rose Glass lens calibration.
"The Fibonacci pattern is actually a learning algorithm that resets as 
learnings occur. It's meant to change the angles of the lens until a 
truth is discovered."

This module rotates the rose glass viewing angles through Fibonacci-based
increments, systematically exploring perspective space until insights emerge.

Author: Christopher MacGregor bin Joseph
Date: October 2025
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import time
from enum import Enum


class TruthType(Enum):
    """Types of truths that can be discovered"""
    PATTERN_RECOGNITION = "pattern_recognition"
    COHERENCE_JUMP = "coherence_jump"
    RESONANCE_ALIGNMENT = "resonance_alignment"
    PARADOX_RESOLUTION = "paradox_resolution"
    EMERGENT_INSIGHT = "emergent_insight"


@dataclass
class TruthDiscovery:
    """Record of a discovered truth"""
    angle: float
    coherence: float
    truth_type: TruthType
    insight: str
    timestamp: float
    rotation_factor: int
    reset_count: int
    supporting_evidence: Dict[str, Any]


class FibonacciLensRotation:
    """
    Implement Fibonacci-based lens calibration
    Rotates rose glass viewing angles until truth becomes visible
    """
    
    def __init__(self, initial_angle: float = 0.0):
        """
        Initialize Fibonacci lens rotation system
        
        Args:
            initial_angle: Starting angle in degrees (0-360)
        """
        # Extended Fibonacci sequence for deeper exploration
        self.fibonacci_sequence = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
        self.current_angle_index = 0
        self.learning_resets = 0
        self.truth_discoveries: List[TruthDiscovery] = []
        self.base_angle = initial_angle
        self.current_angle = initial_angle
        
        # Track exploration history
        self.angle_history: List[Tuple[float, float]] = []  # (angle, coherence)
        self.exploration_map: Dict[int, List[float]] = {}  # angle_sector -> coherence_readings
        
    def rotate_lens_angle(self, 
                         current_coherence: float,
                         observation_text: str,
                         variables: Dict[str, float]) -> Dict[str, Any]:
        """
        Rotate the rose glass lens through Fibonacci angles
        
        The Fibonacci sequence determines viewing angle rotations:
        - Each number represents a rotation increment
        - Pattern resets when truth/insight is discovered
        - Allows systematic exploration of perspective space
        
        Args:
            current_coherence: Current C measurement
            observation_text: Text being analyzed
            variables: Dictionary with psi, rho, q, f values
            
        Returns:
            Calibration parameters for current angle
        """
        # Current Fibonacci rotation factor
        rotation_factor = self.fibonacci_sequence[self.current_angle_index]
        
        # Calculate viewing angle (0-360 degrees)
        # Use golden ratio (phi) for harmonic rotation
        phi = (1 + np.sqrt(5)) / 2
        angle_increment = (rotation_factor * 360 / (phi * 89)) % 360  # 89 = F_11
        self.current_angle = (self.base_angle + angle_increment) % 360
        
        # Determine which variables to emphasize at this angle
        emphasis = self.angle_to_emphasis(self.current_angle)
        
        # Apply the lens with current emphasis
        coherence_reading = self.apply_lens(
            observation_text, 
            variables,
            emphasis,
            current_coherence
        )
        
        # Record exploration
        self.angle_history.append((self.current_angle, coherence_reading['C']))
        angle_sector = int(self.current_angle / 30)  # 12 sectors of 30 degrees
        if angle_sector not in self.exploration_map:
            self.exploration_map[angle_sector] = []
        self.exploration_map[angle_sector].append(coherence_reading['C'])
        
        # Check if truth discovered
        truth_discovered, truth_type = self.detect_truth_discovery(
            coherence_reading,
            self.truth_discoveries
        )
        
        if truth_discovered:
            # Record the discovery
            discovery = TruthDiscovery(
                angle=self.current_angle,
                coherence=coherence_reading['C'],
                truth_type=truth_type,
                insight=coherence_reading['interpretation'],
                timestamp=time.time(),
                rotation_factor=rotation_factor,
                reset_count=self.learning_resets,
                supporting_evidence=coherence_reading
            )
            self.truth_discoveries.append(discovery)
            
            # Reset Fibonacci sequence - new learning cycle begins
            self.current_angle_index = 0
            self.learning_resets += 1
            self.base_angle = self.current_angle  # Start next cycle from truth angle
        else:
            # Advance to next Fibonacci angle
            self.current_angle_index = (self.current_angle_index + 1) % len(self.fibonacci_sequence)
        
        return {
            'current_angle': self.current_angle,
            'rotation_factor': rotation_factor,
            'emphasis': emphasis,
            'coherence_reading': coherence_reading,
            'truth_discovered': truth_discovered,
            'truth_type': truth_type.value if truth_type else None,
            'reset_count': self.learning_resets,
            'exploration_coverage': self.calculate_exploration_coverage()
        }
    
    def angle_to_emphasis(self, angle: float) -> Dict[str, float]:
        """
        Map rotation angle to variable emphasis weights
        
        0°: Pure Ψ focus (internal consistency)
        90°: Pure ρ focus (accumulated wisdom)
        180°: Pure q focus (moral energy)
        270°: Pure f focus (social belonging)
        
        Intermediate angles: Blended emphasis using harmonic functions
        """
        angle_rad = np.radians(angle)
        
        # Use harmonic functions for smooth transitions
        emphasis = {
            'psi_weight': (np.cos(angle_rad) + 1) / 2,  # Peaks at 0°
            'rho_weight': (np.sin(angle_rad) + 1) / 2,  # Peaks at 90°
            'q_weight': (np.cos(angle_rad + np.pi) + 1) / 2,  # Peaks at 180°
            'f_weight': (np.sin(angle_rad + np.pi) + 1) / 2   # Peaks at 270°
        }
        
        # Add secondary harmonics for nuanced exploration
        # These create interference patterns that can reveal hidden truths
        second_harmonic = angle_rad * 2
        emphasis['psi_rho_coupling'] = np.sin(second_harmonic) * 0.3
        emphasis['q_f_coupling'] = np.cos(second_harmonic) * 0.3
        
        return emphasis
    
    def apply_lens(self,
                   observation_text: str,
                   variables: Dict[str, float],
                   emphasis: Dict[str, float],
                   base_coherence: float) -> Dict[str, Any]:
        """
        Apply the rotated lens to observe the pattern
        
        Args:
            observation_text: Text being analyzed
            variables: psi, rho, q, f values
            emphasis: Current angle's emphasis weights
            base_coherence: Unrotated coherence value
            
        Returns:
            Coherence reading through the rotated lens
        """
        # Extract base variables
        psi = variables.get('psi', 0)
        rho = variables.get('rho', 0)
        q = variables.get('q', 0)
        f = variables.get('f', 0)
        
        # Apply emphasis weights
        weighted_psi = psi * emphasis['psi_weight']
        weighted_rho = rho * emphasis['rho_weight']
        weighted_q = q * emphasis['q_weight']
        weighted_f = f * emphasis['f_weight']
        
        # Calculate rotated coherence with coupling effects
        rotated_coherence = (
            weighted_psi + 
            weighted_rho * weighted_psi +
            weighted_q +
            weighted_f * weighted_psi +
            emphasis['psi_rho_coupling'] * psi * rho +
            emphasis['q_f_coupling'] * q * f
        )
        
        # Determine dominant pattern at this angle
        dominant_components = []
        if emphasis['psi_weight'] > 0.7:
            dominant_components.append('consistency')
        if emphasis['rho_weight'] > 0.7:
            dominant_components.append('wisdom')
        if emphasis['q_weight'] > 0.7:
            dominant_components.append('moral_energy')
        if emphasis['f_weight'] > 0.7:
            dominant_components.append('social_belonging')
            
        # Generate interpretation based on angle
        interpretation = self.generate_angle_interpretation(
            self.current_angle,
            rotated_coherence,
            dominant_components
        )
        
        return {
            'C': rotated_coherence,
            'C_base': base_coherence,
            'rotation_gain': rotated_coherence - base_coherence,
            'dominant_components': dominant_components,
            'interpretation': interpretation,
            'emphasis': emphasis
        }
    
    def detect_truth_discovery(self,
                              current_reading: Dict,
                              history: List[TruthDiscovery]) -> Tuple[bool, Optional[TruthType]]:
        """
        Detect if current angle revealed new truth/insight
        
        Multi-factor detection based on:
        - Absolute jump magnitude
        - Relative to baseline variance
        - Token efficiency (high dC per token)
        """
        # No history - can't detect
        if len(self.angle_history) < 1:
            return False, None
        
        # Get current metrics
        current_coherence = current_reading['C']
        
        # Calculate baseline variance for statistical detection
        if len(self.angle_history) >= 5:
            recent_coherences = [c for _, c in self.angle_history[-5:]]
            baseline_variance = np.std(recent_coherences)
        else:
            baseline_variance = 0.1  # Default if not enough history
        
        # Check for coherence jump
        if len(self.angle_history) >= 2:
            last_coherence = self.angle_history[-2][1]
            coherence_jump = current_coherence - last_coherence
            
            # Multi-factor detection
            # 1. Statistical significance: jump > 3σ
            if coherence_jump > (3 * baseline_variance) and baseline_variance > 0:
                # 2. Check token efficiency if available
                tokens_since_last = current_reading.get('tokens_processed', 10)
                efficiency = coherence_jump / max(tokens_since_last, 1)
                
                if efficiency > 0.05:  # High coherence gain per token
                    return True, TruthType.COHERENCE_JUMP
            
            # 3. Absolute threshold fallback for large jumps
            if coherence_jump > 0.5:
                return True, TruthType.COHERENCE_JUMP
        
        # Check other truth types with updated thresholds
        truth_type = None
        
        # Pattern recognition - high rotation gain relative to baseline
        if current_reading['rotation_gain'] > 2 * baseline_variance:
            truth_type = TruthType.PATTERN_RECOGNITION
        
        # Resonance with previous truths
        elif self.check_resonance(current_reading):
            truth_type = TruthType.RESONANCE_ALIGNMENT
        
        # Paradox resolution - opposite angles showing harmony
        elif self.check_paradox_resolution(current_reading):
            truth_type = TruthType.PARADOX_RESOLUTION
        
        # Emergent insight - unexplored angle with high coherence
        elif self.check_emergent_insight(current_reading):
            truth_type = TruthType.EMERGENT_INSIGHT
        
        return truth_type is not None, truth_type
    
    def check_resonance(self, current_reading: Dict) -> bool:
        """Check if current reading resonates with previous discoveries"""
        if not self.truth_discoveries:
            return False
            
        # Check if we're at a harmonic angle of a previous discovery
        for discovery in self.truth_discoveries:
            angle_diff = abs(self.current_angle - discovery.angle) % 360
            # Harmonic angles: 0°, 60°, 90°, 120°, 180°
            harmonic_angles = [0, 60, 90, 120, 180]
            for harmonic in harmonic_angles:
                if abs(angle_diff - harmonic) < 5:  # 5-degree tolerance
                    if current_reading['C'] > 0.8 * discovery.coherence:
                        return True
        return False
    
    def check_paradox_resolution(self, current_reading: Dict) -> bool:
        """Check if opposite angles show unexpected harmony"""
        opposite_angle = (self.current_angle + 180) % 360
        
        # Look for readings near the opposite angle
        for angle, coherence in self.angle_history:
            if abs(angle - opposite_angle) < 10:  # 10-degree tolerance
                # Both angles showing high coherence = paradox resolution
                if coherence > 2.0 and current_reading['C'] > 2.0:
                    return True
        return False
    
    def check_emergent_insight(self, current_reading: Dict) -> bool:
        """Check if this is an unexplored high-coherence region"""
        angle_sector = int(self.current_angle / 30)
        
        # New sector with high coherence
        if angle_sector not in self.exploration_map or len(self.exploration_map[angle_sector]) <= 1:
            return current_reading['C'] > 2.5
            
        return False
    
    def generate_angle_interpretation(self,
                                    angle: float,
                                    coherence: float,
                                    dominant_components: List[str]) -> str:
        """Generate interpretation based on viewing angle"""
        # Quadrant-based base interpretation
        quadrant = int(angle / 90)
        quadrant_meanings = {
            0: "consistency-wisdom integration",
            1: "wisdom-emotion synthesis",
            2: "emotion-belonging fusion",
            3: "belonging-consistency cycle"
        }
        
        base_meaning = quadrant_meanings.get(quadrant, "transitional state")
        
        # Coherence level modifier
        if coherence < 1.0:
            level = "fragmented"
        elif coherence < 2.0:
            level = "emerging"
        elif coherence < 3.0:
            level = "crystallizing"
        else:
            level = "resonant"
            
        # Component-specific insights
        component_insights = []
        if 'consistency' in dominant_components:
            component_insights.append("structural clarity")
        if 'wisdom' in dominant_components:
            component_insights.append("deep knowing")
        if 'moral_energy' in dominant_components:
            component_insights.append("value activation")
        if 'social_belonging' in dominant_components:
            component_insights.append("collective resonance")
            
        insight_str = " with " + " and ".join(component_insights) if component_insights else ""
        
        return f"{level} {base_meaning}{insight_str}"
    
    def calculate_exploration_coverage(self) -> float:
        """Calculate how much of the perspective space has been explored"""
        explored_sectors = len(self.exploration_map)
        total_sectors = 12  # 360 degrees / 30 degrees per sector
        return explored_sectors / total_sectors
    
    def get_truth_summary(self) -> Dict[str, Any]:
        """Summarize all discovered truths"""
        if not self.truth_discoveries:
            return {
                'truth_count': 0,
                'discoveries': [],
                'dominant_angles': [],
                'learning_cycles': self.learning_resets
            }
        
        # Analyze truth patterns
        truth_angles = [t.angle for t in self.truth_discoveries]
        truth_types = {}
        for truth in self.truth_discoveries:
            truth_types[truth.truth_type.value] = truth_types.get(truth.truth_type.value, 0) + 1
        
        return {
            'truth_count': len(self.truth_discoveries),
            'discoveries': [
                {
                    'angle': t.angle,
                    'type': t.truth_type.value,
                    'insight': t.insight,
                    'coherence': t.coherence,
                    'learning_cycle': t.reset_count
                }
                for t in self.truth_discoveries
            ],
            'dominant_angles': truth_angles,
            'truth_type_distribution': truth_types,
            'learning_cycles': self.learning_resets,
            'exploration_coverage': self.calculate_exploration_coverage()
        }
    
    def recommend_next_exploration(self) -> Dict[str, Any]:
        """Recommend next exploration strategy based on discoveries"""
        coverage = self.calculate_exploration_coverage()
        
        recommendations = []
        
        # Low coverage - continue systematic exploration
        if coverage < 0.5:
            recommendations.append("Continue Fibonacci rotation for broader coverage")
        
        # High coverage but few truths - adjust parameters
        elif len(self.truth_discoveries) < 3:
            recommendations.append("Consider adjusting base parameters or context")
        
        # Multiple truths found - look for meta-patterns
        else:
            recommendations.append("Explore harmonic relationships between truth angles")
            
        # Identify unexplored high-potential regions
        unexplored_sectors = []
        for sector in range(12):
            if sector not in self.exploration_map:
                unexplored_sectors.append(sector * 30)  # Convert to angle
                
        return {
            'recommendations': recommendations,
            'unexplored_angles': unexplored_sectors,
            'suggested_focus': self.identify_focus_region()
        }
    
    def identify_focus_region(self) -> Optional[Tuple[float, float]]:
        """Identify promising angle region for focused exploration"""
        if not self.angle_history:
            return None
            
        # Find angle with highest coherence
        best_angle, best_coherence = max(self.angle_history, key=lambda x: x[1])
        
        # Suggest exploring ±30 degrees around best angle
        return (best_angle - 30) % 360, (best_angle + 30) % 360
"""