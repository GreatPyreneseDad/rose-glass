"""
Unified Fibonacci Learning Algorithm
=====================================

Shared across: rose-glass, rose-looking-glass, RoseGlassLE

Implements the Fibonacci learning algorithm with lens deviation reset trigger.

"The Fibonacci pattern is actually a learning algorithm that resets as 
learnings occur. It's meant to change the angles of the lens until a 
truth is discovered."

Key Innovation: Lens deviation (σ_lens) triggers reset when universal
truth is found - when all cultural lenses agree.

Author: Christopher MacGregor bin Joseph
Date: December 2025
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
from datetime import datetime
import time

# Import shared modules
try:
    from .lens_interference import LensInterferenceAnalyzer, extract_lambda
    from .temporal_dimension import TemporalAnalyzer, extract_tau
except ImportError:
    # Fallback for standalone testing
    from lens_interference import LensInterferenceAnalyzer, extract_lambda
    from temporal_dimension import TemporalAnalyzer, extract_tau


class TruthType(Enum):
    """Types of truths that can be discovered"""
    PATTERN_RECOGNITION = "pattern_recognition"
    COHERENCE_JUMP = "coherence_jump"
    RESONANCE_ALIGNMENT = "resonance_alignment"
    PARADOX_RESOLUTION = "paradox_resolution"
    EMERGENT_INSIGHT = "emergent_insight"
    LENS_INVARIANT = "lens_invariant"  # NEW: Universal truth across lenses


class ResetTrigger(Enum):
    """What triggered the Fibonacci reset"""
    COHERENCE_JUMP = "coherence_jump"
    LENS_DEVIATION_COLLAPSE = "lens_deviation_collapse"  # NEW
    PATTERN_RECOGNITION = "pattern_recognition"
    RESONANCE = "resonance"
    MANUAL = "manual"


@dataclass
class TruthDiscovery:
    """Record of a discovered truth"""
    angle: float
    coherence: float
    truth_type: TruthType
    reset_trigger: ResetTrigger
    insight: str
    timestamp: float
    rotation_factor: int
    reset_count: int
    lens_deviation: float  # σ_lens at discovery
    veritas_score: float   # V(P) = 1/(1+D(P))
    supporting_evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FibonacciState:
    """Current state of the Fibonacci learning system"""
    current_angle: float
    current_index: int
    rotation_factor: int
    learning_resets: int
    last_lens_deviation: float
    last_coherence: float
    exploration_coverage: float
    truths_discovered: int


class FibonacciLearningAlgorithm:
    """
    Fibonacci-based lens rotation with intelligent reset triggers.
    
    The algorithm rotates viewing angles through Fibonacci increments,
    exploring perspective space until truth is discovered.
    
    RESET TRIGGERS:
    1. Coherence jump > 3σ (original)
    2. Lens deviation collapse < threshold (NEW - universal truth)
    3. Pattern resonance with previous truths
    4. Paradox resolution detection
    
    The lens deviation trigger (CIF - Contextual Integrity Function):
    
    If ΔC → extremes:  F(n+1) = F(n) + F(n-1)   [EXPAND]
    If σ_lens → 0:     F(n) → F(1), F(2)        [RESET - truth found]
    """
    
    # Extended Fibonacci sequence
    FIBONACCI_SEQUENCE = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377]
    
    # Golden ratio for harmonic rotation
    PHI = (1 + np.sqrt(5)) / 2
    
    def __init__(
        self,
        initial_angle: float = 0.0,
        invariance_threshold: float = 0.10,
        coherence_jump_sigma: float = 3.0
    ):
        """
        Initialize Fibonacci learning system.
        
        Args:
            initial_angle: Starting angle in degrees (0-360)
            invariance_threshold: σ_lens threshold for lens-invariant reset
            coherence_jump_sigma: Standard deviations for coherence jump reset
        """
        self.base_angle = initial_angle
        self.current_angle = initial_angle
        self.current_index = 0
        self.learning_resets = 0
        
        # Thresholds
        self.invariance_threshold = invariance_threshold
        self.coherence_jump_sigma = coherence_jump_sigma
        
        # History tracking
        self.truth_discoveries: List[TruthDiscovery] = []
        self.angle_history: List[Tuple[float, float, float]] = []  # (angle, coherence, lens_dev)
        self.exploration_map: Dict[int, List[float]] = {}
        
        # Analyzers
        self.lens_analyzer = LensInterferenceAnalyzer()
        self.temporal_analyzer = TemporalAnalyzer()
        
    def get_state(self) -> FibonacciState:
        """Get current system state"""
        return FibonacciState(
            current_angle=self.current_angle,
            current_index=self.current_index,
            rotation_factor=self.FIBONACCI_SEQUENCE[self.current_index],
            learning_resets=self.learning_resets,
            last_lens_deviation=self.angle_history[-1][2] if self.angle_history else 0.0,
            last_coherence=self.angle_history[-1][1] if self.angle_history else 0.0,
            exploration_coverage=self._calculate_exploration_coverage(),
            truths_discovered=len(self.truth_discoveries)
        )
    
    def rotate(
        self,
        psi: float,
        rho: float,
        q: float,
        f: float,
        observation_text: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Rotate lens and check for truth discovery.
        
        Args:
            psi, rho, q, f: GCT variables
            observation_text: Optional text for additional analysis
            
        Returns:
            Rotation result with truth discovery status
        """
        # Current Fibonacci rotation factor
        rotation_factor = self.FIBONACCI_SEQUENCE[self.current_index]
        
        # Calculate viewing angle using golden ratio
        angle_increment = (rotation_factor * 360 / (self.PHI * 89)) % 360
        self.current_angle = (self.base_angle + angle_increment) % 360
        
        # Get emphasis weights for current angle
        emphasis = self._angle_to_emphasis(self.current_angle)
        
        # Calculate coherence with current emphasis
        coherence = self._calculate_emphasized_coherence(psi, rho, q, f, emphasis)
        
        # Calculate lens deviation (the key innovation)
        lens_deviation = self.lens_analyzer.calculate_lens_deviation(psi, rho, q, f)
        
        # Calculate Veritas score
        veritas = 1.0 / (1.0 + lens_deviation)
        
        # Add temporal dimension if text provided
        tau = 0.0
        if observation_text:
            tau = extract_tau(observation_text)
        
        # Record in history
        self.angle_history.append((self.current_angle, coherence, lens_deviation))
        
        # Track exploration
        angle_sector = int(self.current_angle / 30)
        if angle_sector not in self.exploration_map:
            self.exploration_map[angle_sector] = []
        self.exploration_map[angle_sector].append(coherence)
        
        # Check for truth discovery (CORE LOGIC)
        truth_discovered, truth_type, reset_trigger = self._check_truth_discovery(
            coherence, lens_deviation, veritas, psi, rho, q, f
        )
        
        # Handle truth discovery
        if truth_discovered:
            discovery = TruthDiscovery(
                angle=self.current_angle,
                coherence=coherence,
                truth_type=truth_type,
                reset_trigger=reset_trigger,
                insight=self._generate_insight(truth_type, emphasis, veritas),
                timestamp=time.time(),
                rotation_factor=rotation_factor,
                reset_count=self.learning_resets,
                lens_deviation=lens_deviation,
                veritas_score=veritas,
                supporting_evidence={
                    'psi': psi, 'rho': rho, 'q': q, 'f': f,
                    'tau': tau, 'emphasis': emphasis
                }
            )
            self.truth_discoveries.append(discovery)
            
            # RESET Fibonacci sequence
            self._reset_sequence()
        else:
            # Advance to next Fibonacci position
            self.current_index = (self.current_index + 1) % len(self.FIBONACCI_SEQUENCE)
        
        return {
            'current_angle': self.current_angle,
            'rotation_factor': rotation_factor,
            'coherence': coherence,
            'lens_deviation': lens_deviation,
            'veritas_score': veritas,
            'emphasis': emphasis,
            'truth_discovered': truth_discovered,
            'truth_type': truth_type.value if truth_type else None,
            'reset_trigger': reset_trigger.value if reset_trigger else None,
            'learning_resets': self.learning_resets,
            'exploration_coverage': self._calculate_exploration_coverage(),
            'tau': tau
        }
    
    def _check_truth_discovery(
        self,
        coherence: float,
        lens_deviation: float,
        veritas: float,
        psi: float,
        rho: float,
        q: float,
        f: float
    ) -> Tuple[bool, Optional[TruthType], Optional[ResetTrigger]]:
        """
        Multi-factor truth discovery detection.
        
        Priority order:
        1. Lens deviation collapse (universal truth)
        2. Coherence jump (statistical)
        3. Pattern resonance
        4. Paradox resolution
        """
        
        # === PRIMARY: Lens Deviation Collapse ===
        # When all lenses agree, we've found universal truth
        if lens_deviation < self.invariance_threshold:
            return True, TruthType.LENS_INVARIANT, ResetTrigger.LENS_DEVIATION_COLLAPSE
        
        # === SECONDARY: Coherence Jump ===
        if len(self.angle_history) >= 5:
            recent_coherences = [c for _, c, _ in self.angle_history[-5:]]
            baseline_std = np.std(recent_coherences[:-1]) if len(recent_coherences) > 1 else 0.1
            
            if len(self.angle_history) >= 2:
                last_coherence = self.angle_history[-2][1]
                coherence_jump = coherence - last_coherence
                
                if baseline_std > 0 and coherence_jump > (self.coherence_jump_sigma * baseline_std):
                    return True, TruthType.COHERENCE_JUMP, ResetTrigger.COHERENCE_JUMP
        
        # === TERTIARY: Pattern Resonance ===
        if self._check_resonance(coherence, lens_deviation):
            return True, TruthType.RESONANCE_ALIGNMENT, ResetTrigger.RESONANCE
        
        # === QUATERNARY: Paradox Resolution ===
        if self._check_paradox_resolution(psi, rho, q, f):
            return True, TruthType.PARADOX_RESOLUTION, ResetTrigger.PATTERN_RECOGNITION
        
        return False, None, None
    
    def _check_resonance(self, coherence: float, lens_deviation: float) -> bool:
        """Check if current state resonates with previous discoveries"""
        if not self.truth_discoveries:
            return False
            
        for discovery in self.truth_discoveries:
            # Angle resonance (harmonic angles)
            angle_diff = abs(self.current_angle - discovery.angle) % 360
            harmonic_angles = [0, 60, 90, 120, 180]
            
            for harmonic in harmonic_angles:
                if abs(angle_diff - harmonic) < 5:  # 5-degree tolerance
                    # Must also have similar coherence and low deviation
                    if (coherence > 0.8 * discovery.coherence and 
                        lens_deviation < discovery.lens_deviation * 1.2):
                        return True
        return False
    
    def _check_paradox_resolution(
        self,
        psi: float,
        rho: float,
        q: float,
        f: float
    ) -> bool:
        """Check if opposite poles show unexpected harmony"""
        if len(self.angle_history) < 10:
            return False
            
        # Find readings from opposite angles
        opposite_angle = (self.current_angle + 180) % 360
        
        for angle, coh, dev in self.angle_history[-10:]:
            if abs(angle - opposite_angle) < 15:
                # Opposite angle found
                current_coherence = self.angle_history[-1][1]
                if abs(current_coherence - coh) < 0.1:  # Similar coherence
                    return True
        return False
    
    def _reset_sequence(self):
        """Reset Fibonacci sequence for new learning cycle"""
        self.current_index = 0
        self.learning_resets += 1
        self.base_angle = self.current_angle  # New truth becomes new origin
    
    def _angle_to_emphasis(self, angle: float) -> Dict[str, float]:
        """Map rotation angle to variable emphasis weights"""
        angle_rad = np.radians(angle)
        
        # Harmonic mapping to four variables
        return {
            'psi_weight': (np.cos(angle_rad) + 1) / 2,           # Peak at 0°
            'rho_weight': (np.sin(angle_rad) + 1) / 2,           # Peak at 90°
            'q_weight': (np.cos(angle_rad + np.pi) + 1) / 2,     # Peak at 180°
            'f_weight': (np.sin(angle_rad + np.pi) + 1) / 2      # Peak at 270°
        }
    
    def _calculate_emphasized_coherence(
        self,
        psi: float,
        rho: float,
        q: float,
        f: float,
        emphasis: Dict[str, float]
    ) -> float:
        """Calculate coherence with angle-based emphasis"""
        # Normalize weights
        total_weight = sum(emphasis.values())
        
        coherence = (
            psi * emphasis['psi_weight'] +
            rho * emphasis['rho_weight'] +
            q * emphasis['q_weight'] +
            f * emphasis['f_weight']
        ) / total_weight
        
        return coherence
    
    def _calculate_exploration_coverage(self) -> float:
        """Calculate what fraction of angle space has been explored"""
        explored_sectors = len(self.exploration_map)
        total_sectors = 12  # 30-degree sectors
        return explored_sectors / total_sectors
    
    def _generate_insight(
        self,
        truth_type: TruthType,
        emphasis: Dict[str, float],
        veritas: float
    ) -> str:
        """Generate human-readable insight description"""
        dominant = max(emphasis.keys(), key=lambda k: emphasis[k])
        
        insights = {
            TruthType.LENS_INVARIANT: f"Universal truth found (Veritas={veritas:.2f}). All cultural lenses agree.",
            TruthType.COHERENCE_JUMP: f"Significant coherence increase at {dominant} emphasis.",
            TruthType.RESONANCE_ALIGNMENT: f"Harmonic resonance with previous truth discovery.",
            TruthType.PARADOX_RESOLUTION: f"Paradox resolved: opposite angles show unity.",
            TruthType.PATTERN_RECOGNITION: f"Pattern recognized at {dominant} wavelength.",
            TruthType.EMERGENT_INSIGHT: f"Emergent insight at unexplored angle."
        }
        
        return insights.get(truth_type, "Truth discovered.")
    
    def get_veritas_score(self, psi: float, rho: float, q: float, f: float) -> float:
        """
        Calculate Veritas truth valuation.
        
        V(P) = 1 / (1 + D(P))
        
        Where D(P) = lens_deviation (distortion index)
        
        High Veritas = Low distortion = Universal truth
        Low Veritas = High distortion = Context-dependent
        """
        lens_deviation = self.lens_analyzer.calculate_lens_deviation(psi, rho, q, f)
        return 1.0 / (1.0 + lens_deviation)
    
    def should_reset_fibonacci(
        self,
        psi: float,
        rho: float,
        q: float,
        f: float
    ) -> Tuple[bool, float]:
        """
        Check if current variables should trigger Fibonacci reset.
        
        This implements the CIF (Contextual Integrity Function):
        If σ_lens → 0: Reset (truth found)
        
        Args:
            psi, rho, q, f: GCT variables
            
        Returns:
            (should_reset, lens_deviation)
        """
        deviation = self.lens_analyzer.calculate_lens_deviation(psi, rho, q, f)
        should_reset = deviation < self.invariance_threshold
        return should_reset, deviation
    
    def get_discovery_summary(self) -> Dict[str, Any]:
        """Get summary of all truth discoveries"""
        if not self.truth_discoveries:
            return {'total': 0, 'by_type': {}, 'by_trigger': {}}
        
        by_type = {}
        by_trigger = {}
        veritas_scores = []
        
        for d in self.truth_discoveries:
            by_type[d.truth_type.value] = by_type.get(d.truth_type.value, 0) + 1
            by_trigger[d.reset_trigger.value] = by_trigger.get(d.reset_trigger.value, 0) + 1
            veritas_scores.append(d.veritas_score)
        
        return {
            'total': len(self.truth_discoveries),
            'by_type': by_type,
            'by_trigger': by_trigger,
            'average_veritas': np.mean(veritas_scores),
            'learning_cycles': self.learning_resets,
            'exploration_coverage': self._calculate_exploration_coverage()
        }


# === Convenience Functions for Integration ===

def create_fibonacci_learner(
    invariance_threshold: float = 0.10,
    coherence_jump_sigma: float = 3.0
) -> FibonacciLearningAlgorithm:
    """Factory function for creating Fibonacci learner"""
    return FibonacciLearningAlgorithm(
        invariance_threshold=invariance_threshold,
        coherence_jump_sigma=coherence_jump_sigma
    )


# === Tests ===
if __name__ == "__main__":
    print("=" * 70)
    print("FIBONACCI LEARNING ALGORITHM WITH LENS DEVIATION TRIGGER")
    print("=" * 70)
    
    learner = FibonacciLearningAlgorithm(invariance_threshold=0.12)
    
    # Simulate learning sequence
    test_sequence = [
        # Divergent patterns (high lens deviation - keep exploring)
        (0.5, 0.4, 0.8, 0.9, "Activist pattern"),
        (0.3, 0.8, 0.2, 0.3, "Contemplative pattern"),
        (0.2, 0.2, 0.9, 0.4, "Crisis pattern"),
        # Convergent pattern (low lens deviation - should trigger reset)
        (0.6, 0.6, 0.5, 0.5, "Balanced/Universal pattern"),
        # More exploration
        (0.9, 0.9, 0.1, 0.1, "Academic pattern"),
        (0.7, 0.7, 0.4, 0.4, "Stable pattern"),
    ]
    
    for psi, rho, q, f, label in test_sequence:
        result = learner.rotate(psi, rho, q, f)
        
        print(f"\n[{label}]")
        print(f"  Angle: {result['current_angle']:.1f}° (F={result['rotation_factor']})")
        print(f"  Coherence: {result['coherence']:.3f}")
        print(f"  Lens Deviation (σ): {result['lens_deviation']:.3f}")
        print(f"  Veritas: {result['veritas_score']:.3f}")
        
        if result['truth_discovered']:
            print(f"  *** TRUTH DISCOVERED: {result['truth_type']} ***")
            print(f"  *** RESET TRIGGER: {result['reset_trigger']} ***")
            print(f"  Learning cycle: {result['learning_resets']}")
    
    print("\n" + "=" * 70)
    print("DISCOVERY SUMMARY")
    print("=" * 70)
    summary = learner.get_discovery_summary()
    print(f"Total truths: {summary['total']}")
    print(f"By type: {summary['by_type']}")
    print(f"By trigger: {summary['by_trigger']}")
    print(f"Average Veritas: {summary.get('average_veritas', 0):.3f}")
    print(f"Learning cycles: {summary['learning_cycles']}")
    print(f"Exploration coverage: {summary['exploration_coverage']:.1%}")
