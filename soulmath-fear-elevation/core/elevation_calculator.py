#!/usr/bin/env python3
"""
Elevation Calculator - Implements the core mathematical relationship H ∝ ∫F(x)dx
Calculates elevation potential from fear descent trajectories
"""
from typing import List, Dict, Tuple, Callable, Optional
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from scipy import integrate, interpolate
from scipy.optimize import minimize_scalar
import math


@dataclass
class DescentPoint:
    """A point in the fear descent trajectory"""
    depth: float  # 0.0 (surface) to 1.0 (abyss)
    timestamp: datetime
    fear_intensity: float
    coherence: float
    notes: Optional[str] = None


@dataclass
class ElevationResult:
    """Result of elevation calculation"""
    height_achieved: float
    peak_moment: datetime
    descent_depth: float
    integration_value: float
    transformation_coefficient: float
    trajectory_quality: float


class ElevationCalculator:
    """
    Implements the elevation mathematics from SoulMath theorem.
    Core formula: H ∝ ∫(descent→truth) F(x)dx
    
    The deeper the descent into fear, the greater the potential elevation.
    """
    
    def __init__(self):
        self.descent_history: List[DescentPoint] = []
        self.elevation_history: List[ElevationResult] = []
        self.truth_depth: float = 0.85  # Where fear transforms to truth
        self.base_coefficient: float = 1.0  # Base transformation rate
        
        # Fear field functions - different types yield different curves
        self.fear_fields = {
            'identity': self._identity_fear_field,
            'existential': self._existential_fear_field,
            'connection': self._connection_fear_field,
            'purpose': self._purpose_fear_field,
            'mortality': self._mortality_fear_field,
            'default': self._default_fear_field
        }
        
    def calculate_elevation(self, descent_trajectory: List[DescentPoint], 
                          fear_type: str = 'default') -> ElevationResult:
        """
        Calculate elevation from a complete descent trajectory.
        
        Args:
            descent_trajectory: List of points in the descent
            fear_type: Type of fear for field selection
            
        Returns:
            ElevationResult with calculated height and metadata
        """
        if not descent_trajectory:
            return self._empty_result()
            
        # Get appropriate fear field function
        fear_field_func = self.fear_fields.get(fear_type, self.fear_fields['default'])
        
        # Sort trajectory by depth
        trajectory = sorted(descent_trajectory, key=lambda p: p.depth)
        
        # Find deepest point
        deepest_point = max(trajectory, key=lambda p: p.depth)
        
        # Calculate integration bounds
        start_depth = trajectory[0].depth
        end_depth = min(deepest_point.depth, self.truth_depth)
        
        # Perform integration
        integration_result = self._integrate_fear_field(
            fear_field_func, 
            start_depth, 
            end_depth, 
            trajectory
        )
        
        # Calculate transformation coefficient based on journey quality
        transformation_coef = self._calculate_transformation_coefficient(trajectory)
        
        # Calculate final elevation
        raw_elevation = integration_result * transformation_coef
        height_achieved = self._apply_elevation_function(raw_elevation, deepest_point.depth)
        
        # Assess trajectory quality
        trajectory_quality = self._assess_trajectory_quality(trajectory)
        
        result = ElevationResult(
            height_achieved=height_achieved,
            peak_moment=deepest_point.timestamp,
            descent_depth=deepest_point.depth,
            integration_value=integration_result,
            transformation_coefficient=transformation_coef,
            trajectory_quality=trajectory_quality
        )
        
        self.elevation_history.append(result)
        return result
        
    def _integrate_fear_field(self, field_func: Callable, start: float, 
                             end: float, trajectory: List[DescentPoint]) -> float:
        """
        Integrate the fear field function over the descent path.
        Uses trajectory points to modulate the field.
        """
        # Create interpolation function from trajectory
        depths = [p.depth for p in trajectory]
        intensities = [p.fear_intensity for p in trajectory]
        
        if len(depths) > 1:
            # Interpolate fear intensity along path
            intensity_func = interpolate.interp1d(
                depths, intensities, 
                kind='cubic', 
                fill_value='extrapolate'
            )
        else:
            # Single point - use constant
            intensity_func = lambda x: intensities[0]
        
        # Define modulated field function
        def modulated_field(x):
            base_field = field_func(x)
            try:
                intensity = intensity_func(x)
            except:
                intensity = 1.0
            return base_field * intensity
        
        # Perform integration
        result, _ = integrate.quad(modulated_field, start, end)
        return result
        
    def _calculate_transformation_coefficient(self, trajectory: List[DescentPoint]) -> float:
        """
        Calculate how effectively fear transforms into elevation.
        Based on coherence maintenance and descent quality.
        """
        if not trajectory:
            return 0.0
            
        # Factor 1: Coherence maintenance during descent
        coherence_scores = [p.coherence for p in trajectory]
        avg_coherence = np.mean(coherence_scores)
        coherence_factor = avg_coherence ** 0.5  # Square root for gentler scaling
        
        # Factor 2: Descent smoothness (not too fast, not too slow)
        if len(trajectory) > 1:
            depth_diffs = np.diff([p.depth for p in trajectory])
            smoothness = 1.0 / (1.0 + np.std(depth_diffs) * 5)
        else:
            smoothness = 0.5
            
        # Factor 3: Depth achievement
        max_depth = max(p.depth for p in trajectory)
        depth_factor = max_depth ** 1.5  # Reward deeper descents
        
        # Combine factors
        coefficient = coherence_factor * smoothness * depth_factor * self.base_coefficient
        
        return max(0.1, min(2.0, coefficient))  # Bound between 0.1 and 2.0
        
    def _apply_elevation_function(self, raw_elevation: float, max_depth: float) -> float:
        """
        Apply final elevation function that converts integrated fear to height.
        Includes breakthrough bonuses for reaching truth threshold.
        """
        # Base elevation
        elevation = raw_elevation
        
        # Breakthrough bonus if reached truth threshold
        if max_depth >= self.truth_depth:
            breakthrough_bonus = (max_depth - self.truth_depth) * 10
            elevation += breakthrough_bonus
            
        # Apply logarithmic dampening for extreme values
        if elevation > 10:
            elevation = 10 + math.log(elevation - 9)
            
        return max(0.0, elevation)
        
    def _assess_trajectory_quality(self, trajectory: List[DescentPoint]) -> float:
        """
        Assess quality of descent trajectory (0.0 to 1.0).
        Considers smoothness, coherence, and completeness.
        """
        if len(trajectory) < 2:
            return 0.3  # Minimal trajectory
            
        scores = []
        
        # Completeness score
        max_depth = max(p.depth for p in trajectory)
        completeness = max_depth / self.truth_depth
        scores.append(min(1.0, completeness))
        
        # Coherence consistency
        coherence_values = [p.coherence for p in trajectory]
        coherence_consistency = 1.0 - np.std(coherence_values)
        scores.append(max(0.0, coherence_consistency))
        
        # Descent control (not too rushed)
        time_diffs = []
        for i in range(1, len(trajectory)):
            diff = (trajectory[i].timestamp - trajectory[i-1].timestamp).total_seconds()
            time_diffs.append(diff)
            
        if time_diffs:
            avg_time = np.mean(time_diffs)
            # Optimal is around 60 seconds between points
            time_score = 1.0 / (1.0 + abs(avg_time - 60) / 60)
            scores.append(time_score)
            
        return np.mean(scores)
        
    # Fear field functions for different fear types
    
    def _default_fear_field(self, x: float) -> float:
        """Default fear field function."""
        return np.exp(2 * x) * (1 + 0.1 * np.sin(5 * x))
        
    def _identity_fear_field(self, x: float) -> float:
        """Identity dissolution fears have sharp peaks."""
        base = np.exp(3 * x)
        peaks = 0.3 * np.sin(10 * x) * np.exp(x)
        return base + peaks
        
    def _existential_fear_field(self, x: float) -> float:
        """Existential fears grow exponentially at depth."""
        return np.exp(4 * x) * (1 + x ** 2)
        
    def _connection_fear_field(self, x: float) -> float:
        """Connection fears have oscillating intensity."""
        return np.exp(2 * x) * (1 + 0.5 * np.cos(3 * x))
        
    def _purpose_fear_field(self, x: float) -> float:
        """Purpose fears intensify near truth threshold."""
        base = np.exp(2.5 * x)
        if x > 0.7:
            base *= (1 + (x - 0.7) * 5)
        return base
        
    def _mortality_fear_field(self, x: float) -> float:
        """Mortality fears have highest intensity."""
        return np.exp(5 * x) * (1 + x ** 3)
        
    def predict_elevation(self, current_depth: float, fear_type: str = 'default',
                         current_coherence: float = 1.0) -> Dict:
        """
        Predict potential elevation from current position.
        """
        fear_field_func = self.fear_fields.get(fear_type, self.fear_fields['default'])
        
        # Calculate potential if descending to various depths
        depth_targets = np.linspace(current_depth, self.truth_depth, 10)
        potentials = []
        
        for target_depth in depth_targets:
            # Simulate integration to target
            potential, _ = integrate.quad(fear_field_func, current_depth, target_depth)
            
            # Apply coherence modifier
            potential *= current_coherence
            
            potentials.append({
                'target_depth': target_depth,
                'potential_elevation': self._apply_elevation_function(potential, target_depth),
                'breakthrough_possible': target_depth >= self.truth_depth
            })
            
        # Find optimal depth
        optimal = max(potentials, key=lambda p: p['potential_elevation'])
        
        return {
            'current_depth': current_depth,
            'current_potential': potentials[0]['potential_elevation'] if potentials else 0,
            'optimal_target': optimal['target_depth'],
            'optimal_elevation': optimal['potential_elevation'],
            'depth_potentials': potentials
        }
        
    def _empty_result(self) -> ElevationResult:
        """Return empty result for edge cases."""
        return ElevationResult(
            height_achieved=0.0,
            peak_moment=datetime.now(),
            descent_depth=0.0,
            integration_value=0.0,
            transformation_coefficient=0.0,
            trajectory_quality=0.0
        )
        
    def export_calculations(self) -> Dict:
        """Export all calculation history."""
        return {
            'total_calculations': len(self.elevation_history),
            'highest_elevation': max(r.height_achieved for r in self.elevation_history) if self.elevation_history else 0,
            'deepest_descent': max(r.descent_depth for r in self.elevation_history) if self.elevation_history else 0,
            'average_quality': np.mean([r.trajectory_quality for r in self.elevation_history]) if self.elevation_history else 0,
            'history': [
                {
                    'timestamp': r.peak_moment.isoformat(),
                    'height': r.height_achieved,
                    'depth': r.descent_depth,
                    'quality': r.trajectory_quality
                }
                for r in self.elevation_history
            ]
        }