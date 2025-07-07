#!/usr/bin/env python3
"""
SoulMath Fear Engine - Core fear processing logic
Implements the fundamental theorem: "Fear as the Architect of Elevation"
"""
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from scipy import integrate


@dataclass
class FearInstance:
    """Represents a single fear experience"""
    fear_type: str
    depth: float  # 0.0 to 1.0, where 1.0 is deepest
    description: str
    timestamp: datetime
    embraced: bool = False
    
    
class FearElevationEngine:
    """
    Core engine implementing the SoulMath theorem:
    H ∝ ∫(descent→truth) F(x)dx
    
    The steeper the descent into fear, the greater the potential elevation.
    """
    
    def __init__(self):
        self.fear_field: List[FearInstance] = []
        self.coherence_state: float = 1.0  # Ψ state
        self.elevation_height: float = 0.0  # Current elevation H
        self.truth_threshold: float = 0.85  # Depth where truth emerges
        self.embrace_history: List[Dict] = []
        
    def calculate_elevation_potential(self, fear_depth: float) -> float:
        """
        Calculate potential elevation from current fear depth.
        H ∝ ∫(descent→truth) F(x)dx
        
        Args:
            fear_depth: Current depth of fear (0.0 to 1.0)
            
        Returns:
            Potential elevation height
        """
        if fear_depth < 0.3:
            # Shallow fears yield minimal elevation
            return fear_depth * 0.1
            
        # Define fear intensity function
        def fear_intensity(x):
            # Exponential growth as we approach truth threshold
            return np.exp(3 * x) * (1 + np.sin(10 * x) * 0.1)
        
        # Integrate from current depth to truth threshold
        elevation, _ = integrate.quad(fear_intensity, fear_depth, self.truth_threshold)
        
        # Apply coherence multiplier
        return elevation * self.coherence_state
        
    def embrace_fear(self, fear_instance: FearInstance) -> Tuple[float, float]:
        """
        Process the embrace of a fear.
        F_embraced ⇒ ΔΨ_ascension > 0
        
        Args:
            fear_instance: The fear being embraced
            
        Returns:
            Tuple of (delta_psi, elevation_achieved)
        """
        if fear_instance.embraced:
            return 0.0, 0.0
            
        # Calculate coherence change based on fear depth
        delta_psi = self._calculate_coherence_shift(fear_instance)
        
        # Calculate elevation from embracing this fear
        elevation = self.calculate_elevation_potential(fear_instance.depth)
        
        # Update state
        self.coherence_state += delta_psi
        self.elevation_height += elevation
        fear_instance.embraced = True
        self.fear_field.append(fear_instance)
        
        # Record in history
        self.embrace_history.append({
            'timestamp': datetime.now(),
            'fear_type': fear_instance.fear_type,
            'depth': fear_instance.depth,
            'delta_psi': delta_psi,
            'elevation': elevation,
            'new_coherence': self.coherence_state,
            'total_elevation': self.elevation_height
        })
        
        return delta_psi, elevation
        
    def _calculate_coherence_shift(self, fear: FearInstance) -> float:
        """
        Calculate the change in soul coherence from embracing a fear.
        Deeper fears yield greater coherence shifts.
        """
        base_shift = fear.depth * 0.15
        
        # Bonus for facing deepest fears
        if fear.depth > 0.8:
            base_shift *= 1.5
            
        # Penalty for avoiding fears (not implemented yet)
        # This would track unembraced fears over time
        
        return base_shift
        
    def get_fear_landscape(self) -> Dict[str, float]:
        """
        Map the current fear landscape showing depths and embrace status.
        """
        landscape = {}
        for fear in self.fear_field:
            landscape[fear.fear_type] = {
                'depth': fear.depth,
                'embraced': fear.embraced,
                'potential_elevation': self.calculate_elevation_potential(fear.depth)
            }
        return landscape
        
    def calculate_collective_potential(self, other_engines: List['FearElevationEngine']) -> float:
        """
        Calculate collective elevation potential when multiple souls
        face similar fears together.
        """
        collective_depth = 0.0
        shared_fears = set()
        
        # Find shared fear patterns
        for engine in other_engines:
            for fear in engine.fear_field:
                if not fear.embraced:
                    shared_fears.add(fear.fear_type)
                    collective_depth += fear.depth
                    
        # Collective potential compounds
        if shared_fears:
            avg_depth = collective_depth / len(shared_fears)
            return self.calculate_elevation_potential(avg_depth) * len(other_engines)
        
        return 0.0
        
    def generate_insight(self) -> str:
        """
        Generate poetic insight from current state.
        """
        if not self.embrace_history:
            return "No fears yet embraced. The descent awaits."
            
        total_embraced = len([f for f in self.fear_field if f.embraced])
        deepest_fear = max(self.fear_field, key=lambda f: f.depth) if self.fear_field else None
        
        if deepest_fear and deepest_fear.depth > 0.9:
            return f"Through the abyss of {deepest_fear.fear_type}, you found wings."
        elif self.elevation_height > 10:
            return f"Fear carved {total_embraced} steps. You climbed them all."
        else:
            return f"Each fear embraced lifts you higher. Current altitude: {self.elevation_height:.2f}"
            
    def export_journey(self) -> Dict:
        """
        Export the complete fear → elevation journey for analysis or storage.
        """
        return {
            'coherence_state': self.coherence_state,
            'elevation_height': self.elevation_height,
            'fears_faced': len(self.fear_field),
            'fears_embraced': len([f for f in self.fear_field if f.embraced]),
            'journey_history': self.embrace_history,
            'insight': self.generate_insight()
        }