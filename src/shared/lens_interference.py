"""
Unified Lens Interference Module
================================

Shared across: rose-glass, rose-looking-glass, RoseGlassLE

Implements λ (lambda) - the lens interference coefficient that measures
how different cultural lenses affect interpretation.

"The same words refract differently through different perspectives."

Author: Christopher MacGregor bin Joseph
Date: December 2025
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum


class InterferenceType(Enum):
    """Types of lens interference patterns"""
    CONSTRUCTIVE = "constructive"     # Lenses reinforce each other
    DESTRUCTIVE = "destructive"       # Lenses contradict each other
    NEUTRAL = "neutral"               # Lenses independent
    RESONANT = "resonant"             # Lenses create harmonic patterns


@dataclass
class LensReading:
    """Reading from a single cultural lens"""
    lens_name: str
    coherence: float
    psi: float
    rho: float
    q: float
    f: float
    weight_profile: Dict[str, float]


@dataclass
class InterferenceAnalysis:
    """Complete interference analysis across lenses"""
    lambda_coefficient: float          # Primary interference measure (0-1)
    interference_type: InterferenceType
    lens_readings: Dict[str, LensReading]
    dominant_lens: str                 # Best-fit lens for this text
    conflicting_lenses: List[Tuple[str, str]]  # Pairs that disagree
    resonant_lenses: List[Tuple[str, str]]     # Pairs that reinforce
    optimal_lens_recommendation: str
    interpretation_stability: float    # How stable across lenses (inverse of λ)
    

class LensInterferenceAnalyzer:
    """
    Analyzes interference patterns across cultural lenses.
    
    λ measures how much interpretation varies across lenses:
    - Low λ (0.0-0.2): Lens-stable (universal patterns)
    - Mid λ (0.2-0.5): Context-dependent
    - High λ (0.5-1.0): Lens-dependent (cultural specificity)
    
    Low λ suggests universal truth; high λ requires cultural sensitivity.
    """
    
    # Standard cultural lens weight profiles
    DEFAULT_LENSES = {
        'modern_academic': {
            'weight_psi': 0.40, 'weight_rho': 0.40,
            'weight_q': 0.10, 'weight_f': 0.10
        },
        'digital_native': {
            'weight_psi': 0.15, 'weight_rho': 0.20,
            'weight_q': 0.35, 'weight_f': 0.30
        },
        'contemplative': {
            'weight_psi': 0.30, 'weight_rho': 0.45,
            'weight_q': 0.15, 'weight_f': 0.10
        },
        'activist': {
            'weight_psi': 0.20, 'weight_rho': 0.15,
            'weight_q': 0.40, 'weight_f': 0.25
        },
        'trauma_informed': {
            'weight_psi': 0.25, 'weight_rho': 0.20,
            'weight_q': 0.35, 'weight_f': 0.20
        },
        'law_enforcement': {
            'weight_psi': 0.35, 'weight_rho': 0.30,
            'weight_q': 0.20, 'weight_f': 0.15
        }
    }
    
    def __init__(self, lenses: Optional[Dict] = None):
        """
        Initialize interference analyzer.
        
        Args:
            lenses: Custom lens definitions or None for defaults
        """
        self.lenses = lenses or self.DEFAULT_LENSES
        
    def calculate_lens_coherence(
        self,
        psi: float,
        rho: float,
        q: float,
        f: float,
        lens_name: str
    ) -> float:
        """
        Calculate coherence through a specific lens.
        
        Args:
            psi, rho, q, f: GCT variables
            lens_name: Name of lens to use
            
        Returns:
            Weighted coherence value
        """
        if lens_name not in self.lenses:
            raise KeyError(f"Unknown lens: {lens_name}")
            
        weights = self.lenses[lens_name]
        
        coherence = (
            psi * weights['weight_psi'] +
            rho * weights['weight_rho'] +
            q * weights['weight_q'] +
            f * weights['weight_f']
        )
        
        return coherence
    
    def analyze_interference(
        self,
        psi: float,
        rho: float,
        q: float,
        f: float,
        include_lenses: Optional[List[str]] = None
    ) -> InterferenceAnalysis:
        """
        Analyze interference across all or specified lenses.
        
        Args:
            psi, rho, q, f: GCT variables
            include_lenses: Specific lenses to analyze (None = all)
            
        Returns:
            InterferenceAnalysis with complete results
        """
        lenses_to_use = include_lenses or list(self.lenses.keys())
        
        # Calculate readings for each lens
        readings = {}
        coherences = []
        
        for lens_name in lenses_to_use:
            coherence = self.calculate_lens_coherence(psi, rho, q, f, lens_name)
            coherences.append(coherence)
            
            readings[lens_name] = LensReading(
                lens_name=lens_name,
                coherence=coherence,
                psi=psi,
                rho=rho,
                q=q,
                f=f,
                weight_profile=self.lenses[lens_name]
            )
        
        # Calculate lambda (standard deviation of coherences)
        lambda_coefficient = np.std(coherences) if len(coherences) > 1 else 0.0
        
        # Normalize to 0-1 range (assuming max practical std ~0.3)
        lambda_normalized = min(lambda_coefficient / 0.3, 1.0)
        
        # Determine interference type
        interference_type = self._classify_interference(coherences)
        
        # Find dominant lens (highest coherence)
        dominant_lens = max(readings.keys(), key=lambda k: readings[k].coherence)
        
        # Find conflicting and resonant pairs
        conflicting = []
        resonant = []
        
        for i, lens1 in enumerate(lenses_to_use):
            for lens2 in lenses_to_use[i+1:]:
                diff = abs(readings[lens1].coherence - readings[lens2].coherence)
                if diff > 0.15:
                    conflicting.append((lens1, lens2))
                elif diff < 0.05:
                    resonant.append((lens1, lens2))
        
        # Recommend optimal lens
        optimal = self._recommend_lens(readings, psi, rho, q, f)
        
        # Interpretation stability (inverse of lambda)
        stability = 1.0 - lambda_normalized
        
        return InterferenceAnalysis(
            lambda_coefficient=lambda_normalized,
            interference_type=interference_type,
            lens_readings=readings,
            dominant_lens=dominant_lens,
            conflicting_lenses=conflicting,
            resonant_lenses=resonant,
            optimal_lens_recommendation=optimal,
            interpretation_stability=stability
        )
    
    def _classify_interference(self, coherences: List[float]) -> InterferenceType:
        """Classify the type of interference pattern"""
        if len(coherences) < 2:
            return InterferenceType.NEUTRAL
            
        std = np.std(coherences)
        mean = np.mean(coherences)
        
        if std < 0.05:
            return InterferenceType.CONSTRUCTIVE  # All lenses agree
        elif std > 0.15:
            return InterferenceType.DESTRUCTIVE   # Lenses disagree
        elif mean > 0.6:
            return InterferenceType.RESONANT      # High agreement at high values
        else:
            return InterferenceType.NEUTRAL
    
    def _recommend_lens(
        self,
        readings: Dict[str, LensReading],
        psi: float,
        rho: float,
        q: float,
        f: float
    ) -> str:
        """Recommend optimal lens based on variable profile"""
        # Find which variable is dominant
        variables = {'psi': psi, 'rho': rho, 'q': q, 'f': f}
        dominant_var = max(variables.keys(), key=lambda k: variables[k])
        
        # Match to lens that weights this variable highest
        best_lens = None
        best_weight = 0.0
        
        for lens_name, weights in self.lenses.items():
            weight_key = f'weight_{dominant_var}'
            if weights.get(weight_key, 0) > best_weight:
                best_weight = weights[weight_key]
                best_lens = lens_name
        
        return best_lens or list(self.lenses.keys())[0]
    
    def calculate_lens_deviation(
        self,
        psi: float,
        rho: float,
        q: float,
        f: float
    ) -> float:
        """
        Calculate standard deviation across all lenses.
        
        This is the Veritas distortion index D(P).
        
        Args:
            psi, rho, q, f: GCT variables
            
        Returns:
            Standard deviation of coherences (0-1 normalized)
        """
        analysis = self.analyze_interference(psi, rho, q, f)
        return analysis.lambda_coefficient
    
    def is_lens_invariant(
        self,
        psi: float,
        rho: float,
        q: float,
        f: float,
        threshold: float = 0.1
    ) -> Tuple[bool, float]:
        """
        Check if pattern is lens-invariant (universal truth).
        
        Args:
            psi, rho, q, f: GCT variables
            threshold: Maximum λ for invariance
            
        Returns:
            (is_invariant, lambda_value)
        """
        deviation = self.calculate_lens_deviation(psi, rho, q, f)
        return deviation < threshold, deviation


def extract_lambda(psi: float, rho: float, q: float, f: float) -> float:
    """
    Convenience function for GCT integration.
    
    Args:
        psi, rho, q, f: GCT variables
        
    Returns:
        λ value (0-1)
    """
    analyzer = LensInterferenceAnalyzer()
    return analyzer.calculate_lens_deviation(psi, rho, q, f)


# === Tests ===
if __name__ == "__main__":
    print("=" * 70)
    print("LENS INTERFERENCE ANALYSIS (λ)")
    print("=" * 70)
    
    analyzer = LensInterferenceAnalyzer()
    
    test_cases = [
        # (psi, rho, q, f, label)
        (0.9, 0.9, 0.1, 0.1, "Academic - high logic/wisdom"),
        (0.5, 0.4, 0.8, 0.9, "Activist - high emotion/social"),
        (0.6, 0.6, 0.5, 0.5, "Balanced - universal"),
        (0.3, 0.8, 0.2, 0.3, "Contemplative - high wisdom"),
        (0.2, 0.2, 0.9, 0.4, "Crisis - high emotion"),
    ]
    
    for psi, rho, q, f, label in test_cases:
        analysis = analyzer.analyze_interference(psi, rho, q, f)
        print(f"\n[{label}]")
        print(f"  Variables: Ψ={psi}, ρ={rho}, q={q}, f={f}")
        print(f"  λ = {analysis.lambda_coefficient:.3f} ({analysis.interference_type.value})")
        print(f"  Stability: {analysis.interpretation_stability:.2%}")
        print(f"  Dominant lens: {analysis.dominant_lens}")
        print(f"  Recommended: {analysis.optimal_lens_recommendation}")
        if analysis.conflicting_lenses:
            print(f"  Conflicts: {analysis.conflicting_lenses[:3]}")
