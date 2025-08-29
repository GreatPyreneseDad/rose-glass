"""
Translation Clarity: Replacing Confidence with Multi-dimensional Understanding
============================================================================

This module reframes "translation confidence" as "translation clarity" with
multiple dimensions, avoiding implications of correctness or accuracy.
"""

from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Optional


class TranslationClarity(Enum):
    """
    Describes how clearly patterns translate through a lens.
    Not about correctness - about visibility and interpretability.
    """
    CRYSTALLINE = "Patterns fully visible - like looking through clear glass"
    FLOWING = "Patterns visible but shifting - like looking through water"  
    MISTY = "Patterns partially visible - like looking through fog"
    OPAQUE = "Patterns barely visible - lens may not fit this expression"
    PRISMATIC = "Multiple equally valid patterns visible - like through a prism"


class TranslationMultiplicity(Enum):
    """
    Describes how many valid translations exist
    """
    SINGULAR = "One primary translation emerges"
    DUAL = "Two equally valid translations present"
    MULTIPLE = "Many valid translations coexist"
    INFINITE = "Unlimited valid translations possible"


@dataclass
class TranslationSpectrum:
    """
    Replaces single confidence score with multi-dimensional spectrum
    """
    clarity: TranslationClarity
    multiplicity: TranslationMultiplicity
    resonance_notes: List[str]  # What resonates through this lens
    interference_notes: List[str]  # What creates interference
    alternative_lenses: List[str]  # Other lenses that might reveal more
    uncertainty_celebration: str  # Why uncertainty is valuable here
    
    def describe(self) -> str:
        """Generate narrative description of translation spectrum"""
        descriptions = {
            TranslationClarity.CRYSTALLINE: 
                "The patterns translate with crystal clarity through this lens.",
            TranslationClarity.FLOWING: 
                "The patterns flow and shift like water - alive and changing.",
            TranslationClarity.MISTY: 
                "The patterns appear through mist - present but softly defined.",
            TranslationClarity.OPAQUE: 
                "The patterns resist this lens - perhaps another would serve better.",
            TranslationClarity.PRISMATIC: 
                "The patterns split into rainbow spectra - many truths at once."
        }
        
        multiplicity_descriptions = {
            TranslationMultiplicity.SINGULAR: 
                "A single translation emerges clearly.",
            TranslationMultiplicity.DUAL: 
                "Two equally valid translations dance together.",
            TranslationMultiplicity.MULTIPLE: 
                "Multiple translations coexist without conflict.",
            TranslationMultiplicity.INFINITE: 
                "Infinite translations spiral outward like galaxies."
        }
        
        base = descriptions[self.clarity]
        multi = multiplicity_descriptions[self.multiplicity]
        
        return f"{base} {multi}\n\n{self.uncertainty_celebration}"


class TranslationClarityAnalyzer:
    """
    Analyzes translation clarity without implying correctness
    """
    
    @staticmethod
    def analyze_clarity(pattern_visibility: Dict[str, float], 
                       lens_name: str) -> TranslationClarity:
        """
        Determine clarity based on pattern visibility.
        High visibility doesn't mean "correct" - means clear through this lens.
        """
        avg_visibility = sum(pattern_visibility.values()) / len(pattern_visibility)
        variance = sum((v - avg_visibility)**2 for v in pattern_visibility.values())
        
        # Crystalline: high visibility, low variance
        if avg_visibility > 0.8 and variance < 0.1:
            return TranslationClarity.CRYSTALLINE
            
        # Prismatic: high visibility but high variance (multiple patterns)
        elif avg_visibility > 0.7 and variance > 0.3:
            return TranslationClarity.PRISMATIC
            
        # Flowing: moderate visibility with movement
        elif 0.4 < avg_visibility < 0.7:
            return TranslationClarity.FLOWING
            
        # Misty: low visibility but present
        elif 0.2 < avg_visibility <= 0.4:
            return TranslationClarity.MISTY
            
        # Opaque: very low visibility
        else:
            return TranslationClarity.OPAQUE
    
    @staticmethod
    def identify_multiplicity(patterns: Dict[str, float], 
                            alternatives: List[str]) -> TranslationMultiplicity:
        """
        Identify how many valid translations exist.
        More alternatives = richer understanding, not confusion.
        """
        if len(alternatives) == 0:
            return TranslationMultiplicity.SINGULAR
        elif len(alternatives) <= 2:
            return TranslationMultiplicity.DUAL
        elif len(alternatives) <= 5:
            return TranslationMultiplicity.MULTIPLE
        else:
            return TranslationMultiplicity.INFINITE
    
    @staticmethod
    def celebrate_uncertainty(clarity: TranslationClarity,
                            multiplicity: TranslationMultiplicity) -> str:
        """
        Generate text explaining why this uncertainty is valuable.
        Uncertainty is a feature, not a bug.
        """
        celebrations = {
            (TranslationClarity.MISTY, TranslationMultiplicity.MULTIPLE): 
                "The mist reveals possibility - many truths shimmer just out of focus, inviting exploration.",
            
            (TranslationClarity.PRISMATIC, TranslationMultiplicity.INFINITE):
                "Like light through a prism, this expression contains infinite spectra. Each viewing angle reveals new beauty.",
            
            (TranslationClarity.OPAQUE, TranslationMultiplicity.MULTIPLE):
                "This lens struggles with these patterns - a reminder that no single lens can see all. The opacity itself teaches humility.",
            
            (TranslationClarity.FLOWING, TranslationMultiplicity.DUAL):
                "Two translations flow together like converging rivers. Neither is wrong; both are true.",
            
            (TranslationClarity.CRYSTALLINE, TranslationMultiplicity.SINGULAR):
                "Even crystal clarity is just one way of seeing. Other lenses would reveal what this clarity conceals."
        }
        
        key = (clarity, multiplicity)
        if key in celebrations:
            return celebrations[key]
        else:
            return "Every translation is a gift, every uncertainty a teacher."
    
    @classmethod
    def create_spectrum(cls, patterns: Dict[str, float],
                       lens_name: str,
                       alternatives: List[str],
                       cultural_context: Dict) -> TranslationSpectrum:
        """
        Create a full translation spectrum analysis
        """
        clarity = cls.analyze_clarity(patterns, lens_name)
        multiplicity = cls.identify_multiplicity(patterns, alternatives)
        
        # Identify what resonates and interferes
        resonance_notes = cls._identify_resonance(patterns, cultural_context)
        interference_notes = cls._identify_interference(patterns, cultural_context)
        
        # Suggest alternative lenses
        alternative_lenses = cls._suggest_alternatives(patterns, lens_name)
        
        # Celebrate the uncertainty
        uncertainty_celebration = cls.celebrate_uncertainty(clarity, multiplicity)
        
        return TranslationSpectrum(
            clarity=clarity,
            multiplicity=multiplicity,
            resonance_notes=resonance_notes,
            interference_notes=interference_notes,
            alternative_lenses=alternative_lenses,
            uncertainty_celebration=uncertainty_celebration
        )
    
    @staticmethod
    def _identify_resonance(patterns: Dict[str, float], 
                          context: Dict) -> List[str]:
        """Identify what resonates well through this lens"""
        resonances = []
        
        if patterns.get('psi', 0) > 0.8:
            resonances.append("Strong harmonic consistency resonates clearly")
        if patterns.get('rho', 0) > 0.7:
            resonances.append("Deep wisdom patterns shine through")
        if patterns.get('q', 0) > 0.6:
            resonances.append("Moral energy pulses with life")
        if patterns.get('f', 0) > 0.7:
            resonances.append("Collective architecture sings in harmony")
            
        return resonances if resonances else ["Unique patterns await discovery"]
    
    @staticmethod
    def _identify_interference(patterns: Dict[str, float], 
                             context: Dict) -> List[str]:
        """Identify what creates interference in translation"""
        interferences = []
        
        # Low values might indicate interference OR different expression
        if patterns.get('psi', 1) < 0.3:
            interferences.append("Non-linear thought creates beautiful static")
        if patterns.get('f', 1) < 0.3:
            interferences.append("Individual voice may seem isolated to collective lens")
            
        return interferences if interferences else ["Clear transmission"]
    
    @staticmethod
    def _suggest_alternatives(patterns: Dict[str, float], 
                            current_lens: str) -> List[str]:
        """Suggest other lenses that might reveal different aspects"""
        suggestions = []
        
        # Pattern-based suggestions
        if patterns.get('q', 0) > 0.8 and 'contemplative' not in current_lens:
            suggestions.append("A contemplative lens might reveal the stillness within motion")
        
        if patterns.get('f', 0) < 0.3 and 'individual' not in current_lens:
            suggestions.append("An individualist lens might better honor this solitary voice")
            
        if patterns.get('psi', 1) < 0.5 and 'indigenous' not in current_lens:
            suggestions.append("An indigenous lens might see circular wisdom where others see inconsistency")
            
        return suggestions if suggestions else ["This lens serves well for now"]


# Example usage showing the shift from confidence to clarity
def demonstrate_clarity_over_confidence():
    """Show how clarity differs from confidence"""
    
    # Same pattern, different lens
    pattern = {'psi': 0.45, 'rho': 0.8, 'q': 0.3, 'f': 0.7}
    
    # Old way (confidence - implies correctness)
    print("OLD: Translation Confidence: 0.73 (HIGH)")
    print("This suggests the translation is 'correct'\n")
    
    # New way (clarity spectrum)
    spectrum = TranslationClarityAnalyzer.create_spectrum(
        patterns=pattern,
        lens_name="western_academic",
        alternatives=["indigenous_oral", "zen_koan", "african_ubuntu"],
        cultural_context={"type": "philosophical"}
    )
    
    print("NEW: Translation Spectrum")
    print(spectrum.describe())
    print(f"\nResonance: {', '.join(spectrum.resonance_notes)}")
    print(f"Interference: {', '.join(spectrum.interference_notes)}")
    print(f"Try also: {', '.join(spectrum.alternative_lenses)}")
    
    print("\nThe shift: From measuring correctness to celebrating multiplicity")


if __name__ == "__main__":
    demonstrate_clarity_over_confidence()