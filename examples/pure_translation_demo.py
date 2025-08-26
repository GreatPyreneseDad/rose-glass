"""
Pure Translation Demo: Seeing Through Different Lenses
=====================================================

This example demonstrates how the Rose Glass translates human expression
patterns without judgment or measurement. The same text appears differently
through different cultural lenses - all interpretations are valid.
"""

import sys
sys.path.append('..')

from src.core.rose_glass_v2 import RoseGlassV2


def demonstrate_pure_translation():
    """Show how translation works across different contexts"""
    
    # Initialize Rose Glass
    glass = RoseGlassV2()
    
    print("=" * 70)
    print("ROSE GLASS PURE TRANSLATION DEMONSTRATION")
    print("=" * 70)
    print("\nRemember: Numbers represent pattern intensity through specific lenses,")
    print("NOT quality measurements. Different lenses reveal different aspects.\n")
    
    # Example 1: A philosophical text pattern
    print("\n1. PHILOSOPHICAL TEXT PATTERN")
    print("-" * 40)
    
    philosophical_pattern = {
        'psi': 0.92,  # High internal consistency
        'rho': 0.88,  # High accumulated wisdom
        'q': 0.25,    # Low emotional activation
        'f': 0.35     # Low social architecture
    }
    
    print("Pattern detected in text:")
    print(f"  Ψ (Consistency): {philosophical_pattern['psi']}")
    print(f"  ρ (Wisdom): {philosophical_pattern['rho']}")
    print(f"  q (Emotion): {philosophical_pattern['q']}")
    print(f"  f (Social): {philosophical_pattern['f']}")
    
    # View through Medieval Islamic lens
    print("\n  Through Medieval Islamic Philosophy lens:")
    glass.select_lens('medieval_islamic')
    result1 = glass.translate_patterns(**philosophical_pattern)
    print(f"    Pattern Intensity: {result1.coherence_construction:.2f}/4.0")
    print(f"    Confidence: {result1.confidence.value}")
    print("    Interpretation: Classic philosophical restraint - emotion")
    print("    deliberately minimized to let reason shine clearly.")
    
    # View through Digital Native lens
    print("\n  Through Digital Native lens:")
    glass.select_lens('digital_native')
    result2 = glass.translate_patterns(**philosophical_pattern)
    print(f"    Pattern Intensity: {result2.coherence_construction:.2f}/4.0")
    print(f"    Confidence: {result2.confidence.value}")
    print("    Interpretation: Low energy, possibly boring or inaccessible.")
    print("    Would benefit from memes and relatable examples.")
    
    # Example 2: Social media activism pattern
    print("\n\n2. SOCIAL MEDIA ACTIVISM PATTERN")
    print("-" * 40)
    
    activism_pattern = {
        'psi': 0.45,  # Lower consistency (emotional)
        'rho': 0.30,  # Lower wisdom (action-focused)
        'q': 0.92,    # Very high moral activation
        'f': 0.88     # Very high social belonging
    }
    
    print("Pattern detected in text:")
    print(f"  Ψ (Consistency): {activism_pattern['psi']}")
    print(f"  ρ (Wisdom): {activism_pattern['rho']}")
    print(f"  q (Emotion): {activism_pattern['q']}")
    print(f"  f (Social): {activism_pattern['f']}")
    
    # View through Digital Native lens
    print("\n  Through Digital Native lens:")
    glass.select_lens('digital_native')
    result3 = glass.translate_patterns(**activism_pattern)
    print(f"    Pattern Intensity: {result3.coherence_construction:.2f}/4.0")
    print(f"    Confidence: {result3.confidence.value}")
    print("    Interpretation: High energy call to action! Strong")
    print("    collective resonance, ready to go viral.")
    
    # View through Buddhist Contemplative lens
    print("\n  Through Buddhist Contemplative lens:")
    glass.select_lens('buddhist_contemplative')
    result4 = glass.translate_patterns(**activism_pattern)
    print(f"    Pattern Intensity: {result4.coherence_construction:.2f}/4.0")
    print(f"    Confidence: {result4.confidence.value}")
    print("    Interpretation: Attachment to outcome creating suffering.")
    print("    High emotional activation may cloud clear seeing.")
    
    # Example 3: Personal storytelling pattern
    print("\n\n3. INDIGENOUS STORYTELLING PATTERN")
    print("-" * 40)
    
    story_pattern = {
        'psi': 0.55,  # Circular consistency
        'rho': 0.75,  # Embedded wisdom
        'q': 0.65,    # Moderate moral teaching
        'f': 0.82     # High collective connection
    }
    
    print("Pattern detected in text:")
    print(f"  Ψ (Consistency): {story_pattern['psi']}")
    print(f"  ρ (Wisdom): {story_pattern['rho']}")
    print(f"  q (Emotion): {story_pattern['q']}")
    print(f"  f (Social): {story_pattern['f']}")
    
    # View through Indigenous Oral lens
    print("\n  Through Indigenous Oral Tradition lens:")
    glass.select_lens('indigenous_oral')
    result5 = glass.translate_patterns(**story_pattern)
    print(f"    Pattern Intensity: {result5.coherence_construction:.2f}/4.0")
    print(f"    Confidence: {result5.confidence.value}")
    print("    Interpretation: Strong teaching story with ancestral wisdom.")
    print("    Circular structure carries deep cultural knowledge.")
    
    # View through Medieval Islamic lens
    print("\n  Through Medieval Islamic Philosophy lens:")
    glass.select_lens('medieval_islamic')
    result6 = glass.translate_patterns(**story_pattern)
    print(f"    Pattern Intensity: {result6.coherence_construction:.2f}/4.0")
    print(f"    Confidence: {result6.confidence.value}")
    print("    Interpretation: Lacks systematic argumentation expected")
    print("    in philosophical discourse. Pattern less visible.")
    
    # Show comparison across all lenses for one pattern
    print("\n\n4. MULTI-LENS COMPARISON")
    print("-" * 40)
    print("Viewing the storytelling pattern through ALL available lenses:")
    
    comparisons = glass.compare_lenses(**story_pattern)
    print("\nPattern intensities by lens:")
    for lens_name, interpretation in comparisons.items():
        print(f"  {lens_name:.<30} {interpretation.coherence_construction:.2f}/4.0")
    
    print("\n" + "=" * 70)
    print("KEY INSIGHTS FROM THIS DEMONSTRATION:")
    print("=" * 70)
    
    insights = [
        "1. The same pattern appears differently through different lenses",
        "2. High intensity through one lens ≠ 'better' communication",
        "3. Low intensity through a lens ≠ 'worse' communication", 
        "4. Each lens reveals different aspects of human expression",
        "5. Choosing the right lens requires cultural awareness",
        "6. Multiple lenses may be needed for full understanding"
    ]
    
    for insight in insights:
        print(f"\n{insight}")
    
    print("\n" + "=" * 70)
    print("ALTERNATIVE READINGS")
    print("=" * 70)
    print("\nThe Rose Glass always provides alternative interpretations:")
    
    # Get alternative readings for the activism pattern
    glass.select_lens('digital_native')
    interpretation = glass.translate_patterns(**activism_pattern)
    
    print("\nFor the activism pattern, alternative readings include:")
    for alt in interpretation.alternative_readings:
        print(f"  • {alt}")
    
    print("\nThese alternatives remind us that patterns can be read many ways.")
    print("The Rose Glass is a tool for understanding, not judgment.")


if __name__ == "__main__":
    demonstrate_pure_translation()