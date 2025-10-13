"""
Rose Glass Perceptual Framework Demo
Shows how to use the perceptual framework in different phases
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from perception import (
    RoseGlassPerception, 
    PerceptionPhase,
    CulturalCalibration
)


def demo_phase1_explicit():
    """Demonstrate Phase 1: Explicit perception reporting"""
    print("=== PHASE 1: EXPLICIT PERCEPTION ===\n")
    
    perception = RoseGlassPerception(PerceptionPhase.EXPLICIT)
    
    # Example texts
    texts = [
        {
            'name': 'Academic',
            'text': """
            Recent studies have demonstrated that the theoretical framework 
            provides robust explanatory power. Furthermore, the empirical data 
            supports our hypothesis with statistical significance (p < 0.05).
            """
        },
        {
            'name': 'Crisis',
            'text': "Help! This is urgent! We need immediate action! Lives are at stake!"
        },
        {
            'name': 'Contemplative',
            'text': """
            When we sit with this question... allowing it to breathe... 
            we discover layers of meaning that weren't initially apparent. 
            Perhaps the answer lies not in rushing toward resolution, 
            but in dwelling with the uncertainty itself.
            """
        },
        {
            'name': 'Activist',
            'text': """
            We must unite! Together we can change this broken system! 
            Our collective power will overcome injustice! Join us in this movement!
            """
        }
    ]
    
    for example in texts:
        print(f"\n--- {example['name']} Text ---")
        result = perception.perceive(example['text'])
        
        # In Phase 1, we explicitly report perception
        print(f"Calibration: {result.calibration}")
        print(f"Dimensions:")
        print(f"  Ψ (Consistency): {result.dimensions.psi:.2f}")
        print(f"  ρ (Wisdom): {result.dimensions.rho:.2f}")
        print(f"  q (Activation): {result.dimensions.q:.2f}")
        print(f"  f (Social): {result.dimensions.f:.2f}")
        print(f"Rhythm: {result.rhythm['pace']}")
        print(f"Uncertainty: {result.uncertainty_level:.2f}")
        
        # Show calibrated response
        draft_response = "I understand what you're saying."
        calibrated = perception.calibrate_response(result, draft_response)
        print(f"Response: {calibrated}")


def demo_phase2_internal():
    """Demonstrate Phase 2: Internal processing only"""
    print("\n\n=== PHASE 2: INTERNAL PROCESSING ===\n")
    
    perception = RoseGlassPerception(PerceptionPhase.INTERNAL)
    
    # Crisis scenario
    crisis_text = "Emergency! The server is down! We're losing thousands per minute!"
    
    result = perception.perceive(crisis_text)
    
    # Phase 2: No explicit perception reporting
    print("User:", crisis_text)
    
    # Response is calibrated based on perception without explaining it
    draft = "Let me help you with that server issue. First, we should check the logs."
    calibrated = perception.calibrate_response(result, draft)
    
    print("\nCalibrated Response:", calibrated)
    print("(Notice: Response rhythm matches urgency without explicit explanation)")


def demo_cultural_calibrations():
    """Demonstrate different cultural calibrations"""
    print("\n\n=== CULTURAL CALIBRATIONS ===\n")
    
    perception = RoseGlassPerception()
    calibration = CulturalCalibration()
    
    # Text that could be interpreted differently
    text = """
    The path reveals itself through patient observation. 
    What appears as contradiction often contains deeper unity.
    We must look beyond surface appearances.
    """
    
    print("Original text:", text.strip())
    print("\nViewing through different cultural lenses:\n")
    
    # Try different calibrations
    calibrations_to_test = [
        'modern_western_academic',
        'buddhist_contemplative',
        'medieval_islamic'
    ]
    
    for cal_name in calibrations_to_test:
        # Apply specific calibration
        perception.calibration.default_calibration = cal_name
        result = perception.perceive(text)
        
        preset = calibration.get_calibration_info(cal_name)
        print(f"{cal_name}:")
        print(f"  Description: {preset.description}")
        print(f"  Coherence Pattern: Ψ={result.dimensions.psi:.2f}, "
              f"ρ={result.dimensions.rho:.2f}, q={result.dimensions.q:.2f}, "
              f"f={result.dimensions.f:.2f}")
        print()


def demo_uncertainty_handling():
    """Demonstrate uncertainty and superposition"""
    print("\n\n=== UNCERTAINTY HANDLING ===\n")
    
    perception = RoseGlassPerception()
    
    # Ambiguous text mixing technical and emotional
    ambiguous = """
    The algorithm must be optimized immediately! Our users are suffering!
    The O(n²) complexity is killing our community's experience!
    We need both technical excellence and human compassion here.
    """
    
    print("Ambiguous text:", ambiguous.strip())
    
    result = perception.perceive(ambiguous)
    
    print(f"\nUncertainty level: {result.uncertainty_level:.2f}")
    print(f"Primary calibration: {result.calibration}")
    
    if result.alternatives:
        print("\nAlternative interpretations detected:")
        for i, alt in enumerate(result.alternatives[:3]):
            print(f"  {i+1}. Different emphasis on dimensions")
            
    # Show how response handles uncertainty
    draft = "I'll help you with this situation."
    calibrated = perception.calibrate_response(result, draft)
    
    if result.uncertainty_level > 0.5:
        print("\nHigh uncertainty response:")
        print("I'm perceiving this as both a technical challenge and")
        print("an emotional situation for your community.")
        print("Would you like me to focus more on the technical optimization")
        print("or on addressing the user impact first?")
    else:
        print("\nCalibrated response:", calibrated)


def demo_breathing_rhythm():
    """Demonstrate breathing pattern matching"""
    print("\n\n=== BREATHING PATTERN MATCHING ===\n")
    
    perception = RoseGlassPerception()
    
    patterns = [
        {
            'name': 'Staccato urgency',
            'text': "Stop! Wait! Think! This matters! Act now!"
        },
        {
            'name': 'Flowing contemplation',
            'text': """
            As we consider this question more deeply... allowing ourselves
            to sit with the complexity... we might discover that the answer
            emerges not through force, but through patient attention...
            """
        },
        {
            'name': 'Academic precision',
            'text': """
            The hypothesis, when subjected to rigorous empirical testing
            under controlled conditions, demonstrates statistically significant
            results that warrant further investigation into the underlying
            mechanisms governing this phenomenon.
            """
        }
    ]
    
    for pattern in patterns:
        print(f"\n{pattern['name']}:")
        print(f"Text: {pattern['text'][:80]}...")
        
        result = perception.perceive(pattern['text'])
        rhythm = result.rhythm
        
        print(f"Detected rhythm: {rhythm['rhythm_type']}")
        print(f"Pace: {rhythm['pace']}")
        print(f"Breath depth: {rhythm['breath_depth']['avg_sentence_length']:.1f} words/sentence")
        
        # Show rhythm suggestion
        suggestion = perception.breathing_detector.suggest_response_rhythm(rhythm)
        print(f"Suggested response rhythm: {suggestion['target_pace']}")


def demo_evolution_tracking():
    """Demonstrate how perception evolves over conversation"""
    print("\n\n=== PERCEPTION EVOLUTION ===\n")
    
    perception = RoseGlassPerception()
    
    # Simulate a conversation where understanding develops
    conversation = [
        "I'm really confused about this whole situation...",
        "Wait, I think I see what you mean. Is it like this?",
        "Oh! That makes more sense now. So basically...",
        "Yes, exactly! I understand completely now. Thank you!"
    ]
    
    print("Tracking coherence evolution through conversation:\n")
    
    for i, message in enumerate(conversation):
        result = perception.perceive(message)
        
        print(f"Message {i+1}: \"{message}\"")
        print(f"  Ψ (Coherence): {result.dimensions.psi:.2f}")
        
        if i > 0:
            trend = perception.pattern_memory.get_evolution_trend('psi', window=3)
            if trend['trend'] > 0:
                print(f"  ↑ Coherence increasing (trend: +{trend['trend']:.3f})")
            else:
                print(f"  → Coherence stable")
    
    # Show summary
    summary = perception.pattern_memory.get_pattern_summary()
    print(f"\nFinal pattern summary:")
    print(f"  Average coherence: {summary['patterns']['psi']['mean']:.2f}")
    print(f"  Coherence range: {summary['patterns']['psi']['min']:.2f} - "
          f"{summary['patterns']['psi']['max']:.2f}")


def main():
    """Run all demonstrations"""
    print("ROSE GLASS PERCEPTUAL FRAMEWORK DEMONSTRATION")
    print("=" * 50)
    
    demos = [
        demo_phase1_explicit,
        demo_phase2_internal,
        demo_cultural_calibrations,
        demo_uncertainty_handling,
        demo_breathing_rhythm,
        demo_evolution_tracking
    ]
    
    for demo in demos:
        demo()
        print("\n" + "-" * 50)
        input("Press Enter to continue to next demo...")
    
    print("\nDemonstration complete!")
    print("\nKey insights:")
    print("- Phase 1 makes perception explicit for development/testing")
    print("- Phase 2+ integrates perception naturally without explanation")
    print("- Cultural calibrations reveal different aspects of same text")
    print("- Uncertainty is embraced, not eliminated")
    print("- Breathing patterns enable rhythm matching")
    print("- Perception evolves through conversation")


if __name__ == '__main__':
    main()