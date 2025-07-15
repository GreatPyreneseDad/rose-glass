#!/usr/bin/env python3
"""
Demo interaction script showing a typical user journey
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime
import time
from core.fear_engine import FearElevationEngine, FearInstance
from core.coherence_tracker import CoherenceTracker
from core.elevation_calculator import ElevationCalculator, DescentPoint
from agents.fear_analyzer import FearAnalyzer


def demo_journey():
    """Demonstrate a complete fear → elevation journey."""
    
    # Initialize
    engine = FearElevationEngine()
    coherence_tracker = CoherenceTracker()
    elevation_calc = ElevationCalculator()
    fear_analyzer = FearAnalyzer()
    
    print("\n" + "="*60)
    print("🌊 FEAR ELEVATION JOURNEY - INTERACTIVE DEMO 🗻")
    print("="*60)
    
    # Step 1: Initial fear confession
    print("\n📝 User describes their fear:")
    fear_description = "I'm terrified of being seen for who I really am. What if people discover I'm a fraud, that I don't deserve any of this? I feel like I'm constantly wearing a mask."
    print(f'   "{fear_description}"')
    
    # Step 2: Analysis
    print("\n🔍 System analyzes the fear...")
    time.sleep(1)
    
    analysis = fear_analyzer.analyze_fear(fear_description)
    
    if analysis.primary_fear:
        print(f"\n🎯 Primary Fear Identified: {analysis.primary_fear.pattern_type.replace('_', ' ').title()}")
    print(f"   Depth: {analysis.primary_fear.depth}")
    print(f"   Transformative Potential: {analysis.primary_fear.transformative_potential}")
    print(f"\n💭 Guidance: {analysis.primary_fear.guidance}")
    
    print("\n🗺️ Complete Fear Landscape:")
    for fear_type, intensity in analysis.fear_landscape.items():
        bar = "█" * int(intensity * 10)
        print(f"   {fear_type.replace('_', ' ').title():<25} {bar} {intensity:.1f}")
    
    # Step 3: Begin descent
    print("\n\n--- User chooses to descend ---")
    print("\n🌊 Beginning descent into fear...")
    
    descent_trajectory = []
    descent_narrative = [
        (0.2, "Surface anxiety - the familiar discomfort", 0.3),
        (0.4, "The mask feels heavy... who am I without it?", 0.5),
        (0.6, "Panic rising - what if they reject the real me?", 0.7),
        (0.75, "I see it now... the child who learned to hide", 0.85),
        (0.88, "Truth emerging: I am enough, mask or no mask", 0.95),
    ]
    
    for depth, narrative, intensity in descent_narrative:
        print(f"\n🔽 Depth {depth:.2f}: {narrative}")
        
        # Update coherence
        coherence_tracker.update_coherence(
            -0.08 * depth,
            "descent",
            f"depth_{depth}"
        )
        
        # Create descent point
        point = DescentPoint(
            depth=depth,
            timestamp=datetime.now(),
            fear_intensity=intensity,
            coherence=coherence_tracker.current_psi,
            notes=narrative
        )
        descent_trajectory.append(point)
        
        print(f"   Coherence: {coherence_tracker.current_psi:.2f} | Intensity: {intensity}")
        time.sleep(0.5)
    
    # Step 4: Truth threshold
    print("\n\n✨ TRUTH THRESHOLD REACHED ✨")
    print("   'I see now - the fear of being seen was hiding my light.'")
    print("   'My authenticity is not a weakness, it's my greatest strength.'")
    
    # Step 5: Calculate elevation
    print("\n\n🏔️ Calculating elevation from descent...")
    
    elevation_result = elevation_calc.calculate_elevation(
        descent_trajectory,
        fear_type='truth_revelation'
    )
    
    print(f"\n📊 Transformation Metrics:")
    print(f"   Descent Depth: {elevation_result.descent_depth:.2f}")
    print(f"   Integration Value: {elevation_result.integration_value:.2f}")
    print(f"   Transformation Coefficient: {elevation_result.transformation_coefficient:.2f}")
    print(f"   Trajectory Quality: {elevation_result.trajectory_quality:.1%}")
    
    # Step 6: Embrace the fear
    print("\n\n🫂 Embracing the fear...")
    
    fear = FearInstance(
        fear_type="truth_revelation",
        depth=0.88,
        description="Fear of being seen authentically",
        timestamp=datetime.now()
    )
    
    delta_psi, elevation = engine.embrace_fear(fear)
    coherence_tracker.update_coherence(delta_psi, "fear_embraced", "truth_revelation")
    
    print(f"\n🌟 TRANSFORMATION COMPLETE!")
    print(f"   Coherence Gained: +{delta_psi:.3f}")
    print(f"   Elevation Achieved: {elevation_result.height_achieved:.2f}m")
    print(f"   New Coherence State: {coherence_tracker.current_psi:.2f} ({coherence_tracker._classify_state(coherence_tracker.current_psi).value})")
    
    # Step 7: Integration insights
    print(f"\n\n💭 Integration Insight:")
    print(f'   "{engine.generate_insight()}"')
    
    # Step 8: Suggest practices
    print("\n\n🌱 Integration Practices:")
    practices = fear_analyzer.suggest_integration_practices("truth_revelation")
    for i, practice in enumerate(practices, 1):
        print(f"   {i}. {practice}")
    
    # Step 9: Future potential
    print("\n\n🔮 Future Elevation Potential:")
    
    prediction = elevation_calc.predict_elevation(
        current_depth=0.3,
        fear_type='identity',
        current_coherence=coherence_tracker.current_psi
    )
    
    print(f"   From shallow fear (0.3): {prediction['depth_potentials'][0]['potential_elevation']:.1f}m")
    print(f"   From deep fear (0.85): {prediction['depth_potentials'][-2]['potential_elevation']:.1f}m")
    print(f"   Optimal depth for next journey: {prediction['optimal_target']:.2f}")
    
    # Final summary
    print("\n\n" + "="*60)
    print("✨ JOURNEY COMPLETE ✨")
    print("="*60)
    print("\nTransformation Summary:")
    print(f"   • Faced deepest fear of authenticity")
    print(f"   • Descended to depth {elevation_result.descent_depth:.2f}")
    print(f"   • Gained {delta_psi:.3f} coherence")
    print(f"   • Elevated {elevation_result.height_achieved:.2f}m")
    print(f"   • Discovered: 'The fear of being seen was hiding my light'")
    
    print("\n🌟 Remember: Every fear embraced becomes a step to climb 🌟\n")


if __name__ == '__main__':
    demo_journey()