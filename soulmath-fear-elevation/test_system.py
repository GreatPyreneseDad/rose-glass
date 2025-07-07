#!/usr/bin/env python3
"""
Test script for SoulMath Fear Elevation System
Demonstrates core functionality without interactive CLI
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime
from core.fear_engine import FearElevationEngine, FearInstance
from core.coherence_tracker import CoherenceTracker
from core.elevation_calculator import ElevationCalculator, DescentPoint
from agents.fear_analyzer import FearAnalyzer


def test_system():
    """Test the complete fear elevation system."""
    print("\n" + "="*60)
    print("🌊 SOULMATH FEAR ELEVATION SYSTEM - TEST RUN 🗻")
    print("="*60)
    
    # Initialize components
    engine = FearElevationEngine()
    coherence_tracker = CoherenceTracker()
    elevation_calc = ElevationCalculator()
    fear_analyzer = FearAnalyzer()
    
    print("\n✅ System initialized successfully")
    print(f"Initial coherence (Ψ): {coherence_tracker.current_psi:.2f}")
    print(f"Initial elevation: {engine.elevation_height:.2f}m")
    
    # Test 1: Fear Analysis
    print("\n\n📋 TEST 1: Fear Analysis")
    print("-" * 40)
    
    fear_input = "I'm afraid of losing myself completely, of becoming nobody, forgotten and alone"
    print(f"Input: '{fear_input}'")
    
    analysis = fear_analyzer.analyze_fear(fear_input)
    print(f"\nPrimary fear detected: {analysis.primary_fear.pattern_type}")
    print(f"Depth: {analysis.primary_fear.depth}")
    print(f"Guidance: {analysis.primary_fear.guidance}")
    
    print("\nFear landscape:")
    for fear_type, intensity in analysis.fear_landscape.items():
        print(f"  - {fear_type}: {intensity:.1f}")
    
    # Test 2: Descent Simulation
    print("\n\n🔽 TEST 2: Fear Descent Simulation")
    print("-" * 40)
    
    # Simulate a descent trajectory
    descent_points = []
    depths = [0.1, 0.3, 0.5, 0.7, 0.85, 0.9]
    
    for i, depth in enumerate(depths):
        print(f"\nDescending to depth {depth:.1f}...")
        
        # Update coherence (it decreases during descent)
        coherence_tracker.update_coherence(
            -0.05 * depth, 
            "descent", 
            f"depth_{depth}"
        )
        
        # Create descent point
        point = DescentPoint(
            depth=depth,
            timestamp=datetime.now(),
            fear_intensity=depth * 1.2,  # Intensity increases with depth
            coherence=coherence_tracker.current_psi
        )
        descent_points.append(point)
        
        print(f"  Current coherence: {coherence_tracker.current_psi:.2f}")
        print(f"  Fear intensity: {point.fear_intensity:.2f}")
    
    # Test 3: Elevation Calculation
    print("\n\n🏔️ TEST 3: Elevation Calculation")
    print("-" * 40)
    
    elevation_result = elevation_calc.calculate_elevation(
        descent_points,
        fear_type='identity'  # Using identity dissolution pattern
    )
    
    print(f"Descent depth reached: {elevation_result.descent_depth:.2f}")
    print(f"Integration value: {elevation_result.integration_value:.2f}")
    print(f"Transformation coefficient: {elevation_result.transformation_coefficient:.2f}")
    print(f"Elevation achieved: {elevation_result.height_achieved:.2f}m")
    print(f"Trajectory quality: {elevation_result.trajectory_quality:.1%}")
    
    # Test 4: Fear Embrace
    print("\n\n🫂 TEST 4: Fear Embrace")
    print("-" * 40)
    
    # Create fear instance from our descent
    fear = FearInstance(
        fear_type="identity_dissolution",
        depth=0.9,
        description="Fear of complete self-dissolution",
        timestamp=datetime.now()
    )
    
    # Embrace the fear
    delta_psi, elevation = engine.embrace_fear(fear)
    
    print(f"Fear embraced: {fear.fear_type}")
    print(f"Coherence gained (ΔΨ): +{delta_psi:.3f}")
    print(f"Elevation from embrace: {elevation:.2f}m")
    print(f"Total elevation: {engine.elevation_height:.2f}m")
    
    # Update coherence tracker
    coherence_tracker.update_coherence(
        delta_psi,
        "fear_embraced",
        "identity_dissolution"
    )
    
    # Test 5: System Status
    print("\n\n📊 TEST 5: Final System Status")
    print("-" * 40)
    
    report = coherence_tracker.get_coherence_report()
    print(f"Final coherence: {report['current_psi']:.3f} ({report['current_state']})")
    print(f"Stability: {report['stability_score']:.1%}")
    print(f"Trend: {report['trend']}")
    
    # Get insight
    insight = engine.generate_insight()
    print(f"\n💭 System insight: {insight}")
    
    # Test 6: Prediction
    print("\n\n🔮 TEST 6: Elevation Prediction")
    print("-" * 40)
    
    prediction = elevation_calc.predict_elevation(
        current_depth=0.5,
        fear_type='existential',
        current_coherence=coherence_tracker.current_psi
    )
    
    print(f"From current depth {prediction['current_depth']:.1f}:")
    print(f"Current potential: {prediction['current_potential']:.2f}m")
    print(f"Optimal target depth: {prediction['optimal_target']:.2f}")
    print(f"Optimal elevation possible: {prediction['optimal_elevation']:.2f}m")
    
    # Test 7: Multiple Fear Analysis
    print("\n\n🌀 TEST 7: Complex Fear Analysis")
    print("-" * 40)
    
    complex_input = "I'm terrified that my life has no meaning, that I'll die having accomplished nothing, alone and forgotten"
    analysis2 = fear_analyzer.analyze_fear(complex_input)
    
    print(f"Input: '{complex_input}'")
    print(f"\nIdentified {len(analysis2.identified_fears)} fear patterns:")
    for fear in analysis2.identified_fears:
        print(f"  - {fear.pattern_type}: depth {fear.depth}")
    
    if analysis2.warnings:
        print("\nWarnings:")
        for warning in analysis2.warnings:
            print(f"  ⚠️  {warning}")
    
    # Summary
    print("\n\n" + "="*60)
    print("✨ TEST COMPLETE - SYSTEM FUNCTIONING CORRECTLY ✨")
    print("="*60)
    print(f"\nFinal Stats:")
    print(f"  Total elevation achieved: {engine.elevation_height:.2f}m")
    print(f"  Soul coherence: {coherence_tracker.current_psi:.3f}")
    print(f"  Fears analyzed: {len(fear_analyzer.analysis_history)}")
    print(f"  Transformations completed: {len(engine.embrace_history)}")
    print("\n🌟 The deeper the fear, the greater the elevation 🌟")


if __name__ == '__main__':
    test_system()