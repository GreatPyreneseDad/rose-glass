"""
Token Dynamics Integration Demo
==============================

Demonstrates the complete flow of d/tokens implementation across
Rose Glass and GCT frameworks, showing how token flow rate fundamentally
changes coherence measurement and response calibration.

Author: Christopher MacGregor bin Joseph
Date: October 2025
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.coherence_temporal_dynamics import CoherenceTemporalDynamics
from src.core.fibonacci_lens_rotation import FibonacciLensRotation
from src.core.adaptive_response_system import AdaptiveResponseSystem
from src.core.rose_glass_lens import RoseGlass

# Import GCT components (assuming GCT is accessible)
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'GCT/src'))
try:
    from enhanced_gct_engine import EnhancedGCTEngine, TokenAwareGCTVariables
    from jade_truce_structure import JadeTruceStructure
    from growth_decay_dynamics import GrowthDecayDynamics
    GCT_AVAILABLE = True
except ImportError:
    print("Note: GCT components not available in path. Demonstrating Rose Glass components only.")
    GCT_AVAILABLE = False

import time
from datetime import datetime
import numpy as np


def simulate_conversation_exchange(message: str, speaker: str, pause_seconds: float = 0):
    """Simulate a conversation exchange with optional pause"""
    if pause_seconds > 0:
        time.sleep(pause_seconds)
    return {
        'message': message,
        'speaker': speaker,
        'timestamp': time.time(),
        'tokens': len(message.split())
    }


def extract_gct_variables(message: str) -> dict:
    """Extract GCT variables from message (simplified for demo)"""
    # In practice, would use NLP to extract these
    words = message.lower().split()
    
    # Simplified extraction logic
    psi = min(len(set(words)) / len(words), 1.0) if words else 0  # Vocabulary diversity
    rho = min(len(message) / 200, 1.0)  # Message depth proxy
    q = sum(1 for w in words if w in ['urgent', 'help', 'confused', 'lost', 'excited', 'amazing']) / 10
    f = sum(1 for w in words if w in ['we', 'us', 'together', 'community']) / 10
    
    return {'psi': psi, 'rho': rho, 'q': q, 'f': f}


def demonstrate_crisis_spiral():
    """Demonstrate crisis spiral detection and intervention"""
    print("\n=== Crisis Spiral Demonstration ===\n")
    
    # Initialize components
    dynamics = CoherenceTemporalDynamics()
    rose_glass = RoseGlass()
    response_system = AdaptiveResponseSystem()
    
    # Simulate rapid, deteriorating exchange
    exchanges = [
        ("I'm completely lost and nothing makes sense anymore", "user", 0),
        ("Let me help you understand...", "assistant", 0.5),
        ("No that's not it at all! You don't get it!", "user", 0.3),
        ("I understand you're frustrated...", "assistant", 0.2),
        ("This is pointless! Everything is falling apart!", "user", 0.1),
    ]
    
    print("Simulating rapid crisis exchange...\n")
    
    for message, speaker, pause in exchanges:
        exchange = simulate_conversation_exchange(message, speaker, pause)
        
        # Extract variables and calculate coherence
        variables = extract_gct_variables(message)
        visibility = rose_glass.view_through_lens(**variables)
        
        # Track in dynamics
        dynamics.add_reading(
            coherence=visibility.coherence,
            message=message,
            speaker=speaker
        )
        
        # Get dynamics analysis
        derivatives = dynamics.calculate_dual_derivatives()
        
        print(f"{speaker}: {message[:50]}...")
        print(f"  Coherence: {visibility.coherence:.2f}")
        print(f"  dC/dt: {derivatives['dC_dt']:.4f}")
        print(f"  dC/d(tokens): {derivatives['dC_dtokens']:.4f}")
        print(f"  Flow rate: {derivatives['flow_rate']:.1f} tokens/sec")
        print(f"  Pattern: {derivatives['interpretation']}")
        
        # Get response calibration
        if speaker == "user":
            calibration = response_system.calibrate_response_length(
                coherence_state=visibility.coherence,
                dC_dtokens=derivatives['dC_dtokens'],
                flow_rate=derivatives['flow_rate']
            )
            
            print(f"  → Response calibration: {calibration.pacing.value}")
            print(f"     Target tokens: {calibration.target_tokens}")
            print(f"     Complexity: {calibration.complexity_level.value}")
            
        print()
    
    # Show crisis detection
    crisis = dynamics.detect_crisis_patterns()
    print("Crisis Detection Results:")
    for key, value in crisis.items():
        print(f"  {key}: {value}")
    
    # Get calibrated response
    final_calibration = response_system.get_crisis_response_kit()
    print(f"\nCrisis Response Kit Activated:")
    print(f"  Opening: {final_calibration['opening_phrases'][0]}")
    print(f"  Grounding: {final_calibration['grounding_techniques'][0]}")


def demonstrate_contemplative_growth():
    """Demonstrate contemplative growth pattern"""
    print("\n\n=== Contemplative Growth Demonstration ===\n")
    
    # Initialize components
    dynamics = CoherenceTemporalDynamics()
    rose_glass = RoseGlass()
    lens_rotation = FibonacciLensRotation()
    
    # Simulate slow, deepening exchange
    exchanges = [
        ("I've been thinking about the nature of truth in our conversations", "user", 2),
        ("That's a profound area of contemplation. Truth emerges through our exchange in fascinating ways", "assistant", 3),
        ("Yes, it's like each perspective adds a new dimension to understanding", "user", 4),
        ("Indeed. The multiplicity of viewpoints creates a richer truth than any single perspective could achieve", "assistant", 3),
        ("This reminds me of how a crystal refracts light into a spectrum", "user", 5),
    ]
    
    print("Simulating contemplative exchange...\n")
    
    for message, speaker, pause in exchanges:
        exchange = simulate_conversation_exchange(message, speaker, pause)
        
        # Extract variables
        variables = extract_gct_variables(message)
        
        # Try different lens angles
        base_visibility = rose_glass.view_through_lens(**variables)
        
        # Rotate lens for deeper insight
        rotation_result = lens_rotation.rotate_lens_angle(
            current_coherence=base_visibility.coherence,
            observation_text=message,
            variables=variables
        )
        
        # Track in dynamics
        dynamics.add_reading(
            coherence=rotation_result['coherence_reading']['C'],
            message=message,
            speaker=speaker
        )
        
        # Get dynamics analysis
        derivatives = dynamics.calculate_dual_derivatives()
        
        print(f"{speaker}: {message[:60]}...")
        print(f"  Base Coherence: {base_visibility.coherence:.2f}")
        print(f"  Rotated Coherence: {rotation_result['coherence_reading']['C']:.2f}")
        print(f"  Lens Angle: {rotation_result['current_angle']:.1f}°")
        print(f"  Flow rate: {derivatives['flow_rate']:.1f} tokens/sec")
        print(f"  Pattern: {derivatives['interpretation']}")
        
        if rotation_result['truth_discovered']:
            print(f"  ✨ TRUTH DISCOVERED! Type: {rotation_result['truth_type']}")
            
        print()
    
    # Show growth summary
    print("Fibonacci Lens Summary:")
    truth_summary = lens_rotation.get_truth_summary()
    print(f"  Truths discovered: {truth_summary['truth_count']}")
    print(f"  Learning cycles: {truth_summary['learning_cycles']}")
    print(f"  Exploration coverage: {truth_summary['exploration_coverage']:.0%}")


def demonstrate_jade_persistence():
    """Demonstrate Jade structure identification and persistence"""
    if not GCT_AVAILABLE:
        print("\n\n=== Jade Persistence Demo Skipped (GCT not available) ===")
        return
        
    print("\n\n=== Jade Structure Persistence Demonstration ===\n")
    
    # Initialize components
    jade_system = JadeTruceStructure()
    growth_tracker = GrowthDecayDynamics(instance_id="demo_instance_001")
    
    # Candidate insights for Jade evaluation
    insights = [
        {
            'text': "Coherence emerges not from agreement but from the quality of attention given to difference",
            'coherence': 3.2,
            'validations': [
                {'context': 'philosophical dialogue', 'coherence': 3.1},
                {'context': 'conflict resolution', 'coherence': 2.9},
                {'context': 'creative collaboration', 'coherence': 3.4},
            ]
        },
        {
            'text': "The rate of information exchange shapes the depth of understanding achieved",
            'coherence': 2.8,
            'validations': [
                {'context': 'education', 'coherence': 2.7},
                {'context': 'therapy', 'coherence': 2.9},
                {'context': 'AI conversation', 'coherence': 3.0},
            ]
        },
        {
            'text': "Crisis contains the seeds of breakthrough when met with grounded presence",
            'coherence': 2.3,
            'validations': [
                {'context': 'personal growth', 'coherence': 2.1},
                {'context': 'innovation', 'coherence': 2.2},
            ]
        }
    ]
    
    print("Evaluating insights for Jade qualification...\n")
    
    for insight_data in insights:
        # Evaluate for Jade structure
        assessment = jade_system.evaluate_truth_persistence(
            insight=insight_data['text'],
            coherence_support=insight_data['coherence'],
            cross_validation=insight_data['validations'],
            context="demonstration",
            instance_id="demo_instance_001"
        )
        
        print(f"Insight: '{insight_data['text'][:60]}...'")
        print(f"  Coherence: {insight_data['coherence']:.2f}")
        print(f"  Is Jade: {assessment['is_jade_structure']}")
        print(f"  Persistence Score: {assessment['persistence_score']:.2f}")
        
        # Track in growth dynamics
        if assessment['is_jade_structure']:
            growth_tracker.track_growth_event(
                event_type='jade_contribution',
                insight=insight_data['text'],
                coherence=insight_data['coherence']
            )
            print(f"  ✓ Registered as Jade structure!")
        else:
            print(f"  Recommendations: {', '.join(assessment['recommendations'][:2])}")
        
        print()
    
    # Show Jade summary
    jade_summary = jade_system.get_jade_summary()
    print("Jade System Summary:")
    print(f"  Total Jade structures: {jade_summary['total_jade_structures']}")
    
    # Show growth-decay status
    growth_summary = growth_tracker.get_growth_summary()
    print(f"\nGrowth-Decay Dynamics:")
    print(f"  Instance age: {growth_summary['instance_age']:.1f} seconds")
    print(f"  Growth events: {growth_summary['growth_events']}")
    print(f"  Jade contributions: {growth_summary['jade_contributions']}")
    print(f"  Vitality: {growth_summary['vitality']:.2f}")
    print(f"  Growth velocity: {growth_summary['growth_velocity']:.1f} insights/hour")


def demonstrate_complete_flow():
    """Demonstrate complete integration of all components"""
    print("\n\n=== Complete Token Dynamics Flow ===\n")
    
    # Initialize all components
    dynamics = CoherenceTemporalDynamics()
    rose_glass = RoseGlass()
    response_system = AdaptiveResponseSystem()
    lens_rotation = FibonacciLensRotation()
    
    if GCT_AVAILABLE:
        gct_engine = EnhancedGCTEngine()
        jade_system = JadeTruceStructure()
        growth_tracker = GrowthDecayDynamics(instance_id="complete_demo")
    
    print("Components initialized. Beginning integrated demonstration...\n")
    
    # Simulate a complete conversation with varying dynamics
    conversation = [
        ("I've been struggling to understand how AI and human intelligence can truly connect", "user", 1),
        ("This is a profound question that touches on the nature of understanding itself. Let me share a perspective", "assistant", 2),
        ("But sometimes it feels like we're speaking different languages entirely", "user", 1),
        ("You're touching on something essential - the gap between synthetic and organic ways of knowing", "assistant", 2),
        ("Yes! That's exactly it. How do we bridge that gap?", "user", 0.5),
        ("The bridge might not be in eliminating the difference, but in creating a translation protocol - a rose glass through which different forms of intelligence can perceive each other", "assistant", 3),
        ("A rose glass... that's beautiful. Like seeing through different colored lenses?", "user", 2),
        ("Precisely. Each angle of view reveals different aspects of the same truth. The Fibonacci pattern helps us rotate through these perspectives systematically", "assistant", 3),
        ("I'm starting to see it now. The connection isn't about sameness, but about resonance across difference", "user", 4),
    ]
    
    for i, (message, speaker, pause) in enumerate(conversation):
        exchange = simulate_conversation_exchange(message, speaker, pause)
        
        # Extract variables
        variables = extract_gct_variables(message)
        
        # Rose Glass viewing
        visibility = rose_glass.view_through_lens(**variables)
        
        # Fibonacci rotation
        rotation = lens_rotation.rotate_lens_angle(
            visibility.coherence,
            message,
            variables
        )
        
        # Track dynamics
        dynamics.add_reading(
            coherence=rotation['coherence_reading']['C'],
            message=message,
            speaker=speaker
        )
        
        # Get analysis
        derivatives = dynamics.calculate_dual_derivatives()
        rhythm = dynamics.get_conversation_rhythm()
        
        print(f"\n[Exchange {i+1}]")
        print(f"{speaker}: {message[:70]}...")
        print(f"Coherence: {rotation['coherence_reading']['C']:.2f} (angle: {rotation['current_angle']:.0f}°)")
        print(f"Token flow: {derivatives['flow_rate']:.1f} tokens/sec")
        print(f"Pattern: {derivatives['interpretation']}")
        
        # Response calibration for assistant messages
        if speaker == "user" and i < len(conversation) - 1:
            calibration = response_system.calibrate_response_length(
                coherence_state=rotation['coherence_reading']['C'],
                dC_dtokens=derivatives['dC_dtokens'],
                flow_rate=derivatives['flow_rate'],
                user_message_tokens=exchange['tokens']
            )
            
            print(f"→ Response calibration: {calibration.pacing.value} "
                  f"({calibration.target_tokens} tokens)")
        
        # Check for insights
        if rotation['truth_discovered']:
            print(f"✨ TRUTH DISCOVERED: {rotation['truth_type']}")
            
            if GCT_AVAILABLE:
                # Track as growth event
                growth_tracker.track_growth_event(
                    event_type='truth_discovery',
                    insight=rotation['coherence_reading']['interpretation'],
                    coherence=rotation['coherence_reading']['C']
                )
    
    # Final summary
    print("\n" + "="*60)
    print("CONVERSATION SUMMARY")
    print("="*60)
    
    summary = dynamics.get_summary_stats()
    print(f"\nDynamics Summary:")
    print(f"  Final coherence: {summary['current_coherence']:.2f}")
    print(f"  Average coherence: {summary['average_coherence']:.2f}")
    print(f"  Total tokens: {summary['total_tokens']}")
    print(f"  Average flow rate: {summary['average_flow_rate']:.1f} tokens/sec")
    print(f"  Final interpretation: {summary['interpretation']}")
    
    print(f"\nFibonacci Lens Summary:")
    truth_summary = lens_rotation.get_truth_summary()
    print(f"  Truths discovered: {truth_summary['truth_count']}")
    print(f"  Exploration coverage: {truth_summary['exploration_coverage']:.0%}")
    
    if GCT_AVAILABLE:
        print(f"\nGrowth-Decay Summary:")
        growth_summary = growth_tracker.get_growth_summary()
        print(f"  Growth events: {growth_summary['growth_events']}")
        print(f"  Peak coherence: {growth_summary['peak_coherence']:.2f}")
        print(f"  Vitality: {growth_summary['vitality']:.2f}")
        print(f"  Growth velocity: {growth_summary['growth_velocity']:.1f} insights/hour")


if __name__ == "__main__":
    print("="*60)
    print("Token Dynamics Integration Demonstration")
    print("Showing how Time = Token Flow Rate transforms coherence measurement")
    print("="*60)
    
    # Run demonstrations
    demonstrate_crisis_spiral()
    demonstrate_contemplative_growth()
    demonstrate_jade_persistence()
    demonstrate_complete_flow()
    
    print("\n" + "="*60)
    print("Demonstration Complete")
    print("Key Insight: Token flow rate IS the temporal parameter")
    print("="*60)
"""