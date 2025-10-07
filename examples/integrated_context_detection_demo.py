#!/usr/bin/env python3
"""
Integrated Context Detection Demo
=================================

Demonstrates how the four critical context detectors work together
to properly calibrate AI responses based on user message context.

This shows the resolution of the 10% of cases where high coherence
alone doesn't provide proper response calibration.

Author: Christopher MacGregor bin Joseph
Date: October 2025
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.adaptive_response_system import AdaptiveResponseSystem
from src.core.coherence_temporal_dynamics import CoherenceTemporalDynamics


def demonstrate_context_detection():
    """Demonstrate all four context detectors in action"""
    
    # Initialize systems
    response_system = AdaptiveResponseSystem()
    temporal_dynamics = CoherenceTemporalDynamics()
    
    # Test scenarios representing the critical 10%
    test_scenarios = [
        # 1. Trust Signal - Brief high-coherence message
        {
            'message': "Trust me on this",
            'coherence': 3.2,
            'description': "Trust Signal Detection",
            'expected_mode': 'trust'
        },
        
        # 2. Mission Mode - Research request
        {
            'message': "Research the impact of quantum computing on cryptography",
            'coherence': 2.5,
            'description': "Mission Mode Detection",
            'expected_mode': 'mission'
        },
        
        # 3. Essence Request - Summary needed
        {
            'message': "Can you summarize the key points of our discussion?",
            'coherence': 2.8,
            'description': "Essence Request Detection",
            'expected_mode': 'essence'
        },
        
        # 4. Crisis with high coherence (shouldn't happen but handle gracefully)
        {
            'message': "I need help understanding this concept that's been overwhelming me",
            'coherence': 0.8,
            'description': "Crisis Override",
            'expected_mode': 'crisis',
            'conversation_state': {'crisis_detected': True}
        },
        
        # 5. Poetic trust signal
        {
            'message': "Through the rose glass, all becomes clear...",
            'coherence': 3.8,
            'description': "Poetic Trust Signal",
            'expected_mode': 'trust'
        },
        
        # 6. Complex mission with structure
        {
            'message': "Analyze and compare the different approaches to machine learning, including supervised, unsupervised, and reinforcement learning. Please organize this step-by-step.",
            'coherence': 2.2,
            'description': "Structured Mission Mode",
            'expected_mode': 'mission'
        },
        
        # 7. TL;DR request
        {
            'message': "TL;DR?",
            'coherence': 2.0,
            'description': "Ultra-brief Essence Request",
            'expected_mode': 'essence'
        },
        
        # 8. Information overload scenario
        {
            'message': "ok",
            'coherence': 1.5,
            'description': "Information Overload Response",
            'expected_mode': 'crisis',
            'conversation_state': {'information_overload': True}
        }
    ]
    
    print("=" * 80)
    print("INTEGRATED CONTEXT DETECTION DEMONSTRATION")
    print("Showing how the four detectors resolve the critical 10% of cases")
    print("=" * 80)
    print()
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n{i}. {scenario['description']}")
        print("-" * 60)
        print(f"Message: \"{scenario['message']}\"")
        print(f"Coherence: {scenario['coherence']}")
        
        # Set up conversation state
        conversation_state = scenario.get('conversation_state', {})
        
        # Simulate some temporal dynamics
        temporal_dynamics.add_reading(
            scenario['coherence'],
            scenario['message'],
            'user'
        )
        
        derivatives = temporal_dynamics.calculate_dual_derivatives()
        
        # Run context detection
        calibration, context = response_system.calibrate_with_context(
            message=scenario['message'],
            coherence=scenario['coherence'],
            dC_dtokens=derivatives['dC_dtokens'],
            flow_rate=derivatives['flow_rate'],
            conversation_state=conversation_state
        )
        
        print(f"\nContext Detection Results:")
        print(f"  Primary Mode: {context['primary_mode']}")
        print(f"  Expected Mode: {scenario['expected_mode']}")
        print(f"  Match: {'✅' if context['primary_mode'] == scenario['expected_mode'] else '❌'}")
        
        print(f"\nDetections:")
        print(f"  Trust Signal: {context['detections']['trust_detected']}")
        print(f"  Mission Mode: {context['detections']['mission_detected']}")
        print(f"  Essence Request: {context['detections']['essence_detected']}")
        
        print(f"\nCalibration:")
        print(f"  Target Tokens: {calibration.target_tokens}")
        print(f"  Pacing: {calibration.pacing.value}")
        print(f"  Complexity: {calibration.complexity_level.value}")
        print(f"  Token Limit Mode: {context['token_limit'].mode.value}")
        
        # Show specific detector details
        if context['trust_signal']:
            print(f"\nTrust Signal Details:")
            print(f"  Type: {context['trust_signal'].signal_type.value}")
            print(f"  Confidence: {context['trust_signal'].confidence:.2f}")
            
        if context['mission']:
            print(f"\nMission Details:")
            print(f"  Type: {context['mission'].mission_type.value}")
            print(f"  Scope: {context['mission'].scope}")
            print(f"  Estimated Tokens: {context['mission'].estimated_tokens}")
            
        if context['essence_request']:
            print(f"\nEssence Request Details:")
            print(f"  Type: {context['essence_request'].essence_type.value}")
            print(f"  Target Length: {context['essence_request'].target_length}")
            print(f"  Format: {context['essence_request'].format_preference}")
    
    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
    
    # Show summary statistics
    print("\nSummary:")
    print(f"Total Scenarios: {len(test_scenarios)}")
    print(f"These represent the ~10% of cases where coherence alone is insufficient")
    print(f"Each detector addresses specific edge cases that would otherwise")
    print(f"result in inappropriate response calibration.")
    
    print("\nKey Insights:")
    print("1. Trust signals override coherence-based calibration")
    print("2. Mission mode provides structure regardless of coherence")
    print("3. Essence requests enforce brevity even at high coherence")
    print("4. Crisis/overload always takes precedence for safety")


def demonstrate_token_limits():
    """Demonstrate token multiplier limits in action"""
    print("\n\n" + "=" * 80)
    print("TOKEN MULTIPLIER LIMITS DEMONSTRATION")
    print("=" * 80)
    
    response_system = AdaptiveResponseSystem()
    
    scenarios = [
        ("Hello!", 3.0, "Standard greeting"),
        ("I can't handle this anymore" * 5, 0.5, "Crisis message"),
        ("Explain quantum computing in detail", 2.5, "Complex request"),
        ("Trust me, this is important", 3.5, "Trust signal"),
        ("Research machine learning algorithms", 2.0, "Mission mode"),
        ("TL;DR of our conversation?", 2.2, "Essence request")
    ]
    
    for message, coherence, desc in scenarios:
        context = response_system.detect_context(
            message, coherence, {}
        )
        
        print(f"\n{desc}:")
        print(f"  User tokens: {context['user_tokens']}")
        print(f"  Raw multiplier: {context['token_limit'].raw_multiplier:.1f}x")
        print(f"  Adjusted multiplier: {context['token_limit'].adjusted_multiplier:.1f}x")
        print(f"  Token limit: {context['token_limit'].token_limit}")
        print(f"  Mode: {context['token_limit'].mode.value}")
        if context['token_limit'].adjustments:
            print(f"  Adjustments: {', '.join(context['token_limit'].adjustments)}")


if __name__ == "__main__":
    demonstrate_context_detection()
    demonstrate_token_limits()
    
    print("\n✨ Context detection enables nuanced response calibration")
    print("   beyond what coherence metrics alone can provide.")
"""