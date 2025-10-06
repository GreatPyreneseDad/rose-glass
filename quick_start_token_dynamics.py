"""
Quick Start: Token Dynamics in Rose Glass & GCT
==============================================

Simple example showing how to use the new d/tokens features
for real-time coherence tracking and response calibration.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src/core'))

from coherence_temporal_dynamics import CoherenceTemporalDynamics
from adaptive_response_system import AdaptiveResponseSystem
from rose_glass_lens import RoseGlass

# Simple example conversation
def analyze_conversation():
    # Initialize components
    dynamics = CoherenceTemporalDynamics()
    rose_glass = RoseGlass()
    response_system = AdaptiveResponseSystem()
    
    print("=== Token Dynamics Quick Start ===\n")
    
    # Simulate a conversation
    messages = [
        ("I'm feeling overwhelmed by all this new technology", "user"),
        ("I understand. Let's take it one step at a time", "assistant"),
        ("But there's so much to learn and I don't know where to start", "user"),
        ("We can begin with what interests you most. What draws your attention?", "assistant"),
    ]
    
    for message, speaker in messages:
        # Extract coherence (simplified)
        word_count = len(message.split())
        coherence = 1.5 + (0.1 * word_count) - (0.05 * message.count('!'))  # Simple heuristic
        
        # Track the message
        dynamics.add_reading(
            coherence=coherence,
            message=message,
            speaker=speaker
        )
        
        # Get dynamics
        analysis = dynamics.calculate_dual_derivatives()
        
        print(f"{speaker}: {message}")
        print(f"  Coherence: {coherence:.2f}")
        print(f"  Flow rate: {analysis['flow_rate']:.1f} tokens/sec")
        print(f"  Pattern: {analysis['interpretation']}")
        
        # If user message, calibrate response
        if speaker == "user":
            calibration = response_system.calibrate_response_length(
                coherence_state=coherence,
                dC_dtokens=analysis['dC_dtokens'],
                flow_rate=analysis['flow_rate']
            )
            print(f"  → Next response: {calibration.target_tokens} tokens, "
                  f"{calibration.pacing.value} pacing")
        print()
    
    # Final summary
    summary = dynamics.get_summary_stats()
    print("\nConversation Summary:")
    print(f"  Average coherence: {summary['average_coherence']:.2f}")
    print(f"  Token derivative: {summary['dC_dtokens']:.4f}")
    print(f"  Interpretation: {summary['interpretation']}")

if __name__ == "__main__":
    analyze_conversation()
"""