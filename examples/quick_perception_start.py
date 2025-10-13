"""
Quick Start Guide for Rose Glass Perceptual Framework
Simple examples to get started immediately
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from perception import RoseGlassPerception


# Example 1: Basic perception
print("Example 1: Basic Perception")
print("-" * 40)

perception = RoseGlassPerception()

text = "We must work together to solve this urgent problem!"
result = perception.perceive(text)

print(f"Text: {text}")
print(f"Detected patterns:")
print(f"  - Internal consistency: {result.dimensions.psi:.2f}")
print(f"  - Wisdom depth: {result.dimensions.rho:.2f}")
print(f"  - Emotional activation: {result.dimensions.q:.2f}")
print(f"  - Social belonging: {result.dimensions.f:.2f}")
print(f"  - Communication pace: {result.rhythm['pace']}")


# Example 2: Calibrating a response
print("\n\nExample 2: Response Calibration")
print("-" * 40)

# High urgency message
urgent_text = "Emergency! Need help immediately! System is down!"
urgent_result = perception.perceive(urgent_text)

# Draft a response
draft_response = """
I understand you're experiencing an issue with the system. 
Let me walk you through a comprehensive troubleshooting process
that will help us identify the root cause of this problem.
"""

# Calibrate for urgency
calibrated = perception.calibrate_response(urgent_result, draft_response)
print(f"User message: {urgent_text}")
print(f"\nOriginal response: {draft_response.strip()}")
print(f"\nCalibrated response: {calibrated.strip()}")
print("(Notice how the response is shortened to match urgency)")


# Example 3: Cultural lens comparison
print("\n\nExample 3: Cultural Lens Comparison")
print("-" * 40)

philosophical_text = """
The nature of truth reveals itself not through assertion but through
patient inquiry. What seems contradictory often points to deeper unity.
"""

print(f"Text: {philosophical_text.strip()}\n")

# Try different cultural lenses
lenses = ['modern_western_academic', 'buddhist_contemplative', 'medieval_islamic']

for lens in lenses:
    perception.calibration.default_calibration = lens
    result = perception.perceive(philosophical_text)
    coherence = (result.dimensions.psi + 
                result.dimensions.rho * result.dimensions.psi +
                result.dimensions.q +
                result.dimensions.f * result.dimensions.psi)
    print(f"{lens}: Coherence pattern = {coherence:.2f}")


# Example 4: Detecting uncertainty
print("\n\nExample 4: Uncertainty Detection")
print("-" * 40)

ambiguous_text = """
The data clearly shows... but I feel in my heart that...
We need rigorous analysis! Yet intuition matters too!
"""

result = perception.perceive(ambiguous_text)

print(f"Text: {ambiguous_text.strip()}")
print(f"\nUncertainty level: {result.uncertainty_level:.2f}")
if result.uncertainty_level > 0.5:
    print("High uncertainty detected - mixed signals between logic and emotion")
    print("Recommendation: Hold multiple interpretations rather than forcing one")


# Example 5: Simple pattern memory
print("\n\nExample 5: Pattern Memory")
print("-" * 40)

messages = [
    "I'm confused...",
    "Starting to understand",
    "Oh, now I see!",
    "This makes perfect sense!"
]

print("Tracking understanding evolution:")
for i, msg in enumerate(messages):
    result = perception.perceive(msg)
    print(f"{i+1}. \"{msg}\" - Coherence: {result.dimensions.psi:.2f}")

summary = perception.pattern_memory.get_pattern_summary()
print(f"\nCoherence improved from {summary['patterns']['psi']['min']:.2f} "
      f"to {summary['patterns']['psi']['max']:.2f}")


print("\n\n" + "=" * 50)
print("Quick Start Complete!")
print("For more detailed examples, see perceptual_framework_demo.py")