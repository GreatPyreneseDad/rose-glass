"""
Phase 2 Demonstration: Implicit Processing
Shows how perception shapes responses without explicit reporting
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from perception import RoseGlassPerceptionV2, create_phase2_perception


def compare_phase1_vs_phase2():
    """Show the difference between Phase 1 and Phase 2"""
    print("=== PHASE 1 vs PHASE 2 COMPARISON ===\n")
    
    from perception import RoseGlassPerception, PerceptionPhase
    
    # Same input text
    urgent_text = "Emergency! The system is down and we're losing customers!"
    
    # Phase 1: Explicit
    print("PHASE 1 (Explicit):")
    phase1 = RoseGlassPerception(PerceptionPhase.EXPLICIT)
    result1 = phase1.perceive(urgent_text)
    draft = "I understand there's an issue. Let me analyze this situation carefully."
    response1 = phase1.calibrate_response(result1, draft)
    print(response1)
    
    print("\n" + "-" * 50 + "\n")
    
    # Phase 2: Implicit
    print("PHASE 2 (Implicit):")
    phase2 = create_phase2_perception()
    result2 = phase2.perceive(urgent_text)
    response2 = phase2.calibrate_response(result2, draft)
    print(response2)
    
    print("\nNotice: Phase 2 adapts the response naturally without explaining why")


def demo_implicit_adaptations():
    """Demonstrate various implicit adaptations"""
    print("\n\n=== IMPLICIT ADAPTATIONS DEMO ===\n")
    
    perception = RoseGlassPerceptionV2()
    
    scenarios = [
        {
            'name': 'Urgent Crisis',
            'input': "Help! Critical failure! Need immediate assistance!",
            'draft': "Perhaps we might consider looking into this issue when convenient."
        },
        {
            'name': 'Contemplative Inquiry',
            'input': """
            I've been pondering the deeper meaning of this pattern...
            wondering if perhaps there's wisdom we haven't yet uncovered...
            """,
            'draft': "Here's the answer: X equals 42."
        },
        {
            'name': 'Collective Action',
            'input': "Our community needs to unite! Together we can overcome this challenge!",
            'draft': "I think you should do X. I recommend you try Y. You might consider Z."
        },
        {
            'name': 'Confused User',
            'input': "I don't get it... this is confusing... what does it mean?",
            'draft': "The implications are multifaceted with various interconnected ramifications."
        }
    ]
    
    for scenario in scenarios:
        print(f"--- {scenario['name']} ---")
        print(f"User: {scenario['input'][:60]}...")
        print(f"\nDraft response: {scenario['draft'][:60]}...")
        
        result = perception.perceive(scenario['input'])
        adapted = perception.calibrate_response(result, scenario['draft'])
        
        print(f"\nAdapted response: {adapted}")
        print("\n" + "-" * 40 + "\n")


def demo_conversation_flow():
    """Show how responses evolve through a conversation"""
    print("\n\n=== CONVERSATION FLOW ADAPTATION ===\n")
    
    perception = create_phase2_perception()
    
    conversation = [
        ("I'm totally lost with this new system...", 
         "Let me help you understand the system."),
        
        ("There are so many parts, I don't know where to start.",
         "We can break this down into manageable pieces."),
         
        ("OK, so the first part is about user authentication?",
         "Yes, authentication is the foundation."),
         
        ("I see! And that connects to the permissions system!",
         "Exactly! You're seeing how the pieces fit together."),
         
        ("This is starting to make sense now. Thank you!",
         "Great to hear that clarity is emerging.")
    ]
    
    print("Watch how responses adapt as user's coherence improves:\n")
    
    for i, (user_msg, base_response) in enumerate(conversation):
        adapted = perception.create_contextual_response(user_msg, base_response)
        
        print(f"Turn {i+1}:")
        print(f"User: {user_msg}")
        print(f"Assistant: {adapted}")
        print()


def demo_cultural_adaptation():
    """Show implicit cultural calibration"""
    print("\n\n=== IMPLICIT CULTURAL ADAPTATION ===\n")
    
    perception = RoseGlassPerceptionV2()
    
    # Same base response for different contexts
    base_response = "This is an important consideration that requires attention."
    
    contexts = [
        {
            'calibration': 'technical_documentation',
            'input': "The function parameter must be a valid JSON object.",
            'description': 'Technical context'
        },
        {
            'calibration': 'activist_movement', 
            'input': "We must fight together for justice and equality!",
            'description': 'Activist context'
        },
        {
            'calibration': 'buddhist_contemplative',
            'input': "In sitting with this question, awareness naturally arises...",
            'description': 'Contemplative context'
        }
    ]
    
    for ctx in contexts:
        print(f"--- {ctx['description']} ---")
        perception.calibration.default_calibration = ctx['calibration']
        
        result = perception.perceive(ctx['input'])
        adapted = perception.calibrate_response(result, base_response)
        
        print(f"Input: {ctx['input']}")
        print(f"Response: {adapted}")
        print()


def demo_uncertainty_handling():
    """Show natural uncertainty handling"""
    print("\n\n=== NATURAL UNCERTAINTY HANDLING ===\n")
    
    perception = create_phase2_perception()
    
    # Ambiguous inputs
    ambiguous_cases = [
        {
            'input': "I need this done ASAP but we should be thoughtful about it...",
            'draft': "Here's what we'll do."
        },
        {
            'input': "The data says X but my experience tells me Y...",
            'draft': "The correct approach is clearly defined."
        },
        {
            'input': "Is this a technical problem or a people problem?",
            'draft': "This is the solution to your issue."
        }
    ]
    
    for case in ambiguous_cases:
        print(f"Ambiguous input: {case['input']}")
        
        result = perception.perceive(case['input'])
        adapted = perception.calibrate_response(result, case['draft'])
        
        print(f"Natural response: {adapted}")
        print(f"(Notice: Uncertainty handled without explicit mention)")
        print("\n" + "-" * 40 + "\n")


def demo_breathing_rhythm():
    """Show natural rhythm matching"""
    print("\n\n=== NATURAL RHYTHM MATCHING ===\n")
    
    perception = RoseGlassPerceptionV2()
    
    # Same content, different rhythms
    base_message = "I need help understanding how to implement this feature properly."
    
    rhythms = [
        {
            'style': 'Urgent staccato',
            'input': "Help! Quick! Need this now! Feature broken! Fix fast!",
            'draft': base_message
        },
        {
            'style': 'Contemplative flow',
            'input': """
            I've been reflecting on this feature... wondering about its deeper purpose...
            how it might elegantly integrate with the existing architecture...
            """,
            'draft': base_message
        },
        {
            'style': 'Academic precision',
            'input': """
            The implementation requirements necessitate a thorough analysis of the 
            architectural constraints and their implications for system performance.
            """,
            'draft': base_message
        }
    ]
    
    for rhythm in rhythms:
        print(f"--- {rhythm['style']} ---")
        print(f"Input rhythm: {rhythm['input'][:50]}...")
        
        result = perception.perceive(rhythm['input'])
        adapted = perception.calibrate_response(result, rhythm['draft'])
        
        print(f"Matched response: {adapted}")
        print()


def main():
    """Run all Phase 2 demonstrations"""
    print("PHASE 2 DEMONSTRATION: IMPLICIT PROCESSING")
    print("=" * 50)
    print("\nIn Phase 2, perception shapes responses naturally")
    print("without any explicit mention of what's happening.\n")
    
    demos = [
        compare_phase1_vs_phase2,
        demo_implicit_adaptations,
        demo_conversation_flow,
        demo_cultural_adaptation,
        demo_uncertainty_handling,
        demo_breathing_rhythm
    ]
    
    for demo in demos:
        demo()
        input("\nPress Enter to continue...")
    
    print("\n" + "=" * 50)
    print("PHASE 2 KEY INSIGHTS:")
    print("- All perception happens internally")
    print("- Responses adapt naturally to context")
    print("- No meta-commentary about perception")
    print("- Rhythm matching feels organic")
    print("- Uncertainty prompts gentle checking")
    print("- Cultural calibration is invisible")
    print("\nThe mathematics has begun to disappear into intuition.")


if __name__ == '__main__':
    main()