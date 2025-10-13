"""
Tests for Phase 2: Internal Processing Only
Validates that perception happens without explicit reporting
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from perception import (
    RoseGlassPerceptionV2,
    create_phase2_perception,
    PerceptionPhase,
    DimensionalPattern,
    Perception
)


class TestPhase2InternalProcessing:
    """Test that Phase 2 operates internally without explicit reporting"""
    
    def test_no_explicit_perception_in_response(self):
        """Ensure responses contain no perception metrics"""
        perception = create_phase2_perception()
        
        # High urgency text
        urgent_text = "This is an emergency! We need help immediately!"
        result = perception.perceive(urgent_text)
        
        # Draft response
        draft = "I understand your concern. Let me think about this carefully."
        
        # Calibrate response
        calibrated = perception.calibrate_response(result, draft)
        
        # Should NOT contain any perception language
        forbidden_terms = ['Ψ', 'psi', 'rho', 'coherence', 'dimension', 
                          'perceiving', 'calibration', 'lens']
        
        for term in forbidden_terms:
            assert term not in calibrated.lower()
            
        # Should be adapted for urgency though
        assert len(calibrated) < len(draft)  # Shortened
        
    def test_implicit_rhythm_matching(self):
        """Test that rhythm is matched without explanation"""
        perception = RoseGlassPerceptionV2()
        
        # Staccato input
        staccato = "Quick! Fast! Now! Help!"
        staccato_result = perception.perceive(staccato)
        
        # Long draft response
        long_draft = """
        I understand you need assistance with this matter. Let me provide you with
        a comprehensive analysis of the situation and outline several potential 
        approaches we might consider exploring together.
        """
        
        adapted = perception.calibrate_response(staccato_result, long_draft)
        
        # Should be shortened but not explain why
        assert len(adapted) < len(long_draft)
        assert 'rhythm' not in adapted.lower()
        assert 'pace' not in adapted.lower()
        assert 'matching' not in adapted.lower()
        
    def test_implicit_collective_adaptation(self):
        """Test shifting to collective frame without explanation"""
        perception = RoseGlassPerceptionV2()
        
        # Collective focused input
        collective_text = "We must work together as a community to solve our shared problems!"
        collective_result = perception.perceive(collective_text)
        
        # Individual focused draft
        draft = "I think you should consider this approach. I recommend you take action."
        
        adapted = perception.calibrate_response(collective_result, draft)
        
        # Should shift to collective language
        assert 'we' in adapted.lower()
        assert adapted.count('I ') < draft.count('I ')
        
        # But shouldn't explain the shift
        assert 'collective' not in adapted.lower()
        assert 'social' not in adapted.lower()
        
    def test_natural_uncertainty_handling(self):
        """Test uncertainty handling without meta-commentary"""
        perception = RoseGlassPerceptionV2()
        
        # Ambiguous text
        ambiguous = """
        The data clearly shows... but I feel in my heart...
        Logic says one thing, intuition says another!
        """
        
        result = perception.perceive(ambiguous)
        draft = "Here's my analysis of your situation."
        
        adapted = perception.calibrate_response(result, draft)
        
        # Should add checking but not explain uncertainty
        assert '?' in adapted  # Should have a question
        assert 'uncertainty' not in adapted.lower()
        assert 'interpretation' not in adapted.lower()
        assert 'ambiguous' not in adapted.lower()


class TestResponseAdaptation:
    """Test the implicit response adaptation system"""
    
    def test_urgency_adaptation(self):
        """Test adaptation for high moral activation"""
        perception = RoseGlassPerceptionV2()
        
        # Create high urgency perception
        urgent_perception = Perception(
            dimensions=DimensionalPattern(psi=0.7, rho=0.5, q=0.9, f=0.6),
            rhythm={'pace': 'rapid'},
            calibration='default'
        )
        
        # Hedging draft
        draft = "Perhaps we might possibly consider maybe taking some action."
        
        adapted = perception.response_adapter.adapt_response(draft, urgent_perception)
        
        # Should remove hedges
        assert 'perhaps' not in adapted.lower()
        assert 'might possibly' not in adapted.lower()
        assert 'maybe' not in adapted.lower()
        
    def test_contemplative_adaptation(self):
        """Test adaptation for contemplative communication"""
        perception = RoseGlassPerceptionV2()
        
        # Create contemplative perception
        contemplative_perception = Perception(
            dimensions=DimensionalPattern(psi=0.8, rho=0.85, q=0.3, f=0.5),
            rhythm={'pace': 'contemplative'},
            calibration='default'
        )
        
        draft = "Here is the answer to your question."
        
        adapted = perception.response_adapter.adapt_response(draft, contemplative_perception)
        
        # Should add natural pauses or depth
        assert '...' in adapted or 'consider' in adapted.lower() or 'perspective' in adapted.lower()
        
    def test_low_coherence_adaptation(self):
        """Test adding structure for low coherence"""
        perception = RoseGlassPerceptionV2()
        
        # Low coherence perception
        low_coherence_perception = Perception(
            dimensions=DimensionalPattern(psi=0.2, rho=0.5, q=0.5, f=0.5),
            rhythm={'pace': 'moderate'},
            calibration='default'
        )
        
        # Unstructured draft
        draft = "This is one point. This is another point. This is a third point."
        
        adapted = perception.response_adapter.adapt_response(draft, low_coherence_perception)
        
        # Should add structure
        structural_markers = ['First', 'Next', 'Also', 'Finally']
        assert any(marker in adapted for marker in structural_markers)


class TestPhase2Integration:
    """Integration tests for Phase 2 functionality"""
    
    def test_perceive_and_respond_flow(self):
        """Test the full perceive and respond flow"""
        perception = RoseGlassPerceptionV2()
        
        user_input = "We urgently need to fix this bug in production!"
        
        def response_generator():
            return "I will analyze the bug and provide recommendations."
        
        final_response = perception.perceive_and_respond(user_input, response_generator)
        
        # Should be adapted but not contain perception language
        assert 'analyze' in final_response or 'help' in final_response
        assert 'perception' not in final_response.lower()
        assert len(final_response) > 0
        
    def test_conversation_flow_tracking(self):
        """Test tracking conversation flow without explicit reporting"""
        perception = RoseGlassPerceptionV2()
        
        messages = [
            "I'm confused about this.",
            "Can you explain more?",
            "Oh, I think I understand now.",
            "Yes, that makes sense!"
        ]
        
        responses = []
        
        for msg in messages:
            base = "Here's more information for you."
            adapted = perception.create_contextual_response(msg, base)
            responses.append(adapted)
            
        # Analyze flow
        flow_analysis = perception.handle_conversation_flow(messages, responses)
        
        assert flow_analysis['coherence_improving'] == True
        
        # But this analysis should never appear in actual responses
        for response in responses:
            assert 'coherence' not in response.lower()
            assert 'improving' not in response.lower()
            
    def test_cultural_calibration_implicit(self):
        """Test that cultural calibration happens implicitly"""
        perception = RoseGlassPerceptionV2()
        
        # Technical text
        tech_text = "The API endpoint returns a JSON object with status codes."
        perception.calibration.default_calibration = 'technical_documentation'
        
        tech_result = perception.perceive(tech_text)
        
        draft = "This might possibly work if you try it."
        adapted = perception.calibrate_response(tech_result, draft)
        
        # Should be more precise but not mention calibration
        assert 'might possibly' not in adapted
        assert 'calibration' not in adapted.lower()
        assert 'technical' not in adapted.lower() or 'technical' in draft.lower()


class TestPhase2Behaviors:
    """Test specific Phase 2 behaviors"""
    
    def test_no_diagnostic_output_in_production(self):
        """Ensure diagnostic methods are not used in responses"""
        perception = RoseGlassPerceptionV2()
        
        text = "Help me understand this concept."
        result = perception.perceive(text)
        
        # This method exists for debugging
        summary = perception.get_adaptation_summary(result)
        
        # But should never be included in actual response
        draft = "Let me explain this concept."
        adapted = perception.calibrate_response(result, draft)
        
        # Response should not contain any diagnostic information
        assert str(summary) not in adapted
        assert 'strategy' not in adapted.lower()
        assert 'dimensional_focus' not in adapted.lower()
        
    def test_gentle_checking_when_uncertain(self):
        """Test that uncertainty prompts gentle checking"""
        perception = RoseGlassPerceptionV2()
        
        # Create high uncertainty perception
        uncertain_perception = Perception(
            dimensions=DimensionalPattern(psi=0.5, rho=0.5, q=0.5, f=0.5),
            rhythm={'pace': 'moderate'},
            calibration='default',
            uncertainty_level=0.7
        )
        
        draft = "This is my understanding of the situation."
        adapted = perception.calibrate_response(uncertain_perception, draft)
        
        # Should add gentle check
        assert '?' in adapted
        assert any(phrase in adapted.lower() for phrase in 
                  ['does this', 'is this', 'resonate', 'capture'])
        
        # But not explain why checking
        assert 'uncertain' not in adapted.lower()
        assert 'checking' not in adapted.lower()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])