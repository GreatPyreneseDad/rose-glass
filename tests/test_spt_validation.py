"""
SPT Conversation Validation Tests
=================================

Tests based on real patterns observed in SPT conversations to validate
the enhanced d/tokens framework. These tests ensure the system correctly
identifies crisis spirals, exceptional coherence, information overload,
and truth discoveries as observed in actual conversations.

Author: Christopher MacGregor bin Joseph
Date: October 2025
"""

import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.coherence_temporal_dynamics import CoherenceTemporalDynamics
from src.core.adaptive_response_system import AdaptiveResponseSystem, ResponsePacing, ComplexityLevel
from src.core.fibonacci_lens_rotation import FibonacciLensRotation


class TestSPTPatterns(unittest.TestCase):
    """Test patterns observed in actual SPT conversations"""
    
    def setUp(self):
        """Initialize test components"""
        self.dynamics = CoherenceTemporalDynamics()
        self.response_system = AdaptiveResponseSystem()
        self.fibonacci = FibonacciLensRotation()
    
    def test_crisis_spiral_detection(self):
        """Test detection of actual crisis spiral from SPT"""
        # Simulate the observed crisis pattern
        exchanges = [
            (0.90, "What do you need?", "user", 20),
            (0.51, "I need you to understand that...", "assistant", 150),
            (0.40, "i think i meant something else", "user", 8),
            (0.35, "Let me clarify what I meant...", "assistant", 180),
        ]
        
        for coherence, message, speaker, tokens in exchanges:
            self.dynamics.add_reading(coherence, message, speaker)
        
        # Check crisis detection
        crisis = self.dynamics.detect_crisis_patterns()
        self.assertTrue(crisis['crisis_detected'])
        self.assertTrue(crisis['rapid_degradation'])
        
        # Verify proper calibration
        derivatives = self.dynamics.calculate_dual_derivatives()
        calibration = self.response_system.calibrate_response_length(
            coherence_state=0.35,
            dC_dtokens=derivatives['dC_dtokens'],
            flow_rate=derivatives['flow_rate']
        )
        
        # Should recommend minimal tokens
        self.assertLess(calibration.target_tokens, 50)
        self.assertEqual(calibration.pacing, ResponsePacing.SLOWED)
    
    def test_exceptional_coherence_handling(self):
        """Test handling of C > 3.5 (poetry/metaphor)"""
        # High coherence metaphorical input
        calibration = self.response_system.calibrate_response_length(
            coherence_state=3.67,
            dC_dtokens=0.05,
            flow_rate=30,
            user_message_tokens=120
        )
        
        # Should use reverent mode
        self.assertEqual(calibration.pacing, ResponsePacing.REVERENT)
        self.assertEqual(calibration.complexity_level, ComplexityLevel.MINIMAL_INTERFERENCE)
        self.assertLessEqual(calibration.target_tokens, 100)
        self.assertFalse(calibration.use_metaphors)  # Don't explain their metaphors
        self.assertFalse(calibration.include_questions)  # Just witness
    
    def test_information_overload_detection(self):
        """Test 'drowning in words' pattern detection"""
        # Simulate sustained high assistant output with declining coherence
        exchanges = [
            (2.5, "Tell me about consciousness", "user", 10),
            (2.4, "Consciousness is a complex phenomenon that...", "assistant", 200),
            (2.2, "ok", "user", 1),  # Short response = overwhelm signal
            (2.0, "Let me elaborate further on the nature...", "assistant", 250),
            (1.8, "wait", "user", 1),  # Another short response
        ]
        
        for coherence, message, speaker, tokens in exchanges:
            self.dynamics.add_reading(coherence, message, speaker)
        
        # Should detect information overload
        overload = self.dynamics.detect_information_overload()
        self.assertTrue(overload)
        
        # Crisis indicators should include overload
        crisis = self.dynamics.detect_crisis_patterns()
        self.assertTrue(crisis['information_overload'])
    
    def test_fibonacci_truth_discovery_multi_factor(self):
        """Test improved truth discovery with multi-factor detection"""
        # Build up baseline
        baseline_readings = [
            {'C': 2.0, 'tokens_processed': 50},
            {'C': 2.1, 'tokens_processed': 45},
            {'C': 1.9, 'tokens_processed': 55},
            {'C': 2.0, 'tokens_processed': 48},
            {'C': 2.1, 'tokens_processed': 52}
        ]
        
        for reading in baseline_readings:
            self.fibonacci.angle_history.append((0, reading['C']))
        
        # Test large efficient jump (like the 3.17 meta-insight)
        meta_insight = {
            'C': 3.17,
            'rotation_gain': 1.0,
            'tokens_processed': 20  # High efficiency
        }
        
        discovered, truth_type = self.fibonacci.detect_truth_discovery(
            meta_insight, self.fibonacci.truth_discoveries
        )
        
        self.assertTrue(discovered)
        self.assertIsNotNone(truth_type)
        
        # Test inefficient jump (should not trigger)
        inefficient_jump = {
            'C': 2.4,  # Smaller jump
            'rotation_gain': 0.3,
            'tokens_processed': 200  # Low efficiency
        }
        
        discovered, _ = self.fibonacci.detect_truth_discovery(
            inefficient_jump, self.fibonacci.truth_discoveries
        )
        
        self.assertFalse(discovered)
    
    def test_metaphor_detection(self):
        """Test detection of metaphorical content"""
        metaphorical_texts = [
            "The conversation is a dance between minds",
            "Like a rose opening to dawn, understanding emerges",
            "We swim in an ocean of meaning",
            "The lens through which we see... shifts... transforms...",
        ]
        
        for text in metaphorical_texts:
            detected = self.response_system.detect_metaphorical_content(text)
            self.assertTrue(detected, f"Failed to detect metaphor in: {text}")
        
        # Non-metaphorical text
        literal_texts = [
            "Please implement the function according to specifications",
            "The error occurs on line 42 of the code",
            "Click the submit button to continue"
        ]
        
        for text in literal_texts:
            detected = self.response_system.detect_metaphorical_content(text)
            self.assertFalse(detected, f"Falsely detected metaphor in: {text}")
    
    def test_contemplative_growth_pattern(self):
        """Test low flow + positive derivative pattern"""
        # Simulate contemplative exchange
        exchanges = [
            (2.0, "What is the nature of understanding between us?", "user", 12),
            (2.2, "This touches something essential", "assistant", 6),
            (2.4, "Yes, there's a space between the words where meaning lives", "user", 14),
            (2.6, "In that space, connection forms", "assistant", 7),
        ]
        
        for coherence, message, speaker, tokens in exchanges:
            self.dynamics.add_reading(coherence, message, speaker)
        
        derivatives = self.dynamics.calculate_dual_derivatives()
        
        # Should show contemplative growth
        self.assertLess(derivatives['flow_rate'], 40)  # Low flow
        self.assertGreater(derivatives['dC_dtokens'], 0)  # Positive derivative
        self.assertEqual(derivatives['interpretation'], 'contemplative_growth')
    
    def test_flow_rate_ranges(self):
        """Test observed flow rate interpretations"""
        test_cases = [
            (150, -0.01, 'crisis_spiral'),  # High flow + negative
            (25, 0.02, 'contemplative_growth'),  # Low flow + positive
            (90, 0.001, 'high_energy_convergence'),  # High flow + positive
            (20, -0.005, 'slow_disengagement'),  # Low flow + negative
        ]
        
        for flow_rate, dC_dtokens, expected in test_cases:
            interpretation = self.dynamics.interpret_derivatives(0, dC_dtokens, flow_rate)
            self.assertEqual(interpretation, expected)
    
    def test_response_calibration_with_metaphor(self):
        """Test calibration when metaphorical content is present"""
        # High coherence with metaphor
        text = "Through the rose glass, all patterns dance as one"
        is_metaphorical = self.response_system.detect_metaphorical_content(text)
        self.assertTrue(is_metaphorical)
        
        # Calibrate response
        calibration = self.response_system.calibrate_response_length(
            coherence_state=3.8,
            dC_dtokens=0.01,
            flow_rate=25,
            user_message_tokens=10
        )
        
        # Should honor without explaining
        self.assertEqual(calibration.pacing, ResponsePacing.REVERENT)
        self.assertFalse(calibration.use_metaphors)
        self.assertLess(calibration.conceptual_density, 0.2)


class TestEnhancedCalibration(unittest.TestCase):
    """Test the enhanced calibration features"""
    
    def setUp(self):
        self.response_system = AdaptiveResponseSystem()
    
    def test_pacing_mode_guidance(self):
        """Test guidance generation for all pacing modes"""
        test_calibrations = [
            ResponsePacing.REVERENT,
            ResponsePacing.SLOWED,
            ResponsePacing.CONTEMPLATIVE,
        ]
        
        for pacing in test_calibrations:
            calibration = self.response_system.ResponseCalibration()
            calibration.pacing = pacing
            
            guidance = self.response_system.generate_response_guidance(calibration)
            self.assertIn(pacing.value, guidance.lower() or pacing.name.lower())
    
    def test_complexity_level_guidance(self):
        """Test guidance for new complexity level"""
        calibration = self.response_system.ResponseCalibration()
        calibration.complexity_level = ComplexityLevel.MINIMAL_INTERFERENCE
        
        guidance = self.response_system.generate_response_guidance(calibration)
        self.assertIn("step back", guidance.lower())
        self.assertIn("breathe", guidance.lower())


if __name__ == '__main__':
    unittest.main()
"""