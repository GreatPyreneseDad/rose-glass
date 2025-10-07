"""
Context Detection Tests
=======================

Comprehensive tests for the four critical context detectors:
- TrustSignalDetector
- MissionModeDetector  
- TokenMultiplierLimiter
- EssenceRequestDetector

These tests ensure proper detection and calibration of the ~10%
of cases where coherence alone is insufficient.

Author: Christopher MacGregor bin Joseph
Date: October 2025
"""

import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.trust_signal_detector import TrustSignalDetector, TrustSignalType
from src.core.mission_mode_detector import MissionModeDetector, MissionType
from src.core.token_multiplier_limiter import TokenMultiplierLimiter, MultiplierMode
from src.core.essence_request_detector import EssenceRequestDetector, EssenceType
from src.core.adaptive_response_system import AdaptiveResponseSystem, ResponsePacing


class TestTrustSignalDetector(unittest.TestCase):
    """Test trust signal detection"""
    
    def setUp(self):
        self.detector = TrustSignalDetector()
    
    def test_direct_trust_detection(self):
        """Test detection of direct trust phrases"""
        test_cases = [
            ("Trust me on this", 3.0, True),
            ("Believe me when I say this matters", 2.8, True),
            ("I promise this is important", 2.6, True),
            ("Listen carefully to what I'm about to say", 2.5, True),
            ("This is just a normal message", 3.0, False)
        ]
        
        for message, coherence, expected in test_cases:
            tokens = len(message.split())
            signal = self.detector.detect_trust_signals(message, coherence, tokens)
            detected = signal is not None
            self.assertEqual(detected, expected, 
                           f"Failed for: '{message}' with C={coherence}")
            
            if signal:
                self.assertEqual(signal.signal_type, TrustSignalType.DIRECT_TRUST)
    
    def test_poetic_trust_signals(self):
        """Test detection of poetic/metaphorical trust signals"""
        test_cases = [
            ("Through the rose glass...", 3.2, TrustSignalType.POETIC_ESSENCE),
            ("Dance of light and shadow", 3.5, TrustSignalType.POETIC_ESSENCE),
            ("Where silence speaks...", 3.0, TrustSignalType.POETIC_ESSENCE)
        ]
        
        for message, coherence, expected_type in test_cases:
            tokens = len(message.split())
            signal = self.detector.detect_trust_signals(message, coherence, tokens)
            self.assertIsNotNone(signal)
            self.assertEqual(signal.signal_type, expected_type)
    
    def test_coherence_threshold(self):
        """Test that coherence thresholds are enforced"""
        # Direct trust requires C >= 2.5
        signal = self.detector.detect_trust_signals("Trust me", 2.4, 2)
        self.assertIsNone(signal)
        
        signal = self.detector.detect_trust_signals("Trust me", 2.5, 2)
        self.assertIsNotNone(signal)
        
        # Poetic requires C >= 3.0
        signal = self.detector.detect_trust_signals("Rose glass dance...", 2.9, 3)
        self.assertIsNone(signal)
        
        signal = self.detector.detect_trust_signals("Rose glass dance...", 3.0, 3)
        self.assertIsNotNone(signal)
    
    def test_brevity_requirement(self):
        """Test that trust signals must be brief"""
        long_message = "Trust me " + " ".join(["on this matter"] * 20)
        tokens = len(long_message.split())
        
        # Over 30 tokens should not trigger
        signal = self.detector.detect_trust_signals(long_message, 3.0, tokens)
        self.assertIsNone(signal)
    
    def test_reverent_mode_trigger(self):
        """Test when reverent mode should be triggered"""
        signal = self.detector.detect_trust_signals("Trust me", 3.0, 2)
        should_trigger = self.detector.should_trigger_reverent_mode(signal, 'standard')
        self.assertTrue(should_trigger)


class TestMissionModeDetector(unittest.TestCase):
    """Test mission mode detection"""
    
    def setUp(self):
        self.detector = MissionModeDetector()
    
    def test_research_mission_detection(self):
        """Test detection of research requests"""
        test_cases = [
            ("Research the history of AI", MissionType.RESEARCH),
            ("Investigate quantum computing applications", MissionType.RESEARCH),
            ("Study the impact of social media", MissionType.RESEARCH),
            ("Look into renewable energy solutions", MissionType.RESEARCH)
        ]
        
        for message, expected_type in test_cases:
            mission = self.detector.detect_mission(message)
            self.assertIsNotNone(mission)
            self.assertEqual(mission.mission_type, expected_type)
    
    def test_analysis_mission_detection(self):
        """Test detection of analysis requests"""
        test_cases = [
            ("Analyze the themes in this poem", MissionType.ANALYSIS),
            ("Break down the components of this system", MissionType.ANALYSIS),
            ("Evaluate the pros and cons", MissionType.ANALYSIS)
        ]
        
        for message, expected_type in test_cases:
            mission = self.detector.detect_mission(message)
            self.assertIsNotNone(mission)
            self.assertEqual(mission.mission_type, expected_type)
    
    def test_scope_determination(self):
        """Test scope detection (comprehensive, focused, quick)"""
        test_cases = [
            ("Give me a comprehensive analysis of ML", 'comprehensive'),
            ("Research everything about quantum computing", 'comprehensive'),
            ("Quick overview of Python", 'quick'),
            ("Brief summary of the main points", 'quick'),
            ("Focus on the specific implementation details", 'focused')
        ]
        
        for message, expected_scope in test_cases:
            mission = self.detector.detect_mission(message)
            if mission:
                self.assertEqual(mission.scope, expected_scope,
                               f"Wrong scope for: {message}")
    
    def test_structure_requirements(self):
        """Test detection of structured output requirements"""
        structured_requests = [
            "Give me a step-by-step guide",
            "List the points in order: 1. First 2. Second",
            "Organize this into sections",
            "Create an outline of the main concepts"
        ]
        
        for message in structured_requests:
            mission = self.detector.detect_mission(message)
            if mission:
                self.assertTrue(mission.requires_structure,
                              f"Should require structure: {message}")
    
    def test_mission_override_decision(self):
        """Test when mission mode should override coherence"""
        mission = self.detector.detect_mission("Research quantum computing")
        should_override = self.detector.should_override_coherence_mode(mission, 2.0)
        self.assertTrue(should_override)
        
        # Low confidence mission with high coherence shouldn't override
        weak_mission = self.detector.detect_mission("Maybe look at this?")
        if weak_mission:
            weak_mission.confidence = 0.3
            should_override = self.detector.should_override_coherence_mode(weak_mission, 3.5)
            self.assertFalse(should_override)


class TestTokenMultiplierLimiter(unittest.TestCase):
    """Test token multiplier limits"""
    
    def setUp(self):
        self.limiter = TokenMultiplierLimiter()
    
    def test_coherence_based_multipliers(self):
        """Test multipliers based on coherence levels"""
        test_cases = [
            (0.5, 0.5),   # Very low coherence
            (1.2, 0.8),   # Low coherence
            (2.2, 1.5),   # Medium coherence
            (3.2, 2.5),   # High coherence
            (3.8, 3.0),   # Very high coherence
        ]
        
        for coherence, expected_mult in test_cases:
            limit = self.limiter.calculate_token_limit(
                user_tokens=100,
                coherence=coherence,
                conversation_state={}
            )
            self.assertAlmostEqual(limit.raw_multiplier, expected_mult, places=1)
    
    def test_crisis_override(self):
        """Test crisis mode token limits"""
        crisis_state = {
            'crisis_detected': True
        }
        
        limit = self.limiter.calculate_token_limit(
            user_tokens=100,
            coherence=3.0,  # High coherence
            conversation_state=crisis_state
        )
        
        # Crisis should severely limit response
        self.assertLessEqual(limit.adjusted_multiplier, 0.3)
        self.assertEqual(limit.mode, MultiplierMode.CRISIS)
    
    def test_information_overload_limit(self):
        """Test information overload limits"""
        overload_state = {
            'information_overload': True
        }
        
        limit = self.limiter.calculate_token_limit(
            user_tokens=50,
            coherence=2.5,
            conversation_state=overload_state
        )
        
        # Should be minimal
        self.assertLessEqual(limit.adjusted_multiplier, 0.2)
        self.assertIn("Information overload", limit.adjustments[0])
    
    def test_mission_mode_expansion(self):
        """Test that mission mode allows expansion"""
        mission_state = {
            'mission_mode': True
        }
        
        limit = self.limiter.calculate_token_limit(
            user_tokens=20,
            coherence=2.0,
            conversation_state=mission_state
        )
        
        # Mission mode should allow at least 2x
        self.assertGreaterEqual(limit.adjusted_multiplier, 2.0)
    
    def test_hard_cap_enforcement(self):
        """Test that hard cap is enforced"""
        # Very long user message with high coherence
        limit = self.limiter.calculate_token_limit(
            user_tokens=300,
            coherence=3.5,
            conversation_state={}
        )
        
        # Should hit hard cap
        self.assertEqual(limit.token_limit, 500)
        self.assertIn("Hard cap applied", limit.adjustments[-1])
    
    def test_user_length_adjustments(self):
        """Test adjustments based on user message length"""
        # Very brief message
        limit_brief = self.limiter.calculate_token_limit(
            user_tokens=5,
            coherence=2.0,
            conversation_state={}
        )
        
        # Long message
        limit_long = self.limiter.calculate_token_limit(
            user_tokens=150,
            coherence=2.0,
            conversation_state={}
        )
        
        # Brief messages get higher multiplier
        self.assertGreater(limit_brief.adjusted_multiplier, 
                          limit_long.adjusted_multiplier)


class TestEssenceRequestDetector(unittest.TestCase):
    """Test essence request detection"""
    
    def setUp(self):
        self.detector = EssenceRequestDetector()
    
    def test_summary_detection(self):
        """Test detection of summary requests"""
        test_cases = [
            ("Can you summarize our discussion?", EssenceType.SUMMARY),
            ("Please sum up the main points", EssenceType.SUMMARY),
            ("Give me a brief recap", EssenceType.SUMMARY),
            ("Overview of what we covered", EssenceType.SUMMARY)
        ]
        
        for message, expected_type in test_cases:
            request = self.detector.detect_essence_request(message)
            self.assertIsNotNone(request)
            self.assertEqual(request.essence_type, expected_type)
    
    def test_key_points_detection(self):
        """Test detection of key points requests"""
        test_cases = [
            ("What are the key points?", EssenceType.KEY_POINTS),
            ("List the main ideas", EssenceType.KEY_POINTS),
            ("Core concepts please", EssenceType.KEY_POINTS)
        ]
        
        for message, expected_type in test_cases:
            request = self.detector.detect_essence_request(message)
            self.assertIsNotNone(request)
            self.assertEqual(request.essence_type, expected_type)
    
    def test_tldr_detection(self):
        """Test detection of TL;DR requests"""
        test_cases = [
            ("TL;DR?", EssenceType.TLDR),
            ("tldr", EssenceType.TLDR),
            ("Give me the short version", EssenceType.TLDR),
            ("In one sentence?", EssenceType.TLDR)
        ]
        
        for message, expected_type in test_cases:
            request = self.detector.detect_essence_request(message)
            self.assertIsNotNone(request)
            self.assertEqual(request.essence_type, expected_type)
    
    def test_format_preferences(self):
        """Test detection of format preferences"""
        test_cases = [
            ("Give me bullet points", 'bullets'),
            ("List the key points", 'bullets'),
            ("Number the main ideas", 'numbered'),
            ("Write a paragraph summary", 'paragraph')
        ]
        
        for message, expected_format in test_cases:
            request = self.detector.detect_essence_request(message)
            if request:
                self.assertEqual(request.format_preference, expected_format)
    
    def test_scope_detection(self):
        """Test scope detection for essence requests"""
        test_cases = [
            ("Summarize our conversation", 'conversation'),
            ("Sum up what we've discussed", 'conversation'),
            ("Key points about quantum computing", 'topic'),
            ("Summarize this document", 'document')
        ]
        
        for message, expected_scope in test_cases:
            request = self.detector.detect_essence_request(message)
            if request:
                self.assertEqual(request.scope, expected_scope)
    
    def test_target_length_detection(self):
        """Test detection of target length preferences"""
        test_cases = [
            ("Very brief summary", 'brief'),
            ("One sentence summary", 'brief'),
            ("Detailed summary", 'detailed'),
            ("Comprehensive overview", 'detailed'),
            ("Standard summary", 'brief')  # Default
        ]
        
        for message, expected_length in test_cases:
            request = self.detector.detect_essence_request(message)
            if request:
                self.assertEqual(request.target_length, expected_length)


class TestIntegratedContextDetection(unittest.TestCase):
    """Test integrated context detection in AdaptiveResponseSystem"""
    
    def setUp(self):
        self.system = AdaptiveResponseSystem()
    
    def test_priority_ordering(self):
        """Test that context modes are prioritized correctly"""
        # Crisis should override everything
        context = self.system.detect_context(
            "Trust me on this important research",
            coherence=3.5,
            conversation_state={'crisis_detected': True}
        )
        self.assertEqual(context['primary_mode'], 'crisis')
        
        # Trust signal should override mission mode
        context = self.system.detect_context(
            "Trust me",
            coherence=3.0,
            conversation_state={}
        )
        self.assertEqual(context['primary_mode'], 'trust')
        
        # Essence should override mission
        context = self.system.detect_context(
            "Summarize this research on AI",
            coherence=2.5,
            conversation_state={}
        )
        self.assertEqual(context['primary_mode'], 'essence')
    
    def test_calibration_with_context(self):
        """Test that calibration properly uses context detection"""
        # Test trust signal calibration
        calibration, context = self.system.calibrate_with_context(
            message="Trust me on this",
            coherence=3.2,
            dC_dtokens=0.01,
            flow_rate=30,
            conversation_state={}
        )
        
        self.assertEqual(calibration.pacing, ResponsePacing.REVERENT)
        self.assertLessEqual(calibration.target_tokens, 100)
        
        # Test mission mode calibration
        calibration, context = self.system.calibrate_with_context(
            message="Research machine learning algorithms comprehensively",
            coherence=2.0,
            dC_dtokens=0.0,
            flow_rate=50,
            conversation_state={}
        )
        
        self.assertEqual(calibration.pacing, ResponsePacing.SYSTEMATIC)
        self.assertGreaterEqual(calibration.target_tokens, 500)
        
        # Test essence request calibration
        calibration, context = self.system.calibrate_with_context(
            message="TL;DR?",
            coherence=2.5,
            dC_dtokens=0.0,
            flow_rate=40,
            conversation_state={}
        )
        
        self.assertEqual(calibration.pacing, ResponsePacing.DISTILLED)
        self.assertLessEqual(calibration.target_tokens, 100)
    
    def test_token_limit_integration(self):
        """Test that token limits are properly applied"""
        # High coherence but with token limiter
        calibration, context = self.system.calibrate_with_context(
            message="Explain this complex topic in detail please",
            coherence=3.0,
            dC_dtokens=0.02,
            flow_rate=60,
            conversation_state={}
        )
        
        # Check that token limit was applied
        token_limit = context['token_limit'].token_limit
        self.assertLessEqual(calibration.target_tokens, token_limit)
    
    def test_edge_case_handling(self):
        """Test handling of edge cases"""
        # Empty message
        calibration, context = self.system.calibrate_with_context(
            message="",
            coherence=2.0,
            dC_dtokens=0.0,
            flow_rate=0,
            conversation_state={}
        )
        self.assertEqual(context['primary_mode'], 'coherence_based')
        
        # Multiple detections - verify priority
        calibration, context = self.system.calibrate_with_context(
            message="Trust me, research and summarize quantum computing",
            coherence=3.2,
            dC_dtokens=0.01,
            flow_rate=40,
            conversation_state={}
        )
        # Trust should win due to priority
        self.assertEqual(context['primary_mode'], 'trust')


if __name__ == '__main__':
    unittest.main()
"""