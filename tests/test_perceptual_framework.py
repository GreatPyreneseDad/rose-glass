"""
Tests for the Rose Glass Perceptual Framework
"""

import pytest
import numpy as np
from datetime import datetime
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from perception.rose_glass_perception import (
    RoseGlassPerception, PerceptionPhase, DimensionalPattern, Perception
)
from perception.pattern_memory import PatternMemory, MemoryEntry
from perception.cultural_calibration import CulturalCalibration, CalibrationPreset
from perception.breathing_patterns import BreathingPatternDetector
from perception.uncertainty_handler import UncertaintyHandler, UncertaintyType


class TestRoseGlassPerception:
    """Test the main RoseGlassPerception class"""
    
    def test_initialization(self):
        """Test basic initialization"""
        perception = RoseGlassPerception()
        assert perception.phase == PerceptionPhase.EXPLICIT
        assert perception.uncertainty_threshold == 0.7
        
    def test_pattern_extraction(self):
        """Test dimensional pattern extraction"""
        perception = RoseGlassPerception()
        
        # Test academic text
        academic_text = """
        Therefore, we can conclude that the hypothesis is supported by the evidence.
        First, the data shows a clear correlation. Second, the theoretical framework
        aligns with our observations. Finally, the results are statistically significant.
        """
        
        pattern = perception._extract_patterns(academic_text)
        
        assert pattern.psi > 0.7  # High internal consistency
        assert pattern.rho > 0.5  # Some wisdom indicators
        assert pattern.q < 0.5   # Low emotional activation
        assert isinstance(pattern, DimensionalPattern)
        
    def test_perception_phases(self):
        """Test different perception phases"""
        perception = RoseGlassPerception(PerceptionPhase.EXPLICIT)
        
        test_text = "We must act now! This is critical for our community."
        result = perception.perceive(test_text)
        
        assert isinstance(result, Perception)
        assert result.dimensions.q > 0.5  # High moral activation
        assert result.dimensions.f > 0.5  # Social belonging
        
        # Test phase transition
        perception.set_phase(PerceptionPhase.INTERNAL)
        assert perception.phase == PerceptionPhase.INTERNAL
        
    def test_cultural_perception_differences(self):
        """Test how perception varies with cultural calibration"""
        perception = RoseGlassPerception()
        
        text = "We must honor our ancestors' wisdom and act together as one."
        
        # Default perception
        default_result = perception.perceive(text)
        
        # Force indigenous calibration
        perception.calibration.default_calibration = 'indigenous_oral'
        indigenous_result = perception.perceive(text)
        
        # Indigenous calibration should show higher f and rho
        assert indigenous_result.dimensions.f > default_result.dimensions.f
        assert indigenous_result.dimensions.rho > default_result.dimensions.rho


class TestPatternMemory:
    """Test the PatternMemory system"""
    
    def test_memory_storage(self):
        """Test storing and retrieving patterns"""
        memory = PatternMemory(max_entries=10)
        
        # Create test pattern
        pattern = DimensionalPattern(psi=0.8, rho=0.7, q=0.5, f=0.6)
        perception = Perception(
            dimensions=pattern,
            rhythm={'pace': 'moderate'},
            calibration='default'
        )
        
        # Store pattern
        memory.update(perception)
        
        assert len(memory) == 1
        assert memory.evolution_tracking['psi'][-1] == 0.8
        
    def test_similarity_scoring(self):
        """Test pattern similarity calculation"""
        memory = PatternMemory()
        
        # Add some patterns
        for i in range(5):
            pattern = DimensionalPattern(
                psi=0.8 + i*0.02,
                rho=0.7,
                q=0.5,
                f=0.6
            )
            perception = Perception(
                dimensions=pattern,
                rhythm={'pace': 'moderate'},
                calibration='default'
            )
            memory.update(perception)
            
        # Test similarity with a close pattern
        similar_pattern = DimensionalPattern(psi=0.81, rho=0.71, q=0.51, f=0.61)
        score = memory.get_similarity_score(similar_pattern)
        assert score > 0.8  # Should be very similar
        
        # Test with distant pattern
        distant_pattern = DimensionalPattern(psi=0.2, rho=0.3, q=0.9, f=0.1)
        score = memory.get_similarity_score(distant_pattern)
        assert score < 0.5  # Should be dissimilar
        
    def test_evolution_tracking(self):
        """Test tracking pattern evolution over time"""
        memory = PatternMemory()
        
        # Simulate evolving patterns
        for i in range(10):
            pattern = DimensionalPattern(
                psi=0.5 + i*0.05,  # Increasing consistency
                rho=0.7,
                q=0.5,
                f=0.6
            )
            perception = Perception(
                dimensions=pattern,
                rhythm={'pace': 'moderate'},
                calibration='default'
            )
            memory.update(perception)
            
        # Check evolution trend
        trend = memory.get_evolution_trend('psi', window=5)
        assert trend['trend'] > 0  # Should show increasing trend
        assert trend['current'] > trend['mean']  # Current above average


class TestCulturalCalibration:
    """Test cultural calibration system"""
    
    def test_calibration_presets(self):
        """Test that calibration presets are loaded correctly"""
        calibration = CulturalCalibration()
        
        # Check default calibrations exist
        assert 'modern_western_academic' in calibration.list_calibrations()
        assert 'medieval_islamic' in calibration.list_calibrations()
        assert 'digital_native' in calibration.list_calibrations()
        
    def test_calibration_application(self):
        """Test applying calibration to patterns"""
        calibration = CulturalCalibration()
        
        # Raw pattern
        raw_pattern = DimensionalPattern(psi=0.5, rho=0.5, q=0.5, f=0.5)
        
        # Apply medieval calibration
        medieval = calibration.apply_calibration(raw_pattern, 'medieval_islamic')
        
        # Medieval should boost psi and rho, reduce q
        assert medieval.psi > raw_pattern.psi  # Higher logical consistency weight
        assert medieval.q < raw_pattern.q      # Lower emotional expression
        
    def test_calibration_suggestion(self):
        """Test automatic calibration suggestion"""
        calibration = CulturalCalibration()
        
        # Technical text
        tech_text = "The function returns a parameter object with the API implementation."
        suggested, confidence = calibration.suggest_calibration(tech_text)
        assert suggested == 'technical_documentation'
        
        # Activist text
        activist_text = "Together we must fight for justice and change in our movement!"
        suggested, confidence = calibration.suggest_calibration(activist_text)
        assert suggested == 'activist_movement'
        
    def test_custom_calibration(self):
        """Test adding custom calibration"""
        calibration = CulturalCalibration()
        
        # Create custom preset
        custom = CalibrationPreset(
            name='test_custom',
            description='Test custom calibration',
            km=0.3,
            ki=0.9,
            coupling_strength=0.2,
            dimension_weights={'psi': 1.1, 'rho': 0.9, 'q': 1.2, 'f': 0.8}
        )
        
        calibration.add_calibration(custom)
        assert 'test_custom' in calibration.list_calibrations()


class TestBreathingPatterns:
    """Test breathing pattern detection"""
    
    def test_pace_detection(self):
        """Test detecting communication pace"""
        detector = BreathingPatternDetector()
        
        # Rapid pace text
        rapid_text = "Quick! Help! We need this now! Fast!"
        rhythm = detector.analyze(rapid_text)
        assert rhythm['pace'] in ['rapid', 'staccato']
        
        # Contemplative pace
        contemplative = """
        When we consider the nature of existence, we find ourselves pondering
        the deep questions that have occupied philosophers throughout the ages,
        leading us to reflect upon our place in the vast cosmos.
        """
        rhythm = detector.analyze(contemplative)
        assert rhythm['pace'] in ['contemplative', 'flowing']
        
    def test_pause_detection(self):
        """Test detecting pauses in text"""
        detector = BreathingPatternDetector()
        
        text_with_pauses = "Well... I think — no, I mean - it's complicated, you know?"
        rhythm = detector.analyze(text_with_pauses)
        
        pauses = rhythm['pause_pattern']
        assert pauses['ellipses'] > 0
        assert pauses['em_dashes'] > 0
        assert pauses['commas'] > 0
        
    def test_rhythm_classification(self):
        """Test overall rhythm classification"""
        detector = BreathingPatternDetector()
        
        # Academic rhythm
        academic = """
        Furthermore, the empirical evidence suggests that the theoretical
        framework, when properly applied to the dataset, yields statistically
        significant results that support our initial hypothesis.
        """
        rhythm = detector.analyze(academic)
        assert rhythm['rhythm_type'] == 'academic'
        
        # Urgent rhythm
        urgent = "Stop! This is critical! We must act now! Please!"
        rhythm = detector.analyze(urgent)
        assert rhythm['rhythm_type'] == 'urgent'
        
    def test_response_rhythm_suggestion(self):
        """Test suggesting appropriate response rhythm"""
        detector = BreathingPatternDetector()
        
        # Crisis communication
        crisis = "Help! Emergency! Need assistance immediately!"
        user_rhythm = detector.analyze(crisis)
        suggestion = detector.suggest_response_rhythm(user_rhythm)
        
        assert suggestion['target_pace'] == 'rapid'
        assert suggestion['sentence_length'] == 'short'
        assert suggestion['emotional_mirroring'] == True


class TestUncertaintyHandler:
    """Test uncertainty handling and superposition"""
    
    def test_uncertainty_assessment(self):
        """Test assessing uncertainty levels"""
        handler = UncertaintyHandler(comfort_threshold=0.7)
        
        # Create divergent patterns
        patterns = [
            ('modern_western_academic', DimensionalPattern(0.8, 0.7, 0.3, 0.4)),
            ('activist_movement', DimensionalPattern(0.5, 0.4, 0.9, 0.8)),
            ('technical_documentation', DimensionalPattern(0.9, 0.5, 0.2, 0.3))
        ]
        
        assessment = handler.assess_uncertainty(patterns, primary_confidence=0.5)
        
        assert assessment['maintain_superposition'] == True
        assert assessment['type'] == UncertaintyType.MIXED_SIGNALS
        assert assessment['divergence'] > 0.3
        
    def test_response_generation(self):
        """Test generating responses under uncertainty"""
        handler = UncertaintyHandler()
        
        from perception.uncertainty_handler import InterpretiveOption
        
        interpretations = [
            InterpretiveOption(
                interpretation="Technical analysis",
                confidence=0.6,
                reasoning="High logical consistency",
                calibration="technical",
                key_indicators=["precise language", "structured format"]
            ),
            InterpretiveOption(
                interpretation="Emotional expression",
                confidence=0.4,
                reasoning="Underlying urgency detected",
                calibration="activist",
                key_indicators=["urgency markers", "value language"]
            )
        ]
        
        uncertainty = {'type': UncertaintyType.CROSS_DOMAIN, 'level': 0.5}
        
        response = handler.generate_response_options(uncertainty, interpretations)
        assert 'strategy' in response
        assert 'options' in response
        assert len(response['options']) > 0
        
    def test_uncertainty_resolution(self):
        """Test resolving uncertainty with clarification"""
        handler = UncertaintyHandler()
        
        initial_uncertainty = {
            'level': 0.8,
            'type': UncertaintyType.INSUFFICIENT_CONTEXT,
            'maintain_superposition': True
        }
        
        # User provides clarification
        clarification = "Yes, exactly! You understood correctly."
        
        resolved = handler.resolve_uncertainty(clarification, initial_uncertainty)
        
        assert resolved['level'] < initial_uncertainty['level']
        assert resolved['resolution_progress'] > 0.3


class TestIntegration:
    """Integration tests for the full framework"""
    
    def test_full_perception_pipeline(self):
        """Test complete perception pipeline"""
        perception = RoseGlassPerception()
        
        # Complex text with multiple dimensions
        text = """
        We must urgently address this crisis together! Research shows that
        immediate action is critical. Our community's future depends on it.
        Let's think carefully but act swiftly - time is running out...
        """
        
        result = perception.perceive(text)
        
        # Should detect multiple dimensions
        assert result.dimensions.q > 0.6  # High urgency
        assert result.dimensions.f > 0.5  # Community focus
        assert result.dimensions.psi > 0.4  # Some logical structure
        
        # Should detect rhythm
        assert result.rhythm['pace'] in ['rapid', 'urgent']
        assert result.rhythm['emotional_acceleration']['pattern'] in ['accelerating', 'crisis']
        
        # Should have some uncertainty (mixed signals)
        assert result.uncertainty_level > 0
        
    def test_response_calibration(self):
        """Test calibrating response based on perception"""
        perception = RoseGlassPerception(PerceptionPhase.INTERNAL)
        
        # User message with high urgency
        user_text = "This is urgent! We need help immediately! Please respond!"
        user_perception = perception.perceive(user_text)
        
        # Draft response
        draft = "I understand your concern. Let me think about this situation carefully and provide you with a comprehensive analysis of the various factors involved."
        
        # Calibrate response
        calibrated = perception.calibrate_response(user_perception, draft)
        
        # Should be shortened for urgency
        assert len(calibrated) < len(draft)
        
    def test_memory_evolution(self):
        """Test how perception evolves through conversation"""
        perception = RoseGlassPerception()
        
        # Simulate conversation progression
        messages = [
            "I'm confused about this topic.",
            "Actually, I think I'm starting to understand.",
            "Yes! Now it makes perfect sense!",
            "Thank you for helping me see clearly."
        ]
        
        perceptions = []
        for msg in messages:
            p = perception.perceive(msg)
            perceptions.append(p)
            
        # Check that coherence increases over time
        psi_values = [p.dimensions.psi for p in perceptions]
        assert psi_values[-1] > psi_values[0]  # Coherence improved
        
        # Check pattern memory captured evolution
        summary = perception.pattern_memory.get_pattern_summary()
        assert summary['patterns']['psi']['trend'] > 0  # Positive trend


if __name__ == '__main__':
    pytest.main([__file__, '-v'])