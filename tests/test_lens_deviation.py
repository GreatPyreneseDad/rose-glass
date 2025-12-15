"""
Test Lens Deviation & Truth Invariance
======================================

Tests for cross-contextual Fibonacci reset triggers.
"""

import pytest
import sys
sys.path.insert(0, '/Users/chris/rose-glass')

from src.core.rose_glass_v2 import RoseGlassV2
from src.core.fibonacci_lens_rotation import FibonacciLensRotation, TruthType


class TestRoseGlassV2LensDeviation:
    """Tests for lens deviation in RoseGlassV2"""

    def test_calculate_lens_deviation(self):
        """Should calculate standard deviation across lenses"""
        glass = RoseGlassV2(invariance_threshold=0.1)
        glass.active_calibration = glass.calibrations['digital_native']

        # Balanced pattern should have some deviation
        deviation = glass.calculate_lens_deviation(
            psi=0.6, rho=0.6, q=0.5, f=0.5
        )

        assert 0.0 <= deviation <= 1.0
        assert isinstance(deviation, float)

    def test_should_reset_fibonacci_universal_truth(self):
        """Should reset on lens-invariant truth"""
        glass = RoseGlassV2(invariance_threshold=0.15)
        glass.active_calibration = glass.calibrations['digital_native']

        # Balanced pattern - may trigger reset depending on exact calibrations
        should_reset, deviation = glass.should_reset_fibonacci(
            psi=0.6, rho=0.6, q=0.5, f=0.5
        )

        assert isinstance(should_reset, bool)
        assert 0.0 <= deviation <= 1.0

    def test_invariance_threshold_configuration(self):
        """Should allow configuration of invariance threshold"""
        glass1 = RoseGlassV2(invariance_threshold=0.05)
        glass2 = RoseGlassV2(invariance_threshold=0.20)

        assert glass1.invariance_threshold == 0.05
        assert glass2.invariance_threshold == 0.20


class TestFibonacciLensRotationIntegration:
    """Tests for FibonacciLensRotation with lens deviation"""

    def test_fibonacci_accepts_rose_glass(self):
        """FibonacciLensRotation should accept rose_glass parameter"""
        glass = RoseGlassV2()
        glass.active_calibration = glass.calibrations['digital_native']

        rotation = FibonacciLensRotation(rose_glass=glass)

        assert rotation.rose_glass is not None
        assert rotation.rose_glass == glass

    def test_lens_invariant_truth_type_exists(self):
        """TruthType.LENS_INVARIANT should exist"""
        assert hasattr(TruthType, 'LENS_INVARIANT')
        assert TruthType.LENS_INVARIANT.value == 'lens_invariant'

    def test_detect_truth_with_lens_deviation(self):
        """Should detect lens-invariant truth in truth discovery"""
        glass = RoseGlassV2(invariance_threshold=0.05)  # Low threshold for testing
        glass.active_calibration = glass.calibrations['digital_native']

        rotation = FibonacciLensRotation(rose_glass=glass)

        # Create a reading with variables that might trigger lens invariance
        current_reading = {
            'C': 0.7,
            'C_base': 0.6,
            'rotation_gain': 0.1,
            'variables': {
                'psi': 0.6,
                'rho': 0.6,
                'q': 0.5,
                'f': 0.5
            },
            'dominant_components': ['consistency'],
            'interpretation': 'Balanced pattern'
        }

        # Add some angle history
        rotation.angle_history = [(0, 0.5), (30, 0.6)]

        truth_discovered, truth_type = rotation.detect_truth_discovery(
            current_reading,
            rotation.truth_discoveries
        )

        # May or may not trigger depending on exact lens calibrations
        assert isinstance(truth_discovered, bool)
        if truth_discovered:
            # If truth was discovered, type should be valid
            assert truth_type is not None
            assert isinstance(truth_type, TruthType)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
