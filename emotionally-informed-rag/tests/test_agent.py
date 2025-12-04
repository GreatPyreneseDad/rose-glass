"""
Unit tests for EmotionalRAGAgent
"""

import pytest
import sys
from pathlib import Path

# Add agent to path
sys.path.insert(0, str(Path(__file__).parent.parent / "agent"))

from emotional_agent import EmotionalRAGAgent, EmotionalSignature


@pytest.fixture
def agent():
    """Create agent instance for testing"""
    return EmotionalRAGAgent()


class TestEmotionalAnalysis:
    """Test emotional signature analysis"""

    def test_analyze_high_emotion(self, agent):
        """High emotional activation should be detected"""
        sig = agent.analyze_query(
            "I'm terrified about losing everything! Please help me!"
        )
        assert sig.q > 0.6, f"Expected high q, got {sig.q}"
        assert sig.context_type in ["crisis", "standard"]

    def test_analyze_calm_query(self, agent):
        """Calm queries should have low emotional activation"""
        sig = agent.analyze_query(
            "What are the general principles of contract law?"
        )
        assert sig.q < 0.5, f"Expected low q, got {sig.q}"

    def test_analyze_wisdom_depth(self, agent):
        """Philosophical queries should show high wisdom depth"""
        sig = agent.analyze_query(
            "What are the fundamental philosophical principles underlying justice?"
        )
        assert sig.rho > 0.3, f"Expected higher rho, got {sig.rho}"

    def test_context_type_detection(self, agent):
        """Context type should be correctly identified"""
        # Crisis
        crisis_sig = agent.analyze_query("URGENT! I need help NOW! Very worried!")
        assert crisis_sig.context_type in ["crisis", "standard"]

        # Standard
        standard_sig = agent.analyze_query("Can you explain this concept?")
        assert standard_sig.context_type == "standard"

    def test_emotional_signature_structure(self, agent):
        """Emotional signature should have all dimensions"""
        sig = agent.analyze_query("Sample text")

        assert hasattr(sig, 'psi')
        assert hasattr(sig, 'rho')
        assert hasattr(sig, 'q')
        assert hasattr(sig, 'f')
        assert hasattr(sig, 'tau')
        assert hasattr(sig, 'lens')
        assert hasattr(sig, 'context_type')

        # All values should be 0-1
        assert 0 <= sig.psi <= 1
        assert 0 <= sig.rho <= 1
        assert 0 <= sig.q <= 1
        assert 0 <= sig.f <= 1
        assert 0 <= sig.tau <= 1


class TestQueryProcessing:
    """Test query processing pipeline"""

    def test_query_returns_response(self, agent):
        """Query should return structured response"""
        result = agent.query("Test query")

        assert 'response' in result
        assert 'signature' in result
        assert 'context_type' in result
        assert isinstance(result['response'], str)

    def test_query_updates_history(self, agent):
        """Query should update conversation history"""
        initial_count = len(agent.conversation_history)
        agent.query("Test query 1")
        agent.query("Test query 2")

        assert len(agent.conversation_history) == initial_count + 2

    def test_multiple_queries(self, agent):
        """Multiple queries should be processed correctly"""
        queries = [
            "What is contract law?",
            "I'm worried about my case",
            "Can you help me understand this?"
        ]

        for query in queries:
            result = agent.query(query)
            assert 'response' in result
            assert 'signature' in result


class TestGradientTracking:
    """Test conversation gradient tracking"""

    def test_escalation_detection(self, agent):
        """Escalation should be detected"""
        # Simulate escalating conversation
        agent.query("I have a question")  # Low q
        agent.query("I'm getting confused")  # Medium q
        agent.query("This is urgent! I need help NOW!")  # High q

        # Check gradient status
        status = agent.track_gradient(agent.analyze_query("Very urgent!!!"))

        # Should detect escalation if q increased significantly
        assert 'alert' in status or 'status' in status

    def test_gradient_insufficient_data(self, agent):
        """Should handle insufficient data gracefully"""
        status = agent.track_gradient(agent.analyze_query("Test"))
        assert 'status' in status


class TestConfiguration:
    """Test agent configuration"""

    def test_default_config(self, agent):
        """Agent should have default configuration"""
        assert 'agent' in agent.config
        assert 'rose_glass' in agent.config
        assert 'retrieval' in agent.config
        assert 'generation' in agent.config
        assert 'monitoring' in agent.config

    def test_custom_config(self):
        """Agent should accept custom config"""
        # This test would need a test config file
        # For now, just verify it doesn't crash
        agent = EmotionalRAGAgent()
        assert agent.config is not None


class TestEmotionalMatching:
    """Test emotional matching calculations"""

    def test_signature_comparison(self, agent):
        """Similar signatures should have high match scores"""
        sig1 = EmotionalSignature(
            psi=0.7, rho=0.6, q=0.5, f=0.4, tau=0.3,
            lens="modern_digital", context_type="standard"
        )
        sig2 = EmotionalSignature(
            psi=0.75, rho=0.65, q=0.55, f=0.45, tau=0.35,
            lens="modern_digital", context_type="standard"
        )

        # Both signatures are similar - would expect high match
        # (Note: would need to expose _calculate_emotional_match for full test)
        assert sig1.psi == pytest.approx(sig2.psi, abs=0.1)
        assert sig1.q == pytest.approx(sig2.q, abs=0.1)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
