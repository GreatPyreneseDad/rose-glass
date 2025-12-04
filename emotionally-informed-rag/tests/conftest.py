"""
Pytest configuration and fixtures for Emotionally-Informed RAG tests
"""

import pytest
import sys
from pathlib import Path

# Add agent to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "agent"))


@pytest.fixture(scope="session")
def test_config():
    """Test configuration"""
    return {
        'agent': {'name': 'test-agent', 'version': '1.0.0'},
        'rose_glass': {'default_lens': 'modern_digital'},
        'retrieval': {
            'qdrant_host': 'localhost',
            'qdrant_port': 6333,
            'elasticsearch_host': 'localhost',
            'elasticsearch_port': 9200,
            'hybrid_alpha': 0.7,
            'top_k': 10
        },
        'generation': {
            'provider': 'mock',
            'model': 'test-model',
            'max_tokens': 1000,
            'token_multiplier_limit': 3.0
        },
        'monitoring': {
            'gradient_tracking': True,
            'escalation_threshold': 0.7,
            'log_level': 'INFO'
        }
    }


@pytest.fixture
def sample_queries():
    """Sample queries for testing"""
    return {
        'high_emotion': "I'm terrified about losing everything! Please help me urgently!",
        'low_emotion': "What are the general principles of contract law?",
        'high_wisdom': "What are the fundamental philosophical underpinnings of justice and equity?",
        'social': "We need to work together as a community to solve this problem",
        'crisis': "URGENT! This is a crisis! I need immediate help!",
        'mission': "I need to systematically analyze all available research on this topic"
    }


@pytest.fixture
def sample_documents():
    """Sample documents for testing"""
    return [
        {
            'id': 'doc1',
            'title': 'Introduction to Contract Law',
            'content': 'Contract law governs agreements between parties...',
            'metadata': {'category': 'legal', 'difficulty': 'beginner'}
        },
        {
            'id': 'doc2',
            'title': 'Trauma-Informed Legal Practice',
            'content': 'Understanding trauma in legal contexts requires empathy and awareness...',
            'metadata': {'category': 'legal', 'difficulty': 'advanced'}
        },
        {
            'id': 'doc3',
            'title': 'Philosophical Foundations of Law',
            'content': 'The philosophical basis of law draws from ancient traditions...',
            'metadata': {'category': 'philosophy', 'difficulty': 'advanced'}
        }
    ]
