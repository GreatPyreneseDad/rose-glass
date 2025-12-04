"""
Emotionally-Informed RAG Agent for Claude Code

Production agent implementing Rose Glass emotional intelligence
with hybrid retrieval-augmented generation.
"""

__version__ = "1.0.0"

from .emotional_agent import EmotionalRAGAgent, EmotionalSignature

__all__ = ['EmotionalRAGAgent', 'EmotionalSignature']
