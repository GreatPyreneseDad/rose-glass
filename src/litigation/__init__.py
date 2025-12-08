"""
Rose Glass Litigation Support Module
===================================

Real-time courtroom coherence analysis.

Components:
- litigation_lens: Core testimony analysis and contradiction detection
- realtime_pipeline: Audio → Transcript → Analysis pipeline (planned)
- claude_integration: Advanced analysis via Claude API (planned)
- delivery: Tablet/earpiece interfaces (planned)

Usage:
    from src.litigation import ContradictionDetector, TestimonyStatement
    
    detector = ContradictionDetector()
    result = detector.add_statement("WITNESS", "I was home that night", 1)
    
Author: Christopher MacGregor bin Joseph
Origin: December 5, 2025
"""

from .litigation_lens import (
    TestimonyStatement,
    ContradictionFlag,
    ContradictionDetector
)

__all__ = [
    'TestimonyStatement',
    'ContradictionFlag', 
    'ContradictionDetector'
]

__version__ = '0.1.0'
__author__ = 'Christopher MacGregor bin Joseph'
