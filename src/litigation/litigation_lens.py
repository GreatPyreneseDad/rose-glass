"""
Rose Glass Litigation Lens Module
================================

Real-time coherence analysis for courtroom testimony.
Detects contradictions, tracks credibility, generates cross-examination prompts.

Author: Christopher MacGregor bin Joseph
Origin: December 5, 2025 - Post-hearing debrief
Status: Initial implementation

"The second chair that never misses."
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import hashlib
import sys
import os

# Add parent directories to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.rose_glass_lens import RoseGlass, PatternVisibility


@dataclass
class TestimonyStatement:
    """
    A single statement from testimony, analyzed through Rose Glass.
    
    Each statement captures:
    - Raw text and metadata
    - GCT variable extraction (Ψ, ρ, q, f)
    - Coherence score through the lens
    - Hash for quick comparison
    """
    speaker: str
    text: str
    timestamp: datetime
    transcript_position: int
    coherence: float
    variables: Dict[str, float]
    hash: str
    
    @classmethod
    def create(cls, speaker: str, text: str, position: int,
               rose_glass: RoseGlass) -> 'TestimonyStatement':
        """Create statement with Rose Glass analysis"""
        
        # Extract GCT variables from text
        psi = cls._extract_psi(text)
        rho = cls._extract_rho(text)
        q = cls._extract_q(text)
        f = cls._extract_f(text)
        
        # View through lens
        visibility = rose_glass.view_through_lens(psi, rho, q, f)
        
        return cls(
            speaker=speaker,
            text=text,
            timestamp=datetime.now(),
            transcript_position=position,
            coherence=visibility.coherence,
            variables={'psi': psi, 'rho': rho, 'q': q, 'f': f},
            hash=hashlib.md5(text.lower().encode()).hexdigest()[:12]
        )
    
    @staticmethod
    def _extract_psi(text: str) -> float:
        """
        Extract internal consistency (Ψ).
        Higher when language is self-consistent.
        Lower when contradicting markers present.
        """
        contradiction_markers = ['but', 'however', 'although', 'except', 'unless']
        consistency_markers = ['therefore', 'because', 'since', 'so', 'thus']
        
        words = text.lower().split()
        if len(words) == 0:
            return 0.5
            
        contradictions = sum(1 for w in words if w in contradiction_markers)
        consistencies = sum(1 for w in words if w in consistency_markers)
        
        return max(0.1, min(0.95, 0.7 + (consistencies - contradictions) * 0.1))
    
    @staticmethod
    def _extract_rho(text: str) -> float:
        """
        Extract accumulated wisdom/specificity (ρ).
        Higher when specific details present.
        Lower when vague language dominates.
        """
        specificity_markers = [
            'exactly', 'specifically', 'precisely', 'approximately',
            'on', 'at', 'during', 'between'  # temporal precision
        ]
        vague_markers = [
            'maybe', 'perhaps', 'possibly', 'sometimes', 'usually',
            'i think', 'i guess', 'sort of', 'kind of', 'probably'
        ]
        
        text_lower = text.lower()
        specific = sum(1 for m in specificity_markers if m in text_lower)
        vague = sum(1 for m in vague_markers if m in text_lower)
        
        return max(0.1, min(0.95, 0.5 + (specific - vague) * 0.15))
    
    @staticmethod
    def _extract_q(text: str) -> float:
        """
        Extract moral activation energy (q).
        Higher when emotional language present.
        Capped by biological optimization.
        """
        emotional_markers = [
            'feel', 'felt', 'afraid', 'scared', 'angry', 'furious',
            'upset', 'hurt', 'love', 'hate', 'worried', 'anxious',
            'terrified', 'devastated', 'thrilled', 'horrified'
        ]
        
        text_lower = text.lower()
        emotional = sum(1 for m in emotional_markers if m in text_lower)
        
        return min(0.95, emotional * 0.15)
    
    @staticmethod
    def _extract_f(text: str) -> float:
        """
        Extract social belonging architecture (f).
        Higher when collective language used.
        Lower when purely individual focus.
        """
        collective_markers = ['we', 'us', 'our', 'together', 'family', 'community']
        individual_markers = ['i', 'me', 'my', 'mine', 'myself']
        
        words = text.lower().split()
        collective = sum(1 for w in words if w in collective_markers)
        individual = sum(1 for w in words if w in individual_markers)
        
        total = collective + individual
        if total == 0:
            return 0.5
            
        return collective / (total + 1)


@dataclass
class ContradictionFlag:
    """
    A detected contradiction between testimony statements.
    
    Types:
    - direct: Explicit denial followed by affirmation (or vice versa)
    - temporal: Conflicting time/location claims
    - logical: Mutually exclusive statements
    - coherence_drop: Significant credibility shift
    """
    statement_a: TestimonyStatement
    statement_b: TestimonyStatement
    contradiction_type: str
    severity: float  # 0.0 - 1.0
    suggested_question: str
    context: str
    
    def to_prompt(self) -> str:
        """Generate formatted cross-examination prompt"""
        return f"""
════════════════════════════════════════════════════════════
CONTRADICTION DETECTED [{self.contradiction_type.upper()}]
Severity: {self.severity:.0%}
════════════════════════════════════════════════════════════

EARLIER ({self.statement_a.timestamp.strftime('%H:%M:%S')}):
"{self.statement_a.text}"
Coherence: {self.statement_a.coherence:.2f}

NOW ({self.statement_b.timestamp.strftime('%H:%M:%S')}):
"{self.statement_b.text}"
Coherence: {self.statement_b.coherence:.2f}

────────────────────────────────────────────────────────────
SUGGESTED QUESTION:
{self.suggested_question}
────────────────────────────────────────────────────────────

Context: {self.context}
"""


class ContradictionDetector:
    """
    Real-time contradiction detection engine.
    
    Tracks all testimony by speaker and cross-references
    new statements against prior testimony.
    """
    
    def __init__(self, sensitivity: float = 0.7):
        """
        Initialize detector.
        
        sensitivity: How aggressive to flag potential contradictions (0-1)
        """
        self.statements_by_speaker: Dict[str, List[TestimonyStatement]] = {}
        self.contradictions: List[ContradictionFlag] = []
        self.sensitivity = sensitivity
        self.rose_glass = RoseGlass()
        
    def add_statement(self, speaker: str, text: str,
                      position: int) -> Optional[ContradictionFlag]:
        """
        Add new testimony statement and check for contradictions.
        
        Returns ContradictionFlag if contradiction detected, else None.
        """
        if not text.strip():
            return None
            
        statement = TestimonyStatement.create(
            speaker=speaker,
            text=text,
            position=position,
            rose_glass=self.rose_glass
        )
        
        if speaker not in self.statements_by_speaker:
            self.statements_by_speaker[speaker] = []
        
        # Check against prior statements
        contradiction = self._check_contradictions(speaker, statement)
        
        # Store statement
        self.statements_by_speaker[speaker].append(statement)
        
        if contradiction:
            self.contradictions.append(contradiction)
            
        return contradiction
    
    def _check_contradictions(self, speaker: str,
                              new_statement: TestimonyStatement) -> Optional[ContradictionFlag]:
        """Check new statement against speaker's history"""
        
        prior_statements = self.statements_by_speaker.get(speaker, [])
        
        for prior in prior_statements:
            # Check direct contradictions
            result = self._check_direct_contradiction(prior, new_statement)
            if result:
                return result
                
            # Check temporal conflicts
            result = self._check_temporal_contradiction(prior, new_statement)
            if result:
                return result
                
            # Check coherence drops
            result = self._check_coherence_drop(prior, new_statement)
            if result:
                return result
                
        return None
    
    def _check_direct_contradiction(self, prior: TestimonyStatement,
                                    current: TestimonyStatement) -> Optional[ContradictionFlag]:
        """Detect direct negation/affirmation contradictions"""
        
        prior_lower = prior.text.lower()
        current_lower = current.text.lower()
        
        never_patterns = ['never', 'did not', "didn't", 'have not', "haven't", 
                         'was not', "wasn't", 'would not', "wouldn't"]
        affirm_patterns = ['when i', 'after i', 'i did', 'i have', 'i was', 'i would']
        
        prior_negates = any(p in prior_lower for p in never_patterns)
        current_affirms = any(p in current_lower for p in affirm_patterns)
        
        current_negates = any(p in current_lower for p in never_patterns)
        prior_affirms = any(p in prior_lower for p in affirm_patterns)
        
        if (prior_negates and current_affirms) or (current_negates and prior_affirms):
            return ContradictionFlag(
                statement_a=prior,
                statement_b=current,
                contradiction_type='direct',
                severity=0.9,
                suggested_question=self._generate_direct_question(prior, current),
                context="Direct contradiction: negation vs affirmation"
            )
            
        return None
    
    def _check_temporal_contradiction(self, prior: TestimonyStatement,
                                      current: TestimonyStatement) -> Optional[ContradictionFlag]:
        """Detect temporal/location conflicts"""
        
        prior_locations = self._extract_locations(prior.text)
        current_locations = self._extract_locations(current.text)
        
        if prior_locations and current_locations:
            if prior_locations.isdisjoint(current_locations):
                return ContradictionFlag(
                    statement_a=prior,
                    statement_b=current,
                    contradiction_type='temporal',
                    severity=0.7,
                    suggested_question=self._generate_temporal_question(prior, current),
                    context="Potentially conflicting locations"
                )
                
        return None
    
    def _check_coherence_drop(self, prior: TestimonyStatement,
                              current: TestimonyStatement) -> Optional[ContradictionFlag]:
        """Detect significant coherence drops"""
        
        drop = prior.coherence - current.coherence
        
        if drop > (0.3 * self.sensitivity):
            return ContradictionFlag(
                statement_a=prior,
                statement_b=current,
                contradiction_type='coherence_drop',
                severity=min(1.0, drop * 1.5),
                suggested_question=self._generate_coherence_question(prior, current),
                context=f"Coherence dropped {drop:.0%} - possible evasion"
            )
            
        return None
    
    def _extract_locations(self, text: str) -> set:
        """Extract location references"""
        # Simplified - production would use spaCy NER
        prepositions = ['at', 'in', 'to', 'from']
        words = text.lower().split()
        locations = set()
        
        for i, word in enumerate(words):
            if word in prepositions and i + 1 < len(words):
                locations.add(words[i + 1])
                
        return locations
    
    def _generate_direct_question(self, prior: TestimonyStatement,
                                  current: TestimonyStatement) -> str:
        return f"""You testified earlier: "{prior.text[:100]}..."
But just now you said: "{current.text[:100]}..."
Can you explain this discrepancy?"""

    def _generate_temporal_question(self, prior: TestimonyStatement,
                                    current: TestimonyStatement) -> str:
        return f"""Earlier you mentioned "{prior.text[:80]}..."
Now you're saying "{current.text[:80]}..."
Which account is accurate?"""

    def _generate_coherence_question(self, prior: TestimonyStatement,
                                     current: TestimonyStatement) -> str:
        return f"""You seemed very certain when you said "{prior.text[:80]}..."
Now you appear less sure. Has something changed?"""

    def get_speaker_timeline(self, speaker: str) -> List[TestimonyStatement]:
        """Get all statements from a speaker"""
        return self.statements_by_speaker.get(speaker, [])
    
    def get_coherence_trend(self, speaker: str) -> List[float]:
        """Get coherence timeline for credibility tracking"""
        statements = self.statements_by_speaker.get(speaker, [])
        return [s.coherence for s in statements]
    
    def export_analysis(self) -> Dict:
        """Export full analysis for documentation"""
        return {
            "total_statements": sum(len(s) for s in self.statements_by_speaker.values()),
            "speakers": list(self.statements_by_speaker.keys()),
            "contradictions_found": len(self.contradictions),
            "contradiction_details": [
                {
                    "type": c.contradiction_type,
                    "severity": c.severity,
                    "speaker": c.statement_a.speaker,
                    "earlier": c.statement_a.text,
                    "later": c.statement_b.text
                }
                for c in self.contradictions
            ]
        }


# === Quick Test ===
if __name__ == "__main__":
    detector = ContradictionDetector()
    
    # Simulate testimony
    test_statements = [
        ("WITNESS", "I have never been to that restaurant"),
        ("WITNESS", "I remember the food was good when I went there"),
        ("WITNESS", "Maybe it was a different place, I'm not sure now"),
    ]
    
    for i, (speaker, text) in enumerate(test_statements):
        result = detector.add_statement(speaker, text, i)
        if result:
            print(result.to_prompt())
            
    print(f"\nTotal contradictions: {len(detector.contradictions)}")
