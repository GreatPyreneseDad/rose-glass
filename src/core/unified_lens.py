"""
Unified Rose Glass Lens
Combines CERATA perception with GCT analysis

Author: Christopher MacGregor bin Joseph
Created: January 2026
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import re


@dataclass
class CoherenceResult:
    """Result of Rose Glass analysis"""
    psi: float      # Internal consistency
    rho: float      # Accumulated wisdom
    q: float        # Emotional activation
    f: float        # Social belonging
    coherence: float
    flags: List[str]
    
    def to_dict(self) -> Dict:
        return {
            'psi': round(self.psi, 3),
            'rho': round(self.rho, 3),
            'q': round(self.q, 3),
            'f': round(self.f, 3),
            'coherence': round(self.coherence, 3),
            'flags': self.flags
        }


class UnifiedRoseGlass:
    """
    Rose Glass lens for both code and text analysis.
    Used by CERATA for prey perception and litigation for testimony analysis.
    
    The Rose Glass is not a quality assessment tool - it is a mathematical lens
    through which synthetic minds can perceive and interpret emotional, social,
    and intellectual patterns of organic intelligence.
    """
    
    def __init__(self, km: float = 0.2, ki: float = 0.8):
        self.km = km  # Saturation constant
        self.ki = ki  # Inhibition constant
        
    def biological_optimization(self, q_raw: float) -> float:
        """
        Apply Michaelis-Menten kinetics to prevent extremism.
        Mimics natural saturation curves in biological systems.
        """
        if q_raw <= 0:
            return 0.0
        return q_raw / (self.km + q_raw + (q_raw**2 / self.ki))
    
    def calculate_coherence(self, psi: float, rho: float, q: float, f: float,
                           coupling_strength: float = 0.15) -> float:
        """
        Calculate coherence score.
        C = Ψ + (ρ × Ψ) + q_opt + (f × Ψ) + coupling
        
        The resulting score (0-4) represents intensity of pattern visibility
        through this lens, not a judgment of quality or worth.
        """
        q_opt = self.biological_optimization(q)
        coupling = coupling_strength * rho * q_opt
        coherence = psi + (rho * psi) + q_opt + (f * psi) + coupling
        return min(coherence, 4.0)
    
    def extract_psi(self, text: str) -> float:
        """
        Extract internal consistency (Ψ) from text.
        Measures harmonic alignment of thoughts and expressions.
        """
        # Check for contradictory markers
        contradiction_markers = [
            'but actually', 'however', 'on the other hand',
            'contradicts', 'inconsistent', 'despite saying',
            'nevertheless', 'yet', 'although'
        ]
        contradiction_count = sum(1 for m in contradiction_markers if m.lower() in text.lower())
        
        # Check structural consistency (sentence length variance)
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        if len(sentences) > 1:
            lengths = [len(s.split()) for s in sentences]
            variance = np.var(lengths) if lengths else 0
            structure_score = max(0, 1 - (variance / 100))
        else:
            structure_score = 0.5
        
        # Check for logical connectors (indicates structured thinking)
        logical_markers = ['therefore', 'because', 'thus', 'hence', 'consequently', 'as a result']
        logical_count = sum(1 for m in logical_markers if m.lower() in text.lower())
        logical_bonus = min(0.2, logical_count * 0.05)
            
        psi = max(0, 1 - (contradiction_count * 0.15)) * structure_score + logical_bonus
        return min(psi, 1.0)
    
    def extract_rho(self, text: str) -> float:
        """
        Extract accumulated wisdom (ρ) from text.
        Measures depth of integrated experience and knowledge.
        """
        # Evidence markers
        evidence_markers = [
            'specifically', 'for example', 'evidence shows',
            'documented', 'recorded', 'verified', 'confirmed',
            'exhibit', 'pursuant to', 'according to'
        ]
        evidence_count = sum(1 for m in evidence_markers if m.lower() in text.lower())
        
        # Specificity: numbers, dates, names
        numbers = len(re.findall(r'\b\d+\b', text))
        dates = len(re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', text))
        dollar_amounts = len(re.findall(r'\$[\d,]+\.?\d*', text))
        percentages = len(re.findall(r'\d+\.?\d*\s*%', text))
        
        # Causal reasoning
        causal_markers = ['because', 'therefore', 'consequently', 'as a result', 'due to', 'since']
        causal_count = sum(1 for m in causal_markers if m.lower() in text.lower())
        
        # Legal/formal citations
        citations = len(re.findall(r'MCL|MCR|USC|CFR|\d+\s+F\.\d+|\d+\s+Mich\.', text))
        
        rho = min(1.0, 
                  (evidence_count * 0.08) + 
                  (numbers * 0.015) + 
                  (dates * 0.1) + 
                  (dollar_amounts * 0.08) +
                  (percentages * 0.05) +
                  (causal_count * 0.08) +
                  (citations * 0.12))
        return rho
    
    def extract_q(self, text: str) -> float:
        """
        Extract emotional activation energy (q) from text.
        Measures emotional and ethical resonance.
        """
        # High-activation words
        high_activation = [
            'never', 'always', 'absolutely', 'completely', 'totally',
            'terrified', 'furious', 'desperate', 'urgent', 'critical',
            'abuse', 'threat', 'danger', 'fear', 'harm', 'attack',
            'victim', 'traumatized', 'horrified', 'outrageous'
        ]
        
        # Moderate activation
        moderate_activation = [
            'concerned', 'worried', 'upset', 'frustrated', 'anxious',
            'important', 'significant', 'serious', 'troubled', 'distressed'
        ]
        
        # Low activation (calming)
        low_activation = [
            'calmly', 'reasonably', 'objectively', 'factually',
            'respectfully', 'appropriately'
        ]
        
        text_lower = text.lower()
        high_count = sum(1 for w in high_activation if w in text_lower)
        mod_count = sum(1 for w in moderate_activation if w in text_lower)
        low_count = sum(1 for w in low_activation if w in text_lower)
        
        # Exclamation marks and caps
        exclamations = text.count('!')
        caps_words = len([w for w in text.split() if w.isupper() and len(w) > 2])
        
        q_raw = ((high_count * 0.12) + 
                 (mod_count * 0.06) + 
                 (exclamations * 0.08) + 
                 (caps_words * 0.04) -
                 (low_count * 0.05))
        return max(0, min(q_raw, 1.0))
    
    def extract_f(self, text: str) -> float:
        """
        Extract social belonging architecture (f) from text.
        Measures patterns of connection and community.
        """
        text_lower = text.lower()
        words = text_lower.split()
        
        # Collective pronouns
        collective = ['we', 'our', 'us', 'together', 'community', 'family', 'children']
        collective_count = sum(1 for w in collective if w in words)
        
        # Isolating language
        isolating = ['alone', 'isolated', 'abandoned', 'rejected', 'excluded']
        isolating_count = sum(1 for w in isolating if w in words)
        
        # Positioning markers
        positioning = ['against', 'versus', 'opposed to', 'in support of', 'aligned with']
        position_count = sum(1 for m in positioning if m in text_lower)
        
        # Relational terms
        relational = ['relationship', 'connection', 'bond', 'trust', 'loyalty', 'father', 'mother', 'daughter', 'son']
        relational_count = sum(1 for w in relational if w in text_lower)
        
        # Court/institutional alignment
        institutional = ['this court', 'your honor', 'pursuant to', 'respectfully']
        institutional_count = sum(1 for m in institutional if m in text_lower)
        
        f = min(1.0, 
                (collective_count * 0.08) + 
                (position_count * 0.1) + 
                (relational_count * 0.08) +
                (institutional_count * 0.1) -
                (isolating_count * 0.1))
        return max(0, f)
    
    def analyze(self, text: str) -> CoherenceResult:
        """
        Full Rose Glass analysis of text.
        Returns CoherenceResult with all dimensions.
        """
        if not text or not text.strip():
            return CoherenceResult(
                psi=0, rho=0, q=0, f=0,
                coherence=0, flags=["EMPTY_INPUT"]
            )
        
        psi = self.extract_psi(text)
        rho = self.extract_rho(text)
        q = self.extract_q(text)
        f = self.extract_f(text)
        
        coherence = self.calculate_coherence(psi, rho, q, f)
        
        # Generate flags
        flags = []
        if psi < 0.4:
            flags.append("LOW_CONSISTENCY")
        if psi < 0.5 and psi >= 0.4:
            flags.append("CONSISTENCY_WARNING")
        if rho < 0.3:
            flags.append("LOW_SPECIFICITY")
        if rho < 0.4 and rho >= 0.3:
            flags.append("SPECIFICITY_WARNING")
        if q > 0.7:
            flags.append("HIGH_EMOTIONAL_ACTIVATION")
        if q > 0.7 and rho < 0.4:
            flags.append("EMOTIONAL_WITHOUT_EVIDENCE")
        if f < 0.2:
            flags.append("ISOLATION_PATTERN")
        if coherence < 1.5:
            flags.append("COHERENCE_WARNING")
        if coherence < 1.0:
            flags.append("COHERENCE_CRITICAL")
            
        return CoherenceResult(
            psi=psi, rho=rho, q=q, f=f,
            coherence=coherence, flags=flags
        )
    
    def compare(self, text_a: str, text_b: str) -> Dict:
        """
        Compare coherence between two texts.
        Useful for detecting contradictions between prior and current testimony.
        """
        result_a = self.analyze(text_a)
        result_b = self.analyze(text_b)
        
        psi_change = result_b.psi - result_a.psi
        rho_change = result_b.rho - result_a.rho
        q_change = result_b.q - result_a.q
        
        delta = {
            'psi_change': round(psi_change, 3),
            'rho_change': round(rho_change, 3),
            'q_change': round(q_change, 3),
            'f_change': round(result_b.f - result_a.f, 3),
            'coherence_change': round(result_b.coherence - result_a.coherence, 3),
            'contradiction_likely': psi_change < -0.2,
            'specificity_dropped': rho_change < -0.15,
            'emotional_spike': q_change > 0.2
        }
        
        return {
            'text_a': result_a.to_dict(),
            'text_b': result_b.to_dict(),
            'delta': delta
        }
    
    def analyze_for_attack(self, text: str) -> Dict:
        """
        Analyze text specifically for identifying attack vectors.
        Returns analysis plus suggested attack strategies.
        """
        result = self.analyze(text)
        
        attack_vectors = []
        
        if result.psi < 0.5:
            attack_vectors.append({
                'type': 'INTERNAL_CONTRADICTION',
                'severity': 1 - result.psi,
                'approach': 'Quote earlier vs later statements, ask which is accurate'
            })
        
        if result.rho < 0.4:
            attack_vectors.append({
                'type': 'LACKS_SPECIFICITY',
                'severity': 1 - result.rho,
                'approach': 'Demand specific dates, amounts, documents'
            })
        
        if result.q > 0.7 and result.rho < 0.5:
            attack_vectors.append({
                'type': 'EMOTIONAL_WITHOUT_EVIDENCE',
                'severity': result.q,
                'approach': 'Acknowledge emotion, redirect to evidence gaps'
            })
        
        if result.coherence < 1.5:
            attack_vectors.append({
                'type': 'COHERENCE_COLLAPSE',
                'severity': 1.5 - result.coherence,
                'approach': 'Let testimony stand, highlight gaps in closing'
            })
        
        return {
            'analysis': result.to_dict(),
            'attack_vectors': attack_vectors,
            'primary_weakness': attack_vectors[0]['type'] if attack_vectors else None
        }


# Singleton instance
_lens = None

def get_lens() -> UnifiedRoseGlass:
    """Get or create the singleton Rose Glass lens instance."""
    global _lens
    if _lens is None:
        _lens = UnifiedRoseGlass()
    return _lens


if __name__ == "__main__":
    # Test
    lens = get_lens()
    
    test_texts = [
        """On January 15, 2025, the defendant specifically stated under oath that 
        he had never contacted the plaintiff. However, documented phone records 
        from Exhibit A show 47 calls between December 2024 and January 2025.""",
        
        """He is ALWAYS lying and has NEVER told the truth! This is absolutely 
        outrageous and completely unacceptable!!!""",
        
        """Pursuant to MCL 722.23, the court shall consider the reasonable 
        preference of the child, if the court considers the child to be of 
        sufficient age to express preference. Specifically, the minor child 
        M.M., age 8, expressed clear preference to reside with Defendant."""
    ]
    
    for i, text in enumerate(test_texts):
        print(f"\n{'='*60}")
        print(f"Test {i+1}:")
        print(f"Text: {text[:80]}...")
        result = lens.analyze(text)
        print(f"Result: {result.to_dict()}")
        
        attack = lens.analyze_for_attack(text)
        print(f"Attack vectors: {[v['type'] for v in attack['attack_vectors']]}")
