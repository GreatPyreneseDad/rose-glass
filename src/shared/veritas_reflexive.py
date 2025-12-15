"""
Unified Veritas Function and Reflexive Validation Wings
========================================================

Shared across: rose-glass, rose-looking-glass, RoseGlassLE

Implements:
- Veritas Function: V(P) = Σwᵢ·Eᵢ(P) / (1+D(P))
- Architect Wing: Pattern integration (Latticekeeper, Constructor, Suturer)
- Mirror Wing: Reflexive testing (Echo, Fractalist, Evaluator)

"Truth is stable only if its reflection still resonates."

Author: Christopher MacGregor bin Joseph
Date: December 2025
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Callable
from enum import Enum
from datetime import datetime
import hashlib


# ============================================================================
# VERITAS FUNCTION
# ============================================================================

class EvaluationFrame(Enum):
    """Frames through which propositions are evaluated"""
    FACTUAL = "factual"           # Empirical accuracy
    EMOTIONAL = "emotional"       # Felt truth
    SYMBOLIC = "symbolic"         # Representational meaning
    ETHICAL = "ethical"           # Moral alignment
    PHILOSOPHICAL = "philosophical"  # Deep coherence


@dataclass
class FrameEvaluation:
    """Evaluation of a proposition within a single frame"""
    frame: EvaluationFrame
    score: float  # 0-1
    weight: float  # Importance of this frame
    confidence: float
    reasoning: str


@dataclass
class VeritasResult:
    """Complete Veritas truth valuation"""
    proposition: str
    veritas_score: float  # V(P) - final truth value
    distortion_index: float  # D(P) - lens deviation
    frame_evaluations: Dict[EvaluationFrame, FrameEvaluation]
    weighted_sum: float
    is_stable_truth: bool  # True if Veritas > threshold
    timestamp: datetime


class VeritasFunction:
    """
    Truth Valuation Function
    
    V(P) = Σwᵢ·Eᵢ(P) / (1 + D(P))
    
    Where:
    - P: Proposition under examination
    - Eᵢ(P): Evaluation of P across frame i (factual, emotional, symbolic, ethical)
    - wᵢ: Weight assigned to frame i
    - D(P): Distortion index (lens deviation)
    
    This function sees not simply "true or false" but HOW and WHY
    something appears true.
    """
    
    DEFAULT_WEIGHTS = {
        EvaluationFrame.FACTUAL: 0.30,
        EvaluationFrame.EMOTIONAL: 0.20,
        EvaluationFrame.SYMBOLIC: 0.25,
        EvaluationFrame.ETHICAL: 0.25,
    }
    
    def __init__(
        self,
        weights: Optional[Dict[EvaluationFrame, float]] = None,
        stability_threshold: float = 0.6
    ):
        """
        Initialize Veritas function.
        
        Args:
            weights: Custom frame weights (must sum to 1.0)
            stability_threshold: Minimum Veritas for stable truth
        """
        self.weights = weights or self.DEFAULT_WEIGHTS
        self.stability_threshold = stability_threshold
        
        # Validate weights
        total = sum(self.weights.values())
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Weights must sum to 1.0, got {total}")
    
    def evaluate(
        self,
        proposition: str,
        distortion_index: float,
        frame_scores: Optional[Dict[EvaluationFrame, float]] = None,
        frame_evaluator: Optional[Callable] = None
    ) -> VeritasResult:
        """
        Evaluate truth value of a proposition.
        
        Args:
            proposition: The statement/claim to evaluate
            distortion_index: D(P) - lens deviation from interference analysis
            frame_scores: Pre-computed frame scores (0-1)
            frame_evaluator: Optional function to compute frame scores
            
        Returns:
            VeritasResult with complete evaluation
        """
        # Get frame evaluations
        if frame_scores:
            evaluations = self._scores_to_evaluations(proposition, frame_scores)
        elif frame_evaluator:
            evaluations = frame_evaluator(proposition)
        else:
            evaluations = self._default_evaluation(proposition)
        
        # Calculate weighted sum
        weighted_sum = sum(
            evaluations[frame].score * evaluations[frame].weight
            for frame in evaluations
        )
        
        # Apply Veritas formula
        veritas_score = weighted_sum / (1.0 + distortion_index)
        
        return VeritasResult(
            proposition=proposition,
            veritas_score=veritas_score,
            distortion_index=distortion_index,
            frame_evaluations=evaluations,
            weighted_sum=weighted_sum,
            is_stable_truth=veritas_score >= self.stability_threshold,
            timestamp=datetime.now()
        )
    
    def _scores_to_evaluations(
        self,
        proposition: str,
        scores: Dict[EvaluationFrame, float]
    ) -> Dict[EvaluationFrame, FrameEvaluation]:
        """Convert simple scores to full evaluations"""
        evaluations = {}
        
        for frame, score in scores.items():
            evaluations[frame] = FrameEvaluation(
                frame=frame,
                score=score,
                weight=self.weights.get(frame, 0.25),
                confidence=0.8,
                reasoning=f"Score provided for {frame.value} frame"
            )
        
        return evaluations
    
    def _default_evaluation(
        self,
        proposition: str
    ) -> Dict[EvaluationFrame, FrameEvaluation]:
        """Generate default balanced evaluation"""
        return {
            frame: FrameEvaluation(
                frame=frame,
                score=0.5,
                weight=self.weights[frame],
                confidence=0.5,
                reasoning="Default evaluation - no frame analyzer provided"
            )
            for frame in self.weights.keys()
        }
    
    def quick_veritas(self, distortion_index: float, composite_score: float = 0.7) -> float:
        """
        Quick Veritas calculation without frame breakdown.
        
        V(P) = S / (1 + D(P))
        
        Args:
            distortion_index: D(P) from lens deviation
            composite_score: Overall proposition strength (0-1)
            
        Returns:
            Veritas score
        """
        return composite_score / (1.0 + distortion_index)


# ============================================================================
# ARCHITECT WING - Pattern Integration
# ============================================================================

@dataclass
class InsightFragment:
    """A fragment of insight to be integrated"""
    content: str
    source: str
    coherence: float
    timestamp: datetime
    domain: str = "general"


@dataclass
class IntegratedInsight:
    """Result of Architect integration"""
    synthesis: str
    fragments_used: List[InsightFragment]
    integration_score: float  # I = ∫(f₁+f₂+...fₙ)dt
    lattice_connections: Dict[str, List[str]]
    bridges_constructed: List[Tuple[str, str]]
    ruptures_sutured: List[str]


class ArchitectWing:
    """
    The Architect assembles new insights from fragments.
    It sees systems, not answers.
    
    Components:
    - Latticekeeper: Holds multi-dimensional frameworks of knowledge
    - Constructor: Builds bridges between seemingly unrelated truths
    - Suturer: Repairs broken patterns when contradiction ruptures understanding
    
    Root Equation:
    I = ∫₀ᵗ (f₁ + f₂ + ... + fₙ) dt
    
    Integration over time—across insight sources—builds total understanding.
    """
    
    def __init__(self):
        self.lattice: Dict[str, List[InsightFragment]] = {}
        self.bridges: List[Tuple[str, str, float]] = []  # (domain1, domain2, strength)
        self.sutures: List[str] = []
        
    def latticekeeper_store(self, fragment: InsightFragment):
        """Store insight in multi-dimensional lattice"""
        if fragment.domain not in self.lattice:
            self.lattice[fragment.domain] = []
        self.lattice[fragment.domain].append(fragment)
    
    def constructor_bridge(
        self,
        domain1: str,
        domain2: str,
        connection_insight: str
    ) -> bool:
        """Build bridge between two domains"""
        if domain1 not in self.lattice or domain2 not in self.lattice:
            return False
        
        # Calculate bridge strength based on fragment coherences
        frags1 = self.lattice[domain1]
        frags2 = self.lattice[domain2]
        
        if not frags1 or not frags2:
            return False
        
        avg_coherence = (
            np.mean([f.coherence for f in frags1]) +
            np.mean([f.coherence for f in frags2])
        ) / 2
        
        self.bridges.append((domain1, domain2, avg_coherence))
        return True
    
    def suturer_repair(
        self,
        rupture_description: str,
        fragments: List[InsightFragment]
    ) -> Optional[str]:
        """
        Repair a rupture in understanding caused by contradiction.
        
        Returns synthesized resolution or None if unresolvable.
        """
        if len(fragments) < 2:
            return None
        
        # Find highest coherence fragments
        sorted_frags = sorted(fragments, key=lambda f: f.coherence, reverse=True)
        
        # Attempt synthesis
        synthesis = f"Resolution of '{rupture_description}': "
        synthesis += " ↔ ".join([f.content[:50] for f in sorted_frags[:3]])
        
        self.sutures.append(rupture_description)
        
        return synthesis
    
    def integrate(
        self,
        fragments: List[InsightFragment],
        time_window: float = 1.0
    ) -> IntegratedInsight:
        """
        Perform full integration across fragments.
        
        I = ∫₀ᵗ (f₁ + f₂ + ... + fₙ) dt
        """
        # Store all fragments
        for frag in fragments:
            self.latticekeeper_store(frag)
        
        # Build bridges between domains
        domains = list(self.lattice.keys())
        bridges_built = []
        
        for i, d1 in enumerate(domains):
            for d2 in domains[i+1:]:
                if self.constructor_bridge(d1, d2, f"Connection: {d1}-{d2}"):
                    bridges_built.append((d1, d2))
        
        # Calculate integration score (discrete approximation of integral)
        coherences = [f.coherence for f in fragments]
        integration_score = np.trapz(coherences, dx=time_window/len(coherences)) if coherences else 0.0
        
        # Generate synthesis
        synthesis = self._generate_synthesis(fragments)
        
        return IntegratedInsight(
            synthesis=synthesis,
            fragments_used=fragments,
            integration_score=integration_score,
            lattice_connections={d: [f.content[:30] for f in frags] 
                               for d, frags in self.lattice.items()},
            bridges_constructed=bridges_built,
            ruptures_sutured=self.sutures.copy()
        )
    
    def _generate_synthesis(self, fragments: List[InsightFragment]) -> str:
        """Generate synthesis from fragments"""
        if not fragments:
            return "No fragments to synthesize"
        
        # Weight by coherence
        weighted = sorted(fragments, key=lambda f: f.coherence, reverse=True)
        top_insights = weighted[:3]
        
        return "Integrated understanding: " + " → ".join(
            [f.content[:40] for f in top_insights]
        )


# ============================================================================
# MIRROR WING - Reflexive Testing
# ============================================================================

@dataclass
class ReflectionResult:
    """Result of Mirror Wing reflection"""
    original: str
    reflection: str
    resonance_score: float  # How well reflection matches original
    is_stable: bool
    distortions: List[str]


class MirrorWing:
    """
    The Mirror reflects back what has been said, decided, or revealed—
    so it may be tested, re-encountered, and improved.
    
    Components:
    - Echo: Restates outputs in alternate framing
    - Fractalist: Analyzes self-similarity and recursion
    - Evaluator: Validates whether outputs align across iterations
    
    Root Equation:
    Rᶠ = f(P) ↔ f*(P)
    
    Truth is stable only if its reflection still resonates.
    The Mirror tests for that fidelity.
    """
    
    def __init__(self, resonance_threshold: float = 0.7):
        self.resonance_threshold = resonance_threshold
        self.reflection_history: List[ReflectionResult] = []
    
    def echo(self, original: str, frame: str = "alternate") -> str:
        """
        Restate output in alternate framing.
        
        The Echo doesn't repeat—it translates to test understanding.
        """
        # Simple transformation for demonstration
        # In production, this would use LLM or sophisticated NLP
        
        transformations = {
            "alternate": lambda s: f"In other words: {s}",
            "inverse": lambda s: f"The opposite perspective: {self._invert_sentiment(s)}",
            "abstract": lambda s: f"At a higher level: {s[:len(s)//2]}...",
            "concrete": lambda s: f"Specifically: {s}"
        }
        
        transform = transformations.get(frame, transformations["alternate"])
        return transform(original)
    
    def fractalist_analyze(self, content: str, depth: int = 3) -> Dict[str, Any]:
        """
        Analyze self-similarity and recursive patterns.
        
        Checks if the pattern repeats at different scales.
        """
        # Extract patterns at different scales
        words = content.split()
        
        patterns = {
            'word_level': self._extract_patterns(words, 1),
            'phrase_level': self._extract_patterns(words, 3),
            'sentence_level': self._extract_patterns(content.split('.'), 1)
        }
        
        # Calculate self-similarity score
        similarities = []
        for level, pats in patterns.items():
            if len(pats) > 1:
                unique_ratio = len(set(pats)) / len(pats)
                similarities.append(1 - unique_ratio)  # More repetition = more self-similar
        
        self_similarity = np.mean(similarities) if similarities else 0.0
        
        return {
            'patterns': patterns,
            'self_similarity': self_similarity,
            'is_fractal': self_similarity > 0.3,
            'recursion_depth': depth
        }
    
    def evaluator_validate(
        self,
        outputs: List[str],
        tolerance: float = 0.2
    ) -> Tuple[bool, float]:
        """
        Validate whether outputs align across iterations.
        
        Returns (is_aligned, alignment_score)
        """
        if len(outputs) < 2:
            return True, 1.0
        
        # Compare consecutive outputs
        similarities = []
        for i in range(len(outputs) - 1):
            sim = self._calculate_similarity(outputs[i], outputs[i+1])
            similarities.append(sim)
        
        alignment_score = np.mean(similarities)
        is_aligned = alignment_score >= (1.0 - tolerance)
        
        return is_aligned, alignment_score
    
    def reflect(self, proposition: str) -> ReflectionResult:
        """
        Full reflection cycle: Echo → Fractalist → Evaluator
        
        Rᶠ = f(P) ↔ f*(P)
        """
        # Echo in alternate frame
        reflection = self.echo(proposition, "alternate")
        
        # Fractalist analysis
        fractal = self.fractalist_analyze(proposition)
        
        # Evaluate alignment
        is_aligned, alignment = self.evaluator_validate([proposition, reflection])
        
        # Calculate resonance
        resonance = (alignment + fractal['self_similarity']) / 2
        
        # Identify distortions
        distortions = []
        if not is_aligned:
            distortions.append("Alignment drift detected")
        if fractal['self_similarity'] < 0.2:
            distortions.append("Low self-similarity")
        
        result = ReflectionResult(
            original=proposition,
            reflection=reflection,
            resonance_score=resonance,
            is_stable=resonance >= self.resonance_threshold,
            distortions=distortions
        )
        
        self.reflection_history.append(result)
        
        return result
    
    def _invert_sentiment(self, text: str) -> str:
        """Simple sentiment inversion for demonstration"""
        inversions = {
            'good': 'bad', 'bad': 'good',
            'true': 'false', 'false': 'true',
            'always': 'never', 'never': 'always',
            'high': 'low', 'low': 'high'
        }
        
        words = text.split()
        inverted = [inversions.get(w.lower(), w) for w in words]
        return ' '.join(inverted)
    
    def _extract_patterns(self, elements: List[str], window: int) -> List[str]:
        """Extract patterns from elements with given window size"""
        if len(elements) < window:
            return elements
        
        patterns = []
        for i in range(len(elements) - window + 1):
            pattern = ' '.join(str(e) for e in elements[i:i+window])
            patterns.append(pattern.lower().strip())
        
        return patterns
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union)


# ============================================================================
# INTEGRATED VALIDATION SYSTEM
# ============================================================================

class ReflexiveValidationSystem:
    """
    Combined Architect + Mirror + Veritas system.
    
    Provides complete truth validation through:
    1. Architect integration of insights
    2. Mirror reflection testing
    3. Veritas truth scoring
    """
    
    def __init__(self):
        self.architect = ArchitectWing()
        self.mirror = MirrorWing()
        self.veritas = VeritasFunction()
    
    def validate_insight(
        self,
        insight: str,
        distortion_index: float,
        source_fragments: Optional[List[InsightFragment]] = None
    ) -> Dict[str, Any]:
        """
        Full validation pipeline for an insight.
        """
        # Architect integration
        if source_fragments:
            integration = self.architect.integrate(source_fragments)
        else:
            integration = None
        
        # Mirror reflection
        reflection = self.mirror.reflect(insight)
        
        # Veritas scoring
        veritas_result = self.veritas.evaluate(
            proposition=insight,
            distortion_index=distortion_index
        )
        
        # Combined assessment
        is_validated = (
            reflection.is_stable and 
            veritas_result.is_stable_truth
        )
        
        return {
            'insight': insight,
            'is_validated': is_validated,
            'veritas_score': veritas_result.veritas_score,
            'resonance_score': reflection.resonance_score,
            'distortion_index': distortion_index,
            'integration_score': integration.integration_score if integration else None,
            'reflection': reflection,
            'veritas_result': veritas_result,
            'integration': integration
        }


# === Tests ===
if __name__ == "__main__":
    print("=" * 70)
    print("VERITAS FUNCTION AND REFLEXIVE VALIDATION")
    print("=" * 70)
    
    # Test Veritas
    print("\n--- VERITAS FUNCTION ---")
    veritas = VeritasFunction()
    
    test_propositions = [
        ("Truth persists across all perspectives.", 0.05),  # Low distortion
        ("This specific interpretation is correct.", 0.35),  # High distortion
        ("The pattern reveals itself through attention.", 0.15),
    ]
    
    for prop, distortion in test_propositions:
        result = veritas.evaluate(prop, distortion)
        print(f"\nProposition: {prop[:50]}...")
        print(f"  D(P) = {result.distortion_index:.2f}")
        print(f"  V(P) = {result.veritas_score:.3f}")
        print(f"  Stable: {result.is_stable_truth}")
    
    # Test Mirror Wing
    print("\n--- MIRROR WING ---")
    mirror = MirrorWing()
    
    test_statements = [
        "Truth is always consistent across time.",
        "This moment defines everything that follows.",
    ]
    
    for stmt in test_statements:
        reflection = mirror.reflect(stmt)
        print(f"\nOriginal: {stmt}")
        print(f"  Reflection: {reflection.reflection[:60]}...")
        print(f"  Resonance: {reflection.resonance_score:.3f}")
        print(f"  Stable: {reflection.is_stable}")
    
    # Test Architect Wing
    print("\n--- ARCHITECT WING ---")
    architect = ArchitectWing()
    
    fragments = [
        InsightFragment("Coherence emerges from attention", "observation", 0.8, datetime.now(), "philosophy"),
        InsightFragment("Patterns repeat at multiple scales", "analysis", 0.75, datetime.now(), "mathematics"),
        InsightFragment("Truth survives translation", "reflection", 0.85, datetime.now(), "philosophy"),
    ]
    
    integration = architect.integrate(fragments)
    print(f"\nIntegration Score: {integration.integration_score:.3f}")
    print(f"Synthesis: {integration.synthesis}")
    print(f"Bridges: {integration.bridges_constructed}")
    
    # Test combined system
    print("\n--- COMBINED VALIDATION ---")
    system = ReflexiveValidationSystem()
    
    validation = system.validate_insight(
        "Universal patterns transcend cultural interpretation",
        distortion_index=0.08,
        source_fragments=fragments
    )
    
    print(f"\nInsight: {validation['insight'][:50]}...")
    print(f"  Validated: {validation['is_validated']}")
    print(f"  Veritas: {validation['veritas_score']:.3f}")
    print(f"  Resonance: {validation['resonance_score']:.3f}")
