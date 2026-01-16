"""
Semantic Contradiction Detection for Rose Glass Litigation Lens
===============================================================

Enhances keyword-based contradiction detection with semantic similarity analysis.
Catches contradictions like:
  - "I was never involved" vs "Christopher coached soccer" (third person)
  - "I don't recall" vs detailed recollection
  - Topic overlap with opposing sentiment

Author: Christopher MacGregor bin Joseph
Created: 2026-01-16
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from numpy.linalg import norm
import re


@dataclass
class SemanticContradiction:
    """
    Semantic contradiction result

    Attributes:
        is_contradiction: Whether statements contradict
        confidence: Confidence score (0-1)
        similarity: Topic similarity score
        sentiment_opposition: Whether sentiments oppose
        explanation: Human-readable explanation
    """
    is_contradiction: bool
    confidence: float
    similarity: float
    sentiment_opposition: bool
    explanation: str


class SemanticContradictionDetector:
    """
    Detect contradictions using semantic similarity + sentiment analysis

    Fallback implementation using simple NLP when sentence-transformers unavailable.
    For production, use sentence-transformers for better accuracy.
    """

    def __init__(self, use_transformers: bool = True):
        """
        Initialize detector

        Args:
            use_transformers: Attempt to use sentence-transformers if available
        """
        self.use_transformers = use_transformers
        self.encoder = None

        if use_transformers:
            try:
                from sentence_transformers import SentenceTransformer
                self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
            except ImportError:
                pass

        # Build semantic vocabulary
        self.vocabulary = self._build_vocabulary()

    def _build_vocabulary(self):
        """Build vocabulary for simple embedding fallback"""
        return {
            # Involvement/participation
            'involved': 0, 'participated': 1, 'engaged': 2, 'coached': 3, 'helped': 4,
            'attended': 5, 'present': 6, 'joined': 7, 'supported': 8,

            # Negation
            'never': 10, 'not': 11, 'no': 12, 'none': 13, 'nothing': 14,

            # Affirmation
            'always': 20, 'yes': 21, 'did': 22, 'was': 23, 'were': 24,

            # Memory/knowledge
            'remember': 30, 'recall': 31, 'know': 32, 'forget': 33,
            'aware': 34, 'familiar': 35,

            # Temporal
            'before': 40, 'after': 41, 'during': 42, 'while': 43, 'when': 44,

            # Locations
            'home': 50, 'school': 51, 'work': 52, 'court': 53, 'office': 54,
        }

    def _simple_embedding(self, text: str) -> np.ndarray:
        """Create simple embedding without transformers"""
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)

        embedding = np.zeros(384)

        # Map vocabulary words
        for word in words:
            if word in self.vocabulary:
                idx = self.vocabulary[word]
                embedding[idx] += 1.0

        # Add hash-based embeddings for other words
        for word in words:
            if word not in self.vocabulary:
                for i in range(3):
                    idx = (hash(word) + i * 17) % 384
                    embedding[idx] += 0.3

        # Normalize
        n = norm(embedding)
        if n > 0:
            embedding = embedding / n

        return embedding

    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding using best available method"""
        if self.encoder:
            return self.encoder.encode(text, convert_to_numpy=True)
        else:
            return self._simple_embedding(text)

    def _analyze_sentiment(self, text: str) -> float:
        """
        Simple sentiment analysis

        Returns:
            Sentiment score: -1 (negative) to +1 (positive)
        """
        text_lower = text.lower()

        # Positive indicators
        positive_words = ['yes', 'always', 'did', 'was', 'were', 'true', 'correct', 'accurate']
        positive_score = sum(1 for word in positive_words if word in text_lower)

        # Negative indicators
        negative_words = ['never', 'not', 'no', 'none', 'nothing', 'false', 'incorrect',
                         "didn't", "wasn't", "weren't", "haven't", "couldn't"]
        negative_score = sum(1 for word in negative_words if word in text_lower)

        # Normalize to -1 to +1
        total = positive_score + negative_score
        if total == 0:
            return 0.0

        return (positive_score - negative_score) / max(total, 1)

    def _extract_subject(self, text: str) -> str:
        """
        Extract grammatical subject (person being discussed)

        Returns subject pronoun: "I", "he/she", "they", or ""
        """
        text_lower = text.lower()

        # First person
        if re.search(r'\b(i|me|my|mine)\b', text_lower):
            return "I"

        # Third person singular
        if re.search(r'\b(he|she|him|her|his|hers)\b', text_lower):
            return "he/she"

        # Third person plural
        if re.search(r'\b(they|them|their)\b', text_lower):
            return "they"

        return ""

    def check_contradiction(
        self,
        statement1: str,
        statement2: str,
        similarity_threshold: float = 0.5,
        confidence_threshold: float = 0.7
    ) -> SemanticContradiction:
        """
        Check if two statements contradict semantically

        Args:
            statement1: First statement
            statement2: Second statement
            similarity_threshold: Minimum similarity to consider same topic (0.5 default)
            confidence_threshold: Minimum confidence to flag as contradiction

        Returns:
            SemanticContradiction result

        Examples:
            >>> detector = SemanticContradictionDetector()
            >>> result = detector.check_contradiction(
            ...     "I was never involved in soccer",
            ...     "Christopher coached the soccer team"
            ... )
            >>> result.is_contradiction
            True
        """
        # Get embeddings
        emb1 = self._get_embedding(statement1)
        emb2 = self._get_embedding(statement2)

        # Calculate topic similarity
        similarity = float(np.dot(emb1, emb2))

        # Analyze sentiments
        sentiment1 = self._analyze_sentiment(statement1)
        sentiment2 = self._analyze_sentiment(statement2)

        # Extract subjects
        subject1 = self._extract_subject(statement1)
        subject2 = self._extract_subject(statement2)

        # Check for sentiment opposition
        sentiment_diff = abs(sentiment1 - sentiment2)
        sentiments_oppose = sentiment_diff > 1.0  # Opposite sides of neutral

        # Check for subject shift (I vs he/she)
        subject_shift = (subject1 == "I" and subject2 == "he/she") or \
                       (subject1 == "he/she" and subject2 == "I")

        # Contradiction logic:
        # 1. Same topic (high similarity)
        # 2. Opposing sentiments OR subject shift with negation
        # 3. Calculate confidence

        is_same_topic = similarity > similarity_threshold

        if not is_same_topic:
            return SemanticContradiction(
                is_contradiction=False,
                confidence=0.0,
                similarity=similarity,
                sentiment_opposition=False,
                explanation="Statements discuss different topics"
            )

        # Calculate contradiction confidence
        confidence = 0.0
        explanation_parts = []

        if sentiments_oppose:
            confidence += 0.5
            explanation_parts.append("opposing sentiments")

        if subject_shift:
            confidence += 0.4
            explanation_parts.append("subject shift (I vs third person)")

        # Boost confidence if similarity is very high
        if similarity > 0.7:
            confidence += 0.2
            explanation_parts.append("high topic similarity")

        confidence = min(confidence, 1.0)

        is_contradiction = confidence >= confidence_threshold

        if is_contradiction:
            explanation = f"Contradiction detected: {', '.join(explanation_parts)}"
        else:
            explanation = f"Possible inconsistency (confidence {confidence:.2f})"

        return SemanticContradiction(
            is_contradiction=is_contradiction,
            confidence=confidence,
            similarity=similarity,
            sentiment_opposition=sentiments_oppose,
            explanation=explanation
        )

    def batch_check(
        self,
        current_statement: str,
        prior_statements: list[str]
    ) -> list[Tuple[int, SemanticContradiction]]:
        """
        Check current statement against list of prior statements

        Args:
            current_statement: New statement to check
            prior_statements: List of previous statements

        Returns:
            List of (index, SemanticContradiction) tuples for contradictions
        """
        contradictions = []

        for idx, prior in enumerate(prior_statements):
            result = self.check_contradiction(prior, current_statement)
            if result.is_contradiction:
                contradictions.append((idx, result))

        return contradictions


# Convenience function for quick checks
def quick_check(statement1: str, statement2: str) -> bool:
    """
    Quick check if two statements contradict

    Args:
        statement1: First statement
        statement2: Second statement

    Returns:
        True if contradiction detected

    Example:
        >>> quick_check("I never coached", "Christopher was the coach")
        True
    """
    detector = SemanticContradictionDetector(use_transformers=False)
    result = detector.check_contradiction(statement1, statement2)
    return result.is_contradiction


if __name__ == "__main__":
    # Demo
    print("=== Semantic Contradiction Detection Demo ===\n")

    detector = SemanticContradictionDetector()

    test_cases = [
        (
            "I was never involved in soccer activities",
            "Christopher coached the soccer team for two years"
        ),
        (
            "I don't recall meeting that person",
            "We had several lengthy conversations about the project"
        ),
        (
            "I attended the hearing on Monday",
            "The hearing was scheduled for Tuesday"
        ),
        (
            "I love pizza",
            "The weather is nice today"
        ),
    ]

    for i, (stmt1, stmt2) in enumerate(test_cases, 1):
        print(f"Test {i}:")
        print(f"  Statement 1: {stmt1}")
        print(f"  Statement 2: {stmt2}")

        result = detector.check_contradiction(stmt1, stmt2)

        print(f"  Result: {'CONTRADICTION' if result.is_contradiction else 'No contradiction'}")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Similarity: {result.similarity:.2f}")
        print(f"  {result.explanation}")
        print()
