#!/usr/bin/env python3
"""
Emotionally Informed RAG Pipeline - Example Implementation

Demonstrates integration of:
- LLM Zoomcamp RAG patterns
- Rose Glass emotional intelligence
- RoseGlassLE advanced features
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np

# Add Rose Glass to path
sys.path.append(str(Path.home() / "rose-glass" / "src"))
sys.path.append(str(Path.home() / "RoseGlassLE" / "src"))

# Rose Glass imports
from core.rose_glass_v2 import RoseGlassV2
from core.trust_signal_detector import TrustSignalDetector
from core.mission_mode_detector import MissionModeDetector

# RoseGlassLE imports
from core.temporal_dimension import TemporalAnalyzer
from core.gradient_tracker import PatternGradientTracker, PatternSnapshot
from cultural_calibrations.neurodivergent_base import (
    AutismSpectrumCalibration,
    ADHDCalibration,
    HighStressTraumaCalibration
)


@dataclass
class EmotionalSignature:
    """Emotional signature of text"""
    psi: float  # Internal consistency
    rho: float  # Wisdom depth
    q: float    # Emotional activation
    f: float    # Social belonging
    tau: float  # Temporal depth
    lens: str   # Cultural lens
    confidence: str


@dataclass
class Document:
    """Document with emotional signature"""
    id: str
    content: str
    title: str
    rag_score: float  # From RAG retrieval
    emotional_signature: Optional[EmotionalSignature] = None
    emotional_match: float = 0.0
    temporal_match: float = 0.0
    wisdom_depth: float = 0.0
    final_score: float = 0.0


class EmotionallyInformedRAG:
    """
    RAG system with emotional intelligence

    Combines:
    - Semantic search (vector embeddings)
    - Keyword search (BM25)
    - Emotional pattern translation (Rose Glass)
    - Temporal depth analysis (RoseGlassLE)
    - Real-time gradient tracking
    """

    def __init__(
        self,
        vector_db_host: str = "localhost",
        vector_db_port: int = 6333,
        es_host: str = "localhost",
        es_port: int = 9200,
        cultural_lens: str = "modern_digital"
    ):
        """Initialize the emotionally informed RAG pipeline"""

        # Initialize Rose Glass components
        self.rose_glass = RoseGlassV2()
        self.rose_glass.select_lens(cultural_lens)

        # Initialize RoseGlassLE components
        self.temporal_analyzer = TemporalAnalyzer()
        self.gradient_tracker = PatternGradientTracker()

        # Initialize context detectors
        self.trust_detector = TrustSignalDetector()
        self.mission_detector = MissionModeDetector()

        # Initialize neurodivergent calibrations
        self.neurodivergent_calibrations = {
            'autism': AutismSpectrumCalibration(),
            'adhd': ADHDCalibration(),
            'high_stress': HighStressTraumaCalibration()
        }

        # RAG database connections (simulated for example)
        self.vector_db = None  # Would be QdrantClient in production
        self.elasticsearch = None  # Would be Elasticsearch in production

        print("✅ Emotionally Informed RAG initialized")
        print(f"   Cultural lens: {cultural_lens}")

    def analyze_query(self, query: str) -> Dict:
        """
        Analyze query through Rose Glass lens

        Returns emotional signature and context type
        """
        print(f"\n🔍 Analyzing query: '{query}'")

        # Detect special context types
        context_type = self._detect_context_type(query)

        # Calculate Rose Glass dimensions (simplified for example)
        emotional_sig = self._calculate_emotional_signature(query)

        # Calculate temporal depth
        temporal_sig = self.temporal_analyzer.analyze(query)
        emotional_sig.tau = temporal_sig.tau

        print(f"   Emotional activation (q): {emotional_sig.q:.2f}")
        print(f"   Wisdom depth (ρ): {emotional_sig.rho:.2f}")
        print(f"   Temporal depth (τ): {emotional_sig.tau:.2f}")
        print(f"   Context type: {context_type}")

        return {
            'emotional_signature': emotional_sig,
            'context_type': context_type,
            'temporal_signature': temporal_sig
        }

    def _detect_context_type(self, query: str) -> str:
        """Detect special query types"""
        if self.trust_detector.detect(query):
            return 'trust_signal'
        elif self.mission_detector.detect(query):
            return 'mission_mode'
        elif 'summarize' in query.lower() or 'essence' in query.lower():
            return 'essence_request'
        else:
            return 'standard'

    def _calculate_emotional_signature(self, text: str) -> EmotionalSignature:
        """
        Calculate emotional signature (simplified)

        In production, use actual Rose Glass ML models
        """
        # Simplified heuristics (replace with actual models)
        words = text.lower().split()

        # q: Emotional activation
        emotional_words = ['urgent', 'critical', 'help', 'please', '!', 'worried', 'concerned']
        q = min(sum(1 for w in words if any(ew in w for ew in emotional_words)) / 10, 1.0)

        # rho: Wisdom depth
        wisdom_words = ['understand', 'philosophy', 'theory', 'principle', 'fundamental']
        rho = min(sum(1 for w in words if any(ww in w for ww in wisdom_words)) / 5, 1.0)

        # psi: Internal consistency
        psi = 0.7 if len(words) > 10 else 0.5  # Longer queries tend to be more structured

        # f: Social belonging
        social_words = ['we', 'us', 'our', 'community', 'together', 'collective']
        f = min(sum(1 for w in words if w in social_words) / 5, 1.0)

        return EmotionalSignature(
            psi=psi,
            rho=rho,
            q=q,
            f=f,
            tau=0.0,  # Will be filled by temporal analyzer
            lens=self.rose_glass.current_lens,
            confidence="moderate"
        )

    def retrieve_documents(self, query: str, limit: int = 10) -> List[Document]:
        """
        Retrieve documents using hybrid search

        In production:
        1. Vector search (Qdrant)
        2. Keyword search (Elasticsearch)
        3. Reciprocal Rank Fusion
        """
        print(f"\n📚 Retrieving documents (hybrid search)...")

        # Simulated retrieval (replace with actual RAG)
        mock_docs = [
            Document(
                id="doc1",
                title="Understanding Emotional Intelligence in AI",
                content="Emotional intelligence in AI systems involves understanding and responding to human emotions...",
                rag_score=0.85
            ),
            Document(
                id="doc2",
                title="Legal Case Analysis: Trauma-Informed Approaches",
                content="When analyzing legal cases involving trauma, it's critical to understand emotional context...",
                rag_score=0.78
            ),
            Document(
                id="doc3",
                title="Technical Guide: Vector Databases",
                content="Vector databases store high-dimensional embeddings for semantic search...",
                rag_score=0.72
            )
        ]

        print(f"   Retrieved {len(mock_docs)} documents")
        return mock_docs

    def analyze_documents(
        self,
        docs: List[Document],
        query_signature: EmotionalSignature
    ) -> List[Document]:
        """
        Analyze documents through Rose Glass and match to query
        """
        print(f"\n🎨 Analyzing documents emotionally...")

        for doc in docs:
            # Calculate document emotional signature
            doc.emotional_signature = self._calculate_emotional_signature(doc.content)

            # Calculate emotional match to query
            doc.emotional_match = self._calculate_emotional_match(
                query_signature,
                doc.emotional_signature
            )

            # Calculate temporal match
            doc.temporal_match = 1.0 - abs(
                query_signature.tau - doc.emotional_signature.tau
            )

            # Store wisdom depth
            doc.wisdom_depth = doc.emotional_signature.rho

            print(f"   {doc.id}: emotional_match={doc.emotional_match:.2f}, "
                  f"wisdom={doc.wisdom_depth:.2f}")

        return docs

    def _calculate_emotional_match(
        self,
        query_sig: EmotionalSignature,
        doc_sig: EmotionalSignature
    ) -> float:
        """
        Calculate how well document emotional signature matches query
        """
        weights = {
            'psi': 0.20,
            'rho': 0.30,
            'q': 0.35,
            'f': 0.15
        }

        match_score = 0.0
        for dim, weight in weights.items():
            q_val = getattr(query_sig, dim)
            d_val = getattr(doc_sig, dim)
            distance = abs(q_val - d_val)
            match_score += (1 - distance) * weight

        return match_score

    def rank_documents(
        self,
        docs: List[Document],
        context_type: str = 'standard'
    ) -> List[Document]:
        """
        Rank documents by combined RAG + emotional scores
        """
        print(f"\n📊 Ranking documents (context: {context_type})...")

        for doc in docs:
            if context_type == 'trust_signal':
                # Prioritize high wisdom, high consistency
                doc.final_score = (
                    doc.rag_score * 0.30 +
                    doc.wisdom_depth * 0.40 +
                    doc.emotional_signature.psi * 0.30
                )
            elif context_type == 'mission_mode':
                # Prioritize relevance and completeness
                doc.final_score = (
                    doc.rag_score * 0.60 +
                    doc.wisdom_depth * 0.25 +
                    doc.emotional_match * 0.15
                )
            else:
                # Standard balanced ranking
                doc.final_score = (
                    doc.rag_score * 0.40 +
                    doc.emotional_match * 0.30 +
                    doc.temporal_match * 0.15 +
                    doc.wisdom_depth * 0.15
                )

            print(f"   {doc.id}: final_score={doc.final_score:.3f}")

        # Sort by final score
        docs.sort(key=lambda d: d.final_score, reverse=True)
        return docs

    def generate_response(
        self,
        query: str,
        context_docs: List[Document],
        query_signature: EmotionalSignature,
        context_type: str
    ) -> str:
        """
        Generate response with emotional grounding

        In production, this would call OpenAI/Claude/Ollama
        """
        print(f"\n💬 Generating response...")

        # Determine style based on emotional signature
        if query_signature.q > 0.7:
            style = "empathetic and emotionally engaged"
        elif query_signature.rho > 0.8:
            style = "philosophically rich and conceptually deep"
        elif query_signature.tau < 0.3:
            style = "current, timely, and action-oriented"
        else:
            style = "balanced and informative"

        print(f"   Response style: {style}")

        # Build context from top documents
        context = "\n\n".join([
            f"[{doc.title}]\n{doc.content}"
            for doc in context_docs[:3]
        ])

        # Simulated LLM call (replace with actual API)
        response = f"""Based on the context provided and understanding your query's emotional signature
(activation: {query_signature.q:.2f}, wisdom depth: {query_signature.rho:.2f}),
I'll respond in a {style} manner.

[Actual LLM response would go here, incorporating the context documents
and matching the emotional tone of the query]

Context sources used: {', '.join(doc.title for doc in context_docs[:3])}
"""

        return response

    def track_conversation(self, query_signature: EmotionalSignature):
        """
        Track conversation gradient for escalation detection
        """
        snapshot = PatternSnapshot(
            timestamp=datetime.now(),
            psi=query_signature.psi,
            rho=query_signature.rho,
            q=query_signature.q,
            f=query_signature.f,
            tau=query_signature.tau,
            pattern_intensity=(query_signature.psi + query_signature.rho +
                             query_signature.q + query_signature.f) / 4
        )

        self.gradient_tracker.add_snapshot(snapshot)

        # Check for escalation
        gradient = self.gradient_tracker.calculate_gradient()
        if gradient:
            prediction = self.gradient_tracker.predict_trajectory(time_horizon=30.0)

            if prediction.intervention_recommended:
                print(f"\n⚠️  INTERVENTION RECOMMENDED: {prediction.intervention_reason}")
                return True

        return False

    def query(self, query: str) -> str:
        """
        Complete RAG pipeline with emotional intelligence

        Main entry point for querying the system
        """
        print("=" * 70)
        print("EMOTIONALLY INFORMED RAG PIPELINE")
        print("=" * 70)

        # 1. Analyze query
        analysis = self.analyze_query(query)
        query_signature = analysis['emotional_signature']
        context_type = analysis['context_type']

        # 2. Track conversation gradient
        intervention_needed = self.track_conversation(query_signature)

        # 3. Retrieve documents (hybrid search)
        docs = self.retrieve_documents(query)

        # 4. Analyze documents emotionally
        docs = self.analyze_documents(docs, query_signature)

        # 5. Rank documents
        ranked_docs = self.rank_documents(docs, context_type)

        # 6. Generate response
        response = self.generate_response(
            query,
            ranked_docs,
            query_signature,
            context_type
        )

        print("\n" + "=" * 70)
        print("RESPONSE:")
        print("=" * 70)
        print(response)

        if intervention_needed:
            print("\n⚠️  Note: Escalation detected in conversation pattern")

        return response


def main():
    """Example usage"""
    print("\n🌹 Initializing Emotionally Informed RAG System...\n")

    # Initialize the system
    rag = EmotionallyInformedRAG(cultural_lens="modern_digital")

    # Example queries with different emotional signatures

    # Query 1: High emotional activation
    print("\n" + "🔴" * 35)
    print("Example 1: High Emotional Activation Query")
    print("🔴" * 35)
    response1 = rag.query(
        "I'm really struggling to understand how AI can help with my legal case. "
        "This is urgent and I'm very worried about the outcome."
    )

    # Query 2: High wisdom depth
    print("\n" + "🔵" * 35)
    print("Example 2: High Wisdom Depth Query")
    print("🔵" * 35)
    response2 = rag.query(
        "What are the fundamental philosophical principles underlying the integration "
        "of emotional intelligence into artificial intelligence systems?"
    )

    # Query 3: Mission mode
    print("\n" + "🟢" * 35)
    print("Example 3: Mission Mode Query")
    print("🟢" * 35)
    response3 = rag.query(
        "Research and analyze all available approaches to building RAG systems "
        "with emotional awareness capabilities."
    )


if __name__ == "__main__":
    main()
