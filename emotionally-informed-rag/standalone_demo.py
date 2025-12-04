#!/usr/bin/env python3
"""
Emotionally Informed RAG Pipeline - Standalone Demo

Self-contained demonstration without external dependencies.
Shows the core concepts of emotional intelligence in RAG.
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
import re


@dataclass
class EmotionalSignature:
    """Emotional signature of text"""
    psi: float  # Internal consistency (0-1)
    rho: float  # Wisdom depth (0-1)
    q: float    # Emotional activation (0-1)
    f: float    # Social belonging (0-1)
    tau: float  # Temporal depth (0-1)
    lens: str   # Cultural lens


@dataclass
class Document:
    """Document with emotional signature and scoring"""
    id: str
    title: str
    content: str
    rag_score: float
    emotional_signature: Optional[EmotionalSignature] = None
    emotional_match: float = 0.0
    temporal_match: float = 0.0
    wisdom_depth: float = 0.0
    final_score: float = 0.0


class SimpleEmotionalAnalyzer:
    """
    Simplified emotional analysis using keyword heuristics

    In production, this would use ML models from Rose Glass
    """

    def __init__(self):
        # Emotional activation indicators
        self.high_emotion_words = {
            'urgent', 'critical', 'help', 'please', 'worried', 'concerned',
            'afraid', 'anxious', 'desperate', 'struggling', 'crisis'
        }

        # Wisdom depth indicators
        self.wisdom_words = {
            'understand', 'philosophy', 'theory', 'principle', 'fundamental',
            'essence', 'nature', 'wisdom', 'insight', 'profound', 'deep'
        }

        # Social belonging indicators
        self.social_words = {
            'we', 'us', 'our', 'community', 'together', 'collective',
            'society', 'people', 'everyone', 'shared'
        }

        # Temporal depth indicators
        self.eternal_markers = {
            'always', 'never', 'eternal', 'timeless', 'ancient', 'forever',
            'generations', 'ages', 'enduring', 'permanent'
        }

        self.immediate_markers = {
            'now', 'today', 'urgent', 'asap', 'immediately', 'right now',
            'breaking', 'latest', 'trending'
        }

    def analyze(self, text: str, lens: str = "modern_digital") -> EmotionalSignature:
        """Analyze text and return emotional signature"""

        text_lower = text.lower()
        words = set(text_lower.split())
        word_count = len(text_lower.split())

        # Calculate q (emotional activation)
        emotion_count = sum(1 for w in words if w in self.high_emotion_words)
        exclamation_count = text.count('!')
        question_count = text.count('?')
        q = min((emotion_count * 0.15 + exclamation_count * 0.1 + question_count * 0.05), 1.0)

        # Calculate rho (wisdom depth)
        wisdom_count = sum(1 for w in words if w in self.wisdom_words)
        avg_word_length = sum(len(w) for w in words) / max(len(words), 1)
        rho = min((wisdom_count * 0.15 + (avg_word_length - 5) * 0.05), 1.0)
        rho = max(0.0, rho)

        # Calculate psi (internal consistency)
        # Longer, more structured text tends to be more consistent
        psi = min(0.5 + (word_count / 100) * 0.3, 1.0)

        # Calculate f (social belonging)
        social_count = sum(1 for w in words if w in self.social_words)
        f = min(social_count * 0.15, 1.0)

        # Calculate tau (temporal depth)
        eternal_count = sum(1 for w in words if w in self.eternal_markers)
        immediate_count = sum(1 for w in words if w in self.immediate_markers)
        tau = min(max(eternal_count * 0.2 - immediate_count * 0.15, 0.0), 1.0)

        return EmotionalSignature(
            psi=psi,
            rho=rho,
            q=q,
            f=f,
            tau=tau,
            lens=lens
        )


class EmotionallyInformedRAG:
    """
    Simplified RAG with emotional intelligence

    Demonstrates core concepts without requiring external services
    """

    def __init__(self, lens: str = "modern_digital"):
        self.analyzer = SimpleEmotionalAnalyzer()
        self.current_lens = lens
        self.conversation_history = []

        # Mock knowledge base
        self.knowledge_base = [
            Document(
                id="doc1",
                title="Understanding Emotional Intelligence in AI Systems",
                content="""Emotional intelligence in AI systems involves the ability to
                understand, interpret, and respond appropriately to human emotions. This goes
                beyond simple sentiment analysis to include deeper aspects like empathy,
                context awareness, and adaptive communication. Advanced systems can detect
                emotional activation, wisdom depth, and social context to provide more
                meaningful and appropriate responses.""",
                rag_score=0.0  # Will be calculated
            ),
            Document(
                id="doc2",
                title="Legal Case Analysis: Trauma-Informed Approaches",
                content="""When analyzing legal cases involving trauma, it's critical to
                understand the emotional context and psychological impact. Trauma-informed
                approaches recognize that individuals may exhibit heightened emotional
                activation, fragmented communication patterns, and urgent need for support.
                Legal professionals must balance emotional sensitivity with procedural
                requirements, ensuring dignity while pursuing justice.""",
                rag_score=0.0
            ),
            Document(
                id="doc3",
                title="RAG Systems: Technical Architecture and Implementation",
                content="""Retrieval-Augmented Generation systems combine information retrieval
                with language generation. The architecture typically includes vector databases
                for semantic search, keyword search engines for precision matching, and
                sophisticated reranking algorithms. Modern RAG systems use hybrid approaches,
                combining multiple retrieval methods to achieve optimal results.""",
                rag_score=0.0
            ),
            Document(
                id="doc4",
                title="Philosophical Foundations of Human-AI Communication",
                content="""The philosophical underpinnings of human-AI communication draw from
                ancient wisdom traditions and modern cognitive science. Understanding requires
                translation, not merely measurement. Different cultural contexts construct
                coherence differently - what appears as wisdom in one tradition may manifest
                differently in another. True artificial intelligence must embrace this
                multiplicity rather than imposing universal standards.""",
                rag_score=0.0
            ),
            Document(
                id="doc5",
                title="Crisis Intervention and De-escalation Techniques",
                content="""Effective crisis intervention requires immediate recognition of
                escalating emotional patterns and rapid, appropriate response. Key indicators
                include rapid increases in emotional activation, loss of coherent communication,
                and expressions of urgency or desperation. Intervention strategies must be
                culturally sensitive, trauma-informed, and focused on immediate stabilization
                while connecting to longer-term support resources.""",
                rag_score=0.0
            )
        ]

    def calculate_rag_score(self, query: str, doc: Document) -> float:
        """Simple keyword-based relevance scoring"""
        query_words = set(query.lower().split())
        doc_words = set(doc.content.lower().split())

        # Jaccard similarity
        intersection = query_words & doc_words
        union = query_words | doc_words

        return len(intersection) / len(union) if union else 0.0

    def calculate_emotional_match(
        self,
        query_sig: EmotionalSignature,
        doc_sig: EmotionalSignature
    ) -> float:
        """Calculate emotional signature similarity"""

        weights = {
            'psi': 0.20,
            'rho': 0.30,
            'q': 0.35,  # Highest weight on emotional activation
            'f': 0.15
        }

        match_score = 0.0
        for dim, weight in weights.items():
            q_val = getattr(query_sig, dim)
            d_val = getattr(doc_sig, dim)
            distance = abs(q_val - d_val)
            match_score += (1 - distance) * weight

        return match_score

    def detect_context_type(self, query: str, query_sig: EmotionalSignature) -> str:
        """Detect special query types"""

        query_lower = query.lower()

        # Trust signal: brief, high-trust expressions
        if query_sig.q > 0.7 and len(query.split()) < 15:
            if 'trust' in query_lower or 'believe' in query_lower:
                return 'trust_signal'

        # Mission mode: research/analysis requests
        if any(word in query_lower for word in ['research', 'analyze', 'investigate', 'explore']):
            return 'mission_mode'

        # Essence request: summary requests
        if any(word in query_lower for word in ['summarize', 'essence', 'brief', 'tldr']):
            return 'essence_request'

        # Crisis: high emotion + urgency
        if query_sig.q > 0.8 and any(word in query_lower for word in ['urgent', 'crisis', 'help', 'emergency']):
            return 'crisis'

        return 'standard'

    def query(self, user_query: str, verbose: bool = True) -> str:
        """Process query through emotional RAG pipeline"""

        if verbose:
            print("=" * 80)
            print("EMOTIONALLY INFORMED RAG PIPELINE")
            print("=" * 80)
            print(f"\n📝 Query: {user_query}\n")

        # 1. Analyze query emotionally
        query_sig = self.analyzer.analyze(user_query, self.current_lens)
        context_type = self.detect_context_type(user_query, query_sig)

        if verbose:
            print("🎨 EMOTIONAL ANALYSIS:")
            print(f"   • Emotional Activation (q): {query_sig.q:.2f}")
            print(f"   • Wisdom Depth (ρ):         {query_sig.rho:.2f}")
            print(f"   • Internal Consistency (Ψ): {query_sig.psi:.2f}")
            print(f"   • Social Belonging (f):     {query_sig.f:.2f}")
            print(f"   • Temporal Depth (τ):       {query_sig.tau:.2f}")
            print(f"   • Context Type:             {context_type}")
            print()

        # 2. Calculate RAG scores for all documents
        for doc in self.knowledge_base:
            doc.rag_score = self.calculate_rag_score(user_query, doc)
            doc.emotional_signature = self.analyzer.analyze(doc.content)
            doc.emotional_match = self.calculate_emotional_match(query_sig, doc.emotional_signature)
            doc.temporal_match = 1.0 - abs(query_sig.tau - doc.emotional_signature.tau)
            doc.wisdom_depth = doc.emotional_signature.rho

        # 3. Calculate final scores based on context type
        if context_type == 'crisis':
            # Prioritize emotional match and immediate relevance
            for doc in self.knowledge_base:
                doc.final_score = (
                    doc.rag_score * 0.30 +
                    doc.emotional_match * 0.50 +
                    doc.temporal_match * 0.20
                )
        elif context_type == 'mission_mode':
            # Prioritize relevance and completeness
            for doc in self.knowledge_base:
                doc.final_score = (
                    doc.rag_score * 0.60 +
                    doc.wisdom_depth * 0.25 +
                    doc.emotional_match * 0.15
                )
        else:
            # Standard balanced scoring
            for doc in self.knowledge_base:
                doc.final_score = (
                    doc.rag_score * 0.40 +
                    doc.emotional_match * 0.30 +
                    doc.temporal_match * 0.15 +
                    doc.wisdom_depth * 0.15
                )

        # 4. Rank documents
        ranked_docs = sorted(self.knowledge_base, key=lambda d: d.final_score, reverse=True)
        top_docs = ranked_docs[:3]

        if verbose:
            print("📊 TOP DOCUMENTS:")
            for i, doc in enumerate(top_docs, 1):
                print(f"\n   {i}. {doc.title}")
                print(f"      • RAG Score:       {doc.rag_score:.3f}")
                print(f"      • Emotional Match: {doc.emotional_match:.3f}")
                print(f"      • Temporal Match:  {doc.temporal_match:.3f}")
                print(f"      • Wisdom Depth:    {doc.wisdom_depth:.3f}")
                print(f"      • FINAL SCORE:     {doc.final_score:.3f}")

        # 5. Generate response based on emotional signature
        response = self.generate_response(user_query, top_docs, query_sig, context_type)

        # 6. Track conversation for gradient analysis
        self.conversation_history.append({
            'timestamp': datetime.now(),
            'query': user_query,
            'signature': query_sig,
            'context_type': context_type
        })

        # Check for escalation
        if len(self.conversation_history) >= 2:
            escalation = self.check_escalation()
            if escalation and verbose:
                print(f"\n⚠️  ESCALATION DETECTED: {escalation}")

        if verbose:
            print("\n" + "=" * 80)
            print("💬 RESPONSE:")
            print("=" * 80)
            print(response)
            print("=" * 80)

        return response

    def generate_response(
        self,
        query: str,
        docs: List[Document],
        query_sig: EmotionalSignature,
        context_type: str
    ) -> str:
        """Generate emotionally appropriate response"""

        # Determine response style
        if context_type == 'crisis':
            style = "immediate, empathetic, and supportive"
            tone = "calm and reassuring"
        elif query_sig.q > 0.7:
            style = "empathetic and emotionally engaged"
            tone = "validating and supportive"
        elif query_sig.rho > 0.7:
            style = "philosophically rich and conceptually deep"
            tone = "thoughtful and nuanced"
        elif context_type == 'mission_mode':
            style = "systematic and comprehensive"
            tone = "analytical and thorough"
        else:
            style = "balanced and informative"
            tone = "clear and helpful"

        # Build response
        response_parts = []

        # Opening (matches emotional tone)
        if context_type == 'crisis':
            response_parts.append("I understand this is urgent and important to you. Let me provide immediate, relevant information:")
        elif query_sig.q > 0.7:
            response_parts.append("I hear the urgency and concern in your question. Here's what I can share:")
        elif query_sig.rho > 0.7:
            response_parts.append("This is a profound question that deserves thoughtful consideration:")
        else:
            response_parts.append("Based on the available information:")

        # Main content from top documents
        response_parts.append(f"\n{docs[0].content[:200]}...")

        # Additional context if relevant
        if len(docs) > 1 and docs[1].final_score > 0.3:
            response_parts.append(f"\nAdditionally: {docs[1].content[:150]}...")

        # Closing (matches context type)
        if context_type == 'crisis':
            response_parts.append("\n\nIf you need immediate assistance, please reach out to appropriate support resources.")
        elif query_sig.q > 0.6:
            response_parts.append("\n\nI hope this information is helpful. Please let me know if you need anything else.")

        # Add sources
        response_parts.append(f"\n\n📚 Sources: {', '.join(doc.title for doc in docs[:2])}")

        return '\n'.join(response_parts)

    def check_escalation(self) -> Optional[str]:
        """Check if conversation is escalating"""

        if len(self.conversation_history) < 2:
            return None

        recent = self.conversation_history[-2:]
        q_change = recent[1]['signature'].q - recent[0]['signature'].q
        psi_change = recent[1]['signature'].psi - recent[0]['signature'].psi

        if q_change > 0.3:
            return "Rapid increase in emotional activation detected"

        if psi_change < -0.3:
            return "Significant loss of coherence detected"

        if recent[1]['signature'].q > 0.85:
            return "Extremely high emotional activation"

        return None


def main():
    """Run demonstration"""

    print("\n" + "🌹" * 40)
    print("EMOTIONALLY INFORMED RAG - STANDALONE DEMO")
    print("🌹" * 40)
    print("\nDemonstrating emotional intelligence in retrieval-augmented generation")
    print("without requiring external services or databases.\n")

    # Initialize system
    rag = EmotionallyInformedRAG(lens="modern_digital")

    # Example 1: High emotional activation
    print("\n" + "🔴" * 40)
    print("EXAMPLE 1: High Emotional Activation Query")
    print("🔴" * 40)

    response1 = rag.query(
        "I'm really struggling to understand how AI can help with my legal case. "
        "This is urgent and I'm very worried about the outcome!"
    )

    input("\n\nPress Enter to continue to Example 2...")

    # Example 2: High wisdom depth
    print("\n\n" + "🔵" * 40)
    print("EXAMPLE 2: High Wisdom Depth Query")
    print("🔵" * 40)

    response2 = rag.query(
        "What are the fundamental philosophical principles underlying the integration "
        "of emotional intelligence into artificial intelligence systems, and how do "
        "different cultural traditions construct coherence differently?"
    )

    input("\n\nPress Enter to continue to Example 3...")

    # Example 3: Mission mode
    print("\n\n" + "🟢" * 40)
    print("EXAMPLE 3: Mission Mode Query")
    print("🟢" * 40)

    response3 = rag.query(
        "Research and analyze all available approaches to building RAG systems "
        "with emotional awareness capabilities."
    )

    input("\n\nPress Enter to continue to Example 4...")

    # Example 4: Escalation sequence
    print("\n\n" + "🟠" * 40)
    print("EXAMPLE 4: Escalation Detection")
    print("🟠" * 40)
    print("\nDemonstrating gradient tracking across conversation turns:\n")

    rag.query("Can you help me understand this legal document?")
    input("\nPress Enter for next turn...")

    rag.query("I'm getting more confused. This is really important!")
    input("\nPress Enter for next turn...")

    rag.query("This is urgent! I need help NOW! I'm very worried!")

    print("\n\n" + "✅" * 40)
    print("DEMO COMPLETE")
    print("✅" * 40)
    print("\nThis demonstration showed:")
    print("  • Emotional signature analysis (Ψ, ρ, q, f, τ)")
    print("  • Context-aware retrieval (crisis, mission, standard)")
    print("  • Emotional matching between queries and documents")
    print("  • Adaptive response generation")
    print("  • Real-time escalation detection")
    print("\nNext steps:")
    print("  1. Integrate actual Rose Glass ML models")
    print("  2. Connect to Qdrant and Elasticsearch")
    print("  3. Add LLM generation (OpenAI/Claude/Ollama)")
    print("  4. Build API server and dashboard")
    print("\n" + "🌹" * 40 + "\n")


if __name__ == "__main__":
    main()
