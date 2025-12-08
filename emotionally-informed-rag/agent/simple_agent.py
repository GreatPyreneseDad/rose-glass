#!/usr/bin/env python3
"""
Simplified Emotionally-Informed RAG Agent for Python 3.13

Works without torch/sentence-transformers and with minimal dependencies.
"""

import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
import re
from collections import Counter
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Import dependencies
try:
    from qdrant_client import QdrantClient
    QDRANT_AVAILABLE = True
except ImportError:
    logger.error("Qdrant not available")
    QDRANT_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    logger.warning("Anthropic not available - responses will be mock")
    ANTHROPIC_AVAILABLE = False


@dataclass
class EmotionalSignature:
    """Emotional signature of text"""
    psi: float  # Internal consistency
    rho: float  # Wisdom depth
    q: float    # Emotional activation
    f: float    # Social belonging
    tau: float = 0.5  # Temporal depth
    lens: str = "modern_digital"
    context_type: str = "standard"


class SimpleEmotionalAgent:
    """Simplified agent for legal document querying with emotional awareness"""

    def __init__(
        self,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        collection_name: str = "knowledge_base",
        anthropic_api_key: Optional[str] = None
    ):
        self.collection_name = collection_name
        self.conversation_history = []

        # Legal vocabulary for embeddings
        self.vocabulary = self._build_legal_vocabulary()

        # Initialize Qdrant
        if QDRANT_AVAILABLE:
            try:
                self.qdrant = QdrantClient(host=qdrant_host, port=qdrant_port)
                logger.info(f"✓ Connected to Qdrant at {qdrant_host}:{qdrant_port}")
            except Exception as e:
                logger.error(f"Failed to connect to Qdrant: {e}")
                self.qdrant = None
        else:
            self.qdrant = None

        # Initialize Anthropic
        if ANTHROPIC_AVAILABLE and anthropic_api_key:
            self.anthropic = anthropic.Anthropic(api_key=anthropic_api_key)
            logger.info("✓ Anthropic API initialized")
        elif ANTHROPIC_AVAILABLE:
            # Try to get from env
            try:
                self.anthropic = anthropic.Anthropic()
                logger.info("✓ Anthropic API initialized from environment")
            except Exception as e:
                logger.warning(f"Anthropic API not configured: {e}")
                self.anthropic = None
        else:
            self.anthropic = None

    def _build_legal_vocabulary(self) -> List[str]:
        """Build vocabulary of common legal and emotional terms"""
        return [
            # Legal terms
            "motion", "court", "case", "defendant", "plaintiff", "evidence", "testimony",
            "hearing", "trial", "judge", "attorney", "counsel", "jury", "verdict",
            "objection", "ruling", "order", "filing", "petition", "complaint",
            "discovery", "deposition", "subpoena", "witness", "document", "exhibit",
            "contract", "agreement", "breach", "damages", "liability", "negligence",
            "custody", "divorce", "family", "child", "support", "visitation",
            "restraining", "protective", "harassment", "abuse", "domestic",
            "maslowsky", "macgregor", "fardell",  # Case names
            # Emotional/trauma-informed terms
            "trauma", "emotional", "distress", "anxiety", "fear", "worried", "concerned",
            "urgent", "crisis", "help", "support", "safety", "protection", "harm",
            "stress", "difficult", "challenging", "overwhelmed", "frustrated",
            "understand", "listen", "empathy", "compassion", "respect",
        ]

    def _simple_embedding(self, text: str) -> np.ndarray:
        """Create simple TF-IDF-style embedding"""
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        word_counts = Counter(words)

        embedding = np.zeros(384)

        # Map vocabulary terms to embedding dimensions
        for idx, vocab_term in enumerate(self.vocabulary[:384]):
            if vocab_term in word_counts:
                tf = word_counts[vocab_term] / len(words) if words else 0
                embedding[idx] = tf

        # Add hash-based embedding for non-vocabulary terms
        non_vocab_words = [w for w in words if w not in self.vocabulary]
        for word in non_vocab_words[:50]:
            hash_val = hash(word)
            for i in range(3):
                idx = (hash_val + i * 17) % 384
                embedding[idx] += 0.1

        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def analyze_emotional_signature(self, text: str) -> EmotionalSignature:
        """Analyze emotional signature of text"""
        text_lower = text.lower()

        # Emotional activation (q)
        crisis_words = ['urgent', 'worried', 'help', 'crisis', 'emergency', 'scared', 'terrified', 'desperate']
        q_score = min(1.0, sum(1 for word in crisis_words if word in text_lower) * 0.2 + 0.3)

        # Wisdom depth (rho)
        wisdom_words = ['understand', 'consider', 'reflect', 'principle', 'philosophy', 'wisdom', 'insight']
        rho_score = min(1.0, sum(1 for word in wisdom_words if word in text_lower) * 0.15 + 0.3)

        # Social belonging (f)
        social_words = ['we', 'together', 'community', 'family', 'relationship', 'support']
        f_score = min(1.0, sum(1 for word in social_words if word in text_lower) * 0.15 + 0.3)

        # Internal consistency (psi)
        psi_score = 0.6

        # Temporal depth (tau)
        temporal_words = ['past', 'future', 'history', 'timeline', 'ongoing', 'current']
        tau_score = min(1.0, sum(1 for word in temporal_words if word in text_lower) * 0.15 + 0.4)

        # Context type detection
        context_type = "standard"
        if q_score > 0.7:
            context_type = "crisis"
        elif any(word in text_lower for word in ['research', 'analyze', 'study', 'comprehensive']):
            context_type = "mission"

        return EmotionalSignature(
            psi=psi_score,
            rho=rho_score,
            q=q_score,
            f=f_score,
            tau=tau_score,
            context_type=context_type
        )

    def retrieve_documents(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve relevant documents from Qdrant"""
        if not self.qdrant:
            logger.warning("Qdrant not available - cannot retrieve documents")
            return []

        try:
            # Create query embedding
            query_embedding = self._simple_embedding(query)

            # Search Qdrant
            results = self.qdrant.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                limit=top_k
            )

            documents = []
            for result in results:
                documents.append({
                    'content': result.payload.get('content', ''),
                    'title': result.payload.get('title', 'Unknown'),
                    'score': result.score,
                    'emotional_signature': {
                        'psi': result.payload.get('psi', 0.5),
                        'rho': result.payload.get('rho', 0.5),
                        'q': result.payload.get('q', 0.5),
                        'f': result.payload.get('f', 0.5)
                    }
                })

            logger.info(f"Retrieved {len(documents)} documents")
            return documents

        except Exception as e:
            logger.error(f"Document retrieval failed: {e}")
            return []

    def generate_response(
        self,
        query: str,
        context_docs: List[Dict],
        emotional_signature: EmotionalSignature
    ) -> str:
        """Generate response using Claude API"""

        if not self.anthropic:
            # Mock response
            return f"""[MOCK RESPONSE - Anthropic API not configured]

Based on your query about: "{query}"

I found {len(context_docs)} relevant documents with emotional activation level q={emotional_signature.q:.2f}.

Context type: {emotional_signature.context_type}

To get real responses, set ANTHROPIC_API_KEY environment variable."""

        # Build context from documents
        context_text = "\n\n".join([
            f"Document: {doc['title']}\n{doc['content'][:500]}..."
            for doc in context_docs[:3]
        ])

        # Build system prompt with emotional awareness
        system_prompt = f"""You are an emotionally-informed legal assistant. You have access to legal documents and provide empathetic, accurate responses.

Current emotional context:
- Emotional activation (q): {emotional_signature.q:.2f}
- Wisdom depth (ρ): {emotional_signature.rho:.2f}
- Social belonging (f): {emotional_signature.f:.2f}
- Context type: {emotional_signature.context_type}

When responding:
1. Be empathetic and trauma-informed
2. Provide accurate legal information from the context
3. Adjust tone based on emotional activation level
4. If this is a crisis (high q), prioritize immediate, supportive information"""

        # Build user message
        user_message = f"""Based on these legal documents:

{context_text}

Please answer: {query}"""

        try:
            # Call Claude API
            response = self.anthropic.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_message}
                ]
            )

            return response.content[0].text

        except Exception as e:
            logger.error(f"Claude API call failed: {e}")
            return f"Error generating response: {e}"

    def query(self, user_query: str) -> Dict[str, Any]:
        """Process a query through the full RAG pipeline"""

        logger.info(f"\nProcessing query: {user_query}")

        # 1. Analyze emotional signature
        signature = self.analyze_emotional_signature(user_query)
        logger.info(f"Emotional signature: q={signature.q:.2f}, ρ={signature.rho:.2f}, context={signature.context_type}")

        # 2. Retrieve relevant documents
        documents = self.retrieve_documents(user_query, top_k=5)

        # 3. Generate response
        response = self.generate_response(user_query, documents, signature)

        # 4. Track conversation history
        self.conversation_history.append({
            'query': user_query,
            'signature': signature,
            'num_docs': len(documents),
            'response': response
        })

        return {
            'response': response,
            'signature': signature,
            'documents': documents,
            'context_type': signature.context_type
        }


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(description='Simple Emotionally-Informed RAG Agent')
    parser.add_argument('query', nargs='?', help='Query to process')
    parser.add_argument('-i', '--interactive', action='store_true', help='Interactive mode')
    parser.add_argument('--host', default='localhost', help='Qdrant host')
    parser.add_argument('--port', type=int, default=6333, help='Qdrant port')
    parser.add_argument('--collection', default='knowledge_base', help='Collection name')

    args = parser.parse_args()

    # Initialize agent
    try:
        agent = SimpleEmotionalAgent(
            qdrant_host=args.host,
            qdrant_port=args.port,
            collection_name=args.collection
        )
    except Exception as e:
        logger.error(f"Failed to initialize agent: {e}")
        return 1

    # Interactive mode
    if args.interactive:
        print("\n" + "="*60)
        print("Emotionally-Informed Legal RAG Agent")
        print("="*60)
        print("\nType 'quit' or 'exit' to end session\n")

        while True:
            try:
                query = input("\n📝 Your query: ").strip()

                if query.lower() in ['quit', 'exit', 'q']:
                    print("\nGoodbye!")
                    break

                if not query:
                    continue

                result = agent.query(query)

                print(f"\n{'='*60}")
                print(f"Context: {result['context_type'].upper()} | "
                      f"q={result['signature'].q:.2f} | "
                      f"ρ={result['signature'].rho:.2f}")
                print(f"{'='*60}")
                print(f"\n{result['response']}")
                print(f"\n{'─'*60}")
                print(f"📚 Based on {len(result['documents'])} documents")
                print(f"{'─'*60}")

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                logger.error(f"Error processing query: {e}")

    # Single query mode
    elif args.query:
        result = agent.query(args.query)
        print(f"\n{result['response']}")
        print(f"\n[Context: {result['context_type']}, Documents: {len(result['documents'])}]")

    else:
        parser.print_help()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
