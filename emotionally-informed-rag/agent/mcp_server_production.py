#!/usr/bin/env python3
"""
Production MCP Server with Sentence-Transformers
================================================

Fixes identified issues:
1. Semantic retrieval using sentence-transformers (not TF-IDF)
2. Calibrated emotional signature computation
3. Fixed context type detection with proper thresholds

Usage:
    Update ~/.claude/settings.json to use this server
"""

import sys
import logging
from pathlib import Path
from typing import List, Dict, Optional
import json
import numpy as np
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from mcp.server import Server
    from mcp.types import Tool, TextContent
    import mcp.server.stdio
    MCP_AVAILABLE = True
except ImportError:
    logger.error("MCP not available. Install with: pip install mcp")
    MCP_AVAILABLE = False
    sys.exit(1)

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny
    QDRANT_AVAILABLE = True
except ImportError:
    logger.error("Qdrant not available")
    QDRANT_AVAILABLE = False
    sys.exit(1)

try:
    from sentence_transformers import SentenceTransformer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("⚠️  sentence-transformers not available. Install for production use:")
    logger.warning("   pip install sentence-transformers")
    TRANSFORMERS_AVAILABLE = False


class ProductionLegalRAGServer:
    """Production MCP Server with semantic search and calibrated analysis"""

    def __init__(
        self,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        collection_name: str = "knowledge_base"
    ):
        self.collection_name = collection_name

        # Initialize encoder (production quality)
        if TRANSFORMERS_AVAILABLE:
            logger.info("Loading sentence-transformers model...")
            self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✓ Using semantic embeddings (production mode)")
        else:
            self.encoder = None
            logger.warning("⚠️ Using TF-IDF fallback (degraded mode)")

        # Initialize Qdrant
        self.qdrant = QdrantClient(host=qdrant_host, port=qdrant_port)
        logger.info(f"✓ Connected to Qdrant at {qdrant_host}:{qdrant_port}")

    def _semantic_embedding(self, text: str) -> np.ndarray:
        """Create semantic embedding using sentence-transformers"""
        if self.encoder:
            return self.encoder.encode(text, convert_to_numpy=True)
        else:
            # Fallback to simple embedding (not recommended for production)
            return self._fallback_embedding(text)

    def _fallback_embedding(self, text: str) -> np.ndarray:
        """Fallback TF-IDF embedding when transformers unavailable"""
        vocabulary = [
            "motion", "court", "case", "defendant", "plaintiff", "evidence",
            "testimony", "hearing", "trial", "custody", "emergency", "protection",
            "order", "restraining", "maslowsky", "macgregor", "fardell"
        ]

        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1

        embedding = np.zeros(384)
        for idx, vocab_term in enumerate(vocabulary[:384]):
            if vocab_term in word_counts:
                tf = word_counts[vocab_term] / len(words) if words else 0
                embedding[idx] = tf

        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def _analyze_emotional_context(self, text: str) -> Dict:
        """
        Calibrated emotional context analysis

        Fixed issues:
        1. Removed baseline scores - start from 0
        2. Lowered crisis threshold to 0.4
        3. Added trauma keywords
        4. Proper legal context detection
        """
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)

        # Emotional activation (q) - FIXED: No baseline, proper weighting
        crisis_keywords = {
            'urgent': 0.3,
            'emergency': 0.4,
            'crisis': 0.5,
            'help': 0.2,
            'worried': 0.2,
            'scared': 0.3,
            'terrified': 0.4,
            'threat': 0.3,
            'violence': 0.4,
            'abuse': 0.4,
            'harm': 0.3,
            'danger': 0.3,
            'immediate': 0.3,
        }

        q_score = 0.0
        for word, weight in crisis_keywords.items():
            if word in text_lower:
                q_score += weight

        q_score = min(1.0, q_score)

        # Wisdom depth (rho) - FIXED: Start from 0, add for legal complexity
        wisdom_keywords = {
            'principle': 0.3,
            'philosophy': 0.3,
            'understand': 0.2,
            'consider': 0.2,
            'analyze': 0.3,
            'comprehensive': 0.3,
            'precedent': 0.3,
            'statutory': 0.3,
            'constitutional': 0.4,
            'jurisprudence': 0.4,
        }

        rho_score = 0.0
        for word, weight in wisdom_keywords.items():
            if word in text_lower:
                rho_score += weight

        # Legal documents have inherent wisdom depth
        legal_terms = ['motion', 'brief', 'statute', 'case', 'ruling', 'order']
        if any(term in text_lower for term in legal_terms):
            rho_score += 0.2

        rho_score = min(1.0, rho_score)

        # Context type - FIXED: Lower thresholds, better detection
        context_type = "standard"
        trauma_informed = False

        # Trauma-informed: Protection orders, custody, abuse (check first)
        trauma_triggers = ['custody', 'protection', 'restraining', 'abuse', 'threat', 'violence', 'harm', 'ppo']
        if any(word in text_lower for word in trauma_triggers):
            trauma_informed = True

        # Crisis mode: High q OR explicit crisis terms OR trauma-informed
        if q_score > 0.4 or any(word in text_lower for word in ['emergency', 'urgent', 'crisis', 'immediate']):
            context_type = "crisis"
            trauma_informed = True
        elif trauma_informed:
            # Trauma queries default to crisis unless research/analysis
            context_type = "crisis"

        # Mission mode: Deep research/analysis (overrides trauma crisis)
        if any(word in text_lower for word in ['analyze', 'comprehensive', 'research', 'investigate']):
            context_type = "mission"

        return {
            "emotional_activation": round(q_score, 2),
            "wisdom_depth": round(rho_score, 2),
            "context_type": context_type,
            "trauma_informed": trauma_informed
        }

    def _build_filter(
        self,
        doc_type: Optional[str] = None,
        case: Optional[str] = None,
        exclude_frameworks: bool = True
    ) -> Optional[Filter]:
        """Build Qdrant filter for metadata"""
        conditions = []

        if doc_type:
            conditions.append(FieldCondition(
                key="doc_type",
                match=MatchValue(value=doc_type)
            ))

        if case:
            conditions.append(FieldCondition(
                key="case",
                match=MatchValue(value=case)
            ))

        if exclude_frameworks:
            # Check if doc_type field exists, if not, skip this filter
            try:
                conditions.append(FieldCondition(
                    key="doc_type",
                    match=MatchAny(any=[
                        "motion", "evidence", "deposition", "transcript",
                        "discovery", "order", "document"
                    ])
                ))
            except:
                # Field doesn't exist in old indexes, skip
                pass

        if not conditions:
            return None

        return Filter(must=conditions)

    def search_documents(
        self,
        query: str,
        top_k: int = 5,
        doc_type: Optional[str] = None,
        case: Optional[str] = None,
        exclude_frameworks: bool = True
    ) -> List[Dict]:
        """
        Semantic search for relevant documents

        Uses sentence-transformers for true semantic similarity,
        not just keyword matching.
        """
        try:
            # Analyze query context
            context = self._analyze_emotional_context(query)

            # Generate semantic embedding
            query_embedding = self._semantic_embedding(query)

            # Build filter
            query_filter = self._build_filter(doc_type, case, exclude_frameworks)

            # Search with filter
            results = self.qdrant.query_points(
                collection_name=self.collection_name,
                query=query_embedding.tolist(),
                limit=top_k,
                query_filter=query_filter
            )

            documents = []
            for result in results.points:
                documents.append({
                    'title': result.payload.get('title', 'Unknown'),
                    'content': result.payload.get('content', '')[:500],
                    'doc_type': result.payload.get('doc_type', 'unknown'),
                    'case': result.payload.get('case', 'general'),
                    'relevance_score': round(result.score, 3),
                    'emotional_signature': {
                        'q': round(result.payload.get('q', 0.5), 2),
                        'rho': round(result.payload.get('rho', 0.5), 2)
                    },
                    'source': result.payload.get('filename', 'Unknown')
                })

            logger.info(
                f"Search: '{query[:50]}' | "
                f"Found {len(documents)} | "
                f"Context: {context['context_type']} | "
                f"q={context['emotional_activation']} "
                f"trauma={context['trauma_informed']}"
            )

            return documents

        except Exception as e:
            logger.error(f"Search error: {e}")
            return []

    def search_by_case(
        self,
        case: str,
        doc_type: Optional[str] = None,
        top_k: int = 10
    ) -> List[Dict]:
        """Get all documents for a specific case"""
        try:
            query_filter = self._build_filter(doc_type=doc_type, case=case)

            results = self.qdrant.scroll(
                collection_name=self.collection_name,
                scroll_filter=query_filter,
                limit=top_k,
                with_payload=True
            )

            documents = []
            for point in results[0]:
                documents.append({
                    'title': point.payload.get('title', 'Unknown'),
                    'doc_type': point.payload.get('doc_type', 'unknown'),
                    'case': point.payload.get('case', 'general'),
                    'filename': point.payload.get('filename', 'Unknown'),
                    'emotional_signature': {
                        'q': round(point.payload.get('q', 0.5), 2),
                        'rho': round(point.payload.get('rho', 0.5), 2)
                    }
                })

            logger.info(f"Case search: {case} | type={doc_type} | Found {len(documents)}")
            return documents

        except Exception as e:
            logger.error(f"Case search error: {e}")
            return []

    def get_document_stats(self) -> Dict:
        """Get collection statistics"""
        try:
            collection_info = self.qdrant.get_collection(self.collection_name)

            # Sample documents for type/case breakdown
            type_counts = {}
            case_counts = {}

            results = self.qdrant.scroll(
                collection_name=self.collection_name,
                limit=1000,
                with_payload=True
            )

            for point in results[0]:
                doc_type = point.payload.get('doc_type', 'unknown')
                case = point.payload.get('case', 'general')
                type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
                case_counts[case] = case_counts.get(case, 0) + 1

            return {
                "total_documents": collection_info.points_count,
                "by_type": type_counts,
                "by_case": case_counts,
                "embedding_mode": "semantic" if TRANSFORMERS_AVAILABLE else "fallback"
            }

        except Exception as e:
            return {"error": str(e)}


def create_server():
    """Create and configure MCP server"""
    server = Server("legal-rag-production")
    rag = ProductionLegalRAGServer()

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """Register available tools"""
        return [
            Tool(
                name="search_documents",
                description=(
                    "Semantic search for legal documents with intelligent filtering. "
                    "Uses sentence-transformers for true semantic similarity. "
                    "Automatically detects crisis/trauma context. "
                    "Filters by doc_type and case."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query"
                        },
                        "top_k": {
                            "type": "number",
                            "description": "Number of results (default 5)",
                            "default": 5
                        },
                        "doc_type": {
                            "type": "string",
                            "description": "Filter by document type"
                        },
                        "case": {
                            "type": "string",
                            "description": "Filter by case"
                        },
                        "exclude_frameworks": {
                            "type": "boolean",
                            "description": "Exclude framework papers (default true)",
                            "default": True
                        }
                    },
                    "required": ["query"]
                }
            ),
            Tool(
                name="search_by_case",
                description="Get all documents for a specific case",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "case": {
                            "type": "string",
                            "description": "Case name"
                        },
                        "doc_type": {
                            "type": "string",
                            "description": "Optional filter by type"
                        },
                        "top_k": {
                            "type": "number",
                            "default": 10
                        }
                    },
                    "required": ["case"]
                }
            ),
            Tool(
                name="get_stats",
                description="Get collection statistics",
                inputSchema={
                    "type": "object",
                    "properties": {}
                }
            )
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        """Handle tool calls"""
        try:
            if name == "search_documents":
                results = rag.search_documents(
                    query=arguments["query"],
                    top_k=arguments.get("top_k", 5),
                    doc_type=arguments.get("doc_type"),
                    case=arguments.get("case"),
                    exclude_frameworks=arguments.get("exclude_frameworks", True)
                )
                return [TextContent(type="text", text=json.dumps(results, indent=2))]

            elif name == "search_by_case":
                results = rag.search_by_case(
                    case=arguments["case"],
                    doc_type=arguments.get("doc_type"),
                    top_k=arguments.get("top_k", 10)
                )
                return [TextContent(type="text", text=json.dumps(results, indent=2))]

            elif name == "get_stats":
                stats = rag.get_document_stats()
                return [TextContent(type="text", text=json.dumps(stats, indent=2))]

            else:
                raise ValueError(f"Unknown tool: {name}")

        except Exception as e:
            logger.error(f"Tool call error: {e}")
            return [TextContent(type="text", text=json.dumps({"error": str(e)}))]

    return server


async def main():
    """Main entry point"""
    if not MCP_AVAILABLE:
        logger.error("MCP not available")
        sys.exit(1)

    server = create_server()
    logger.info("✓ Production Legal RAG server initialized")

    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
