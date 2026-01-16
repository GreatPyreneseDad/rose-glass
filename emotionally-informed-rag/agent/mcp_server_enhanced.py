#!/usr/bin/env python3
"""
Enhanced MCP Server for Emotionally-Informed Legal RAG
======================================================

Adds intelligent filtering by doc_type, case, and speaker.
Prevents framework papers from polluting case-specific searches.

Usage:
    Update ~/.claude/settings.json to use this server instead of mcp_server_simple.py
"""

import sys
import logging
from pathlib import Path
from typing import List, Dict, Optional
import json
import numpy as np
import re
from collections import Counter

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


class EnhancedLegalRAGServer:
    """Enhanced MCP Server with intelligent filtering"""

    def __init__(
        self,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        collection_name: str = "knowledge_base"
    ):
        self.collection_name = collection_name

        # Legal vocabulary for embeddings
        self.vocabulary = self._build_legal_vocabulary()

        # Initialize Qdrant
        self.qdrant = QdrantClient(host=qdrant_host, port=qdrant_port)
        logger.info(f"✓ Connected to Qdrant at {qdrant_host}:{qdrant_port}")

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
            "maslowsky", "macgregor", "fardell", "stokes", "kowitz",
            # Emotional terms
            "trauma", "emotional", "distress", "anxiety", "fear", "worried", "concerned",
            "urgent", "crisis", "help", "support", "safety", "protection", "harm",
        ]

    def _simple_embedding(self, text: str) -> np.ndarray:
        """Create simple TF-IDF-style embedding"""
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        word_counts = Counter(words)

        embedding = np.zeros(384)

        for idx, vocab_term in enumerate(self.vocabulary[:384]):
            if vocab_term in word_counts:
                tf = word_counts[vocab_term] / len(words) if words else 0
                embedding[idx] = tf

        non_vocab_words = [w for w in words if w not in self.vocabulary]
        for word in non_vocab_words[:50]:
            hash_val = hash(word)
            for i in range(3):
                idx = (hash_val + i * 17) % 384
                embedding[idx] += 0.1

        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def _build_filter(
        self,
        doc_type: Optional[str] = None,
        case: Optional[str] = None,
        exclude_frameworks: bool = False
    ) -> Optional[Filter]:
        """
        Build Qdrant filter for metadata

        Args:
            doc_type: Filter by document type (motion, evidence, deposition, etc.)
            case: Filter by case (maslowsky, fardell, kowitz, etc.)
            exclude_frameworks: Exclude GCT/philosophical papers

        Returns:
            Qdrant Filter object or None
        """
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
            # Exclude documents with doc_type='framework'
            conditions.append(FieldCondition(
                key="doc_type",
                match=MatchAny(any=[
                    "motion", "evidence", "deposition", "transcript",
                    "discovery", "order", "document"
                ])
            ))

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
        Search for relevant documents with intelligent filtering

        Args:
            query: Search query
            top_k: Number of results to return
            doc_type: Filter by document type
            case: Filter by case
            exclude_frameworks: Exclude GCT framework papers (default True)

        Returns:
            List of matching documents
        """
        try:
            query_embedding = self._simple_embedding(query)

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
                    'content': result.payload.get('content', '')[:500],  # Truncate for display
                    'doc_type': result.payload.get('doc_type', 'unknown'),
                    'case': result.payload.get('case', 'general'),
                    'relevance_score': round(result.score, 3),
                    'emotional_signature': {
                        'q': round(result.payload.get('q', 0.5), 2),
                        'rho': round(result.payload.get('rho', 0.5), 2)
                    },
                    'source': result.payload.get('filename', 'Unknown')
                })

            logger.info(f"Search: '{query[:50]}' | Found {len(documents)} docs | Filters: type={doc_type} case={case}")
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
        """
        Get all documents for a specific case

        Args:
            case: Case name (maslowsky, fardell, kowitz, stokes)
            doc_type: Optional filter by document type
            top_k: Number of results

        Returns:
            List of case documents
        """
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

            logger.info(f"Case search: {case} | type={doc_type} | Found {len(documents)} docs")
            return documents

        except Exception as e:
            logger.error(f"Case search error: {e}")
            return []

    def get_document_stats(self) -> Dict:
        """Get collection statistics with type breakdown"""
        try:
            collection_info = self.qdrant.get_collection(self.collection_name)

            # Get type breakdown
            type_counts = {}
            case_counts = {}

            results = self.qdrant.scroll(
                collection_name=self.collection_name,
                limit=1000,  # Sample first 1000
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
                "vector_size": collection_info.config.params.vectors.size
            }

        except Exception as e:
            logger.error(f"Stats error: {e}")
            return {"error": str(e)}


def create_server():
    """Create and configure MCP server"""
    server = Server("legal-rag-enhanced")
    rag = EnhancedLegalRAGServer()

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """Register available tools"""
        return [
            Tool(
                name="search_documents",
                description=(
                    "Search legal documents with intelligent filtering. "
                    "Automatically excludes framework papers unless specified. "
                    "Supports filtering by doc_type (motion, evidence, deposition, "
                    "transcript, discovery, order) and case (maslowsky, fardell, "
                    "kowitz, stokes)."
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
                            "description": "Filter by document type",
                            "enum": ["motion", "evidence", "deposition", "transcript",
                                   "discovery", "order", "document"]
                        },
                        "case": {
                            "type": "string",
                            "description": "Filter by case",
                            "enum": ["maslowsky", "fardell", "kowitz", "stokes", "general"]
                        },
                        "exclude_frameworks": {
                            "type": "boolean",
                            "description": "Exclude GCT framework papers (default true)",
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
                            "description": "Case name",
                            "enum": ["maslowsky", "fardell", "kowitz", "stokes"]
                        },
                        "doc_type": {
                            "type": "string",
                            "description": "Optional filter by document type"
                        },
                        "top_k": {
                            "type": "number",
                            "description": "Number of results",
                            "default": 10
                        }
                    },
                    "required": ["case"]
                }
            ),
            Tool(
                name="get_stats",
                description="Get collection statistics with type and case breakdown",
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
                return [TextContent(
                    type="text",
                    text=json.dumps(results, indent=2)
                )]

            elif name == "search_by_case":
                results = rag.search_by_case(
                    case=arguments["case"],
                    doc_type=arguments.get("doc_type"),
                    top_k=arguments.get("top_k", 10)
                )
                return [TextContent(
                    type="text",
                    text=json.dumps(results, indent=2)
                )]

            elif name == "get_stats":
                stats = rag.get_document_stats()
                return [TextContent(
                    type="text",
                    text=json.dumps(stats, indent=2)
                )]

            else:
                raise ValueError(f"Unknown tool: {name}")

        except Exception as e:
            logger.error(f"Tool call error: {e}")
            return [TextContent(
                type="text",
                text=json.dumps({"error": str(e)})
            )]

    return server


async def main():
    """Main entry point"""
    if not MCP_AVAILABLE:
        logger.error("MCP not available")
        sys.exit(1)

    server = create_server()
    logger.info("✓ Enhanced Legal RAG server initialized")

    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
