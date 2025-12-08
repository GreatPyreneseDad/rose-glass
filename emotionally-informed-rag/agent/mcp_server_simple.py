#!/usr/bin/env python3
"""
MCP Server for Emotionally-Informed Legal RAG

Exposes legal document retrieval to Claude Desktop via Model Context Protocol.
Claude Desktop handles the LLM interactions, this server provides the retrieval.
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
    QDRANT_AVAILABLE = True
except ImportError:
    logger.error("Qdrant not available")
    QDRANT_AVAILABLE = False
    sys.exit(1)


class LegalRAGServer:
    """MCP Server for legal document retrieval"""

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
            "maslowsky", "macgregor", "fardell",
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

    def analyze_emotional_context(self, text: str) -> Dict:
        """Analyze emotional signature of query"""
        text_lower = text.lower()

        # Emotional activation (q)
        crisis_words = ['urgent', 'worried', 'help', 'crisis', 'emergency', 'scared', 'terrified']
        q_score = min(1.0, sum(1 for word in crisis_words if word in text_lower) * 0.2 + 0.3)

        # Wisdom depth (rho)
        wisdom_words = ['understand', 'consider', 'principle', 'philosophy', 'why']
        rho_score = min(1.0, sum(1 for word in wisdom_words if word in text_lower) * 0.15 + 0.3)

        # Context type
        context_type = "standard"
        if q_score > 0.7:
            context_type = "crisis"
        elif any(word in text_lower for word in ['research', 'analyze', 'comprehensive']):
            context_type = "mission"

        return {
            "emotional_activation": round(q_score, 2),
            "wisdom_depth": round(rho_score, 2),
            "context_type": context_type,
            "trauma_informed": q_score > 0.6
        }

    def search_documents(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for relevant documents"""
        try:
            query_embedding = self._simple_embedding(query)

            results = self.qdrant.query_points(
                collection_name=self.collection_name,
                query=query_embedding.tolist(),
                limit=top_k
            )

            documents = []
            for result in results.points:
                documents.append({
                    'title': result.payload.get('title', 'Unknown'),
                    'content': result.payload.get('content', ''),
                    'relevance_score': round(result.score, 3),
                    'emotional_signature': {
                        'q': round(result.payload.get('q', 0.5), 2),
                        'rho': round(result.payload.get('rho', 0.5), 2)
                    },
                    'source': result.payload.get('source', 'Unknown')
                })

            return documents

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def get_collection_stats(self) -> Dict:
        """Get statistics about the document collection"""
        try:
            collection_info = self.qdrant.get_collection(self.collection_name)
            return {
                "total_documents": collection_info.points_count,
                "collection_name": self.collection_name,
                "vector_size": collection_info.config.params.vectors.size
            }
        except Exception as e:
            return {"error": str(e)}


# Initialize MCP Server
app = Server("legal-rag")
rag_server = LegalRAGServer()


@app.list_tools()
async def list_tools() -> List[Tool]:
    """List available MCP tools"""
    return [
        Tool(
            name="search_legal_documents",
            description="""Search through legal case documents with emotional awareness.

            Use this tool to find relevant legal documents from the Maslowsky, Fardell, and MacGregor cases.
            The search understands emotional context and provides trauma-informed results.

            Best for: Finding specific case information, legal precedents, or relevant documents.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (e.g., 'custody issues in Maslowsky case')"
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of documents to return (default: 5)",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="analyze_emotional_context",
            description="""Analyze the emotional context and urgency of a query or text.

            Returns emotional activation levels, wisdom depth, and context type (standard/crisis/mission).
            Use this to understand if a query requires trauma-informed handling.

            Best for: Understanding emotional state before responding, detecting crisis situations.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text to analyze for emotional context"
                    }
                },
                "required": ["text"]
            }
        ),
        Tool(
            name="get_collection_info",
            description="""Get information about the legal document collection.

            Returns total number of documents and collection statistics.

            Best for: Understanding what documents are available.""",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> List[TextContent]:
    """Handle tool calls from Claude Desktop"""

    try:
        if name == "search_legal_documents":
            query = arguments.get("query", "")
            num_results = arguments.get("num_results", 5)

            if not query:
                return [TextContent(type="text", text="Error: Query is required")]

            # Analyze emotional context
            emotional_context = rag_server.analyze_emotional_context(query)

            # Search documents
            documents = rag_server.search_documents(query, top_k=num_results)

            # Format response
            response = f"""**Emotional Context Analysis:**
- Emotional Activation (q): {emotional_context['emotional_activation']}
- Wisdom Depth (ρ): {emotional_context['wisdom_depth']}
- Context Type: {emotional_context['context_type']}
- Trauma-Informed: {"Yes" if emotional_context['trauma_informed'] else "No"}

**Found {len(documents)} relevant documents:**

"""

            for i, doc in enumerate(documents, 1):
                response += f"""
---
**{i}. {doc['title']}** (relevance: {doc['relevance_score']})
Emotional signature: q={doc['emotional_signature']['q']}, ρ={doc['emotional_signature']['rho']}

{doc['content'][:800]}...

Source: {doc['source']}
---
"""

            if not documents:
                response += "\nNo documents found matching the query."

            return [TextContent(type="text", text=response)]

        elif name == "analyze_emotional_context":
            text = arguments.get("text", "")

            if not text:
                return [TextContent(type="text", text="Error: Text is required")]

            context = rag_server.analyze_emotional_context(text)

            response = f"""**Emotional Context Analysis:**

- **Emotional Activation (q):** {context['emotional_activation']}
  {"🔴 HIGH - Crisis/urgent context" if context['emotional_activation'] > 0.7 else "🟡 MEDIUM" if context['emotional_activation'] > 0.5 else "🟢 LOW - Calm context"}

- **Wisdom Depth (ρ):** {context['wisdom_depth']}
  {"Deep philosophical/reflective query" if context['wisdom_depth'] > 0.5 else "Practical information query"}

- **Context Type:** {context['context_type'].upper()}

- **Trauma-Informed Approach:** {"REQUIRED - Use empathetic, supportive language" if context['trauma_informed'] else "Standard approach appropriate"}

**Recommendation:** {"Prioritize immediate, actionable support. Be extra empathetic." if context['context_type'] == 'crisis' else "Comprehensive research-based response appropriate" if context['context_type'] == 'mission' else "Standard balanced response"}
"""

            return [TextContent(type="text", text=response)]

        elif name == "get_collection_info":
            stats = rag_server.get_collection_stats()

            response = f"""**Legal Document Collection Statistics:**

- Total Documents: {stats.get('total_documents', 'Unknown')}
- Collection: {stats.get('collection_name', 'Unknown')}
- Vector Dimension: {stats.get('vector_size', 'Unknown')}

**Available Cases:**
- Maslowsky v. MacGregor
- Fardell Case
- MacGregor Case Files
- Various legal hearings and documents

The collection includes discovery documents, motions, hearings, and case files with emotional signature tagging for trauma-informed retrieval.
"""

            return [TextContent(type="text", text=response)]

        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

    except Exception as e:
        logger.error(f"Tool execution error: {e}")
        return [TextContent(type="text", text=f"Error: {str(e)}")]


async def main():
    """Run MCP server"""
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
