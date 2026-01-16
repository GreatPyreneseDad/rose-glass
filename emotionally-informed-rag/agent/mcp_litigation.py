#!/usr/bin/env python3
"""
MCP Server for Real-Time Litigation Support

Exposes Rose Glass litigation tools to Claude Desktop:
- Analyze testimony for contradictions
- Track witness coherence in real-time
- Generate cross-examination questions
- Search legal documents for impeachment evidence
"""

import sys
import logging
from pathlib import Path
from typing import List, Dict, Optional
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))  # agent directory for mcp_server_simple

try:
    from mcp.server import Server
    from mcp.types import Tool, TextContent
    import mcp.server.stdio
    MCP_AVAILABLE = True
except ImportError:
    logger.error("MCP not available. Install with: pip install mcp")
    MCP_AVAILABLE = False
    sys.exit(1)

# Import litigation components
try:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "rose-glass/src"))
    from litigation.litigation_lens import TestimonyStatement, ContradictionDetector, ContradictionFlag
    from core.rose_glass_lens import RoseGlass
    LITIGATION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Litigation module not available: {e}")
    LITIGATION_AVAILABLE = False

# Import document retrieval
try:
    from mcp_server_simple import LegalRAGServer
    RAG_AVAILABLE = True
except ImportError:
    logger.warning("RAG server not available")
    RAG_AVAILABLE = False


class LitigationMCPServer:
    """MCP Server combining litigation analysis + document retrieval"""

    def __init__(self):
        # Initialize Rose Glass
        if LITIGATION_AVAILABLE:
            self.rose_glass = RoseGlass()
            self.detector = ContradictionDetector(self.rose_glass)
            logger.info("✓ Litigation components initialized")
        else:
            self.rose_glass = None
            self.detector = None
            logger.warning("⚠ Running without litigation components")

        # Initialize document retrieval
        if RAG_AVAILABLE:
            self.rag_server = LegalRAGServer()
            logger.info("✓ Legal document retrieval initialized")
        else:
            self.rag_server = None
            logger.warning("⚠ Running without document retrieval")

        # Track testimony for the session
        self.session_statements: List[TestimonyStatement] = []
        self.session_contradictions: List[ContradictionFlag] = []

    def analyze_testimony(self, speaker: str, text: str) -> Dict:
        """Analyze a single statement from testimony"""
        if not LITIGATION_AVAILABLE:
            return {"error": "Litigation components not available"}

        # Create statement with Rose Glass analysis
        statement = TestimonyStatement.create(
            speaker=speaker,
            text=text,
            position=len(self.session_statements),
            rose_glass=self.rose_glass
        )

        # Add to session
        self.session_statements.append(statement)

        # Check for contradictions
        new_flags = self.detector.add_statement(statement)
        self.session_contradictions.extend(new_flags)

        return {
            "statement_id": len(self.session_statements) - 1,
            "speaker": speaker,
            "coherence": round(statement.coherence, 3),
            "variables": {
                "psi": round(statement.variables['psi'], 3),
                "rho": round(statement.variables['rho'], 3),
                "q": round(statement.variables['q'], 3),
                "f": round(statement.variables['f'], 3)
            },
            "contradictions_found": len(new_flags),
            "total_statements": len(self.session_statements),
            "total_flags": len(self.session_contradictions)
        }

    def get_contradictions(self, min_severity: float = 0.5) -> List[Dict]:
        """Get all detected contradictions above severity threshold"""
        if not self.session_contradictions:
            return []

        filtered = [
            {
                "severity": round(flag.severity, 3),
                "type": flag.flag_type,
                "current_statement": flag.current_statement.text,
                "conflicting_statement": flag.conflicting_statement.text,
                "suggested_question": flag.suggested_question,
                "speaker": flag.current_statement.speaker,
                "timestamps": {
                    "current": flag.current_statement.timestamp.isoformat(),
                    "conflicting": flag.conflicting_statement.timestamp.isoformat()
                }
            }
            for flag in self.session_contradictions
            if flag.severity >= min_severity
        ]

        # Sort by severity (highest first)
        filtered.sort(key=lambda x: x['severity'], reverse=True)
        return filtered

    def generate_cross_exam_questions(self, speaker: str, top_n: int = 5) -> List[str]:
        """Generate cross-examination questions for a speaker"""
        # Get speaker's contradictions
        speaker_flags = [
            flag for flag in self.session_contradictions
            if flag.current_statement.speaker == speaker
        ]

        if not speaker_flags:
            return []

        # Sort by severity
        speaker_flags.sort(key=lambda x: x.severity, reverse=True)

        # Take top N
        return [flag.suggested_question for flag in speaker_flags[:top_n]]

    def search_for_impeachment(self, claim: str, top_k: int = 3) -> List[Dict]:
        """Search legal documents for evidence to impeach a claim"""
        if not self.rag_server:
            return []

        # Search with high emotional activation (urgent context)
        documents = self.rag_server.search_documents(claim, top_k=top_k)

        return [
            {
                "title": doc["title"],
                "excerpt": doc["content"][:500] + "...",
                "relevance": doc["relevance_score"],
                "source": doc["source"]
            }
            for doc in documents
        ]

    def get_session_summary(self) -> Dict:
        """Get summary of current testimony session"""
        if not self.session_statements:
            return {
                "total_statements": 0,
                "speakers": [],
                "contradictions": 0,
                "average_coherence": 0
            }

        speakers = list(set(s.speaker for s in self.session_statements))
        avg_coherence = sum(s.coherence for s in self.session_statements) / len(self.session_statements)

        return {
            "total_statements": len(self.session_statements),
            "speakers": speakers,
            "contradictions": len(self.session_contradictions),
            "high_severity_flags": len([f for f in self.session_contradictions if f.severity > 0.7]),
            "average_coherence": round(avg_coherence, 3),
            "coherence_by_speaker": {
                speaker: round(
                    sum(s.coherence for s in self.session_statements if s.speaker == speaker) /
                    len([s for s in self.session_statements if s.speaker == speaker]),
                    3
                )
                for speaker in speakers
            }
        }

    def clear_session(self):
        """Clear current testimony session"""
        self.session_statements = []
        self.session_contradictions = []
        if self.detector:
            self.detector = ContradictionDetector(self.rose_glass)


# Initialize MCP Server
app = Server("litigation-support")
litigation_server = LitigationMCPServer()


@app.list_tools()
async def list_tools() -> List[Tool]:
    """List available litigation MCP tools"""
    return [
        Tool(
            name="analyze_testimony",
            description="""Analyze a statement from testimony through Rose Glass.

            Extracts GCT variables (Ψ, ρ, q, f), calculates coherence, detects contradictions.
            Automatically tracks all statements in the session for pattern analysis.

            Use during hearings to get real-time coherence analysis.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "speaker": {
                        "type": "string",
                        "description": "Name of speaker (e.g., 'Witness Smith', 'Opposing Counsel')"
                    },
                    "text": {
                        "type": "string",
                        "description": "Exact text of the testimony statement"
                    }
                },
                "required": ["speaker", "text"]
            }
        ),
        Tool(
            name="get_contradictions",
            description="""Get all detected contradictions from current testimony session.

            Returns contradictions sorted by severity with suggested cross-examination questions.
            Filter by minimum severity (0-1 scale).

            Use to review all contradictions before cross-examination.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "min_severity": {
                        "type": "number",
                        "description": "Minimum severity threshold (0-1). Default: 0.5",
                        "default": 0.5
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="generate_cross_exam_questions",
            description="""Generate cross-examination questions for a specific speaker.

            Based on detected contradictions and coherence drops.
            Returns top questions sorted by severity/impact.

            Use when preparing to cross-examine a witness.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "speaker": {
                        "type": "string",
                        "description": "Name of speaker to generate questions for"
                    },
                    "top_n": {
                        "type": "integer",
                        "description": "Number of questions to generate. Default: 5",
                        "default": 5
                    }
                },
                "required": ["speaker"]
            }
        ),
        Tool(
            name="search_for_impeachment",
            description="""Search legal documents for evidence to impeach a claim.

            Searches the 2,672-document collection for evidence contradicting a witness statement.
            Returns relevant documents that could be used for impeachment.

            Use when witness makes a claim you want to challenge with filed evidence.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "claim": {
                        "type": "string",
                        "description": "The claim or statement to find contradicting evidence for"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of documents to return. Default: 3",
                        "default": 3
                    }
                },
                "required": ["claim"]
            }
        ),
        Tool(
            name="get_session_summary",
            description="""Get summary statistics for current testimony session.

            Shows total statements, speakers, contradictions, average coherence by speaker.

            Use to get overview of testimony session.""",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="clear_session",
            description="""Clear current testimony session and start fresh.

            Removes all tracked statements and contradictions.
            Use at the start of a new hearing or witness examination.""",
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
        if name == "analyze_testimony":
            speaker = arguments.get("speaker", "Unknown")
            text = arguments.get("text", "")

            if not text:
                return [TextContent(type="text", text="Error: Text is required")]

            result = litigation_server.analyze_testimony(speaker, text)

            if "error" in result:
                return [TextContent(type="text", text=f"Error: {result['error']}")]

            response = f"""**Testimony Analysis - Statement #{result['statement_id']}**

**Speaker:** {result['speaker']}

**Rose Glass Variables:**
- Ψ (Internal Consistency): {result['variables']['psi']}
- ρ (Wisdom/Specificity): {result['variables']['rho']}
- q (Emotional Activation): {result['variables']['q']}
- f (Social Belonging): {result['variables']['f']}

**Coherence:** {result['coherence']} {"🟢" if result['coherence'] > 0.7 else "🟡" if result['coherence'] > 0.5 else "🔴"}

**Contradictions:** {result['contradictions_found']} new flag(s) detected

**Session Stats:** {result['total_statements']} statements, {result['total_flags']} total flags
"""

            if result['contradictions_found'] > 0:
                response += f"\n⚠️ **ALERT:** {result['contradictions_found']} contradiction(s) detected! Use `get_contradictions` to review."

            return [TextContent(type="text", text=response)]

        elif name == "get_contradictions":
            min_severity = arguments.get("min_severity", 0.5)
            contradictions = litigation_server.get_contradictions(min_severity)

            if not contradictions:
                return [TextContent(type="text", text=f"No contradictions found above severity {min_severity}")]

            response = f"**Detected Contradictions (≥{min_severity} severity)**\n\n"

            for i, flag in enumerate(contradictions, 1):
                response += f"""---
**#{i} - Severity: {flag['severity']}** {"🔴" if flag['severity'] > 0.8 else "🟡"}

**Type:** {flag['type']}
**Speaker:** {flag['speaker']}

**Current Statement** ({flag['timestamps']['current']}):
> {flag['current_statement']}

**Conflicting Statement** ({flag['timestamps']['conflicting']}):
> {flag['conflicting_statement']}

**Suggested Question:**
> {flag['suggested_question']}

"""

            return [TextContent(type="text", text=response)]

        elif name == "generate_cross_exam_questions":
            speaker = arguments.get("speaker", "")
            top_n = arguments.get("top_n", 5)

            if not speaker:
                return [TextContent(type="text", text="Error: Speaker is required")]

            questions = litigation_server.generate_cross_exam_questions(speaker, top_n)

            if not questions:
                return [TextContent(type="text", text=f"No contradictions detected for {speaker}")]

            response = f"**Cross-Examination Questions for {speaker}**\n\n"
            for i, question in enumerate(questions, 1):
                response += f"{i}. {question}\n\n"

            return [TextContent(type="text", text=response)]

        elif name == "search_for_impeachment":
            claim = arguments.get("claim", "")
            top_k = arguments.get("top_k", 3)

            if not claim:
                return [TextContent(type="text", text="Error: Claim is required")]

            documents = litigation_server.search_for_impeachment(claim, top_k)

            if not documents:
                return [TextContent(type="text", text="No relevant impeachment evidence found")]

            response = f"**Impeachment Evidence for:** \"{claim}\"\n\n"

            for i, doc in enumerate(documents, 1):
                response += f"""---
**{i}. {doc['title']}** (Relevance: {doc['relevance']})

{doc['excerpt']}

Source: {doc['source']}
---

"""

            return [TextContent(type="text", text=response)]

        elif name == "get_session_summary":
            summary = litigation_server.get_session_summary()

            response = f"""**Testimony Session Summary**

**Statements:** {summary['total_statements']}
**Speakers:** {', '.join(summary['speakers']) if summary['speakers'] else 'None'}
**Total Contradictions:** {summary['contradictions']}
**High Severity Flags:** {summary.get('high_severity_flags', 0)} (>0.7)
**Average Coherence:** {summary['average_coherence']}

**Coherence by Speaker:**
"""
            for speaker, coherence in summary.get('coherence_by_speaker', {}).items():
                indicator = "🟢" if coherence > 0.7 else "🟡" if coherence > 0.5 else "🔴"
                response += f"- {speaker}: {coherence} {indicator}\n"

            return [TextContent(type="text", text=response)]

        elif name == "clear_session":
            litigation_server.clear_session()
            return [TextContent(type="text", text="✓ Session cleared. Ready for new testimony.")]

        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

    except Exception as e:
        logger.error(f"Tool execution error: {e}")
        import traceback
        traceback.print_exc()
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
