#!/usr/bin/env python3
"""
MCP Server exposing Emotional RAG Agent tools to Claude Code

This server implements the Model Context Protocol (MCP) to expose
the Emotionally-Informed RAG Agent as tools that Claude Code can use.
"""

import json
import sys
from pathlib import Path

# Add agent to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from mcp.server import Server
    from mcp.types import Tool, TextContent
    MCP_AVAILABLE = True
except ImportError:
    print("Warning: MCP not available. Install with: pip install mcp", file=sys.stderr)
    MCP_AVAILABLE = False

from emotional_agent import EmotionalRAGAgent

# Initialize MCP server
if MCP_AVAILABLE:
    app = Server("emotional-rag-agent")
else:
    # Fallback for testing
    class MockServer:
        def tool(self):
            def decorator(func):
                return func
            return decorator
    app = MockServer()

# Initialize agent
try:
    agent = EmotionalRAGAgent()
    print(f"✓ Emotional RAG Agent initialized", file=sys.stderr)
except Exception as e:
    print(f"⚠ Agent initialization warning: {e}", file=sys.stderr)
    agent = None


@app.tool()
async def analyze_emotional_signature(text: str) -> str:
    """
    Analyze text through Rose Glass framework.

    Returns emotional signature with dimensions:
    - psi: internal consistency (0-1)
    - rho: wisdom depth (0-1)
    - q: emotional activation (0-1)
    - f: social belonging (0-1)
    - tau: temporal depth (0-1)
    - context_type: trust/mission/crisis/standard

    Example:
        analyze_emotional_signature("I'm very worried about this legal case")
    """
    if agent is None:
        return json.dumps({"error": "Agent not initialized"})

    try:
        signature = agent.analyze_query(text)
        return json.dumps({
            'psi': signature.psi,
            'rho': signature.rho,
            'q': signature.q,
            'f': signature.f,
            'tau': signature.tau,
            'lens': signature.lens,
            'context_type': signature.context_type
        }, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@app.tool()
async def query_with_emotion(query: str) -> str:
    """
    Process query through emotionally-informed RAG.

    Retrieves documents matching emotional signature and generates
    contextually appropriate response.

    The response includes:
    - response: generated text
    - signature: emotional analysis
    - documents_used: number of documents retrieved
    - gradient_status: escalation monitoring
    - context_type: detected context

    Example:
        query_with_emotion("What are trauma-informed approaches in legal cases?")
    """
    if agent is None:
        return json.dumps({"error": "Agent not initialized"})

    try:
        result = agent.query(query)
        return json.dumps(result, indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


@app.tool()
async def check_escalation_risk() -> str:
    """
    Check conversation gradient for escalation indicators.

    Returns risk assessment and recommendations based on
    tracking emotional activation (q) changes over time.

    Returns:
    - status: normal or escalation_detected
    - risk: escalation risk score (if applicable)
    - recommendation: suggested action

    Example:
        check_escalation_risk()
    """
    if agent is None:
        return json.dumps({"error": "Agent not initialized"})

    try:
        if len(agent.pattern_history) < 3:
            return json.dumps({
                "status": "insufficient_data",
                "snapshots": len(agent.pattern_history),
                "message": "Need at least 3 conversation turns for gradient analysis"
            })

        # Get recent emotional activation values
        recent_q = [s.q for s in agent.pattern_history[-3:]]
        q_increase = recent_q[-1] - recent_q[0]

        if q_increase > 0.3:
            return json.dumps({
                "alert": "escalation_detected",
                "risk": q_increase,
                "recent_q_values": recent_q,
                "recommendation": "Consider intervention or human support",
                "threshold": 0.3
            }, indent=2)
        else:
            return json.dumps({
                "status": "normal",
                "risk": q_increase,
                "recent_q_values": recent_q
            }, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})


@app.tool()
async def get_conversation_history() -> str:
    """
    Retrieve conversation history with emotional signatures.

    Returns all conversation turns with:
    - query: user input
    - response: agent output
    - signature: emotional analysis
    - timestamp: when the interaction occurred

    Example:
        get_conversation_history()
    """
    if agent is None:
        return json.dumps({"error": "Agent not initialized"})

    try:
        return json.dumps({
            "conversation_count": len(agent.conversation_history),
            "history": agent.conversation_history
        }, indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


@app.tool()
async def switch_cultural_lens(lens: str) -> str:
    """
    Switch cultural calibration lens.

    Available lenses:
    - modern_digital: Digital native communication
    - modern_academic: Academic writing
    - indigenous_oral: Oral storytelling patterns
    - trauma_informed: High-stress contexts
    - neurodivergent_autism: Autism spectrum calibration
    - neurodivergent_adhd: ADHD communication patterns

    Example:
        switch_cultural_lens("trauma_informed")
    """
    if agent is None:
        return json.dumps({"error": "Agent not initialized"})

    try:
        valid_lenses = [
            'modern_digital', 'modern_academic', 'indigenous_oral',
            'trauma_informed', 'neurodivergent_autism', 'neurodivergent_adhd'
        ]

        if lens not in valid_lenses:
            return json.dumps({
                "error": f"Invalid lens: {lens}",
                "valid_lenses": valid_lenses
            })

        agent.config['rose_glass']['default_lens'] = lens
        return json.dumps({
            "lens": lens,
            "status": "switched",
            "message": f"Cultural lens changed to: {lens}"
        }, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@app.tool()
async def get_agent_status() -> str:
    """
    Get agent system status and capabilities.

    Returns information about:
    - Available services (Qdrant, Elasticsearch, Claude)
    - Configuration
    - Conversation state
    - Version info

    Example:
        get_agent_status()
    """
    if agent is None:
        return json.dumps({"error": "Agent not initialized"})

    try:
        return json.dumps({
            "agent": {
                "name": agent.config['agent']['name'],
                "version": agent.config['agent']['version']
            },
            "services": {
                "qdrant": agent.qdrant is not None,
                "elasticsearch": agent.elasticsearch is not None,
                "claude": agent.claude is not None
            },
            "conversation": {
                "turns": len(agent.conversation_history),
                "pattern_snapshots": len(agent.pattern_history)
            },
            "config": {
                "lens": agent.config['rose_glass']['default_lens'],
                "escalation_threshold": agent.config['monitoring']['escalation_threshold'],
                "top_k": agent.config['retrieval']['top_k']
            }
        }, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


# For standalone testing
def main():
    """Test the MCP server tools"""
    import asyncio

    print("Testing MCP Server Tools\n")
    print("=" * 60)

    async def test_tools():
        # Test 1: Analyze emotional signature
        print("\n1. Testing emotional signature analysis...")
        result = await analyze_emotional_signature(
            "I'm really worried about this case and need urgent help!"
        )
        print(result)

        # Test 2: Query with emotion
        print("\n2. Testing query with emotion...")
        result = await query_with_emotion(
            "What are trauma-informed approaches in legal cases?"
        )
        print(result)

        # Test 3: Check status
        print("\n3. Testing agent status...")
        result = await get_agent_status()
        print(result)

        # Test 4: Switch lens
        print("\n4. Testing lens switch...")
        result = await switch_cultural_lens("trauma_informed")
        print(result)

        # Test 5: Escalation check
        print("\n5. Testing escalation check...")
        result = await check_escalation_risk()
        print(result)

    asyncio.run(test_tools())
    print("\n" + "=" * 60)
    print("MCP Server tools test complete")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        main()
    elif MCP_AVAILABLE:
        import asyncio
        print("Starting MCP server...", file=sys.stderr)
        asyncio.run(app.run())
    else:
        print("MCP not available. Install with: pip install mcp")
        print("Running in test mode instead...")
        main()
