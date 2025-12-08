# Emotionally-Informed RAG Agent

**Production-ready Retrieval-Augmented Generation with Rose Glass emotional intelligence**

Part of the [Rose Glass Framework](https://github.com/GreatPyreneseDad/rose-glass) - A cultural translation layer for AI systems.

---

## Overview

This is a complete RAG (Retrieval-Augmented Generation) implementation that combines:

- **Rose Glass Framework** - Emotional pattern translation (Ψ, ρ, q, f, τ dimensions)
- **Hybrid Vector Search** - Qdrant (semantic) + Elasticsearch (keyword) + RRF fusion
- **Real-Time Litigation Support** - Contradiction detection, coherence tracking, cross-examination prompts
- **Claude Desktop Integration** - MCP servers for seamless AI assistance
- **Trauma-Informed Retrieval** - Emotionally-aware document search and response generation
- **2,600+ Document Collection** - Pre-indexed legal case files with emotional signatures

**Status:** ✅ Production-ready, actively used in legal practice

---

## What's Included

### Core Components

```
emotionally-informed-rag/
├── agent/
│   ├── emotional_agent.py          # Full RAG agent (500+ lines)
│   ├── simple_agent.py              # Simplified agent (Python 3.13 compatible)
│   ├── mcp_server_simple.py         # MCP server for document retrieval
│   ├── mcp_litigation.py            # MCP server for litigation support
│   ├── tools/
│   │   ├── ingest.py                # Document ingestion with embeddings
│   │   └── ingest_simple.py         # Simplified ingestion (no torch required)
│   └── config/
│       └── agent_config.yaml        # Agent configuration
├── tests/
│   ├── test_agent.py                # Comprehensive test suite
│   └── conftest.py                  # Pytest fixtures
├── api/
│   └── main.py                      # FastAPI server (optional)
├── docker-compose.yml               # Infrastructure (Qdrant, Elasticsearch, Redis)
├── standalone_demo.py               # Zero-setup demo
└── example_implementation.py        # Integration examples
```

### Documentation

- **[QUICKSTART.md](./QUICKSTART.md)** - Get running in 5 minutes
- **[M4_MAX_BUILD_GUIDE.md](./M4_MAX_BUILD_GUIDE.md)** - Complete M4 Max setup
- **[ARCHITECTURE.md](./ARCHITECTURE.md)** - System architecture deep-dive
- **[CLAUDE_DESKTOP_SETUP.md](./CLAUDE_DESKTOP_SETUP.md)** - MCP integration guide
- **[LITIGATION_SUPPORT_GUIDE.md](./LITIGATION_SUPPORT_GUIDE.md)** - Real-time courtroom AI

---

## Quick Start

### 30-Second Demo (No Setup)

```bash
cd emotionally-informed-rag
python3 standalone_demo.py
```

This works immediately with zero dependencies!

### 5-Minute Setup (Full System)

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start infrastructure
docker compose up -d

# 4. Run agent
python3 agent/simple_agent.py --interactive
```

### Claude Desktop Integration

1. **Copy MCP config:**
   ```bash
   cp claude_desktop_config.json ~/Library/Application\ Support/Claude/
   ```

2. **Restart Claude Desktop**

3. **Available tools:**
   - `search_legal_documents` - Search 2,600+ documents
   - `analyze_emotional_context` - Emotional signature analysis
   - `analyze_testimony` - Real-time contradiction detection
   - `generate_cross_exam_questions` - Auto-generate questions
   - `search_for_impeachment` - Find contradicting evidence

See [CLAUDE_DESKTOP_SETUP.md](./CLAUDE_DESKTOP_SETUP.md) for full guide.

---

## Key Features

### 1. Emotional Signature Analysis

Every query and document is analyzed through Rose Glass dimensions:

- **Ψ (psi)** - Internal consistency
- **ρ (rho)** - Wisdom depth
- **q** - Emotional activation
- **f** - Social belonging
- **τ (tau)** - Temporal depth

This enables **trauma-informed responses** that adapt to emotional context.

### 2. Hybrid Retrieval

Combines three retrieval methods for better results than pure vector or keyword search alone.

### 3. Real-Time Litigation Support

Built for actual courtroom use - tracks coherence, detects contradictions, generates questions.

### 4. Context-Type Detection

Automatically detects query context (Crisis/Mission/Standard) and adapts response.

### 5. Gradient Tracking

Monitors conversation patterns for escalation and early intervention.

---

## Architecture

See [ARCHITECTURE.md](./ARCHITECTURE.md) for complete system design.

---

## Use Cases

- Legal research & case preparation
- Real-time courtroom support
- Crisis response with trauma-informed handling
- Comprehensive research queries

---

## Configuration

Edit `agent/config/agent_config.yaml` to customize:
- Cultural lens (trauma_informed, modern_digital, etc.)
- Retrieval parameters
- LLM provider (Anthropic or Ollama)
- Monitoring thresholds

---

## Testing

```bash
pytest tests/ -v
pytest tests/ --cov=agent --cov-report=html
```

---

## License

MIT License - See [LICENSE](../LICENSE) in main repo.

---

*"Understanding precedes judgment. Translation enables understanding."* 🌹

**Ready to use!** See [QUICKSTART.md](./QUICKSTART.md) to get started.
