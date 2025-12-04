# Emotionally-Informed RAG - Quick Start Guide

Get started with the Emotionally-Informed RAG Agent in 5 minutes.

---

## What You've Got

A complete **production-ready RAG system** with emotional intelligence:

- ✅ **Working standalone demo** (no setup needed)
- ✅ **Production agent** with Claude Code integration
- ✅ **MCP server** for tool exposure
- ✅ **Document ingestion** with emotional tagging
- ✅ **Full test suite**
- ✅ **Docker infrastructure**
- ✅ **M4 Max optimized** (ARM64 native)

---

## 30-Second Test (No Setup)

```bash
# Navigate to project
cd ~/rose-glass/emotionally-informed-rag

# Run standalone demo
python3 standalone_demo.py
```

This works immediately! No Docker, no databases, no dependencies.

---

## 5-Minute Setup (Production Agent)

### Step 1: Install Dependencies

```bash
cd ~/rose-glass/emotionally-informed-rag

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### Step 2: Run Interactive Agent

```bash
# Run agent in interactive mode
python3 agent/emotional_agent.py --interactive
```

Try these queries:
- `"I'm really worried about my legal case"` (high emotion)
- `"What are philosophical principles of justice?"` (high wisdom)
- `"Research all approaches to contract law"` (mission mode)

---

## 10-Minute Setup (Full Stack)

### Step 1: Start Infrastructure

```bash
cd ~/rose-glass/emotionally-informed-rag

# Start Qdrant, Elasticsearch, Redis
docker compose up -d

# Verify services
curl http://localhost:6333/health      # Qdrant
curl http://localhost:9200             # Elasticsearch
redis-cli ping                         # Redis
```

### Step 2: Ingest Documents

```bash
# Activate venv
source venv/bin/activate

# Ingest your documents
python3 agent/tools/ingest.py ~/Documents/legal-cases/

# Or ingest specific file
python3 agent/tools/ingest.py ~/Documents/important.txt --title "Important Doc"
```

### Step 3: Query with Full RAG

```bash
# Run with full retrieval
python3 agent/emotional_agent.py --interactive

# Or single query
python3 agent/emotional_agent.py "What are trauma-informed legal approaches?"
```

---

## Claude Code Integration (MCP)

### Step 1: Install Claude Code

```bash
# Install Claude Code CLI (if not already installed)
npm install -g @anthropic-ai/claude-code

# Authenticate
claude auth login
```

### Step 2: Configure MCP Server

Create `.claude/settings.json` in your project:

```json
{
  "mcpServers": {
    "emotional-rag": {
      "command": "python3",
      "args": ["agent/mcp_server.py"],
      "cwd": "${workspaceFolder}",
      "env": {
        "PYTHONPATH": "${workspaceFolder}/../src:${workspaceFolder}/../../RoseGlassLE/src"
      }
    }
  }
}
```

### Step 3: Use in Claude Code

```bash
cd ~/rose-glass/emotionally-informed-rag
claude
```

Then use the tools:
```
> Use analyze_emotional_signature to analyze: "I'm very worried about this"
> Query with emotional awareness: "What are my legal options?"
> Check escalation risk in this conversation
```

---

## Testing

### Run Test Suite

```bash
cd ~/rose-glass/emotionally-informed-rag
source venv/bin/activate

# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_agent.py::TestEmotionalAnalysis -v

# With coverage
pytest tests/ --cov=agent --cov-report=html
```

### Test MCP Server

```bash
# Test MCP tools
python3 agent/mcp_server.py test
```

---

## Configuration

### Agent Config

Edit `agent/config/agent_config.yaml`:

```yaml
rose_glass:
  default_lens: "modern_digital"  # or trauma_informed, neurodivergent_autism, etc.

retrieval:
  top_k: 10  # Number of documents to retrieve
  hybrid_alpha: 0.7  # Vector vs keyword weight

generation:
  provider: "anthropic"  # or "ollama" for local
  model: "claude-sonnet-4-20250514"
  max_tokens: 4096

monitoring:
  gradient_tracking: true
  escalation_threshold: 0.7
```

### Cultural Lenses

Available lenses:
- `modern_digital` - Digital native communication
- `modern_academic` - Academic writing
- `trauma_informed` - High-stress contexts
- `neurodivergent_autism` - Autism spectrum
- `neurodivergent_adhd` - ADHD patterns
- `indigenous_oral` - Oral storytelling

Switch lens via MCP:
```python
switch_cultural_lens("trauma_informed")
```

---

## Architecture Overview

```
Query → Emotional Analysis → Hybrid Retrieval → Context Assembly → LLM Generation
         (Rose Glass)         (Vector + Keyword)   (Match scores)   (Claude/Ollama)
              ↓                      ↓                    ↓              ↓
         Ψ, ρ, q, f, τ          Qdrant + ES          Emotional       Response
         Context type           + RRF fusion         matching        + Monitoring
```

---

## Common Use Cases

### 1. Legal Document Analysis

```bash
# Ingest case files
python3 agent/tools/ingest.py ~/legal-cases/ --extensions .txt .pdf .md

# Query with trauma-informed lens
python3 agent/emotional_agent.py "Analyze the emotional impact of this custody case"
```

### 2. Research Assistant

```bash
# Mission mode queries
python3 agent/emotional_agent.py "Research all approaches to rehabilitative justice"
```

### 3. Crisis Support

```bash
# High-activation queries automatically detected
python3 agent/emotional_agent.py "I need urgent help with this situation!"
# → System detects crisis context
# → Prioritizes immediate, empathetic response
# → Monitors for escalation
```

---

## Troubleshooting

### Agent won't start
```bash
# Check dependencies
pip install -r requirements.txt

# Check Python path
export PYTHONPATH="$(pwd)/../src:$(pwd)/../../RoseGlassLE/src:$PYTHONPATH"
```

### Docker services not running
```bash
# Check status
docker compose ps

# Restart
docker compose restart

# View logs
docker compose logs qdrant
```

### Rose Glass imports fail
```bash
# Verify Rose Glass is available
python3 -c "import sys; sys.path.insert(0, '../src'); from core.rose_glass_v2 import RoseGlassV2; print('✓ OK')"
```

### MCP server won't connect
```bash
# Test directly
python3 agent/mcp_server.py test

# Check logs
cat ~/.claude/logs/mcp-emotional-rag.log
```

---

## Next Steps

1. **Read the full guides:**
   - `M4_MAX_BUILD_GUIDE.md` - Complete M4 Max setup
   - `ARCHITECTURE.md` - System architecture
   - `SETUP_GUIDE.md` - Detailed deployment

2. **Customize for your use case:**
   - Add your documents to Qdrant
   - Configure cultural lens
   - Tune retrieval parameters

3. **Integrate with your workflow:**
   - Use MCP tools in Claude Code
   - Build custom tools
   - Add domain-specific calibrations

4. **Deploy to production:**
   - Set up monitoring
   - Configure authentication
   - Scale infrastructure

---

## Support & Resources

- **Documentation**: See all `.md` files in this directory
- **Tests**: `pytest tests/ -v`
- **Examples**: `standalone_demo.py`, `example_implementation.py`
- **GitHub**: https://github.com/GreatPyreneseDad/rose-glass

---

## Key Commands Cheat Sheet

```bash
# Standalone demo (no setup)
python3 standalone_demo.py

# Interactive agent
python3 agent/emotional_agent.py -i

# Single query
python3 agent/emotional_agent.py "your query here"

# Ingest documents
python3 agent/tools/ingest.py /path/to/docs/

# Run tests
pytest tests/ -v

# Test MCP server
python3 agent/mcp_server.py test

# Start infrastructure
docker compose up -d

# Start API server
uvicorn api.main:app --reload
```

---

*"Understanding precedes judgment. Translation enables understanding."*

**Ready to use! 🌹**
