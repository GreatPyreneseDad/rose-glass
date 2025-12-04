# Emotionally-Informed RAG Agent Build Guide
## MacBook M4 Max + Claude Code Implementation

*Translation without measurement. Understanding without judgment.*

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      CLAUDE CODE (Terminal)                      │
│  • Natural language → Agent commands                            │
│  • Code generation, debugging, iteration                        │
│  • Direct filesystem access                                     │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              EMOTIONALLY-INFORMED RAG AGENT                      │
│  • Rose Glass emotional analysis                                │
│  • Hybrid retrieval (Qdrant + Elasticsearch)                    │
│  • Gradient tracking for conversation flow                      │
│  • Cultural lens calibration                                    │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LOCAL INFRASTRUCTURE                          │
│  Qdrant (ARM64) │ Elasticsearch │ Redis │ Ollama (optional)    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: M4 Max Environment Setup

### 1.1 System Prerequisites

```bash
# Verify M4 Max architecture
uname -m  # Should return: arm64

# Install Homebrew (if not present)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Essential tools
brew install git python@3.11 docker colima node
```

### 1.2 Docker on Apple Silicon

Colima provides better ARM64 performance than Docker Desktop:

```bash
# Start Colima with M4 Max optimized resources
colima start \
  --cpu 8 \
  --memory 32 \
  --disk 100 \
  --arch aarch64 \
  --vm-type vz \
  --vz-rosetta

# Verify Docker
docker --version
docker run hello-world
```

### 1.3 Claude Code Installation

```bash
# Install Claude Code CLI
npm install -g @anthropic-ai/claude-code

# Authenticate (requires API key)
claude auth login

# Verify installation
claude --version
```

---

## Phase 2: Repository Structure

### 2.1 Project Organization

```bash
# Create workspace
mkdir -p ~/emotional-rag-workspace
cd ~/emotional-rag-workspace

# Clone repositories
git clone https://github.com/GreatPyreneseDad/rose-glass.git
git clone https://github.com/GreatPyreneseDad/RoseGlassLE.git

# Clone LLM Zoomcamp for RAG patterns
git clone https://github.com/DataTalksClub/llm-zoomcamp.git
```

### 2.2 Directory Structure

```
~/emotional-rag-workspace/
├── rose-glass/                 # Core emotional analysis
│   ├── emotionally-informed-rag/  # RAG integration (this project)
│   └── src/core/
│       ├── rose_glass_v2.py
│       ├── trust_signal_detector.py
│       └── mission_mode_detector.py
├── RoseGlassLE/               # Extended features
│   └── src/
│       ├── core/
│       │   ├── temporal_dimension.py
│       │   └── gradient_tracker.py
│       └── cultural_calibrations/
│           └── neurodivergent_base.py
├── llm-zoomcamp/              # RAG patterns reference
└── agent/                     # Production agent implementation
    ├── emotional_agent.py
    ├── mcp_server.py
    ├── tools/
    └── config/
```

---

## Phase 3: Python Environment

### 3.1 Native ARM64 Environment

```bash
cd ~/emotional-rag-workspace/rose-glass/emotionally-informed-rag

# Create virtual environment with system Python
python3.11 -m venv venv --system-site-packages
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip wheel setuptools
```

### 3.2 Install Dependencies

```bash
# Use the existing requirements.txt
pip install -r requirements.txt

# Additional dependencies for M4 Max optimization
pip install \
  torch torchvision torchaudio \
  --extra-index-url https://download.pytorch.org/whl/cpu

# Anthropic SDK for Claude integration
pip install anthropic

# MCP server dependencies
pip install mcp
```

### 3.3 Configure PYTHONPATH

```bash
# Add to ~/.zshrc or ~/.bashrc
export EMOTIONAL_RAG_HOME="$HOME/emotional-rag-workspace"
export PYTHONPATH="$EMOTIONAL_RAG_HOME/rose-glass/src:$EMOTIONAL_RAG_HOME/RoseGlassLE/src:$EMOTIONAL_RAG_HOME/rose-glass/emotionally-informed-rag:$PYTHONPATH"

# Reload shell
source ~/.zshrc
```

---

## Phase 4: Infrastructure Services

### 4.1 Use Existing Docker Compose

The repository already includes `docker-compose.yml`. Update it for M4 Max optimization if needed:

```bash
cd ~/emotional-rag-workspace/rose-glass/emotionally-informed-rag

# Review docker-compose.yml
cat docker-compose.yml

# Start all services
docker compose up -d

# Verify health
curl http://localhost:6333/health      # Qdrant
curl http://localhost:9200/_cluster/health  # Elasticsearch
redis-cli ping                          # Redis
```

---

## Phase 5: Agent Implementation

See `agent/emotional_agent.py` in this repository for the complete implementation.

Key components:
- **EmotionalRAGAgent**: Main agent class
- **analyze_query()**: Rose Glass emotional analysis
- **retrieve_documents()**: Hybrid search with emotional matching
- **generate_response()**: Context-aware response generation via Claude

---

## Phase 6: Claude Code Integration

### 6.1 MCP Server Setup

The MCP server (`agent/mcp_server.py`) exposes tools to Claude Code:

- `analyze_emotional_signature` - Analyze text through Rose Glass
- `query_with_emotion` - RAG query with emotional matching
- `check_escalation_risk` - Monitor conversation gradient
- `get_conversation_history` - Retrieve conversation with signatures
- `switch_cultural_lens` - Change cultural calibration

### 6.2 Claude Code Configuration

Create `.claude/settings.json` in the project root to configure MCP:

```json
{
  "mcpServers": {
    "emotional-rag": {
      "command": "python",
      "args": ["agent/mcp_server.py"],
      "cwd": "${workspaceFolder}",
      "env": {
        "PYTHONPATH": "${workspaceFolder}/../src:${workspaceFolder}/../../RoseGlassLE/src"
      }
    }
  }
}
```

---

## Phase 7: Document Ingestion

### 7.1 Ingest Your Knowledge Base

```bash
cd ~/emotional-rag-workspace/rose-glass/emotionally-informed-rag

# Ingest documents with emotional signatures
python agent/tools/ingest.py ~/Documents/legal-cases/
python agent/tools/ingest.py ~/Documents/research/

# Verify ingestion
curl http://localhost:6333/collections/knowledge_base
```

### 7.2 Ingest to Elasticsearch

```bash
# Also index in Elasticsearch for keyword search
python agent/tools/ingest_elasticsearch.py ~/Documents/legal-cases/
```

---

## Phase 8: Running the Agent

### 8.1 Quick Test

```bash
# Activate environment
cd ~/emotional-rag-workspace/rose-glass/emotionally-informed-rag
source venv/bin/activate

# Run standalone demo (no external services required)
python standalone_demo.py
```

### 8.2 Full Agent (with infrastructure)

```bash
# Ensure services are running
docker compose ps

# Run agent interactively
python agent/emotional_agent.py --interactive
```

### 8.3 Via Claude Code

```bash
# Start Claude Code in project directory
cd ~/emotional-rag-workspace/rose-glass/emotionally-informed-rag
claude
```

Then in Claude Code:
```
> Use the emotional-rag tools to analyze: "I'm very worried about my legal case"
> Query the knowledge base about contract law with emotional awareness
> Check escalation risk in this conversation
```

---

## Phase 9: Testing & Validation

### 9.1 Run Test Suite

```bash
cd ~/emotional-rag-workspace/rose-glass/emotionally-informed-rag

# Run all tests
pytest tests/ -v

# Run specific test module
pytest tests/test_agent.py -v

# With coverage
pytest tests/ --cov=agent --cov-report=html
```

### 9.2 Integration Tests

```bash
# Test full pipeline
pytest tests/integration/ -v

# Test with real infrastructure
pytest tests/integration/ --live-services
```

---

## Performance Optimization for M4 Max

### Metal Acceleration

```python
import torch

# Use Metal Performance Shaders for local models
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Metal acceleration")
else:
    device = torch.device("cpu")
```

### Batch Processing

```python
# Optimize for M4 Max Neural Engine
from sentence_transformers import SentenceTransformer

encoder = SentenceTransformer('all-MiniLM-L6-v2')

# Large batch sizes work well on M4 Max
embeddings = encoder.encode(
    documents,
    batch_size=64,  # M4 Max can handle this easily
    show_progress_bar=True,
    device=device
)
```

### Memory Configuration

```bash
# Optimize Docker memory for M4 Max (64GB+)
# Edit docker-compose.yml:

elasticsearch:
  environment:
    - ES_JAVA_OPTS=-Xms4g -Xmx8g  # Increase for M4 Max

qdrant:
  # Qdrant automatically uses available memory
```

---

## Troubleshooting

### Common Issues on M4 Max

**Docker containers not starting:**
```bash
# Reset Colima
colima stop
colima delete
colima start --cpu 8 --memory 32 --arch aarch64 --vm-type vz --vz-rosetta
```

**Qdrant connection refused:**
```bash
# Check logs
docker logs emotional-rag-qdrant

# Restart service
docker compose restart qdrant
```

**Rose Glass import errors:**
```bash
# Verify PYTHONPATH
echo $PYTHONPATH | tr ':' '\n'

# Test imports
python -c "import sys; sys.path.insert(0, '$HOME/emotional-rag-workspace/rose-glass/src'); from core.rose_glass_v2 import RoseGlassV2; print('✓ Rose Glass OK')"
```

**PyTorch not using Metal:**
```bash
# Verify MPS availability
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"

# If false, reinstall PyTorch for M4 Max
pip install --upgrade torch torchvision torchaudio
```

---

## Advanced Features

### 1. Local LLM with Ollama

```bash
# Install Ollama
brew install ollama

# Pull models optimized for M4 Max
ollama pull mistral
ollama pull llama3

# Configure agent to use Ollama
# Edit agent/config/agent_config.yaml:
generation:
  provider: "ollama"
  model: "mistral"
  base_url: "http://localhost:11434"
```

### 2. Streaming Responses

```bash
# Run API server with streaming
cd ~/emotional-rag-workspace/rose-glass/emotionally-informed-rag
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Monitoring Dashboard

```bash
# Access Grafana
open http://localhost:3000

# Default credentials: admin/admin
# Import dashboard from monitoring/grafana-dashboards/
```

---

## Production Deployment Checklist

- [ ] Set secure API keys in `.env`
- [ ] Configure CORS properly in `api/main.py`
- [ ] Enable authentication/authorization
- [ ] Set up SSL/TLS certificates
- [ ] Configure logging to external service
- [ ] Set up automated backups for Qdrant/Elasticsearch
- [ ] Configure monitoring alerts in Prometheus
- [ ] Load test the system (use `locust` or `k6`)
- [ ] Document incident response procedures
- [ ] Create runbook for common issues

---

## Next Steps

1. **Add Ollama integration** - Run mistral/llama3 locally
2. **Implement streaming** - FastAPI websocket for real-time responses
3. **Build monitoring dashboard** - Grafana with Rose Glass metrics
4. **Add authentication** - JWT tokens for API access
5. **Create evaluation suite** - Measure emotional match accuracy
6. **Deploy to production** - Kubernetes or cloud hosting

---

## Resources

- **Rose Glass Documentation**: See `../README.md`
- **Architecture Details**: See `ARCHITECTURE.md`
- **Setup Guide**: See `SETUP_GUIDE.md`
- **API Documentation**: http://localhost:8000/docs (when running)
- **LLM Zoomcamp**: https://github.com/DataTalksClub/llm-zoomcamp

---

## Support

For issues or questions:
- Check the logs: `docker compose logs`
- Review troubleshooting section above
- Run standalone demo to verify concepts: `python standalone_demo.py`
- Open an issue on GitHub

---

*"Understanding precedes judgment. Translation enables understanding."*

**Built with ❤️ integrating Rose Glass + RAG on M4 Max**
