# Using Claude Desktop with Legal RAG

Your Emotionally-Informed Legal RAG system is now configured to work with Claude Desktop!

## What's Set Up

✅ **2,106 legal documents** ingested from your Legal_Cases folder
✅ **Qdrant vector database** running with emotional signatures
✅ **MCP Server** configured to expose retrieval tools to Claude Desktop
✅ **Claude Desktop config** installed at `~/Library/Application Support/Claude/claude_desktop_config.json`

## How It Works

Instead of paying for API calls, **Claude Desktop acts as the LLM** and uses your local MCP server to retrieve legal documents. You get:

- **Free LLM inference** (uses your Claude Desktop subscription)
- **Private document retrieval** (your legal docs never leave your machine)
- **Emotional awareness** (queries are analyzed for trauma-informed responses)
- **Full conversational AI** (Claude Desktop's native interface)

## Usage

### 1. Restart Claude Desktop

If Claude Desktop is running, quit and restart it to load the MCP configuration.

### 2. Available Tools

Claude Desktop now has access to these tools:

#### `search_legal_documents`
Search through your legal cases with emotional awareness.

**Example prompts:**
- "Search for custody issues in the Maslowsky case"
- "Find documents about discovery in Fardell case"
- "What evidence exists about harassment?"

#### `analyze_emotional_context`
Analyze text for emotional activation and trauma-informed needs.

**Example prompts:**
- "Analyze the emotional context of this email: [paste email]"
- "Is this query showing signs of crisis?"

#### `get_collection_info`
Get statistics about your document collection.

**Example prompts:**
- "How many legal documents are available?"
- "Show me collection statistics"

### 3. Sample Conversation

```
You: Search for key issues in the Maslowsky case

Claude Desktop: [Uses search_legal_documents tool]
I found several relevant documents. The key issues include:
1. Custody disputes...
2. Discovery requests...
[Full response with document excerpts]

You: That sounds urgent - analyze the emotional context

Claude Desktop: [Uses analyze_emotional_context tool]
Emotional Analysis:
- Activation: 0.65 (Medium-High)
- Context: Standard
- Recommendation: Use supportive, clear language
```

### 4. Checking Tools Are Available

In Claude Desktop, you can ask:
```
"What MCP tools do you have available?"
```

Claude should list the legal-rag tools. If not, check:

1. **Config file exists:** `cat ~/Library/Application\ Support/Claude/claude_desktop_config.json`
2. **Docker is running:** `docker ps | grep qdrant`
3. **Python path is correct:** Test with `source venv/bin/activate && python3 agent/mcp_server_simple.py`

## Architecture

```
┌─────────────────┐
│  Claude Desktop │ ← You interact here (FREE)
│   (LLM Engine)  │
└────────┬────────┘
         │ Uses MCP Protocol
         ↓
┌─────────────────┐
│   MCP Server    │ ← Runs locally in your venv
│  (Python 3.13)  │
└────────┬────────┘
         │ Queries
         ↓
┌─────────────────┐
│     Qdrant      │ ← Your 2,106 legal documents
│  Vector Store   │    with emotional signatures
└─────────────────┘
```

## Benefits Over API-based Agent

| Feature | MCP + Claude Desktop | API Agent |
|---------|---------------------|-----------|
| **LLM Cost** | Free (Desktop sub) | $$ API calls |
| **Privacy** | 100% local retrieval | Docs sent to API |
| **Interface** | Native Desktop UI | Custom CLI |
| **Conversation** | Full context memory | Limited history |
| **Emotional Analysis** | ✅ Built-in | ✅ Built-in |

## Troubleshooting

### Tools Not Appearing

```bash
# Check config
cat ~/Library/Application\ Support/Claude/claude_desktop_config.json

# Test MCP server directly
source venv/bin/activate
python3 agent/mcp_server_simple.py
# Should start without errors
```

### "Connection Failed" Error

```bash
# Ensure Docker is running
docker ps | grep qdrant

# If not running, start it
cd /Users/chris/rose-glass/emotionally-informed-rag
docker compose up -d qdrant elasticsearch
```

### Documents Not Found

```bash
# Check collection stats
curl http://localhost:6333/collections/knowledge_base
# Should show 2106 points
```

## Adding More Documents

To ingest additional legal documents:

```bash
cd /Users/chris/rose-glass/emotionally-informed-rag
source venv/bin/activate

# Ingest a directory
python3 agent/tools/ingest_simple.py /path/to/new/docs --extensions .txt .md .pdf

# Ingest a single file
python3 agent/tools/ingest_simple.py /path/to/document.pdf
```

Then just continue using Claude Desktop - it will automatically see the new documents.

## What You Can Ask

### Legal Research
- "Summarize the Maslowsky custody case"
- "Find all references to protective orders"
- "What discovery was requested in Fardell?"

### Document Finding
- "Search for hearing transcripts from December"
- "Find documents mentioning MacGregor"
- "Show me motion filings related to sanctions"

### Emotional Analysis
- "This client email seems distressed - analyze it"
- "Is this query showing signs of trauma?"
- "What's the emotional context of this situation?"

### Meta Queries
- "How many documents do you have access to?"
- "What cases are in the collection?"
- "Show me collection statistics"

## Privacy & Security

✅ **All document retrieval happens locally** - your legal docs never leave your machine
✅ **Claude Desktop sends NO document content to Anthropic** - only tool calls and results
✅ **Conversation context stays in Claude Desktop** - full privacy control
✅ **Qdrant runs in Docker on localhost** - no external connections

---

**You're all set! Open Claude Desktop and start asking about your legal cases.** 🌹

The system combines Claude's intelligence with your private legal document collection, using emotional awareness to provide trauma-informed responses.
