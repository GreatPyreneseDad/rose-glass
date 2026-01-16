#!/bin/bash
# Re-index Legal Documents with Enhanced Metadata
# Usage: ./reindex_with_metadata.sh

set -e

VENV_PYTHON="/Users/chris/rose-glass/emotionally-informed-rag/venv/bin/python"
INGEST_SCRIPT="/Users/chris/rose-glass/emotionally-informed-rag/agent/tools/ingest_enhanced.py"
LEGAL_DOCS="/Users/chris/Desktop/Legal_Cases"
QDRANT_HOST="localhost"
QDRANT_PORT="6333"
COLLECTION="knowledge_base"

echo "╔══════════════════════════════════════════════════════╗"
echo "║  Rose Glass Enhanced Re-Indexing                     ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# Check Qdrant is running
echo "[1/4] Checking Qdrant status..."
if ! curl -s "http://$QDRANT_HOST:$QDRANT_PORT" > /dev/null; then
    echo "❌ Qdrant not running at $QDRANT_HOST:$QDRANT_PORT"
    echo "Start with: docker start emotional-rag-qdrant"
    exit 1
fi
echo "✓ Qdrant is running"

# Get current document count
CURRENT_COUNT=$(curl -s "http://$QDRANT_HOST:$QDRANT_PORT/collections/$COLLECTION" | python3 -c "import sys, json; print(json.load(sys.stdin)['result']['points_count'])")
echo "✓ Current document count: $CURRENT_COUNT"
echo ""

# Backup collection
echo "[2/4] Creating snapshot backup..."
SNAPSHOT_RESULT=$(curl -s -X POST "http://$QDRANT_HOST:$QDRANT_PORT/collections/$COLLECTION/snapshots" | python3 -c "import sys, json; print(json.load(sys.stdin)['result']['name'])")
echo "✓ Snapshot created: $SNAPSHOT_RESULT"
echo ""

# Optional: Clear collection (comment out to append instead)
read -p "Clear existing collection? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "[3/4] Clearing collection..."
    curl -s -X DELETE "http://$QDRANT_HOST:$QDRANT_PORT/collections/$COLLECTION" > /dev/null
    echo "✓ Collection cleared"
    sleep 2
else
    echo "[3/4] Keeping existing documents (will merge)"
fi
echo ""

# Re-index with enhanced metadata
echo "[4/4] Re-indexing with enhanced metadata..."
echo "Source: $LEGAL_DOCS"
echo "Collection: $COLLECTION"
echo "Exclude: node_modules, venv, __pycache__, .git, .DS_Store"
echo ""

$VENV_PYTHON $INGEST_SCRIPT \
    "$LEGAL_DOCS" \
    --host "$QDRANT_HOST" \
    --port "$QDRANT_PORT" \
    --collection "$COLLECTION" \
    --extensions .txt .md .pdf \
    --exclude node_modules venv __pycache__ .git .DS_Store

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  Re-indexing Complete!                               ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# Get new stats
NEW_COUNT=$(curl -s "http://$QDRANT_HOST:$QDRANT_PORT/collections/$COLLECTION" | python3 -c "import sys, json; print(json.load(sys.stdin)['result']['points_count'])")
echo "Documents indexed: $NEW_COUNT"
echo ""

# Get type breakdown
echo "Checking metadata..."
$VENV_PYTHON -c "
from qdrant_client import QdrantClient
import json

c = QdrantClient(host='$QDRANT_HOST', port=$QDRANT_PORT)
results = c.scroll('$COLLECTION', limit=100, with_payload=True)

type_counts = {}
case_counts = {}

for point in results[0]:
    doc_type = point.payload.get('doc_type', 'unknown')
    case = point.payload.get('case', 'general')
    type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
    case_counts[case] = case_counts.get(case, 0) + 1

print('\nDocument Types (sample):')
for k, v in sorted(type_counts.items()):
    print(f'  {k}: {v}')

print('\nCases (sample):')
for k, v in sorted(case_counts.items()):
    print(f'  {k}: {v}')
"

echo ""
echo "Next steps:"
echo "1. Update ~/.claude/settings.json to use mcp_server_enhanced.py"
echo "2. Restart Claude Desktop"
echo "3. Test: 'Search for custody motions in Fardell case'"
echo ""
echo "Rollback command (if needed):"
echo "curl -X POST \"http://$QDRANT_HOST:$QDRANT_PORT/collections/$COLLECTION/snapshots/$SNAPSHOT_RESULT/recover\""
