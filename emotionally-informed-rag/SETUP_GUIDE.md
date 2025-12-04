# Emotionally Informed RAG - Setup Guide

Complete setup instructions for running the emotionally informed RAG system.

## Prerequisites

### Required Software
- Docker & Docker Compose
- Python 3.10+
- Git

### Required Repositories
Already cloned in your environment:
- `/Users/chris/llm-zoomcamp` - RAG patterns
- `/Users/chris/rose-glass` - Emotional intelligence
- `/Users/chris/RoseGlassLE` - Advanced features
- `/Users/chris/emotionally-informed-rag` - This project

## Quick Start

### 1. Start Services with Docker Compose

```bash
cd /Users/chris/emotionally-informed-rag

# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f
```

This will start:
- **Qdrant** on port 6333 (vector database)
- **Elasticsearch** on port 9200 (keyword search)
- **Redis** on port 6379 (caching)
- **API Server** on port 8000 (REST API)
- **Prometheus** on port 9090 (metrics)
- **Grafana** on port 3000 (dashboards)

### 2. Verify Services

```bash
# Check Qdrant
curl http://localhost:6333/health

# Check Elasticsearch
curl http://localhost:9200/_cluster/health

# Check API
curl http://localhost:8000/health

# Check Prometheus
open http://localhost:9090

# Check Grafana (login: admin/admin)
open http://localhost:3000
```

### 3. Run Standalone Demo (No Docker Required)

```bash
cd /Users/chris/emotionally-informed-rag
python3 standalone_demo.py
```

This demonstrates the core concepts without requiring any external services.

## Development Setup

### 1. Create Virtual Environment

```bash
cd /Users/chris/emotionally-informed-rag
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Environment Variables

```bash
# Create .env file
cat > .env << EOF
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Database Configuration
QDRANT_HOST=localhost
QDRANT_PORT=6333

ELASTICSEARCH_HOST=localhost
ELASTICSEARCH_PORT=9200

REDIS_HOST=localhost
REDIS_PORT=6379

# LLM Configuration (optional)
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here

# Cultural Lens
DEFAULT_CULTURAL_LENS=modern_digital

# Logging
LOG_LEVEL=INFO
EOF
```

### 4. Run API Server Locally

```bash
# With auto-reload for development
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Or using python
python api/main.py
```

### 5. Access API Documentation

Open http://localhost:8000/docs for interactive Swagger UI

## API Usage Examples

### Example 1: Simple Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How can AI help with legal document analysis?",
    "cultural_lens": "modern_digital"
  }'
```

### Example 2: High Emotional Activation Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "I need urgent help! This legal case is very important to me!",
    "cultural_lens": "modern_digital",
    "max_documents": 5
  }'
```

### Example 3: Analyze Text Emotionally

```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "I am deeply concerned about the philosophical implications of artificial intelligence",
    "lens": "modern_academic"
  }'
```

### Example 4: List Available Lenses

```bash
curl http://localhost:8000/lenses
```

## Python Client Example

```python
import requests

# Initialize client
base_url = "http://localhost:8000"

# Query the system
response = requests.post(
    f"{base_url}/query",
    json={
        "query": "What are trauma-informed approaches in legal cases?",
        "cultural_lens": "modern_digital",
        "max_documents": 5
    }
)

result = response.json()

print(f"Emotional Analysis:")
print(f"  q (activation): {result['emotional_analysis']['q']}")
print(f"  ρ (wisdom): {result['emotional_analysis']['rho']}")
print(f"  Context: {result['context_type']}")
print(f"\nResponse: {result['response']}")
```

## Monitoring

### Prometheus Metrics

Access Prometheus at http://localhost:9090

Key metrics:
- `queries_total` - Total queries processed
- `queries_success` - Successful queries
- `queries_error` - Failed queries
- `avg_processing_time_ms` - Average processing time
- `escalations_detected` - Number of escalations detected

### Grafana Dashboards

Access Grafana at http://localhost:3000 (admin/admin)

Create dashboards for:
- Query volume and latency
- Emotional signature distributions
- Escalation events
- Document retrieval performance

## Data Management

### Index Documents to Qdrant

```python
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

client = QdrantClient("localhost", port=6333)
encoder = SentenceTransformer('all-MiniLM-L6-v2')

# Create collection
client.create_collection(
    collection_name="knowledge_base",
    vectors_config={
        "size": 384,  # all-MiniLM-L6-v2 dimension
        "distance": "Cosine"
    }
)

# Index documents
documents = [
    {"id": 1, "text": "Your document text here"},
    # ... more documents
]

for doc in documents:
    vector = encoder.encode(doc["text"])
    client.upsert(
        collection_name="knowledge_base",
        points=[{
            "id": doc["id"],
            "vector": vector.tolist(),
            "payload": {"text": doc["text"]}
        }]
    )
```

### Index Documents to Elasticsearch

```python
from elasticsearch import Elasticsearch

es = Elasticsearch(["http://localhost:9200"])

# Create index
es.indices.create(
    index="knowledge_base",
    body={
        "mappings": {
            "properties": {
                "content": {"type": "text"},
                "title": {"type": "text"},
                "emotional_signature": {"type": "object"}
            }
        }
    }
)

# Index documents
docs = [
    {"title": "Document 1", "content": "Content here"},
    # ... more documents
]

for i, doc in enumerate(docs):
    es.index(index="knowledge_base", id=i, document=doc)
```

## Troubleshooting

### Services Not Starting

```bash
# Check Docker logs
docker-compose logs

# Restart specific service
docker-compose restart api

# Rebuild and restart
docker-compose up -d --build
```

### Port Already in Use

```bash
# Find process using port
lsof -i :8000

# Kill process
kill -9 <PID>
```

### Import Errors

```bash
# Ensure PYTHONPATH includes Rose Glass
export PYTHONPATH="${PYTHONPATH}:/Users/chris/rose-glass/src:/Users/chris/RoseGlassLE/src"
```

### Database Connection Issues

```bash
# Check if services are running
docker-compose ps

# Test connectivity
curl http://localhost:6333/health
curl http://localhost:9200/_cluster/health
redis-cli ping
```

## Testing

### Run Unit Tests

```bash
pytest tests/
```

### Run Integration Tests

```bash
pytest tests/integration/
```

### Test Emotional Analysis

```bash
python -c "
from standalone_demo import SimpleEmotionalAnalyzer
analyzer = SimpleEmotionalAnalyzer()
sig = analyzer.analyze('I am very worried about this!')
print(f'q={sig.q:.2f}, rho={sig.rho:.2f}')
"
```

## Performance Optimization

### Caching Strategy

```python
# Use Redis for caching frequent queries
import redis

r = redis.Redis(host='localhost', port=6379, decode_responses=True)

# Cache query results
r.setex(query_hash, 3600, json.dumps(result))  # 1 hour TTL
```

### Vector Index Optimization

```python
# Use HNSW index for better performance
client.create_collection(
    collection_name="knowledge_base",
    vectors_config={
        "size": 384,
        "distance": "Cosine"
    },
    hnsw_config={
        "m": 16,
        "ef_construct": 100
    }
)
```

## Deployment

### Production Checklist

- [ ] Set secure API keys in `.env`
- [ ] Configure CORS properly
- [ ] Enable authentication/authorization
- [ ] Set up SSL/TLS certificates
- [ ] Configure logging to external service
- [ ] Set up automated backups
- [ ] Configure monitoring alerts
- [ ] Load test the system
- [ ] Document incident response procedures

### Docker Production Deploy

```bash
# Build for production
docker-compose -f docker-compose.prod.yml build

# Deploy
docker-compose -f docker-compose.prod.yml up -d

# Scale API servers
docker-compose -f docker-compose.prod.yml up -d --scale api=3
```

## Next Steps

1. **Implement Production RAG Engine**
   - Complete vector search integration
   - Implement hybrid search with RRF
   - Add document reranking

2. **Integrate Rose Glass ML Models**
   - Replace heuristics with actual models
   - Add all cultural calibrations
   - Implement gradient tracking

3. **Build Frontend Dashboard**
   - Real-time query interface
   - Emotional signature visualization
   - Escalation monitoring

4. **Add Authentication**
   - API key management
   - Rate limiting
   - Usage tracking

5. **Comprehensive Testing**
   - Unit tests for all components
   - Integration tests for pipeline
   - Load testing
   - User acceptance testing

## Support

For issues or questions:
- Check the logs: `docker-compose logs`
- Review ARCHITECTURE.md for design details
- Run the standalone demo to verify concepts
- Open an issue on GitHub

---

**Built with ❤️ using Rose Glass + RAG**
