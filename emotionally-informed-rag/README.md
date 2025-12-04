# Emotionally Informed RAG Agent Pipeline

An advanced Retrieval-Augmented Generation (RAG) system that integrates **emotional intelligence** through the Rose Glass framework, enabling AI to understand and respond to the emotional, social, and wisdom dimensions of both queries and retrieved documents.

## Overview

This system combines three powerful technologies:

1. **LLM Zoomcamp RAG Patterns** - State-of-the-art retrieval techniques
   - Vector search with Qdrant
   - Keyword search with Elasticsearch
   - Hybrid search with RRF (Reciprocal Rank Fusion)
   - Document reranking

2. **Rose Glass** - Emotional pattern translation framework
   - 4 core dimensions: Ψ (consistency), ρ (wisdom), q (emotion), f (social)
   - Cultural calibrations
   - Context detection (trust signals, mission mode, etc.)

3. **RoseGlassLE** - Advanced temporal and real-time tracking
   - τ (Tau) - Temporal depth dimension
   - λ (Lambda) - Lens interference analysis
   - Gradient tracking for escalation detection
   - Neurodivergent calibrations

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for complete system design and implementation details.

## Quick Start

### Prerequisites

1. **Clone Required Repositories**

```bash
# This repo
cd ~/emotionally-informed-rag

# LLM Zoomcamp (already cloned)
# ~/llm-zoomcamp

# Rose Glass (already cloned)
# ~/rose-glass

# RoseGlassLE (already cloned)
# ~/RoseGlassLE
```

2. **Install Dependencies**

```bash
pip install -r requirements.txt
```

3. **Start Required Services**

```bash
# Qdrant (Vector Database)
docker run -p 6333:6333 -p 6334:6334 \
   -v "$(pwd)/qdrant_storage:/qdrant/storage:z" \
   qdrant/qdrant

# Elasticsearch
docker run -it --rm --name elasticsearch \
   -p 9200:9200 -p 9300:9300 \
   -e "discovery.type=single-node" \
   -e "xpack.security.enabled=false" \
   docker.elastic.co/elasticsearch/elasticsearch:8.9.0
```

4. **Run Example**

```bash
python example_implementation.py
```

## Example Usage

```python
from emotionally_informed_rag import EmotionallyInformedRAG

# Initialize the system
rag = EmotionallyInformedRAG(cultural_lens="modern_digital")

# Query with high emotional activation
response = rag.query(
    "I'm really struggling to understand how AI can help with my legal case. "
    "This is urgent and I'm very worried about the outcome."
)

# The system will:
# 1. Detect high emotional activation (q=0.8)
# 2. Retrieve relevant documents
# 3. Prioritize emotionally aligned content
# 4. Generate empathetic, validating response
# 5. Track conversation gradient for escalation
```

## Key Features

### 🎯 Emotional Query Analysis
- Detects emotional activation, wisdom depth, social framing
- Identifies context type (trust signal, mission mode, etc.)
- Calculates temporal depth

### 📚 Emotionally-Aware Retrieval
- Hybrid search (vector + keyword)
- Emotional signature matching
- Temporal alignment
- Wisdom depth prioritization

### 💬 Adaptive Response Generation
- Tone matching to query emotional signature
- Cultural lens calibration
- Token multiplier limiting (max 3x input)
- Real-time response adaptation

### 📊 Conversation Monitoring
- Gradient tracking (detect escalation)
- Pattern consistency checking
- Intervention recommendations
- Crisis detection

### 🧠 Neurodivergent Support
- Autism spectrum calibrations
- ADHD communication patterns
- High-stress/trauma contexts
- Non-pathologizing approach

## Use Cases

1. **Mental Health Support Chatbot**
   - Detect emotional distress
   - Match empathetic responses
   - Escalate when needed

2. **Legal Document Analysis** *(Your use case!)*
   - Understand emotional weight of cases
   - Trauma-informed responses
   - Detect escalation in communications

3. **Academic Research Assistant**
   - Prioritize high-wisdom sources
   - Match philosophical depth
   - Balance breadth vs depth

4. **Customer Support**
   - Detect frustration/urgency
   - Adapt tone dynamically
   - Predict escalation

## Configuration

### Cultural Lenses

Available lenses (from Rose Glass):
- `modern_digital` - Digital native communication
- `modern_academic` - Academic writing
- `medieval_islamic` - Islamic philosophy tradition
- `indigenous_oral` - Oral tradition patterns
- `buddhist_contemplative` - Contemplative teachings

```python
rag = EmotionallyInformedRAG(cultural_lens="modern_academic")
```

### Neurodivergent Calibrations

```python
from cultural_calibrations.neurodivergent_base import AutismSpectrumCalibration

# Apply autism spectrum calibration
calibration = AutismSpectrumCalibration()
rag.rose_glass.apply_calibration(calibration)
```

## Project Structure

```
emotionally-informed-rag/
├── ARCHITECTURE.md           # Complete architecture documentation
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── example_implementation.py # Example code
├── src/                      # Source code (to be built)
│   ├── core/
│   │   ├── rag_engine.py
│   │   ├── emotional_analyzer.py
│   │   └── response_generator.py
│   ├── integrations/
│   │   ├── qdrant_client.py
│   │   └── elasticsearch_client.py
│   └── utils/
└── tests/                    # Test suite
```

## Development Roadmap

- [x] Architecture design
- [x] Example implementation
- [ ] Production RAG engine
- [ ] API server
- [ ] Web dashboard
- [ ] Comprehensive testing
- [ ] Performance optimization
- [ ] Documentation
- [ ] Deployment guide

## Ethical Considerations

### What We Don't Do
❌ Profile or identify individuals
❌ Judge quality of human expression
❌ Impose cultural norms
❌ Store conversations without consent

### What We Do
✅ Translate patterns for understanding
✅ Respect cultural diversity
✅ Support neurodivergent communication
✅ Enable mutual understanding
✅ Maintain transparency

## Performance Targets

- **Latency:** <500ms p95
- **Throughput:** >100 req/sec
- **Accuracy:** >85% emotional match
- **Escalation Detection:** >90% true positive rate

## References

### Source Repositories
- **LLM Zoomcamp:** `/Users/chris/llm-zoomcamp`
- **Rose Glass:** `/Users/chris/rose-glass`
- **RoseGlassLE:** `/Users/chris/RoseGlassLE`

### Papers & Resources
- Grounded Coherence Theory (GCT)
- Ibn Rushd - The Incoherence of the Incoherence
- Reciprocal Rank Fusion Algorithm
- Hybrid Search Techniques

## Contributing

Contributions welcome! Areas of interest:
- Additional cultural calibrations
- Performance optimizations
- Test coverage
- Documentation improvements
- Real-world use case validation

## License

[To be determined - coordinate with Rose Glass licensing]

## Contact

For questions or collaboration opportunities, please open an issue.

---

**"Understanding precedes judgment. Translation enables understanding."**

*Powered by Rose Glass + RAG*
