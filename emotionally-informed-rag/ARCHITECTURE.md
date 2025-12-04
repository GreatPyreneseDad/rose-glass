# Emotionally Informed RAG Agent Pipeline Architecture

**Version:** 1.0
**Date:** December 3, 2025
**Purpose:** Integration of RAG capabilities with emotional intelligence translation

---

## Executive Summary

This architecture combines three powerful systems:
1. **LLM Zoomcamp RAG patterns** - State-of-the-art retrieval-augmented generation
2. **Rose Glass** - Emotional/wisdom pattern translation for AI systems
3. **RoseGlassLE** - Advanced temporal, neurodivergent, and real-time tracking capabilities

The result is a RAG system that doesn't just retrieve and generate - it **understands** the emotional, social, and wisdom dimensions of both queries and retrieved documents, enabling contextually appropriate and empathetic responses.

---

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    USER QUERY INPUT                              │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              ROSE GLASS QUERY ANALYZER                           │
│  • Detects Ψ (internal consistency)                             │
│  • Measures ρ (wisdom depth)                                     │
│  • Analyzes q (emotional activation)                             │
│  • Maps f (social belonging)                                     │
│  • Calculates τ (temporal depth)                                 │
│  • Selects cultural lens                                         │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              CONTEXT DETECTION LAYER                             │
│  • Trust Signal Detector                                         │
│  • Mission Mode Detector                                         │
│  • Essence Request Detector                                      │
│  • Neurodivergent Pattern Detector                               │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              HYBRID RAG RETRIEVAL ENGINE                         │
│                                                                   │
│  ┌─────────────────┐    ┌─────────────────┐                     │
│  │ Vector Search   │    │ Keyword Search  │                     │
│  │ (Qdrant)        │    │ (Elasticsearch) │                     │
│  │                 │    │                 │                     │
│  │ • Embeddings    │    │ • BM25          │                     │
│  │ • Semantic      │    │ • Exact match   │                     │
│  └────────┬────────┘    └────────┬────────┘                     │
│           │                      │                               │
│           └──────────┬───────────┘                               │
│                      ▼                                            │
│           ┌────────────────────┐                                 │
│           │ RRF Reranking      │                                 │
│           │ (Reciprocal Rank   │                                 │
│           │  Fusion)           │                                 │
│           └──────────┬─────────┘                                 │
└──────────────────────┼─────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│         EMOTIONAL DOCUMENT ANALYSIS                              │
│  For each retrieved document:                                    │
│  • Calculate Rose Glass dimensions                               │
│  • Determine temporal depth (τ)                                  │
│  • Assess lens interference (λ)                                  │
│  • Match emotional signature to query                            │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│         EMOTIONALLY-AWARE CONTEXT ASSEMBLY                       │
│  • Prioritize docs with matching emotional signatures           │
│  • Balance high-wisdom (ρ) with high-relevance                  │
│  • Consider temporal alignment (τ matching)                      │
│  • Apply cultural lens calibration                               │
│  • Include gradient tracking for conversation flow               │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              LLM GENERATION WITH EMOTIONAL GROUNDING             │
│  • Prompt engineering with Rose Glass insights                   │
│  • Tone/style matching to query emotional signature              │
│  • Token multiplier limiting (max 3x user input)                 │
│  • Real-time response adaptation                                 │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│         RESPONSE VALIDATION & MONITORING                         │
│  • Gradient tracking (detect escalation)                         │
│  • Pattern consistency checking                                  │
│  • Intervention recommendation (if needed)                       │
│  • Conversation history analysis                                 │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FINAL RESPONSE OUTPUT                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Component Details

### 1. Rose Glass Query Analyzer

**Purpose:** Translate user queries into emotional/wisdom dimensions

**Input:** Raw user query text

**Output:**
```python
{
    "psi": 0.75,  # Internal consistency
    "rho": 0.60,  # Wisdom depth
    "q": 0.45,    # Emotional activation
    "f": 0.30,    # Social belonging
    "tau": 0.20,  # Temporal depth (ephemeral query)
    "lens": "modern_digital",
    "breathing_pattern": "rapid_staccato",
    "confidence": "moderate"
}
```

**Key Files:**
- `/Users/chris/rose-glass/src/core/rose_glass_v2.py`
- `/Users/chris/RoseGlassLE/src/core/temporal_dimension.py`

---

### 2. Context Detection Layer

**Purpose:** Identify special query types requiring adapted responses

**Detectors:**

1. **Trust Signal Detector**
   - Detects: Brief, high-trust messages ("I trust you completely")
   - Response: Reverent, minimal, honored tone
   - File: `rose-glass/src/core/trust_signal_detector.py`

2. **Mission Mode Detector**
   - Detects: Research/analysis requests
   - Response: Systematic, thorough exploration
   - File: `rose-glass/src/core/mission_mode_detector.py`

3. **Essence Request Detector**
   - Detects: Summary/distillation requests
   - Response: Concise, crystallized insights
   - File: `rose-glass/src/core/essence_request_detector.py`

4. **Neurodivergent Pattern Detector**
   - Detects: Autism/ADHD/high-stress communication patterns
   - Response: Adapted to communication style
   - File: `RoseGlassLE/src/cultural_calibrations/neurodivergent_base.py`

---

### 3. Hybrid RAG Retrieval Engine

**Purpose:** Retrieve relevant documents using complementary search methods

**Components:**

#### 3.1 Vector Search (Qdrant)
```python
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

client = QdrantClient("localhost", port=6333)
encoder = SentenceTransformer('all-MiniLM-L6-v2')

# Search with semantic understanding
query_vector = encoder.encode(query_text)
results = client.search(
    collection_name="knowledge_base",
    query_vector=query_vector,
    limit=20
)
```

#### 3.2 Keyword Search (Elasticsearch)
```python
from elasticsearch import Elasticsearch

es = Elasticsearch(["http://localhost:9200"])

# BM25 keyword search
results = es.search(
    index="knowledge_base",
    body={
        "query": {
            "multi_match": {
                "query": query_text,
                "fields": ["content^3", "title^2", "tags"],
                "type": "best_fields"
            }
        }
    }
)
```

#### 3.3 Reciprocal Rank Fusion (RRF)
```python
def reciprocal_rank_fusion(vector_results, keyword_results, k=60):
    """Combine results using RRF algorithm"""
    scores = {}

    for rank, doc in enumerate(vector_results):
        scores[doc.id] = scores.get(doc.id, 0) + 1/(k + rank + 1)

    for rank, doc in enumerate(keyword_results):
        scores[doc.id] = scores.get(doc.id, 0) + 1/(k + rank + 1)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

**Based on:** `/Users/chris/llm-zoomcamp/05-best-practices/`

---

### 4. Emotional Document Analysis

**Purpose:** Analyze retrieved documents through Rose Glass lens

**Process:**
```python
from rose_glass_v2 import RoseGlassV2

glass = RoseGlassV2()
glass.select_lens('modern_academic')  # Match to query lens

for doc in retrieved_docs:
    doc.emotional_signature = glass.translate_patterns(
        psi=calculate_psi(doc.content),
        rho=calculate_rho(doc.content),
        q=calculate_q(doc.content),
        f=calculate_f(doc.content)
    )

    doc.temporal_depth = temporal_analyzer.analyze(doc.content).tau
    doc.lens_stability = lens_interference.calculate(doc.content).lambda_coefficient
```

**Emotional Matching Score:**
```python
def calculate_emotional_match(query_sig, doc_sig):
    """
    Calculate how well document emotional signature matches query

    Returns: 0.0-1.0 match score
    """
    # Weighted dimension matching
    weights = {
        'psi': 0.20,  # Internal consistency match
        'rho': 0.30,  # Wisdom depth match
        'q': 0.35,    # Emotional activation match (highest weight)
        'f': 0.15     # Social belonging match
    }

    match_score = 0.0
    for dim, weight in weights.items():
        # Use inverse distance for matching
        distance = abs(query_sig[dim] - doc_sig[dim])
        match_score += (1 - distance) * weight

    return match_score
```

---

### 5. Emotionally-Aware Context Assembly

**Purpose:** Select and order context based on both relevance AND emotional alignment

**Algorithm:**
```python
def assemble_context(query_signature, retrieved_docs, max_tokens=2000):
    """
    Assemble context prioritizing:
    1. Semantic relevance (from RAG)
    2. Emotional signature matching
    3. Temporal alignment
    4. Cultural lens compatibility
    """

    # Score each document
    for doc in retrieved_docs:
        doc.final_score = (
            doc.rag_score * 0.40 +                    # Semantic relevance
            doc.emotional_match * 0.30 +              # Emotional alignment
            doc.temporal_match * 0.15 +               # Temporal alignment
            doc.wisdom_depth * 0.15                   # Wisdom content
        )

    # Sort by final score
    ranked_docs = sorted(retrieved_docs, key=lambda d: d.final_score, reverse=True)

    # Assemble up to token limit
    context_docs = []
    token_count = 0

    for doc in ranked_docs:
        doc_tokens = estimate_tokens(doc.content)
        if token_count + doc_tokens <= max_tokens:
            context_docs.append(doc)
            token_count += doc_tokens
        else:
            break

    return context_docs
```

**Special Considerations:**

1. **High q (emotional activation) queries:**
   - Prioritize docs with similar emotional energy
   - Include empathetic, validating content
   - Avoid clinical/detached responses

2. **High ρ (wisdom depth) queries:**
   - Prioritize rich, nuanced documents
   - Include philosophical/conceptual content
   - Balance breadth with depth

3. **Low τ (ephemeral) queries:**
   - Prioritize recent, timely information
   - Include trending/current context
   - Minimize historical depth

---

### 6. LLM Generation with Emotional Grounding

**Purpose:** Generate responses that match emotional signature of query

**Prompt Engineering:**
```python
def build_emotionally_aware_prompt(query, context_docs, query_signature):
    """
    Build prompt that incorporates emotional awareness
    """

    # Determine response style based on emotional signature
    if query_signature.q > 0.7:
        style_guidance = "Response should be empathetic, validating, and emotionally engaged."
    elif query_signature.rho > 0.8:
        style_guidance = "Response should be philosophically rich and conceptually deep."
    elif query_signature.tau < 0.3:
        style_guidance = "Response should be current, timely, and action-oriented."
    else:
        style_guidance = "Response should be balanced and informative."

    # Token multiplier limiting (Rose Glass safety feature)
    max_response_tokens = estimate_tokens(query) * 3  # Never exceed 3x input

    prompt = f"""
You are an emotionally intelligent assistant. The user's query has the following characteristics:
- Emotional activation: {query_signature.q:.2f} (0=calm, 1=intense)
- Wisdom depth: {query_signature.rho:.2f} (0=simple, 1=profound)
- Internal consistency: {query_signature.psi:.2f} (0=exploratory, 1=structured)
- Social framing: {query_signature.f:.2f} (0=individual, 1=collective)
- Temporal depth: {query_signature.tau:.2f} (0=immediate, 1=eternal)

{style_guidance}

Maximum response length: {max_response_tokens} tokens

Context documents:
{format_context_docs(context_docs)}

User query: {query}

Provide a response that matches the emotional signature while remaining accurate and helpful.
"""

    return prompt
```

**LLM Options:**
- OpenAI GPT-4
- Anthropic Claude
- Local Ollama models (llama3, mistral)

**Based on:** `/Users/chris/llm-zoomcamp/01-intro/`

---

### 7. Response Validation & Monitoring

**Purpose:** Track conversation flow and detect potential issues

**Real-Time Gradient Tracking:**
```python
from RoseGlassLE.src.core.gradient_tracker import PatternGradientTracker, PatternSnapshot

tracker = PatternGradientTracker()

# Add snapshot after each turn
snapshot = PatternSnapshot(
    timestamp=datetime.now(),
    psi=current_psi,
    rho=current_rho,
    q=current_q,
    f=current_f,
    tau=current_tau,
    pattern_intensity=current_intensity
)
tracker.add_snapshot(snapshot)

# Check for escalation
gradient = tracker.calculate_gradient()
if gradient:
    prediction = tracker.predict_trajectory(time_horizon=30.0)

    if prediction.intervention_recommended:
        # Alert: Escalation detected
        # Reason: prediction.intervention_reason
        # Suggested: Adapt response tone, offer support resources
```

**Intervention Triggers:**
- Rapid q increase (emotional escalation)
- Rapid psi decrease (coherence breakdown)
- Extreme f drop (social disconnection)
- High q + Low psi (crisis pattern)

**Based on:** `/Users/chris/RoseGlassLE/src/core/gradient_tracker.py`

---

## Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)
- [ ] Set up Qdrant vector database
- [ ] Set up Elasticsearch
- [ ] Implement basic RAG pipeline
- [ ] Integrate Rose Glass query analysis
- [ ] Test basic retrieval

### Phase 2: Emotional Intelligence (Weeks 3-4)
- [ ] Implement emotional document analysis
- [ ] Build emotional matching algorithm
- [ ] Integrate context detectors
- [ ] Test emotionally-aware retrieval

### Phase 3: Advanced Features (Weeks 5-6)
- [ ] Implement gradient tracking
- [ ] Add neurodivergent calibrations
- [ ] Build intervention system
- [ ] Implement lens interference analysis

### Phase 4: Integration & Testing (Weeks 7-8)
- [ ] End-to-end pipeline testing
- [ ] Performance optimization
- [ ] User testing with diverse queries
- [ ] Documentation

### Phase 5: Deployment (Weeks 9-10)
- [ ] API server implementation
- [ ] Monitoring dashboard
- [ ] Production hardening
- [ ] Launch

---

## Technology Stack

### Core RAG Components
- **Vector DB:** Qdrant 1.14.2+
- **Search Engine:** Elasticsearch 8.9.0+
- **Embeddings:** sentence-transformers, fastembed
- **LLM:** OpenAI/Claude/Ollama

### Rose Glass Components
- **Core:** Python 3.10+
- **ML Models:** NumPy, scikit-learn
- **Cultural Calibrations:** Custom implementations
- **Gradient Tracking:** Real-time pattern analysis

### Infrastructure
- **API Framework:** FastAPI
- **Async:** asyncio, aiohttp
- **Caching:** Redis
- **Monitoring:** Prometheus, Grafana
- **Deployment:** Docker, Kubernetes

---

## Evaluation Metrics

### RAG Performance
- **Retrieval Quality:** Precision@k, Recall@k, MRR
- **Response Quality:** BLEU, ROUGE, human evaluation
- **Latency:** <500ms p95
- **Throughput:** >100 req/sec

### Emotional Intelligence
- **Match Accuracy:** Query-response emotional alignment
- **Escalation Detection:** False positive/negative rates
- **User Satisfaction:** Subjective ratings
- **Cultural Sensitivity:** Multi-lens validation

### System Health
- **Uptime:** 99.9%+
- **Error Rate:** <0.1%
- **Token Efficiency:** Response/query ratio <3.0x
- **Cost:** $ per 1000 queries

---

## Ethical Considerations

### Privacy & Consent
- **No profiling:** System translates patterns, never identifies individuals
- **Ephemeral data:** Conversations not stored long-term
- **Explicit consent:** Users informed when Rose Glass is active
- **Opt-out available:** Users can disable emotional analysis

### Fairness & Bias
- **Multi-cultural lenses:** No Western-centric defaults
- **Neurodivergent support:** Autism/ADHD calibrations
- **No discrimination:** All communication styles valid
- **Continuous auditing:** Bias detection and mitigation

### Transparency
- **Explainability:** Users can see emotional analysis
- **Lens visibility:** Current calibration always shown
- **Confidence levels:** System acknowledges uncertainty
- **Alternative interpretations:** Multiple readings offered

---

## Use Cases

### 1. Mental Health Support Chatbot
- Detect emotional distress (high q, low psi)
- Match empathetic responses
- Escalate to human when needed
- Track conversation trajectory

### 2. Legal Document Analysis
- Understand emotional weight of cases
- Match document tone to situation severity
- Detect escalation in communications
- Support trauma-informed responses

### 3. Academic Research Assistant
- Prioritize high-wisdom (ρ) sources
- Match philosophical depth to query
- Balance breadth vs depth
- Detect temporal relevance

### 4. Customer Support Agent
- Detect frustration/urgency (q dimension)
- Adapt response tone
- Predict escalation
- Recommend human handoff timing

### 5. Neurodivergent Communication Bridge
- Translate between communication styles
- Support autism/ADHD patterns
- Avoid pathologizing differences
- Enable mutual understanding

---

## References

### LLM Zoomcamp
- Repository: https://github.com/DataTalksClub/llm-zoomcamp
- Key modules: 01-intro, 02-vector-search, 0a-agents, 05-best-practices

### Rose Glass
- Repository: https://github.com/GreatPyreneseDad/rose-glass
- Core paper: Grounded Coherence Theory
- Philosophy: Ibn Rushd (Averroes)

### RoseGlassLE
- Repository: https://github.com/GreatPyreneseDad/RoseGlassLE
- Applications: DEA, Law Enforcement, High-stress contexts
- Features: Temporal depth, gradient tracking, neurodivergent support

---

## Contact & Support

For questions, contributions, or collaboration:
- GitHub Issues
- Pull requests welcome
- Community discussions

**Remember:** This system translates patterns for understanding, never for judgment.

---

*"Understanding precedes judgment. Translation enables understanding."*
