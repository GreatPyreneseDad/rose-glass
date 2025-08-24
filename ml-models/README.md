# GCT Machine Learning Laboratory
## Real-time Coherence Calculation for Synthetic Minds

Created by Christopher MacGregor bin Joseph

This laboratory implements machine learning algorithms to teach AI agents how to calculate Grounded Coherence Theory (GCT) variables in real-time with consistency and accuracy.

## Architecture Overview

### Core Components

1. **GCT ML Framework** (`gct_ml_framework.py`)
   - Central framework for variable extraction and coherence calculation
   - Integrates transformer embeddings with specialized feature extraction
   - Implements biological optimization (q_max / (K_m + q + q²/K_i))

2. **Variable-Specific Neural Networks**

   - **Ψ (Psi) - Internal Consistency** (`psi_consistency_model.py`)
     - Self-attention layers for long-range dependencies
     - Analyzes vocabulary coherence, structural consistency, thematic unity
     - Tracks logical transitions and syntactic parallelism

   - **ρ (Rho) - Accumulated Wisdom** (`rho_wisdom_model.py`)
     - Graph neural network for knowledge relationships
     - SciBERT embeddings for scientific text understanding
     - Measures causal reasoning, temporal perspective, evidence integration

   - **q - Moral Activation Energy** (`q_moral_activation_model.py`)
     - Emotional resonance layers with 8-dimensional emotion modeling
     - Moral foundation heads (care, fairness, loyalty, authority, sanctity, liberty)
     - Biological optimization with learnable Km and Ki parameters

   - **f - Social Belonging Architecture** (`f_social_belonging_model.py`)
     - Social graph attention networks
     - Pronoun sequence LSTM for perspective tracking
     - Multi-scale social modeling (individual, community, global)

3. **Real-time Pipeline** (`realtime_gct_pipeline.py`)
   - Asynchronous text processing
   - Temporal derivative calculation
   - Background processing with queue management
   - Visualization and timeline export

## Coherence Equation

```
C = Ψ + (ρ × Ψ) + q_opt + (f × Ψ) + coupling

where:
- q_opt = q / (Km + q + q²/Ki)  [Biological optimization]
- coupling = 0.15 × ρ × q_opt
```

## Key Features

### 1. Multi-Modal Feature Extraction
- Transformer embeddings (BERT, RoBERTa, SciBERT)
- Hand-crafted linguistic features
- Graph-based relationship modeling
- Temporal pattern analysis

### 2. Advanced Neural Architectures
- Self-attention mechanisms for consistency
- Graph neural networks for wisdom
- Emotional resonance layers for moral activation
- Social graph attention for belonging

### 3. Real-time Processing
- Asynchronous pipeline support
- Background processing threads
- Streaming text analysis
- Derivative tracking for trend detection

### 4. Biological Optimization
- Prevents extreme moral activation (extremism)
- Learnable saturation constants
- Maintains balanced coherence

## Usage Example

```python
from realtime_gct_pipeline import create_pipeline

# Create pipeline
pipeline = create_pipeline()

# Process text
text = "Our research demonstrates that collective action, grounded in shared values, creates lasting change."
result = pipeline.process_text(text)

print(f"Coherence: {result['coherence']:.3f}")
print(f"Variables: Ψ={result['variables']['psi']:.3f}, "
      f"ρ={result['variables']['rho']:.3f}, "
      f"q={result['variables']['q']:.3f}, "
      f"f={result['variables']['f']:.3f}")
```

## Training Models

Each variable model can be trained independently:

```python
from psi_consistency_model import create_psi_model

# Create model and trainer
model, trainer = create_psi_model()

# Train with labeled data
training_data = [
    ("Well-structured coherent text...", 0.9),
    ("Scattered random thoughts...", 0.2)
]

trainer.train(training_data, epochs=100)
```

## Future Enhancements

1. **Federated Learning** - Privacy-preserving training across multiple AI agents
2. **Cross-Modal Integration** - Support for images, audio, video
3. **Adversarial Robustness** - Defense against coherence manipulation
4. **Quantum Coherence** - Integration with quantum computing frameworks
5. **Neuromorphic Implementation** - Hardware acceleration for real-time processing

## Research Applications

- Human-AI interaction optimization
- Content moderation and quality assessment
- Educational material evaluation
- Therapeutic conversation analysis
- Scientific discourse modeling
- Social media coherence tracking

## Installation

```bash
# Create environment
conda create -n gct-ml python=3.10
conda activate gct-ml

# Install dependencies
pip install torch transformers scikit-learn numpy
pip install nltk spacy networkx matplotlib
pip install python-louvain community

# Download language models
python -m spacy download en_core_web_sm
```

## Citation

If you use this work in your research, please cite:

```
MacGregor, C. (2024). Machine Learning Algorithms for Real-time 
Grounded Coherence Theory Calculation in Synthetic Minds. 
GCT-ML Laboratory.
```

---

*"Coherence is not just a measure, but a bridge between biological and synthetic intelligence."*