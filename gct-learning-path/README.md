# GCT Adaptive Learning Path Generator

Leverage Grounded Coherence Theory (GCT) to create personalized e-learning sequences that flow naturally in topic, difficulty, and cognitive load—making every learner's journey intuitive, engaging, and optimally structured.

## 🎯 Overview

Traditional learning platforms present content in rigid blocks, causing abrupt jumps in complexity that discourage learners. This module uses GCT's coherence metrics to:

- **Quantify transition quality** between learning modules
- **Optimize learning paths** for individual learners
- **Maintain flow states** throughout the learning journey
- **Adapt to real-time performance** and feedback

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/GreatPyreneseDad/GCT.git
cd GCT/gct-learning-path

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the API
python src/api.py
```

The API will be available at `http://localhost:8000`

## 📖 How It Works

### Coherence Formula for Learning

```
C(m1→m2) = ψ(topic_similarity) + ρ(knowledge_building) + q_opt(difficulty_progression) 
            + f(flow_maintenance) + α(personalization)
```

Where:
- **ψ (psi)**: Thematic coherence between topics
- **ρ (rho)**: Knowledge building with redundancy penalty
- **q_opt**: Optimal difficulty progression (Zone of Proximal Development)
- **f (flow)**: Maintaining learner engagement and momentum
- **α (alpha)**: Personalization based on learner profile

## 🔧 Core Components

### 1. Course Repository (`repository.py`)
- Manages learning module storage and retrieval
- Supports SQLite, JSON, and API backends
- Tracks engagement metrics and relationships

### 2. Metadata Extractor (`metadata_extractor.py`)
- Embeds topics into semantic space using sentence transformers
- Analyzes cognitive complexity using Bloom's taxonomy
- Identifies prerequisite chains and learning patterns

### 3. GCT Engine (`gct_engine.py`)
- Calculates coherence scores for module transitions
- Evaluates each component (ψ, ρ, q_opt, f, α)
- Creates transition graphs for pathfinding

### 4. Path Optimizer (`path_optimizer.py`)
- Uses A* search with coherence heuristics
- Falls back to beam search for complex paths
- Generates alternative paths with different optimization focuses

### 5. API (`api.py`)
- RESTful endpoints for path generation
- Learner profile management
- Real-time feedback processing

## 📊 API Endpoints

### Generate Learning Path
```bash
POST /generate-path
{
  "learner_id": "u123",
  "start_module": "intro_python",
  "target_module": "advanced_ml",
  "max_steps": 10
}

# Response
{
  "path": ["intro_python", "data_types", "functions", ...],
  "coherence_score": 0.85,
  "estimated_duration": 420,
  "difficulty_progression": [0.2, 0.25, 0.35, ...],
  "explanation": {
    "overview": "This path contains 8 modules...",
    "key_transitions": [...],
    "strengths": [...]
  }
}
```

### Submit Feedback
```bash
POST /feedback
{
  "learner_id": "u123",
  "module_id": "functions",
  "completion_time": 45,
  "difficulty_rating": 3.5,
  "engagement_rating": 4.0,
  "quiz_score": 0.85
}
```

### Get Recommendations
```bash
GET /recommendations/{learner_id}

# Response
{
  "recommendations": [
    {"module_id": "lists", "coherence": 0.92},
    {"module_id": "loops", "coherence": 0.87}
  ],
  "based_on": "functions"
}
```

## 🧪 Example Usage

### Python Client Example
```python
import requests

# Create learner profile
profile = {
    "learner_id": "alice",
    "skill_level": 0.3,
    "learning_style": "visual",
    "goals": ["become a data scientist"],
    "interests": ["machine_learning", "visualization"]
}

requests.put("http://localhost:8000/learner/alice", json=profile)

# Generate learning path
path_request = {
    "learner_id": "alice",
    "start_module": "python_basics",
    "target_module": "neural_networks",
    "max_steps": 15
}

response = requests.post("http://localhost:8000/generate-path", json=path_request)
path_data = response.json()

print(f"Your learning journey ({path_data['coherence_score']:.2f} coherence):")
for i, module in enumerate(path_data['path']):
    print(f"{i+1}. {module} (difficulty: {path_data['difficulty_progression'][i]:.2f})")
```

## 📈 Visualization Dashboard

```python
# Run the Streamlit dashboard
streamlit run dashboard.py
```

Features:
- Learning path visualization
- Coherence heatmaps
- Progress tracking
- Performance analytics

## 🔬 Advanced Features

### Alternative Path Generation
The system can generate multiple paths optimized for different goals:
- **Topic-focused**: Maximizes thematic coherence
- **Difficulty-focused**: Ensures smooth progression
- **Time-efficient**: Minimizes total duration
- **Exploration**: Includes diverse topics

### Real-time Adaptation
- Adjusts difficulty based on quiz performance
- Modifies pace based on completion times
- Suggests breaks when fatigue is detected
- Recommends review when performance drops

### Team Learning
- Synchronize learning paths for study groups
- Identify complementary skill sets
- Optimize collaborative project assignments

## 🏗️ Architecture

```
┌─────────────────────┐        ┌─────────────────────┐
│  Course Repository  │───▶───▶│ Metadata Extractor  │
│ (videos, quizzes)   │        │ (topics, difficulty)│
└─────────────────────┘        └─────────────────────┘
                                          │
                                          ▼
                               ┌─────────────────────┐
                               │   GCT Coherence     │
                               │      Engine         │
                               └─────────────────────┘
                                          │
                                          ▼
                               ┌─────────────────────┐
                               │  Learning Path      │
                               │    Optimizer        │
                               └─────────────────────┘
                                          │
                                          ▼
                               ┌─────────────────────┐
                               │   FastAPI REST      │
                               │      Service        │
                               └─────────────────────┘
```

## 📚 Module Database Schema

### Modules Table
- `module_id`: Unique identifier
- `title`: Module name
- `topic_tags`: JSON array of topics
- `difficulty`: 0-1 normalized score
- `prerequisites`: JSON array of module IDs
- `cognitive_load`: Estimated mental effort
- `engagement_score`: Historical engagement average

### Relationships Table
- `from_module`: Source module ID
- `to_module`: Target module ID
- `relationship_type`: prerequisite|builds_on|related
- `strength`: Relationship strength (0-1)

## 🎯 Use Cases

### Corporate Training
- Onboard new employees with personalized paths
- Upskill teams based on project requirements
- Track organizational learning metrics

### Academic Institutions
- Adaptive curriculum for diverse learners
- Prerequisite verification and planning
- Performance-based acceleration

### MOOC Platforms
- Reduce dropout rates with coherent progressions
- Personalize massive courses for individuals
- A/B test different learning sequences

## 🔒 Privacy & Ethics

- All learner data is processed locally by default
- Anonymized analytics for path improvement
- Transparent coherence calculations
- No hidden profiling or discrimination

## 🛠️ Configuration

### GCT Weights (`config.yaml`)
```yaml
weights:
  psi: 0.3      # Topic similarity importance
  rho: 0.2      # Knowledge building
  q_opt: 0.2    # Difficulty progression
  flow: 0.2     # Engagement maintenance
  alpha: 0.1    # Personalization

parameters:
  optimal_difficulty_step: 0.1
  max_difficulty_jump: 0.3
  redundancy_threshold: 0.8
```

## 🚧 Future Enhancements

- **Multi-modal learning**: Adapt to video, text, and interactive content
- **Collaborative filtering**: Learn from successful peer paths
- **Quantum superposition**: Explore multiple paths simultaneously
- **AR/VR integration**: Coherence in immersive environments
- **Emotional intelligence**: Adapt to learner mood and stress

## 📄 License

MIT License - See LICENSE file

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

*"The mind is not a vessel to be filled, but a fire to be kindled." - Plutarch*

*Learning paths should kindle that fire, not extinguish it with incoherent jumps.*