"""
GCT Machine Learning Framework
Core framework for training AI agents to calculate GCT coherence variables in real-time

Author: Christopher MacGregor bin Joseph
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, AutoModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GCTVariables:
    """Core GCT variables for coherence calculation"""
    psi: float  # Ψ - Internal Consistency (0-1)
    rho: float  # ρ - Accumulated Wisdom (0-1)
    q: float    # q - Moral Activation Energy (0-1)
    f: float    # f - Social Belonging Architecture (0-1)
    timestamp: datetime
    text: str
    metadata: Optional[Dict] = None


@dataclass 
class GCTCoherence:
    """GCT coherence calculation results"""
    coherence: float
    q_optimized: float
    components: Dict[str, float]
    derivatives: Dict[str, float]
    timestamp: datetime


class GCTFeatureExtractor:
    """Extract features from text for GCT variable calculation"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        
        # Lexicons for feature extraction
        self.lexicons = self._load_lexicons()
        
    def _load_lexicons(self) -> Dict[str, List[str]]:
        """Load specialized lexicons for each GCT variable"""
        return {
            'consistency': {
                'transitions': ['therefore', 'thus', 'hence', 'consequently', 'furthermore',
                               'moreover', 'however', 'nevertheless', 'although', 'whereas'],
                'coherence_markers': ['firstly', 'secondly', 'finally', 'in conclusion',
                                    'to summarize', 'in other words', 'specifically'],
            },
            'wisdom': {
                'knowledge': ['research', 'study', 'analysis', 'evidence', 'data', 'findings',
                            'pattern', 'correlation', 'insight', 'understanding', 'principle',
                            'theory', 'framework', 'methodology', 'empirical', 'hypothesis'],
                'causal': ['because', 'therefore', 'thus', 'hence', 'consequently', 'leads to',
                          'results in', 'causes', 'due to', 'as a result', 'triggers'],
                'temporal': ['historically', 'previously', 'traditionally', 'evolution',
                           'development', 'progression', 'over time', 'gradually'],
            },
            'moral': {
                'values': ['should', 'must', 'ought', 'right', 'wrong', 'good', 'bad',
                          'ethical', 'moral', 'value', 'principle', 'justice', 'fair',
                          'responsibility', 'duty', 'honor', 'integrity', 'honest'],
                'urgency': ['urgent', 'immediately', 'now', 'critical', 'essential',
                           'vital', 'crucial', 'imperative', 'necessary', 'pressing'],
                'action': ['act', 'action', 'do', 'make', 'change', 'fight', 'stand',
                          'defend', 'protect', 'build', 'create', 'improve', 'transform'],
            },
            'social': {
                'collective': ['we', 'us', 'our', 'ours', 'together', 'community', 'team',
                             'group', 'collective', 'shared', 'common', 'mutual', 'unity'],
                'relationship': ['relationship', 'connection', 'bond', 'friendship', 'trust',
                               'support', 'help', 'care', 'empathy', 'compassion'],
                'cultural': ['culture', 'society', 'tradition', 'heritage', 'custom',
                           'norm', 'belief', 'practice', 'ritual', 'identity'],
            }
        }
    
    def extract_features(self, text: str) -> Dict[str, np.ndarray]:
        """Extract multi-modal features from text"""
        # Basic text statistics
        words = text.lower().split()
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        # Transformer embeddings
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, 
                                  padding=True, max_length=512)
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        
        # Linguistic features
        features = {
            'embeddings': embeddings,
            'text_length': len(text),
            'word_count': len(words),
            'sentence_count': len(sentences),
            'avg_sentence_length': len(words) / max(len(sentences), 1),
            'unique_word_ratio': len(set(words)) / max(len(words), 1),
        }
        
        # Lexicon-based features
        for var_name, lexicon_dict in self.lexicons.items():
            for feature_name, word_list in lexicon_dict.items():
                count = sum(1 for word in words if word in word_list)
                features[f'{var_name}_{feature_name}_count'] = count
                features[f'{var_name}_{feature_name}_density'] = count / max(len(words), 1)
        
        # Structural features
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['capitalized_words'] = sum(1 for word in text.split() if word.isupper() and len(word) > 1)
        
        return features


class GCTVariablePredictor(nn.Module):
    """Neural network for predicting individual GCT variables"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [256, 128, 64]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())  # Output in [0, 1]
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze()


class GCTMLEngine:
    """Main engine for ML-based GCT coherence calculation"""
    
    def __init__(self, model_dir: str = "/Users/chris/GCT-ML-Lab/models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        self.feature_extractor = GCTFeatureExtractor()
        self.scalers = {}
        self.models = {}
        
        # GCT parameters for biological optimization
        self.km = 0.2  # Saturation constant
        self.ki = 0.8  # Inhibition constant
        
        # Initialize models for each variable
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize ML models for each GCT variable"""
        # Determine feature dimension
        sample_features = self.feature_extractor.extract_features("Sample text")
        feature_vector = self._prepare_feature_vector(sample_features)
        input_dim = len(feature_vector)
        
        variables = ['psi', 'rho', 'q', 'f']
        for var in variables:
            self.models[var] = GCTVariablePredictor(input_dim)
            self.scalers[var] = StandardScaler()
            
    def _prepare_feature_vector(self, features: Dict) -> np.ndarray:
        """Convert feature dict to numerical vector"""
        # Extract numerical features
        numerical_features = []
        
        # Add embedding features
        if 'embeddings' in features:
            numerical_features.extend(features['embeddings'])
        
        # Add other numerical features
        for key, value in features.items():
            if key != 'embeddings' and isinstance(value, (int, float)):
                numerical_features.append(value)
                
        return np.array(numerical_features)
    
    def train_variable_model(self, variable: str, training_data: List[Tuple[str, float]], 
                           epochs: int = 100, batch_size: int = 32):
        """Train ML model for a specific GCT variable"""
        logger.info(f"Training model for {variable}")
        
        model = self.models[variable]
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # Prepare training data
        X = []
        y = []
        
        for text, target_value in training_data:
            features = self.feature_extractor.extract_features(text)
            feature_vector = self._prepare_feature_vector(features)
            X.append(feature_vector)
            y.append(target_value)
        
        X = np.array(X)
        y = np.array(y)
        
        # Fit scaler
        X_scaled = self.scalers[variable].fit_transform(X)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_scaled)
        y_tensor = torch.FloatTensor(y)
        
        # Training loop
        model.train()
        for epoch in range(epochs):
            # Shuffle data
            perm = torch.randperm(len(X_tensor))
            X_shuffled = X_tensor[perm]
            y_shuffled = y_tensor[perm]
            
            total_loss = 0
            for i in range(0, len(X_tensor), batch_size):
                batch_X = X_shuffled[i:i+batch_size]
                batch_y = y_shuffled[i:i+batch_size]
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}, Loss: {total_loss/len(X_tensor):.4f}")
        
        # Save model
        model_path = self.model_dir / f"{variable}_model.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'scaler_state': self.scalers[variable].__dict__
        }, model_path)
        
    def predict_variables(self, text: str) -> GCTVariables:
        """Predict all GCT variables for given text"""
        features = self.feature_extractor.extract_features(text)
        feature_vector = self._prepare_feature_vector(features)
        
        predictions = {}
        
        for var, model in self.models.items():
            model.eval()
            with torch.no_grad():
                # Scale features
                scaled_features = self.scalers[var].transform([feature_vector])
                tensor_features = torch.FloatTensor(scaled_features)
                
                # Predict
                prediction = model(tensor_features).item()
                predictions[var] = prediction
        
        return GCTVariables(
            psi=predictions['psi'],
            rho=predictions['rho'],
            q=predictions['q'],
            f=predictions['f'],
            timestamp=datetime.now(),
            text=text
        )
    
    def calculate_coherence(self, variables: GCTVariables) -> GCTCoherence:
        """Calculate coherence using GCT formula with biological optimization"""
        # Apply biological optimization to q
        q_opt = variables.q / (self.km + variables.q + (variables.q**2 / self.ki))
        
        # Calculate component contributions
        base = variables.psi
        wisdom_amplification = variables.rho * variables.psi
        social_amplification = variables.f * variables.psi
        coupling = 0.15 * variables.rho * q_opt  # Coupling strength
        
        # Total coherence
        coherence = base + wisdom_amplification + q_opt + social_amplification + coupling
        
        components = {
            'base': base,
            'wisdom_amplification': wisdom_amplification,
            'moral_energy': q_opt,
            'social_amplification': social_amplification,
            'coupling': coupling
        }
        
        return GCTCoherence(
            coherence=coherence,
            q_optimized=q_opt,
            components=components,
            derivatives={},  # TODO: Add temporal derivatives
            timestamp=variables.timestamp
        )
    
    def analyze_text_realtime(self, text: str) -> Dict:
        """Complete real-time analysis pipeline"""
        # Predict variables
        variables = self.predict_variables(text)
        
        # Calculate coherence
        coherence = self.calculate_coherence(variables)
        
        return {
            'variables': {
                'psi': variables.psi,
                'rho': variables.rho,
                'q': variables.q,
                'f': variables.f
            },
            'coherence': coherence.coherence,
            'q_optimized': coherence.q_optimized,
            'components': coherence.components,
            'timestamp': coherence.timestamp.isoformat()
        }
    
    def save_models(self):
        """Save all trained models"""
        for var, model in self.models.items():
            model_path = self.model_dir / f"{var}_model.pth"
            torch.save({
                'model_state_dict': model.state_dict(),
                'scaler_state': self.scalers[var].__dict__
            }, model_path)
            logger.info(f"Saved {var} model to {model_path}")
    
    def load_models(self):
        """Load pre-trained models"""
        for var in ['psi', 'rho', 'q', 'f']:
            model_path = self.model_dir / f"{var}_model.pth"
            if model_path.exists():
                checkpoint = torch.load(model_path)
                self.models[var].load_state_dict(checkpoint['model_state_dict'])
                
                # Restore scaler state
                for key, value in checkpoint['scaler_state'].items():
                    setattr(self.scalers[var], key, value)
                    
                logger.info(f"Loaded {var} model from {model_path}")


def main():
    """Example usage"""
    engine = GCTMLEngine()
    
    # Example training data (would need real labeled data)
    sample_training_data = [
        ("This research demonstrates a clear pattern of evidence supporting our hypothesis.", 0.8),
        ("We must act now to protect our community's future together.", 0.9),
        ("Random words without clear connection or meaning scattered.", 0.2),
    ]
    
    # Train models (would need much more data in practice)
    # engine.train_variable_model('psi', sample_training_data)
    
    # Real-time analysis
    test_text = """
    Our research has uncovered significant patterns in how communities build resilience. 
    We must work together to implement these findings, as they clearly demonstrate 
    the importance of collective action. Therefore, it is our responsibility to act now.
    """
    
    result = engine.analyze_text_realtime(test_text)
    print(f"Real-time GCT Analysis: {json.dumps(result, indent=2)}")


if __name__ == "__main__":
    main()