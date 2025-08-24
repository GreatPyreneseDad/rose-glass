"""
q (Charge) - Moral Activation Energy ML Model
Neural network for calculating the moral drive and action potential in text

This model focuses on:
- Value-driven language and moral imperatives
- Urgency and call-to-action indicators
- Emotional intensity calibration
- Action orientation vs contemplation
- Biological optimization to prevent extremism
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from transformers import AutoModel, AutoTokenizer
from transformers import pipeline
import re
from collections import Counter
from dataclasses import dataclass


@dataclass
class MoralDimension:
    """Represents different dimensions of moral activation"""
    care_harm: float
    fairness_cheating: float
    loyalty_betrayal: float
    authority_subversion: float
    sanctity_degradation: float
    liberty_oppression: float


class EmotionalResonanceLayer(nn.Module):
    """Custom layer for modeling emotional resonance and moral activation"""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_emotions: int = 8):
        super().__init__()
        
        # Emotion-specific processors
        self.emotion_processors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim // 2)
            ) for _ in range(num_emotions)
        ])
        
        # Resonance matrix for emotion interactions
        self.resonance_matrix = nn.Parameter(torch.randn(num_emotions, num_emotions) * 0.1)
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim // 2 * num_emotions, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
    def forward(self, x):
        # Process through emotion-specific pathways
        emotion_outputs = []
        for processor in self.emotion_processors:
            emotion_outputs.append(processor(x))
        
        # Stack emotions
        emotions = torch.stack(emotion_outputs, dim=1)  # (batch, num_emotions, hidden//2)
        
        # Apply resonance interactions
        resonance = torch.matmul(F.softmax(self.resonance_matrix, dim=1), emotions.transpose(1, 2))
        resonated = emotions + resonance.transpose(1, 2) * 0.1
        
        # Flatten and project
        flattened = resonated.view(x.size(0), -1)
        return self.output_projection(flattened)


class MoralActivationFeatureExtractor:
    """Extract features related to moral activation and value-driven language"""
    
    def __init__(self):
        # Moral foundations lexicon
        self.moral_foundations = {
            'care': ['care', 'protect', 'safety', 'harm', 'suffer', 'pain', 'compassion',
                    'nurture', 'empathy', 'kindness', 'cruel', 'hurt', 'damage'],
            'fairness': ['fair', 'equal', 'justice', 'rights', 'deserve', 'earn', 'honest',
                        'cheat', 'fraud', 'bias', 'discriminate', 'prejudice'],
            'loyalty': ['loyal', 'betray', 'traitor', 'solidarity', 'patriot', 'family',
                       'community', 'unity', 'together', 'abandon', 'desert'],
            'authority': ['authority', 'respect', 'tradition', 'obey', 'duty', 'law',
                         'order', 'chaos', 'rebel', 'subvert', 'hierarchy'],
            'sanctity': ['sacred', 'pure', 'clean', 'holy', 'dignity', 'noble', 'virtue',
                        'contaminate', 'disgust', 'degrade', 'pervert', 'vile'],
            'liberty': ['freedom', 'liberty', 'choice', 'autonomy', 'independent', 'oppress',
                       'tyranny', 'coerce', 'control', 'dominate', 'enslaved']
        }
        
        # Action orientation lexicon
        self.action_lexicon = {
            'immediate': ['now', 'immediately', 'urgent', 'today', 'instantly', 'quickly',
                         'rapidly', 'without delay', 'at once', 'right away'],
            'imperative': ['must', 'need', 'have to', 'should', 'ought', 'required',
                          'necessary', 'essential', 'vital', 'crucial', 'imperative'],
            'mobilization': ['act', 'action', 'do', 'make', 'fight', 'stand', 'rise',
                            'defend', 'protect', 'build', 'create', 'change', 'transform'],
            'consequence': ['or else', 'otherwise', 'if not', 'consequences', 'at stake',
                           'risk', 'danger', 'threat', 'peril', 'jeopardy']
        }
        
        # Emotional intensity markers
        self.intensity_markers = {
            'amplifiers': ['very', 'extremely', 'absolutely', 'completely', 'totally',
                          'utterly', 'entirely', 'deeply', 'profoundly', 'incredibly'],
            'exclamations': ['!', '!!', '!!!'],
            'capitals': r'\b[A-Z]{3,}\b',
            'repetition': r'\b(\w+)\s+\1\b'
        }
        
        # Load emotion classifier
        try:
            self.emotion_classifier = pipeline("text-classification", 
                                             model="j-hartmann/emotion-english-distilroberta-base",
                                             return_all_scores=True)
        except:
            self.emotion_classifier = None
            
    def extract_moral_features(self, text: str) -> Dict[str, float]:
        """Extract features related to moral activation"""
        
        features = {}
        words = text.lower().split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # 1. Moral Foundation Scores
        for foundation, keywords in self.moral_foundations.items():
            count = sum(1 for word in words if word in keywords)
            features[f'moral_{foundation}_count'] = count
            features[f'moral_{foundation}_density'] = count / max(len(words), 1)
            
            # Check for moral conflict (opposing values mentioned)
            positive_words = keywords[:len(keywords)//2]
            negative_words = keywords[len(keywords)//2:]
            pos_count = sum(1 for word in words if word in positive_words)
            neg_count = sum(1 for word in words if word in negative_words)
            features[f'moral_{foundation}_conflict'] = min(pos_count, neg_count) / max(len(words), 1)
        
        # 2. Action Orientation Scores
        for category, patterns in self.action_lexicon.items():
            count = sum(1 for pattern in patterns 
                       for _ in re.finditer(r'\b' + pattern + r'\b', text.lower()))
            features[f'action_{category}_count'] = count
            features[f'action_{category}_density'] = count / max(len(sentences), 1)
        
        # 3. Emotional Intensity
        # Exclamation marks
        exclamation_count = text.count('!')
        features['exclamation_intensity'] = min(exclamation_count / max(len(sentences), 1), 1.0)
        
        # Capitalized words (excluding sentence starts)
        caps_words = re.findall(self.intensity_markers['capitals'], text)
        features['capitalization_intensity'] = len(caps_words) / max(len(words), 1)
        
        # Intensifier usage
        intensifier_count = sum(1 for word in words 
                               if word in self.intensity_markers['amplifiers'])
        features['intensifier_density'] = intensifier_count / max(len(words), 1)
        
        # 4. Moral Certainty vs Nuance
        certainty_words = ['definitely', 'certainly', 'absolutely', 'undoubtedly', 'clearly']
        nuance_words = ['perhaps', 'maybe', 'possibly', 'might', 'could', 'sometimes']
        
        certainty_count = sum(1 for word in words if word in certainty_words)
        nuance_count = sum(1 for word in words if word in nuance_words)
        
        features['moral_certainty'] = certainty_count / max(len(words), 1)
        features['moral_nuance'] = nuance_count / max(len(words), 1)
        features['certainty_balance'] = 1 - abs(certainty_count - nuance_count) / max(certainty_count + nuance_count, 1)
        
        # 5. Call to Action Patterns
        cta_patterns = [
            r'we must', r'we need to', r'we should', r'let us', r"let's",
            r'join us', r'stand with', r'fight for', r'defend our', r'protect our'
        ]
        cta_count = sum(1 for pattern in cta_patterns 
                       for _ in re.finditer(pattern, text.lower()))
        features['call_to_action_score'] = cta_count / max(len(sentences), 1)
        
        # 6. Moral Argument Structure
        moral_connectors = ['because', 'therefore', 'since', 'as', 'for this reason']
        moral_reasoning_count = sum(1 for connector in moral_connectors 
                                   if connector in text.lower())
        features['moral_reasoning_density'] = moral_reasoning_count / max(len(sentences), 1)
        
        # 7. Emotion Classification
        if self.emotion_classifier and len(text) > 10:
            try:
                emotions = self.emotion_classifier(text[:512])[0]  # Limit length
                for emotion in emotions:
                    features[f'emotion_{emotion["label"]}'] = emotion['score']
            except:
                # Fallback if classifier fails
                for emotion in ['anger', 'fear', 'joy', 'sadness', 'surprise', 'disgust']:
                    features[f'emotion_{emotion}'] = 0.0
        
        # 8. Moral Stakes Indicators
        stakes_words = ['life', 'death', 'survival', 'future', 'children', 'generation',
                       'civilization', 'humanity', 'world', 'planet', 'existence']
        stakes_count = sum(1 for word in words if word in stakes_words)
        features['moral_stakes_level'] = stakes_count / max(len(words), 1)
        
        # 9. Group Identity Markers
        group_pronouns = ['we', 'us', 'our', 'they', 'them', 'their']
        group_count = sum(1 for word in words if word in group_pronouns)
        features['group_identity_strength'] = group_count / max(len(words), 1)
        
        # 10. Temporal Urgency
        future_words = ['will', 'going to', 'about to', 'soon', 'imminent']
        past_words = ['was', 'were', 'had', 'used to', 'previously']
        present_words = ['now', 'currently', 'today', 'present', 'immediate']
        
        future_count = sum(1 for word in words if word in future_words)
        present_count = sum(1 for word in words if word in present_words)
        
        features['temporal_urgency'] = present_count / max(future_count + present_count + 1, 1)
        
        return features


class QMoralActivationModel(nn.Module):
    """Deep learning model for q (Moral Activation Energy) prediction with biological optimization"""
    
    def __init__(self, feature_dim: int, embedding_dim: int = 768,
                 hidden_dims: List[int] = [512, 256, 128]):
        super().__init__()
        
        # Feature processing branch with moral-specific architecture
        self.moral_feature_processor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dims[0], hidden_dims[1])
        )
        
        # Embedding processing with emotional resonance
        self.embedding_processor = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.ReLU()
        )
        
        # Emotional resonance layer
        self.emotional_resonance = EmotionalResonanceLayer(
            hidden_dims[0], 
            hidden_dims[1], 
            num_emotions=8
        )
        
        # Moral foundations modeling
        self.moral_foundation_heads = nn.ModuleList([
            nn.Linear(hidden_dims[1], 16) for _ in range(6)  # 6 moral foundations
        ])
        
        # Integration network
        combined_dim = hidden_dims[1] * 2 + 16 * 6  # features + resonance + foundations
        
        self.integration_network = nn.Sequential(
            nn.Linear(combined_dim, hidden_dims[2]),
            nn.BatchNorm1d(hidden_dims[2]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dims[2], 64),
            nn.ReLU()
        )
        
        # Biological optimization parameters (learnable)
        self.km = nn.Parameter(torch.tensor(0.2))
        self.ki = nn.Parameter(torch.tensor(0.8))
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, features: torch.Tensor, embeddings: torch.Tensor):
        # Process moral features
        moral_features = self.moral_feature_processor(features)
        
        # Process embeddings with emotional resonance
        embed_processed = self.embedding_processor(embeddings)
        emotional_features = self.emotional_resonance(embed_processed)
        
        # Compute moral foundation activations
        foundation_outputs = []
        for head in self.moral_foundation_heads:
            foundation_outputs.append(head(moral_features))
        foundations_concat = torch.cat(foundation_outputs, dim=1)
        
        # Combine all features
        combined = torch.cat([moral_features, emotional_features, foundations_concat], dim=1)
        
        # Integration
        integrated = self.integration_network(combined)
        
        # Raw moral activation
        q_raw = self.output_layer(integrated).squeeze()
        
        # Apply biological optimization inline
        # This prevents extreme moral activation (extremism)
        q_optimized = q_raw / (self.km + q_raw + (q_raw ** 2) / self.ki)
        
        return q_optimized


class QTrainer:
    """Trainer for the Q moral activation model"""
    
    def __init__(self, model: QMoralActivationModel, device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.feature_extractor = MoralActivationFeatureExtractor()
        
        # Use RoBERTa for better emotion understanding
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        self.transformer = AutoModel.from_pretrained("roberta-base").to(device)
        self.transformer.eval()
        
    def prepare_batch(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare batch of texts for model input"""
        
        # Extract moral activation features
        all_features = []
        for text in texts:
            features = self.feature_extractor.extract_moral_features(text)
            feature_vector = list(features.values())
            all_features.append(feature_vector)
        
        features_tensor = torch.FloatTensor(all_features).to(self.device)
        
        # Extract embeddings
        with torch.no_grad():
            inputs = self.tokenizer(texts, return_tensors="pt", truncation=True,
                                  padding=True, max_length=512).to(self.device)
            outputs = self.transformer(**inputs)
            
            # Use last 4 layers for richer representation
            hidden_states = outputs.last_hidden_state
            embeddings = hidden_states.mean(dim=1)  # Average pooling
        
        return features_tensor, embeddings
    
    def train(self, train_data: List[Tuple[str, float]], 
              val_data: Optional[List[Tuple[str, float]]] = None,
              epochs: int = 200, batch_size: int = 16, lr: float = 0.0003):
        """Train the Q model with moral activation specific optimizations"""
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        
        # Learning rate scheduling
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=lr,
            epochs=epochs,
            steps_per_epoch=len(train_data) // batch_size
        )
        
        # Custom loss function that considers biological optimization
        def moral_activation_loss(predictions, targets):
            # Base MSE loss
            mse = F.mse_loss(predictions, targets)
            
            # Regularization to prevent extreme activation
            extremity_penalty = torch.mean((predictions - 0.5).pow(4)) * 0.05
            
            # Encourage smooth activation curves
            if len(predictions) > 1:
                smoothness = torch.mean(torch.abs(predictions[1:] - predictions[:-1])) * 0.02
            else:
                smoothness = 0
            
            return mse + extremity_penalty + smoothness
        
        best_val_loss = float('inf')
        patience = 20
        patience_counter = 0
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            # Shuffle training data
            np.random.shuffle(train_data)
            
            for i in range(0, len(train_data), batch_size):
                batch = train_data[i:i + batch_size]
                texts, targets = zip(*batch)
                
                # Prepare inputs
                features, embeddings = self.prepare_batch(list(texts))
                targets_tensor = torch.FloatTensor(targets).to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                predictions = self.model(features, embeddings)
                loss = moral_activation_loss(predictions, targets_tensor)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / (len(train_data) / batch_size)
            
            # Validation
            if val_data and epoch % 10 == 0:
                val_loss = self.evaluate(val_data, batch_size)
                print(f"Epoch {epoch}: Train Loss = {avg_loss:.4f}, Val Loss = {val_loss:.4f}")
                print(f"  Km = {self.model.km.item():.3f}, Ki = {self.model.ki.item():.3f}")
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    torch.save(self.model.state_dict(), '/Users/chris/GCT-ML-Lab/models/q_best.pth')
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch}")
                        break
            elif epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss = {avg_loss:.4f}")
    
    def evaluate(self, data: List[Tuple[str, float]], batch_size: int = 16) -> float:
        """Evaluate the model"""
        self.model.eval()
        total_loss = 0
        criterion = nn.MSELoss()
        
        with torch.no_grad():
            for i in range(0, len(data), batch_size):
                batch = data[i:i + batch_size]
                texts, targets = zip(*batch)
                
                features, embeddings = self.prepare_batch(list(texts))
                targets_tensor = torch.FloatTensor(targets).to(self.device)
                
                predictions = self.model(features, embeddings)
                loss = criterion(predictions, targets_tensor)
                total_loss += loss.item()
        
        return total_loss / (len(data) / batch_size)
    
    def predict(self, text: str) -> Tuple[float, Dict[str, float]]:
        """Predict Q value and return additional insights"""
        self.model.eval()
        
        with torch.no_grad():
            features, embeddings = self.prepare_batch([text])
            q = self.model(features, embeddings).item()
            
            # Extract feature importance
            feature_dict = self.feature_extractor.extract_moral_features(text)
            
        return q, feature_dict


def create_q_model(feature_dim: int = 50) -> Tuple[QMoralActivationModel, QTrainer]:
    """Create and return a Q model and trainer"""
    model = QMoralActivationModel(feature_dim=feature_dim)
    trainer = QTrainer(model)
    return model, trainer


if __name__ == "__main__":
    # Example usage
    model, trainer = create_q_model()
    
    # Example training data showing different levels of moral activation
    sample_data = [
        ("We must act NOW to save our children's future! This is not a drill - our very survival depends on immediate action!", 0.95),
        ("It would be nice if people considered being more environmentally conscious when convenient.", 0.3),
        ("Our moral duty demands that we stand up for justice. The time for action is now, and we cannot afford to wait any longer.", 0.85),
        ("Some people think one way, others think differently. Both perspectives have merit.", 0.2),
        ("This is a CRITICAL moment in history! We MUST unite to defend our values or risk losing everything we hold dear!", 0.9),
    ]
    
    # Test prediction
    test_text = """
    The time has come for us to take a stand. Our children's future hangs in the balance, 
    and we cannot afford to remain silent. We must act with courage and conviction to 
    protect what matters most. This is our moral obligation.
    """
    # q_score, features = trainer.predict(test_text)
    # print(f"Predicted q (Moral Activation Energy): {q_score:.3f}")