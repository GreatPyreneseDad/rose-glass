"""
Ψ (Psi) - Internal Consistency ML Model
Specialized neural network for calculating internal narrative consistency

This model focuses on:
- Vocabulary coherence and diversity
- Structural consistency (sentence patterns)
- Thematic unity and flow
- Logical transitions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from transformers import AutoModel, AutoTokenizer
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter
import re


class PsiAttentionLayer(nn.Module):
    """Self-attention layer for capturing long-range dependencies in text"""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.scale = np.sqrt(hidden_dim)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, hidden_dim)
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, V)
        
        return attended, attention_weights


class ConsistencyFeatureExtractor:
    """Extract specialized features for internal consistency analysis"""
    
    def __init__(self):
        # Download required NLTK data
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
        except:
            pass
        
        # Transition patterns for logical flow
        self.transition_patterns = {
            'sequential': ['first', 'second', 'third', 'next', 'then', 'finally'],
            'causal': ['because', 'therefore', 'thus', 'hence', 'as a result'],
            'contrast': ['however', 'although', 'despite', 'nevertheless', 'whereas'],
            'addition': ['furthermore', 'moreover', 'additionally', 'also', 'besides'],
            'conclusion': ['in conclusion', 'to summarize', 'overall', 'in summary']
        }
        
    def extract_consistency_features(self, text: str) -> Dict[str, float]:
        """Extract features specifically relevant to internal consistency"""
        
        # Tokenization
        sentences = sent_tokenize(text)
        words = word_tokenize(text.lower())
        
        features = {}
        
        # 1. Vocabulary Consistency Metrics
        unique_words = set(words)
        total_words = len(words)
        
        features['vocabulary_diversity'] = len(unique_words) / max(total_words, 1)
        features['type_token_ratio'] = len(unique_words) / max(total_words, 1)
        
        # Word frequency distribution (measure of repetition patterns)
        word_freq = Counter(words)
        freq_values = list(word_freq.values())
        features['word_freq_mean'] = np.mean(freq_values) if freq_values else 0
        features['word_freq_std'] = np.std(freq_values) if freq_values else 0
        
        # 2. Sentence Structure Consistency
        sentence_lengths = [len(word_tokenize(sent)) for sent in sentences]
        if sentence_lengths:
            features['avg_sentence_length'] = np.mean(sentence_lengths)
            features['sentence_length_variance'] = np.var(sentence_lengths)
            features['sentence_length_consistency'] = 1 / (1 + np.std(sentence_lengths))
        else:
            features['avg_sentence_length'] = 0
            features['sentence_length_variance'] = 0
            features['sentence_length_consistency'] = 0
        
        # 3. Thematic Consistency (key term repetition)
        # Find content words (length > 4) that appear multiple times
        content_words = [w for w in words if len(w) > 4]
        content_freq = Counter(content_words)
        repeated_themes = [w for w, c in content_freq.items() if c > 1]
        
        features['thematic_repetition_ratio'] = len(repeated_themes) / max(len(unique_words), 1)
        features['avg_theme_frequency'] = np.mean([content_freq[w] for w in repeated_themes]) if repeated_themes else 0
        
        # 4. Logical Flow Indicators
        transition_counts = {}
        for category, patterns in self.transition_patterns.items():
            count = sum(1 for pattern in patterns if pattern in text.lower())
            transition_counts[category] = count
            features[f'transition_{category}_count'] = count
        
        features['total_transitions'] = sum(transition_counts.values())
        features['transition_density'] = features['total_transitions'] / max(len(sentences), 1)
        
        # 5. Paragraph Structure (if text has newlines)
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        features['paragraph_count'] = len(paragraphs)
        if len(paragraphs) > 1:
            para_lengths = [len(word_tokenize(p)) for p in paragraphs]
            features['paragraph_balance'] = 1 / (1 + np.std(para_lengths))
        else:
            features['paragraph_balance'] = 1
        
        # 6. Pronoun Consistency (tracking subject continuity)
        pronouns = ['i', 'you', 'he', 'she', 'it', 'we', 'they']
        pronoun_counts = {p: words.count(p) for p in pronouns}
        
        # Check if there's a dominant perspective
        max_pronoun_count = max(pronoun_counts.values()) if pronoun_counts.values() else 0
        total_pronouns = sum(pronoun_counts.values())
        features['pronoun_consistency'] = max_pronoun_count / max(total_pronouns, 1)
        
        # 7. Temporal Consistency
        temporal_markers = ['now', 'then', 'before', 'after', 'during', 'while', 
                          'yesterday', 'today', 'tomorrow', 'always', 'never']
        temporal_count = sum(1 for word in words if word in temporal_markers)
        features['temporal_marker_density'] = temporal_count / max(total_words, 1)
        
        # 8. Syntactic Parallelism (similar sentence structures)
        if len(sentences) > 1:
            # Simple approach: check for similar sentence starts
            sentence_starts = [sent.split()[0].lower() if sent.split() else '' for sent in sentences]
            start_freq = Counter(sentence_starts)
            parallel_structures = sum(1 for count in start_freq.values() if count > 1)
            features['syntactic_parallelism'] = parallel_structures / len(sentences)
        else:
            features['syntactic_parallelism'] = 0
        
        return features


class PsiConsistencyModel(nn.Module):
    """Deep learning model for Ψ (Internal Consistency) prediction"""
    
    def __init__(self, feature_dim: int, embedding_dim: int = 768, 
                 hidden_dims: List[int] = [512, 256, 128]):
        super().__init__()
        
        # Feature processing branch
        self.feature_processor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Embedding processing branch (for transformer embeddings)
        self.embedding_processor = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Attention mechanism for feature importance
        self.attention = PsiAttentionLayer(hidden_dims[0])
        
        # Combined processing
        combined_dim = hidden_dims[0] * 2  # Features + embeddings
        
        layers = []
        prev_dim = combined_dim
        for hidden_dim in hidden_dims[1:]:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        # Output layer with sigmoid for [0, 1] range
        layers.extend([
            nn.Linear(prev_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        ])
        
        self.output_network = nn.Sequential(*layers)
        
    def forward(self, features: torch.Tensor, embeddings: torch.Tensor):
        # Process hand-crafted features
        feat_processed = self.feature_processor(features)
        
        # Process transformer embeddings
        embed_processed = self.embedding_processor(embeddings)
        
        # Combine branches
        combined = torch.cat([feat_processed, embed_processed], dim=1)
        
        # Apply attention if we have sequence dimension
        if len(combined.shape) == 3:
            attended, _ = self.attention(combined)
            combined = attended.mean(dim=1)  # Pool over sequence
        
        # Final prediction
        psi = self.output_network(combined)
        
        return psi.squeeze()


class PsiTrainer:
    """Trainer for the Psi consistency model"""
    
    def __init__(self, model: PsiConsistencyModel, device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.feature_extractor = ConsistencyFeatureExtractor()
        
        # For transformer embeddings
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.transformer = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(device)
        self.transformer.eval()
        
    def prepare_batch(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare a batch of texts for model input"""
        
        # Extract hand-crafted features
        all_features = []
        for text in texts:
            features = self.feature_extractor.extract_consistency_features(text)
            feature_vector = list(features.values())
            all_features.append(feature_vector)
        
        features_tensor = torch.FloatTensor(all_features).to(self.device)
        
        # Extract transformer embeddings
        with torch.no_grad():
            inputs = self.tokenizer(texts, return_tensors="pt", truncation=True,
                                  padding=True, max_length=512).to(self.device)
            outputs = self.transformer(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)  # Average pooling
        
        return features_tensor, embeddings
    
    def train(self, train_data: List[Tuple[str, float]], val_data: Optional[List[Tuple[str, float]]] = None,
              epochs: int = 100, batch_size: int = 16, lr: float = 0.001):
        """Train the Psi model"""
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
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
                loss = criterion(predictions, targets_tensor)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / (len(train_data) / batch_size)
            
            # Validation
            if val_data and epoch % 10 == 0:
                val_loss = self.evaluate(val_data, batch_size)
                print(f"Epoch {epoch}: Train Loss = {avg_loss:.4f}, Val Loss = {val_loss:.4f}")
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
    
    def predict(self, text: str) -> float:
        """Predict Psi value for a single text"""
        self.model.eval()
        
        with torch.no_grad():
            features, embeddings = self.prepare_batch([text])
            psi = self.model(features, embeddings).item()
        
        return psi


def create_psi_model(feature_dim: int = 30) -> Tuple[PsiConsistencyModel, PsiTrainer]:
    """Create and return a Psi model and trainer"""
    model = PsiConsistencyModel(feature_dim=feature_dim)
    trainer = PsiTrainer(model)
    return model, trainer


if __name__ == "__main__":
    # Example usage
    model, trainer = create_psi_model()
    
    # Example training data (would need real labeled data)
    sample_data = [
        ("This is a well-structured argument. First, we establish the premise. Then, we explore the implications. Finally, we draw conclusions based on the evidence presented.", 0.9),
        ("Random thoughts. Sky blue. Dogs bark. Pizza good. No connection between ideas whatsoever.", 0.2),
        ("The research demonstrates clear patterns. Furthermore, these patterns align with our hypothesis. Therefore, we can conclude that our theory holds merit.", 0.85),
    ]
    
    # Train model
    # trainer.train(sample_data, epochs=50)
    
    # Test prediction
    test_text = "This analysis begins with examining the data. The data reveals interesting patterns. These patterns suggest important conclusions."
    # psi_score = trainer.predict(test_text)
    # print(f"Predicted Ψ (Internal Consistency): {psi_score:.3f}")