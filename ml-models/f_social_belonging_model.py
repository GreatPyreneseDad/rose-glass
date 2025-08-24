"""
f (Frequency) - Social Belonging Architecture ML Model
Neural network for measuring collective identity and social resonance

This model focuses on:
- Collective vs individual perspective
- Community bonds and relationships  
- Cultural identity markers
- Social scale and reach
- Group dynamics and cohesion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from transformers import AutoModel, AutoTokenizer
import networkx as nx
import re
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import community  # python-louvain for community detection


class SocialGraphAttention(nn.Module):
    """Graph attention network for modeling social relationships"""
    
    def __init__(self, node_features: int, edge_features: int, 
                 hidden_dim: int, num_heads: int = 4):
        super().__init__()
        
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads
        
        # Multi-head attention components
        self.W_nodes = nn.Linear(node_features, hidden_dim)
        self.W_edges = nn.Linear(edge_features, hidden_dim // 2)
        
        # Attention mechanisms
        self.attention_weights = nn.Parameter(torch.randn(num_heads, self.head_dim * 2))
        self.attention_bias = nn.Parameter(torch.zeros(num_heads))
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, nodes, edges, adjacency_matrix):
        batch_size = nodes.size(0)
        num_nodes = nodes.size(1)
        
        # Transform node features
        node_features = self.W_nodes(nodes)  # (batch, num_nodes, hidden_dim)
        node_features = node_features.view(batch_size, num_nodes, self.num_heads, self.head_dim)
        
        # Compute attention scores
        attention_scores = []
        for head in range(self.num_heads):
            # Get head-specific features
            head_features = node_features[:, :, head, :]  # (batch, num_nodes, head_dim)
            
            # Compute pairwise attention
            scores = torch.zeros(batch_size, num_nodes, num_nodes).to(nodes.device)
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if adjacency_matrix[i, j] > 0:
                        combined = torch.cat([head_features[:, i], head_features[:, j]], dim=-1)
                        score = torch.matmul(combined, self.attention_weights[head]) + self.attention_bias[head]
                        scores[:, i, j] = score
            
            attention_scores.append(scores)
        
        # Apply softmax to get attention weights
        attention_scores = torch.stack(attention_scores, dim=1)  # (batch, heads, nodes, nodes)
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Apply attention to aggregate features
        aggregated = torch.zeros_like(node_features)
        for head in range(self.num_heads):
            for i in range(num_nodes):
                weighted_sum = torch.zeros(batch_size, self.head_dim).to(nodes.device)
                for j in range(num_nodes):
                    if adjacency_matrix[i, j] > 0:
                        weighted_sum += attention_weights[:, head, i, j].unsqueeze(-1) * node_features[:, j, head, :]
                aggregated[:, i, head, :] = weighted_sum
        
        # Reshape and project
        aggregated = aggregated.view(batch_size, num_nodes, self.hidden_dim)
        output = self.output_projection(aggregated)
        
        return output, attention_weights


class SocialBelongingFeatureExtractor:
    """Extract features related to social belonging and collective identity"""
    
    def __init__(self):
        # Social pronoun categories
        self.pronouns = {
            'collective': ['we', 'us', 'our', 'ours', 'ourselves'],
            'individual': ['i', 'me', 'my', 'mine', 'myself'],
            'other': ['they', 'them', 'their', 'theirs'],
            'inclusive': ['everyone', 'everybody', 'all', 'together', 'collectively']
        }
        
        # Social relationship markers
        self.relationships = {
            'family': ['family', 'parent', 'child', 'mother', 'father', 'brother', 
                      'sister', 'cousin', 'relative', 'kin'],
            'community': ['community', 'neighbor', 'citizen', 'member', 'resident',
                         'local', 'town', 'city', 'village'],
            'friendship': ['friend', 'companion', 'ally', 'partner', 'colleague',
                          'peer', 'mate', 'buddy', 'pal'],
            'professional': ['team', 'organization', 'company', 'department', 'staff',
                           'employee', 'coworker', 'associate'],
            'cultural': ['culture', 'tradition', 'heritage', 'customs', 'values',
                        'belief', 'practice', 'ritual', 'ceremony'],
            'national': ['nation', 'country', 'homeland', 'patriot', 'fellow',
                        'countrymen', 'society', 'public']
        }
        
        # Social action verbs
        self.social_actions = {
            'cooperation': ['cooperate', 'collaborate', 'work together', 'join',
                           'unite', 'combine', 'partner', 'team up'],
            'support': ['support', 'help', 'assist', 'aid', 'care', 'nurture',
                       'protect', 'defend', 'stand by', 'back up'],
            'communication': ['share', 'discuss', 'talk', 'communicate', 'express',
                            'listen', 'understand', 'empathize'],
            'belonging': ['belong', 'fit in', 'part of', 'member of', 'identify with',
                         'connected', 'included', 'welcomed']
        }
        
        # Scale indicators
        self.scale_markers = {
            'small': ['group', 'circle', 'handful', 'few', 'several'],
            'medium': ['community', 'organization', 'movement', 'network'],
            'large': ['society', 'nation', 'world', 'global', 'humanity', 'everyone']
        }
        
        # Initialize TF-IDF for semantic similarity
        self.tfidf = TfidfVectorizer(max_features=100, ngram_range=(1, 2))
        
    def extract_social_features(self, text: str) -> Dict[str, float]:
        """Extract features related to social belonging"""
        
        features = {}
        words = text.lower().split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # 1. Pronoun Analysis
        pronoun_counts = {cat: 0 for cat in self.pronouns}
        for word in words:
            for category, pronoun_list in self.pronouns.items():
                if word in pronoun_list:
                    pronoun_counts[category] += 1
        
        total_pronouns = sum(pronoun_counts.values())
        for category, count in pronoun_counts.items():
            features[f'pronoun_{category}_count'] = count
            features[f'pronoun_{category}_ratio'] = count / max(total_pronouns, 1)
        
        # Collective vs Individual ratio
        collective_score = pronoun_counts['collective'] + pronoun_counts['inclusive']
        individual_score = pronoun_counts['individual']
        features['collective_individual_ratio'] = collective_score / max(individual_score + 1, 1)
        
        # 2. Relationship Network Analysis
        relationship_graph = self._build_relationship_graph(text)
        
        if relationship_graph.number_of_nodes() > 0:
            # Graph metrics
            features['social_network_size'] = relationship_graph.number_of_nodes()
            features['social_connections'] = relationship_graph.number_of_edges()
            features['network_density'] = nx.density(relationship_graph)
            
            # Centrality measures
            if relationship_graph.number_of_nodes() > 1:
                centrality = nx.degree_centrality(relationship_graph)
                features['avg_centrality'] = np.mean(list(centrality.values()))
                features['max_centrality'] = max(centrality.values())
            else:
                features['avg_centrality'] = 0
                features['max_centrality'] = 0
            
            # Community detection
            try:
                communities = community.best_partition(relationship_graph)
                features['num_communities'] = len(set(communities.values()))
            except:
                features['num_communities'] = 1
        else:
            features['social_network_size'] = 0
            features['social_connections'] = 0
            features['network_density'] = 0
            features['avg_centrality'] = 0
            features['max_centrality'] = 0
            features['num_communities'] = 0
        
        # 3. Social Relationship Categories
        for category, markers in self.relationships.items():
            count = sum(1 for word in words if word in markers)
            features[f'relationship_{category}_count'] = count
            features[f'relationship_{category}_density'] = count / max(len(words), 1)
        
        # 4. Social Action Patterns
        for action_type, actions in self.social_actions.items():
            count = sum(1 for action in actions 
                       for _ in re.finditer(r'\b' + action + r'\b', text.lower()))
            features[f'social_action_{action_type}'] = count / max(len(sentences), 1)
        
        # 5. Scale of Social Reference
        scale_scores = {scale: 0 for scale in self.scale_markers}
        for scale, markers in self.scale_markers.items():
            for marker in markers:
                scale_scores[scale] += text.lower().count(marker)
        
        total_scale = sum(scale_scores.values())
        for scale, score in scale_scores.items():
            features[f'social_scale_{scale}'] = score / max(total_scale, 1)
        
        # 6. Inclusivity Indicators
        inclusive_terms = ['all', 'everyone', 'anybody', 'nobody', 'together', 
                          'universal', 'shared', 'common', 'mutual']
        exclusive_terms = ['only', 'just', 'except', 'but not', 'excluding',
                          'elite', 'select', 'chosen', 'special']
        
        inclusive_count = sum(1 for word in words if word in inclusive_terms)
        exclusive_count = sum(1 for word in words if word in exclusive_terms)
        
        features['inclusivity_score'] = inclusive_count / max(len(words), 1)
        features['exclusivity_score'] = exclusive_count / max(len(words), 1)
        features['inclusivity_balance'] = (inclusive_count - exclusive_count) / max(inclusive_count + exclusive_count + 1, 1)
        
        # 7. Shared Values and Beliefs
        value_terms = ['value', 'believe', 'principle', 'ideal', 'norm', 'standard',
                      'ethic', 'moral', 'conviction', 'faith']
        value_count = sum(1 for word in words if word in value_terms)
        features['shared_values_density'] = value_count / max(len(words), 1)
        
        # 8. Emotional Connection
        connection_terms = ['love', 'care', 'empathy', 'compassion', 'sympathy',
                           'bond', 'connection', 'attachment', 'affection']
        connection_count = sum(1 for word in words if word in connection_terms)
        features['emotional_connection_score'] = connection_count / max(len(words), 1)
        
        # 9. Temporal Continuity (past-present-future community)
        past_community = ['ancestors', 'forefathers', 'heritage', 'tradition', 'historically']
        present_community = ['currently', 'now', 'today', 'present', 'existing']
        future_community = ['children', 'generations', 'future', 'legacy', 'tomorrow']
        
        past_count = sum(1 for word in words if word in past_community)
        present_count = sum(1 for word in words if word in present_community)
        future_count = sum(1 for word in words if word in future_community)
        
        temporal_span = (min(past_count, 1) + min(present_count, 1) + min(future_count, 1)) / 3
        features['temporal_community_span'] = temporal_span
        
        # 10. Identity Markers
        identity_terms = ['identity', 'who we are', 'defines us', 'makes us',
                         'proud', 'honor', 'dignity', 'character']
        identity_count = sum(1 for term in identity_terms 
                           for _ in re.finditer(term, text.lower()))
        features['identity_strength'] = identity_count / max(len(sentences), 1)
        
        return features
    
    def _build_relationship_graph(self, text: str) -> nx.Graph:
        """Build a graph of social relationships mentioned in text"""
        G = nx.Graph()
        
        # Simple entity extraction (would use NER in production)
        # Look for patterns like "X and Y", "X with Y", etc.
        patterns = [
            r'(\w+) and (\w+)',
            r'(\w+) with (\w+)',
            r'(\w+) or (\w+)',
            r'(\w+), (\w+)',
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text.lower())
            for match in matches:
                entity1, entity2 = match.groups()
                # Filter out common words
                if len(entity1) > 2 and len(entity2) > 2:
                    G.add_edge(entity1, entity2)
        
        return G


class FSocialBelongingModel(nn.Module):
    """Deep learning model for f (Social Belonging) prediction"""
    
    def __init__(self, feature_dim: int, embedding_dim: int = 768,
                 hidden_dims: List[int] = [512, 256, 128]):
        super().__init__()
        
        # Social feature processing
        self.social_feature_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU()
        )
        
        # Embedding processing with social attention
        self.embedding_processor = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Social graph attention (simplified version)
        self.graph_attention = SocialGraphAttention(
            node_features=32,
            edge_features=16,
            hidden_dim=hidden_dims[1],
            num_heads=4
        )
        
        # Pronoun perspective modeling
        self.pronoun_embeddings = nn.Embedding(10, 32)  # Different pronoun types
        self.pronoun_lstm = nn.LSTM(32, hidden_dims[2] // 2, bidirectional=True, batch_first=True)
        
        # Multi-scale social modeling
        self.scale_processors = nn.ModuleList([
            nn.Linear(hidden_dims[1], 64) for _ in range(3)  # small, medium, large scale
        ])
        
        # Integration network
        combined_dim = hidden_dims[1] + hidden_dims[0] + hidden_dims[2] + 64 * 3
        
        self.integration_net = nn.Sequential(
            nn.Linear(combined_dim, hidden_dims[2]),
            nn.BatchNorm1d(hidden_dims[2]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dims[2], 64),
            nn.ReLU()
        )
        
        # Output with calibration
        self.output_net = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        # Learned calibration parameters
        self.scale_factor = nn.Parameter(torch.tensor(1.0))
        self.offset = nn.Parameter(torch.tensor(0.0))
        
    def forward(self, features: torch.Tensor, embeddings: torch.Tensor,
                pronoun_sequence: Optional[torch.Tensor] = None,
                graph_data: Optional[Dict] = None):
        
        # Process social features
        social_features = self.social_feature_net(features)
        
        # Process embeddings
        embed_features = self.embedding_processor(embeddings)
        
        # Process pronoun sequences if available
        if pronoun_sequence is not None:
            pronoun_embeds = self.pronoun_embeddings(pronoun_sequence)
            lstm_out, _ = self.pronoun_lstm(pronoun_embeds)
            pronoun_features = lstm_out.mean(dim=1)  # Average over sequence
        else:
            pronoun_features = torch.zeros(features.size(0), self.pronoun_lstm.hidden_size * 2).to(features.device)
        
        # Multi-scale processing
        scale_outputs = []
        for scale_processor in self.scale_processors:
            scale_out = scale_processor(social_features)
            scale_outputs.append(scale_out)
        scale_features = torch.cat(scale_outputs, dim=1)
        
        # Combine all features
        combined = torch.cat([
            social_features,
            embed_features,
            pronoun_features,
            scale_features
        ], dim=1)
        
        # Integration
        integrated = self.integration_net(combined)
        
        # Output with calibration
        raw_output = self.output_net(integrated)
        calibrated = torch.clamp(
            raw_output * self.scale_factor + self.offset,
            min=0.0, max=1.0
        )
        
        return calibrated.squeeze()


class FTrainer:
    """Trainer for the F social belonging model"""
    
    def __init__(self, model: FSocialBelongingModel, device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.feature_extractor = SocialBelongingFeatureExtractor()
        
        # Use BERT for better contextual understanding
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.transformer = AutoModel.from_pretrained("bert-base-uncased").to(device)
        self.transformer.eval()
        
    def prepare_batch(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare batch of texts for model input"""
        
        # Extract social belonging features
        all_features = []
        all_pronoun_sequences = []
        
        for text in texts:
            features = self.feature_extractor.extract_social_features(text)
            feature_vector = list(features.values())
            all_features.append(feature_vector)
            
            # Extract pronoun sequence
            words = text.lower().split()
            pronoun_map = {
                'we': 0, 'us': 1, 'our': 2, 'i': 3, 'me': 4,
                'they': 5, 'them': 6, 'you': 7, 'all': 8, 'everyone': 9
            }
            pronoun_seq = [pronoun_map.get(w, -1) for w in words if w in pronoun_map]
            if pronoun_seq:
                all_pronoun_sequences.append(pronoun_seq[:50])  # Limit length
            else:
                all_pronoun_sequences.append([0])  # Default
        
        features_tensor = torch.FloatTensor(all_features).to(self.device)
        
        # Pad pronoun sequences
        max_len = max(len(seq) for seq in all_pronoun_sequences)
        padded_sequences = []
        for seq in all_pronoun_sequences:
            padded = seq + [0] * (max_len - len(seq))
            padded_sequences.append(padded)
        
        pronoun_tensor = torch.LongTensor(padded_sequences).to(self.device)
        
        # Extract embeddings
        with torch.no_grad():
            inputs = self.tokenizer(texts, return_tensors="pt", truncation=True,
                                  padding=True, max_length=512).to(self.device)
            outputs = self.transformer(**inputs)
            
            # Use pooler output for sentence-level representation
            embeddings = outputs.pooler_output
        
        return features_tensor, embeddings, pronoun_tensor
    
    def train(self, train_data: List[Tuple[str, float]], 
              val_data: Optional[List[Tuple[str, float]]] = None,
              epochs: int = 150, batch_size: int = 16, lr: float = 0.0005):
        """Train the F model with social belonging specific optimizations"""
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        
        # Cosine annealing with warm restarts
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=30, T_mult=2
        )
        
        # Custom loss emphasizing moderate social belonging
        def social_belonging_loss(predictions, targets):
            # Base MSE loss
            mse = F.mse_loss(predictions, targets)
            
            # Encourage values in moderate range (not too isolated, not too enmeshed)
            ideal_range_center = 0.6
            range_penalty = torch.mean((predictions - ideal_range_center).pow(2)) * 0.05
            
            # Smooth distribution penalty
            if len(predictions) > 1:
                variance = torch.var(predictions)
                variance_penalty = torch.abs(variance - 0.1) * 0.1  # Target variance of 0.1
            else:
                variance_penalty = 0
            
            return mse + range_penalty + variance_penalty
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            # Shuffle training data
            np.random.shuffle(train_data)
            
            for i in range(0, len(train_data), batch_size):
                batch = train_data[i:i + batch_size]
                texts, targets = zip(*batch)
                
                # Prepare inputs
                features, embeddings, pronouns = self.prepare_batch(list(texts))
                targets_tensor = torch.FloatTensor(targets).to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                predictions = self.model(features, embeddings, pronouns)
                loss = social_belonging_loss(predictions, targets_tensor)
                
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
                print(f"  Scale = {self.model.scale_factor.item():.3f}, Offset = {self.model.offset.item():.3f}")
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(self.model.state_dict(), '/Users/chris/GCT-ML-Lab/models/f_best.pth')
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
                
                features, embeddings, pronouns = self.prepare_batch(list(texts))
                targets_tensor = torch.FloatTensor(targets).to(self.device)
                
                predictions = self.model(features, embeddings, pronouns)
                loss = criterion(predictions, targets_tensor)
                total_loss += loss.item()
        
        return total_loss / (len(data) / batch_size)
    
    def predict(self, text: str) -> Tuple[float, Dict[str, float]]:
        """Predict F value and return social insights"""
        self.model.eval()
        
        with torch.no_grad():
            features, embeddings, pronouns = self.prepare_batch([text])
            f_score = self.model(features, embeddings, pronouns).item()
            
            # Extract feature insights
            feature_dict = self.feature_extractor.extract_social_features(text)
            
        return f_score, feature_dict


def create_f_model(feature_dim: int = 60) -> Tuple[FSocialBelongingModel, FTrainer]:
    """Create and return an F model and trainer"""
    model = FSocialBelongingModel(feature_dim=feature_dim)
    trainer = FTrainer(model)
    return model, trainer


if __name__ == "__main__":
    # Example usage
    model, trainer = create_f_model()
    
    # Example training data showing different levels of social belonging
    sample_data = [
        ("We stand together as a community, supporting each other through these challenges. Our shared values and mutual respect make us stronger.", 0.9),
        ("I work alone and prefer it that way. Don't need anyone else.", 0.2),
        ("Our team collaborates effectively, bringing together diverse perspectives to achieve our common goals. Everyone contributes their unique strengths.", 0.85),
        ("Society is just a collection of individuals pursuing their own interests.", 0.3),
        ("Together with our neighbors and friends, we're building a better future for our children and generations to come.", 0.95),
    ]
    
    # Test prediction
    test_text = """
    As members of this community, we share a common purpose and vision. Our collective 
    efforts have created meaningful change, and by working together, we continue to 
    strengthen the bonds that unite us. This is our shared journey.
    """
    # f_score, features = trainer.predict(test_text)
    # print(f"Predicted f (Social Belonging): {f_score:.3f}")