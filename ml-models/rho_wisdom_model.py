"""
ρ (Rho) - Accumulated Wisdom ML Model
Neural network for measuring depth of understanding and learned insights

This model focuses on:
- Causal reasoning patterns
- Temporal perspective and evolution of thought
- Evidence-based reasoning
- Complexity of analysis
- Integration of multiple perspectives
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from transformers import AutoModel, AutoTokenizer
import re
from collections import defaultdict
import spacy
from datetime import datetime


class WisdomGraphNetwork(nn.Module):
    """Graph neural network component for modeling knowledge relationships"""
    
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int):
        super().__init__()
        self.node_transform = nn.Linear(node_dim, hidden_dim)
        self.edge_transform = nn.Linear(edge_dim, hidden_dim)
        
        # Message passing layers
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Node update
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        
    def forward(self, node_features, edge_features, adjacency):
        # Transform features
        nodes = self.node_transform(node_features)
        edges = self.edge_transform(edge_features)
        
        # Message passing
        for _ in range(3):  # 3 iterations
            messages = []
            for i in range(nodes.size(0)):
                # Aggregate messages from neighbors
                neighbors = adjacency[i].nonzero().squeeze()
                if neighbors.numel() > 0:
                    neighbor_feats = nodes[neighbors]
                    edge_feats = edges[i, neighbors] if edges.dim() > 1 else edges
                    
                    # Concatenate source, edge, target features
                    combined = torch.cat([
                        nodes[i].unsqueeze(0).expand(neighbor_feats.size(0), -1),
                        edge_feats,
                        neighbor_feats
                    ], dim=-1)
                    
                    # Compute messages
                    msg = self.message_mlp(combined).mean(dim=0)
                    messages.append(msg)
                else:
                    messages.append(torch.zeros_like(nodes[i]))
            
            # Update nodes
            messages = torch.stack(messages)
            nodes = self.gru(messages, nodes)
        
        return nodes


class WisdomFeatureExtractor:
    """Extract features related to accumulated wisdom and depth of understanding"""
    
    def __init__(self):
        # Load spaCy for advanced NLP
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            print("SpaCy model not found. Some features will be limited.")
            self.nlp = None
        
        # Knowledge indicators
        self.knowledge_patterns = {
            'research': ['research', 'study', 'investigation', 'analysis', 'examination',
                        'experiment', 'observation', 'findings', 'results', 'data'],
            'evidence': ['evidence', 'proof', 'demonstrate', 'indicate', 'suggest',
                        'reveal', 'show', 'confirm', 'validate', 'support'],
            'theory': ['theory', 'hypothesis', 'principle', 'concept', 'framework',
                      'model', 'paradigm', 'approach', 'methodology', 'system'],
            'causation': ['because', 'therefore', 'thus', 'hence', 'consequently',
                         'as a result', 'leads to', 'causes', 'due to', 'results in'],
            'temporal': ['historically', 'previously', 'traditionally', 'originally',
                        'evolved', 'developed', 'progressed', 'changed', 'transformed'],
            'integration': ['however', 'although', 'while', 'whereas', 'despite',
                           'nevertheless', 'furthermore', 'moreover', 'additionally'],
            'certainty': ['clearly', 'obviously', 'certainly', 'definitely', 'undoubtedly',
                         'perhaps', 'possibly', 'might', 'could', 'seems']
        }
        
        # Citation patterns
        self.citation_patterns = [
            r'\(\d{4}\)',  # (2023)
            r'\[\d+\]',    # [1]
            r'et al\.',    # et al.
            r'according to',
            r'as .+ noted',
            r'research by',
            r'study by'
        ]
        
    def extract_wisdom_features(self, text: str) -> Dict[str, float]:
        """Extract features indicating accumulated wisdom"""
        
        features = {}
        words = text.lower().split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # 1. Knowledge Depth Indicators
        for category, patterns in self.knowledge_patterns.items():
            count = sum(1 for word in words if word in patterns)
            features[f'knowledge_{category}_count'] = count
            features[f'knowledge_{category}_density'] = count / max(len(words), 1)
        
        # 2. Causal Reasoning Complexity
        causal_chains = self._extract_causal_chains(text)
        features['causal_chain_count'] = len(causal_chains)
        features['avg_causal_chain_length'] = np.mean([len(c) for c in causal_chains]) if causal_chains else 0
        features['max_causal_chain_length'] = max([len(c) for c in causal_chains]) if causal_chains else 0
        
        # 3. Evidence Integration
        citation_count = sum(1 for pattern in self.citation_patterns 
                           for _ in re.finditer(pattern, text, re.IGNORECASE))
        features['citation_count'] = citation_count
        features['citation_density'] = citation_count / max(len(sentences), 1)
        
        # 4. Analytical Complexity
        # Sentence complexity based on subordinate clauses
        complex_sentences = 0
        for sent in sentences:
            if any(marker in sent.lower() for marker in ['which', 'that', 'who', 'whom', 'whose', 'when', 'where']):
                complex_sentences += 1
        features['complex_sentence_ratio'] = complex_sentences / max(len(sentences), 1)
        
        # Average sentence depth (words per sentence)
        features['avg_sentence_depth'] = len(words) / max(len(sentences), 1)
        
        # 5. Temporal Perspective
        temporal_count = sum(1 for word in words if word in self.knowledge_patterns['temporal'])
        features['temporal_perspective_score'] = temporal_count / max(len(words), 1)
        
        # Historical references
        historical_patterns = ['year', 'century', 'decade', 'era', 'period', 'age']
        historical_count = sum(1 for word in words if word in historical_patterns)
        features['historical_depth'] = historical_count / max(len(words), 1)
        
        # 6. Multi-perspective Integration
        contrast_markers = self.knowledge_patterns['integration']
        contrast_count = sum(1 for word in words if word in contrast_markers)
        features['perspective_integration'] = contrast_count / max(len(sentences), 1)
        
        # 7. Conceptual Density
        # Ratio of abstract/conceptual words
        abstract_words = ['concept', 'idea', 'theory', 'principle', 'notion', 'aspect',
                         'dimension', 'factor', 'element', 'component', 'characteristic']
        abstract_count = sum(1 for word in words if word in abstract_words)
        features['conceptual_density'] = abstract_count / max(len(words), 1)
        
        # 8. Argument Structure
        if self.nlp:
            doc = self.nlp(text)
            # Count different dependency patterns
            dep_patterns = defaultdict(int)
            for token in doc:
                dep_patterns[token.dep_] += 1
            
            # Specific patterns indicating reasoning
            features['subject_relations'] = dep_patterns.get('nsubj', 0) / max(len(doc), 1)
            features['object_relations'] = dep_patterns.get('dobj', 0) / max(len(doc), 1)
            features['modifier_density'] = (dep_patterns.get('amod', 0) + dep_patterns.get('advmod', 0)) / max(len(doc), 1)
        
        # 9. Knowledge Building Patterns
        building_phrases = ['builds on', 'extends', 'develops', 'expands', 'elaborates',
                           'based on', 'derived from', 'grounded in', 'rooted in']
        building_count = sum(1 for phrase in building_phrases if phrase in text.lower())
        features['knowledge_building_score'] = building_count / max(len(sentences), 1)
        
        # 10. Certainty Calibration
        high_certainty = sum(1 for word in words if word in self.knowledge_patterns['certainty'][:5])
        low_certainty = sum(1 for word in words if word in self.knowledge_patterns['certainty'][5:])
        features['certainty_balance'] = 1 - abs(high_certainty - low_certainty) / max(high_certainty + low_certainty, 1)
        
        return features
    
    def _extract_causal_chains(self, text: str) -> List[List[str]]:
        """Extract causal reasoning chains from text"""
        causal_chains = []
        sentences = re.split(r'[.!?]+', text)
        
        current_chain = []
        causal_markers = self.knowledge_patterns['causation']
        
        for sent in sentences:
            if any(marker in sent.lower() for marker in causal_markers):
                current_chain.append(sent.strip())
            else:
                if len(current_chain) > 1:
                    causal_chains.append(current_chain)
                current_chain = [sent.strip()] if sent.strip() else []
        
        if len(current_chain) > 1:
            causal_chains.append(current_chain)
        
        return causal_chains


class RhoWisdomModel(nn.Module):
    """Deep learning model for ρ (Accumulated Wisdom) prediction"""
    
    def __init__(self, feature_dim: int, embedding_dim: int = 768,
                 hidden_dims: List[int] = [512, 256, 128], use_graph: bool = True):
        super().__init__()
        
        self.use_graph = use_graph
        
        # Feature processing with residual connections
        self.feature_blocks = nn.ModuleList()
        prev_dim = feature_dim
        for hidden_dim in hidden_dims[:2]:
            block = nn.Sequential(
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            )
            self.feature_blocks.append(block)
            prev_dim = hidden_dim
        
        # Embedding processing with attention
        self.embedding_attention = nn.MultiheadAttention(embedding_dim, num_heads=8, batch_first=True)
        self.embedding_processor = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Graph network for knowledge relationships (optional)
        if use_graph:
            self.graph_network = WisdomGraphNetwork(
                node_dim=hidden_dims[1],
                edge_dim=32,
                hidden_dim=hidden_dims[1]
            )
        
        # Wisdom-specific layers
        combined_dim = hidden_dims[0] + hidden_dims[1]
        
        self.wisdom_processor = nn.Sequential(
            nn.Linear(combined_dim, hidden_dims[2]),
            nn.BatchNorm1d(hidden_dims[2]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dims[2], 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # Output with careful normalization for wisdom score
        self.output_layer = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        # Calibration layer for fine-tuning
        self.calibration = nn.Parameter(torch.ones(1))
        self.calibration_bias = nn.Parameter(torch.zeros(1))
        
    def forward(self, features: torch.Tensor, embeddings: torch.Tensor, 
                graph_data: Optional[Dict] = None):
        
        # Process features through residual blocks
        feat_out = features
        feat_residuals = []
        for block in self.feature_blocks:
            feat_out = block(feat_out)
            feat_residuals.append(feat_out)
        
        # Process embeddings with self-attention
        if len(embeddings.shape) == 2:
            embeddings = embeddings.unsqueeze(1)  # Add sequence dimension
        
        attended_embeddings, _ = self.embedding_attention(embeddings, embeddings, embeddings)
        embed_processed = self.embedding_processor(attended_embeddings.squeeze(1))
        
        # Apply graph network if available
        if self.use_graph and graph_data is not None:
            graph_features = self.graph_network(
                graph_data['nodes'],
                graph_data['edges'],
                graph_data['adjacency']
            )
            # Aggregate graph features
            feat_out = feat_out + graph_features.mean(dim=0)
        
        # Combine processed features
        combined = torch.cat([embed_processed, feat_residuals[-1]], dim=1)
        
        # Process through wisdom-specific layers
        wisdom_features = self.wisdom_processor(combined)
        
        # Generate final wisdom score with calibration
        raw_score = self.output_layer(wisdom_features)
        calibrated_score = torch.clamp(
            raw_score * self.calibration + self.calibration_bias,
            min=0.0, max=1.0
        )
        
        return calibrated_score.squeeze()


class RhoTrainer:
    """Trainer for the Rho wisdom model"""
    
    def __init__(self, model: RhoWisdomModel, device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.feature_extractor = WisdomFeatureExtractor()
        
        # Transformer for embeddings
        self.tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
        self.transformer = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased").to(device)
        self.transformer.eval()
        
    def prepare_batch(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare batch of texts for model input"""
        
        # Extract wisdom features
        all_features = []
        for text in texts:
            features = self.feature_extractor.extract_wisdom_features(text)
            feature_vector = list(features.values())
            all_features.append(feature_vector)
        
        features_tensor = torch.FloatTensor(all_features).to(self.device)
        
        # Extract embeddings using SciBERT for better scientific text understanding
        with torch.no_grad():
            inputs = self.tokenizer(texts, return_tensors="pt", truncation=True,
                                  padding=True, max_length=512).to(self.device)
            outputs = self.transformer(**inputs)
            
            # Use CLS token embedding + mean pooling
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            mean_embeddings = outputs.last_hidden_state.mean(dim=1)
            embeddings = (cls_embeddings + mean_embeddings) / 2
        
        return features_tensor, embeddings
    
    def train(self, train_data: List[Tuple[str, float]], 
              val_data: Optional[List[Tuple[str, float]]] = None,
              epochs: int = 150, batch_size: int = 16, lr: float = 0.0005):
        """Train the Rho model with wisdom-specific optimizations"""
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        # Custom loss that emphasizes gradual wisdom accumulation
        def wisdom_loss(predictions, targets):
            mse = F.mse_loss(predictions, targets)
            # Penalize extreme predictions
            extremity_penalty = torch.mean(torch.abs(predictions - 0.5) ** 2) * 0.1
            return mse + extremity_penalty
        
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
                features, embeddings = self.prepare_batch(list(texts))
                targets_tensor = torch.FloatTensor(targets).to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                predictions = self.model(features, embeddings)
                loss = wisdom_loss(predictions, targets_tensor)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
            
            scheduler.step()
            avg_loss = total_loss / (len(train_data) / batch_size)
            
            # Validation
            if val_data and epoch % 10 == 0:
                val_loss = self.evaluate(val_data, batch_size)
                print(f"Epoch {epoch}: Train Loss = {avg_loss:.4f}, Val Loss = {val_loss:.4f}")
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    # Save best model
                    torch.save(self.model.state_dict(), '/Users/chris/GCT-ML-Lab/models/rho_best.pth')
            elif epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss = {avg_loss:.4f}")
    
    def predict(self, text: str) -> float:
        """Predict Rho value for a single text"""
        self.model.eval()
        
        with torch.no_grad():
            features, embeddings = self.prepare_batch([text])
            rho = self.model(features, embeddings).item()
        
        return rho


def create_rho_model(feature_dim: int = 35) -> Tuple[RhoWisdomModel, RhoTrainer]:
    """Create and return a Rho model and trainer"""
    model = RhoWisdomModel(feature_dim=feature_dim, use_graph=False)  # Graph features optional
    trainer = RhoTrainer(model)
    return model, trainer


if __name__ == "__main__":
    # Example usage
    model, trainer = create_rho_model()
    
    # Example training data demonstrating different levels of wisdom
    sample_data = [
        ("Research conducted over the past decade has revealed complex patterns in neural development. These findings build upon earlier work by Smith et al. (2019), suggesting that synaptic plasticity evolves through distinct phases. Consequently, our understanding of learning mechanisms has fundamentally transformed.", 0.9),
        ("The thing is good. People like it. It works well. Everyone should use it.", 0.2),
        ("Historical analysis demonstrates that technological advancement follows predictable cycles. As noted by multiple researchers, innovation patterns from the industrial revolution mirror current digital transformation trends. This suggests underlying principles governing societal adaptation to new technologies.", 0.85),
    ]
    
    # Test prediction
    test_text = """
    Building upon foundational theories in cognitive science, recent investigations have uncovered 
    nuanced relationships between memory consolidation and sleep cycles. These findings not only 
    extend previous work but also challenge traditional assumptions about learning processes.
    """
    # rho_score = trainer.predict(test_text)
    # print(f"Predicted ρ (Accumulated Wisdom): {rho_score:.3f}")