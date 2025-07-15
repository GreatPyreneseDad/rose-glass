"""
Metadata Extraction and Semantic Analysis
Extracts and analyzes learning module metadata for coherence calculation
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import json
from pathlib import Path


class MetadataExtractor:
    """Extracts and processes metadata from learning modules"""
    
    def __init__(self, ontology_model: Optional[str] = None):
        """
        Initialize metadata extractor with optional ontology model
        
        Args:
            ontology_model: Path to pre-trained model or model name
                          Default uses sentence-transformers
        """
        self.ontology_model = ontology_model or 'all-MiniLM-L6-v2'
        
        # Initialize text embedding model
        try:
            self.embedding_model = SentenceTransformer(self.ontology_model)
        except:
            print(f"Failed to load {self.ontology_model}, using TF-IDF fallback")
            self.embedding_model = None
        
        # Knowledge graph for topic relationships
        self.topic_hierarchy = self._load_topic_hierarchy()
        
        # Difficulty calibration parameters
        self.difficulty_factors = {
            'vocabulary_complexity': 0.2,
            'concept_density': 0.3,
            'prerequisite_depth': 0.3,
            'cognitive_operations': 0.2
        }
        
        # Cognitive load taxonomy (Bloom's revised)
        self.cognitive_levels = {
            'remember': 1.0,
            'understand': 2.0,
            'apply': 3.0,
            'analyze': 4.0,
            'evaluate': 5.0,
            'create': 6.0
        }
    
    def _load_topic_hierarchy(self) -> Dict:
        """Load or create topic hierarchy/ontology"""
        # Simplified topic hierarchy for demonstration
        return {
            'programming': {
                'children': ['python', 'javascript', 'algorithms', 'data_structures'],
                'parent': 'computer_science',
                'related': ['software_engineering', 'web_development']
            },
            'python': {
                'children': ['syntax', 'functions', 'classes', 'modules'],
                'parent': 'programming',
                'related': ['data_science', 'automation']
            },
            'machine_learning': {
                'children': ['supervised', 'unsupervised', 'deep_learning', 'nlp'],
                'parent': 'artificial_intelligence',
                'related': ['data_science', 'statistics']
            },
            'data_science': {
                'children': ['data_analysis', 'visualization', 'statistics'],
                'parent': 'computer_science',
                'related': ['machine_learning', 'python']
            }
        }
    
    def extract_topic_vectors(self, modules: List[Dict]) -> pd.DataFrame:
        """
        Embed each module's topic tags into a semantic space
        
        Args:
            modules: List of module dictionaries
            
        Returns:
            DataFrame with columns:
            - module_id
            - topic_vector (embedded representation)
            - primary_topic
            - topic_breadth (diversity of topics)
        """
        results = []
        
        for module in modules:
            module_id = module['module_id']
            
            # Combine all text for embedding
            text_content = self._combine_module_text(module)
            
            # Generate embedding
            if self.embedding_model:
                embedding = self.embedding_model.encode(text_content)
            else:
                # Fallback to TF-IDF
                embedding = self._tfidf_embedding(text_content, modules)
            
            # Analyze topic distribution
            topic_analysis = self._analyze_topics(module['topic_tags'])
            
            results.append({
                'module_id': module_id,
                'topic_vector': embedding,
                'primary_topic': topic_analysis['primary'],
                'topic_breadth': topic_analysis['breadth'],
                'topic_depth': topic_analysis['depth'],
                'semantic_richness': self._calculate_semantic_richness(module)
            })
        
        return pd.DataFrame(results)
    
    def _combine_module_text(self, module: Dict) -> str:
        """Combine all text content from module for embedding"""
        text_parts = [
            module.get('title', ''),
            module.get('description', ''),
            ' '.join(module.get('topic_tags', [])),
            ' '.join(module.get('learning_objectives', []))
        ]
        
        return ' '.join(filter(None, text_parts))
    
    def _tfidf_embedding(self, text: str, all_modules: List[Dict]) -> np.ndarray:
        """Fallback TF-IDF embedding when transformer model unavailable"""
        # Create corpus from all modules
        corpus = [self._combine_module_text(m) for m in all_modules]
        
        vectorizer = TfidfVectorizer(max_features=100)
        tfidf_matrix = vectorizer.fit_transform(corpus)
        
        # Find index of current text
        text_vector = vectorizer.transform([text])
        
        return text_vector.toarray()[0]
    
    def _analyze_topics(self, topic_tags: List[str]) -> Dict:
        """Analyze topic distribution and relationships"""
        if not topic_tags:
            return {'primary': 'general', 'breadth': 0, 'depth': 0}
        
        # Find primary topic (most specific)
        primary_topic = topic_tags[0]
        
        # Calculate breadth (number of distinct top-level topics)
        top_level_topics = set()
        for topic in topic_tags:
            # Find root topic
            current = topic
            while current in self.topic_hierarchy:
                parent = self.topic_hierarchy[current].get('parent')
                if parent:
                    current = parent
                else:
                    break
            top_level_topics.add(current)
        
        breadth = len(top_level_topics)
        
        # Calculate depth (average depth in hierarchy)
        depths = []
        for topic in topic_tags:
            depth = 0
            current = topic
            while current in self.topic_hierarchy:
                parent = self.topic_hierarchy[current].get('parent')
                if parent:
                    depth += 1
                    current = parent
                else:
                    break
            depths.append(depth)
        
        avg_depth = np.mean(depths) if depths else 0
        
        return {
            'primary': primary_topic,
            'breadth': breadth,
            'depth': avg_depth
        }
    
    def _calculate_semantic_richness(self, module: Dict) -> float:
        """Calculate semantic richness of module content"""
        # Factors contributing to richness
        factors = {
            'description_length': len(module.get('description', '').split()),
            'num_objectives': len(module.get('learning_objectives', [])),
            'num_topics': len(module.get('topic_tags', [])),
            'has_prerequisites': len(module.get('prerequisites', [])) > 0
        }
        
        # Normalize and weight factors
        richness = (
            min(factors['description_length'] / 100, 1.0) * 0.3 +
            min(factors['num_objectives'] / 5, 1.0) * 0.3 +
            min(factors['num_topics'] / 3, 1.0) * 0.2 +
            (1.0 if factors['has_prerequisites'] else 0.0) * 0.2
        )
        
        return richness
    
    def assign_difficulty_scores(self, modules: List[Dict]) -> pd.DataFrame:
        """
        Normalize difficulty, duration, and cognitive load metrics
        
        Args:
            modules: List of module dictionaries
            
        Returns:
            DataFrame with columns:
            - module_id
            - normalized_difficulty (0-1)
            - cognitive_complexity (0-1)
            - time_investment (normalized duration)
            - prerequisite_complexity
        """
        results = []
        
        # First pass: collect statistics
        all_durations = [m['duration_minutes'] for m in modules]
        duration_stats = {
            'min': min(all_durations),
            'max': max(all_durations),
            'mean': np.mean(all_durations),
            'std': np.std(all_durations)
        }
        
        # Build prerequisite graph
        prereq_graph = self._build_prerequisite_graph(modules)
        
        for module in modules:
            module_id = module['module_id']
            
            # Normalize difficulty (already 0-1)
            base_difficulty = module.get('difficulty', 0.5)
            
            # Calculate cognitive complexity
            cognitive_complexity = self._calculate_cognitive_complexity(module)
            
            # Normalize duration
            duration = module['duration_minutes']
            normalized_duration = (duration - duration_stats['min']) / (
                duration_stats['max'] - duration_stats['min'] + 1
            )
            
            # Calculate prerequisite complexity
            prereq_complexity = self._calculate_prerequisite_complexity(
                module_id, prereq_graph
            )
            
            # Combined difficulty score
            combined_difficulty = (
                base_difficulty * 0.4 +
                cognitive_complexity * 0.3 +
                prereq_complexity * 0.2 +
                normalized_duration * 0.1
            )
            
            results.append({
                'module_id': module_id,
                'normalized_difficulty': combined_difficulty,
                'cognitive_complexity': cognitive_complexity,
                'time_investment': normalized_duration,
                'prerequisite_complexity': prereq_complexity,
                'cognitive_load': module.get('cognitive_load', 0.5),
                'base_difficulty': base_difficulty
            })
        
        return pd.DataFrame(results)
    
    def _calculate_cognitive_complexity(self, module: Dict) -> float:
        """Calculate cognitive complexity based on learning objectives"""
        objectives = module.get('learning_objectives', [])
        
        if not objectives:
            return 0.5  # Default medium complexity
        
        # Analyze cognitive verbs in objectives
        complexity_scores = []
        
        cognitive_verbs = {
            'remember': ['identify', 'recall', 'recognize', 'list'],
            'understand': ['explain', 'describe', 'summarize', 'interpret'],
            'apply': ['use', 'implement', 'solve', 'demonstrate'],
            'analyze': ['analyze', 'compare', 'contrast', 'examine'],
            'evaluate': ['assess', 'critique', 'judge', 'recommend'],
            'create': ['design', 'construct', 'develop', 'formulate']
        }
        
        for objective in objectives:
            objective_lower = objective.lower()
            max_level = 1.0
            
            for level, verbs in cognitive_verbs.items():
                if any(verb in objective_lower for verb in verbs):
                    max_level = max(max_level, self.cognitive_levels[level])
            
            complexity_scores.append(max_level / 6.0)  # Normalize to 0-1
        
        return np.mean(complexity_scores) if complexity_scores else 0.5
    
    def _build_prerequisite_graph(self, modules: List[Dict]) -> Dict[str, List[str]]:
        """Build directed graph of prerequisites"""
        graph = {}
        
        for module in modules:
            module_id = module['module_id']
            prerequisites = module.get('prerequisites', [])
            
            # Add reverse edges for easier traversal
            for prereq in prerequisites:
                if prereq not in graph:
                    graph[prereq] = []
                graph[prereq].append(module_id)
        
        return graph
    
    def _calculate_prerequisite_complexity(self, 
                                         module_id: str, 
                                         prereq_graph: Dict[str, List[str]]) -> float:
        """Calculate complexity based on prerequisite chain depth"""
        # Find maximum depth in prerequisite chain
        visited = set()
        
        def get_depth(mod_id: str) -> int:
            if mod_id in visited:
                return 0
            visited.add(mod_id)
            
            if mod_id not in prereq_graph:
                return 0
            
            depths = [get_depth(child) + 1 for child in prereq_graph[mod_id]]
            return max(depths) if depths else 0
        
        max_depth = get_depth(module_id)
        
        # Normalize (assume max reasonable depth is 10)
        return min(max_depth / 10.0, 1.0)
    
    def calculate_topic_similarity_matrix(self, topic_vectors_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate pairwise topic similarities between modules
        
        Returns:
            DataFrame with similarity scores between all module pairs
        """
        module_ids = topic_vectors_df['module_id'].values
        vectors = np.vstack(topic_vectors_df['topic_vector'].values)
        
        # Calculate cosine similarity
        similarity_matrix = cosine_similarity(vectors)
        
        # Create DataFrame
        similarity_df = pd.DataFrame(
            similarity_matrix,
            index=module_ids,
            columns=module_ids
        )
        
        return similarity_df
    
    def extract_learning_patterns(self, 
                                modules: List[Dict],
                                learner_history: Optional[List[Dict]] = None) -> Dict:
        """
        Extract patterns useful for personalization
        
        Args:
            modules: Available modules
            learner_history: Optional history of completed modules
            
        Returns:
            Dictionary of extracted patterns
        """
        patterns = {
            'topic_clusters': self._identify_topic_clusters(modules),
            'difficulty_progression': self._analyze_difficulty_progression(modules),
            'common_paths': self._identify_common_learning_paths(modules)
        }
        
        if learner_history:
            patterns['learner_preferences'] = self._analyze_learner_preferences(
                learner_history
            )
        
        return patterns
    
    def _identify_topic_clusters(self, modules: List[Dict]) -> List[List[str]]:
        """Identify natural topic clusters"""
        # Group modules by primary topic
        topic_groups = {}
        
        for module in modules:
            primary_topic = module.get('topic_tags', ['general'])[0]
            if primary_topic not in topic_groups:
                topic_groups[primary_topic] = []
            topic_groups[primary_topic].append(module['module_id'])
        
        return list(topic_groups.values())
    
    def _analyze_difficulty_progression(self, modules: List[Dict]) -> Dict:
        """Analyze natural difficulty progression patterns"""
        # Sort modules by difficulty
        sorted_modules = sorted(modules, key=lambda m: m.get('difficulty', 0))
        
        # Identify difficulty tiers
        tiers = {
            'beginner': [m['module_id'] for m in sorted_modules if m.get('difficulty', 0) < 0.33],
            'intermediate': [m['module_id'] for m in sorted_modules if 0.33 <= m.get('difficulty', 0) < 0.67],
            'advanced': [m['module_id'] for m in sorted_modules if m.get('difficulty', 0) >= 0.67]
        }
        
        return tiers
    
    def _identify_common_learning_paths(self, modules: List[Dict]) -> List[List[str]]:
        """Identify common prerequisite chains"""
        paths = []
        
        # Build prerequisite graph
        prereq_map = {m['module_id']: m.get('prerequisites', []) for m in modules}
        
        # Find modules with no prerequisites (starting points)
        start_modules = [m_id for m_id, prereqs in prereq_map.items() if not prereqs]
        
        # Trace paths from each starting point
        for start in start_modules:
            path = self._trace_dependency_path(start, prereq_map)
            if len(path) > 1:
                paths.append(path)
        
        return paths
    
    def _trace_dependency_path(self, start: str, prereq_map: Dict[str, List[str]]) -> List[str]:
        """Trace a dependency path from a starting module"""
        path = [start]
        
        # Find modules that depend on current module
        dependents = [m_id for m_id, prereqs in prereq_map.items() if start in prereqs]
        
        if dependents:
            # Choose the most logical next step (simplified)
            next_module = dependents[0]
            path.extend(self._trace_dependency_path(next_module, prereq_map))
        
        return path
    
    def _analyze_learner_preferences(self, learner_history: List[Dict]) -> Dict:
        """Analyze learner preferences from history"""
        preferences = {
            'preferred_duration': np.mean([h.get('duration', 30) for h in learner_history]),
            'preferred_difficulty': np.mean([h.get('difficulty', 0.5) for h in learner_history]),
            'completion_rate': len([h for h in learner_history if h.get('completed', False)]) / len(learner_history),
            'favorite_topics': self._extract_favorite_topics(learner_history)
        }
        
        return preferences
    
    def _extract_favorite_topics(self, learner_history: List[Dict]) -> List[str]:
        """Extract most engaged topics from history"""
        topic_engagement = {}
        
        for entry in learner_history:
            for topic in entry.get('topics', []):
                if topic not in topic_engagement:
                    topic_engagement[topic] = 0
                topic_engagement[topic] += entry.get('engagement_score', 1)
        
        # Sort by engagement
        sorted_topics = sorted(topic_engagement.items(), key=lambda x: x[1], reverse=True)
        
        return [topic for topic, _ in sorted_topics[:5]]