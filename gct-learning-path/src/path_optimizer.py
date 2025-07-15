"""
Learning Path Optimization
Uses graph search algorithms to find optimal learning sequences
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
import networkx as nx
from heapq import heappush, heappop
from collections import defaultdict
import json


@dataclass
class LearningPath:
    """Represents a complete learning path"""
    modules: List[str]
    total_coherence: float
    estimated_duration: int  # minutes
    difficulty_curve: List[float]
    metadata: Dict


class LearningPathOptimizer:
    """Optimizes learning paths using GCT coherence scores"""
    
    def __init__(self, gct_engine, learner_profile: Dict):
        """
        Initialize path optimizer
        
        Args:
            gct_engine: Initialized GCTEngine instance
            learner_profile: {
                'learner_id': str,
                'skill_level': float (0-1),
                'learning_style': str,
                'goals': List[str],
                'constraints': Dict,
                'history': Dict
            }
        """
        self.gct_engine = gct_engine
        self.learner_profile = learner_profile
        
        # Path optimization parameters
        self.params = {
            'max_path_length': 20,
            'beam_width': 10,
            'coherence_weight': 0.7,
            'duration_weight': 0.2,
            'personalization_weight': 0.1,
            'exploration_bonus': 0.05
        }
        
        # Learning constraints
        self.constraints = learner_profile.get('constraints', {})
        self.max_daily_minutes = self.constraints.get('max_daily_minutes', 120)
        self.preferred_module_duration = self.constraints.get('preferred_duration', 30)
        
        # Performance tracking
        self.path_cache = {}
        self.feedback_history = []
    
    def build_path(self, 
                  start_module: str,
                  target_module: str,
                  module_graph: nx.DiGraph,
                  module_metadata: pd.DataFrame,
                  max_steps: Optional[int] = None) -> LearningPath:
        """
        Build optimal learning path using A* search with coherence heuristic
        
        Args:
            start_module: Starting module ID
            target_module: Target module ID
            module_graph: NetworkX graph with coherence weights
            module_metadata: DataFrame with module information
            max_steps: Maximum path length
            
        Returns:
            LearningPath object
        """
        max_steps = max_steps or self.params['max_path_length']
        
        # Check cache
        cache_key = f"{start_module}->{target_module}"
        if cache_key in self.path_cache:
            cached_path = self.path_cache[cache_key]
            if len(cached_path.modules) <= max_steps:
                return cached_path
        
        # Use A* search with coherence-based heuristic
        try:
            path_modules = self._astar_search(
                start_module, target_module, module_graph, module_metadata, max_steps
            )
            
            if not path_modules:
                # Fallback to beam search if A* fails
                path_modules = self._beam_search(
                    start_module, target_module, module_graph, module_metadata, max_steps
                )
        except:
            # Final fallback: shortest path
            try:
                path_modules = nx.shortest_path(module_graph, start_module, target_module)
                if len(path_modules) > max_steps:
                    path_modules = path_modules[:max_steps]
            except:
                path_modules = [start_module]  # No path found
        
        # Calculate path metrics
        learning_path = self._create_learning_path(path_modules, module_metadata)
        
        # Cache the result
        self.path_cache[cache_key] = learning_path
        
        return learning_path
    
    def _astar_search(self,
                     start: str,
                     target: str,
                     graph: nx.DiGraph,
                     metadata: pd.DataFrame,
                     max_steps: int) -> List[str]:
        """A* search with coherence heuristic"""
        
        # Priority queue: (f_score, g_score, current_node, path)
        frontier = [(0, 0, start, [start])]
        visited = set()
        
        # Get target features for heuristic
        target_features = metadata[metadata['module_id'] == target].iloc[0]
        target_vector = target_features.get('topic_vector', np.zeros(100))
        target_difficulty = target_features.get('normalized_difficulty', 0.5)
        
        while frontier:
            f_score, g_score, current, path = heappop(frontier)
            
            if current == target:
                return path
            
            if current in visited or len(path) >= max_steps:
                continue
            
            visited.add(current)
            
            # Explore neighbors
            for neighbor in graph.successors(current):
                if neighbor not in visited:
                    # Calculate g_score (actual cost)
                    edge_data = graph[current][neighbor]
                    coherence = edge_data.get('coherence', 0.5)
                    new_g_score = g_score + (1 - coherence)  # Cost is inverse of coherence
                    
                    # Calculate heuristic h_score
                    h_score = self._calculate_heuristic(
                        neighbor, target, metadata, target_vector, target_difficulty
                    )
                    
                    # f_score = g_score + h_score
                    new_f_score = new_g_score + h_score
                    
                    heappush(frontier, (new_f_score, new_g_score, neighbor, path + [neighbor]))
        
        return []  # No path found
    
    def _calculate_heuristic(self,
                           current: str,
                           target: str,
                           metadata: pd.DataFrame,
                           target_vector: np.ndarray,
                           target_difficulty: float) -> float:
        """Calculate heuristic distance to target"""
        
        current_features = metadata[metadata['module_id'] == current].iloc[0]
        
        # Topic distance
        current_vector = current_features.get('topic_vector', np.zeros(100))
        topic_distance = np.linalg.norm(current_vector - target_vector)
        
        # Difficulty distance
        current_difficulty = current_features.get('normalized_difficulty', 0.5)
        difficulty_distance = abs(current_difficulty - target_difficulty)
        
        # Prerequisites distance (simplified)
        current_prereqs = set(current_features.get('prerequisites', []))
        target_prereqs = set(metadata[metadata['module_id'] == target].iloc[0].get('prerequisites', []))
        prereq_distance = len(target_prereqs - current_prereqs) * 0.1
        
        # Combined heuristic
        h_score = (
            topic_distance * 0.5 +
            difficulty_distance * 0.3 +
            prereq_distance * 0.2
        )
        
        return h_score
    
    def _beam_search(self,
                    start: str,
                    target: str,
                    graph: nx.DiGraph,
                    metadata: pd.DataFrame,
                    max_steps: int) -> List[str]:
        """Beam search for path finding"""
        
        beam_width = self.params['beam_width']
        
        # Initialize beam with start node
        beam = [(0, [start])]  # (score, path)
        
        for step in range(max_steps - 1):
            new_beam = []
            
            for score, path in beam:
                current = path[-1]
                
                if current == target:
                    return path
                
                # Explore successors
                for neighbor in graph.successors(current):
                    if neighbor not in path:  # Avoid cycles
                        edge_data = graph[current][neighbor]
                        coherence = edge_data.get('coherence', 0.5)
                        
                        # Calculate path score
                        new_score = score + coherence
                        
                        # Add target bonus
                        if neighbor == target:
                            new_score += 10  # Strong preference for reaching target
                        
                        new_beam.append((new_score, path + [neighbor]))
            
            # Keep top beam_width paths
            new_beam.sort(key=lambda x: x[0], reverse=True)
            beam = new_beam[:beam_width]
            
            if not beam:
                break
        
        # Return best path that gets closest to target
        best_path = max(beam, key=lambda x: x[0])[1] if beam else [start]
        
        # If we didn't reach target, try to append direct path if exists
        if best_path[-1] != target:
            try:
                extension = nx.shortest_path(graph, best_path[-1], target)
                if len(best_path) + len(extension) - 1 <= max_steps:
                    best_path.extend(extension[1:])
            except:
                pass
        
        return best_path
    
    def _create_learning_path(self, 
                            modules: List[str],
                            metadata: pd.DataFrame) -> LearningPath:
        """Create LearningPath object with calculated metrics"""
        
        if not modules:
            return LearningPath([], 0, 0, [], {})
        
        # Calculate total coherence
        total_coherence = 0
        if len(modules) > 1:
            for i in range(len(modules) - 1):
                # Get coherence from GCT engine
                from_data = metadata[metadata['module_id'] == modules[i]].iloc[0].to_dict()
                to_data = metadata[metadata['module_id'] == modules[i+1]].iloc[0].to_dict()
                
                transition_score = self.gct_engine.score_transition(
                    from_data, to_data, self.learner_profile.get('state', {})
                )
                total_coherence += transition_score['coherence']
            
            total_coherence /= (len(modules) - 1)  # Average coherence
        else:
            total_coherence = 1.0
        
        # Calculate duration
        durations = []
        difficulties = []
        
        for module_id in modules:
            module_data = metadata[metadata['module_id'] == module_id].iloc[0]
            durations.append(module_data.get('duration_minutes', 30))
            difficulties.append(module_data.get('normalized_difficulty', 0.5))
        
        total_duration = sum(durations)
        
        # Path metadata
        path_metadata = {
            'average_module_duration': np.mean(durations),
            'difficulty_variance': np.var(difficulties),
            'covers_prerequisites': self._check_prerequisites(modules, metadata),
            'personalization_score': self._calculate_path_personalization(modules, metadata)
        }
        
        return LearningPath(
            modules=modules,
            total_coherence=total_coherence,
            estimated_duration=total_duration,
            difficulty_curve=difficulties,
            metadata=path_metadata
        )
    
    def _check_prerequisites(self, modules: List[str], metadata: pd.DataFrame) -> bool:
        """Check if path covers all prerequisites"""
        covered = set(modules)
        
        for module_id in modules:
            module_data = metadata[metadata['module_id'] == module_id].iloc[0]
            prerequisites = set(module_data.get('prerequisites', []))
            
            if not prerequisites.issubset(covered):
                return False
        
        return True
    
    def _calculate_path_personalization(self, 
                                      modules: List[str],
                                      metadata: pd.DataFrame) -> float:
        """Calculate how well path matches learner profile"""
        
        if not self.learner_profile:
            return 0.5
        
        scores = []
        
        # Goal alignment
        learner_goals = set(' '.join(self.learner_profile.get('goals', [])).lower().split())
        
        for module_id in modules:
            module_data = metadata[metadata['module_id'] == module_id].iloc[0]
            module_objectives = ' '.join(module_data.get('learning_objectives', [])).lower()
            
            # Simple keyword matching
            matches = sum(1 for goal in learner_goals if goal in module_objectives)
            scores.append(min(matches / (len(learner_goals) + 1), 1.0))
        
        return np.mean(scores) if scores else 0.5
    
    def update_on_feedback(self, module_id: str, feedback: Dict):
        """
        Update optimizer based on learner feedback
        
        Args:
            module_id: Completed module
            feedback: {
                'completion_time': int,
                'difficulty_rating': float (1-5),
                'engagement_rating': float (1-5),
                'quiz_score': float (0-1),
                'would_recommend': bool
            }
        """
        # Store feedback
        self.feedback_history.append({
            'module_id': module_id,
            'timestamp': pd.Timestamp.now(),
            'feedback': feedback
        })
        
        # Update learner profile based on feedback
        self._update_learner_profile(module_id, feedback)
        
        # Adjust path optimization parameters
        self._adjust_parameters(feedback)
        
        # Clear cache to force recalculation with new data
        self.path_cache.clear()
    
    def _update_learner_profile(self, module_id: str, feedback: Dict):
        """Update learner profile based on feedback"""
        
        # Update skill level estimate
        quiz_score = feedback.get('quiz_score', 0.7)
        current_skill = self.learner_profile.get('skill_level', 0.5)
        
        # Exponential moving average
        alpha = 0.1
        new_skill = (1 - alpha) * current_skill + alpha * quiz_score
        self.learner_profile['skill_level'] = new_skill
        
        # Update learning pace preference
        actual_time = feedback.get('completion_time', 30)
        if 'constraints' not in self.learner_profile:
            self.learner_profile['constraints'] = {}
        
        preferred_duration = self.learner_profile['constraints'].get('preferred_duration', 30)
        new_preferred = (1 - alpha) * preferred_duration + alpha * actual_time
        self.learner_profile['constraints']['preferred_duration'] = new_preferred
        
        # Track performance trends
        if 'performance_history' not in self.learner_profile:
            self.learner_profile['performance_history'] = []
        
        self.learner_profile['performance_history'].append({
            'module_id': module_id,
            'score': quiz_score,
            'engagement': feedback.get('engagement_rating', 3) / 5
        })
    
    def _adjust_parameters(self, feedback: Dict):
        """Adjust optimization parameters based on feedback"""
        
        difficulty_rating = feedback.get('difficulty_rating', 3)
        engagement_rating = feedback.get('engagement_rating', 3)
        
        # If content was too difficult, reduce optimal difficulty step
        if difficulty_rating >= 4:
            self.gct_engine.params['optimal_difficulty_step'] *= 0.95
        elif difficulty_rating <= 2:
            self.gct_engine.params['optimal_difficulty_step'] *= 1.05
        
        # If engagement was low, increase personalization weight
        if engagement_rating <= 2:
            self.params['personalization_weight'] = min(
                self.params['personalization_weight'] * 1.1, 0.3
            )
    
    def generate_alternative_paths(self,
                                 start_module: str,
                                 target_module: str,
                                 module_graph: nx.DiGraph,
                                 module_metadata: pd.DataFrame,
                                 n_alternatives: int = 3) -> List[LearningPath]:
        """Generate multiple alternative learning paths"""
        
        alternatives = []
        
        # Method 1: Vary coherence weights
        original_weights = self.gct_engine.weights
        weight_variations = [
            {'psi': 0.4, 'rho': 0.1, 'q_opt': 0.3, 'flow': 0.1, 'alpha': 0.1},  # Topic-focused
            {'psi': 0.2, 'rho': 0.2, 'q_opt': 0.4, 'flow': 0.1, 'alpha': 0.1},  # Difficulty-focused
            {'psi': 0.2, 'rho': 0.2, 'q_opt': 0.2, 'flow': 0.2, 'alpha': 0.2},  # Balanced
        ]
        
        for weights in weight_variations[:n_alternatives]:
            # Temporarily change weights
            self.gct_engine.weights = type(original_weights)(**weights)
            
            # Rebuild graph with new weights
            score_matrix = self.gct_engine.score_transitions(module_metadata)
            new_graph = self.gct_engine.create_transition_graph(score_matrix)
            
            # Find path
            path = self.build_path(start_module, target_module, new_graph, module_metadata)
            alternatives.append(path)
        
        # Restore original weights
        self.gct_engine.weights = original_weights
        
        # Remove duplicates
        unique_paths = []
        seen_paths = set()
        
        for path in alternatives:
            path_tuple = tuple(path.modules)
            if path_tuple not in seen_paths:
                seen_paths.add(path_tuple)
                unique_paths.append(path)
        
        return unique_paths
    
    def explain_path_choice(self, path: LearningPath, module_metadata: pd.DataFrame) -> Dict:
        """Generate explanation for why this path was chosen"""
        
        explanation = {
            'overview': f"This learning path contains {len(path.modules)} modules with an average coherence of {path.total_coherence:.2f}",
            'duration': f"Estimated time: {path.estimated_duration} minutes ({path.estimated_duration/60:.1f} hours)",
            'difficulty_progression': self._explain_difficulty_progression(path.difficulty_curve),
            'key_transitions': [],
            'strengths': [],
            'considerations': []
        }
        
        # Analyze key transitions
        if len(path.modules) > 1:
            for i in range(len(path.modules) - 1):
                from_module = path.modules[i]
                to_module = path.modules[i + 1]
                
                from_data = module_metadata[module_metadata['module_id'] == from_module].iloc[0]
                to_data = module_metadata[module_metadata['module_id'] == to_module].iloc[0]
                
                transition = self.gct_engine.score_transition(
                    from_data.to_dict(), 
                    to_data.to_dict(),
                    self.learner_profile.get('state', {})
                )
                
                if transition['coherence'] > 0.8:
                    explanation['key_transitions'].append(
                        f"Excellent transition from '{from_module}' to '{to_module}' "
                        f"(coherence: {transition['coherence']:.2f})"
                    )
        
        # Identify strengths
        if path.total_coherence > 0.7:
            explanation['strengths'].append("High overall coherence ensures smooth learning flow")
        
        if path.metadata.get('covers_prerequisites', False):
            explanation['strengths'].append("All prerequisites are properly covered")
        
        if path.metadata.get('personalization_score', 0) > 0.6:
            explanation['strengths'].append("Well-aligned with your learning goals")
        
        # Identify considerations
        if path.estimated_duration > 300:
            explanation['considerations'].append(
                "This is a longer path - consider breaking it into multiple sessions"
            )
        
        if np.var(path.difficulty_curve) > 0.2:
            explanation['considerations'].append(
                "Difficulty varies significantly - be prepared for challenging transitions"
            )
        
        return explanation
    
    def _explain_difficulty_progression(self, difficulty_curve: List[float]) -> str:
        """Generate explanation of difficulty progression"""
        
        if not difficulty_curve:
            return "No difficulty information available"
        
        if len(difficulty_curve) == 1:
            return f"Single module with difficulty level {difficulty_curve[0]:.2f}"
        
        # Analyze trend
        trend = np.polyfit(range(len(difficulty_curve)), difficulty_curve, 1)[0]
        
        if abs(trend) < 0.01:
            return "Maintains consistent difficulty throughout"
        elif trend > 0:
            return f"Gradually increases in difficulty (slope: {trend:.3f})"
        else:
            return f"Includes easier review sections (slope: {trend:.3f})"