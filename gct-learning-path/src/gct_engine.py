"""
GCT Engine for Learning Path Coherence
Calculates coherence scores for learning module transitions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.spatial.distance import cosine
import networkx as nx


@dataclass
class GCTWeights:
    """Weights for GCT coherence calculation"""
    psi: float = 0.3      # Baseline thematic coherence
    rho: float = 0.2      # Overlap/redundancy penalty
    q_opt: float = 0.2    # Ideal next-step quality
    flow: float = 0.2     # Flow adjustment for learner state
    alpha: float = 0.1    # Personalization tweak


class LearningGCTEngine:
    """
    GCT Engine adapted for learning path optimization
    
    Coherence formula for learning transitions:
    C(m1→m2) = ψ(topic_similarity) + ρ(knowledge_building) + q_opt(difficulty_progression) 
                + f(flow_maintenance) + α(personalization)
    """
    
    def __init__(self, weights: Optional[GCTWeights] = None):
        """
        Initialize GCT engine with component weights
        
        Args:
            weights: GCTWeights object or None for defaults
        """
        self.weights = weights or GCTWeights()
        
        # Learning-specific parameters
        self.params = {
            'optimal_difficulty_step': 0.1,  # Ideal difficulty increase
            'max_difficulty_jump': 0.3,      # Maximum acceptable difficulty jump
            'redundancy_threshold': 0.8,     # Topic similarity threshold for redundancy
            'flow_decay_rate': 0.1,          # How quickly flow state decays
            'personalization_strength': 0.5   # How much to weight personal preferences
        }
        
        # Component calculators
        self.component_calculators = {
            'psi': self._calculate_thematic_coherence,
            'rho': self._calculate_knowledge_building,
            'q_opt': self._calculate_difficulty_progression,
            'flow': self._calculate_flow_maintenance,
            'alpha': self._calculate_personalization
        }
    
    def score_transition(self, 
                        from_module: Dict,
                        to_module: Dict,
                        learner_state: Optional[Dict] = None,
                        context: Optional[Dict] = None) -> Dict[str, float]:
        """
        Score a single module transition
        
        Args:
            from_module: Current module metadata
            to_module: Target module metadata
            learner_state: Current learner state (performance, preferences, etc.)
            context: Additional context (time constraints, goals, etc.)
            
        Returns:
            Dictionary with overall coherence and component scores
        """
        components = {}
        
        # Calculate each component
        components['psi'] = self._calculate_thematic_coherence(from_module, to_module)
        components['rho'] = self._calculate_knowledge_building(from_module, to_module)
        components['q_opt'] = self._calculate_difficulty_progression(from_module, to_module)
        components['flow'] = self._calculate_flow_maintenance(from_module, to_module, learner_state)
        components['alpha'] = self._calculate_personalization(to_module, learner_state)
        
        # Calculate weighted coherence
        coherence = sum(
            getattr(self.weights, comp) * score 
            for comp, score in components.items()
        )
        
        # Apply context modifiers
        if context:
            coherence = self._apply_context_modifiers(coherence, context)
        
        return {
            'coherence': min(1.0, max(0.0, coherence)),
            'components': components,
            'transition_quality': self._assess_transition_quality(components)
        }
    
    def score_transitions(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute pairwise coherence scores for all module transitions
        
        Args:
            feature_df: DataFrame with module features including:
                       - module_id
                       - topic_vector
                       - normalized_difficulty
                       - cognitive_complexity
                       - prerequisites
                       
        Returns:
            DataFrame with transition scores
        """
        modules = feature_df.to_dict('records')
        n_modules = len(modules)
        
        # Initialize score matrix
        scores = np.zeros((n_modules, n_modules))
        components = {comp: np.zeros((n_modules, n_modules)) 
                     for comp in ['psi', 'rho', 'q_opt', 'flow', 'alpha']}
        
        # Calculate pairwise scores
        for i in range(n_modules):
            for j in range(n_modules):
                if i != j:  # No self-transitions
                    result = self.score_transition(modules[i], modules[j])
                    scores[i, j] = result['coherence']
                    
                    for comp, value in result['components'].items():
                        components[comp][i, j] = value
        
        # Create result DataFrame
        module_ids = feature_df['module_id'].values
        score_df = pd.DataFrame(scores, index=module_ids, columns=module_ids)
        
        # Add component DataFrames as attributes
        score_df.components = {
            comp: pd.DataFrame(matrix, index=module_ids, columns=module_ids)
            for comp, matrix in components.items()
        }
        
        return score_df
    
    def _calculate_thematic_coherence(self, from_module: Dict, to_module: Dict) -> float:
        """
        Calculate ψ - baseline thematic coherence
        
        Based on:
        - Topic similarity
        - Conceptual relatedness
        - Domain consistency
        """
        # Topic vector similarity
        if 'topic_vector' in from_module and 'topic_vector' in to_module:
            topic_similarity = 1 - cosine(from_module['topic_vector'], 
                                         to_module['topic_vector'])
        else:
            # Fallback to tag overlap
            from_tags = set(from_module.get('topic_tags', []))
            to_tags = set(to_module.get('topic_tags', []))
            
            if from_tags or to_tags:
                topic_similarity = len(from_tags.intersection(to_tags)) / len(from_tags.union(to_tags))
            else:
                topic_similarity = 0.5
        
        # Boost for explicit prerequisites
        prereq_bonus = 0.2 if from_module['module_id'] in to_module.get('prerequisites', []) else 0
        
        # Penalty for domain jumps
        from_domain = from_module.get('primary_topic', '').split('_')[0]
        to_domain = to_module.get('primary_topic', '').split('_')[0]
        domain_penalty = 0.1 if from_domain != to_domain else 0
        
        psi = topic_similarity + prereq_bonus - domain_penalty
        
        return max(0.0, min(1.0, psi))
    
    def _calculate_knowledge_building(self, from_module: Dict, to_module: Dict) -> float:
        """
        Calculate ρ - knowledge building with redundancy penalty
        
        Rewards:
        - Progressive depth
        - Complementary coverage
        
        Penalizes:
        - Excessive overlap
        - Circular dependencies
        """
        # Check for redundancy
        topic_similarity = 1 - cosine(from_module.get('topic_vector', [0]), 
                                     to_module.get('topic_vector', [0]))
        
        if topic_similarity > self.params['redundancy_threshold']:
            # High overlap - check if it's progressive
            difficulty_delta = to_module.get('normalized_difficulty', 0.5) - \
                             from_module.get('normalized_difficulty', 0.5)
            
            if difficulty_delta > 0:
                # Progressive deepening
                rho = 0.5 + difficulty_delta
            else:
                # Redundant or regressive
                rho = 0.2 * (1 - topic_similarity)
        else:
            # Complementary topics
            # Check for conceptual bridges
            shared_prereqs = set(from_module.get('prerequisites', [])).intersection(
                set(to_module.get('prerequisites', []))
            )
            
            if shared_prereqs:
                # Building on common foundation
                rho = 0.7
            else:
                # Moderate building potential
                rho = 0.4 + 0.2 * topic_similarity
        
        return max(0.0, min(1.0, rho))
    
    def _calculate_difficulty_progression(self, from_module: Dict, to_module: Dict) -> float:
        """
        Calculate q_opt - ideal difficulty progression
        
        Optimal when:
        - Difficulty increases slightly (zone of proximal development)
        - Cognitive complexity matches learner capacity
        - Time investment is appropriate
        """
        # Difficulty progression
        from_diff = from_module.get('normalized_difficulty', 0.5)
        to_diff = to_module.get('normalized_difficulty', 0.5)
        diff_delta = to_diff - from_diff
        
        # Optimal progression curve (bell curve around ideal step)
        optimal_step = self.params['optimal_difficulty_step']
        max_step = self.params['max_difficulty_jump']
        
        if diff_delta < 0:
            # Regression - sometimes necessary but not ideal
            q_diff = 0.3 + 0.2 * (1 + diff_delta)  # Less penalty for small regression
        elif diff_delta <= optimal_step:
            # Near optimal
            q_diff = 0.9 - abs(diff_delta - optimal_step) * 2
        elif diff_delta <= max_step:
            # Acceptable but challenging
            q_diff = 0.7 - (diff_delta - optimal_step) * 2
        else:
            # Too large a jump
            q_diff = 0.2
        
        # Cognitive complexity alignment
        from_cog = from_module.get('cognitive_complexity', 0.5)
        to_cog = to_module.get('cognitive_complexity', 0.5)
        cog_delta = to_cog - from_cog
        
        # Similar curve for cognitive progression
        if abs(cog_delta) < 0.2:
            q_cog = 0.8
        else:
            q_cog = 0.4 + 0.4 * (1 - min(abs(cog_delta), 1))
        
        # Time investment consideration
        from_time = from_module.get('time_investment', 0.5)
        to_time = to_module.get('time_investment', 0.5)
        
        # Prefer consistent time investment
        time_consistency = 1 - min(abs(to_time - from_time), 1)
        
        # Weighted combination
        q_opt = 0.5 * q_diff + 0.3 * q_cog + 0.2 * time_consistency
        
        return max(0.0, min(1.0, q_opt))
    
    def _calculate_flow_maintenance(self, 
                                   from_module: Dict, 
                                   to_module: Dict,
                                   learner_state: Optional[Dict] = None) -> float:
        """
        Calculate f - flow state maintenance
        
        Considers:
        - Engagement continuity
        - Momentum preservation
        - Fatigue management
        """
        if not learner_state:
            # Default flow assumption
            return 0.6
        
        # Current flow level
        current_flow = learner_state.get('flow_level', 0.5)
        
        # Engagement match
        from_engagement = from_module.get('engagement_score', 0.7)
        to_engagement = to_module.get('engagement_score', 0.7)
        engagement_continuity = 1 - abs(to_engagement - from_engagement) / 2
        
        # Learning momentum
        recent_performance = learner_state.get('recent_performance', 0.7)
        confidence_level = learner_state.get('confidence', 0.6)
        momentum = (recent_performance + confidence_level) / 2
        
        # Fatigue factor
        session_duration = learner_state.get('session_minutes', 0)
        fatigue = min(session_duration / 120, 1.0)  # Max fatigue at 2 hours
        
        # Variety bonus (prevents monotony)
        recent_topics = learner_state.get('recent_topics', [])
        to_topic = to_module.get('primary_topic', '')
        variety_bonus = 0.1 if to_topic not in recent_topics else 0
        
        # Calculate flow maintenance
        flow = (
            current_flow * 0.3 +
            engagement_continuity * 0.3 +
            momentum * 0.2 +
            (1 - fatigue) * 0.1 +
            variety_bonus +
            0.1  # Base flow
        )
        
        return max(0.0, min(1.0, flow))
    
    def _calculate_personalization(self,
                                 to_module: Dict,
                                 learner_state: Optional[Dict] = None) -> float:
        """
        Calculate α - personalization adjustment
        
        Based on:
        - Learning style match
        - Interest alignment
        - Goal relevance
        """
        if not learner_state:
            return 0.5  # Neutral personalization
        
        alpha_components = []
        
        # Learning style match
        learner_style = learner_state.get('learning_style', 'balanced')
        module_style = to_module.get('content_type', 'mixed')
        
        style_match = {
            ('visual', 'video'): 0.9,
            ('reading', 'text'): 0.9,
            ('kinesthetic', 'interactive'): 0.9,
            ('balanced', 'mixed'): 0.7
        }.get((learner_style, module_style), 0.5)
        
        alpha_components.append(style_match)
        
        # Interest alignment
        learner_interests = set(learner_state.get('interests', []))
        module_topics = set(to_module.get('topic_tags', []))
        
        if learner_interests and module_topics:
            interest_overlap = len(learner_interests.intersection(module_topics)) / \
                             len(learner_interests.union(module_topics))
            alpha_components.append(interest_overlap)
        
        # Goal relevance
        learner_goals = learner_state.get('learning_goals', [])
        module_objectives = to_module.get('learning_objectives', [])
        
        if learner_goals and module_objectives:
            # Simple keyword matching (could be enhanced with NLP)
            goal_keywords = set(' '.join(learner_goals).lower().split())
            objective_keywords = set(' '.join(module_objectives).lower().split())
            
            keyword_overlap = len(goal_keywords.intersection(objective_keywords)) / \
                            (len(goal_keywords) + 1)
            alpha_components.append(min(keyword_overlap * 2, 1.0))
        
        # Performance history with similar modules
        performance_history = learner_state.get('module_performance', {})
        similar_modules = [m for m in performance_history 
                          if any(t in module_topics for t in m.get('topics', []))]
        
        if similar_modules:
            avg_performance = np.mean([m.get('score', 0.7) for m in similar_modules])
            alpha_components.append(avg_performance)
        
        # Weight personalization by configured strength
        alpha = np.mean(alpha_components) if alpha_components else 0.5
        alpha = 0.5 + (alpha - 0.5) * self.params['personalization_strength']
        
        return max(0.0, min(1.0, alpha))
    
    def _assess_transition_quality(self, components: Dict[str, float]) -> str:
        """Assess overall transition quality based on components"""
        
        avg_score = np.mean(list(components.values()))
        min_score = min(components.values())
        
        if avg_score >= 0.8 and min_score >= 0.6:
            return "excellent"
        elif avg_score >= 0.6 and min_score >= 0.4:
            return "good"
        elif avg_score >= 0.4:
            return "acceptable"
        else:
            return "poor"
    
    def _apply_context_modifiers(self, coherence: float, context: Dict) -> float:
        """Apply contextual modifiers to coherence score"""
        
        # Time pressure modifier
        if context.get('time_constrained', False):
            # Prefer shorter modules when time constrained
            coherence *= 0.9
        
        # Learning streak modifier
        streak_days = context.get('learning_streak', 0)
        if streak_days > 7:
            # Bonus for maintaining streak
            coherence *= 1.05
        
        # Special event modifiers (e.g., exam preparation)
        if context.get('exam_prep', False):
            # Prefer review and practice modules
            coherence *= 1.1
        
        return coherence
    
    def create_transition_graph(self, 
                              score_matrix: pd.DataFrame,
                              threshold: float = 0.5) -> nx.DiGraph:
        """
        Create a directed graph of feasible transitions
        
        Args:
            score_matrix: Pairwise coherence scores
            threshold: Minimum coherence for edge inclusion
            
        Returns:
            NetworkX directed graph
        """
        G = nx.DiGraph()
        
        # Add nodes
        modules = score_matrix.index
        G.add_nodes_from(modules)
        
        # Add edges where coherence exceeds threshold
        for from_module in modules:
            for to_module in modules:
                if from_module != to_module:
                    coherence = score_matrix.loc[from_module, to_module]
                    
                    if coherence >= threshold:
                        G.add_edge(from_module, to_module, 
                                 weight=1-coherence,  # Use distance for pathfinding
                                 coherence=coherence)
        
        return G
    
    def recommend_next_modules(self,
                             current_module: str,
                             score_matrix: pd.DataFrame,
                             n_recommendations: int = 5,
                             learner_state: Optional[Dict] = None) -> List[Tuple[str, float]]:
        """
        Recommend next modules based on coherence scores
        
        Returns:
            List of (module_id, coherence_score) tuples
        """
        # Get scores for transitions from current module
        if current_module not in score_matrix.index:
            return []
        
        transition_scores = score_matrix.loc[current_module].copy()
        
        # Remove self-transition
        transition_scores = transition_scores[transition_scores.index != current_module]
        
        # Apply learner state adjustments if provided
        if learner_state:
            completed_modules = learner_state.get('completed_modules', [])
            # Remove already completed modules
            transition_scores = transition_scores[~transition_scores.index.isin(completed_modules)]
            
            # Boost modules matching current momentum
            if learner_state.get('high_performance', False):
                # Could boost more challenging modules
                pass
        
        # Sort by coherence and return top N
        recommendations = transition_scores.nlargest(n_recommendations)
        
        return [(module_id, score) for module_id, score in recommendations.items()]