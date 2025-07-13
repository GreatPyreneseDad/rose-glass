#!/usr/bin/env python3
"""
Context-Aware GCT Moderation System
Enhanced moderation with coherence drift detection and contextual understanding
"""

import asyncio
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
from dataclasses import dataclass, field
import torch
from transformers import AutoTokenizer, AutoModel
import networkx as nx
from collections import defaultdict
import logging


@dataclass
class Comment:
    """Represents a comment with metadata"""
    id: str
    author_id: str
    text: str
    timestamp: datetime
    parent_id: Optional[str] = None
    thread_id: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class Thread:
    """Represents a discussion thread"""
    id: str
    title: str
    created_at: datetime
    comments: List[Comment]
    participants: List[str]
    metadata: Dict = field(default_factory=dict)


@dataclass
class User:
    """Represents a user with history"""
    id: str
    username: str
    created_at: datetime
    comment_history: List[Comment]
    coherence_trajectory: List[float]
    network: List[str]  # Connected users
    metadata: Dict = field(default_factory=dict)


@dataclass
class ModerationDecision:
    """Moderation decision with explanation"""
    action: str  # 'approve', 'flag', 'remove', 'shadowban'
    confidence: float
    explanation: str
    coherence_analysis: Dict
    context_factors: Dict
    recommended_interventions: List[str] = field(default_factory=list)


class UserBehaviorTracker:
    """Track user behavior and coherence over time"""
    
    def __init__(self, database=None):
        self.db = database
        self.user_cache = {}
        self.coherence_window = 30  # days
        
    async def get_user_trajectory(self, user: User) -> Dict:
        """Get user's coherence trajectory and behavioral patterns"""
        # Get recent history
        recent_comments = [
            c for c in user.comment_history
            if c.timestamp > datetime.now() - timedelta(days=self.coherence_window)
        ]
        
        if len(recent_comments) < 5:
            return {
                'trajectory_type': 'insufficient_data',
                'coherence_trend': 0.0,
                'volatility': 0.0,
                'baseline_coherence': 0.5
            }
        
        # Calculate trajectory metrics
        coherences = user.coherence_trajectory[-len(recent_comments):]
        
        # Trend (linear regression slope)
        x = np.arange(len(coherences))
        trend = np.polyfit(x, coherences, 1)[0]
        
        # Volatility
        volatility = np.std(coherences)
        
        # Baseline (median)
        baseline = np.median(coherences)
        
        # Classify trajectory
        trajectory_type = self._classify_trajectory(trend, volatility, baseline)
        
        return {
            'trajectory_type': trajectory_type,
            'coherence_trend': trend,
            'volatility': volatility,
            'baseline_coherence': baseline,
            'recent_coherences': coherences
        }
    
    def _classify_trajectory(self, trend: float, volatility: float, baseline: float) -> str:
        """Classify user trajectory pattern"""
        if trend > 0.01 and volatility < 0.1:
            return 'improving_stable'
        elif trend > 0.01 and volatility >= 0.1:
            return 'improving_volatile'
        elif trend < -0.01 and volatility < 0.1:
            return 'declining_stable'
        elif trend < -0.01 and volatility >= 0.1:
            return 'declining_volatile'
        elif volatility > 0.2:
            return 'chaotic'
        else:
            return 'stable'


class CoherenceAnalyzer:
    """Analyze coherence in conversational context"""
    
    def __init__(self, gct_engine):
        self.gct_engine = gct_engine
        self.context_window = 5  # Comments to consider for context
        
    def analyze_thread(self, thread: Thread) -> Dict:
        """Analyze coherence flow in a thread"""
        if len(thread.comments) < 2:
            return {
                'thread_coherence': 0.5,
                'coherence_flow': [],
                'disruption_points': []
            }
        
        # Calculate coherence for each comment
        coherences = []
        for comment in thread.comments:
            coherence = self.gct_engine.calculate_coherence_from_text(comment.text)
            coherences.append({
                'comment_id': comment.id,
                'author_id': comment.author_id,
                'coherence': coherence,
                'timestamp': comment.timestamp
            })
        
        # Analyze flow
        coherence_values = [c['coherence'] for c in coherences]
        thread_coherence = np.mean(coherence_values)
        
        # Detect disruption points (sudden drops)
        disruption_points = []
        for i in range(1, len(coherence_values)):
            if coherence_values[i] < coherence_values[i-1] - 0.3:
                disruption_points.append(i)
        
        return {
            'thread_coherence': thread_coherence,
            'coherence_flow': coherences,
            'disruption_points': disruption_points,
            'coherence_variance': np.var(coherence_values)
        }


class ContextAwareGCTModerator:
    """Enhanced moderation with deep contextual understanding"""
    
    def __init__(self, model_name: str = "microsoft/deberta-v3-base"):
        # Transformer model for semantic understanding
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        
        # Components
        self.coherence_analyzer = CoherenceAnalyzer(self)
        self.behavior_tracker = UserBehaviorTracker()
        
        # Thresholds
        self.coherence_thresholds = {
            'critical_low': 0.2,
            'warning_low': 0.35,
            'healthy': 0.5
        }
        
        self.drift_thresholds = {
            'rapid': 0.3,    # Coherence change in single comment
            'gradual': 0.2   # Coherence change over thread
        }
        
        self.logger = logging.getLogger(__name__)
    
    async def analyze_with_context(self, 
                                  comment: Comment,
                                  thread: Thread,
                                  user: User) -> ModerationDecision:
        """
        Contextual moderation using coherence drift detection
        
        Args:
            comment: The comment to moderate
            thread: The thread context
            user: The user posting the comment
            
        Returns:
            ModerationDecision with detailed analysis
        """
        # 1. Analyze thread coherence
        thread_analysis = self.coherence_analyzer.analyze_thread(thread)
        
        # 2. Get user behavior trajectory
        user_trajectory = await self.behavior_tracker.get_user_trajectory(user)
        
        # 3. Calculate comment coherence
        comment_coherence = self.calculate_coherence_from_text(comment.text)
        
        # 4. Detect coherence drift
        drift_analysis = self.calculate_coherence_drift(
            comment_coherence,
            thread_analysis,
            user_trajectory
        )
        
        # 5. Check for coordinated behavior
        coordination_score = await self.detect_coordination(
            comment,
            thread,
            user
        )
        
        # 6. Analyze semantic toxicity with context
        toxicity_analysis = await self.analyze_contextual_toxicity(
            comment,
            thread
        )
        
        # 7. Generate decision
        decision = self.generate_moderation_decision(
            comment_coherence,
            drift_analysis,
            coordination_score,
            toxicity_analysis,
            user_trajectory
        )
        
        return decision
    
    def calculate_coherence_from_text(self, text: str) -> float:
        """Calculate coherence using enhanced NLP"""
        # This would use the enhanced NLP extractor
        # Simplified for example
        from src.enhanced_nlp_extractor import EnhancedNLPExtractor
        
        extractor = EnhancedNLPExtractor()
        psi = extractor.extract_psi_enhanced(text)
        rho = extractor.extract_rho_enhanced(text)
        q = extractor.extract_q_enhanced(text)
        f = extractor.extract_f_enhanced(text)
        
        # Simple coherence calculation
        coherence = (psi + rho + q + f) / 4
        
        return coherence
    
    def calculate_coherence_drift(self,
                                comment_coherence: float,
                                thread_analysis: Dict,
                                user_trajectory: Dict) -> Dict:
        """Detect various types of coherence drift"""
        
        # Drift from thread baseline
        thread_drift = abs(comment_coherence - thread_analysis['thread_coherence'])
        
        # Drift from user baseline
        user_drift = abs(comment_coherence - user_trajectory['baseline_coherence'])
        
        # Sudden drift (from previous comment in thread)
        sudden_drift = 0.0
        if thread_analysis['coherence_flow']:
            last_coherence = thread_analysis['coherence_flow'][-1]['coherence']
            sudden_drift = abs(comment_coherence - last_coherence)
        
        # Classify drift type
        drift_type = 'normal'
        if sudden_drift > self.drift_thresholds['rapid']:
            drift_type = 'sudden_shift'
        elif thread_drift > self.drift_thresholds['gradual'] and user_drift > self.drift_thresholds['gradual']:
            drift_type = 'anomalous'
        elif user_trajectory['trajectory_type'] == 'declining_volatile' and comment_coherence < 0.3:
            drift_type = 'deteriorating'
        
        return {
            'drift_type': drift_type,
            'thread_drift': thread_drift,
            'user_drift': user_drift,
            'sudden_drift': sudden_drift,
            'is_anomalous': drift_type != 'normal'
        }
    
    async def detect_coordination(self,
                                comment: Comment,
                                thread: Thread,
                                user: User) -> float:
        """Detect coordinated inauthentic behavior"""
        coordination_signals = []
        
        # 1. Temporal clustering
        user_comments_in_thread = [
            c for c in thread.comments 
            if c.author_id in user.network
        ]
        
        if len(user_comments_in_thread) > 3:
            # Check if network members posted in quick succession
            timestamps = [c.timestamp for c in user_comments_in_thread]
            time_diffs = [
                (timestamps[i+1] - timestamps[i]).total_seconds() 
                for i in range(len(timestamps)-1)
            ]
            
            if time_diffs and np.mean(time_diffs) < 300:  # 5 minutes average
                coordination_signals.append(0.3)
        
        # 2. Content similarity
        if user_comments_in_thread:
            similarities = []
            comment_embedding = await self._get_embedding(comment.text)
            
            for other_comment in user_comments_in_thread[-5:]:  # Last 5 comments
                other_embedding = await self._get_embedding(other_comment.text)
                similarity = self._cosine_similarity(comment_embedding, other_embedding)
                similarities.append(similarity)
            
            if similarities and np.mean(similarities) > 0.8:
                coordination_signals.append(0.4)
        
        # 3. Coherence synchronization
        network_coherences = []
        for network_user_id in user.network[:10]:  # Check up to 10 network members
            network_comments = [
                c for c in thread.comments 
                if c.author_id == network_user_id
            ]
            if network_comments:
                network_coherence = np.mean([
                    self.calculate_coherence_from_text(c.text) 
                    for c in network_comments[-3:]
                ])
                network_coherences.append(network_coherence)
        
        if network_coherences and np.std(network_coherences) < 0.1:
            coordination_signals.append(0.3)
        
        # Aggregate coordination score
        coordination_score = sum(coordination_signals)
        
        return min(coordination_score, 1.0)
    
    async def analyze_contextual_toxicity(self,
                                        comment: Comment,
                                        thread: Thread) -> Dict:
        """Analyze toxicity considering context"""
        # Get comment embedding
        comment_embedding = await self._get_embedding(comment.text)
        
        # Get context embeddings (previous comments)
        context_comments = thread.comments[-self.coherence_analyzer.context_window:]
        context_embeddings = [
            await self._get_embedding(c.text) 
            for c in context_comments
        ]
        
        # Calculate semantic shift
        semantic_coherence = 1.0
        if context_embeddings:
            similarities = [
                self._cosine_similarity(comment_embedding, ctx_emb)
                for ctx_emb in context_embeddings
            ]
            semantic_coherence = np.mean(similarities)
        
        # Detect adversarial patterns
        adversarial_score = 0.0
        
        # Check for topic hijacking (low semantic coherence + high confidence)
        if semantic_coherence < 0.3 and len(comment.text.split()) > 50:
            adversarial_score += 0.3
        
        # Check for baiting (questions with negative emotion)
        if comment.text.count('?') > 2 and self._detect_negative_emotion(comment.text):
            adversarial_score += 0.2
        
        # Check for gaslighting patterns
        gaslighting_phrases = [
            "you're imagining", "that never happened", "you're overreacting",
            "you're too sensitive", "you're crazy"
        ]
        if any(phrase in comment.text.lower() for phrase in gaslighting_phrases):
            adversarial_score += 0.4
        
        return {
            'semantic_coherence': semantic_coherence,
            'adversarial_score': adversarial_score,
            'is_toxic': adversarial_score > 0.5,
            'toxicity_type': self._classify_toxicity(adversarial_score, semantic_coherence)
        }
    
    def generate_moderation_decision(self,
                                   comment_coherence: float,
                                   drift_analysis: Dict,
                                   coordination_score: float,
                                   toxicity_analysis: Dict,
                                   user_trajectory: Dict) -> ModerationDecision:
        """Generate moderation decision based on all factors"""
        
        # Decision logic
        action = 'approve'
        confidence = 0.9
        factors = []
        interventions = []
        
        # Check coherence breakdown
        if comment_coherence < self.coherence_thresholds['critical_low']:
            action = 'remove'
            confidence = 0.95
            factors.append('coherence_breakdown')
            interventions.append('suggest_user_break')
        
        # Check drift patterns
        elif drift_analysis['drift_type'] == 'deteriorating':
            action = 'flag'
            confidence = 0.8
            factors.append('user_deterioration')
            interventions.append('wellness_check')
        
        # Check coordination
        elif coordination_score > 0.7:
            action = 'shadowban'
            confidence = 0.85
            factors.append('coordinated_behavior')
            interventions.append('network_investigation')
        
        # Check toxicity
        elif toxicity_analysis['is_toxic']:
            action = 'remove'
            confidence = 0.9
            factors.append(toxicity_analysis['toxicity_type'])
            interventions.append('warning_message')
        
        # Check anomalous but not harmful
        elif drift_analysis['is_anomalous'] and comment_coherence > 0.5:
            action = 'flag'
            confidence = 0.7
            factors.append('positive_anomaly')
            interventions.append('monitor_thread')
        
        # Generate explanation
        explanation = self._generate_explanation(
            action, factors, comment_coherence, drift_analysis
        )
        
        return ModerationDecision(
            action=action,
            confidence=confidence,
            explanation=explanation,
            coherence_analysis={
                'comment_coherence': comment_coherence,
                'drift_analysis': drift_analysis,
                'user_trajectory': user_trajectory
            },
            context_factors={
                'coordination_score': coordination_score,
                'toxicity_analysis': toxicity_analysis
            },
            recommended_interventions=interventions
        )
    
    async def _get_embedding(self, text: str) -> np.ndarray:
        """Get text embedding using transformer model"""
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            max_length=512, 
            truncation=True,
            padding=True
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use CLS token embedding
            embedding = outputs.last_hidden_state[:, 0, :].numpy().squeeze()
        
        return embedding
    
    def _cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate cosine similarity between embeddings"""
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    
    def _detect_negative_emotion(self, text: str) -> bool:
        """Simple negative emotion detection"""
        negative_words = {
            'hate', 'angry', 'furious', 'disgusted', 'horrible',
            'terrible', 'awful', 'stupid', 'idiotic', 'pathetic'
        }
        
        text_lower = text.lower()
        return any(word in text_lower for word in negative_words)
    
    def _classify_toxicity(self, adversarial_score: float, semantic_coherence: float) -> str:
        """Classify type of toxicity"""
        if adversarial_score > 0.7:
            return 'severe_adversarial'
        elif adversarial_score > 0.5 and semantic_coherence < 0.3:
            return 'topic_hijacking'
        elif adversarial_score > 0.5:
            return 'baiting'
        else:
            return 'mild_adversarial'
    
    def _generate_explanation(self,
                            action: str,
                            factors: List[str],
                            coherence: float,
                            drift_analysis: Dict) -> str:
        """Generate human-readable explanation"""
        
        explanations = {
            'coherence_breakdown': f"Comment shows severe coherence breakdown ({coherence:.2f})",
            'user_deterioration': "User showing concerning decline in coherence",
            'coordinated_behavior': "Suspected coordinated inauthentic behavior detected",
            'topic_hijacking': "Comment appears to hijack thread topic",
            'baiting': "Comment contains baiting or provocative content",
            'positive_anomaly': "Unusual but potentially valuable contribution"
        }
        
        explanation_parts = [explanations.get(f, f) for f in factors]
        
        if drift_analysis['is_anomalous']:
            explanation_parts.append(
                f"Coherence drift detected: {drift_analysis['drift_type']}"
            )
        
        return f"{action.capitalize()} - " + "; ".join(explanation_parts)


async def example_usage():
    """Example of using the context-aware moderator"""
    moderator = ContextAwareGCTModerator()
    
    # Create example data
    comment = Comment(
        id="c123",
        author_id="u456",
        text="I completely disagree with everything said here. You're all wrong!",
        timestamp=datetime.now()
    )
    
    thread = Thread(
        id="t789",
        title="Discussion on climate policy",
        created_at=datetime.now() - timedelta(hours=2),
        comments=[],  # Would contain previous comments
        participants=["u456", "u789", "u012"]
    )
    
    user = User(
        id="u456",
        username="example_user",
        created_at=datetime.now() - timedelta(days=180),
        comment_history=[],  # Would contain history
        coherence_trajectory=[0.5, 0.48, 0.45, 0.4, 0.35],
        network=["u789", "u012"]
    )
    
    # Analyze
    decision = await moderator.analyze_with_context(comment, thread, user)
    
    print(f"Decision: {decision.action}")
    print(f"Confidence: {decision.confidence:.2f}")
    print(f"Explanation: {decision.explanation}")
    print(f"Recommended actions: {', '.join(decision.recommended_interventions)}")


if __name__ == "__main__":
    asyncio.run(example_usage())