"""
Interaction Quality Monitor for Rose Glass Framework
===================================================

Focuses on improving communication quality rather than profiling users.
Measures mutual understanding and engagement without identity assumptions.

Author: Christopher MacGregor bin Joseph
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from collections import deque
import json


@dataclass
class InteractionMetrics:
    """Metrics for a single interaction exchange"""
    timestamp: datetime
    mutual_understanding: float
    engagement_level: float
    satisfaction_signals: float
    confusion_indicators: float
    response_relevance: float
    communication_efficiency: float
    
    @property
    def overall_quality(self) -> float:
        """Calculate overall interaction quality (0-1)"""
        positive_factors = (
            self.mutual_understanding * 0.3 +
            self.engagement_level * 0.2 +
            self.satisfaction_signals * 0.2 +
            self.response_relevance * 0.2 +
            self.communication_efficiency * 0.1
        )
        
        # Confusion reduces quality
        quality = positive_factors * (1 - self.confusion_indicators * 0.5)
        return max(0, min(1, quality))


@dataclass
class ConversationQuality:
    """Overall conversation quality tracking"""
    metrics_history: deque
    quality_trend: float
    avg_understanding: float
    avg_engagement: float
    total_exchanges: int
    successful_clarifications: int
    unresolved_confusions: int
    
    def get_summary(self) -> Dict:
        """Get conversation quality summary"""
        return {
            'overall_quality': self.calculate_overall_quality(),
            'quality_trend': self.quality_trend,
            'understanding_level': self.avg_understanding,
            'engagement_level': self.avg_engagement,
            'success_rate': self.calculate_success_rate(),
            'needs_attention': self.identify_attention_areas()
        }
    
    def calculate_overall_quality(self) -> float:
        """Calculate overall conversation quality"""
        if not self.metrics_history:
            return 0.5
            
        recent_metrics = list(self.metrics_history)[-10:]
        avg_quality = np.mean([m.overall_quality for m in recent_metrics])
        
        # Factor in success rate
        success_rate = self.calculate_success_rate()
        
        return avg_quality * 0.7 + success_rate * 0.3
    
    def calculate_success_rate(self) -> float:
        """Calculate clarification success rate"""
        total_clarifications = self.successful_clarifications + self.unresolved_confusions
        
        if total_clarifications == 0:
            return 1.0  # No clarifications needed is good
            
        return self.successful_clarifications / total_clarifications
    
    def identify_attention_areas(self) -> List[str]:
        """Identify areas needing attention"""
        areas = []
        
        if self.avg_understanding < 0.5:
            areas.append("Low mutual understanding - consider clarifying communication")
            
        if self.avg_engagement < 0.3:
            areas.append("Low engagement - topic or style may need adjustment")
            
        if self.quality_trend < -0.1:
            areas.append("Declining quality - conversation may need reset")
            
        if self.unresolved_confusions > 2:
            areas.append("Multiple unresolved confusions - address directly")
            
        return areas


class InteractionQualityMonitor:
    """
    Monitor and improve synthetic-organic communication quality.
    Focus on effectiveness, not user profiling.
    """
    
    def __init__(self, history_size: int = 100):
        self.conversation_quality = ConversationQuality(
            metrics_history=deque(maxlen=history_size),
            quality_trend=0.0,
            avg_understanding=0.5,
            avg_engagement=0.5,
            total_exchanges=0,
            successful_clarifications=0,
            unresolved_confusions=0
        )
        
        # Understanding indicators
        self.understanding_markers = {
            'positive': ['i see', 'makes sense', 'got it', 'understand', 'clear', 
                        'ah', 'oh i see', 'right', 'exactly', 'yes'],
            'negative': ['confused', "don't understand", 'what do you mean', 
                        'unclear', 'lost me', 'huh', "doesn't make sense", 
                        'can you explain', 'what?']
        }
        
        # Engagement indicators
        self.engagement_markers = {
            'high': ['tell me more', 'interesting', 'go on', 'what else', 
                    'how about', 'and then', 'really?', 'wow'],
            'low': ['ok', 'sure', 'whatever', 'i guess', 'if you say so', 
                   'mhm', 'uh huh']
        }
        
        # Satisfaction indicators
        self.satisfaction_markers = {
            'positive': ['thanks', 'helpful', 'great', 'perfect', 'awesome', 
                        'appreciated', 'good', 'nice', 'excellent'],
            'negative': ['unhelpful', 'frustrating', 'annoying', 'useless', 
                        'waste of time', 'not what i wanted']
        }
    
    def measure_interaction_quality(self, 
                                  human_text: str,
                                  synthetic_response: str,
                                  previous_exchange: Optional[Tuple[str, str]] = None) -> InteractionMetrics:
        """
        Measure quality of a single interaction exchange.
        
        Args:
            human_text: The human's message
            synthetic_response: The AI's response
            previous_exchange: Previous (human, ai) tuple if available
            
        Returns:
            InteractionMetrics for this exchange
        """
        metrics = InteractionMetrics(
            timestamp=datetime.now(),
            mutual_understanding=self._calculate_understanding_indicators(human_text, synthetic_response),
            engagement_level=self._measure_engagement_depth(human_text),
            satisfaction_signals=self._detect_satisfaction_feedback(human_text),
            confusion_indicators=self._detect_clarification_needs(human_text, synthetic_response),
            response_relevance=self._measure_response_relevance(human_text, synthetic_response, previous_exchange),
            communication_efficiency=self._calculate_communication_efficiency(human_text, synthetic_response)
        )
        
        # Update conversation tracking
        self._update_conversation_quality(metrics)
        
        return metrics
    
    def _calculate_understanding_indicators(self, human_text: str, ai_response: str) -> float:
        """
        Calculate mutual understanding based on explicit markers.
        No assumptions about what understanding "should" look like.
        """
        human_lower = human_text.lower()
        
        # Check for understanding markers in human response
        positive_count = sum(1 for marker in self.understanding_markers['positive'] 
                           if marker in human_lower)
        negative_count = sum(1 for marker in self.understanding_markers['negative'] 
                           if marker in human_lower)
        
        # Also check if AI asked clarifying questions
        ai_questions = ai_response.count('?')
        clarification_phrases = ['could you clarify', 'do you mean', 'to make sure i understand']
        ai_clarifications = sum(1 for phrase in clarification_phrases if phrase in ai_response.lower())
        
        # Calculate understanding score
        if positive_count + negative_count == 0:
            # No explicit markers - neutral
            base_score = 0.5
        else:
            base_score = positive_count / (positive_count + negative_count)
        
        # Adjust based on AI's clarification attempts
        if ai_clarifications > 0:
            # AI is actively trying to understand
            base_score = min(1.0, base_score + 0.2)
        
        return base_score
    
    def _measure_engagement_depth(self, human_text: str) -> float:
        """
        Measure engagement without judging communication style.
        Some people engage deeply with few words, others with many.
        """
        text_lower = human_text.lower()
        
        # Engagement through markers
        high_engagement = sum(1 for marker in self.engagement_markers['high'] 
                            if marker in text_lower)
        low_engagement = sum(1 for marker in self.engagement_markers['low'] 
                           if marker in text_lower)
        
        # Engagement through questions
        questions = human_text.count('?')
        
        # Engagement through length (normalized)
        word_count = len(human_text.split())
        length_engagement = min(word_count / 50, 1.0)  # 50 words = high engagement
        
        # Combine factors
        marker_score = 0.5  # neutral default
        if high_engagement + low_engagement > 0:
            marker_score = high_engagement / (high_engagement + low_engagement)
        
        question_score = min(questions / 3, 1.0)  # 3 questions = high engagement
        
        # Weighted combination
        engagement = (
            marker_score * 0.4 +
            question_score * 0.3 +
            length_engagement * 0.3
        )
        
        return engagement
    
    def _detect_satisfaction_feedback(self, human_text: str) -> float:
        """Detect satisfaction signals from explicit feedback"""
        text_lower = human_text.lower()
        
        positive = sum(1 for marker in self.satisfaction_markers['positive'] 
                      if marker in text_lower)
        negative = sum(1 for marker in self.satisfaction_markers['negative'] 
                      if marker in text_lower)
        
        if positive + negative == 0:
            return 0.5  # Neutral
            
        return positive / (positive + negative)
    
    def _detect_clarification_needs(self, human_text: str, ai_response: str) -> float:
        """
        Detect if clarification is needed based on confusion markers.
        High score = high confusion.
        """
        confusion_score = 0.0
        
        # Direct confusion markers
        confusion_phrases = ["i don't understand", "what do you mean", "confused",
                           "that doesn't make sense", "can you explain", "huh?",
                           "i'm lost", "what?", "come again?"]
        
        human_lower = human_text.lower()
        for phrase in confusion_phrases:
            if phrase in human_lower:
                confusion_score += 0.3
        
        # Repetition of question (if we detect the same question pattern)
        if '?' in human_text and ai_response:
            # Simple heuristic: if human asks similar question after AI response
            human_words = set(human_text.lower().split())
            if len(human_words) < 20:  # Short message
                confusion_score += 0.2
        
        # Multiple question marks
        if human_text.count('?') > 2:
            confusion_score += 0.1
        
        return min(confusion_score, 1.0)
    
    def _measure_response_relevance(self, human_text: str, ai_response: str, 
                                  previous_exchange: Optional[Tuple[str, str]]) -> float:
        """
        Measure how relevant the AI response is to the human input.
        Based on topic overlap and question answering.
        """
        # Extract key terms from human text
        human_words = set(human_text.lower().split())
        ai_words = set(ai_response.lower().split())
        
        # Remove common words
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 
                       'to', 'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was',
                       'are', 'been', 'be', 'have', 'has', 'had', 'do', 'does',
                       'did', 'will', 'would', 'should', 'could', 'may', 'might'}
        
        human_content = human_words - common_words
        ai_content = ai_words - common_words
        
        # Calculate overlap
        if not human_content:
            return 0.5  # Can't determine relevance
            
        overlap = len(human_content & ai_content)
        relevance = min(overlap / len(human_content), 1.0)
        
        # Check if question was answered
        if '?' in human_text:
            # AI should provide some form of answer
            if '.' in ai_response or '!' in ai_response:
                relevance = min(relevance + 0.2, 1.0)
        
        return relevance
    
    def _calculate_communication_efficiency(self, human_text: str, ai_response: str) -> float:
        """
        Measure communication efficiency without penalizing style preferences.
        Efficiency = achieving understanding with appropriate effort.
        """
        human_length = len(human_text.split())
        ai_length = len(ai_response.split())
        
        # Very short or very long responses may be inefficient
        if ai_length < 5:
            length_efficiency = 0.5  # Too brief?
        elif ai_length > 200:
            length_efficiency = 0.5  # Too verbose?
        else:
            # Good range
            length_efficiency = 1.0
        
        # Response proportionality
        if human_length > 0:
            proportion = ai_length / human_length
            if 0.5 <= proportion <= 3.0:
                proportion_efficiency = 1.0
            else:
                proportion_efficiency = 0.5
        else:
            proportion_efficiency = 0.5
        
        return (length_efficiency + proportion_efficiency) / 2
    
    def _update_conversation_quality(self, metrics: InteractionMetrics):
        """Update overall conversation quality tracking"""
        self.conversation_quality.metrics_history.append(metrics)
        self.conversation_quality.total_exchanges += 1
        
        # Update running averages
        recent_metrics = list(self.conversation_quality.metrics_history)[-20:]
        
        self.conversation_quality.avg_understanding = np.mean(
            [m.mutual_understanding for m in recent_metrics]
        )
        self.conversation_quality.avg_engagement = np.mean(
            [m.engagement_level for m in recent_metrics]
        )
        
        # Calculate quality trend
        if len(recent_metrics) >= 5:
            recent_quality = [m.overall_quality for m in recent_metrics[-5:]]
            older_quality = [m.overall_quality for m in recent_metrics[-10:-5]]
            
            if older_quality:
                self.conversation_quality.quality_trend = (
                    np.mean(recent_quality) - np.mean(older_quality)
                )
        
        # Track clarification success
        if metrics.confusion_indicators > 0.5:
            # Check if next exchange shows understanding
            if len(self.conversation_quality.metrics_history) > 1:
                previous = self.conversation_quality.metrics_history[-2]
                if previous.confusion_indicators > 0.5 and metrics.mutual_understanding > 0.7:
                    self.conversation_quality.successful_clarifications += 1
                else:
                    self.conversation_quality.unresolved_confusions += 1
    
    def get_quality_report(self) -> Dict:
        """Get comprehensive quality report for current conversation"""
        return self.conversation_quality.get_summary()
    
    def suggest_improvements(self) -> List[str]:
        """
        Suggest improvements based on quality metrics.
        Focus on communication effectiveness, not user change.
        """
        suggestions = []
        report = self.get_quality_report()
        
        if report['understanding_level'] < 0.5:
            suggestions.append(
                "Consider using simpler language or breaking down complex concepts"
            )
            suggestions.append(
                "Ask clarifying questions to ensure mutual understanding"
            )
        
        if report['engagement_level'] < 0.3:
            suggestions.append(
                "Try asking open-ended questions to encourage engagement"
            )
            suggestions.append(
                "Match the user's energy and interest level"
            )
        
        if report['quality_trend'] < -0.1:
            suggestions.append(
                "Consider acknowledging any confusion and starting fresh"
            )
            suggestions.append(
                "Summarize key points to realign the conversation"
            )
        
        # Check for specific problem patterns
        recent_metrics = list(self.conversation_quality.metrics_history)[-5:]
        if recent_metrics:
            avg_relevance = np.mean([m.response_relevance for m in recent_metrics])
            if avg_relevance < 0.5:
                suggestions.append(
                    "Focus responses more directly on user's questions and topics"
                )
        
        return suggestions
    
    def reset_conversation(self):
        """Reset quality tracking for new conversation"""
        self.conversation_quality = ConversationQuality(
            metrics_history=deque(maxlen=100),
            quality_trend=0.0,
            avg_understanding=0.5,
            avg_engagement=0.5,
            total_exchanges=0,
            successful_clarifications=0,
            unresolved_confusions=0
        )
    
    def export_metrics(self, filepath: str):
        """Export conversation metrics for analysis"""
        metrics_data = {
            'conversation_summary': self.get_quality_report(),
            'total_exchanges': self.conversation_quality.total_exchanges,
            'improvement_suggestions': self.suggest_improvements(),
            'detailed_metrics': [
                {
                    'timestamp': m.timestamp.isoformat(),
                    'understanding': m.mutual_understanding,
                    'engagement': m.engagement_level,
                    'satisfaction': m.satisfaction_signals,
                    'confusion': m.confusion_indicators,
                    'relevance': m.response_relevance,
                    'efficiency': m.communication_efficiency,
                    'overall_quality': m.overall_quality
                }
                for m in self.conversation_quality.metrics_history
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(metrics_data, f, indent=2)


def demonstrate_quality_monitoring():
    """Demonstrate the interaction quality monitoring system"""
    monitor = InteractionQualityMonitor()
    
    # Simulate conversation exchanges
    exchanges = [
        # Good understanding
        ("How do I implement a binary search tree?",
         "A binary search tree is a data structure where each node has at most two children. The left child contains values less than the parent, and the right child contains values greater. Would you like to see the implementation code?"),
        
        # Positive engagement
        ("Yes! Show me the code please. This is really helpful!",
         "Here's a Python implementation of a binary search tree with insert and search methods: [code example]. The key insight is maintaining the ordering property during insertion."),
        
        # Some confusion
        ("Wait, I don't understand the recursive part. What's happening there?",
         "Let me clarify the recursion. When we insert a value, we compare it with the current node. If it's smaller, we go left; if larger, we go right. We repeat this process until we find an empty spot."),
        
        # Understanding achieved
        ("Oh I see! So it keeps going down until it finds where to put the new value. That makes sense now.",
         "Exactly! You've got it. The recursion naturally maintains the tree's ordering property. Would you like to explore how deletion works too?"),
        
        # Continued engagement
        ("Sure, but first - how do we handle duplicates?",
         "Great question! There are several approaches for handling duplicates in a BST: 1) Reject duplicates, 2) Always insert to the left or right, or 3) Keep a count at each node. Which approach interests you most?")
    ]
    
    print("=== Interaction Quality Monitoring Demo ===\n")
    
    previous = None
    for i, (human, ai) in enumerate(exchanges):
        print(f"\nExchange {i+1}:")
        print(f"Human: {human}")
        print(f"AI: {ai}")
        
        # Measure quality
        metrics = monitor.measure_interaction_quality(human, ai, previous)
        
        print(f"\nMetrics:")
        print(f"  Understanding: {metrics.mutual_understanding:.2f}")
        print(f"  Engagement: {metrics.engagement_level:.2f}")
        print(f"  Satisfaction: {metrics.satisfaction_signals:.2f}")
        print(f"  Confusion: {metrics.confusion_indicators:.2f}")
        print(f"  Overall Quality: {metrics.overall_quality:.2f}")
        
        previous = (human, ai)
    
    # Get final report
    print("\n=== Conversation Quality Report ===")
    report = monitor.get_quality_report()
    print(json.dumps(report, indent=2))
    
    print("\n=== Improvement Suggestions ===")
    suggestions = monitor.suggest_improvements()
    for suggestion in suggestions:
        print(f"- {suggestion}")


if __name__ == "__main__":
    demonstrate_quality_monitoring()