"""
Communication Style Adapter for Rose Glass Framework
====================================================

Ethical adaptation to communication patterns without identity profiling.
Focuses on HOW things are expressed, not WHO is expressing them.

Core Principles:
- No demographic profiling
- No assumption-based categorization
- Focus on expressed needs, not inferred characteristics
- Respect for privacy and dignity
- Pattern recognition without identity attribution

Author: Christopher MacGregor bin Joseph
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import re
from collections import Counter


@dataclass
class CommunicationPattern:
    """Observable communication patterns without identity assumptions"""
    expression_style: Dict[str, float]
    cognitive_patterns: Dict[str, float]
    emotional_expression: Dict[str, float]
    interaction_goals: Dict[str, str]
    conversation_dynamics: Dict[str, float]
    timestamp: datetime


class CommunicationStyleAdapter:
    """
    Adapt to observable communication patterns without profiling.
    Enhances synthetic-organic translation by matching communication styles.
    """
    
    def __init__(self):
        # Communication style indicators (not identity markers)
        self.style_indicators = {
            'formality': {
                'formal': ['therefore', 'furthermore', 'nevertheless', 'consequently'],
                'informal': ['like', 'kinda', 'gonna', 'yeah', 'okay']
            },
            'directness': {
                'direct': ['i want', 'i need', 'please do', 'must', 'will you'],
                'indirect': ['perhaps', 'maybe', 'it might be', 'could possibly']
            },
            'detail': {
                'high': ['specifically', 'precisely', 'exactly', 'in particular'],
                'low': ['basically', 'simply', 'just', 'overall']
            }
        }
        
    def extract_communication_patterns(self, text: str, 
                                     interaction_history: Optional[List[str]] = None) -> CommunicationPattern:
        """
        Extract communication style patterns without demographic profiling.
        Focus on HOW things are expressed, not WHO is expressing them.
        """
        return CommunicationPattern(
            expression_style=self.analyze_communication_style(text),
            cognitive_patterns=self.detect_reasoning_approaches(text),
            emotional_expression=self.measure_affect_patterns(text),
            interaction_goals=self.identify_stated_needs(text),
            conversation_dynamics=self.analyze_dialogue_patterns(text, interaction_history),
            timestamp=datetime.now()
        )
    
    def analyze_communication_style(self, text: str) -> Dict[str, float]:
        """
        Identify communication preferences without cultural/social assumptions.
        """
        words = text.lower().split()
        sentences = re.split(r'[.!?]+', text)
        
        return {
            'directness_level': self._measure_explicit_vs_implicit(text, words),
            'formality_register': self._detect_formality_preference(words),
            'detail_orientation': self._measure_elaboration_level(sentences),
            'abstraction_level': self._detect_concrete_vs_abstract(words)
        }
    
    def _measure_explicit_vs_implicit(self, text: str, words: List[str]) -> float:
        """Measure how direct vs indirect the communication is"""
        direct_count = sum(1 for phrase in self.style_indicators['directness']['direct'] 
                          if phrase in text.lower())
        indirect_count = sum(1 for phrase in self.style_indicators['directness']['indirect'] 
                            if phrase in text.lower())
        
        total = direct_count + indirect_count
        if total == 0:
            return 0.5  # Neutral
        
        return direct_count / total
    
    def _detect_formality_preference(self, words: List[str]) -> float:
        """Detect formality level without making cultural assumptions"""
        formal_count = sum(1 for word in words 
                          if word in self.style_indicators['formality']['formal'])
        informal_count = sum(1 for word in words 
                            if word in self.style_indicators['formality']['informal'])
        
        total = formal_count + informal_count
        if total == 0:
            return 0.5  # Neutral
        
        return formal_count / total
    
    def _measure_elaboration_level(self, sentences: List[str]) -> float:
        """Measure how much detail is provided"""
        if not sentences:
            return 0.5
            
        avg_length = np.mean([len(s.split()) for s in sentences if s])
        # Normalize to 0-1 scale (assuming 30 words is high detail)
        return min(avg_length / 30, 1.0)
    
    def _detect_concrete_vs_abstract(self, words: List[str]) -> float:
        """Detect preference for concrete vs abstract language"""
        # Simple heuristic: longer words tend to be more abstract
        avg_word_length = np.mean([len(w) for w in words]) if words else 0
        # Normalize (assuming 8 letters is highly abstract)
        return min(avg_word_length / 8, 1.0)
    
    def detect_reasoning_approaches(self, text: str) -> Dict[str, float]:
        """
        Detect reasoning patterns without judging validity.
        All reasoning styles are valid in their contexts.
        """
        patterns = {
            'linear': self._detect_linear_reasoning(text),
            'narrative': self._detect_narrative_reasoning(text),
            'associative': self._detect_associative_reasoning(text),
            'analytical': self._detect_analytical_reasoning(text)
        }
        
        # Normalize so they sum to 1
        total = sum(patterns.values())
        if total > 0:
            patterns = {k: v/total for k, v in patterns.items()}
            
        return patterns
    
    def _detect_linear_reasoning(self, text: str) -> float:
        """Detect step-by-step logical progression"""
        linear_markers = ['first', 'second', 'then', 'therefore', 'thus', 'consequently']
        return sum(1 for marker in linear_markers if marker in text.lower()) / len(linear_markers)
    
    def _detect_narrative_reasoning(self, text: str) -> float:
        """Detect story-based reasoning"""
        narrative_markers = ['once', 'when i', 'remember', 'story', 'example', 'like when']
        return sum(1 for marker in narrative_markers if marker in text.lower()) / len(narrative_markers)
    
    def _detect_associative_reasoning(self, text: str) -> float:
        """Detect connection-based reasoning"""
        associative_markers = ['reminds me', 'similar to', 'like', 'connected', 'related']
        return sum(1 for marker in associative_markers if marker in text.lower()) / len(associative_markers)
    
    def _detect_analytical_reasoning(self, text: str) -> float:
        """Detect analytical reasoning"""
        analytical_markers = ['analyze', 'examine', 'consider', 'evaluate', 'assess']
        return sum(1 for marker in analytical_markers if marker in text.lower()) / len(analytical_markers)
    
    def measure_affect_patterns(self, text: str) -> Dict[str, float]:
        """
        Measure emotional expression patterns without judging appropriateness.
        """
        # Simple emotion indicators (could be enhanced with sentiment analysis)
        emotion_words = {
            'positive': ['happy', 'joy', 'love', 'excited', 'grateful', 'hope'],
            'negative': ['sad', 'angry', 'frustrated', 'worried', 'fear', 'upset'],
            'neutral': ['think', 'believe', 'understand', 'know', 'see', 'consider']
        }
        
        words = text.lower().split()
        emotion_counts = {
            emotion: sum(1 for word in words if word in indicators)
            for emotion, indicators in emotion_words.items()
        }
        
        total_emotion = sum(emotion_counts.values())
        
        return {
            'emotional_expressiveness': total_emotion / max(len(words), 1),
            'emotional_valence': self._calculate_valence(emotion_counts),
            'comfort_with_emotion': self._assess_emotional_comfort(text)
        }
    
    def _calculate_valence(self, emotion_counts: Dict[str, int]) -> float:
        """Calculate overall emotional valence (-1 to 1)"""
        positive = emotion_counts.get('positive', 0)
        negative = emotion_counts.get('negative', 0)
        total = positive + negative
        
        if total == 0:
            return 0.0
            
        return (positive - negative) / total
    
    def _assess_emotional_comfort(self, text: str) -> float:
        """Assess comfort with emotional expression"""
        emotional_phrases = ['i feel', 'makes me', 'i\'m feeling', 'emotionally']
        count = sum(1 for phrase in emotional_phrases if phrase in text.lower())
        return min(count / 3, 1.0)  # Normalize
    
    def identify_stated_needs(self, text: str) -> Dict[str, str]:
        """
        Focus only on explicitly expressed needs and goals.
        No inference of hidden agendas or unstated motivations.
        """
        needs = {
            'expressed_goals': self._extract_stated_objectives(text),
            'requested_assistance': self._identify_help_requests(text),
            'information_seeking': self._detect_questions(text),
            'emotional_expression': self._identify_feeling_statements(text)
        }
        
        return {k: v for k, v in needs.items() if v}  # Only return non-empty needs
    
    def _extract_stated_objectives(self, text: str) -> str:
        """Extract explicitly stated goals"""
        goal_patterns = [
            r'i want to (\w+.*?)(?:\.|$)',
            r'i need to (\w+.*?)(?:\.|$)',
            r'my goal is (\w+.*?)(?:\.|$)',
            r'i\'m trying to (\w+.*?)(?:\.|$)'
        ]
        
        for pattern in goal_patterns:
            match = re.search(pattern, text.lower())
            if match:
                return match.group(1)
        return ""
    
    def _identify_help_requests(self, text: str) -> str:
        """Identify explicit requests for help"""
        help_patterns = [
            r'can you help(.*?)(?:\?|$)',
            r'please help(.*?)(?:\?|$)',
            r'i need help with(.*?)(?:\?|$)',
            r'could you(.*?)(?:\?|$)'
        ]
        
        for pattern in help_patterns:
            match = re.search(pattern, text.lower())
            if match:
                return match.group(1).strip()
        return ""
    
    def _detect_questions(self, text: str) -> str:
        """Detect information-seeking questions"""
        questions = re.findall(r'([^.!?]*\?)', text)
        return '; '.join(questions) if questions else ""
    
    def _identify_feeling_statements(self, text: str) -> str:
        """Identify explicit feeling expressions"""
        feeling_patterns = [
            r'i feel (\w+)',
            r'i\'m feeling (\w+)',
            r'makes me feel (\w+)',
            r'i\'m (\w+) about'
        ]
        
        feelings = []
        for pattern in feeling_patterns:
            matches = re.findall(pattern, text.lower())
            feelings.extend(matches)
            
        return ', '.join(feelings) if feelings else ""
    
    def analyze_dialogue_patterns(self, text: str, 
                                history: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Analyze conversation dynamics without making assumptions.
        """
        return {
            'turn_taking_preference': self._assess_turn_length(text, history),
            'responsiveness': self._measure_responsiveness(text, history),
            'topic_consistency': self._measure_topic_consistency(text, history),
            'engagement_level': self._measure_engagement(text)
        }
    
    def _assess_turn_length(self, text: str, history: Optional[List[str]]) -> float:
        """Assess preference for long vs short conversational turns"""
        current_length = len(text.split())
        
        if not history:
            return 0.5  # Neutral
            
        avg_history_length = np.mean([len(h.split()) for h in history[-5:]])
        
        # Compare current to average
        if current_length > avg_history_length * 1.5:
            return 0.8  # Prefers longer turns
        elif current_length < avg_history_length * 0.5:
            return 0.2  # Prefers shorter turns
        else:
            return 0.5  # Balanced
    
    def _measure_responsiveness(self, text: str, history: Optional[List[str]]) -> float:
        """Measure how responsive the text is to previous context"""
        if not history:
            return 0.5
            
        # Simple check: does text reference previous content?
        previous = ' '.join(history[-3:]) if len(history) >= 3 else ' '.join(history)
        previous_words = set(previous.lower().split())
        current_words = set(text.lower().split())
        
        overlap = len(previous_words & current_words)
        return min(overlap / 10, 1.0)  # Normalize
    
    def _measure_topic_consistency(self, text: str, history: Optional[List[str]]) -> float:
        """Measure consistency in topic focus"""
        # This is a simplified version - could use topic modeling
        return 0.7  # Placeholder
    
    def _measure_engagement(self, text: str) -> float:
        """Measure engagement level from text"""
        engagement_markers = ['!', '?', 'really', 'very', 'definitely', 'absolutely']
        count = sum(1 for marker in engagement_markers if marker in text)
        return min(count / 5, 1.0)
    
    def calibrate_variables_for_communication(self, patterns: CommunicationPattern) -> Dict[str, float]:
        """
        Adjust GCT variables based on communication patterns, not identity.
        Returns adjustment factors for each variable.
        """
        adjustments = {
            'psi_adjustment': self._calibrate_for_expression_style(patterns.expression_style),
            'rho_adjustment': self._calibrate_for_reasoning_pattern(patterns.cognitive_patterns),
            'q_adjustment': self._calibrate_for_emotional_expression(patterns.emotional_expression),
            'f_adjustment': self._calibrate_for_interaction_style(patterns.conversation_dynamics)
        }
        
        return adjustments
    
    def _calibrate_for_expression_style(self, style: Dict[str, float]) -> float:
        """
        Adjust consistency detection based on communication style.
        Some styles naturally appear less "consistent" without being incoherent.
        """
        adjustment = 0.0
        
        # Indirect communication shouldn't be penalized
        if style['directness_level'] < 0.3:
            adjustment -= 0.1
            
        # High abstraction may appear less consistent
        if style['abstraction_level'] > 0.7:
            adjustment -= 0.05
            
        return adjustment
    
    def _calibrate_for_reasoning_pattern(self, cognitive_patterns: Dict[str, float]) -> float:
        """
        Recognize different valid reasoning approaches.
        All styles have their own wisdom.
        """
        adjustment = 0.0
        
        # Narrative reasoning carries wisdom differently
        if cognitive_patterns.get('narrative', 0) > 0.5:
            adjustment += 0.1
            
        # Associative reasoning may seem less "accumulated"
        if cognitive_patterns.get('associative', 0) > 0.5:
            adjustment += 0.05
            
        return adjustment
    
    def _calibrate_for_emotional_expression(self, emotional_patterns: Dict[str, float]) -> float:
        """
        Adjust for different comfort levels with emotional expression.
        """
        adjustment = 0.0
        
        # Low emotional expression doesn't mean low moral activation
        if emotional_patterns['comfort_with_emotion'] < 0.3:
            adjustment += 0.1
            
        return adjustment
    
    def _calibrate_for_interaction_style(self, dynamics: Dict[str, float]) -> float:
        """
        Adjust for different interaction preferences.
        """
        adjustment = 0.0
        
        # Short turns don't mean low social connection
        if dynamics.get('turn_taking_preference', 0.5) < 0.3:
            adjustment += 0.05
            
        return adjustment
    
    def adapt_response_style(self, base_response: str, 
                           patterns: CommunicationPattern) -> str:
        """
        Adapt synthetic response to match communication preferences.
        Focus on accessibility and clarity, not assumptions about identity.
        """
        adapted = base_response
        
        # Match formality level
        formality = patterns.expression_style.get('formality_register', 0.5)
        if formality > 0.7:
            adapted = self._increase_formality(adapted)
        elif formality < 0.3:
            adapted = self._decrease_formality(adapted)
        
        # Match detail level
        detail = patterns.expression_style.get('detail_orientation', 0.5)
        if detail > 0.7:
            adapted = self._add_elaboration(adapted)
        elif detail < 0.3:
            adapted = self._simplify_message(adapted)
        
        # Respect emotional expression preferences
        emotion_comfort = patterns.emotional_expression.get('comfort_with_emotion', 0.5)
        if emotion_comfort < 0.3:
            adapted = self._reduce_emotional_content(adapted)
        elif emotion_comfort > 0.7:
            adapted = self._add_emotional_acknowledgment(adapted)
        
        return adapted
    
    def _increase_formality(self, text: str) -> str:
        """Make language more formal"""
        replacements = {
            "can't": "cannot",
            "won't": "will not",
            "it's": "it is",
            "let's": "let us",
            " ok ": " acceptable ",
            " okay ": " acceptable "
        }
        
        formal_text = text
        for informal, formal in replacements.items():
            formal_text = formal_text.replace(informal, formal)
            
        return formal_text
    
    def _decrease_formality(self, text: str) -> str:
        """Make language more casual"""
        # This is simplified - in practice would be more sophisticated
        return text.replace("Therefore", "So").replace("However", "But")
    
    def _add_elaboration(self, text: str) -> str:
        """Add more detail to response"""
        # In practice, this would intelligently expand on key points
        return f"{text} (I can provide more specific details if helpful.)"
    
    def _simplify_message(self, text: str) -> str:
        """Simplify the message"""
        # In practice, this would intelligently simplify
        sentences = text.split('. ')
        if len(sentences) > 2:
            # Keep only most important sentences
            return '. '.join(sentences[:2]) + '.'
        return text
    
    def _reduce_emotional_content(self, text: str) -> str:
        """Reduce emotional language"""
        emotional_words = ['feel', 'feeling', 'emotion', 'heart', 'soul']
        neutral_text = text
        
        for word in emotional_words:
            neutral_text = neutral_text.replace(word, '')
            
        return neutral_text.strip()
    
    def _add_emotional_acknowledgment(self, text: str) -> str:
        """Add emotional validation"""
        if 'understand' in text.lower():
            return text.replace('understand', 'understand how you feel about')
        return f"I appreciate you sharing this. {text}"