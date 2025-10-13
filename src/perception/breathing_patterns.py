"""
Breathing Pattern Detection
Perceives rhythm, pace, and natural pauses in human communication
"""

from typing import Dict, List, Optional, Tuple
import re
import numpy as np
from collections import Counter


class BreathingPatternDetector:
    """
    Detects and analyzes the 'breathing patterns' in text:
    - Rhythm and pace
    - Natural pauses
    - Emotional acceleration/deceleration
    - Sentence and paragraph structure
    """
    
    def __init__(self):
        self.current_rhythm = None
        self.rhythm_history = []
        
    def analyze(self, text: str) -> Dict[str, any]:
        """
        Analyze breathing patterns in text
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary containing rhythm profile
        """
        sentences = self._split_sentences(text)
        paragraphs = text.split('\n\n')
        
        rhythm_profile = {
            'pace': self._determine_pace(sentences),
            'pause_pattern': self._detect_pauses(text),
            'breath_depth': self._measure_breath_depth(sentences, paragraphs),
            'emotional_acceleration': self._detect_acceleration(sentences),
            'rhythm_type': self._classify_rhythm(text),
            'sentence_stats': self._analyze_sentence_structure(sentences),
            'punctuation_density': self._calculate_punctuation_density(text)
        }
        
        self.current_rhythm = rhythm_profile
        self.rhythm_history.append(rhythm_profile)
        
        return rhythm_profile
        
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences, handling various punctuation"""
        # Simple sentence splitter - could be enhanced with NLTK
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
        
    def _determine_pace(self, sentences: List[str]) -> str:
        """
        Determine overall pace of communication
        
        Returns: 'rapid', 'moderate', 'contemplative', 'staccato', 'flowing'
        """
        if not sentences:
            return 'moderate'
            
        # Calculate average words per sentence
        word_counts = [len(s.split()) for s in sentences]
        avg_words = np.mean(word_counts) if word_counts else 10
        
        # Calculate variance in sentence length
        variance = np.std(word_counts) if len(word_counts) > 1 else 0
        
        # Short sentences = rapid/staccato
        if avg_words < 8:
            return 'staccato' if variance < 3 else 'rapid'
        # Long sentences = contemplative/flowing  
        elif avg_words > 20:
            return 'contemplative' if variance < 5 else 'flowing'
        # Medium sentences
        else:
            return 'moderate'
            
    def _detect_pauses(self, text: str) -> Dict[str, int]:
        """Detect various pause indicators in text"""
        pauses = {
            'ellipses': len(re.findall(r'\.{2,}', text)),
            'em_dashes': text.count('—') + text.count(' - '),
            'commas': text.count(','),
            'semicolons': text.count(';'),
            'parentheses': text.count('('),
            'paragraph_breaks': text.count('\n\n')
        }
        
        # Calculate pause density
        total_words = len(text.split())
        total_pauses = sum(pauses.values())
        pauses['density'] = total_pauses / total_words if total_words > 0 else 0
        
        return pauses
        
    def _measure_breath_depth(self, sentences: List[str], paragraphs: List[str]) -> Dict[str, float]:
        """Measure the 'depth' of breaths (length of segments)"""
        sentence_lengths = [len(s.split()) for s in sentences]
        paragraph_lengths = [len(p.split()) for p in paragraphs if p.strip()]
        
        return {
            'avg_sentence_length': np.mean(sentence_lengths) if sentence_lengths else 0,
            'max_sentence_length': max(sentence_lengths) if sentence_lengths else 0,
            'min_sentence_length': min(sentence_lengths) if sentence_lengths else 0,
            'avg_paragraph_length': np.mean(paragraph_lengths) if paragraph_lengths else 0,
            'breath_variation': np.std(sentence_lengths) if len(sentence_lengths) > 1 else 0
        }
        
    def _detect_acceleration(self, sentences: List[str]) -> Dict[str, any]:
        """Detect emotional acceleration or deceleration patterns"""
        if len(sentences) < 3:
            return {'pattern': 'stable', 'intensity': 0.0}
            
        # Analyze sentence length progression
        lengths = [len(s.split()) for s in sentences]
        
        # Calculate rolling average
        window_size = min(3, len(lengths))
        rolling_avg = []
        for i in range(len(lengths) - window_size + 1):
            window = lengths[i:i + window_size]
            rolling_avg.append(np.mean(window))
            
        if len(rolling_avg) < 2:
            return {'pattern': 'stable', 'intensity': 0.0}
            
        # Detect trend
        x = np.arange(len(rolling_avg))
        coefficients = np.polyfit(x, rolling_avg, 1)
        slope = coefficients[0]
        
        # Analyze exclamation marks and questions
        exclamations = sum(s.count('!') for s in sentences)
        questions = sum(s.count('?') for s in sentences)
        
        # Determine pattern
        if abs(slope) < 0.5:
            pattern = 'stable'
        elif slope > 0.5:
            pattern = 'decelerating'  # Sentences getting longer
        else:
            pattern = 'accelerating'  # Sentences getting shorter
            
        # Check for crisis pattern (short sentences + exclamations)
        avg_length = np.mean(lengths)
        if avg_length < 10 and exclamations > len(sentences) * 0.3:
            pattern = 'crisis'
            
        intensity = min(1.0, abs(slope) / 5.0)
        
        return {
            'pattern': pattern,
            'intensity': intensity,
            'slope': slope,
            'exclamation_rate': exclamations / len(sentences) if sentences else 0,
            'question_rate': questions / len(sentences) if sentences else 0
        }
        
    def _classify_rhythm(self, text: str) -> str:
        """
        Classify the overall rhythm pattern
        
        Returns one of:
        - 'academic': Long, complex sentences with formal structure
        - 'conversational': Mixed lengths, natural flow
        - 'poetic': Rhythmic patterns, repetition
        - 'urgent': Short bursts, high punctuation
        - 'narrative': Story-like flow
        """
        sentences = self._split_sentences(text)
        
        if not sentences:
            return 'conversational'
            
        # Calculate features
        avg_length = np.mean([len(s.split()) for s in sentences])
        exclamation_rate = text.count('!') / len(sentences)
        question_rate = text.count('?') / len(sentences)
        comma_rate = text.count(',') / len(text.split())
        
        # Check for repetitive structures (poetic)
        words = text.lower().split()
        word_freq = Counter(words)
        repetition_score = sum(1 for count in word_freq.values() if count > 2) / len(word_freq)
        
        # Classification logic
        if avg_length > 25 and comma_rate > 0.1:
            return 'academic'
        elif exclamation_rate > 0.3 or (avg_length < 10 and question_rate > 0.2):
            return 'urgent'
        elif repetition_score > 0.1:
            return 'poetic'
        elif 'once upon' in text.lower() or 'and then' in text.lower():
            return 'narrative'
        else:
            return 'conversational'
            
    def _analyze_sentence_structure(self, sentences: List[str]) -> Dict[str, any]:
        """Detailed analysis of sentence structure"""
        if not sentences:
            return {'count': 0}
            
        lengths = [len(s.split()) for s in sentences]
        
        return {
            'count': len(sentences),
            'avg_length': np.mean(lengths),
            'std_length': np.std(lengths) if len(lengths) > 1 else 0,
            'shortest': min(lengths),
            'longest': max(lengths),
            'variety_score': np.std(lengths) / np.mean(lengths) if np.mean(lengths) > 0 else 0
        }
        
    def _calculate_punctuation_density(self, text: str) -> Dict[str, float]:
        """Calculate density of various punctuation marks"""
        total_chars = len(text) if text else 1
        
        return {
            'overall': len(re.findall(r'[.,;:!?—\-()"]', text)) / total_chars,
            'emotional': (text.count('!') + text.count('?')) / total_chars,
            'pausal': (text.count(',') + text.count(';') + text.count('—')) / total_chars,
            'quotation': text.count('"') / total_chars
        }
        
    def get_rhythm_profile(self) -> Optional[Dict[str, any]]:
        """Get the current rhythm profile"""
        return self.current_rhythm
        
    def suggest_response_rhythm(self, user_rhythm: Dict[str, any]) -> Dict[str, any]:
        """
        Suggest appropriate response rhythm based on user's pattern
        
        Args:
            user_rhythm: The user's detected rhythm profile
            
        Returns:
            Suggested rhythm parameters for response
        """
        suggestions = {
            'target_pace': 'moderate',
            'sentence_length': 'medium',
            'paragraph_breaks': 'normal',
            'emotional_mirroring': False
        }
        
        # Match pace for crisis/urgent situations
        if user_rhythm.get('pace') in ['rapid', 'staccato'] or \
           user_rhythm.get('rhythm_type') == 'urgent':
            suggestions['target_pace'] = 'rapid'
            suggestions['sentence_length'] = 'short'
            suggestions['emotional_mirroring'] = True
            
        # Allow space for contemplative communication
        elif user_rhythm.get('pace') == 'contemplative':
            suggestions['target_pace'] = 'contemplative'
            suggestions['sentence_length'] = 'long'
            suggestions['paragraph_breaks'] = 'frequent'
            
        # Match academic rhythm
        elif user_rhythm.get('rhythm_type') == 'academic':
            suggestions['target_pace'] = 'moderate'
            suggestions['sentence_length'] = 'complex'
            
        return suggestions
        
    def apply_rhythm_to_response(self, response: str, target_rhythm: Dict[str, any]) -> str:
        """
        Apply rhythm adjustments to a response
        
        Args:
            response: Original response text
            target_rhythm: Target rhythm parameters
            
        Returns:
            Rhythm-adjusted response
        """
        if target_rhythm.get('sentence_length') == 'short':
            # Split long sentences
            sentences = self._split_sentences(response)
            shortened = []
            for sentence in sentences:
                words = sentence.split()
                if len(words) > 15:
                    # Split into multiple sentences
                    chunks = [words[i:i+10] for i in range(0, len(words), 10)]
                    shortened.extend([' '.join(chunk) + '.' for chunk in chunks])
                else:
                    shortened.append(sentence + '.')
            response = ' '.join(shortened)
            
        elif target_rhythm.get('sentence_length') == 'long':
            # Combine short sentences where appropriate
            sentences = self._split_sentences(response)
            if len(sentences) > 1:
                combined = []
                i = 0
                while i < len(sentences):
                    if i + 1 < len(sentences) and len(sentences[i].split()) < 10:
                        # Combine with next sentence
                        combined.append(f"{sentences[i]}, and {sentences[i+1]}")
                        i += 2
                    else:
                        combined.append(sentences[i])
                        i += 1
                response = '. '.join(combined) + '.'
                
        # Add paragraph breaks for contemplative rhythm
        if target_rhythm.get('paragraph_breaks') == 'frequent':
            sentences = self._split_sentences(response)
            if len(sentences) > 3:
                # Add break every 2-3 sentences
                paragraphs = []
                for i in range(0, len(sentences), 3):
                    para = '. '.join(sentences[i:i+3])
                    if para and not para.endswith('.'):
                        para += '.'
                    paragraphs.append(para)
                response = '\n\n'.join(paragraphs)
                
        return response
        
    def get_breathing_summary(self) -> str:
        """Get a human-readable summary of current breathing pattern"""
        if not self.current_rhythm:
            return "No rhythm detected yet"
            
        rhythm = self.current_rhythm
        pace = rhythm.get('pace', 'moderate')
        rhythm_type = rhythm.get('rhythm_type', 'conversational')
        
        descriptions = {
            'rapid': 'quick and energetic',
            'staccato': 'short, punctuated bursts',
            'contemplative': 'slow and thoughtful',
            'flowing': 'smooth and continuous',
            'moderate': 'balanced and steady'
        }
        
        return f"The communication has a {descriptions.get(pace, pace)} pace with a {rhythm_type} rhythm."