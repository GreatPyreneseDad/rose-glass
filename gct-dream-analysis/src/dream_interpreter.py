"""
AI-Powered Dream Interpreter using GCT Analysis
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import re
from collections import defaultdict

class DreamInterpreter:
    """Advanced dream interpretation using GCT coherence patterns"""
    
    def __init__(self):
        self.jungian_archetypes = {
            'shadow': {
                'keywords': ['dark', 'evil', 'monster', 'enemy', 'opposite', 'hidden'],
                'meaning': 'rejected aspects of self requiring integration'
            },
            'anima_animus': {
                'keywords': ['woman', 'man', 'feminine', 'masculine', 'opposite sex'],
                'meaning': 'contrasexual aspects of the psyche'
            },
            'wise_old': {
                'keywords': ['elder', 'teacher', 'guide', 'wizard', 'sage', 'mentor'],
                'meaning': 'wisdom and guidance from the unconscious'
            },
            'great_mother': {
                'keywords': ['mother', 'nurturing', 'earth', 'womb', 'ocean'],
                'meaning': 'nurturing and creative aspects'
            },
            'hero': {
                'keywords': ['hero', 'warrior', 'fighter', 'brave', 'quest'],
                'meaning': 'ego consciousness on a journey of individuation'
            },
            'trickster': {
                'keywords': ['fool', 'joker', 'clown', 'chaos', 'mischief'],
                'meaning': 'catalyst for change and transformation'
            }
        }
        
        self.emotional_landscapes = {
            'fear': ['afraid', 'scared', 'terrified', 'anxious', 'panic', 'dread'],
            'joy': ['happy', 'joyful', 'elated', 'excited', 'bliss', 'euphoric'],
            'anger': ['angry', 'furious', 'rage', 'frustrated', 'annoyed', 'mad'],
            'sadness': ['sad', 'depressed', 'grief', 'sorrow', 'melancholy', 'despair'],
            'love': ['love', 'affection', 'care', 'compassion', 'tender', 'warm'],
            'shame': ['ashamed', 'guilty', 'embarrassed', 'humiliated', 'regret']
        }
        
        self.transformation_indicators = [
            'became', 'transformed', 'changed', 'morphed', 'evolved',
            'turned into', 'shifted', 'metamorphosis', 'transition'
        ]
        
        self.integration_words = [
            'accepted', 'embraced', 'understood', 'realized', 'integrated',
            'unified', 'whole', 'complete', 'balanced', 'harmonized'
        ]
    
    def interpret_coherence_pattern(self, coherence_history: List[float], 
                                   states_history: List[str]) -> Dict:
        """Interpret coherence patterns over time"""
        if len(coherence_history) < 3:
            return {"pattern": "insufficient_data", "interpretation": "More dreams needed for pattern analysis"}
        
        # Calculate trend
        trend = np.polyfit(range(len(coherence_history)), coherence_history, 1)[0]
        avg_coherence = np.mean(coherence_history)
        volatility = np.std(coherence_history)
        
        # Identify pattern
        if trend > 0.02 and volatility < 0.2:
            pattern = "ascending_integration"
            interpretation = "You're in a phase of psychological integration and growth. Your subconscious is successfully processing and incorporating new insights."
        elif trend < -0.02 and volatility < 0.2:
            pattern = "gradual_dissolution"
            interpretation = "Old psychological structures are dissolving to make way for new growth. This is often uncomfortable but necessary."
        elif volatility > 0.3:
            pattern = "chaotic_transformation"
            interpretation = "You're undergoing rapid psychological changes. High volatility indicates active unconscious processing."
        elif avg_coherence > 0.7 and volatility < 0.1:
            pattern = "stable_integration"
            interpretation = "Your psyche is in a stable, integrated state. This is a good time for conscious work and decision-making."
        elif avg_coherence < 0.4:
            pattern = "fragmentation"
            interpretation = "Low coherence suggests psychological fragmentation. Consider what conflicts or stressors need attention."
        else:
            pattern = "dynamic_equilibrium"
            interpretation = "Your psyche is in dynamic balance, processing experiences while maintaining stability."
        
        # Analyze state transitions
        state_transitions = self._analyze_state_transitions(states_history)
        
        return {
            "pattern": pattern,
            "interpretation": interpretation,
            "trend": "improving" if trend > 0.01 else "declining" if trend < -0.01 else "stable",
            "volatility": "high" if volatility > 0.3 else "moderate" if volatility > 0.15 else "low",
            "avg_coherence": avg_coherence,
            "dominant_transitions": state_transitions
        }
    
    def _analyze_state_transitions(self, states: List[str]) -> Dict[str, int]:
        """Analyze common state transitions"""
        transitions = defaultdict(int)
        
        for i in range(1, len(states)):
            transition = f"{states[i-1]} → {states[i]}"
            transitions[transition] += 1
        
        # Return top 3 transitions
        sorted_transitions = sorted(transitions.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_transitions[:3])
    
    def generate_symbol_insights(self, symbol_frequency: Dict[str, int],
                               symbol_evolution: Dict[str, List[Tuple[datetime, float]]]) -> List[Dict]:
        """Generate insights about recurring symbols"""
        insights = []
        
        for symbol, frequency in symbol_frequency.items():
            if frequency < 3:
                continue  # Skip rare symbols
            
            # Check if symbol matches archetypes
            archetype_match = None
            for archetype, data in self.jungian_archetypes.items():
                if any(keyword in symbol.lower() for keyword in data['keywords']):
                    archetype_match = archetype
                    break
            
            # Analyze symbol evolution if available
            evolution_pattern = "stable"
            if symbol in symbol_evolution and len(symbol_evolution[symbol]) > 2:
                coherences = [c for _, c in symbol_evolution[symbol]]
                trend = np.polyfit(range(len(coherences)), coherences, 1)[0]
                evolution_pattern = "strengthening" if trend > 0.01 else "weakening" if trend < -0.01 else "stable"
            
            insight = {
                "symbol": symbol,
                "frequency": frequency,
                "archetype": archetype_match,
                "evolution": evolution_pattern,
                "interpretation": self._interpret_symbol(symbol, frequency, archetype_match, evolution_pattern)
            }
            
            insights.append(insight)
        
        return sorted(insights, key=lambda x: x['frequency'], reverse=True)
    
    def _interpret_symbol(self, symbol: str, frequency: int, 
                         archetype: Optional[str], evolution: str) -> str:
        """Generate interpretation for a specific symbol"""
        base_interpretation = f"The symbol '{symbol}' appears {frequency} times in your dreams"
        
        if archetype:
            arch_data = self.jungian_archetypes[archetype]
            base_interpretation += f", representing {arch_data['meaning']}"
        
        if evolution == "strengthening":
            base_interpretation += ". Its increasing coherence suggests growing integration of this aspect."
        elif evolution == "weakening":
            base_interpretation += ". Its decreasing coherence may indicate resolution or avoidance."
        else:
            base_interpretation += ". Its stable presence indicates an ongoing psychological theme."
        
        # Add specific symbol interpretations
        symbol_interpretations = {
            'water': "Water often represents emotions and the unconscious. Pay attention to its state (calm, turbulent, etc.).",
            'flying': "Flying dreams often relate to freedom, transcendence, or escaping limitations.",
            'falling': "Falling can represent loss of control, fear of failure, or letting go.",
            'death': "Death in dreams usually symbolizes transformation and the end of one phase of life.",
            'animal': "Animals represent instinctual aspects of the psyche. Note the specific animal and your relationship to it.",
            'house': "Houses often represent the self or psyche. Different rooms may represent different aspects of consciousness.",
            'child': "Children in dreams can represent new beginnings, innocence, or undeveloped aspects of self."
        }
        
        if symbol.lower() in symbol_interpretations:
            base_interpretation += f" {symbol_interpretations[symbol.lower()]}"
        
        return base_interpretation
    
    def analyze_emotional_progression(self, dreams_data: List[Dict]) -> Dict:
        """Analyze emotional progression across dreams"""
        if not dreams_data:
            return {"message": "No dreams to analyze"}
        
        emotion_timeline = []
        emotion_counts = defaultdict(int)
        
        for dream in dreams_data:
            emotions = self._extract_emotions(dream.get('narrative', ''))
            dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0] if emotions else 'neutral'
            
            emotion_timeline.append({
                'date': dream.get('dream_date', datetime.now()),
                'emotion': dominant_emotion,
                'intensity': dream.get('q_raw', 0.5)
            })
            
            emotion_counts[dominant_emotion] += 1
        
        # Analyze emotional patterns
        emotional_stability = self._calculate_emotional_stability(emotion_timeline)
        dominant_emotions = sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        
        interpretation = self._interpret_emotional_pattern(dominant_emotions, emotional_stability)
        
        return {
            'dominant_emotions': dominant_emotions,
            'emotional_stability': emotional_stability,
            'timeline': emotion_timeline[-10:],  # Last 10 for visualization
            'interpretation': interpretation
        }
    
    def _extract_emotions(self, text: str) -> Dict[str, int]:
        """Extract emotions from text"""
        text_lower = text.lower()
        emotion_scores = {}
        
        for emotion, keywords in self.emotional_landscapes.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                emotion_scores[emotion] = score
        
        return emotion_scores
    
    def _calculate_emotional_stability(self, timeline: List[Dict]) -> float:
        """Calculate emotional stability score"""
        if len(timeline) < 2:
            return 0.5
        
        # Count emotion changes
        changes = 0
        for i in range(1, len(timeline)):
            if timeline[i]['emotion'] != timeline[i-1]['emotion']:
                changes += 1
        
        # Stability is inverse of change rate
        change_rate = changes / (len(timeline) - 1)
        return 1 - change_rate
    
    def _interpret_emotional_pattern(self, dominant_emotions: List[Tuple[str, int]], 
                                   stability: float) -> str:
        """Interpret emotional patterns"""
        if not dominant_emotions:
            return "No clear emotional patterns detected."
        
        primary_emotion = dominant_emotions[0][0]
        
        interpretations = {
            'fear': "Fear is dominant in your dreams, suggesting anxiety or unresolved concerns in waking life.",
            'joy': "Joy dominates your dreams, indicating positive psychological integration and life satisfaction.",
            'anger': "Anger in dreams often represents suppressed frustration or the need to assert boundaries.",
            'sadness': "Sadness in dreams may indicate grief processing or the need to accept losses.",
            'love': "Love in dreams reflects connection needs and positive relational patterns.",
            'shame': "Shame in dreams suggests self-judgment that needs compassionate attention."
        }
        
        base_interpretation = interpretations.get(primary_emotion, 
            f"{primary_emotion.capitalize()} is your dominant dream emotion.")
        
        if stability > 0.7:
            base_interpretation += " Your emotional patterns are stable, suggesting good emotional regulation."
        elif stability < 0.3:
            base_interpretation += " High emotional variability suggests active psychological processing."
        
        return base_interpretation
    
    def generate_personalized_recommendations(self, analysis_data: Dict) -> List[str]:
        """Generate personalized recommendations based on dream patterns"""
        recommendations = []
        
        # Based on coherence
        avg_coherence = analysis_data.get('avg_coherence', 0.5)
        if avg_coherence < 0.4:
            recommendations.append("Consider keeping a more detailed dream journal to improve recall and coherence.")
            recommendations.append("Practice meditation or relaxation before sleep to enhance dream clarity.")
        elif avg_coherence > 0.7:
            recommendations.append("Your high dream coherence suggests readiness for lucid dreaming practices.")
        
        # Based on dominant state
        dominant_state = analysis_data.get('dominant_state', '')
        state_recommendations = {
            'chaotic': "Try establishing a regular sleep schedule to stabilize dream patterns.",
            'emotional_processing': "Your dreams are processing intense emotions. Consider journaling or therapy.",
            'recurring_pattern': "Recurring patterns suggest unresolved issues. Reflect on what these symbols mean to you.",
            'shadow_work': "You're integrating shadow aspects. Be patient and compassionate with yourself.",
            'lucid_integration': "Continue lucid dreaming practices. Try setting intentions before sleep."
        }
        
        if dominant_state in state_recommendations:
            recommendations.append(state_recommendations[dominant_state])
        
        # Based on symbols
        top_symbols = analysis_data.get('top_symbols', [])
        if any('water' in str(s).lower() for s in top_symbols):
            recommendations.append("Water symbols suggest focus on emotional awareness and flow.")
        if any('death' in str(s).lower() or 'dying' in str(s).lower() for s in top_symbols):
            recommendations.append("Death symbols indicate transformation. Embrace change in your waking life.")
        
        # Based on trend
        if analysis_data.get('coherence_trend') == 'declining':
            recommendations.append("Coherence is declining. Ensure adequate sleep and stress management.")
        elif analysis_data.get('coherence_trend') == 'improving':
            recommendations.append("Your improving coherence shows psychological growth. Keep up your practices.")
        
        # General recommendations
        if len(recommendations) < 3:
            recommendations.extend([
                "Draw or write about significant dream images to deepen understanding.",
                "Share dreams with trusted friends to gain new perspectives.",
                "Notice emotional patterns in dreams and their connection to daily life."
            ])
        
        return recommendations[:5]  # Return top 5 recommendations
    
    def detect_psychological_themes(self, dreams_narrative: List[str]) -> Dict[str, float]:
        """Detect major psychological themes across dreams"""
        themes = {
            'individuation': 0,
            'shadow_integration': 0,
            'relationship_dynamics': 0,
            'creative_expression': 0,
            'spiritual_awakening': 0,
            'trauma_processing': 0,
            'power_dynamics': 0,
            'identity_formation': 0
        }
        
        # Keywords for each theme
        theme_keywords = {
            'individuation': ['self', 'becoming', 'whole', 'complete', 'journey', 'path', 'discover'],
            'shadow_integration': ['dark', 'shadow', 'hidden', 'rejected', 'embrace', 'accept', 'integrate'],
            'relationship_dynamics': ['love', 'family', 'friend', 'partner', 'connection', 'separation', 'together'],
            'creative_expression': ['create', 'art', 'music', 'dance', 'paint', 'write', 'build', 'design'],
            'spiritual_awakening': ['light', 'divine', 'sacred', 'spiritual', 'transcend', 'enlighten', 'cosmic'],
            'trauma_processing': ['hurt', 'pain', 'wound', 'heal', 'past', 'memory', 'escape', 'trapped'],
            'power_dynamics': ['control', 'power', 'dominate', 'submit', 'authority', 'rebel', 'free'],
            'identity_formation': ['who', 'identity', 'name', 'role', 'mask', 'true', 'false', 'authentic']
        }
        
        # Analyze each dream
        for narrative in dreams_narrative:
            text_lower = narrative.lower()
            
            # Check for transformation
            has_transformation = any(word in text_lower for word in self.transformation_indicators)
            
            # Score each theme
            for theme, keywords in theme_keywords.items():
                score = sum(1 for keyword in keywords if keyword in text_lower)
                if has_transformation and theme in ['individuation', 'shadow_integration']:
                    score *= 1.5  # Boost for transformation themes
                themes[theme] += score
        
        # Normalize scores
        total = sum(themes.values())
        if total > 0:
            themes = {k: v/total for k, v in themes.items()}
        
        # Return only significant themes
        return {k: v for k, v in themes.items() if v > 0.1}