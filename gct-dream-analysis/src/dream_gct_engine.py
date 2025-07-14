"""
Dream GCT Engine - Adapting Grounded Coherence Theory for Dream Analysis
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import re
from collections import defaultdict

# Import core GCT components from parent project
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../gct-market-sentiment/src'))
from gct_engine import GCTParameters, GCTResult


@dataclass
class DreamVariables:
    """Variables extracted from dream narratives"""
    psi: float      # Clarity/vividness of dream imagery
    rho: float      # Symbolic depth and archetypal wisdom
    q_raw: float    # Raw emotional intensity
    f: float        # Social/relational content
    timestamp: datetime
    
    # Dream-specific additions
    lucidity: float  # Level of conscious awareness in dream
    recurring: float # Strength of recurring elements
    shadow: float    # Integration of shadow aspects


@dataclass 
class DreamSymbol:
    """Represents a symbol or archetype in dreams"""
    name: str
    frequency: int
    emotional_charge: float
    contexts: List[str]
    evolution: List[Tuple[datetime, float]]  # Track symbol coherence over time


class DreamGCTEngine:
    """Specialized GCT engine for dream analysis"""
    
    def __init__(self):
        self.params = GCTParameters()
        # Adjust parameters for dream context
        self.params.km = 0.4  # Higher saturation for symbolic content
        self.params.coupling_strength = 0.25  # Stronger coupling in dreams
        
        # Dream-specific parameters
        self.lucidity_weight = 0.2
        self.shadow_integration_factor = 1.5
        self.symbol_database = self._initialize_symbols()
        
        # History for pattern analysis
        self.dream_history = []
        self.symbol_evolution = defaultdict(list)
        
    def _initialize_symbols(self) -> Dict[str, DreamSymbol]:
        """Initialize common dream symbols and archetypes"""
        symbols = {
            'water': DreamSymbol('water', 0, 0.0, [], []),
            'flying': DreamSymbol('flying', 0, 0.0, [], []),
            'falling': DreamSymbol('falling', 0, 0.0, [], []),
            'death': DreamSymbol('death', 0, 0.0, [], []),
            'birth': DreamSymbol('birth', 0, 0.0, [], []),
            'animal': DreamSymbol('animal', 0, 0.0, [], []),
            'shadow_figure': DreamSymbol('shadow_figure', 0, 0.0, [], []),
            'light': DreamSymbol('light', 0, 0.0, [], []),
            'maze': DreamSymbol('maze', 0, 0.0, [], []),
            'mirror': DreamSymbol('mirror', 0, 0.0, [], []),
        }
        return symbols
    
    def extract_dream_variables(self, dream_text: str, 
                              sleep_quality: Optional[float] = None,
                              lucid: bool = False) -> DreamVariables:
        """Extract GCT variables from dream narrative"""
        
        # Clarity (psi) - based on descriptive detail
        detail_words = ['vivid', 'clear', 'bright', 'detailed', 'sharp', 'distinct']
        vague_words = ['blurry', 'unclear', 'foggy', 'vague', 'hazy', 'confused']
        
        psi = self._calculate_clarity(dream_text, detail_words, vague_words)
        
        # Wisdom (rho) - archetypal and symbolic content
        rho = self._calculate_symbolic_depth(dream_text)
        
        # Emotional charge (q) - intensity of emotions
        q_raw = self._calculate_emotional_intensity(dream_text)
        
        # Social (f) - presence of others and relationships
        f = self._calculate_social_content(dream_text)
        
        # Lucidity factor
        lucidity = 1.0 if lucid else self._detect_lucidity(dream_text)
        
        # Recurring elements
        recurring = self._calculate_recurrence_strength(dream_text)
        
        # Shadow integration
        shadow = self._calculate_shadow_integration(dream_text)
        
        return DreamVariables(
            psi=psi,
            rho=rho,
            q_raw=q_raw,
            f=f,
            timestamp=datetime.now(),
            lucidity=lucidity,
            recurring=recurring,
            shadow=shadow
        )
    
    def _calculate_clarity(self, text: str, positive: List[str], negative: List[str]) -> float:
        """Calculate dream clarity/vividness"""
        text_lower = text.lower()
        
        positive_count = sum(1 for word in positive if word in text_lower)
        negative_count = sum(1 for word in negative if word in text_lower)
        
        # Length factor - longer descriptions tend to be clearer
        length_factor = min(len(text.split()) / 100, 1.0)
        
        # Sensory details
        sensory_words = ['saw', 'heard', 'felt', 'touched', 'smelled', 'tasted', 
                        'color', 'sound', 'texture', 'warm', 'cold']
        sensory_count = sum(1 for word in sensory_words if word in text_lower)
        
        clarity = (positive_count - negative_count + sensory_count/2 + length_factor) / 10
        return max(0, min(1, clarity))
    
    def _calculate_symbolic_depth(self, text: str) -> float:
        """Calculate archetypal and symbolic content"""
        text_lower = text.lower()
        
        # Check for archetypal symbols
        symbol_count = 0
        for symbol_name, symbol in self.symbol_database.items():
            if symbol_name in text_lower:
                symbol_count += 1
                symbol.frequency += 1
                symbol.contexts.append(text[:100])  # Store context snippet
        
        # Transformation words
        transform_words = ['became', 'transformed', 'changed', 'morphed', 'shifted',
                          'evolved', 'turned into', 'metamorphosis']
        transform_count = sum(1 for word in transform_words if word in text_lower)
        
        # Depth indicators
        depth_words = ['realized', 'understood', 'meaning', 'symbol', 'represented',
                      'signified', 'deeper', 'beneath', 'hidden']
        depth_count = sum(1 for word in depth_words if word in text_lower)
        
        rho = (symbol_count * 2 + transform_count + depth_count) / 10
        return max(0, min(1, rho))
    
    def _calculate_emotional_intensity(self, text: str) -> float:
        """Calculate emotional charge in dream"""
        text_lower = text.lower()
        
        # Intense emotions
        intense_emotions = {
            'terror': 3, 'horror': 3, 'panic': 3, 'dread': 2,
            'ecstasy': 3, 'bliss': 3, 'euphoria': 3,
            'rage': 3, 'fury': 3, 'anguish': 3,
            'love': 2, 'hate': 2, 'fear': 2, 'joy': 2
        }
        
        # Mild emotions
        mild_emotions = {
            'worried': 1, 'concerned': 1, 'happy': 1, 'sad': 1,
            'anxious': 1, 'calm': 0.5, 'peaceful': 0.5
        }
        
        intensity = 0
        for emotion, weight in intense_emotions.items():
            intensity += text_lower.count(emotion) * weight
        
        for emotion, weight in mild_emotions.items():
            intensity += text_lower.count(emotion) * weight
        
        # Exclamation marks indicate intensity
        intensity += text.count('!') * 0.5
        
        return max(0, min(1, intensity / 10))
    
    def _calculate_social_content(self, text: str) -> float:
        """Calculate social and relational content"""
        text_lower = text.lower()
        
        # Social indicators
        social_words = ['people', 'person', 'friend', 'family', 'mother', 'father',
                       'brother', 'sister', 'child', 'stranger', 'crowd', 'group',
                       'together', 'alone', 'isolated', 'connected']
        
        social_count = sum(1 for word in social_words if word in text_lower)
        
        # Pronouns indicating others
        other_pronouns = len(re.findall(r'\b(he|she|they|them|we|us)\b', text_lower))
        
        # Interaction verbs
        interaction_verbs = ['talked', 'said', 'told', 'asked', 'helped', 'hugged',
                           'fought', 'argued', 'kissed', 'touched']
        interaction_count = sum(1 for verb in interaction_verbs if verb in text_lower)
        
        f = (social_count + other_pronouns/10 + interaction_count) / 10
        return max(0, min(1, f))
    
    def _detect_lucidity(self, text: str) -> float:
        """Detect level of lucidity in dream"""
        text_lower = text.lower()
        
        lucidity_markers = ['realized i was dreaming', 'knew it was a dream',
                          'became lucid', 'conscious that', 'aware that',
                          'decided to', 'chose to', 'controlled']
        
        lucidity_score = sum(1 for marker in lucidity_markers if marker in text_lower)
        return min(1.0, lucidity_score / 3)
    
    def _calculate_recurrence_strength(self, text: str) -> float:
        """Calculate strength of recurring elements"""
        if len(self.dream_history) < 2:
            return 0.0
        
        # Check for recurring symbols
        current_symbols = set()
        text_lower = text.lower()
        for symbol in self.symbol_database:
            if symbol in text_lower:
                current_symbols.add(symbol)
        
        # Compare with recent dreams
        recurring_count = 0
        for past_dream in self.dream_history[-10:]:  # Last 10 dreams
            past_symbols = past_dream.get('symbols', set())
            recurring_count += len(current_symbols.intersection(past_symbols))
        
        return min(1.0, recurring_count / 10)
    
    def _calculate_shadow_integration(self, text: str) -> float:
        """Calculate integration of shadow aspects"""
        text_lower = text.lower()
        
        # Shadow indicators
        shadow_words = ['dark', 'shadow', 'hidden', 'rejected', 'denied',
                       'monster', 'demon', 'evil', 'enemy', 'opposite']
        shadow_count = sum(1 for word in shadow_words if word in text_lower)
        
        # Integration indicators
        integration_words = ['accepted', 'embraced', 'understood', 'befriended',
                           'merged', 'integrated', 'reconciled', 'faced']
        integration_count = sum(1 for word in integration_words if word in text_lower)
        
        # Higher score for integration of shadow
        if shadow_count > 0 and integration_count > 0:
            return min(1.0, (shadow_count + integration_count * 2) / 5)
        elif shadow_count > 0:
            return min(0.5, shadow_count / 5)
        else:
            return 0.0
    
    def calculate_dream_coherence(self, variables: DreamVariables) -> DreamGCTResult:
        """Calculate dream coherence using enhanced GCT formula"""
        
        # Optimize emotional charge for dream context
        q_opt = self._optimize_emotional_charge(variables.q_raw, variables.shadow)
        
        # Enhanced dream coherence formula
        base_coherence = (
            variables.psi + 
            (variables.rho * variables.psi) + 
            q_opt + 
            (variables.f * variables.psi)
        )
        
        # Dream-specific enhancements
        lucidity_bonus = variables.lucidity * self.lucidity_weight
        shadow_bonus = variables.shadow * self.shadow_integration_factor * variables.psi
        recurring_factor = 1 + (variables.recurring * 0.2)  # Recurring dreams have higher significance
        
        coherence = (base_coherence + lucidity_bonus + shadow_bonus) * recurring_factor
        
        # Normalize
        coherence = coherence / (1 + variables.rho + variables.f + self.lucidity_weight + self.shadow_integration_factor)
        
        # Calculate derivatives if we have history
        dc_dt, d2c_dt2 = self._calculate_derivatives(coherence)
        
        # Determine dream state
        dream_state = self._classify_dream_state(coherence, dc_dt, variables)
        
        components = {
            'base_psi': variables.psi,
            'wisdom_contrib': variables.rho * variables.psi,
            'emotion_contrib': q_opt,
            'social_contrib': variables.f * variables.psi,
            'lucidity_bonus': lucidity_bonus,
            'shadow_bonus': shadow_bonus,
            'recurring_factor': recurring_factor
        }
        
        result = DreamGCTResult(
            coherence=coherence,
            q_opt=q_opt,
            dc_dt=dc_dt,
            d2c_dt2=d2c_dt2,
            dream_state=dream_state,
            components=components,
            insights=self._generate_insights(variables, coherence, dream_state)
        )
        
        # Store in history
        self._update_history(variables, result)
        
        return result
    
    def _optimize_emotional_charge(self, q_raw: float, shadow: float) -> float:
        """Optimize emotional charge for dream context"""
        # In dreams, high emotional charge with shadow integration is positive
        if shadow > 0.5:
            return q_raw * (1 + shadow)  # Amplify when shadow is integrated
        else:
            return q_raw / (1 + self.params.ki * q_raw)  # Standard optimization
    
    def _calculate_derivatives(self, coherence: float) -> Tuple[float, float]:
        """Calculate coherence derivatives"""
        if len(self.dream_history) < 2:
            return 0.0, 0.0
        
        # Get recent coherence values
        recent_coherences = [d['coherence'] for d in self.dream_history[-5:]]
        recent_coherences.append(coherence)
        
        # Simple finite differences
        if len(recent_coherences) >= 2:
            dc_dt = recent_coherences[-1] - recent_coherences[-2]
        else:
            dc_dt = 0.0
        
        if len(recent_coherences) >= 3:
            d2c_dt2 = (recent_coherences[-1] - 2*recent_coherences[-2] + recent_coherences[-3])
        else:
            d2c_dt2 = 0.0
        
        return dc_dt, d2c_dt2
    
    def _classify_dream_state(self, coherence: float, dc_dt: float, 
                            variables: DreamVariables) -> str:
        """Classify the dream state based on coherence patterns"""
        
        if variables.lucidity > 0.7:
            return "lucid_integration"
        elif variables.shadow > 0.7:
            return "shadow_work" 
        elif coherence > 0.8:
            if dc_dt > 0.1:
                return "breakthrough"
            else:
                return "integrated"
        elif coherence > 0.6:
            if dc_dt > 0.05:
                return "processing"
            elif dc_dt < -0.05:
                return "fragmenting"
            else:
                return "stable"
        elif coherence > 0.4:
            if variables.recurring > 0.5:
                return "recurring_pattern"
            else:
                return "exploring"
        else:
            if variables.q_raw > 0.7:
                return "emotional_processing"
            else:
                return "chaotic"
    
    def _generate_insights(self, variables: DreamVariables, coherence: float, 
                         dream_state: str) -> List[str]:
        """Generate insights based on dream analysis"""
        insights = []
        
        # Coherence-based insights
        if coherence > 0.8:
            insights.append("High dream coherence indicates strong psychological integration.")
        elif coherence < 0.3:
            insights.append("Low coherence suggests internal conflicts requiring attention.")
        
        # State-based insights
        state_insights = {
            "lucid_integration": "Lucid dreaming shows high conscious awareness and control.",
            "shadow_work": "You're actively integrating rejected aspects of yourself.",
            "breakthrough": "A psychological breakthrough is occurring in your subconscious.",
            "integrated": "Your psyche is in a state of harmony and balance.",
            "processing": "Your mind is actively processing recent experiences.",
            "fragmenting": "Some psychological defenses may be breaking down.",
            "stable": "Your subconscious is in a steady, balanced state.",
            "recurring_pattern": "Recurring elements suggest unresolved issues.",
            "exploring": "You're exploring new psychological territory.",
            "emotional_processing": "Intense emotions are being processed and released.",
            "chaotic": "Your subconscious is in a state of reorganization."
        }
        
        if dream_state in state_insights:
            insights.append(state_insights[dream_state])
        
        # Variable-specific insights
        if variables.psi < 0.3:
            insights.append("Low clarity may indicate stress or poor sleep quality.")
        elif variables.psi > 0.8:
            insights.append("Exceptional dream vividness suggests heightened awareness.")
        
        if variables.rho > 0.7:
            insights.append("Rich symbolic content indicates deep psychological work.")
        
        if variables.f < 0.2:
            insights.append("Low social content may reflect feelings of isolation.")
        elif variables.f > 0.8:
            insights.append("High social content suggests focus on relationships.")
        
        return insights
    
    def _update_history(self, variables: DreamVariables, result: 'DreamGCTResult'):
        """Update dream history for pattern tracking"""
        # Extract current symbols
        symbols = set()
        for symbol in self.symbol_database:
            if self.symbol_database[symbol].frequency > 0:
                symbols.add(symbol)
                # Track evolution
                self.symbol_evolution[symbol].append(
                    (variables.timestamp, result.coherence)
                )
        
        self.dream_history.append({
            'timestamp': variables.timestamp,
            'coherence': result.coherence,
            'state': result.dream_state,
            'symbols': symbols,
            'variables': variables,
            'result': result
        })
        
        # Keep only last 100 dreams
        if len(self.dream_history) > 100:
            self.dream_history = self.dream_history[-100:]
    
    def get_pattern_analysis(self) -> Dict:
        """Analyze patterns across dream history"""
        if len(self.dream_history) < 5:
            return {"message": "Not enough dreams for pattern analysis"}
        
        # Coherence trends
        coherences = [d['coherence'] for d in self.dream_history]
        avg_coherence = np.mean(coherences)
        coherence_trend = np.polyfit(range(len(coherences)), coherences, 1)[0]
        
        # Most common state
        states = [d['state'] for d in self.dream_history]
        state_counts = defaultdict(int)
        for state in states:
            state_counts[state] += 1
        dominant_state = max(state_counts, key=state_counts.get)
        
        # Symbol frequency
        all_symbols = defaultdict(int)
        for dream in self.dream_history:
            for symbol in dream['symbols']:
                all_symbols[symbol] += 1
        
        top_symbols = sorted(all_symbols.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'avg_coherence': avg_coherence,
            'coherence_trend': 'improving' if coherence_trend > 0.01 else 'declining' if coherence_trend < -0.01 else 'stable',
            'dominant_state': dominant_state,
            'state_distribution': dict(state_counts),
            'top_symbols': top_symbols,
            'total_dreams_analyzed': len(self.dream_history)
        }


@dataclass
class DreamGCTResult(GCTResult):
    """Enhanced GCT result for dreams"""
    dream_state: str
    insights: List[str]
    
    def __init__(self, coherence: float, q_opt: float, dc_dt: float, 
                 d2c_dt2: float, dream_state: str, components: Dict[str, float],
                 insights: List[str]):
        super().__init__(
            coherence=coherence,
            q_opt=q_opt,
            dc_dt=dc_dt,
            d2c_dt2=d2c_dt2,
            sentiment="",  # Not used for dreams
            components=components
        )
        self.dream_state = dream_state
        self.insights = insights