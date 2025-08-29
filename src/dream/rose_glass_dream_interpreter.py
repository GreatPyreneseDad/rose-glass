"""
Rose Glass Dream Interpreter: Translating Dream Patterns Through Cultural Lenses
==============================================================================

This module adapts dream analysis to the Rose Glass framework, viewing dreams
as patterns to be translated rather than measured. Each cultural lens reveals
different aspects of the dream's meaning.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict
import re

from ..core.rose_glass_v2 import RoseGlassV2, PatternInterpretation
from ..core.translation_clarity import TranslationClarityAnalyzer, TranslationSpectrum


@dataclass
class DreamPatterns:
    """Patterns detected in dream narratives"""
    psi: float      # Narrative coherence/clarity
    rho: float      # Symbolic/archetypal depth  
    q: float        # Emotional intensity
    f: float        # Relational/social content
    
    # Dream-specific patterns
    lucidity: float  # Conscious awareness level
    cyclical: float  # Recurring/circular themes
    liminal: float   # Threshold/transformation states


@dataclass  
class DreamSymbol:
    """A symbol appearing in dreams"""
    symbol: str
    appearances: List[Dict]  # Context of each appearance
    cultural_meanings: Dict[str, str]  # Meanings across cultures
    personal_resonance: Optional[str] = None


class RoseGlassDreamInterpreter:
    """
    Interprets dreams through multiple cultural lenses using Rose Glass.
    No judgment, only translation of patterns into understanding.
    """
    
    def __init__(self):
        self.rose_glass = RoseGlassV2()
        self.clarity_analyzer = TranslationClarityAnalyzer()
        
        # Universal dream motifs (appear across cultures)
        self.universal_motifs = {
            'water': "Emotional life, unconscious depths, purification",
            'flying': "Freedom, transcendence, rising above",
            'falling': "Loss of control, surrender, descent", 
            'death': "Transformation, ending, rebirth",
            'birth': "New beginnings, creative potential",
            'animals': "Instinctual nature, guides, aspects of self",
            'light/dark': "Consciousness/unconscious, known/unknown",
            'journey': "Life path, individuation, quest"
        }
        
        # Emotional resonances in dreams
        self.dream_emotions = {
            'fear': ['afraid', 'scared', 'terrified', 'anxious', 'dread', 'panic'],
            'joy': ['happy', 'joyful', 'elated', 'bliss', 'excited', 'peaceful'],
            'anger': ['angry', 'rage', 'furious', 'frustrated', 'annoyed'],
            'sadness': ['sad', 'grief', 'sorrow', 'melancholy', 'lonely'],
            'wonder': ['amazed', 'awed', 'curious', 'fascinated', 'mystified'],
            'love': ['love', 'affection', 'care', 'tender', 'warm', 'connected']
        }
        
        # Transformation indicators
        self.transformation_words = [
            'became', 'transformed', 'changed', 'morphed', 'shifted',
            'turned into', 'evolved', 'dissolved', 'emerged'
        ]
        
    def extract_dream_patterns(self, dream_text: str, 
                             dream_context: Optional[Dict] = None) -> DreamPatterns:
        """
        Extract patterns from dream narrative.
        Not measuring quality, but identifying patterns for translation.
        """
        text_lower = dream_text.lower()
        words = text_lower.split()
        
        # Narrative coherence (how the story flows)
        psi = self._assess_narrative_flow(dream_text)
        
        # Symbolic depth (presence of archetypal content)
        rho = self._assess_symbolic_density(dream_text)
        
        # Emotional intensity
        q = self._assess_emotional_charge(dream_text)
        
        # Social/relational content
        f = self._assess_relational_content(dream_text)
        
        # Dream-specific patterns
        lucidity = self._detect_lucidity_markers(dream_text)
        cyclical = self._detect_cyclical_patterns(dream_text)
        liminal = self._detect_liminal_states(dream_text)
        
        return DreamPatterns(
            psi=psi, rho=rho, q=q, f=f,
            lucidity=lucidity, cyclical=cyclical, liminal=liminal
        )
    
    def _assess_narrative_flow(self, text: str) -> float:
        """Assess how coherently the dream narrative flows"""
        sentences = text.split('.')
        if len(sentences) < 2:
            return 0.5
        
        # Look for connecting words
        connectors = ['then', 'next', 'after', 'before', 'suddenly', 
                     'meanwhile', 'because', 'so', 'but', 'and']
        
        connector_count = sum(1 for word in connectors 
                            if word in text.lower())
        
        # Check for scene transitions
        transition_phrases = ['found myself', 'i was', 'scene changed',
                            'suddenly i', 'then i']
        transitions = sum(1 for phrase in transition_phrases 
                         if phrase in text.lower())
        
        # Calculate flow (not quality, just pattern)
        flow_score = min((connector_count + transitions) / len(sentences), 1.0)
        
        # Adjust for dream logic (non-linear can still flow)
        if 'somehow' in text.lower() or 'just knew' in text.lower():
            flow_score = max(flow_score, 0.6)  # Dream logic is valid
            
        return flow_score
    
    def _assess_symbolic_density(self, text: str) -> float:
        """Assess density of symbolic/archetypal content"""
        symbol_count = 0
        text_lower = text.lower()
        
        # Check universal motifs
        for motif in self.universal_motifs:
            if motif in text_lower:
                symbol_count += 1
        
        # Check for other symbolic content
        symbolic_words = [
            'symbol', 'sign', 'meaning', 'represent', 'stood for',
            'felt like', 'reminded me', 'archetype', 'myth'
        ]
        
        for word in symbolic_words:
            if word in text_lower:
                symbol_count += 0.5
        
        # Numbers, colors, directions often symbolic
        if re.search(r'\b(three|seven|four|twelve)\b', text_lower):
            symbol_count += 0.5
        if re.search(r'\b(red|white|black|gold|blue|green)\b', text_lower):
            symbol_count += 0.5
        if re.search(r'\b(north|south|east|west|up|down)\b', text_lower):
            symbol_count += 0.5
            
        # Normalize by text length
        word_count = len(text.split())
        density = min(symbol_count / (word_count / 100), 1.0)
        
        return density
    
    def _assess_emotional_charge(self, text: str) -> float:
        """Assess emotional intensity in the dream"""
        text_lower = text.lower()
        emotion_count = 0
        intensity_multiplier = 1.0
        
        # Count emotion words
        for emotion_type, words in self.dream_emotions.items():
            for word in words:
                if word in text_lower:
                    emotion_count += 1
        
        # Intensity modifiers
        intensifiers = ['very', 'extremely', 'incredibly', 'overwhelmingly',
                       'intensely', 'deeply', 'profoundly']
        for intensifier in intensifiers:
            if intensifier in text_lower:
                intensity_multiplier = 1.3
                break
        
        # Physical sensations often indicate high emotion
        physical_words = ['heart', 'breath', 'shaking', 'trembling', 
                         'sweating', 'frozen', 'paralyzed']
        physical_count = sum(1 for word in physical_words if word in text_lower)
        
        # Calculate charge
        base_charge = min(emotion_count / 10, 1.0)
        physical_boost = min(physical_count * 0.1, 0.3)
        
        return min((base_charge + physical_boost) * intensity_multiplier, 1.0)
    
    def _assess_relational_content(self, text: str) -> float:
        """Assess social/relational content in dreams"""
        text_lower = text.lower()
        
        # Relationship indicators
        people_words = ['mother', 'father', 'friend', 'family', 'stranger',
                       'crowd', 'people', 'someone', 'everyone', 'nobody',
                       'brother', 'sister', 'child', 'elder', 'ancestor']
        
        interaction_words = ['talked', 'said', 'told', 'asked', 'helped',
                           'hugged', 'fought', 'followed', 'met', 'saw']
        
        collective_words = ['we', 'us', 'our', 'together', 'community',
                          'group', 'circle', 'gathering']
        
        people_count = sum(1 for word in people_words if word in text_lower)
        interaction_count = sum(1 for word in interaction_words if word in text_lower)
        collective_count = sum(1 for word in collective_words if word in text_lower)
        
        # Being alone is also relational (absence of relation)
        if 'alone' in text_lower or 'nobody' in text_lower:
            # This is significant relational content (aloneness)
            return 0.3
        
        # Calculate relational density
        total_relational = people_count + interaction_count + (collective_count * 2)
        word_count = len(text.split())
        
        return min(total_relational / (word_count / 20), 1.0)
    
    def _detect_lucidity_markers(self, text: str) -> float:
        """Detect markers of lucid awareness in dreams"""
        lucidity_phrases = [
            'realized i was dreaming', 'knew it was a dream',
            'became aware', 'conscious that', 'decided to',
            'i could control', 'i chose to', 'i wondered'
        ]
        
        questioning_phrases = [
            'why', 'how', 'what if', 'i wondered', 'strange that'
        ]
        
        text_lower = text.lower()
        lucidity_count = sum(1 for phrase in lucidity_phrases if phrase in text_lower)
        question_count = sum(1 for phrase in questioning_phrases if phrase in text_lower)
        
        return min((lucidity_count * 0.3 + question_count * 0.1), 1.0)
    
    def _detect_cyclical_patterns(self, text: str) -> float:
        """Detect recurring or cyclical elements"""
        text_lower = text.lower()
        
        # Direct repetition words
        repetition_words = ['again', 'kept', 'recurring', 'repeat',
                          'same', 'return', 'back to', 'cycle']
        
        rep_count = sum(1 for word in repetition_words if word in text_lower)
        
        # Check for repeated content
        sentences = text.split('.')
        repeated_elements = 0
        
        # Simple check for repeated key words
        words = text_lower.split()
        word_counts = defaultdict(int)
        for word in words:
            if len(word) > 4:  # Skip small words
                word_counts[word] += 1
        
        # Count words that appear multiple times
        repeated_elements = sum(1 for count in word_counts.values() if count > 2)
        
        return min((rep_count * 0.2 + repeated_elements * 0.05), 1.0)
    
    def _detect_liminal_states(self, text: str) -> float:
        """Detect threshold/transformation states"""
        text_lower = text.lower()
        
        liminal_words = ['threshold', 'door', 'gate', 'bridge', 'crossing',
                        'between', 'neither', 'both', 'edge', 'border',
                        'twilight', 'dawn', 'dusk']
        
        transformation_count = sum(1 for word in self.transformation_words 
                                 if word in text_lower)
        liminal_count = sum(1 for word in liminal_words if word in text_lower)
        
        return min((transformation_count * 0.2 + liminal_count * 0.15), 1.0)
    
    def translate_dream(self, dream_text: str, 
                       selected_lens: Optional[str] = None,
                       dream_context: Optional[Dict] = None) -> Dict:
        """
        Translate dream patterns through selected cultural lens.
        Returns multiple interpretations, not judgments.
        """
        # Extract patterns
        patterns = self.extract_dream_patterns(dream_text, dream_context)
        
        # Convert to Rose Glass format
        rose_patterns = {
            'psi': patterns.psi,
            'rho': patterns.rho,
            'q': patterns.q,
            'f': patterns.f
        }
        
        # If no lens selected, try multiple
        if not selected_lens:
            return self._multi_lens_translation(dream_text, rose_patterns, patterns)
        
        # Select specific lens
        self.rose_glass.select_lens(selected_lens)
        
        # Translate patterns
        interpretation = self.rose_glass.translate_patterns(
            **rose_patterns,
            text_sample=dream_text
        )
        
        # Extract symbols
        symbols = self._extract_dream_symbols(dream_text)
        
        # Generate dream-specific insights
        insights = self._generate_dream_insights(patterns, interpretation, symbols)
        
        # Analyze translation clarity
        clarity_spectrum = self.clarity_analyzer.create_spectrum(
            patterns=rose_patterns,
            lens_name=selected_lens,
            alternatives=interpretation.alternative_readings,
            cultural_context=dream_context or {}
        )
        
        return {
            'patterns': patterns,
            'interpretation': interpretation,
            'symbols': symbols,
            'insights': insights,
            'clarity_spectrum': clarity_spectrum,
            'dream_specific': {
                'lucidity_level': self._interpret_lucidity(patterns.lucidity),
                'cyclical_nature': self._interpret_cycles(patterns.cyclical),
                'liminal_state': self._interpret_liminality(patterns.liminal)
            }
        }
    
    def _multi_lens_translation(self, dream_text: str, 
                               rose_patterns: Dict, 
                               dream_patterns: DreamPatterns) -> Dict:
        """Translate dream through multiple lenses for comparison"""
        translations = {}
        
        # Try different lenses
        lenses = ['medieval_islamic', 'indigenous_oral', 
                 'buddhist_contemplative', 'digital_native']
        
        for lens in lenses:
            if lens in self.rose_glass.calibrations:
                self.rose_glass.select_lens(lens)
                interpretation = self.rose_glass.translate_patterns(
                    **rose_patterns,
                    text_sample=dream_text
                )
                translations[lens] = interpretation
        
        # Find which lens provides clearest translation
        best_lens = None
        best_intensity = 0
        
        for lens, interp in translations.items():
            if interp.coherence_construction > best_intensity:
                best_intensity = interp.coherence_construction
                best_lens = lens
        
        # Extract symbols once
        symbols = self._extract_dream_symbols(dream_text)
        
        return {
            'multi_lens_view': translations,
            'clearest_lens': best_lens,
            'symbols': symbols,
            'patterns': dream_patterns,
            'insight': "Different cultural lenses reveal different aspects of your dream. "
                      "No single interpretation is 'correct' - each offers valuable perspective."
        }
    
    def _extract_dream_symbols(self, dream_text: str) -> List[DreamSymbol]:
        """Extract significant symbols from dream"""
        symbols = []
        text_lower = dream_text.lower()
        
        # Check universal motifs
        for motif, meaning in self.universal_motifs.items():
            if motif in text_lower:
                # Find context
                sentences = dream_text.split('.')
                contexts = []
                for sent in sentences:
                    if motif in sent.lower():
                        contexts.append(sent.strip())
                
                symbol = DreamSymbol(
                    symbol=motif,
                    appearances=[{'context': ctx, 'position': i} 
                               for i, ctx in enumerate(contexts)],
                    cultural_meanings={
                        'universal': meaning,
                        'western': self._get_western_meaning(motif),
                        'eastern': self._get_eastern_meaning(motif),
                        'indigenous': self._get_indigenous_meaning(motif)
                    }
                )
                symbols.append(symbol)
        
        return symbols
    
    def _get_western_meaning(self, symbol: str) -> str:
        """Get Western interpretation of symbol"""
        western_meanings = {
            'water': "Emotions, unconscious mind, purification",
            'flying': "Freedom, escaping limitations, ambition",
            'falling': "Loss of control, failure anxiety, letting go",
            'death': "End of a phase, transformation, fear of loss",
            'animals': "Instincts, shadow aspects, natural wisdom"
        }
        return western_meanings.get(symbol, "Personal significance varies")
    
    def _get_eastern_meaning(self, symbol: str) -> str:
        """Get Eastern interpretation of symbol"""
        eastern_meanings = {
            'water': "Flow of qi/prana, flexibility, the Dao",
            'flying': "Spiritual liberation, transcendence of maya",
            'falling': "Surrender to the universe, ego dissolution",
            'death': "Impermanence, rebirth, karma completion",
            'animals': "Spirit guides, past life connections"
        }
        return eastern_meanings.get(symbol, "Context determines meaning")
    
    def _get_indigenous_meaning(self, symbol: str) -> str:
        """Get Indigenous interpretation of symbol"""
        indigenous_meanings = {
            'water': "Life force, cleansing, ancestors' realm",
            'flying': "Spirit journey, shamanic flight, freedom",
            'falling': "Descent to underworld, initiation",
            'death': "Transition to spirit world, ancestor wisdom",
            'animals': "Power animals, teachers, clan totems"
        }
        return indigenous_meanings.get(symbol, "Requires cultural context")
    
    def _generate_dream_insights(self, patterns: DreamPatterns,
                               interpretation: PatternInterpretation,
                               symbols: List[DreamSymbol]) -> List[str]:
        """Generate insights specific to dream analysis"""
        insights = []
        
        # Pattern-based insights
        if patterns.psi > 0.8:
            insights.append("Your dream shows clear narrative structure, suggesting "
                          "active integration of experiences.")
        elif patterns.psi < 0.4:
            insights.append("The fragmented narrative may indicate processing of "
                          "complex or conflicting experiences.")
        
        if patterns.rho > 0.7:
            insights.append("Rich symbolic content suggests deep psychological work. "
                          "These symbols deserve reflection.")
        
        if patterns.q > 0.8:
            insights.append("High emotional intensity indicates this dream touches "
                          "important psychological material.")
        
        if patterns.lucidity > 0.5:
            insights.append("Lucidity markers suggest growing consciousness within "
                          "the dream state. This is often spiritually significant.")
        
        # Symbol insights
        if symbols:
            symbol_names = [s.symbol for s in symbols]
            insights.append(f"Key symbols appearing: {', '.join(symbol_names)}. "
                          "Each carries multiple cultural meanings.")
        
        # Lens-specific insight
        lens_name = interpretation.lens_used.name
        insights.append(f"Through the {lens_name} lens: {interpretation.cultural_notes}")
        
        return insights
    
    def _interpret_lucidity(self, level: float) -> str:
        """Interpret lucidity level"""
        if level < 0.2:
            return "Fully immersed in dream consciousness"
        elif level < 0.5:
            return "Moments of questioning or awareness"
        elif level < 0.8:
            return "Significant lucid awareness present"
        else:
            return "High lucidity - conscious participation in dream"
    
    def _interpret_cycles(self, level: float) -> str:
        """Interpret cyclical patterns"""
        if level < 0.2:
            return "Linear dream narrative"
        elif level < 0.5:
            return "Some recurring elements or themes"
        elif level < 0.8:
            return "Strong cyclical patterns - possible recurring dream elements"
        else:
            return "Highly cyclical - may be processing persistent themes"
    
    def _interpret_liminality(self, level: float) -> str:
        """Interpret liminal/threshold states"""
        if level < 0.2:
            return "Stable dream environment"
        elif level < 0.5:
            return "Some transitional elements present"
        elif level < 0.8:
            return "Significant threshold crossing - transformation occurring"
        else:
            return "Highly liminal - major psychological transition indicated"
    
    def generate_dream_report(self, translation_result: Dict) -> str:
        """Generate a narrative dream interpretation report"""
        patterns = translation_result['patterns']
        
        report = f"""
ROSE GLASS DREAM TRANSLATION
============================

Dream Pattern Summary:
- Narrative Flow: {patterns.psi:.2f} - {self._describe_pattern(patterns.psi, 'flow')}
- Symbolic Density: {patterns.rho:.2f} - {self._describe_pattern(patterns.rho, 'symbols')}
- Emotional Charge: {patterns.q:.2f} - {self._describe_pattern(patterns.q, 'emotion')}
- Relational Content: {patterns.f:.2f} - {self._describe_pattern(patterns.f, 'social')}

Dream-Specific Qualities:
{translation_result['dream_specific']['lucidity_level']}
{translation_result['dream_specific']['cyclical_nature']}
{translation_result['dream_specific']['liminal_state']}
"""
        
        # Add interpretation through lens
        if 'interpretation' in translation_result:
            interp = translation_result['interpretation']
            report += f"\n{interp.get_narrative()}\n"
        
        # Add symbols
        if translation_result.get('symbols'):
            report += "\nSignificant Symbols:\n"
            for symbol in translation_result['symbols']:
                report += f"\n{symbol.symbol.upper()}:\n"
                for culture, meaning in symbol.cultural_meanings.items():
                    report += f"  {culture}: {meaning}\n"
        
        # Add insights
        if translation_result.get('insights'):
            report += "\nDream Insights:\n"
            for insight in translation_result['insights']:
                report += f"- {insight}\n"
        
        # Add clarity spectrum
        if 'clarity_spectrum' in translation_result:
            spectrum = translation_result['clarity_spectrum']
            report += f"\nTranslation Clarity: {spectrum.clarity.value}\n"
            report += f"{spectrum.uncertainty_celebration}\n"
        
        report += "\n" + "="*50 + "\n"
        report += "Remember: Dreams are personal. These translations offer perspectives, "
        report += "not definitive meanings. Your own understanding is most valuable.\n"
        
        return report
    
    def _describe_pattern(self, value: float, pattern_type: str) -> str:
        """Describe pattern intensity without judgment"""
        descriptions = {
            'flow': {
                0.8: "highly coherent narrative",
                0.5: "dream logic flow",
                0.0: "fragmented impressions"
            },
            'symbols': {
                0.8: "rich archetypal content",
                0.5: "moderate symbolic presence",
                0.0: "literal/direct imagery"
            },
            'emotion': {
                0.8: "intense emotional content",
                0.5: "moderate emotional tone",
                0.0: "emotionally neutral"
            },
            'social': {
                0.8: "highly relational/collective",
                0.5: "balanced self/other presence",
                0.0: "solitary journey"
            }
        }
        
        desc_set = descriptions.get(pattern_type, {})
        for threshold in sorted(desc_set.keys(), reverse=True):
            if value >= threshold:
                return desc_set[threshold]
        return desc_set[0.0]


# Example usage
def demonstrate_dream_translation():
    """Demonstrate Rose Glass dream translation"""
    interpreter = RoseGlassDreamInterpreter()
    
    # Example dream
    dream_text = """
    I found myself in a vast library that seemed to stretch infinitely upward. 
    The books were glowing with soft light, and when I touched one, I could 
    feel the knowledge flowing into me like water. Suddenly my grandmother 
    appeared, but she was young again. She smiled and handed me a golden key, 
    saying "This opens what you've always known." Then I was flying above 
    a city I'd never seen before, yet it felt like home. I realized I was 
    dreaming and decided to dive into the ocean below, where I transformed 
    into a dolphin and swam with my ancestors.
    """
    
    # Translate through multiple lenses
    result = interpreter.translate_dream(dream_text)
    
    # Generate report
    report = interpreter.generate_dream_report(result)
    print(report)
    
    # Show multi-lens view
    if 'multi_lens_view' in result:
        print("\nMULTI-LENS PERSPECTIVE:")
        for lens, interp in result['multi_lens_view'].items():
            print(f"\n{lens}: {interp.coherence_construction:.2f}/4.0")
            print(f"  {interp.alternative_readings[0]}")


if __name__ == "__main__":
    demonstrate_dream_translation()