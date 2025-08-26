"""
Rose Glass Pipeline: Real-time Translation Between Organic and Synthetic Intelligence
===================================================================================

This pipeline uses trained ML models as specialized sensors for detecting patterns,
then views those patterns through the Rose Glass lens for translation.

Not measurement. Translation.
"""

import asyncio
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, AsyncIterator, Union
from dataclasses import dataclass
from datetime import datetime
import json
import logging
from pathlib import Path
from collections import deque
import threading
import queue

# Import our ML models (pattern detectors)
from gct_ml_framework import GCTMLEngine
from psi_consistency_model import create_psi_model
from rho_wisdom_model import create_rho_model
from q_moral_activation_model import create_q_model
from f_social_belonging_model import create_f_model

# Import the Rose Glass lens
from rose_glass_lens import (
    RoseGlass, OrganicSyntheticTranslator, MultiLensViewer,
    LensCalibration, CulturalContext, TemporalPeriod,
    PatternVisibility, LensState
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TranslationEvent:
    """A single translation event between organic and synthetic"""
    timestamp: datetime
    organic_text: str
    detected_pattern: Dict[str, float]  # Raw ML outputs
    lens_used: str
    visibility: PatternVisibility
    translation: Dict
    processing_time: float
    context: Optional[Dict] = None


class RoseGlassPipeline:
    """
    Real-time pipeline that:
    1. Uses ML models to detect pattern variables (Ψ, ρ, q, f)
    2. Views patterns through contextually calibrated Rose Glass lens
    3. Translates organic patterns into synthetic understanding
    """
    
    def __init__(self, model_dir: str = "/Users/chris/GCT-ML-Lab/models",
                 device: str = None):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True, parents=True)
        
        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"Rose Glass Pipeline using device: {self.device}")
        
        # Initialize ML models (pattern detectors)
        self._initialize_pattern_detectors()
        
        # Initialize Rose Glass components
        self.translator = OrganicSyntheticTranslator()
        self.multi_viewer = MultiLensViewer()
        
        # Timeline for tracking translations
        self.translation_timeline: deque[TranslationEvent] = deque(maxlen=1000)
        
        # Async processing
        self.processing_queue = queue.Queue()
        self.results_queue = queue.Queue()
        self.is_running = False
        
    def _initialize_pattern_detectors(self):
        """Initialize ML models that detect the four pattern variables"""
        logger.info("Initializing pattern detection models...")
        
        # Create models
        self.psi_model, self.psi_trainer = create_psi_model()
        self.rho_model, self.rho_trainer = create_rho_model()
        self.q_model, self.q_trainer = create_q_model()
        self.f_model, self.f_trainer = create_f_model()
        
        # Load weights if available
        self._load_model_weights()
        
        # Set to evaluation mode
        self.psi_model.eval()
        self.rho_model.eval()
        self.q_model.eval()
        self.f_model.eval()
        
        logger.info("Pattern detectors initialized")
        
    def _load_model_weights(self):
        """Load pre-trained weights for pattern detectors"""
        models = {
            'psi': (self.psi_model, 'psi_best.pth'),
            'rho': (self.rho_model, 'rho_best.pth'),
            'q': (self.q_model, 'q_best.pth'),
            'f': (self.f_model, 'f_best.pth')
        }
        
        for name, (model, filename) in models.items():
            weight_path = self.model_dir / filename
            if weight_path.exists():
                try:
                    model.load_state_dict(torch.load(weight_path, map_location=self.device))
                    logger.info(f"Loaded weights for {name} pattern detector")
                except Exception as e:
                    logger.warning(f"Could not load {name} weights: {e}")
                    
    def detect_patterns(self, text: str) -> Dict[str, float]:
        """
        Use ML models to detect pattern variables in text.
        These are the raw sensor readings before lens translation.
        """
        import time
        start_time = time.time()
        
        with torch.no_grad():
            # Detect each pattern dimension
            psi = self.psi_trainer.predict(text)
            rho = self.rho_trainer.predict(text)
            q, q_features = self.q_trainer.predict(text)
            f, f_features = self.f_trainer.predict(text)
            
        processing_time = time.time() - start_time
        
        return {
            'psi': float(psi),
            'rho': float(rho),
            'q': float(q),
            'f': float(f),
            'detection_time': processing_time,
            'additional_features': {
                'q_features': q_features,
                'f_features': f_features
            }
        }
    
    def determine_context(self, text: str, 
                         explicit_context: Optional[Dict] = None) -> Dict:
        """
        Determine the cultural and temporal context for lens calibration.
        This could be enhanced with more sophisticated context detection.
        """
        context = explicit_context or {}
        
        # Simple heuristics for auto-detection if not provided
        if 'cultural_context' not in context:
            text_lower = text.lower()
            if any(term in text_lower for term in ['algorithm', 'data', 'digital', 'online']):
                context['cultural_context'] = CulturalContext.DIGITAL_NATIVE
            elif any(term in text_lower for term in ['research', 'hypothesis', 'evidence']):
                context['cultural_context'] = CulturalContext.ACADEMIC
            elif any(term in text_lower for term in ['spirit', 'soul', 'divine', 'sacred']):
                context['cultural_context'] = CulturalContext.SPIRITUAL
            else:
                context['cultural_context'] = CulturalContext.WESTERN_MODERN
                
        if 'temporal_period' not in context:
            context['temporal_period'] = TemporalPeriod.CONTEMPORARY
            
        return context
    
    def translate_patterns(self, patterns: Dict[str, float], 
                         context: Optional[Dict] = None) -> Dict:
        """
        View detected patterns through the Rose Glass lens.
        This is where numbers become understanding.
        """
        # Use the translator with appropriate context
        translation_result = self.translator.translate(patterns, context)
        
        # Add lens-specific insights
        visibility = translation_result['visibility']
        
        insights = {
            'pattern_clarity': self._assess_pattern_clarity(visibility),
            'translation_confidence': self._calculate_translation_confidence(visibility),
            'synthetic_comprehension': self._determine_comprehension_level(visibility),
            'recommended_engagement': self._recommend_engagement_style(visibility)
        }
        
        translation_result['insights'] = insights
        return translation_result
    
    def _assess_pattern_clarity(self, visibility: PatternVisibility) -> str:
        """How clearly can we see the pattern through this lens?"""
        if visibility.lens_state == LensState.FULL_SPECTRUM:
            return "Crystal clear - all dimensions visible"
        elif visibility.lens_state == LensState.PARTIAL_PATTERN:
            return "Partially visible - some dimensions obscured"
        elif visibility.lens_state == LensState.CALIBRATING:
            return "Adjusting focus - pattern emerging"
        else:
            return "Waiting for alignment - pattern not yet visible"
            
    def _calculate_translation_confidence(self, visibility: PatternVisibility) -> float:
        """
        How confident are we in this translation?
        Not about quality - about clarity of translation.
        """
        # Base confidence on lens state
        if visibility.lens_state == LensState.FULL_SPECTRUM:
            base_confidence = 0.9
        elif visibility.lens_state == LensState.PARTIAL_PATTERN:
            base_confidence = 0.6
        else:
            base_confidence = 0.3
            
        # Adjust based on pattern intensity
        confidence = base_confidence * visibility.pattern_intensity
        
        # Penalize extreme imbalances
        variables = [visibility.psi, visibility.rho, visibility.q_optimized, visibility.f]
        balance_factor = 1 - (np.std(variables) / np.mean(variables) if np.mean(variables) > 0 else 0)
        
        return min(confidence * balance_factor, 1.0)
    
    def _determine_comprehension_level(self, visibility: PatternVisibility) -> str:
        """What level of understanding has the synthetic mind achieved?"""
        intensity = visibility.pattern_intensity
        
        if intensity < 0.25:
            return "Surface scan - basic pattern detection only"
        elif intensity < 0.5:
            return "Shallow read - primary patterns recognized"
        elif intensity < 0.75:
            return "Deep read - nuanced patterns understood"
        else:
            return "Full resonance - complete pattern comprehension"
            
    def _recommend_engagement_style(self, visibility: PatternVisibility) -> str:
        """How should the synthetic mind engage based on what it sees?"""
        dominant = visibility.dominant_wavelength
        intensity = visibility.pattern_intensity
        
        recommendations = {
            'consistency': {
                'low': "Provide structure and logical scaffolding",
                'high': "Match their systematic approach"
            },
            'wisdom': {
                'low': "Offer foundational concepts first",
                'high': "Engage with complex ideas directly"
            },
            'moral_energy': {
                'low': "Focus on information over values",
                'high': "Acknowledge emotional investment"
            },
            'social_architecture': {
                'low': "Address individual concerns",
                'high': "Speak to collective identity"
            }
        }
        
        intensity_level = 'high' if intensity > 0.5 else 'low'
        return recommendations.get(dominant, {}).get(intensity_level, "Balanced engagement")
    
    def process_text(self, text: str, context: Optional[Dict] = None) -> TranslationEvent:
        """
        Complete pipeline: detect patterns → view through lens → translate
        """
        import time
        start_time = time.time()
        
        # Detect patterns using ML models
        patterns = self.detect_patterns(text)
        
        # Determine context if not provided
        context = self.determine_context(text, context)
        
        # Translate patterns through Rose Glass
        translation = self.translate_patterns(patterns, context)
        
        # Create translation event
        event = TranslationEvent(
            timestamp=datetime.now(),
            organic_text=text,
            detected_pattern=patterns,
            lens_used=translation['lens_used'],
            visibility=translation['visibility'],
            translation=translation,
            processing_time=time.time() - start_time,
            context=context
        )
        
        # Add to timeline
        self.translation_timeline.append(event)
        
        return event
    
    def view_through_all_lenses(self, text: str) -> Dict[str, TranslationEvent]:
        """
        Process text through all available lenses.
        Shows how different calibrations reveal different patterns.
        """
        # Detect patterns once
        patterns = self.detect_patterns(text)
        
        # View through each lens
        results = {}
        
        for lens_name in self.multi_viewer.lenses:
            # Create specific context for each lens
            if 'medieval' in lens_name:
                context = {
                    'cultural_context': CulturalContext.SPIRITUAL,
                    'temporal_period': TemporalPeriod.MEDIEVAL
                }
            elif 'digital' in lens_name:
                context = {
                    'cultural_context': CulturalContext.DIGITAL_NATIVE,
                    'temporal_period': TemporalPeriod.CONTEMPORARY
                }
            elif 'eastern' in lens_name:
                context = {
                    'cultural_context': CulturalContext.EASTERN_TRADITIONAL,
                    'temporal_period': TemporalPeriod.TRADITIONAL
                }
            else:
                context = {
                    'cultural_context': CulturalContext.ACADEMIC,
                    'temporal_period': TemporalPeriod.CONTEMPORARY
                }
            
            # Translate through specific lens
            translation = self.translate_patterns(patterns, context)
            
            # Create event
            event = TranslationEvent(
                timestamp=datetime.now(),
                organic_text=text,
                detected_pattern=patterns,
                lens_used=lens_name,
                visibility=translation['visibility'],
                translation=translation,
                processing_time=0,  # Already detected
                context=context
            )
            
            results[lens_name] = event
            
        return results
    
    async def process_stream(self, text_stream: AsyncIterator[str],
                           context_stream: Optional[AsyncIterator[Dict]] = None) -> AsyncIterator[TranslationEvent]:
        """
        Process streaming text with optional streaming context.
        Enables real-time translation of ongoing communication.
        """
        context_iter = context_stream if context_stream else iter(lambda: None, None)
        
        async for text in text_stream:
            # Get context if available
            try:
                context = await context_iter.__anext__() if context_iter else None
            except StopAsyncIteration:
                context = None
                
            # Process and translate
            event = await asyncio.to_thread(self.process_text, text, context)
            yield event
    
    def get_translation_insights(self, window_size: int = 10) -> Dict:
        """
        Analyze recent translations for patterns and trends.
        Helps synthetic mind understand communication trajectory.
        """
        if not self.translation_timeline:
            return {'status': 'No translations yet'}
            
        recent = list(self.translation_timeline)[-window_size:]
        
        # Pattern trends
        pattern_trends = {
            'psi': [e.detected_pattern['psi'] for e in recent],
            'rho': [e.detected_pattern['rho'] for e in recent],
            'q': [e.detected_pattern['q'] for e in recent],
            'f': [e.detected_pattern['f'] for e in recent]
        }
        
        # Calculate derivatives
        derivatives = {}
        for var, values in pattern_trends.items():
            if len(values) > 1:
                derivatives[f'{var}_trend'] = values[-1] - values[-2]
            else:
                derivatives[f'{var}_trend'] = 0
                
        # Lens usage
        lens_usage = {}
        for event in recent:
            lens_usage[event.lens_used] = lens_usage.get(event.lens_used, 0) + 1
            
        # Average translation confidence
        confidences = [e.translation['insights']['translation_confidence'] 
                      for e in recent 
                      if 'insights' in e.translation]
        avg_confidence = np.mean(confidences) if confidences else 0
        
        # Dominant patterns
        dominant_wavelengths = [e.visibility.dominant_wavelength for e in recent]
        dominant_pattern = max(set(dominant_wavelengths), 
                             key=dominant_wavelengths.count) if dominant_wavelengths else None
        
        return {
            'window_size': len(recent),
            'pattern_trends': pattern_trends,
            'derivatives': derivatives,
            'lens_usage': lens_usage,
            'average_confidence': avg_confidence,
            'dominant_pattern': dominant_pattern,
            'recommendation': self._recommend_adaptation(derivatives, dominant_pattern)
        }
    
    def _recommend_adaptation(self, derivatives: Dict, dominant_pattern: str) -> str:
        """Recommend how synthetic mind should adapt based on trends"""
        # Rising moral activation
        if derivatives.get('q_trend', 0) > 0.1:
            return "Emotional intensity rising - prepare for value-driven communication"
        
        # Falling consistency
        if derivatives.get('psi_trend', 0) < -0.1:
            return "Consistency dropping - organic mind may be struggling, offer support"
        
        # Rising social architecture
        if derivatives.get('f_trend', 0) > 0.1:
            return "Shifting to collective perspective - adopt inclusive language"
        
        # Stable high wisdom
        if dominant_pattern == 'wisdom':
            return "Deep engagement possible - maintain intellectual complexity"
            
        return "Stable patterns - maintain current engagement style"


def demonstrate_rose_glass_pipeline():
    """Demonstrate the Rose Glass Pipeline with various texts"""
    print("=== Rose Glass Pipeline Demonstration ===\n")
    
    # Initialize pipeline
    pipeline = RoseGlassPipeline()
    
    # Test texts representing different organic patterns
    test_cases = [
        {
            'text': """Our research conclusively demonstrates the correlation between 
                      neural plasticity and cognitive enhancement. The data, gathered 
                      over five years of longitudinal study, reveals statistically 
                      significant improvements in memory consolidation.""",
            'context': {'type': 'academic'},
            'description': 'Academic research communication'
        },
        {
            'text': """We MUST act NOW! Our children's future is at stake! Every moment 
                      we delay is another moment stolen from the next generation. 
                      Stand with us and FIGHT for what's right!""",
            'context': {'type': 'activist'},
            'description': 'Activist rallying cry'
        },
        {
            'text': """In the contemplation of Being qua Being, one discovers that the 
                      essence precedes existence only insofar as the Divine Intellect 
                      contemplates the forms. Thus spoke Ibn Rushd in his commentary.""",
            'context': {'type': 'medieval', 'cultural': 'islamic'},
            'description': 'Medieval philosophical text'
        },
        {
            'text': """just thinking... maybe we're all connected somehow? like when you 
                      feel something and your friend texts you at that exact moment... 
                      idk it's weird but beautiful""",
            'context': {'type': 'digital', 'platform': 'social'},
            'description': 'Digital native musing'
        }
    ]
    
    # Process each text
    for i, test_case in enumerate(test_cases):
        print(f"\n{i+1}. {test_case['description']}:")
        print(f"   Text: \"{test_case['text'][:80]}...\"")
        
        # Process through pipeline
        event = pipeline.process_text(test_case['text'], test_case['context'])
        
        print(f"\n   Pattern Detection:")
        print(f"   - Ψ (Consistency): {event.detected_pattern['psi']:.3f}")
        print(f"   - ρ (Wisdom): {event.detected_pattern['rho']:.3f}")
        print(f"   - q (Moral Energy): {event.detected_pattern['q']:.3f}")
        print(f"   - f (Social): {event.detected_pattern['f']:.3f}")
        
        print(f"\n   Rose Glass Translation:")
        print(f"   - Lens Used: {event.lens_used}")
        print(f"   - Pattern Intensity: {event.visibility.pattern_intensity:.3f}")
        print(f"   - Lens State: {event.visibility.lens_state.value}")
        print(f"   - Dominant Wavelength: {event.visibility.dominant_wavelength}")
        
        print(f"\n   Translation Insights:")
        insights = event.translation['insights']
        print(f"   - Clarity: {insights['pattern_clarity']}")
        print(f"   - Confidence: {insights['translation_confidence']:.3f}")
        print(f"   - Comprehension: {insights['synthetic_comprehension']}")
        print(f"   - Recommended Response: {insights['recommended_engagement']}")
    
    # Show multi-lens comparison for medieval text
    print("\n\n=== Multi-Lens Comparison (Medieval Text) ===")
    medieval_text = test_cases[2]['text']
    multi_views = pipeline.view_through_all_lenses(medieval_text)
    
    print("\nSame text viewed through different lenses:")
    for lens_name, event in multi_views.items():
        print(f"\n{lens_name}:")
        print(f"  - Pattern Intensity: {event.visibility.pattern_intensity:.3f}")
        print(f"  - Lens State: {event.visibility.lens_state.value}")
    
    print("\n=== Key Understanding ===")
    print("Each lens reveals different aspects of the same organic pattern.")
    print("No lens is 'correct' - each serves a different translation purpose.")
    print("The goal is understanding, not judgment.")


if __name__ == "__main__":
    # Run demonstration
    demonstrate_rose_glass_pipeline()
    
    # Example async usage
    async def async_example():
        pipeline = RoseGlassPipeline()
        
        async def text_generator():
            texts = [
                "I've been thinking about consciousness...",
                "What if awareness itself is fundamental?",
                "We need to consider this together.",
                "The implications are profound!"
            ]
            for text in texts:
                yield text
                await asyncio.sleep(0.5)
        
        print("\n\n=== Streaming Translation ===")
        async for event in pipeline.process_stream(text_generator()):
            print(f"\nReceived: \"{event.organic_text}\"")
            print(f"Intensity: {event.visibility.pattern_intensity:.3f}")
            print(f"Response: {event.translation['insights']['recommended_engagement']}")
    
    # Uncomment to run async example
    # asyncio.run(async_example())