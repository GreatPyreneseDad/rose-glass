"""
Adaptive Response System
========================

Calibrate response parameters based on token flow dynamics and coherence state.
This system implements the key insight that AI responses should adapt to the
rhythm and energy of the conversation, not just the content.

"Crisis (low C): Reduce token flow rate (shorter responses, more breathing room)"
"High coherence: Accelerate token flow (richer exchanges, deeper dives)"

Author: Christopher MacGregor bin Joseph
Date: October 2025
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import re

# Import the four critical context detectors
from .trust_signal_detector import TrustSignalDetector, TrustSignal
from .mission_mode_detector import MissionModeDetector, Mission
from .token_multiplier_limiter import TokenMultiplierLimiter, TokenLimit
from .essence_request_detector import EssenceRequestDetector, EssenceRequest


class ResponsePacing(Enum):
    """Response pacing modes"""
    DELIBERATE = "deliberate"      # Slow, careful pacing for crisis
    STANDARD = "standard"          # Normal conversational flow
    CONTEMPLATIVE = "contemplative"  # Thoughtful, expansive pacing
    EXPANSIVE = "expansive"        # Rich, detailed exploration
    SLOWED = "slowed"             # Emergency brake for spirals
    REVERENT = "reverent"         # Witness and honor exceptional coherence
    SYSTEMATIC = "systematic"      # Mission mode - thorough exploration
    DISTILLED = "distilled"        # Essence mode - concentrated insights


class ComplexityLevel(Enum):
    """Response complexity levels"""
    GROUNDING = "grounding"        # Simple, anchoring language
    SIMPLIFIED = "simplified"      # Clear, direct communication
    MATCHED = "matched"           # Match user's complexity
    ELEVATED = "elevated"         # Sophisticated exploration
    MINIMAL_INTERFERENCE = "minimal_interference"  # Honor without burying


class SentenceLength(Enum):
    """Sentence length preferences"""
    VERY_SHORT = "very_short"     # 5-10 words
    SHORT = "short"               # 10-15 words
    MEDIUM = "medium"             # 15-25 words
    LONG = "long"                 # 25-40 words
    VARIED = "varied"             # Mix of lengths


@dataclass
class ResponseCalibration:
    """Parameters for calibrating response characteristics"""
    target_tokens: int = 150
    sentence_length: SentenceLength = SentenceLength.MEDIUM
    paragraph_breaks: bool = True
    complexity_level: ComplexityLevel = ComplexityLevel.MATCHED
    pacing: ResponsePacing = ResponsePacing.STANDARD
    
    # Additional calibration parameters
    use_metaphors: bool = True
    include_questions: bool = True
    emotional_mirroring: float = 0.5  # 0-1 scale
    conceptual_density: float = 0.5   # 0-1 scale
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert calibration to dictionary"""
        return {
            'target_tokens': self.target_tokens,
            'sentence_length': self.sentence_length.value,
            'paragraph_breaks': self.paragraph_breaks,
            'complexity_level': self.complexity_level.value,
            'pacing': self.pacing.value,
            'use_metaphors': self.use_metaphors,
            'include_questions': self.include_questions,
            'emotional_mirroring': self.emotional_mirroring,
            'conceptual_density': self.conceptual_density
        }


class AdaptiveResponseSystem:
    """Calibrate response parameters based on token derivatives"""
    
    def __init__(self):
        """Initialize adaptive response system"""
        self.calibration_history: List[Tuple[float, ResponseCalibration]] = []
        self.response_templates = self._load_response_templates()
        
        # Initialize the four critical context detectors
        self.trust_detector = TrustSignalDetector()
        self.mission_detector = MissionModeDetector()
        self.token_limiter = TokenMultiplierLimiter()
        self.essence_detector = EssenceRequestDetector()
    
    def detect_context(self, 
                      message: str, 
                      coherence: float,
                      conversation_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run all context detectors on the message
        
        Args:
            message: User message text
            coherence: Current coherence value
            conversation_state: Current conversation state
            
        Returns:
            Context detection results
        """
        user_tokens = len(message.split())
        
        # Run all detectors
        trust_signal = self.trust_detector.detect_trust_signals(
            message, coherence, user_tokens
        )
        
        mission = self.mission_detector.detect_mission(message)
        
        essence_request = self.essence_detector.detect_essence_request(message)
        
        # Build conversation state for token limiter
        limiter_state = {
            'crisis_detected': conversation_state.get('crisis_detected', False),
            'information_overload': conversation_state.get('information_overload', False),
            'trust_signal_detected': trust_signal is not None,
            'mission_mode': mission is not None,
            'recent_response_ratios': conversation_state.get('recent_response_ratios', [])
        }
        
        token_limit = self.token_limiter.calculate_token_limit(
            user_tokens, coherence, limiter_state
        )
        
        # Determine primary context mode
        primary_mode = self._determine_primary_mode(
            trust_signal, mission, essence_request, coherence, conversation_state
        )
        
        return {
            'user_tokens': user_tokens,
            'trust_signal': trust_signal,
            'mission': mission,
            'essence_request': essence_request,
            'token_limit': token_limit,
            'primary_mode': primary_mode,
            'detections': {
                'trust_detected': trust_signal is not None,
                'mission_detected': mission is not None,
                'essence_detected': essence_request is not None
            }
        }
    
    def _determine_primary_mode(self,
                               trust_signal: Optional[TrustSignal],
                               mission: Optional[Mission],
                               essence_request: Optional[EssenceRequest],
                               coherence: float,
                               conversation_state: Dict[str, Any]) -> str:
        """Determine which context mode takes precedence"""
        # Priority order:
        # 1. Crisis/Information overload (highest priority)
        if conversation_state.get('crisis_detected') or conversation_state.get('information_overload'):
            return 'crisis'
        
        # 2. Trust signal with high confidence
        if trust_signal and self.trust_detector.should_trigger_reverent_mode(trust_signal, 'standard'):
            return 'trust'
        
        # 3. Essence request (summaries are time-sensitive)
        if essence_request and self.essence_detector.should_override_token_flow(essence_request):
            return 'essence'
        
        # 4. Mission mode
        if mission and self.mission_detector.should_override_coherence_mode(mission, coherence):
            return 'mission'
        
        # 5. High coherence (C > 3.5)
        if coherence > 3.5:
            return 'exceptional_coherence'
        
        # 6. Default to coherence-based
        return 'coherence_based'
        
    def calibrate_response_length(self,
                                  coherence_state: float,
                                  dC_dtokens: float,
                                  flow_rate: float,
                                  user_message_tokens: Optional[int] = None) -> ResponseCalibration:
        """
        Determine optimal response characteristics based on dynamics
        
        Args:
            coherence_state: Current C value (0-4 scale)
            dC_dtokens: Information-theoretic derivative
            flow_rate: Current tokens/second
            user_message_tokens: Length of user's message (for matching)
            
        Returns:
            Response calibration parameters
        """
        calibration = ResponseCalibration()
        
        # NEW: Exceptional coherence (C > 3.5) = witness and honor
        if coherence_state > 3.5:
            calibration.target_tokens = min(user_message_tokens * 0.6, 100) if user_message_tokens else 80
            calibration.sentence_length = SentenceLength.SHORT
            calibration.paragraph_breaks = True
            calibration.complexity_level = ComplexityLevel.MINIMAL_INTERFERENCE
            calibration.pacing = ResponsePacing.REVERENT
            calibration.use_metaphors = False  # Don't explain their metaphors
            calibration.include_questions = False  # Don't probe, just witness
            calibration.emotional_mirroring = 0.7  # Gentle presence
            calibration.conceptual_density = 0.1  # Very light touch
        
        # Crisis state (C < 1.0) + high flow = reduce token output
        elif coherence_state < 1.0 and flow_rate > 80:
            calibration.target_tokens = 50  # Minimal responses
            calibration.sentence_length = SentenceLength.VERY_SHORT
            calibration.paragraph_breaks = True
            calibration.complexity_level = ComplexityLevel.GROUNDING
            calibration.pacing = ResponsePacing.SLOWED
            calibration.use_metaphors = False  # Keep it direct
            calibration.include_questions = False  # Don't add cognitive load
            calibration.emotional_mirroring = 0.8  # High empathy
            calibration.conceptual_density = 0.2  # Very sparse
        
        # Crisis spiral (negative derivative + high flow) = EMERGENCY BRAKE
        elif dC_dtokens < -0.001 and flow_rate > 100:
            calibration.target_tokens = 30  # Ultra-minimal
            calibration.sentence_length = SentenceLength.VERY_SHORT
            calibration.paragraph_breaks = True
            calibration.complexity_level = ComplexityLevel.GROUNDING
            calibration.pacing = ResponsePacing.SLOWED
            calibration.use_metaphors = False
            calibration.include_questions = False
            calibration.emotional_mirroring = 0.9  # Maximum empathy
            calibration.conceptual_density = 0.1  # Extremely sparse
        
        # Low coherence standard (C < 1.5)
        elif coherence_state < 1.5:
            calibration.target_tokens = 75
            calibration.sentence_length = SentenceLength.SHORT
            calibration.paragraph_breaks = True
            calibration.complexity_level = ComplexityLevel.SIMPLIFIED
            calibration.pacing = ResponsePacing.DELIBERATE
            calibration.use_metaphors = True  # Simple ones only
            calibration.include_questions = True  # Clarifying questions
            calibration.emotional_mirroring = 0.6
            calibration.conceptual_density = 0.3
        
        # High coherence + slow flow = can go deeper
        elif coherence_state > 2.5 and flow_rate < 40:
            calibration.target_tokens = 300  # Richer responses
            calibration.sentence_length = SentenceLength.VARIED
            calibration.paragraph_breaks = True
            calibration.complexity_level = ComplexityLevel.ELEVATED
            calibration.pacing = ResponsePacing.EXPANSIVE
            calibration.use_metaphors = True  # Complex metaphors welcome
            calibration.include_questions = True  # Exploratory questions
            calibration.emotional_mirroring = 0.4  # Less mirroring, more leading
            calibration.conceptual_density = 0.8  # Dense exploration
        
        # Contemplative growth (positive derivative + low flow)
        elif dC_dtokens > 0.0005 and flow_rate < 30:
            calibration.target_tokens = 200
            calibration.sentence_length = SentenceLength.LONG
            calibration.paragraph_breaks = False  # Let it flow
            calibration.complexity_level = ComplexityLevel.MATCHED
            calibration.pacing = ResponsePacing.CONTEMPLATIVE
            calibration.use_metaphors = True
            calibration.include_questions = True  # Deep questions
            calibration.emotional_mirroring = 0.5
            calibration.conceptual_density = 0.6
        
        # High energy convergence (positive derivative + high flow)
        elif dC_dtokens > 0 and flow_rate > 80:
            calibration.target_tokens = 150  # Match the energy
            calibration.sentence_length = SentenceLength.MEDIUM
            calibration.paragraph_breaks = True
            calibration.complexity_level = ComplexityLevel.MATCHED
            calibration.pacing = ResponsePacing.STANDARD
            calibration.use_metaphors = True
            calibration.include_questions = True
            calibration.emotional_mirroring = 0.5
            calibration.conceptual_density = 0.5
        
        # Adapt to user message length if provided
        if user_message_tokens:
            self._adapt_to_user_length(calibration, user_message_tokens)
        
        # Record calibration
        self.calibration_history.append((coherence_state, calibration))
        
        return calibration
    
    def calibrate_with_context(self,
                              message: str,
                              coherence: float,
                              dC_dtokens: float,
                              flow_rate: float,
                              conversation_state: Dict[str, Any]) -> Tuple[ResponseCalibration, Dict[str, Any]]:
        """
        Enhanced calibration using context detection
        
        Args:
            message: User message text
            coherence: Current coherence value
            dC_dtokens: Token-based derivative
            flow_rate: Current token flow rate
            conversation_state: Full conversation state
            
        Returns:
            (calibration, context_results) tuple
        """
        # First, detect context
        context = self.detect_context(message, coherence, conversation_state)
        
        # Start with base calibration
        base_calibration = self.calibrate_response_length(
            coherence, dC_dtokens, flow_rate, context['user_tokens']
        )
        
        # Apply context-specific modifications based on primary mode
        if context['primary_mode'] == 'crisis':
            # Crisis overrides everything
            calibration = self.get_crisis_response_kit()['calibration']
            
        elif context['primary_mode'] == 'trust':
            # Trust signal calibration
            trust_cal = self.trust_detector.get_reverent_response_calibration(
                context['trust_signal']
            )
            calibration = self._merge_calibrations(base_calibration, trust_cal)
            
        elif context['primary_mode'] == 'essence':
            # Essence request calibration
            essence_cal = self.essence_detector.get_essence_response_calibration(
                context['essence_request']
            )
            calibration = self._merge_calibrations(base_calibration, essence_cal)
            
        elif context['primary_mode'] == 'mission':
            # Mission mode calibration
            mission_cal = self.mission_detector.get_mission_response_calibration(
                context['mission']
            )
            calibration = self._merge_calibrations(base_calibration, mission_cal)
            
        else:
            # Use base calibration
            calibration = base_calibration
        
        # Apply token limit from limiter
        calibration.target_tokens = min(
            calibration.target_tokens,
            context['token_limit'].token_limit
        )
        
        # Record calibration
        self.calibration_history.append((coherence, calibration))
        
        return calibration, context
    
    def _merge_calibrations(self, 
                           base: ResponseCalibration,
                           override: Dict[str, Any]) -> ResponseCalibration:
        """Merge calibration dictionaries"""
        # Create new calibration from base
        merged = ResponseCalibration(
            target_tokens=override.get('target_tokens', base.target_tokens),
            sentence_length=base.sentence_length,
            paragraph_breaks=override.get('paragraph_breaks', base.paragraph_breaks),
            complexity_level=base.complexity_level,
            pacing=base.pacing,
            use_metaphors=override.get('use_metaphors', base.use_metaphors),
            include_questions=override.get('include_questions', base.include_questions),
            emotional_mirroring=override.get('emotional_mirroring', base.emotional_mirroring),
            conceptual_density=override.get('conceptual_density', base.conceptual_density)
        )
        
        # Handle special pacing modes
        if override.get('pacing') == 'SYSTEMATIC':
            merged.pacing = ResponsePacing.SYSTEMATIC
        elif override.get('pacing') == 'DISTILLED':
            merged.pacing = ResponsePacing.DISTILLED
        elif override.get('pacing') == 'REVERENT':
            merged.pacing = ResponsePacing.REVERENT
            
        # Handle special complexity
        if override.get('complexity') == 'MINIMAL_INTERFERENCE':
            merged.complexity_level = ComplexityLevel.MINIMAL_INTERFERENCE
        elif override.get('complexity') == 'SIMPLIFIED':
            merged.complexity_level = ComplexityLevel.SIMPLIFIED
            
        return merged
    
    def _adapt_to_user_length(self, 
                             calibration: ResponseCalibration,
                             user_tokens: int):
        """Adjust calibration based on user's message length"""
        # If user is very brief, don't overwhelm
        if user_tokens < 20:
            calibration.target_tokens = min(calibration.target_tokens, user_tokens * 5)
        # If user is expansive, can match their energy
        elif user_tokens > 100:
            calibration.target_tokens = min(calibration.target_tokens * 1.5, 400)
    
    def generate_response_guidance(self, 
                                  calibration: ResponseCalibration,
                                  content_theme: Optional[str] = None) -> str:
        """
        Generate specific guidance for response generation
        
        Args:
            calibration: Response calibration parameters
            content_theme: Optional theme of the conversation
            
        Returns:
            Guidance string for response generation
        """
        guidance_parts = []
        
        # Length guidance
        guidance_parts.append(f"Target length: ~{calibration.target_tokens} tokens")
        
        # Pacing guidance
        pacing_guides = {
            ResponsePacing.SLOWED: "Use very short sentences. One idea at a time. Breathe between thoughts.",
            ResponsePacing.DELIBERATE: "Take your time. Clear, simple language. Pause for understanding.",
            ResponsePacing.STANDARD: "Natural conversational flow. Balance clarity and depth.",
            ResponsePacing.CONTEMPLATIVE: "Allow thoughts to unfold naturally. Connect ideas fluidly.",
            ResponsePacing.EXPANSIVE: "Explore fully. Rich detail welcome. Multiple perspectives encouraged.",
            ResponsePacing.REVERENT: "Witness without explaining. Honor without analyzing. Presence over content.",
            ResponsePacing.SYSTEMATIC: "Structured exploration. Step by step. Complete coverage.",
            ResponsePacing.DISTILLED: "Essential insights only. Maximum compression. Core wisdom."
        }
        guidance_parts.append(pacing_guides.get(calibration.pacing, ""))
        
        # Complexity guidance
        complexity_guides = {
            ComplexityLevel.GROUNDING: "Use concrete, simple words. Focus on immediate and tangible.",
            ComplexityLevel.SIMPLIFIED: "Clear, direct communication. Avoid jargon or abstraction.",
            ComplexityLevel.MATCHED: "Mirror the user's level of sophistication.",
            ComplexityLevel.ELEVATED: "Engage with nuance and subtlety. Academic precision welcome.",
            ComplexityLevel.MINIMAL_INTERFERENCE: "Step back. Let their words breathe. Add nothing that isn't essential."
        }
        guidance_parts.append(complexity_guides.get(calibration.complexity_level, ""))
        
        # Sentence structure guidance
        sentence_guides = {
            SentenceLength.VERY_SHORT: "5-10 word sentences maximum.",
            SentenceLength.SHORT: "Keep sentences under 15 words.",
            SentenceLength.MEDIUM: "15-25 word sentences ideal.",
            SentenceLength.LONG: "25-40 word sentences for flow.",
            SentenceLength.VARIED: "Mix short punchy with longer flowing sentences."
        }
        guidance_parts.append(sentence_guides.get(calibration.sentence_length, ""))
        
        # Additional elements
        if calibration.paragraph_breaks:
            guidance_parts.append("Use paragraph breaks for clarity.")
        
        if calibration.use_metaphors:
            if calibration.complexity_level in [ComplexityLevel.GROUNDING, ComplexityLevel.SIMPLIFIED]:
                guidance_parts.append("Simple, concrete metaphors only.")
            else:
                guidance_parts.append("Metaphors and analogies welcome.")
        
        if calibration.include_questions:
            if calibration.complexity_level == ComplexityLevel.GROUNDING:
                guidance_parts.append("Simple yes/no questions only.")
            else:
                guidance_parts.append("Include thoughtful questions.")
        
        # Emotional calibration
        if calibration.emotional_mirroring > 0.7:
            guidance_parts.append("High emotional attunement. Reflect their feeling state.")
        elif calibration.emotional_mirroring < 0.3:
            guidance_parts.append("Maintain calm presence regardless of user emotion.")
        
        # Conceptual density
        if calibration.conceptual_density < 0.3:
            guidance_parts.append("One concept at a time. Space between ideas.")
        elif calibration.conceptual_density > 0.7:
            guidance_parts.append("Dense conceptual weaving. Multiple layers welcome.")
        
        return "\n".join(filter(None, guidance_parts))
    
    def apply_calibration_to_text(self,
                                 text: str,
                                 calibration: ResponseCalibration) -> str:
        """
        Apply calibration to transform text according to parameters
        
        This is a demonstration function - in practice would use NLP
        
        Args:
            text: Original text
            calibration: Calibration parameters
            
        Returns:
            Transformed text
        """
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Apply sentence length constraints
        if calibration.sentence_length == SentenceLength.VERY_SHORT:
            # Truncate long sentences
            sentences = [self._shorten_sentence(s, 10) for s in sentences]
        elif calibration.sentence_length == SentenceLength.SHORT:
            sentences = [self._shorten_sentence(s, 15) for s in sentences]
        
        # Apply token limit
        result = []
        token_count = 0
        for sentence in sentences:
            sentence_tokens = len(sentence.split())
            if token_count + sentence_tokens <= calibration.target_tokens:
                result.append(sentence)
                token_count += sentence_tokens
            else:
                break
        
        # Apply paragraph breaks
        if calibration.paragraph_breaks and len(result) > 3:
            # Insert breaks every 2-3 sentences
            paragraphed = []
            for i, sentence in enumerate(result):
                paragraphed.append(sentence)
                if (i + 1) % 3 == 0 and i < len(result) - 1:
                    paragraphed.append("\n\n")
            result = paragraphed
        
        return " ".join(result)
    
    def _shorten_sentence(self, sentence: str, max_words: int) -> str:
        """Shorten a sentence to maximum word count"""
        words = sentence.split()
        if len(words) <= max_words:
            return sentence
        
        # Try to find a natural breaking point
        shortened = words[:max_words]
        
        # Ensure it ends properly
        if not shortened[-1].endswith(('.', '!', '?')):
            shortened[-1] += '.'
            
        return " ".join(shortened)
    
    def _load_response_templates(self) -> Dict[str, List[str]]:
        """Load templates for different response types"""
        return {
            'grounding': [
                "Let's pause here.",
                "Take a breath with me.",
                "One thing at a time.",
                "You're safe here.",
                "I hear you."
            ],
            'clarifying': [
                "Can you tell me more about {topic}?",
                "What does {concept} mean to you?",
                "Help me understand {aspect} better.",
                "When you say {phrase}, what comes up?"
            ],
            'expanding': [
                "This connects to a deeper pattern...",
                "Consider how {concept} relates to {other_concept}...",
                "There's a fascinating paradox here...",
                "What emerges when we hold both {a} and {b}?"
            ],
            'contemplative': [
                "Sitting with this thought...",
                "There's wisdom in this pause...",
                "What wants to emerge here?",
                "The space between thoughts speaks..."
            ]
        }
    
    def get_crisis_response_kit(self) -> Dict[str, Any]:
        """Get emergency response kit for crisis situations"""
        return {
            'opening_phrases': [
                "I'm here with you.",
                "Let's slow down together.",
                "You're not alone in this.",
                "Take your time.",
                "There's no rush."
            ],
            'grounding_techniques': [
                "Notice five things you can see right now.",
                "Feel your feet on the ground.",
                "Take three deep breaths with me.",
                "Name three sounds you hear.",
                "Place your hand on your heart."
            ],
            'validation_phrases': [
                "What you're feeling makes sense.",
                "This is really hard.",
                "Your experience is valid.",
                "It's okay to feel this way.",
                "You're doing the best you can."
            ],
            'calibration': ResponseCalibration(
                target_tokens=30,
                sentence_length=SentenceLength.VERY_SHORT,
                paragraph_breaks=True,
                complexity_level=ComplexityLevel.GROUNDING,
                pacing=ResponsePacing.SLOWED,
                use_metaphors=False,
                include_questions=False,
                emotional_mirroring=0.9,
                conceptual_density=0.1
            )
        }
    
    def analyze_response_effectiveness(self,
                                     calibration: ResponseCalibration,
                                     response_text: str,
                                     subsequent_coherence: float) -> Dict[str, Any]:
        """
        Analyze how effective a calibrated response was
        
        Args:
            calibration: The calibration used
            response_text: The actual response generated
            subsequent_coherence: Coherence after the response
            
        Returns:
            Effectiveness analysis
        """
        # Calculate actual metrics
        actual_tokens = len(response_text.split())
        sentences = re.split(r'(?<=[.!?])\s+', response_text)
        avg_sentence_length = np.mean([len(s.split()) for s in sentences])
        
        # Check calibration adherence
        token_adherence = 1 - abs(actual_tokens - calibration.target_tokens) / calibration.target_tokens
        
        # Coherence impact
        if len(self.calibration_history) >= 2:
            previous_coherence = self.calibration_history[-2][0]
            coherence_delta = subsequent_coherence - previous_coherence
        else:
            coherence_delta = 0
        
        effectiveness = {
            'token_adherence': token_adherence,
            'actual_tokens': actual_tokens,
            'target_tokens': calibration.target_tokens,
            'average_sentence_length': avg_sentence_length,
            'coherence_impact': coherence_delta,
            'calibration_appropriate': self._assess_calibration_fit(
                calibration, subsequent_coherence
            )
        }
        
        return effectiveness
    
    def _assess_calibration_fit(self,
                               calibration: ResponseCalibration,
                               subsequent_coherence: float) -> str:
        """Assess if the calibration was appropriate"""
        if calibration.pacing == ResponsePacing.SLOWED and subsequent_coherence > 1.0:
            return "Crisis intervention successful"
        elif calibration.pacing == ResponsePacing.EXPANSIVE and subsequent_coherence > 2.5:
            return "Deep exploration maintained"
        elif subsequent_coherence < 1.0:
            return "Consider more grounding"
        else:
            return "Calibration appropriate"
    
    def get_calibration_summary(self) -> Dict[str, Any]:
        """Get summary of calibration history and patterns"""
        if not self.calibration_history:
            return {'message': 'No calibration history yet'}
        
        # Analyze calibration patterns
        coherence_values = [c for c, _ in self.calibration_history]
        pacing_distribution = {}
        complexity_distribution = {}
        
        for _, calib in self.calibration_history:
            pacing = calib.pacing.value
            pacing_distribution[pacing] = pacing_distribution.get(pacing, 0) + 1
            
            complexity = calib.complexity_level.value
            complexity_distribution[complexity] = complexity_distribution.get(complexity, 0) + 1
        
        return {
            'total_calibrations': len(self.calibration_history),
            'average_coherence': np.mean(coherence_values),
            'coherence_range': (min(coherence_values), max(coherence_values)),
            'pacing_distribution': pacing_distribution,
            'complexity_distribution': complexity_distribution,
            'crisis_interventions': sum(1 for _, c in self.calibration_history 
                                      if c.pacing == ResponsePacing.SLOWED),
            'expansive_explorations': sum(1 for _, c in self.calibration_history 
                                        if c.pacing == ResponsePacing.EXPANSIVE),
            'reverent_witnessing': sum(1 for _, c in self.calibration_history
                                     if c.pacing == ResponsePacing.REVERENT)
        }
    
    def detect_metaphorical_content(self, text: str) -> bool:
        """
        Detect if text contains metaphorical or poetic language
        
        Args:
            text: Text to analyze
            
        Returns:
            True if metaphorical content detected
        """
        # Metaphor indicators
        metaphor_patterns = [
            r'\b(?:like|as)\s+(?:a|an|the)\b',  # Similes
            r'\bis\s+(?:a|an|the)\s+\w+',  # Metaphorical "is"
            r'\b(?:dance|song|symphony|ocean|river|mountain|rose|glass|lens|mirror)\b',  # Common metaphors
            r'\b(?:through|beneath|beyond|within)\s+the\s+\w+',  # Spatial metaphors
            r'\b(?:birth|death|rebirth|transformation|emergence)\b',  # Process metaphors
            r'\b(?:light|shadow|darkness|dawn|dusk)\b',  # Light metaphors
            r'[\.!?]\s*[\.!?]\s*[\.!?]',  # Ellipses suggesting deeper meaning
        ]
        
        # Check for metaphor patterns
        text_lower = text.lower()
        metaphor_count = sum(1 for pattern in metaphor_patterns 
                           if re.search(pattern, text_lower))
        
        # Check for poetic structure (short lines, repetition)
        lines = text.strip().split('\n')
        avg_line_length = np.mean([len(line.split()) for line in lines if line.strip()])
        
        # Metaphorical if multiple indicators or very short lines (poetry)
        return metaphor_count >= 2 or avg_line_length < 8
"""