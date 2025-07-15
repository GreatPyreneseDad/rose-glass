#!/usr/bin/env python3
"""
AI-Powered Fear Detector using Ollama
Analyzes dialogue, monologue, and written communication to detect underlying fears
"""
import json
import requests
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import re


@dataclass
class DetectedFear:
    """Represents a fear detected by AI analysis"""
    fear_type: str
    confidence: float  # 0.0 to 1.0
    evidence: List[str]  # Specific quotes/phrases that indicate this fear
    depth_estimate: float  # 0.0 to 1.0
    context: str  # Additional context about the fear


@dataclass
class CommunicationAnalysis:
    """Complete analysis of a communication sample"""
    detected_fears: List[DetectedFear]
    emotional_tone: str
    fear_indicators: List[str]
    suggested_approach: str
    raw_analysis: str  # Full AI response


class AIFearDetector:
    """
    Uses Ollama LLM to detect fears in written communication.
    Analyzes dialogue, monologue, emails, texts, etc.
    """
    
    def __init__(self, model: str = "llama3.2:latest"):
        self.model = model
        self.ollama_url = "http://localhost:11434/api/generate"
        
        # Fear detection prompt template
        self.system_prompt = """You are an expert psychologist specializing in fear analysis and depth psychology. Your task is to analyze written communication to detect underlying fears.

When analyzing text, look for:
1. Explicit fear statements ("I'm afraid of...", "I worry about...")
2. Implicit fear indicators (avoidance language, anxiety markers, defensive statements)
3. Emotional subtext revealing hidden fears
4. Patterns of concern or preoccupation
5. Metaphors and imagery suggesting fear

For each fear detected, identify:
- The core fear archetype it represents
- Specific evidence (quotes) from the text
- Estimated depth (0.0-1.0, where 1.0 is deepest/most fundamental)
- Confidence level in your detection (0.0-1.0)

Common fear archetypes to consider:
- Identity Dissolution: Fear of losing sense of self
- Existential Void: Fear of meaninglessness
- Connection Loss: Fear of abandonment/isolation
- Purpose Absence: Fear of wasted life
- Mortality Terror: Fear of death
- Control Loss: Fear of chaos/uncertainty
- Inadequacy Shadow: Fear of not being enough
- Truth Revelation: Fear of being truly seen

Respond in JSON format."""

        # Conversation analysis prompt
        self.conversation_prompt = """Analyze this conversation/dialogue for underlying fears. Pay special attention to:
- What each speaker avoids saying
- Defensive responses
- Topic changes that suggest discomfort
- Emotional reactions
- Power dynamics revealing insecurities

For dialogue, note which speaker exhibits which fears."""

    def analyze_text(self, text: str, is_dialogue: bool = False) -> CommunicationAnalysis:
        """
        Analyze text for underlying fears using AI.
        
        Args:
            text: The communication to analyze
            is_dialogue: Whether this is a dialogue between multiple people
            
        Returns:
            Complete analysis with detected fears
        """
        # Prepare the prompt
        analysis_type = "dialogue" if is_dialogue else "monologue/written communication"
        
        prompt = f"""{self.system_prompt}

{self.conversation_prompt if is_dialogue else ''}

Analyze the following {analysis_type} for underlying fears:

---
{text}
---

Provide a comprehensive analysis in JSON format with the following structure:
{{
    "detected_fears": [
        {{
            "fear_type": "archetype name",
            "confidence": 0.0-1.0,
            "evidence": ["quote 1", "quote 2"],
            "depth_estimate": 0.0-1.0,
            "context": "explanation"
        }}
    ],
    "emotional_tone": "overall emotional state",
    "fear_indicators": ["indicator 1", "indicator 2"],
    "suggested_approach": "therapeutic or self-help approach"
}}"""

        try:
            # Call Ollama API
            response = requests.post(self.ollama_url, json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "temperature": 0.7,
                "max_tokens": 1500
            })
            
            if response.status_code == 200:
                result = response.json()
                ai_response = result.get('response', '')
                
                # Parse the JSON response
                analysis = self._parse_ai_response(ai_response)
                analysis.raw_analysis = ai_response
                
                return analysis
            else:
                return self._fallback_analysis(text)
                
        except Exception as e:
            print(f"AI analysis error: {e}")
            return self._fallback_analysis(text)
    
    def _parse_ai_response(self, response: str) -> CommunicationAnalysis:
        """Parse the AI's JSON response into our data structure."""
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = {}
            
            # Parse detected fears
            detected_fears = []
            for fear_data in data.get('detected_fears', []):
                detected_fears.append(DetectedFear(
                    fear_type=fear_data.get('fear_type', 'unknown'),
                    confidence=float(fear_data.get('confidence', 0.5)),
                    evidence=fear_data.get('evidence', []),
                    depth_estimate=float(fear_data.get('depth_estimate', 0.5)),
                    context=fear_data.get('context', '')
                ))
            
            return CommunicationAnalysis(
                detected_fears=detected_fears,
                emotional_tone=data.get('emotional_tone', 'neutral'),
                fear_indicators=data.get('fear_indicators', []),
                suggested_approach=data.get('suggested_approach', ''),
                raw_analysis=response
            )
            
        except Exception as e:
            print(f"Parse error: {e}")
            return CommunicationAnalysis(
                detected_fears=[],
                emotional_tone="unknown",
                fear_indicators=[],
                suggested_approach="",
                raw_analysis=response
            )
    
    def _fallback_analysis(self, text: str) -> CommunicationAnalysis:
        """Fallback pattern-based analysis if AI fails."""
        detected_fears = []
        fear_indicators = []
        
        # Simple pattern matching
        text_lower = text.lower()
        
        # Fear keywords and their mappings
        fear_patterns = {
            'identity': ['who am i', 'losing myself', 'identity', 'don\'t know myself'],
            'abandonment': ['alone', 'lonely', 'abandoned', 'leave me', 'rejection'],
            'failure': ['not good enough', 'failure', 'inadequate', 'worthless'],
            'death': ['dying', 'death', 'mortality', 'end of life'],
            'meaningless': ['pointless', 'no meaning', 'why bother', 'doesn\'t matter']
        }
        
        for fear_type, patterns in fear_patterns.items():
            matches = [p for p in patterns if p in text_lower]
            if matches:
                detected_fears.append(DetectedFear(
                    fear_type=fear_type,
                    confidence=0.6,
                    evidence=matches,
                    depth_estimate=0.5,
                    context="Pattern-based detection"
                ))
                fear_indicators.extend(matches)
        
        return CommunicationAnalysis(
            detected_fears=detected_fears,
            emotional_tone="uncertain",
            fear_indicators=fear_indicators,
            suggested_approach="Consider deeper analysis with AI",
            raw_analysis="Fallback analysis used"
        )
    
    def analyze_conversation_dynamics(self, dialogue: List[Dict[str, str]]) -> Dict:
        """
        Analyze fear dynamics in a conversation.
        
        Args:
            dialogue: List of {'speaker': 'name', 'text': 'what they said'}
            
        Returns:
            Analysis of fear dynamics between speakers
        """
        # Format dialogue for analysis
        formatted_dialogue = "\n".join([
            f"{turn['speaker']}: {turn['text']}" 
            for turn in dialogue
        ])
        
        # Analyze as dialogue
        analysis = self.analyze_text(formatted_dialogue, is_dialogue=True)
        
        # Additional dynamics analysis
        dynamics_prompt = f"""Analyze the fear dynamics between speakers in this conversation:

{formatted_dialogue}

Focus on:
1. How fears manifest differently in each speaker
2. Fear-based interaction patterns
3. Defensive or avoidant communication
4. Power dynamics related to fear

Respond with insights about the relational fear dynamics."""

        try:
            response = requests.post(self.ollama_url, json={
                "model": self.model,
                "prompt": dynamics_prompt,
                "stream": False,
                "temperature": 0.7
            })
            
            if response.status_code == 200:
                dynamics_insight = response.json().get('response', '')
            else:
                dynamics_insight = "Unable to analyze dynamics"
                
        except:
            dynamics_insight = "Error analyzing dynamics"
        
        return {
            'analysis': analysis,
            'dynamics': dynamics_insight,
            'speaker_count': len(set(turn['speaker'] for turn in dialogue))
        }
    
    def suggest_fear_exploration(self, detected_fear: DetectedFear) -> Dict:
        """
        Generate specific exploration suggestions for a detected fear.
        """
        prompt = f"""Given this detected fear:
Type: {detected_fear.fear_type}
Evidence: {', '.join(detected_fear.evidence[:3])}
Depth: {detected_fear.depth_estimate}
Context: {detected_fear.context}

Suggest:
1. Three clarifying questions to explore this fear deeper
2. A simple exercise to begin working with this fear
3. An affirmation or reframe to consider

Be specific and practical."""

        try:
            response = requests.post(self.ollama_url, json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "temperature": 0.8,
                "max_tokens": 500
            })
            
            if response.status_code == 200:
                suggestions = response.json().get('response', '')
                
                # Parse suggestions
                questions = re.findall(r'(?:1\.|question).*?:(.*?)(?=\d\.|$)', suggestions, re.IGNORECASE | re.DOTALL)
                exercise = re.search(r'(?:2\.|exercise).*?:(.*?)(?=\d\.|$)', suggestions, re.IGNORECASE | re.DOTALL)
                affirmation = re.search(r'(?:3\.|affirmation|reframe).*?:(.*?)$', suggestions, re.IGNORECASE | re.DOTALL)
                
                return {
                    'questions': [q.strip() for q in questions] if questions else ["What does this fear protect you from?"],
                    'exercise': exercise.group(1).strip() if exercise else "Sit with this fear for 5 minutes without trying to change it.",
                    'affirmation': affirmation.group(1).strip() if affirmation else "This fear is a teacher, not an enemy.",
                    'raw_suggestions': suggestions
                }
            else:
                return self._default_suggestions(detected_fear)
                
        except Exception as e:
            return self._default_suggestions(detected_fear)
    
    def _default_suggestions(self, detected_fear: DetectedFear) -> Dict:
        """Default suggestions if AI fails."""
        return {
            'questions': [
                "When did you first notice this fear?",
                "What would happen if this fear came true?",
                "What does this fear protect you from?"
            ],
            'exercise': "Write about this fear for 10 minutes without stopping.",
            'affirmation': "I am learning from this fear.",
            'raw_suggestions': "Default suggestions"
        }