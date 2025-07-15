#!/usr/bin/env python3
"""
Fear Analyzer Agent - Identifies and analyzes fear patterns
Maps user inputs to deep fear archetypes and provides descent guidance
"""
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import re
import json
from collections import defaultdict


@dataclass(frozen=True)
class FearPattern:
    """Represents a recognized fear pattern"""
    pattern_type: str
    depth: float  # 0.0 to 1.0
    frequency: str  # 'high', 'medium', 'low'
    description: str
    keywords: Tuple[str, ...]
    transformative_potential: float
    guidance: str


@dataclass
class AnalysisResult:
    """Result of fear analysis"""
    identified_fears: List[FearPattern]
    primary_fear: Optional[FearPattern]
    fear_landscape: Dict[str, float]
    recommended_approach: str
    warnings: List[str]
    coherence_risk: float


class FearAnalyzer:
    """
    Analyzes user inputs to identify deep fear patterns.
    Maps surface anxieties to archetypal fears.
    """
    
    def __init__(self):
        # Core fear patterns database
        self.fear_patterns = {
            "identity_dissolution": FearPattern(
                pattern_type="identity_dissolution",
                depth=0.9,
                frequency="high",
                description="Fear of losing sense of self, ego death, identity crisis",
                keywords=("who am i", "losing myself", "identity", "ego", "dissolve", "disappear", "nobody", "forgotten"),
                transformative_potential=0.95,
                guidance="This fear guards the doorway to true self. Embrace dissolution to find what cannot be dissolved."
            ),
            
            "existential_void": FearPattern(
                pattern_type="existential_void",
                depth=1.0,
                frequency="medium",
                description="Fear of meaninglessness, cosmic insignificance, existential dread",
                keywords=("meaningless", "pointless", "void", "nothing matters", "why exist", "cosmic", "insignificant", "absurd"),
                transformative_potential=1.0,
                guidance="The void is not empty - it is pregnant with all possibility. Dive deep to find the ground of being."
            ),
            
            "connection_loss": FearPattern(
                pattern_type="connection_loss",
                depth=0.8,
                frequency="high", 
                description="Fear of abandonment, isolation, severed connections",
                keywords=("alone", "abandoned", "isolated", "rejected", "unloved", "disconnected", "lonely", "left behind"),
                transformative_potential=0.85,
                guidance="True connection begins with embracing aloneness. In solitude, find the thread that connects all."
            ),
            
            "purpose_absence": FearPattern(
                pattern_type="purpose_absence",
                depth=0.85,
                frequency="medium",
                description="Fear of purposelessness, wasted life, unfulfilled potential",
                keywords=("purpose", "wasted", "potential", "direction", "lost", "aimless", "unfulfilled", "meant to"),
                transformative_potential=0.9,
                guidance="Purpose emerges from purposelessness. Release the need to know, and knowing will find you."
            ),
            
            "mortality_terror": FearPattern(
                pattern_type="mortality_terror",
                depth=0.95,
                frequency="high",
                description="Fear of death, annihilation, non-existence",
                keywords=("death", "die", "mortality", "finite", "ending", "cease", "extinct", "terminal"),
                transformative_potential=0.98,
                guidance="Death is the ultimate teacher. Face it fully to discover what in you cannot die."
            ),
            
            "control_loss": FearPattern(
                pattern_type="control_loss",
                depth=0.75,
                frequency="high",
                description="Fear of chaos, uncertainty, loss of control",
                keywords=("control", "chaos", "uncertain", "unpredictable", "powerless", "helpless", "random", "out of control"),
                transformative_potential=0.8,
                guidance="Control is illusion. Surrender to the flow and find power in powerlessness."
            ),
            
            "inadequacy_shadow": FearPattern(
                pattern_type="inadequacy_shadow",
                depth=0.7,
                frequency="high",
                description="Fear of not being enough, fundamental unworthiness",
                keywords=("not enough", "inadequate", "worthless", "failure", "imposter", "fraud", "inferior", "broken"),
                transformative_potential=0.75,
                guidance="Your inadequacy is the crack where light enters. Embrace imperfection as the path to wholeness."
            ),
            
            "truth_revelation": FearPattern(
                pattern_type="truth_revelation",
                depth=0.88,
                frequency="low",
                description="Fear of seeing/being seen truly, exposure of authentic self",
                keywords=("truth", "exposed", "seen", "revealed", "authentic", "naked", "vulnerable", "real self"),
                transformative_potential=0.92,
                guidance="Truth burns away all that is false. Let yourself be seen to see yourself clearly."
            )
        }
        
        # Surface to deep fear mapping
        self.surface_to_deep_map = {
            "job loss": ["purpose_absence", "inadequacy_shadow", "control_loss"],
            "relationship": ["connection_loss", "inadequacy_shadow", "truth_revelation"],
            "health": ["mortality_terror", "control_loss", "identity_dissolution"],
            "money": ["control_loss", "inadequacy_shadow", "purpose_absence"],
            "future": ["existential_void", "control_loss", "mortality_terror"],
            "past": ["identity_dissolution", "truth_revelation", "inadequacy_shadow"],
            "social": ["connection_loss", "truth_revelation", "inadequacy_shadow"],
            "performance": ["inadequacy_shadow", "purpose_absence", "truth_revelation"]
        }
        
        self.analysis_history = []
        
    def analyze_fear(self, user_input: str, context: Optional[Dict] = None) -> AnalysisResult:
        """
        Analyze user input to identify fear patterns.
        
        Args:
            user_input: Raw user description of fears/anxieties
            context: Optional context about user state
            
        Returns:
            Comprehensive fear analysis result
        """
        # Normalize input
        normalized_input = user_input.lower()
        
        # Identify fear patterns
        identified_fears = self._identify_patterns(normalized_input)
        
        # Map surface fears to deep patterns
        deep_fears = self._map_to_deep_fears(normalized_input, identified_fears)
        
        # Build fear landscape
        fear_landscape = self._build_fear_landscape(identified_fears + deep_fears)
        
        # Determine primary fear
        primary_fear = self._determine_primary_fear(identified_fears + deep_fears)
        
        # Generate approach recommendation
        approach = self._recommend_approach(primary_fear, fear_landscape)
        
        # Assess risks and warnings
        warnings, coherence_risk = self._assess_risks(fear_landscape, context)
        
        result = AnalysisResult(
            identified_fears=list(set(identified_fears + deep_fears)),
            primary_fear=primary_fear,
            fear_landscape=fear_landscape,
            recommended_approach=approach,
            warnings=warnings,
            coherence_risk=coherence_risk
        )
        
        self.analysis_history.append({
            'timestamp': datetime.now(),
            'input': user_input,
            'result': result
        })
        
        return result
        
    def _identify_patterns(self, text: str) -> List[FearPattern]:
        """Identify fear patterns based on keywords."""
        identified = []
        
        for pattern_name, pattern in self.fear_patterns.items():
            # Check for keyword matches
            matches = sum(1 for keyword in pattern.keywords if keyword in text)
            
            # If significant match, include pattern
            if matches >= 1:
                identified.append(pattern)
                
        return identified
        
    def _map_to_deep_fears(self, text: str, already_identified: List[FearPattern]) -> List[FearPattern]:
        """Map surface-level fears to deeper patterns."""
        deep_fears = []
        identified_types = {f.pattern_type for f in already_identified}
        
        # Check for surface fear indicators
        for surface, deep_list in self.surface_to_deep_map.items():
            if surface in text:
                for deep_pattern_name in deep_list:
                    if deep_pattern_name not in identified_types:
                        deep_fears.append(self.fear_patterns[deep_pattern_name])
                        identified_types.add(deep_pattern_name)
                        
        return deep_fears
        
    def _build_fear_landscape(self, fears: List[FearPattern]) -> Dict[str, float]:
        """Build a map of fear types to their intensities."""
        landscape = defaultdict(float)
        
        for fear in fears:
            # Use depth as base intensity
            landscape[fear.pattern_type] = max(
                landscape[fear.pattern_type],
                fear.depth
            )
            
        return dict(landscape)
        
    def _determine_primary_fear(self, fears: List[FearPattern]) -> Optional[FearPattern]:
        """Determine the primary/deepest fear to address."""
        if not fears:
            return None
            
        # Sort by transformative potential and depth
        sorted_fears = sorted(
            fears,
            key=lambda f: (f.transformative_potential * f.depth),
            reverse=True
        )
        
        return sorted_fears[0]
        
    def _recommend_approach(self, primary_fear: Optional[FearPattern], 
                          landscape: Dict[str, float]) -> str:
        """Generate approach recommendation based on fear analysis."""
        if not primary_fear:
            return "Begin with gentle self-inquiry. What stirs beneath the surface of your awareness?"
            
        if primary_fear.depth > 0.9:
            return f"Deep fear detected: {primary_fear.pattern_type}. Approach with reverence. Build coherence before descent. {primary_fear.guidance}"
            
        elif len(landscape) > 3:
            return "Multiple fear patterns active. Focus on one at a time. Start with the fear that feels most alive right now."
            
        else:
            return f"Primary fear: {primary_fear.pattern_type}. {primary_fear.guidance} Begin descent when breathing is stable."
            
    def _assess_risks(self, landscape: Dict[str, float], 
                     context: Optional[Dict]) -> Tuple[List[str], float]:
        """Assess risks and generate warnings."""
        warnings = []
        risk_score = 0.0
        
        # Check for high-depth fears
        deep_fears = [f for f, d in landscape.items() if d > 0.85]
        if len(deep_fears) > 2:
            warnings.append("Multiple deep fears active. Consider working with a guide.")
            risk_score += 0.3
            
        # Check for existential void without preparation
        if "existential_void" in landscape and len(landscape) == 1:
            warnings.append("Existential void detected as sole fear. Ensure strong coherence base.")
            risk_score += 0.2
            
        # Check context for additional risk factors
        if context:
            if context.get('coherence', 1.0) < 0.5:
                warnings.append("Low coherence detected. Stabilize before deep descent.")
                risk_score += 0.4
                
            if context.get('recent_crisis', False):
                warnings.append("Recent crisis noted. Allow integration time before new descent.")
                risk_score += 0.2
                
        # Cap risk score
        risk_score = min(risk_score, 1.0)
        
        return warnings, risk_score
        
    def get_fear_info(self, fear_type: str) -> Optional[FearPattern]:
        """Get detailed information about a specific fear pattern."""
        return self.fear_patterns.get(fear_type)
        
    def suggest_integration_practices(self, fear_type: str) -> List[str]:
        """Suggest practices for integrating specific fears."""
        practices = {
            "identity_dissolution": [
                "Mirror gazing meditation - 20 minutes",
                "Write 'I am' statements, then cross each out",
                "Practice saying 'I don't know who I am' with peace"
            ],
            "existential_void": [
                "Sit with the question 'Why?' without seeking answers",
                "Contemplate the vastness of space for 15 minutes",
                "Journal on 'What if nothing matters?' as liberation"
            ],
            "connection_loss": [
                "Practice being alone without distraction for 1 hour",
                "Write a letter to your loneliness as a friend",
                "Meditate on the space between breaths"
            ],
            "mortality_terror": [
                "Contemplate a photo of yourself as a child",
                "Write your own eulogy with love",
                "Breathe as if each breath could be the last"
            ]
        }
        
        return practices.get(fear_type, ["Sit with the fear. Breathe. Notice what arises."])
        
    def export_analysis(self) -> Dict:
        """Export analysis history and patterns."""
        return {
            'total_analyses': len(self.analysis_history),
            'fear_frequency': self._calculate_fear_frequency(),
            'common_patterns': self._identify_common_patterns(),
            'depth_progression': self._track_depth_progression()
        }
        
    def _calculate_fear_frequency(self) -> Dict[str, int]:
        """Calculate frequency of each fear type in history."""
        frequency = defaultdict(int)
        for analysis in self.analysis_history:
            for fear in analysis['result'].identified_fears:
                frequency[fear.pattern_type] += 1
        return dict(frequency)
        
    def _identify_common_patterns(self) -> List[Tuple[str, str]]:
        """Identify commonly co-occurring fears."""
        pairs = defaultdict(int)
        for analysis in self.analysis_history:
            fears = [f.pattern_type for f in analysis['result'].identified_fears]
            for i in range(len(fears)):
                for j in range(i + 1, len(fears)):
                    pair = tuple(sorted([fears[i], fears[j]]))
                    pairs[pair] += 1
                    
        return sorted(pairs.items(), key=lambda x: x[1], reverse=True)[:5]
        
    def _track_depth_progression(self) -> List[float]:
        """Track how fear depths progress over time."""
        depths = []
        for analysis in self.analysis_history:
            if analysis['result'].primary_fear:
                depths.append(analysis['result'].primary_fear.depth)
        return depths