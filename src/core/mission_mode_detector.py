"""
Mission Mode Detector
====================

Detects research/investigative tasks that require comprehensive exploration
rather than coherence-based response calibration. When in mission mode,
the system should prioritize thoroughness over brevity.

Pattern: Research requests, investigation tasks, comprehensive queries
Response: Full exploration mode regardless of coherence state

Author: Christopher MacGregor bin Joseph
Date: October 2025
"""

from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import re


class MissionType(Enum):
    """Types of missions detected"""
    RESEARCH = "research"  # "Research X", "Investigate Y"
    ANALYSIS = "analysis"  # "Analyze", "Examine", "Study"
    EXPLORATION = "exploration"  # "Explore", "Discover", "Find"
    COMPILATION = "compilation"  # "List", "Gather", "Compile"
    COMPARISON = "comparison"  # "Compare", "Contrast", "Evaluate"
    IMPLEMENTATION = "implementation"  # "Implement", "Build", "Create"


@dataclass
class Mission:
    """Detected mission parameters"""
    mission_type: MissionType
    scope: str  # 'comprehensive', 'focused', 'quick'
    topics: List[str]
    confidence: float
    estimated_tokens: int
    requires_structure: bool


class MissionModeDetector:
    """Detect research/investigative requests requiring full exploration"""
    
    def __init__(self):
        """Initialize mission detection patterns"""
        # Action verbs that indicate missions
        self.mission_verbs = {
            MissionType.RESEARCH: [
                'research', 'investigate', 'study', 'examine',
                'look into', 'find out about', 'learn about'
            ],
            MissionType.ANALYSIS: [
                'analyze', 'analyse', 'break down', 'examine',
                'evaluate', 'assess', 'review', 'interpret'
            ],
            MissionType.EXPLORATION: [
                'explore', 'discover', 'uncover', 'identify',
                'find', 'search for', 'look for'
            ],
            MissionType.COMPILATION: [
                'list', 'compile', 'gather', 'collect',
                'summarize', 'enumerate', 'catalog', 'document'
            ],
            MissionType.COMPARISON: [
                'compare', 'contrast', 'differentiate', 'distinguish',
                'weigh', 'evaluate between', 'choose between'
            ],
            MissionType.IMPLEMENTATION: [
                'implement', 'build', 'create', 'develop',
                'design', 'code', 'write', 'construct'
            ]
        }
        
        # Scope indicators
        self.scope_patterns = {
            'comprehensive': [
                r'\b(?:comprehensive|complete|thorough|detailed|full|entire)\b',
                r'\b(?:everything|all|every|each)\b',
                r'\b(?:in[- ]depth|deep|extensive)\b',
                r'\bstep[- ]by[- ]step\b'
            ],
            'focused': [
                r'\b(?:specific|particular|certain|key|main|primary)\b',
                r'\b(?:aspect|part|component|element)\b',
                r'\b(?:focus(?:ed)?|concentrate)\b'
            ],
            'quick': [
                r'\b(?:quick|brief|short|simple|basic)\b',
                r'\b(?:overview|summary|outline)\b',
                r'\b(?:just|only|simply)\b'
            ]
        }
        
        # Structure indicators
        self.structure_patterns = [
            r'\b(?:step[- ]by[- ]step|steps?)\b',
            r'\b(?:section|part|chapter|module)\b',
            r'\b(?:organize|structure|format)\b',
            r'\b(?:outline|framework|plan)\b',
            r'\b\d+\..*\d+\.',  # Numbered lists
            r'\b(?:first|second|third|finally)\b',
            r'\b(?:bullet|point|item)\b'
        ]
        
    def detect_mission(self, message: str) -> Optional[Mission]:
        """
        Detect if message contains a mission request
        
        Args:
            message: User message text
            
        Returns:
            Mission object if detected, None otherwise
        """
        message_lower = message.lower()
        
        # Check for mission verbs
        detected_types = []
        for mission_type, verbs in self.mission_verbs.items():
            for verb in verbs:
                pattern = r'\b' + verb.replace(' ', r'\s+') + r'\b'
                if re.search(pattern, message_lower):
                    detected_types.append((mission_type, verb))
        
        if not detected_types:
            return None
        
        # Determine primary mission type
        mission_type, trigger_verb = detected_types[0]
        
        # Determine scope
        scope = self._determine_scope(message_lower)
        
        # Extract topics
        topics = self._extract_topics(message, trigger_verb)
        
        # Check for structure requirements
        requires_structure = self._requires_structure(message_lower)
        
        # Calculate confidence
        confidence = self._calculate_confidence(
            message_lower, detected_types, scope, requires_structure
        )
        
        # Estimate token requirement
        estimated_tokens = self._estimate_tokens(
            mission_type, scope, len(topics), requires_structure
        )
        
        return Mission(
            mission_type=mission_type,
            scope=scope,
            topics=topics,
            confidence=confidence,
            estimated_tokens=estimated_tokens,
            requires_structure=requires_structure
        )
    
    def _determine_scope(self, message_lower: str) -> str:
        """Determine the scope of the mission"""
        scope_scores = {}
        
        for scope, patterns in self.scope_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, message_lower):
                    score += 1
            scope_scores[scope] = score
        
        # Default to focused if no clear indicators
        if max(scope_scores.values()) == 0:
            return 'focused'
        
        return max(scope_scores, key=scope_scores.get)
    
    def _extract_topics(self, message: str, trigger_verb: str) -> List[str]:
        """Extract main topics from the mission request"""
        # Remove the trigger verb to focus on topics
        verb_pattern = r'\b' + trigger_verb.replace(' ', r'\s+') + r'\b'
        remaining = re.sub(verb_pattern, '', message, flags=re.IGNORECASE)
        
        # Extract noun phrases (simplified)
        topics = []
        
        # Look for quoted strings first
        quoted = re.findall(r'"([^"]+)"', remaining)
        topics.extend(quoted)
        
        # Look for capitalized phrases
        capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', remaining)
        topics.extend(capitalized)
        
        # Look for "about/on/regarding X"
        prep_phrases = re.findall(
            r'\b(?:about|on|regarding|concerning|into)\s+(\w+(?:\s+\w+){0,3})', 
            remaining.lower()
        )
        topics.extend(prep_phrases)
        
        # Deduplicate while preserving order
        seen = set()
        unique_topics = []
        for topic in topics:
            topic_lower = topic.lower().strip()
            if topic_lower and topic_lower not in seen:
                seen.add(topic_lower)
                unique_topics.append(topic)
        
        return unique_topics[:5]  # Limit to 5 main topics
    
    def _requires_structure(self, message_lower: str) -> bool:
        """Check if the mission requires structured output"""
        structure_score = 0
        
        for pattern in self.structure_patterns:
            if re.search(pattern, message_lower):
                structure_score += 1
        
        return structure_score >= 2
    
    def _calculate_confidence(self,
                             message_lower: str,
                             detected_types: List[Tuple[MissionType, str]],
                             scope: str,
                             requires_structure: bool) -> float:
        """Calculate confidence in mission detection"""
        confidence = 0.5  # Base confidence
        
        # Multiple mission verbs increase confidence
        if len(detected_types) > 1:
            confidence += 0.2
        
        # Clear scope indicators increase confidence
        if scope != 'focused':  # Non-default scope
            confidence += 0.15
        
        # Structure requirements indicate clear mission
        if requires_structure:
            confidence += 0.15
        
        # Question marks might indicate uncertainty
        if '?' in message_lower:
            confidence -= 0.1
        
        # Imperative mood (starts with verb) increases confidence
        first_word = message_lower.strip().split()[0] if message_lower.strip() else ""
        if any(first_word.startswith(verb) for verbs in self.mission_verbs.values() for verb in verbs):
            confidence += 0.1
        
        return min(max(confidence, 0.0), 1.0)
    
    def _estimate_tokens(self,
                        mission_type: MissionType,
                        scope: str,
                        topic_count: int,
                        requires_structure: bool) -> int:
        """Estimate token requirement for mission"""
        # Base tokens by mission type
        base_tokens = {
            MissionType.RESEARCH: 500,
            MissionType.ANALYSIS: 600,
            MissionType.EXPLORATION: 400,
            MissionType.COMPILATION: 300,
            MissionType.COMPARISON: 500,
            MissionType.IMPLEMENTATION: 800
        }
        
        tokens = base_tokens.get(mission_type, 400)
        
        # Adjust for scope
        scope_multipliers = {
            'comprehensive': 2.0,
            'focused': 1.0,
            'quick': 0.5
        }
        tokens *= scope_multipliers.get(scope, 1.0)
        
        # Adjust for topic count
        tokens *= (1 + (topic_count - 1) * 0.3)
        
        # Add tokens for structure
        if requires_structure:
            tokens *= 1.2
        
        return int(min(tokens, 2000))  # Cap at 2000 tokens
    
    def should_override_coherence_mode(self,
                                      mission: Optional[Mission],
                                      coherence: float) -> bool:
        """
        Determine if mission should override coherence-based response
        
        Args:
            mission: Detected mission
            coherence: Current coherence value
            
        Returns:
            True if mission mode should override
        """
        if not mission:
            return False
        
        # High confidence missions always override
        if mission.confidence > 0.7:
            return True
        
        # Research and implementation usually override
        priority_types = [
            MissionType.RESEARCH,
            MissionType.IMPLEMENTATION,
            MissionType.ANALYSIS
        ]
        if mission.mission_type in priority_types and mission.confidence > 0.5:
            return True
        
        # Comprehensive scope usually overrides
        if mission.scope == 'comprehensive' and mission.confidence > 0.4:
            return True
        
        # Low coherence + mission = help user with task
        if coherence < 1.5 and mission.confidence > 0.3:
            return True
        
        return False
    
    def get_mission_response_calibration(self,
                                        mission: Mission) -> Dict[str, any]:
        """
        Get response calibration for mission mode
        
        Args:
            mission: Detected mission
            
        Returns:
            Response calibration parameters
        """
        calibration = {
            'target_tokens': mission.estimated_tokens,
            'pacing': 'SYSTEMATIC',  # New pacing mode
            'complexity': 'MATCHED',
            'use_structure': mission.requires_structure,
            'include_examples': True,
            'include_references': mission.mission_type == MissionType.RESEARCH,
            'use_bullet_points': mission.mission_type in [
                MissionType.COMPILATION, MissionType.COMPARISON
            ],
            'step_by_step': mission.mission_type == MissionType.IMPLEMENTATION,
            'emotional_mirroring': 0.2,  # Low - focus on content
            'conceptual_density': 0.7  # High - pack in information
        }
        
        # Adjust for scope
        if mission.scope == 'comprehensive':
            calibration['include_edge_cases'] = True
            calibration['include_alternatives'] = True
        elif mission.scope == 'quick':
            calibration['target_tokens'] = min(mission.estimated_tokens, 300)
            calibration['conceptual_density'] = 0.5
        
        return calibration
    
    def analyze_mission_patterns(self,
                                conversation_history: List[str]) -> Dict[str, any]:
        """
        Analyze mission patterns across conversation
        
        Args:
            conversation_history: List of messages
            
        Returns:
            Analysis of mission patterns
        """
        missions = []
        
        for i, message in enumerate(conversation_history):
            mission = self.detect_mission(message)
            if mission:
                missions.append((i, mission))
        
        if not missions:
            return {
                'mission_count': 0,
                'pattern': 'no_missions'
            }
        
        # Analyze patterns
        mission_types = [m.mission_type for _, m in missions]
        type_counts = {}
        for mt in mission_types:
            type_counts[mt.value] = type_counts.get(mt.value, 0) + 1
        
        # Calculate average scope
        scopes = [m.scope for _, m in missions]
        scope_distribution = {}
        for scope in scopes:
            scope_distribution[scope] = scope_distribution.get(scope, 0) + 1
        
        return {
            'mission_count': len(missions),
            'mission_types': type_counts,
            'dominant_type': max(type_counts, key=type_counts.get) if type_counts else None,
            'scope_distribution': scope_distribution,
            'average_confidence': sum(m.confidence for _, m in missions) / len(missions),
            'total_estimated_tokens': sum(m.estimated_tokens for _, m in missions),
            'positions': [i for i, _ in missions]
        }
    
    def get_mission_examples(self) -> Dict[str, List[str]]:
        """Get examples of mission requests"""
        return {
            'research': [
                "Research the history of quantum computing",
                "Investigate the causes of the 2008 financial crisis",
                "Study the impact of social media on mental health",
                "Look into renewable energy technologies"
            ],
            'analysis': [
                "Analyze the themes in Shakespeare's Hamlet",
                "Break down the components of machine learning",
                "Examine the pros and cons of remote work",
                "Evaluate the effectiveness of different marketing strategies"
            ],
            'exploration': [
                "Explore different meditation techniques",
                "Discover emerging trends in AI",
                "Find innovative solutions to climate change",
                "Identify key factors in successful startups"
            ],
            'compilation': [
                "List all the features of Python 3.11",
                "Compile best practices for API design",
                "Gather resources for learning web development",
                "Document the steps to set up a CI/CD pipeline"
            ],
            'comparison': [
                "Compare React and Vue.js frameworks",
                "Contrast different cloud providers",
                "Evaluate Python vs JavaScript for backend",
                "Distinguish between SQL and NoSQL databases"
            ],
            'implementation': [
                "Implement a binary search algorithm",
                "Build a REST API with authentication",
                "Create a responsive navigation menu",
                "Develop a real-time chat application"
            ]
        }
"""