"""
Essence Request Detector
========================

Detects requests for summaries, key points, or essential insights.
These requests should receive concise, focused responses regardless
of coherence state or token flow dynamics.

Pattern: "Summarize", "Key points", "Main ideas", "In essence"
Response: Distilled wisdom, 100-200 tokens max

Author: Christopher MacGregor bin Joseph
Date: October 2025
"""

from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import re


class EssenceType(Enum):
    """Types of essence requests"""
    SUMMARY = "summary"  # Summarize, sum up, recap
    KEY_POINTS = "key_points"  # Key points, main points, highlights
    ESSENCE = "essence"  # In essence, essentially, core of
    TAKEAWAYS = "takeaways"  # Takeaways, lessons, insights
    BOTTOM_LINE = "bottom_line"  # Bottom line, upshot, gist
    TLDR = "tldr"  # TL;DR, brief version


@dataclass
class EssenceRequest:
    """Detected essence request parameters"""
    essence_type: EssenceType
    scope: str  # 'conversation', 'topic', 'document'
    target_length: str  # 'brief', 'moderate', 'detailed'
    confidence: float
    focus_areas: List[str]
    format_preference: str  # 'paragraph', 'bullets', 'numbered'


class EssenceRequestDetector:
    """Detect requests for summaries and essential insights"""
    
    def __init__(self):
        """Initialize essence request patterns"""
        # Essence request indicators
        self.essence_patterns = {
            EssenceType.SUMMARY: [
                r'\bsummariz[es]?\b',
                r'\bsum(?:med)?\s+up\b',
                r'\brecap\b',
                r'\bbrief(?:ly)?\s+(?:explain|describe)\b',
                r'\boverview\b'
            ],
            EssenceType.KEY_POINTS: [
                r'\bkey\s+(?:points?|ideas?|concepts?)\b',
                r'\bmain\s+(?:points?|ideas?|concepts?|takeaways?)\b',
                r'\b(?:most\s+)?important\s+(?:points?|things?|aspects?)\b',
                r'\bhighlights?\b',
                r'\bcore\s+(?:ideas?|concepts?|principles?)\b'
            ],
            EssenceType.ESSENCE: [
                r'\bin\s+essence\b',
                r'\bessentially\b',
                r'\b(?:the\s+)?essence\s+(?:of|is)\b',
                r'\bfundamentally\b',
                r'\bat\s+(?:its|the)\s+(?:core|heart)\b'
            ],
            EssenceType.TAKEAWAYS: [
                r'\btakeaways?\b',
                r'\blessons?\s+learned\b',
                r'\bkey\s+(?:insights?|learnings?)\b',
                r'\bwhat\s+(?:I|we)\s+(?:learned|discovered)\b'
            ],
            EssenceType.BOTTOM_LINE: [
                r'\bbottom\s+line\b',
                r'\bupshot\b',
                r'\bgist\b',
                r'\bnet\s+(?:result|effect)\b',
                r'\bwhat\s+(?:it|this)\s+(?:means|boils\s+down\s+to)\b'
            ],
            EssenceType.TLDR: [
                r'\btl;?dr\b',
                r'\btoo\s+long[,;]?\s+didn\'?t\s+read\b',
                r'\bshort\s+version\b',
                r'\bquick\s+version\b',
                r'\bin\s+(?:a|one)\s+(?:sentence|paragraph)\b'
            ]
        }
        
        # Scope indicators
        self.scope_patterns = {
            'conversation': [
                r'\b(?:this|our)\s+(?:conversation|discussion|chat)\b',
                r'\bwhat\s+we(?:\'ve)?\s+(?:discussed|talked\s+about)\b',
                r'\beverything\s+(?:we\'ve\s+)?(?:covered|said)\b',
                r'\bso\s+far\b'
            ],
            'topic': [
                r'\babout\s+\w+',
                r'\bregarding\s+\w+',
                r'\b(?:this|that)\s+(?:topic|subject|theme)\b',
                r'\bon\s+\w+'
            ],
            'document': [
                r'\b(?:this|the)\s+(?:document|article|paper|text)\b',
                r'\bwhat\s+(?:I|you)\s+(?:just\s+)?(?:read|wrote|shared)\b',
                r'\b(?:above|below)\s+(?:text|content)\b'
            ]
        }
        
        # Format preferences
        self.format_patterns = {
            'bullets': [
                r'\bbullet(?:s|ed)?\s*(?:points?|list)?\b',
                r'\blist\b',
                r'\bpoints?\b'
            ],
            'numbered': [
                r'\bnumbered\b',
                r'\b\d+\.\s*\w+',
                r'\bsteps?\b',
                r'\bordered\b'
            ],
            'paragraph': [
                r'\bparagraph\b',
                r'\bnarrative\b',
                r'\bprose\b'
            ]
        }
        
        # Length indicators
        self.length_patterns = {
            'brief': [
                r'\b(?:very\s+)?(?:brief|short|quick|concise)\b',
                r'\b(?:one|1)\s+(?:sentence|line|paragraph)\b',
                r'\bfew\s+words\b'
            ],
            'moderate': [
                r'\b(?:few|several)\s+(?:sentences|paragraphs)\b',
                r'\bmoderate\s+(?:length|detail)\b'
            ],
            'detailed': [
                r'\b(?:detailed|comprehensive|thorough)\b',
                r'\b(?:full|complete)\s+summary\b',
                r'\beverything\b'
            ]
        }
        
    def detect_essence_request(self, message: str) -> Optional[EssenceRequest]:
        """
        Detect if message requests summary or essence
        
        Args:
            message: User message text
            
        Returns:
            EssenceRequest if detected, None otherwise
        """
        message_lower = message.lower()
        
        # Check for essence patterns
        detected_types = []
        for essence_type, patterns in self.essence_patterns.items():
            for pattern in patterns:
                if re.search(pattern, message_lower):
                    detected_types.append(essence_type)
                    break
        
        if not detected_types:
            return None
        
        # Primary type is first detected
        primary_type = detected_types[0]
        
        # Determine scope
        scope = self._determine_scope(message_lower)
        
        # Determine target length
        target_length = self._determine_length(message_lower)
        
        # Extract focus areas
        focus_areas = self._extract_focus_areas(message)
        
        # Determine format preference
        format_preference = self._determine_format(message_lower)
        
        # Calculate confidence
        confidence = self._calculate_confidence(
            message_lower, detected_types, scope, target_length
        )
        
        return EssenceRequest(
            essence_type=primary_type,
            scope=scope,
            target_length=target_length,
            confidence=confidence,
            focus_areas=focus_areas,
            format_preference=format_preference
        )
    
    def _determine_scope(self, message_lower: str) -> str:
        """Determine the scope of the essence request"""
        scope_scores = {}
        
        for scope, patterns in self.scope_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, message_lower):
                    score += 1
            scope_scores[scope] = score
        
        # Default to topic if no clear indicators
        if max(scope_scores.values()) == 0:
            return 'topic'
        
        return max(scope_scores, key=scope_scores.get)
    
    def _determine_length(self, message_lower: str) -> str:
        """Determine target length preference"""
        for length, patterns in self.length_patterns.items():
            for pattern in patterns:
                if re.search(pattern, message_lower):
                    return length
        
        # Default to brief for essence requests
        return 'brief'
    
    def _determine_format(self, message_lower: str) -> str:
        """Determine format preference"""
        for format_type, patterns in self.format_patterns.items():
            for pattern in patterns:
                if re.search(pattern, message_lower):
                    return format_type
        
        # Default to paragraph
        return 'paragraph'
    
    def _extract_focus_areas(self, message: str) -> List[str]:
        """Extract specific areas to focus on"""
        focus_areas = []
        
        # Look for "especially", "particularly", "focusing on"
        focus_patterns = [
            r'\b(?:especially|particularly|specifically)\s+(?:about|regarding|on)\s+(\w+(?:\s+\w+){0,2})',
            r'\bfocus(?:ing)?\s+on\s+(\w+(?:\s+\w+){0,2})',
            r'\b(?:main|key)\s+(\w+)\s+(?:only|specifically)',
        ]
        
        for pattern in focus_patterns:
            matches = re.findall(pattern, message.lower())
            focus_areas.extend(matches)
        
        # Look for quoted focus areas
        quoted = re.findall(r'"([^"]+)"', message)
        focus_areas.extend(quoted)
        
        # Deduplicate
        seen = set()
        unique_areas = []
        for area in focus_areas:
            area_lower = area.lower().strip()
            if area_lower and area_lower not in seen:
                seen.add(area_lower)
                unique_areas.append(area)
        
        return unique_areas[:3]  # Limit to 3 focus areas
    
    def _calculate_confidence(self,
                             message_lower: str,
                             detected_types: List[EssenceType],
                             scope: str,
                             target_length: str) -> float:
        """Calculate confidence in essence request detection"""
        confidence = 0.6  # Base confidence
        
        # Multiple essence indicators increase confidence
        if len(detected_types) > 1:
            confidence += 0.2
        
        # Clear scope increases confidence
        if scope != 'topic':  # Non-default scope
            confidence += 0.1
        
        # Explicit length request increases confidence
        if target_length in ['brief', 'detailed']:
            confidence += 0.1
        
        # Direct commands increase confidence
        if message_lower.strip().startswith(('summarize', 'give me', 'what are')):
            confidence += 0.1
        
        # Question format
        if '?' in message_lower:
            confidence += 0.05
        
        return min(confidence, 1.0)
    
    def get_essence_response_calibration(self,
                                        request: EssenceRequest) -> Dict[str, any]:
        """
        Get response calibration for essence request
        
        Args:
            request: Detected essence request
            
        Returns:
            Response calibration parameters
        """
        # Base token limits by length preference
        token_limits = {
            'brief': 100,
            'moderate': 200,
            'detailed': 300
        }
        
        calibration = {
            'target_tokens': token_limits.get(request.target_length, 150),
            'format': request.format_preference,
            'pacing': 'DISTILLED',  # New pacing mode for essence
            'complexity': 'SIMPLIFIED',
            'include_examples': False,  # No examples in summaries
            'use_transitions': request.format_preference == 'paragraph',
            'numbered_points': request.format_preference == 'numbered',
            'bullet_points': request.format_preference == 'bullets',
            'focus_areas': request.focus_areas,
            'emotional_mirroring': 0.1,  # Minimal emotion
            'conceptual_density': 0.9  # High density - pack insights
        }
        
        # Adjust for essence type
        type_adjustments = {
            EssenceType.TLDR: {
                'target_tokens': 50,
                'ultra_concise': True
            },
            EssenceType.BOTTOM_LINE: {
                'target_tokens': 75,
                'direct_conclusion': True
            },
            EssenceType.KEY_POINTS: {
                'extract_main_ideas': True,
                'hierarchical': True
            },
            EssenceType.TAKEAWAYS: {
                'action_oriented': True,
                'practical_focus': True
            }
        }
        
        if request.essence_type in type_adjustments:
            calibration.update(type_adjustments[request.essence_type])
        
        # Scope adjustments
        if request.scope == 'conversation':
            calibration['retrospective'] = True
            calibration['chronological'] = False  # Focus on themes not timeline
        elif request.scope == 'document':
            calibration['structured_extraction'] = True
        
        return calibration
    
    def should_override_token_flow(self,
                                  request: Optional[EssenceRequest]) -> bool:
        """
        Determine if essence request should override token flow calibration
        
        Args:
            request: Detected essence request
            
        Returns:
            True if should override normal calibration
        """
        if not request:
            return False
        
        # High confidence essence requests always override
        if request.confidence > 0.7:
            return True
        
        # TL;DR always overrides
        if request.essence_type == EssenceType.TLDR:
            return True
        
        # Bottom line with good confidence overrides
        if request.essence_type == EssenceType.BOTTOM_LINE and request.confidence > 0.5:
            return True
        
        return False
    
    def format_essence_response(self,
                              content: List[str],
                              request: EssenceRequest) -> str:
        """
        Format essence content according to request preferences
        
        Args:
            content: List of key points/insights
            request: Original essence request
            
        Returns:
            Formatted response string
        """
        if request.format_preference == 'bullets':
            # Bullet point format
            formatted = "\n".join(f"• {point}" for point in content)
            
        elif request.format_preference == 'numbered':
            # Numbered list format
            formatted = "\n".join(f"{i+1}. {point}" 
                                for i, point in enumerate(content))
            
        else:  # paragraph
            # Paragraph format with transitions
            if len(content) == 1:
                formatted = content[0]
            elif len(content) == 2:
                formatted = f"{content[0]}. Additionally, {content[1].lower()}"
            else:
                # First point
                formatted = content[0]
                # Middle points
                for point in content[1:-1]:
                    formatted += f". Furthermore, {point.lower()}"
                # Last point
                formatted += f". Finally, {content[-1].lower()}"
            
            # Ensure ends with period
            if not formatted.endswith('.'):
                formatted += '.'
        
        # Add scope context if needed
        if request.scope == 'conversation':
            formatted = "From our conversation: " + formatted
        elif request.scope == 'document':
            formatted = "Document summary: " + formatted
        
        return formatted
    
    def analyze_essence_patterns(self,
                                conversation_history: List[str]) -> Dict[str, any]:
        """
        Analyze essence request patterns in conversation
        
        Args:
            conversation_history: List of messages
            
        Returns:
            Analysis of essence patterns
        """
        essence_requests = []
        
        for i, message in enumerate(conversation_history):
            request = self.detect_essence_request(message)
            if request:
                essence_requests.append((i, request))
        
        if not essence_requests:
            return {
                'essence_count': 0,
                'pattern': 'no_essence_requests'
            }
        
        # Analyze patterns
        types = [r.essence_type for _, r in essence_requests]
        type_counts = {}
        for et in types:
            type_counts[et.value] = type_counts.get(et.value, 0) + 1
        
        # Scope analysis
        scopes = [r.scope for _, r in essence_requests]
        scope_counts = {}
        for scope in scopes:
            scope_counts[scope] = scope_counts.get(scope, 0) + 1
        
        # Format preferences
        formats = [r.format_preference for _, r in essence_requests]
        format_counts = {}
        for fmt in formats:
            format_counts[fmt] = format_counts.get(fmt, 0) + 1
        
        return {
            'essence_count': len(essence_requests),
            'type_distribution': type_counts,
            'scope_distribution': scope_counts,
            'format_preferences': format_counts,
            'average_confidence': sum(r.confidence for _, r in essence_requests) / len(essence_requests),
            'positions': [i for i, _ in essence_requests],
            'pattern': self._determine_pattern(essence_requests, len(conversation_history))
        }
    
    def _determine_pattern(self, 
                          essence_requests: List[Tuple[int, EssenceRequest]],
                          total_messages: int) -> str:
        """Determine the pattern of essence requests"""
        if len(essence_requests) == 1:
            # Single request - check position
            position = essence_requests[0][0]
            if position < total_messages * 0.3:
                return 'early_summary'
            elif position > total_messages * 0.7:
                return 'closing_summary'
            else:
                return 'mid_conversation_summary'
        
        # Multiple requests
        positions = [i for i, _ in essence_requests]
        
        # Regular intervals?
        if len(positions) >= 3:
            intervals = [positions[i+1] - positions[i] 
                        for i in range(len(positions)-1)]
            if max(intervals) - min(intervals) <= 2:
                return 'periodic_summaries'
        
        # Clustering at end?
        if all(p > total_messages * 0.7 for p in positions[-2:]):
            return 'conclusion_focused'
        
        return 'scattered_summaries'
    
    def get_essence_examples(self) -> Dict[str, List[str]]:
        """Get examples of essence requests"""
        return {
            'summary': [
                "Can you summarize our discussion?",
                "Please sum up the main points",
                "Give me a brief recap",
                "What's the overview of what we covered?"
            ],
            'key_points': [
                "What are the key points?",
                "List the main ideas",
                "What are the most important concepts?",
                "Highlight the core principles"
            ],
            'essence': [
                "What's the essence of this?",
                "In essence, what does this mean?",
                "What's at the core here?",
                "Fundamentally, what are we saying?"
            ],
            'takeaways': [
                "What are the main takeaways?",
                "What lessons did we learn?",
                "Key insights from this?",
                "What should I remember?"
            ],
            'bottom_line': [
                "What's the bottom line?",
                "What's the upshot?",
                "What does this boil down to?",
                "Net-net, what's the conclusion?"
            ],
            'tldr': [
                "TL;DR?",
                "Give me the short version",
                "In one sentence?",
                "Quick summary please"
            ]
        }
"""