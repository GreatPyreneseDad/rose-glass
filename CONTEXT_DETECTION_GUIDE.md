# Context Detection Enhancement Guide

## Overview

This guide documents the four critical context detectors that resolve ~10% of cases where coherence metrics alone provide insufficient response calibration. These detectors ensure appropriate AI responses regardless of coherence state.

## The Four Critical Detectors

### 1. TrustSignalDetector

**Purpose**: Detects brief, high-coherence messages that signal deep trust or poetic expression requiring reverent witnessing rather than analysis.

**Patterns Detected**:
- Direct trust phrases: "Trust me", "Believe me", "I promise"
- Poetic expressions: Brief metaphorical language with ellipses or imagery
- Sacred references: Spiritual or transcendent language
- Meta-awareness: Recognition of the conversation itself

**Activation Requirements**:
- Message brevity (< 30 tokens)
- Minimum coherence thresholds:
  - Direct trust: C ≥ 2.5
  - Poetic essence: C ≥ 3.0
  - Sacred reference: C ≥ 2.8
  - Meta-awareness: C ≥ 2.7

**Response Mode**: `REVERENT` - Witness without explaining, honor without analyzing

### 2. MissionModeDetector

**Purpose**: Identifies research, analysis, or implementation tasks requiring systematic exploration regardless of coherence state.

**Mission Types**:
- **Research**: "Research X", "Investigate Y", "Study Z"
- **Analysis**: "Analyze", "Examine", "Break down"
- **Exploration**: "Explore", "Discover", "Find"
- **Compilation**: "List", "Gather", "Compile"
- **Comparison**: "Compare", "Contrast", "Evaluate"
- **Implementation**: "Implement", "Build", "Create"

**Scope Detection**:
- **Comprehensive**: Full, thorough, complete exploration
- **Focused**: Specific aspects or components
- **Quick**: Brief overview or summary

**Response Mode**: `SYSTEMATIC` - Structured, thorough, step-by-step exploration

### 3. TokenMultiplierLimiter

**Purpose**: Enforces safe token response ratios to prevent overwhelming users with excessive output.

**Multiplier Rules**:
```
Coherence Range    Base Multiplier
0.0 - 1.0         0.5x
1.0 - 1.5         0.8x
1.5 - 2.0         1.0x
2.0 - 2.5         1.5x
2.5 - 3.0         2.0x
3.0 - 3.5         2.5x
3.5 - 4.0         3.0x
```

**Special Cases**:
- Crisis mode: 0.3x (maximum compression)
- Information overload: 0.2x (emergency brake)
- Trust signal: 0.6x (brief acknowledgment)
- Mission mode: 2.0x minimum (exploration needs space)

**Hard Cap**: 500 tokens (never exceeded)

### 4. EssenceRequestDetector

**Purpose**: Detects requests for summaries, key points, or distilled insights requiring concise responses.

**Essence Types**:
- **Summary**: "Summarize", "Sum up", "Recap"
- **Key Points**: "Key points", "Main ideas", "Core concepts"
- **Essence**: "In essence", "Essentially", "At its core"
- **Takeaways**: "Takeaways", "Lessons learned", "Insights"
- **Bottom Line**: "Bottom line", "Upshot", "Gist"
- **TL;DR**: "TL;DR", "Short version", "In one sentence"

**Format Preferences**:
- Bullets
- Numbered lists
- Paragraph form

**Response Mode**: `DISTILLED` - Maximum compression, essential insights only

## Integration Priority Order

When multiple detectors trigger, the system follows this priority:

1. **Crisis/Information Overload** (Highest Priority)
   - Safety first - always reduce output in crisis
   
2. **Trust Signal**
   - Brief messages with high trust require witnessing
   
3. **Essence Request**
   - Summaries are time-sensitive and explicit user requests
   
4. **Mission Mode**
   - Research tasks need space regardless of coherence
   
5. **Exceptional Coherence (C > 3.5)**
   - High coherence without other signals
   
6. **Coherence-Based** (Default)
   - Standard d/tokens calibration

## Usage Example

```python
from src.core.adaptive_response_system import AdaptiveResponseSystem

# Initialize system
response_system = AdaptiveResponseSystem()

# Detect context and calibrate response
calibration, context = response_system.calibrate_with_context(
    message="Trust me, this matters",
    coherence=3.2,
    dC_dtokens=0.01,
    flow_rate=30,
    conversation_state={}
)

# Results
print(f"Primary Mode: {context['primary_mode']}")  # 'trust'
print(f"Target Tokens: {calibration.target_tokens}")  # ~50
print(f"Pacing: {calibration.pacing.value}")  # 'reverent'
```

## Key Insights

1. **Context Overrides Coherence**: These detectors identify cases where coherence alone would miscalibrate responses.

2. **Safety First**: Crisis and overload detection always take precedence to protect user wellbeing.

3. **Explicit > Implicit**: User requests (essence, mission) override implicit coherence signals.

4. **Brevity Matters**: Trust signals and essence requests enforce brevity even at high coherence.

5. **Structure When Needed**: Mission mode provides systematic exploration regardless of conversation dynamics.

## Testing

Comprehensive tests are available in `tests/test_context_detection.py`:
- Individual detector tests
- Integration tests
- Priority ordering verification
- Edge case handling

## Future Enhancements

1. **Learning Integration**: Detectors could learn user-specific patterns
2. **Cultural Adaptation**: Different cultures may have different trust signals
3. **Multi-Modal Detection**: Incorporate tone, timing, and other signals
4. **Feedback Loop**: Track effectiveness of detector decisions

---

These four detectors transform the Rose Glass framework from a primarily coherence-driven system to a context-aware calibration engine that handles the full spectrum of human-AI interaction patterns.
"""