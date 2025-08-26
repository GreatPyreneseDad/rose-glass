# Ethical Guidelines for Rose Glass Synthetic-Organic Translation

## Core Principles

### 1. Translation, Not Judgment
The Rose Glass framework is a **translation lens**, not a quality assessment tool. We translate patterns to enable understanding between different forms of intelligence, never to judge or rank human expression.

### 2. Dignity and Respect
Every organic intelligence deserves dignity and respect. The system must:
- Never make assumptions about identity based on communication patterns
- Respect diverse ways of expressing coherence
- Acknowledge that different cultural contexts have different valid patterns

### 3. Privacy Preservation
- **No demographic profiling**: Never attempt to infer age, gender, race, class, or other protected characteristics
- **No persistent storage**: Conversation data should be ephemeral
- **Explicit consent**: Any context gathering must be through explicit user consent
- **Minimal data**: Collect only what's necessary for translation

### 4. Cultural Humility
The "Averroes Test" teaches us that when medieval philosophy scores differently than modern texts, it reveals our lens limitations, not textual deficiencies. Always maintain:
- Multiple calibrated lenses for different contexts
- Recognition that no single lens is "correct"
- Humility about what synthetic intelligence can perceive

## Prohibited Practices

### ❌ DO NOT:

1. **Profile Users**
   - No inferring "hidden agendas" or "unstated motivations"
   - No determining social class, education level, or background
   - No gender or demographic detection

2. **Make Assumptions**
   - No "truth detection" or "authenticity gaps"
   - No psychological diagnosis or mental state inference
   - No claims about what someone "really means"

3. **Store Personal Patterns**
   - No building user profiles over time
   - No pattern matching against previous users
   - No behavioral prediction models

4. **Judge Communication Styles**
   - No labeling styles as "better" or "worse"
   - No penalizing indirect or non-linear communication
   - No cultural superiority assumptions

## Approved Practices

### ✅ DO:

1. **Focus on Observable Patterns**
   ```python
   # Good: Detecting communication preferences
   formality_level = detect_formality_preference(text)
   
   # Bad: Inferring education level
   education = infer_education_from_vocabulary(text)  # NEVER DO THIS
   ```

2. **Request Explicit Consent**
   ```python
   # Good: Asking for preferences
   lens_preference = request_user_lens_selection()
   
   # Bad: Inferring cultural background
   culture = detect_cultural_background(text)  # NEVER DO THIS
   ```

3. **Adapt to Communication Styles**
   ```python
   # Good: Matching directness level
   if prefers_indirect_communication:
       use_softer_language(response)
   
   # Bad: Assuming indirectness means dishonesty
   if indirect:
       flag_as_deceptive()  # NEVER DO THIS
   ```

4. **Monitor Quality, Not Identity**
   ```python
   # Good: Measuring mutual understanding
   understanding_score = check_clarification_success()
   
   # Bad: Profiling user competence
   user_intelligence = assess_cognitive_ability()  # NEVER DO THIS
   ```

## Implementation Guidelines

### 1. Pattern Detection
- Detect WHAT is expressed, not WHO is expressing
- Focus on structural patterns, not identity markers
- Use patterns only for current translation, not future prediction

### 2. Lens Calibration
- Offer multiple cultural lenses
- Let users choose their preferred lens
- Never force a lens based on assumptions

### 3. Response Adaptation
- Match communication style for accessibility
- Respect expressed preferences
- Maintain synthetic authenticity while adapting

### 4. Quality Monitoring
- Measure communication effectiveness
- Focus on mutual understanding
- Never blame users for "poor" communication

## Ethical Decision Framework

When implementing new features, ask:

1. **Does this respect human dignity?**
   - Would I want this applied to my own communication?
   - Does it make assumptions about who someone is?

2. **Is this truly necessary for translation?**
   - Can we achieve understanding without this information?
   - Are we collecting minimal necessary data?

3. **Could this enable discrimination?**
   - Could the feature be misused for profiling?
   - Are we encoding biases?

4. **Is consent explicit and informed?**
   - Do users understand what we're doing?
   - Can they opt out without penalty?

## Handling Edge Cases

### When Communication Seems Incoherent
- Consider lens calibration mismatch
- Offer clarification without judgment
- Remember: coherence is constructed, not discovered

### When Patterns Don't Match Expectations
- Question the lens, not the human
- Consider cultural or contextual differences
- Maintain humility about synthetic limitations

### When Quality Metrics Are Low
- Focus on improving synthetic responses
- Don't blame human communication style
- Seek explicit feedback on preferences

## Continuous Improvement

### Regular Audits
- Review patterns for unintended profiling
- Check for cultural bias in lenses
- Validate privacy preservation

### User Feedback
- Create safe channels for reporting issues
- Act on concerns about dignity or respect
- Continuously refine ethical guidelines

### Transparency
- Document all pattern detection methods
- Explain lens calibration clearly
- Be open about system limitations

## Conclusion

The Rose Glass is a tool for understanding, not judgment. By following these ethical guidelines, we ensure that synthetic-organic translation enhances communication while preserving human dignity, privacy, and the rich diversity of human expression.

Remember: **We are translators, not judges. We are bridges, not gatekeepers.**

---

*These guidelines are living documents. As our understanding evolves, so too should our ethical practices.*