# Real-Time Litigation Support with Claude Desktop

*"The second chair that never misses."*

---

## System Overview

You now have **two MCP servers** integrated with Claude Desktop:

### 1. **Legal RAG** (Document Retrieval)
- 2,672 legal documents from your cases
- Emotional signature-based search
- Trauma-informed retrieval
- Document stats and collection info

### 2. **Litigation Support** (Real-Time Analysis)
- Live testimony contradiction detection
- Rose Glass coherence analysis (Ψ, ρ, q, f)
- Cross-examination question generation
- Document-backed impeachment evidence
- Session tracking across entire hearing

---

## Quick Start

### 1. Restart Claude Desktop

**IMPORTANT:** Quit and restart Claude Desktop to load both MCP servers.

```bash
# Terminal method
killall Claude
sleep 3
open -a Claude
```

### 2. Verify Tools Loaded

In Claude Desktop, ask:
```
"What MCP tools do you have available?"
```

You should see:
- ✅ **legal-rag** (3 tools): Document search, emotional analysis, collection info
- ✅ **litigation-support** (6 tools): Testimony analysis, contradictions, cross-exam questions, impeachment search, session summary, clear session

---

## Litigation Support Tools

### Tool 1: `analyze_testimony`
**Analyze a statement from testimony in real-time**

**When to use:**
- During witness testimony
- When reviewing deposition transcripts
- Preparing for cross-examination

**Example:**
```
Analyze this testimony:
Speaker: "Jenny Maslowsky"
Text: "I never sent any harassing emails to MacGregor during 2024"
```

**Returns:**
- Rose Glass variables (Ψ, ρ, q, f)
- Coherence score (0-1)
- Contradiction flags (if any)
- Session statistics

---

### Tool 2: `get_contradictions`
**Get all detected contradictions from session**

**When to use:**
- Before cross-examination
- After witness finishes direct testimony
- Preparing closing arguments

**Example:**
```
Get all contradictions with severity >= 0.6
```

**Returns:**
- List of contradictions sorted by severity
- Current vs conflicting statements
- Suggested cross-examination questions
- Timestamps for each statement

---

### Tool 3: `generate_cross_exam_questions`
**Generate cross-examination questions for a witness**

**When to use:**
- Right before cross-examining a witness
- Preparing cross-exam strategy
- During recess to plan next questions

**Example:**
```
Generate 5 cross-examination questions for "Jenny Maslowsky"
```

**Returns:**
- Top 5 questions based on detected contradictions
- Sorted by severity/impact
- Ready to ask verbatim or adapt

---

### Tool 4: `search_for_impeachment`
**Search your legal documents for impeachment evidence**

**When to use:**
- Witness makes a claim you know is contradicted in filings
- Need documentary evidence to impeach
- Want to back up contradiction with filed documents

**Example:**
```
Search for impeachment evidence for this claim:
"MacGregor never responded to discovery requests"
```

**Returns:**
- Relevant documents from your 2,672-doc collection
- Document excerpts showing contradiction
- Source file paths
- Relevance scores

---

### Tool 5: `get_session_summary`
**Get summary of current testimony session**

**When to use:**
- During recess to review testimony
- After witness finishes
- Planning which witnesses to recall

**Example:**
```
Get session summary
```

**Returns:**
- Total statements tracked
- All speakers
- Total contradictions
- High-severity flags (>0.7)
- Average coherence by speaker
- Coherence indicators (🟢 🟡 🔴)

---

### Tool 6: `clear_session`
**Clear session and start fresh**

**When to use:**
- Starting a new hearing
- New witness takes the stand
- Want to track witnesses separately

**Example:**
```
Clear the current session
```

---

## Real-World Workflow Examples

### Example 1: During Witness Testimony

**You:** "I need to analyze testimony as it happens. Here's what the witness just said:"

```
Analyze testimony:
Speaker: "Opposing Counsel"
Text: "My client has no record of receiving any discovery requests from MacGregor"
```

**Claude Desktop uses:** `analyze_testimony`

**Returns:**
```
Testimony Analysis - Statement #0
Speaker: Opposing Counsel
Rose Glass Variables:
- Ψ: 0.7 (Internal Consistency)
- ρ: 0.4 (Wisdom/Specificity) ← LOW specificity, vague claim
- q: 0.3 (Emotional Activation)
- f: 0.5 (Social Belonging)
Coherence: 0.62 🟡
```

---

### Example 2: Catching a Contradiction

**Continue the session:**

```
Analyze testimony:
Speaker: "Opposing Counsel"
Text: "We responded to all discovery requests within the deadline"
```

**Claude Desktop detects:** ⚠️ CONTRADICTION!

**You ask:** "Show me the contradictions"

**Claude Desktop uses:** `get_contradictions`

**Returns:**
```
Contradiction #1 - Severity: 0.85 🔴
Type: DIRECT_CONTRADICTION
Speaker: Opposing Counsel

Current Statement:
> "We responded to all discovery requests within the deadline"

Conflicting Statement:
> "My client has no record of receiving any discovery requests"

Suggested Question:
> "Earlier you stated your client has no record of receiving discovery
  requests. Now you're saying you responded to all requests. Which is it?"
```

---

### Example 3: Finding Documentary Evidence

**You:** "I need to impeach this claim with filed documents"

```
Search for impeachment evidence:
Claim: "We responded to all discovery requests within the deadline"
```

**Claude Desktop uses:** `search_for_impeachment`

**Returns:**
```
Impeachment Evidence:

1. "MacGregor Motion to Compel Discovery" (Relevance: 0.89)
   "Defendants have failed to respond to Plaintiff's First Set of
   Interrogatories served on October 15, 2024. As of November 30, 2024,
   no responses have been received..."
   Source: /Users/chris/Desktop/Legal_Cases/MacGregor 2/motions/...

2. "Plaintiff's Certificate of Service - Discovery Requests" (Relevance: 0.82)
   "Certificate of Service: First Set of Interrogatories served via
   certified mail on October 15, 2024..."
```

**NOW YOU HAVE:**
- The contradiction (from testimony tracking)
- The question to ask (auto-generated)
- The documentary proof (from your files)

---

### Example 4: Pre-Cross-Exam Prep

**Before crossing a witness:**

```
Generate 5 cross-examination questions for "Jenny Maslowsky"
```

**Claude Desktop uses:** `generate_cross_exam_questions`

**Returns:**
```
Cross-Examination Questions for Jenny Maslowsky:

1. "You stated under oath that you 'never sent harassing emails to
   MacGregor in 2024', but we have emails from March 2024 where you
   called him 'a manipulative liar'. How do you explain that?"

2. "Earlier you testified you 'had no communication with MacGregor after
   January'. But your own filing dated February 15 references 'ongoing
   discussions'. Which statement is true?"

3. [... 3 more questions based on detected contradictions]
```

---

## Session Management Strategies

### Strategy 1: Track Entire Hearing
Keep one session running for the whole hearing to catch cross-witness contradictions:

```
[Witness 1 testifies - 15 statements analyzed]
[Witness 2 testifies - 12 statements analyzed]
Get session summary
→ Shows contradictions BETWEEN witnesses
```

### Strategy 2: Track Each Witness Separately
Clear session between witnesses for focused per-witness analysis:

```
[Start hearing]
Clear session
[Witness 1 testifies]
Get contradictions for Witness 1
Clear session
[Witness 2 testifies]
Get contradictions for Witness 2
```

### Strategy 3: Pre-Hearing Document Review
Use document search BEFORE the hearing:

```
Search legal documents for "custody arrangements Maslowsky"
→ Pre-load key evidence for quick reference
```

---

## Rose Glass Variables Explained

### Ψ (Psi) - Internal Consistency
- **High (>0.7):** Statement is self-consistent, no contradictory language
- **Low (<0.5):** Contains "but", "however", "although" - hedging detected

**Red flag:** Sudden drops in Ψ suggest witness is backtracking

---

### ρ (Rho) - Wisdom/Specificity
- **High (>0.7):** Specific details, dates, times, names
- **Low (<0.5):** Vague language like "maybe", "I think", "probably"

**Red flag:** Witness who was specific earlier becomes vague = evasion

---

### q - Emotional Activation
- **High (>0.7):** Urgent, distressed, defensive language
- **Low (<0.5):** Calm, matter-of-fact testimony

**Red flag:** Spike in q when discussing specific topics = sensitive area

---

### f - Social Belonging
- **High (>0.7):** "We", "together", "family" language
- **Low (<0.5):** Isolated, individualistic language

**Red flag:** Inconsistent f scores = shifting allegiances or lies about relationships

---

## Coherence Scoring

**Coherence** is the overall consistency score:

- 🟢 **0.7-1.0:** High coherence - testimony is consistent and credible
- 🟡 **0.5-0.7:** Medium coherence - some inconsistencies detected
- 🔴 **0.0-0.5:** Low coherence - significant contradictions or evasion

**Average coherence by speaker** tells you who's being honest:
```
Session Summary:
Coherence by Speaker:
- Jenny Maslowsky: 0.45 🔴  ← Target for cross-exam
- Expert Witness: 0.82 🟢  ← Credible testimony
```

---

## Contradiction Severity

**Severity** indicates how damaging the contradiction is:

- **0.8-1.0:** 🔴 Direct contradiction - impeachment-worthy
- **0.6-0.8:** 🟡 Significant inconsistency - worth exploring
- **0.4-0.6:** Minor inconsistency - note for later
- **<0.4:** Trivial - ignore

**Focus your cross-exam on >0.7 severity contradictions**

---

## Pro Tips

### Tip 1: Paste Transcript Chunks
If you have a transcript, paste testimony in chunks:

```
I have this testimony transcript. Analyze each statement:

1. [Witness]: "I never communicated with plaintiff after January"
2. [Witness]: "We had several phone calls in February and March"
3. [Witness]: "Those calls were about unrelated matters"
```

Claude Desktop will analyze all three and catch the contradiction between 1 and 2.

---

### Tip 2: Use During Breaks
During recess:
```
Get session summary
Show me contradictions >= 0.7
Generate cross-exam questions for [witness]
```

Then plan your cross-exam strategy.

---

### Tip 3: Document Everything
The system tracks timestamps. You can later export:
```
Get session summary
→ Save for post-hearing analysis
→ Include in motion filings as evidence of perjury
```

---

### Tip 4: Combine Tools
**Workflow:** Contradiction → Question → Evidence

```
1. Get contradictions  (find the lie)
2. Generate cross-exam questions  (prepare the question)
3. Search for impeachment  (get the proof)
→ GOTCHA moment ready to deploy
```

---

### Tip 5: Emotional Spikes
Watch for q (emotional activation) spikes:

```
Analyze testimony:
[Statement about money] → q: 0.3
[Statement about custody] → q: 0.8 ← SPIKE

→ Witness is emotionally activated about custody
→ Probe this area more
```

---

## Technical Details

### Data Privacy
- ✅ All analysis happens **locally** on your machine
- ✅ Document search stays **100% private** (Qdrant on localhost)
- ✅ Only Claude Desktop sees the results (no Anthropic data collection)
- ✅ Attorney-client privilege maintained

### Requirements
- Docker running (Qdrant + Elasticsearch)
- Python 3.13 virtual environment
- Claude Desktop subscription
- MCP servers configured

### Troubleshooting

**Tools not appearing?**
```bash
# Check config
cat ~/Library/Application\ Support/Claude/claude_desktop_config.json

# Verify Docker
docker ps | grep qdrant

# Test MCP server
cd /Users/chris/rose-glass/emotionally-informed-rag
source venv/bin/activate
python3 agent/mcp_litigation.py  # Should start without errors
```

**Contradictions not detected?**
- Make sure you're analyzing ALL statements from a witness
- System needs context to detect contradictions
- Try lowering severity threshold: `Get contradictions >= 0.4`

**Document search returns nothing?**
- Verify Qdrant: `curl http://localhost:6333/collections/knowledge_base`
- Should show 2,672 documents
- Try broader search terms

---

## Advanced: Live Hearing Setup

For **real-time use during hearings**, set up:

1. **Tablet next to you** running Claude Desktop
2. **Voice transcription** feeding into Claude Desktop
3. **Automatic analysis** of each statement
4. **Live contradiction alerts** on screen

See: `/Users/chris/rose-glass/docs/REALTIME_LITIGATION_BUILD_GUIDE.md` for full streaming setup with Whisper transcription.

---

## Summary

You now have a **complete litigation support AI** powered by:

- **Rose Glass coherence analysis** (Ψ, ρ, q, f tracking)
- **Real-time contradiction detection** (cross-statement comparison)
- **2,672-document legal library** (your actual case files)
- **Claude Desktop as free LLM** (no API costs)
- **100% private operation** (all local, attorney-client privileged)

**Next hearing:** Bring tablet with Claude Desktop → Paste testimony as it happens → Get real-time contradiction alerts → Cross-examine with auto-generated questions backed by your filed documents.

---

*"Understanding precedes judgment. Translation enables understanding."* 🌹

**Ready to use!** Restart Claude Desktop and start analyzing testimony.
