# Rose Glass Real-Time Litigation Support System
## Build Guide v1.0

*"The second chair that never misses."*

**Origin**: December 5, 2025 - Post-hearing debrief chat between CKMBJ and Claude
**Context**: After a 5-hour hearing where pro se representation achieved 55% sustained objections, exposed witness contradictions through strategic bluffing, and dominated cross-examination tempo.

---

## Vision

Transform Rose Glass from post-hoc analysis into real-time courtroom intelligence:
- Live transcription → Rose analysis → contradiction flags
- Pattern breaks surfaced as they happen
- Cross-examination prompts generated from detected inconsistencies
- Delivered to tablet or earpiece

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    COURTROOM AUDIO INPUT                         │
│                   (Microphone / Court Feed)                      │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│              TRANSCRIPTION LAYER (Streaming)                     │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│   │   Whisper   │  │  Deepgram   │  │ AssemblyAI  │            │
│   │  (Local)    │  │  (Cloud)    │  │  (Cloud)    │            │
│   └─────────────┘  └─────────────┘  └─────────────┘            │
└─────────────────────┬───────────────────────────────────────────┘
                      │ Text chunks (streaming)
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                 ROSE GLASS LITIGATION LENS                       │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Speaker Diarization → Track testimony by witness         │  │
│  │  Temporal Anchoring → Timestamp all statements            │  │
│  │  GCT Variables → Ψ, ρ, q, f per utterance                │  │
│  │  Coherence Tracking → Rolling window analysis             │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│              CONTRADICTION DETECTION ENGINE                      │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Statement Registry → All testimony indexed               │  │
│  │  Semantic Similarity → Detect conflicting claims          │  │
│  │  Temporal Conflicts → "I never" vs prior "I did"         │  │
│  │  Document Cross-Ref → Compare to filed exhibits          │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│              CROSS-EXAMINATION PROMPT GENERATOR                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Flag Type → Contradiction / Coherence Drop / Evasion    │  │
│  │  Context Package → Prior statement + current statement   │  │
│  │  Suggested Question → "Earlier you said X, now Y?"       │  │
│  │  Priority Score → How damaging is this contradiction?    │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                   DELIVERY INTERFACE                             │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│   │   Tablet    │  │  Earpiece   │  │  Laptop     │            │
│   │  (Visual)   │  │  (Audio)    │  │  (Full UI)  │            │
│   └─────────────┘  └─────────────┘  └─────────────┘            │
└─────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Transcription Integration

### Option A: Local Whisper (Privacy-First)
```python
# litigation_transcriber.py

import whisper
import numpy as np
import sounddevice as sd
from queue import Queue
from threading import Thread
import time

class LocalWhisperTranscriber:
    """
    Local transcription using OpenAI Whisper.
    No data leaves the device - critical for attorney-client privilege.
    """
    
    def __init__(self, model_size: str = "base"):
        """
        model_size options: tiny, base, small, medium, large
        - tiny: Fastest, ~1GB VRAM, good for real-time
        - base: Balanced, ~1GB VRAM, recommended starting point
        - small: Better accuracy, ~2GB VRAM
        - medium: High accuracy, ~5GB VRAM
        - large: Best accuracy, ~10GB VRAM
        """
        self.model = whisper.load_model(model_size)
        self.audio_queue = Queue()
        self.is_running = False
        self.sample_rate = 16000
        self.chunk_duration = 5  # seconds per chunk
        
    def audio_callback(self, indata, frames, time_info, status):
        """Callback for sounddevice stream"""
        if status:
            print(f"Audio status: {status}")
        self.audio_queue.put(indata.copy())
        
    def start_stream(self):
        """Start capturing audio"""
        self.is_running = True
        self.stream = sd.InputStream(
            channels=1,
            samplerate=self.sample_rate,
            callback=self.audio_callback,
            blocksize=int(self.sample_rate * self.chunk_duration)
        )
        self.stream.start()
        
    def transcribe_chunk(self, audio_data: np.ndarray) -> dict:
        """Transcribe a single audio chunk"""
        result = self.model.transcribe(
            audio_data.flatten().astype(np.float32),
            language="en",
            fp16=False  # Set True if GPU available
        )
        return {
            "text": result["text"],
            "segments": result["segments"],
            "timestamp": time.time()
        }
        
    def stream_transcriptions(self):
        """Generator yielding transcriptions as they're produced"""
        audio_buffer = []
        
        while self.is_running:
            if not self.audio_queue.empty():
                chunk = self.audio_queue.get()
                audio_buffer.append(chunk)
                
                # Process when buffer has enough data
                if len(audio_buffer) >= 1:
                    combined = np.concatenate(audio_buffer)
                    result = self.transcribe_chunk(combined)
                    audio_buffer = []
                    yield result
```

### Option B: Deepgram Cloud (Lower Latency)
```python
# deepgram_transcriber.py

import asyncio
from deepgram import Deepgram
import json

class DeepgramTranscriber:
    """
    Cloud-based transcription with speaker diarization.
    Lower latency than local, but data leaves device.
    """
    
    def __init__(self, api_key: str):
        self.client = Deepgram(api_key)
        self.websocket = None
        
    async def connect(self, on_transcript):
        """Connect to Deepgram streaming API"""
        try:
            self.websocket = await self.client.transcription.live({
                'punctuate': True,
                'diarize': True,  # Speaker identification
                'interim_results': True,
                'language': 'en',
                'model': 'nova-2',  # Best accuracy
                'smart_format': True
            })
            
            self.websocket.registerHandler(
                self.websocket.event.TRANSCRIPT_RECEIVED,
                on_transcript
            )
            
            return self.websocket
            
        except Exception as e:
            print(f"Connection error: {e}")
            raise
            
    async def send_audio(self, audio_data: bytes):
        """Send audio chunk to Deepgram"""
        if self.websocket:
            await self.websocket.send(audio_data)
```

---

## Phase 2: Litigation Lens Module

```python
# src/litigation/litigation_lens.py

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import deque
import hashlib

from ..core.rose_glass_lens import RoseGlass, PatternVisibility
from ..core.rose_glass_v2 import RoseGlassV2


@dataclass
class TestimonyStatement:
    """A single statement from testimony"""
    speaker: str
    text: str
    timestamp: datetime
    transcript_position: int
    coherence: float
    variables: Dict[str, float]  # Ψ, ρ, q, f
    hash: str  # For quick comparison
    
    @classmethod
    def create(cls, speaker: str, text: str, position: int, 
               rose_glass: RoseGlass) -> 'TestimonyStatement':
        """Create statement with Rose Glass analysis"""
        
        # Analyze through lens
        # In production, use ML models for variable extraction
        psi = cls._extract_psi(text)
        rho = cls._extract_rho(text)
        q = cls._extract_q(text)
        f = cls._extract_f(text)
        
        visibility = rose_glass.view_through_lens(psi, rho, q, f)
        
        return cls(
            speaker=speaker,
            text=text,
            timestamp=datetime.now(),
            transcript_position=position,
            coherence=visibility.coherence,
            variables={'psi': psi, 'rho': rho, 'q': q, 'f': f},
            hash=hashlib.md5(text.lower().encode()).hexdigest()[:12]
        )
    
    @staticmethod
    def _extract_psi(text: str) -> float:
        """Extract internal consistency (simplified)"""
        # Check for self-contradicting language
        contradiction_markers = ['but', 'however', 'although', 'except']
        consistency_markers = ['therefore', 'because', 'since', 'so']
        
        words = text.lower().split()
        contradictions = sum(1 for w in words if w in contradiction_markers)
        consistencies = sum(1 for w in words if w in consistency_markers)
        
        if len(words) == 0:
            return 0.5
            
        return max(0.1, min(0.95, 0.7 + (consistencies - contradictions) * 0.1))
    
    @staticmethod
    def _extract_rho(text: str) -> float:
        """Extract accumulated wisdom/specificity"""
        # Specific details indicate depth
        specificity_markers = [
            'exactly', 'specifically', 'precisely', 
            'on', 'at', 'during'  # temporal markers
        ]
        vague_markers = [
            'maybe', 'perhaps', 'possibly', 'sometimes',
            'i think', 'i guess', 'sort of', 'kind of'
        ]
        
        text_lower = text.lower()
        specific = sum(1 for m in specificity_markers if m in text_lower)
        vague = sum(1 for m in vague_markers if m in text_lower)
        
        return max(0.1, min(0.95, 0.5 + (specific - vague) * 0.15))
    
    @staticmethod
    def _extract_q(text: str) -> float:
        """Extract moral activation energy"""
        emotional_markers = [
            'feel', 'felt', 'afraid', 'scared', 'angry',
            'upset', 'hurt', 'love', 'hate', 'worried'
        ]
        
        text_lower = text.lower()
        emotional = sum(1 for m in emotional_markers if m in text_lower)
        
        return min(0.95, emotional * 0.15)
    
    @staticmethod
    def _extract_f(text: str) -> float:
        """Extract social belonging architecture"""
        collective_markers = ['we', 'us', 'our', 'together', 'family']
        individual_markers = ['i', 'me', 'my', 'mine']
        
        words = text.lower().split()
        collective = sum(1 for w in words if w in collective_markers)
        individual = sum(1 for w in words if w in individual_markers)
        
        if collective + individual == 0:
            return 0.5
            
        return collective / (collective + individual + 1)



@dataclass
class ContradictionFlag:
    """A detected contradiction between statements"""
    statement_a: TestimonyStatement
    statement_b: TestimonyStatement
    contradiction_type: str  # 'direct', 'temporal', 'logical', 'emotional'
    severity: float  # 0.0 - 1.0
    suggested_question: str
    context: str
    
    def to_prompt(self) -> str:
        """Generate cross-examination prompt"""
        return f"""
CONTRADICTION DETECTED [{self.contradiction_type.upper()}]
Severity: {self.severity:.0%}

EARLIER: "{self.statement_a.text}"
  - Timestamp: {self.statement_a.timestamp.strftime('%H:%M:%S')}
  - Coherence: {self.statement_a.coherence:.2f}

NOW: "{self.statement_b.text}"
  - Timestamp: {self.statement_b.timestamp.strftime('%H:%M:%S')}
  - Coherence: {self.statement_b.coherence:.2f}

SUGGESTED QUESTION:
{self.suggested_question}

CONTEXT: {self.context}
"""


class ContradictionDetector:
    """
    Detects contradictions in testimony as they occur.
    Tracks all statements per speaker and cross-references.
    """
    
    def __init__(self, similarity_threshold: float = 0.7):
        self.statements_by_speaker: Dict[str, List[TestimonyStatement]] = {}
        self.contradictions: List[ContradictionFlag] = []
        self.similarity_threshold = similarity_threshold
        self.rose_glass = RoseGlass()
        
    def add_statement(self, speaker: str, text: str, 
                      position: int) -> Optional[ContradictionFlag]:
        """
        Add a new statement and check for contradictions.
        Returns contradiction if found.
        """
        statement = TestimonyStatement.create(
            speaker=speaker,
            text=text,
            position=position,
            rose_glass=self.rose_glass
        )
        
        if speaker not in self.statements_by_speaker:
            self.statements_by_speaker[speaker] = []
            
        # Check against prior statements from same speaker
        contradiction = self._check_contradictions(speaker, statement)
        
        # Store statement
        self.statements_by_speaker[speaker].append(statement)
        
        if contradiction:
            self.contradictions.append(contradiction)
            return contradiction
            
        return None
    
    def _check_contradictions(self, speaker: str, 
                              new_statement: TestimonyStatement) -> Optional[ContradictionFlag]:
        """Check new statement against speaker's history"""
        
        prior_statements = self.statements_by_speaker.get(speaker, [])
        
        for prior in prior_statements:
            # Check for direct contradictions
            direct = self._check_direct_contradiction(prior, new_statement)
            if direct:
                return direct
                
            # Check for temporal contradictions
            temporal = self._check_temporal_contradiction(prior, new_statement)
            if temporal:
                return temporal
                
            # Check for coherence drops
            coherence = self._check_coherence_drop(prior, new_statement)
            if coherence:
                return coherence
                
        return None
    
    def _check_direct_contradiction(self, prior: TestimonyStatement,
                                    current: TestimonyStatement) -> Optional[ContradictionFlag]:
        """
        Detect direct contradictions like:
        "I never did X" vs "When I did X..."
        """
        prior_lower = prior.text.lower()
        current_lower = current.text.lower()
        
        # Negation patterns
        never_patterns = ['never', 'did not', "didn't", 'have not', "haven't"]
        affirmation_patterns = ['when i', 'after i', 'i did', 'i have']
        
        prior_negates = any(p in prior_lower for p in never_patterns)
        current_affirms = any(p in current_lower for p in affirmation_patterns)
        
        if prior_negates and current_affirms:
            # Extract what's being negated/affirmed
            return ContradictionFlag(
                statement_a=prior,
                statement_b=current,
                contradiction_type='direct',
                severity=0.9,
                suggested_question=self._generate_direct_question(prior, current),
                context="Direct contradiction: negation followed by affirmation"
            )
            
        return None
    
    def _check_temporal_contradiction(self, prior: TestimonyStatement,
                                      current: TestimonyStatement) -> Optional[ContradictionFlag]:
        """
        Detect temporal contradictions like:
        "I was at home" vs "I was at the store" (same timeframe implied)
        """
        # Simplified: check for conflicting location/time claims
        # In production, use NLP for entity extraction
        
        location_words_prior = self._extract_locations(prior.text)
        location_words_current = self._extract_locations(current.text)
        
        if location_words_prior and location_words_current:
            if location_words_prior != location_words_current:
                return ContradictionFlag(
                    statement_a=prior,
                    statement_b=current,
                    contradiction_type='temporal',
                    severity=0.7,
                    suggested_question=self._generate_temporal_question(prior, current),
                    context="Possible conflicting locations/times"
                )
                
        return None
    
    def _check_coherence_drop(self, prior: TestimonyStatement,
                              current: TestimonyStatement) -> Optional[ContradictionFlag]:
        """
        Detect significant coherence drops indicating evasion or stress.
        """
        drop = prior.coherence - current.coherence
        
        if drop > 0.3:  # Significant drop
            return ContradictionFlag(
                statement_a=prior,
                statement_b=current,
                contradiction_type='coherence_drop',
                severity=min(1.0, drop),
                suggested_question=self._generate_coherence_question(prior, current),
                context=f"Coherence dropped {drop:.0%} - possible evasion"
            )
            
        return None
    
    def _extract_locations(self, text: str) -> set:
        """Extract location references from text"""
        # Simplified - in production use spaCy NER
        location_markers = ['at', 'in', 'to', 'from', 'home', 'work', 'store']
        words = text.lower().split()
        locations = set()
        
        for i, word in enumerate(words):
            if word in ['at', 'in', 'to', 'from'] and i + 1 < len(words):
                locations.add(words[i + 1])
                
        return locations
    
    def _generate_direct_question(self, prior: TestimonyStatement,
                                  current: TestimonyStatement) -> str:
        """Generate cross-examination question for direct contradiction"""
        return f"""You testified earlier that "{prior.text[:100]}..."
        
But just now you said "{current.text[:100]}..."

Can you explain this discrepancy?"""

    def _generate_temporal_question(self, prior: TestimonyStatement,
                                    current: TestimonyStatement) -> str:
        """Generate question for temporal/location contradiction"""
        return f"""Earlier you mentioned being "{prior.text[:80]}..."

Now you're saying "{current.text[:80]}..."

Which account is accurate?"""

    def _generate_coherence_question(self, prior: TestimonyStatement,
                                     current: TestimonyStatement) -> str:
        """Generate question targeting coherence drop"""
        return f"""You seemed very certain when you said "{prior.text[:80]}..."

Now you appear less sure. Why is that?"""


---

## Phase 3: Real-Time Pipeline Integration

```python
# src/litigation/realtime_pipeline.py

import asyncio
from typing import AsyncIterator, Callable, Optional
from dataclasses import dataclass
from datetime import datetime
import json


@dataclass
class PipelineOutput:
    """Output from real-time litigation pipeline"""
    transcript: str
    speaker: str
    timestamp: datetime
    coherence: float
    contradiction: Optional[ContradictionFlag]
    cross_prompt: Optional[str]
    priority: str  # 'low', 'medium', 'high', 'critical'
    

class RealtimeLitigationPipeline:
    """
    Complete real-time pipeline:
    Audio → Transcription → Rose Analysis → Contradiction Detection → Prompts
    """
    
    def __init__(self, 
                 transcriber_type: str = "whisper",
                 api_key: Optional[str] = None):
        """
        Initialize pipeline.
        
        transcriber_type: "whisper" (local) or "deepgram" (cloud)
        api_key: Required for cloud transcription
        """
        self.transcriber_type = transcriber_type
        self.api_key = api_key
        
        self.detector = ContradictionDetector()
        self.rose_glass = RoseGlass()
        self.statement_count = 0
        self.current_speaker = "WITNESS"
        
        self._setup_transcriber()
        
    def _setup_transcriber(self):
        """Initialize the transcription engine"""
        if self.transcriber_type == "whisper":
            self.transcriber = LocalWhisperTranscriber(model_size="base")
        elif self.transcriber_type == "deepgram":
            if not self.api_key:
                raise ValueError("API key required for Deepgram")
            self.transcriber = DeepgramTranscriber(self.api_key)
        else:
            raise ValueError(f"Unknown transcriber: {self.transcriber_type}")
    
    def set_speaker(self, speaker: str):
        """Set current speaker (for when diarization isn't available)"""
        self.current_speaker = speaker
        
    async def process_stream(self) -> AsyncIterator[PipelineOutput]:
        """
        Main processing loop.
        Yields PipelineOutput objects as testimony is processed.
        """
        if self.transcriber_type == "whisper":
            self.transcriber.start_stream()
            
            for transcription in self.transcriber.stream_transcriptions():
                output = await self._process_transcription(transcription)
                yield output
                
        elif self.transcriber_type == "deepgram":
            async def handle_transcript(result):
                output = await self._process_transcription(result)
                await self._output_queue.put(output)
                
            self._output_queue = asyncio.Queue()
            await self.transcriber.connect(handle_transcript)
            
            while True:
                output = await self._output_queue.get()
                yield output
    
    async def _process_transcription(self, transcription: dict) -> PipelineOutput:
        """Process a single transcription chunk through Rose Glass"""
        text = transcription.get("text", "").strip()
        speaker = transcription.get("speaker", self.current_speaker)
        
        if not text:
            return PipelineOutput(
                transcript="",
                speaker=speaker,
                timestamp=datetime.now(),
                coherence=0.0,
                contradiction=None,
                cross_prompt=None,
                priority="low"
            )
        
        self.statement_count += 1
        
        # Add to contradiction detector
        contradiction = self.detector.add_statement(
            speaker=speaker,
            text=text,
            position=self.statement_count
        )
        
        # Get latest coherence
        if speaker in self.detector.statements_by_speaker:
            latest = self.detector.statements_by_speaker[speaker][-1]
            coherence = latest.coherence
        else:
            coherence = 0.5
            
        # Determine priority
        if contradiction and contradiction.severity > 0.8:
            priority = "critical"
        elif contradiction and contradiction.severity > 0.5:
            priority = "high"
        elif contradiction:
            priority = "medium"
        else:
            priority = "low"
            
        return PipelineOutput(
            transcript=text,
            speaker=speaker,
            timestamp=datetime.now(),
            coherence=coherence,
            contradiction=contradiction,
            cross_prompt=contradiction.to_prompt() if contradiction else None,
            priority=priority
        )
    
    def get_all_contradictions(self) -> List[ContradictionFlag]:
        """Get all detected contradictions"""
        return self.detector.contradictions
    
    def get_speaker_coherence_history(self, speaker: str) -> List[float]:
        """Get coherence timeline for a speaker"""
        statements = self.detector.statements_by_speaker.get(speaker, [])
        return [s.coherence for s in statements]
    
    def export_transcript(self, filepath: str):
        """Export full transcript with analysis"""
        export = {
            "generated": datetime.now().isoformat(),
            "total_statements": self.statement_count,
            "contradictions_found": len(self.detector.contradictions),
            "speakers": {}
        }
        
        for speaker, statements in self.detector.statements_by_speaker.items():
            export["speakers"][speaker] = [
                {
                    "position": s.transcript_position,
                    "timestamp": s.timestamp.isoformat(),
                    "text": s.text,
                    "coherence": s.coherence,
                    "variables": s.variables
                }
                for s in statements
            ]
            
        export["contradictions"] = [
            {
                "type": c.contradiction_type,
                "severity": c.severity,
                "statement_a": c.statement_a.text,
                "statement_b": c.statement_b.text,
                "suggested_question": c.suggested_question
            }
            for c in self.detector.contradictions
        ]
        
        with open(filepath, 'w') as f:
            json.dump(export, f, indent=2)
```

---

## Phase 4: Delivery Interface

### Tablet Interface (Flask + WebSocket)
```python
# src/litigation/delivery/tablet_server.py

from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import asyncio
from threading import Thread

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

pipeline = None  # Initialize globally


@app.route('/')
def index():
    return render_template('litigation_dashboard.html')


@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('status', {'message': 'Connected to Rose Glass Litigation Support'})


@socketio.on('start_listening')
def handle_start():
    global pipeline
    pipeline = RealtimeLitigationPipeline(transcriber_type="whisper")
    
    def run_pipeline():
        async def stream():
            async for output in pipeline.process_stream():
                socketio.emit('update', {
                    'transcript': output.transcript,
                    'speaker': output.speaker,
                    'coherence': output.coherence,
                    'priority': output.priority,
                    'cross_prompt': output.cross_prompt
                })
        asyncio.run(stream())
    
    Thread(target=run_pipeline, daemon=True).start()
    emit('status', {'message': 'Listening started'})


@socketio.on('set_speaker')
def handle_speaker(data):
    if pipeline:
        pipeline.set_speaker(data['speaker'])
        emit('status', {'message': f'Speaker set to {data["speaker"]}'})


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)
```

### Dashboard Template
```html
<!-- templates/litigation_dashboard.html -->
<!DOCTYPE html>
<html>
<head>
    <title>Rose Glass - Litigation Support</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    <style>
        body { 
            font-family: -apple-system, BlinkMacSystemFont, sans-serif;
            background: #1a1a2e; 
            color: #eee;
            padding: 20px;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { 
            display: flex; 
            justify-content: space-between; 
            align-items: center;
            border-bottom: 1px solid #333;
            padding-bottom: 10px;
        }
        .coherence-meter {
            width: 200px;
            height: 20px;
            background: #333;
            border-radius: 10px;
            overflow: hidden;
        }
        .coherence-fill {
            height: 100%;
            transition: width 0.3s, background 0.3s;
        }
        .transcript-area {
            background: #0f0f1a;
            border-radius: 8px;
            padding: 15px;
            margin: 20px 0;
            height: 200px;
            overflow-y: auto;
        }
        .alert-area {
            background: #2a0a0a;
            border: 2px solid #ff4444;
            border-radius: 8px;
            padding: 15px;
            margin: 20px 0;
            display: none;
        }
        .alert-area.active { display: block; }
        .alert-area.critical { 
            animation: pulse 1s infinite;
        }
        @keyframes pulse {
            0% { border-color: #ff4444; }
            50% { border-color: #ff0000; }
            100% { border-color: #ff4444; }
        }
        .prompt-text {
            font-family: 'Courier New', monospace;
            white-space: pre-wrap;
            font-size: 14px;
        }
        .speaker-buttons button {
            padding: 10px 20px;
            margin: 5px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        .speaker-buttons button.active { background: #4CAF50; color: white; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🌹 Rose Glass Litigation Support</h1>
            <div class="coherence-meter">
                <div class="coherence-fill" id="coherence-fill" style="width: 50%; background: #4CAF50;"></div>
            </div>
        </div>
        
        <div class="speaker-buttons">
            <button onclick="setSpeaker('WITNESS')" id="btn-witness">WITNESS</button>
            <button onclick="setSpeaker('PETITIONER')" id="btn-petitioner">PETITIONER</button>
            <button onclick="setSpeaker('RESPONDENT')" id="btn-respondent">RESPONDENT</button>
            <button onclick="setSpeaker('ATTORNEY')" id="btn-attorney">OPP. COUNSEL</button>
        </div>
        
        <div class="transcript-area" id="transcript"></div>
        
        <div class="alert-area" id="alert-area">
            <h3>⚠️ CONTRADICTION DETECTED</h3>
            <div class="prompt-text" id="cross-prompt"></div>
        </div>
        
        <button onclick="startListening()" style="padding: 15px 30px; font-size: 18px;">
            🎙️ START LISTENING
        </button>
    </div>
    
    <script>
        const socket = io();
        let currentSpeaker = 'WITNESS';
        
        socket.on('update', (data) => {
            // Update transcript
            const transcript = document.getElementById('transcript');
            transcript.innerHTML += `<p><strong>${data.speaker}:</strong> ${data.transcript}</p>`;
            transcript.scrollTop = transcript.scrollHeight;
            
            // Update coherence meter
            const fill = document.getElementById('coherence-fill');
            const percent = data.coherence * 100;
            fill.style.width = percent + '%';
            fill.style.background = percent > 60 ? '#4CAF50' : percent > 30 ? '#ff9800' : '#f44336';
            
            // Show alert if contradiction
            if (data.cross_prompt) {
                const alert = document.getElementById('alert-area');
                const prompt = document.getElementById('cross-prompt');
                
                alert.classList.add('active');
                if (data.priority === 'critical') {
                    alert.classList.add('critical');
                }
                prompt.textContent = data.cross_prompt;
                
                // Auto-hide after 30 seconds
                setTimeout(() => {
                    alert.classList.remove('active', 'critical');
                }, 30000);
            }
        });
        
        function startListening() {
            socket.emit('start_listening');
        }
        
        function setSpeaker(speaker) {
            currentSpeaker = speaker;
            socket.emit('set_speaker', { speaker: speaker });
            
            document.querySelectorAll('.speaker-buttons button').forEach(b => b.classList.remove('active'));
            document.getElementById('btn-' + speaker.toLowerCase()).classList.add('active');
        }
    </script>
</body>
</html>
```

---

## Phase 5: Claude API Integration for Advanced Analysis

```python
# src/litigation/claude_integration.py

import anthropic
from typing import List, Dict, Optional

class ClaudeLitigationAnalyzer:
    """
    Use Claude API for advanced contradiction analysis
    and cross-examination question generation.
    """
    
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.system_prompt = """You are an expert litigation support AI analyzing courtroom testimony in real-time.

Your role:
1. Identify contradictions between statements
2. Generate strategic cross-examination questions
3. Flag credibility issues
4. Track narrative coherence

When generating questions:
- Be precise and targeted
- Reference specific testimony
- Avoid leading questions (unless in cross-examination of adverse witness)
- Consider Michigan Rules of Evidence

You are integrated with Rose Glass, a coherence translation system that provides:
- Ψ (Psi): Internal consistency score
- ρ (Rho): Wisdom/specificity depth
- q: Emotional activation
- f: Social belonging patterns

Use these metrics to inform your analysis."""

    async def analyze_contradiction(self, 
                                    statement_a: str,
                                    statement_b: str,
                                    context: Dict) -> Dict:
        """Deep analysis of a potential contradiction"""
        
        message = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=self.system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": f"""Analyze this potential contradiction:

EARLIER STATEMENT:
"{statement_a}"
Coherence: {context.get('coherence_a', 'N/A')}
Time: {context.get('time_a', 'N/A')}

CURRENT STATEMENT:
"{statement_b}"
Coherence: {context.get('coherence_b', 'N/A')}
Time: {context.get('time_b', 'N/A')}

GCT Variables:
- Ψ change: {context.get('psi_change', 'N/A')}
- ρ change: {context.get('rho_change', 'N/A')}
- q change: {context.get('q_change', 'N/A')}

Provide:
1. Is this a true contradiction? (Yes/No/Partial)
2. Type of contradiction (direct/temporal/logical/emotional)
3. Severity (0.0-1.0)
4. Three targeted cross-examination questions
5. Recommended follow-up strategy"""
                }
            ]
        )
        
        return {
            "analysis": message.content[0].text,
            "model": "claude-sonnet-4-20250514"
        }
    
    async def generate_cross_questions(self, 
                                       testimony_history: List[str],
                                       current_statement: str,
                                       case_context: str) -> List[str]:
        """Generate cross-examination questions based on testimony pattern"""
        
        history_text = "\n".join([f"- {t}" for t in testimony_history[-10:]])
        
        message = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=512,
            system=self.system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": f"""Case Context: {case_context}

Recent Testimony History:
{history_text}

Current Statement: "{current_statement}"

Generate 3 strategic cross-examination questions that:
1. Exploit any inconsistencies
2. Test the credibility of claims
3. Advance my case theory

Format: Just the questions, numbered 1-3."""
                }
            ]
        )
        
        # Parse questions from response
        lines = message.content[0].text.strip().split('\n')
        questions = [l.lstrip('0123456789. ') for l in lines if l.strip()]
        
        return questions[:3]
```

---

## Installation & Setup

### Requirements
```txt
# requirements-litigation.txt

# Transcription
openai-whisper>=20230918
sounddevice>=0.4.6
deepgram-sdk>=2.12.0

# Core Rose Glass
torch>=2.0.0
transformers>=4.30.0
numpy>=1.24.0

# Web Interface
flask>=2.3.0
flask-socketio>=5.3.0
python-socketio>=5.8.0

# Claude Integration
anthropic>=0.18.0

# NLP (for advanced extraction)
spacy>=3.5.0
sentence-transformers>=2.2.0
```

### Quick Start
```bash
# 1. Clone Rose Glass if needed
cd /Users/chris/rose-glass

# 2. Install litigation dependencies
pip install -r requirements-litigation.txt

# 3. Download Whisper model
python -c "import whisper; whisper.load_model('base')"

# 4. Download spaCy model (optional, for advanced NER)
python -m spacy download en_core_web_sm

# 5. Start the tablet server
python -m src.litigation.delivery.tablet_server

# 6. Open browser to http://localhost:5000
```

---

## Integration with Existing Rose Glass

The litigation module extends the core Rose Glass pipeline:

```
rose-glass/
├── src/
│   ├── core/
│   │   ├── rose_glass_lens.py      # Core lens (unchanged)
│   │   ├── rose_glass_v2.py        # Enhanced lens (unchanged)
│   │   └── rose_glass_pipeline.py  # Base pipeline (unchanged)
│   │
│   ├── litigation/                 # NEW MODULE
│   │   ├── __init__.py
│   │   ├── litigation_lens.py      # TestimonyStatement, ContradictionFlag
│   │   ├── contradiction_detector.py
│   │   ├── realtime_pipeline.py    # RealtimeLitigationPipeline
│   │   ├── claude_integration.py   # Claude API for advanced analysis
│   │   └── delivery/
│   │       ├── tablet_server.py
│   │       └── templates/
│   │           └── litigation_dashboard.html
│   │
│   └── ... (existing modules)
│
├── docs/
│   ├── REALTIME_LITIGATION_BUILD_GUIDE.md  # This document
│   └── ... (existing docs)
│
└── requirements-litigation.txt
```

---

## Development Roadmap

### Phase 1: MVP (December 2025)
- [x] Architecture design
- [ ] Local Whisper transcription
- [ ] Basic contradiction detection
- [ ] Tablet dashboard

### Phase 2: Enhancement (January 2026)
- [ ] Claude API integration
- [ ] Document cross-reference (compare to filed exhibits)
- [ ] Speaker diarization
- [ ] Coherence trend visualization

### Phase 3: Production (Q1 2026)
- [ ] Court-approved recording integration
- [ ] Historical case pattern matching
- [ ] Multi-case witness tracking
- [ ] Export to legal document format

---

## Legal Considerations

### Attorney-Client Privilege
- Local processing (Whisper) keeps data on device
- Cloud processing requires confidentiality agreements
- Audio recordings may require court permission

### Evidence Preservation
- All transcripts should be preserved as work product
- Timestamps enable synchronization with official record
- Export functionality for case file integration

### Court Rules
- Check local rules on electronic devices in courtroom
- Some courts restrict wireless communication
- Tablet interface designed for silent operation

---

## Origin Story

This build guide emerged from a post-hearing debrief on December 5, 2025. After a five-hour hearing where pro se representation achieved:

- 55% sustained objections against opposing counsel
- Successful cross-examination using strategic bluffing
- Witness contradictions exposed through pattern recognition
- Institutional credibility established with the court

The insight: the coherence detection already happening intuitively during cross-examination could be systematized and enhanced with real-time analysis.

*"The second chair that never misses."*

---

**Author**: Christopher MacGregor bin Joseph  
**Framework**: Rose Glass / GCT  
**Date**: December 5, 2025  
**Status**: Architecture Complete, Implementation Ready
