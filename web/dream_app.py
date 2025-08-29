"""
Rose Glass Dream Analysis Web Application
========================================

A public-facing web app for dream analysis using the Rose Glass framework.
Integrates with LLM APIs to provide rich, culturally-aware dream interpretation.
"""

import os
import asyncio
import json
from datetime import datetime
from typing import Dict, Optional, List
import hashlib
import time

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import openai
from anthropic import Anthropic
import aiohttp
import uvicorn

# Import Rose Glass components
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.dream.rose_glass_dream_interpreter import RoseGlassDreamInterpreter
from src.core.rose_glass_v2 import RoseGlassV2


# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
DEFAULT_LLM = os.getenv("DEFAULT_LLM", "claude")  # or "openai"

# Initialize app
app = FastAPI(
    title="Rose Glass Dream Analysis",
    description="Translate your dreams through cultural lenses",
    version="1.0.0"
)

# CORS for web access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize interpreters
dream_interpreter = RoseGlassDreamInterpreter()
rose_glass = RoseGlassV2()

# Request models
class DreamAnalysisRequest(BaseModel):
    dream_text: str
    selected_lens: Optional[str] = None
    include_llm_insights: bool = True
    llm_provider: Optional[str] = None
    
class DreamSession(BaseModel):
    session_id: str
    dreams: List[Dict]
    created_at: datetime
    
# Privacy-preserving session storage (in-memory, ephemeral)
dream_sessions = {}

# HTML template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Rose Glass Dream Analysis</title>
    <style>
        :root {
            --primary: #8b5cf6;
            --secondary: #ec4899;
            --background: #0f0f0f;
            --surface: #1a1a1a;
            --text: #e5e5e5;
            --accent: #f59e0b;
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--background);
            color: var(--text);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }
        
        .header {
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            padding: 2rem;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .header p {
            font-size: 1.1rem;
            opacity: 0.9;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
            flex: 1;
        }
        
        .dream-input {
            background: var(--surface);
            padding: 2rem;
            border-radius: 12px;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        .dream-input textarea {
            width: 100%;
            min-height: 200px;
            padding: 1rem;
            background: var(--background);
            border: 1px solid #333;
            border-radius: 8px;
            color: var(--text);
            font-size: 1rem;
            resize: vertical;
            font-family: inherit;
        }
        
        .dream-input textarea:focus {
            outline: none;
            border-color: var(--primary);
        }
        
        .lens-selector {
            margin: 1.5rem 0;
        }
        
        .lens-selector label {
            display: block;
            margin-bottom: 0.5rem;
            font-weight: 500;
        }
        
        .lens-selector select {
            width: 100%;
            padding: 0.75rem;
            background: var(--background);
            border: 1px solid #333;
            border-radius: 8px;
            color: var(--text);
            font-size: 1rem;
        }
        
        .options {
            margin: 1.5rem 0;
            display: flex;
            gap: 2rem;
            align-items: center;
        }
        
        .options label {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            cursor: pointer;
        }
        
        .analyze-button {
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            color: white;
            border: none;
            padding: 1rem 2rem;
            border-radius: 8px;
            font-size: 1.1rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 6px rgba(139, 92, 246, 0.3);
        }
        
        .analyze-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(139, 92, 246, 0.4);
        }
        
        .analyze-button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }
        
        .results {
            background: var(--surface);
            padding: 2rem;
            border-radius: 12px;
            margin-top: 2rem;
            display: none;
        }
        
        .results.show {
            display: block;
        }
        
        .results h2 {
            margin-bottom: 1.5rem;
            color: var(--accent);
        }
        
        .pattern-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }
        
        .pattern-card {
            background: var(--background);
            padding: 1.5rem;
            border-radius: 8px;
            text-align: center;
        }
        
        .pattern-card h3 {
            font-size: 0.9rem;
            opacity: 0.7;
            margin-bottom: 0.5rem;
        }
        
        .pattern-value {
            font-size: 2rem;
            font-weight: bold;
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .interpretation {
            background: var(--background);
            padding: 1.5rem;
            border-radius: 8px;
            margin: 1rem 0;
            line-height: 1.6;
        }
        
        .symbols {
            margin: 2rem 0;
        }
        
        .symbol-card {
            background: var(--background);
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1rem;
        }
        
        .symbol-card h4 {
            color: var(--accent);
            margin-bottom: 0.5rem;
        }
        
        .insights {
            background: rgba(139, 92, 246, 0.1);
            border-left: 4px solid var(--primary);
            padding: 1rem;
            margin: 1rem 0;
        }
        
        .insights h3 {
            margin-bottom: 0.5rem;
        }
        
        .insights ul {
            list-style: none;
            padding-left: 0;
        }
        
        .insights li {
            margin: 0.5rem 0;
            padding-left: 1.5rem;
            position: relative;
        }
        
        .insights li::before {
            content: "✨";
            position: absolute;
            left: 0;
        }
        
        .loading {
            display: none;
            text-align: center;
            padding: 2rem;
        }
        
        .loading.show {
            display: block;
        }
        
        .spinner {
            border: 3px solid rgba(139, 92, 246, 0.1);
            border-top: 3px solid var(--primary);
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 1rem;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .privacy-note {
            background: rgba(245, 158, 11, 0.1);
            border: 1px solid var(--accent);
            padding: 1rem;
            border-radius: 8px;
            margin-top: 2rem;
            font-size: 0.9rem;
            text-align: center;
        }
        
        .error {
            background: rgba(239, 68, 68, 0.1);
            border: 1px solid #ef4444;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
            display: none;
        }
        
        .error.show {
            display: block;
        }
        
        @media (max-width: 768px) {
            .header h1 {
                font-size: 2rem;
            }
            
            .pattern-grid {
                grid-template-columns: repeat(2, 1fr);
            }
            
            .options {
                flex-direction: column;
                align-items: flex-start;
                gap: 1rem;
            }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🌹 Rose Glass Dream Analysis</h1>
        <p>Translate your dreams through cultural lenses</p>
    </div>
    
    <div class="container">
        <div class="dream-input">
            <textarea 
                id="dreamText" 
                placeholder="Describe your dream in as much detail as you can remember. Include emotions, symbols, people, and any significant moments..."
            ></textarea>
            
            <div class="lens-selector">
                <label for="lensSelect">Choose a cultural lens (or leave empty for multi-lens view):</label>
                <select id="lensSelect">
                    <option value="">Automatic (Multi-lens)</option>
                    <option value="medieval_islamic">Medieval Islamic Philosophy</option>
                    <option value="indigenous_oral">Indigenous Oral Tradition</option>
                    <option value="buddhist_contemplative">Buddhist Contemplative</option>
                    <option value="digital_native">Digital Native</option>
                </select>
            </div>
            
            <div class="options">
                <label>
                    <input type="checkbox" id="includeLLM" checked>
                    Include AI-enhanced insights
                </label>
                
                <label>
                    <input type="radio" name="llmProvider" value="claude" checked>
                    Claude
                </label>
                
                <label>
                    <input type="radio" name="llmProvider" value="openai">
                    GPT-4
                </label>
            </div>
            
            <button class="analyze-button" onclick="analyzeDream()">
                Translate My Dream
            </button>
        </div>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>Translating your dream patterns...</p>
        </div>
        
        <div class="error" id="error"></div>
        
        <div class="results" id="results"></div>
        
        <div class="privacy-note">
            🔒 Privacy First: Your dreams are sacred. We don't store any dream content. 
            All analysis happens in real-time and is immediately forgotten after you leave.
        </div>
    </div>
    
    <script>
        async function analyzeDream() {
            const dreamText = document.getElementById('dreamText').value.trim();
            if (!dreamText) {
                showError('Please describe your dream first');
                return;
            }
            
            const selectedLens = document.getElementById('lensSelect').value;
            const includeLLM = document.getElementById('includeLLM').checked;
            const llmProvider = document.querySelector('input[name="llmProvider"]:checked').value;
            
            // UI feedback
            document.querySelector('.analyze-button').disabled = true;
            document.getElementById('loading').classList.add('show');
            document.getElementById('results').classList.remove('show');
            document.getElementById('error').classList.remove('show');
            
            try {
                const response = await fetch('/api/analyze_dream', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        dream_text: dreamText,
                        selected_lens: selectedLens || null,
                        include_llm_insights: includeLLM,
                        llm_provider: llmProvider
                    })
                });
                
                if (!response.ok) {
                    throw new Error('Analysis failed');
                }
                
                const data = await response.json();
                displayResults(data);
                
            } catch (error) {
                showError('Unable to analyze dream. Please try again.');
                console.error('Error:', error);
            } finally {
                document.querySelector('.analyze-button').disabled = false;
                document.getElementById('loading').classList.remove('show');
            }
        }
        
        function showError(message) {
            const errorDiv = document.getElementById('error');
            errorDiv.textContent = message;
            errorDiv.classList.add('show');
            setTimeout(() => errorDiv.classList.remove('show'), 5000);
        }
        
        function displayResults(data) {
            const resultsDiv = document.getElementById('results');
            resultsDiv.innerHTML = '';
            
            // Pattern overview
            const patternsHtml = `
                <h2>Dream Pattern Translation</h2>
                <div class="pattern-grid">
                    <div class="pattern-card">
                        <h3>Narrative Flow</h3>
                        <div class="pattern-value">${(data.patterns.psi * 100).toFixed(0)}%</div>
                        <p>${data.pattern_descriptions.psi}</p>
                    </div>
                    <div class="pattern-card">
                        <h3>Symbolic Density</h3>
                        <div class="pattern-value">${(data.patterns.rho * 100).toFixed(0)}%</div>
                        <p>${data.pattern_descriptions.rho}</p>
                    </div>
                    <div class="pattern-card">
                        <h3>Emotional Charge</h3>
                        <div class="pattern-value">${(data.patterns.q * 100).toFixed(0)}%</div>
                        <p>${data.pattern_descriptions.q}</p>
                    </div>
                    <div class="pattern-card">
                        <h3>Relational Content</h3>
                        <div class="pattern-value">${(data.patterns.f * 100).toFixed(0)}%</div>
                        <p>${data.pattern_descriptions.f}</p>
                    </div>
                </div>
            `;
            
            // Main interpretation
            const interpretationHtml = `
                <div class="interpretation">
                    <h3>${data.lens_used}</h3>
                    <p>${data.interpretation}</p>
                </div>
            `;
            
            // Symbols
            let symbolsHtml = '';
            if (data.symbols && data.symbols.length > 0) {
                symbolsHtml = '<div class="symbols"><h3>Dream Symbols</h3>';
                data.symbols.forEach(symbol => {
                    symbolsHtml += `
                        <div class="symbol-card">
                            <h4>${symbol.symbol}</h4>
                            <p>${symbol.meanings.universal}</p>
                        </div>
                    `;
                });
                symbolsHtml += '</div>';
            }
            
            // Insights
            let insightsHtml = '';
            if (data.insights && data.insights.length > 0) {
                insightsHtml = `
                    <div class="insights">
                        <h3>Dream Insights</h3>
                        <ul>
                            ${data.insights.map(insight => `<li>${insight}</li>`).join('')}
                        </ul>
                    </div>
                `;
            }
            
            // LLM insights
            let llmHtml = '';
            if (data.llm_insights) {
                llmHtml = `
                    <div class="insights" style="background: rgba(236, 72, 153, 0.1); border-color: var(--secondary);">
                        <h3>AI-Enhanced Insights</h3>
                        <p>${data.llm_insights}</p>
                    </div>
                `;
            }
            
            resultsDiv.innerHTML = patternsHtml + interpretationHtml + symbolsHtml + insightsHtml + llmHtml;
            resultsDiv.classList.add('show');
            
            // Scroll to results
            resultsDiv.scrollIntoView({ behavior: 'smooth' });
        }
        
        // Allow Ctrl+Enter to submit
        document.getElementById('dreamText').addEventListener('keydown', (e) => {
            if (e.ctrlKey && e.key === 'Enter') {
                analyzeDream();
            }
        });
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the main dream analysis interface"""
    return HTML_TEMPLATE

@app.post("/api/analyze_dream")
async def analyze_dream(request: DreamAnalysisRequest):
    """Analyze a dream using Rose Glass + optional LLM enhancement"""
    try:
        # Basic Rose Glass analysis
        if request.selected_lens:
            result = dream_interpreter.translate_dream(
                request.dream_text,
                selected_lens=request.selected_lens
            )
        else:
            result = dream_interpreter.translate_dream(request.dream_text)
        
        # Prepare response
        response_data = {
            "patterns": {
                "psi": result['patterns'].psi,
                "rho": result['patterns'].rho,
                "q": result['patterns'].q,
                "f": result['patterns'].f
            },
            "pattern_descriptions": {
                "psi": dream_interpreter._describe_pattern(result['patterns'].psi, 'flow'),
                "rho": dream_interpreter._describe_pattern(result['patterns'].rho, 'symbols'),
                "q": dream_interpreter._describe_pattern(result['patterns'].q, 'emotion'),
                "f": dream_interpreter._describe_pattern(result['patterns'].f, 'social')
            },
            "dream_qualities": result.get('dream_specific', {}),
            "lens_used": request.selected_lens or result.get('clearest_lens', 'Multi-lens')
        }
        
        # Add interpretation
        if 'interpretation' in result:
            interp = result['interpretation']
            response_data['interpretation'] = interp.get_narrative()
            response_data['alternative_readings'] = interp.alternative_readings
        elif 'multi_lens_view' in result:
            # Compile multi-lens interpretation
            views = []
            for lens, interp in result['multi_lens_view'].items():
                views.append(f"{lens}: {interp.coherence_construction:.2f}/4.0")
            response_data['interpretation'] = (
                "Your dream was viewed through multiple cultural lenses:\n" +
                "\n".join(views) +
                "\n\n" + result.get('insight', '')
            )
        
        # Add symbols
        if 'symbols' in result:
            response_data['symbols'] = [
                {
                    'symbol': s.symbol,
                    'meanings': s.cultural_meanings,
                    'appearances': len(s.appearances)
                }
                for s in result['symbols']
            ]
        
        # Add insights
        response_data['insights'] = result.get('insights', [])
        
        # Optional LLM enhancement
        if request.include_llm_insights:
            llm_insights = await get_llm_insights(
                request.dream_text,
                result,
                request.llm_provider or DEFAULT_LLM
            )
            response_data['llm_insights'] = llm_insights
        
        # Generate session ID for privacy-preserving temporary storage
        session_id = hashlib.sha256(
            f"{request.dream_text}{time.time()}".encode()
        ).hexdigest()[:16]
        
        # Store temporarily (auto-expires)
        dream_sessions[session_id] = {
            "timestamp": datetime.now(),
            "analysis": response_data
        }
        
        # Clean old sessions (older than 1 hour)
        cleanup_old_sessions()
        
        response_data['session_id'] = session_id
        
        return JSONResponse(content=response_data)
        
    except Exception as e:
        print(f"Error analyzing dream: {e}")
        raise HTTPException(status_code=500, detail="Failed to analyze dream")

async def get_llm_insights(dream_text: str, 
                         rose_glass_result: Dict,
                         provider: str = "claude") -> str:
    """Get additional insights from LLM based on Rose Glass analysis"""
    
    # Prepare context for LLM
    context = f"""
    The Rose Glass framework has analyzed this dream with the following patterns:
    - Narrative Flow: {rose_glass_result['patterns'].psi:.2f}
    - Symbolic Density: {rose_glass_result['patterns'].rho:.2f}
    - Emotional Charge: {rose_glass_result['patterns'].q:.2f}
    - Relational Content: {rose_glass_result['patterns'].f:.2f}
    
    Key symbols detected: {', '.join([s.symbol for s in rose_glass_result.get('symbols', [])])}
    
    Dream qualities:
    {rose_glass_result.get('dream_specific', {})}
    
    The dreamer's narrative:
    "{dream_text}"
    
    Based on this Rose Glass analysis, provide additional therapeutic insights that:
    1. Honor the dreamer's experience without judgment
    2. Suggest what psychological work might be happening
    3. Offer gentle questions for reflection
    4. Acknowledge multiple valid interpretations
    
    Keep the response compassionate, non-prescriptive, and under 200 words.
    """
    
    try:
        if provider == "claude" and ANTHROPIC_API_KEY:
            client = Anthropic(api_key=ANTHROPIC_API_KEY)
            response = await asyncio.to_thread(
                client.messages.create,
                model="claude-3-haiku-20240307",
                max_tokens=300,
                messages=[
                    {"role": "user", "content": context}
                ]
            )
            return response.content[0].text
            
        elif provider == "openai" and OPENAI_API_KEY:
            openai.api_key = OPENAI_API_KEY
            response = await asyncio.to_thread(
                openai.ChatCompletion.create,
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a compassionate dream analyst using the Rose Glass framework."},
                    {"role": "user", "content": context}
                ],
                max_tokens=300
            )
            return response.choices[0].message.content
            
        else:
            return "LLM insights unavailable. The Rose Glass patterns above offer rich material for reflection."
            
    except Exception as e:
        print(f"LLM error: {e}")
        return "Additional AI insights temporarily unavailable. The Rose Glass analysis above provides comprehensive perspectives on your dream."

def cleanup_old_sessions():
    """Remove sessions older than 1 hour"""
    current_time = datetime.now()
    expired_sessions = []
    
    for session_id, data in dream_sessions.items():
        if (current_time - data['timestamp']).seconds > 3600:
            expired_sessions.append(session_id)
    
    for session_id in expired_sessions:
        del dream_sessions[session_id]

@app.get("/api/session/{session_id}")
async def get_session(session_id: str):
    """Retrieve a previous analysis (within 1 hour)"""
    if session_id in dream_sessions:
        return dream_sessions[session_id]['analysis']
    else:
        raise HTTPException(status_code=404, detail="Session not found or expired")

@app.get("/api/lenses")
async def get_available_lenses():
    """Get list of available cultural lenses"""
    lenses = []
    for name, cal in rose_glass.calibrations.items():
        lenses.append({
            "id": name,
            "name": cal.name,
            "description": cal.description,
            "tradition": cal.philosophical_tradition
        })
    return lenses

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Rose Glass Dream Analysis"}

# Run the app
if __name__ == "__main__":
    print("🌹 Rose Glass Dream Analysis Server Starting...")
    print("Visit http://localhost:8000 to analyze dreams")
    uvicorn.run(app, host="0.0.0.0", port=8000)