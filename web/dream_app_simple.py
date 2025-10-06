"""
Rose Glass Dream Analysis Web Application (Simplified)
====================================================

A simplified version that works without ML dependencies.
"""

import os
from datetime import datetime
from typing import Dict, Optional, List
import json
import hashlib
import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

# Initialize app
app = FastAPI(
    title="Rose Glass Dream Analysis",
    description="Translate your dreams through cultural lenses",
    version="1.0.0"
)

# CORS for web access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request models
class DreamAnalysisRequest(BaseModel):
    dream_text: str
    selected_lens: Optional[str] = None

# HTML template (same as before)
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
                <label for="lensSelect">Choose a cultural lens:</label>
                <select id="lensSelect">
                    <option value="multi">Multi-lens View (See All)</option>
                    <option value="medieval_islamic">Medieval Islamic Philosophy</option>
                    <option value="indigenous_oral">Indigenous Oral Tradition</option>
                    <option value="buddhist_contemplative">Buddhist Contemplative</option>
                    <option value="digital_native">Digital Native</option>
                </select>
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
                        selected_lens: selectedLens
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
                            <p>${symbol.meaning}</p>
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
            
            resultsDiv.innerHTML = patternsHtml + interpretationHtml + symbolsHtml + insightsHtml;
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

# Simplified dream analyzer
def analyze_dream_simple(dream_text: str, lens: str = "multi") -> Dict:
    """Simple dream analysis without ML dependencies"""
    
    # Extract basic patterns from text
    text_lower = dream_text.lower()
    words = dream_text.split()
    
    # Pattern detection (simplified)
    psi = min(len([w for w in words if len(w) > 4]) / len(words), 1.0) if words else 0.5
    rho = len([w for w in ['water', 'flying', 'falling', 'death', 'light', 'dark', 'door', 'key'] if w in text_lower]) * 0.15
    q = len([w for w in ['afraid', 'happy', 'sad', 'angry', 'love', 'fear', 'joy'] if w in text_lower]) * 0.1
    f = len([w for w in ['we', 'us', 'they', 'family', 'friend', 'people'] if w in text_lower]) * 0.05
    
    # Normalize
    rho = min(rho, 1.0)
    q = min(q, 1.0)
    f = min(f, 1.0)
    
    # Extract symbols
    universal_symbols = {
        'water': "Emotional life, unconscious depths, purification",
        'flying': "Freedom, transcendence, rising above limitations",
        'falling': "Loss of control, surrender, letting go", 
        'death': "Transformation, ending of one phase, rebirth",
        'light': "Consciousness, clarity, spiritual awakening",
        'dark': "Unknown aspects, mystery, the unconscious",
        'door': "Transition, opportunity, passage between states",
        'key': "Solution, access to hidden knowledge, power"
    }
    
    found_symbols = []
    for symbol, meaning in universal_symbols.items():
        if symbol in text_lower:
            found_symbols.append({'symbol': symbol.title(), 'meaning': meaning})
    
    # Generate insights based on patterns
    insights = []
    if psi > 0.7:
        insights.append("Your dream shows clear narrative structure, suggesting active integration of experiences")
    if rho > 0.5:
        insights.append("Rich symbolic content indicates deep psychological processing")
    if q > 0.6:
        insights.append("High emotional content - this dream touches important feelings")
    if f > 0.5:
        insights.append("Strong relational themes suggest processing of social connections")
    
    # Lens-specific interpretations
    interpretations = {
        'medieval_islamic': f"Through the Medieval Islamic lens: This dream shows {'high' if rho > 0.5 else 'moderate'} wisdom content with {'restrained' if q < 0.4 else 'significant'} emotional expression. In the tradition of Ibn Rushd, dreams are seen as the soul's journey through symbolic realms.",
        'indigenous_oral': f"Through the Indigenous lens: Your dream speaks of {'collective' if f > 0.5 else 'individual'} journey with {'many' if len(found_symbols) > 2 else 'few'} spirit guides. The {'circular' if 'return' in text_lower or 'again' in text_lower else 'linear'} pattern suggests {'ongoing' if rho > 0.5 else 'new'} teachings.",
        'buddhist_contemplative': f"Through the Buddhist lens: The dream reveals {'attachment' if q > 0.6 else 'detachment'} patterns and {'clear' if psi > 0.7 else 'clouded'} awareness. {'Transformation' if any(s in text_lower for s in ['change', 'became', 'transform']) else 'Stability'} themes point to the impermanent nature of all phenomena.",
        'digital_native': f"Through the Digital Native lens: This dream has {'high' if q > 0.5 and f > 0.5 else 'low'} virality potential with {'strong' if q > 0.6 else 'moderate'} emotional hooks. The narrative {'flows' if psi > 0.6 else 'fragments'} like {'a thread' if psi > 0.7 else 'scattered posts'}."
    }
    
    # Select interpretation
    if lens == "multi":
        interpretation = "Multi-lens view:\n\n" + "\n\n".join([f"• {interp}" for interp in interpretations.values()])
    else:
        interpretation = interpretations.get(lens, "Please select a valid lens")
    
    return {
        'patterns': {
            'psi': psi,
            'rho': rho,
            'q': q,
            'f': f
        },
        'pattern_descriptions': {
            'psi': 'highly coherent narrative' if psi > 0.7 else 'dream logic flow' if psi > 0.4 else 'fragmented impressions',
            'rho': 'rich archetypal content' if rho > 0.6 else 'moderate symbolic presence' if rho > 0.3 else 'literal imagery',
            'q': 'intense emotional content' if q > 0.6 else 'moderate emotional tone' if q > 0.3 else 'emotionally neutral',
            'f': 'highly relational' if f > 0.6 else 'balanced self/other' if f > 0.3 else 'solitary journey'
        },
        'lens_used': lens.replace('_', ' ').title(),
        'interpretation': interpretation,
        'symbols': found_symbols[:5],  # Limit to 5 symbols
        'insights': insights
    }

@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the main dream analysis interface"""
    return HTML_TEMPLATE

@app.post("/api/analyze_dream")
async def analyze_dream(request: DreamAnalysisRequest):
    """Analyze a dream using simplified Rose Glass approach"""
    try:
        result = analyze_dream_simple(request.dream_text, request.selected_lens or "multi")
        return JSONResponse(content=result)
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Failed to analyze dream")

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Rose Glass Dream Analysis (Simple)"}

if __name__ == "__main__":
    port = 8002
    print("🌹 Rose Glass Dream Analysis Server Starting...")
    print(f"Visit http://localhost:{port} to analyze dreams")
    uvicorn.run(app, host="0.0.0.0", port=port)