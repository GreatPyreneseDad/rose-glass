"""
GCT-Style Dream Analysis Web Application
=======================================

Uses Rose Glass internally to help LLMs understand dreams,
but presents results in human-friendly GCT format.
"""

import os
import asyncio
from datetime import datetime
from typing import Dict, Optional, List
import json
import re
from collections import defaultdict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import openai
from anthropic import Anthropic
import uvicorn
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Initialize app
app = FastAPI(
    title="GCT Dream Analysis",
    description="Deep dream analysis using Grounded Coherence Theory",
    version="2.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class DreamAnalysisRequest(BaseModel):
    dream_text: str
    llm_provider: str = "claude"
    include_symbols: bool = True
    include_coherence: bool = True

# HTML template with GCT styling
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GCT Dream Analysis</title>
    <style>
        :root {
            --primary: #6366f1;
            --secondary: #8b5cf6;
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
            --background: #0f0f23;
            --surface: #1a1a2e;
            --text: #eee;
            --accent: #00d4ff;
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Monaco', 'Consolas', monospace;
            background: var(--background);
            color: var(--text);
            min-height: 100vh;
            background-image: 
                radial-gradient(circle at 25% 25%, #2a2a3e 0%, transparent 50%),
                radial-gradient(circle at 75% 75%, #1a1a2e 0%, transparent 50%);
        }
        
        .header {
            background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
            padding: 2rem;
            text-align: center;
            box-shadow: 0 4px 20px rgba(99, 102, 241, 0.3);
        }
        
        .header h1 {
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
            letter-spacing: 2px;
        }
        
        .header p {
            font-size: 1.1rem;
            opacity: 0.95;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
        }
        
        .dream-input {
            background: var(--surface);
            padding: 2rem;
            border-radius: 12px;
            margin-bottom: 2rem;
            border: 1px solid rgba(99, 102, 241, 0.2);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }
        
        .dream-input textarea {
            width: 100%;
            min-height: 250px;
            padding: 1rem;
            background: rgba(15, 15, 35, 0.8);
            border: 1px solid rgba(99, 102, 241, 0.3);
            border-radius: 8px;
            color: var(--text);
            font-size: 1rem;
            resize: vertical;
            font-family: inherit;
            line-height: 1.6;
        }
        
        .dream-input textarea:focus {
            outline: none;
            border-color: var(--accent);
            box-shadow: 0 0 20px rgba(0, 212, 255, 0.2);
        }
        
        .options {
            margin: 1.5rem 0;
            display: flex;
            gap: 2rem;
            align-items: center;
            flex-wrap: wrap;
        }
        
        .options label {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            cursor: pointer;
            font-size: 0.95rem;
        }
        
        .analyze-button {
            background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
            color: white;
            border: none;
            padding: 1rem 2.5rem;
            border-radius: 8px;
            font-size: 1.1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 1px;
            box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);
        }
        
        .analyze-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(99, 102, 241, 0.6);
        }
        
        .analyze-button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }
        
        .results {
            display: none;
        }
        
        .results.show {
            display: block;
        }
        
        .coherence-panel {
            background: var(--surface);
            padding: 2rem;
            border-radius: 12px;
            margin-bottom: 2rem;
            border: 1px solid rgba(99, 102, 241, 0.2);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }
        
        .coherence-panel h2 {
            color: var(--accent);
            margin-bottom: 1.5rem;
            font-size: 1.8rem;
        }
        
        .coherence-metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }
        
        .metric-card {
            background: rgba(15, 15, 35, 0.6);
            padding: 1.5rem;
            border-radius: 8px;
            border: 1px solid rgba(0, 212, 255, 0.2);
            transition: all 0.3s ease;
        }
        
        .metric-card:hover {
            transform: translateY(-2px);
            border-color: var(--accent);
            box-shadow: 0 4px 20px rgba(0, 212, 255, 0.2);
        }
        
        .metric-card h3 {
            font-size: 0.9rem;
            color: var(--accent);
            margin-bottom: 0.5rem;
            opacity: 0.8;
        }
        
        .metric-value {
            font-size: 2.5rem;
            font-weight: bold;
            background: linear-gradient(135deg, var(--primary), var(--accent));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }
        
        .metric-desc {
            font-size: 0.85rem;
            opacity: 0.7;
            line-height: 1.4;
        }
        
        .coherence-score {
            text-align: center;
            margin: 2rem 0;
            padding: 2rem;
            background: rgba(15, 15, 35, 0.8);
            border-radius: 12px;
            border: 2px solid var(--accent);
        }
        
        .coherence-score h3 {
            color: var(--accent);
            margin-bottom: 1rem;
            font-size: 1.3rem;
        }
        
        .score-display {
            font-size: 4rem;
            font-weight: bold;
            background: linear-gradient(135deg, var(--primary), var(--accent));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-shadow: 0 0 40px rgba(0, 212, 255, 0.5);
        }
        
        .score-interpretation {
            margin-top: 1rem;
            font-size: 1.1rem;
            color: var(--warning);
        }
        
        .analysis-section {
            background: var(--surface);
            padding: 2rem;
            border-radius: 12px;
            margin-bottom: 2rem;
            border: 1px solid rgba(139, 92, 246, 0.2);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }
        
        .analysis-section h2 {
            color: var(--secondary);
            margin-bottom: 1.5rem;
            font-size: 1.8rem;
        }
        
        .dream-interpretation {
            background: rgba(15, 15, 35, 0.6);
            padding: 2rem;
            border-radius: 8px;
            line-height: 1.8;
            font-size: 1.05rem;
            border-left: 4px solid var(--secondary);
        }
        
        .symbols-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1.5rem;
            margin-top: 1.5rem;
        }
        
        .symbol-card {
            background: rgba(15, 15, 35, 0.6);
            padding: 1.5rem;
            border-radius: 8px;
            border: 1px solid rgba(139, 92, 246, 0.3);
            transition: all 0.3s ease;
        }
        
        .symbol-card:hover {
            transform: scale(1.02);
            border-color: var(--secondary);
            box-shadow: 0 4px 20px rgba(139, 92, 246, 0.3);
        }
        
        .symbol-card h4 {
            color: var(--warning);
            margin-bottom: 0.5rem;
            font-size: 1.2rem;
        }
        
        .psychological-themes {
            margin-top: 2rem;
        }
        
        .theme-bar {
            margin-bottom: 1rem;
        }
        
        .theme-bar h4 {
            color: var(--text);
            margin-bottom: 0.5rem;
            font-size: 0.95rem;
        }
        
        .theme-progress {
            background: rgba(15, 15, 35, 0.8);
            border-radius: 8px;
            overflow: hidden;
            height: 24px;
            box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.3);
        }
        
        .theme-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--primary), var(--secondary));
            transition: width 1s ease;
            display: flex;
            align-items: center;
            padding-left: 10px;
            color: white;
            font-size: 0.85rem;
            font-weight: 600;
        }
        
        .recommendations {
            background: rgba(16, 185, 129, 0.1);
            border: 1px solid var(--success);
            padding: 1.5rem;
            border-radius: 8px;
            margin-top: 2rem;
        }
        
        .recommendations h3 {
            color: var(--success);
            margin-bottom: 1rem;
        }
        
        .recommendations ul {
            list-style: none;
        }
        
        .recommendations li {
            margin: 0.8rem 0;
            padding-left: 1.5rem;
            position: relative;
        }
        
        .recommendations li::before {
            content: "→";
            position: absolute;
            left: 0;
            color: var(--success);
            font-weight: bold;
        }
        
        .loading {
            display: none;
            text-align: center;
            padding: 4rem 2rem;
        }
        
        .loading.show {
            display: block;
        }
        
        .spinner {
            border: 3px solid rgba(99, 102, 241, 0.2);
            border-top: 3px solid var(--primary);
            border-radius: 50%;
            width: 60px;
            height: 60px;
            animation: spin 1s linear infinite;
            margin: 0 auto 2rem;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .loading-text {
            font-size: 1.2rem;
            color: var(--primary);
            animation: pulse 1.5s ease-in-out infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 0.8; }
            50% { opacity: 1; }
        }
        
        .error {
            background: rgba(239, 68, 68, 0.1);
            border: 1px solid var(--danger);
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
            display: none;
        }
        
        .error.show {
            display: block;
        }
        
        .privacy-note {
            text-align: center;
            margin-top: 3rem;
            padding: 1rem;
            opacity: 0.7;
            font-size: 0.9rem;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>GCT Dream Analysis</h1>
        <p>Deep psychological insights through Grounded Coherence Theory</p>
    </div>
    
    <div class="container">
        <div class="dream-input">
            <textarea 
                id="dreamText" 
                placeholder="Describe your dream in detail. Include emotions, symbols, people, settings, and any transformations or significant moments. The more detail you provide, the deeper the analysis..."
            ></textarea>
            
            <div class="options">
                <label>
                    <input type="checkbox" id="includeSymbols" checked>
                    Analyze symbols & archetypes
                </label>
                
                <label>
                    <input type="checkbox" id="includeCoherence" checked>
                    Calculate coherence patterns
                </label>
                
                <label>
                    <input type="radio" name="llmProvider" value="claude" checked>
                    Claude (recommended)
                </label>
                
                <label>
                    <input type="radio" name="llmProvider" value="openai">
                    GPT-4
                </label>
            </div>
            
            <button class="analyze-button" onclick="analyzeDream()">
                Analyze Dream
            </button>
        </div>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <div class="loading-text">Analyzing your dream patterns...</div>
        </div>
        
        <div class="error" id="error"></div>
        
        <div class="results" id="results"></div>
        
        <div class="privacy-note">
            🔐 Your privacy is sacred. Dreams are analyzed in real-time and never stored.
        </div>
    </div>
    
    <script>
        async function analyzeDream() {
            const dreamText = document.getElementById('dreamText').value.trim();
            if (!dreamText) {
                showError('Please describe your dream first');
                return;
            }
            
            const includeSymbols = document.getElementById('includeSymbols').checked;
            const includeCoherence = document.getElementById('includeCoherence').checked;
            const llmProvider = document.querySelector('input[name="llmProvider"]:checked').value;
            
            // UI feedback
            document.querySelector('.analyze-button').disabled = true;
            document.getElementById('loading').classList.add('show');
            document.getElementById('results').classList.remove('show');
            document.getElementById('error').classList.remove('show');
            
            try {
                const response = await fetch('/api/analyze_dream_gct', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        dream_text: dreamText,
                        llm_provider: llmProvider,
                        include_symbols: includeSymbols,
                        include_coherence: includeCoherence
                    })
                });
                
                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(error.detail || 'Analysis failed');
                }
                
                const data = await response.json();
                displayResults(data);
                
            } catch (error) {
                showError('Unable to analyze dream: ' + error.message);
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
            
            // Coherence panel
            if (data.coherence_analysis) {
                const coherenceHtml = `
                    <div class="coherence-panel">
                        <h2>Coherence Analysis</h2>
                        <div class="coherence-metrics">
                            <div class="metric-card">
                                <h3>Ψ - Internal Consistency</h3>
                                <div class="metric-value">${(data.coherence_analysis.psi * 100).toFixed(0)}%</div>
                                <div class="metric-desc">${data.coherence_analysis.psi_description}</div>
                            </div>
                            <div class="metric-card">
                                <h3>ρ - Accumulated Wisdom</h3>
                                <div class="metric-value">${(data.coherence_analysis.rho * 100).toFixed(0)}%</div>
                                <div class="metric-desc">${data.coherence_analysis.rho_description}</div>
                            </div>
                            <div class="metric-card">
                                <h3>q - Moral/Emotional Energy</h3>
                                <div class="metric-value">${(data.coherence_analysis.q * 100).toFixed(0)}%</div>
                                <div class="metric-desc">${data.coherence_analysis.q_description}</div>
                            </div>
                            <div class="metric-card">
                                <h3>f - Social Architecture</h3>
                                <div class="metric-value">${(data.coherence_analysis.f * 100).toFixed(0)}%</div>
                                <div class="metric-desc">${data.coherence_analysis.f_description}</div>
                            </div>
                        </div>
                        
                        <div class="coherence-score">
                            <h3>Overall Coherence Score</h3>
                            <div class="score-display">${data.coherence_analysis.overall_coherence.toFixed(2)}</div>
                            <div class="score-interpretation">${data.coherence_analysis.coherence_state}</div>
                        </div>
                    </div>
                `;
                resultsDiv.innerHTML += coherenceHtml;
            }
            
            // Main interpretation
            const interpretationHtml = `
                <div class="analysis-section">
                    <h2>Dream Interpretation</h2>
                    <div class="dream-interpretation">
                        ${data.interpretation}
                    </div>
                </div>
            `;
            resultsDiv.innerHTML += interpretationHtml;
            
            // Symbols and archetypes
            if (data.symbols && data.symbols.length > 0) {
                let symbolsHtml = `
                    <div class="analysis-section">
                        <h2>Symbols & Archetypes</h2>
                        <div class="symbols-grid">
                `;
                
                data.symbols.forEach(symbol => {
                    symbolsHtml += `
                        <div class="symbol-card">
                            <h4>${symbol.symbol}</h4>
                            <p>${symbol.interpretation}</p>
                            ${symbol.archetype ? `<p><em>Archetype: ${symbol.archetype}</em></p>` : ''}
                        </div>
                    `;
                });
                
                symbolsHtml += '</div></div>';
                resultsDiv.innerHTML += symbolsHtml;
            }
            
            // Psychological themes
            if (data.psychological_themes) {
                let themesHtml = `
                    <div class="analysis-section">
                        <h2>Psychological Themes</h2>
                        <div class="psychological-themes">
                `;
                
                Object.entries(data.psychological_themes).forEach(([theme, value]) => {
                    const percentage = (value * 100).toFixed(0);
                    themesHtml += `
                        <div class="theme-bar">
                            <h4>${theme.replace(/_/g, ' ').toUpperCase()}</h4>
                            <div class="theme-progress">
                                <div class="theme-fill" style="width: ${percentage}%">${percentage}%</div>
                            </div>
                        </div>
                    `;
                });
                
                themesHtml += '</div></div>';
                resultsDiv.innerHTML += themesHtml;
            }
            
            // Recommendations
            if (data.recommendations && data.recommendations.length > 0) {
                let recsHtml = `
                    <div class="analysis-section">
                        <h2>Insights & Recommendations</h2>
                        <div class="recommendations">
                            <h3>For deeper understanding:</h3>
                            <ul>
                `;
                
                data.recommendations.forEach(rec => {
                    recsHtml += `<li>${rec}</li>`;
                });
                
                recsHtml += '</ul></div></div>';
                resultsDiv.innerHTML += recsHtml;
            }
            
            resultsDiv.classList.add('show');
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

def extract_gct_patterns(dream_text: str) -> Dict:
    """Extract GCT patterns from dream text"""
    text_lower = dream_text.lower()
    sentences = dream_text.split('.')
    words = dream_text.split()
    
    # Ψ (Psi) - Internal Consistency
    # Look for narrative coherence, logical flow
    connectors = ['then', 'next', 'because', 'so', 'therefore', 'thus', 'after']
    connector_count = sum(1 for word in connectors if word in text_lower)
    scene_changes = text_lower.count('suddenly') + text_lower.count('then i')
    
    psi = min(0.2 + (connector_count * 0.1) + (len(sentences) / 20) - (scene_changes * 0.05), 1.0)
    psi = max(psi, 0.1)
    
    # ρ (Rho) - Accumulated Wisdom
    # Look for symbols, archetypes, deep meanings
    wisdom_words = ['realized', 'understood', 'wisdom', 'truth', 'meaning', 'symbol', 
                    'represented', 'ancient', 'knowledge', 'learned', 'insight']
    archetype_words = ['mother', 'father', 'child', 'hero', 'shadow', 'wise', 'elder',
                      'death', 'birth', 'transformation', 'journey', 'guide']
    
    wisdom_score = sum(1 for word in wisdom_words if word in text_lower) * 0.15
    archetype_score = sum(1 for word in archetype_words if word in text_lower) * 0.1
    
    rho = min(wisdom_score + archetype_score + 0.3, 1.0)
    
    # q - Moral/Emotional Activation
    emotion_words = {
        'fear': ['afraid', 'scared', 'terrified', 'anxious', 'panic', 'horror'],
        'joy': ['happy', 'joy', 'elated', 'excited', 'wonderful', 'bliss'],
        'anger': ['angry', 'rage', 'furious', 'mad', 'annoyed', 'frustrated'],
        'sadness': ['sad', 'depressed', 'grief', 'sorrow', 'lonely', 'melancholy'],
        'love': ['love', 'affection', 'care', 'tender', 'warm', 'compassion']
    }
    
    emotion_count = 0
    for emotion_type, words_list in emotion_words.items():
        emotion_count += sum(1 for word in words_list if word in text_lower)
    
    moral_words = ['should', 'must', 'right', 'wrong', 'good', 'evil', 'moral', 'ethical']
    moral_count = sum(1 for word in moral_words if word in text_lower)
    
    q = min((emotion_count * 0.08) + (moral_count * 0.1) + 0.2, 1.0)
    
    # f - Social Architecture
    social_words = ['family', 'friend', 'mother', 'father', 'brother', 'sister',
                   'people', 'crowd', 'group', 'together', 'alone', 'community',
                   'we', 'us', 'our', 'they', 'them']
    
    social_count = sum(1 for word in social_words if word in text_lower)
    f = min((social_count * 0.05) + 0.2, 1.0)
    
    return {
        'psi': psi,
        'rho': rho,
        'q': q,
        'f': f
    }

def extract_symbols(dream_text: str) -> List[Dict]:
    """Extract symbols and their meanings"""
    text_lower = dream_text.lower()
    
    universal_symbols = {
        'water': "Emotions, unconscious mind, purification, life force",
        'flying': "Freedom, transcendence, escaping limitations, spiritual elevation",
        'falling': "Loss of control, fear of failure, surrender, letting go",
        'death': "Transformation, end of a phase, rebirth, major change",
        'birth': "New beginnings, creative potential, fresh start",
        'fire': "Passion, destruction, transformation, purification",
        'mountain': "Challenges, spiritual journey, achievement, perspective",
        'ocean': "Vast unconscious, emotions, the unknown, collective unconscious",
        'snake': "Transformation, healing, hidden fears, kundalini energy",
        'house': "The self, psyche, different aspects of personality",
        'door': "Opportunities, transitions, choices, passages",
        'key': "Solutions, access to hidden knowledge, power, understanding",
        'mirror': "Self-reflection, truth, illusion, how others see you",
        'bridge': "Transition, connection, overcoming obstacles",
        'child': "Inner child, innocence, new beginnings, vulnerability",
        'animal': "Instincts, primal nature, specific qualities of that animal",
        'light': "Consciousness, understanding, hope, spiritual awakening",
        'darkness': "Unknown, unconscious, fear, hidden aspects",
        'tree': "Growth, life, connection between earth and sky, family",
        'road': "Life path, journey, choices, direction"
    }
    
    jungian_archetypes = {
        'shadow': ['dark figure', 'enemy', 'villain', 'monster', 'evil'],
        'anima/animus': ['mysterious woman', 'mysterious man', 'opposite sex'],
        'wise old man/woman': ['elder', 'teacher', 'guide', 'sage', 'mentor'],
        'great mother': ['mother', 'nurturing', 'protective woman', 'goddess'],
        'hero': ['hero', 'warrior', 'savior', 'fighter'],
        'trickster': ['joker', 'fool', 'clown', 'shapeshifter']
    }
    
    found_symbols = []
    
    # Check for universal symbols
    for symbol, meaning in universal_symbols.items():
        if symbol in text_lower:
            found_symbols.append({
                'symbol': symbol.title(),
                'interpretation': meaning,
                'archetype': None
            })
    
    # Check for Jungian archetypes
    for archetype, indicators in jungian_archetypes.items():
        for indicator in indicators:
            if indicator in text_lower:
                found_symbols.append({
                    'symbol': indicator.title(),
                    'interpretation': f"Represents the {archetype} archetype",
                    'archetype': archetype.title()
                })
                break
    
    return found_symbols[:8]  # Limit to 8 most relevant symbols

async def get_llm_dream_analysis(dream_text: str, patterns: Dict, symbols: List[Dict], 
                                provider: str = "claude") -> Dict:
    """Get deep dream analysis from LLM using Rose Glass patterns internally"""
    
    # Create context with GCT patterns to help LLM understand
    context = f"""You are an expert dream analyst trained in Grounded Coherence Theory (GCT) and Jungian psychology. 
    
I've analyzed this dream using GCT and found these patterns:
- Ψ (Internal Consistency): {patterns['psi']:.2f} - The dream's narrative coherence
- ρ (Accumulated Wisdom): {patterns['rho']:.2f} - Symbolic and archetypal depth  
- q (Moral/Emotional Energy): {patterns['q']:.2f} - Emotional intensity and moral activation
- f (Social Architecture): {patterns['f']:.2f} - Relational and collective themes

Key symbols detected: {', '.join([s['symbol'] for s in symbols])}

The dreamer wrote:
"{dream_text}"

Provide a deep, personalized dream interpretation that:
1. Explains what psychological work the dream is doing
2. Identifies the main themes and their significance
3. Discusses the symbols in the context of the dreamer's psyche
4. Suggests what the unconscious might be processing
5. Offers insights for personal growth

Be specific to THIS dream, not generic. Reference the actual content and patterns.
Write in a warm, insightful tone that respects the dreamer's experience.
About 300-400 words.
"""
    
    try:
        if provider == "claude" and ANTHROPIC_API_KEY:
            client = Anthropic(api_key=ANTHROPIC_API_KEY)
            response = client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=800,
                temperature=0.7,
                messages=[
                    {"role": "user", "content": context}
                ]
            )
            return {"interpretation": response.content[0].text}
            
        elif provider == "openai" and OPENAI_API_KEY:
            openai.api_key = OPENAI_API_KEY
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a compassionate dream analyst expert in GCT and Jungian psychology."},
                    {"role": "user", "content": context}
                ],
                max_tokens=800,
                temperature=0.7
            )
            return {"interpretation": response.choices[0].message.content}
            
        else:
            # Fallback interpretation
            return {
                "interpretation": f"""Based on the GCT analysis of your dream:

Your dream shows {'strong' if patterns['psi'] > 0.7 else 'moderate' if patterns['psi'] > 0.4 else 'fragmented'} narrative coherence (Ψ={patterns['psi']:.2f}), suggesting {'clear psychological processing' if patterns['psi'] > 0.7 else 'complex inner work' if patterns['psi'] > 0.4 else 'deep unconscious processing'}.

The {'high' if patterns['rho'] > 0.7 else 'moderate' if patterns['rho'] > 0.4 else 'emerging'} wisdom content (ρ={patterns['rho']:.2f}) indicates {'rich symbolic material and archetypal themes' if patterns['rho'] > 0.7 else 'developing insight and understanding' if patterns['rho'] > 0.4 else 'new psychological territory being explored'}.

Your emotional activation (q={patterns['q']:.2f}) is {'intense' if patterns['q'] > 0.7 else 'moderate' if patterns['q'] > 0.4 else 'contained'}, showing {'significant emotional processing' if patterns['q'] > 0.7 else 'balanced emotional engagement' if patterns['q'] > 0.4 else 'intellectual or observational stance'}.

The social dimension (f={patterns['f']:.2f}) reveals {'strong relational themes' if patterns['f'] > 0.7 else 'balanced individual/collective elements' if patterns['f'] > 0.4 else 'primarily individual journey'}.

Key symbols like {', '.join([s['symbol'] for s in symbols[:3]])} suggest themes of transformation, self-discovery, and integration of unconscious material.

Consider journaling about these symbols and what they mean to you personally."""
            }
            
    except Exception as e:
        print(f"LLM error: {e}")
        return {
            "interpretation": "Dream analysis requires API access. The patterns show interesting psychological dynamics worth exploring through journaling or therapy."
        }

def detect_psychological_themes(dream_text: str) -> Dict[str, float]:
    """Detect major psychological themes in the dream"""
    themes = {
        'individuation': 0,
        'shadow_integration': 0,
        'relationship_dynamics': 0,
        'creative_expression': 0,
        'spiritual_awakening': 0,
        'trauma_processing': 0,
        'power_dynamics': 0,
        'identity_formation': 0
    }
    
    text_lower = dream_text.lower()
    
    # Theme detection logic
    theme_indicators = {
        'individuation': ['self', 'becoming', 'whole', 'journey', 'path', 'discover', 'true'],
        'shadow_integration': ['dark', 'shadow', 'hidden', 'enemy', 'opposite', 'integrate'],
        'relationship_dynamics': ['mother', 'father', 'family', 'friend', 'love', 'together'],
        'creative_expression': ['create', 'art', 'music', 'dance', 'build', 'color', 'design'],
        'spiritual_awakening': ['light', 'divine', 'spiritual', 'transcend', 'awaken', 'enlighten'],
        'trauma_processing': ['fear', 'hurt', 'pain', 'escape', 'trapped', 'past', 'memory'],
        'power_dynamics': ['control', 'power', 'authority', 'dominate', 'submit', 'free'],
        'identity_formation': ['who am i', 'identity', 'mask', 'role', 'authentic', 'pretend']
    }
    
    for theme, keywords in theme_indicators.items():
        score = sum(1 for keyword in keywords if keyword in text_lower)
        themes[theme] = min(score * 0.2, 1.0)
    
    # Normalize
    total = sum(themes.values())
    if total > 0:
        themes = {k: v/total for k, v in themes.items()}
    
    # Return only significant themes
    return {k: v for k, v in themes.items() if v > 0.1}

def generate_recommendations(patterns: Dict, themes: Dict) -> List[str]:
    """Generate personalized recommendations based on analysis"""
    recommendations = []
    
    # Based on coherence patterns
    if patterns['psi'] < 0.4:
        recommendations.append("Keep a more detailed dream journal to improve dream recall and narrative clarity")
    if patterns['rho'] > 0.7:
        recommendations.append("Your dreams contain rich symbolic content - consider exploring these symbols through active imagination or art")
    if patterns['q'] > 0.7:
        recommendations.append("High emotional content suggests important processing - ensure you have emotional support")
    if patterns['f'] < 0.3:
        recommendations.append("Dreams focus on individual journey - this may be a time for self-reflection")
    
    # Based on themes
    if themes.get('shadow_integration', 0) > 0.2:
        recommendations.append("Shadow work appears prominent - consider what aspects of yourself you may be rejecting")
    if themes.get('spiritual_awakening', 0) > 0.2:
        recommendations.append("Spiritual themes are emerging - meditation or contemplative practices may be helpful")
    if themes.get('trauma_processing', 0) > 0.2:
        recommendations.append("If trauma themes persist, consider working with a qualified therapist")
    
    # General recommendations
    recommendations.extend([
        "Draw or paint significant dream images to deepen understanding",
        "Before sleep, set an intention to receive clarity on dream messages",
        "Share dreams with trusted friends or a dream group for new perspectives"
    ])
    
    return recommendations[:5]

@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the main interface"""
    return HTML_TEMPLATE

@app.post("/api/analyze_dream_gct")
async def analyze_dream_gct(request: DreamAnalysisRequest):
    """Analyze dream using GCT framework with LLM enhancement"""
    try:
        # Extract GCT patterns
        patterns = extract_gct_patterns(request.dream_text)
        
        # Extract symbols
        symbols = extract_symbols(request.dream_text) if request.include_symbols else []
        
        # Get LLM interpretation
        llm_result = await get_llm_dream_analysis(
            request.dream_text,
            patterns,
            symbols,
            request.llm_provider
        )
        
        # Prepare response
        response_data = {
            "interpretation": llm_result["interpretation"]
        }
        
        # Add coherence analysis if requested
        if request.include_coherence:
            # Calculate overall coherence using GCT formula
            psi, rho, q, f = patterns['psi'], patterns['rho'], patterns['q'], patterns['f']
            
            # Biological optimization of q
            km, ki = 0.2, 0.8
            q_opt = q / (km + q + (q**2 / ki))
            
            # Calculate coherence
            coherence = psi + (rho * psi) + q_opt + (f * psi) + (0.15 * rho * q_opt)
            coherence = min(coherence, 4.0)
            
            # Determine state
            if coherence < 0.5:
                state = "Fragmented - Deep unconscious processing"
            elif coherence < 1.5:
                state = "Emerging - Patterns forming" 
            elif coherence < 2.5:
                state = "Integrated - Clear psychological work"
            else:
                state = "Highly Coherent - Deep integration occurring"
            
            response_data["coherence_analysis"] = {
                "psi": psi,
                "psi_description": "High narrative coherence" if psi > 0.7 else "Moderate flow" if psi > 0.4 else "Dream logic",
                "rho": rho,
                "rho_description": "Rich symbolic content" if rho > 0.7 else "Moderate symbolism" if rho > 0.4 else "Emerging symbols",
                "q": q,
                "q_description": "Intense emotions" if q > 0.7 else "Moderate feelings" if q > 0.4 else "Calm observation",
                "f": f,
                "f_description": "Strong social themes" if f > 0.7 else "Balanced self/other" if f > 0.4 else "Individual focus",
                "overall_coherence": coherence,
                "coherence_state": state
            }
        
        # Add symbols if requested
        if request.include_symbols and symbols:
            response_data["symbols"] = symbols
        
        # Detect psychological themes
        themes = detect_psychological_themes(request.dream_text)
        if themes:
            response_data["psychological_themes"] = themes
        
        # Generate recommendations
        response_data["recommendations"] = generate_recommendations(patterns, themes)
        
        return JSONResponse(content=response_data)
        
    except Exception as e:
        print(f"Error analyzing dream: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = 8004
    print("🧠 GCT Dream Analysis Server Starting...")
    print(f"Visit http://localhost:{port} to analyze dreams")
    uvicorn.run(app, host="0.0.0.0", port=port)