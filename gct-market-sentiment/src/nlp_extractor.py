"""
NLP module for extracting GCT variables from financial news
"""

import re
import spacy
import numpy as np
from typing import Dict, List, Tuple, Set
from collections import Counter
from dataclasses import dataclass
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer


@dataclass
class Article:
    """Financial news article"""
    id: str
    timestamp: str
    source: str
    title: str
    body: str
    tickers: List[str] = None


class GCTVariableExtractor:
    """Extract GCT variables (ψ, ρ, q, f) from financial text"""
    
    def __init__(self):
        # Load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            print("Downloading spaCy model...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")
            
        # Initialize NLTK sentiment analyzer
        try:
            self.sia = SentimentIntensityAnalyzer()
        except:
            nltk.download('vader_lexicon')
            self.sia = SentimentIntensityAnalyzer()
            
        # Financial domain keywords
        self.financial_entities = {
            'bullish_terms': {'surge', 'rally', 'boom', 'growth', 'gain', 'rise', 
                             'breakthrough', 'expansion', 'profit', 'revenue'},
            'bearish_terms': {'crash', 'plunge', 'decline', 'loss', 'fall', 'drop',
                             'recession', 'downturn', 'deficit', 'bankruptcy'},
            'uncertainty_terms': {'volatile', 'uncertain', 'risk', 'concern', 'worry',
                                'fluctuate', 'unstable', 'turbulent'}
        }
        
    def extract_psi(self, text: str) -> float:
        """
        Extract ψ (clarity/precision of narrative)
        Based on:
        - Sentence structure complexity
        - Use of specific vs vague language
        - Factual density
        """
        doc = self.nlp(text)
        
        # Metrics for clarity
        sentences = list(doc.sents)
        if not sentences:
            return 0.0
            
        # Average sentence length (moderate is clearer)
        avg_sent_length = np.mean([len(sent.text.split()) for sent in sentences])
        clarity_from_length = 1.0 / (1.0 + abs(avg_sent_length - 15) / 10)
        
        # Proportion of concrete entities (companies, numbers, dates)
        entities = [ent for ent in doc.ents if ent.label_ in ['ORG', 'MONEY', 'PERCENT', 'DATE']]
        entity_density = len(entities) / len(doc)
        
        # Use of hedging language (reduces clarity)
        hedge_words = {'maybe', 'perhaps', 'possibly', 'might', 'could', 'seems', 'appears'}
        hedge_count = sum(1 for token in doc if token.text.lower() in hedge_words)
        hedge_penalty = 1.0 / (1.0 + hedge_count / 10)
        
        # Combine metrics
        psi = (clarity_from_length + entity_density + hedge_penalty) / 3
        return min(max(psi, 0.0), 1.0)
        
    def extract_rho(self, text: str) -> float:
        """
        Extract ρ (reflective depth/nuance)
        Based on:
        - Causal reasoning markers
        - Conditional statements
        - Multiple perspective indicators
        """
        doc = self.nlp(text)
        
        # Causal markers
        causal_markers = {'because', 'therefore', 'consequently', 'thus', 'hence',
                         'as a result', 'due to', 'leads to', 'causes'}
        causal_count = sum(1 for token in doc if token.text.lower() in causal_markers)
        
        # Conditional/hypothetical thinking
        conditional_markers = {'if', 'when', 'unless', 'whereas', 'although', 'however',
                              'despite', 'nevertheless', 'on the other hand'}
        conditional_count = sum(1 for token in doc if token.text.lower() in conditional_markers)
        
        # Comparative analysis
        comparative_markers = {'compared to', 'relative to', 'versus', 'unlike',
                              'in contrast', 'similarly', 'likewise'}
        comparative_count = sum(1 for chunk in doc.noun_chunks 
                               if any(marker in chunk.text.lower() for marker in comparative_markers))
        
        # Normalize by text length
        text_length = len(doc)
        if text_length == 0:
            return 0.0
            
        rho = (causal_count + conditional_count * 1.5 + comparative_count * 2) / (text_length / 100)
        return min(max(rho, 0.0), 1.0)
        
    def extract_q_raw(self, text: str) -> float:
        """
        Extract q_raw (emotional charge)
        Based on:
        - Sentiment intensity
        - Emotional language
        - Market reaction terms
        """
        # VADER sentiment scores
        sentiment = self.sia.polarity_scores(text)
        
        # Emotional intensity from compound score
        emotional_intensity = abs(sentiment['compound'])
        
        # Check for strong financial emotional terms
        text_lower = text.lower()
        strong_positive = sum(1 for term in self.financial_entities['bullish_terms'] 
                             if term in text_lower)
        strong_negative = sum(1 for term in self.financial_entities['bearish_terms'] 
                             if term in text_lower)
        uncertainty = sum(1 for term in self.financial_entities['uncertainty_terms'] 
                         if term in text_lower)
        
        # Exclamation marks indicate emotional charge
        exclamation_count = text.count('!')
        
        # Combine metrics
        term_intensity = (strong_positive + strong_negative + uncertainty * 0.5) / 10
        q_raw = emotional_intensity * 0.6 + term_intensity * 0.3 + min(exclamation_count / 5, 0.1)
        
        return min(max(q_raw, 0.0), 1.0)
        
    def extract_f(self, text: str) -> float:
        """
        Extract f (social belonging signal)
        Based on:
        - Collective language ("we", "us", "our")
        - Market consensus references
        - Community/sector mentions
        """
        doc = self.nlp(text)
        text_lower = text.lower()
        
        # Collective pronouns
        collective_pronouns = {'we', 'us', 'our', 'everyone', 'investors', 'traders',
                              'market', 'sector', 'industry', 'community'}
        collective_count = sum(1 for token in doc if token.text.lower() in collective_pronouns)
        
        # Consensus language
        consensus_terms = {'consensus', 'agrees', 'widely', 'most', 'majority',
                          'trend', 'momentum', 'sentiment', 'outlook'}
        consensus_count = sum(1 for term in consensus_terms if term in text_lower)
        
        # Social proof indicators
        social_proof = {'analysts say', 'experts believe', 'reports indicate',
                       'sources confirm', 'data shows', 'survey finds'}
        social_count = sum(1 for phrase in social_proof if phrase in text_lower)
        
        # Normalize
        text_length = len(doc)
        if text_length == 0:
            return 0.0
            
        f = (collective_count + consensus_count * 1.5 + social_count * 2) / (text_length / 50)
        return min(max(f, 0.0), 1.0)
        
    def extract_tickers(self, text: str) -> List[str]:
        """
        Extract stock ticker symbols from text
        """
        # Common pattern: $TICKER or (TICKER)
        ticker_pattern = r'\$([A-Z]{1,5})\b|\(([A-Z]{1,5})\)'
        matches = re.findall(ticker_pattern, text)
        
        # Flatten and deduplicate
        tickers = []
        for match in matches:
            ticker = match[0] if match[0] else match[1]
            if ticker and 2 <= len(ticker) <= 5:
                tickers.append(ticker)
                
        # Also check for common company names to ticker mapping
        company_ticker_map = {
            'apple': 'AAPL', 'microsoft': 'MSFT', 'amazon': 'AMZN',
            'google': 'GOOGL', 'meta': 'META', 'tesla': 'TSLA',
            'nvidia': 'NVDA', 'netflix': 'NFLX', 'jpmorgan': 'JPM',
            'goldman': 'GS', 'berkshire': 'BRK', 'walmart': 'WMT'
        }
        
        text_lower = text.lower()
        for company, ticker in company_ticker_map.items():
            if company in text_lower:
                tickers.append(ticker)
                
        return list(set(tickers))
        
    def process_article(self, article: Article) -> Dict:
        """
        Process a financial article and extract all GCT variables
        """
        # Combine title and body for analysis
        full_text = f"{article.title} {article.body}"
        
        # Extract GCT variables
        psi = self.extract_psi(full_text)
        rho = self.extract_rho(full_text)
        q_raw = self.extract_q_raw(full_text)
        f = self.extract_f(full_text)
        
        # Extract tickers if not provided
        tickers = article.tickers or self.extract_tickers(full_text)
        
        return {
            'article_id': article.id,
            'timestamp': article.timestamp,
            'psi': psi,
            'rho': rho,
            'q_raw': q_raw,
            'f': f,
            'tickers': tickers,
            'source': article.source
        }
