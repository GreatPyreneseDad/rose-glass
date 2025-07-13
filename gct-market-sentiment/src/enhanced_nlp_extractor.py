#!/usr/bin/env python3
"""
Enhanced NLP Extractor with Advanced Language Processing
Implements review recommendations for improved variable extraction
"""

import spacy
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import re
from textstat import flesch_reading_ease, gunning_fog
from collections import Counter
import networkx as nx


@dataclass
class ArgumentStructure:
    """Represents the argumentative structure of text"""
    claims: List[str]
    evidence: List[str]
    relations: List[Tuple[str, str, str]]  # (claim, evidence, relation_type)
    complexity_score: float


@dataclass
class DiscourseRelations:
    """Represents discourse relations in text"""
    causal_relations: int
    contrastive_relations: int
    elaborative_relations: int
    temporal_relations: int
    coherence_score: float


class EnhancedNLPExtractor:
    """Advanced NLP extractor implementing review recommendations"""
    
    def __init__(self, model_name: str = "en_core_web_lg"):
        self.nlp = spacy.load(model_name)
        
        # Add custom pipeline components
        self.nlp.add_pipe("sentencizer")
        
        # Discourse markers
        self.discourse_markers = {
            'causal': ['because', 'therefore', 'thus', 'hence', 'consequently', 
                      'as a result', 'due to', 'owing to'],
            'contrastive': ['however', 'but', 'although', 'whereas', 'while', 
                           'on the other hand', 'in contrast', 'nevertheless'],
            'elaborative': ['for example', 'specifically', 'in particular', 
                           'namely', 'that is', 'in other words'],
            'temporal': ['then', 'next', 'after', 'before', 'subsequently', 
                        'meanwhile', 'finally', 'initially']
        }
        
        # Epistemic markers for certainty/hedging
        self.epistemic_markers = {
            'certainty': ['definitely', 'certainly', 'clearly', 'obviously', 
                         'undoubtedly', 'surely', 'absolutely'],
            'hedging': ['perhaps', 'maybe', 'possibly', 'might', 'could', 
                       'seems', 'appears', 'suggests', 'likely']
        }
        
    def extract_psi_enhanced(self, text: str) -> float:
        """
        Enhanced clarity extraction with syntactic complexity
        ψ (psi) - Moral clarity and coherent expression
        """
        doc = self.nlp(text)
        
        # Syntactic complexity analysis
        syntax_score = self._calculate_syntactic_complexity(doc)
        
        # Argument structure analysis
        arg_structure = self._analyze_argument_structure(doc)
        
        # Discourse coherence
        discourse_score = self._analyze_discourse_coherence(doc)
        
        # Readability metrics
        readability = self._calculate_readability(text)
        
        # Integrate signals
        psi = self._integrate_psi_signals(
            syntax_score, 
            arg_structure.complexity_score,
            discourse_score,
            readability
        )
        
        return min(max(psi, 0.0), 1.0)
    
    def extract_rho_enhanced(self, text: str) -> float:
        """
        Enhanced reflective depth extraction
        ρ (rho) - Wisdom through experience and reflection
        """
        doc = self.nlp(text)
        
        # Discourse marker analysis
        discourse_relations = self._identify_discourse_relations(doc)
        
        # Argument complexity
        arg_structure = self._analyze_argument_structure(doc)
        
        # Epistemic stance analysis
        epistemic_score = self._analyze_epistemic_stance(doc)
        
        # Temporal depth (references to past/future)
        temporal_depth = self._calculate_temporal_depth(doc)
        
        # Integrate signals
        rho = self._integrate_rho_signals(
            discourse_relations,
            arg_structure,
            epistemic_score,
            temporal_depth
        )
        
        return min(max(rho, 0.0), 1.0)
    
    def extract_q_enhanced(self, text: str) -> float:
        """
        Enhanced questioning frequency extraction
        q - Rate of engaging with moral questions
        """
        doc = self.nlp(text)
        
        # Direct questions
        direct_questions = self._count_questions(doc)
        
        # Implicit questioning (uncertainty, exploration)
        implicit_questions = self._detect_implicit_questioning(doc)
        
        # Moral/ethical terms
        moral_engagement = self._detect_moral_engagement(doc)
        
        # Normalize by text length
        total_questions = direct_questions + implicit_questions
        q = (total_questions / len(doc)) * moral_engagement
        
        return min(max(q * 10, 0.0), 1.0)  # Scale appropriately
    
    def extract_f_enhanced(self, text: str) -> float:
        """
        Enhanced challenge frequency extraction
        f - Frequency of confronting moral challenges
        """
        doc = self.nlp(text)
        
        # Challenge indicators
        challenge_score = self._detect_challenges(doc)
        
        # Conflict language
        conflict_score = self._detect_conflict_language(doc)
        
        # Growth language
        growth_score = self._detect_growth_language(doc)
        
        # Integrate signals
        f = (challenge_score + conflict_score + growth_score) / 3.0
        
        return min(max(f, 0.0), 1.0)
    
    def _calculate_syntactic_complexity(self, doc) -> float:
        """Calculate syntactic complexity of text"""
        if len(doc) == 0:
            return 0.0
            
        # Average sentence length
        sent_lengths = [len(sent) for sent in doc.sents]
        avg_sent_length = np.mean(sent_lengths) if sent_lengths else 0
        
        # Dependency tree depth
        max_depths = []
        for sent in doc.sents:
            depths = []
            for token in sent:
                depth = 0
                current = token
                while current.head != current:
                    depth += 1
                    current = current.head
                depths.append(depth)
            if depths:
                max_depths.append(max(depths))
        
        avg_tree_depth = np.mean(max_depths) if max_depths else 0
        
        # Subordinate clause ratio
        subord_count = sum(1 for token in doc if token.dep_ in ['advcl', 'ccomp', 'xcomp'])
        clause_ratio = subord_count / len(doc) if len(doc) > 0 else 0
        
        # Normalize and combine
        complexity = (
            min(avg_sent_length / 30, 1.0) * 0.3 +  # Normalize to 30 words
            min(avg_tree_depth / 10, 1.0) * 0.4 +   # Normalize to depth 10
            min(clause_ratio * 10, 1.0) * 0.3       # Scale clause ratio
        )
        
        return complexity
    
    def _analyze_argument_structure(self, doc) -> ArgumentStructure:
        """Analyze argumentative structure of text"""
        claims = []
        evidence = []
        relations = []
        
        # Simple claim detection (statements with strong verbs)
        claim_verbs = {'argue', 'claim', 'believe', 'assert', 'maintain', 'contend'}
        evidence_markers = {'evidence', 'study', 'research', 'data', 'example', 'shows'}
        
        for sent in doc.sents:
            sent_text = sent.text.lower()
            
            # Detect claims
            if any(verb in sent_text for verb in claim_verbs):
                claims.append(sent.text)
            
            # Detect evidence
            if any(marker in sent_text for marker in evidence_markers):
                evidence.append(sent.text)
            
            # Detect claim-evidence relations
            if claims and evidence:
                # Simple heuristic: evidence follows claim
                if len(claims) > len(relations):
                    relations.append((claims[-1], sent.text, 'supports'))
        
        # Calculate complexity score
        complexity = 0.0
        if claims:
            complexity += 0.3
        if evidence:
            complexity += 0.3
        if relations:
            complexity += 0.4
            
        return ArgumentStructure(claims, evidence, relations, complexity)
    
    def _identify_discourse_relations(self, doc) -> DiscourseRelations:
        """Identify discourse relations in text"""
        text_lower = doc.text.lower()
        
        # Count different relation types
        causal = sum(1 for marker in self.discourse_markers['causal'] 
                    if marker in text_lower)
        contrastive = sum(1 for marker in self.discourse_markers['contrastive'] 
                         if marker in text_lower)
        elaborative = sum(1 for marker in self.discourse_markers['elaborative'] 
                         if marker in text_lower)
        temporal = sum(1 for marker in self.discourse_markers['temporal'] 
                      if marker in text_lower)
        
        # Calculate coherence score
        total_markers = causal + contrastive + elaborative + temporal
        sentences = len(list(doc.sents))
        
        coherence_score = min(total_markers / max(sentences, 1), 1.0)
        
        return DiscourseRelations(
            causal, contrastive, elaborative, temporal, coherence_score
        )
    
    def _analyze_epistemic_stance(self, doc) -> float:
        """Analyze epistemic stance (certainty vs hedging)"""
        text_lower = doc.text.lower()
        
        certainty_count = sum(1 for marker in self.epistemic_markers['certainty'] 
                             if marker in text_lower)
        hedging_count = sum(1 for marker in self.epistemic_markers['hedging'] 
                           if marker in text_lower)
        
        # Balance of certainty and appropriate hedging indicates sophistication
        total = certainty_count + hedging_count
        if total == 0:
            return 0.3  # Neutral
            
        # Optimal is some certainty with appropriate hedging
        balance = abs(certainty_count - hedging_count) / total
        epistemic_score = 1.0 - (balance * 0.5)  # Less extreme is better
        
        return epistemic_score
    
    def _calculate_temporal_depth(self, doc) -> float:
        """Calculate temporal depth of reflection"""
        past_tense = sum(1 for token in doc if token.tag_ in ['VBD', 'VBN'])
        future_refs = sum(1 for token in doc if token.text.lower() in 
                         ['will', 'would', 'future', 'tomorrow', 'later'])
        
        temporal_markers = past_tense + future_refs
        temporal_depth = min(temporal_markers / len(doc), 1.0) if len(doc) > 0 else 0
        
        return temporal_depth
    
    def _calculate_readability(self, text: str) -> float:
        """Calculate readability metrics"""
        if not text or len(text.split()) < 10:
            return 0.5
            
        try:
            # Flesch Reading Ease (0-100, higher is easier)
            fre = flesch_reading_ease(text)
            # Normalize to 0-1 (invert so higher = more sophisticated)
            fre_norm = 1.0 - (min(max(fre, 0), 100) / 100)
            
            # Gunning Fog (years of education needed)
            fog = gunning_fog(text)
            # Normalize to 0-1 (12-16 years is optimal)
            fog_norm = min(max(fog - 8, 0) / 8, 1.0)
            
            return (fre_norm + fog_norm) / 2
            
        except:
            return 0.5
    
    def _count_questions(self, doc) -> int:
        """Count direct questions in text"""
        return sum(1 for sent in doc.sents if sent.text.strip().endswith('?'))
    
    def _detect_implicit_questioning(self, doc) -> int:
        """Detect implicit questioning and uncertainty"""
        questioning_words = {'wonder', 'consider', 'ponder', 'reflect', 'examine',
                           'explore', 'investigate', 'question', 'curious'}
        
        count = sum(1 for token in doc 
                   if token.lemma_ in questioning_words)
        
        return count
    
    def _detect_moral_engagement(self, doc) -> float:
        """Detect engagement with moral/ethical concepts"""
        moral_terms = {'moral', 'ethical', 'right', 'wrong', 'should', 'ought',
                      'value', 'principle', 'virtue', 'justice', 'fairness',
                      'responsibility', 'duty', 'integrity'}
        
        moral_count = sum(1 for token in doc 
                         if token.lemma_ in moral_terms)
        
        engagement = min(moral_count / len(doc), 1.0) if len(doc) > 0 else 0
        return max(engagement * 5, 0.2)  # Scale and ensure minimum
    
    def _detect_challenges(self, doc) -> float:
        """Detect language indicating challenges"""
        challenge_words = {'challenge', 'difficult', 'struggle', 'obstacle',
                          'problem', 'issue', 'confront', 'face', 'overcome'}
        
        count = sum(1 for token in doc if token.lemma_ in challenge_words)
        return min(count / len(doc) * 10, 1.0) if len(doc) > 0 else 0
    
    def _detect_conflict_language(self, doc) -> float:
        """Detect conflict-related language"""
        conflict_words = {'conflict', 'tension', 'dilemma', 'paradox',
                         'contradiction', 'opposing', 'versus', 'debate'}
        
        count = sum(1 for token in doc if token.lemma_ in conflict_words)
        return min(count / len(doc) * 10, 1.0) if len(doc) > 0 else 0
    
    def _detect_growth_language(self, doc) -> float:
        """Detect growth and development language"""
        growth_words = {'grow', 'develop', 'evolve', 'learn', 'improve',
                       'progress', 'advance', 'mature', 'strengthen'}
        
        count = sum(1 for token in doc if token.lemma_ in growth_words)
        return min(count / len(doc) * 10, 1.0) if len(doc) > 0 else 0
    
    def _integrate_psi_signals(self, syntax: float, argument: float, 
                              discourse: float, readability: float) -> float:
        """Integrate multiple signals for psi calculation"""
        # Weighted combination
        psi = (
            syntax * 0.25 +
            argument * 0.25 +
            discourse * 0.30 +
            readability * 0.20
        )
        return psi
    
    def _integrate_rho_signals(self, discourse: DiscourseRelations,
                              argument: ArgumentStructure,
                              epistemic: float,
                              temporal: float) -> float:
        """Integrate multiple signals for rho calculation"""
        # Weighted combination emphasizing coherence and depth
        rho = (
            discourse.coherence_score * 0.30 +
            argument.complexity_score * 0.25 +
            epistemic * 0.25 +
            temporal * 0.20
        )
        return rho


if __name__ == "__main__":
    # Test the enhanced extractor
    extractor = EnhancedNLPExtractor()
    
    test_text = """
    I believe that ethical decision-making requires careful consideration 
    of multiple perspectives. For example, recent studies have shown that 
    when we face moral dilemmas, our initial intuitions might conflict with 
    reasoned analysis. However, through reflection and dialogue, we can 
    develop more nuanced understanding. This process, though challenging, 
    ultimately strengthens our moral reasoning capabilities.
    """
    
    print(f"Psi (clarity): {extractor.extract_psi_enhanced(test_text):.3f}")
    print(f"Rho (wisdom): {extractor.extract_rho_enhanced(test_text):.3f}")
    print(f"Q (questioning): {extractor.extract_q_enhanced(test_text):.3f}")
    print(f"F (challenges): {extractor.extract_f_enhanced(test_text):.3f}")