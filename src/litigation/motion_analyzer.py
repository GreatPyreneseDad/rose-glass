"""
Motion Analysis Pipeline
Analyzes legal motions through Rose Glass for drafting optimization

Author: Christopher MacGregor bin Joseph
Created: January 2026
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re
import argparse

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.unified_lens import get_lens, CoherenceResult


class MotionAnalyzer:
    """
    Analyze legal motions for coherence optimization.
    Used for both attacking opponent filings and improving own drafts.
    """
    
    def __init__(self):
        self.lens = get_lens()
        
    def analyze_motion(self, text: str) -> Dict:
        """
        Full motion analysis with paragraph-level breakdown.
        Returns overall coherence plus per-paragraph analysis.
        """
        # Overall analysis
        overall = self.lens.analyze(text)
        
        # Paragraph analysis
        paragraphs = self._split_paragraphs(text)
        paragraph_results = []
        
        for i, para in enumerate(paragraphs):
            if len(para.strip()) < 20:
                continue
            result = self.lens.analyze(para)
            attack_vectors = self._identify_attack_vectors(result)
            
            paragraph_results.append({
                'index': i,
                'preview': para[:100] + '...' if len(para) > 100 else para,
                'full_text': para,
                'analysis': result.to_dict(),
                'attack_vectors': attack_vectors
            })
        
        # Identify weakest paragraphs
        weak_paragraphs = [
            p for p in paragraph_results 
            if p['analysis']['coherence'] < 1.5 or p['analysis']['psi'] < 0.4
        ]
        
        # Sort by weakness (lowest coherence first)
        weak_paragraphs.sort(key=lambda x: x['analysis']['coherence'])
        
        return {
            'overall': overall.to_dict(),
            'paragraph_count': len(paragraph_results),
            'paragraphs': paragraph_results,
            'weak_points': weak_paragraphs,
            'attack_summary': self._summarize_attacks(paragraph_results)
        }
    
    def _split_paragraphs(self, text: str) -> List[str]:
        """Split motion into paragraphs."""
        # Split on double newlines or numbered paragraphs
        paragraphs = re.split(r'\n\s*\n|\n\s*\d+\.\s+', text)
        return [p.strip() for p in paragraphs if p.strip()]
    
    def _identify_attack_vectors(self, result: CoherenceResult) -> List[str]:
        """Identify potential attack vectors from analysis."""
        vectors = []
        
        if result.psi < 0.5:
            vectors.append("INTERNAL_CONTRADICTION")
        if result.rho < 0.4:
            vectors.append("LACKS_SPECIFICITY")
        if result.q > 0.7 and result.rho < 0.5:
            vectors.append("EMOTIONAL_WITHOUT_EVIDENCE")
        if 'COHERENCE_WARNING' in result.flags:
            vectors.append("COHERENCE_COLLAPSE")
        if result.psi < 0.4 and result.q > 0.6:
            vectors.append("EMOTIONAL_INCONSISTENCY")
            
        return vectors
    
    def _summarize_attacks(self, paragraph_results: List[Dict]) -> Dict:
        """Summarize attack opportunities across all paragraphs."""
        all_vectors = []
        for p in paragraph_results:
            all_vectors.extend(p['attack_vectors'])
        
        from collections import Counter
        vector_counts = Counter(all_vectors)
        
        # Find paragraphs with most vulnerabilities
        most_vulnerable = sorted(
            [p for p in paragraph_results if p['attack_vectors']],
            key=lambda x: len(x['attack_vectors']),
            reverse=True
        )[:3]
        
        return {
            'total_weak_paragraphs': len([p for p in paragraph_results if p['attack_vectors']]),
            'total_paragraphs': len(paragraph_results),
            'weakness_ratio': len([p for p in paragraph_results if p['attack_vectors']]) / max(len(paragraph_results), 1),
            'attack_vector_frequency': dict(vector_counts),
            'primary_weakness': vector_counts.most_common(1)[0][0] if vector_counts else None,
            'most_vulnerable_paragraphs': [p['index'] for p in most_vulnerable]
        }
    
    def optimize_draft(self, draft: str) -> Dict:
        """
        Analyze own draft and suggest optimizations.
        Returns analysis plus specific improvement suggestions.
        """
        analysis = self.analyze_motion(draft)
        
        suggestions = []
        
        for para in analysis['paragraphs']:
            a = para['analysis']
            para_idx = para['index']
            
            if a['psi'] < 0.6:
                suggestions.append({
                    'paragraph': para_idx,
                    'dimension': 'psi',
                    'current_value': a['psi'],
                    'issue': 'Internal consistency low',
                    'suggestion': 'Check for contradictory statements or unclear logic. Ensure claims follow logically.',
                    'priority': 'HIGH' if a['psi'] < 0.4 else 'MEDIUM'
                })
                
            if a['rho'] < 0.5:
                suggestions.append({
                    'paragraph': para_idx,
                    'dimension': 'rho',
                    'current_value': a['rho'],
                    'issue': 'Specificity low',
                    'suggestion': 'Add specific dates, dollar amounts, exhibit references, or statutory citations.',
                    'priority': 'HIGH' if a['rho'] < 0.3 else 'MEDIUM'
                })
                
            if a['q'] > 0.7:
                suggestions.append({
                    'paragraph': para_idx,
                    'dimension': 'q',
                    'current_value': a['q'],
                    'issue': 'High emotional activation',
                    'suggestion': 'Consider reducing emotional language for judicial audience. Replace adjectives with evidence.',
                    'priority': 'MEDIUM'
                })
                
            if a['f'] < 0.3:
                suggestions.append({
                    'paragraph': para_idx,
                    'dimension': 'f',
                    'current_value': a['f'],
                    'issue': 'Low institutional alignment',
                    'suggestion': 'Add references to court standards, statutory requirements, or procedural rules.',
                    'priority': 'LOW'
                })
        
        # Sort by priority
        priority_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
        suggestions.sort(key=lambda x: priority_order[x['priority']])
        
        return {
            'analysis': analysis,
            'suggestions': suggestions,
            'high_priority_count': len([s for s in suggestions if s['priority'] == 'HIGH']),
            'overall_recommendation': self._overall_recommendation(analysis['overall'])
        }
    
    def _overall_recommendation(self, overall: Dict) -> str:
        """Generate overall recommendation based on analysis."""
        c = overall['coherence']
        psi = overall['psi']
        rho = overall['rho']
        q = overall['q']
        
        if c > 2.5 and psi > 0.7:
            return "Strong draft. Minor refinements may help but document is ready for filing."
        elif c > 2.0:
            return "Good draft. Review flagged paragraphs for targeted improvements."
        elif c > 1.5:
            return "Adequate draft. Address high-priority suggestions before filing."
        else:
            if psi < 0.5:
                return "Draft needs revision: Internal consistency issues detected. Review for contradictions."
            elif rho < 0.4:
                return "Draft needs revision: Add specific evidence, dates, and citations."
            elif q > 0.7:
                return "Draft needs revision: Reduce emotional language, increase factual support."
            else:
                return "Draft needs significant revision. Multiple coherence issues detected."
    
    def compare_filings(self, our_filing: str, their_filing: str) -> Dict:
        """
        Compare our filing against opponent's for strategic advantage.
        Identifies relative strengths and attack opportunities.
        """
        our_analysis = self.analyze_motion(our_filing)
        their_analysis = self.analyze_motion(their_filing)
        
        # Calculate advantages
        our_c = our_analysis['overall']['coherence']
        their_c = their_analysis['overall']['coherence']
        
        our_psi = our_analysis['overall']['psi']
        their_psi = their_analysis['overall']['psi']
        
        our_rho = our_analysis['overall']['rho']
        their_rho = their_analysis['overall']['rho']
        
        advantages = []
        if our_psi > their_psi + 0.1:
            advantages.append("Our filing has stronger internal consistency")
        if our_rho > their_rho + 0.1:
            advantages.append("Our filing has better evidentiary support")
        if their_analysis['overall']['q'] > 0.6 and their_rho < 0.5:
            advantages.append("Their filing relies on emotion over evidence")
        
        vulnerabilities = []
        if our_psi < their_psi - 0.1:
            vulnerabilities.append("Our filing may have consistency issues to address")
        if our_rho < their_rho - 0.1:
            vulnerabilities.append("Consider adding more specific evidence")
        
        return {
            'our_analysis': our_analysis,
            'their_analysis': their_analysis,
            'coherence_advantage': round(our_c - their_c, 3),
            'consistency_advantage': round(our_psi - their_psi, 3),
            'evidence_advantage': round(our_rho - their_rho, 3),
            'our_advantages': advantages,
            'our_vulnerabilities': vulnerabilities,
            'their_weak_points': their_analysis['weak_points'],
            'strategic_targets': [
                {
                    'paragraph': p['index'],
                    'preview': p['preview'],
                    'attack_vectors': p['attack_vectors'],
                    'coherence': p['analysis']['coherence']
                }
                for p in their_analysis['paragraphs']
                if p['attack_vectors']
            ]
        }
    
    def generate_attack_questions(self, their_filing: str, max_questions: int = 5) -> List[Dict]:
        """
        Generate cross-examination style questions targeting weak points in opponent's filing.
        """
        analysis = self.analyze_motion(their_filing)
        questions = []
        
        for para in analysis['paragraphs']:
            if not para['attack_vectors']:
                continue
                
            a = para['analysis']
            preview = para['preview']
            
            if 'INTERNAL_CONTRADICTION' in para['attack_vectors']:
                questions.append({
                    'type': 'contradiction',
                    'target_paragraph': para['index'],
                    'question': f"Your filing states '{preview[:50]}...' - how do you reconcile this with [prior statement]?",
                    'severity': 1 - a['psi']
                })
            
            if 'LACKS_SPECIFICITY' in para['attack_vectors']:
                questions.append({
                    'type': 'specificity',
                    'target_paragraph': para['index'],
                    'question': f"Regarding '{preview[:50]}...' - can you provide specific dates and documentation?",
                    'severity': 1 - a['rho']
                })
            
            if 'EMOTIONAL_WITHOUT_EVIDENCE' in para['attack_vectors']:
                questions.append({
                    'type': 'evidence_demand',
                    'target_paragraph': para['index'],
                    'question': f"Setting aside characterizations, what documentary evidence supports '{preview[:50]}...'?",
                    'severity': a['q']
                })
        
        # Sort by severity and return top N
        questions.sort(key=lambda x: x['severity'], reverse=True)
        return questions[:max_questions]


def main():
    """Command-line interface for motion analysis."""
    parser = argparse.ArgumentParser(description='Analyze legal motions through Rose Glass')
    parser.add_argument('--input', '-i', type=str, help='Input file to analyze')
    parser.add_argument('--optimize', '-o', action='store_true', help='Run optimization analysis')
    parser.add_argument('--compare', '-c', type=str, help='Compare against opponent filing')
    parser.add_argument('--questions', '-q', type=int, default=0, help='Generate N attack questions')
    parser.add_argument('--text', '-t', type=str, help='Direct text input')
    
    args = parser.parse_args()
    
    analyzer = MotionAnalyzer()
    
    # Get text input
    if args.input:
        with open(args.input, 'r') as f:
            text = f.read()
    elif args.text:
        text = args.text
    else:
        print("Error: Provide --input file or --text")
        sys.exit(1)
    
    # Run appropriate analysis
    if args.compare:
        with open(args.compare, 'r') as f:
            their_text = f.read()
        result = analyzer.compare_filings(text, their_text)
        print("\n=== FILING COMPARISON ===")
        print(f"Coherence Advantage: {result['coherence_advantage']:+.3f}")
        print(f"Our Advantages: {result['our_advantages']}")
        print(f"Our Vulnerabilities: {result['our_vulnerabilities']}")
        print(f"Their Weak Points: {len(result['their_weak_points'])}")
        
    elif args.optimize:
        result = analyzer.optimize_draft(text)
        print("\n=== DRAFT OPTIMIZATION ===")
        print(f"Overall Coherence: {result['analysis']['overall']['coherence']:.3f}")
        print(f"Recommendation: {result['overall_recommendation']}")
        print(f"\nHigh Priority Issues: {result['high_priority_count']}")
        for s in result['suggestions'][:5]:
            print(f"  Para {s['paragraph']}: {s['issue']} ({s['priority']})")
            
    elif args.questions > 0:
        questions = analyzer.generate_attack_questions(text, args.questions)
        print(f"\n=== ATTACK QUESTIONS ({len(questions)}) ===")
        for q in questions:
            print(f"\n[{q['type'].upper()}] Para {q['target_paragraph']}:")
            print(f"  {q['question']}")
            
    else:
        result = analyzer.analyze_motion(text)
        print("\n=== MOTION ANALYSIS ===")
        print(f"Overall: {result['overall']}")
        print(f"Paragraphs: {result['paragraph_count']}")
        print(f"Weak Points: {len(result['weak_points'])}")
        print(f"Attack Summary: {result['attack_summary']}")


if __name__ == "__main__":
    main()
