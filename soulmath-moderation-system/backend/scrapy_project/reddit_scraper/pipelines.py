import json
from datetime import datetime
from typing import Dict, Any

class SoulMathAnalysisPipeline:
    """Analyzes scraped content using SoulMath coherence metrics"""
    
    def process_item(self, item, spider):
        text = item.get('text', '') or item.get('title', '')
        if text:
            analysis = self.analyze_moderation_risk(text)
            
            # Add SoulMath analysis to item
            item['psi'] = analysis['psi']
            item['rho'] = analysis['rho']
            item['coherence'] = analysis['coherence']
            item['toxicity_risk'] = analysis['toxicity_risk']
            item['manipulation_risk'] = analysis['manipulation_risk']
            item['extremism_risk'] = analysis['extremism_risk']
            item['spam_risk'] = analysis['spam_risk']
            item['harassment_risk'] = analysis['harassment_risk']
            item['discourse_collapse'] = analysis['discourse_collapse']
            item['escalation_risk'] = analysis['escalation_risk']
            item['overall_risk'] = analysis['overall_risk']
        
        # Add timestamp
        item['scraped_at'] = datetime.utcnow().isoformat()
        
        return item
    
    def analyze_moderation_risk(self, text: str) -> Dict[str, float]:
        """SoulMath coherence-based content analysis"""
        words = text.lower().split()
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        # Core coherence metrics
        avg_words_per_sentence = len(words) / max(len(sentences), 1)
        psi = min(1.0, 0.3 + (avg_words_per_sentence / 20))
        unique_words = len(set(words))
        rho = min(1.0, unique_words / max(len(words), 1) + 0.2)
        
        # Toxicity detection
        toxic_patterns = [
            'hate', 'kill', 'die', 'stupid', 'idiot', 'loser', 'waste', 'trash',
            'scum', 'moron', 'worthless', 'disgusting', 'retard', 'pathetic'
        ]
        
        harassment_patterns = [
            'you should', 'why don\'t you', 'go back to', 'your kind', 'typical',
            'always complaining', 'people like you', 'cry more'
        ]
        
        toxic_score = sum(1 for w in words if w in toxic_patterns)
        harassment_score = sum(1 for p in harassment_patterns if p in text.lower())
        q_toxic = min(1.0, (toxic_score + harassment_score) / max(len(words), 1) * 20)
        
        # Manipulation detection
        manipulation_patterns = [
            'everyone knows', 'obviously', 'wake up', 'sheep', 'brainwashed',
            'any reasonable person', 'it\'s clear that', 'open your eyes'
        ]
        
        manipulation_score = sum(1 for p in manipulation_patterns if p in text.lower())
        f_manipulation = min(1.0, manipulation_score / 10)
        
        # Extremism detection
        extremism_patterns = [
            'destroy the', 'war on', 'invasion', 'replacement',
            'eliminate', 'purge', 'they want to', 'cleanse', 'taking over'
        ]
        
        extremism_score = sum(1 for p in extremism_patterns if p in text.lower())
        extremism_risk = min(1.0, extremism_score / 5)
        
        # Spam detection
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        max_repetition = max(word_counts.values()) if word_counts else 0
        spam_risk = 1.0 if max_repetition > len(words) * 0.3 else 0.0
        
        # Advanced metrics
        discourse_collapse = max(0, 1 - (psi * (1 - q_toxic) * rho) / 3)
        escalation_risk = (q_toxic + f_manipulation + extremism_risk + harassment_score/3) / 4
        overall_risk = escalation_risk * 1.1  # Witness multiplier
        
        return {
            'psi': round(psi, 3),
            'rho': round(rho, 3),
            'coherence': round(psi, 3),
            'toxicity_risk': round(q_toxic, 3),
            'manipulation_risk': round(f_manipulation, 3),
            'extremism_risk': round(extremism_risk, 3),
            'spam_risk': round(spam_risk, 3),
            'harassment_risk': round(min(1.0, harassment_score / 3), 3),
            'discourse_collapse': round(discourse_collapse, 3),
            'escalation_risk': round(escalation_risk, 3),
            'overall_risk': round(overall_risk, 3)
        }

class JsonWriterPipeline:
    """Writes items to JSON file"""
    
    def open_spider(self, spider):
        self.file = open('reddit_moderation_data.jsonl', 'w')
    
    def close_spider(self, spider):
        self.file.close()
    
    def process_item(self, item, spider):
        line = json.dumps(dict(item)) + "\n"
        self.file.write(line)
        return item