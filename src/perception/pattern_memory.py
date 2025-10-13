"""
Pattern Memory System
Tracks perception patterns over time to enable learning and adaptation
"""

from typing import Dict, List, Optional, Any
from collections import deque
from dataclasses import dataclass, field
import numpy as np
from datetime import datetime
import json


@dataclass
class MemoryEntry:
    """Single entry in pattern memory"""
    timestamp: datetime
    pattern: 'DimensionalPattern'
    context: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    calibration: str = 'default'


class PatternMemory:
    """
    Stores and retrieves historical perception patterns
    Enables learning from past interactions and pattern evolution tracking
    """
    
    def __init__(self, max_entries: int = 1000):
        self.max_entries = max_entries
        self.memory: deque = deque(maxlen=max_entries)
        self.pattern_clusters: Dict[str, List[MemoryEntry]] = {}
        self.evolution_tracking: Dict[str, List[float]] = {
            'psi': [],
            'rho': [],
            'q': [],
            'f': []
        }
        
    def update(self, perception: 'Perception'):
        """
        Add a new perception to memory
        
        Args:
            perception: The perception to store
        """
        from .rose_glass_perception import DimensionalPattern
        
        entry = MemoryEntry(
            timestamp=datetime.now(),
            pattern=perception.dimensions,
            context={'calibration': perception.calibration},
            confidence=1.0 - perception.uncertainty_level,
            calibration=perception.calibration
        )
        
        self.memory.append(entry)
        
        # Update evolution tracking
        self.evolution_tracking['psi'].append(perception.dimensions.psi)
        self.evolution_tracking['rho'].append(perception.dimensions.rho)
        self.evolution_tracking['q'].append(perception.dimensions.q)
        self.evolution_tracking['f'].append(perception.dimensions.f)
        
        # Maintain size limits
        for key in self.evolution_tracking:
            if len(self.evolution_tracking[key]) > self.max_entries:
                self.evolution_tracking[key] = self.evolution_tracking[key][-self.max_entries:]
                
        # Update clusters
        self._update_clusters(entry)
        
    def get_similarity_score(self, pattern: 'DimensionalPattern') -> float:
        """
        Calculate how similar a pattern is to historical patterns
        
        Args:
            pattern: Pattern to compare
            
        Returns:
            Similarity score between 0 and 1
        """
        if not self.memory:
            return 0.5  # No history, neutral score
            
        # Get recent patterns (last 20)
        recent_patterns = list(self.memory)[-20:]
        
        similarities = []
        for entry in recent_patterns:
            # Calculate euclidean distance in 4D space
            distance = np.sqrt(
                (entry.pattern.psi - pattern.psi) ** 2 +
                (entry.pattern.rho - pattern.rho) ** 2 +
                (entry.pattern.q - pattern.q) ** 2 +
                (entry.pattern.f - pattern.f) ** 2
            )
            
            # Convert distance to similarity (max distance is sqrt(4) = 2)
            similarity = 1 - (distance / 2)
            
            # Weight by confidence
            weighted_similarity = similarity * entry.confidence
            similarities.append(weighted_similarity)
            
        return np.mean(similarities) if similarities else 0.5
        
    def get_evolution_trend(self, dimension: str, window: int = 10) -> Dict[str, float]:
        """
        Get the evolution trend for a specific dimension
        
        Args:
            dimension: One of 'psi', 'rho', 'q', 'f'
            window: Number of recent entries to analyze
            
        Returns:
            Dict with trend information
        """
        if dimension not in self.evolution_tracking:
            raise ValueError(f"Unknown dimension: {dimension}")
            
        values = self.evolution_tracking[dimension]
        
        if len(values) < 2:
            return {
                'trend': 0.0,
                'stability': 1.0,
                'current': values[-1] if values else 0.5
            }
            
        recent_values = values[-window:]
        
        # Calculate trend (positive = increasing, negative = decreasing)
        if len(recent_values) >= 2:
            x = np.arange(len(recent_values))
            coefficients = np.polyfit(x, recent_values, 1)
            trend = coefficients[0]
        else:
            trend = 0.0
            
        # Calculate stability (inverse of variance)
        stability = 1.0 / (1.0 + np.std(recent_values))
        
        return {
            'trend': trend,
            'stability': stability,
            'current': recent_values[-1],
            'mean': np.mean(recent_values),
            'std': np.std(recent_values)
        }
        
    def find_similar_contexts(self, 
                            current_pattern: 'DimensionalPattern', 
                            threshold: float = 0.8) -> List[MemoryEntry]:
        """
        Find historical contexts similar to current pattern
        
        Args:
            current_pattern: Pattern to match
            threshold: Similarity threshold (0-1)
            
        Returns:
            List of similar memory entries
        """
        similar_entries = []
        
        for entry in self.memory:
            # Calculate similarity
            distance = np.sqrt(
                (entry.pattern.psi - current_pattern.psi) ** 2 +
                (entry.pattern.rho - current_pattern.rho) ** 2 +
                (entry.pattern.q - current_pattern.q) ** 2 +
                (entry.pattern.f - current_pattern.f) ** 2
            )
            
            similarity = 1 - (distance / 2)
            
            if similarity >= threshold:
                similar_entries.append(entry)
                
        # Sort by similarity (most similar first)
        similar_entries.sort(
            key=lambda e: self._calculate_similarity(e.pattern, current_pattern),
            reverse=True
        )
        
        return similar_entries[:10]  # Return top 10
        
    def _calculate_similarity(self, pattern1: 'DimensionalPattern', 
                            pattern2: 'DimensionalPattern') -> float:
        """Calculate similarity between two patterns"""
        distance = np.sqrt(
            (pattern1.psi - pattern2.psi) ** 2 +
            (pattern1.rho - pattern2.rho) ** 2 +
            (pattern1.q - pattern2.q) ** 2 +
            (pattern1.f - pattern2.f) ** 2
        )
        return 1 - (distance / 2)
        
    def _update_clusters(self, entry: MemoryEntry):
        """Update pattern clusters for quick retrieval"""
        # Simple clustering by dominant dimension
        dominant_dim = self._get_dominant_dimension(entry.pattern)
        
        if dominant_dim not in self.pattern_clusters:
            self.pattern_clusters[dominant_dim] = []
            
        self.pattern_clusters[dominant_dim].append(entry)
        
        # Maintain cluster size
        if len(self.pattern_clusters[dominant_dim]) > 100:
            self.pattern_clusters[dominant_dim] = \
                self.pattern_clusters[dominant_dim][-100:]
                
    def _get_dominant_dimension(self, pattern: 'DimensionalPattern') -> str:
        """Determine which dimension is most prominent"""
        values = {
            'psi': pattern.psi,
            'rho': pattern.rho,
            'q': pattern.q,
            'f': pattern.f
        }
        
        # Normalize to deviation from 0.5 (neutral)
        deviations = {k: abs(v - 0.5) for k, v in values.items()}
        
        return max(deviations, key=deviations.get)
        
    def get_pattern_summary(self) -> Dict[str, Any]:
        """Get summary statistics of stored patterns"""
        if not self.memory:
            return {'entries': 0, 'patterns': {}}
            
        recent_entries = list(self.memory)[-100:]  # Last 100
        
        summary = {
            'entries': len(self.memory),
            'patterns': {}
        }
        
        for dim in ['psi', 'rho', 'q', 'f']:
            values = [getattr(e.pattern, dim) for e in recent_entries]
            summary['patterns'][dim] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'trend': self.get_evolution_trend(dim)['trend']
            }
            
        return summary
        
    def export_memory(self, filepath: str):
        """Export memory to JSON file"""
        data = {
            'entries': [
                {
                    'timestamp': e.timestamp.isoformat(),
                    'pattern': {
                        'psi': e.pattern.psi,
                        'rho': e.pattern.rho,
                        'q': e.pattern.q,
                        'f': e.pattern.f
                    },
                    'confidence': e.confidence,
                    'calibration': e.calibration
                }
                for e in self.memory
            ],
            'summary': self.get_pattern_summary()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
            
    def import_memory(self, filepath: str):
        """Import memory from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        # Clear existing memory
        self.memory.clear()
        self.evolution_tracking = {
            'psi': [],
            'rho': [],
            'q': [],
            'f': []
        }
        
        # Reconstruct memory
        from .rose_glass_perception import DimensionalPattern
        
        for entry_data in data['entries']:
            pattern = DimensionalPattern(
                psi=entry_data['pattern']['psi'],
                rho=entry_data['pattern']['rho'],
                q=entry_data['pattern']['q'],
                f=entry_data['pattern']['f']
            )
            
            entry = MemoryEntry(
                timestamp=datetime.fromisoformat(entry_data['timestamp']),
                pattern=pattern,
                confidence=entry_data['confidence'],
                calibration=entry_data.get('calibration', 'default')
            )
            
            self.memory.append(entry)
            
            # Update evolution tracking
            for dim in ['psi', 'rho', 'q', 'f']:
                self.evolution_tracking[dim].append(getattr(pattern, dim))
                
    def __len__(self):
        return len(self.memory)
        
    def clear(self):
        """Clear all memory"""
        self.memory.clear()
        self.pattern_clusters.clear()
        for key in self.evolution_tracking:
            self.evolution_tracking[key].clear()