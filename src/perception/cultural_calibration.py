"""
Cultural Calibration System
Adjusts perception parameters based on cultural and temporal context
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class CalibrationPreset:
    """Defines calibration parameters for a specific cultural/temporal context"""
    name: str
    description: str
    km: float = 0.2  # Saturation constant
    ki: float = 0.8  # Inhibition constant
    coupling_strength: float = 0.15
    dimension_weights: Dict[str, float] = None
    
    def __post_init__(self):
        if self.dimension_weights is None:
            self.dimension_weights = {
                'psi': 1.0,
                'rho': 1.0, 
                'q': 1.0,
                'f': 1.0
            }


class CulturalCalibration:
    """
    Manages cultural calibration presets and applies appropriate adjustments
    to perception based on context
    """
    
    def __init__(self):
        self.calibrations = self._initialize_presets()
        self.default_calibration = 'modern_western_academic'
        
    def _initialize_presets(self) -> Dict[str, CalibrationPreset]:
        """Initialize default cultural calibration presets"""
        presets = {
            'modern_western_academic': CalibrationPreset(
                name='modern_western_academic',
                description='Default calibration for modern Western academic discourse',
                km=0.2,
                ki=0.8,
                coupling_strength=0.15,
                dimension_weights={
                    'psi': 1.2,  # High value on logical consistency
                    'rho': 1.0,  # Moderate wisdom appreciation
                    'q': 0.7,    # Lower emotional expression
                    'f': 0.8     # Moderate social architecture
                }
            ),
            
            'medieval_islamic': CalibrationPreset(
                name='medieval_islamic',
                description='Calibration for medieval Islamic philosophical texts',
                km=1.2,
                ki=0.5,
                coupling_strength=0.7,
                dimension_weights={
                    'psi': 1.5,  # Very high logical rigor
                    'rho': 1.3,  # Deep wisdom tradition
                    'q': 0.4,    # Restrained emotion
                    'f': 0.9     # Community wisdom
                }
            ),
            
            'indigenous_oral': CalibrationPreset(
                name='indigenous_oral',
                description='Calibration for indigenous oral tradition transcripts',
                km=0.8,
                ki=1.2,
                coupling_strength=0.5,
                dimension_weights={
                    'psi': 0.7,  # Circular rather than linear logic
                    'rho': 1.5,  # Deep ancestral wisdom
                    'q': 1.2,    # Integrated emotion
                    'f': 1.5     # Strong collective focus
                }
            ),
            
            'digital_native': CalibrationPreset(
                name='digital_native',
                description='Calibration for rapid digital communication',
                km=0.1,
                ki=1.5,
                coupling_strength=0.1,
                dimension_weights={
                    'psi': 0.8,  # Fragmented is acceptable
                    'rho': 0.6,  # Less emphasis on accumulated wisdom
                    'q': 1.3,    # High emotional expression
                    'f': 1.1     # Network-based social patterns
                }
            ),
            
            'buddhist_contemplative': CalibrationPreset(
                name='buddhist_contemplative',
                description='Calibration for Buddhist contemplative teachings',
                km=0.5,
                ki=0.5,
                coupling_strength=0.3,
                dimension_weights={
                    'psi': 0.9,  # Paradox is feature not bug
                    'rho': 1.4,  # Deep wisdom tradition
                    'q': 0.8,    # Balanced emotion
                    'f': 1.0     # Sangha connection
                }
            ),
            
            'activist_movement': CalibrationPreset(
                name='activist_movement',
                description='Calibration for activist and movement communication',
                km=0.3,
                ki=2.0,
                coupling_strength=0.2,
                dimension_weights={
                    'psi': 0.9,
                    'rho': 0.8,
                    'q': 1.5,    # High moral activation
                    'f': 1.4     # Strong collective action
                }
            ),
            
            'technical_documentation': CalibrationPreset(
                name='technical_documentation',
                description='Calibration for technical and API documentation',
                km=0.1,
                ki=0.5,
                coupling_strength=0.05,
                dimension_weights={
                    'psi': 1.5,  # Extreme precision required
                    'rho': 0.8,  # Some experience helpful
                    'q': 0.3,    # Minimal emotion
                    'f': 0.5     # Individual understanding
                }
            ),
            
            'therapeutic_dialogue': CalibrationPreset(
                name='therapeutic_dialogue',
                description='Calibration for therapeutic and counseling contexts',
                km=0.4,
                ki=1.0,
                coupling_strength=0.25,
                dimension_weights={
                    'psi': 0.8,  # Emotional logic valued
                    'rho': 1.1,  # Life experience important
                    'q': 1.3,    # Emotional expression central
                    'f': 0.9     # Balance individual/relational
                }
            )
        }
        
        return presets
        
    def apply_calibration(self, 
                         pattern: 'DimensionalPattern', 
                         calibration_name: str) -> 'DimensionalPattern':
        """
        Apply cultural calibration to raw perception patterns
        
        Args:
            pattern: Raw dimensional pattern
            calibration_name: Name of calibration to apply
            
        Returns:
            Calibrated pattern
        """
        if calibration_name not in self.calibrations:
            calibration_name = self.default_calibration
            
        preset = self.calibrations[calibration_name]
        
        # Apply dimension weights
        calibrated_psi = pattern.psi * preset.dimension_weights['psi']
        calibrated_rho = pattern.rho * preset.dimension_weights['rho']
        calibrated_q = pattern.q * preset.dimension_weights['q']
        calibrated_f = pattern.f * preset.dimension_weights['f']
        
        # Apply biological optimization to q
        calibrated_q = self._optimize_q(calibrated_q, preset.km, preset.ki)
        
        # Apply coupling effects
        coupling = preset.coupling_strength * np.mean([
            calibrated_psi,
            calibrated_rho,
            calibrated_f
        ])
        
        # Create calibrated pattern
        from .rose_glass_perception import DimensionalPattern
        
        return DimensionalPattern(
            psi=min(1.0, calibrated_psi),
            rho=min(1.0, calibrated_rho),
            q=min(1.0, calibrated_q),
            f=min(1.0, calibrated_f),
            confidence=pattern.confidence,
            metadata={
                'calibration': calibration_name,
                'coupling': coupling,
                'original': pattern
            }
        )
        
    def _optimize_q(self, q: float, km: float, ki: float) -> float:
        """Apply biological optimization function"""
        return q / (km + q + (q ** 2) / ki)
        
    def get_applicable_calibrations(self, context: Dict) -> List[str]:
        """
        Determine which calibrations might apply based on context
        
        Args:
            context: Context information (language, domain, etc.)
            
        Returns:
            List of applicable calibration names
        """
        applicable = []
        
        # Always include default
        applicable.append(self.default_calibration)
        
        # Add others based on context clues
        context_lower = str(context).lower()
        
        if any(word in context_lower for word in ['technical', 'api', 'documentation']):
            applicable.append('technical_documentation')
            
        if any(word in context_lower for word in ['activist', 'movement', 'justice']):
            applicable.append('activist_movement')
            
        if any(word in context_lower for word in ['medieval', 'islamic', 'arabic']):
            applicable.append('medieval_islamic')
            
        if any(word in context_lower for word in ['indigenous', 'oral', 'story']):
            applicable.append('indigenous_oral')
            
        if any(word in context_lower for word in ['digital', 'online', 'internet']):
            applicable.append('digital_native')
            
        if any(word in context_lower for word in ['buddhist', 'contemplative', 'meditation']):
            applicable.append('buddhist_contemplative')
            
        if any(word in context_lower for word in ['therapy', 'counseling', 'healing']):
            applicable.append('therapeutic_dialogue')
            
        return list(set(applicable))  # Remove duplicates
        
    def compare_calibrations(self, pattern: 'DimensionalPattern') -> Dict[str, float]:
        """
        Compare how a pattern appears through different calibrations
        
        Args:
            pattern: Pattern to analyze
            
        Returns:
            Dict of calibration names to coherence constructions
        """
        results = {}
        
        for cal_name, preset in self.calibrations.items():
            calibrated = self.apply_calibration(pattern, cal_name)
            
            # Calculate coherence construction
            coherence = (calibrated.psi + 
                        (calibrated.rho * calibrated.psi) + 
                        calibrated.q + 
                        (calibrated.f * calibrated.psi))
            
            results[cal_name] = coherence
            
        return results
        
    def add_calibration(self, preset: CalibrationPreset):
        """Add a new cultural calibration preset"""
        self.calibrations[preset.name] = preset
        
    def list_calibrations(self) -> List[str]:
        """List all available calibration names"""
        return list(self.calibrations.keys())
        
    def get_calibration_info(self, name: str) -> Optional[CalibrationPreset]:
        """Get detailed information about a calibration"""
        return self.calibrations.get(name)
        
    def suggest_calibration(self, text: str) -> Tuple[str, float]:
        """
        Suggest the most appropriate calibration based on text analysis
        
        Args:
            text: Text to analyze
            
        Returns:
            Tuple of (calibration_name, confidence)
        """
        # Simple heuristic-based suggestion
        # In later phases, this would use ML
        
        scores = {}
        text_lower = text.lower()
        
        # Check for technical language
        tech_words = ['function', 'api', 'parameter', 'return', 'implementation']
        tech_score = sum(1 for word in tech_words if word in text_lower)
        scores['technical_documentation'] = tech_score / 5
        
        # Check for activist language
        activist_words = ['justice', 'movement', 'together', 'fight', 'change']
        activist_score = sum(1 for word in activist_words if word in text_lower)
        scores['activist_movement'] = activist_score / 5
        
        # Check for contemplative language
        contemplative_words = ['mindful', 'present', 'awareness', 'being', 'peace']
        contemplative_score = sum(1 for word in contemplative_words if word in text_lower)
        scores['buddhist_contemplative'] = contemplative_score / 5
        
        # Check for digital native patterns
        if len(text) < 280 and any(char in text for char in ['@', '#', '👍', '😊']):
            scores['digital_native'] = 0.8
            
        # Default to modern western academic if no strong signals
        if all(score < 0.4 for score in scores.values()):
            return (self.default_calibration, 0.6)
            
        # Return highest scoring calibration
        best_cal = max(scores, key=scores.get)
        confidence = scores[best_cal]
        
        return (best_cal, confidence)
        
    def export_calibration(self, name: str) -> Dict:
        """Export calibration as dictionary for sharing"""
        if name not in self.calibrations:
            raise ValueError(f"Calibration '{name}' not found")
            
        preset = self.calibrations[name]
        
        return {
            'name': preset.name,
            'description': preset.description,
            'km': preset.km,
            'ki': preset.ki,
            'coupling_strength': preset.coupling_strength,
            'dimension_weights': preset.dimension_weights
        }
        
    def import_calibration(self, calibration_data: Dict):
        """Import calibration from dictionary"""
        preset = CalibrationPreset(
            name=calibration_data['name'],
            description=calibration_data['description'],
            km=calibration_data['km'],
            ki=calibration_data['ki'],
            coupling_strength=calibration_data['coupling_strength'],
            dimension_weights=calibration_data['dimension_weights']
        )
        
        self.add_calibration(preset)