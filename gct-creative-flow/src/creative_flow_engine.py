"""
GCT Creative Flow Analysis System
Real-time creativity enhancement through coherence tracking
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import asyncio
from collections import deque
from scipy import signal
from scipy.stats import entropy
import json
import sys
import os

# Import GCT components from parent project
sys.path.append(os.path.join(os.path.dirname(__file__), '../../gct-market-sentiment/src'))
from gct_engine import GCTEngine, GCTVariables, GCTResult, GCTParameters


class CreativeState(Enum):
    """Creative process states based on Wallas's model + coherence dynamics"""
    PREPARATION = "preparation"
    INCUBATION = "incubation"
    ILLUMINATION = "illumination"
    VERIFICATION = "verification"
    EXPLORATION = "exploration"
    FLOW = "flow"
    BLOCKED = "blocked"
    TRANSITION = "transition"


@dataclass
class CreativeMetrics:
    """Metrics specific to creative work"""
    novelty_score: float  # 0-1, originality of output
    fluency_rate: float   # Ideas/concepts per minute
    flexibility_index: float  # Diversity of approaches
    elaboration_depth: float  # Detail and development level
    convergence_ratio: float  # Exploration vs refinement balance
    flow_intensity: float  # Depth of flow state
    breakthrough_probability: float  # Likelihood of creative breakthrough


@dataclass
class BiometricData:
    """Biometric indicators of creative state"""
    hrv: float  # Heart rate variability
    eeg_alpha: float  # Alpha wave activity (8-12 Hz)
    eeg_theta: float  # Theta wave activity (4-8 Hz)
    eeg_gamma: float  # Gamma wave activity (30-100 Hz)
    gsr: float  # Galvanic skin response
    eye_movement_entropy: float  # Randomness of eye movements
    posture_stability: float  # Physical stillness/movement


class CreativeFlowEngine(GCTEngine):
    """Extended GCT engine for creative process analysis"""
    
    def __init__(self):
        super().__init__()
        
        # Creative-specific parameters
        self.creative_params = {
            'flow_threshold': 0.75,
            'breakthrough_threshold': 0.15,  # dC/dt threshold
            'exploration_q_range': (0.5, 0.8),
            'incubation_rho_range': (0.6, 0.9),
            'flow_stability_window': 300  # seconds
        }
        
        # State transition rules based on coherence patterns
        self.state_rules = {
            CreativeState.EXPLORATION: {
                'psi': (0.3, 0.5),
                'q': (0.6, 0.8),
                'dc_dt': (-0.1, 0.1)
            },
            CreativeState.INCUBATION: {
                'rho': (0.7, 0.9),
                'q': (0.2, 0.4),
                'dc_dt': (-0.05, 0.05)
            },
            CreativeState.ILLUMINATION: {
                'psi': (0.8, 1.0),
                'dc_dt': (0.1, 1.0),
                'd2c_dt2': (0.05, 1.0)
            },
            CreativeState.FLOW: {
                'coherence': (0.75, 1.0),
                'dc_dt': (-0.02, 0.02),  # Stable
                'q_opt': (0.5, 0.7)
            },
            CreativeState.BLOCKED: {
                'coherence': (0, 0.3),
                'q': (0.8, 1.0),  # High frustration
                'psi': (0, 0.3)
            }
        }
        
        # History tracking for pattern analysis
        self.creative_history = deque(maxlen=1000)
        self.breakthrough_history = []
        self.flow_episodes = []
        
    def analyze_creative_state(self, 
                              variables: GCTVariables,
                              biometrics: Optional[BiometricData] = None,
                              creative_output: Optional[Dict] = None) -> Dict:
        """
        Comprehensive creative state analysis
        
        Args:
            variables: Standard GCT variables
            biometrics: Optional biometric data
            creative_output: Optional analysis of creative work
            
        Returns:
            Dictionary with creative state analysis
        """
        # Get base GCT analysis
        gct_result = self.analyze(variables)
        
        # Determine creative state
        creative_state = self._classify_creative_state(gct_result, biometrics)
        
        # Calculate creative metrics
        creative_metrics = self._calculate_creative_metrics(
            gct_result, biometrics, creative_output
        )
        
        # Detect breakthrough potential
        breakthrough_prob = self._detect_breakthrough_potential(
            gct_result, self.creative_history
        )
        creative_metrics.breakthrough_probability = breakthrough_prob
        
        # Flow state analysis
        flow_analysis = self._analyze_flow_state(gct_result, biometrics)
        
        # Generate recommendations
        recommendations = self._generate_creative_recommendations(
            creative_state, gct_result, creative_metrics
        )
        
        # Store in history
        analysis_result = {
            'timestamp': variables.timestamp,
            'gct_result': gct_result,
            'creative_state': creative_state,
            'creative_metrics': creative_metrics,
            'breakthrough_probability': breakthrough_prob,
            'flow_analysis': flow_analysis,
            'recommendations': recommendations
        }
        
        self.creative_history.append(analysis_result)
        
        # Check for breakthrough occurrence
        if breakthrough_prob > self.creative_params['breakthrough_threshold']:
            self.breakthrough_history.append(analysis_result)
        
        return analysis_result
    
    def _classify_creative_state(self, 
                                gct_result: GCTResult,
                                biometrics: Optional[BiometricData]) -> CreativeState:
        """Classify current creative state based on coherence patterns"""
        
        # Extract component values
        psi_value = gct_result.components.get('psi', 0)
        q_value = gct_result.components.get('q_raw', 0)
        rho_value = gct_result.components.get('rho', 0)
        
        # Check each state's criteria
        for state, criteria in self.state_rules.items():
            matches = True
            
            # Check coherence criteria
            if 'coherence' in criteria:
                min_c, max_c = criteria['coherence']
                if not (min_c <= gct_result.coherence <= max_c):
                    matches = False
            
            # Check component criteria
            if 'psi' in criteria:
                min_val, max_val = criteria['psi']
                if not (min_val <= psi_value <= max_val):
                    matches = False
                    
            if 'q' in criteria:
                min_val, max_val = criteria['q']
                if not (min_val <= q_value <= max_val):
                    matches = False
                    
            if 'rho' in criteria:
                min_val, max_val = criteria['rho']
                if not (min_val <= rho_value <= max_val):
                    matches = False
            
            # Check derivative criteria
            if 'dc_dt' in criteria:
                min_dc, max_dc = criteria['dc_dt']
                if not (min_dc <= gct_result.dc_dt <= max_dc):
                    matches = False
            
            if matches:
                return state
        
        # Use biometrics for additional classification if available
        if biometrics:
            if biometrics.eeg_theta > 0.7 and biometrics.eeg_alpha > 0.6:
                return CreativeState.FLOW
            elif biometrics.hrv < 0.3 and biometrics.gsr > 0.8:
                return CreativeState.BLOCKED
        
        return CreativeState.TRANSITION
    
    def _calculate_creative_metrics(self,
                                  gct_result: GCTResult,
                                  biometrics: Optional[BiometricData],
                                  creative_output: Optional[Dict]) -> CreativeMetrics:
        """Calculate creativity-specific metrics"""
        
        # Get component values
        psi_value = gct_result.components.get('psi', 0)
        q_value = gct_result.components.get('q_raw', 0)
        rho_value = gct_result.components.get('rho', 0)
        
        # Novelty from low psi + high q (exploration)
        novelty = (1 - psi_value) * q_value
        
        # Fluency from coherence stability
        recent_coherences = [h['gct_result'].coherence 
                           for h in list(self.creative_history)[-10:]]
        fluency = 1 - np.std(recent_coherences) if recent_coherences else 0.5
        
        # Flexibility from coherence variability over longer window
        long_coherences = [h['gct_result'].coherence 
                         for h in list(self.creative_history)[-50:]]
        flexibility = entropy(np.histogram(long_coherences, bins=10)[0] + 1e-10) / np.log(10) \
                     if len(long_coherences) > 20 else 0.5
        
        # Elaboration from rho (wisdom/depth)
        elaboration = rho_value
        
        # Convergence ratio from derivative direction
        convergence = 0.5 + (gct_result.dc_dt / 0.2)  # Positive = convergent
        convergence = max(0, min(1, convergence))
        
        # Flow intensity from coherence + stability
        flow_intensity = gct_result.coherence * fluency if gct_result.coherence > 0.7 else 0
        
        # Breakthrough probability (will be set by detect_breakthrough_potential)
        breakthrough_prob = 0.0
        
        # Enhance with biometrics if available
        if biometrics:
            # Alpha/theta ratio indicates creative state
            alpha_theta_ratio = biometrics.eeg_alpha / (biometrics.eeg_theta + 0.1)
            flow_intensity *= (1 + alpha_theta_ratio) / 2
            
            # Eye movement entropy indicates exploration
            novelty *= (1 + biometrics.eye_movement_entropy) / 2
        
        return CreativeMetrics(
            novelty_score=min(1.0, novelty),
            fluency_rate=fluency,
            flexibility_index=flexibility,
            elaboration_depth=min(1.0, elaboration),
            convergence_ratio=convergence,
            flow_intensity=min(1.0, flow_intensity),
            breakthrough_probability=breakthrough_prob
        )
    
    def _detect_breakthrough_potential(self, 
                                     gct_result: GCTResult,
                                     history: deque) -> float:
        """Detect potential for creative breakthrough"""
        
        if len(history) < 10:
            return 0.0
        
        # Pattern 1: Incubation followed by rapid rise
        recent_states = [h['creative_state'] for h in list(history)[-10:]]
        incubation_count = recent_states.count(CreativeState.INCUBATION)
        
        # Pattern 2: High acceleration
        acceleration_factor = abs(gct_result.d2c_dt2) * 5
        
        # Pattern 3: Component divergence then convergence
        recent_components = [h['gct_result'].components for h in list(history)[-5:]]
        if len(recent_components) >= 2:
            component_lists = []
            for c in recent_components:
                component_lists.append([
                    c.get('psi', 0), 
                    c.get('q_raw', 0),
                    c.get('rho', 0), 
                    c.get('f', 0)
                ])
            component_variance = np.var(component_lists)
        else:
            component_variance = 0
        
        # Pattern 4: Unusual coherence after low period
        if len(history) >= 20:
            old_coherence = np.mean([h['gct_result'].coherence 
                                   for h in list(history)[-20:-10]])
            coherence_jump = gct_result.coherence - old_coherence
        else:
            coherence_jump = 0
        
        # Combine factors
        breakthrough_prob = (
            (incubation_count / 10) * 0.3 +
            min(1.0, acceleration_factor) * 0.3 +
            min(1.0, component_variance) * 0.2 +
            max(0, min(1.0, coherence_jump * 2)) * 0.2
        )
        
        return breakthrough_prob
    
    def _analyze_flow_state(self,
                          gct_result: GCTResult,
                          biometrics: Optional[BiometricData]) -> Dict:
        """Detailed flow state analysis"""
        
        flow_indicators = {
            'coherence_level': gct_result.coherence > self.creative_params['flow_threshold'],
            'stability': abs(gct_result.dc_dt) < 0.02,
            'optimal_activation': 0.5 <= gct_result.q_opt <= 0.7,
            'balanced_components': np.std([
                gct_result.components.get('psi', 0),
                gct_result.components.get('q_raw', 0),
                gct_result.components.get('rho', 0),
                gct_result.components.get('f', 0)
            ]) < 0.2
        }
        
        # Biometric indicators
        if biometrics:
            flow_indicators.update({
                'alpha_theta_sync': abs(biometrics.eeg_alpha - biometrics.eeg_theta) < 0.2,
                'optimal_arousal': 0.4 <= biometrics.hrv <= 0.7,
                'focused_attention': biometrics.eye_movement_entropy < 0.3,
                'relaxed_body': biometrics.posture_stability > 0.7
            })
        
        # Calculate flow score
        flow_score = sum(flow_indicators.values()) / len(flow_indicators)
        
        # Check for flow episode
        in_flow = flow_score > 0.7
        
        # Track flow duration if in flow
        if in_flow and self.flow_episodes:
            last_episode = self.flow_episodes[-1]
            if not last_episode.get('ended'):
                # Continue current episode
                last_episode['duration'] = (
                    datetime.now() - last_episode['start_time']
                ).total_seconds()
            else:
                # Start new episode
                self.flow_episodes.append({
                    'start_time': datetime.now(),
                    'duration': 0,
                    'ended': False
                })
        elif in_flow:
            # Start first episode
            self.flow_episodes.append({
                'start_time': datetime.now(),
                'duration': 0,
                'ended': False
            })
        elif not in_flow and self.flow_episodes and not self.flow_episodes[-1].get('ended'):
            # End current episode
            self.flow_episodes[-1]['ended'] = True
        
        return {
            'in_flow': in_flow,
            'flow_score': flow_score,
            'flow_indicators': flow_indicators,
            'current_episode_duration': (
                self.flow_episodes[-1]['duration'] 
                if self.flow_episodes and not self.flow_episodes[-1].get('ended')
                else 0
            )
        }
    
    def _generate_creative_recommendations(self,
                                         state: CreativeState,
                                         gct_result: GCTResult,
                                         metrics: CreativeMetrics) -> List[Dict]:
        """Generate personalized recommendations for enhancing creativity"""
        
        recommendations = []
        
        # Get component values
        psi_value = gct_result.components.get('psi', 0)
        q_value = gct_result.components.get('q_raw', 0)
        rho_value = gct_result.components.get('rho', 0)
        f_value = gct_result.components.get('f', 0)
        
        # State-specific recommendations
        if state == CreativeState.BLOCKED:
            if psi_value < 0.3:  # Low clarity
                recommendations.append({
                    'type': 'clarity_boost',
                    'action': 'Take a 10-minute walk or do a simple organizational task',
                    'rationale': 'Physical movement or simple tasks can restore mental clarity',
                    'urgency': 'high'
                })
            if q_value > 0.8:  # High emotional charge
                recommendations.append({
                    'type': 'emotional_regulation',
                    'action': 'Practice 5 minutes of deep breathing or progressive relaxation',
                    'rationale': 'Reducing emotional intensity can unblock creative flow',
                    'urgency': 'high'
                })
        
        elif state == CreativeState.EXPLORATION:
            if metrics.convergence_ratio > 0.7:  # Too convergent
                recommendations.append({
                    'type': 'divergence_prompt',
                    'action': 'Try random word association or explore unrelated domains',
                    'rationale': 'Increasing divergent thinking can lead to novel connections',
                    'urgency': 'medium'
                })
        
        elif state == CreativeState.FLOW:
            recommendations.append({
                'type': 'flow_maintenance',
                'action': 'Continue current activity, minimize interruptions',
                'rationale': f'You\'ve been in flow for {metrics.flow_intensity:.0f} minutes',
                'urgency': 'low'
            })
        
        # Breakthrough recommendations
        if metrics.breakthrough_probability > 0.5:
            recommendations.append({
                'type': 'breakthrough_preparation',
                'action': 'Have capture tools ready, stay open to unexpected connections',
                'rationale': f'Breakthrough probability: {metrics.breakthrough_probability:.1%}',
                'urgency': 'medium'
            })
        
        # General coherence optimization
        if gct_result.coherence < 0.5:
            component_values = {
                'clarity': psi_value,
                'emotion': q_value,
                'wisdom': rho_value,
                'social': f_value
            }
            
            weakest = min(component_values, key=component_values.get)
            
            component_boosts = {
                'clarity': {
                    'action': 'Organize your workspace or create a clear problem statement',
                    'rationale': 'Improving environmental and mental clarity'
                },
                'emotion': {
                    'action': 'Connect with why this creative work matters to you',
                    'rationale': 'Reconnecting with intrinsic motivation'
                },
                'wisdom': {
                    'action': 'Review past successful projects or seek expert input',
                    'rationale': 'Drawing on accumulated knowledge and experience'
                },
                'social': {
                    'action': 'Share your work-in-progress or collaborate with others',
                    'rationale': 'Social connection can energize creative work'
                }
            }
            
            if weakest in component_boosts:
                recommendations.append({
                    'type': f'{weakest}_enhancement',
                    'action': component_boosts[weakest]['action'],
                    'rationale': component_boosts[weakest]['rationale'],
                    'urgency': 'medium'
                })
        
        return recommendations
    
    def predict_creative_trajectory(self, 
                                   current_state: Dict,
                                   horizon_minutes: int = 30) -> Dict:
        """Predict future creative states and recommend interventions"""
        
        if len(self.creative_history) < 10:
            return {'prediction': 'insufficient_data'}
        
        # Analyze recent patterns
        recent_states = [h['creative_state'] for h in list(self.creative_history)[-20:]]
        recent_coherences = [h['gct_result'].coherence for h in list(self.creative_history)[-20:]]
        
        # Fit simple trend
        x = np.arange(len(recent_coherences))
        z = np.polyfit(x, recent_coherences, 2)
        p = np.poly1d(z)
        
        # Project forward
        future_points = horizon_minutes // 5  # Assume 5-minute intervals
        future_x = np.arange(len(recent_coherences), len(recent_coherences) + future_points)
        predicted_coherences = p(future_x)
        
        # Predict state transitions
        predicted_states = []
        for pred_coherence in predicted_coherences:
            if pred_coherence > 0.75:
                predicted_states.append(CreativeState.FLOW)
            elif pred_coherence < 0.3:
                predicted_states.append(CreativeState.BLOCKED)
            elif abs(p.deriv()(future_x[0])) > 0.02:
                predicted_states.append(CreativeState.TRANSITION)
            else:
                predicted_states.append(current_state['creative_state'])
        
        # Identify intervention points
        interventions = []
        for i, (current, future) in enumerate(zip(predicted_states[:-1], predicted_states[1:])):
            if current != future:
                if future == CreativeState.BLOCKED:
                    interventions.append({
                        'time': (i + 1) * 5,
                        'type': 'prevent_block',
                        'suggestion': 'Take a break or switch creative modes'
                    })
                elif future == CreativeState.FLOW:
                    interventions.append({
                        'time': (i + 1) * 5,
                        'type': 'prepare_flow',
                        'suggestion': 'Clear distractions and prepare for deep work'
                    })
        
        return {
            'predicted_trajectory': list(zip(future_x * 5, predicted_coherences)),
            'predicted_states': predicted_states,
            'recommended_interventions': interventions,
            'confidence': 0.7 if len(self.creative_history) > 50 else 0.5
        }


class CreativeEnvironmentOptimizer:
    """Optimize physical and digital environment for creativity"""
    
    def __init__(self):
        self.environment_profiles = {
            CreativeState.EXPLORATION: {
                'lighting': {'temperature': 5000, 'brightness': 0.7},  # Cool, bright
                'sound': {'type': 'ambient', 'volume': 0.3, 'variation': 'high'},
                'visual': {'complexity': 'high', 'color_variety': 'high'},
                'interruptions': 'allowed',
                'tools': ['whiteboard', 'random_stimuli', 'multiple_media']
            },
            CreativeState.FLOW: {
                'lighting': {'temperature': 4000, 'brightness': 0.6},  # Neutral, moderate
                'sound': {'type': 'binaural', 'frequency': 'theta', 'volume': 0.2},
                'visual': {'complexity': 'minimal', 'color_variety': 'low'},
                'interruptions': 'blocked',
                'tools': ['focused_workspace', 'single_medium']
            },
            CreativeState.INCUBATION: {
                'lighting': {'temperature': 3000, 'brightness': 0.4},  # Warm, dim
                'sound': {'type': 'nature', 'volume': 0.2, 'variation': 'low'},
                'visual': {'complexity': 'medium', 'color_variety': 'medium'},
                'interruptions': 'minimal',
                'tools': ['notebook', 'comfortable_seating']
            }
        }
    
    def optimize_environment(self, current_state: CreativeState, 
                           target_state: Optional[CreativeState] = None) -> Dict:
        """Generate environment optimization recommendations"""
        
        if target_state:
            profile = self.environment_profiles.get(target_state, {})
            transition_time = self._estimate_transition_time(current_state, target_state)
        else:
            profile = self.environment_profiles.get(current_state, {})
            transition_time = 0
        
        recommendations = {
            'immediate_changes': [],
            'gradual_changes': [],
            'transition_time': transition_time
        }
        
        # Immediate environmental changes
        if 'lighting' in profile:
            recommendations['immediate_changes'].append({
                'type': 'lighting',
                'settings': profile['lighting'],
                'action': f"Adjust lighting to {profile['lighting']['temperature']}K "
                         f"at {profile['lighting']['brightness']*100}% brightness"
            })
        
        if 'sound' in profile:
            recommendations['immediate_changes'].append({
                'type': 'sound',
                'settings': profile['sound'],
                'action': f"Play {profile['sound']['type']} sounds at "
                         f"{profile['sound']['volume']*100}% volume"
            })
        
        # Gradual changes for state transition
        if target_state and current_state != target_state:
            recommendations['gradual_changes'] = self._plan_transition(
                current_state, target_state, transition_time
            )
        
        return recommendations
    
    def _estimate_transition_time(self, 
                                current_state: CreativeState,
                                target_state: CreativeState) -> int:
        """Estimate time needed for state transition"""
        
        transition_times = {
            (CreativeState.BLOCKED, CreativeState.EXPLORATION): 15,
            (CreativeState.EXPLORATION, CreativeState.FLOW): 10,
            (CreativeState.FLOW, CreativeState.VERIFICATION): 5,
            (CreativeState.INCUBATION, CreativeState.ILLUMINATION): 20,
        }
        
        return transition_times.get((current_state, target_state), 10)
    
    def _plan_transition(self,
                        current_state: CreativeState,
                        target_state: CreativeState,
                        duration_minutes: int) -> List[Dict]:
        """Plan gradual transition between states"""
        
        steps = []
        
        # Example transition plan
        if current_state == CreativeState.BLOCKED and target_state == CreativeState.EXPLORATION:
            steps = [
                {
                    'time': 0,
                    'action': 'Stand up and stretch for 2 minutes',
                    'rationale': 'Physical movement breaks mental patterns'
                },
                {
                    'time': 2,
                    'action': 'Change physical location if possible',
                    'rationale': 'New environment stimulates new thinking'
                },
                {
                    'time': 5,
                    'action': 'Browse unrelated creative work for inspiration',
                    'rationale': 'External stimuli can trigger new associations'
                },
                {
                    'time': 10,
                    'action': 'Start with small, playful creative exercises',
                    'rationale': 'Low-pressure activities rebuild creative confidence'
                }
            ]
        
        return steps