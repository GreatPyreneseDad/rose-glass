#!/usr/bin/env python3
"""
Soul Coherence Tracker - Monitors and tracks soul coherence state (Ψ)
Implements real-time coherence calculations and stability monitoring
"""
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import numpy as np
from enum import Enum


class CoherenceState(Enum):
    """Coherence state classifications"""
    FRAGMENTED = "fragmented"      # Ψ < 0.3
    UNSTABLE = "unstable"          # 0.3 ≤ Ψ < 0.5
    EMERGING = "emerging"          # 0.5 ≤ Ψ < 0.7
    STABLE = "stable"              # 0.7 ≤ Ψ < 0.9
    HARMONIZED = "harmonized"      # Ψ ≥ 0.9


@dataclass
class CoherenceEvent:
    """Represents a coherence-affecting event"""
    timestamp: datetime
    event_type: str
    delta_psi: float
    trigger: str
    notes: Optional[str] = None


@dataclass
class CoherenceSnapshot:
    """Point-in-time coherence state"""
    timestamp: datetime
    psi_value: float
    state: CoherenceState
    stability_score: float
    recent_delta: float
    resonance_frequency: float = 0.0


class CoherenceTracker:
    """
    Tracks soul coherence (Ψ) over time.
    Monitors stability, detects patterns, and provides warnings.
    """
    
    def __init__(self, initial_psi: float = 1.0):
        self.current_psi: float = initial_psi
        self.history: deque = deque(maxlen=1000)  # Rolling history
        self.events: List[CoherenceEvent] = []
        self.snapshots: List[CoherenceSnapshot] = []
        self.stability_window: int = 10  # Events to consider for stability
        self.warning_threshold: float = 0.3  # Warn if Ψ drops below
        self.critical_threshold: float = 0.15  # Critical state threshold
        
        # Take initial snapshot
        self._take_snapshot()
        
    def update_coherence(self, delta_psi: float, event_type: str, 
                        trigger: str, notes: Optional[str] = None) -> CoherenceSnapshot:
        """
        Update coherence state with a delta change.
        
        Args:
            delta_psi: Change in coherence (-1.0 to 1.0)
            event_type: Type of event causing change
            trigger: What triggered the event
            notes: Optional additional context
            
        Returns:
            Current coherence snapshot after update
        """
        # Record event
        event = CoherenceEvent(
            timestamp=datetime.now(),
            event_type=event_type,
            delta_psi=delta_psi,
            trigger=trigger,
            notes=notes
        )
        self.events.append(event)
        
        # Update coherence with bounds
        old_psi = self.current_psi
        self.current_psi = max(0.0, min(2.0, self.current_psi + delta_psi))
        
        # Record in history
        self.history.append({
            'timestamp': event.timestamp,
            'psi': self.current_psi,
            'delta': delta_psi,
            'event_type': event_type
        })
        
        # Check for warnings
        self._check_coherence_warnings(old_psi, self.current_psi)
        
        # Take snapshot
        return self._take_snapshot()
        
    def _take_snapshot(self) -> CoherenceSnapshot:
        """Create and store a coherence snapshot."""
        snapshot = CoherenceSnapshot(
            timestamp=datetime.now(),
            psi_value=self.current_psi,
            state=self._classify_state(self.current_psi),
            stability_score=self._calculate_stability(),
            recent_delta=self._calculate_recent_delta(),
            resonance_frequency=self._calculate_resonance()
        )
        self.snapshots.append(snapshot)
        return snapshot
        
    def _classify_state(self, psi: float) -> CoherenceState:
        """Classify coherence value into state category."""
        if psi < 0.3:
            return CoherenceState.FRAGMENTED
        elif psi < 0.5:
            return CoherenceState.UNSTABLE
        elif psi < 0.7:
            return CoherenceState.EMERGING
        elif psi < 0.9:
            return CoherenceState.STABLE
        else:
            return CoherenceState.HARMONIZED
            
    def _calculate_stability(self) -> float:
        """
        Calculate stability score based on recent variance.
        Returns 0.0 (chaotic) to 1.0 (perfectly stable).
        """
        if len(self.history) < 3:
            return 0.5  # Neutral if insufficient data
            
        recent_values = [h['psi'] for h in list(self.history)[-self.stability_window:]]
        
        if len(recent_values) < 2:
            return 0.5
            
        # Calculate variance-based stability
        variance = np.var(recent_values)
        # Convert variance to stability score (inverse relationship)
        stability = 1.0 / (1.0 + variance * 10)
        
        return stability
        
    def _calculate_recent_delta(self) -> float:
        """Calculate net coherence change over recent window."""
        if len(self.history) < 2:
            return 0.0
            
        recent = list(self.history)[-self.stability_window:]
        if recent:
            return sum(h['delta'] for h in recent)
        return 0.0
        
    def _calculate_resonance(self) -> float:
        """
        Calculate resonance frequency - how often coherence oscillates
        around a stable point. Higher = more resonant/stable.
        """
        if len(self.history) < 10:
            return 0.0
            
        recent_values = [h['psi'] for h in list(self.history)[-20:]]
        
        # Find oscillation frequency using zero-crossing method
        mean_psi = np.mean(recent_values)
        centered = [v - mean_psi for v in recent_values]
        
        # Count zero crossings
        zero_crossings = 0
        for i in range(1, len(centered)):
            if centered[i-1] * centered[i] < 0:
                zero_crossings += 1
                
        # Normalize to frequency
        resonance = zero_crossings / len(centered)
        
        # Stable resonance is moderate frequency (not too high, not too low)
        optimal_resonance = 0.3
        return 1.0 - abs(resonance - optimal_resonance) / optimal_resonance
        
    def _check_coherence_warnings(self, old_psi: float, new_psi: float):
        """Check for coherence warning conditions."""
        warnings = []
        
        # Rapid drop warning
        if old_psi - new_psi > 0.2:
            warnings.append({
                'type': 'rapid_drop',
                'message': f'Rapid coherence drop: {old_psi:.2f} → {new_psi:.2f}',
                'severity': 'high'
            })
            
        # Threshold warnings
        if new_psi < self.critical_threshold:
            warnings.append({
                'type': 'critical_low',
                'message': f'CRITICAL: Coherence at {new_psi:.2f}',
                'severity': 'critical'
            })
        elif new_psi < self.warning_threshold:
            warnings.append({
                'type': 'low_coherence',
                'message': f'Warning: Low coherence at {new_psi:.2f}',
                'severity': 'medium'
            })
            
        # Instability warning
        if self._calculate_stability() < 0.3:
            warnings.append({
                'type': 'unstable',
                'message': 'Coherence state highly unstable',
                'severity': 'medium'
            })
            
        return warnings
        
    def get_coherence_report(self) -> Dict:
        """Generate comprehensive coherence report."""
        current_snapshot = self.snapshots[-1] if self.snapshots else None
        
        # Calculate trends
        trend = "stable"
        if len(self.snapshots) > 1:
            recent_delta = self.snapshots[-1].psi_value - self.snapshots[-10].psi_value if len(self.snapshots) > 10 else self.snapshots[-1].psi_value - self.snapshots[0].psi_value
            if recent_delta > 0.1:
                trend = "ascending"
            elif recent_delta < -0.1:
                trend = "descending"
                
        # Find peak and trough
        if self.snapshots:
            peak = max(self.snapshots, key=lambda s: s.psi_value)
            trough = min(self.snapshots, key=lambda s: s.psi_value)
        else:
            peak = trough = None
            
        return {
            'current_psi': self.current_psi,
            'current_state': current_snapshot.state.value if current_snapshot else None,
            'stability_score': current_snapshot.stability_score if current_snapshot else 0.0,
            'trend': trend,
            'recent_events': len(self.events),
            'peak_coherence': peak.psi_value if peak else self.current_psi,
            'lowest_coherence': trough.psi_value if trough else self.current_psi,
            'resonance_frequency': current_snapshot.resonance_frequency if current_snapshot else 0.0,
            'warnings': self._check_coherence_warnings(self.current_psi, self.current_psi)
        }
        
    def predict_trajectory(self, steps: int = 5) -> List[float]:
        """
        Predict future coherence trajectory based on recent patterns.
        Simple linear extrapolation for now.
        """
        if len(self.history) < 3:
            return [self.current_psi] * steps
            
        # Calculate recent trend
        recent = list(self.history)[-10:]
        times = list(range(len(recent)))
        values = [h['psi'] for h in recent]
        
        # Simple linear regression
        z = np.polyfit(times, values, 1)
        slope = z[0]
        
        # Project forward
        predictions = []
        future_psi = self.current_psi
        for i in range(steps):
            future_psi = max(0.0, min(2.0, future_psi + slope))
            predictions.append(future_psi)
            
        return predictions
        
    def export_data(self) -> Dict:
        """Export all coherence data for analysis."""
        return {
            'current_psi': self.current_psi,
            'total_events': len(self.events),
            'snapshots': [
                {
                    'timestamp': s.timestamp.isoformat(),
                    'psi': s.psi_value,
                    'state': s.state.value,
                    'stability': s.stability_score
                }
                for s in self.snapshots
            ],
            'events': [
                {
                    'timestamp': e.timestamp.isoformat(),
                    'type': e.event_type,
                    'delta_psi': e.delta_psi,
                    'trigger': e.trigger
                }
                for e in self.events
            ],
            'report': self.get_coherence_report()
        }