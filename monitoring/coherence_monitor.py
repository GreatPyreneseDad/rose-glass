#!/usr/bin/env python3
"""
GCT Performance Monitoring Module
Tracks model performance, drift, and anomalies in production
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from prometheus_client import Counter, Histogram, Gauge, Summary
from dataclasses import dataclass, asdict
import json
import asyncio
from collections import deque


# Prometheus metrics
COHERENCE_CALCULATIONS = Counter(
    'gct_coherence_calculations_total',
    'Total number of coherence calculations',
    ['entity_type', 'source']
)

COHERENCE_VALUES = Histogram(
    'gct_coherence_values',
    'Distribution of coherence values',
    ['entity_type'],
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
)

PROCESSING_TIME = Summary(
    'gct_processing_time_seconds',
    'Time spent processing coherence calculations',
    ['operation']
)

MODEL_DRIFT = Gauge(
    'gct_model_drift',
    'Measure of model prediction drift',
    ['model_component']
)

ANOMALY_ALERTS = Counter(
    'gct_anomaly_alerts_total',
    'Total anomaly alerts generated',
    ['alert_type', 'severity']
)

ERROR_RATE = Counter(
    'gct_errors_total',
    'Total errors encountered',
    ['error_type', 'component']
)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics"""
    timestamp: datetime
    avg_processing_time: float
    total_calculations: int
    error_rate: float
    coherence_distribution: Dict[str, float]
    drift_scores: Dict[str, float]
    anomaly_count: int


@dataclass
class DriftAlert:
    """Alert for model drift detection"""
    timestamp: datetime
    component: str
    drift_score: float
    baseline_performance: float
    current_performance: float
    severity: str
    recommendation: str


class CoherenceMonitor:
    """Monitor GCT system performance and health"""
    
    def __init__(self, 
                 baseline_window_days: int = 30,
                 drift_threshold: float = 0.1,
                 anomaly_threshold: float = 3.0):
        self.baseline_window = timedelta(days=baseline_window_days)
        self.drift_threshold = drift_threshold
        self.anomaly_threshold = anomaly_threshold
        
        # Performance tracking
        self.calculation_history = deque(maxlen=10000)
        self.error_history = deque(maxlen=1000)
        self.coherence_baselines = {}
        
        # Drift detection
        self.performance_baselines = {
            'psi_extractor': deque(maxlen=1000),
            'rho_extractor': deque(maxlen=1000),
            'q_extractor': deque(maxlen=1000),
            'f_extractor': deque(maxlen=1000),
            'coherence_engine': deque(maxlen=1000)
        }
        
        self.logger = logging.getLogger(__name__)
        
    def track_calculation(self, 
                         entity_type: str,
                         entity_id: str,
                         coherence: float,
                         processing_time: float,
                         source: str = "api"):
        """Track a coherence calculation"""
        # Update metrics
        COHERENCE_CALCULATIONS.labels(entity_type=entity_type, source=source).inc()
        COHERENCE_VALUES.labels(entity_type=entity_type).observe(coherence)
        PROCESSING_TIME.labels(operation="calculate_coherence").observe(processing_time)
        
        # Store in history
        self.calculation_history.append({
            'timestamp': datetime.now(),
            'entity_type': entity_type,
            'entity_id': entity_id,
            'coherence': coherence,
            'processing_time': processing_time,
            'source': source
        })
        
        # Check for anomalies
        if self.detect_anomaly(entity_type, coherence):
            self._handle_anomaly(entity_type, entity_id, coherence)
    
    def track_error(self, error_type: str, component: str, error_message: str):
        """Track system errors"""
        ERROR_RATE.labels(error_type=error_type, component=component).inc()
        
        self.error_history.append({
            'timestamp': datetime.now(),
            'error_type': error_type,
            'component': component,
            'message': error_message
        })
        
        self.logger.error(f"Error in {component}: {error_type} - {error_message}")
    
    def track_component_performance(self, 
                                  component: str,
                                  performance_score: float):
        """Track individual component performance for drift detection"""
        if component in self.performance_baselines:
            self.performance_baselines[component].append({
                'timestamp': datetime.now(),
                'score': performance_score
            })
            
            # Check for drift
            drift_score = self._calculate_drift(component)
            MODEL_DRIFT.labels(model_component=component).set(drift_score)
            
            if drift_score > self.drift_threshold:
                self._handle_drift(component, drift_score)
    
    def detect_anomaly(self, entity_type: str, coherence: float) -> bool:
        """Detect if coherence value is anomalous"""
        if entity_type not in self.coherence_baselines:
            self._update_baseline(entity_type)
            return False
        
        baseline = self.coherence_baselines[entity_type]
        if baseline['count'] < 100:  # Need sufficient data
            return False
        
        # Calculate z-score
        z_score = abs(coherence - baseline['mean']) / baseline['std']
        
        return z_score > self.anomaly_threshold
    
    def _calculate_drift(self, component: str) -> float:
        """Calculate drift score for a component"""
        history = list(self.performance_baselines[component])
        
        if len(history) < 100:
            return 0.0
        
        # Split into old and recent
        midpoint = len(history) // 2
        old_scores = [h['score'] for h in history[:midpoint]]
        recent_scores = [h['score'] for h in history[midpoint:]]
        
        # Calculate drift using KL divergence approximation
        old_mean = np.mean(old_scores)
        recent_mean = np.mean(recent_scores)
        
        if old_mean == 0:
            return 0.0
        
        drift = abs(recent_mean - old_mean) / old_mean
        
        return drift
    
    def _update_baseline(self, entity_type: str):
        """Update coherence baseline for entity type"""
        recent_calcs = [
            calc for calc in self.calculation_history
            if calc['entity_type'] == entity_type and
            calc['timestamp'] > datetime.now() - self.baseline_window
        ]
        
        if len(recent_calcs) < 30:
            return
        
        coherences = [calc['coherence'] for calc in recent_calcs]
        
        self.coherence_baselines[entity_type] = {
            'mean': np.mean(coherences),
            'std': np.std(coherences),
            'median': np.median(coherences),
            'count': len(coherences),
            'updated': datetime.now()
        }
    
    def _handle_anomaly(self, entity_type: str, entity_id: str, coherence: float):
        """Handle detected anomaly"""
        severity = 'warning' if coherence > 0.1 else 'critical'
        
        ANOMALY_ALERTS.labels(
            alert_type='coherence_anomaly',
            severity=severity
        ).inc()
        
        self.logger.warning(
            f"Coherence anomaly detected - Type: {entity_type}, "
            f"ID: {entity_id}, Value: {coherence:.3f}"
        )
    
    def _handle_drift(self, component: str, drift_score: float):
        """Handle detected drift"""
        history = list(self.performance_baselines[component])
        recent_performance = np.mean([h['score'] for h in history[-50:]])
        baseline_performance = np.mean([h['score'] for h in history[:50]])
        
        alert = DriftAlert(
            timestamp=datetime.now(),
            component=component,
            drift_score=drift_score,
            baseline_performance=baseline_performance,
            current_performance=recent_performance,
            severity='high' if drift_score > 0.2 else 'medium',
            recommendation=self._get_drift_recommendation(component, drift_score)
        )
        
        ANOMALY_ALERTS.labels(
            alert_type='model_drift',
            severity=alert.severity
        ).inc()
        
        self.logger.warning(f"Model drift detected: {asdict(alert)}")
    
    def _get_drift_recommendation(self, component: str, drift_score: float) -> str:
        """Get recommendation for handling drift"""
        if drift_score > 0.3:
            return f"Urgent: Retrain {component} with recent data"
        elif drift_score > 0.2:
            return f"Review {component} performance and consider retraining"
        else:
            return f"Monitor {component} closely for continued drift"
    
    def get_performance_report(self) -> PerformanceMetrics:
        """Generate performance report"""
        recent_calcs = [
            calc for calc in self.calculation_history
            if calc['timestamp'] > datetime.now() - timedelta(hours=1)
        ]
        
        if not recent_calcs:
            return PerformanceMetrics(
                timestamp=datetime.now(),
                avg_processing_time=0.0,
                total_calculations=0,
                error_rate=0.0,
                coherence_distribution={},
                drift_scores={},
                anomaly_count=0
            )
        
        # Calculate metrics
        processing_times = [calc['processing_time'] for calc in recent_calcs]
        coherences = [calc['coherence'] for calc in recent_calcs]
        
        # Error rate
        recent_errors = [
            err for err in self.error_history
            if err['timestamp'] > datetime.now() - timedelta(hours=1)
        ]
        error_rate = len(recent_errors) / max(len(recent_calcs), 1)
        
        # Coherence distribution
        coherence_dist = {
            'p10': np.percentile(coherences, 10),
            'p25': np.percentile(coherences, 25),
            'p50': np.percentile(coherences, 50),
            'p75': np.percentile(coherences, 75),
            'p90': np.percentile(coherences, 90)
        }
        
        # Drift scores
        drift_scores = {}
        for component in self.performance_baselines:
            drift_scores[component] = self._calculate_drift(component)
        
        return PerformanceMetrics(
            timestamp=datetime.now(),
            avg_processing_time=np.mean(processing_times),
            total_calculations=len(recent_calcs),
            error_rate=error_rate,
            coherence_distribution=coherence_dist,
            drift_scores=drift_scores,
            anomaly_count=sum(1 for calc in recent_calcs 
                            if self.detect_anomaly(calc['entity_type'], calc['coherence']))
        )
    
    async def continuous_monitoring(self, interval_seconds: int = 300):
        """Run continuous monitoring loop"""
        while True:
            try:
                # Update baselines
                for entity_type in set(calc['entity_type'] 
                                     for calc in self.calculation_history):
                    self._update_baseline(entity_type)
                
                # Generate report
                report = self.get_performance_report()
                
                # Log summary
                self.logger.info(
                    f"Performance Report - Calculations: {report.total_calculations}, "
                    f"Avg Time: {report.avg_processing_time:.3f}s, "
                    f"Error Rate: {report.error_rate:.2%}"
                )
                
                # Check system health
                if report.error_rate > 0.05:
                    self.logger.warning("High error rate detected")
                
                if any(score > self.drift_threshold 
                      for score in report.drift_scores.values()):
                    self.logger.warning("Model drift detected in one or more components")
                
                await asyncio.sleep(interval_seconds)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(interval_seconds)


def setup_monitoring() -> CoherenceMonitor:
    """Set up monitoring instance"""
    monitor = CoherenceMonitor(
        baseline_window_days=30,
        drift_threshold=0.1,
        anomaly_threshold=3.0
    )
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    return monitor


# Example usage
if __name__ == "__main__":
    monitor = setup_monitoring()
    
    # Simulate some calculations
    import random
    
    for i in range(100):
        monitor.track_calculation(
            entity_type="stock",
            entity_id=f"STOCK_{i % 10}",
            coherence=random.gauss(0.5, 0.1),
            processing_time=random.gauss(0.1, 0.02)
        )
    
    # Get report
    report = monitor.get_performance_report()
    print(f"Performance Report: {asdict(report)}")
    
    # Run continuous monitoring
    # asyncio.run(monitor.continuous_monitoring())