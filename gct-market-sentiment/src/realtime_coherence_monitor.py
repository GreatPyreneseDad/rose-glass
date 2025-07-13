#!/usr/bin/env python3
"""
Real-Time Coherence Monitoring System
Implements WebSocket streaming and anomaly detection for market coherence
"""

import asyncio
import websockets
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, AsyncGenerator, Optional, Callable
import numpy as np
from collections import deque
import aioredis
from dataclasses import dataclass, asdict
import pandas as pd


@dataclass
class CoherenceUpdate:
    """Real-time coherence update"""
    timestamp: datetime
    entity_id: str
    entity_type: str  # 'stock', 'sector', 'market'
    coherence: float
    psi: float
    rho: float
    q: float
    f: float
    truth_cost: float
    emotion: str
    anomaly_score: float
    metadata: Dict


@dataclass
class CoherenceAlert:
    """Alert for coherence anomalies"""
    timestamp: datetime
    entity_id: str
    alert_type: str  # 'drift', 'spike', 'breakdown', 'emergence'
    severity: str    # 'info', 'warning', 'critical'
    current_coherence: float
    expected_coherence: float
    deviation: float
    message: str
    recommended_action: Optional[str] = None


class RealTimeCoherenceMonitor:
    """Real-time coherence monitoring with anomaly detection"""
    
    def __init__(self, 
                 ws_url: str,
                 redis_url: str = "redis://localhost:6379",
                 alert_callback: Optional[Callable] = None):
        self.ws_url = ws_url
        self.redis_url = redis_url
        self.alert_callback = alert_callback
        self.logger = logging.getLogger(__name__)
        
        # Time series storage for anomaly detection
        self.coherence_history = {}  # entity_id -> deque of CoherenceUpdate
        self.history_window = 100     # Keep last 100 updates per entity
        
        # Anomaly detection parameters
        self.drift_threshold = 0.15   # Coherence drift threshold
        self.spike_threshold = 3.0    # Standard deviations for spike
        self.breakdown_threshold = 0.3 # Coherence below this is breakdown
        
        # Redis for distributed state
        self.redis = None
        
        # Performance metrics
        self.metrics = {
            'updates_processed': 0,
            'alerts_generated': 0,
            'processing_time_ms': deque(maxlen=1000)
        }
    
    async def initialize(self):
        """Initialize connections and resources"""
        # Connect to Redis
        self.redis = await aioredis.create_redis_pool(self.redis_url)
        
        # Load historical baselines
        await self._load_baselines()
        
        self.logger.info("Real-time monitor initialized")
    
    async def stream_coherence_updates(self, 
                                     entities: List[str],
                                     entity_type: str = 'stock') -> AsyncGenerator[CoherenceUpdate, None]:
        """
        Stream real-time coherence updates via WebSocket
        
        Args:
            entities: List of entity IDs to monitor
            entity_type: Type of entities (stock, sector, etc.)
            
        Yields:
            CoherenceUpdate objects as they arrive
        """
        async with websockets.connect(self.ws_url) as ws:
            # Subscribe to entities
            await ws.send(json.dumps({
                'action': 'subscribe',
                'entities': entities,
                'type': entity_type
            }))
            
            while True:
                try:
                    # Receive update
                    message = await asyncio.wait_for(ws.recv(), timeout=30.0)
                    data = json.loads(message)
                    
                    # Process update
                    update = await self._process_update(data)
                    
                    # Check for anomalies
                    alerts = await self._detect_anomalies(update)
                    
                    # Handle alerts
                    for alert in alerts:
                        await self._handle_alert(alert)
                    
                    # Update metrics
                    self.metrics['updates_processed'] += 1
                    
                    yield update
                    
                except asyncio.TimeoutError:
                    # Send heartbeat
                    await ws.send(json.dumps({'action': 'heartbeat'}))
                    
                except websockets.ConnectionClosed:
                    self.logger.warning("WebSocket connection closed, reconnecting...")
                    await asyncio.sleep(5)
                    break
                    
                except Exception as e:
                    self.logger.error(f"Error processing update: {e}")
                    continue
    
    async def monitor_market_coherence(self) -> AsyncGenerator[Dict, None]:
        """
        Monitor overall market coherence with sector breakdowns
        
        Yields:
            Market-wide coherence summaries
        """
        sectors = ['Technology', 'Healthcare', 'Finance', 'Energy', 
                  'Consumer', 'Industrial', 'Materials', 'Utilities']
        
        while True:
            try:
                # Get sector coherences from Redis
                sector_coherences = {}
                for sector in sectors:
                    key = f"coherence:sector:{sector}"
                    value = await self.redis.get(key)
                    if value:
                        sector_coherences[sector] = float(value)
                
                # Calculate market aggregate
                if sector_coherences:
                    market_coherence = np.mean(list(sector_coherences.values()))
                    
                    # Detect market-wide patterns
                    patterns = self._detect_market_patterns(sector_coherences)
                    
                    yield {
                        'timestamp': datetime.now(),
                        'market_coherence': market_coherence,
                        'sector_coherences': sector_coherences,
                        'patterns': patterns,
                        'health_status': self._assess_market_health(market_coherence)
                    }
                
                await asyncio.sleep(10)  # Update every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Error monitoring market: {e}")
                await asyncio.sleep(10)
    
    async def _process_update(self, data: Dict) -> CoherenceUpdate:
        """Process incoming data into CoherenceUpdate"""
        start_time = datetime.now()
        
        # Extract coherence variables
        update = CoherenceUpdate(
            timestamp=datetime.fromisoformat(data['timestamp']),
            entity_id=data['entity_id'],
            entity_type=data.get('entity_type', 'stock'),
            coherence=data['coherence'],
            psi=data['psi'],
            rho=data['rho'],
            q=data['q'],
            f=data['f'],
            truth_cost=data.get('truth_cost', 0.0),
            emotion=data.get('emotion', 'NEUTRAL'),
            anomaly_score=0.0,  # Will be calculated
            metadata=data.get('metadata', {})
        )
        
        # Update history
        if update.entity_id not in self.coherence_history:
            self.coherence_history[update.entity_id] = deque(maxlen=self.history_window)
        self.coherence_history[update.entity_id].append(update)
        
        # Store in Redis for distributed access
        await self._store_update_redis(update)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        self.metrics['processing_time_ms'].append(processing_time)
        
        return update
    
    async def _detect_anomalies(self, update: CoherenceUpdate) -> List[CoherenceAlert]:
        """Detect various types of coherence anomalies"""
        alerts = []
        history = list(self.coherence_history.get(update.entity_id, []))
        
        if len(history) < 10:  # Need minimum history
            return alerts
        
        # Calculate statistics
        coherences = [h.coherence for h in history[:-1]]  # Exclude current
        mean_coherence = np.mean(coherences)
        std_coherence = np.std(coherences)
        
        # 1. Detect coherence drift
        if len(history) >= 20:
            recent_mean = np.mean([h.coherence for h in history[-10:-1]])
            older_mean = np.mean([h.coherence for h in history[-20:-10]])
            drift = abs(recent_mean - older_mean)
            
            if drift > self.drift_threshold:
                alerts.append(CoherenceAlert(
                    timestamp=update.timestamp,
                    entity_id=update.entity_id,
                    alert_type='drift',
                    severity='warning',
                    current_coherence=update.coherence,
                    expected_coherence=older_mean,
                    deviation=drift,
                    message=f"Coherence drift detected: {drift:.3f} change",
                    recommended_action="Review recent events affecting entity"
                ))
        
        # 2. Detect coherence spikes
        if std_coherence > 0:
            z_score = abs(update.coherence - mean_coherence) / std_coherence
            update.anomaly_score = z_score
            
            if z_score > self.spike_threshold:
                alerts.append(CoherenceAlert(
                    timestamp=update.timestamp,
                    entity_id=update.entity_id,
                    alert_type='spike',
                    severity='warning' if z_score < 4 else 'critical',
                    current_coherence=update.coherence,
                    expected_coherence=mean_coherence,
                    deviation=z_score,
                    message=f"Coherence spike: {z_score:.1f} standard deviations",
                    recommended_action="Investigate unusual activity or news"
                ))
        
        # 3. Detect coherence breakdown
        if update.coherence < self.breakdown_threshold:
            alerts.append(CoherenceAlert(
                timestamp=update.timestamp,
                entity_id=update.entity_id,
                alert_type='breakdown',
                severity='critical',
                current_coherence=update.coherence,
                expected_coherence=mean_coherence,
                deviation=mean_coherence - update.coherence,
                message=f"Coherence breakdown: {update.coherence:.3f}",
                recommended_action="Immediate attention required - check for crisis"
            ))
        
        # 4. Detect coherence emergence (positive anomaly)
        if update.coherence > 0.8 and mean_coherence < 0.6:
            alerts.append(CoherenceAlert(
                timestamp=update.timestamp,
                entity_id=update.entity_id,
                alert_type='emergence',
                severity='info',
                current_coherence=update.coherence,
                expected_coherence=mean_coherence,
                deviation=update.coherence - mean_coherence,
                message=f"Coherence emergence: {update.coherence:.3f}",
                recommended_action="Positive development - monitor for sustainability"
            ))
        
        return alerts
    
    async def _handle_alert(self, alert: CoherenceAlert):
        """Handle generated alerts"""
        self.metrics['alerts_generated'] += 1
        
        # Log alert
        self.logger.warning(f"Alert: {alert.alert_type} for {alert.entity_id} - {alert.message}")
        
        # Store in Redis
        alert_key = f"alert:{alert.entity_id}:{alert.timestamp.isoformat()}"
        await self.redis.setex(
            alert_key,
            86400,  # 24 hour TTL
            json.dumps(asdict(alert), default=str)
        )
        
        # Publish to alert channel
        await self.redis.publish(
            f"alerts:{alert.severity}",
            json.dumps(asdict(alert), default=str)
        )
        
        # Call custom callback if provided
        if self.alert_callback:
            await self.alert_callback(alert)
    
    async def _store_update_redis(self, update: CoherenceUpdate):
        """Store update in Redis for distributed access"""
        # Current coherence
        await self.redis.set(
            f"coherence:{update.entity_type}:{update.entity_id}",
            update.coherence
        )
        
        # Time series data
        ts_key = f"timeseries:{update.entity_id}"
        await self.redis.zadd(
            ts_key,
            update.timestamp.timestamp(),
            json.dumps(asdict(update), default=str)
        )
        
        # Trim old data (keep 24 hours)
        cutoff = (datetime.now() - timedelta(days=1)).timestamp()
        await self.redis.zremrangebyscore(ts_key, 0, cutoff)
    
    async def _load_baselines(self):
        """Load historical baselines for anomaly detection"""
        # In production, load from database
        # For now, we'll start fresh
        pass
    
    def _detect_market_patterns(self, sector_coherences: Dict[str, float]) -> List[str]:
        """Detect market-wide patterns from sector coherences"""
        patterns = []
        
        coherence_values = list(sector_coherences.values())
        
        # Synchronized movement
        if np.std(coherence_values) < 0.1:
            if np.mean(coherence_values) > 0.6:
                patterns.append("synchronized_bullish")
            elif np.mean(coherence_values) < 0.4:
                patterns.append("synchronized_bearish")
        
        # Sector rotation
        if np.std(coherence_values) > 0.3:
            patterns.append("sector_rotation")
        
        # Flight to quality
        defensive = ['Healthcare', 'Utilities', 'Consumer']
        cyclical = ['Technology', 'Finance', 'Industrial']
        
        defensive_avg = np.mean([sector_coherences.get(s, 0.5) for s in defensive])
        cyclical_avg = np.mean([sector_coherences.get(s, 0.5) for s in cyclical])
        
        if defensive_avg > cyclical_avg + 0.2:
            patterns.append("flight_to_quality")
        elif cyclical_avg > defensive_avg + 0.2:
            patterns.append("risk_on")
        
        return patterns
    
    def _assess_market_health(self, market_coherence: float) -> str:
        """Assess overall market health"""
        if market_coherence > 0.7:
            return "VERY_HEALTHY"
        elif market_coherence > 0.5:
            return "HEALTHY"
        elif market_coherence > 0.3:
            return "NEUTRAL"
        elif market_coherence > 0.2:
            return "STRESSED"
        else:
            return "CRISIS"
    
    async def get_performance_metrics(self) -> Dict:
        """Get monitor performance metrics"""
        avg_processing_time = (
            np.mean(self.metrics['processing_time_ms'])
            if self.metrics['processing_time_ms'] else 0
        )
        
        return {
            'updates_processed': self.metrics['updates_processed'],
            'alerts_generated': self.metrics['alerts_generated'],
            'avg_processing_time_ms': avg_processing_time,
            'entities_monitored': len(self.coherence_history),
            'uptime_seconds': (datetime.now() - self.start_time).total_seconds()
        }
    
    async def cleanup(self):
        """Clean up resources"""
        if self.redis:
            self.redis.close()
            await self.redis.wait_closed()


async def example_alert_handler(alert: CoherenceAlert):
    """Example alert handler"""
    print(f"ALERT: {alert.severity.upper()} - {alert.entity_id}")
    print(f"  Type: {alert.alert_type}")
    print(f"  Message: {alert.message}")
    print(f"  Action: {alert.recommended_action}")
    
    # In production, could:
    # - Send email/SMS notifications
    # - Trigger automated trading actions
    # - Update dashboards
    # - Log to monitoring systems


async def main():
    """Example usage"""
    monitor = RealTimeCoherenceMonitor(
        ws_url="wss://market-data.example.com/coherence",
        alert_callback=example_alert_handler
    )
    
    await monitor.initialize()
    
    # Monitor specific stocks
    stocks = ['AAPL', 'GOOGL', 'TSLA', 'NVDA']
    
    async for update in monitor.stream_coherence_updates(stocks):
        print(f"{update.entity_id}: C={update.coherence:.3f}, "
              f"Anomaly={update.anomaly_score:.1f}, "
              f"Emotion={update.emotion}")


if __name__ == "__main__":
    asyncio.run(main())