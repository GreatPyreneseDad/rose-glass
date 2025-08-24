"""
Real-time GCT Coherence Computation Pipeline
Integrates all GCT variable models for live coherence calculation

This pipeline:
- Loads all trained models (Ψ, ρ, q, f)
- Processes text in real-time
- Calculates coherence with biological optimization
- Provides temporal tracking and derivatives
- Enables consistency monitoring
"""

import asyncio
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging
from pathlib import Path
import matplotlib.pyplot as plt
from collections import deque
import threading
import queue
import time

# Import our models
from gct_ml_framework import GCTVariables, GCTCoherence
from psi_consistency_model import create_psi_model
from rho_wisdom_model import create_rho_model
from q_moral_activation_model import create_q_model
from f_social_belonging_model import create_f_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CoherenceSnapshot:
    """Single point in coherence timeline"""
    timestamp: datetime
    text: str
    variables: GCTVariables
    coherence: float
    q_optimized: float
    components: Dict[str, float]
    processing_time: float


@dataclass
class CoherenceTimeline:
    """Tracks coherence over time for derivative calculations"""
    snapshots: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    def add_snapshot(self, snapshot: CoherenceSnapshot):
        self.snapshots.append(snapshot)
    
    def get_derivatives(self, window_size: int = 5) -> Tuple[float, float]:
        """Calculate first and second derivatives of coherence"""
        if len(self.snapshots) < 3:
            return 0.0, 0.0
        
        # Get recent snapshots
        recent = list(self.snapshots)[-window_size:]
        
        # Extract time and coherence values
        times = [(s.timestamp - recent[0].timestamp).total_seconds() for s in recent]
        coherences = [s.coherence for s in recent]
        
        # Calculate derivatives using numpy
        if len(times) >= 2:
            dc_dt = np.gradient(coherences, times)[-1]
        else:
            dc_dt = 0.0
            
        if len(times) >= 3:
            d2c_dt2 = np.gradient(np.gradient(coherences, times), times)[-1]
        else:
            d2c_dt2 = 0.0
            
        return dc_dt, d2c_dt2


class RealtimeGCTPipeline:
    """Main pipeline for real-time GCT coherence computation"""
    
    def __init__(self, model_dir: str = "/Users/chris/GCT-ML-Lab/models",
                 device: str = None):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True, parents=True)
        
        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"Using device: {self.device}")
        
        # Initialize models
        self._initialize_models()
        
        # Timeline tracking
        self.timeline = CoherenceTimeline()
        
        # Biological optimization parameters
        self.km = 0.2
        self.ki = 0.8
        
        # Processing queue for async operation
        self.processing_queue = queue.Queue()
        self.results_queue = queue.Queue()
        self.is_running = False
        
    def _initialize_models(self):
        """Initialize all GCT variable models"""
        logger.info("Initializing GCT models...")
        
        # Create models
        self.psi_model, self.psi_trainer = create_psi_model()
        self.rho_model, self.rho_trainer = create_rho_model()
        self.q_model, self.q_trainer = create_q_model()
        self.f_model, self.f_trainer = create_f_model()
        
        # Load pre-trained weights if available
        self._load_model_weights()
        
        # Set models to evaluation mode
        self.psi_model.eval()
        self.rho_model.eval()
        self.q_model.eval()
        self.f_model.eval()
        
        logger.info("Models initialized successfully")
        
    def _load_model_weights(self):
        """Load pre-trained model weights if available"""
        models = {
            'psi': (self.psi_model, 'psi_best.pth'),
            'rho': (self.rho_model, 'rho_best.pth'),
            'q': (self.q_model, 'q_best.pth'),
            'f': (self.f_model, 'f_best.pth')
        }
        
        for name, (model, filename) in models.items():
            weight_path = self.model_dir / filename
            if weight_path.exists():
                try:
                    model.load_state_dict(torch.load(weight_path, map_location=self.device))
                    logger.info(f"Loaded pre-trained weights for {name} model")
                except Exception as e:
                    logger.warning(f"Could not load weights for {name}: {e}")
            else:
                logger.warning(f"No pre-trained weights found for {name} at {weight_path}")
    
    def calculate_variables(self, text: str) -> GCTVariables:
        """Calculate all GCT variables for given text"""
        start_time = time.time()
        
        with torch.no_grad():
            # Predict each variable
            psi = self.psi_trainer.predict(text)
            rho = self.rho_trainer.predict(text)
            q, _ = self.q_trainer.predict(text)
            f, _ = self.f_trainer.predict(text)
        
        processing_time = time.time() - start_time
        
        return GCTVariables(
            psi=psi,
            rho=rho,
            q=q,
            f=f,
            timestamp=datetime.now(),
            text=text,
            metadata={'processing_time': processing_time}
        )
    
    def calculate_coherence(self, variables: GCTVariables) -> GCTCoherence:
        """Calculate coherence from GCT variables"""
        # Apply biological optimization to q
        q_opt = variables.q / (self.km + variables.q + (variables.q**2 / self.ki))
        
        # Calculate component contributions
        base = variables.psi
        wisdom_amplification = variables.rho * variables.psi
        social_amplification = variables.f * variables.psi
        coupling = 0.15 * variables.rho * q_opt
        
        # Total coherence
        coherence = base + wisdom_amplification + q_opt + social_amplification + coupling
        
        # Get derivatives from timeline
        dc_dt, d2c_dt2 = self.timeline.get_derivatives()
        
        components = {
            'base': base,
            'wisdom_amplification': wisdom_amplification,
            'moral_energy': q_opt,
            'social_amplification': social_amplification,
            'coupling': coupling
        }
        
        derivatives = {
            'dc_dt': dc_dt,
            'd2c_dt2': d2c_dt2,
            'trend': 'increasing' if dc_dt > 0.01 else 'decreasing' if dc_dt < -0.01 else 'stable'
        }
        
        return GCTCoherence(
            coherence=coherence,
            q_optimized=q_opt,
            components=components,
            derivatives=derivatives,
            timestamp=variables.timestamp
        )
    
    def process_text(self, text: str) -> Dict:
        """Process single text and return full analysis"""
        # Calculate variables
        variables = self.calculate_variables(text)
        
        # Calculate coherence
        coherence_result = self.calculate_coherence(variables)
        
        # Create snapshot
        snapshot = CoherenceSnapshot(
            timestamp=datetime.now(),
            text=text,
            variables=variables,
            coherence=coherence_result.coherence,
            q_optimized=coherence_result.q_optimized,
            components=coherence_result.components,
            processing_time=variables.metadata.get('processing_time', 0)
        )
        
        # Add to timeline
        self.timeline.add_snapshot(snapshot)
        
        # Return comprehensive results
        return {
            'timestamp': snapshot.timestamp.isoformat(),
            'variables': {
                'psi': variables.psi,
                'rho': variables.rho,
                'q': variables.q,
                'f': variables.f
            },
            'coherence': coherence_result.coherence,
            'q_optimized': coherence_result.q_optimized,
            'components': coherence_result.components,
            'derivatives': coherence_result.derivatives,
            'processing_time': snapshot.processing_time,
            'interpretation': self._interpret_coherence(coherence_result)
        }
    
    def _interpret_coherence(self, coherence: GCTCoherence) -> Dict:
        """Provide human-readable interpretation of coherence results"""
        level = self._get_coherence_level(coherence.coherence)
        
        interpretation = {
            'level': level,
            'score': f"{coherence.coherence:.3f}",
            'trend': coherence.derivatives['trend'],
            'dominant_component': max(coherence.components.items(), key=lambda x: x[1])[0],
            'balance': self._assess_balance(coherence.components)
        }
        
        # Specific insights
        insights = []
        
        if coherence.components['base'] < 0.3:
            insights.append("Low internal consistency - narrative may lack coherent structure")
        elif coherence.components['base'] > 0.7:
            insights.append("Strong internal consistency - well-structured narrative")
            
        if coherence.components['wisdom_amplification'] > 0.5:
            insights.append("High wisdom density - deep analytical content")
            
        if coherence.q_optimized > 0.7:
            insights.append("Strong moral activation - value-driven content")
        elif coherence.q_optimized < 0.3:
            insights.append("Low moral energy - lacks call to action")
            
        if coherence.components['social_amplification'] > 0.5:
            insights.append("Strong social belonging - collective perspective")
        elif coherence.components['social_amplification'] < 0.2:
            insights.append("Limited social connection - individualistic tone")
            
        interpretation['insights'] = insights
        
        return interpretation
    
    def _get_coherence_level(self, coherence: float) -> str:
        """Categorize coherence level"""
        if coherence >= 3.5:
            return "Exceptional"
        elif coherence >= 2.5:
            return "Very High"
        elif coherence >= 1.5:
            return "High"
        elif coherence >= 1.0:
            return "Moderate"
        elif coherence >= 0.5:
            return "Low"
        else:
            return "Very Low"
    
    def _assess_balance(self, components: Dict[str, float]) -> str:
        """Assess balance between components"""
        values = list(components.values())
        std_dev = np.std(values)
        
        if std_dev < 0.1:
            return "Well-balanced"
        elif std_dev < 0.3:
            return "Moderately balanced"
        else:
            return "Imbalanced"
    
    async def process_stream(self, text_stream: AsyncIterator[str]) -> AsyncIterator[Dict]:
        """Process streaming text input asynchronously"""
        async for text in text_stream:
            result = await asyncio.to_thread(self.process_text, text)
            yield result
    
    def start_background_processor(self):
        """Start background processing thread"""
        if self.is_running:
            return
            
        self.is_running = True
        self.processor_thread = threading.Thread(target=self._background_processor)
        self.processor_thread.start()
        logger.info("Background processor started")
    
    def _background_processor(self):
        """Background thread for processing texts"""
        while self.is_running:
            try:
                # Get text from queue (timeout to check is_running)
                text = self.processing_queue.get(timeout=1.0)
                
                # Process text
                result = self.process_text(text)
                
                # Put result in results queue
                self.results_queue.put(result)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in background processor: {e}")
    
    def stop_background_processor(self):
        """Stop background processing thread"""
        self.is_running = False
        if hasattr(self, 'processor_thread'):
            self.processor_thread.join()
        logger.info("Background processor stopped")
    
    def submit_text(self, text: str) -> str:
        """Submit text for background processing"""
        if not self.is_running:
            self.start_background_processor()
            
        task_id = f"{datetime.now().timestamp()}"
        self.processing_queue.put(text)
        return task_id
    
    def get_results(self, timeout: float = None) -> Optional[Dict]:
        """Get processed results from queue"""
        try:
            return self.results_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def export_timeline(self, filepath: str):
        """Export timeline data for analysis"""
        data = []
        for snapshot in self.timeline.snapshots:
            data.append({
                'timestamp': snapshot.timestamp.isoformat(),
                'coherence': snapshot.coherence,
                'psi': snapshot.variables.psi,
                'rho': snapshot.variables.rho,
                'q': snapshot.variables.q,
                'f': snapshot.variables.f,
                'q_optimized': snapshot.q_optimized,
                'components': snapshot.components,
                'text_preview': snapshot.text[:100] + '...' if len(snapshot.text) > 100 else snapshot.text
            })
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Timeline exported to {filepath}")
    
    def visualize_timeline(self, save_path: Optional[str] = None):
        """Create visualization of coherence timeline"""
        if len(self.timeline.snapshots) < 2:
            logger.warning("Not enough data points for visualization")
            return
            
        # Extract data
        snapshots = list(self.timeline.snapshots)
        times = [(s.timestamp - snapshots[0].timestamp).total_seconds() / 60 for s in snapshots]  # Minutes
        coherences = [s.coherence for s in snapshots]
        psis = [s.variables.psi for s in snapshots]
        rhos = [s.variables.rho for s in snapshots]
        qs = [s.variables.q for s in snapshots]
        fs = [s.variables.f for s in snapshots]
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Plot coherence
        ax1.plot(times, coherences, 'b-', linewidth=2, label='Coherence')
        ax1.set_ylabel('Coherence Score')
        ax1.set_title('GCT Coherence Over Time')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot individual variables
        ax2.plot(times, psis, 'r-', label='Ψ (Consistency)', alpha=0.7)
        ax2.plot(times, rhos, 'g-', label='ρ (Wisdom)', alpha=0.7)
        ax2.plot(times, qs, 'b-', label='q (Moral Energy)', alpha=0.7)
        ax2.plot(times, fs, 'm-', label='f (Social Belonging)', alpha=0.7)
        ax2.set_xlabel('Time (minutes)')
        ax2.set_ylabel('Variable Score')
        ax2.set_title('GCT Variables Over Time')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to {save_path}")
        else:
            plt.show()


def create_pipeline(device: Optional[str] = None) -> RealtimeGCTPipeline:
    """Create and return a real-time GCT pipeline"""
    return RealtimeGCTPipeline(device=device)


async def example_streaming():
    """Example of streaming text processing"""
    pipeline = create_pipeline()
    
    async def text_generator():
        """Simulate streaming text input"""
        texts = [
            "Our research reveals fundamental patterns in human cognition.",
            "We must act together to protect our shared future.",
            "The evidence clearly demonstrates the need for immediate action.",
            "As a community, we have the power to create positive change."
        ]
        
        for text in texts:
            yield text
            await asyncio.sleep(1)  # Simulate delay
    
    # Process stream
    async for result in pipeline.process_stream(text_generator()):
        print(f"\nCoherence: {result['coherence']:.3f}")
        print(f"Level: {result['interpretation']['level']}")
        print(f"Trend: {result['interpretation']['trend']}")


def main():
    """Example usage of real-time pipeline"""
    # Create pipeline
    pipeline = create_pipeline()
    
    # Test texts
    test_texts = [
        """
        Through careful analysis of the data, we have discovered patterns that 
        challenge our previous understanding. These findings, while preliminary, 
        suggest a need to reconsider our theoretical framework. Further research 
        is essential to validate these observations.
        """,
        """
        NOW is the time for ACTION! We cannot stand by while our future crumbles. 
        Every one of us must rise up and fight for what we believe in. Together, 
        we are unstoppable!
        """,
        """
        Random thoughts. Sky blue. Pizza tastes good. Cars go fast. 
        Weather changes. Time passes.
        """
    ]
    
    # Process texts
    for i, text in enumerate(test_texts):
        print(f"\n{'='*60}")
        print(f"Text {i+1}:")
        print(f"{'='*60}")
        
        result = pipeline.process_text(text)
        
        print(f"\nVariables:")
        for var, value in result['variables'].items():
            print(f"  {var}: {value:.3f}")
        
        print(f"\nCoherence: {result['coherence']:.3f}")
        print(f"Level: {result['interpretation']['level']}")
        print(f"Balance: {result['interpretation']['balance']}")
        
        print(f"\nInsights:")
        for insight in result['interpretation']['insights']:
            print(f"  - {insight}")
        
        print(f"\nProcessing time: {result['processing_time']:.3f}s")
    
    # Export timeline
    pipeline.export_timeline("/Users/chris/GCT-ML-Lab/timeline_demo.json")
    
    # Create visualization
    pipeline.visualize_timeline("/Users/chris/GCT-ML-Lab/coherence_timeline.png")


if __name__ == "__main__":
    # Run synchronous example
    main()
    
    # Run async example
    # asyncio.run(example_streaming())