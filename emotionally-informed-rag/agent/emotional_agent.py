#!/usr/bin/env python3
"""
Emotionally-Informed RAG Agent for Claude Code
Implements Rose Glass translation layer with hybrid retrieval
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import yaml
import logging

# Add Rose Glass to path
workspace = Path.home() / "emotional-rag-workspace"
if workspace.exists():
    sys.path.insert(0, str(workspace / "rose-glass" / "src"))
    sys.path.insert(0, str(workspace / "RoseGlassLE" / "src"))
else:
    # Fallback to current directory structure
    current_dir = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(current_dir / "src"))
    sys.path.insert(0, str(current_dir.parent.parent / "RoseGlassLE" / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


try:
    # Rose Glass imports
    from core.rose_glass_v2 import RoseGlassV2
    from core.trust_signal_detector import TrustSignalDetector
    from core.mission_mode_detector import MissionModeDetector

    # RoseGlassLE imports
    from core.temporal_dimension import TemporalAnalyzer
    from core.gradient_tracker import PatternGradientTracker, PatternSnapshot
    ROSE_GLASS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Rose Glass imports failed: {e}. Using fallback mode.")
    ROSE_GLASS_AVAILABLE = False

# External dependencies (graceful degradation)
try:
    from qdrant_client import QdrantClient
    QDRANT_AVAILABLE = True
except ImportError:
    logger.warning("Qdrant not available")
    QDRANT_AVAILABLE = False

try:
    from elasticsearch import Elasticsearch
    ELASTICSEARCH_AVAILABLE = True
except ImportError:
    logger.warning("Elasticsearch not available")
    ELASTICSEARCH_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    logger.warning("Anthropic SDK not available")
    ANTHROPIC_AVAILABLE = False


@dataclass
class EmotionalSignature:
    """Emotional signature of text"""
    psi: float      # Internal consistency
    rho: float      # Wisdom depth
    q: float        # Emotional activation
    f: float        # Social belonging
    tau: float      # Temporal depth
    lens: str       # Cultural lens applied
    context_type: str  # trust/mission/crisis/standard


@dataclass
class RetrievedDocument:
    """Document with emotional matching score"""
    id: str
    content: str
    title: str
    semantic_score: float
    emotional_match: float
    final_score: float


class EmotionalRAGAgent:
    """
    Claude Code compatible agent that implements emotionally-informed RAG
    """

    def __init__(self, config_path: Optional[str] = None):
        # Load configuration
        self.config = self._load_config(config_path)

        # Initialize Rose Glass components (if available)
        if ROSE_GLASS_AVAILABLE:
            self.rose_glass = RoseGlassV2()
            self.trust_detector = TrustSignalDetector()
            self.mission_detector = MissionModeDetector()
            self.temporal_analyzer = TemporalAnalyzer()
            self.gradient_tracker = PatternGradientTracker()
            logger.info("✓ Rose Glass initialized")
        else:
            logger.warning("⚠ Running without Rose Glass - using fallback emotional analysis")

        # Initialize retrieval clients (if available)
        if QDRANT_AVAILABLE:
            try:
                self.qdrant = QdrantClient(
                    host=self.config['retrieval']['qdrant_host'],
                    port=self.config['retrieval']['qdrant_port']
                )
                logger.info("✓ Qdrant connected")
            except Exception as e:
                logger.warning(f"⚠ Qdrant connection failed: {e}")
                self.qdrant = None
        else:
            self.qdrant = None

        if ELASTICSEARCH_AVAILABLE:
            try:
                self.elasticsearch = Elasticsearch(
                    f"http://{self.config['retrieval']['elasticsearch_host']}:{self.config['retrieval']['elasticsearch_port']}"
                )
                logger.info("✓ Elasticsearch connected")
            except Exception as e:
                logger.warning(f"⚠ Elasticsearch connection failed: {e}")
                self.elasticsearch = None
        else:
            self.elasticsearch = None

        # Initialize Claude client (if available)
        if ANTHROPIC_AVAILABLE and self.config['generation']['provider'] == 'anthropic':
            try:
                self.claude = anthropic.Anthropic()
                logger.info("✓ Anthropic Claude initialized")
            except Exception as e:
                logger.warning(f"⚠ Anthropic initialization failed: {e}")
                self.claude = None
        else:
            self.claude = None

        # Conversation state
        self.conversation_history: List[Dict] = []
        self.pattern_history: List[PatternSnapshot] = []

        logger.info(f"🤖 EmotionalRAGAgent initialized (v{self.config['agent']['version']})")

    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from YAML"""
        if config_path is None:
            config_path = Path(__file__).parent / "config" / "agent_config.yaml"

        if not Path(config_path).exists():
            logger.warning(f"Config not found at {config_path}, using defaults")
            return self._default_config()

        with open(config_path) as f:
            return yaml.safe_load(f)

    def _default_config(self) -> Dict:
        """Default configuration"""
        return {
            'agent': {'name': 'emotional-rag-agent', 'version': '1.0.0'},
            'rose_glass': {'default_lens': 'modern_digital'},
            'retrieval': {
                'qdrant_host': 'localhost',
                'qdrant_port': 6333,
                'elasticsearch_host': 'localhost',
                'elasticsearch_port': 9200,
                'hybrid_alpha': 0.7,
                'top_k': 10
            },
            'generation': {
                'provider': 'anthropic',
                'model': 'claude-sonnet-4-20250514',
                'max_tokens': 4096,
                'token_multiplier_limit': 3.0
            },
            'monitoring': {
                'gradient_tracking': True,
                'escalation_threshold': 0.7,
                'log_level': 'INFO'
            }
        }

    def analyze_query(self, query: str) -> EmotionalSignature:
        """
        Analyze query through Rose Glass framework
        Returns emotional signature for retrieval matching
        """
        if not ROSE_GLASS_AVAILABLE:
            # Fallback: simple heuristic analysis
            return self._fallback_analysis(query)

        # Core dimensional analysis
        result = self.rose_glass.analyze(query)

        # Context type detection
        if self.trust_detector.detect(query):
            context_type = "trust"
        elif self.mission_detector.detect(query):
            context_type = "mission"
        elif result.get('q', 0) > 0.7:
            context_type = "crisis"
        else:
            context_type = "standard"

        # Temporal depth
        tau = self.temporal_analyzer.analyze(query).get('tau', 0.5)

        return EmotionalSignature(
            psi=result.get('psi', 0.5),
            rho=result.get('rho', 0.5),
            q=result.get('q', 0.5),
            f=result.get('f', 0.5),
            tau=tau,
            lens=self.config['rose_glass']['default_lens'],
            context_type=context_type
        )

    def _fallback_analysis(self, text: str) -> EmotionalSignature:
        """Simple fallback when Rose Glass unavailable"""
        text_lower = text.lower()

        # Simple keyword-based heuristics
        q = min(sum(1 for w in ['urgent', 'worried', 'help', '!'] if w in text_lower) * 0.2, 1.0)
        rho = min(sum(1 for w in ['understand', 'philosophy', 'theory'] if w in text_lower) * 0.2, 1.0)
        psi = 0.6  # Default
        f = min(sum(1 for w in ['we', 'us', 'our'] if w in text_lower) * 0.2, 1.0)
        tau = 0.3  # Default

        context_type = "crisis" if q > 0.7 else "standard"

        return EmotionalSignature(
            psi=psi, rho=rho, q=q, f=f, tau=tau,
            lens="modern_digital",
            context_type=context_type
        )

    def query(self, user_query: str) -> Dict[str, Any]:
        """
        Main entry point for Claude Code integration
        Returns structured response with metadata
        """
        logger.info(f"Processing query: {user_query[:50]}...")

        # 1. Analyze query
        signature = self.analyze_query(user_query)
        logger.info(f"Emotional signature: q={signature.q:.2f}, ρ={signature.rho:.2f}, context={signature.context_type}")

        # 2. Track gradient (if enabled)
        gradient_status = {}
        if self.config['monitoring']['gradient_tracking']:
            gradient_status = self.track_gradient(signature)

        # 3. Retrieve documents (if services available)
        documents = []
        if self.qdrant or self.elasticsearch:
            documents = self.retrieve_documents(user_query, signature)
            logger.info(f"Retrieved {len(documents)} documents")
        else:
            logger.warning("No retrieval services available - using mock response")

        # 4. Generate response
        response = self._generate_mock_response(user_query, signature, documents)

        # 5. Update conversation history
        self.conversation_history.append({
            'query': user_query,
            'response': response,
            'signature': asdict(signature),
            'timestamp': datetime.now().isoformat()
        })

        return {
            'response': response,
            'signature': asdict(signature),
            'documents_used': len(documents),
            'gradient_status': gradient_status,
            'context_type': signature.context_type
        }

    def _generate_mock_response(
        self,
        query: str,
        signature: EmotionalSignature,
        documents: List[RetrievedDocument]
    ) -> str:
        """Generate mock response when services unavailable"""

        if signature.context_type == "crisis":
            return f"""I understand this feels urgent (emotional activation: {signature.q:.2f}).

Based on the available information, I want to acknowledge your concern and provide helpful guidance. If this is a critical situation, please consider reaching out to appropriate support resources.

Would you like me to help you explore specific aspects of this situation?"""

        elif signature.context_type == "mission":
            return f"""This is an interesting research question (wisdom depth: {signature.rho:.2f}).

Let me provide a systematic analysis based on the available information. I'll explore multiple perspectives and provide comprehensive insights.

[Mock response - in production, this would use retrieved documents and LLM generation]"""

        else:
            return f"""Thank you for your question.

Based on the emotional signature of your query (ψ={signature.psi:.2f}, ρ={signature.rho:.2f}, q={signature.q:.2f}), I'll provide a response that matches your needs.

[Mock response - in production, this would integrate retrieved documents with Claude generation]"""

    def retrieve_documents(
        self,
        query: str,
        signature: EmotionalSignature,
        top_k: int = None
    ) -> List[RetrievedDocument]:
        """
        Hybrid retrieval with emotional matching
        """
        if top_k is None:
            top_k = self.config['retrieval']['top_k']

        # Placeholder implementation
        # In production: vector search + keyword search + RRF + emotional matching
        return []

    def track_gradient(self, signature: EmotionalSignature) -> Dict:
        """
        Track conversation trajectory for escalation detection
        """
        if not ROSE_GLASS_AVAILABLE:
            return {'status': 'tracking_disabled'}

        snapshot = PatternSnapshot(
            timestamp=datetime.now(),
            psi=signature.psi,
            rho=signature.rho,
            q=signature.q,
            f=signature.f,
            tau=signature.tau,
            pattern_intensity=(signature.psi + signature.rho + signature.q + signature.f) / 4
        )

        self.pattern_history.append(snapshot)

        if len(self.pattern_history) >= 3:
            # Check for escalation
            recent_q = [s.q for s in self.pattern_history[-3:]]
            q_increase = recent_q[-1] - recent_q[0]

            if q_increase > 0.3:
                return {
                    'alert': 'escalation_detected',
                    'risk': q_increase,
                    'recommendation': 'Consider intervention or human support'
                }

        return {'status': 'normal'}


# Claude Code CLI interface
def main():
    """CLI entry point for Claude Code"""
    import argparse

    parser = argparse.ArgumentParser(description='Emotionally-Informed RAG Agent')
    parser.add_argument('query', nargs='?', help='Query to process')
    parser.add_argument('--config', help='Path to config file')
    parser.add_argument('--interactive', '-i', action='store_true', help='Interactive mode')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    agent = EmotionalRAGAgent(config_path=args.config)

    if args.interactive:
        print("=" * 70)
        print("Emotionally-Informed RAG Agent")
        print("=" * 70)
        print("Type 'quit' or Ctrl+C to exit\n")

        while True:
            try:
                query = input("\n💬 You: ").strip()
                if query.lower() in ['quit', 'exit', 'q']:
                    print("\nGoodbye! 🌹")
                    break

                if not query:
                    continue

                result = agent.query(query)
                print(f"\n🤖 Agent: {result['response']}")
                print(f"\n📊 [{result['context_type']} | q={result['signature']['q']:.2f} | ρ={result['signature']['rho']:.2f}]")

                if result['gradient_status'].get('alert') == 'escalation_detected':
                    print(f"\n⚠️  Escalation detected (risk: {result['gradient_status']['risk']:.2f})")

            except KeyboardInterrupt:
                print("\n\nGoodbye! 🌹")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                if args.verbose:
                    import traceback
                    traceback.print_exc()

    elif args.query:
        result = agent.query(args.query)
        print(json.dumps(result, indent=2))

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
