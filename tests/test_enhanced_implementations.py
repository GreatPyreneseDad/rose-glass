#!/usr/bin/env python3
"""
Test Suite for Enhanced GCT Implementations
Tests for all optimizations based on review feedback
"""

import pytest
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import sys
import os

# Add project directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'gct-market-sentiment', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'validation'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'soulmath-moderation-system', 'src'))


class TestEnhancedNLPExtractor:
    """Test enhanced NLP extraction capabilities"""
    
    @pytest.fixture
    def extractor(self):
        from enhanced_nlp_extractor import EnhancedNLPExtractor
        return EnhancedNLPExtractor()
    
    def test_psi_extraction_with_complexity(self, extractor):
        """Test psi extraction with syntactic complexity"""
        # Simple text
        simple_text = "This is good. I like it."
        simple_psi = extractor.extract_psi_enhanced(simple_text)
        
        # Complex text with subordinate clauses
        complex_text = """
        Although the situation presents significant challenges that we must 
        carefully consider, I believe that through systematic analysis and 
        thoughtful deliberation, we can develop solutions that address the 
        core issues while maintaining our fundamental principles.
        """
        complex_psi = extractor.extract_psi_enhanced(complex_text)
        
        # Complex text should have higher psi
        assert complex_psi > simple_psi
        assert 0 <= simple_psi <= 1
        assert 0 <= complex_psi <= 1
    
    def test_rho_extraction_with_discourse(self, extractor):
        """Test rho extraction with discourse relations"""
        # Text without discourse markers
        simple_text = "The market went up. Stocks gained value."
        simple_rho = extractor.extract_rho_enhanced(simple_text)
        
        # Text with rich discourse relations
        discourse_text = """
        The market initially showed weakness; however, strong earnings reports 
        led to a reversal. Specifically, tech stocks rallied because investors 
        regained confidence. Consequently, the broader indices followed suit, 
        although some sectors lagged behind.
        """
        discourse_rho = extractor.extract_rho_enhanced(discourse_text)
        
        assert discourse_rho > simple_rho
        assert 0 <= simple_rho <= 1
        assert 0 <= discourse_rho <= 1
    
    def test_q_extraction_with_moral_engagement(self, extractor):
        """Test questioning frequency with moral engagement"""
        # Text without questions or moral terms
        neutral_text = "The weather is sunny today. Birds are singing."
        neutral_q = extractor.extract_q_enhanced(neutral_text)
        
        # Text with questions and moral engagement
        questioning_text = """
        What are our ethical obligations in this situation? Should we prioritize 
        fairness over efficiency? I wonder if we're considering all the moral 
        implications of our decision.
        """
        questioning_q = extractor.extract_q_enhanced(questioning_text)
        
        assert questioning_q > neutral_q
        assert questioning_q > 0.5  # High engagement expected
    
    def test_f_extraction_with_challenges(self, extractor):
        """Test challenge frequency extraction"""
        # Text without challenges
        easy_text = "Everything is going smoothly. No problems at all."
        easy_f = extractor.extract_f_enhanced(easy_text)
        
        # Text with challenges and growth
        challenge_text = """
        We face significant obstacles in implementing this solution. The 
        conflict between stakeholders creates a challenging dilemma. However, 
        by confronting these difficulties, we can grow and develop better 
        approaches.
        """
        challenge_f = extractor.extract_f_enhanced(challenge_text)
        
        assert challenge_f > easy_f
        assert challenge_f > 0.3  # Significant challenge content
    
    def test_argument_structure_detection(self, extractor):
        """Test argument structure analysis"""
        argumentative_text = """
        I argue that sustainable investing is crucial for long-term success. 
        Recent research shows that ESG-focused portfolios outperform traditional 
        ones. For example, a study by Harvard Business School found 15% higher 
        returns. This evidence clearly supports the claim that sustainability 
        matters.
        """
        
        doc = extractor.nlp(argumentative_text)
        arg_structure = extractor._analyze_argument_structure(doc)
        
        assert len(arg_structure.claims) > 0
        assert len(arg_structure.evidence) > 0
        assert arg_structure.complexity_score > 0.5


class TestGCTValidator:
    """Test validation suite functionality"""
    
    @pytest.fixture
    def validator(self):
        from gct_validator import GCTValidator
        return GCTValidator()
    
    def test_test_retest_reliability(self, validator):
        """Test reliability calculation"""
        # Create mock data
        subjects = ['user1', 'user2', 'user3', 'user4', 'user5']
        
        # Mock scores with high reliability
        test_scores = {'user1': 0.7, 'user2': 0.6, 'user3': 0.8, 
                      'user4': 0.5, 'user5': 0.65}
        retest_scores = {'user1': 0.72, 'user2': 0.58, 'user3': 0.82, 
                        'user4': 0.48, 'user5': 0.67}
        
        with patch.object(validator, '_get_scores') as mock_get_scores:
            mock_get_scores.side_effect = [test_scores, retest_scores]
            
            result = validator.test_retest_reliability(subjects)
            
            assert result.metric_name == "test_retest_reliability_coherence"
            assert result.value > 0.8  # Should show good reliability
            assert result.n_samples == 5
            assert "reliability" in result.interpretation.lower()
    
    def test_internal_consistency(self, validator):
        """Test Cronbach's alpha calculation"""
        # Create item responses with good internal consistency
        np.random.seed(42)
        n_subjects = 50
        n_items = 10
        
        # Generate correlated items
        base_scores = np.random.normal(0.5, 0.1, n_subjects)
        item_responses = pd.DataFrame()
        
        for i in range(n_items):
            noise = np.random.normal(0, 0.05, n_subjects)
            item_responses[f'item_{i}'] = base_scores + noise
        
        result = validator.internal_consistency(item_responses)
        
        assert result.value > 0.7  # Should show acceptable consistency
        assert result.n_samples == n_subjects
        assert "α" in result.interpretation
    
    def test_convergent_validity(self, validator):
        """Test convergent validity with external measure"""
        # Create correlated scores
        np.random.seed(42)
        n = 30
        
        gct_base = np.random.normal(0.5, 0.15, n)
        external_base = gct_base + np.random.normal(0, 0.1, n)
        
        gct_scores = {f'user_{i}': float(gct_base[i]) for i in range(n)}
        external_scores = {f'user_{i}': float(external_base[i]) for i in range(n)}
        
        result = validator.convergent_validity(
            gct_scores, 
            external_scores, 
            "wellbeing_scale"
        )
        
        assert result.value > 0.5  # Should show moderate correlation
        assert result.p_value < 0.05  # Should be significant
        assert "correlation" in result.interpretation
    
    def test_predictive_validity(self, validator):
        """Test predictive validity"""
        # Create data where baseline predicts outcome
        np.random.seed(42)
        n = 50
        
        baseline = np.random.normal(0.5, 0.15, n)
        # Outcome partially determined by baseline
        outcomes = 0.7 * baseline + 0.3 * np.random.normal(0.5, 0.1, n)
        
        baseline_scores = {f'user_{i}': float(baseline[i]) for i in range(n)}
        outcome_scores = {f'user_{i}': float(outcomes[i]) for i in range(n)}
        
        result = validator.predictive_validity(
            baseline_scores,
            outcome_scores,
            "performance",
            30  # 30 days lag
        )
        
        assert result.value > 0.3  # Should explain >30% variance
        assert result.p_value < 0.05
        assert "variance" in result.interpretation


class TestRealTimeMonitor:
    """Test real-time monitoring capabilities"""
    
    @pytest.fixture
    async def monitor(self):
        from realtime_coherence_monitor import RealTimeCoherenceMonitor
        monitor = RealTimeCoherenceMonitor(
            ws_url="wss://test.example.com",
            redis_url="redis://localhost:6379"
        )
        # Mock Redis
        monitor.redis = AsyncMock()
        return monitor
    
    @pytest.mark.asyncio
    async def test_coherence_drift_detection(self, monitor):
        """Test drift detection in coherence"""
        from realtime_coherence_monitor import CoherenceUpdate
        
        # Create history with stable coherence
        entity_id = "AAPL"
        for i in range(20):
            update = CoherenceUpdate(
                timestamp=datetime.now() - timedelta(minutes=20-i),
                entity_id=entity_id,
                entity_type="stock",
                coherence=0.5 + np.random.normal(0, 0.05),
                psi=0.5, rho=0.5, q=0.5, f=0.5,
                truth_cost=0.2,
                emotion="NEUTRAL",
                anomaly_score=0.0,
                metadata={}
            )
            monitor.coherence_history[entity_id].append(update)
        
        # Add drifting update
        drift_update = CoherenceUpdate(
            timestamp=datetime.now(),
            entity_id=entity_id,
            entity_type="stock",
            coherence=0.75,  # Significant increase
            psi=0.7, rho=0.7, q=0.7, f=0.7,
            truth_cost=0.1,
            emotion="GREED",
            anomaly_score=0.0,
            metadata={}
        )
        
        alerts = await monitor._detect_anomalies(drift_update)
        
        assert len(alerts) > 0
        assert any(alert.alert_type == 'spike' for alert in alerts)
    
    @pytest.mark.asyncio
    async def test_market_pattern_detection(self, monitor):
        """Test market-wide pattern detection"""
        sector_coherences = {
            'Technology': 0.7,
            'Healthcare': 0.72,
            'Finance': 0.69,
            'Energy': 0.71,
            'Consumer': 0.68,
            'Industrial': 0.70
        }
        
        patterns = monitor._detect_market_patterns(sector_coherences)
        
        assert 'synchronized_bullish' in patterns
        assert 'sector_rotation' not in patterns  # Low variance
    
    @pytest.mark.asyncio
    async def test_performance_metrics(self, monitor):
        """Test performance metric tracking"""
        monitor.start_time = datetime.now() - timedelta(hours=1)
        monitor.metrics['updates_processed'] = 1000
        monitor.metrics['alerts_generated'] = 25
        monitor.metrics['processing_time_ms'].extend([5.2, 6.1, 4.8, 5.5, 6.0])
        
        metrics = await monitor.get_performance_metrics()
        
        assert metrics['updates_processed'] == 1000
        assert metrics['alerts_generated'] == 25
        assert 4 < metrics['avg_processing_time_ms'] < 7
        assert metrics['uptime_seconds'] > 3500


class TestContextAwareModerator:
    """Test context-aware moderation"""
    
    @pytest.fixture
    async def moderator(self):
        from context_aware_moderator import ContextAwareGCTModerator
        with patch('context_aware_moderator.AutoModel'):
            with patch('context_aware_moderator.AutoTokenizer'):
                return ContextAwareGCTModerator()
    
    @pytest.mark.asyncio
    async def test_coherence_drift_classification(self, moderator):
        """Test drift type classification"""
        from context_aware_moderator import Comment, Thread, User
        
        # Create test data
        comment = Comment(
            id="c1",
            author_id="u1",
            text="This discussion has completely derailed!",
            timestamp=datetime.now()
        )
        
        thread_analysis = {
            'thread_coherence': 0.6,
            'coherence_flow': [
                {'coherence': 0.6}, {'coherence': 0.58}, {'coherence': 0.62}
            ]
        }
        
        user_trajectory = {
            'baseline_coherence': 0.5,
            'trajectory_type': 'stable'
        }
        
        # Test with sudden drop
        with patch.object(moderator, 'calculate_coherence_from_text', return_value=0.2):
            drift = moderator.calculate_coherence_drift(
                0.2,  # Low coherence
                thread_analysis,
                user_trajectory
            )
            
            assert drift['drift_type'] == 'sudden_shift'
            assert drift['is_anomalous']
            assert drift['sudden_drift'] > 0.3
    
    @pytest.mark.asyncio
    async def test_coordination_detection(self, moderator):
        """Test coordinated behavior detection"""
        from context_aware_moderator import Comment, Thread, User
        
        # Create coordinated comments
        base_time = datetime.now()
        thread = Thread(
            id="t1",
            title="Test Thread",
            created_at=base_time - timedelta(hours=1),
            comments=[
                Comment("c1", "u2", "Great point!", base_time - timedelta(minutes=5), thread_id="t1"),
                Comment("c2", "u3", "I agree completely!", base_time - timedelta(minutes=4), thread_id="t1"),
                Comment("c3", "u4", "Exactly right!", base_time - timedelta(minutes=3), thread_id="t1"),
            ],
            participants=["u1", "u2", "u3", "u4"],
            metadata={}
        )
        
        user = User(
            id="u1",
            username="test_user",
            created_at=base_time - timedelta(days=30),
            comment_history=[],
            coherence_trajectory=[0.5] * 10,
            network=["u2", "u3", "u4"],  # Connected to commenters
            metadata={}
        )
        
        comment = Comment(
            id="c4",
            author_id="u1",
            text="Absolutely correct!",
            timestamp=base_time
        )
        
        # Mock embedding similarity
        with patch.object(moderator, '_get_embedding', new_callable=AsyncMock) as mock_embed:
            mock_embed.return_value = np.array([0.5] * 768)  # Mock BERT embedding
            
            coordination_score = await moderator.detect_coordination(
                comment, thread, user
            )
            
            assert coordination_score > 0.3  # Should detect temporal clustering
    
    @pytest.mark.asyncio
    async def test_moderation_decision_generation(self, moderator):
        """Test decision generation logic"""
        
        # Test coherence breakdown
        decision = moderator.generate_moderation_decision(
            comment_coherence=0.15,  # Very low
            drift_analysis={'drift_type': 'normal', 'is_anomalous': False},
            coordination_score=0.1,
            toxicity_analysis={'is_toxic': False},
            user_trajectory={'trajectory_type': 'stable'}
        )
        
        assert decision.action == 'remove'
        assert 'coherence_breakdown' in decision.coherence_analysis
        assert decision.confidence > 0.9
        
        # Test coordinated behavior
        decision = moderator.generate_moderation_decision(
            comment_coherence=0.5,
            drift_analysis={'drift_type': 'normal', 'is_anomalous': False},
            coordination_score=0.8,  # High coordination
            toxicity_analysis={'is_toxic': False},
            user_trajectory={'trajectory_type': 'stable'}
        )
        
        assert decision.action == 'shadowban'
        assert 'network_investigation' in decision.recommended_interventions


class TestIntegration:
    """Integration tests for the full system"""
    
    @pytest.mark.asyncio
    async def test_market_sentiment_pipeline(self):
        """Test full market sentiment analysis pipeline"""
        from enhanced_nlp_extractor import EnhancedNLPExtractor
        
        extractor = EnhancedNLPExtractor()
        
        market_text = """
        The Federal Reserve's decision to maintain interest rates reflects 
        careful consideration of multiple economic factors. However, market 
        participants are questioning whether this approach adequately addresses 
        emerging inflationary pressures. Recent data suggests that while 
        employment remains strong, there are growing challenges in supply 
        chain dynamics that could impact future growth.
        """
        
        # Extract all variables
        psi = extractor.extract_psi_enhanced(market_text)
        rho = extractor.extract_rho_enhanced(market_text)
        q = extractor.extract_q_enhanced(market_text)
        f = extractor.extract_f_enhanced(market_text)
        
        # Calculate coherence
        coherence = (psi + rho + q + f) / 4
        
        # All values should be reasonable
        assert 0.3 < psi < 0.8
        assert 0.3 < rho < 0.8
        assert 0.1 < q < 0.6
        assert 0.2 < f < 0.7
        assert 0.2 < coherence < 0.7
    
    def test_validation_workflow(self):
        """Test complete validation workflow"""
        from gct_validator import GCTValidator
        
        validator = GCTValidator()
        
        # Generate synthetic test data
        np.random.seed(42)
        n_subjects = 100
        n_items = 20
        
        # Create item responses with factor structure
        factors = np.random.normal(0, 1, (n_subjects, 4))  # 4 factors
        loadings = np.random.uniform(0.3, 0.8, (n_items, 4))
        noise = np.random.normal(0, 0.2, (n_subjects, n_items))
        
        item_responses = pd.DataFrame(
            np.dot(factors, loadings.T) + noise,
            columns=[f'item_{i}' for i in range(n_items)]
        )
        
        # Test internal consistency
        consistency_result = validator.internal_consistency(item_responses)
        assert consistency_result.value > 0.6
        
        # Test factor structure
        factor_result = validator.factor_structure_validity(item_responses, expected_factors=4)
        assert 'loadings' in factor_result
        assert factor_result['n_factors'] == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])