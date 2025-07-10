"""
Test script to verify GCT Market Sentiment system
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from datetime import datetime
from src.gct_engine import GCTEngine, GCTVariables
from src.nlp_extractor import GCTVariableExtractor, Article
from src.analysis_pipeline import GCTAnalysisPipeline


def test_gct_engine():
    """Test GCT coherence calculations"""
    print("Testing GCT Engine...")
    
    engine = GCTEngine()
    
    # Test case 1: Bullish narrative
    variables = GCTVariables(
        psi=0.8,    # High clarity
        rho=0.7,    # Good reflection
        q_raw=0.6,  # Moderate emotion
        f=0.5,      # Some social signal
        timestamp=datetime.now()
    )
    
    result = engine.analyze(variables)
    print(f"Bullish test - Coherence: {result.coherence:.3f}, Sentiment: {result.sentiment}")
    
    # Test case 2: Bearish narrative (lower values)
    variables2 = GCTVariables(
        psi=0.3,    # Low clarity
        rho=0.2,    # Poor reflection
        q_raw=0.8,  # High emotion (fear)
        f=0.1,      # Low social cohesion
        timestamp=datetime.now()
    )
    
    result2 = engine.analyze(variables2)
    print(f"Bearish test - Coherence: {result2.coherence:.3f}, Sentiment: {result2.sentiment}")
    
    return True


def test_nlp_extractor():
    """Test NLP variable extraction"""
    print("\nTesting NLP Extractor...")
    
    extractor = GCTVariableExtractor()
    
    # Test article
    article = Article(
        id="test_1",
        timestamp=datetime.now().isoformat(),
        source="Test",
        title="Apple Stock Surges on Strong iPhone Sales, Analysts Optimistic",
        body="Apple Inc. (AAPL) reported record-breaking iPhone sales this quarter, "
             "exceeding analyst expectations. The tech giant's revenue grew 15% year-over-year. "
             "Most analysts agree this signals continued strength in consumer demand. "
             "We believe Apple is well-positioned for future growth.",
        tickers=["AAPL"]
    )
    
    results = extractor.process_article(article)
    
    print(f"Extracted variables:")
    print(f"  ψ (clarity): {results['psi']:.3f}")
    print(f"  ρ (reflection): {results['rho']:.3f}")
    print(f"  q (emotion): {results['q_raw']:.3f}")
    print(f"  f (social): {results['f']:.3f}")
    print(f"  Tickers: {results['tickers']}")
    
    return True


def test_pipeline():
    """Test full analysis pipeline"""
    print("\nTesting Analysis Pipeline...")
    
    # Use mock client
    pipeline = GCTAnalysisPipeline(use_mock=True)
    
    # Get mock article
    articles = pipeline.tiingo.get_historical_news(limit=1)
    
    if articles:
        article = articles[0]
        print(f"Processing article: {article['title']}")
        
        # Process through pipeline
        results = pipeline.process_article(article)
        
        for ticker, gct_result in results.items():
            print(f"\n{ticker}:")
            print(f"  Coherence: {gct_result.coherence:.3f}")
            print(f"  dC/dt: {gct_result.dc_dt:.3f}")
            print(f"  Sentiment: {gct_result.sentiment}")
            
    # Test market summary
    summary = pipeline.get_market_summary()
    print(f"\nMarket Summary: {summary['overall_stats']}")
    
    return True


if __name__ == "__main__":
    print("="*50)
    print("GCT Market Sentiment System Test")
    print("="*50)
    
    try:
        # Run tests
        test_gct_engine()
        test_nlp_extractor()
        test_pipeline()
        
        print("\n✅ All tests passed!")
        print("\nSystem is ready. Run 'streamlit run app.py' to start the dashboard.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()