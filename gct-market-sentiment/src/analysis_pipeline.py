"""
Main analysis pipeline for GCT Market Sentiment
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional
from collections import defaultdict

from .gct_engine import GCTEngine, GCTVariables, SectorGCTEngine
from .nlp_extractor import GCTVariableExtractor, Article
from .database import GCTDatabase
from .tiingo_client import TiingoClient, MockTiingoClient


class GCTAnalysisPipeline:
    """Main pipeline for processing news through GCT analysis"""

    def __init__(self, api_token: Optional[str] = None, use_mock: bool = False):
        # Initialize components
        self.db = GCTDatabase()
        self.extractor = GCTVariableExtractor()

        # Tiingo client
        if use_mock:
            self.tiingo = MockTiingoClient()
        else:
            self.tiingo = TiingoClient(api_token)

        # GCT engines by ticker/sector
        self.engines = defaultdict(lambda: GCTEngine())
        self.sector_engines = {}

        # Ticker to sector mapping (would be loaded from config/API)
        self.ticker_sectors = {
            "AAPL": "tech",
            "MSFT": "tech",
            "GOOGL": "tech",
            "NVDA": "tech",
            "JPM": "finance",
            "GS": "finance",
            "BAC": "finance",
            "XOM": "energy",
            "CVX": "energy",
            "COP": "energy",
            "JNJ": "healthcare",
            "PFE": "healthcare",
            "UNH": "healthcare",
        }

    def get_engine_for_ticker(self, ticker: str) -> GCTEngine:
        """Get appropriate GCT engine for a ticker"""
        sector = self.ticker_sectors.get(ticker, "default")

        if sector not in self.sector_engines:
            self.sector_engines[sector] = SectorGCTEngine(sector)

        return self.sector_engines[sector]

    def process_article(self, article_data: Dict) -> Dict[str, List[Dict]]:
        """
        Process a single article through the pipeline

        Returns:
            Dict mapping tickers to their GCT results
        """
        # Store article in database
        self.db.insert_article(article_data)

        # Create Article object
        article = Article(
            id=article_data["id"],
            timestamp=article_data["timestamp"],
            source=article_data["source"],
            title=article_data["title"],
            body=article_data["body"],
            tickers=article_data.get("tickers", []),
        )

        # Extract GCT variables
        nlp_results = self.extractor.process_article(article)

        # Process for each ticker mentioned
        ticker_results = {}
        timestamp = datetime.fromisoformat(article.timestamp.replace("Z", "+00:00"))

        for ticker in nlp_results["tickers"]:
            # Get appropriate engine
            engine = self.get_engine_for_ticker(ticker)

            # Create GCT variables
            variables = GCTVariables(
                psi=nlp_results["psi"],
                rho=nlp_results["rho"],
                q_raw=nlp_results["q_raw"],
                f=nlp_results["f"],
                timestamp=timestamp,
            )

            # Analyze
            result = engine.analyze(variables)

            # Store results
            score_data = {
                "article_id": article.id,
                "timestamp": timestamp,
                "psi": variables.psi,
                "rho": variables.rho,
                "q_raw": variables.q_raw,
                "f": variables.f,
                "q_opt": result.q_opt,
                "coherence": result.coherence,
                "dc_dt": result.dc_dt,
                "d2c_dt2": result.d2c_dt2,
                "sentiment": result.sentiment,
                "components": result.components,
            }

            self.db.insert_gct_score(score_data)

            # Update ticker timeline
            self.db.update_ticker_timeline(
                ticker=ticker,
                timestamp=timestamp,
                coherence=result.coherence,
                dc_dt=result.dc_dt,
                sentiment=result.sentiment,
            )

            ticker_results[ticker] = result

        return ticker_results

    def backfill_historical(self, days: int = 7, tickers: Optional[List[str]] = None):
        """
        Backfill historical news data

        Args:
            days: Number of days to backfill
            tickers: Specific tickers to focus on
        """
        print(f"Backfilling {days} days of historical news...")

        # Fetch historical news
        articles = self.tiingo.get_historical_news(tickers=tickers, limit=1000)

        print(f"Processing {len(articles)} articles...")

        # Process each article
        for i, article in enumerate(articles):
            try:
                results = self.process_article(article)

                if i % 10 == 0:
                    print(f"Processed {i+1}/{len(articles)} articles")

            except Exception as e:
                print(f"Error processing article {article.get('id')}: {e}")

        print("Backfill complete!")

    async def stream_realtime(self):
        """Start streaming real-time news"""
        print("Starting real-time news stream...")

        async def process_streamed_article(article_data: Dict):
            """Callback for processing streamed articles"""
            try:
                results = self.process_article(article_data)

                # Log significant movements
                for ticker, result in results.items():
                    if abs(result.dc_dt) > 0.1:
                        print(
                            f"ALERT: {ticker} showing {result.sentiment} signal "
                            f"(dc/dt = {result.dc_dt:.3f})"
                        )

            except Exception as e:
                print(f"Error processing streamed article: {e}")

        # Start streaming
        await self.tiingo.stream_realtime_news(process_streamed_article)

    def get_market_summary(self) -> Dict:
        """Get current market sentiment summary"""
        stats = self.db.get_coherence_stats()

        # Get top movers
        top_bullish = self.db.get_top_movers("bullish", limit=5)
        top_bearish = self.db.get_top_movers("bearish", limit=5)

        return {
            "overall_stats": stats,
            "top_bullish": top_bullish,
            "top_bearish": top_bearish,
            "timestamp": datetime.now().isoformat(),
        }

    def detect_coherence_spikes(self, threshold: float = 0.1) -> List[Dict]:
        """Detect tickers with coherence spikes"""
        spikes = []

        # Check each sector engine
        for sector, engine in self.sector_engines.items():
            for timestamp, coherence in engine.history[-10:]:  # Last 10 points
                _, d2c_dt2 = engine.compute_derivatives(coherence, timestamp)

                if abs(d2c_dt2) > threshold:
                    spikes.append(
                        {
                            "sector": sector,
                            "timestamp": timestamp,
                            "coherence": coherence,
                            "d2c_dt2": d2c_dt2,
                            "type": "acceleration" if d2c_dt2 > 0 else "deceleration",
                        }
                    )

        return spikes
