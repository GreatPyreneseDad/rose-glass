import sys
import os
import pandas as pd
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'gct-stock-analyzer', 'src'))
from gct_stock_analyzer import GCTAnalyzer


def test_gct_analyzer_basic():
    dates = pd.date_range(end=datetime.utcnow(), periods=5)
    df = pd.DataFrame({
        "timestamp": dates,
        "open": [1, 2, 3, 4, 5],
        "high": [1, 2, 3, 4, 5],
        "low": [1, 2, 3, 4, 5],
        "close": [1, 2, 3, 4, 5],
        "volume": [100, 100, 100, 100, 100],
    })
    analyzer = GCTAnalyzer(df)
    res = analyzer.analyze()
    assert set(res.keys()) == {"coherence", "dC_dt", "sentiment"}
    assert isinstance(res["coherence"], float)
    assert isinstance(res["dC_dt"], float)
    assert res["sentiment"] in {"bullish", "bearish", "neutral"}
