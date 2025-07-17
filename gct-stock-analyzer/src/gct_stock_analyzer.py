import os
import requests
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional, List

import pandas as pd
import numpy as np
from scipy.signal import savgol_filter

TIINGO_URL = "https://api.tiingo.com/tiingo/daily/{symbol}/prices"
TIINGO_TOKEN = os.getenv("TIINGO_API_KEY")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "stock_data")


@dataclass
class StockData:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class StockFetcher:
    """Fetch daily stock data from Tiingo"""

    def __init__(self, token: Optional[str] = None):
        self.token = token or TIINGO_TOKEN
        if not self.token:
            raise ValueError("TIINGO_API_KEY environment variable not set")

    def fetch(self, symbol: str, start: datetime, end: datetime) -> List[StockData]:
        url = TIINGO_URL.format(symbol=symbol)
        params = {
            "token": self.token,
            "startDate": start.strftime("%Y-%m-%d"),
            "endDate": end.strftime("%Y-%m-%d"),
        }
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:  # Fallback to empty list on failure
            print(f"API request failed: {exc}")
            data = []

        records = []
        for row in data:
            records.append(
                StockData(
                    timestamp=datetime.fromisoformat(row["date"].split("T")[0]),
                    open=row.get("open", 0.0),
                    high=row.get("high", 0.0),
                    low=row.get("low", 0.0),
                    close=row.get("close", 0.0),
                    volume=row.get("volume", 0.0),
                )
            )
        return records


class DataManager:
    """Manage per-stock CSV storage"""

    def __init__(self, symbol: str):
        self.symbol = symbol.lower()
        os.makedirs(DATA_DIR, exist_ok=True)
        self.path = os.path.join(DATA_DIR, f"{self.symbol}_data.csv")

    def load(self) -> pd.DataFrame:
        if os.path.exists(self.path):
            return pd.read_csv(self.path, parse_dates=["timestamp"])
        columns = ["timestamp", "open", "high", "low", "close", "volume"]
        return pd.DataFrame(columns=columns)

    def append(self, rows: List[StockData]):
        df = self.load()
        new_rows = [r.__dict__ for r in rows]
        if new_rows:
            df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
            df.drop_duplicates(subset=["timestamp"], keep="last", inplace=True)
            df.sort_values("timestamp", inplace=True)
            df.to_csv(self.path, index=False)


class GCTAnalyzer:
    """Compute simple GCT coherence metrics"""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def analyze(self) -> dict:
        if self.df.empty:
            return {"coherence": 0.0, "dC_dt": 0.0, "sentiment": "neutral"}

        self.df["return"] = self.df["close"].pct_change().fillna(0)
        self.df["psi"] = self.df["return"].abs()
        self.df["rho"] = self.df["return"].rolling(5).mean().fillna(0)
        vol_ma = self.df["volume"].rolling(5).mean().fillna(self.df["volume"])
        self.df["q_raw"] = self.df["volume"] / vol_ma
        self.df["q_opt"] = -4 * (self.df["q_raw"] - 0.667) ** 2 + 1
        self.df["f"] = 0.5
        self.df["C"] = (
            self.df["psi"] + self.df["rho"] + self.df["q_opt"] + self.df["f"]
        )

        window = 5 if len(self.df) >= 5 else len(self.df) - 1
        if window % 2 == 0:
            window -= 1
        if window >= 3:
            smoothed = savgol_filter(self.df["C"], window_length=window, polyorder=2)
        else:
            smoothed = self.df["C"].values
        dC = np.gradient(smoothed)

        coherence = float(self.df["C"].iloc[-1])
        derivative = float(dC[-1]) if len(dC) else 0.0
        sentiment = "bullish" if derivative > 0 else "bearish" if derivative < 0 else "neutral"

        return {
            "coherence": coherence,
            "dC_dt": derivative,
            "sentiment": sentiment,
        }


def update_stock(symbol: str, days: int = 30) -> dict:
    manager = DataManager(symbol)
    df = manager.load()
    if df.empty:
        start = datetime.utcnow() - timedelta(days=days)
    else:
        last = df["timestamp"].max()
        start = last + timedelta(days=1)
    end = datetime.utcnow()
    fetcher = StockFetcher()
    rows = fetcher.fetch(symbol, start=start, end=end)
    manager.append(rows)
    df = manager.load()
    analysis = GCTAnalyzer(df).analyze()
    return analysis


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GCT Stock Sentiment Analyzer")
    parser.add_argument("--symbol", required=True, help="Stock symbol, e.g. AAPL")
    args = parser.parse_args()

    results = update_stock(args.symbol)
    print(f"Sentiment for {args.symbol}: {results['sentiment']}")
    print(f"Coherence: {results['coherence']:.3f}  dC/dt: {results['dC_dt']:.3f}")
