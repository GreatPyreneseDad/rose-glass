# GCT Stock Sentiment Analyzer

A standalone module for evaluating the market sentiment of a single stock using Grounded Coherence Theory (GCT). The tool retrieves historical data from the Tiingo API, maintains a per-stock CSV record, and computes a coherence-based sentiment signal.

## Features

- Fetch daily OHLCV data for any stock symbol via Tiingo
- Append new data to a dedicated CSV file under `stock_data/`
- Calculate a simplified GCT coherence score with bio-inspired emotional optimization (q_opt at 0.667)
- Output bullish, bearish or neutral sentiment based on the coherence derivative
- Command line interface for quick usage

## Installation

1. Clone this repository.
2. Set the `TIINGO_API_KEY` environment variable with your Tiingo token.
3. Run the analyzer:

```bash
python gct_stock_analyzer.py --symbol AAPL
```

Historical data for the symbol will be stored in `stock_data/aapl_data.csv`.

## Development

The module is self contained and does not depend on other parts of the repository. It requires `pandas`, `numpy`, `requests`, and `scipy`. Tests can be executed with `pytest` from the repository root.

## License

MIT
